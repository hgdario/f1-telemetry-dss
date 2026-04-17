"""
TelemetryTrace.py — TALOS F1 Telemetry System
==============================================
Módulo de visualización de telemetría individual por vuelta.

Uso desde el enrutador principal:
    from modules import TelemetryTrace as trace
    trace.render_telemetry_trace(session, driver, lap_number)
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import fastf1
from fastf1.core import Session, Lap
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES DE DISEÑO
# ─────────────────────────────────────────────────────────────────────────────
F1_RED        = "#E8002D"
F1_WHITE      = "#FFFFFF"
ACCENT_AMBER  = "#FFA500"   # Freno
ACCENT_GREEN  = "#39FF14"   # Acelerador
ACCENT_CYAN   = "#00D2FF"   # Velocidad
ACCENT_PURPLE = "#C77DFF"   # RPM
ACCENT_YELLOW = "#FFD600"   # Marchas

BG_DARK       = "#0E0E0F"
BG_SURFACE    = "#1A1A1F"
BG_PANEL      = "#111115"
GRID_COLOR    = "rgba(255,255,255,0.06)"
ZERO_LINE     = "rgba(255,255,255,0.12)"

PLOTLY_FONT   = dict(family="'JetBrains Mono', 'Courier New', monospace",
                     color=F1_WHITE, size=11)

TYRE_COLORS = {
    "SUPERSOFT": "#E8002D", 
    "ULTRASOFT": "#C77DFF",     
    "HYPERSOFT": "#FF66B2",    
    "SOFT"    : "#E8002D",
    "MEDIUM"  : "#FFF200",
    "HARD"    : "#EBEBEB",
    "INTER"   : "#43B02A",
    "WET"     : "#0067FF",
    "UNKNOWN" : "#888888",
}

TYRE_ICONS = {
    "HYPERSOFT": "●",
    "SUPERSOFT": "🔴",
    "ULTRASOFT": "🟣",
    "SOFT"  : "🔴",
    "MEDIUM": "🟡",
    "HARD"  : "⚪",
    "INTER" : "🟢",
    "WET"   : "🔵",
}

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS DE FORMATO
# ─────────────────────────────────────────────────────────────────────────────

def _format_laptime(td: pd.Timedelta) -> str:
    """Convierte un Timedelta a string MM:SS.mmm."""
    try:
        total_seconds = td.total_seconds()
        minutes   = int(total_seconds // 60)
        seconds   = int(total_seconds % 60)
        millis    = int(round((total_seconds % 1) * 1000))
        return f"{minutes:01d}:{seconds:02d}.{millis:03d}"
    except Exception:
        return "—"

def _get_real_mid_throttle_mask(tel: pd.DataFrame, threshold: float | None = None) -> pd.Series:
    """
    Detecta las muestras en las que el piloto mantiene un acelerador parcial estable.
    
    El umbral de derivada es ahora adaptativo: si no se proporciona, se calcula como el
    percentil 70 de la derivada absoluta de la vuelta. Esto evita el valor fijo de 25.0
    que era arbitrario e ignoraba el ritmo de muestreo real de la telemetría.
    """
    dt    = tel['Time'].dt.total_seconds().diff()
    d_thr = tel['Throttle'].diff()
    derivative = (d_thr / dt).fillna(0)

    if threshold is None:
        # Umbral adaptativo: percentil 70 de la derivada absoluta de esta vuelta
        threshold = float(np.percentile(derivative.abs().dropna(), 70))
        threshold = max(threshold, 5.0)   # mínimo de seguridad

    is_mid_position = (tel['Throttle'] > 0) & (tel['Throttle'] < 99)
    is_pedal_stable = derivative.abs() < threshold

    return is_mid_position & is_pedal_stable


# ─────────────────────────────────────────────────────────────────────────────
# MOTOR DE INFERENCIAS — SISTEMA FUZZY + BASELINE DE SESIÓN
# ─────────────────────────────────────────────────────────────────────────────

def _extract_pedal_stats(tel: pd.DataFrame) -> dict:
    """
    Calcula los porcentajes de tiempo en cada fase de pedal para un DataFrame de
    telemetría. Función auxiliar compartida entre el histograma y el motor de inferencias.

    Retorna dict con claves: p_full, p_part, p_coast, p_brk  (valores 0-100).
    """
    brake_col = tel['Brake'].copy()
    if brake_col.dtype == bool or set(brake_col.dropna().unique()).issubset({0, 1, True, False}):
        brake_val = brake_col.astype(float) * 100
    else:
        brake_val = brake_col.copy()

    throttle_col = tel['Throttle']
    mask_mid_gas = _get_real_mid_throttle_mask(tel)

    if 'Distance' in tel.columns:
        dist_delta = tel['Distance'].diff().fillna(0).clip(lower=0)
    else:
        dist_delta = pd.Series(1, index=tel.index) # Fallback a tiempo si falla
    
    total_dist = dist_delta.sum()
    if total_dist == 0: total_dist = 1

    #Sumamos metros recorridos.
    dist_full  = dist_delta[throttle_col >= 98].sum()
    dist_part  = dist_delta[mask_mid_gas].sum()
    dist_brk   = dist_delta[brake_val > 0].sum()
    dist_coast = dist_delta[(throttle_col == 0) & (brake_val == 0)].sum()

    return {
        "p_full" : (dist_full  / total_dist) * 100,
        "p_part" : (dist_part  / total_dist) * 100,
        "p_coast": (dist_coast / total_dist) * 100,
        "p_brk"  : (dist_brk   / total_dist) * 100,
    }


def _compute_session_baseline(session, driver: str) -> dict | None:
    """
    Calcula la media y desviación típica de las métricas de pedal sobre todas las
    vueltas rápidas válidas del piloto en la sesión.

    Esto convierte los thresholds fijos (¿por qué 72%?) en referencias relativas al
    propio circuito y piloto: una vuelta con p_full=68% en Mónaco puede ser altísima,
    mientras que en Monza es lo normal.

    Retorna None si no hay suficientes vueltas para calcular baseline.
    """
    try:
        laps = session.laps.pick_drivers(driver).pick_quicklaps()
        if len(laps) < 3:
            return None

        records = {"p_full": [], "p_part": [], "p_coast": [], "p_brk": []}

        for _, lap in laps.iterrows():
            try:
                tel_lap = lap.get_telemetry()
                if tel_lap.empty:
                    continue
                stats = _extract_pedal_stats(tel_lap)
                for k in records:
                    records[k].append(stats[k])
            except Exception:
                continue

        if not records["p_full"] or len(records["p_full"]) < 3:
            return None

        "Tomamos la media, varianza estandar y los percentiles"
        return {
            k: {
                "mean": float(np.mean(v)),
                "std" : float(np.std(v)) + 1e-6,   # evitar div/0
                "p25" : float(np.percentile(v, 25)),
                "p75" : float(np.percentile(v, 75)),
            }
            for k, v in records.items()
        }

    except Exception:
        return None


def _sigmoid_score(value: float, center: float, steepness: float = 0.35) -> float:
    """
    Convierte un valor continuo en una puntuación 0-1 con transición suave.
    
    - score ≈ 0.5  →  el valor está en el centro (neutro)
    - score ≈ 1.0  →  valor muy por encima del centro
    - score ≈ 0.0  →  valor muy por debajo del centro
    
    """
    return float(1 / (1 + np.exp(-steepness * (value - center))))


def _zscore_relative(value: float, baseline_entry: dict) -> float:
    """Desviación del valor respecto a la media de la sesión."""
    return (value - baseline_entry["mean"]) / baseline_entry["std"]


def _rpm_score(tel: pd.DataFrame) -> float:
    """
    Puntuación 0-1 de exigencia de motor basada en el percentil 90 de RPM.
    Los coches de F1 moderno tienen límite ~15.000 rpm; se normaliza sobre eso.
    """
    if "RPM" not in tel.columns:
        return 0.5  # sin datos → neutro
    rpm_series = tel["RPM"].dropna()
    if rpm_series.empty:
        return 0.5
    p90 = float(np.percentile(rpm_series, 90))
    return min(p90 / 14_500, 1.0)


def _gear_aggression_score(tel: pd.DataFrame) -> float:
    """
    Puntuación 0-1 de agresividad mecánica basada en el número de cambios de marcha
    por kilómetro. Muchos cambios → circuito técnico o estilo muy reactivo.
    """
    if "nGear" not in tel.columns or "Distance" not in tel.columns:
        return 0.5
    gear   = tel["nGear"].dropna()
    dist_m = tel["Distance"].dropna()
    if gear.empty or dist_m.empty or dist_m.max() < 100:
        return 0.5
    n_changes = int(gear.diff().abs().gt(0).sum())
    km        = dist_m.max() / 1000
    shifts_per_km = n_changes / km
    # ~4 cambios/km es bajo (circuitos rápidos); ~12 es alto (circuitos urbanos)
    return _sigmoid_score(shifts_per_km, center=7.0, steepness=0.25)


def _brake_aggressiveness_score(tel: pd.DataFrame) -> float:
    """
    Mide la brutalidad de las frenadas a partir del gradiente negativo de velocidad.
    Un gradiente muy negativo (deceleración repentina) indica frenadas tardías y agresivas.
    """
    if "Speed" not in tel.columns or "Time" not in tel.columns:
        return 0.5
    speed = tel["Speed"].dropna()
    if speed.empty:
        return 0.5
    dt        = tel["Time"].dt.total_seconds().diff().fillna(0.1)
    d_speed   = speed.diff().fillna(0)
    decel     = (d_speed / dt).clip(upper=0)            # solo valores negativos
    p10_decel = float(np.percentile(decel.dropna(), 10)) # el 10% más agresivo
    # −60 km/h/s es suave; −200 km/h/s es muy agresivo
    return _sigmoid_score(-p10_decel, center=120, steepness=0.015)


def _generate_driving_summary(tel: pd.DataFrame, session=None, driver: str = "") -> str:
    """
    Motor de inferencias de estilo de conducción.
    ─────────────────────────────────────────────────────────────
    1. NORMALIZACIÓN RELATIVA A LA SESIÓN: si se pasa `session` y `driver`,
       los porcentajes se comparan contra la baseline del propio piloto en
       ese circuito, no contra valores fijos universales.

    2. LÓGICA DIFUSA (sigmoide): en lugar de if/elif binarios, cada señal
       produce una puntuación continua 0-1, eliminando saltos artificiales.

    3. SCORES COMPUESTOS: las señales individuales se combinan en 4 dimensiones
       interpretables (Agresividad, Eficiencia, Tracción, Frenada).

    4. MÁS CANALES: además de Throttle y Brake, usa RPM, Speed y Gear.
    """
    # ── 1. Extraer stats de pedal ──────────────────────────────────────────
    stats = _extract_pedal_stats(tel)
    p_full  = stats["p_full"]
    p_part  = stats["p_part"]
    p_coast = stats["p_coast"]
    p_brk   = stats["p_brk"]

    if p_full + p_part + p_coast + p_brk == 0:
        return "No hay datos suficientes para generar un perfil de conducción."

    # ── 2. Baseline de sesión (si está disponible) ─────────────────────────
    baseline = None
    if session is not None and driver:
        baseline = _compute_session_baseline(session, driver)

    # ── 3. Centros de referencia: sesión o valores universales de fallback ──
    if baseline:
        center_full  = baseline["p_full"]["mean"]
        center_coast = baseline["p_coast"]["mean"]
        center_part  = baseline["p_part"]["mean"]
        center_brk   = baseline["p_brk"]["mean"]
    else:
        # Fallback: medias "universales" de referencia para F1 moderno
        center_full  = 63.0
        center_coast = 5.5
        center_part  = 9.0
        center_brk   = 18.0

    # ── 4. Scores fuzzy individuales (0 = bajo, 1 = alto) ──────────────────
    s_full        = _sigmoid_score(p_full,  center=center_full,  steepness=0.22)
    s_coast       = _sigmoid_score(p_coast, center=center_coast, steepness=0.45)
    s_part        = _sigmoid_score(p_part,  center=center_part,  steepness=0.30)
    s_brk         = _sigmoid_score(p_brk,   center=center_brk,   steepness=0.22)
    s_rpm         = _rpm_score(tel)
    s_gear_agg    = _gear_aggression_score(tel)
    s_brake_agg   = _brake_aggressiveness_score(tel)

    # ── 5. Scores compuestos (ponderados) ──────────────────────────────────
    score_agresividad = (
        0.40 * s_full       +   # a fondo = motor exigido
        0.25 * s_rpm        +   # altas RPM = ritmo alto
        0.20 * (1 - s_coast)+   # menos coasting = más agresivo
        0.15 * s_brake_agg      # frenadas tardías = riesgo/velocidad
    )

    score_eficiencia = (
        0.45 * s_coast      +   # coasting = ahorro deliberado
        0.30 * (1 - s_brk)  +   # menos freno = más fluido
        0.25 * (1 - s_part)     # menos medio-gas = aceleración más limpia
    )

    score_traccion = (
        0.60 * s_part       +   # medio-gas prolongado = dificultad de tracción
        0.40 * s_gear_agg       # muchos cambios = circuito técnico
    )

    score_frenada = (
        0.55 * s_brake_agg  +   # gradiente de velocidad
        0.45 * s_brk            # tiempo en freno
    )

    # ── 6. Contexto adicional si hay baseline ─────────────────────────────
    contexto_baseline = ""
    if baseline:
        z_full  = _zscore_relative(p_full,  baseline["p_full"])
        z_coast = _zscore_relative(p_coast, baseline["p_coast"])
        if abs(z_full) > 1.5:
            dir_full = "superior" if z_full > 0 else "inferior"
            contexto_baseline = (
                f" Esta vuelta es notablemente {dir_full} a su media de sesión "
                f"({baseline['p_full']['mean']:.1f}% a fondo de media vs "
                f"{p_full:.1f}% en esta vuelta)."
            )
        elif abs(z_coast) > 1.5:
            dir_coast = "más" if z_coast > 0 else "menos"
            contexto_baseline = (
                f" El coasting es {dir_coast} pronunciado que en el resto de sus vueltas "
                f"({p_coast:.1f}% vs media de {baseline['p_coast']['mean']:.1f}%)."
            )

    # ── 7. Generación de insights en lenguaje natural ──────────────────────
    insights = []

    # — Agresividad / Motor —
    if score_agresividad > 0.70:
        insights.append(
            f"una exigencia de motor muy alta ({p_full:.0f}% a fondo, RPM en percentil 90 elevado). "
            "El piloto está en modo ataque pleno — perfil típico de vuelta de clasificación "
            "o circuito de alta velocidad como Monza o Silverstone"
        )
    elif score_agresividad < 0.35:
        insights.append(
            f"un perfil conservador de motor ({p_full:.0f}% a fondo). "
            "El piloto gestiona recursos, posiblemente por estrategia de carrera "
            "larga, tráfico, o condiciones de agarre limitadas"
        )

    # — Eficiencia / Coasting —
    if score_eficiencia > 0.68:
        insights.append(
            f"una gestión energética muy marcada ({p_coast:.1f}% en coasting). "
            "El 'Lift & Coast' sugiere estrategia activa de ahorro de combustible "
            "o cuidado de frenos, común en vueltas finales de stint largo"
        )
    elif score_eficiencia < 0.30 and score_agresividad > 0.55:
        insights.append(
            f"conducción al límite con transiciones pedal-pedal muy directas "
            f"({p_coast:.1f}% de coasting). Pasa del acelerador al freno en milisegundos "
            "sin dejar rodar el coche por inercia"
        )

    # — Tracción —
    if score_traccion > 0.65:
        insights.append(
            f"una demanda de tracción elevada ({p_part:.0f}% en medio-gas). "
            "El piloto modula el acelerador constantemente en la salida de curvas, "
            "señal de grip limitado por temperatura de neumático, desgaste, o trazado técnico"
        )

    # — Frenada —
    if score_frenada > 0.72:
        insights.append(
            f"frenadas muy agresivas y tardías ({p_brk:.0f}% del tiempo frenando). "
            "El gradiente de deceleración es elevado, lo que indica que el piloto busca "
            "el punto de frenada más tarde posible — alta confianza en el coche o "
            "intento de adelantamiento"
        )
    elif score_frenada < 0.30:
        insights.append(
            f"frenadas progresivas y suaves ({p_brk:.0f}% del tiempo frenando). "
            "Conducción fluida que cuida los neumáticos y distribución de temperatura"
        )

    # ── 8. Construir texto final ───────────────────────────────────────────
    if not insights:
        return (
            f"El piloto muestra un perfil de conducción equilibrado para este circuito "
            f"({p_full:.0f}% a fondo · {p_coast:.1f}% coasting · {p_brk:.0f}% frenando), "
            "sin excesos en ninguna dimensión. Ritmo de carrera estándar."
            + contexto_baseline
        )

    if len(insights) == 1:
        resumen = insights[0]
    elif len(insights) == 2:
        resumen = insights[0] + " y " + insights[1]
    else:
        resumen = ", ".join(insights[:-1]) + " y " + insights[-1]

    return f"El análisis de telemetría revela {resumen}.{contexto_baseline}"

def _format_gap(gap_td: Optional[pd.Timedelta]) -> str:
    """Convierte un gap al pole a string +S.mmm."""
    if gap_td is None:
        return "—"
    try:
        secs = gap_td.total_seconds()
        if secs == 0:
            return "POLE"
        return f"+{secs:.3f}s"
    except Exception:
        return "—"


def _safe_series(tel: pd.DataFrame, channel: str) -> Optional[pd.Series]:
    """Devuelve la serie si existe y tiene datos; None en caso contrario."""
    if channel not in tel.columns:
        return None
    s = tel[channel].dropna()
    return s if not s.empty else None


# ─────────────────────────────────────────────────────────────────────────────
# EXTRACCIÓN DE DATOS DE VUELTA
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=600, show_spinner=False)
def _get_lap_data(
    session_key: str,
    driver: str,
    lap_number: int,
) -> dict:
    """
    Extrae métricas y telemetría de una vuelta. Cacheado por sesión+piloto+vuelta.
    `session_key` es un identificador único de la sesión (ej. "2024_Monza_R").
    """
    raise NotImplementedError(
        "_get_lap_data se llama internamente a través de render_telemetry_trace; "
        "no llames esta función directamente."
    )


def _extract_lap_metrics(session: Session, driver: str, lap_number: int) -> dict:
    """
    Extrae métricas de contexto para el panel de vuelta.
    Retorna un dict con: lap_time, vmax, compound, tyre_life, gap_to_pole.
    """
    result = {
        "lap_time"    : None,
        "vmax"        : None,
        "compound"    : "UNKNOWN",
        "tyre_life"   : None,
        "gap_to_pole" : None,
        "driver_name" : driver,
        "team_color"  : F1_RED,
    }

    try:
        laps = session.laps.pick_drivers(driver)
        lap: Lap = laps[laps["LapNumber"] == lap_number].iloc[0]

        result["lap_time"] = lap.get("LapTime")

        # Neumático
        compound = lap.get("Compound", "UNKNOWN")
        result["compound"] = str(compound).upper() if pd.notna(compound) else "UNKNOWN"

        tyre_life = lap.get("TyreLife")
        result["tyre_life"] = int(tyre_life) if pd.notna(tyre_life) else None

        # Gap to Pole: tomamos la vuelta más rápida de la sesión
        try:
            fastest = session.laps.pick_fastest()
            pole_time = fastest["LapTime"]
            lap_time  = lap["LapTime"]
            if pd.notna(pole_time) and pd.notna(lap_time):
                gap = lap_time - pole_time
                result["gap_to_pole"] = gap if gap.total_seconds() >= 0 else pd.Timedelta(0)
        except Exception:
            pass

        # Velocidad punta desde telemetría
        try:
            tel = lap.get_telemetry().add_distance()
            speed_series = _safe_series(tel, "Speed")
            if speed_series is not None:
                result["vmax"] = round(float(speed_series.max()), 1)
        except Exception:
            pass

        # Color de equipo
        try:
            result["team_color"] = "#" + session.results.loc[
                session.results["Abbreviation"] == driver, "TeamColor"
            ].iloc[0]
        except Exception:
            result["team_color"] = F1_RED

    except (IndexError, KeyError, Exception):
        pass

    return result


def _extract_telemetry(session: Session, driver: str, lap_number: int) -> Optional[pd.DataFrame]:
    """
    Descarga la telemetría de una vuelta específica y añade columna Distance.
    Retorna DataFrame o None si falla.
    """
    try:
        laps = session.laps.pick_drivers(driver)
        lap: Lap = laps[laps["LapNumber"] == lap_number].iloc[0]
        tel = lap.get_telemetry().add_distance()
        if tel.empty:
            return None
        return tel
    except Exception as e:
        st.warning(f"⚠️ No se pudo cargar la telemetría: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# PANEL DE MÉTRICAS
# ─────────────────────────────────────────────────────────────────────────────

def _render_metrics_dashboard(metrics: dict, lap_number: int) -> None:
    """
    Renderiza el panel de contexto de vuelta con métricas clave.
    Diseño tipo 'data terminal': compacto, denso en información, sin relleno.
    """
    compound     = metrics["compound"]
    tyre_color   = TYRE_COLORS.get(compound, TYRE_COLORS["UNKNOWN"])
    tyre_icon    = TYRE_ICONS.get(compound, "⚫")
    team_color   = metrics.get("team_color", F1_RED)
    lap_time_str = _format_laptime(metrics["lap_time"]) if metrics["lap_time"] else "—"
    gap_str      = _format_gap(metrics.get("gap_to_pole"))
    vmax_str     = f"{metrics['vmax']} km/h" if metrics["vmax"] else "—"
    tyre_life_str = f"{metrics['tyre_life']} vueltas" if metrics["tyre_life"] else "—"

    # Cabecera: Piloto + Vuelta
    st.markdown(
        f"""
        <div style="
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 10px 0 6px 0;
            border-bottom: 2px solid {team_color};
            margin-bottom: 16px;
        ">
            <span style="
                font-size: 22px;
                font-weight: 800;
                letter-spacing: 3px;
                color: {F1_WHITE};
                font-family: 'JetBrains Mono', monospace;
            ">{metrics['driver_name']}</span>
            <span style="
                font-size: 13px;
                color: rgba(255,255,255,0.45);
                font-family: 'JetBrains Mono', monospace;
                letter-spacing: 2px;
            ">LAP {lap_number:02d}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Métricas en 4 columnas
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="⏱ TIEMPO DE VUELTA",
            value=lap_time_str,
            help="Tiempo total de la vuelta seleccionada"
        )

    with col2:
        st.metric(
            label="🚀 VELOCIDAD PUNTA",
            value=vmax_str,
            help="Velocidad máxima registrada en la vuelta"
        )

    with col3:
        # Neumático con color de compuesto dinámico para el icono
        st.markdown(
            f"""
            <div style="padding: 4px 0;">
                <p style="
                    font-size: 12px;
                    color: rgba(255,255,255,0.55);
                    margin: 0 0 6px 0;
                    font-family: 'JetBrains Mono', monospace;
                    letter-spacing: 1px;
                    text-transform: uppercase;
                "><span style="color: {tyre_color}; font-size: 14px;">{tyre_icon}</span> NEUMÁTICO</p>
                <p style="
                    font-size: 22px;
                    font-weight: 700;
                    color: {tyre_color};
                    margin: 0;
                    font-family: 'JetBrains Mono', monospace;
                    letter-spacing: 2px;
                ">{compound}</p>
                <p style="
                    font-size: 12px;
                    color: rgba(255,255,255,0.4);
                    margin: 4px 0 0 0;
                    font-family: 'JetBrains Mono', monospace;
                ">vida: {tyre_life_str}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col4:
        delta_color = "off" if gap_str in ("—", "POLE") else "inverse"
        st.metric(
            label="🏁 GAP TO POLE",
            value=gap_str,
            help="Diferencia respecto a la vuelta más rápida de la sesión"
        )


# ─────────────────────────────────────────────────────────────────────────────
# GRÁFICO DE TELEMETRÍA
# ─────────────────────────────────────────────────────────────────────────────

def _build_telemetry_figure(tel: pd.DataFrame, team_color: str) -> go.Figure:
    """
    Construye la figura Plotly con 4 subplots verticales compartiendo eje X (Distancia).
    Subplots: Velocidad | Acelerador+Freno | Marchas | RPM
    """
    dist = tel["Distance"]

    # Extraer canales de forma segura
    speed    = _safe_series(tel, "Speed")
    throttle = _safe_series(tel, "Throttle")
    brake    = _safe_series(tel, "Brake")
    gear     = _safe_series(tel, "nGear")
    rpm      = _safe_series(tel, "RPM")

    # Decidir qué filas mostrar y sus alturas relativas
    rows_config = [
        {"label": "SPEED",         "available": speed    is not None},
        {"label": "THR / BRK",     "available": throttle is not None or brake is not None},
        {"label": "GEAR",          "available": gear     is not None},
        {"label": "RPM",           "available": rpm      is not None},
    ]
    visible_rows = [r for r in rows_config if r["available"]]
    n_rows = len(visible_rows)

    if n_rows == 0:
        return go.Figure()

    row_heights = []
    for r in visible_rows:
        if r["label"] == "SPEED":
            row_heights.append(0.38)
        elif r["label"] == "THR / BRK":
            row_heights.append(0.26)
        elif r["label"] == "GEAR":
            row_heights.append(0.18)
        else:
            row_heights.append(0.18)

    # Normalizar alturas
    total = sum(row_heights)
    row_heights = [h / total for h in row_heights]

    subplot_titles = [r["label"] for r in visible_rows]

    fig = make_subplots(
        rows=n_rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=subplot_titles,
        row_heights=row_heights,
    )

    row_idx = {r["label"]: i + 1 for i, r in enumerate(visible_rows)}

    # ── Velocidad ──────────────────────────────────────────────────────────
    if speed is not None and "SPEED" in row_idx:
        r = row_idx["SPEED"]
        fig.add_trace(
            go.Scatter(
                x=dist.loc[speed.index],
                y=speed,
                mode="lines",
                name="Speed (km/h)",
                line=dict(color=ACCENT_CYAN, width=2),
                fill="tozeroy",
                fillcolor="rgba(0,210,255,0.08)",
                hovertemplate="<b>%{y:.0f} km/h</b><extra>Speed</extra>",
            ),
            row=r, col=1,
        )
        # Anotación de Vmax
        vmax_idx = speed.idxmax()
        fig.add_annotation(
            x=float(dist.loc[vmax_idx]),
            y=float(speed.loc[vmax_idx]),
            text=f"V<sub>max</sub> {speed.loc[vmax_idx]:.0f}",
            showarrow=True,
            arrowhead=2,
            arrowcolor=ACCENT_CYAN,
            font=dict(color=ACCENT_CYAN, size=10, family="JetBrains Mono, monospace"),
            bgcolor=BG_SURFACE,
            bordercolor=ACCENT_CYAN,
            borderwidth=1,
            row=r, col=1,
            ax=40, ay=-30,
        )

    # ── Acelerador + Freno ─────────────────────────────────────────────────
    if "THR / BRK" in row_idx:
        r = row_idx["THR / BRK"]
        if throttle is not None:
            # Normalizar 0-100 si viene en 0-1
            thr_vals = throttle.copy()
            if thr_vals.max() <= 1.0:
                thr_vals = thr_vals * 100
            fig.add_trace(
                go.Scatter(
                    x=dist.loc[thr_vals.index],
                    y=thr_vals,
                    mode="lines",
                    name="Throttle %",
                    line=dict(color=ACCENT_GREEN, width=1.5),
                    fill="tozeroy",
                    fillcolor="rgba(57,255,20,0.10)",
                    hovertemplate="<b>%{y:.0f}%</b><extra>Throttle</extra>",
                ),
                row=r, col=1,
            )
        if brake is not None:
            # El freno puede ser bool (True/False) o float (0-1)
            brk_vals = brake.copy()
            if brk_vals.dtype == bool or set(brk_vals.dropna().unique()).issubset({0, 1, True, False}):
                brk_vals = brk_vals.astype(float) * 100
            elif brk_vals.max() <= 1.0:
                brk_vals = brk_vals * 100
            fig.add_trace(
                go.Scatter(
                    x=dist.loc[brk_vals.index],
                    y=brk_vals,
                    mode="lines",
                    name="Brake %",
                    line=dict(color=ACCENT_AMBER, width=1.5),
                    fill="tozeroy",
                    fillcolor="rgba(255,165,0,0.12)",
                    hovertemplate="<b>%{y:.0f}%</b><extra>Brake</extra>",
                ),
                row=r, col=1,
            )

    # ── Marchas ────────────────────────────────────────────────────────────
    if gear is not None and "GEAR" in row_idx:
        r = row_idx["GEAR"]
        fig.add_trace(
            go.Scatter(
                x=dist.loc[gear.index],
                y=gear,
                mode="lines",
                name="Gear",
                line=dict(color=ACCENT_YELLOW, width=1.5, shape="hv"),  # escalón
                hovertemplate="<b>Gear %{y:.0f}</b><extra></extra>",
            ),
            row=r, col=1,
        )

    # ── RPM ────────────────────────────────────────────────────────────────
    if rpm is not None and "RPM" in row_idx:
        r = row_idx["RPM"]
        fig.add_trace(
            go.Scatter(
                x=dist.loc[rpm.index],
                y=rpm,
                mode="lines",
                name="RPM",
                line=dict(color=ACCENT_PURPLE, width=1.5),
                fill="tozeroy",
                fillcolor="rgba(199,125,255,0.07)",
                hovertemplate="<b>%{y:,.0f} rpm</b><extra>RPM</extra>",
            ),
            row=r, col=1,
        )

    # ── Layout global ──────────────────────────────────────────────────────
    total_height = max(520, n_rows * 160)

    fig.update_layout(
        height=total_height,
        paper_bgcolor=BG_DARK,
        plot_bgcolor=BG_PANEL,
        font=PLOTLY_FONT,
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor=BG_SURFACE,
            bordercolor="rgba(255,255,255,0.15)",
            font=dict(family="JetBrains Mono, monospace", color=F1_WHITE, size=11),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="right",
            x=1,
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            font=dict(size=10),
        ),
        margin=dict(l=54, r=24, t=40, b=40),
        showlegend=True,
    )

    # ── Estilos de ejes compartidos ────────────────────────────────────────
    axis_common = dict(
        showgrid=True,
        gridcolor=GRID_COLOR,
        gridwidth=1,
        zeroline=True,
        zerolinecolor=ZERO_LINE,
        zerolinewidth=1,
        linecolor="rgba(255,255,255,0.12)",
        tickfont=dict(family="JetBrains Mono, monospace", size=9, color="rgba(255,255,255,0.5)"),
        title_font=dict(size=10, color="rgba(255,255,255,0.6)"),
    )

    for i in range(1, n_rows + 1):
        ykey = "yaxis" if i == 1 else f"yaxis{i}"
        xkey = "xaxis" if i == 1 else f"xaxis{i}"
        fig.update_layout(
            **{
                ykey: {**axis_common},
                xkey: {
                    **axis_common,
                    "title": "Distancia (m)" if i == n_rows else "",
                    "ticksuffix": " m" if i == n_rows else "",
                },
            }
        )

    # Ajuste fino por canal
    if "SPEED" in row_idx:
        r = row_idx["SPEED"]
        ykey = "yaxis" if r == 1 else f"yaxis{r}"
        fig.update_layout(**{ykey: {**axis_common, "ticksuffix": " km/h"}})

    if "GEAR" in row_idx:
        r = row_idx["GEAR"]
        ykey = "yaxis" if r == 1 else f"yaxis{r}"
        fig.update_layout(**{
            ykey: {
                **axis_common,
                "dtick": 1,
                "range": [0, 9],
                "tickvals": list(range(1, 9)),
            }
        })

    if "RPM" in row_idx:
        r = row_idx["RPM"]
        ykey = "yaxis" if r == 1 else f"yaxis{r}"
        fig.update_layout(**{ykey: {**axis_common, "ticksuffix": " rpm"}})

    # Títulos de subplot en mono
    for annotation in fig.layout.annotations:
        annotation.update(
            font=dict(family="JetBrains Mono, monospace", size=10,
                      color="rgba(255,255,255,0.45)"),
            x=0,
            xanchor="left",
        )

    # --- AÑADIR ETIQUETAS DE CURVAS ---
    try:
        # Recuperamos la información del circuito desde la sesión
        # Nota: La sesión debe estar cargada previamente
        circuit_info = tel.session.get_circuit_info()
        
        for _, corner in circuit_info.corners.iterrows():
            # 1. Línea vertical punteada que atraviesa todos los subplots
            fig.add_vline(
                x=corner['Distance'],
                line_width=1,
                line_dash="dash",
                line_color="rgba(255, 255, 255, 0.2)",
                layer="below"
            )
            
            # 2. Etiqueta con el número de curva (T1, T2...) arriba del todo
            fig.add_annotation(
                x=corner['Distance'],
                y=1.02, # Un poco por encima del primer gráfico
                xref="x",
                yref="paper",
                text=f"T{corner['Number']}",
                showarrow=False,
                font=dict(
                    family="JetBrains Mono, monospace",
                    size=10,
                    color="rgba(255, 255, 255, 0.5)"
                ),
                align="center"
            )
    except Exception as e:
        # Si por algún motivo no hay datos de mapa, la app sigue funcionando
        pass
    return fig

# ── Histograma ────────────────────────────────────────

def _build_pedal_histogram(tel: pd.DataFrame) -> go.Figure:
    # Usamos el helper compartido para obtener los porcentajes reales
    stats = _extract_pedal_stats(tel)

    p_full  = stats["p_full"]
    p_part  = stats["p_part"]
    p_brk   = stats["p_brk"]
    p_coast = stats["p_coast"]
    
    # Calculamos los metros de "Transición" (el porcentaje huérfano para llegar a 100)
    p_trans = max(0, 100.0 - (p_full + p_part + p_brk + p_coast))

    # Añadimos la quinta categoría al gráfico
    labels = ['A FONDO', 'MEDIO GAS', 'FRENANDO', 'COASTING', 'TRANSICIÓN']
    values = [p_full, p_part, p_brk, p_coast, p_trans]
    
    # Le damos un color casi transparente a la transición para que no ensucie
    colors = [ACCENT_GREEN, '#206020', ACCENT_AMBER, '#50506A', 'rgba(255,255,255,0.05)']

    fig = go.Figure()

    # 1. El donut anclado a la mitad derecha
    fig.add_trace(go.Pie(
        labels=labels,
        values=values,
        hole=0.45,
        domain=dict(x=[0.5, 1.0]),
        marker=dict(colors=colors, line=dict(color=BG_DARK, width=2)),
        textinfo='percent',
        hoverinfo='label+percent',
        textfont=dict(family="JetBrains Mono, monospace", size=14, color=F1_WHITE),
        sort=False # IMPORTANTE: Evita que Plotly cambie el orden de los colores
    ))

    # 2. La leyenda actualizada
    legend_text = (
        "<b>Distribución de la Vuelta</b><br><br>"
        "<span style='font-size: 11px; color: rgba(255,255,255,0.6);'>Porcentaje de distancia en cada fase de pilotaje.</span><br><br>"
        f"<span style='color: {ACCENT_GREEN};'>■</span> A fondo (Acelerador > 98%)<br>"
        f"<span style='color: {ACCENT_AMBER};'>■</span> Frenando: Fase de deceleración<br>"
        f"<span style='color: #206020;'>■</span> Medio Gas: Tracción parcial estable<br>"
        "<span style='color: #50506A;'>■</span> Coasting: Ni acelerador ni freno<br>"
        "<span style='color: rgba(255,255,255,0.2);'>■</span> Transición: Cambios bruscos de pedal"
    )

    fig.add_annotation(
        x=0, y=0.5,
        xref="paper", yref="paper",
        text=legend_text,
        showarrow=False,
        align="left",
        font=dict(family="JetBrains Mono, monospace", size=12, color=F1_WHITE)
    )

    fig.update_layout(
        height=220,
        margin=dict(t=20, b=20, l=20, r=20),
        paper_bgcolor=BG_DARK,
        plot_bgcolor=BG_DARK,
        showlegend=False
    )
    return fig
# ─────────────────────────────────────────────────────────────────────────────
# PUNTO DE ENTRADA PÚBLICO
# ─────────────────────────────────────────────────────────────────────────────

def render_telemetry_trace(
    session: Session,
    driver: str,
    lap_number: int,
) -> None:
    """
    Punto de entrada principal del módulo TelemetryTrace.

    Renders:
        1. Panel de métricas de contexto (tiempo, Vmax, neumático, gap al pole).
        2. Gráfico de telemetría multi-canal compartiendo eje X de Distancia.

    Parámetros
    ----------
    session    : fastf1.core.Session ya cargada (con load_telemetry=True).
    driver     : Abreviatura del piloto (ej. "VER", "HAM").
    lap_number : Número de vuelta a analizar.
    """
    # 1. Spinner de carga
    with st.spinner(f"Cargando telemetría — {driver} · Vuelta {lap_number}"):

        # 2. Métricas de contexto
        metrics = _extract_lap_metrics(session, driver, lap_number)

        # 3. Telemetría raw
        tel = _extract_telemetry(session, driver, lap_number)

    # 4. Panel de métricas
    _render_metrics_dashboard(metrics, lap_number)

    st.divider()

    # 5. Validación antes del gráfico
    if tel is None or tel.empty:
        st.error(
            "❌ No hay datos de telemetría disponibles para esta vuelta. "
            "Verifica que la sesión fue cargada con `load_telemetry=True`."
        )
        return

    if "Distance" not in tel.columns:
        st.error("❌ El canal 'Distance' no está disponible en los datos de telemetría.")
        return

    # 6. Gráfico
    st.markdown(
        "<p style='font-family: JetBrains Mono, monospace; font-size: 11px; "
        "letter-spacing: 2px; color: rgba(255,255,255,0.35); margin-bottom: 8px;'>"
        "TELEMETRY TRACE · DISTANCIA vs CANALES</p>",
        unsafe_allow_html=True,
    )

    fig = _build_telemetry_figure(tel, metrics.get("team_color", F1_RED))
    st.plotly_chart(fig, use_container_width=True, config={
        "displayModeBar"  : True,
        "displaylogo"     : False,
        "modeBarButtonsToRemove": ["select2d", "lasso2d", "autoScale2d"],
        "toImageButtonOptions": {
            "format"  : "png",
            "filename": f"talos_trace_{driver}_lap{lap_number}",
            "scale"   : 2,
        },
    })
    # ---------------------------------------------------------
    # 7. ANÁLISIS DE PILOTAJE (USO DE PEDALES)
    # ---------------------------------------------------------
    st.divider()
    st.caption("ESTILO DE CONDUCCIÓN · USO DE PEDALES")

    # Renderizamos la gráfica que ahora contiene también el texto
    fig_pedals = _build_pedal_histogram(tel)
    st.plotly_chart(fig_pedals, use_container_width=True, config={"displayModeBar": False})

    # ---------------------------------------------------------
    # 8. CONCLUSIÓN PARA STAKEHOLDERS
    # ---------------------------------------------------------
    summary_text = _generate_driving_summary(tel, session=session, driver=driver)
    
    st.markdown(
        f"""
        <div style="background-color: {BG_PANEL}; padding: 15px; border-left: 3px solid {ACCENT_CYAN}; border-radius: 2px; margin-top: 15px;">
            <p style="font-family: 'Titillium Web', sans-serif; font-size: 14px; color: {F1_WHITE}; margin: 0;">
                <span style="color: {ACCENT_CYAN}; font-size: 16px;">💡 <b>ESTILO DE CONDUCCIÓN:</b></span><br>
                <span style="color: rgba(255,255,255,0.8); line-height: 1.5;">{summary_text}</span>
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )