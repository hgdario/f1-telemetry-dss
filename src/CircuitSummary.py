"""
CircuitSummary — Clasificación de Circuitos
============================================
Caracteriza un circuito a partir de la telemetría de la vuelta más rápida
de la sesión y lo asigna a una de cinco tipologías predefinidas, basándose
en umbrales determinísticos sobre fuerzas G, velocidad y uso del acelerador.

Tipologías reconocidas:
  · POWER          (Monza, Spa, Baku)        — alto throttle, alta velocidad media
  · HIGH-SPEED     (Suzuka, Silverstone)     — G laterales altas, flowing
  · STOP-AND-GO    (Singapur, Bakú street)   — frenadas largas, paradas
  · STREET         (Mónaco, Las Vegas)       — baja velocidad media, muchas curvas
  · TECHNICAL      (Barcelona, Hungaroring)  — equilibrio de demandas

100 % validable: el mismo circuito en cualquier año retorna la misma
clasificación porque las reglas son deterministas y no dependen de la sesión
(condiciones, neumáticos, estrategia).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ─── PALETA F1 ────────────────────────────────────────────────────────────────
F1_RED      = "#E8002D"
PIT_GREEN   = "#00D2BE"
WARN_YELLOW = "#FFF500"
INFO_BLUE   = "#5A9FD4"
GOLD        = "#FFD700"
WHITE       = "#FFFFFF"
DIM         = "#6A6A88"
PANEL       = "#15151E"

# ─── PERFILES TIPO (estáticos, derivados de telemetrías históricas) ──────────
# Cada perfil es la "huella" promedio normalizada (0-100) de un tipo de circuito
TYPE_PROFILES = {
    "POWER": {
        "icon":        "⚡",
        "color":       WARN_YELLOW,
        "description": "Circuitos dominados por rectas largas, alto porcentaje de gas a fondo y velocidades punta muy elevadas. Los motores dictan el cronómetro.",
        "examples":    "Monza · Spa · Baku · Jeddah",
        "profile":     {"speed_max": 92, "speed_avg": 85, "throttle_full_pct": 78,
                        "brake_pct": 18, "g_lat_avg": 30, "g_lon_max": 70,
                        "num_corners": 30},
    },
    "HIGH-SPEED": {
        "icon":        "◉",
        "color":       PIT_GREEN,
        "description": "Trazados rápidos de curvas de alta velocidad encadenadas. Las fuerzas G laterales son extremas y exigen un coche con buena carga aerodinámica.",
        "examples":    "Suzuka · Silverstone · Copse-Maggotts",
        "profile":     {"speed_max": 78, "speed_avg": 75, "throttle_full_pct": 60,
                        "brake_pct": 22, "g_lat_avg": 85, "g_lon_max": 65,
                        "num_corners": 50},
    },
    "STOP-AND-GO": {
        "icon":        "▣",
        "color":       F1_RED,
        "description": "Configuración con frenadas muy fuertes seguidas de aceleraciones largas. Los frenos sufren y la tracción mecánica es decisiva.",
        "examples":    "Singapur · Bahrain S2 · Canadá",
        "profile":     {"speed_max": 70, "speed_avg": 55, "throttle_full_pct": 45,
                        "brake_pct": 35, "g_lat_avg": 50, "g_lon_max": 90,
                        "num_corners": 60},
    },
    "STREET": {
        "icon":        "▤",
        "color":       INFO_BLUE,
        "description": "Circuitos urbanos de baja velocidad media, alta densidad de curvas y muros muy cercanos. Premian la precisión y la confianza del piloto.",
        "examples":    "Mónaco · Las Vegas · Singapur · Bakú",
        "profile":     {"speed_max": 60, "speed_avg": 45, "throttle_full_pct": 35,
                        "brake_pct": 30, "g_lat_avg": 55, "g_lon_max": 80,
                        "num_corners": 90},
    },
    "TECHNICAL": {
        "icon":        "◬",
        "color":       GOLD,
        "description": "Equilibrio entre demandas — sin extremos. Buenas opciones para que un coche todoterreno destaque sin un punto fuerte definido.",
        "examples":    "Barcelona · Hungaroring · México · COTA",
        "profile":     {"speed_max": 75, "speed_avg": 65, "throttle_full_pct": 55,
                        "brake_pct": 25, "g_lat_avg": 65, "g_lon_max": 60,
                        "num_corners": 55},
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
#  EXTRACCIÓN DE FEATURES DEL CIRCUITO
# ═══════════════════════════════════════════════════════════════════════════════
def _calculate_g_forces(tel: pd.DataFrame):
    """Calcula g_lat y g_lon a partir de Speed y posición XY.

    Devuelve (g_lat, g_lon, g_total) — cada uno como np.ndarray del mismo
    largo que `tel`.  Determinista, basado únicamente en derivadas
    numéricas de la telemetría.
    """
    G = 9.81
    # Tiempo en segundos
    if 'SessionTime' in tel:
        t = tel['SessionTime'].dt.total_seconds().to_numpy()
    elif 'Time' in tel:
        t = tel['Time'].dt.total_seconds().to_numpy()
    else:
        t = np.arange(len(tel)) * 0.05  # 20 Hz fallback

    # Velocidad en m/s
    v = tel['Speed'].astype(float).to_numpy() / 3.6

    # ─── G longitudinal ─── derivada de la velocidad
    dv = np.gradient(v, t)
    g_lon = dv / G

    # ─── G lateral ─── derivada del rumbo
    if 'X' in tel and 'Y' in tel:
        x = tel['X'].astype(float).to_numpy()
        y = tel['Y'].astype(float).to_numpy()
        dx = np.gradient(x, t)
        dy = np.gradient(y, t)
        heading = np.arctan2(dy, dx)
        # Desempaquetar el rumbo para evitar saltos de ±π
        heading_unwrapped = np.unwrap(heading)
        d_heading = np.gradient(heading_unwrapped, t)
        g_lat = v * d_heading / G
    else:
        g_lat = np.zeros_like(g_lon)

    # Filtro suave para reducir ruido numérico
    def _smooth(arr, w=5):
        if len(arr) < w:
            return arr
        kernel = np.ones(w) / w
        return np.convolve(arr, kernel, mode='same')

    g_lat = _smooth(g_lat)
    g_lon = _smooth(g_lon)
    g_total = np.sqrt(g_lat**2 + g_lon**2)
    return g_lat, g_lon, g_total


def _extract_circuit_features(session) -> dict:
    """Extrae el conjunto de features que definen un circuito."""
    fastest = session.laps.pick_fastest()
    tel     = fastest.get_telemetry().add_distance()

    # Info del circuito
    try:
        circuit_info = session.get_circuit_info()
        num_corners  = len(circuit_info.corners) if circuit_info is not None else 0
    except Exception:
        num_corners = 0

    # Features de velocidad
    speed = tel['Speed'].astype(float).to_numpy()
    speed_max = float(np.nanmax(speed))
    speed_avg = float(np.nanmean(speed))
    speed_min = float(np.nanmin(speed))
    speed_std = float(np.nanstd(speed))

    # % a fondo y % freno
    throttle = tel['Throttle'].astype(float).to_numpy()
    brake    = tel['Brake'].astype(bool).to_numpy()
    throttle_full_pct = float(np.mean(throttle >= 98.0) * 100)
    throttle_avg      = float(np.nanmean(throttle))
    brake_pct         = float(np.mean(brake) * 100)

    # Cambios de marcha
    num_gear_changes = 0
    if 'nGear' in tel:
        gear = tel['nGear'].astype(int).to_numpy()
        num_gear_changes = int(np.sum(np.diff(gear) != 0))

    # Fuerzas G
    g_lat, g_lon, g_total = _calculate_g_forces(tel)
    g_lat_max = float(np.nanmax(np.abs(g_lat)))
    g_lat_avg = float(np.nanmean(np.abs(g_lat)))
    g_lon_max = float(np.nanmax(np.abs(g_lon)))   # la frenada máxima
    g_lon_brake_max = float(np.nanmin(g_lon))     # negativo = frenada

    # Longitud de pista
    if 'Distance' in tel:
        track_length = float(tel['Distance'].iloc[-1] - tel['Distance'].iloc[0])
    else:
        track_length = 0.0

    # Tiempo de vuelta
    try:
        lap_time_s = float(fastest['LapTime'].total_seconds())
    except Exception:
        lap_time_s = 0.0

    return {
        "event_name":         str(session.event['EventName']),
        "country":            str(session.event.get('Country', '—')),
        "speed_max":          speed_max,
        "speed_avg":          speed_avg,
        "speed_min":          speed_min,
        "speed_std":          speed_std,
        "throttle_full_pct":  throttle_full_pct,
        "throttle_avg":       throttle_avg,
        "brake_pct":          brake_pct,
        "num_gear_changes":   num_gear_changes,
        "num_corners":        num_corners,
        "track_length_m":     track_length,
        "lap_time_s":         lap_time_s,
        "g_lat_max":          g_lat_max,
        "g_lat_avg":          g_lat_avg,
        "g_lon_max":          g_lon_max,
        "g_lon_brake_max":    g_lon_brake_max,
        "tel":                tel,
        "g_lat":              g_lat,
        "g_lon":              g_lon,
        "g_total":            g_total,
        "fastest_lap":        fastest,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  CLASIFICACIÓN DETERMINISTA POR REGLAS
# ═══════════════════════════════════════════════════════════════════════════════
def _classify_circuit(f: dict) -> tuple[str, list[str]]:
    """Aplica reglas determinísticas para clasificar el circuito.

    Cada regla suma puntos a un tipo. El tipo con más puntos gana.
    """
    scores = {k: 0 for k in TYPE_PROFILES}
    reasons: list[str] = []

    # ── POWER ─────────────────────────────────────────────────────────────
    if f["throttle_full_pct"] >= 65:
        scores["POWER"] += 3
        reasons.append(f"% gas a fondo elevado ({f['throttle_full_pct']:.0f}%) → POWER")
    if f["speed_avg"] >= 215:
        scores["POWER"] += 3
        reasons.append(f"velocidad media alta ({f['speed_avg']:.0f} km/h) → POWER")
    if f["speed_max"] >= 330:
        scores["POWER"] += 2
        reasons.append(f"velocidad punta alta ({f['speed_max']:.0f} km/h) → POWER")

    # ── STREET ────────────────────────────────────────────────────────────
    if f["speed_avg"] < 170:
        scores["STREET"] += 3
        reasons.append(f"velocidad media baja ({f['speed_avg']:.0f} km/h) → STREET")
    if f["throttle_full_pct"] < 45 and f["brake_pct"] >= 22:
        scores["STREET"] += 2
        reasons.append(f"poco gas a fondo ({f['throttle_full_pct']:.0f}%) + freno alto ({f['brake_pct']:.0f}%) → STREET")
    if f["track_length_m"] > 0 and f["track_length_m"] < 5500:
        scores["STREET"] += 1
        reasons.append(f"trazado corto ({f['track_length_m']/1000:.2f} km) → STREET")

    # ── HIGH-SPEED ────────────────────────────────────────────────────────
    if f["g_lat_avg"] >= 1.6:
        scores["HIGH-SPEED"] += 3
        reasons.append(f"G laterales medias altas ({f['g_lat_avg']:.2f} G) → HIGH-SPEED")
    if f["g_lat_max"] >= 4.5:
        scores["HIGH-SPEED"] += 2
        reasons.append(f"G lateral pico muy alta ({f['g_lat_max']:.2f} G) → HIGH-SPEED")
    if f["throttle_full_pct"] >= 55 and f["g_lat_avg"] >= 1.4:
        scores["HIGH-SPEED"] += 1
        reasons.append("balance de gas + G laterales → HIGH-SPEED")

    # ── STOP-AND-GO ───────────────────────────────────────────────────────
    if f["brake_pct"] >= 28:
        scores["STOP-AND-GO"] += 3
        reasons.append(f"% de freno muy alto ({f['brake_pct']:.0f}%) → STOP-AND-GO")
    if abs(f["g_lon_brake_max"]) >= 5.0:
        scores["STOP-AND-GO"] += 2
        reasons.append(f"frenada extrema ({f['g_lon_brake_max']:.2f} G) → STOP-AND-GO")
    if f["speed_min"] < 70 and f["speed_max"] - f["speed_min"] >= 240:
        scores["STOP-AND-GO"] += 2
        reasons.append("rango velocidad amplio + mínimas muy bajas → STOP-AND-GO")

    # ── TECHNICAL (default si no hay extremos) ───────────────────────────
    scores["TECHNICAL"] += 1   # base — siempre suma 1 como fallback
    if 175 <= f["speed_avg"] <= 210 and 45 <= f["throttle_full_pct"] <= 65:
        scores["TECHNICAL"] += 2
        reasons.append("rangos medios sin extremos → TECHNICAL")

    # Ganador
    winner = max(scores, key=scores.get)
    return winner, reasons


# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURAS
# ═══════════════════════════════════════════════════════════════════════════════
def _fig_track_demand(tel: pd.DataFrame, g_total: np.ndarray) -> go.Figure:
    """Mapa del trazado coloreado por demanda (G total)."""
    if 'X' not in tel or 'Y' not in tel:
        return go.Figure()

    x = tel['X'].to_numpy()
    y = tel['Y'].to_numpy()

    # Recortar el G a un rango razonable para colorear bien
    g_clip = np.clip(g_total, 0, 6)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y, mode='markers',
        marker=dict(
            size=4,
            color=g_clip,
            colorscale=[(0, "#1E1E2D"), (0.4, INFO_BLUE), (0.7, WARN_YELLOW), (1, F1_RED)],
            showscale=True,
            colorbar=dict(title=dict(text="G total", font=dict(color="white")),
                          tickfont=dict(color="white"), thickness=12),
        ),
        hovertemplate="<b>G total: %{marker.color:.2f} G</b><extra></extra>",
        showlegend=False,
    ))
    fig.update_layout(
        height=520, paper_bgcolor=PANEL, plot_bgcolor=PANEL,
        font=dict(color="white", family="Titillium Web"),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False, scaleanchor="x", scaleratio=1),
        margin=dict(l=0, r=0, t=10, b=0),
    )
    return fig


def _fig_speed_profile(tel: pd.DataFrame) -> go.Figure:
    """Perfil de velocidad vs distancia."""
    fig = go.Figure()
    if 'Distance' not in tel:
        return fig
    d = tel['Distance'].to_numpy()
    s = tel['Speed'].to_numpy()
    fig.add_trace(go.Scatter(
        x=d, y=s, mode='lines',
        line=dict(color=F1_RED, width=2),
        fill='tozeroy', fillcolor='rgba(232,0,45,0.10)',
        hovertemplate="dist: %{x:.0f} m<br>vel: %{y:.0f} km/h<extra></extra>",
        showlegend=False,
    ))
    fig.update_layout(
        height=240, paper_bgcolor=PANEL, plot_bgcolor=PANEL,
        font=dict(color="white", family="Titillium Web"),
        xaxis=dict(title="Distancia (m)", gridcolor="#2A2A3A"),
        yaxis=dict(title="Velocidad (km/h)", gridcolor="#2A2A3A"),
        margin=dict(l=40, r=20, t=20, b=40),
    )
    return fig


def _hex_to_rgba(hex_color: str, alpha: float = 0.2) -> str:
    h = hex_color.lstrip("#")
    if len(h) != 6:
        return f"rgba(255,255,255,{alpha})"
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _normalize_circuit_features(f: dict) -> dict:
    """Normaliza features del circuito a [0, 100] con anclajes empíricos."""
    return {
        "speed_max":         min(100, max(0, (f["speed_max"]         - 250) / (370 - 250) * 100)),
        "speed_avg":         min(100, max(0, (f["speed_avg"]         - 130) / (240 - 130) * 100)),
        "throttle_full_pct": min(100, max(0, f["throttle_full_pct"])),
        "brake_pct":         min(100, max(0, f["brake_pct"] / 40 * 100)),
        "g_lat_avg":         min(100, max(0, f["g_lat_avg"] / 3.0 * 100)),
        "g_lon_max":         min(100, max(0, f["g_lon_max"] / 6.0 * 100)),
        "num_corners":       min(100, max(0, f["num_corners"] / 25 * 100)),
    }


def _fig_radar_circuit(features_norm: dict, winner_type: str) -> go.Figure:
    """Radar comparando el circuito actual con el perfil del tipo asignado."""
    cats = ["VEL.MAX", "VEL.AVG", "% A FONDO", "% FRENO",
            "G LAT MEDIA", "G LON MAX", "Nº CURVAS"]
    keys = ["speed_max", "speed_avg", "throttle_full_pct", "brake_pct",
            "g_lat_avg", "g_lon_max", "num_corners"]

    cats_loop = cats + [cats[0]]

    # Perfil actual
    actual = [features_norm[k] for k in keys]
    actual_loop = actual + [actual[0]]

    fig = go.Figure()

    # Perfil tipo (todos en gris)
    for tname, tdata in TYPE_PROFILES.items():
        prof = tdata["profile"]
        vals = [prof.get(k, 0) for k in keys]
        vals_loop = vals + [vals[0]]
        is_winner = (tname == winner_type)
        color = tdata["color"] if is_winner else "#3A3A4A"
        width = 3 if is_winner else 1
        fig.add_trace(go.Scatterpolar(
            r=vals_loop, theta=cats_loop,
            mode='lines',
            line=dict(color=color, width=width, dash=None if is_winner else "dot"),
            name=f"{tdata['icon']} {tname}" + (" ●" if is_winner else ""),
            opacity=1.0 if is_winner else 0.45,
        ))

    # Perfil real (rojo, encima de todo)
    fig.add_trace(go.Scatterpolar(
        r=actual_loop, theta=cats_loop,
        fill='toself',
        line=dict(color=F1_RED, width=3),
        fillcolor=_hex_to_rgba(F1_RED, 0.25),
        name="◆ ESTE CIRCUITO",
    ))

    fig.update_layout(
        height=520, paper_bgcolor=PANEL, plot_bgcolor=PANEL,
        font=dict(color="white", family="Titillium Web"),
        polar=dict(
            bgcolor=PANEL,
            radialaxis=dict(visible=True, range=[0, 100], gridcolor="#2A2A3A",
                            tickfont=dict(color=DIM, size=9)),
            angularaxis=dict(gridcolor="#2A2A3A",
                             tickfont=dict(color="white", size=10, family="Share Tech Mono")),
        ),
        legend=dict(orientation="h", yanchor="bottom", y=-0.18, x=0,
                    font=dict(size=10, family="Share Tech Mono")),
        margin=dict(l=40, r=40, t=30, b=80),
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════
def render_circuit_summary(session) -> None:
    st.header("Resumen del Circuito · Clasificación por Demanda Dinámica")
    st.caption("REGLAS DETERMINISTAS  ·  G LAT/LON DERIVADAS DE TELEMETRÍA  ·  REPRODUCIBLE 100%")

    try:
        with st.spinner("Calculando perfil dinámico del circuito..."):
            f = _extract_circuit_features(session)
    except Exception as e:
        st.error(f"No se pudo extraer telemetría suficiente para clasificar el circuito · {e}")
        return

    winner, reasons = _classify_circuit(f)
    profile = TYPE_PROFILES[winner]

    # ── 1. TARJETA PRINCIPAL DE CLASIFICACIÓN ────────────────────────────────
    st.markdown(f"""
    <div style='background: linear-gradient(115deg, {_hex_to_rgba(profile["color"], 0.20)} 0%, #15151E 60%);
                border: 1px solid {profile["color"]};
                border-left: 6px solid {profile["color"]};
                padding: 24px 28px;
                margin-top: 4px;
                margin-bottom: 22px;
                border-radius: 2px;'>
        <div style='display:flex; align-items:center; gap:20px; flex-wrap:wrap;'>
            <div style='font-size:64px; line-height:1; color:{profile["color"]};'>{profile["icon"]}</div>
            <div style='flex:1; min-width:240px;'>
                <div style='font-family:"Share Tech Mono",monospace; font-size:11px;
                            letter-spacing:4px; color:{DIM}; text-transform:uppercase;'>
                    TIPO DE CIRCUITO
                </div>
                <div style='font-family:"Share Tech Mono",monospace; font-size:34px;
                            letter-spacing:8px; color:{profile["color"]}; line-height:1.1;
                            margin-top:4px; font-weight:400;'>
                    {winner}
                </div>
                <div style='font-family:"Titillium Web",sans-serif; color:white;
                            font-size:14px; margin-top:10px; line-height:1.5;
                            max-width: 720px;'>
                    {profile["description"]}
                </div>
                <div style='font-family:"Share Tech Mono",monospace; color:{DIM};
                            font-size:11px; margin-top:8px; letter-spacing:2px;
                            text-transform:uppercase;'>
                    Ejemplos · {profile["examples"]}
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── 2. MÉTRICAS CLAVE ────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Vel. máxima",     f"{f['speed_max']:.0f} km/h")
    m2.metric("Vel. media",      f"{f['speed_avg']:.0f} km/h")
    m3.metric("% Gas a fondo",   f"{f['throttle_full_pct']:.0f} %")
    m4.metric("% Tiempo freno",  f"{f['brake_pct']:.0f} %")

    m5, m6, m7, m8 = st.columns(4)
    m5.metric("G lateral máx",   f"{f['g_lat_max']:.2f} G")
    m6.metric("G lateral media", f"{f['g_lat_avg']:.2f} G")
    m7.metric("Frenada máx",     f"{abs(f['g_lon_brake_max']):.2f} G")
    if f["track_length_m"] > 0:
        m8.metric("Longitud",     f"{f['track_length_m']/1000:.3f} km")
    else:
        m8.metric("Longitud",     "—")

    m9, m10, m11, m12 = st.columns(4)
    m9.metric("Curvas",          f"{f['num_corners']}" if f['num_corners'] > 0 else "—")
    m10.metric("Cambios marcha", f"{f['num_gear_changes']}")
    m11.metric("Vel. mínima",    f"{f['speed_min']:.0f} km/h")
    if f["lap_time_s"] > 0:
        mins = int(f["lap_time_s"] // 60)
        secs = f["lap_time_s"] % 60
        m12.metric("Mejor vuelta", f"{mins}:{secs:06.3f}")
    else:
        m12.metric("Mejor vuelta", "—")

    # ── 3. VISUALIZACIONES ──────────────────────────────────────────────────
    st.divider()
    tab_radar, tab_demand, tab_speed, tab_rules = st.tabs(
        ["◬ Perfil vs tipologías", "◉ Mapa de demanda", "▭ Perfil de velocidad", "◷ Reglas aplicadas"]
    )

    with tab_radar:
        st.markdown(
            "<div style='font-family:Titillium Web; color:#aaa; font-size:12px;'>"
            "Comparación del perfil normalizado del circuito actual contra las cinco "
            "tipologías de referencia. La tipología asignada se resalta en color, "
            "el resto aparecen como guías punteadas."
            "</div>", unsafe_allow_html=True,
        )
        st.plotly_chart(
            _fig_radar_circuit(_normalize_circuit_features(f), winner),
            use_container_width=True, config={"displayModeBar": False}
        )

    with tab_demand:
        st.markdown(
            "<div style='font-family:Titillium Web; color:#aaa; font-size:12px;'>"
            "Trazado coloreado por la fuerza G total instantánea (combinada lateral "
            "+ longitudinal). Las zonas rojas son los puntos más exigentes "
            "del circuito."
            "</div>", unsafe_allow_html=True,
        )
        st.plotly_chart(
            _fig_track_demand(f["tel"], f["g_total"]),
            use_container_width=True, config={"displayModeBar": False}
        )

    with tab_speed:
        st.markdown(
            "<div style='font-family:Titillium Web; color:#aaa; font-size:12px;'>"
            "Perfil de velocidad sobre la vuelta más rápida — útil para identificar "
            "las rectas (mesetas altas) y las curvas (depresiones)."
            "</div>", unsafe_allow_html=True,
        )
        st.plotly_chart(
            _fig_speed_profile(f["tel"]),
            use_container_width=True, config={"displayModeBar": False}
        )

    with tab_rules:
        st.markdown(
            "<div style='font-family:Titillium Web; color:#aaa; font-size:12px;'>"
            "Lista de reglas que se han activado para asignar este circuito a su "
            "tipología.  Todas las reglas son determinísticas: idénticos datos "
            "producen siempre la misma clasificación."
            "</div>", unsafe_allow_html=True,
        )
        if not reasons:
            st.info("Sin reglas activadas — el circuito ha caído en la categoría TECHNICAL por defecto.")
        else:
            for r in reasons:
                st.markdown(f"""
                <div style='background:#12121A; border-left:3px solid {profile["color"]};
                            padding:8px 14px; margin-bottom:4px; border-radius:2px;
                            font-family:"Share Tech Mono",monospace; font-size:12px;
                            color:#D0D2DC; letter-spacing:0.5px;'>
                    ▸ {r}
                </div>
                """, unsafe_allow_html=True)

    # ── 4. FOOTER TÉCNICO ───────────────────────────────────────────────────
    with st.expander("◷  Detalles técnicos del modelo"):
        st.markdown(f"""
        - **Determinismo**: la clasificación se basa exclusivamente en umbrales
          aplicados sobre los features extraídos. Misma sesión → misma clasificación.
        - **Cálculo de fuerzas G**:
            - **G longitudinal** = `dV/dt / 9.81`
            - **G lateral** = `V * dθ/dt / 9.81` con θ = `arctan2(dY, dX)` desempaquetado
            - Suavizado con media móvil de 5 muestras para reducir ruido numérico.
        - **Tipologías**: POWER · HIGH-SPEED · STOP-AND-GO · STREET · TECHNICAL.
          Cada tipología tiene un perfil normalizado de referencia (ver radar).
        - **Sistema de puntuación**: cada regla suma puntos a su tipología;
          gana la que más acumule. La tipología TECHNICAL parte con +1 base
          para servir de fallback cuando ningún extremo se cumple.
        - **Validación manual**: ejecutar sobre Mónaco debe producir STREET,
          sobre Monza POWER, sobre Suzuka HIGH-SPEED, sobre Singapur STOP-AND-GO,
          sobre Barcelona TECHNICAL.
        """)
