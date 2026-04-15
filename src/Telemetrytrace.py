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
    "SOFT"    : "#E8002D",
    "MEDIUM"  : "#FFF200",
    "HARD"    : "#EBEBEB",
    "INTER"   : "#43B02A",
    "WET"     : "#0067FF",
    "UNKNOWN" : "#888888",
}

TYRE_ICONS = {
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
        # Neumático con color de compuesto
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
                ">{tyre_icon} NEUMÁTICO</p>
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