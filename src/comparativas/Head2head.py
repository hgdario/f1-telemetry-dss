"""
LapOverlay.py — TALOS F1 Superposición de Vueltas H2H

"""


from __future__ import annotations
import ui_assets

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.ndimage import uniform_filter1d
from typing import Optional
from fastf1.core import Session

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────────────────────────────────────────

BG_DARK    = "#0E0E0F"
BG_MAP     = "#13131A"
BG_PANEL   = "#111115"
BG_SURFACE = "#1A1A2E"
F1_WHITE   = "#FFFFFF"
F1_RED     = "#E8002D"
GREY_ZONE  = "rgba(100,100,120,0.5)"   # zona en disputa
MONO_FONT  = "'JetBrains Mono', 'Courier New', monospace"

ACCENT_CYAN   = "#00D2FF"
ACCENT_GREEN  = "#39FF14"
ACCENT_AMBER  = "#FFA500"
ACCENT_PURPLE = "#C77DFF"

# Puntos de la rejilla de interpolación
GRID_N = 2500

# Ventana de suavizado para asignación de zona (en nº de puntos de rejilla)
SMOOTH_WIN = 9

# Velocidad mínima para considerar un punto "en disputa" (gap < umbral)
DISPUTE_THRESHOLD_KMH = 2.0


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_time(seconds: float) -> str:
    if np.isnan(seconds) or seconds <= 0:
        return "—"
    m  = int(seconds // 60)
    s  = int(seconds % 60)
    ms = int(round((seconds % 1) * 1000))
    return f"{m}:{s:02d}.{ms:03d}"


def _fmt_delta(delta_s: float) -> str:
    sign = "+" if delta_s > 0 else ""
    return f"{sign}{delta_s:.3f}s"


def _get_team_color(sesion: Session, driver: str) -> str:
    try:
        return f"#{sesion.get_driver(driver)['TeamColor']}"
    except Exception:
        return "#888888"


def _get_full_name(sesion: Session, driver: str) -> str:
    try:
        info = sesion.get_driver(driver)
        return f"{info.get('FirstName','')} {info.get('LastName', driver)}".strip()
    except Exception:
        return driver


def _normalize_brake(b: np.ndarray) -> np.ndarray:
    if b.dtype == bool or set(np.unique(b[~np.isnan(b)])).issubset({0, 1}):
        return b.astype(float) * 100
    return b * 100 if b.max() <= 1.0 else b.copy()


def _exterior_coords(x, y, mid_x, mid_y, idx, offset):
    i_n = min(idx+5, len(x)-1); i_p = max(idx-5, 0)
    dx, dy = x[i_n]-x[i_p], y[i_n]-y[i_p]
    nx, ny = -dy, dx
    mag = np.sqrt(nx**2+ny**2) or 1.0
    nx /= mag; ny /= mag
    if (np.sqrt((x[idx]+nx*offset-mid_x)**2+(y[idx]+ny*offset-mid_y)**2) <
            np.sqrt((x[idx]-mid_x)**2+(y[idx]-mid_y)**2)):
        nx, ny = -nx, -ny
    return float(x[idx]+nx*offset), float(y[idx]+ny*offset)


# ─────────────────────────────────────────────────────────────────────────────
# EXTRACCIÓN E INTERPOLACIÓN
# ─────────────────────────────────────────────────────────────────────────────

def _extract_lap(sesion: Session, driver: str, lap_number: int) -> Optional[pd.DataFrame]:
    try:
        laps = sesion.laps.pick_drivers(driver)
        lap  = laps[laps["LapNumber"] == lap_number].iloc[0]
        tel  = lap.get_telemetry().add_distance()
        return tel if not tel.empty and "X" in tel.columns else None
    except Exception:
        return None


def _interpolate(tel: pd.DataFrame, grid: np.ndarray) -> dict[str, np.ndarray]:
    d = tel["Distance"].values.astype(float)
    t = tel["Time"].dt.total_seconds().values.astype(float)
    return {
        "x"       : np.interp(grid, d, tel["X"].values.astype(float)),
        "y"       : np.interp(grid, d, tel["Y"].values.astype(float)),
        "speed"   : np.interp(grid, d, tel["Speed"].values.astype(float)),
        "throttle": np.interp(grid, d, tel["Throttle"].values.astype(float)),
        "brake"   : np.interp(grid, d, _normalize_brake(tel["Brake"].values).astype(float)),
        "rpm"     : np.interp(grid, d, tel["RPM"].values.astype(float)),
        "gear"    : np.round(np.interp(grid, d, tel["nGear"].values.astype(float))).astype(int),
        "drs"     : np.interp(grid, d, tel["DRS"].values.astype(float)),
        "time_s"  : np.interp(grid, d, t),
    }


# ─────────────────────────────────────────────────────────────────────────────
# ZONA DE DOMINIO: quién va más rápido en cada punto del circuito
# ─────────────────────────────────────────────────────────────────────────────

def _compute_dominance(
    speed_a: np.ndarray,
    speed_b: np.ndarray,
) -> np.ndarray:
    """
    Retorna un array de etiquetas: 1=A gana, -1=B gana, 0=disputa.

    Proceso:
      1. Diferencia de velocidad puntual: gap[i] = speed_a[i] - speed_b[i]
      2. Suavizado por ventana deslizante (uniform_filter1d) para evitar
         parpadeo en puntos individuales donde el muestreo crea ruido.
      3. Si |gap_suavizado| < DISPUTE_THRESHOLD → zona en disputa (0).
    """
    gap        = speed_a - speed_b
    gap_smooth = uniform_filter1d(gap, size=SMOOTH_WIN)

    dominance  = np.where(
        np.abs(gap_smooth) < DISPUTE_THRESHOLD_KMH, 0,
        np.where(gap_smooth > 0, 1, -1)
    )
    return dominance


# ─────────────────────────────────────────────────────────────────────────────
# MAPA H2H COLOREADO POR DOMINIO
# ─────────────────────────────────────────────────────────────────────────────

def _build_h2h_map(
    da: dict, db: dict,
    color_a: str, color_b: str,
    label_a: str, label_b: str,
    dominance: np.ndarray,
    dist_grid: np.ndarray,
    circuit_info,
    show_corners: bool,
) -> go.Figure:
    """
    Mapa del circuito dividido en segmentos continuos por zona de dominio.

    Técnica de segmentación
    ────────────────────────
    En lugar de un trace por punto (lento), agrupamos puntos consecutivos
    del mismo dominador en segmentos. Cada vez que cambia el dominador se
    abre un nuevo trace. Esto minimiza el número de traces (típicamente 15-30)
    manteniendo la misma calidad visual.
    """
    x = da["x"]; y = da["y"]
    mid_x, mid_y = float(np.mean(x)), float(np.mean(y))
    range_xy = max(x.max()-x.min(), y.max()-y.min()) / 1.24

    fig = go.Figure()

    # ── Capa base ─────────────────────────────────────────────────────────
    fig.add_trace(go.Scattergl(
        x=x, y=y, mode="lines",
        line=dict(color="rgba(80,80,106,0.30)", width=10),
        hoverinfo="skip", showlegend=False, name="_base",
    ))

    # ── Segmentos por dominio ─────────────────────────────────────────────
    color_map = {1: color_a, -1: color_b, 0: GREY_ZONE}
    name_map  = {1: label_a, -1: label_b, 0: "Disputa"}

    # Detectar cambios de dominador
    changes = np.where(np.diff(dominance, prepend=dominance[0]))[0]
    seg_starts = np.concatenate([[0], changes])
    seg_ends   = np.concatenate([changes, [len(dominance)]])

    already_in_legend = set()
    for s, e in zip(seg_starts, seg_ends):
        dom    = int(dominance[s])
        color  = color_map[dom]
        name   = name_map[dom]
        in_leg = name not in already_in_legend

        # +1 punto de overlap para que no haya gap visual entre segmentos
        e_ext  = min(e + 1, len(x))

        # Hover: velocidades en el punto central del segmento
        mid_i  = (s + e) // 2
        v_a    = float(da["speed"][mid_i])
        v_b    = float(db["speed"][mid_i])
        hover  = (
            f"<b>{name}</b><br>"
            f"{label_a}: {v_a:.1f} km/h<br>"
            f"{label_b}: {v_b:.1f} km/h<br>"
            f"Gap: {abs(v_a-v_b):.1f} km/h<br>"
            f"Dist: {dist_grid[s]:.0f}–{dist_grid[min(e,len(dist_grid)-1)]:.0f} m"
        )

        fig.add_trace(go.Scattergl(
            x=x[s:e_ext], y=y[s:e_ext],
            mode="lines",
            line=dict(color=color, width=6),
            name=name,
            legendgroup=name,
            showlegend=in_leg,
            hovertemplate=hover + "<extra></extra>",
        ))
        already_in_legend.add(name)

    # ── Marcadores de inicio/fin de vuelta ────────────────────────────────
    dx_m = float(x[1]-x[0]); dy_m = float(y[1]-y[0])
    mag  = np.sqrt(dx_m**2+dy_m**2) or 1.0
    nmx, nmy = -dy_m/mag, dx_m/mag
    meta_w   = 200.0
    fig.add_trace(go.Scatter(
        x=[x[0]-nmx*meta_w, x[0]+nmx*meta_w],
        y=[y[0]-nmy*meta_w, y[0]+nmy*meta_w],
        mode="lines", line=dict(color=F1_WHITE, width=3),
        hoverinfo="skip", showlegend=False, name="_meta",
    ))

    # ── Etiquetas de curva ────────────────────────────────────────────────
    if show_corners and circuit_info is not None:
        dist_max = float(dist_grid[-1])
        for _, row in circuit_info.corners.iterrows():
            d_c   = float(row["Distance"])
            if d_c > dist_max:
                continue
            idx_c = int(np.argmin(np.abs(dist_grid - d_c)))
            cx, cy = _exterior_coords(x, y, mid_x, mid_y, idx_c, 240)
            fig.add_trace(go.Scatter(
                x=[cx], y=[cy], mode="markers+text",
                marker=dict(size=18, color=F1_WHITE, symbol="circle",
                            line=dict(color=F1_RED, width=1.5)),
                text=[str(int(row["Number"]))],
                textfont=dict(size=9, color="#0E0E0F", family=MONO_FONT),
                textposition="middle center",
                hoverinfo="skip", showlegend=False,
            ))

    fig.update_layout(
        height=580,
        paper_bgcolor=BG_DARK, plot_bgcolor=BG_MAP,
        font=dict(family=MONO_FONT, color=F1_WHITE, size=11),
        showlegend=True,
        legend=dict(
            orientation="h", y=1.03, x=0.5, xanchor="center",
            bgcolor="rgba(14,14,15,0.7)",
            bordercolor="rgba(255,255,255,0.1)", borderwidth=1,
            font=dict(size=11),
        ),
        margin=dict(l=10, r=10, t=40, b=10),
        hoverlabel=dict(bgcolor=BG_SURFACE, bordercolor="rgba(255,255,255,0.15)",
                        font=dict(family=MONO_FONT, color=F1_WHITE, size=11)),
        xaxis=dict(range=[mid_x-range_xy, mid_x+range_xy],
                   scaleanchor="y", scaleratio=1,
                   showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(range=[mid_y-range_xy*1.4, mid_y+range_xy*0.7],
                   showgrid=False, zeroline=False, showticklabels=False),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# TELEMETRÍA SUPERPUESTA — 4 canales
# ─────────────────────────────────────────────────────────────────────────────

def _build_telemetry_overlay(
    da: dict, db: dict,
    dist_grid: np.ndarray,
    color_a: str, color_b: str,
    label_a: str, label_b: str,
    circuit_info,
) -> go.Figure:
    """
    5 subplots verticales compartiendo eje X de distancia:
      Speed · Throttle · Brake · Gear · RPM
    Ambas vueltas superpuestas en cada panel.
    """
    # Detectar curvas para vlines
    corner_dists = []
    if circuit_info is not None:
        dist_max = float(dist_grid[-1])
        for _, row in circuit_info.corners.iterrows():
            if float(row["Distance"]) <= dist_max:
                corner_dists.append(float(row["Distance"]))

    fig = make_subplots(
        rows=5, cols=1, shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=["SPEED (km/h)", "THROTTLE (%)", "BRAKE (%)", "GEAR", "RPM"],
        row_heights=[0.30, 0.20, 0.15, 0.15, 0.20],
    )

    def add_pair(channel, row, suffix="", dash_b="dot"):
        y_a = da[channel]; y_b = db[channel]
        fig.add_trace(go.Scatter(
            x=dist_grid, y=y_a,
            mode="lines", line=dict(color=color_a, width=1.8),
            name=label_a, legendgroup=label_a,
            showlegend=(row == 1),
            hovertemplate=f"<b>{label_a}</b>  %{{y:.1f}}{suffix}<extra></extra>",
        ), row=row, col=1)
        fig.add_trace(go.Scatter(
            x=dist_grid, y=y_b,
            mode="lines", line=dict(color=color_b, width=1.8, dash=dash_b),
            name=label_b, legendgroup=label_b,
            showlegend=(row == 1),
            hovertemplate=f"<b>{label_b}</b>  %{{y:.1f}}{suffix}<extra></extra>",
        ), row=row, col=1)

    add_pair("speed",    1, " km/h")
    add_pair("throttle", 2, "%")
    add_pair("brake",    3, "%")
    add_pair("gear",     4)
    add_pair("rpm",      5, " rpm")

    # Vlines de curvas
    for d_c in corner_dists:
        for r in range(1, 6):
            fig.add_vline(x=d_c, line_width=0.6, line_dash="dash",
                          line_color="rgba(255,255,255,0.1)", row=r, col=1)

    # Etiquetas de curva encima del primer subplot
    if circuit_info is not None:
        for _, row in circuit_info.corners.iterrows():
            d_c = float(row["Distance"])
            if d_c > float(dist_grid[-1]):
                continue
            fig.add_annotation(
                x=d_c, y=1.02, xref="x", yref="paper",
                text=f"T{int(row['Number'])}",
                showarrow=False,
                font=dict(size=8, color="rgba(255,255,255,0.35)", family=MONO_FONT),
            )

    ax_common = dict(
        showgrid=True, gridcolor="rgba(255,255,255,0.05)",
        zeroline=False,
        tickfont=dict(size=8, color="rgba(255,255,255,0.4)", family=MONO_FONT),
    )

    fig.update_layout(
        height=700,
        paper_bgcolor=BG_DARK, plot_bgcolor=BG_PANEL,
        font=dict(family=MONO_FONT, color=F1_WHITE, size=10),
        hovermode="x unified",
        legend=dict(orientation="h", y=1.04, x=0.5, xanchor="center",
                    bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
        margin=dict(l=54, r=20, t=50, b=40),
        hoverlabel=dict(bgcolor=BG_SURFACE, font=dict(family=MONO_FONT, size=10)),
        xaxis5=dict(**ax_common, title="Distancia (m)", ticksuffix=" m"),
    )
    for i in range(1, 6):
        fig.update_layout(**{f"xaxis{i if i>1 else ''}": ax_common})
        fig.update_layout(**{f"yaxis{i if i>1 else ''}": ax_common})

    for ann in fig.layout.annotations:
        ann.update(font=dict(family=MONO_FONT, size=9,
                             color="rgba(255,255,255,0.40)"),
                   x=0, xanchor="left")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# DELTA ACUMULADO + MINI-SECTORES H2H
# ─────────────────────────────────────────────────────────────────────────────

def _build_delta_and_dominance(
    da: dict, db: dict,
    dist_grid: np.ndarray,
    dominance: np.ndarray,
    color_a: str, color_b: str,
    label_a: str, label_b: str,
    circuit_info,
) -> go.Figure:
    """
    Gráfico combinado de 2 filas:
      Row 1: Delta acumulado (A − B) con relleno por ganador
      Row 2: Barra de dominio visual (como el indicador de MotoGP TV)
    """
    delta = da["time_s"] - db["time_s"]

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.75, 0.25],
        vertical_spacing=0.05,
        subplot_titles=["DELTA ACUMULADO (A − B)", "DOMINIO POR ZONA"],
    )

    # ── Delta relleno por colores ─────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=dist_grid, y=np.where(delta >= 0, delta, 0),
        fill="tozeroy", mode="lines",
        line=dict(color=color_b, width=0),
        fillcolor=f"rgba({ui_assets.hex_rgb(color_b)},0.25)",
        hoverinfo="skip", showlegend=False,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=dist_grid, y=np.where(delta <= 0, delta, 0),
        fill="tozeroy", mode="lines",
        line=dict(color=color_a, width=0),
        fillcolor=f"rgba({ui_assets.hex_rgb(color_a)},0.25)",
        hoverinfo="skip", showlegend=False,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=dist_grid, y=delta,
        mode="lines", line=dict(color=F1_WHITE, width=1.5),
        hovertemplate="Dist %{x:.0f}m<br>Δ %{y:.3f}s<extra></extra>",
        showlegend=False, name="delta",
    ), row=1, col=1)
    fig.add_hline(y=0, line_color="rgba(255,255,255,0.2)", line_width=1, row=1, col=1)

    # Anotación final
    final_d = float(delta[-1])
    winner  = label_a if final_d < 0 else label_b
    w_col   = color_a if final_d < 0 else color_b
    fig.add_annotation(
        x=float(dist_grid[-1]), y=final_d,
        text=f"<b>{winner}  {_fmt_delta(final_d if final_d < 0 else -final_d)}</b>",
        showarrow=True, arrowhead=2, arrowcolor=w_col,
        font=dict(color=w_col, size=10, family=MONO_FONT),
        bgcolor=BG_SURFACE, bordercolor=w_col, borderwidth=1,
        ax=-60, ay=-30, row=1, col=1,
    )

    # ── Barra de dominio (indicator bar estilo TV) ─────────────────────────
    # Mapeamos dominance a valores: 1=A, -1=B, 0=disputa
    dom_vals = dominance.astype(float)
    color_seq = [
        color_a if v > 0 else (color_b if v < 0 else GREY_ZONE)
        for v in dom_vals
    ]
    fig.add_trace(go.Scatter(
        x=dist_grid, y=np.ones(len(dist_grid)),
        mode="markers",
        marker=dict(size=8, color=color_seq, symbol="square"),
        hoverinfo="skip", showlegend=False,
    ), row=2, col=1)

    ax = dict(showgrid=False, zeroline=False,
              tickfont=dict(size=8, color="rgba(255,255,255,0.4)", family=MONO_FONT))
    fig.update_layout(
        height=300,
        paper_bgcolor=BG_DARK, plot_bgcolor=BG_PANEL,
        font=dict(family=MONO_FONT, color=F1_WHITE, size=10),
        margin=dict(l=54, r=20, t=30, b=40),
        showlegend=False, hovermode="x",
        xaxis2=dict(**ax, title="Distancia (m)", ticksuffix=" m"),
        yaxis={**ax, "zeroline": True, "zerolinecolor": "rgba(255,255,255,0.2)", "ticksuffix": "s"},
        yaxis2=dict(showgrid=False, zeroline=False, showticklabels=False),
    )
    for ann in fig.layout.annotations:
        ann.update(font=dict(family=MONO_FONT, size=9, color="rgba(255,255,255,0.40)"),
                   x=0, xanchor="left")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# KPI TABLE
# ─────────────────────────────────────────────────────────────────────────────

def _render_h2h_kpis(
    da: dict, db: dict,
    label_a: str, label_b: str,
    color_a: str, color_b: str,
    lap_time_a: float, lap_time_b: float,
    dominance: np.ndarray,
    dist_grid: np.ndarray,
) -> None:
    delta_final = lap_time_a - lap_time_b
    winner      = label_a if delta_final < 0 else label_b
    w_color     = color_a if delta_final < 0 else color_b

    # Distancia dominada por cada uno
    n = len(dominance)
    pct_a  = (dominance > 0).sum() / n * 100
    pct_b  = (dominance < 0).sum() / n * 100
    pct_d  = (dominance == 0).sum() / n * 100

    # Diferencia de velocidad punta
    vmax_a = float(da["speed"].max())
    vmax_b = float(db["speed"].max())

    st.markdown(
        f"""
        <div style="
            background:{BG_SURFACE};
            border-left: 4px solid {w_color};
            padding: 14px 20px;
            border-radius: 3px;
            margin-bottom: 16px;
            font-family:{MONO_FONT};
        ">
            <span style="font-size:11px;color:rgba(255,255,255,0.4);letter-spacing:2px;">
                GANADOR H2H
            </span><br>
            <span style="font-size:22px;font-weight:800;color:{w_color};">
                {winner}
            </span>
            <span style="font-size:14px;color:rgba(255,255,255,0.6);margin-left:12px;">
                {_fmt_delta(delta_final)}
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c = st.columns(6)
    c[0].metric(f"⏱ {label_a}", _fmt_time(lap_time_a))
    c[1].metric(f"⏱ {label_b}", _fmt_time(lap_time_b))
    c[2].metric(f"Dominio {label_a}", f"{pct_a:.0f}% circuito")
    c[3].metric(f"Dominio {label_b}", f"{pct_b:.0f}% circuito")
    c[4].metric("V Punta",
                f"{label_a} {vmax_a:.0f}" if vmax_a >= vmax_b else f"{label_b} {vmax_b:.0f}",
                f"+{abs(vmax_a-vmax_b):.1f} km/h")
    c[5].metric("Zona disputa", f"{pct_d:.0f}% circuito")


# ─────────────────────────────────────────────────────────────────────────────
# SELECTOR DE PILOTO/VUELTA
# ─────────────────────────────────────────────────────────────────────────────

def _lap_selector(sesion: Session, prefix: str, label: str,
                  avoid_driver: Optional[str] = None) -> tuple[str, int, float]:
    drivers_list  = list(sesion.drivers)
    driver_names  = {d: sesion.get_driver(d).get("FullName", d) for d in drivers_list}

    if avoid_driver and avoid_driver in drivers_list:
        default_idx = next((i for i, d in enumerate(drivers_list) if d != avoid_driver), 0)
    else:
        default_idx = 0

    driver = st.selectbox(label, drivers_list, index=default_idx,
                          format_func=lambda x: driver_names.get(x, x),
                          key=f"{prefix}_drv")
    try:
        laps = sesion.laps.pick_drivers(driver).pick_accurate()
        fastest_n = laps.pick_fastest()["LapNumber"]
        lap_list  = sorted(laps["LapNumber"].tolist())
        def_idx   = lap_list.index(fastest_n) if fastest_n in lap_list else 0
        lap_n = st.selectbox(
            "Vuelta", lap_list, index=def_idx,
            format_func=lambda x: f"V{int(x)} ⏱ Best" if x == fastest_n else f"V{int(x)}",
            key=f"{prefix}_lap",
        )
        lap_time = float(
            laps[laps["LapNumber"] == lap_n].iloc[0]["LapTime"].total_seconds()
        )
    except Exception:
        lap_n    = st.number_input("Vuelta", 1, 80, 1, key=f"{prefix}_lap_num")
        lap_time = 0.0

    return str(driver), int(lap_n), lap_time


# ─────────────────────────────────────────────────────────────────────────────
# PUNTO DE ENTRADA PÚBLICO
# ─────────────────────────────────────────────────────────────────────────────

def render_lap_overlay(sesion: Session, corners: bool = True) -> None:

    st.markdown(
        "<p style='font-family:JetBrains Mono,monospace;font-size:11px;"
        "letter-spacing:2px;color:rgba(255,255,255,0.35);margin-bottom:8px;'>"
        "SUPERPOSICIÓN H2H · MAPA DE DOMINIO POR ZONA</p>",
        unsafe_allow_html=True,
    )

    # ── Selectores ────────────────────────────────────────────────────────
    col_a, col_sep, col_b = st.columns([5, 1, 5])
    with col_a:
        st.markdown(
            "<p style='font-family:JetBrains Mono,monospace;font-size:10px;"
            "letter-spacing:2px;color:rgba(255,255,255,0.45);'>PILOTO A</p>",
            unsafe_allow_html=True,
        )
        driver_a, lap_a, lap_time_a = _lap_selector(sesion, "lo_a", "Piloto A")
    with col_sep:
        st.markdown(
            "<div style='text-align:center;padding-top:32px;font-family:JetBrains Mono,"
            "monospace;font-size:18px;color:rgba(255,255,255,0.25);'>VS</div>",
            unsafe_allow_html=True,
        )
    with col_b:
        st.markdown(
            "<p style='font-family:JetBrains Mono,monospace;font-size:10px;"
            "letter-spacing:2px;color:rgba(255,255,255,0.45);'>PILOTO B</p>",
            unsafe_allow_html=True,
        )
        driver_b, lap_b, lap_time_b = _lap_selector(sesion, "lo_b", "Piloto B",
                                                     avoid_driver=driver_a)

    st.divider()

    # ── Carga de Telemetría ───────────────────────────────────────────────
    with st.spinner("Cargando telemetría H2H..."):
        tel_a = _extract_lap(sesion, driver_a, lap_a)
        tel_b = _extract_lap(sesion, driver_b, lap_b)

    if tel_a is None:
        st.error(f"❌ No se pudo cargar telemetría de {driver_a} · V{lap_a}")
        return
    if tel_b is None:
        st.error(f"❌ No se pudo cargar telemetría de {driver_b} · V{lap_b}")
        return

    # Interpolación espacial y dominio
    dist_max  = min(float(tel_a["Distance"].max()), float(tel_b["Distance"].max()))
    dist_grid = np.linspace(0, dist_max, GRID_N)
    da        = _interpolate(tel_a, dist_grid)
    db        = _interpolate(tel_b, dist_grid)
    dominance = _compute_dominance(da["speed"], db["speed"])

    # Colores 
    color_a = _get_team_color(sesion, driver_a)
    color_b = _get_team_color(sesion, driver_b)
    # Evitar colores idénticos si son del mismo equipo
    if color_a.lower() == color_b.lower():
        color_b = ACCENT_CYAN

    label_a = _get_full_name(sesion,driver_a) if driver_a != driver_b else f"{ _get_full_name(sesion,driver_a)}·V{lap_a}"
    label_b = _get_full_name(sesion,driver_b) if driver_a != driver_b else f"{_get_full_name(sesion,driver_b)}·V{lap_b}"

    try:
        circuit_info = sesion.get_circuit_info()
    except Exception:
        circuit_info = None

    # ── 1. KPIs cabecera ──────────────────────────────────────────────────
    _render_h2h_kpis(da, db, label_a, label_b, color_a, color_b,
                     lap_time_a, lap_time_b, dominance, dist_grid)

    st.divider()

    # ── 2. Mapa de Dominio ────────────────────────────────────────────────
    st.markdown(
        "<p style='font-family:JetBrains Mono,monospace;font-size:12px;"
        "letter-spacing:2px;color:rgba(255,255,255,0.7);'>🗺️ MAPA DE DOMINIO H2H</p>",
        unsafe_allow_html=True,
    )
    fig_map = _build_h2h_map(
        da, db, color_a, color_b, label_a, label_b,
        dominance, dist_grid, circuit_info, corners,
    )
    st.plotly_chart(fig_map, use_container_width=True, config={
        "displayModeBar": True, "displaylogo": False,
        "toImageButtonOptions": {"format": "png", "scale": 2,
                                 "filename": f"talos_h2h_{driver_a}_vs_{driver_b}"},
    })
    
    # ── 3. Telemetría Superpuesta ─────────────────────────────────────────
    st.markdown(
        "<p style='font-family:JetBrains Mono,monospace;font-size:12px;"
        "letter-spacing:2px;color:rgba(255,255,255,0.7);margin-top:20px;'>"
        "📊 TELEMETRÍA SUPERPUESTA · LÍNEA CONTINUA = A · PUNTEADA = B</p>",
        unsafe_allow_html=True,
    )
    fig_tel = _build_telemetry_overlay(
        da, db, dist_grid, color_a, color_b, label_a, label_b, circuit_info,
    )
    st.plotly_chart(fig_tel, use_container_width=True, config={"displayModeBar": False})

    # ── 4. Delta Acumulado ────────────────────────────────────────────────
    st.markdown(
        "<p style='font-family:JetBrains Mono,monospace;font-size:12px;"
        "letter-spacing:2px;color:rgba(255,255,255,0.7);margin-top:20px;'>"
        "⏱️ DELTA ACUMULADO</p>",
        unsafe_allow_html=True,
    )
    fig_delta = _build_delta_and_dominance(
        da, db, dist_grid, dominance, color_a, color_b, label_a, label_b, circuit_info,
    )
    st.plotly_chart(fig_delta, use_container_width=True, config={"displayModeBar": False})
