from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional
from fastf1.core import Session

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────────────────────────────────────────

BG_DARK    = "#0E0E0F"
BG_MAP     = "#13131A"
BG_PANEL   = "#111115"
BG_SURFACE = "#1A1A1F"
F1_WHITE   = "#FFFFFF"
F1_RED     = "#E8002D"
GREY_TRACK = "#50506A"
MONO_FONT  = "'JetBrains Mono', 'Courier New', monospace"

# Frames objetivo para la animación — equilibrio calidad/tamaño
TARGET_FRAMES = 400

# Ventana de scroll de los paneles de pedal (metros a cada lado del cursor)
SCROLL_WINDOW_M = 900

# Tamaño del marcador de cada coche
CAR_MARKER_SIZE = 14

# Colores de fallback si el equipo no está disponible
FALLBACK_COLORS = ["#00D2FF", "#FF6B35", "#C77DFF", "#39FF14"]

# Colorscale del heatmap de velocidad del circuito base
COLORSCALE_SPEED = [
    [0.00, "#1a1a2e"], [0.30, "#16213e"],
    [0.60, "#374785"], [0.85, "#50506A"],
    [1.00, "#888888"],
]


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_time(td: Optional[pd.Timedelta]) -> str:
    if td is None or (isinstance(td, float) and np.isnan(td)):
        return "—"
    try:
        t  = td.total_seconds()
        m  = int(t // 60)
        s  = int(t % 60)
        ms = int(round((t % 1) * 1000))
        return f"{m}:{s:02d}.{ms:03d}"
    except Exception:
        return "—"


def _fmt_delta(delta_s: float) -> str:
    """Formatea el delta de tiempo como +/-S.mmm"""
    if abs(delta_s) < 0.001:
        return "±0.000s"
    sign = "+" if delta_s > 0 else ""
    return f"{sign}{delta_s:.3f}s"


def _hex_rgb(hex_color: str) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"{r},{g},{b}"


def _get_team_color(sesion: Session, driver_code: str) -> str:
    try:
        info = sesion.get_driver(driver_code)
        return f"#{info['TeamColor']}"
    except Exception:
        return FALLBACK_COLORS[hash(driver_code) % len(FALLBACK_COLORS)]


def _normalize_brake(brake_raw: np.ndarray) -> np.ndarray:
    if brake_raw.dtype == bool or set(np.unique(brake_raw[~np.isnan(brake_raw)])).issubset({0, 1}):
        return brake_raw.astype(float) * 100
    if brake_raw.max() <= 1.0:
        return brake_raw * 100
    return brake_raw.copy()


def _exterior_coords(
    x: np.ndarray, y: np.ndarray,
    mid_x: float, mid_y: float,
    idx: int, offset: float,
) -> tuple[float, float]:
    i_next = min(idx + 5, len(x) - 1)
    i_prev = max(idx - 5, 0)
    dx, dy = x[i_next] - x[i_prev], y[i_next] - y[i_prev]
    nx, ny = -dy, dx
    mag = np.sqrt(nx**2 + ny**2)
    if mag != 0:
        nx /= mag; ny /= mag
    if (np.sqrt((x[idx] + nx * offset - mid_x)**2 + (y[idx] + ny * offset - mid_y)**2) <
            np.sqrt((x[idx] - mid_x)**2 + (y[idx] - mid_y)**2)):
        nx, ny = -nx, -ny
    return float(x[idx] + nx * offset), float(y[idx] + ny * offset)


def _corner_vlines(corner_dists: list[float]) -> tuple[list, list]:
    vx, vy = [], []
    for d in corner_dists:
        vx += [d, d, None]
        vy += [0, 105, None]
    return vx, vy


# ─────────────────────────────────────────────────────────────────────────────
# EXTRACCIÓN E INTERPOLACIÓN EN REJILLA DE DISTANCIA COMÚN
# ─────────────────────────────────────────────────────────────────────────────

def _extract_lap_telemetry(
    sesion: Session,
    driver_code: str,
    lap_number: int,
) -> Optional[pd.DataFrame]:
    """
    Extrae la telemetría de una vuelta específica y añade Distance.
    Retorna None si falla.
    """
    try:
        laps = sesion.laps.pick_drivers(driver_code)
        lap  = laps[laps["LapNumber"] == lap_number].iloc[0]
        tel  = lap.get_telemetry().add_distance()
        if tel.empty:
            return None
        return tel
    except Exception as e:
        return None


def _interpolate_on_grid(
    tel: pd.DataFrame,
    dist_grid: np.ndarray,
    ref_dist_max: float,
) -> dict[str, np.ndarray]:
    src_dist = tel["Distance"].values.astype(float)
    
    # Normalización por porcentaje para evitar fallos de trazada más corta/larga
    d_max = src_dist[-1] if (len(src_dist) > 0 and src_dist[-1] > 0) else 1.0
    frac_orig = src_dist / d_max
    frac_grid = dist_grid / ref_dist_max

    channels: dict[str, np.ndarray] = {}

    # Posición en el plano XY
    channels["x"] = np.interp(frac_grid, frac_orig, tel["X"].values.astype(float))
    channels["y"] = np.interp(frac_grid, frac_orig, tel["Y"].values.astype(float))

    # Velocidad
    channels["speed"] = np.interp(frac_grid, frac_orig, tel["Speed"].values.astype(float))

    # Tiempo acumulado en segundos — clave para el cálculo del delta
    # Se fuerza a empezar en 0.0s para evitar offsets de inicio
    time_s = tel["Time"].dt.total_seconds().values.astype(float)
    time_s = time_s - time_s[0]
    channels["time_s"] = np.interp(frac_grid, frac_orig, time_s)

    # Pedales
    throttle = tel["Throttle"].values.astype(float)
    channels["throttle"] = np.interp(frac_grid, frac_orig, throttle)

    brake_raw = _normalize_brake(tel["Brake"].values)
    channels["brake"] = np.interp(frac_grid, frac_orig, brake_raw.astype(float))

    # Motor
    channels["rpm"]  = np.interp(frac_grid, frac_orig, tel["RPM"].values.astype(float))
    channels["gear"] = np.round(np.interp(frac_grid, frac_orig, tel["nGear"].values.astype(float))).astype(int)

    # DRS
    channels["drs"] = np.interp(frac_grid, frac_orig, tel["DRS"].values.astype(float))

    return channels


# ─────────────────────────────────────────────────────────────────────────────
# CONSTRUCCIÓN DE LA FIGURA PLOTLY
# ─────────────────────────────────────────────────────────────────────────────

def _build_ghost_figure(
    # Car A
    data_a: dict[str, np.ndarray],
    color_a: str,
    label_a: str,
    lap_time_a: str,
    # Car B
    data_b: dict[str, np.ndarray],
    color_b: str,
    label_b: str,
    lap_time_b: str,
    # Circuit
    dist_grid: np.ndarray,
    circuit_info,
    show_corners: bool,
) -> go.Figure:
    
    N      = len(dist_grid)
    step   = max(1, N // TARGET_FRAMES)
    fidxs  = list(range(0, N, step))

    x_base  = data_a["x"]
    y_base  = data_a["y"]
    mid_x   = float(np.mean(x_base))
    mid_y   = float(np.mean(y_base))
    dist_max = float(dist_grid[-1])

    # ── Curvas ────────────────────────────────────────────────────────────
    corner_dists: list[float] = []
    lbl_x, lbl_y, lbl_t      = [], [], []

    if circuit_info is not None:
        for _, row in circuit_info.corners.iterrows():
            d_c = float(row["Distance"])
            if d_c > dist_max:
                continue
            corner_dists.append(d_c)
            if show_corners:
                idx_c = int(np.argmin(np.abs(dist_grid - d_c)))
                cx, cy = _exterior_coords(x_base, y_base, mid_x, mid_y, idx_c, 250)
                lbl_x.append(cx); lbl_y.append(cy)
                lbl_t.append(str(int(row["Number"])))

    vx_thr, vy_thr = _corner_vlines(corner_dists)
    vx_brk, vy_brk = _corner_vlines(corner_dists)

    # ── Línea de meta ─────────────────────────────────────────────────────
    dx_m   = float(x_base[1] - x_base[0])
    dy_m   = float(y_base[1] - y_base[0])
    mag_m  = np.sqrt(dx_m**2 + dy_m**2) or 1.0
    nmx, nmy = -dy_m / mag_m, dx_m / mag_m
    meta_w   = 200.0

    # ── Subplots (Proporciones Clásicas) ──────────────────────────────────
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"colspan": 2, "type": "xy"}, None],
               [{"type": "xy"},               {"type": "xy"}]],
        row_heights=[0.80, 0.20],
        vertical_spacing=0.003,
        horizontal_spacing=0.04,
    )

    # ── #0 Circuito base (LINEA SÓLIDA LIMPIA) ────────────────────────────
    fig.add_trace(go.Scatter(
        x=x_base, y=y_base, mode="lines",
        line=dict(color=GREY_TRACK, width=6),
        opacity=0.5, hoverinfo="skip", showlegend=False, name="_circuit",
    ), row=1, col=1)

    # ── #1 Etiquetas de curva (Bolitas Blancas) ───────────────────────────
    if show_corners and lbl_x:
        fig.add_trace(go.Scatter(
            x=lbl_x, y=lbl_y,
            mode="markers+text",
            text=lbl_t,
            textfont=dict(size=9, color="#0E0E0F", family=MONO_FONT),
            textposition="middle center",
            marker=dict(size=18, color=F1_WHITE, symbol="circle",
                        line=dict(color=F1_RED, width=1.5)),
            hoverinfo="skip",
            showlegend=False,
            name="_corners",
        ), row=1, col=1)
    else:
        fig.add_trace(go.Scatter(x=[], y=[], showlegend=False,
                                 hoverinfo="skip", name="_corners_empty"), row=1, col=1)

    # ── #2 Línea de meta ─────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=[x_base[0] - nmx * meta_w, x_base[0] + nmx * meta_w],
        y=[y_base[0] - nmy * meta_w, y_base[0] + nmy * meta_w],
        mode="lines",
        line=dict(color=F1_WHITE, width=3),
        hoverinfo="skip",
        showlegend=False,
        name="_meta",
    ), row=1, col=1)

    # ── #3 Acelerador A ──────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=dist_grid, y=data_a["throttle"],
        mode="lines", line=dict(color=color_a, width=1.5),
        fill="tozeroy", fillcolor=f"rgba({_hex_rgb(color_a)},0.20)",
        opacity=0.85, hoverinfo="skip", showlegend=False, name=f"_thr_{label_a}",
    ), row=2, col=1)

    # ── #4 Acelerador B ──────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=dist_grid, y=data_b["throttle"],
        mode="lines", line=dict(color=color_b, width=1.5, dash="dot"),
        fill="tozeroy", fillcolor=f"rgba({_hex_rgb(color_b)},0.10)",
        opacity=0.85, hoverinfo="skip", showlegend=False, name=f"_thr_{label_b}",
    ), row=2, col=1)

    # ── #5 Vlines curvas (Acelerador) ────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=vx_thr, y=vy_thr, mode="lines",
        line=dict(color="rgba(136,136,160,0.30)", width=0.8, dash="dash"),
        hoverinfo="skip", showlegend=False, name="_thr_vlines",
    ), row=2, col=1)

    # ── #6 Freno A ───────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=dist_grid, y=data_a["brake"],
        mode="lines", line=dict(color=color_a, width=1.5),
        fill="tozeroy", fillcolor=f"rgba({_hex_rgb(color_a)},0.20)",
        opacity=0.85, hoverinfo="skip", showlegend=False, name=f"_brk_{label_a}",
    ), row=2, col=2)

    # ── #7 Freno B ───────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=dist_grid, y=data_b["brake"],
        mode="lines", line=dict(color=color_b, width=1.5, dash="dot"),
        fill="tozeroy", fillcolor=f"rgba({_hex_rgb(color_b)},0.10)",
        opacity=0.85, hoverinfo="skip", showlegend=False, name=f"_brk_{label_b}",
    ), row=2, col=2)

    # ── #8 Vlines curvas (Freno) ─────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=vx_brk, y=vy_brk, mode="lines",
        line=dict(color="rgba(136,136,160,0.30)", width=0.8, dash="dash"),
        hoverinfo="skip", showlegend=False, name="_brk_vlines",
    ), row=2, col=2)

    # ── #9 Marcador coche A ──────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=[], y=[], mode="markers",
        marker=dict(size=CAR_MARKER_SIZE, color=color_a, symbol="circle",
                    line=dict(color=F1_WHITE, width=2)),
        hoverinfo="skip", showlegend=True, legendgroup=label_a,
        name=f"{label_a}  {lap_time_a}",
    ), row=1, col=1)

    # ── #10 Marcador coche B ─────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=[], y=[], mode="markers",
        marker=dict(size=CAR_MARKER_SIZE, color=color_b, symbol="circle",
                    line=dict(color=F1_WHITE, width=2)),
        hoverinfo="skip", showlegend=True, legendgroup=label_b,
        name=f"{label_b}  {lap_time_b}",
    ), row=1, col=1)

    # ── #11 & #12 Cursores ────────────────────────────────────────────────
    fig.add_trace(go.Scatter(x=[], y=[0, 105], mode="lines", line=dict(color=F1_WHITE, width=1.5), hoverinfo="skip", showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=[], y=[0, 105], mode="lines", line=dict(color=F1_WHITE, width=1.5), hoverinfo="skip", showlegend=False), row=2, col=2)

    # ── ESTADO INICIAL DEL HUD ─────────────────────────────────────────────
    hud_text_init = (
        f"<span style='color:{F1_WHITE}'><b>Δ IGUAL</b></span><br>"
        f"──────────────────────<br>"
        f"<span style='color:{color_a}'><b>{label_a}</b></span>  "
        f"<span style='color:{color_b}'><b>{label_b}</b></span><br>"
        f"V: <span style='color:{color_a}'>  0.0</span>  <span style='color:{color_b}'>  0.0</span> km/h<br>"
        f"G: <span style='color:{color_a}'>0</span>  <span style='color:{color_b}'>0</span><br>"
        f"RPM: <span style='color:{color_a}'>    0</span>  <span style='color:{color_b}'>    0</span><br>"
        f"DRS: <span style='color:{color_a}'>—</span>  <span style='color:{color_b}'>—</span>"
    )

    # ─────────────────────────────────────────────────────────────────────
    # FRAMES
    # ─────────────────────────────────────────────────────────────────────
    frames = []

    for i, idx in enumerate(fidxs):
        idx = min(idx, N - 1)
        cur_dist = float(dist_grid[idx])
        cur_time = float(data_a["time_s"][idx])

        # ── LA CÁMARA ESTILO SUPER MARIO (TAMAÑO DE ZOOM FIJO) ──
        window_size = SCROLL_WINDOW_M * 2

        v_min = cur_dist - SCROLL_WINDOW_M
        v_max = cur_dist + SCROLL_WINDOW_M

        # Clampeamos los bordes pero forzando que el tamaño de la ventana NUNCA cambie
        if v_min < 0:
            v_min = 0
            v_max = window_size
        if v_max > dist_max:
            v_max = dist_max
            v_min = max(0.0, dist_max - window_size)
        # ────────────────────────────────────────────────────────

        delta_at_point = float(data_a["time_s"][idx] - data_b["time_s"][idx])
        
        if delta_at_point > 0.05:
            delta_color = F1_RED
            delta_who   = f"{label_a} +{delta_at_point:.3f}s"
        elif delta_at_point < -0.05:
            delta_color = "#39FF14"
            delta_who   = f"{label_a} {delta_at_point:.3f}s"
        else:
            delta_color = F1_WHITE
            delta_who   = "IGUAL"

        v_a, v_b   = float(data_a["speed"][idx]), float(data_b["speed"][idx])
        g_a, g_b   = int(data_a["gear"][idx]), int(data_b["gear"][idx])
        r_a, r_b   = float(data_a["rpm"][idx]), float(data_b["rpm"][idx])
        drs_a = "OPEN" if data_a["drs"][idx] > 8 else "—"
        drs_b = "OPEN" if data_b["drs"][idx] > 8 else "—"

        hud_text = (
            f"<span style='color:{delta_color}'><b>Δ {delta_who}</b></span><br>"
            f"──────────────────────<br>"
            f"<span style='color:{color_a}'><b>{label_a}</b></span>  <span style='color:{color_b}'><b>{label_b}</b></span><br>"
            f"V: <span style='color:{color_a}'>{v_a:5.1f}</span>  <span style='color:{color_b}'>{v_b:5.1f}</span> km/h<br>"
            f"G: <span style='color:{color_a}'>{g_a}</span>  <span style='color:{color_b}'>{g_b}</span><br>"
            f"RPM: <span style='color:{color_a}'>{r_a:5.0f}</span>  <span style='color:{color_b}'>{r_b:5.0f}</span><br>"
            f"DRS: <span style='color:{color_a}'>{drs_a}</span>  <span style='color:{color_b}'>{drs_b}</span>"
        )

        # Novedad: Interpolar físicamente dónde estaba el coche B en el tiempo del Coche A
        x_b_ghost = float(np.interp(cur_time, data_b["time_s"], data_b["x"]))
        y_b_ghost = float(np.interp(cur_time, data_b["time_s"], data_b["y"]))

        frame_data = [
            go.Scatter(x=[float(data_a["x"][idx])], y=[float(data_a["y"][idx])]),
            go.Scatter(x=[x_b_ghost], y=[y_b_ghost]), # Aquí usamos el fantasma real
            go.Scatter(x=[cur_dist, cur_dist], y=[0, 105]),
            go.Scatter(x=[cur_dist, cur_dist], y=[0, 105]),
        ]

        frame_layout = {
            "xaxis2": {"range": [v_min, v_max]}, # ¡Pásale solo el range!
            "xaxis3": {"range": [v_min, v_max]},
            "annotations": [dict(
                x=0.14, y=0.98, xref="paper", yref="paper",
                text=hud_text, showarrow=False, align="left",
                font=dict(family=MONO_FONT, size=14, color=F1_WHITE),
                bgcolor="rgba(0,0,0,0)", borderwidth=0,
            )],
        }
        frames.append(go.Frame(data=frame_data, layout=frame_layout, traces=[9, 10, 11, 12], name=str(i)))

    fig.frames = frames

    # ── Controles de animación ─────────────────────────────────────────────
    frame_ms = 40   # ~25 fps
    slider_steps = []
    tick_every   = max(1, len(fidxs) // 80)
    
    for i in range(0, len(fidxs), tick_every):
        d_i = float(dist_grid[min(fidxs[i], N - 1)])
        slider_steps.append(dict(
            args=[[str(i)], {"frame": {"duration": frame_ms, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}],
            label=f"{d_i/1000:.1f}km", method="animate",
        ))

    updatemenus = [dict(
        type="buttons", showactive=False, y=1.06, x=0.0, xanchor="left", yanchor="top", pad=dict(t=0, r=10),
        buttons=[
            dict(label="▶ PLAY", method="animate", args=[None, {"frame": {"duration": frame_ms, "redraw": False}, "fromcurrent": True, "transition": {"duration": 0}}]),
            dict(label="⏸ PAUSE", method="animate", args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}]),
        ],
    )]

    sliders = [dict(
        active=0, transition=dict(duration=0), pad=dict(b=10, t=5), len=1.0, x=0.0, y=0.0, steps=slider_steps,
        bgcolor="rgba(255,255,255,0.05)", bordercolor="rgba(255,255,255,0.1)", tickcolor="rgba(255,255,255,0.3)",
        font=dict(size=8, color="rgba(255,255,255,0.4)", family=MONO_FONT),
        currentvalue=dict(prefix="Dist: ", visible=True, font=dict(size=10, color="rgba(255,255,255,0.5)", family=MONO_FONT)),
    )]

    # ── Layout global (EL ESTILO LIMPIO GIGANTE) ──────────────────────────
    x_min, x_max = np.min(x_base), np.max(x_base)
    y_min, y_max = np.min(y_base), np.max(y_base)
    margen_x = (x_max - x_min) * 0.02
    margen_y = (y_max - y_min) * 0.02

    axis_panel = dict(showgrid=False, zeroline=False, showticklabels=False, showline=True, linecolor="rgba(136,136,160,0.25)", linewidth=1)

    fig.update_layout(
        height=900,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family=MONO_FONT, color=F1_WHITE, size=11),
        updatemenus=updatemenus,
        sliders=sliders,
        showlegend=True,
        legend=dict(orientation="v", yanchor="top", y=1.0, xanchor="left", x=1.02, bgcolor="rgba(14,14,15,0.7)", font=dict(size=10)),
        margin=dict(l=10, r=10, t=10, b=10),
        hoverlabel=dict(bgcolor="rgba(255,255,255,0.15)", bordercolor="rgba(255,255,255,0.15)", font=dict(family=MONO_FONT, color=F1_WHITE, size=11)),
        # Mapa 2D Ceñido
        xaxis=dict(range=[x_min - margen_x, x_max + margen_x], showgrid=False, zeroline=False, visible=False),
        yaxis=dict(range=[y_min - margen_y, y_max + margen_y], showgrid=False, zeroline=False, visible=False, scaleanchor="x", scaleratio=1),
        # Paneles pedal
        xaxis2={**axis_panel, "range": [0, SCROLL_WINDOW_M * 2]},
        yaxis2=dict(range=[0, 105], showgrid=True, gridcolor="rgba(255,255,255,0.05)", zeroline=False, showticklabels=False, title=dict(text="THROT", font=dict(size=9, color="rgba(255,255,255,0.5)"))),
        xaxis3={**axis_panel, "range": [0, SCROLL_WINDOW_M * 2]},
        yaxis3=dict(range=[0, 105], showgrid=True, gridcolor="rgba(255,255,255,0.05)", zeroline=False, showticklabels=False, title=dict(text="BRAKE", font=dict(size=9, color="rgba(255,255,255,0.5)"))),
        # HUD Inicial Integrado (Sin Cajas)
        annotations=[dict(
            x=0.14, y=0.98, xref="paper", yref="paper",
            text=hud_text_init, showarrow=False, align="left",
            font=dict(family=MONO_FONT, size=14, color=F1_WHITE),
            bgcolor="rgba(0,0,0,0)", borderwidth=0,
        )]
    )

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# GRÁFICO DELTA VS DISTANCIA (estático, debajo de la animación)
# ─────────────────────────────────────────────────────────────────────────────

def _build_delta_chart(
    data_a: dict[str, np.ndarray],
    data_b: dict[str, np.ndarray],
    dist_grid: np.ndarray,
    color_a: str,
    color_b: str,
    label_a: str,
    label_b: str,
) -> go.Figure:
    """
    Gráfico de delta acumulado (A − B) a lo largo de la distancia.

    La línea cruza el cero donde los dos pilotos se igualan.
    Área verde = A va por delante.
    Área roja  = B va por delante.

    Interpretación: la pendiente del delta en cada zona indica dónde se gana
    o se pierde tiempo. Pendiente negativa → A gana en ese sector.
    """
    delta = data_a["time_s"] - data_b["time_s"]

    # Separar en positivo y negativo para rellenar con colores distintos
    delta_pos = np.where(delta >= 0, delta,  0)
    delta_neg = np.where(delta <= 0, delta,  0)

    fig = go.Figure()

    # Relleno zona positiva (B ganando)
    fig.add_trace(go.Scatter(
        x=dist_grid, y=delta_pos,
        mode="lines",
        line=dict(color=color_b, width=0),
        fill="tozeroy",
        fillcolor=f"rgba({_hex_rgb(color_b)},0.25)",
        hoverinfo="skip",
        showlegend=False,
        name="_delta_pos",
    ))

    # Relleno zona negativa (A ganando)
    fig.add_trace(go.Scatter(
        x=dist_grid, y=delta_neg,
        mode="lines",
        line=dict(color=color_a, width=0),
        fill="tozeroy",
        fillcolor=f"rgba({_hex_rgb(color_a)},0.25)",
        hoverinfo="skip",
        showlegend=False,
        name="_delta_neg",
    ))

    # Línea de delta con tooltip
    hover_text = [
        f"<b>Dist: {d:.0f} m</b><br>Δ {_fmt_delta(v)}<br>"
        f"<span style='color:{color_a if v < 0 else color_b}'>"
        f"{'→ ' + label_a + ' adelante' if v < 0 else '→ ' + label_b + ' adelante'}"
        f"</span>"
        for d, v in zip(dist_grid, delta)
    ]

    fig.add_trace(go.Scatter(
        x=dist_grid, y=delta,
        mode="lines",
        line=dict(color=F1_WHITE, width=1.5),
        text=hover_text,
        hovertemplate="%{text}<extra></extra>",
        showlegend=False,
        name="delta",
    ))

    # Línea de cero
    fig.add_hline(y=0, line_width=1, line_color="rgba(255,255,255,0.2)")

    # Anotación final: quién gana y por cuánto
    final_delta = float(delta[-1])
    winner      = label_a if final_delta < 0 else label_b
    winner_col  = color_a if final_delta < 0 else color_b
    margin      = abs(final_delta)

    fig.add_annotation(
        x=float(dist_grid[-1]), y=float(delta[-1]),
        text=f"<b>{winner}  {_fmt_delta(-margin if final_delta < 0 else margin)}</b>",
        showarrow=True, arrowhead=2, arrowcolor=winner_col,
        font=dict(color=winner_col, size=10, family=MONO_FONT),
        bgcolor=BG_SURFACE, bordercolor=winner_col, borderwidth=1,
        ax=-60, ay=-30,
    )

    fig.update_layout(
        height=160,
        paper_bgcolor=BG_DARK,
        plot_bgcolor=BG_PANEL,
        font=dict(family=MONO_FONT, color=F1_WHITE, size=10),
        margin=dict(l=54, r=24, t=10, b=40),
        showlegend=False,
        hovermode="x unified",
        hoverlabel=dict(bgcolor=BG_SURFACE, bordercolor="rgba(255,255,255,0.15)",
                        font=dict(family=MONO_FONT, color=F1_WHITE, size=10)),
        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)",
                   zeroline=False, ticksuffix=" m",
                   tickfont=dict(size=8, color="rgba(255,255,255,0.4)")),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)",
                   zeroline=False, ticksuffix="s",
                   tickfont=dict(size=8, color="rgba(255,255,255,0.4)"),
                   title=dict(text=f"Δ {label_a}−{label_b}",
                              font=dict(size=9, color="rgba(255,255,255,0.5)"))),
    )

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# SELECTOR DE PILOTO + VUELTA (reutilizable)
# ─────────────────────────────────────────────────────────────────────────────

def _driver_lap_selector(
    sesion: Session,
    key_prefix: str,
    label: str,
    default_driver: Optional[str] = None,
    default_driver_b: Optional[str] = None,
) -> tuple[str, int]:
    """
    Widget de selección de piloto + vuelta. Devuelve (driver_code, lap_number).
    key_prefix asegura que los widgets de A y B son independientes en session_state.
    """
    driver_names = {
        d: sesion.get_driver(d).get("FullName", str(d))
        for d in sesion.drivers
    }

    # Driver por defecto
    drivers_list = list(sesion.drivers)
    if default_driver and default_driver in drivers_list:
        def_drv_idx = drivers_list.index(default_driver)
    elif default_driver_b and default_driver_b in drivers_list:
        # Para el segundo piloto, intentamos evitar repetir el primero
        candidates = [d for d in drivers_list if d != default_driver_b]
        def_drv_idx = drivers_list.index(candidates[0]) if candidates else 0
    else:
        def_drv_idx = 0

    driver = st.selectbox(
        label,
        options=drivers_list,
        index=def_drv_idx,
        format_func=lambda x: driver_names.get(x, x),
        key=f"{key_prefix}_drv",
    )

    # Vueltas disponibles para ese piloto
    try:
        laps_drv = sesion.laps.pick_drivers(driver).pick_accurate()
        fastest_n = laps_drv.pick_fastest()["LapNumber"]
        lap_list  = sorted(laps_drv["LapNumber"].tolist())
        def_idx   = lap_list.index(fastest_n) if fastest_n in lap_list else 0

        lap_n = st.selectbox(
            "Vuelta",
            options=lap_list,
            index=def_idx,
            format_func=lambda x: (
                f"V{int(x)}  ⏱ Best" if x == fastest_n else f"V{int(x)}"
            ),
            key=f"{key_prefix}_lap",
        )
    except Exception:
        lap_n = st.number_input("Vuelta (manual)", min_value=1, value=1,
                                key=f"{key_prefix}_lap_manual")

    return str(driver), int(lap_n)


# ─────────────────────────────────────────────────────────────────────────────
# PUNTO DE ENTRADA PÚBLICO
# ─────────────────────────────────────────────────────────────────────────────

def render_ghost_car(sesion: Session, corners: bool = True) -> None:
    """
    Punto de entrada principal del módulo Ghost Car.

    Renders:
      1. Selectores de piloto/vuelta para A y B (pueden ser el mismo piloto
         con vueltas distintas para self-comparison).
      2. Animación 2D sincronizada por distancia con HUD dual y delta en tiempo real.
      3. Gráfico de delta acumulado vs distancia.

    Parámetros
    ──────────
    sesion  : fastf1.core.Session ya cargada (con load_telemetry=True).
    corners : si se muestran las etiquetas de curva en el mapa.
    """
    # ── 1. Cabecera ───────────────────────────────────────────────────────
    st.markdown(
        "<p style='font-family: JetBrains Mono, monospace; font-size: 11px; "
        "letter-spacing: 2px; color: rgba(255,255,255,0.35); margin-bottom: 8px;'>"
        "GHOST CAR · COMPARATIVA POR DISTANCIA</p>",
        unsafe_allow_html=True,
    )

    # ── 2. Selectores de piloto y vuelta ──────────────────────────────────
    col_a, col_sep, col_b = st.columns([5, 1, 5])

    drivers_list = list(sesion.drivers)
    default_a    = drivers_list[0] if drivers_list else None
    default_b    = drivers_list[1] if len(drivers_list) > 1 else drivers_list[0]

    with col_a:
        st.markdown(
            "<p style='font-family: JetBrains Mono, monospace; font-size: 10px; "
            "letter-spacing: 2px; color: rgba(255,255,255,0.45);'>COCHE A</p>",
            unsafe_allow_html=True,
        )
        driver_a, lap_a = _driver_lap_selector(
            sesion, key_prefix="ghost_a",
            label="Piloto A",
            default_driver=default_a,
        )

    with col_sep:
        st.markdown("<div style='text-align:center;padding-top:32px;"
                    "font-family:JetBrains Mono,monospace;font-size:18px;"
                    "color:rgba(255,255,255,0.25);'>VS</div>", unsafe_allow_html=True)

    with col_b:
        st.markdown(
            "<p style='font-family: JetBrains Mono, monospace; font-size: 10px; "
            "letter-spacing: 2px; color: rgba(255,255,255,0.45);'>COCHE B</p>",
            unsafe_allow_html=True,
        )
        driver_b, lap_b = _driver_lap_selector(
            sesion, key_prefix="ghost_b",
            label="Piloto B",
            default_driver=default_b,
            default_driver_b=driver_a,
        )

    # Aviso self-comparison
    if driver_a == driver_b and lap_a == lap_b:
        st.info("⚠️ Has seleccionado el mismo piloto y la misma vuelta. "
                "Prueba distintas vueltas del mismo piloto para comparar evolución de ritmo.")



    # ── 3. Carga de telemetría ─────────────────────────────────────────────
    with st.spinner(f"Cargando y sincronizando telemetría — {driver_a} V{lap_a}  vs  {driver_b} V{lap_b}"):
        tel_a = _extract_lap_telemetry(sesion, driver_a, lap_a)
        tel_b = _extract_lap_telemetry(sesion, driver_b, lap_b)

    if tel_a is None:
        st.error(f"❌ No se pudo cargar la telemetría de {driver_a} · V{lap_a}.")
        return
    if tel_b is None:
        st.error(f"❌ No se pudo cargar la telemetría de {driver_b} · V{lap_b}.")
        return

    # ── 4. Rejilla de distancia común ─────────────────────────────────────
    dist_max_a = float(tel_a["Distance"].max())
    dist_max_b = float(tel_b["Distance"].max())
    
    # Usamos la distancia máxima del coche de referencia (A)
    ref_dist_max = dist_max_a

    # Resolución: ~2000 puntos para suavidad sin peso excesivo
    n_grid    = 2000
    dist_grid = np.linspace(0, ref_dist_max, n_grid)

    # ── 5. Interpolación en rejilla común ─────────────────────────────────
    data_a = _interpolate_on_grid(tel_a, dist_grid, ref_dist_max)
    data_b = _interpolate_on_grid(tel_b, dist_grid, ref_dist_max)

    # ── 6. Colores y metadatos ─────────────────────────────────────────────
    color_a = _get_team_color(sesion, driver_a)
    color_b = _get_team_color(sesion, driver_b)

    # Si mismo equipo (o mismo piloto), forzar color diferente al segundo
    if color_a.lower() == color_b.lower():
        color_b = FALLBACK_COLORS[1]   # azul cian como diferenciador

    lap_time_a = _fmt_time(
        sesion.laps.pick_drivers(driver_a)[
            sesion.laps.pick_drivers(driver_a)["LapNumber"] == lap_a
        ].iloc[0].get("LapTime")
    )
    lap_time_b = _fmt_time(
        sesion.laps.pick_drivers(driver_b)[
            sesion.laps.pick_drivers(driver_b)["LapNumber"] == lap_b
        ].iloc[0].get("LapTime")
    )


    # Etiqueta compacta para el HUD y la leyenda
    laps_a = sesion.laps.pick_drivers(driver_a)
    row_a  = laps_a[laps_a["LapNumber"] == lap_a].iloc[0]
    laps_b = sesion.laps.pick_drivers(driver_b)
    row_b  = laps_b[laps_b["LapNumber"] == lap_b].iloc[0]
    abbr_a = str(row_a["Driver"])
    abbr_b = str(row_b["Driver"])
    label_a = abbr_a if driver_a != driver_b else f"{abbr_a}·V{lap_a}"
    label_b = abbr_b if driver_a != driver_b else f"{abbr_b}·V{lap_b}"

   

    # ── 7. Circuit info ────────────────────────────────────────────────────
    try:
        circuit_info = sesion.get_circuit_info()
    except Exception:
        circuit_info = None
    
    # ── 10. Tabla de KPIs comparativos ────────────────────────────────────
    st.markdown(
        "<p style='font-family: JetBrains Mono, monospace; font-size: 11px; "
        "letter-spacing: 2px; color: rgba(255,255,255,0.35); margin-bottom: 8px;'>"
        "KPIs COMPARATIVOS</p>",
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4, c5 = st.columns(5)

    delta_final  = float(data_a["time_s"][-1] - data_b["time_s"][-1])
    winner_str   = (f"{label_a}" if delta_final < 0 else f"{label_b}")
    delta_display = f"{_fmt_delta(delta_final)} {winner_str}"

    c1.metric(
        "Δ TIEMPO TOTAL",
        delta_display,
        help="Delta acumulado al final de la distancia común (A − B)"
    )
    c2.metric(
        f"V MÁX  {label_a}",
        f"{float(data_a['speed'].max()):.1f} km/h",
    )
    c3.metric(
        f"V MÁX  {label_b}",
        f"{float(data_b['speed'].max()):.1f} km/h",
    )
    c4.metric(
        f"V MEDIA  {label_a}",
        f"{float(data_a['speed'].mean()):.1f} km/h",
    )
    c5.metric(
        f"V MEDIA  {label_b}",
        f"{float(data_b['speed'].mean()):.1f} km/h",
    )

    st.divider()

    # ── 8. Animación ──────────────────────────────────────────────────────
    fig = _build_ghost_figure(
        data_a=data_a, color_a=color_a, label_a=label_a, lap_time_a=lap_time_a,
        data_b=data_b, color_b=color_b, label_b=label_b, lap_time_b=lap_time_b,
        dist_grid=dist_grid,
        circuit_info=circuit_info,
        show_corners=corners,
    )

    st.plotly_chart(fig, use_container_width=True, config={
        "displayModeBar"         : True,
        "displaylogo"            : False,
        "modeBarButtonsToRemove" : ["select2d", "lasso2d", "autoScale2d"],
        "toImageButtonOptions"   : {
            "format"  : "png",
            "filename": f"talos_ghost_{driver_a}_vs_{driver_b}",
            "scale"   : 2,
        },
    })

    # ── 9. Delta chart ─────────────────────────────────────────────────────
    st.markdown(
        "<p style='font-family: JetBrains Mono, monospace; font-size: 11px; "
        "letter-spacing: 2px; color: rgba(255,255,255,0.35); margin: 8px 0 4px;'>"
        "DELTA ACUMULADO · A − B</p>",
        unsafe_allow_html=True,
    )

    fig_delta = _build_delta_chart(
        data_a=data_a, data_b=data_b, dist_grid=dist_grid,
        color_a=color_a, color_b=color_b,
        label_a=label_a, label_b=label_b,
    )
    st.plotly_chart(fig_delta, use_container_width=True,
                    config={"displayModeBar": False})