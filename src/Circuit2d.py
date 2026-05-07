from __future__ import annotations
import ui_assets

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
F1_WHITE   = "#FFFFFF"
F1_RED     = "#E8002D"
DRS_GREEN  = "#00E887"
GREY_TRACK = "#50506A"

MONO_FONT  = "'JetBrains Mono', 'Courier New', monospace"

# Animación: target de frames. 1000 es para máxima fluidez
TARGET_FRAMES = 1000

# Ventana de scroll de los paneles de pedal (metros a cada lado del cursor)
SCROLL_WINDOW_M = 900

TYRE_COLORS = {
    "SOFT": "#E8002D", "MEDIUM": "#FFF200", "HARD": "#EBEBEB",
    "INTER": "#43B02A", "WET": "#0067FF",
    "SUPERSOFT": "#E8002D", "ULTRASOFT": "#C77DFF", "HYPERSOFT": "#FF66B2",
    "UNKNOWN": "#888888",
}

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_time(td: Optional[pd.Timedelta]) -> str:
    """Timedelta → MM:SS.mmm"""
    if td is None or pd.isna(td):
        return "—"
    try:
        t = td.total_seconds()
        m, s = int(t // 60), int(t % 60)
        ms = int(round((t % 1) * 1000))
        return f"{m}:{s:02d}.{ms:03d}"
    except Exception:
        return "—"

def _exterior_coords(
    x: np.ndarray, y: np.ndarray,
    mid_x: float, mid_y: float,
    idx: int, offset: float,
) -> tuple[float, float]:
    i_next = min(idx + 5, len(x) - 1)
    i_prev = max(idx - 5, 0)
    dx = x[i_next] - x[i_prev]
    dy = y[i_next] - y[i_prev]
    nx, ny = -dy, dx
    mag = np.sqrt(nx**2 + ny**2)
    if mag != 0:
        nx /= mag; ny /= mag

    if (np.sqrt((x[idx] + nx * offset - mid_x)**2 + (y[idx] + ny * offset - mid_y)**2) <
            np.sqrt((x[idx] - mid_x)**2 + (y[idx] - mid_y)**2)):
        nx, ny = -nx, -ny

    return float(x[idx] + nx * offset), float(y[idx] + ny * offset)

def _corner_vlines_arrays(
    corner_distances: list[float], y_min: float = 0, y_max: float = 105
) -> tuple[list, list]:
    vx, vy = [], []
    for d in corner_distances:
        vx += [d, d, None]
        vy += [y_min, y_max, None]
    return vx, vy

def _normalize_brake(brake_raw: np.ndarray) -> np.ndarray:
    if brake_raw.dtype == bool or set(np.unique(brake_raw[~np.isnan(brake_raw)])).issubset({0, 1}):
        return brake_raw.astype(float) * 100
    if brake_raw.max() <= 1.0:
        return brake_raw * 100
    return brake_raw.copy()

def _is_qualifying(sesion: Session) -> bool:
    try:
        name = str(sesion.session_info.get("Type", "")).lower()
        return any(k in name for k in ("qualifying", "sprint qualifying"))
    except Exception:
        return False

# ─────────────────────────────────────────────────────────────────────────────
# CONSTRUCCIÓN DE LA FIGURA PLOTLY CON FRAMES
# ─────────────────────────────────────────────────────────────────────────────

def _build_animated_figure(
    x: np.ndarray,
    y: np.ndarray,
    dist_array: np.ndarray,
    vel: np.ndarray,
    throttle: np.ndarray,
    brake_pct: np.ndarray,
    rpm: np.ndarray,
    gear: np.ndarray,
    drs: np.ndarray,
    circuit_info,
    show_corners: bool,
    coche_color: str,
    driver: str,
    team: str,
    compound: str,
    tyre_age: int,
    lap_time_str: str,
    lap_time_seg: float,
    t_sec, s1_idx, s2_idx, c_s1, c_s2, c_s3,
    lap_delta, d_s1, d_s2, d_s3,
) -> go.Figure:

    N          = len(x)
    mid_x      = float(np.mean(x))
    mid_y      = float(np.mean(y))
    dist_max   = float(dist_array[-1])

    step         = max(1, N // TARGET_FRAMES)
    frame_indices = list(range(0, N, step))
    n_frames     = len(frame_indices)

    brake_binary = (brake_pct > 0).astype(int)
    brake_starts = np.where(np.diff(brake_binary) > 0)[0]

    drs_binary = (drs > 8).astype(int)
    drs_starts = np.where(np.diff(drs_binary) > 0)[0]

    dx_m   = float(x[1] - x[0])
    dy_m   = float(y[1] - y[0])
    mag_m  = np.sqrt(dx_m**2 + dy_m**2)
    nmx, nmy = -dy_m / mag_m, dx_m / mag_m
    meta_w = 200.0
    meta_x = [x[0] - nmx * meta_w, x[0] + nmx * meta_w]
    meta_y = [y[0] - nmy * meta_w, y[0] + nmy * meta_w]

    corner_dists = []
    corner_labels_x, corner_labels_y, corner_labels_t = [], [], []
    if circuit_info is not None:
        for _, row in circuit_info.corners.iterrows():
            d_c = row["Distance"]
            corner_dists.append(float(d_c))
            if show_corners:
                idx_c = int(np.argmin(np.abs(dist_array - d_c)))
                cx, cy = _exterior_coords(x, y, mid_x, mid_y, idx_c, 250)
                corner_labels_x.append(cx)
                corner_labels_y.append(cy)
                corner_labels_t.append(str(int(row["Number"])))

    vx_thr, vy_thr = _corner_vlines_arrays(corner_dists, 0, 105)
    vx_brk, vy_brk = _corner_vlines_arrays(corner_dists, 0, 105)

    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"colspan": 2, "type": "xy"}, None],
               [{"type": "xy"},               {"type": "xy"}]],
        row_heights=[0.8, 0.2],
        vertical_spacing=0.003,
        horizontal_spacing=0.04,
    )

    fig.add_trace(go.Scatter(
        x=x, y=y, mode="lines",
        line=dict(color=GREY_TRACK, width=6),
        opacity=0.5, hoverinfo="skip", showlegend=False, name="_circuit",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=x[brake_starts] if len(brake_starts) else [],
        y=y[brake_starts] if len(brake_starts) else [],
        mode="markers", marker=dict(symbol="triangle-down", size=8, color=F1_RED, opacity=0.85, line=dict(color="rgba(0,0,0,0.3)", width=0.5)),
        name="Braking", hovertemplate="<b>Braking</b><extra></extra>", showlegend=True,
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=x[drs_starts] if len(drs_starts) else [],
        y=y[drs_starts] if len(drs_starts) else [],
        mode="markers", marker=dict(symbol="circle", size=8, color=DRS_GREEN, opacity=0.85, line=dict(color="rgba(0,0,0,0.3)", width=0.5)),
        name="DRS Open", hovertemplate="<b>DRS Open</b><extra></extra>", showlegend=True,
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=meta_x, y=meta_y, mode="lines",
        line=dict(color=F1_WHITE, width=3),
        hoverinfo="skip", showlegend=False, name="_meta",
    ), row=1, col=1)

    if show_corners and corner_labels_x:
        fig.add_trace(go.Scatter(
            x=corner_labels_x, y=corner_labels_y, mode="text", text=corner_labels_t,
            textfont=dict(size=9, color="#0E0E0F", family=MONO_FONT),
            textposition="middle center",
            marker=dict(size=18, color=F1_WHITE, symbol="circle", line=dict(color=F1_RED, width=1.5)),
            hoverinfo="skip", showlegend=False, name="_corners",
        ), row=1, col=1)
        fig.data[-1].update(mode="markers+text")
    else:
        fig.add_trace(go.Scatter(x=[], y=[], showlegend=False, hoverinfo="skip", name="_corners_empty"), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=dist_array, y=throttle, mode="lines", line=dict(color="#39FF14", width=1.5),
        fill="tozeroy", fillcolor="rgba(57,255,20,0.15)", hoverinfo="skip", showlegend=False, name="_thr_fill",
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=vx_thr, y=vy_thr, mode="lines", line=dict(color="rgba(136,136,160,0.35)", width=0.8, dash="dash"),
        hoverinfo="skip", showlegend=False, name="_thr_vlines",
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=dist_array, y=brake_pct, mode="lines", line=dict(color=F1_RED, width=1.5),
        fill="tozeroy", fillcolor="rgba(232,0,45,0.15)", hoverinfo="skip", showlegend=False, name="_brk_fill",
    ), row=2, col=2)

    fig.add_trace(go.Scatter(
        x=vx_brk, y=vy_brk, mode="lines", line=dict(color="rgba(136,136,160,0.35)", width=0.8, dash="dash"),
        hoverinfo="skip", showlegend=False, name="_brk_vlines",
    ), row=2, col=2)

    fig.add_trace(go.Scattergl(
        x=[], y=[], mode="lines", line=dict(color=coche_color, width=3),
        hoverinfo="skip", showlegend=False, name="_trail",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=[], y=[], mode="markers", marker=dict(size=12, color=coche_color, symbol="circle", line=dict(color=F1_WHITE, width=2)),
        hoverinfo="skip", showlegend=False, name="_car",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=[], y=[0, 105], mode="lines", line=dict(color=F1_WHITE, width=2), hoverinfo="skip", showlegend=False, name="_thr_cursor",
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=[], y=[0, 105], mode="lines", line=dict(color=F1_WHITE, width=2), hoverinfo="skip", showlegend=False, name="_brk_cursor",
    ), row=2, col=2)


    # ── FORMATO DE DELTAS (ROJO/VERDE) ─────────────────────────────────────
    def format_delta_mono(d):
        if d is None or pd.isna(d): return "&nbsp;&nbsp;&nbsp;&nbsp;-&nbsp;&nbsp;&nbsp;&nbsp;"
        if d > 0.001:
            return f"<span style='color:#FF3333'>{d:+.3f}</span>"
        elif d < -0.001:
            return f"<span style='color:#33FF33'>{d:+.3f}</span>"
        else:
            return f"<span style='color:#33FF33'>0.000</span>"

    delta_lap_str = format_delta_mono(lap_delta)

    # ── ESTADO INICIAL (PARA LA CARGA ANTES DE DAR PLAY) ───────────────────
    idx_init = frame_indices[0] if len(frame_indices) > 0 else 0
    t_init = t_sec[idx_init] if len(t_sec) > 0 else 0
    timer_init_str = f"{int(t_init // 60)}:{t_init % 60:06.3f}"
    
    hud_text_init = (
        f"<b>{driver}</b>  [{team}]<br>"
        f"TYRES : {compound}  age {tyre_age} vueltas<br>"
        f"LAP   : {lap_time_str}<br>"
        f"────────────────────<br>"
        f"SPEED : 0.0 km/h<br>"
        f"RPM   : 0<br>"
        f"GEAR  : 0<br>"
        f"DRS   : CLOSED<br>"
        f"THROT : 0.0 %<br>"
        f"BRAKE : 0.0 %"
    )

    fom_hud_init = (
        f"<span style='font-size:13px; color:#AAAAAA'>LAP TIME</span><br><br>"
        f"<span style='font-size:28px; color:{F1_WHITE}'><b>{timer_init_str}</b></span><br>"
        f"<span style='font-size:16px'>&nbsp;&nbsp;&nbsp;&nbsp;-&nbsp;&nbsp;&nbsp;&nbsp;</span><br><br><br>"
        f"<span style='font-size:13px; color:#AAAAAA'>"
        f"&nbsp;&nbsp;&nbsp;&nbsp;<b>S1</b>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<b>S2</b>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<b>S3</b><br></span>"
        f"<span style='font-size:15px'>&nbsp;&nbsp;&nbsp;&nbsp;-&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-&nbsp;&nbsp;&nbsp;&nbsp;</span>"
    )

    # ── FRAMES ─────────────────────────────────────────────────────────────
    frames = []

    for i, idx in enumerate(frame_indices):
        idx = min(idx, N - 1)

        current_time = t_sec[idx]
        mins = int(current_time // 60)
        secs = current_time % 60
        timer_str = f"{mins}:{secs:06.3f}"

        disp_s1 = format_delta_mono(d_s1) if idx >= s1_idx else "&nbsp;&nbsp;&nbsp;&nbsp;-&nbsp;&nbsp;&nbsp;&nbsp;"
        disp_s2 = format_delta_mono(d_s2) if idx >= s2_idx else "&nbsp;&nbsp;&nbsp;&nbsp;-&nbsp;&nbsp;&nbsp;&nbsp;"
        disp_s3 = format_delta_mono(d_s3) if idx >= (N-2) else "&nbsp;&nbsp;&nbsp;&nbsp;-&nbsp;&nbsp;&nbsp;&nbsp;"
        disp_lap_delta = delta_lap_str if idx >= (N-2) else "&nbsp;&nbsp;&nbsp;&nbsp;-&nbsp;&nbsp;&nbsp;&nbsp;"

        fom_hud = (
            f"<span style='font-size:13px; color:#AAAAAA'>LAP TIME</span><br><br>"
            f"<span style='font-size:28px; color:{F1_WHITE}'><b>{timer_str}</b></span><br>"
            f"<span style='font-size:16px'>{disp_lap_delta}</span><br><br><br>"
            f"<span style='font-size:13px; color:#AAAAAA'>"
            f"&nbsp;&nbsp;&nbsp;&nbsp;<b>S1</b>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<b>S2</b>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<b>S3</b><br></span>"
            f"<span style='font-size:15px'>{disp_s1}&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{disp_s2}&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{disp_s3}</span>"
        )

        trail_x = x[0:idx].tolist()
        trail_y = y[0:idx].tolist()

        cur_dist = float(dist_array[idx])
        v_min    = max(0.0, cur_dist - SCROLL_WINDOW_M)
        v_max    = min(dist_max, cur_dist + SCROLL_WINDOW_M)

        v_kmh    = float(vel[idx])
        r_rpm    = float(rpm[idx])
        g_gear   = int(gear[idx])
        drs_stat = "OPEN" if drs[idx] > 8 else "CLOSED"
        thr_pct  = float(throttle[idx])
        brk_pct  = float(brake_pct[idx])

        hud_text = (
            f"<b>{driver}</b>  [{team}]<br>"
            f"TYRES : {compound}  age {tyre_age} vueltas<br>"
            f"LAP   : {lap_time_str}<br>"
            f"────────────────────<br>"
            f"SPEED : {v_kmh:6.1f} km/h<br>"
            f"RPM   : {r_rpm:6.0f}<br>"
            f"GEAR  : {g_gear}<br>"
            f"DRS   : {drs_stat}<br>"
            f"THROT : {thr_pct:5.1f} %<br>"
            f"BRAKE : {brk_pct:5.1f} %"
        )

        frame_data = [
            go.Scattergl(x=trail_x, y=trail_y),
            go.Scatter(x=[float(x[idx])], y=[float(y[idx])]),
            go.Scatter(x=[cur_dist, cur_dist], y=[0, 105]),
            go.Scatter(x=[cur_dist, cur_dist], y=[0, 105]),
        ]

        frame_layout = {
            "xaxis2": {"range": [v_min, v_max], "showgrid": False, "showticklabels": False, "zeroline": False},
            "xaxis3": {"range": [v_min, v_max], "showgrid": False, "showticklabels": False, "zeroline": False},
            "annotations": [
                dict(
                    x=0.14, y=1, xref="paper", yref="paper", 
                    text=hud_text, showarrow=False, align="left",
                    font=dict(family=MONO_FONT, size=15, color=F1_WHITE),
                    bgcolor="rgba(0,0,0,0)", borderwidth=0, borderpad=8,
                ),
                dict(
                    x=0.98, y=0.98, xref="paper", yref="paper", 
                    text=fom_hud, showarrow=False, align="center",
                    font=dict(family=MONO_FONT),
                    bgcolor="rgba(0,0,0,0)", borderwidth=0,          
                )
            ],
        }

        frames.append(go.Frame(data=frame_data, layout=frame_layout, traces=[9, 10, 11, 12], name=str(i)))

    fig.frames = frames

    # ── Slider y botones de reproducción ──────────────────────────────────
    frame_duration_ms = int((lap_time_seg * 1000) / n_frames)

    slider_steps = []
    step_every   = max(1, n_frames // 80)
    for i in range(0, n_frames, step_every):
        idx_i   = frame_indices[min(i, n_frames - 1)]
        dist_i  = float(dist_array[min(idx_i, N - 1)])
        slider_steps.append(dict(
            args=[[str(i)], {"frame": {"duration": frame_duration_ms, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}],
            label=f"{dist_i/1000:.1f}km", method="animate",
        ))

    updatemenus = [dict(
        type="buttons", showactive=False, y=1.06, x=0.0, xanchor="left", yanchor="top", pad=dict(t=0, r=10),
        buttons=[
            dict(label="▶ PLAY", method="animate", args=[None, {"frame": {"duration": frame_duration_ms, "redraw": False}, "fromcurrent": True, "transition": {"duration": 0}}]),
            dict(label="⏸ PAUSE", method="animate", args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}]),
        ],
    )]

    sliders = [dict(
        active=0, transition=dict(duration=0), pad=dict(b=10, t=5), len=1.0, x=0.0, y=0.0, steps=slider_steps,
        bgcolor="rgba(255,255,255,0.05)", bordercolor="rgba(255,255,255,0.1)", tickcolor="rgba(255,255,255,0.3)",
        font=dict(size=8, color="rgba(255,255,255,0.4)", family=MONO_FONT),
        currentvalue=dict(prefix="Dist: ", visible=True, font=dict(size=10, color="rgba(255,255,255,0.5)", family=MONO_FONT)),
    )]

    # ── Layout global ──────────────────────────────────────────────────────
    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)
    margen_x = (x_max - x_min) * 0.02
    margen_y = (y_max - y_min) * 0.02

    axis_panel = dict(showgrid=False, zeroline=False, showticklabels=False, linecolor="rgba(136,136,160,0.3)", linewidth=1, showline=True)

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
        xaxis=dict(range=[x_min - margen_x, x_max + margen_x], showgrid=False, zeroline=False, visible=False),
        yaxis=dict(range=[y_min - margen_y, y_max + margen_y], showgrid=False, zeroline=False, visible=False, scaleanchor="x", scaleratio=1),
        xaxis2={**axis_panel, "range": [0, dist_max]},
        yaxis2=dict(range=[0, 105], showgrid=True, gridcolor="rgba(255,255,255,0.05)", zeroline=False, showticklabels=False, title=dict(text="THROT", font=dict(size=9, color="#39FF14"))),
        xaxis3={**axis_panel, "range": [0, dist_max]},
        yaxis3=dict(range=[0, 105], showgrid=True, gridcolor="rgba(255,255,255,0.05)", zeroline=False, showticklabels=False, title=dict(text="BRAKE", font=dict(size=9, color=F1_RED))),
        # Anotaciones iniciales (para que se vean antes de dar al Play)
        annotations=[
            dict(
                x=0.14, y=1, xref="paper", yref="paper", 
                text=hud_text_init, showarrow=False, align="left",
                font=dict(family=MONO_FONT, size=15, color=F1_WHITE),
                bgcolor="rgba(0,0,0,0)", bordercolor=coche_color, borderwidth=0, borderpad=8,
            ),
            dict(
                x=0.98, y=0.98, xref="paper", yref="paper", 
                text=fom_hud_init, showarrow=False, align="center",
                font=dict(family=MONO_FONT),
                bgcolor="rgba(0,0,0,0)", borderwidth=0,          
            )
        ]
    )

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# PUNTO DE ENTRADA PÚBLICO
# ─────────────────────────────────────────────────────────────────────────────

def render_interactive_sim(
    sesion: Session,
    driver_code: str,
    show_corners: bool,
    lap_number: int,
) -> None:

    with st.spinner(f"Preparando simulación — {driver_code} · Vuelta {lap_number}"):

        # ── 1. Telemetría ─────────────────────────────────────────────────
        try:
            vueltas_piloto = sesion.laps.pick_drivers(driver_code)
            pole           = vueltas_piloto[vueltas_piloto["LapNumber"] == lap_number].iloc[0]
            telemetria     = pole.get_telemetry().add_distance()
        except Exception as e:
            st.error(f"❌ No se pudo cargar la telemetría: {e}")
            return

        driver       = str(pole["Driver"])
        team         = str(pole["Team"])
        compound     = str(pole.get("Compound", "UNKNOWN")).upper()
        tyre_age     = int(pole["TyreLife"]) if pd.notna(pole.get("TyreLife")) else 0
        lap_time_str = _fmt_time(pole.get("LapTime"))
        lap_time_seg = pole.get("LapTime").total_seconds()

        # ── LÓGICA FOM (TIEMPOS, COLORES Y DELTAS) ───────────────────────
        t_sec = telemetria['Time'].dt.total_seconds().values
        
        s1_td = pole['Sector1Time']
        s2_td = pole['Sector2Time']
        s3_td = pole['Sector3Time']

        s1_end = s1_td.total_seconds() if pd.notnull(s1_td) else 0
        s2_end = s1_end + (s2_td.total_seconds() if pd.notnull(s2_td) else 0)

        s1_idx = np.searchsorted(t_sec, s1_end)
        s2_idx = np.searchsorted(t_sec, s2_end)

        driver_best = vueltas_piloto.pick_fastest()
        overall_best = sesion.laps.pick_fastest()

        def get_sector_color(lap_sec, drv_sec, all_sec):
            if pd.isnull(lap_sec): return "⬛"
            if pd.notnull(all_sec) and lap_sec <= all_sec: return "🟪"
            if pd.notnull(drv_sec) and lap_sec <= drv_sec: return "🟩"
            return "🟨"

        c_s1 = get_sector_color(s1_td, driver_best.get('Sector1Time'), overall_best.get('Sector1Time'))
        c_s2 = get_sector_color(s2_td, driver_best.get('Sector2Time'), overall_best.get('Sector2Time'))
        c_s3 = get_sector_color(s3_td, driver_best.get('Sector3Time'), overall_best.get('Sector3Time'))

        if pd.notnull(pole.get('LapTime')) and pd.notnull(driver_best.get('LapTime')):
            lap_delta = pole['LapTime'].total_seconds() - driver_best['LapTime'].total_seconds()
        else:
            lap_delta = None

        def get_sector_delta(lap_sec, drv_sec):
            if pd.notnull(lap_sec) and pd.notnull(drv_sec):
                return lap_sec.total_seconds() - drv_sec.total_seconds()
            return None

        d_s1 = get_sector_delta(s1_td, driver_best.get('Sector1Time'))
        d_s2 = get_sector_delta(s2_td, driver_best.get('Sector2Time'))
        d_s3 = get_sector_delta(s3_td, driver_best.get('Sector3Time'))


        # ── 2. Arrays de telemetría ───────────────────────────────────────
        x         = np.array(telemetria["X"].values,        dtype=float)
        y         = np.array(telemetria["Y"].values,        dtype=float)
        dist_arr  = np.array(telemetria["Distance"].values, dtype=float)
        vel       = np.array(telemetria["Speed"].values,    dtype=float)
        rpm       = np.array(telemetria["RPM"].values,      dtype=float)
        gear      = np.array(telemetria["nGear"].values,    dtype=float)
        throttle  = np.array(telemetria["Throttle"].values, dtype=float)
        drs       = np.array(telemetria["DRS"].values,      dtype=float)
        brake_raw = np.array(telemetria["Brake"].values)
        brake_pct = _normalize_brake(brake_raw)

        # ── 3. Color de equipo ─────────────────────────────────────────────
        try:
            driver_info  = sesion.get_driver(driver)
            coche_color  = f"#{driver_info['TeamColor']}"
        except Exception:
            coche_color  = F1_RED

        # ── 4. Circuit info ────────────────────────────────────────────────
        try:
            circuit_info = sesion.get_circuit_info()
        except Exception:
            circuit_info = None

    # ── 5. Construir figura Plotly ────────────────────────────────────────
    fig = _build_animated_figure(
        x=x, y=y, dist_array=dist_arr,
        vel=vel, throttle=throttle, brake_pct=brake_pct,
        rpm=rpm, gear=gear, drs=drs,
        circuit_info=circuit_info,
        show_corners=show_corners,
        coche_color=coche_color,
        driver=driver, team=team,
        compound=compound, tyre_age=tyre_age,
        lap_time_str=lap_time_str,
        lap_time_seg=lap_time_seg,
        t_sec=t_sec, s1_idx=s1_idx, s2_idx=s2_idx,
        c_s1=c_s1, c_s2=c_s2, c_s3=c_s3,
        lap_delta=lap_delta, d_s1=d_s1, d_s2=d_s2, d_s3=d_s3,
    )

    st.plotly_chart(fig, use_container_width=True, config={
        "displayModeBar"         : True,
        "displaylogo"            : False,
        "modeBarButtonsToRemove" : ["select2d", "lasso2d", "autoScale2d"],
        "toImageButtonOptions"   : {
            "format"  : "png",
            "filename": f"talos_sim_{driver}_lap{lap_number}",
            "scale"   : 2,
        },
    })
