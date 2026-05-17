"""
TeamTelemetry.py — TALOS F1 Team Telemetry System
=================================================
Módulo que replica la interfaz SAP de Mercedes McLaren,
mostrando telemetría y estado de los dos pilotos de un equipo
de forma simétrica.
"""

from __future__ import annotations

import os
import base64
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from fastf1.core import Session

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES DE DISEÑO
# ─────────────────────────────────────────────────────────────────────────────
BG_DARK       = "#0E0E0F"
BG_SURFACE    = "#1A1A1F"
BG_PANEL      = "#111115"
F1_WHITE      = "#FFFFFF"
F1_RED        = "#E8002D"
ACCENT_AMBER  = "#FFA500"
ACCENT_CYAN   = "#00D2FF"

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

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _format_laptime(td: pd.Timedelta) -> str:
    if pd.isna(td):
        return "—"
    try:
        total_seconds = td.total_seconds()
        minutes   = int(total_seconds // 60)
        seconds   = int(total_seconds % 60)
        millis    = int(round((total_seconds % 1) * 1000))
        if minutes > 0:
            return f"{minutes:01d}:{seconds:02d}.{millis:03d}"
        return f"{seconds:02d}.{millis:03d}"
    except Exception:
        return "—"

def _get_team_color(session: Session, driver: str) -> str:
    try:
        return f"#{session.get_driver(driver)['TeamColor']}"
    except Exception:
        return F1_WHITE

def _get_full_name(session: Session, driver: str) -> str:
    try:
        info = session.get_driver(driver)
        return f"{info.get('FirstName','')} {info.get('LastName', driver)}".strip()
    except Exception:
        return driver

def _get_teams_and_drivers(session: Session) -> dict:
    """Extrae un diccionario de {TeamName: [Driver1, Driver2]}"""
    teams = {}
    try:
        results = session.results
        for _, row in results.iterrows():
            team = row.get("TeamName")
            driver = row.get("Abbreviation")
            if pd.notna(team) and pd.notna(driver):
                if team not in teams:
                    teams[team] = []
                if len(teams[team]) < 2:
                    teams[team].append(driver)
    except Exception:
        pass
    return {k: v for k, v in teams.items() if len(v) == 2}

def _get_team_car_image(team_name: str, color_hex: str) -> str:
    """Retorna la imagen del coche para el equipo (PNG personalizado si existe, sino SVG colorizado)."""
    # Mapeo de equipos a archivos PNG personalizados
    TEAM_CAR_IMAGES = {
        "Mercedes": "mercedes.png",
        "Ferrari": "ferrari.png",
        "McLaren": "mclaren.png",
        "Red Bull Racing": "redbull.png",
        "Aston Martin": "astonmartin.png",
    }

    # assets/ está en src/, no en src/comparativas/ → subir un nivel
    assets_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets")

    # Intentar cargar PNG personalizado si existe para este equipo
    png_filename = TEAM_CAR_IMAGES.get(team_name)
    if png_filename:
        png_path = os.path.join(assets_dir, png_filename)
        if os.path.exists(png_path):
            try:
                with open(png_path, "rb") as f:
                    png_b64 = base64.b64encode(f.read()).decode('utf-8')
                return f"data:image/png;base64,{png_b64}"
            except Exception:
                pass

    # Fallback: SVG colorizado
    svg_path = os.path.join(assets_dir, "IABFx01.svg")
    try:
        with open(svg_path, "r", encoding="utf-8") as f:
            svg_content = f.read()
        svg_content = svg_content.replace('fill="#000000"', f'fill="{color_hex}"')
        svg_b64 = base64.b64encode(svg_content.encode('utf-8')).decode('utf-8')
        return f"data:image/svg+xml;base64,{svg_b64}"
    except Exception as e:
        return ""

# ─────────────────────────────────────────────────────────────────────────────
# EXTRACCIÓN DE DATOS
# ─────────────────────────────────────────────────────────────────────────────

def _extract_lap_data(session: Session, driver: str, lap_number: int) -> dict:
    """Extrae métricas generales de la vuelta seleccionada."""
    data = {
        "lap_time": None,
        "best_lap": None,
        "vmax": 0.0,
        "compound": "UNKNOWN",
        "tyre_life": 0,
        "position": "—",
        "number": driver,
        "throttle_avg": 0.0,
        "brake_avg": 0.0,
        "gear_avg": 0.0,
        "drs": "OFF",
        "valid": False
    }
    
    try:
        info = session.get_driver(driver)
        data["number"] = info.get("DriverNumber", driver)
        pos = info.get("Position")
        data["position"] = str(int(pos)) if pd.notna(pos) else "—"
        
        laps = session.laps.pick_drivers(driver)
        if laps.empty: return data
        
        # Mejor vuelta de la sesión para el piloto
        fastest_lap = laps.pick_fastest()
        data["best_lap"] = fastest_lap.get("LapTime")
        
        # Vuelta específica
        lap_rows = laps[laps["LapNumber"] == lap_number]
        if lap_rows.empty: return data
        
        lap = lap_rows.iloc[0]
        data["lap_time"] = lap.get("LapTime")
        
        compound = lap.get("Compound")
        data["compound"] = str(compound).upper() if pd.notna(compound) else "UNKNOWN"
        
        tyre_life = lap.get("TyreLife")
        data["tyre_life"] = int(tyre_life) if pd.notna(tyre_life) else 0
        
        fresh = lap.get("FreshTyre")
        data["fresh"] = "NEW" if fresh == True else ("USED" if fresh == False else "")
        
        data["s1"] = lap.get("Sector1Time")
        data["s2"] = lap.get("Sector2Time")
        data["s3"] = lap.get("Sector3Time")
        
        # Telemetría
        tel = lap.get_telemetry()
        if not tel.empty:
            data["valid"] = True
            
            if "Speed" in tel.columns:
                data["vmax"] = float(tel["Speed"].max())
            
            if "Throttle" in tel.columns:
                data["throttle_avg"] = float(tel["Throttle"].mean())
                
            if "Brake" in tel.columns:
                b = tel["Brake"]
                if b.dtype == bool or set(b.dropna().unique()).issubset({0, 1, True, False}):
                    data["brake_avg"] = float((b.astype(float) * 100).mean())
                else:
                    data["brake_avg"] = float(b.mean() * 100 if b.max() <= 1.0 else b.mean())
                    
            if "nGear" in tel.columns:
                data["gear_avg"] = float(tel["nGear"].mean())
                
            if "DRS" in tel.columns:
                drs_active = tel["DRS"] >= 10
                data["drs"] = "ON" if drs_active.any() else "OFF"
                
    except Exception as e:
        print(f"Error extracting data for {driver} L{lap_number}: {e}")
        
    return data

def _build_circuit_map(session: Session, drv1: str, drv2: str, t_start: pd.Timestamp, t_end: pd.Timestamp) -> go.Figure:
    """Dibuja el mapa del circuito animado con ambos pilotos en una ventana de tiempo absoluta."""
    fig = go.Figure()
    
    try:
        laps_drv1 = session.laps.pick_drivers(drv1)
        laps_drv2 = session.laps.pick_drivers(drv2)
        
        # Filtrar vueltas cercanas para no cargar la telemetría de toda la carrera
        window_start = t_start - pd.Timedelta(minutes=3)
        window_end = t_end + pd.Timedelta(minutes=3)
        
        valid_laps1 = laps_drv1[(laps_drv1['Time'] >= window_start) & (laps_drv1['Time'] <= window_end)]
        valid_laps2 = laps_drv2[(laps_drv2['Time'] >= window_start) & (laps_drv2['Time'] <= window_end)]
        
        if valid_laps1.empty and valid_laps2.empty:
            return fig
            
        tel1 = valid_laps1.get_telemetry().slice_by_time(t_start, t_end) if not valid_laps1.empty else pd.DataFrame()
        tel2 = valid_laps2.get_telemetry().slice_by_time(t_start, t_end) if not valid_laps2.empty else pd.DataFrame()
        
        color1 = _get_team_color(session, drv1)
        color2 = "#FFFFFF" 
        
        # Tiempos en segundos relativos al inicio de la ventana
        t1_sec = (tel1["SessionTime"] - t_start).dt.total_seconds().values if not tel1.empty and "SessionTime" in tel1.columns else np.array([])
        t2_sec = (tel2["SessionTime"] - t_start).dt.total_seconds().values if not tel2.empty and "SessionTime" in tel2.columns else np.array([])
        
        max_t = (t_end - t_start).total_seconds()
        t_grid = np.linspace(0, max_t, 150)
        
        # Interpolar (usamos valores constantes en los bordes si falta data)
        x1_anim = np.interp(t_grid, t1_sec, tel1['X'].values, left=tel1['X'].values[0], right=tel1['X'].values[-1]) if len(t1_sec) else np.zeros_like(t_grid)
        y1_anim = np.interp(t_grid, t1_sec, tel1['Y'].values, left=tel1['Y'].values[0], right=tel1['Y'].values[-1]) if len(t1_sec) else np.zeros_like(t_grid)
        
        x2_anim = np.interp(t_grid, t2_sec, tel2['X'].values, left=tel2['X'].values[0], right=tel2['X'].values[-1]) if len(t2_sec) else np.zeros_like(t_grid)
        y2_anim = np.interp(t_grid, t2_sec, tel2['Y'].values, left=tel2['Y'].values[0], right=tel2['Y'].values[-1]) if len(t2_sec) else np.zeros_like(t_grid)
        
        # Coordenadas base para el circuito estático (usamos la vuelta más rápida de la sesión para tener el circuito completo perfecto)
        fastest = session.laps.pick_fastest()
        base_tel = fastest.get_telemetry()
        x_base = base_tel['X'].values
        y_base = base_tel['Y'].values
        
        # Lógica de colores por Banderas (TrackStatus)
        track_color = "rgba(255,255,255,0.2)"
        try:
            ts = session.track_status
            ts_active = ts[ts['Time'] <= t_end]
            if not ts_active.empty:
                last_before = ts[ts['Time'] <= t_start]
                ts_relevant = ts_active[ts_active['Time'] >= t_start]
                if not last_before.empty:
                    ts_relevant = pd.concat([last_before.tail(1), ts_relevant])
                
                s_vals = ts_relevant['Status'].astype(str).tolist()
                
                # Prioridad de color: Red > SC > VSC > Yellow
                if any('5' in s for s in s_vals): # Red flag
                    track_color = "#E8002D"
                elif any('4' in s for s in s_vals): # Safety Car
                    track_color = "#FF8C00" # Dark Orange
                elif any('6' in s or '7' in s for s in s_vals): # VSC
                    track_color = "#FFA500" # Orange
                elif any('2' in s for s in s_vals): # Yellow
                    track_color = "#FFE119" # Yellow
        except Exception:
            pass

        fig.add_trace(go.Scatter(
            x=x_base, y=y_base,
            mode="lines",
            line=dict(color=track_color, width=4),
            hoverinfo="skip",
            showlegend=False
        ))
        
        # Trace estático: Línea de meta
        if len(x_base) > 1:
            dx = x_base[1] - x_base[0]
            dy = y_base[1] - y_base[0]
            mag = np.sqrt(dx**2 + dy**2) or 1
            nx, ny = -dy / mag, dx / mag
            meta_w = 400
            fig.add_trace(go.Scatter(
                x=[x_base[0] - nx * meta_w, x_base[0] + nx * meta_w],
                y=[y_base[0] - ny * meta_w, y_base[0] + ny * meta_w],
                mode="lines",
                line=dict(color=F1_RED, width=3),
                hoverinfo="skip",
                showlegend=False
            ))
            
        # Traces para los coches
        fig.add_trace(go.Scatter(
            x=[x1_anim[0]], y=[y1_anim[0]], mode="markers",
            marker=dict(size=12, color=color1, symbol="circle", line=dict(color=F1_WHITE, width=1.5)),
            name=drv1
        ))
        
        fig.add_trace(go.Scatter(
            x=[x2_anim[0]], y=[y2_anim[0]], mode="markers",
            marker=dict(size=12, color=color2, symbol="circle", line=dict(color=BG_DARK, width=1.5)),
            name=drv2
        ))
        
        # Frames de animación
        frames = []
        for i in range(len(t_grid)):
            frames.append(go.Frame(
                data=[
                    go.Scatter(x=[x1_anim[i]], y=[y1_anim[i]]),
                    go.Scatter(x=[x2_anim[i]], y=[y2_anim[i]])
                ],
                traces=[2, 3],
                name=str(i)
            ))
        fig.frames = frames
        
        # Controles
        frame_ms = int((max_t * 1000) / len(t_grid))
        
        slider_steps = []
        for i in range(len(t_grid)):
            slider_steps.append(dict(
                args=[[str(i)], {"frame": {"duration": frame_ms, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}],
                label=f"{t_grid[i]:.1f}s", method="animate",
            ))

        updatemenus = [dict(
            type="buttons", showactive=False,
            x=0.5, y=-0.15, xanchor="center", yanchor="top",
            direction="left",
            buttons=[
                dict(label="▶ PLAY", method="animate", args=[None, {"frame": {"duration": frame_ms, "redraw": False}, "fromcurrent": True, "transition": {"duration": 0}}]),
                dict(label="⏸ PAUSE", method="animate", args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}]),
            ],
            font=dict(color=F1_WHITE, size=10)
        )]

        sliders = [dict(
            active=0, transition=dict(duration=0), pad=dict(t=30),
            x=0.0, y=-0.25, len=1.0, steps=slider_steps,
            font=dict(color=F1_WHITE, size=9),
            currentvalue=dict(prefix="Time: ", font=dict(color=F1_WHITE, size=10)),
        )]
        
        mid_x, mid_y = np.mean(x_base), np.mean(y_base)
        range_xy = max(x_base.max()-x_base.min(), y_base.max()-y_base.min()) / 1.8

        fig.update_layout(
            height=600,
            margin=dict(l=0, r=0, t=20, b=80),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            updatemenus=updatemenus,
            sliders=sliders,
            dragmode="pan",
            xaxis=dict(range=[mid_x-range_xy, mid_x+range_xy],
                       scaleanchor="y", scaleratio=1,
                       showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(range=[mid_y-range_xy, mid_y+range_xy],
                       showgrid=False, zeroline=False, showticklabels=False),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, font=dict(color=F1_WHITE))
        )
    except Exception as e:
        print("Map error:", e)
        
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# UI BUILDERS
# ─────────────────────────────────────────────────────────────────────────────

def _metric_html(label: str, value: str, align: str = "center") -> str:
    """Genera un métrico compacto en HTML para evitar los cortes de texto de st.metric."""
    return f'''
    <div style="text-align:{align}; margin-bottom:15px;">
        <div style="font-size:10px; color:rgba(255,255,255,0.5); letter-spacing:1px; margin-bottom:2px;">{label}</div>
        <div style="font-size:18px; font-weight:600; color:#FFF; font-family:'JetBrains Mono', monospace; line-height:1.2;">{value}</div>
    </div>
    '''

def _render_driver_panel(session: Session, driver: str, lap_n: int, is_left: bool):
    """Renderiza la mitad del panel para un piloto (izquierdo o derecho)."""

    data = _extract_lap_data(session, driver, lap_n)
    color = _get_team_color(session, driver)
    full_name = _get_full_name(session, driver).upper()
    team_name = session.get_driver(driver).get("TeamName", "")
    
    # Extraer primer nombre y apellido para el estilo SAP
    name_parts = full_name.split()
    first_name = name_parts[0] if name_parts else ""
    last_name = " ".join(name_parts[1:]) if len(name_parts) > 1 else driver
    
    # Configuración de alineación HTML (solo para textos pequeños)
    align = "left" if is_left else "right"
    
    # --- FILA 1: CABECERA ---
    if is_left:
        c1, c2 = st.columns([3, 1])
        with c1:
            st.caption("DRIVER")
            st.markdown(f"### <span style='color:{ACCENT_CYAN}; font-weight:300;'>{first_name}</span> {last_name}", unsafe_allow_html=True)
            st.markdown(f"<span style='background:{color}; color:#fff; padding:2px 8px; border-radius:2px; font-size:12px; font-weight:bold;'>{data['number']}</span>", unsafe_allow_html=True)
        with c2:
            st.caption("POS & LAP")
            st.markdown(f"<h1 style='margin:0; padding:0; line-height:1;'>P{data['position']} <span style='font-size:16px; color:rgba(255,255,255,0.4);'>L{lap_n}</span></h1>", unsafe_allow_html=True)
    else:
        c1, c2 = st.columns([1, 3])
        with c1:
            st.caption("POS & LAP")
            st.markdown(f"<h1 style='margin:0; padding:0; line-height:1;'>P{data['position']} <span style='font-size:16px; color:rgba(255,255,255,0.4);'>L{lap_n}</span></h1>", unsafe_allow_html=True)
        with c2:
            st.markdown(f"<div style='text-align:right;'>", unsafe_allow_html=True)
            st.caption("DRIVER")
            st.markdown(f"<div style='text-align:right;'><h3><span style='color:{ACCENT_CYAN}; font-weight:300;'>{first_name}</span> {last_name}</h3></div>", unsafe_allow_html=True)
            st.markdown(f"<div style='text-align:right;'><span style='background:{color}; color:#fff; padding:2px 8px; border-radius:2px; font-size:12px; font-weight:bold;'>{data['number']}</span></div>", unsafe_allow_html=True)
            st.markdown(f"</div>", unsafe_allow_html=True)
            
    st.divider()

    # --- FILA 2: SVG/PNG + TIEMPOS ---
    img_src = _get_team_car_image(team_name, color)
    transform = "transform: scaleX(-1);" if not is_left else ""
    img_html = f'<div style="text-align:center; padding: 20px 0;"><img src="{img_src}" width="280px" style="{transform}"></div>'

    st.markdown(img_html, unsafe_allow_html=True)
    
    c_metrics1, c_metrics2, c_metrics3 = st.columns(3)
    with c_metrics1:
        st.markdown(_metric_html("LAP", _format_laptime(data['lap_time'])), unsafe_allow_html=True)
    with c_metrics2:
        st.markdown(_metric_html("BEST", _format_laptime(data['best_lap'])), unsafe_allow_html=True)
    with c_metrics3:
        st.markdown(_metric_html("VMAX", f"{data['vmax']:.0f}"), unsafe_allow_html=True)

    st.divider()

    # --- FILA 3: ENGINE & INPUTS ---
    thr_pct = min(100, max(0, data["throttle_avg"]))
    brk_pct = min(100, max(0, data["brake_avg"]))
    
    st.caption("ENGINE & INPUTS")
    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown(_metric_html("AVG GEAR", f"{data['gear_avg']:.1f}", align="left" if is_left else "right"), unsafe_allow_html=True)
        drs_used = "DRS USED" if data['drs'] == 'ON' else "NO DRS"
        drs_color = ACCENT_CYAN if data['drs'] == 'ON' else 'rgba(255,255,255,0.4)'
        st.markdown(f"<div style='text-align:center; background:rgba(255,255,255,0.1); padding:4px; border-radius:2px; margin-top:5px;'><span style='color:{drs_color}; font-weight:bold; font-size:12px;'>{drs_used}</span></div>", unsafe_allow_html=True)
    with c2:
        st.progress(thr_pct / 100.0, text=f"Throttle Avg {thr_pct:.0f}%")
        st.progress(brk_pct / 100.0, text=f"Brake Avg {brk_pct:.0f}%")

    st.divider()

    # --- FILA 4: TYRES ---
    tyre_color = TYRE_COLORS.get(data["compound"], TYRE_COLORS["UNKNOWN"])
    st.caption("TYRE LIFE")
    c1, c2 = st.columns([1, 3])
    with c1:
        st.markdown(f"""
        <div style='border:2px solid {tyre_color}; border-radius:50%; width:40px; height:40px; display:flex; align-items:center; justify-content:center; margin:auto;'>
            <span style='font-size:16px; font-weight:700;'>{data['tyre_life']}</span>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div style='padding-top:8px;'><span style='color:{tyre_color}; font-weight:700; font-size:16px; letter-spacing:1px;'>{data['compound']}</span> <span style='color:rgba(255,255,255,0.4); font-size:12px; margin-left:10px;'>{data['fresh']}</span></div>", unsafe_allow_html=True)
        
        def fmt_s(t):
            if pd.isna(t): return "—"
            return f"{t.total_seconds():.3f}s"
            
        s1, s2, s3 = fmt_s(data.get('s1')), fmt_s(data.get('s2')), fmt_s(data.get('s3'))
        
        st.markdown(f"""
<div style='display:flex; gap:15px; margin-top:5px; font-size:11px; font-family:"JetBrains Mono", monospace; color:rgba(255,255,255,0.6);'>
    <div>S1 <span style='color:white; font-weight:bold;'>{s1}</span></div>
    <div>S2 <span style='color:white; font-weight:bold;'>{s2}</span></div>
    <div>S3 <span style='color:white; font-weight:bold;'>{s3}</span></div>
</div>
""", unsafe_allow_html=True)

def _render_center_panel(session: Session, team_name: str, drv1: str, lap1: int, drv2: str, lap2: int, corners: bool, t_start: pd.Timestamp, t_end: pd.Timestamp):
    """Renderiza el panel central (clima, circuito)."""
    
    # Extraer clima actual
    air_temp = "—"
    track_temp = "—"
    wind_speed = "—"
    
    try:
        weather = session.weather_data
        if not weather.empty:
            latest = weather.iloc[-1]
            air_temp = f"{latest['AirTemp']:.1f}°C"
            track_temp = f"{latest['TrackTemp']:.1f}°C"
            wind_speed = f"{latest['WindSpeed']:.1f} m/s"
    except Exception:
        pass
        
    st.markdown(f"<div style='text-align:center;'><span style='color:rgba(255,255,255,0.5); letter-spacing:3px; font-size:14px; text-transform:uppercase;'>{session.event.get('EventName', 'F1')}</span></div>", unsafe_allow_html=True)
    st.markdown(f"<div style='text-align:center;'><h2 style='letter-spacing:2px; text-transform:uppercase; margin-top:0;'>{team_name}</h2></div>", unsafe_allow_html=True)
    
    st.divider()
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(_metric_html("AIR TEMP", air_temp), unsafe_allow_html=True)
    with c2:
        st.markdown(_metric_html("TRACK TEMP", track_temp), unsafe_allow_html=True)
    with c3:
        st.markdown(_metric_html("WIND", wind_speed), unsafe_allow_html=True)
    
    fig = _build_circuit_map(session, drv1, drv2, t_start, t_end)
    st.plotly_chart(fig, use_container_width=True, config={
        "displayModeBar": False,
        "scrollZoom": True,
        "displaylogo": False,
        "modeBarButtonsToRemove": ["lasso2d", "select2d"]
    })
    
    # Mensaje de Race Control
    rcm_msg = "TRACK CLEAR"
    try:
        rcm = session.race_control_messages
        if not rcm.empty:
            past_msgs = rcm[rcm['Time'] <= t_end]
            if not past_msgs.empty:
                rcm_msg = past_msgs.iloc[-1]['Message']
    except Exception:
        pass
        
    st.markdown(f"""<div style='text-align:center; padding:15px; background:rgba(255,255,255,0.05); border-radius:4px; margin-top: 10px;'>
    <div style='color:rgba(255,255,255,0.5); font-size:10px; letter-spacing:2px; margin-bottom:5px;'>LATEST RACE CONTROL MESSAGE</div>
    <div style='color:#FFF; font-family:"JetBrains Mono", monospace; font-size:14px; font-weight:bold;'>{rcm_msg}</div>
</div>""", unsafe_allow_html=True)
    
    # Tiempo Acumulado (Race Time) y Gap en la Vuelta de Referencia
    try:
        ld1 = session.laps.pick_drivers(drv1)
        ld2 = session.laps.pick_drivers(drv2)
        
        # Inicio de carrera absoluto para restar el tiempo de formación
        race_start = session.laps[session.laps['LapNumber'] == 1]['LapStartTime'].min()
        
        row1_ref = ld1[ld1["LapNumber"] == lap1]
        row2_ref = ld2[ld2["LapNumber"] == lap1]
        
        t1_total = row1_ref.iloc[0].get('Time') if not row1_ref.empty else None
        t2_total = row2_ref.iloc[0].get('Time') if not row2_ref.empty else None
        
        race_t1 = t1_total - race_start if pd.notna(t1_total) and pd.notna(race_start) else None
        race_t2 = t2_total - race_start if pd.notna(t2_total) and pd.notna(race_start) else None
        
        def fmt_race_time(td):
            if pd.isna(td) or td is None: return "—"
            ts = td.total_seconds()
            h = int(ts // 3600)
            m = int((ts % 3600) // 60)
            s = int(ts % 60)
            ms = int((ts % 1) * 1000)
            if h > 0:
                return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"
            return f"{m:02d}:{s:02d}.{ms:03d}"
            
        t1_str = fmt_race_time(race_t1)
        t2_str = fmt_race_time(race_t2)
        
        # Calcular GAP real en la misma vuelta
        if not row1_ref.empty and not row2_ref.empty:
            gap_total = (t1_total - t2_total).total_seconds()
            if gap_total < 0:
                gap_text = f"◀◀ {drv1} LEADS BY {abs(gap_total):.3f}s"
                gap_col = "#39FF14"
            else:
                gap_text = f"▶▶ {drv2} LEADS BY {abs(gap_total):.3f}s"
                gap_col = "#39FF14"
        else:
            # Uno de los dos no llegó a esta vuelta
            if row1_ref.empty and not row2_ref.empty:
                diff = lap1 - ld1['LapNumber'].max()
                gap_text = f"▶▶ {drv2} LEADS BY {int(diff)} LAP(s)"
                gap_col = "#39FF14"
                t1_str = "LAPPED / DNF"
            elif not row1_ref.empty and row2_ref.empty:
                diff = lap1 - ld2['LapNumber'].max()
                gap_text = f"◀◀ {drv1} LEADS BY {int(diff)} LAP(s)"
                gap_col = "#39FF14"
                t2_str = "LAPPED / DNF"
            else:
                gap_text = "N/A"
                gap_col = "rgba(255,255,255,0.4)"
            
        st.markdown(f"""<div style='display:flex; flex-direction:column; padding:25px 20px; background:rgba(255,255,255,0.05); border-radius:4px; margin-top: 15px; min-height: 120px;'>
<div style='text-align:center; color:rgba(255,255,255,0.5); font-size:10px; letter-spacing:2px; margin-bottom:20px;'>TOTAL RACE TIME & GAP (AT LAP {lap1})</div>
<div style='display:flex; justify-content:space-between; align-items:center;'>
<div style='text-align:left;'>
<div style='color:rgba(255,255,255,0.5); font-size:10px; letter-spacing:1px;'>{drv1} TIME</div>
<div style='color:#FFF; font-family:"JetBrains Mono", monospace; font-size:20px; font-weight:bold;'>{t1_str}</div>
</div>
<div style='text-align:center; flex-grow:1; margin:0 15px;'>
<div style='color:{gap_col}; font-family:"JetBrains Mono", monospace; font-size:14px; font-weight:bold; background:rgba(255,255,255,0.1); padding:6px 12px; border-radius:20px; display:inline-block;'>
{gap_text}
</div>
</div>
<div style='text-align:right;'>
<div style='color:rgba(255,255,255,0.5); font-size:10px; letter-spacing:1px;'>{drv2} TIME</div>
<div style='color:#FFF; font-family:"JetBrains Mono", monospace; font-size:20px; font-weight:bold;'>{t2_str}</div>
</div>
</div>
</div>""", unsafe_allow_html=True)
    except Exception:
        pass

# ─────────────────────────────────────────────────────────────────────────────
# MAIN RENDER FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def render_team_telemetry(session: Session, corners: bool = False):
    # 1. Obtener equipos
    teams_dict = _get_teams_and_drivers(session)
    if not teams_dict:
        st.warning("⚠️ No se encontraron equipos con 2 pilotos en esta sesión.")
        return
        
    team_names = list(teams_dict.keys())
    
    # 2. Selectores superiores (Solo 1 selector de vuelta)
    col_sel1, col_sel2 = st.columns([1, 2])
    with col_sel1:
        selected_team = st.selectbox("Equipo", team_names)
        
    drivers = teams_dict[selected_team]
    drv1, drv2 = drivers[0], drivers[1]
    
    # Obtener laps para el selector basado en drv1
    try:
        laps_drv1 = session.laps.pick_drivers(drv1).pick_accurate()
        laps_list = [int(l) for l in laps_drv1["LapNumber"].tolist()]
        if not laps_list: laps_list = [1]
    except Exception:
        laps_list = [1]
        
    with col_sel2:
        lap = st.selectbox(f"Vuelta de referencia ({drv1})", laps_list)
        
    # Obtener tiempos absolutos de la vuelta seleccionada
    try:
        laps_drv1_all = session.laps.pick_drivers(drv1)
        lap_data1 = laps_drv1_all[laps_drv1_all["LapNumber"] == lap].iloc[0]
        t_end = lap_data1['Time']
        t_start = lap_data1['LapStartTime'] if pd.notna(lap_data1.get('LapStartTime')) else t_end - lap_data1['LapTime']
        if pd.isna(t_start):
            t_start = t_end - pd.Timedelta(seconds=120)
    except Exception:
        st.error("No se pudieron determinar los tiempos de la vuelta.")
        return
        
    # Buscar en qué vuelta estaba drv2 en el instante t_start
    lap2 = lap
    try:
        laps_drv2 = session.laps.pick_drivers(drv2)
        for _, l2 in laps_drv2.iterrows():
            l2_end = l2['Time']
            l2_start = l2['LapStartTime'] if pd.notna(l2.get('LapStartTime')) else l2_end - l2['LapTime']
            if pd.isna(l2_start):
                l2_start = l2_end - pd.Timedelta(seconds=120)
            if pd.notna(l2_start) and pd.notna(l2_end):
                if l2_start <= t_start < l2_end:
                    lap2 = int(l2['LapNumber'])
                    break
    except Exception:
        pass
        
    st.divider()

    # 3. Layout Principal (SAP Style)
    col_l, col_c, col_r = st.columns([3, 4, 3], gap="large")
    
    with col_l:
        _render_driver_panel(session, drv1, lap, is_left=True)
        
    with col_c:
        _render_center_panel(session, selected_team, drv1, lap, drv2, lap2, corners, t_start, t_end)
        
    with col_r:
        _render_driver_panel(session, drv2, lap2, is_left=False)
