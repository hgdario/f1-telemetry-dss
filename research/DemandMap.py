import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d

# --- PALETA DE INGENIERÍA OFICIAL ---
C_BRAKE = (255, 0, 0)      # Rojo
C_ACCEL = (0, 150, 255)    # Azul
C_LAT   = (255, 220, 0)    # Amarillo
C_BASE  = (60, 60, 65)     # Gris

def _get_dynamic_color(val, vmin, vmax, mode):
    if mode == "G Longitudinal":
        if val < -0.1:
            f = np.clip(val / vmin, 0, 1) if vmin < 0 else 0
            c = [int(C_BRAKE[j]*f + C_BASE[j]*(1-f)) for j in range(3)]
        elif val > 0.1:
            f = np.clip(val / vmax, 0, 1) if vmax > 0 else 0
            c = [int(C_BASE[j]*(1-f) + C_ACCEL[j]*f) for j in range(3)]
        else:
            c = C_BASE
    else:
        f = np.clip((abs(val) - 0) / (vmax - 0), 0, 1) if vmax > 0 else 0
        c = [int(C_BASE[j]*(1-f) + C_LAT[j]*f) for j in range(3)]
    return f'rgb({c[0]},{c[1]},{c[2]})'

def render_demand_map(session):
    st.title("🛰️ Mapa Maestro de Exigencia del Trazado")
    st.write("Análisis físico promediado del Top 10 (Límite del Chasis).")

    # 1. CÁLCULO DE LA MEDIA (TOP 10)
    with st.spinner("Procesando telemetría maestra..."):
        # Obtenemos la vuelta rápida de referencia para saber la distancia total
        ref_lap = session.laps.pick_fastest()
        ref_tel = ref_lap.get_telemetry().add_distance()
        max_dist = ref_tel['Distance'].max()
        
        # Creamos la rejilla de 1000 puntos basada en la distancia real
        dist_grid = np.linspace(0, max_dist, 1000)
        
        top_10 = session.results.head(10)['Abbreviation'].tolist()
        all_lon, all_lat = [], []
        
        for drv in top_10:
            try:
                # Cargamos la mejor vuelta de cada uno de los top 10
                t = session.laps.pick_driver(drv).pick_fastest().get_telemetry().add_distance()
                v = t['Speed'].values / 3.6
                time = t['Time'].dt.total_seconds().values
                
                # Cálculo de Gs
                gl = np.gradient(v, time) / 9.81
                # G Lateral promediada por dirección
                dy = np.gradient(t['Y'])
                dx = np.gradient(t['X'])
                heading = np.unwrap(np.arctan2(dy, dx))
                gt = (v * np.gradient(heading, time)) / 9.81
                
                # Interpolación para que todos los pilotos "midan" lo mismo
                all_lon.append(interp1d(t['Distance'], gl, fill_value="extrapolate")(dist_grid))
                all_lat.append(interp1d(t['Distance'], gt, fill_value="extrapolate")(dist_grid))
            except:
                continue

        # Promedios y suavizado
        avg_lon = gaussian_filter1d(np.mean(all_lon, axis=0), sigma=4)
        avg_lat = gaussian_filter1d(np.mean(all_lat, axis=0), sigma=4)
        avg_tot = np.sqrt(avg_lon**2 + avg_lat**2)
        
        # Mapeamos las coordenadas X, Y del piloto de referencia a nuestra rejilla
        map_x = interp1d(ref_tel['Distance'], ref_tel['X'], fill_value="extrapolate")(dist_grid)
        map_y = interp1d(ref_tel['Distance'], ref_tel['Y'], fill_value="extrapolate")(dist_grid)

    # 2. RENDERIZADO
    mode = st.radio("Analizar Vector de Fuerza (Promedio):", ["G Longitudinal", "G Lateral", "G Total"], horizontal=True)
    data = {"G Longitudinal": avg_lon, "G Lateral": avg_lat, "G Total": avg_tot}[mode]
    vmin, vmax = data.min(), data.max()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=map_x, y=map_y, mode='lines', line=dict(color='#111', width=12), hoverinfo='skip'))

    # Segmentación para evitar la "telaraña"
    N_BINS = 40 
    bins = np.linspace(vmin, vmax, N_BINS + 1)
    for i in range(N_BINS):
        mask = (data[:-1] >= bins[i]) & (data[:-1] < bins[i+1])
        if not mask.any(): continue
        v_mid = (bins[i] + bins[i+1]) / 2
        color = _get_dynamic_color(v_mid, vmin, vmax, mode)
        
        xs, ys = [], []
        for idx in np.where(mask)[0]:
            xs += [map_x[idx], map_x[idx+1], None]
            ys += [map_y[idx], map_y[idx+1], None]
            
        fig.add_trace(go.Scatter(x=xs, y=ys, mode='lines', line=dict(color=color, width=6), hoverinfo='skip', showlegend=False))

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        height=750, margin=dict(l=0,r=0,t=0,b=0),
        xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x", scaleratio=1)
    )
    st.plotly_chart(fig, use_container_width=True)

    # 3. MÉTRICAS DE SEVERIDAD
    st.subheader("📋 Resumen de Exigencia")
    c1, c2 = st.columns(2)
    c1.metric("G Lateral Máx (Promedio)", f"{np.max(np.abs(avg_lat)):.2f} G")
    c2.metric("G Frenada Máx (Promedio)", f"{np.min(avg_lon):.2f} G")