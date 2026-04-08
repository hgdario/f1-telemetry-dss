import streamlit as st
import streamlit.components.v1 as components
import fastf1
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.gridspec as gridspec
import numpy as np
import matplotlib as mpl

def render_3d_circuit(session, driver_code):
    # 1. Aumentamos el límite de memoria para animaciones (Adiós al error de 20MB)
    mpl.rcParams['animation.embed_limit'] = 100.0 # Subimos a 100MB

    # 2. Carga de datos de alta precisión
    with st.spinner(f"Procesando telemetría de alta resolución para {driver_code}..."):
        lap = session.laps.pick_driver(driver_code).pick_fastest()
        telemetry = lap.get_telemetry()
    
    x = telemetry['X'].values
    y = telemetry['Y'].values
    dist = telemetry['Distance'].values
    vel = telemetry['Speed'].values
    
    # Parámetros de la animación para que sea fluida
    N = len(x)
    FPS = 25
    duracion_seg = lap['LapTime'].total_seconds()
    total_frames = int(duracion_seg * FPS)
    Salto = max(1, N // total_frames)
    final_frames = N // Salto

    # 3. Gráfico estilo Dashboard F1
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(15, 8), facecolor='black')
    gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1])
    
    # Mapa 2D con alta resolución
    ax_map = fig.add_subplot(gs[0])
    ax_map.axis('off')
    # Dibujamos el trazado completo de fondo en gris oscuro
    ax_map.plot(x, y, color='#333333', linewidth=2, alpha=0.7) 
    
    # Elementos dinámicos (Punto más pequeño, estela más fina)
    point, = ax_map.plot([], [], 'ro', markersize=4, zorder=5)
    line, = ax_map.plot([], [], color='red', linewidth=1.5, alpha=0.9, zorder=4)
    
    # Gráfica de velocidad (HUD inferior)
    ax_speed = fig.add_subplot(gs[1], facecolor='#111111')
    ax_speed.plot(dist, vel, color='white', linewidth=1, alpha=0.2)
    v_cursor = ax_speed.axvline(x=0, color='red', linewidth=1)
    ax_speed.set_ylabel("KM/H", color='gray', fontsize=8)
    ax_speed.tick_params(labelsize=7, colors='gray')

    # 4. Función Update (Sin pérdida de datos)
    def update(frame):
        idx = (frame * Salto)
        if idx >= N: idx = N - 1
        
        point.set_data([x[idx]], [y[idx]])
        line.set_data(x[:idx], y[:idx])
        v_cursor.set_xdata([dist[idx]])
        return point, line, v_cursor

    ani = animation.FuncAnimation(fig, update, frames=final_frames, interval=1000/FPS, blit=True)

    # 5. Renderizado
    html_content = ani.to_jshtml()
    components.html(html_content, height=850, scrolling=False)
    plt.close(fig)