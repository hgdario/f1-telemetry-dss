import fastf1
import fastf1.plotting
import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.gridspec as gridspec
from scipy.signal import savgol_filter

def render_interactive_sim(sesion, driver_code, show_corners):
    mpl.rcParams['animation.embed_limit'] = 100.0
    
    Pole = sesion.laps.pick_drivers(driver_code).pick_fastest()
    telemetria = Pole.get_telemetry()
    Driver = Pole['Driver']
    Team = Pole['Team']
    TiempoVuelta = Pole['LapTime']
    TiempoVueltaSeg = TiempoVuelta.total_seconds()

    compound = Pole['Compound']
    tyre_age = Pole['TyreLife']

    vel = np.array(telemetria['Speed'].values)
    rpm = np.array(telemetria['RPM'].values)
    gear = np.array(telemetria['nGear'].values)
    throttle = np.array(telemetria['Throttle'].values) # 0-100
    brake = np.array(telemetria['Brake'].values)       # 0-100
    drs = np.array(telemetria['DRS'].values)           # 8+ es abierto


    print(f"Vuelta cargada: {Driver} - {TiempoVuelta}")

    driver_info = sesion.get_driver(Driver)
    Team_Color = driver_info['TeamColor']

    #Driver color
    try:
        coche_color = f"#{Team_Color}"
    except:
        coche_color = 'purple'

    # RAW Data (Solo X e Y)
    x = np.array(telemetria['X'].values)
    y = np.array(telemetria['Y'].values)

    
    dist_array = np.array(telemetria['Distance'].values)
    Velocidad = np.array(telemetria['Speed'].values)

    circuit_info = sesion.get_circuit_info()

    print(" Pintando el circuito base con la vuelta de referencia...")

    #Animation Params
    FPS = 25
    total_frames = int(TiempoVueltaSeg * FPS)
    N = len(x)
    Salto = max(1, int(N / total_frames))
    final_frames = N // Salto
    Intervalo_Real = (TiempoVueltaSeg * 1000) / final_frames

    # --- Split Screen ---
    fig = plt.figure(figsize=(16, 9), facecolor="#1C1C27")

    #Grid intacto
    gs = gridspec.GridSpec(5, 2, figure=fig, height_ratios=[1,1,1,1,1.5])

    #MAPA 2D
    ax = fig.add_subplot(gs[:5, :], facecolor="#1C1C27")
    ax.axis('off')

    #Centering Cam 2D 
    mid_x, mid_y = np.mean(x), np.mean(y)
    range_x = x.max() - x.min()
    range_y = y.max() - y.min()

    maxRange = max(range_x, range_y) / 1.24
    ax.set_xlim(mid_x - maxRange, mid_x + maxRange)
    ax.set_ylim(mid_y - maxRange * 1.5, mid_y + maxRange * 0.7)

    #Coloring Map
    ax.plot(x, y, color='#50506A', linewidth=6, alpha=0.5)

    distancia_offset = 100
    #Curves numbers
    if show_corners:
        for _, row in circuit_info.corners.iterrows():
            num_curva = row['Number']
            dist_curva = row['Distance']

            # Encontramos el índice de la curva
            idx = (np.abs(dist_array - dist_curva)).argmin()
            
            # Tomamos un rango de puntos para una tangente más suave (evita saltos)
            idx_next = min(idx + 5, len(x) - 1)
            idx_prev = max(idx - 5, 0)
            
            dx = x[idx_next] - x[idx_prev]
            dy = y[idx_next] - y[idx_prev]
            
            # Vector normal perpendicular: (-dy, dx)
            nx, ny = -dy, dx
            
            # Normalizamos el vector
            mag = np.sqrt(nx**2 + ny**2)
            if mag != 0:
                nx /= mag
                ny /= mag
            
            # LÓGICA DE DIRECCIÓN: 
            # Si al sumar el offset nos acercamos al centro del mapa, invertimos el vector
            # Esto asegura que el número siempre salga hacia el EXTERIOR del circuito
            pos_test_x = x[idx] + nx * distancia_offset
            pos_test_y = y[idx] + ny * distancia_offset
            
            dist_centro_actual = np.sqrt((x[idx] - mid_x)**2 + (y[idx] - mid_y)**2)
            dist_centro_test = np.sqrt((pos_test_x - mid_x)**2 + (pos_test_y - mid_y)**2)
            
            if dist_centro_test < dist_centro_actual:
                nx, ny = -nx, -ny

            # Posición definitiva
            cx = x[idx] + nx * distancia_offset
            cy = y[idx] + ny * distancia_offset

            ax.text(cx, cy, 
                    str(num_curva), 
                    color='black', 
                    fontsize=9, 
                    weight='bold',
                    ha='center', va='center',
                    bbox=dict(boxstyle='circle,pad=0.2', facecolor='white', alpha=0.9, edgecolor='none'))

    # Car
    car_marker, = ax.plot([], [], marker='o', color=coche_color, markersize=5, zorder=20)
    # Trail
    car_trail, = ax.plot([], [], color=coche_color, linewidth=3, zorder=10)

    #Throttle Graphic
    ax_thr = fig.add_subplot(gs[4,0], facecolor = "#1C1C27")
    ax_thr.set_ylabel('THRO',color= 'lime', fontsize=10,fontweight='bold')
    ax_thr.set_ylim(-5, 105)

    for spine in ax_thr.spines.values(): spine.set_visible(False)
    ax_thr.spines['bottom'].set_color('#333333') 
    ax_thr.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    ax_thr.plot(dist_array,throttle,color='lime',linewidth=2)
    ax_thr.fill_between(dist_array, throttle, color='lime', alpha=0.2)
    cursor_thr = ax_thr.axvline(x=0, color='white', linewidth=2, linestyle='-')

    #Brake Graphic
    ax_brk = fig.add_subplot(gs[4,1],facecolor="#1C1C27")
    ax_brk.set_ylabel('BRAKE', color='red', fontsize=10, fontweight='bold')
    ax_brk.set_ylim(-5, 105)

    for spine in ax_brk.spines.values(): spine.set_visible(False)
    ax_brk.tick_params(left=False, bottom=True, labelleft=False, labelbottom=False, colors='gray')

    ax_brk.plot(dist_array, brake * 100, color='red', linewidth=2)
    ax_brk.fill_between(dist_array, brake * 100, color='red', alpha=0.4)
    cursor_brk = ax_brk.axvline(x=0, color='white', linewidth=2, linestyle='-')

    #HUD
    hud = ax.text(0.05, 0.85, "", transform=ax.transAxes, color='white', 
                fontdict={'family': 'monospace', 'weight': 'bold', 'size': 14})

    def update(frame):
        idx = (frame * Salto) % N 
        if idx < Salto and frame > 0:
            idx = N-1
        
        current_dist = dist_array[idx]
        
        # Movimiento coche e estela (X e Y)
        car_marker.set_data([x[idx]], [y[idx]])
        car_trail.set_data(x[0:idx], y[0:idx])
        
        window_size = 800
        view_min = current_dist - window_size
        view_max = current_dist + window_size

        ax_thr.set_xlim(view_min,view_max)
        ax_brk.set_xlim(view_min, view_max)

        cursor_thr.set_xdata([current_dist])
        cursor_brk.set_xdata([current_dist])

        v = vel[idx]
        g = gear[idx]
        r = rpm[idx]
        drs_stat = "OPEN" if drs[idx] > 8 else "CLOSED"
        
        hud_text = (
            f"DRIVER : {Driver} [{Team}]\n"
            f"TYRES  : {compound} (Age: {tyre_age} Laps) \n"
            f"LAPTIME: {TiempoVueltaSeg:.2f} s\n"
            f"--------------------------\n"
            f"SPEED  : {v:3.0f} km/h\n"
            f"RPM    : {r:5.0f}\n"
            f"GEAR   : {g}\n"
            f"DRS    : {drs_stat}\n"
        )
        
        hud.set_text(hud_text)

        return car_marker, car_trail, hud, cursor_thr, cursor_brk

    # --- 4. EJECUTAR ---
# --- RENDERIZADO PARA STREAMLIT ---
    ani = animation.FuncAnimation(fig, update, frames=final_frames, interval=Intervalo_Real, blit=False)
    
    # 1. Extraemos el HTML crudo de la animación
    html_animacion = ani.to_jshtml()
    
    # 2. Lo envolvemos en el div centrador (Opción 2)
    html_centrado = f"""
    <div style="display: flex; justify-content: center; width: 100%;">
        {html_animacion}
    </div>
    """
    
    # 3. Lo pasamos a Streamlit con la altura que ya tenías
    components.html(html_centrado, height=1600, scrolling=True)
    
    plt.close(fig) # Cerramos la figura para liberar memoria