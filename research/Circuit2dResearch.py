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

def render_interactive_sim(sesion, driver_code, show_corners, lap_number):
    mpl.rcParams['animation.embed_limit'] = 100.0
    
    vueltas_piloto = sesion.laps.pick_drivers(driver_code)
    Pole = vueltas_piloto[vueltas_piloto['LapNumber'] == lap_number].iloc[0]
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
    # --- [NUEVO] Función de offset automático ---
    mid_x, mid_y = np.mean(x), np.mean(y)

    def exterior_coords(idx, offset_dist):
        idx_next = min(idx + 5, len(x) - 1)
        idx_prev = max(idx - 5, 0)
        dx = x[idx_next] - x[idx_prev]
        dy = y[idx_next] - y[idx_prev]
        nx, ny = -dy, dx 
        mag = np.sqrt(nx**2 + ny**2)
        if mag != 0:
            nx /= mag; ny /= mag
        if np.sqrt((x[idx] + nx * offset_dist - mid_x)**2 + (y[idx] + ny * offset_dist - mid_y)**2) < \
           np.sqrt((x[idx] - mid_x)**2 + (y[idx] - mid_y)**2):
            nx, ny = -nx, -ny
        return x[idx] + nx * offset_dist, y[idx] + ny * offset_dist
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

    brake_binary = (brake > 0).astype(int)
    brake_starts = np.where(np.diff(brake_binary) > 0)[0] 
    ax.scatter(x[brake_starts], y[brake_starts],
           marker='v', color='#E8002D', s=40, zorder=15, alpha=0.8)
    
    DRS_binary = (drs > 8).astype(int)
    DRS_starts = np.where(np.diff(DRS_binary) > 0)[0] 
    ax.scatter(x[DRS_starts], y[DRS_starts],
           marker='o', color="#00E887", s=40, zorder=15, alpha=0.8)
    
    # --- [NUEVO] Línea de Meta ---
    start_x, start_y = x[0], y[0]
    dx_m = x[1] - x[0]
    dy_m = y[1] - y[0]
    mag_m = np.sqrt(dx_m**2 + dy_m**2)
    nmx, nmy = -dy_m/mag_m, dx_m/mag_m 
    
    meta_w = 200
    ax.plot([start_x - nmx*meta_w, start_x + nmx*meta_w], 
            [start_y - nmy*meta_w, start_y + nmy*meta_w], 
            color='white', linewidth=3, zorder=10)

    if show_corners:
        circuit_info = sesion.get_circuit_info()
        for _, row in circuit_info.corners.iterrows():
            idx = (np.abs(dist_array - row['Distance'])).argmin()
            
            # Llamada exacta a tu función
            cx, cy = exterior_coords(idx, 250)
            
            ax.text(cx, cy, str(row['Number']), 
                    color='#15151E', fontsize=9, weight='black',
                    ha='center', va='center', zorder=20,
                    bbox=dict(boxstyle='circle,pad=0.3', 
                              facecolor='white', 
                              edgecolor='#E8002D', 
                              linewidth=1.5))

    # Car
    car_marker, = ax.plot([], [], marker='o', color=coche_color, markersize=5, zorder=20)
    # Trail
    car_trail, = ax.plot([], [], color=coche_color, linewidth=3, zorder=10)

    # ── THROTTLE — solo ejes X e Y visibles ──────────────────────────────────
    ax_thr = fig.add_subplot(gs[4, 0], facecolor="#1C1C27")
    ax_thr.set_ylabel('THRO', color='lime', fontsize=10, fontweight='bold')
    ax_thr.set_ylim(0, 105)


    for spine in ax_thr.spines.values():
        spine.set_visible(False)
    ax_thr.spines['bottom'].set_visible(True)
    ax_thr.spines['bottom'].set_color('#8888A0')   # eje X más claro
    ax_thr.spines['left'].set_visible(True)
    ax_thr.spines['left'].set_color('#8888A0')     # eje Y más claro
    ax_thr.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    ax_thr.plot(dist_array, throttle, color='lime', linewidth=2)
    ax_thr.fill_between(dist_array, throttle, color='lime', alpha=0.2)
    cursor_thr = ax_thr.axvline(x=0, color='white', linewidth=2, linestyle='-')

    # ── BRAKE — solo ejes X e Y visibles ─────────────────────────────────────
    ax_brk = fig.add_subplot(gs[4, 1], facecolor="#1C1C27")
    ax_brk.set_ylabel('BRAKE', color='red', fontsize=10, fontweight='bold')
    ax_brk.set_ylim(0, 105)

    for spine in ax_brk.spines.values():
        spine.set_visible(False)
    ax_brk.spines['bottom'].set_visible(True)
    ax_brk.spines['bottom'].set_color('#8888A0')   # eje X más claro
    ax_brk.spines['left'].set_visible(True)
    ax_brk.spines['left'].set_color('#8888A0')     # eje Y más claro
    ax_brk.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    ax_brk.plot(dist_array, brake * 100, color='red', linewidth=2)
    ax_brk.fill_between(dist_array, brake * 100, color='red', alpha=0.4)
    cursor_brk = ax_brk.axvline(x=0, color='white', linewidth=2, linestyle='-')

    # ── Etiquetas de curva en ambas gráficas ──────────────────────────────────
    for _, row in circuit_info.corners.iterrows():
        dist_curva = row['Distance']
        txt = f"T{row['Number']}"

        ax_thr.axvline(x=dist_curva, color='#8888A0', linestyle='--', linewidth=0.8, alpha=0.4)
        ax_brk.axvline(x=dist_curva, color='#8888A0', linestyle='--', linewidth=0.8, alpha=0.4)

        ax_thr.text(dist_curva, 0, txt, color='#1C1C27', fontsize=7,
                    ha='center', va='bottom', weight='bold', clip_on=True,
                    bbox=dict(boxstyle='square,pad=0.25', facecolor='#8888A0',
                              edgecolor='none', alpha=0.9))
        ax_brk.text(dist_curva, 0, txt, color='#1C1C27', fontsize=7,
                    ha='center', va='bottom', weight='bold', clip_on=True,
                    bbox=dict(boxstyle='square,pad=0.25', facecolor='#8888A0',
                              edgecolor='none', alpha=0.9))

    # ─────────────────────────────────────────────────────────────────────────
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
        view_min = max(0, current_dist - window_size)
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
    
    # 2. Lo envolvemos en el div centrador
    html_centrado = f"""
    <div style="display: flex; justify-content: center; width: 100%;">
        {html_animacion}
    </div>
    """
    
    # 3. Lo pasamos a Streamlit con la altura que ya tenías
    components.html(html_centrado, height=1000, scrolling=False)
    
    plt.close(fig) # Cerramos la figura para liberar memoria