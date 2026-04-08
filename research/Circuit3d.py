import fastf1
import fastf1.plotting
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import savgol_filter

#Initial Configuration
fastf1.Cache.enable_cache('C:/Users/hgdar/Desktop/TFG - Unnamed\Cache')
fastf1.plotting.setup_mpl(misc_mpl_mods=False)
print("###### Cargando datos ###### \n")

sesion = fastf1.get_session(2026, 'Australia', 'Q')
sesion.load()

#Data
Pole = sesion.laps.pick_driver('RUS').pick_fastest()
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
    # If Fails, generic color
    coche_color = 'purple'

# RAW Data
Z_SCALE = 5.0
x = np.array(telemetria['X'].values)
y = np.array(telemetria['Y'].values)
zRaw = np.array(telemetria['Z'].values)


z = savgol_filter(zRaw, window_length= 51, polyorder=2) * Z_SCALE

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
fig = plt.figure(figsize=(16, 9), facecolor='black')

#Grid, 0-3 Rows for Map, Columns For Throttle and Brake
gs = gridspec.GridSpec(5, 2, figure=fig, height_ratios=[1,1,1,1,1.5])

#3D MAP
ax = fig.add_subplot(gs[:5, :], projection='3d', facecolor='black')
ax.axis('off')

#Centering Cam
mid_x, mid_y, mid_z = np.mean(x),np.mean(y),np.mean(z)
maxRange = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 1.8

z_span = z.max() - z.min()
z_margin = z_span * 1.5  

#Centering cam
ax.set_xlim(mid_x - maxRange, mid_x + maxRange)
ax.set_ylim(mid_y - maxRange, mid_y + maxRange)
ax.set_zlim(mid_z - z_margin, mid_z + z_margin)


#Isometric View
ax.view_init(elev=45,azim=-45)

#Coloring Map
ax.plot(x, y, z, color='#222222', linewidth=4.5, alpha=0.5)

#Curves numbers
text_z_offset = 100
text_x_offset = 0
text_y_offset = 0
for _, row in circuit_info.corners.iterrows():
    num_curva = row['Number']
    dist_curva = row['Distance']

    idx_curva = (np.abs(dist_array - dist_curva)).argmin()
    cx,cy,cz = x[idx_curva],y[idx_curva],z[idx_curva]

    ax.plot([cx, cx + text_x_offset], [cy, cy + text_y_offset], [cz, cz + text_z_offset], color='white', linewidth=0.5, alpha=0.5)
    ax.text(cx, cy, cz + text_z_offset, 
            str(num_curva), 
            color='black', 
            fontsize=9, 
            weight='bold',
            horizontalalignment='center',
            verticalalignment='center',
            bbox=dict(boxstyle='circle,pad=0.2', facecolor='white', alpha=0.9, edgecolor='none'))      
    

# Car
car_marker, = ax.plot([], [], [], marker='o', color=coche_color, markersize=5, zorder=20)
# Trail
car_trail, = ax.plot([], [], [], color=coche_color, linewidth=3, zorder=10)

#Throttle Graphic
ax_thr = fig.add_subplot(gs[4,0], facecolor = 'black')
ax_thr.set_ylabel('THRO',color= 'lime', fontsize=10,fontweight='bold')
ax_thr.set_ylim(-5, 105)

for spine in ax_thr.spines.values(): spine.set_visible(False)
ax_thr.spines['bottom'].set_color('#333333') # Division Line
ax_thr.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

#Static Throttle Data
ax_thr.plot(dist_array,throttle,color='lime',linewidth=2)
ax_thr.fill_between(dist_array, throttle, color='lime', alpha=0.2)
cursor_thr = ax_thr.axvline(x=0, color='white', linewidth=2, linestyle='-')

#Brake Graphic
ax_brk = fig.add_subplot(gs[4,1],facecolor='black')
ax_brk.set_ylabel('BRAKE', color='red', fontsize=10, fontweight='bold')
ax_brk.set_ylim(-5, 105)

for spine in ax_brk.spines.values(): spine.set_visible(False)
ax_brk.tick_params(left=False, bottom=True, labelleft=False, labelbottom=False, colors='gray')

#Static Brake Data
ax_brk.plot(dist_array, brake * 100, color='red', linewidth=2)
ax_brk.fill_between(dist_array, brake * 100, color='red', alpha=0.4)
cursor_brk = ax_brk.axvline(x=0, color='white', linewidth=2, linestyle='-')

#HUD
font_props = {'family': 'monospace', 'weight': 'bold', 'size': 14}
hud = ax.text2D(0.05, 0.75, "", transform=ax.transAxes, color='white', fontdict=font_props)

def update(frame):
    #Puntos que saltamos, si cambiamos salto cambiaremos el muestreo
    idx = (frame * Salto) % N 
    #Si es el ultimo punto paramos en el final
    if idx < Salto and frame > 0:
        idx = N-1
    
    current_dist = dist_array[idx]
    #Movimiento coche
    car_marker.set_data([x[idx]],[y[idx]])
    car_marker.set_3d_properties([z[idx]])

    #Movimiento estela
    car_trail.set_data(x[0:idx],y[0:idx])
    car_trail.set_3d_properties(z[0:idx])
    
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
        f"\n"
    )
    
    hud.set_text(hud_text)

    return car_marker, car_trail,hud,cursor_thr,cursor_brk

# --- 4. EJECUTAR ---
print(" Lanzando animación Modo TV...")
# Calculamos exactamente cuántos frames necesitamos


ani = animation.FuncAnimation(fig, update, frames=final_frames, interval=Intervalo_Real, blit=False)
plt.show()