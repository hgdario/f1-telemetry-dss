import fastf1
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

# 1. Configuración y Carga
fastf1.Cache.enable_cache('cache')

print("1. Cargando datos...")
session = fastf1.get_session(2023, 'Barcelona', 'Q')
session.load()

# Obtener metadata del circuito (Curvas, Marshals, Rotación)
circuit_info = session.get_circuit_info()

# 2. Obtenemos las vueltas
lap_ver = session.laps.pick_driver('VER').pick_fastest()
lap_alo = session.laps.pick_driver('ALO').pick_fastest()

tel_ver = lap_ver.get_telemetry()
tel_alo = lap_alo.get_telemetry()

# --- FUNCIÓN DE ROTACIÓN (Matriz de Rotación 2D) ---
def rotate(x, y, angle):
    radians = np.radians(angle)
    # Fórmula estándar de rotación de coordenadas
    x_new = x * np.cos(radians) - y * np.sin(radians)
    y_new = x * np.sin(radians) + y * np.cos(radians)
    return x_new, y_new
# ---------------------------------------------------

# 3. Interpolación y Cálculo de Delta
total_distance = tel_ver['Distance'].max()
distancia_comun = np.linspace(0, total_distance, num=4000)

# Datos interpolados (todavía sin rotar)
vel_ver_interp = np.interp(distancia_comun, tel_ver['Distance'], tel_ver['Speed'])
vel_alo_interp = np.interp(distancia_comun, tel_alo['Distance'], tel_alo['Speed'])
x_raw = np.interp(distancia_comun, tel_ver['Distance'], tel_ver['X'])
y_raw = np.interp(distancia_comun, tel_ver['Distance'], tel_ver['Y'])

delta_velocidad = vel_alo_interp - vel_ver_interp

# 4. APLICAMOS LA ROTACIÓN A LA PISTA
track_x, track_y = rotate(x_raw, y_raw, circuit_info.rotation)

# -------------------------------------------------------------------------
# NUEVO: CÁLCULO DE VECTORES PARA DESPLAZAMIENTO (OFFSET)
# -------------------------------------------------------------------------
# 1. Calculamos la derivada (la dirección de la pista en cada punto)
dx = np.gradient(track_x)
dy = np.gradient(track_y)

# 2. Normalizamos (hacemos que los vectores midan 1 unidad)
len_d = np.sqrt(dx**2 + dy**2)
dx /= len_d
dy /= len_d

# 3. Calculamos la NORMAL (Perpendicular): Giramos 90 grados (-dy, dx)
# Esto nos da la dirección "hacia afuera" (o adentro, depende del giro)
nx = -dy
ny = dx

# CONFIGURACIÓN DE DISTANCIA (Juega con esto si se van muy lejos)
OFFSET_MARSHAL = 150  # Metros de separación para las luces
OFFSET_CORNER = 200   # Metros de separación para los números (más lejos)
# -------------------------------------------------------------------------

points = np.array([track_x, track_y]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
delta_velocidad_segmentos = delta_velocidad[:-1] 

# 5. Visualización
fig, ax = plt.subplots(figsize=(12, 10), facecolor='black')
ax.set_facecolor('black')

# A) Pintamos la Trazada
norm = plt.Normalize(-20, 20) 
lc = LineCollection(segments, cmap='RdYlGn', norm=norm, linestyle='-', linewidth=4) # Bajé un poco el grosor
lc.set_array(delta_velocidad_segmentos)
ax.add_collection(lc)

# B) Pintamos los MARSHAL LIGHTS (Desplazados)
marshal_dist = circuit_info.marshal_lights['Distance']

# Buscamos el índice más cercano en nuestro array interpolado para cada marshal
m_indices = [np.abs(distancia_comun - d).argmin() for d in marshal_dist]

# Aplicamos el desplazamiento usando el vector normal en ESE índice
m_x = track_x[m_indices] + nx[m_indices] * OFFSET_MARSHAL
m_y = track_y[m_indices] + ny[m_indices] * OFFSET_MARSHAL

ax.scatter(m_x, m_y, s=15, c='dodgerblue', marker='.', label='Marshal Light', zorder=5)

# C) Pintamos las CURVAS (Más Desplazadas)
corner_dist = circuit_info.corners['Distance']
c_indices = [np.abs(distancia_comun - d).argmin() for d in corner_dist]

# Aplicamos un offset MAYOR para que estén detrás de los marshals
c_x = track_x[c_indices] + nx[c_indices] * OFFSET_CORNER
c_y = track_y[c_indices] + ny[c_indices] * OFFSET_CORNER

for i, numero in enumerate(circuit_info.corners['Number']):
    txt = ax.text(c_x[i], c_y[i], str(numero), 
            color='white', fontsize=8, ha='center', va='center', fontweight='bold')
    # Fondo más discreto (sin borde blanco grueso)
    txt.set_bbox(dict(facecolor='black', alpha=0.6, edgecolor='none', boxstyle='circle,pad=0.1'))

# Ajustes Finales
buffer = 400
ax.set_xlim(track_x.min() - buffer, track_x.max() + buffer)
ax.set_ylim(track_y.min() - buffer, track_y.max() + buffer)
ax.set_aspect('equal') 
plt.axis('off')

# Barra de Color
cbar = plt.colorbar(lc, ax=ax, fraction=0.03, pad=0.04)
cbar.set_label('Delta (km/h)', color='white', size=9)
cbar.ax.yaxis.set_tick_params(color='white', labelsize=8)

plt.title(f"Barcelona 2023 - Mapa de Calor + Infraestructura", color='white', size=14, fontweight='bold')

plt.show()