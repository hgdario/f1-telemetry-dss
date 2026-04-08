import fastf1
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

fastf1.Cache.enable_cache('C:/Users/hgdar/Desktop/TFG - Unnamed\Cache')
print("###### Cargando datos ###### \n")

sesion = fastf1.get_session(2023, 'Monaco', 'Q')
sesion.load()

Pole = sesion.laps.pick_fastest()
telemetria = Pole.get_telemetry()

print(f"Vuelta cargada: {Pole['Driver']} - {Pole['LapTime']}")

x = np.array(telemetria['X'].values)
y = np.array(telemetria['Y'].values)
Velocidad = np.array(telemetria['Speed'].values)

points = np.array([x, y]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)


fig, ax = plt.subplots(sharex=True, sharey=True, figsize=(12, 6.75))


norm = plt.Normalize(Velocidad.min(), Velocidad.max())
lc = LineCollection(segments, cmap='plasma', norm=norm, linestyle='-', linewidth=5)
lc.set_array(Velocidad)

# Añadimos la línea al gráfico
line = ax.add_collection(lc)


ax.add_collection(lc)
ax.autoscale()
ax.set_facecolor('black') 
fig.patch.set_facecolor('black') 
plt.axis('off') 


cb = fig.colorbar(line, ax=ax, location='right', pad=0.05)
cb.set_label('Velocidad (km/h)', color='white')
cb.ax.yaxis.set_tick_params(color='white')
plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color='white')

plt.title(f"Telemetría GPS: {Pole['Driver']} - Mónaco Q3", color='white', size=15)
print("Generando mapa de calor...")
plt.show()