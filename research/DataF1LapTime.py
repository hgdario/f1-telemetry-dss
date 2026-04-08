import fastf1
import matplotlib.pyplot as plt

fastf1.Cache.enable_cache('C:/Users/hgdar/Desktop/TFG - Unnamed\Cache')
print("###### Cargando datos ###### \n")

sesion = fastf1.get_session(2023, 'Monaco', 'R')
sesion.load()
laps_ALO = sesion.laps.pick_drivers("ALO")
laps_VER = sesion.laps.pick_drivers("VER")


plt.figure(figsize=(12,6))

plt.plot(laps_VER['LapNumber'], laps_VER['LapTime'].dt.total_seconds(), label='Verstappen', color='blue', linewidth=2)
plt.plot(laps_ALO['LapNumber'], laps_ALO['LapTime'].dt.total_seconds(), label='Alonso', color='green', linewidth=2)

plt.title("Mónaco 2023 grand prix")
plt.xlabel("Número de vuelta")
plt.ylabel("Tiempo de vuelta (s)")
plt.legend
plt.grid(True)

plt.show()