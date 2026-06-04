"""
conftest.py — helpers de telemetría sintética para los tests.

La aplicación trabaja sobre DataFrames de telemetría de FastF1. Para testear las
funciones de cálculo de forma determinista (sin red ni FastF1), construimos
telemetría sintética con física conocida: así podemos comparar la salida del
código contra el resultado analítico esperado.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

G = 9.81  # gravedad estándar (m/s^2), igual que en el código de producción


def _time_col(n: int, dt: float) -> pd.Series:
    """Columna Time como timedelta, muestreada a paso dt."""
    return pd.to_timedelta(np.arange(n) * dt, unit="s")


def straight_accel(v0_kmh: float, v1_kmh: float, T: float, n: int = 200) -> pd.DataFrame:
    """
    Recta con aceleración longitudinal constante (sin X/Y → G lateral nula).

    Velocidad lineal de v0 a v1 en T segundos. La aceleración esperada es
    a = (v1 - v0) / T (m/s^2), por lo que G_lon = a / g constante.
    """
    dt = T / (n - 1)
    v_ms = np.linspace(v0_kmh / 3.6, v1_kmh / 3.6, n)
    return pd.DataFrame({"Time": _time_col(n, dt), "Speed": v_ms * 3.6})


def constant_circle(radius_m: float, v_kmh: float, n: int = 300) -> pd.DataFrame:
    """
    Trayectoria circular a velocidad constante (X, Y de un círculo).

    Para un círculo de radio r recorrido a velocidad v, la aceleración lateral es
    a_lat = v^2 / r, constante, por lo que G_lat = v^2 / (r * g). La velocidad es
    constante, así que G_lon debe ser ~0.
    """
    v_ms = v_kmh / 3.6
    omega = v_ms / radius_m            # rad/s
    T = (2.0 * np.pi) / omega          # una vuelta completa como mucho
    dt = T / (n - 1)
    t = np.arange(n) * dt
    theta = omega * t
    return pd.DataFrame({
        "Time":  _time_col(n, dt),
        "Speed": np.full(n, v_ms * 3.6),
        "X":     radius_m * np.cos(theta),
        "Y":     radius_m * np.sin(theta),
    })


def synthetic_lap(n: int = 300, *, throttle=None, brake=None,
                  speed_kmh=None, gear=None) -> pd.DataFrame:
    """
    Vuelta sintética con TODOS los canales que consumen los clasificadores y la
    interpolación. Por defecto un coche rodando a 200 km/h, gas pleno, sin frenar.
    Cada canal puede sobreescribirse pasando un array de longitud n.
    """
    dt = 0.27  # ~3.7 Hz, la frecuencia real de la telemetría pública
    speed = np.full(n, 200.0) if speed_kmh is None else np.asarray(speed_kmh, float)
    distance = np.cumsum(speed / 3.6 * dt)  # integral de v → distancia (m)
    return pd.DataFrame({
        "Time":     _time_col(n, dt),
        "Distance": distance,
        "Speed":    speed,
        "Throttle": np.full(n, 100.0) if throttle is None else np.asarray(throttle, float),
        "Brake":    np.zeros(n, dtype=bool) if brake is None else np.asarray(brake),
        "nGear":    np.full(n, 7, dtype=int) if gear is None else np.asarray(gear, int),
        "RPM":      np.full(n, 11000.0),
        "X":        np.linspace(0, distance[-1], n),
        "Y":        np.zeros(n),
        "DRS":      np.zeros(n),
    })


@pytest.fixture
def lap_full_throttle():
    return synthetic_lap()
