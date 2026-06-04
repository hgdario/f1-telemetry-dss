"""
Tests de la interpolación/sincronización sobre rejilla (Head2head).

_interpolate remuestrea la telemetría de una vuelta sobre una rejilla de
distancia común, base de toda comparación entre vueltas. Se verifica con señales
lineales (interpolación exacta) y los casos de normalización del freno.
"""

import numpy as np
import pandas as pd
import pytest

from comparativas.Head2head import _interpolate, _normalize_brake
from tests.conftest import synthetic_lap


# ── _normalize_brake ────────────────────────────────────────────────────────

def test_normalize_brake_booleano():
    b = np.array([True, False, True, False])
    assert np.array_equal(_normalize_brake(b), [100, 0, 100, 0])


def test_normalize_brake_cero_uno():
    b = np.array([0.0, 1.0, 0.0, 1.0])
    assert np.array_equal(_normalize_brake(b), [0, 100, 0, 100])


def test_normalize_brake_ya_en_porcentaje():
    b = np.array([0.0, 50.0, 100.0])
    assert np.array_equal(_normalize_brake(b), [0, 50, 100])


# ── _interpolate ────────────────────────────────────────────────────────────

def test_longitud_de_salida_igual_a_la_rejilla():
    tel = synthetic_lap(n=300)
    grid = np.linspace(tel["Distance"].min(), tel["Distance"].max(), 500)
    out = _interpolate(tel, grid)
    assert all(len(v) == 500 for v in out.values())


def test_interpolacion_lineal_exacta_en_speed():
    # Speed lineal con la distancia → la interpolación debe reproducirla
    tel = synthetic_lap(n=300, speed_kmh=np.linspace(100, 300, 300))
    grid = np.linspace(tel["Distance"].min(), tel["Distance"].max(), 200)
    out = _interpolate(tel, grid)
    esperado = np.interp(grid, tel["Distance"].values, tel["Speed"].values)
    assert np.allclose(out["speed"], esperado)


def test_gear_es_entero():
    tel = synthetic_lap(n=300, gear=np.full(300, 5))
    grid = np.linspace(tel["Distance"].min(), tel["Distance"].max(), 100)
    out = _interpolate(tel, grid)
    assert np.issubdtype(out["gear"].dtype, np.integer)
    assert np.all(out["gear"] == 5)


def test_canales_presentes():
    tel = synthetic_lap(n=200)
    grid = np.linspace(tel["Distance"].min(), tel["Distance"].max(), 50)
    out = _interpolate(tel, grid)
    for canal in ("x", "y", "speed", "throttle", "brake", "rpm", "gear", "drs", "time_s"):
        assert canal in out


def test_tiempo_monotono_creciente():
    tel = synthetic_lap(n=300)
    grid = np.linspace(tel["Distance"].min(), tel["Distance"].max(), 200)
    out = _interpolate(tel, grid)
    assert np.all(np.diff(out["time_s"]) >= 0)
