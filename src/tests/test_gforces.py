"""
Tests del cálculo de fuerzas G (_calculate_g_forces, GGDiagram).

Estrategia: alimentar telemetría sintética con física conocida y comprobar que
las G calculadas coinciden con el resultado analítico. Se evalúa la zona central
del array para evitar los efectos de borde del filtro Savitzky-Golay.
"""

import numpy as np
import pytest

from dinamica_vehicular.GGDiagram import _calculate_g_forces
from tests.conftest import straight_accel, constant_circle, synthetic_lap, G


def _central(arr):
    """Recorte central (evita bordes del suavizado)."""
    n = len(arr)
    return arr[n // 4: 3 * n // 4]


def test_aceleracion_constante_da_glon_esperado():
    # De 100 a 200 km/h en 5 s: a = (200-100)/3.6/5 m/s^2
    a = (200 - 100) / 3.6 / 5.0
    g_lon_esperado = a / G
    tel = _calculate_g_forces(straight_accel(100, 200, T=5.0, n=300))
    assert np.allclose(_central(tel["g_lon"].to_numpy()), g_lon_esperado, atol=0.05)


def test_frenada_da_glon_negativo():
    tel = _calculate_g_forces(straight_accel(250, 80, T=4.0, n=300))
    assert np.median(_central(tel["g_lon"].to_numpy())) < -0.1


def test_velocidad_constante_glon_casi_cero():
    tel = _calculate_g_forces(straight_accel(180, 180, T=5.0, n=300))
    assert np.allclose(_central(tel["g_lon"].to_numpy()), 0.0, atol=0.02)


def test_recta_sin_xy_no_genera_glat():
    # straight_accel no aporta X/Y → la G lateral debe ser exactamente 0
    tel = _calculate_g_forces(straight_accel(120, 120, T=5.0, n=300))
    assert np.all(tel["g_lat"].to_numpy() == 0.0)


def test_circulo_da_glat_esperado():
    # Círculo r=100 m a 108 km/h (30 m/s): a_lat = v^2/r = 9 m/s^2
    r, v_kmh = 100.0, 108.0
    g_lat_esperado = (v_kmh / 3.6) ** 2 / r / G
    tel = _calculate_g_forces(constant_circle(r, v_kmh, n=400))
    g_lat_central = np.abs(_central(tel["g_lat"].to_numpy()))
    assert np.allclose(g_lat_central, g_lat_esperado, atol=0.12)


def test_glat_se_anula_a_baja_velocidad():
    # Círculo a 20 km/h (< 30 km/h): el código anula la G lateral
    tel = _calculate_g_forces(constant_circle(100.0, 20.0, n=300))
    assert np.all(tel["g_lat"].to_numpy() == 0.0)


def test_gtotal_es_norma_de_componentes():
    tel = _calculate_g_forces(constant_circle(80.0, 130.0, n=300))
    esperado = np.sqrt(tel["g_lat"] ** 2 + tel["g_lon"] ** 2)
    assert np.allclose(tel["g_total"], esperado)


def test_clip_a_limite_fisico():
    tel = _calculate_g_forces(constant_circle(80.0, 130.0, n=300))
    assert tel["g_total"].max() <= 6.0 + 1e-9
    assert tel["g_lat"].abs().max() <= 6.0 + 1e-9
    assert tel["g_lon"].abs().max() <= 6.0 + 1e-9


def test_pocas_muestras_devuelve_nan():
    tel = _calculate_g_forces(synthetic_lap(n=5))
    assert tel["g_lat"].isna().all()
    assert tel["g_lon"].isna().all()


def test_sin_columna_speed_devuelve_nan():
    tel = synthetic_lap(n=200).drop(columns=["Speed"])
    out = _calculate_g_forces(tel)
    assert out["g_lon"].isna().all()
