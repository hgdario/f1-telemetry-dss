"""
Tests de la extracción del vector de features de pilotaje (_extract_lap_features,
LapTypeClassifier). Es la entrada común de los clasificadores de vuelta.

Se construyen vueltas sintéticas con un comportamiento dominante conocido (gas
pleno, coasting, frenada) y se comprueba que la feature correspondiente domina.
"""

import numpy as np
import pytest

from LapTypeClassifier import _extract_lap_features
from tests.conftest import synthetic_lap


def test_gas_pleno_da_p_full_alto():
    feats = _extract_lap_features(synthetic_lap(n=300))  # throttle 100% por defecto
    assert feats is not None
    assert feats["p_full"] > 95
    assert feats["p_brk"] == 0
    assert feats["p_coast"] == 0


def test_coasting_da_p_coast_alto():
    # Throttle 0 y freno 0 en toda la vuelta → coasting puro
    tel = synthetic_lap(n=300, throttle=np.zeros(300),
                        brake=np.zeros(300, dtype=bool))
    feats = _extract_lap_features(tel)
    assert feats["p_coast"] > 95


def test_frenada_da_p_brk_alto():
    tel = synthetic_lap(n=300, throttle=np.zeros(300),
                        brake=np.ones(300, dtype=bool))
    feats = _extract_lap_features(tel)
    assert feats["p_brk"] > 95


def test_porcentajes_en_rango_valido():
    feats = _extract_lap_features(synthetic_lap(n=300))
    for k in ("p_full", "p_part", "p_coast", "p_brk"):
        assert 0.0 <= feats[k] <= 100.0


def test_todas_las_features_presentes():
    feats = _extract_lap_features(synthetic_lap(n=300))
    esperadas = {"p_full", "p_part", "p_coast", "p_brk", "decel_p10",
                 "shifts_per_km", "throttle_avg", "coast_avg_len"}
    assert set(feats.keys()) == esperadas


def test_shifts_per_km_cuenta_cambios():
    # Alternar marcha 6/7 en cada muestra → muchos cambios; vs marcha fija → 0
    n = 300
    gear_alt = np.tile([6, 7], n // 2)
    f_alt = _extract_lap_features(synthetic_lap(n=n, gear=gear_alt))
    f_fix = _extract_lap_features(synthetic_lap(n=n, gear=np.full(n, 7)))
    assert f_alt["shifts_per_km"] > f_fix["shifts_per_km"]
    assert f_fix["shifts_per_km"] == 0


def test_pocas_muestras_devuelve_none():
    assert _extract_lap_features(synthetic_lap(n=50)) is None


def test_falta_columna_devuelve_none():
    tel = synthetic_lap(n=300).drop(columns=["Brake"])
    assert _extract_lap_features(tel) is None


def test_distancia_nula_devuelve_none():
    tel = synthetic_lap(n=300)
    tel["Distance"] = 0.0  # sin avance → total_dist <= 0
    assert _extract_lap_features(tel) is None
