"""
Tests del clasificador de circuitos (CircuitClassifier).

El camino principal es un lookup determinista sobre CIRCUIT_FEATURES, ideal para
testear sin red: cada circuito conocido debe caer en su cluster esperado y la
resolución de nombres debe ser robusta a variantes.
"""

import pytest

import CircuitClassifier as CC


# ── _resolve_circuit_key ────────────────────────────────────────────────────

def test_resuelve_nombre_exacto():
    assert CC._resolve_circuit_key("Monaco Grand Prix") == "Monaco Grand Prix"


def test_resuelve_case_insensitive():
    assert CC._resolve_circuit_key("monaco grand prix") == "Monaco Grand Prix"


def test_resuelve_sin_grand_prix():
    assert CC._resolve_circuit_key("Monaco") == "Monaco Grand Prix"


def test_nombre_desconocido_da_none():
    assert CC._resolve_circuit_key("Nürburgring") is None


# ── classify_circuit (lookup determinista) ──────────────────────────────────

def test_monaco_cae_en_su_cluster():
    # Mónaco es matemáticamente único → cluster propio (id 4)
    res = CC.classify_circuit("Monaco Grand Prix")
    assert res["cluster_id"] == 4
    assert res["label"] == "Urbano extremo"
    assert res["method"] == "lookup"


def test_monza_es_low_drag():
    res = CC.classify_circuit("Italian Grand Prix")
    assert res["cluster_id"] == 0  # Low Drag


def test_todos_los_circuitos_conocidos_clasifican():
    for nombre, (_, _, _, cid) in CC.CIRCUIT_FEATURES.items():
        res = CC.classify_circuit(nombre)
        assert res is not None
        assert res["cluster_id"] == cid           # coincide con el lookup
        assert res["cluster_id"] in CC.CLUSTER_LABELS


def test_features_devueltas_coinciden_con_lookup():
    g_lat, g_lon, cambios, _ = CC.CIRCUIT_FEATURES["Monaco Grand Prix"]
    feats = CC.classify_circuit("Monaco Grand Prix")["features"]
    assert feats["g_lat_mean"] == pytest.approx(g_lat)
    assert feats["g_lon_mean"] == pytest.approx(g_lon)
    assert feats["cambios_marcha_km"] == pytest.approx(cambios)


def test_desconocido_sin_sesion_da_none():
    assert CC.classify_circuit("Circuito Inventado") is None
