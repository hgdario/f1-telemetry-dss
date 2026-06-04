"""
Tests del clasificador de tipo de vuelta (LapTypeClassifier).

Cubre la resolución de Gran Premio, la clasificación determinista sobre
telemetría sintética y el manejo de casos límite (telemetría insuficiente y
vueltas atípicas que no deben forzarse a ningún cluster).
"""

import numpy as np
import pytest

import LapTypeClassifier as LT
from tests.conftest import synthetic_lap


# ── _resolve_gp_key ─────────────────────────────────────────────────────────

def test_resuelve_gp_exacto():
    assert LT._resolve_gp_key("Italian") == "Italian"


def test_resuelve_gp_con_sufijo_grand_prix():
    assert LT._resolve_gp_key("Italian Grand Prix") == "Italian"


def test_resuelve_gp_case_insensitive():
    assert LT._resolve_gp_key("MONACO") == "Monaco"


def test_gp_desconocido_da_none():
    assert LT._resolve_gp_key("Portuguese") is None


# ── classify_lap ────────────────────────────────────────────────────────────

def test_clasifica_devuelve_estructura_completa():
    # Verifica el contrato de salida (no el cluster concreto): dict con los
    # campos esperados y un cluster_id válido del catálogo.
    cls = LT.classify_lap(synthetic_lap(n=300), "Italian Grand Prix")
    assert cls is not None
    assert cls["cluster_id"] in LT.CLUSTER_LABELS
    assert {"features", "distance", "label", "method"} <= cls.keys()


def test_telemetria_insuficiente_da_none():
    assert LT.classify_lap(synthetic_lap(n=50), "Italian Grand Prix") is None


def test_gp_desconocido_usa_fallback_pero_clasifica():
    cls = LT.classify_lap(synthetic_lap(n=300), "Gran Premio Inexistente")
    assert cls is not None
    assert cls["method"] == "live_fallback"


def test_gp_mean_explicito_marca_metodo_live_session():
    gp_mean = LT.GLOBAL_FALLBACK_MEAN
    cls = LT.classify_lap(synthetic_lap(n=300), "Italian Grand Prix", gp_mean=gp_mean)
    assert cls["method"] == "live_session"


def test_vuelta_lejana_a_todo_centroide_es_atipica():
    # Mecanismo de detección de atípicos: forzamos una línea base (gp_mean)
    # absurdamente lejana, de modo que el residual de cualquier vuelta quede a
    # muchísimas sigma de TODOS los centroides. El clasificador debe marcarla
    # como atípica (-1) en vez de forzarla al cluster más cercano.
    gp_mean = np.full(len(LT.FEATURE_COLS), -1e6)
    cls = LT.classify_lap(synthetic_lap(n=300), "Italian Grand Prix", gp_mean=gp_mean)
    assert cls is not None
    assert cls["cluster_id"] == -1
    assert cls["label"] == LT.CLUSTER_LABELS[-1]


def test_vuelta_en_la_media_del_gp_no_es_atipica():
    # Contraprueba determinista: si la línea base es la propia vuelta, el residual
    # es 0 y queda dentro del umbral de TODOS los centroides → NO es atípica.
    tel = synthetic_lap(n=300)
    feats = LT._extract_lap_features(tel)
    gp_mean = np.array([feats[c] for c in LT.FEATURE_COLS])
    cls = LT.classify_lap(tel, "Italian Grand Prix", gp_mean=gp_mean)
    assert cls["cluster_id"] != -1
