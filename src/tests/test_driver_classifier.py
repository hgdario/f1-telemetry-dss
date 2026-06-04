"""
Tests del clasificador de firma de pilotaje (DriverStyleClassifier).

Este clasificador funciona por lookup sobre driver_signatures.csv (firmas
precalculadas). Los tests usan los pilotos realmente presentes en el CSV; si el
CSV no se ha generado, se omiten en lugar de fallar.
"""

import numpy as np
import pytest

import DriverStyleClassifier as DS


# Pilotos disponibles en el dataset entrenado (vacío si no hay CSV)
_DRIVERS = list(DS.DRIVER_FEATURES.keys())
_skip_si_vacio = pytest.mark.skipif(not _DRIVERS, reason="driver_signatures.csv no disponible")


def test_piloto_desconocido_da_none():
    assert DS.classify_driver("ZZZ") is None


def test_codigo_vacio_da_none():
    assert DS.classify_driver("") is None


@_skip_si_vacio
def test_piloto_conocido_devuelve_firma():
    cls = DS.classify_driver(_DRIVERS[0])
    assert cls is not None
    assert "cluster_id" in cls and "label" in cls
    assert set(cls["features"].keys()) == set(DS.FEATURE_COLS)


@_skip_si_vacio
def test_label_coincide_con_cluster():
    for code in _DRIVERS:
        cls = DS.classify_driver(code)
        assert cls["label"] == DS.CLUSTER_LABELS[cls["cluster_id"]]


@_skip_si_vacio
def test_cluster_id_valido():
    for code in _DRIVERS:
        cls = DS.classify_driver(code)
        assert cls["cluster_id"] in DS.CLUSTER_LABELS


@_skip_si_vacio
def test_hibrido_si_lejos_del_centroide():
    # Si un piloto se marca como híbrido (-2), debe traer la info del más cercano
    for code in _DRIVERS:
        cls = DS.classify_driver(code)
        if cls["cluster_id"] == -2:
            assert "nearest_cluster" in cls
            assert "distance" in cls and cls["distance"] > DS.OUTLIER_THRESHOLD_Z


def test_centroides_en_espacio_pca_2d():
    # El pipeline proyecta a 2 componentes principales: los centroides deben ser 2D
    if DS._CLUSTER_CENTROIDS:
        for c in DS._CLUSTER_CENTROIDS.values():
            assert c.shape == (2,)
