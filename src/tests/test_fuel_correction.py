"""
Tests de la corrección por carga de combustible (fuel_corrected_laptime).

La función resta una penalización proporcional al combustible restante en cada
vuelta. Comprobamos el comportamiento físico y los valores analíticos exactos.
"""

import numpy as np
import pandas as pd

from informacion_sesion.Strat_Grid import (
    fuel_corrected_laptime, FUEL_KG_INICIAL, SEG_POR_KG,
)


def test_ultima_vuelta_no_se_corrige():
    # En la última vuelta el depósito está ~vacío → corrección nula
    corr = fuel_corrected_laptime(90.0, lap_number=50, total_laps=50)
    assert corr == 90.0


def test_primera_vuelta_descuenta_casi_todo_el_deposito():
    # Vuelta 1 de 50: kg_restante = 110*(1 - 1/50) = 107.8 → resta 107.8*0.03
    esperado = 90.0 - FUEL_KG_INICIAL * (1 - 1 / 50) * SEG_POR_KG
    assert np.isclose(fuel_corrected_laptime(90.0, 1, 50), esperado)


def test_vuelta_temprana_se_corrige_mas_que_tardia():
    temprana = fuel_corrected_laptime(90.0, 5, 60)
    tardia   = fuel_corrected_laptime(90.0, 55, 60)
    # Más combustible a bordo => mayor descuento => tiempo corregido menor
    assert temprana < tardia


def test_correccion_siempre_reduce_o_iguala_el_tiempo():
    for lap in range(1, 51):
        assert fuel_corrected_laptime(90.0, lap, 50) <= 90.0


def test_vectorizado_sobre_serie():
    laps = pd.Series([1, 25, 50])
    tiempos = pd.Series([91.0, 90.0, 89.5])
    out = fuel_corrected_laptime(tiempos, laps, 50)
    assert len(out) == 3
    assert np.isclose(out.iloc[2], 89.5)  # última vuelta intacta


def test_parametros_personalizados():
    # Con sec_per_kg = 0, no debe corregir nada
    assert fuel_corrected_laptime(90.0, 1, 50, sec_per_kg=0.0) == 90.0
