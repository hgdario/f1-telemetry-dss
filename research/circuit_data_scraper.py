"""
circuit_data_scraper.py — TALOS F1 · Recopilador de Características por Circuito
=================================================================================

Recorre todas las carreras de 2018-2025 y extrae características FÍSICAS de cada
circuito. Un circuito se caracteriza por su geometría, no por el estilo del piloto.

Características extraídas (SOLO del circuito):
  · g_lat_mean      — G lateral media (qué tan rápidas son sus curvas)
  · g_lon_mean      — G longitudinal media (cuánto frena/acelera el trazado)
  · g_total_mean    — Carga física total media
  · cambios_marcha_km — Cambios de marcha por km (tecnicidad del trazado)
  · longitud_km     — Longitud de la vuelta

muestreo:
  - Vueltas 5 a 25 de CADA piloto (descarta vuelta 1 y degradación extrema)
  - Media de todos los pilotos → característica estable del CIRCUITO
  - Promediado por año: un circuito tendrá una fila por año
"""

import argparse
import os
import sys
import time

import fastf1
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

# ── Importar _calculate_g_forces desde src ────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from GGDiagram import _calculate_g_forces

# ── Caché de FastF1 ───────────────────────────────────────────────────────────
CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "f1_cache")
os.makedirs(CACHE_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)

# ── Salida ────────────────────────────────────────────────────────────────────
OUTPUT_CSV = os.path.join(os.path.dirname(__file__), "circuit_features_raw.csv")

# Vueltas que consideramos "ritmo normal" de carrera
LAP_MIN = 5
LAP_MAX = 25


# ─────────────────────────────────────────────────────────────────────────────
# EXTRACCIÓN DE CARACTERÍSTICAS
# ─────────────────────────────────────────────────────────────────────────────

def extract_features(year: int, event_name: str) -> dict | None:
    """
    Carga la carrera y extrae características físicas del circuito.

    Promedia vueltas 5-25 de todos los pilotos para obtener
    una caracterización estable del trazado, independiente del estilo
    de conducción de cada piloto.

    Devuelve None si no hay telemetría suficiente.
    """
    try:
        # ── 1. Cargar sesión completa (necesitamos telemetría de varios pilotos)
        session = fastf1.get_session(year, event_name, "R")
        session.load(laps=True, telemetry=True, weather=False, messages=False)

        if session.laps.empty:
            print(f"  ⚠  Sin vueltas — {year} {event_name}")
            return None

        # ── 2. Acumuladores por vuelta ────────────────────────────────────
        all_g_lat       = []
        all_g_lon       = []
        all_g_total     = []
        all_cambios_km  = []
        longitud_km     = None   # Constante del circuito, la leemos una vez

        # ── 3. Recorrer pilotos y sus vueltas normales ────────────────────
        for driver in session.drivers:
            try:
                driver_laps = session.laps.pick_drivers(driver)
                if driver_laps.empty:
                    continue

                # Seleccionar vueltas de ritmo estable (descarta V1 y final de stint)
                max_lap  = int(driver_laps["LapNumber"].max())
                lap_ceil = min(LAP_MAX, max_lap)

                normal_laps = driver_laps[
                    (driver_laps["LapNumber"] >= LAP_MIN) &
                    (driver_laps["LapNumber"] <= lap_ceil)
                ]

                if normal_laps.empty:
                    continue

                for _, lap in normal_laps.iterrows():
                    try:
                        tel = lap.get_telemetry().add_distance()

                        # Mínimo de puntos para que valga la pena
                        if len(tel) < 100 or "Speed" not in tel.columns:
                            continue
                        if "X" not in tel.columns or "Y" not in tel.columns:
                            continue

                        # ── Calcular fuerzas G ────────────────────────────
                        tel = _calculate_g_forces(tel)

                        if "g_lat" not in tel.columns:
                            continue

                        # Acumular medias de la vuelta
                        all_g_lat.append(float(tel["g_lat"].abs().mean()))
                        all_g_lon.append(float(tel["g_lon"].abs().mean()))
                        all_g_total.append(float(tel["g_total"].mean()))

                        # ── Cambios de marcha por km ──────────────────────
                        if "nGear" in tel.columns and "Distance" in tel.columns:
                            gear     = tel["nGear"].dropna()
                            dist_km  = tel["Distance"].max() / 1000
                            if dist_km > 0:
                                cambios = int(gear.diff().abs().gt(0).sum())
                                all_cambios_km.append(cambios / dist_km)

                        # ── Longitud del circuito (solo una vez) ──────────
                        if longitud_km is None and "Distance" in tel.columns:
                            longitud_km = round(tel["Distance"].max() / 1000, 3)

                    except Exception:
                        continue

            except Exception:
                continue

        # ── 4. Necesitamos al menos 10 vueltas para un promedio fiable ────
        if len(all_g_lat) < 10:
            print(f"  ⚠  Datos insuficientes ({len(all_g_lat)} vueltas) — {year} {event_name}")
            return None

        return {
            "year":             year,
            "circuito":         event_name,
            "n_vueltas":        len(all_g_lat),            # Cuántas vueltas se usaron
            "g_lat_mean":       round(np.mean(all_g_lat),   3),
            "g_lon_mean":       round(np.mean(all_g_lon),   3),
            "g_total_mean":     round(np.mean(all_g_total), 3),
            "cambios_marcha_km": round(np.mean(all_cambios_km) if all_cambios_km else 0, 2),
            "longitud_km":      longitud_km if longitud_km else 0.0,
        }

    except Exception as e:
        print(f"  ✗  Error — {year} {event_name}: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# GUARDADO INCREMENTAL
# ─────────────────────────────────────────────────────────────────────────────

def save_incremental(records: list[dict]) -> None:
    """
    Guarda o actualiza el CSV incrementalmente.
    Clave única: year + circuito. Evita duplicados.
    """
    df_new = pd.DataFrame(records)

    if os.path.exists(OUTPUT_CSV):
        df_old = pd.read_csv(OUTPUT_CSV)
        df_combined = pd.concat([df_old, df_new], ignore_index=True)
        df_combined.drop_duplicates(subset=["year", "circuito"], keep="last", inplace=True)
    else:
        df_combined = df_new

    df_combined.to_csv(OUTPUT_CSV, index=False)


def already_downloaded(year: int, event_name: str, existing: pd.DataFrame) -> bool:
    """Comprueba si ya tenemos datos de esta sesión en el CSV."""
    if existing.empty:
        return False
    return not existing[
        (existing["year"] == year) &
        (existing["circuito"] == event_name)
    ].empty


# ─────────────────────────────────────────────────────────────────────────────
# LOOP PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

def run(years: list[int]) -> None:

    # Cargar CSV existente para reanudación automática
    if os.path.exists(OUTPUT_CSV):
        existing = pd.read_csv(OUTPUT_CSV)
        print(f"📂 CSV existente — {len(existing)} registros ya descargados")
    else:
        existing = pd.DataFrame()
        print("📂 Empezando desde cero")

    records_buffer = []
    total_ok    = 0
    total_skip  = 0
    total_error = 0

    for year in years:
        print(f"\n{'='*55}")
        print(f"  AÑO {year}")
        print(f"{'='*55}")

        try:
            schedule = fastf1.get_event_schedule(year, include_testing=False)
        except Exception as e:
            print(f"  ✗ No se pudo obtener calendario {year}: {e}")
            continue

        n_events = len(schedule)

        for idx, (_, event) in enumerate(schedule.iterrows(), start=1):
            event_name = event["EventName"]
            t0 = time.time()

            # Saltar si ya está descargado
            if already_downloaded(year, event_name, existing):
                print(f"  [{idx:02d}/{n_events}] ⏭  Ya existe — {event_name}")
                total_skip += 1
                continue

            print(f"  [{idx:02d}/{n_events}] ⬇  {event_name}...", end=" ", flush=True)

            features = extract_features(year, event_name)
            elapsed  = time.time() - t0

            if features:
                records_buffer.append(features)
                print(f"✓  {features['n_vueltas']} vueltas · ({elapsed:.1f}s)")
                total_ok += 1
            else:
                total_error += 1
                print(f"✗  ({elapsed:.1f}s)")

            # Pausa entre sesiones para no saturar la API (500 calls/h)
            # Una sesión hace ~200-400 llamadas → esperamos 10s entre carreras
            time.sleep(10)

            # Guardar cada 5 sesiones
            if len(records_buffer) >= 5:
                save_incremental(records_buffer)
                existing = pd.read_csv(OUTPUT_CSV)
                records_buffer = []
                print(f"  💾 Guardado parcial — {len(existing)} registros totales")

    # Guardar lo que quede
    if records_buffer:
        save_incremental(records_buffer)

    # Resumen
    print(f"\n{'='*55}")
    print(f"  FINALIZADO")
    print(f"  ✓  Descargados:  {total_ok}")
    print(f"  ⏭  Ya existían: {total_skip}")
    print(f"  ✗  Errores:     {total_error}")
    print(f"  CSV: {OUTPUT_CSV}")
    print(f"{'='*55}\n")

    if os.path.exists(OUTPUT_CSV):
        df = pd.read_csv(OUTPUT_CSV)
        print(f"Vista previa ({len(df)} filas):")
        print(df[["year", "circuito", "g_lat_mean", "g_lon_mean", "cambios_marcha_km", "longitud_km"]].to_string(index=False))


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extrae características físicas de circuitos F1")
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=list(range(2018, 2026)),
        help="Años a procesar (ej: --years 2022 2023 2024)",
    )
    args = parser.parse_args()

    print(f"\n🏎  TALOS · Circuit Data Scraper")
    print(f"   Años:   {args.years}")
    print(f"   Caché:  {CACHE_DIR}")
    print(f"   Salida: {OUTPUT_CSV}")
    print(f"   Vueltas usadas: {LAP_MIN}–{LAP_MAX} por piloto\n")

    run(args.years)
