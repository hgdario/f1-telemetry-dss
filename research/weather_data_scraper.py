"""
weather_data_scraper.py — TALOS F1 · Temperatura de asfalto por fin de semana
==============================================================================

Recorre las CARRERAS de cada temporada y extrae la temperatura media (asfalto y
aire) de cada fin de semana. Sirve para poner la temp. de asfalto de una carrera
EN CONTEXTO frente al resto de carreras de su temporada (barra relativa del
Resumen de Sesión).

Diseño (mismo patrón que circuit_data_scraper.py):
  · Coste ÚNICO y offline → vuelca a CSV → la app solo lee el CSV.
  · Carga SOLO el stream de clima (weather=True, sin laps/telemetría/mensajes),
    así que cada sesión son ~1-2 llamadas a la API, no cientos.
  · Resumible: salta lo ya descargado (clave year + circuito).

Salida: research/data/track_temp_by_weekend.csv
  year, round, circuito, track_temp_mean, air_temp_mean, n_samples
"""

import argparse
import os
import time

import fastf1
import pandas as pd

# ── Caché de FastF1 ───────────────────────────────────────────────────────────
CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "f1_cache")
os.makedirs(CACHE_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)

# ── Salida ────────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)
OUTPUT_CSV = os.path.join(DATA_DIR, "track_temp_by_weekend.csv")


# ─────────────────────────────────────────────────────────────────────────────
# EXTRACCIÓN
# ─────────────────────────────────────────────────────────────────────────────

def extract_weather(year: int, event_name: str, round_n) -> dict | None:
    """Carga SOLO el clima de la carrera y devuelve las medias de la sesión.

    Devuelve None si la sesión no tiene datos de clima.
    """
    try:
        session = fastf1.get_session(year, event_name, "R")
        # Solo clima: ligero. Nada de laps/telemetría/mensajes.
        session.load(laps=False, telemetry=False, weather=True, messages=False)

        w = session.weather_data
        if w is None or w.empty or "TrackTemp" not in w.columns:
            print(f"  ⚠  Sin clima — {year} {event_name}")
            return None

        track = float(w["TrackTemp"].mean())
        air   = float(w["AirTemp"].mean()) if "AirTemp" in w.columns else None

        return {
            "year":            year,
            "round":           int(round_n) if pd.notna(round_n) else None,
            "circuito":        event_name,
            "track_temp_mean": round(track, 2),
            "air_temp_mean":   round(air, 2) if air is not None else None,
            "n_samples":       int(len(w)),
        }

    except Exception as e:
        print(f"  ✗  Error — {year} {event_name}: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# GUARDADO INCREMENTAL
# ─────────────────────────────────────────────────────────────────────────────

def save_incremental(records: list[dict]) -> None:
    """Guarda/actualiza el CSV. Clave única: year + circuito."""
    df_new = pd.DataFrame(records)
    if os.path.exists(OUTPUT_CSV):
        df_old = pd.read_csv(OUTPUT_CSV)
        df = pd.concat([df_old, df_new], ignore_index=True)
        df.drop_duplicates(subset=["year", "circuito"], keep="last", inplace=True)
    else:
        df = df_new
    df.sort_values(["year", "round"], inplace=True)
    df.to_csv(OUTPUT_CSV, index=False)


def already_downloaded(year: int, event_name: str, existing: pd.DataFrame) -> bool:
    if existing.empty:
        return False
    return not existing[
        (existing["year"] == year) & (existing["circuito"] == event_name)
    ].empty


# ─────────────────────────────────────────────────────────────────────────────
# LOOP PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

def run(years: list[int]) -> None:
    if os.path.exists(OUTPUT_CSV):
        existing = pd.read_csv(OUTPUT_CSV)
        print(f"📂 CSV existente — {len(existing)} registros ya descargados")
    else:
        existing = pd.DataFrame()
        print("📂 Empezando desde cero")

    buffer = []
    total_ok = total_skip = total_error = 0

    for year in years:
        print(f"\n{'='*55}\n  AÑO {year}\n{'='*55}")
        try:
            schedule = fastf1.get_event_schedule(year, include_testing=False)
        except Exception as e:
            print(f"  ✗ No se pudo obtener calendario {year}: {e}")
            continue

        n_events = len(schedule)
        for idx, (_, event) in enumerate(schedule.iterrows(), start=1):
            event_name = event["EventName"]
            round_n    = event.get("RoundNumber")

            if already_downloaded(year, event_name, existing):
                print(f"  [{idx:02d}/{n_events}] ⏭  Ya existe — {event_name}")
                total_skip += 1
                continue

            print(f"  [{idx:02d}/{n_events}] ⬇  {event_name}...", end=" ", flush=True)
            t0 = time.time()
            rec = extract_weather(year, event_name, round_n)
            elapsed = time.time() - t0

            if rec:
                buffer.append(rec)
                air_s = f"{rec['air_temp_mean']:.1f}" if rec["air_temp_mean"] is not None else "—"
                print(f"✓  asfalto {rec['track_temp_mean']:.1f}°C · aire {air_s}°C ({elapsed:.1f}s)")
                total_ok += 1
            else:
                total_error += 1
                print(f"✗  ({elapsed:.1f}s)")

            # Clima = 1-2 llamadas; pausa corta para no saturar la API
            time.sleep(3)

            if len(buffer) >= 5:
                save_incremental(buffer)
                existing = pd.read_csv(OUTPUT_CSV)
                buffer = []
                print(f"  💾 Guardado parcial — {len(existing)} registros totales")

    if buffer:
        save_incremental(buffer)

    print(f"\n{'='*55}\n  FINALIZADO")
    print(f"  ✓  Descargados: {total_ok}")
    print(f"  ⏭  Ya existían: {total_skip}")
    print(f"  ✗  Errores:     {total_error}")
    print(f"  CSV: {OUTPUT_CSV}\n{'='*55}\n")

    if os.path.exists(OUTPUT_CSV):
        df = pd.read_csv(OUTPUT_CSV)
        print(f"Vista previa ({len(df)} filas):")
        print(df.to_string(index=False))


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extrae temperatura de asfalto por carrera")
    parser.add_argument(
        "--years", nargs="+", type=int, default=[2022, 2023, 2024, 2025],
        help="Años a procesar (ej: --years 2024)",
    )
    args = parser.parse_args()

    print("\n🌡  TALOS · Weather Data Scraper")
    print(f"   Años:   {args.years}")
    print(f"   Caché:  {CACHE_DIR}")
    print(f"   Salida: {OUTPUT_CSV}\n")

    run(args.years)
