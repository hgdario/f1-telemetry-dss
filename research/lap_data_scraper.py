"""
lap_data_scraper.py — TALOS F1 · Recopilador de Estilos de Pilotaje por Vuelta
================================================================================

Extrae features de pilotaje (uso de pedales, frenadas, cambios) de cada vuelta
rápida válida. A diferencia del scraper de circuitos (que promedia muchas vueltas
para describir el TRAZADO), aquí cada fila es UNA vuelta — describimos el ESTILO.

Características extraídas (SOLO del piloto, no del circuito):
  · p_full           — % distancia a fondo (throttle ≥ 98)
  · p_part           — % distancia con medio-gas estable
  · p_coast          — % distancia coasting (sin acel ni freno)
  · p_brk            — % distancia frenando
  · decel_p10        — percentil 10 del gradiente de velocidad (frenadas extremas)
  · shifts_per_km    — cambios de marcha por km
  · throttle_avg     — media de throttle cuando no es 0
  · coast_avg_len    — duración media de un segmento de coasting (segundos)
  · speed_std_norm   — std de velocidad / vmax  (suavidad de la conducción)

Metadata (NO entra al modelo, solo para validación posterior):
  · year, gp, session_type, driver, lap_number, lap_time_s, compound,
    is_quicklap, stint_position

Muestreo estratificado (Navaja de Ockam — pocos GPs pero diversos):
  - 8 GPs por año cubriendo todo el espectro físico:
    Monza, Mónaco, Bahrain, Silverstone, Hungaroring, Spa, Singapur, Abu Dhabi
  - 3 sesiones por GP: FP3, Q, R
  - Todas las quicklaps válidas (FastF1 ya filtra outliers de tiempo)

Uso:
  python lap_data_scraper.py                 # años por defecto 2022-2025
  python lap_data_scraper.py --years 2024    # solo un año
  python lap_data_scraper.py --gps Monaco    # solo un GP (sub-string match)
"""

import argparse
import os
import sys
import time

import fastf1
import numpy as np
import pandas as pd

# ── Importar helper de cálculo de fases de pedal desde src ────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# ── Caché de FastF1 ───────────────────────────────────────────────────────────
CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "f1_cache")
os.makedirs(CACHE_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)

# ── Salida ────────────────────────────────────────────────────────────────────
DATA_DIR   = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)
OUTPUT_CSV = os.path.join(DATA_DIR, "lap_features_raw.csv")

# ── Configuración del muestreo ───────────────────────────────────────────────
# 8 GPs estratificados por tipo de trazado (cubren los 5 clusters del circuit classifier)
TARGET_GPS: list[str] = [
    "Italian",        # Monza   → low drag
    "Monaco",         # Mónaco  → atípico técnico
    "Bahrain",        # Bahrain → stop-and-go
    "British",        # Silverstone → aero efficiency
    "Hungarian",      # Hungaroring → balanced
    "Belgian",        # Spa     → mixto velocidad/curvas
    "Singapore",      # Singapur → urbano nocturno
    "Abu Dhabi",      # Abu Dhabi → balanced moderno
]

# Sesiones a procesar por GP
TARGET_SESSIONS = ["FP3", "Q", "R"]

# Filtros mínimos por vuelta
MIN_TELEMETRY_POINTS = 200   # vueltas con menos puntos = telemetría corrupta
MIN_LAP_TIME_S       = 60    # vueltas absurdamente cortas (errores)
MAX_LAP_TIME_S       = 300   # vueltas absurdamente largas (in/out severas, SC)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS DE EXTRACCIÓN DE FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def _adaptive_throttle_threshold(tel: pd.DataFrame) -> float:
    """
    Umbral de derivada para detectar throttle parcial estable.
    Adaptativo a la frecuencia de muestreo de cada vuelta (P70 de |dThr/dt|).
    """
    dt    = tel["Time"].dt.total_seconds().diff()
    d_thr = tel["Throttle"].diff()
    derivative = (d_thr / dt).fillna(0).abs().dropna()
    if derivative.empty:
        return 5.0
    return max(float(np.percentile(derivative, 70)), 5.0)


def _coast_avg_length(coast_mask: pd.Series, tel: pd.DataFrame) -> float:
    """
    Duración media (segundos) de los segmentos consecutivos de coasting.
    Mide si el piloto hace "lift & coast" largo y deliberado vs. transiciones cortas.
    """
    if not coast_mask.any():
        return 0.0
    # Detectar arranques de segmento (shift con fill_value=False evita el downcast bool→object)
    starts = coast_mask & ~coast_mask.shift(1, fill_value=False)
    group_id = starts.cumsum() * coast_mask
    durations = []
    for gid in group_id[group_id > 0].unique():
        seg = tel[group_id == gid]
        if len(seg) < 2:
            continue
        dur = (seg["Time"].iloc[-1] - seg["Time"].iloc[0]).total_seconds()
        durations.append(dur)
    return float(np.mean(durations)) if durations else 0.0


def extract_lap_features(tel: pd.DataFrame) -> dict | None:
    """
    Extrae el vector de 9 features de pilotaje desde la telemetría de UNA vuelta.

    Retorna None si los datos no son suficientes para una extracción fiable.
    """
    # ── Sanidad básica ────────────────────────────────────────────────────────
    if len(tel) < MIN_TELEMETRY_POINTS:
        return None
    needed = {"Throttle", "Brake", "Speed", "Time", "Distance", "nGear"}
    if not needed.issubset(tel.columns):
        return None

    # ── Normalizar Brake a 0-100 (puede venir bool, 0-1, 0-100) ───────────────
    brake = tel["Brake"].copy()
    if brake.dtype == bool or set(brake.dropna().unique()).issubset({0, 1, True, False}):
        brake = brake.astype(float) * 100
    elif brake.max() <= 1.0:
        brake = brake * 100

    # ── Distance delta por punto (peso de cada muestra) ──────────────────────
    dist_delta = tel["Distance"].diff().fillna(0).clip(lower=0)
    total_dist = float(dist_delta.sum())
    if total_dist <= 0:
        return None

    throttle = tel["Throttle"]

    # ── Máscaras de fase de pedal ─────────────────────────────────────────────
    dt = tel["Time"].dt.total_seconds().diff()
    d_thr = throttle.diff()
    derivative = (d_thr / dt).fillna(0)
    thr_threshold = _adaptive_throttle_threshold(tel)

    mask_full  = throttle >= 98
    mask_part  = (throttle > 0) & (throttle < 99) & (derivative.abs() < thr_threshold)
    mask_brk   = brake > 0
    mask_coast = (throttle == 0) & (brake == 0)

    # ── % de distancia en cada fase ───────────────────────────────────────────
    p_full  = float(dist_delta[mask_full].sum()  / total_dist * 100)
    p_part  = float(dist_delta[mask_part].sum()  / total_dist * 100)
    p_brk   = float(dist_delta[mask_brk].sum()   / total_dist * 100)
    p_coast = float(dist_delta[mask_coast].sum() / total_dist * 100)

    # ── Frenada extrema: percentil 10 del gradiente de velocidad ──────────────
    decel = (tel["Speed"].diff() / dt).fillna(0).clip(upper=0)
    decel_p10 = float(np.percentile(decel.dropna(), 10)) if decel.notna().any() else 0.0

    # ── Cambios de marcha por km ──────────────────────────────────────────────
    gear = tel["nGear"].dropna()
    km   = total_dist / 1000
    shifts_per_km = float(gear.diff().abs().gt(0).sum() / km) if km > 0 else 0.0

    # ── Throttle medio cuando no está a 0 (ritmo medio de aceleración) ────────
    thr_pos = throttle[throttle > 0]
    throttle_avg = float(thr_pos.mean()) if not thr_pos.empty else 0.0

    # ── Duración media de los segmentos de coasting (s) ───────────────────────
    coast_avg_len = _coast_avg_length(mask_coast, tel)

    # ── Variabilidad de velocidad normalizada ─────────────────────────────────
    vmax = float(tel["Speed"].max())
    speed_std_norm = float(tel["Speed"].std() / vmax) if vmax > 0 else 0.0

    return {
        "p_full":         round(p_full,         3),
        "p_part":         round(p_part,         3),
        "p_coast":        round(p_coast,        3),
        "p_brk":          round(p_brk,          3),
        "decel_p10":      round(decel_p10,      2),  # km/h/s, negativo
        "shifts_per_km":  round(shifts_per_km,  3),
        "throttle_avg":   round(throttle_avg,   2),
        "coast_avg_len":  round(coast_avg_len,  3),  # segundos
        "speed_std_norm": round(speed_std_norm, 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# SCRAPING POR SESIÓN
# ─────────────────────────────────────────────────────────────────────────────

def scrape_session(year: int, gp_name: str, sess_code: str) -> list[dict]:
    """
    Procesa una sesión completa. Retorna lista de dicts (uno por vuelta válida).
    """
    records = []

    try:
        session = fastf1.get_session(year, gp_name, sess_code)
        session.load(laps=True, telemetry=True, weather=False, messages=False)
    except Exception as e:
        print(f"    ✗ Sesión no disponible — {e}")
        return records

    if session.laps.empty:
        print(f"    ⚠ Sin vueltas en sesión")
        return records

    # Solo quicklaps (FastF1 filtra automáticamente vueltas anómalas por tiempo)
    laps = session.laps.pick_quicklaps()
    if laps.empty:
        print(f"    ⚠ Sin quicklaps válidas")
        return records

    n_valid = 0
    for _, lap in laps.iterrows():
        try:
            lap_time = lap.get("LapTime")
            if pd.isna(lap_time):
                continue
            lap_time_s = lap_time.total_seconds()
            if not (MIN_LAP_TIME_S <= lap_time_s <= MAX_LAP_TIME_S):
                continue

            tel = lap.get_telemetry().add_distance()
            feats = extract_lap_features(tel)
            if feats is None:
                continue

            # Metadata (NO entra al modelo)
            compound = lap.get("Compound", "UNKNOWN")
            stint    = lap.get("Stint", None)
            tyre_age = lap.get("TyreLife", None)

            records.append({
                # ── Identificación ────────────────────────────────────────────
                "year":            year,
                "gp":              gp_name,
                "session_type":    sess_code,
                "driver":          str(lap.get("Driver", "")),
                "lap_number":      int(lap.get("LapNumber", 0)),
                "lap_time_s":      round(lap_time_s, 3),
                # ── Metadata neumático (validación posterior) ────────────────
                "compound":        str(compound).upper() if pd.notna(compound) else "UNKNOWN",
                "stint":           int(stint) if pd.notna(stint) else 0,
                "tyre_age":        int(tyre_age) if pd.notna(tyre_age) else 0,
                # ── Features de pilotaje (lo que entra al modelo) ────────────
                **feats,
            })
            n_valid += 1

        except Exception:
            continue

    print(f"    ✓ {n_valid} vueltas extraídas")
    return records


# ─────────────────────────────────────────────────────────────────────────────
# GUARDADO INCREMENTAL
# ─────────────────────────────────────────────────────────────────────────────

def save_incremental(records: list[dict]) -> None:
    """
    Guarda o actualiza el CSV. Clave única: year + gp + session_type + driver + lap_number.
    """
    if not records:
        return

    df_new = pd.DataFrame(records)
    if os.path.exists(OUTPUT_CSV):
        df_old = pd.read_csv(OUTPUT_CSV)
        df = pd.concat([df_old, df_new], ignore_index=True)
        df.drop_duplicates(
            subset=["year", "gp", "session_type", "driver", "lap_number"],
            keep="last",
            inplace=True,
        )
    else:
        df = df_new

    df.to_csv(OUTPUT_CSV, index=False)


def already_scraped(year: int, gp: str, sess: str, existing: pd.DataFrame) -> bool:
    """¿Esta sesión completa ya está en el CSV?"""
    if existing.empty:
        return False
    return not existing[
        (existing["year"] == year) &
        (existing["gp"]   == gp)   &
        (existing["session_type"] == sess)
    ].empty


# ─────────────────────────────────────────────────────────────────────────────
# LOOP PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

def run(years: list[int], gps_filter: list[str] | None = None) -> None:
    if os.path.exists(OUTPUT_CSV):
        existing = pd.read_csv(OUTPUT_CSV)
        print(f"📂 CSV existente — {len(existing)} vueltas ya extraídas")
    else:
        existing = pd.DataFrame()
        print("📂 Empezando desde cero")

    total_sessions_ok    = 0
    total_sessions_skip  = 0
    total_sessions_error = 0
    total_laps           = 0

    # Filtrar GPs si el usuario pasa --gps
    gp_targets = TARGET_GPS
    if gps_filter:
        gp_targets = [
            gp for gp in TARGET_GPS
            if any(f.lower() in gp.lower() for f in gps_filter)
        ]
        print(f"🔎 Filtrado a GPs: {gp_targets}\n")

    for year in years:
        print(f"\n{'='*60}")
        print(f"  AÑO {year}")
        print(f"{'='*60}")

        for gp in gp_targets:
            for sess in TARGET_SESSIONS:
                t0 = time.time()
                tag = f"{year} {gp:<12s} {sess}"

                if already_scraped(year, gp, sess, existing):
                    print(f"  ⏭ {tag}  (ya extraído)")
                    total_sessions_skip += 1
                    continue

                print(f"  ⬇ {tag}")
                records = scrape_session(year, gp, sess)
                elapsed = time.time() - t0

                if records:
                    save_incremental(records)
                    existing = pd.read_csv(OUTPUT_CSV)
                    total_sessions_ok += 1
                    total_laps += len(records)
                    print(f"    💾 Guardado — {len(records)} vueltas · ({elapsed:.1f}s)")
                else:
                    total_sessions_error += 1
                    print(f"    ✗ Sin datos · ({elapsed:.1f}s)")

                # Cortesía con la API (límite ~500 calls/h)
                time.sleep(8)

    # ── Resumen final ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  FINALIZADO")
    print(f"  ✓  Sesiones OK:    {total_sessions_ok}")
    print(f"  ⏭  Ya existían:    {total_sessions_skip}")
    print(f"  ✗  Errores:        {total_sessions_error}")
    print(f"  📊 Vueltas nuevas: {total_laps}")
    print(f"  CSV: {OUTPUT_CSV}")
    print(f"{'='*60}\n")

    if os.path.exists(OUTPUT_CSV):
        df = pd.read_csv(OUTPUT_CSV)
        print(f"Total acumulado: {len(df)} vueltas")
        print(f"\nDistribución por año / sesión:")
        print(df.groupby(["year", "session_type"]).size().to_string())


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extrae features de pilotaje vuelta a vuelta para clustering de estilos"
    )
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=list(range(2022, 2026)),
        help="Años a procesar (default: 2022-2025)",
    )
    parser.add_argument(
        "--gps",
        nargs="+",
        type=str,
        default=None,
        help="Filtrar a un subconjunto de GPs (sub-string match). Ej: --gps Monaco Italian",
    )
    args = parser.parse_args()

    print(f"\n🏎  TALOS · Lap Data Scraper")
    print(f"   Años:      {args.years}")
    print(f"   GPs:       {args.gps or 'todos los 8 estratificados'}")
    print(f"   Sesiones:  {TARGET_SESSIONS}")
    print(f"   Caché:     {CACHE_DIR}")
    print(f"   Salida:    {OUTPUT_CSV}")
    print(f"\n   Estimación: ~{len(args.years) * len(TARGET_GPS) * len(TARGET_SESSIONS)} sesiones · "
          f"~{len(args.years) * len(TARGET_GPS) * len(TARGET_SESSIONS) * 200} vueltas\n")

    run(args.years, gps_filter=args.gps)
