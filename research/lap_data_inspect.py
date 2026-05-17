"""
lap_data_inspect.py — Diagnóstico rápido del CSV del scraper.

Se puede lanzar EN PARALELO al scraper (lee el último estado guardado).
Muestra:
  - Total de vueltas
  - Cobertura year × GP × sesión (matriz de qué tenemos)
  - Qué falta de la matriz objetivo
  - Estadísticas básicas de features
  - Diversidad de pilotos
"""

import os
import pandas as pd

CSV_PATH = os.path.join(os.path.dirname(__file__), "data", "lap_features_raw.csv")

# Matriz objetivo (lo que el scraper PRETENDE recoger)
TARGET_GPS = [
    "Italian", "Monaco", "Bahrain", "British",
    "Hungarian", "Belgian", "Singapore", "Abu Dhabi",
]
TARGET_SESSIONS = ["FP3", "Q", "R"]
TARGET_YEARS = [2022, 2023, 2024, 2025]

FEATURE_COLS = [
    "p_full", "p_part", "p_coast", "p_brk",
    "decel_p10", "shifts_per_km", "throttle_avg",
    "coast_avg_len", "speed_std_norm",
]


def main():
    if not os.path.exists(CSV_PATH):
        print(f"❌ No existe {CSV_PATH} — el scraper aún no ha generado nada.")
        return

    df = pd.read_csv(CSV_PATH)
    print(f"\n{'='*60}")
    print(f"  📊 ESTADO DEL CSV — {len(df)} vueltas")
    print(f"{'='*60}\n")

    # ── 1. Cobertura por año ─────────────────────────────────────────────
    print("📅 VUELTAS POR AÑO")
    print(df["year"].value_counts().sort_index().to_string())
    print()

    # ── 2. Cobertura por sesión ──────────────────────────────────────────
    print("🏁 VUELTAS POR SESIÓN")
    print(df["session_type"].value_counts().to_string())
    print()

    # ── 3. Matriz year × GP × session ─────────────────────────────────────
    print("🗺  MATRIZ year × GP × session (count)")
    pivot = (
        df.groupby(["year", "gp", "session_type"])
        .size()
        .unstack(fill_value=0)
    )
    print(pivot.to_string())
    print()

    # ── 4. Qué FALTA de la matriz objetivo ───────────────────────────────
    expected = {
        (y, g, s)
        for y in TARGET_YEARS
        for g in TARGET_GPS
        for s in TARGET_SESSIONS
    }
    have = set(
        zip(df["year"], df["gp"], df["session_type"])
    )
    missing = sorted(expected - have)

    print(f"🚫 SESIONES PENDIENTES ({len(missing)} de {len(expected)})")
    if missing:
        # Agrupar por año para visualizar mejor
        from collections import defaultdict
        by_year = defaultdict(list)
        for y, g, s in missing:
            by_year[y].append(f"{g} {s}")
        for y in sorted(by_year):
            print(f"  {y}: {', '.join(by_year[y])}")
    else:
        print("  ✓ Matriz completa")
    print()

    # ── 5. Pilotos ────────────────────────────────────────────────────────
    print(f"👤 PILOTOS ÚNICOS: {df['driver'].nunique()}")
    print(df["driver"].value_counts().to_string())
    print()

    # ── 6. Estadísticas de features ──────────────────────────────────────
    print("📈 ESTADÍSTICAS DE FEATURES")
    desc = df[FEATURE_COLS].describe().T[["mean", "std", "min", "25%", "50%", "75%", "max"]]
    print(desc.round(2).to_string())
    print()

    # ── 7. Salud por GP (varianza de p_full → ¿hay diversidad de estilos?) ─
    print("🔬 VARIANZA DE p_full POR GP (alta = mezcla de ataque/gestión)")
    var_by_gp = df.groupby("gp")["p_full"].agg(["mean", "std", "count"]).round(2)
    print(var_by_gp.to_string())
    print()


if __name__ == "__main__":
    main()
