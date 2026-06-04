"""
validate_classifiers.py — TALOS F1 · Validación de los 3 Clasificadores
========================================================================

Ejecuta (desde la carpeta research/):
    python validate_classifiers.py

No modifica ningún script de entrenamiento ni CSV existente.
Genera: research/data/validation_report.csv

Métricas por clasificador:
  CircuitClassifier (25 circuitos)  — LOO CV + internas
  LapTypeClassifier (18k vueltas)   — 5-fold CV + temporal CV + internas
  DriverStyleClassifier (20 pilotos) — LOO CV + temporal CV + internas
"""

from __future__ import annotations

import os
import warnings
import numpy as np
import pandas as pd

from sklearn.cluster      import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.metrics import (
    silhouette_score, silhouette_samples,
    calinski_harabasz_score, davies_bouldin_score,
    adjusted_rand_score, classification_report, accuracy_score,
)
from scipy.optimize import linear_sum_assignment

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
HERE     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
SRC_DATA = os.path.join(HERE, "..", "src", "data")
OUT_CSV  = os.path.join(DATA_DIR, "validation_report.csv")

# ── Features (idénticas a los training scripts) ────────────────────────────────
CIRCUIT_FEATS = ["g_lat_mean", "g_lon_mean", "cambios_marcha_km"]
LAP_FEATS     = ["p_full", "p_part", "p_coast", "p_brk",
                 "decel_p10", "shifts_per_km", "throttle_avg", "coast_avg_len"]
K_CIRCUIT, K_LAP, K_DRIVER = 5, 4, 4


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def align_labels(y_ref: np.ndarray, y_new: np.ndarray, k: int) -> np.ndarray:
    """
    Alinea y_new con y_ref usando el algoritmo húngaro.
    Necesario porque K-means puede numerar clusters distinto en cada run.
    """
    cm = np.zeros((k, k), dtype=int)
    for r, n in zip(y_ref, y_new):
        if 0 <= r < k and 0 <= n < k:
            cm[r, n] += 1
    row_ind, col_ind = linear_sum_assignment(-cm)
    mapping = {col_ind[i]: row_ind[i] for i in range(len(row_ind))}
    return np.array([mapping.get(int(l), int(l)) for l in y_new])


def internal_metrics(X: np.ndarray, labels: np.ndarray) -> dict:
    """Silhouette, Calinski-Harabász, Davies-Bouldin y % bien asignados."""
    sil     = silhouette_score(X, labels)
    ch      = calinski_harabasz_score(X, labels)
    db      = davies_bouldin_score(X, labels)
    samples = silhouette_samples(X, labels)
    return {
        "silhouette":        round(float(sil), 4),
        "calinski_harabasz": round(float(ch),  2),
        "davies_bouldin":    round(float(db),  4),
        "pct_bien_asignados": round(float((samples > 0.5).mean() * 100), 1),
    }


def run_loo(X: np.ndarray, y_ref: np.ndarray, k: int,
            n_init: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """
    Leave-One-Out CV. Devuelve (y_true, y_pred) alineados con y_ref.
    """
    y_true, y_pred = [], []
    loo = LeaveOneOut()
    for train_idx, test_idx in loo.split(X):
        km = KMeans(n_clusters=k, random_state=42, n_init=n_init)
        km.fit(X[train_idx])
        # Alinear IDs de este run con los del modelo completo
        labels_aligned = align_labels(y_ref[train_idx], km.labels_, k)
        # Construir mapping new_id → aligned_id
        cm = np.zeros((k, k), dtype=int)
        for r, n in zip(y_ref[train_idx], km.labels_):
            if 0 <= r < k and 0 <= n < k:
                cm[r, n] += 1
        _, col_ind = linear_sum_assignment(-cm)
        row_ind = np.arange(k)
        # row_ind[i] es el cluster ref, col_ind[i] es el cluster loo
        mapping = {col_ind[i]: row_ind[i] for i in range(k)}
        raw = int(km.predict(X[test_idx])[0])
        y_true.append(int(y_ref[test_idx[0]]))
        y_pred.append(mapping.get(raw, raw))
    return np.array(y_true), np.array(y_pred)


def header(title: str) -> None:
    print(f"\n{'=' * 65}")
    print(f"  {title}")
    print(f"{'=' * 65}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. CIRCUIT CLASSIFIER
# ─────────────────────────────────────────────────────────────────────────────

def validate_circuits() -> dict:
    header("1. CIRCUIT CLASSIFIER  — LOO CV (K=5, 25 circuitos)")

    df_raw      = pd.read_csv(os.path.join(DATA_DIR, "circuit_features_raw.csv"))
    df_clusters = pd.read_csv(os.path.join(DATA_DIR, "circuit_clusters.csv"))

    # Agregación por circuito (media de años) — igual que circuit_classifier.py
    df_c = (df_raw.groupby("circuito")[CIRCUIT_FEATS]
                  .mean().reset_index()
                  .merge(df_clusters[["circuito", "cluster", "label"]], on="circuito"))

    X    = StandardScaler().fit_transform(df_c[CIRCUIT_FEATS].values)
    y    = df_c["cluster"].astype(int).values

    # Métricas internas (modelo completo)
    met = internal_metrics(X, y)
    print(f"\n  Silhouette:          {met['silhouette']}")
    print(f"  Calinski-Harabász:   {met['calinski_harabasz']}")
    print(f"  Davies-Bouldin:      {met['davies_bouldin']}")
    print(f"  Bien asignados:      {met['pct_bien_asignados']}%")

    # Silhouette por cluster
    sil_s = silhouette_samples(X, y)
    print("\n  Silhouette por cluster:")
    for cid in sorted(np.unique(y)):
        m = y == cid
        names = df_c.loc[m, "circuito"].str.replace(" Grand Prix", "").tolist()
        label = df_c.loc[m, "label"].iloc[0]
        print(f"    {cid} {label:<20s} sil={sil_s[m].mean():.3f}  [{', '.join(names)}]")

    # LOO CV
    y_true, y_pred = run_loo(X, y, K_CIRCUIT)
    acc = accuracy_score(y_true, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)
    met["accuracy_loo"] = round(acc, 4)
    met["ari_loo"]      = round(ari, 4)

    print(f"\n  LOO Accuracy: {acc:.2%}")
    print(f"  LOO ARI:      {ari:.4f}")

    labels_by_id = dict(zip(df_clusters["cluster"], df_clusters["label"]))
    tnames = [labels_by_id.get(i, str(i)) for i in range(K_CIRCUIT)]
    print("\n  Classification report (LOO):")
    print(classification_report(y_true, y_pred, target_names=tnames, zero_division=0))

    return {"classifier": "circuit", **met}


# ─────────────────────────────────────────────────────────────────────────────
# 2. LAP TYPE CLASSIFIER
# ─────────────────────────────────────────────────────────────────────────────

def validate_laps() -> dict:
    header("2. LAP TYPE CLASSIFIER — 5-Fold CV + Temporal CV (K=4, ~18k vueltas)")

    df = pd.read_csv(os.path.join(DATA_DIR, "lap_features_raw.csv"))
    df_asgn = pd.read_csv(os.path.join(DATA_DIR, "lap_type_assignments.csv"))

    # Residualizar por GP — igual que lap_type_classifier.py
    for col in LAP_FEATS:
        df[col] = df[col] - df.groupby("gp")[col].transform("mean")
    df = df.dropna(subset=LAP_FEATS).reset_index(drop=True)
    df_asgn = df_asgn.reset_index(drop=True)

    # Alinear assignments al mismo índice
    key = ["year", "gp", "session_type", "driver", "lap_number"]
    df_merged = df.merge(df_asgn[key + ["cluster"]], on=key, how="left")

    scaler = StandardScaler()
    X = scaler.fit_transform(df_merged[LAP_FEATS].values)
    y_full = df_merged["cluster"].values  # puede tener NaN en merge

    # Modelo completo (para métricas internas)
    km_full = KMeans(n_clusters=K_LAP, random_state=42, n_init=20)
    y_full_pred = km_full.fit_predict(X)
    met = internal_metrics(X, y_full_pred)
    print(f"\n  Vueltas: {len(X)}")
    print(f"  Silhouette:          {met['silhouette']}")
    print(f"  Calinski-Harabász:   {met['calinski_harabasz']}")
    print(f"  Davies-Bouldin:      {met['davies_bouldin']}")
    print(f"  Bien asignados:      {met['pct_bien_asignados']}%")

    # ── 5-fold CV ─────────────────────────────────────────────────────────────
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    sil_train_folds, sil_test_folds, ari_folds = [], [], []

    for train_idx, test_idx in kf.split(X):
        X_tr, X_te = X[train_idx], X[test_idx]
        km = KMeans(n_clusters=K_LAP, random_state=42, n_init=20)
        km.fit(X_tr)
        labels_te = km.predict(X_te)

        sil_train_folds.append(silhouette_score(X_tr, km.labels_))
        sil_test_folds.append(silhouette_score(X_te, labels_te))
        ari_folds.append(adjusted_rand_score(y_full_pred[test_idx], labels_te))

    met["sil_train_5fold"]   = round(float(np.mean(sil_train_folds)), 4)
    met["sil_test_5fold"]    = round(float(np.mean(sil_test_folds)),  4)
    met["sil_test_5fold_std"]= round(float(np.std(sil_test_folds)),   4)
    met["ari_5fold"]         = round(float(np.mean(ari_folds)),       4)

    print(f"\n  5-fold CV:")
    print(f"    Silhouette train: {met['sil_train_5fold']:.4f}")
    print(f"    Silhouette test:  {met['sil_test_5fold']:.4f} ± {met['sil_test_5fold_std']:.4f}")
    print(f"    ARI (test vs full model): {met['ari_5fold']:.4f}")

    # ── Temporal CV ───────────────────────────────────────────────────────────
    train_m = df_merged["year"] <= 2023
    test_m  = df_merged["year"] >= 2024

    if test_m.sum() >= 100:
        sc2 = StandardScaler()
        X_tr_t = sc2.fit_transform(df_merged.loc[train_m, LAP_FEATS].values)
        X_te_t = sc2.transform(df_merged.loc[test_m,  LAP_FEATS].values)
        km_t = KMeans(n_clusters=K_LAP, random_state=42, n_init=20)
        km_t.fit(X_tr_t)
        labels_te_t = km_t.predict(X_te_t)
        sil_t = silhouette_score(X_te_t, labels_te_t)
        ari_t = adjusted_rand_score(y_full_pred[test_m], labels_te_t)
        met["sil_temporal"] = round(float(sil_t), 4)
        met["ari_temporal"] = round(float(ari_t), 4)
        print(f"\n  Temporal CV (≤2023 → ≥2024):")
        print(f"    Train: {train_m.sum()} vueltas  · Test: {test_m.sum()} vueltas")
        print(f"    Silhouette test: {sil_t:.4f}")
        print(f"    ARI temporal:    {ari_t:.4f}")
    else:
        met["sil_temporal"] = float("nan")
        met["ari_temporal"] = float("nan")
        print(f"\n  Temporal CV: sin datos 2024+ suficientes en el CSV")

    return {"classifier": "laps", **met}


# ─────────────────────────────────────────────────────────────────────────────
# 3. DRIVER STYLE CLASSIFIER
# ─────────────────────────────────────────────────────────────────────────────

def validate_drivers() -> dict:
    header("3. DRIVER STYLE CLASSIFIER — LOO CV + Temporal CV (K=4, ~20 pilotos)")

    df = pd.read_csv(os.path.join(SRC_DATA, "driver_signatures.csv"), index_col=0)
    Xs = StandardScaler().fit_transform(df[LAP_FEATS].values)
    X  = PCA(n_components=2, random_state=42).fit_transform(Xs)  # modelo desplegado: PCA-2
    y  = df["cluster"].astype(int).values

    # Métricas internas
    met = internal_metrics(X, y)
    print(f"\n  Pilotos: {len(df)}")
    print(f"  Silhouette:          {met['silhouette']}")
    print(f"  Calinski-Harabász:   {met['calinski_harabasz']}")
    print(f"  Davies-Bouldin:      {met['davies_bouldin']}")
    print(f"  Bien asignados:      {met['pct_bien_asignados']}%")

    # Silhouette por cluster
    sil_s = silhouette_samples(X, y)
    print("\n  Silhouette por cluster:")
    for cid in sorted(np.unique(y)):
        m = y == cid
        label = df.loc[m, "label"].iloc[0]
        drivers = df.index[m].tolist()
        print(f"    {cid} {label:<15s} sil={sil_s[m].mean():.3f}  [{', '.join(drivers)}]")

    # LOO CV
    y_true, y_pred = run_loo(X, y, K_DRIVER, n_init=50)
    acc = accuracy_score(y_true, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)
    met["accuracy_loo"] = round(acc, 4)
    met["ari_loo"]      = round(ari, 4)

    print(f"\n  LOO Accuracy: {acc:.2%}")
    print(f"  LOO ARI:      {ari:.4f}")

    tnames = [df.loc[df["cluster"] == i, "label"].iloc[0]
              if (df["cluster"] == i).any() else str(i) for i in range(K_DRIVER)]
    print("\n  Classification report (LOO):")
    print(classification_report(y_true, y_pred, target_names=tnames, zero_division=0))

    # ── Temporal CV ───────────────────────────────────────────────────────────
    df_raw = pd.read_csv(os.path.join(DATA_DIR, "lap_features_raw.csv"))
    for col in LAP_FEATS:
        df_raw[col] = df_raw[col] - df_raw.groupby(["gp", "session_type"])[col].transform("mean")
    df_raw = df_raw.dropna(subset=LAP_FEATS)

    sigs_train = df_raw[df_raw["year"] <= 2023].groupby("driver")[LAP_FEATS].mean()
    sigs_test  = df_raw[df_raw["year"] >= 2024].groupby("driver")[LAP_FEATS].mean()
    common     = sigs_train.index.intersection(sigs_test.index)

    if len(common) >= K_DRIVER:
        sc2 = StandardScaler()
        pca2 = PCA(n_components=2, random_state=42)
        X_tr_t = pca2.fit_transform(sc2.fit_transform(sigs_train.loc[common].values))
        X_te_t = pca2.transform(sc2.transform(sigs_test.loc[common].values))
        km_t   = KMeans(n_clusters=K_DRIVER, random_state=42, n_init=50)
        km_t.fit(X_tr_t)
        labels_te_t = km_t.predict(X_te_t)
        sil_t = silhouette_score(X_te_t, labels_te_t)

        # ARI vs etiquetas originales
        y_orig_common = df.loc[df.index.isin(common), "cluster"].values
        labels_aligned = align_labels(y_orig_common, labels_te_t, K_DRIVER)
        ari_t = adjusted_rand_score(y_orig_common, labels_aligned)

        met["sil_temporal"] = round(float(sil_t), 4)
        met["ari_temporal"] = round(float(ari_t), 4)
        print(f"\n  Temporal CV (≤2023 → ≥2024):")
        print(f"    Pilotos comunes: {len(common)}")
        print(f"    Silhouette test: {sil_t:.4f}")
        print(f"    ARI temporal:    {ari_t:.4f}")
    else:
        met["sil_temporal"] = float("nan")
        met["ari_temporal"] = float("nan")
        print(f"\n  Temporal CV: pilotos comunes insuficientes ({len(common)})")

    return {"classifier": "drivers", **met}


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  TALOS F1 — Validación de los 3 Clasificadores K-means       ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    results = [
        validate_circuits(),
        validate_laps(),
        validate_drivers(),
    ]

    df_report = pd.DataFrame(results).set_index("classifier")
    df_report.to_csv(OUT_CSV)

    print(f"\n{'=' * 65}")
    print("  RESUMEN FINAL")
    print(f"{'=' * 65}")
    cols = ["silhouette", "calinski_harabasz", "davies_bouldin",
            "pct_bien_asignados", "accuracy_loo", "sil_test_5fold", "sil_temporal", "ari_temporal"]
    print(df_report[[c for c in cols if c in df_report.columns]].to_string())
    print(f"\n  Reporte guardado en: {OUT_CSV}")
