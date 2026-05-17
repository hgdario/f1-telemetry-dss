"""
lap_type_classifier.py — TALOS F1 · K-means de Tipos de Vuelta
================================================================

A partir de lap_features_raw.csv (generado por lap_data_scraper.py),
clasifica cada vuelta en uno de N grupos naturales de pilotaje.

DIFERENCIA CLAVE respecto al CircuitClassifier:
  Antes de clusterizar RESIDUALIZAMOS por GP — restamos a cada vuelta
  la media de su circuito. Sin esto, los clusters reflejan el CIRCUITO
  en lugar del TIPO DE VUELTA (un 55% p_full es "ataque" en Mónaco pero
  "gestión" en Monza). Después de residualizar, los clusters salen
  comparables entre trazados.

Pipeline:
  PASO 0 — Cargar CSV y descartar features inútiles (speed_std_norm)
  PASO 1 — Residualizar features por GP
  PASO 2 — StandardScaler sobre residuales
  PASO 3 — Isolation Forest para detectar outliers
  PASO 4 — Elbow + Silhouette → elegir K
  PASO 5 — K-means con K elegido
  PASO 6 — Imprimir clusters para etiquetar
  PASO 7 — PCA 2D para validación visual
  PASO 8 — Guardar resultados + imprimir constantes para el módulo live

Uso:
  python lap_type_classifier.py
"""

import os
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# ── Rutas ─────────────────────────────────────────────────────────────────────
DATA_DIR          = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)
INPUT_CSV         = os.path.join(DATA_DIR, "lap_features_raw.csv")
OUTPUT_SUMMARY    = os.path.join(DATA_DIR, "lap_type_clusters_summary.csv")
OUTPUT_ASSIGNMENT = os.path.join(DATA_DIR, "lap_type_assignments.csv")
OUTPUT_GP_MEANS   = os.path.join(DATA_DIR, "lap_type_gp_means.csv")

# ── Features que entran al modelo ─────────────────────────────────────────────
# speed_std_norm queda fuera: std=0.01 → es prácticamente una constante
# y no aporta señal al K-means (todas las vueltas serían iguales en ese eje)
FEATURE_COLS = [
    "p_full",
    "p_part",
    "p_coast",
    "p_brk",
    "decel_p10",
    "shifts_per_km",
    "throttle_avg",
    "coast_avg_len",
]

# Default K (lo decide elbow+silhouette en runtime, esto es el fallback)
K_DEFAULT = 4


# ─────────────────────────────────────────────────────────────────────────────
# PASO 0 — CARGAR
# ─────────────────────────────────────────────────────────────────────────────

def load_csv(csv_path: str) -> pd.DataFrame:
    """Carga el CSV de vueltas y filtra columnas necesarias."""
    df = pd.read_csv(csv_path)

    # Validar features
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas en el CSV: {missing}")

    # Filtrar NaN en features (raro pero por si acaso)
    n_before = len(df)
    df = df.dropna(subset=FEATURE_COLS)
    if len(df) < n_before:
        print(f"  ⚠  Descartadas {n_before - len(df)} filas con NaN en features")

    print(f"✓ CSV cargado: {len(df)} vueltas · "
          f"{df['gp'].nunique()} GPs · "
          f"{df['session_type'].nunique()} tipos sesión · "
          f"{df['driver'].nunique()} pilotos\n")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# PASO 1 — RESIDUALIZAR POR GP
# ─────────────────────────────────────────────────────────────────────────────

def residualize_by_gp(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Resta la media del GP a cada feature, vuelta por vuelta.

    El objetivo es eliminar el efecto del CIRCUITO de los datos. Después
    de residualizar, una vuelta de Mónaco con p_full=50% (media Monaco=44%)
    tendrá residual=+6, mientras que una vuelta de Monza con p_full=70%
    (media Monza=75%) tendrá residual=-5. Ahora son comparables: la primera
    es "más ataque de lo normal", la segunda "menos".

    Retorna:
      - df_residual: DataFrame con features residualizadas (mismas columnas)
      - gp_means:    DataFrame con la media de cada feature por GP (para inferencia)
    """
    gp_means = df.groupby("gp")[FEATURE_COLS].mean()
    print("✓ Medias por GP (de las que se restará):")
    print(gp_means.round(2).to_string())
    print()

    df_residual = df.copy()
    for col in FEATURE_COLS:
        df_residual[col] = df[col] - df.groupby("gp")[col].transform("mean")

    print(f"✓ Residualización aplicada · ahora cada vuelta es 'cuánto se desvía '")
    print(f"  de la media de su GP en cada feature\n")

    return df_residual, gp_means


# ─────────────────────────────────────────────────────────────────────────────
# PASO 2 — STANDARDSCALER
# ─────────────────────────────────────────────────────────────────────────────

def normalize(df_residual: pd.DataFrame) -> tuple[np.ndarray, StandardScaler]:
    """
    Normaliza los residuales a media 0 y std 1. Necesario porque las features
    tienen escalas muy distintas (p_full std~12, coast_avg_len std~0.15).
    """
    scaler = StandardScaler()
    X = scaler.fit_transform(df_residual[FEATURE_COLS])
    print(f"✓ StandardScaler aplicado sobre residuales")
    print(f"  Mean por feature (debe ser ≈0): {scaler.mean_.round(3)}")
    print(f"  Std por feature  (los pesos):   {scaler.scale_.round(3)}\n")
    return X, scaler


# ─────────────────────────────────────────────────────────────────────────────
# PASO 3 — ISOLATION FOREST · DETECCIÓN DE OUTLIERS
# ─────────────────────────────────────────────────────────────────────────────

def detect_outliers(X: np.ndarray, contamination: float = 0.05) -> np.ndarray:
    """
    Detecta outliers con Isolation Forest sobre los residuales escalados.

    Vueltas tras SC, in/out laps, telemetría corrupta etc tienen perfiles
    extremos que tiran de los centroides del K-means. Las separamos antes.

    contamination=0.05 → asume que ~5% de las vueltas son anómalas.
    Retorna máscara booleana: True = inlier (vuelta normal), False = outlier.
    """
    iso = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_estimators=200,
    )
    preds = iso.fit_predict(X)
    inlier_mask = preds == 1
    n_outliers = (~inlier_mask).sum()

    print(f"✓ Isolation Forest · {n_outliers} outliers detectados "
          f"({n_outliers/len(X)*100:.1f}% del dataset)")
    print(f"  Estos no entran al K-means pero se etiquetarán como 'Atípico'\n")

    return inlier_mask


# ─────────────────────────────────────────────────────────────────────────────
# PASO 4 — ELBOW + SILHOUETTE → ELEGIR K
# ─────────────────────────────────────────────────────────────────────────────

def plot_elbow_and_silhouette(X: np.ndarray, ks: range = range(2, 9)) -> None:
    """
    Prueba K de 2 a 8 y muestra:
      - Inercia (curva descendente) → elbow method
      - Silhouette score (curva con máximo) → calidad de separación
    El K óptimo está en el codo de la inercia Y maximiza la silhouette.
    """
    inertias    = []
    silhouettes = []

    # Subsampleo para silhouette si el dataset es grande (~13k → 5k)
    if len(X) > 5000:
        rng = np.random.default_rng(42)
        sample_idx = rng.choice(len(X), 5000, replace=False)
        X_sample = X[sample_idx]
    else:
        X_sample = X

    print("  Calculando elbow + silhouette (puede tardar)...")
    for k in ks:
        km = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        # Silhouette se calcula sobre la muestra (mucho más rápido)
        labels_sample = km.predict(X_sample)
        sil = silhouette_score(X_sample, labels_sample)
        silhouettes.append(sil)
        print(f"    K={k}: inertia={km.inertia_:.0f} · silhouette={sil:.3f}")

    # Doble eje Y: inercia + silhouette
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(ks), y=inertias,
        mode="lines+markers",
        name="Inercia (↓ mejor)",
        marker=dict(size=10, color="#E8002D"),
        line=dict(color="#E8002D", width=2),
        yaxis="y",
    ))
    fig.add_trace(go.Scatter(
        x=list(ks), y=silhouettes,
        mode="lines+markers",
        name="Silhouette (↑ mejor)",
        marker=dict(size=10, color="#39FF14"),
        line=dict(color="#39FF14", width=2),
        yaxis="y2",
    ))
    fig.update_layout(
        title="Elbow + Silhouette · K óptimo",
        xaxis=dict(title="K (nº clusters)", dtick=1, color="rgba(255,255,255,0.6)"),
        yaxis=dict(title="Inercia", color="#E8002D", side="left"),
        yaxis2=dict(title="Silhouette", color="#39FF14", side="right", overlaying="y"),
        paper_bgcolor="#0E0E0F",
        plot_bgcolor="#111115",
        font=dict(family="JetBrains Mono, monospace", color="white"),
        width=850, height=470,
        legend=dict(bgcolor="rgba(0,0,0,0.4)", bordercolor="rgba(255,255,255,0.2)"),
    )
    fig.show()
    print("\n✓ Curvas mostradas — elige K en el codo de inercia que también "
          "maximice silhouette\n")


# ─────────────────────────────────────────────────────────────────────────────
# PASO 5 — K-MEANS
# ─────────────────────────────────────────────────────────────────────────────

def run_kmeans(X: np.ndarray, k: int) -> tuple[np.ndarray, KMeans]:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = kmeans.fit_predict(X)
    print(f"✓ K-means (K={k}) · inercia={kmeans.inertia_:.0f}")
    counts = {i: int((labels == i).sum()) for i in range(k)}
    print(f"  Vueltas por cluster: {counts}\n")
    return labels, kmeans


# ─────────────────────────────────────────────────────────────────────────────
# PASO 6 — IMPRIMIR CLUSTERS PARA ETIQUETAR
# ─────────────────────────────────────────────────────────────────────────────

def print_clusters(df_assigned: pd.DataFrame, gp_means: pd.DataFrame) -> None:
    """
    Imprime el perfil de cada cluster:
      - Medias de features RESIDUALES (lo que define el cluster)
      - Composición por GP y por sesión (validación cualitativa)
      - Tiempo medio de vuelta (sanity check)
    Tras leer esto, el usuario etiqueta cada cluster a mano.
    """
    print("=" * 72)
    print("  CLUSTERS — perfila cada grupo para etiquetarlo a mano")
    print("=" * 72)

    for cluster_id in sorted(df_assigned["cluster"].unique()):
        if cluster_id == -1:
            continue  # outliers van al final
        grupo = df_assigned[df_assigned["cluster"] == cluster_id]

        print(f"\n┌─ CLUSTER {cluster_id} ({len(grupo)} vueltas) {'─' * 40}")
        print(f"│  RESIDUALES MEDIOS (vs media del GP):")
        for col in FEATURE_COLS:
            mean_res = grupo[f"{col}_res"].mean()
            sign = "+" if mean_res >= 0 else ""
            print(f"│    {col:<18s} {sign}{mean_res:>7.2f}")

        # Composición por tipo de sesión (Q/R/FP3)
        sess_dist = grupo["session_type"].value_counts(normalize=True) * 100
        print(f"│")
        print(f"│  Composición por sesión:")
        for sess, pct in sess_dist.items():
            print(f"│    {sess:<6s} {pct:>5.1f}%")

        # GPs más representados (top 3)
        gp_dist = grupo["gp"].value_counts().head(3)
        print(f"│  Top GPs: {', '.join(f'{g} ({c})' for g, c in gp_dist.items())}")

        # Lap time medio
        if "lap_time_s" in grupo.columns:
            print(f"│  Tiempo medio vuelta: {grupo['lap_time_s'].mean():.2f}s")

        print(f"└{'─' * 60}")

    # Outliers
    n_outliers = (df_assigned["cluster"] == -1).sum()
    if n_outliers > 0:
        print(f"\n┌─ OUTLIERS ({n_outliers} vueltas) — detectados por Isolation Forest")
        out = df_assigned[df_assigned["cluster"] == -1]
        sess_dist = out["session_type"].value_counts(normalize=True) * 100
        print(f"│  Composición: {dict(sess_dist.round(1))}")
        print(f"│  Top GPs: {', '.join(out['gp'].value_counts().head(3).index)}")
        print(f"│  Probables vueltas atípicas: in/out, SC, telemetría dañada")
        print(f"└{'─' * 60}")

    print()


# ─────────────────────────────────────────────────────────────────────────────
# PASO 7 — PCA 2D PARA VALIDACIÓN VISUAL
# ─────────────────────────────────────────────────────────────────────────────

CLUSTER_COLORS = ["#E8002D", "#00D2FF", "#39FF14", "#C77DFF", "#FFA500", "#FFD600"]


def plot_pca(df_assigned: pd.DataFrame, X: np.ndarray,
             inlier_mask: np.ndarray, cluster_labels: dict[int, str]) -> None:
    """
    Reduce a 2D con PCA y dibuja un scatter coloreado por cluster.
    Sirve para validar visualmente que los clusters están separados.
    También imprime las loadings de cada feature en PC1 y PC2.
    """
    pca = PCA(n_components=2)
    coords = pca.fit_transform(X)
    var_explained = pca.explained_variance_ratio_

    print(f"\n✓ PCA 2D · varianza explicada: PC1={var_explained[0]*100:.1f}%  "
          f"PC2={var_explained[1]*100:.1f}%  · Total={var_explained.sum()*100:.1f}%")
    print(f"\n  LOADINGS (peso de cada feature en cada componente):")
    print(f"  {'Feature':<20} {'PC1':>8}  {'PC2':>8}")
    for i, feat in enumerate(FEATURE_COLS):
        print(f"  {feat:<20} {pca.components_[0][i]:>+8.3f}  {pca.components_[1][i]:>+8.3f}")
    print()

    # Plot
    fig = go.Figure()

    # Inliers coloreados por cluster
    inlier_df = df_assigned[inlier_mask].copy()
    inlier_coords = coords  # X ya era solo inliers
    for cid in sorted(inlier_df["cluster"].unique()):
        if cid == -1:
            continue
        mask = inlier_df["cluster"] == cid
        label = cluster_labels.get(cid, f"Cluster {cid}")
        # Subsample para no saturar el plot
        idx = np.where(mask)[0]
        if len(idx) > 800:
            rng = np.random.default_rng(42)
            idx = rng.choice(idx, 800, replace=False)
        fig.add_trace(go.Scatter(
            x=inlier_coords[idx, 0], y=inlier_coords[idx, 1],
            mode="markers",
            name=f"{label} ({mask.sum()})",
            marker=dict(
                size=5,
                color=CLUSTER_COLORS[cid % len(CLUSTER_COLORS)],
                opacity=0.6,
                line=dict(width=0),
            ),
            hovertemplate=f"<b>{label}</b><br>PC1=%{{x:.2f}}<br>PC2=%{{y:.2f}}<extra></extra>",
        ))

    fig.update_layout(
        title=dict(
            text="Clusters de Tipos de Vuelta · PCA 2D",
            font=dict(size=18, color="white"),
        ),
        xaxis=dict(
            title=f"PC1 ({var_explained[0]*100:.1f}% var)",
            gridcolor="rgba(255,255,255,0.08)",
            color="rgba(255,255,255,0.6)",
            zeroline=True, zerolinecolor="rgba(255,255,255,0.15)",
        ),
        yaxis=dict(
            title=f"PC2 ({var_explained[1]*100:.1f}% var)",
            gridcolor="rgba(255,255,255,0.08)",
            color="rgba(255,255,255,0.6)",
            zeroline=True, zerolinecolor="rgba(255,255,255,0.15)",
        ),
        paper_bgcolor="#0E0E0F",
        plot_bgcolor="#111115",
        font=dict(family="JetBrains Mono, monospace", color="white"),
        legend=dict(bgcolor="rgba(0,0,0,0.4)", bordercolor="rgba(255,255,255,0.2)"),
        width=900, height=650,
    )
    fig.show()


# ─────────────────────────────────────────────────────────────────────────────
# PASO 8 — GUARDAR
# ─────────────────────────────────────────────────────────────────────────────

def save_results(
    df_assigned: pd.DataFrame,
    gp_means: pd.DataFrame,
    kmeans: KMeans,
    scaler: StandardScaler,
    cluster_labels: dict[int, str],
) -> None:
    """
    Guarda 3 CSVs:
      - lap_type_assignments.csv: una fila por vuelta con cluster + label
      - lap_type_clusters_summary.csv: una fila por cluster con stats
      - lap_type_gp_means.csv: medias por GP para residualizar en inferencia
    """
    # 1. Asignaciones por vuelta (con metadata)
    cols_keep = ["year", "gp", "session_type", "driver", "lap_number",
                 "lap_time_s", "compound", "cluster"]
    df_out = df_assigned[cols_keep].copy()
    df_out["label"] = df_out["cluster"].map(cluster_labels)
    df_out.to_csv(OUTPUT_ASSIGNMENT, index=False)
    print(f"💾 Asignaciones: {OUTPUT_ASSIGNMENT}")

    # 2. Resumen por cluster (medias de features residualizadas)
    summary = []
    for cid in sorted(df_assigned["cluster"].unique()):
        grupo = df_assigned[df_assigned["cluster"] == cid]
        row = {
            "cluster": cid,
            "label": cluster_labels.get(cid, "Atípico"),
            "n_laps": len(grupo),
        }
        for col in FEATURE_COLS:
            row[f"{col}_res_mean"] = round(float(grupo[f"{col}_res"].mean()), 3)
            row[f"{col}_raw_mean"] = round(float(grupo[col].mean()), 3)
        summary.append(row)
    pd.DataFrame(summary).to_csv(OUTPUT_SUMMARY, index=False)
    print(f"💾 Resumen:      {OUTPUT_SUMMARY}")

    # 3. Medias por GP (para residualización en inferencia)
    gp_means.to_csv(OUTPUT_GP_MEANS)
    print(f"💾 GP means:     {OUTPUT_GP_MEANS}")

    # 4. Constantes para copiar al módulo live (LapTypeClassifier.py)
    print(f"\n{'=' * 72}")
    print(f"  CONSTANTES PARA src/LapTypeClassifier.py (copia-pega)")
    print(f"{'=' * 72}")
    print(f"\nFEATURE_COLS = {FEATURE_COLS!r}")
    print(f"\nFEATURE_MEAN (de StandardScaler — debe ser ≈0 por residualización):")
    print(f"  np.array({list(scaler.mean_.round(4))})")
    print(f"\nFEATURE_STD (de StandardScaler):")
    print(f"  np.array({list(scaler.scale_.round(4))})")
    print(f"\nCLUSTER_CENTROIDS (en espacio residualizado+escalado):")
    print(f"{{")
    for i, c in enumerate(kmeans.cluster_centers_):
        print(f"  {i}: np.array({list(c.round(4))}),")
    print(f"}}")
    print(f"\nCLUSTER_LABELS:")
    print(f"{cluster_labels}\n")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not os.path.exists(INPUT_CSV):
        print(f"❌ No se encuentra {INPUT_CSV}")
        print("   Ejecuta primero: python lap_data_scraper.py")
        sys.exit(1)

    print(f"\n🏎  TALOS · Lap Type Classifier")
    print(f"   Input:  {INPUT_CSV}")
    print(f"   Output: 3 CSVs en research/")
    print(f"   Features (8): {FEATURE_COLS}\n")
    print("─" * 72)

    # ── Pasos 0-2 ───────────────────────────────────────────────────────
    df = load_csv(INPUT_CSV)
    df_residual, gp_means = residualize_by_gp(df)
    X, scaler = normalize(df_residual)

    # ── Paso 3: outliers ────────────────────────────────────────────────
    print("─" * 72)
    print("  ISOLATION FOREST")
    print("─" * 72)
    inlier_mask = detect_outliers(X, contamination=0.05)
    X_clean = X[inlier_mask]
    df_clean = df.iloc[inlier_mask].reset_index(drop=True)
    df_residual_clean = df_residual.iloc[inlier_mask].reset_index(drop=True)

    # ── Paso 4: elbow + silhouette ──────────────────────────────────────
    print("─" * 72)
    print("  ELBOW + SILHOUETTE — cerrar la ventana para continuar")
    print("─" * 72)
    plot_elbow_and_silhouette(X_clean)

    # ── Paso 5: K-means ─────────────────────────────────────────────────
    k_input = input(f"¿Qué K usamos? (default {K_DEFAULT}): ").strip()
    k = int(k_input) if k_input else K_DEFAULT

    labels, kmeans = run_kmeans(X_clean, k)

    # Construir DataFrame final: assignments + features RAW + features RESIDUAL
    df_assigned = df_clean.copy()
    df_assigned["cluster"] = labels
    for col in FEATURE_COLS:
        df_assigned[f"{col}_res"] = df_residual_clean[col].values

    # Añadir los outliers con cluster=-1
    df_outliers = df.iloc[~inlier_mask].copy()
    df_outliers["cluster"] = -1
    for col in FEATURE_COLS:
        df_outliers[f"{col}_res"] = df_residual.iloc[~inlier_mask][col].values
    df_assigned = pd.concat([df_assigned, df_outliers], ignore_index=True)

    # ── Paso 6: imprimir clusters ───────────────────────────────────────
    print_clusters(df_assigned, gp_means)

    # ── Etiquetado manual ───────────────────────────────────────────────
    print("─" * 72)
    print("  ETIQUETADO — escribe un nombre para cada cluster")
    print("  (ej: Ataque puro, Ritmo de carrera, Lift & Coast, Vuelta defensiva)")
    print("─" * 72)
    cluster_labels = {}
    for cid in sorted(set(labels)):
        nombre = input(f"  Nombre para Cluster {cid}: ").strip()
        cluster_labels[cid] = nombre if nombre else f"Cluster {cid}"
    cluster_labels[-1] = "Atípico"  # outliers automáticamente

    # ── Paso 7: PCA visual ──────────────────────────────────────────────
    plot_pca(df_assigned, X_clean, inlier_mask, cluster_labels)

    # ── Paso 8: guardar ─────────────────────────────────────────────────
    save_results(df_assigned, gp_means, kmeans, scaler, cluster_labels)

    print("\n✅ Listo. Lanza el plot del PCA y revisa la separación visual.")
    print("   Si los clusters se solapan mucho → revisa K o features.")
    print("   Si están bien separados → copia las constantes al módulo live.\n")
