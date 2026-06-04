"""
driver_style_classifier.py — TALOS F1 · K-means de Estilos de Piloto
======================================================================

A partir de lap_features_raw.csv (generado por lap_data_scraper.py),
descubre arquetipos de pilotaje agrupando a los pilotos por su FIRMA
DE ESTILO — la deviación media de cada feature respecto a lo "esperado"
en cada (circuito, tipo de sesión).

DIFERENCIA CLAVE vs el LapTypeClassifier (Fase A):

  Fase A: residualizar por GP        → describe la EJECUCIÓN de una vuelta
  Fase C: residualizar por (GP, sesión) → quita TAMBIÉN el efecto de Q vs R

  Después de residualizar por (GP, sesión), lo que queda en cada vuelta es:
  - El estilo personal del piloto (la firma)
  - + ruido aleatorio

  Al promediar muchas vueltas del mismo piloto, el ruido se cancela
  y queda solo la firma. Eso es lo que clusterizamos.

Pipeline:
  PASO 0 — Cargar CSV, filtrar pilotos con pocas vueltas
  PASO 1 — Residualizar features por (GP, session_type)
  PASO 2 — Agrupar por piloto: media de residuales = firma del piloto
  PASO 3 — StandardScaler sobre firmas
  PASO 4 — Elbow + Silhouette → elegir K
  PASO 5 — K-means con K elegido
  PASO 6 — Imprimir clusters con pilotos para etiquetar
  PASO 7 — PCA 2D scatter con nombres de pilotos (validación visual)
  PASO 8 — Guardar resultados + imprimir constantes para módulo live

Uso:
  python driver_style_classifier.py
"""

import os
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# ── Rutas ─────────────────────────────────────────────────────────────────────
# Input + artefactos intermedios viven en research/data/ (training pipeline)
DATA_DIR          = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)
INPUT_CSV         = os.path.join(DATA_DIR, "lap_features_raw.csv")
OUTPUT_CLUSTERS   = os.path.join(DATA_DIR, "driver_clusters.csv")

# El CSV de producción va directamente a src/data/ (lo lee DriverStyleClassifier)
SRC_DATA_DIR      = os.path.join(os.path.dirname(__file__), "..", "src", "data")
os.makedirs(SRC_DATA_DIR, exist_ok=True)
OUTPUT_SIGNATURES = os.path.join(SRC_DATA_DIR, "driver_signatures.csv")

# Features (mismas 8 que en Fase A)
FEATURE_COLS = [
    "p_full", "p_part", "p_coast", "p_brk",
    "decel_p10", "shifts_per_km", "throttle_avg", "coast_avg_len",
]

# Mínimo de vueltas por piloto para entrar al modelo (estabilidad estadística)
MIN_LAPS_PER_DRIVER = 100

# K default (lo decide elbow+silhouette en runtime)
K_DEFAULT = 4


# ─────────────────────────────────────────────────────────────────────────────
# PASO 0 — CARGAR Y FILTRAR
# ─────────────────────────────────────────────────────────────────────────────

def load_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas en el CSV: {missing}")

    df = df.dropna(subset=FEATURE_COLS)

    # Filtrar pilotos con pocas vueltas (no podemos sacar firma estable de < 100 vueltas)
    counts = df["driver"].value_counts()
    valid_drivers = counts[counts >= MIN_LAPS_PER_DRIVER].index.tolist()
    n_removed = df["driver"].nunique() - len(valid_drivers)

    print(f"✓ CSV cargado: {len(df)} vueltas · {df['driver'].nunique()} pilotos")
    if n_removed > 0:
        excluded = counts[counts < MIN_LAPS_PER_DRIVER].to_dict()
        print(f"  ⚠ Excluidos {n_removed} pilotos con < {MIN_LAPS_PER_DRIVER} vueltas:")
        for d, c in excluded.items():
            print(f"      {d}: {c} vueltas")
    print(f"✓ Pilotos válidos: {len(valid_drivers)}\n")

    df = df[df["driver"].isin(valid_drivers)].reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# PASO 1 — RESIDUALIZAR POR (GP, SESSION_TYPE)
# ─────────────────────────────────────────────────────────────────────────────

def residualize_by_context(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resta a cada vuelta la media de su contexto exacto (gp + session_type).

    Esto quita el efecto de "qué tipo de vuelta es" — una Q3 en Monza y una
    race lap en Mónaco están en grupos distintos. Después, lo que queda en
    el residual es purely el estilo personal del piloto.
    """
    df_residual = df.copy()
    for col in FEATURE_COLS:
        df_residual[col] = df[col] - df.groupby(["gp", "session_type"])[col].transform("mean")

    print(f"✓ Residualización por (GP, session_type) aplicada · "
          f"{df.groupby(['gp', 'session_type']).ngroups} buckets de contexto\n")
    return df_residual


# ─────────────────────────────────────────────────────────────────────────────
# PASO 2 — AGREGAR POR PILOTO (FIRMA = MEDIA DE RESIDUALES)
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_by_driver(df_residual: pd.DataFrame) -> pd.DataFrame:
    """
    Una fila por piloto. Cada columna es la media del residual de ese piloto
    sobre todas sus vueltas. Esto ES la firma de estilo personal.
    """
    signatures = df_residual.groupby("driver")[FEATURE_COLS].mean()
    counts = df_residual["driver"].value_counts()
    signatures["n_laps"] = counts

    print(f"✓ Firmas calculadas para {len(signatures)} pilotos\n")
    print("Vista previa (ordenado por p_full residual, ataque ↓):")
    print(signatures.sort_values("p_full", ascending=False).round(2).to_string())
    print()

    return signatures


# ─────────────────────────────────────────────────────────────────────────────
# PASO 3 — STANDARDSCALER
# ─────────────────────────────────────────────────────────────────────────────

def normalize_signatures(signatures: pd.DataFrame) -> tuple[np.ndarray, StandardScaler]:
    """
    Normaliza las firmas para que cada feature pese igual en el K-means.
    No usamos n_laps en el modelo, solo lo conservamos como metadata.
    """
    scaler = StandardScaler()
    X = scaler.fit_transform(signatures[FEATURE_COLS])
    print(f"✓ StandardScaler aplicado sobre firmas")
    print(f"  Std por feature (los pesos efectivos): {scaler.scale_.round(3)}\n")
    return X, scaler


# ─────────────────────────────────────────────────────────────────────────────
# PASO 4 — ELBOW + SILHOUETTE
# ─────────────────────────────────────────────────────────────────────────────

def plot_elbow_and_silhouette(X: np.ndarray, ks: range = range(2, 7)) -> None:
    """Mismo gráfico dual-axis que en Fase A."""
    inertias    = []
    silhouettes = []

    print("  Calculando elbow + silhouette...")
    for k in ks:
        km = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        # Con ~20 pilotos no hace falta subsample
        sil = silhouette_score(X, labels)
        silhouettes.append(sil)
        print(f"    K={k}: inertia={km.inertia_:.1f} · silhouette={sil:.3f}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(ks), y=inertias, mode="lines+markers",
        name="Inercia (↓ mejor)",
        marker=dict(size=10, color="#E8002D"),
        line=dict(color="#E8002D", width=2),
        yaxis="y",
    ))
    fig.add_trace(go.Scatter(
        x=list(ks), y=silhouettes, mode="lines+markers",
        name="Silhouette (↑ mejor)",
        marker=dict(size=10, color="#39FF14"),
        line=dict(color="#39FF14", width=2),
        yaxis="y2",
    ))
    fig.update_layout(
        title="Elbow + Silhouette · K óptimo para estilos de piloto",
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
    print()


# ─────────────────────────────────────────────────────────────────────────────
# PASO 5 — K-MEANS
# ─────────────────────────────────────────────────────────────────────────────

def run_kmeans(X: np.ndarray, k: int) -> tuple[np.ndarray, KMeans]:
    # Clusterizar en el espacio PCA-2 (igual que el modelo desplegado). Las 8
    # features están muy correlacionadas; proyectar a sus 2 componentes
    # principales (~65% de varianza) casi duplica la separación de los clusters.
    Xp = PCA(n_components=2, random_state=42).fit_transform(X)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=50)
    labels = kmeans.fit_predict(Xp)
    print(f"✓ K-means (K={k}) sobre PCA-2 · inercia={kmeans.inertia_:.2f}")
    counts = {i: int((labels == i).sum()) for i in range(k)}
    print(f"  Pilotos por cluster: {counts}\n")
    return labels, kmeans


# ─────────────────────────────────────────────────────────────────────────────
# PASO 6 — IMPRIMIR CLUSTERS CON NOMBRES DE PILOTOS
# ─────────────────────────────────────────────────────────────────────────────

def print_clusters(signatures: pd.DataFrame) -> None:
    """
    Imprime cada cluster con los pilotos que lo componen + sus firmas.
    Este es el momento clave de etiquetado a mano.
    """
    print("=" * 72)
    print("  CLUSTERS DE ESTILO — etiqueta cada grupo según los pilotos que aparecen")
    print("=" * 72)

    for cid in sorted(signatures["cluster"].unique()):
        grupo = signatures[signatures["cluster"] == cid].sort_values("n_laps", ascending=False)

        print(f"\n┌─ CLUSTER {cid} ({len(grupo)} pilotos) {'─' * 40}")
        print(f"│  PILOTOS: {', '.join(grupo.index.tolist())}")
        print(f"│")
        print(f"│  FIRMA MEDIA (residual vs media de contexto):")
        for col in FEATURE_COLS:
            mean_res = grupo[col].mean()
            sign = "+" if mean_res >= 0 else ""
            print(f"│    {col:<18s} {sign}{mean_res:>7.3f}")

        # Top 3 pilotos del cluster por número de vueltas (las firmas más estables)
        print(f"│")
        print(f"│  Firmas detalladas (top 3 por nº vueltas):")
        for driver, row in grupo.head(3).iterrows():
            vals = " · ".join(
                f"{c[:6]} {row[c]:+.2f}"
                for c in ["p_full", "p_part", "p_coast", "p_brk"]
            )
            print(f"│    {driver} ({int(row['n_laps'])} laps): {vals}")
        print(f"└{'─' * 60}")

    print()


# ─────────────────────────────────────────────────────────────────────────────
# PASO 7 — PCA 2D CON NOMBRES DE PILOTOS
# ─────────────────────────────────────────────────────────────────────────────

CLUSTER_COLORS = ["#E8002D", "#00D2FF", "#39FF14", "#C77DFF", "#FFA500", "#FFD600"]


def plot_pca(signatures: pd.DataFrame, X: np.ndarray, cluster_labels: dict[int, str]) -> None:
    """
    Scatter PCA 2D con cada piloto etiquetado. Validación visual brutal:
    si VER y PER caen en clusters distintos hay diferencia de estilos real,
    si HAM y ALO caen juntos, validamos el cluster "smooth", etc.
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

    fig = go.Figure()
    for cid in sorted(signatures["cluster"].unique()):
        mask = signatures["cluster"] == cid
        label = cluster_labels.get(cid, f"Cluster {cid}")
        fig.add_trace(go.Scatter(
            x=coords[mask, 0], y=coords[mask, 1],
            mode="markers+text",
            text=signatures.index[mask].tolist(),
            textposition="top center",
            textfont=dict(size=12, color="rgba(255,255,255,0.9)", family="JetBrains Mono"),
            name=f"{label} ({mask.sum()})",
            marker=dict(
                size=14,
                color=CLUSTER_COLORS[cid % len(CLUSTER_COLORS)],
                line=dict(color="white", width=1),
            ),
            hovertemplate=f"<b>%{{text}}</b><br>{label}<br>PC1=%{{x:.2f}}<br>PC2=%{{y:.2f}}<extra></extra>",
        ))

    fig.update_layout(
        title=dict(text="Estilos de Pilotaje · PCA 2D", font=dict(size=18, color="white")),
        xaxis=dict(
            title=f"PC1 ({var_explained[0]*100:.1f}% var)",
            gridcolor="rgba(255,255,255,0.08)", color="rgba(255,255,255,0.6)",
            zeroline=True, zerolinecolor="rgba(255,255,255,0.15)",
        ),
        yaxis=dict(
            title=f"PC2 ({var_explained[1]*100:.1f}% var)",
            gridcolor="rgba(255,255,255,0.08)", color="rgba(255,255,255,0.6)",
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
# PASO 8 — GUARDAR + IMPRIMIR CONSTANTES
# ─────────────────────────────────────────────────────────────────────────────

def save_results(
    signatures: pd.DataFrame,
    kmeans: KMeans,
    scaler: StandardScaler,
    cluster_labels: dict[int, str],
) -> None:
    # 1. Firmas (1 fila por piloto, todas las features)
    out_sig = signatures.copy()
    out_sig["label"] = out_sig["cluster"].map(cluster_labels)
    out_sig.to_csv(OUTPUT_SIGNATURES)
    print(f"💾 Firmas: {OUTPUT_SIGNATURES}")

    # 2. Resumen de clusters (1 fila por cluster, medias de firmas)
    summary = []
    for cid in sorted(signatures["cluster"].unique()):
        grupo = signatures[signatures["cluster"] == cid]
        row = {
            "cluster": cid,
            "label": cluster_labels.get(cid, "?"),
            "n_drivers": len(grupo),
            "drivers": ",".join(grupo.index.tolist()),
        }
        for col in FEATURE_COLS:
            row[f"{col}_mean"] = round(float(grupo[col].mean()), 3)
        summary.append(row)
    pd.DataFrame(summary).to_csv(OUTPUT_CLUSTERS, index=False)
    print(f"💾 Resumen:  {OUTPUT_CLUSTERS}")

    # 3. Constantes para el módulo live
    print(f"\n{'=' * 72}")
    print(f"  CONSTANTES PARA src/DriverStyleClassifier.py (copia-pega)")
    print(f"{'=' * 72}")

    print(f"\nFEATURE_COLS = {FEATURE_COLS!r}")

    print(f"\nFEATURE_STD (del StandardScaler):")
    print(f"  np.array({[float(x) for x in scaler.scale_.round(4)]})")

    print(f"\nCLUSTER_CENTROIDS (espacio escalado):")
    print(f"{{")
    for i, c in enumerate(kmeans.cluster_centers_):
        vals = [float(x) for x in c.round(4)]
        print(f"  {i}: np.array({vals}),")
    print(f"}}")

    print(f"\nCLUSTER_LABELS:")
    print(f"  {{")
    for cid, lbl in sorted(cluster_labels.items()):
        print(f"    {cid}: \"{lbl}\",")
    print(f"  }}")

    print(f"\nDRIVER_FEATURES (firma + cluster_id):")
    print(f"{{")
    for driver in sorted(signatures.index.tolist()):
        row = signatures.loc[driver]
        vals = [round(float(row[c]), 4) for c in FEATURE_COLS]
        print(f"  \"{driver}\": ({', '.join(map(str, vals))}, {int(row['cluster'])}),")
    print(f"}}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not os.path.exists(INPUT_CSV):
        print(f"❌ No se encuentra {INPUT_CSV}")
        print("   Ejecuta primero: python lap_data_scraper.py")
        sys.exit(1)

    print(f"\n🏎  TALOS · Driver Style Classifier")
    print(f"   Input:           {INPUT_CSV}")
    print(f"   Min laps/driver: {MIN_LAPS_PER_DRIVER}")
    print(f"   Features (8):    {FEATURE_COLS}\n")
    print("─" * 72)

    # ── Pasos 0-3 ───────────────────────────────────────────────────────
    df = load_csv(INPUT_CSV)
    df_residual = residualize_by_context(df)
    signatures = aggregate_by_driver(df_residual)
    X, scaler = normalize_signatures(signatures)

    # ── Paso 4: elbow + silhouette ──────────────────────────────────────
    print("─" * 72)
    print("  ELBOW + SILHOUETTE — cerrar la ventana para continuar")
    print("─" * 72)
    plot_elbow_and_silhouette(X)

    # ── Paso 5: K-means ─────────────────────────────────────────────────
    k_input = input(f"¿Qué K usamos? (default {K_DEFAULT}): ").strip()
    k = int(k_input) if k_input else K_DEFAULT

    labels, kmeans = run_kmeans(X, k)
    signatures["cluster"] = labels

    # ── Paso 6: imprimir clusters ───────────────────────────────────────
    print_clusters(signatures)

    # ── Etiquetado manual ───────────────────────────────────────────────
    print("─" * 72)
    print("  ETIQUETADO — escribe un nombre para cada cluster")
    print("  (sugerencias: Atacador puro · Smooth · Modulador · Conservador · ...)")
    print("─" * 72)
    cluster_labels = {}
    for cid in sorted(set(labels)):
        nombre = input(f"  Nombre para Cluster {cid}: ").strip()
        cluster_labels[cid] = nombre if nombre else f"Cluster {cid}"

    # ── Paso 7: PCA visual con nombres ──────────────────────────────────
    plot_pca(signatures, X, cluster_labels)

    # ── Paso 8: guardar ─────────────────────────────────────────────────
    save_results(signatures, kmeans, scaler, cluster_labels)

    print("\n✅ Listo. Revisa el PCA — los pilotos del mismo cluster deben estar")
    print("   cerca, y debes ver agrupaciones intuitivas (HAM con ALO, VER con GAS, etc.).")
    print("   Si los grupos son raros → K mal o features insuficientes.")
    print("   Si parecen lógicos → copia las constantes a src/DriverStyleClassifier.py\n")
