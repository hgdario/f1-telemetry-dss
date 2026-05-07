"""
circuit_classifier.py — TALOS F1 · K-means Clustering de Circuitos
====================================================================

A partir de circuit_features_raw.csv (generado por circuit_data_scraper.py),
clasifica automáticamente los circuitos de F1 en grupos naturales.

Pipeline:
  PASO 1 — Agrupar por circuito (media de todos los años disponibles)
  PASO 2 — Normalizar características (StandardScaler → media 0, std 1)
  PASO 3 — K-means con k=4
  PASO 4 — Imprimir clusters para etiquetar manualmente
  PASO 5 — Visualizar con Plotly (scatter interactivo)
  PASO 6 — Guardar resultado en circuit_clusters.csv

Uso:
  python circuit_classifier.py
"""

import os
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ── Rutas ─────────────────────────────────────────────────────────────────────
INPUT_CSV  = os.path.join(os.path.dirname(__file__), "circuit_features_raw.csv")
OUTPUT_CSV = os.path.join(os.path.dirname(__file__), "circuit_clusters.csv")

# Características que definen el CIRCUITO (no el piloto)
FEATURE_COLS = ["g_lat_mean", "g_lon_mean", "cambios_marcha_km"]
# EXCLUIDOS:
#   g_total_mean    → derivado de sqrt(g_lat²+g_lon²): multicolinealidad, daría triple peso a las G
#   longitud_km     → variable estratégica (nº vueltas, pit stop), no define el ADN del trazado

# Número de clusters (default — se puede cambiar en tiempo de ejecución)
K = 3


# ─────────────────────────────────────────────────────────────────────────────
# PASO 1 — CARGAR Y AGRUPAR POR CIRCUITO
# ─────────────────────────────────────────────────────────────────────────────

def load_and_aggregate(csv_path: str) -> pd.DataFrame:
    """
    Carga el CSV y promedia todos los años de un mismo circuito.
    Resultado: 1 fila por circuito con sus características medias.
    """
    df = pd.read_csv(csv_path)
    print(f"✓ CSV cargado: {len(df)} filas ({df['year'].nunique()} años · {df['circuito'].nunique()} circuitos únicos)")
    print(f"  Años:      {sorted(df['year'].unique())}")
    print(f"  Columnas:  {list(df.columns)}\n")

    # Agrupar por nombre de circuito → media de todos los años
    # longitud_km se mantiene como dato informativo (no entra en el modelo)
    agg_cols = FEATURE_COLS + ["n_vueltas", "longitud_km"]
    df_circuit = (
        df.groupby("circuito")[agg_cols]
        .mean()
        .reset_index()
    )
    df_circuit["n_años"] = df.groupby("circuito")["year"].nunique().values

    print(f"✓ Tras agrupar: {len(df_circuit)} circuitos únicos\n")
    print(df_circuit[["circuito", "g_lat_mean", "g_lon_mean", "cambios_marcha_km", "longitud_km", "n_años"]]
          .sort_values("g_lat_mean", ascending=False)
          .to_string(index=False))
    print()

    return df_circuit


# ─────────────────────────────────────────────────────────────────────────────
# PASO 2 — NORMALIZAR
# ─────────────────────────────────────────────────────────────────────────────

def normalize(df_circuit: pd.DataFrame) -> tuple[np.ndarray, StandardScaler]:
    """
    Normaliza las características a media 0 y std 1.
    Necesario para que K-means no esté dominado por la escala de cada variable.
    """
    scaler = StandardScaler()
    X = scaler.fit_transform(df_circuit[FEATURE_COLS])
    print(f"✓ Normalización aplicada (StandardScaler)")
    print(f"  Medias originales:   { {c: round(df_circuit[c].mean(),3) for c in FEATURE_COLS} }")
    print(f"  Std originales:      { {c: round(df_circuit[c].std(),3)  for c in FEATURE_COLS} }\n")
    return X, scaler


# ─────────────────────────────────────────────────────────────────────────────
# PASO 3 — K-MEANS
# ─────────────────────────────────────────────────────────────────────────────

def run_kmeans(X: np.ndarray, k: int = K) -> np.ndarray:
    """
    Aplica K-means con k clusters.
    random_state=42 para reproducibilidad.
    n_init=20 para evitar mínimos locales.
    """
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = kmeans.fit_predict(X)

    inertia = kmeans.inertia_
    print(f"✓ K-means completado (k={k})")
    print(f"  Inertia (suma distancias²): {inertia:.3f}")
    print(f"  Circuitos por cluster: { {i: int((labels==i).sum()) for i in range(k)} }\n")

    return labels, kmeans


# ─────────────────────────────────────────────────────────────────────────────
# PASO 4 — IMPRIMIR CLUSTERS PARA ETIQUETAR
# ─────────────────────────────────────────────────────────────────────────────

def print_clusters(df_circuit: pd.DataFrame) -> None:
    """
    Imprime cada cluster con sus circuitos y medias de características.
    El objetivo es que puedas leer esto y etiquetar cada cluster manualmente.
    """
    print("=" * 65)
    print("  CLUSTERS — lee esto para etiquetar manualmente cada grupo")
    print("=" * 65)

    for cluster_id in sorted(df_circuit["cluster"].unique()):
        grupo = df_circuit[df_circuit["cluster"] == cluster_id].sort_values("g_lat_mean", ascending=False)

        print(f"\n┌─ CLUSTER {cluster_id} ({len(grupo)} circuitos) {'─'*35}")
        print(f"│  Circuitos: {', '.join(grupo['circuito'].str.replace(' Grand Prix','').values)}")
        print(f"│")
        print(f"│  g_lat_mean        (curvas rápidas): {grupo['g_lat_mean'].mean():.3f}")
        print(f"│  g_lon_mean        (frenadas):       {grupo['g_lon_mean'].mean():.3f}")
        print(f"│  cambios_marcha_km (tecnicidad):     {grupo['cambios_marcha_km'].mean():.2f}")
        print(f"│  longitud_km       (info, no modelo):{grupo['longitud_km'].mean():.3f}")
        print(f"└{'─'*50}")

    print()


# ─────────────────────────────────────────────────────────────────────────────
# PASO 3b — ELBOW METHOD (para elegir K óptimo)
# ─────────────────────────────────────────────────────────────────────────────

def plot_elbow(X: np.ndarray) -> None:
    """
    Prueba K de 2 a 10 y pinta la inercia.
    El 'codo' (donde la curva deja de bajar bruscamente) es el K óptimo.
    """
    ks      = range(2, 11)
    inertias = []

    for k in ks:
        km = KMeans(n_clusters=k, random_state=42, n_init=20)
        km.fit(X)
        inertias.append(km.inertia_)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(ks),
        y=inertias,
        mode="lines+markers",
        marker=dict(size=10, color="#E8002D"),
        line=dict(color="#E8002D", width=2),
        hovertemplate="K=%{x}<br>Inercia=%{y:.2f}<extra></extra>",
    ))
    fig.update_layout(
        title="Elbow Method — K óptimo",
        xaxis=dict(title="Número de clusters (K)", dtick=1, color="rgba(255,255,255,0.6)", gridcolor="rgba(255,255,255,0.08)"),
        yaxis=dict(title="Inercia (suma distancias²)",  color="rgba(255,255,255,0.6)", gridcolor="rgba(255,255,255,0.08)"),
        paper_bgcolor="#0E0E0F",
        plot_bgcolor="#111115",
        font=dict(family="JetBrains Mono, monospace", color="white"),
        width=700,
        height=420,
    )
    fig.show()
    print("✓ Elbow Method mostrado — elige el K donde la curva 'dobla'\n")


# ─────────────────────────────────────────────────────────────────────────────
# PASO 5 — VISUALIZAR CON PLOTLY
# ─────────────────────────────────────────────────────────────────────────────

CLUSTER_COLORS = ["#E8002D", "#00D2FF", "#39FF14", "#C77DFF"]
CLUSTER_SYMBOLS = ["circle", "diamond", "square", "star"]

def plot_clusters(df_circuit: pd.DataFrame) -> None:
    """
    Scatter interactivo: G lateral (curvas) vs Cambios de marcha (tecnicidad).
    Estos dos ejes resumen mejor la personalidad de un circuito.
    """
    fig = go.Figure()

    for cluster_id in sorted(df_circuit["cluster"].unique()):
        grupo = df_circuit[df_circuit["cluster"] == cluster_id]
        label = grupo["label"].iloc[0] if "label" in grupo.columns else f"Cluster {cluster_id}"

        fig.add_trace(go.Scatter(
            x=grupo["g_lat_mean"],
            y=grupo["cambios_marcha_km"],
            mode="markers+text",
            name=label,
            text=grupo["circuito"].str.replace(" Grand Prix", ""),
            textposition="top center",
            textfont=dict(size=9, color="rgba(255,255,255,0.7)"),
            marker=dict(
                size=12,
                color=CLUSTER_COLORS[cluster_id % len(CLUSTER_COLORS)],
                symbol=CLUSTER_SYMBOLS[cluster_id % len(CLUSTER_SYMBOLS)],
                line=dict(color="white", width=1),
            ),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "G Lateral: %{x:.3f} G<br>"
                "Cambios/km: %{y:.2f}<br>"
                f"Cluster: {label}"
                "<extra></extra>"
            ),
        ))

    fig.update_layout(
        title=dict(
            text="Clasificación de Circuitos F1 · K-means",
            font=dict(size=18, color="white"),
        ),
        xaxis=dict(
            title="G Lateral Media (curvas rápidas →)",
            gridcolor="rgba(255,255,255,0.08)",
            color="rgba(255,255,255,0.6)",
            zeroline=False,
        ),
        yaxis=dict(
            title="Cambios de Marcha / km (tecnicidad →)",
            gridcolor="rgba(255,255,255,0.08)",
            color="rgba(255,255,255,0.6)",
            zeroline=False,
        ),
        paper_bgcolor="#0E0E0F",
        plot_bgcolor="#111115",
        font=dict(family="JetBrains Mono, monospace", color="white"),
        legend=dict(
            bgcolor="rgba(0,0,0,0.4)",
            bordercolor="rgba(255,255,255,0.2)",
            borderwidth=1,
        ),
        width=1000,
        height=650,
    )

    fig.show()
    fig.write_html(os.path.join(os.path.dirname(__file__), "circuit_clusters.html"))
    print("✓ Gráfico guardado en research/circuit_clusters.html\n")


# ─────────────────────────────────────────────────────────────────────────────
# PASO 6 — GUARDAR
# ─────────────────────────────────────────────────────────────────────────────

def save_results(df_circuit: pd.DataFrame) -> None:
    df_circuit.to_csv(OUTPUT_CSV, index=False)
    print(f"✓ Resultado guardado en {OUTPUT_CSV}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    if not os.path.exists(INPUT_CSV):
        print(f"❌ No se encuentra {INPUT_CSV}")
        print("   Ejecuta primero: python circuit_data_scraper.py")
        sys.exit(1)

    print(f"\n🏎  TALOS · Circuit Classifier")
    print(f"   Input:  {INPUT_CSV}")
    print(f"   Output: {OUTPUT_CSV}")
    print(f"   K = {K} clusters\n")
    print("─" * 55)

    # ── Pasos 1-2 ─────────────────────────────────────────────
    df_circuit  = load_and_aggregate(INPUT_CSV)
    X, scaler   = normalize(df_circuit)

    # ── Paso 3b: Elbow method ─────────────────────────────────
    print("─" * 55)
    print("  ELBOW METHOD — cerrando esta ventana continúa el script")
    print("─" * 55)
    plot_elbow(X)

    # ── Paso 3: K-means con K elegido ─────────────────────────
    k_elegido = int(input(f"¿Qué K usamos? (default {K}): ").strip() or K)
    labels, kmeans   = run_kmeans(X, k=k_elegido)
    df_circuit["cluster"] = labels

    # ── Paso 4: imprimir para etiquetar ───────────────────────
    print_clusters(df_circuit)

    # ── Etiquetar manualmente (dinámico para cualquier K) ──────
    print("─" * 55)
    print("  ETIQUETADO — escribe un nombre para cada cluster")
    print("  (ejemplos: Urbano/Técnico, Mixto, Alta velocidad, Rápido…)")
    print("─" * 55)
    cluster_labels = {}
    for cid in sorted(df_circuit["cluster"].unique()):
        nombre = input(f"  Nombre para Cluster {cid}: ").strip()
        cluster_labels[cid] = nombre if nombre else f"Cluster {cid}"
    df_circuit["label"] = df_circuit["cluster"].map(cluster_labels)

    # ── Pasos 5-6 ─────────────────────────────────────────────
    plot_clusters(df_circuit)
    save_results(df_circuit)
