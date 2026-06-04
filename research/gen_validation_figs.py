"""
Genera las figuras de validación de clasificadores (cap. 5.3) sobre los datos
reales de entrenamiento. Salida: PNG en docs/Borrador_Memoria/.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

DATA = os.path.join(os.path.dirname(__file__), "data")
OUT  = os.path.join(os.path.dirname(__file__), "..", "docs", "Borrador_Memoria")
FEATS = ["p_full", "p_part", "p_coast", "p_brk",
         "decel_p10", "shifts_per_km", "throttle_avg", "coast_avg_len"]
PALETTE = ["#E8002D", "#00A0D2", "#39A900", "#9B59B6", "#F39C12"]
plt.rcParams.update({"font.size": 11, "figure.dpi": 150})


def fig_scatter_circuitos():
    df = pd.read_csv(os.path.join(DATA, "circuit_clusters.csv"))
    X = df[["g_lat_mean", "g_lon_mean", "cambios_marcha_km"]].to_numpy()
    Xs = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2, random_state=42)
    P = pca.fit_transform(Xs)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for cid in sorted(df["cluster"].unique()):
        m = df["cluster"] == cid
        lab = df.loc[m, "label"].iloc[0]
        ax.scatter(P[m, 0], P[m, 1], s=90, color=PALETTE[cid % len(PALETTE)],
                   label=f"{lab}", edgecolor="white", linewidth=0.6, zorder=3)
    for i, name in enumerate(df["circuito"]):
        ax.annotate(name.replace(" Grand Prix", ""), (P[i, 0], P[i, 1]),
                    fontsize=7, alpha=0.7, xytext=(4, 3), textcoords="offset points")
    var = pca.explained_variance_ratio_
    ax.set_xlabel(f"Componente principal 1 ({var[0]*100:.0f}% var.)")
    ax.set_ylabel(f"Componente principal 2 ({var[1]*100:.0f}% var.)")
    ax.legend(loc="best", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    p = os.path.join(OUT, "cluster_circuitos.png")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    print("OK", p, "| var PCA:", np.round(var, 3), "acum:", round(var.sum(), 3))


def fig_scatter_pilotaje():
    df = pd.read_csv(os.path.join("..", "src", "data", "driver_signatures.csv"), index_col=0)
    X = df[FEATS].to_numpy()
    Xs = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2, random_state=42)
    P = pca.fit_transform(Xs)
    labels = {0: "Modulador", 1: "Smooth", 2: "Reactivo", 3: "Late braker"}
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for cid in sorted(df["cluster"].unique()):
        m = df["cluster"] == cid
        ax.scatter(P[m, 0], P[m, 1], s=80, color=PALETTE[cid % len(PALETTE)],
                   label=labels.get(cid, str(cid)), edgecolor="white", linewidth=0.6, zorder=3)
    for i, name in enumerate(df.index):
        ax.annotate(str(name), (P[i, 0], P[i, 1]), fontsize=7, alpha=0.7,
                    xytext=(4, 3), textcoords="offset points")
    var = pca.explained_variance_ratio_
    ax.set_xlabel(f"Componente principal 1 ({var[0]*100:.0f}% var.)")
    ax.set_ylabel(f"Componente principal 2 ({var[1]*100:.0f}% var.)")
    ax.legend(loc="best", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    p = os.path.join(OUT, "cluster_pilotaje.png")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    print("OK", p, "| var PCA:", np.round(var, 3), "acum:", round(var.sum(), 3))


def fig_elbow_laps():
    df = pd.read_csv(os.path.join(DATA, "lap_features_raw.csv"))
    df = df.dropna(subset=FEATS)
    # Residualizar por GP (restar la media de cada Gran Premio)
    res = df.copy()
    res[FEATS] = df.groupby("gp")[FEATS].transform(lambda s: s - s.mean())
    Xs = StandardScaler().fit_transform(res[FEATS].to_numpy())
    ks = range(2, 9)
    inertias, sils = [], []
    for k in ks:
        km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(Xs)
        inertias.append(km.inertia_)
        sils.append(silhouette_score(Xs, km.labels_, sample_size=3000, random_state=42))
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(list(ks), inertias, "o-", color="#2C3E50", label="Inercia (codo)")
    ax1.set_xlabel("Número de clústeres $k$")
    ax1.set_ylabel("Inercia", color="#2C3E50")
    ax1.axvline(4, color="#E8002D", ls="--", alpha=0.7)
    ax2 = ax1.twinx()
    ax2.plot(list(ks), sils, "s-", color="#E8002D", label="Silhouette")
    ax2.set_ylabel("Silhouette", color="#E8002D")
    ax1.grid(True, alpha=0.2)
    fig.tight_layout()
    p = os.path.join(OUT, "elbow_laps.png")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    print("OK", p, "| k=4 inercia:", round(inertias[2], 0), "sil:", round(sils[2], 4))


def _elbow_plot(Xs, k_opt, fname):
    ks = range(2, 9)
    inertias, sils = [], []
    for k in ks:
        km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(Xs)
        inertias.append(km.inertia_)
        sils.append(silhouette_score(Xs, km.labels_))
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(list(ks), inertias, "o-", color="#2C3E50")
    ax1.set_xlabel("Número de clústeres $k$")
    ax1.set_ylabel("Inercia", color="#2C3E50")
    ax1.axvline(k_opt, color="#E8002D", ls="--", alpha=0.7)
    ax2 = ax1.twinx()
    ax2.plot(list(ks), sils, "s-", color="#E8002D")
    ax2.set_ylabel("Silhouette", color="#E8002D")
    ax1.grid(True, alpha=0.2)
    fig.tight_layout()
    p = os.path.join(OUT, fname)
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    print("OK", p, "| sil k_opt:", round(sils[k_opt - 2], 4))


def fig_elbow_circuitos():
    df = pd.read_csv(os.path.join(DATA, "circuit_clusters.csv"))
    Xs = StandardScaler().fit_transform(
        df[["g_lat_mean", "g_lon_mean", "cambios_marcha_km"]].to_numpy())
    _elbow_plot(Xs, 5, "elbow_circuitos.png")


def fig_elbow_pilotaje():
    df = pd.read_csv(os.path.join("..", "src", "data", "driver_signatures.csv"), index_col=0)
    Xs = StandardScaler().fit_transform(df[FEATS].to_numpy())
    P = PCA(n_components=2, random_state=42).fit_transform(Xs)
    _elbow_plot(P, 4, "elbow_pilotaje.png")


if __name__ == "__main__":
    fig_scatter_circuitos()
    fig_scatter_pilotaje()
    fig_elbow_laps()
    fig_elbow_circuitos()
    fig_elbow_pilotaje()
    print("DONE")
