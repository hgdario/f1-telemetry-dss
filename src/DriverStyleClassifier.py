"""
DriverStyleClassifier.py — TALOS · Clasificador de Firma de Pilotaje
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go


# ─────────────────────────────────────────────────────────────────────────────
# DATOS DEL MODELO ENTRENADO (K=4, ~16k vueltas · 2022–2025)
# Las firmas de piloto se cargan de research/data/driver_signatures.csv
# al hacer import. El CSV es el resultado de driver_style_classifier.py.
# ─────────────────────────────────────────────────────────────────────────────

CLUSTER_LABELS: dict[int, str] = {
   -2: "Híbrido",
   -1: "No clasificado",
    0: "Modulador",
    1: "Smooth",
    2: "Reactivo",
    3: "Late braker",
}

CLUSTER_COLORS: dict[int, str] = {
   -2: "#B8BACA",   # gris cálido — pertenencia ambigua
   -1: "#9497A8",   # gris frío — fuera del dataset
    0: "#E8002D",   # rojo — modulación agresiva
    1: "#00D2FF",   # cyan — suave/cerebral
    2: "#39FF14",   # verde — reactivo/instintivo
    3: "#C77DFF",   # violeta — frenador comprometido
}

CLUSTER_TAGLINES: dict[int, str] = {
   -2: "Perfil sin pertenencia clara · Se encuentra entre clusters",
   -1: "Piloto fuera del dataset entrenado · No podemos asignarlo a un cluster sin datos suficientes",
    0: "Mucho medio gas · Modulador de tracción",
    1: "Inputs eficientes · transiciones suaves",
    2: "Acelerador binario · Acelera a fondo o hace coasting",
    3: "Frenadas tardías · Frenada profunda acercandose al vértice",
}

CLUSTER_EXPLAIN: dict[int, str] = {
   -2: ("La firma de este piloto está a más de 2σ del centroide más cercano en "
        "el espacio de las 8 features. K-means lo asigna al cluster más próximo, "
        "pero su perfil no encaja limpiamente: combina rasgos de varios clusters. "
        "Habitualmente son pilotos en coches muy limitados, o que adoptaron un "
        "estilo conservador por estrategia de equipo."),
   -1: ("Este piloto no formaba parte del dataset de entrenamiento (≥100 "
        "vueltas en 2022-2025 sobre los 8 GPs muestreados). No tenemos firma "
        "suficiente para clasificarlo de forma fiable."),
    0: ("Modulación constante del throttle en curva. Estos pilotos negocian "
        "el agarre punto a punto en lugar de ir tope-o-cero. Reduce el riesgo "
        "de pérdida de tracción pero baja la velocidad media. Habitualmente "
        "asociado a coches con potencia difícil de domar o con balance "
        "delantero. Hay un fuerte sesgo de equipo: los 3 pilotos del cluster "
        "han pilotado para Ferrari."),
    1: ("Inputs progresivos y transiciones suaves. Frenadas menos agresivas "
        "y throttle medio bajo. Es el perfil 'cerebral' clásico: menos input "
        "agresivo pero más eficiente en gestión de neumático y consistencia "
        "vuelta a vuelta. Hamilton es el caso más claro de este tipo de piloto."),
    2: ("Conducción reactiva e instintiva. Mucho coasting y poco medio gas: "
        "el piloto va a fondo o levanta del todo, raramente modula. Throttle "
        "medio alto compensado por momentos de inercia. Familia Red Bull "
        "domina este cluster — puede haber sesgo de coche."),
    3: ("Frenador tardío. Compromiso máximo en el punto de frenada con "
        "gradiente de deceleración alto. p_full elevado porque va a fondo "
        "hasta que ya no se puede más. Estilo asociado a pilotos veteranos "
        "que entienden bien la mecánica del coche."),
}

# Umbral de "Híbrido": si la firma del piloto está a >N σ del centroide del
# cluster que K-means le asignó (en espacio escalado 8D), lo etiquetamos como
# perfil sin pertenencia clara en lugar de forzarlo al cluster más cercano.
# 2.0σ es generoso pero deja fuera los borderlines verdaderos como SAR.
OUTLIER_THRESHOLD_Z: float = 2.0

# Features que componen la firma (mismo orden que en lap_data_scraper)
FEATURE_COLS = [
    "p_full", "p_part", "p_coast", "p_brk",
    "decel_p10", "shifts_per_km", "throttle_avg", "coast_avg_len",
]

# Etiquetas legibles de cada feature para la UI
FEATURE_LABELS: dict[str, str] = {
    "p_full":        "A FONDO",
    "p_part":        "MEDIO GAS",
    "p_coast":       "COASTING",
    "p_brk":         "FRENO",
    "decel_p10":     "FRENADAS",
    "shifts_per_km": "CAMBIOS/KM",
    "throttle_avg":  "THROTTLE MEDIO",
    "coast_avg_len": "COAST LARGO",
}

# Interpretación direccional: cuando z>0, qué significa para el piloto
FEATURE_HI: dict[str, str] = {
    "p_full":        "más tiempo a fondo",
    "p_part":        "más modulación de pedal",
    "p_coast":       "más coasting",
    "p_brk":         "más tiempo frenando",
    "decel_p10":     "frenadas más suaves",   # decel_p10 menos negativo = menos brusco
    "shifts_per_km": "más cambios por km",
    "throttle_avg":  "throttle medio más alto",
    "coast_avg_len": "segmentos de coast largos",
}
FEATURE_LO: dict[str, str] = {
    "p_full":        "menos tiempo a fondo",
    "p_part":        "menos modulación",
    "p_coast":       "menos coasting",
    "p_brk":         "menos tiempo frenando",
    "decel_p10":     "frenadas más bruscas",
    "shifts_per_km": "menos cambios por km",
    "throttle_avg":  "throttle medio más bajo",
    "coast_avg_len": "segmentos de coast cortos",
}


# ─────────────────────────────────────────────────────────────────────────────
# CARGA DE FIRMAS Y CÁLCULO DE CENTROIDES (al import)
# ─────────────────────────────────────────────────────────────────────────────

_CSV_PATH = os.path.join(
    os.path.dirname(__file__), "data", "driver_signatures.csv"
)


def _load_driver_features() -> dict[str, tuple]:
    """
    Carga driver_signatures.csv → dict[driver_code, (f1, f2, ..., f8, cluster_id)].
    Los valores son los RESIDUALES CRUDOS (no escalados); el escalado se hace
    abajo al calcular los centroides.
    """
    if not os.path.exists(_CSV_PATH):
        return {}
    try:
        df = pd.read_csv(_CSV_PATH, index_col=0)
        result: dict[str, tuple] = {}
        for driver, row in df.iterrows():
            feats = tuple(float(row[c]) for c in FEATURE_COLS)
            cid = int(row["cluster"])
            result[str(driver)] = (*feats, cid)
        return result
    except Exception:
        return {}


def _compute_scaler_and_centroids(
    features: dict[str, tuple],
) -> tuple[np.ndarray, np.ndarray, dict[int, np.ndarray]]:
    """
    Reproduce el StandardScaler + medias-por-cluster del entrenamiento offline.

    """
    if not features:
        return np.zeros(8), np.ones(8), {}

    drivers = list(features.keys())
    raw_matrix = np.array([list(features[d][:-1]) for d in drivers])  # (N, 8)
    cluster_ids = np.array([features[d][-1] for d in drivers])         # (N,)

    feat_mean = raw_matrix.mean(axis=0)
    feat_std  = raw_matrix.std(axis=0) + 1e-12
    scaled    = (raw_matrix - feat_mean) / feat_std

    centroids: dict[int, np.ndarray] = {}
    for cid in np.unique(cluster_ids):
        mask = cluster_ids == cid
        centroids[int(cid)] = scaled[mask].mean(axis=0)

    return feat_mean, feat_std, centroids


DRIVER_FEATURES: dict[str, tuple] = _load_driver_features()
_FEATURE_MEAN, _FEATURE_STD, _CLUSTER_CENTROIDS = _compute_scaler_and_centroids(DRIVER_FEATURES)


# ─────────────────────────────────────────────────────────────────────────────
# CLASIFICACIÓN — lookup + umbral de "Híbrido" si está lejos del centroide
# ─────────────────────────────────────────────────────────────────────────────

def classify_driver(driver_code: str) -> dict | None:
    """
    Devuelve la firma + cluster de un piloto.
    · None si el piloto no está en el dataset.
    · cluster_id = -2 ("Híbrido") si su firma está a > OUTLIER_THRESHOLD_Z σ
      del centroide del cluster asignado (perfil sin pertenencia clara).
    """
    if not driver_code or driver_code not in DRIVER_FEATURES:
        return None

    *raw_feats, cid_assigned = DRIVER_FEATURES[driver_code]
    feats_dict = {col: float(raw_feats[i]) for i, col in enumerate(FEATURE_COLS)}

    # Si no se pudieron computar centroides (CSV vacío al import), devolver lookup directo
    if not _CLUSTER_CENTROIDS:
        return {
            "driver":     driver_code,
            "cluster_id": cid_assigned,
            "label":      CLUSTER_LABELS.get(cid_assigned, "?"),
            "features":   feats_dict,
        }

    # Escalar la firma del piloto al mismo espacio que el K-means
    raw = np.array(raw_feats)
    scaled = (raw - _FEATURE_MEAN) / _FEATURE_STD
    centroid = _CLUSTER_CENTROIDS[cid_assigned]
    distance = float(np.linalg.norm(scaled - centroid))

    # Umbral: si la firma está demasiado lejos de su centroide, no fingir
    if distance > OUTLIER_THRESHOLD_Z:
        return {
            "driver":          driver_code,
            "cluster_id":      -2,           # Híbrido
            "label":           CLUSTER_LABELS[-2],
            "features":        feats_dict,
            "distance":        distance,
            "nearest_cluster": cid_assigned,
            "nearest_label":   CLUSTER_LABELS[cid_assigned],
        }

    return {
        "driver":     driver_code,
        "cluster_id": cid_assigned,
        "label":      CLUSTER_LABELS[cid_assigned],
        "features":   feats_dict,
        "distance":   distance,
    }


def _peers_in_cluster(driver_code: str, cluster_id: int) -> list[str]:
    """Otros pilotos del mismo cluster, excluyendo al actual y los Híbridos."""
    if cluster_id < 0:
        return []
    return [
        d for d, vals in DRIVER_FEATURES.items()
        if int(vals[-1]) == cluster_id and d != driver_code
    ]


# ─────────────────────────────────────────────────────────────────────────────
# VIZ · DIVERGING BARS (sin radar, distintivo del resto del TelemetryTrace)
# ─────────────────────────────────────────────────────────────────────────────

def _build_signature_bars(features: dict[str, float], cluster_id: int) -> go.Figure:
    """
    Diverging horizontal bar chart de la firma del piloto.
    Cada barra = una feature, eje X = σ respecto a la media del calendario.
    """
    color = CLUSTER_COLORS.get(cluster_id, "#9497A8")

    # Ordenar por |valor| descendente → las features más distintivas arriba
    ordered = sorted(features.items(), key=lambda kv: abs(kv[1]), reverse=True)
    labels = [FEATURE_LABELS[c] for c, _ in ordered]
    values = [v for _, v in ordered]

    # Color de cada barra: cluster color si la |σ| es notable, gris si ~media
    bar_colors = [
        color if abs(v) > 0.4 else "rgba(150,150,150,0.35)"
        for v in values
    ]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=labels,
        x=values,
        orientation="h",
        marker=dict(color=bar_colors, line=dict(color="rgba(0,0,0,0)", width=0)),
        text=[f"{v:+.2f}σ" for v in values],
        textposition="outside",
        textfont=dict(family="JetBrains Mono, monospace", size=10,
                      color="rgba(255,255,255,0.85)"),
        hovertemplate="<b>%{y}</b><br>%{x:+.2f}σ vs media calendario<extra></extra>",
        showlegend=False,
    ))

    # Línea vertical en 0 (la media)
    fig.add_vline(x=0, line_width=1.5, line_color="rgba(255,255,255,0.35)")

    # Bandas suaves a ±1σ y ±2σ para referencia visual
    for x_band in [-2, -1, 1, 2]:
        fig.add_vline(x=x_band, line_width=0.5, line_color="rgba(255,255,255,0.08)",
                      line_dash="dot")

    # Rango simétrico, suficiente para captar los extremos de Ferrari (~+5σ)
    x_max = max(5.0, max(abs(v) for v in values) * 1.15)

    fig.update_layout(
        height=300,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=70, t=10, b=30),
        xaxis=dict(
            range=[-x_max, x_max],
            zeroline=False,
            gridcolor="rgba(255,255,255,0.04)",
            tickfont=dict(color="rgba(255,255,255,0.45)", size=9,
                          family="JetBrains Mono, monospace"),
            ticksuffix="σ",
        ),
        yaxis=dict(
            tickfont=dict(color="rgba(255,255,255,0.75)", size=10,
                          family="JetBrains Mono, monospace"),
            autorange="reversed",  # primera feature arriba (la más distintiva)
            showgrid=False,
        ),
        font=dict(family="Inter, system-ui, sans-serif"),
        bargap=0.45,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# RENDER · UI MINIMAL
# ─────────────────────────────────────────────────────────────────────────────

def _render_html(html: str) -> None:
    if hasattr(st, "html"):
        st.html(html)
    else:
        st.markdown(html, unsafe_allow_html=True)


def _interpret_residual(feat: str, value: float) -> str:
    """Devuelve una frase legible de qué significa el residual para esta feature."""
    if abs(value) < 0.3:
        return "≈ media calendario"
    if value > 0:
        return FEATURE_HI[feat]
    return FEATURE_LO[feat]


def render_driver_dna(driver_code: str) -> None:
    """Renderiza el bloque de firma de pilotaje — espejo de render_lap_dna."""
    cls = classify_driver(driver_code)
    if cls is None:
        _render_html(
            f'<div class="ss-empty"><b>Firma no disponible</b> — el piloto '
            f'<code>{driver_code or "?"}</code> no formaba parte del entrenamiento '
            f'(o el CSV de firmas no se ha generado todavía). Ejecuta '
            f'<code>research/driver_style_classifier.py</code> para entrenar.</div>'
        )
        return

    cid       = cls["cluster_id"]
    label     = cls["label"]
    feats     = cls["features"]
    color     = CLUSTER_COLORS[cid]
    is_hybrid = (cid == -2)

    # ── (1) Cabecera con label + tagline + código del piloto ─────────
    # Para híbridos, la tagline informa del cluster más cercano
    if is_hybrid:
        near_label = cls.get("nearest_label", "?")
        tagline = (
            f"Perfil sin pertenencia clara · más cercano a {near_label.upper()} "
            f"({cls.get('distance', 0):.2f}σ)"
        )
    else:
        tagline = CLUSTER_TAGLINES[cid]

    _render_html(
        f'<div class="dna-line" style="--dna-color:{color};">'
        f'<span class="dna-line-name">{label}</span>'
        f'<span class="dna-line-tag">{tagline}</span>'
        f'<span class="dna-line-conf">{driver_code}</span>'
        f'</div>'
    )

    # ── (2) Bars + Insight ───────────────────────────────────────────
    col_bars, col_insight = st.columns([1.4, 1])

    with col_bars:
        st.plotly_chart(
            _build_signature_bars(feats, cid),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    with col_insight:
        # Top 3 features más distintivas (las primeras del bar chart)
        top3 = sorted(feats.items(), key=lambda kv: abs(kv[1]), reverse=True)[:3]
        bullets = ""
        for feat_name, val in top3:
            interp = _interpret_residual(feat_name, val)
            bullets += (
                f'<li><b>{FEATURE_LABELS[feat_name]}</b> '
                f'<span class="meta">{val:+.2f}σ</span> — {interp}</li>'
            )

        # Peers: si es híbrido, mostramos los del cluster más cercano para contexto
        peer_cid = cls.get("nearest_cluster", cid) if is_hybrid else cid
        peers = _peers_in_cluster(driver_code, peer_cid)
        peers_label = "Cluster más cercano" if is_hybrid else "Mismo cluster"
        peers_str = ", ".join(peers) if peers else "—"

        _render_html(
            f'<div class="dna-insight">'
            f'<p>{CLUSTER_EXPLAIN[cid]}</p>'
            f'<p><b>Rasgos más distintivos:</b></p>'
            f'<ul style="margin: 0.2em 0 0.6em 1.2em; padding: 0; '
            f'color: #C8CAD8; font-family: Inter, system-ui, sans-serif;">'
            f'{bullets}</ul>'
            f'<p><b>{peers_label}:</b> {peers_str} '
            f'<span class="meta">· firma calculada sobre todas las quicklaps '
            f'2022-2025 en 8 GPs</span></p>'
            f'</div>'
        )
