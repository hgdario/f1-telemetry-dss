from __future__ import annotations

import numpy as np
import streamlit as st
import plotly.graph_objects as go

from dinamica_vehicular.GGDiagram import _calculate_g_forces


# ─────────────────────────────────────────────────────────────────────────────
# DATOS DEL MODELO ENTRENADO (K=5, 25 circuitos con años 2022–2025)
# ─────────────────────────────────────────────────────────────────────────────

CLUSTER_LABELS: dict[int, str] = {
   -1: "Atípico",
    0: "Low Drag",
    1: "Stop-and-Go",
    2: "Aero Efficiency",
    3: "Balanced Chassis",
    4: "Monaco",
}

CLUSTER_COLORS: dict[int, str] = {
   -1: "#9497A8",   # gris neutro
    0: "#FF6B35",
    1: "#00D7B5",
    2: "#3B82F6",
    3: "#B138DD",
    4: "#E8002D",
}

CLUSTER_TAGLINES: dict[int, str] = {
   -1: "Layout fuera de la distribución del calendario moderno",
    0: "Velocidad punta con la mínima carga aerodinámica",
    1: "Acelerar, frenar fuerte, repetir",
    2: "Curvas rápidas con un downforce eficiente",
    3: "Coche completo y sin debilidades",
    4: "Atípico con clase propia",
}

CLUSTER_EXPLAIN: dict[int, str] = {
   -1: ("Las características de este circuito se desvían más de 2.5σ del centroide "
        "más cercano, por lo que no se puede asignar a ninguno de los "
        "5 clusters. Suele ocurrir con circuitos pre-2022, configuraciones alternativas "
        "(p.ej. el circuito de Bahrain exterior en 2020) o circuitos no recurrentes."),
    0: ("Circuitos donde manda la potencia del motor. Largas rectas y mínimas G. "
        "Los equipos ponen alerones con una carga mínima para ganar velocidad punta."),
    1: ("Perfil 'recta + frenada fuerte + curva lenta'. Frenadas violentas y muchos cambios "
        "de marcha. La gestión térmica de frenos y la salida de curva son críticas."),
    2: ("Curvas rápidas y sostenidas (no de frenón fuerte). Importa más cuánto "
        "downforce generas con poca pérdida aerodinámica que la frenada bruta."),
    3: ("Circuito mixto: curvas rápidas, lentas, cambios de elevación y frenadas variadas. "
        "Gana el coche más versátil, no el más especializado."),
    4: ("Mónaco es matemáticamente único: tiene el doble de cambios de marcha por km que "
        "casi cualquier otro trazado, en menos de 3,3 km. El clasificador con K-means lo aísla en su "
        "propio cluster"),
}

CLUSTER_EXAMPLES: dict[int, str] = {
   -1: "Bahrain Outer (2020) · Mugello (2020) · Imola (2020)",
    0: "Monza · Las Vegas",
    1: "Baku · Singapore · Canadá",
    2: "Spa · Silverstone · Suzuka",
    3: "Hungaroring · Zandvoort · COTA",
    4: "Mónaco (sólo)",
}

# Umbral de outlier: si el circuito está a más de 2.5 σ del centroide más cercano lo clasificamos como anomalía
OUTLIER_THRESHOLD_Z: float = 2.5

# Centroides en espacio crudo (medias por cluster)
CLUSTER_CENTROIDS: dict[int, np.ndarray] = {
    0: np.array([0.954, 0.630,  6.39]),
    1: np.array([0.954, 0.730,  8.85]),
    2: np.array([1.330, 0.586,  6.62]),
    3: np.array([1.338, 0.669,  7.97]),
    4: np.array([1.204, 0.751, 12.04]),
}

# Estadísticos del scaler original (StandardScaler ddof=0)
FEATURE_MEAN = np.array([1.1919, 0.6595, 7.8190])
FEATURE_STD  = np.array([0.2010, 0.0660, 1.4076])

# Lookup directo: features REALES de cada circuito + cluster_id
CIRCUIT_FEATURES: dict[str, tuple[float, float, float, int]] = {
    "Abu Dhabi Grand Prix":      (1.2592, 0.6663,  7.7975, 3),
    "Australian Grand Prix":     (1.1930, 0.5988,  7.0775, 2),
    "Austrian Grand Prix":       (1.0368, 0.7195,  8.0775, 1),
    "Azerbaijan Grand Prix":     (0.8395, 0.7575,  9.0250, 1),
    "Bahrain Grand Prix":        (0.9675, 0.7262,  8.7025, 1),
    "Belgian Grand Prix":        (1.2703, 0.5712,  5.8175, 2),
    "British Grand Prix":        (1.3440, 0.5445,  6.2575, 2),
    "Canadian Grand Prix":       (0.8698, 0.7372,  9.7900, 1),
    "Chinese Grand Prix":        (1.2160, 0.6520,  8.2550, 3),
    "Dutch Grand Prix":          (1.5585, 0.6370,  7.8300, 3),
    "Emilia Romagna Grand Prix": (1.0473, 0.6993,  8.3567, 1),
    "French Grand Prix":         (1.2340, 0.5810,  7.3400, 2),
    "Hungarian Grand Prix":      (1.3290, 0.7552,  8.8725, 3),
    "Italian Grand Prix":        (0.9130, 0.6110,  6.0700, 0),
    "Japanese Grand Prix":       (1.3958, 0.5660,  6.5600, 2),
    "Las Vegas Grand Prix":      (0.9947, 0.6480,  6.7133, 0),
    "Mexico City Grand Prix":    (0.9683, 0.6985,  7.7500, 1),
    "Miami Grand Prix":          (1.2692, 0.6280,  6.7800, 2),
    "Monaco Grand Prix":         (1.2035, 0.7510, 12.0425, 4),
    "Qatar Grand Prix":          (1.5033, 0.6080,  6.4800, 2),
    "Saudi Arabian Grand Prix":  (1.4318, 0.5912,  6.6325, 2),
    "Singapore Grand Prix":      (0.9490, 0.7712, 10.2050, 1),
    "Spanish Grand Prix":        (1.4110, 0.6568,  6.9825, 3),
    "São Paulo Grand Prix":      (1.2857, 0.6355,  7.8575, 3),
    "United States Grand Prix":  (1.3062, 0.6775,  8.2025, 3),
}


# ─────────────────────────────────────────────────────────────────────────────
# CÁLCULO LIVE (cuando el circuito no está en CIRCUIT_FEATURES)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_live_features(session) -> dict[str, float] | None:
    try:
        fastest = session.laps.pick_fastest()
        tel = fastest.get_telemetry().add_distance()
        if len(tel) < 100 or "Speed" not in tel.columns:
            return None
        tel = _calculate_g_forces(tel)
        if "g_lat" not in tel.columns:
            return None
        cambios_km = 0.0
        if "nGear" in tel.columns and "Distance" in tel.columns:
            gear = tel["nGear"].dropna()
            dist_km = tel["Distance"].max() / 1000
            if dist_km > 0:
                cambios_km = float(gear.diff().abs().gt(0).sum()) / dist_km
        return {
            "g_lat_mean":        float(tel["g_lat"].abs().mean()),
            "g_lon_mean":        float(tel["g_lon"].abs().mean()),
            "cambios_marcha_km": cambios_km,
        }
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# CLASIFICACIÓN
# ─────────────────────────────────────────────────────────────────────────────





def _resolve_circuit_key(event_name: str) -> str | None:
    """
    Busca el key de CIRCUIT_FEATURES que corresponde al nombre del evento. Retorna el nombre o None
    """
    # 1. Match exacto (camino feliz)
    if event_name in CIRCUIT_FEATURES:
        return event_name

    # 2. Match case-insensitive
    ev_low = event_name.lower()
    for key in CIRCUIT_FEATURES:
        if key.lower() == ev_low:
            return key

    # 3. Match sin 'Grand Prix' / 'Grand-Prix' → solo país/ubicación
    def _strip(s: str) -> str:
        return s.lower().replace("grand prix", "").replace("grand-prix", "").strip()

    ev_strip = _strip(event_name)
    for key in CIRCUIT_FEATURES:
        if _strip(key) == ev_strip:
            return key

    return None


@st.cache_data(show_spinner=False)
def classify_circuit(event_name: str, _session=None) -> dict | None:
    """Clasifica un circuito en uno de los 5 clusters."""
    # Camino preferido: lookup con features reales del CSV
    key = _resolve_circuit_key(event_name)
    if key is not None:
        g_lat, g_lon, cambios, cid = CIRCUIT_FEATURES[key]
        return {
            "cluster_id": cid,
            "label":      CLUSTER_LABELS[cid],
            "method":     "lookup",
            "features": {
                "g_lat_mean":        float(g_lat),
                "g_lon_mean":        float(g_lon),
                "cambios_marcha_km": float(cambios),
            },
        }

    # Fallback: features computadas en vivo
    if _session is None:
        return None
    feats = _compute_live_features(_session)
    if feats is None:
        return None
    feat_vec = np.array([feats["g_lat_mean"], feats["g_lon_mean"], feats["cambios_marcha_km"]])
    z = (feat_vec - FEATURE_MEAN) / FEATURE_STD
    distances = {
        cid: float(np.linalg.norm(z - (centroid - FEATURE_MEAN) / FEATURE_STD))
        for cid, centroid in CLUSTER_CENTROIDS.items()
    }
    cid_best = min(distances, key=distances.get)
    d_min    = distances[cid_best]

    # Outlier check: si el circuito está demasiado lejos del centroide más cercano,
    # marcarlo como Atípico en lugar de forzarlo a un cluster que no encaja
    if d_min > OUTLIER_THRESHOLD_Z:
        return {
            "cluster_id": -1,
            "label":      CLUSTER_LABELS[-1],
            "method":     "live",
            "features":   feats,
            "distance":   d_min,  # diagnóstico: distancia al centroide más cercano
        }

    return {
        "cluster_id": cid_best,
        "label":      CLUSTER_LABELS[cid_best],
        "method":     "live",
        "features":   feats,
    }


# ─────────────────────────────────────────────────────────────────────────────
# RADAR CHART · COMPARATIVO VS LOS 5 CENTROIDES
# ─────────────────────────────────────────────────────────────────────────────

def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _build_radar(features: dict[str, float], cid: int) -> go.Figure:
    """Radar 3 ejes con min-max scaling y margen 5% para legibilidad."""
    color = CLUSTER_COLORS[cid]
    pad = 0.05
    ranges = [
        (0.84 - pad*(1.56-0.84),   1.56 + pad*(1.56-0.84)),
        (0.54 - pad*(0.78-0.54),   0.78 + pad*(0.78-0.54)),
        (5.82 - pad*(12.04-5.82), 12.04 + pad*(12.04-5.82)),
    ]
    def s(v, axis):
        lo, hi = ranges[axis]
        return float(np.clip((v - lo) / (hi - lo), 0.0, 1.0))

    fig = go.Figure()

    # Otros centroides como referencia (gris)
    for ocid, centroid in CLUSTER_CENTROIDS.items():
        if ocid == cid:
            continue
        r_vals = [s(centroid[0], 0), s(centroid[1], 1), s(centroid[2], 2)]
        fig.add_trace(go.Scatterpolar(
            r=r_vals + [r_vals[0]],
            theta=["G Lateral", "G Longitudinal", "Cambios/km", "G Lateral"],
            line=dict(color="rgba(255,255,255,0.10)", width=1),
            name=CLUSTER_LABELS[ocid],
            hovertemplate=f"<b>{CLUSTER_LABELS[ocid]}</b><extra></extra>",
            showlegend=False,
        ))

    # Circuito actual
    g_lat_n   = s(features["g_lat_mean"], 0)
    g_lon_n   = s(features["g_lon_mean"], 1)
    cambios_n = s(features["cambios_marcha_km"], 2)
    fig.add_trace(go.Scatterpolar(
        r=[g_lat_n, g_lon_n, cambios_n, g_lat_n],
        theta=["G Lateral", "G Longitudinal", "Cambios/km", "G Lateral"],
        fill="toself",
        fillcolor=_hex_to_rgba(color, 0.20),
        line=dict(color=color, width=2.5),
        name="Este circuito",
        hovertemplate=(
            "<b>Este circuito</b><br>"
            f"G Lat: {features['g_lat_mean']:.2f} G<br>"
            f"G Lon: {features['g_lon_mean']:.2f} G<br>"
            f"Cambios: {features['cambios_marcha_km']:.1f}/km"
            "<extra></extra>"
        ),
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1.05], showticklabels=False, ticks="",
                            gridcolor="rgba(255,255,255,0.06)"),
            angularaxis=dict(gridcolor="rgba(255,255,255,0.08)",
                             tickfont=dict(color="#9497A8", size=11, family="JetBrains Mono")),
            bgcolor="rgba(0,0,0,0)",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=30, r=30, t=20, b=20),
        height=320,
        showlegend=False,
        font=dict(family="Inter, system-ui, sans-serif", color="#E8E9EF"),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# RENDER UI
# ─────────────────────────────────────────────────────────────────────────────

def _render_html(html: str) -> None:
    if hasattr(st, "html"):
        st.html(html)
    else:
        st.markdown(html, unsafe_allow_html=True)


def render_circuit_dna(session) -> None:
    """Renderiza el bloque ADN del circuito."""
    event_name = session.event["EventName"]
    cls = classify_circuit(event_name, _session=session)

    if cls is None:
        st.markdown('<div class="ss-empty">ADN no disponible — datos insuficientes.</div>',
                    unsafe_allow_html=True)
        return

    cid    = cls["cluster_id"]
    label  = cls["label"]
    feats  = cls["features"]
    color  = CLUSTER_COLORS[cid]
    method = cls["method"]
    is_outlier = (cid == -1)

    if is_outlier:
        badge_text = "ATÍPICO"
    elif method == "lookup":
        badge_text = "HISTÓRICO"
    else:
        badge_text = "EN VIVO"

    # z-scores (cuántas σ se desvía cada feature de la media global)
    z_lat = (feats["g_lat_mean"]        - FEATURE_MEAN[0]) / FEATURE_STD[0]
    z_lon = (feats["g_lon_mean"]        - FEATURE_MEAN[1]) / FEATURE_STD[1]
    z_cmb = (feats["cambios_marcha_km"] - FEATURE_MEAN[2]) / FEATURE_STD[2]

    def _z_class(z: float) -> str:
        if z >  0.5: return "hi"
        if z < -0.5: return "lo"
        return ""
    def _z_text(z: float) -> str:
        if abs(z) < 0.05: return "= media"
        sign = "+" if z > 0 else "−"
        return f"{sign}{abs(z):.1f} σ"

    # ── (1) Línea-cabecera ────────────────────────────────────────────
    _render_html(
        f'<div class="dna-line" style="--dna-color:{color};">'
        f'<span class="dna-line-name">{label}</span>'
        f'<span class="dna-line-tag">{CLUSTER_TAGLINES[cid]}</span>'
        f'<span class="dna-line-conf">{badge_text}</span>'
        f'</div>'
    )

    # ── (2) 3 stats inline + radar a la derecha ──────────────────────
    col_stats, col_radar = st.columns([1.4, 1])

    with col_stats:
        _render_html(
            f'<div class="dna-grid" style="--dna-color:{color};">'
            f'<div class="dna-stat">'
            f'<div class="dna-stat-name">G LATERAL</div>'
            f'<div class="dna-stat-value">{feats["g_lat_mean"]:.2f}<span class="dna-stat-unit">G</span></div>'
            f'<div class="dna-stat-z {_z_class(z_lat)}">{_z_text(z_lat)} vs calendario</div>'
            f'</div>'
            f'<div class="dna-stat">'
            f'<div class="dna-stat-name">G LONGITUDINAL</div>'
            f'<div class="dna-stat-value">{feats["g_lon_mean"]:.2f}<span class="dna-stat-unit">G</span></div>'
            f'<div class="dna-stat-z {_z_class(z_lon)}">{_z_text(z_lon)} vs calendario</div>'
            f'</div>'
            f'<div class="dna-stat">'
            f'<div class="dna-stat-name">CAMBIOS / KM</div>'
            f'<div class="dna-stat-value">{feats["cambios_marcha_km"]:.1f}<span class="dna-stat-unit">/km</span></div>'
            f'<div class="dna-stat-z {_z_class(z_cmb)}">{_z_text(z_cmb)} vs calendario</div>'
            f'</div>'
            f'</div>'
        )

        # ── (3) Insight breve ─────────────────────────────────────────
        if is_outlier:
            method_note = f"Distancia al centroide más cercano: {cls.get('distance', 0):.2f}σ (umbral: {OUTLIER_THRESHOLD_Z}σ)"
            examples_label = "Ejemplos de outliers conocidos"
        else:
            method_note = ("Media histórica 2022–2025"
                           if cls["method"] == "lookup"
                           else "Estimado en vivo desde la vuelta más rápida")
            examples_label = "Ejemplos del mismo grupo"

        _render_html(
            f'<div class="dna-insight">'
            f'<p>{CLUSTER_EXPLAIN[cid]}</p>'
            f'<p><b>{examples_label}:</b> {CLUSTER_EXAMPLES[cid]} '
            f'<span class="meta">· {method_note}</span></p>'
            f'</div>'
        )

    with col_radar:
        # Para outliers no tiene sentido el radar (no hay cluster al que pertenezca)
        if not is_outlier:
            st.plotly_chart(_build_radar(feats, cid),
                            use_container_width=True,
                            config={"displayModeBar": False})
