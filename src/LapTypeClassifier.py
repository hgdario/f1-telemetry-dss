"""
LapTypeClassifier.py — TALOS · Clasificador de Tipo de Vuelta
==============================================================

NO es un módulo de la app (no se registra en NAV). Es una utilidad que
TelemetryTrace.py llama para mostrar el "tipo de vuelta" y un insight
explicando por qué pertenece a su cluster.

Clusters generados con K-means (k=4) sobre 8 features de pilotaje,
residualizadas por GP (resta de la media del circuito) y normalizadas
con StandardScaler:
    p_full, p_part, p_coast, p_brk, decel_p10,
    shifts_per_km, throttle_avg, coast_avg_len

DIFERENCIA vs CircuitClassifier:
  - El CircuitClassifier describe el TRAZADO (físico, fijo por circuito).
  - Este describe la EJECUCIÓN de una vuelta (cambia vuelta a vuelta).
  - No hay "lookup" — todas las vueltas se computan en vivo desde telemetría.

API pública:
    classify_lap(tel, event_name) → dict
    render_lap_dna(tel, event_name) → renderiza la sección
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go


# ─────────────────────────────────────────────────────────────────────────────
# DATOS DEL MODELO ENTRENADO (K=4, ~16k vueltas · 2022–2025)
# ─────────────────────────────────────────────────────────────────────────────

CLUSTER_LABELS: dict[int, str] = {
   -1: "Atípica",
    0: "Ritmo de carrera",
    1: "Lift & Coast",
    2: "Tracción comprometida",
    3: "Vuelta de ataque",
}

CLUSTER_COLORS: dict[int, str] = {
   -1: "#9497A8",   # gris neutro
    0: "#00D7B5",   # cyan — ritmo normal
    1: "#FF6B35",   # naranja — gestión
    2: "#FFD600",   # amarillo — precaución/grip
    3: "#E8002D",   # rojo — ataque
}

CLUSTER_TAGLINES: dict[int, str] = {
   -1: "Perfil estadísticamente anómalo · in/out, SC o telemetría dañada",
    0: "Ritmo activo · frenadas reales sin gestionar",
    1: "Gestión activa · ahorro de combustible/frenos/neumático",
    2: "Modulación constante · tracción limitada o tráfico",
    3: "Al límite · clasificación o vuelta explosiva",
}

CLUSTER_EXPLAIN: dict[int, str] = {
   -1: ("Vuelta con perfil estadísticamente extremo. Habitual en in-lap, "
        "out-lap, Safety Car o telemetría con huecos. No se puede clasificar "
        "como tipo de vuelta normal."),
    0: ("Vuelta de carrera con frenadas activas. El piloto no se reserva ni "
        "gestiona en exceso: frena progresivamente, mantiene throttle controlado "
        "y respeta el desgaste sin renunciar al ritmo. Perfil típico del medio "
        "de un stint."),
    1: ("Vuelta de gestión activa. Mayor coasting (cero acelerador, cero freno) "
        "y segmentos de inercia más largos. El piloto está ahorrando combustible, "
        "cuidando frenos o protegiendo neumáticos. Habitual en finales de stint "
        "largo o cuando hay margen de tiempo sobre el rival inmediato."),
    2: ("Mucha modulación de throttle (medio gas constante) y throttle medio bajo. "
        "Indica grip limitado: neumático desgastado, frío, asfalto húmedo, o "
        "tráfico (dirty air). El piloto negocia el agarre vuelta a vuelta en "
        "lugar de ir a tope-cero-tope."),
    3: ("Vuelta al límite. Throttle a fondo máximo, mínimo coasting y frenadas "
        "más comprimidas. Captura sesiones de clasificación, vueltas calientes "
        "de FP3 y race laps explosivas (primera vuelta, undercut, defensa de "
        "posición con neumático nuevo)."),
}

CLUSTER_EXAMPLES: dict[int, str] = {
   -1: "In/out laps · vueltas tras Safety Car · telemetría dañada",
    0: "Mid-stint en cualquier carrera con ritmo controlado",
    1: "Stints largos en Singapur, Mónaco, Hungría · final de stint en cualquier GP",
    2: "Neumático muy desgastado · condiciones húmedas · seguir a otro coche",
    3: "Q3 · FP3 hot lap · primera vuelta de carrera · neumático fresco",
}

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES DEL MODELO (extraídas de research/data/lap_type_gp_means.csv
# y del print del classifier offline)
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_COLS = [
    "p_full", "p_part", "p_coast", "p_brk",
    "decel_p10", "shifts_per_km", "throttle_avg", "coast_avg_len",
]

# StandardScaler.scale_ aprendido sobre los residuales (FEATURE_MEAN es 0 por construcción)
FEATURE_STD = np.array([5.357, 5.1539, 1.8519, 2.2554, 4.7045, 0.9191, 4.1116, 0.1521])

# Centroides en espacio residualizado + escalado
CLUSTER_CENTROIDS: dict[int, np.ndarray] = {
    0: np.array([-0.3467, -0.1672, -0.0684,  0.6852, -0.6598,  0.5575, -0.0196,  0.0474]),
    1: np.array([-0.4120, -0.1653,  0.8627, -0.3443,  0.5326, -0.3675,  0.1599,  0.5131]),
    2: np.array([-0.2508,  0.7083, -0.7172,  0.4175,  0.2829,  0.0545, -0.9872, -0.6788]),
    3: np.array([ 1.1279, -0.4374, -0.5425, -0.5691, -0.1910, -0.1893,  0.7290, -0.2672]),
}

# Media de cada feature por GP (para residualizar al entrar una vuelta nueva)
GP_MEANS: dict[str, np.ndarray] = {
    "Abu Dhabi": np.array([63.7287,  3.0759,  2.5207, 13.8502, -43.3456,  7.7619, 82.6115, 0.3384]),
    "Bahrain":   np.array([61.3449,  2.9651,  2.3484, 17.0073, -51.4911,  8.7079, 83.3760, 0.3404]),
    "Belgian":   np.array([67.5404,  2.5153,  1.9722, 12.2794, -34.5554,  5.7987, 81.7996, 0.3760]),
    "British":   np.array([66.9872,  3.1071,  1.6804, 13.0066, -33.6363,  5.7529, 82.3082, 0.3777]),
    "Hungarian": np.array([50.9188,  7.4973,  3.1532, 21.1387, -47.3092,  8.8260, 76.0143, 0.3556]),
    "Italian":   np.array([74.7527,  2.1003,  1.9373, 11.5702, -38.7563,  5.9089, 88.0302, 0.3141]),
    "Monaco":    np.array([43.9236,  9.6008,  3.6560, 25.8088, -54.5189, 12.3401, 69.9489, 0.3737]),
    "Singapore": np.array([49.4650, 10.0989,  3.3473, 21.4938, -54.3997, 10.3875, 73.1972, 0.3113]),
}

# Fallback para GPs no entrenados: media global del dataset (descrita en lap_data_inspect)
GLOBAL_FALLBACK_MEAN = np.array([60.04, 5.05, 2.59, 17.09, -44.90, 8.23, 79.92, 0.35])

# Umbral de outlier: si la distancia a TODOS los centroides es > N σ en espacio
# escalado, no se asigna a ningún cluster → cluster_id = -1
OUTLIER_THRESHOLD_Z: float = 3.5


# ─────────────────────────────────────────────────────────────────────────────
# EXTRACCIÓN DE FEATURES (clon del scraper, adaptado a una sola vuelta)
# ─────────────────────────────────────────────────────────────────────────────

def _adaptive_throttle_threshold(tel: pd.DataFrame) -> float:
    """Umbral de derivada P70 para detectar medio-gas estable."""
    dt = tel["Time"].dt.total_seconds().diff()
    d_thr = tel["Throttle"].diff()
    derivative = (d_thr / dt).fillna(0).abs().dropna()
    if derivative.empty:
        return 5.0
    return max(float(np.percentile(derivative, 70)), 5.0)


def _coast_avg_length(coast_mask: pd.Series, tel: pd.DataFrame) -> float:
    """Duración media (s) de los segmentos consecutivos de coasting."""
    if not coast_mask.any():
        return 0.0
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


def _extract_lap_features(tel: pd.DataFrame) -> dict | None:
    """
    Extrae el vector de 8 features de pilotaje desde la telemetría de una vuelta.
    Retorna None si los datos no son suficientes.
    """
    if len(tel) < 100:
        return None
    needed = {"Throttle", "Brake", "Speed", "Time", "Distance", "nGear"}
    if not needed.issubset(tel.columns):
        return None

    # Brake → 0-100 (robusto a bool/float/0-1/0-100)
    brake = tel["Brake"].copy()
    if brake.dtype == bool or set(brake.dropna().unique()).issubset({0, 1, True, False}):
        brake = brake.astype(float) * 100
    elif brake.max() <= 1.0:
        brake = brake * 100

    dist_delta = tel["Distance"].diff().fillna(0).clip(lower=0)
    total_dist = float(dist_delta.sum())
    if total_dist <= 0:
        return None

    throttle = tel["Throttle"]
    dt = tel["Time"].dt.total_seconds().diff()
    d_thr = throttle.diff()
    derivative = (d_thr / dt).fillna(0)
    thr_threshold = _adaptive_throttle_threshold(tel)

    mask_full  = throttle >= 98
    mask_part  = (throttle > 0) & (throttle < 99) & (derivative.abs() < thr_threshold)
    mask_brk   = brake > 0
    mask_coast = (throttle == 0) & (brake == 0)

    p_full  = float(dist_delta[mask_full].sum()  / total_dist * 100)
    p_part  = float(dist_delta[mask_part].sum()  / total_dist * 100)
    p_brk   = float(dist_delta[mask_brk].sum()   / total_dist * 100)
    p_coast = float(dist_delta[mask_coast].sum() / total_dist * 100)

    decel = (tel["Speed"].diff() / dt).fillna(0).clip(upper=0)
    decel_p10 = float(np.percentile(decel.dropna(), 10)) if decel.notna().any() else 0.0

    gear = tel["nGear"].dropna()
    km   = total_dist / 1000
    shifts_per_km = float(gear.diff().abs().gt(0).sum() / km) if km > 0 else 0.0

    thr_pos = throttle[throttle > 0]
    throttle_avg = float(thr_pos.mean()) if not thr_pos.empty else 0.0

    coast_avg_len = _coast_avg_length(mask_coast, tel)

    return {
        "p_full": p_full,
        "p_part": p_part,
        "p_coast": p_coast,
        "p_brk": p_brk,
        "decel_p10": decel_p10,
        "shifts_per_km": shifts_per_km,
        "throttle_avg": throttle_avg,
        "coast_avg_len": coast_avg_len,
    }


# ─────────────────────────────────────────────────────────────────────────────
# RESOLUCIÓN DE GP (FastF1 da "Italian Grand Prix", nuestras keys son "Italian")
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_gp_key(event_name: str) -> str | None:
    """
    Devuelve el key de GP_MEANS correspondiente al event_name de FastF1.
    Prueba en orden: exacto → sin 'Grand Prix' → case-insensitive.
    """
    if event_name in GP_MEANS:
        return event_name

    def _strip(s: str) -> str:
        return s.lower().replace("grand prix", "").strip()

    ev_strip = _strip(event_name)
    for key in GP_MEANS:
        if _strip(key) == ev_strip:
            return key
    return None


# ─────────────────────────────────────────────────────────────────────────────
# CLASIFICACIÓN
# ─────────────────────────────────────────────────────────────────────────────

def classify_lap(tel: pd.DataFrame, event_name: str) -> dict | None:
    """
    Clasifica una vuelta en uno de los 4 clusters de tipo de vuelta.

    Retorna dict con: cluster_id, label, features, gp_used, distance, method.
    """
    feats = _extract_lap_features(tel)
    if feats is None:
        return None

    feat_vec = np.array([feats[c] for c in FEATURE_COLS])

    # Residualizar por GP (si conocido) o por media global (fallback)
    gp_key = _resolve_gp_key(event_name)
    if gp_key is not None:
        gp_mean = GP_MEANS[gp_key]
        method  = "live"
    else:
        gp_mean = GLOBAL_FALLBACK_MEAN
        method  = "live_fallback"

    residual = feat_vec - gp_mean
    z = residual / FEATURE_STD   # FEATURE_MEAN = 0 por construcción

    # Distancia a cada centroide
    distances = {
        cid: float(np.linalg.norm(z - centroid))
        for cid, centroid in CLUSTER_CENTROIDS.items()
    }
    cid_best = min(distances, key=distances.get)
    d_min    = distances[cid_best]

    # Outlier check
    if d_min > OUTLIER_THRESHOLD_Z:
        return {
            "cluster_id": -1,
            "label":      CLUSTER_LABELS[-1],
            "features":   feats,
            "gp_used":    gp_key or "fallback",
            "distance":   d_min,
            "method":     method,
        }

    return {
        "cluster_id": cid_best,
        "label":      CLUSTER_LABELS[cid_best],
        "features":   feats,
        "gp_used":    gp_key or "fallback",
        "distance":   d_min,
        "method":     method,
    }


# ─────────────────────────────────────────────────────────────────────────────
# RADAR CHART · COMPARATIVO VS LOS 4 CENTROIDES
# ─────────────────────────────────────────────────────────────────────────────

def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# Etiquetas legibles del radar (mismas 8 features pero con nombres bonitos)
_RADAR_LABELS = ["A fondo %", "Medio gas %", "Coasting %", "Freno %",
                 "Frenada bruta", "Cambios/km", "Throttle medio", "Coast largo"]


def _build_lap_radar(z_vec: np.ndarray, cid: int) -> go.Figure:
    """
    Radar 8 ejes mostrando la vuelta (residual+escalado) vs los 4 centroides.
    Los ejes están en z-score (σ desde la media del GP), simétrico en [-2.5, +2.5].
    """
    if cid < 0:
        cid = 0  # para evitar KeyError en color
    color = CLUSTER_COLORS[cid]

    # Escala simétrica → 0 = centro (media del GP); ±valor σ desde ahí
    Z_MAX = 2.5
    def to_radial(z): return float(np.clip((z + Z_MAX) / (2 * Z_MAX), 0.0, 1.0))

    fig = go.Figure()

    # Centroides de los demás clusters en gris translúcido
    for ocid, centroid in CLUSTER_CENTROIDS.items():
        if ocid == cid:
            continue
        r_vals = [to_radial(v) for v in centroid]
        fig.add_trace(go.Scatterpolar(
            r=r_vals + [r_vals[0]],
            theta=_RADAR_LABELS + [_RADAR_LABELS[0]],
            line=dict(color="rgba(255,255,255,0.10)", width=1),
            name=CLUSTER_LABELS[ocid],
            hovertemplate=f"<b>{CLUSTER_LABELS[ocid]}</b><extra></extra>",
            showlegend=False,
        ))

    # Vuelta actual
    r_lap = [to_radial(v) for v in z_vec]
    fig.add_trace(go.Scatterpolar(
        r=r_lap + [r_lap[0]],
        theta=_RADAR_LABELS + [_RADAR_LABELS[0]],
        fill="toself",
        fillcolor=_hex_to_rgba(color, 0.20),
        line=dict(color=color, width=2.5),
        name="Esta vuelta",
        hovertemplate="<b>Esta vuelta</b><br>%{theta}: %{customdata:+.2f}σ<extra></extra>",
        customdata=z_vec,
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1.0], showticklabels=False, ticks="",
                            gridcolor="rgba(255,255,255,0.06)"),
            angularaxis=dict(gridcolor="rgba(255,255,255,0.08)",
                             tickfont=dict(color="#9497A8", size=10, family="JetBrains Mono")),
            bgcolor="rgba(0,0,0,0)",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=30, r=30, t=20, b=20),
        height=340,
        showlegend=False,
        font=dict(family="Inter, system-ui, sans-serif", color="#E8E9EF"),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# RENDER · UI MINIMAL (cabecera + 4 stats + insight + radar)
# ─────────────────────────────────────────────────────────────────────────────

def _render_html(html: str) -> None:
    """Helper: usa st.html si está disponible (Streamlit ≥1.33), si no markdown."""
    if hasattr(st, "html"):
        st.html(html)
    else:
        st.markdown(html, unsafe_allow_html=True)


def render_lap_dna(tel: pd.DataFrame, event_name: str) -> None:
    """Renderiza el bloque de tipo de vuelta — espejo de render_circuit_dna."""
    cls = classify_lap(tel, event_name)
    if cls is None:
        _render_html('<div class="ss-empty">Tipo de vuelta no disponible — '
                     'telemetría insuficiente.</div>')
        return

    cid    = cls["cluster_id"]
    label  = cls["label"]
    feats  = cls["features"]
    color  = CLUSTER_COLORS[cid]
    method = cls["method"]
    is_outlier = (cid == -1)

    if is_outlier:
        badge_text = "ATÍPICA"
    elif method == "live_fallback":
        badge_text = "ESTIMADO"
    else:
        badge_text = "EN VIVO"

    # Vector en espacio z-score (residual / std, contra GP de la vuelta)
    feat_vec = np.array([feats[c] for c in FEATURE_COLS])
    gp_key   = _resolve_gp_key(event_name)
    gp_mean  = GP_MEANS[gp_key] if gp_key in GP_MEANS else GLOBAL_FALLBACK_MEAN
    z_vec    = (feat_vec - gp_mean) / FEATURE_STD

    z_full  = float(z_vec[0])
    z_coast = float(z_vec[2])
    z_brk   = float(z_vec[3])
    z_shft  = float(z_vec[5])

    def _z_class(z: float) -> str:
        if z >  0.5: return "hi"
        if z < -0.5: return "lo"
        return ""
    def _z_text(z: float) -> str:
        if abs(z) < 0.05: return "= media GP"
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

    # ── (2) 4 stats + radar a la derecha ─────────────────────────────
    col_stats, col_radar = st.columns([1.4, 1])

    with col_stats:
        _render_html(
            f'<div class="dna-grid" style="--dna-color:{color}; grid-template-columns: repeat(4, 1fr);">'
            f'<div class="dna-stat">'
            f'<div class="dna-stat-name">A FONDO</div>'
            f'<div class="dna-stat-value">{feats["p_full"]:.1f}<span class="dna-stat-unit">%</span></div>'
            f'<div class="dna-stat-z {_z_class(z_full)}">{_z_text(z_full)} vs GP</div>'
            f'</div>'
            f'<div class="dna-stat">'
            f'<div class="dna-stat-name">COASTING</div>'
            f'<div class="dna-stat-value">{feats["p_coast"]:.1f}<span class="dna-stat-unit">%</span></div>'
            f'<div class="dna-stat-z {_z_class(z_coast)}">{_z_text(z_coast)} vs GP</div>'
            f'</div>'
            f'<div class="dna-stat">'
            f'<div class="dna-stat-name">FRENO</div>'
            f'<div class="dna-stat-value">{feats["p_brk"]:.1f}<span class="dna-stat-unit">%</span></div>'
            f'<div class="dna-stat-z {_z_class(z_brk)}">{_z_text(z_brk)} vs GP</div>'
            f'</div>'
            f'<div class="dna-stat">'
            f'<div class="dna-stat-name">CAMBIOS/KM</div>'
            f'<div class="dna-stat-value">{feats["shifts_per_km"]:.1f}<span class="dna-stat-unit">/km</span></div>'
            f'<div class="dna-stat-z {_z_class(z_shft)}">{_z_text(z_shft)} vs GP</div>'
            f'</div>'
            f'</div>'
        )

        # ── (3) Insight breve ─────────────────────────────────────────
        if is_outlier:
            note = f"Distancia al centroide más cercano: {cls['distance']:.2f}σ"
        elif method == "live_fallback":
            note = f"GP '{event_name}' no en el dataset de entrenamiento · "\
                   f"residualización contra media global del calendario"
        else:
            note = f"Residualizado contra la media de {cls['gp_used']}"

        _render_html(
            f'<div class="dna-insight">'
            f'<p>{CLUSTER_EXPLAIN[cid]}</p>'
            f'<p><b>Cuándo aparece:</b> {CLUSTER_EXAMPLES[cid]} '
            f'<span class="meta">· {note}</span></p>'
            f'</div>'
        )

    with col_radar:
        if not is_outlier:
            st.plotly_chart(_build_lap_radar(z_vec, cid),
                            use_container_width=True,
                            config={"displayModeBar": False})
