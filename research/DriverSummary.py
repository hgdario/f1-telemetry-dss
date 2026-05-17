"""
DriverSummary — Clustering de Estilos de Conducción
====================================================
"""

from __future__ import annotations
import ui_assets

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ─── PALETA F1 ────────────────────────────────────────────────────────────────
F1_RED      = "#E8002D"
PIT_GREEN   = "#00D2BE"
WARN_YELLOW = "#FFF500"
INFO_BLUE   = "#5A9FD4"
GOLD        = "#FFD700"
WHITE       = "#FFFFFF"
DIM         = "#6A6A88"
PANEL       = "#15151E"

CLUSTER_PALETTE = [F1_RED, PIT_GREEN, INFO_BLUE, WARN_YELLOW, GOLD]

# Catálogo legible de features
FEATURE_LABELS = {
    "speed_max":        "VEL. MÁX",
    "speed_avg":        "VEL. MEDIA",
    "speed_std":        "VAR. VEL.",
    "throttle_full_pct":"% A FONDO",
    "throttle_avg":     "ACEL. MEDIA",
    "brake_pct":        "% FRENO",
    "lap_consistency":  "CONSISTENCIA",
    "gear_change_rate": "CAMBIOS MARCHA",
    "drs_usage_pct":    "% DRS",
    "coast_pct":        "% COASTING",
}


# ═══════════════════════════════════════════════════════════════════════════════
#  EXTRACCIÓN DE FEATURES POR PILOTO
# ═══════════════════════════════════════════════════════════════════════════════
def _extract_driver_features(session) -> pd.DataFrame:
    """Extrae 10 features de comportamiento por piloto sobre su mejor vuelta."""
    rows: list[dict] = []

    for drv_id in session.drivers:
        try:
            drv_laps = session.laps.pick_drivers(drv_id).pick_accurate()
            if drv_laps.empty:
                continue
            best = drv_laps.pick_fastest()
            tel  = best.get_telemetry()
            if tel.empty or len(tel) < 50:
                continue

            # Tiempos de vuelta del piloto — para la consistencia
            lap_times_s = (
                drv_laps['LapTime']
                .dropna()
                .dt.total_seconds()
                .values
            )

            # ─── Features ────────────────────────────────────────────────
            speed = tel['Speed'].astype(float).values
            throt = tel['Throttle'].astype(float).values
            brake = tel['Brake'].astype(bool).values
            gear  = tel['nGear'].astype(int).values if 'nGear' in tel else None
            drs   = tel['DRS'].astype(int).values   if 'DRS'   in tel else None

            # Velocidad
            speed_max = float(np.nanmax(speed))
            speed_avg = float(np.nanmean(speed))
            speed_std = float(np.nanstd(speed))

            # Throttle / freno
            throttle_full_pct = float(np.mean(throt >= 98.0) * 100)  # % a fondo
            throttle_avg      = float(np.nanmean(throt))
            brake_pct         = float(np.mean(brake) * 100)

            # Coasting (sin gas y sin freno)
            coasting = (throt < 5.0) & (~brake)
            coast_pct = float(np.mean(coasting) * 100)

            # Cambios de marcha por segundo
            if gear is not None and len(gear) > 1:
                gear_changes = int(np.sum(np.diff(gear) != 0))
                # Duración aprox. en segundos (Distance / Speed promedio)
                dist = tel['Distance'].iloc[-1] - tel['Distance'].iloc[0] if 'Distance' in tel else None
                if dist and speed_avg > 0:
                    duration_s = float(dist) / (speed_avg / 3.6)
                    gear_change_rate = gear_changes / duration_s if duration_s > 0 else 0.0
                else:
                    gear_change_rate = 0.0
            else:
                gear_change_rate = 0.0

            # DRS — valores >9 normalmente significan abierto
            if drs is not None:
                drs_usage_pct = float(np.mean(drs >= 10) * 100)
            else:
                drs_usage_pct = 0.0

            # Consistencia de vueltas (más bajo = más consistente)
            #   se invierte para que valores ALTOS indiquen ALTA consistencia
            if len(lap_times_s) >= 3:
                lap_std = float(np.std(lap_times_s))
                lap_consistency = float(np.exp(-lap_std))   # 0..1
            else:
                lap_consistency = float('nan')

            # Info contextual
            info = session.get_driver(drv_id)
            rows.append({
                "driver":       drv_id,
                "abbr":         info.get("Abbreviation", str(drv_id)),
                "full_name":    info.get("FullName", str(drv_id)),
                "team":         info.get("TeamName", "—"),
                "team_color":   "#" + str(info.get("TeamColor", "888888")).lstrip("#"),
                "position":     info.get("Position", None),
                # Features
                "speed_max":         speed_max,
                "speed_avg":         speed_avg,
                "speed_std":         speed_std,
                "throttle_full_pct": throttle_full_pct,
                "throttle_avg":      throttle_avg,
                "brake_pct":         brake_pct,
                "coast_pct":         coast_pct,
                "gear_change_rate":  gear_change_rate,
                "drs_usage_pct":     drs_usage_pct,
                "lap_consistency":   lap_consistency,
            })
        except Exception:
            continue

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
#  K-MEANS  (determinista)
# ═══════════════════════════════════════════════════════════════════════════════
def _run_kmeans(df: pd.DataFrame, k: int, feature_cols: list[str]):
    """Devuelve (labels, scaled_X, centroids_unscaled, silhouette, scaler, pca)."""
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score
    from sklearn.decomposition import PCA

    X = df[feature_cols].fillna(df[feature_cols].mean()).values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(Xs)

    sil = silhouette_score(Xs, labels) if k > 1 and len(set(labels)) > 1 else float('nan')

    pca = PCA(n_components=2, random_state=42)
    Xp = pca.fit_transform(Xs)

    # Centroides en el espacio original
    centroids_unscaled = scaler.inverse_transform(km.cluster_centers_)

    return labels, Xp, centroids_unscaled, sil, scaler, km


# ═══════════════════════════════════════════════════════════════════════════════
#  INTERPRETACIÓN AUTOMÁTICA DE CLUSTERS
# ═══════════════════════════════════════════════════════════════════════════════
def _label_clusters(centroids: np.ndarray, feature_cols: list[str]) -> list[str]:
    """Asigna una etiqueta legible a cada cluster según sus centroides relativos."""
    labels: list[str] = []
    z = (centroids - centroids.mean(axis=0)) / (centroids.std(axis=0) + 1e-9)

    for i in range(centroids.shape[0]):
        zi = dict(zip(feature_cols, z[i]))

        # Heurísticas — cada una mira sólo el signo del z-score
        speed_idx     = zi.get("speed_avg", 0) + zi.get("speed_max", 0)
        aggression    = zi.get("brake_pct", 0) + zi.get("throttle_full_pct", 0)
        consistency   = zi.get("lap_consistency", 0)
        coasting      = zi.get("coast_pct", 0)

        if consistency > 0.6 and aggression < 0.2:
            tag = "CONSISTENTE"
        elif aggression > 0.6 and speed_idx > 0:
            tag = "AGRESIVO"
        elif speed_idx > 0.6 and consistency > 0:
            tag = "VELOZ-EQUILIBRADO"
        elif coasting > 0.4:
            tag = "CONSERVADOR"
        elif aggression > 0.3:
            tag = "ATACANTE"
        else:
            tag = "INTERMEDIO"
        labels.append(tag)

    # Evita etiquetas duplicadas
    seen: dict[str, int] = {}
    out: list[str] = []
    for t in labels:
        if t in seen:
            seen[t] += 1
            out.append(f"{t}-{seen[t]}")
        else:
            seen[t] = 1
            out.append(t)
    return out


# ═══════════════════════════════════════════════════════════════════════════════
#  VALIDACIÓN — correlación cluster vs posición final
# ═══════════════════════════════════════════════════════════════════════════════
def _validate_with_results(df: pd.DataFrame, labels) -> dict:
    """Calcula correlación entre cluster asignado y posición final."""
    out = {"available": False, "spearman": None, "n": 0, "note": ""}
    try:
        df = df.copy()
        df["cluster"] = labels
        df_with_pos = df[df["position"].notna()].copy()
        if len(df_with_pos) < 4:
            out["note"] = "Pocos pilotos con posición final"
            return out
        df_with_pos["position"] = df_with_pos["position"].astype(float)

        # Spearman entre cluster_id y posición — sólo sirve si los clusters
        # están ordenados (lo hacemos por velocidad media descendente)
        cl_speed = df_with_pos.groupby("cluster")["speed_avg"].mean().sort_values(ascending=False)
        rank_map = {c: i for i, c in enumerate(cl_speed.index)}
        df_with_pos["cluster_rank"] = df_with_pos["cluster"].map(rank_map)

        rho = df_with_pos[["cluster_rank", "position"]].corr(method="spearman").iloc[0, 1]
        out["available"] = True
        out["spearman"]  = float(rho)
        out["n"]         = len(df_with_pos)
        return out
    except Exception as e:
        out["note"] = str(e)
        return out


# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURAS
# ═══════════════════════════════════════════════════════════════════════════════
def _fig_pca_scatter(df: pd.DataFrame, Xp, labels, cluster_names) -> go.Figure:
    fig = go.Figure()
    for cl in sorted(set(labels)):
        mask = labels == cl
        color = CLUSTER_PALETTE[cl % len(CLUSTER_PALETTE)]
        sub = df.loc[mask]
        fig.add_trace(go.Scatter(
            x=Xp[mask, 0], y=Xp[mask, 1],
            mode="markers+text",
            marker=dict(size=22, color=color, line=dict(color="#000", width=1.5),
                        symbol="circle"),
            text=sub["abbr"].values,
            textposition="middle center",
            textfont=dict(color="white", size=10, family="Share Tech Mono"),
            name=f"{cluster_names[cl]}  ({mask.sum()})",
            customdata=sub[["full_name", "team"]].values,
            hovertemplate=(
                "<b>%{text}</b><br>"
                "%{customdata[0]} · %{customdata[1]}"
                "<extra></extra>"
            ),
        ))
    fig.update_layout(
        height=480, paper_bgcolor=PANEL, plot_bgcolor=PANEL,
        font=dict(color="white", family="Titillium Web"),
        xaxis=dict(title="PC1", gridcolor="#2A2A3A", zerolinecolor="#2A2A3A"),
        yaxis=dict(title="PC2", gridcolor="#2A2A3A", zerolinecolor="#2A2A3A"),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0,
                    font=dict(size=11, family="Share Tech Mono")),
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def _fig_radar_clusters(df: pd.DataFrame, labels, feature_cols, cluster_names) -> go.Figure:
    """Radar normalizado (0-100) con un trazo por cluster."""
    fig = go.Figure()
    df = df.copy()
    df["cluster"] = labels

    # Normalizar features a 0-100 para visualización
    nf = df[feature_cols].copy()
    nf = (nf - nf.min()) / (nf.max() - nf.min() + 1e-9) * 100

    cats = [FEATURE_LABELS.get(c, c) for c in feature_cols]
    cats_loop = cats + [cats[0]]

    for cl in sorted(set(labels)):
        means = nf.loc[df["cluster"] == cl].mean().values.tolist()
        means_loop = means + [means[0]]
        color = CLUSTER_PALETTE[cl % len(CLUSTER_PALETTE)]
        fig.add_trace(go.Scatterpolar(
            r=means_loop, theta=cats_loop,
            fill='toself', name=f"{cluster_names[cl]}",
            line=dict(color=color, width=2),
            fillcolor=ui_assets.hex_to_rgba(color, 0.18),
        ))
    fig.update_layout(
        height=480, paper_bgcolor=PANEL, plot_bgcolor=PANEL,
        font=dict(color="white", family="Titillium Web"),
        polar=dict(
            bgcolor=PANEL,
            radialaxis=dict(visible=True, range=[0, 100], gridcolor="#2A2A3A",
                            tickfont=dict(color=DIM, size=9)),
            angularaxis=dict(gridcolor="#2A2A3A",
                             tickfont=dict(color="white", size=10, family="Share Tech Mono")),
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0,
                    font=dict(size=11, family="Share Tech Mono")),
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig


def _fig_silhouette_curve(df: pd.DataFrame, feature_cols) -> go.Figure:
    """Curva de Silhouette para K=2..6 — ayuda a justificar la K elegida."""
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score

    X = df[feature_cols].fillna(df[feature_cols].mean()).values
    Xs = StandardScaler().fit_transform(X)
    ks = list(range(2, min(7, len(df))))
    scores = []
    for k in ks:
        km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(Xs)
        try:
            scores.append(silhouette_score(Xs, km.labels_))
        except Exception:
            scores.append(np.nan)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ks, y=scores, mode="lines+markers",
        line=dict(color=F1_RED, width=2),
        marker=dict(size=10, color=F1_RED, line=dict(color=WHITE, width=1)),
        hovertemplate="K=%{x}<br>Silhouette=%{y:.3f}<extra></extra>",
    ))
    if scores:
        best_k = ks[int(np.nanargmax(scores))]
        fig.add_vline(x=best_k, line_dash="dash", line_color=PIT_GREEN,
                      annotation_text=f"ÓPTIMO · K={best_k}",
                      annotation_font_color=PIT_GREEN)
    fig.update_layout(
        height=260, paper_bgcolor=PANEL, plot_bgcolor=PANEL,
        font=dict(color="white", family="Titillium Web"),
        xaxis=dict(title="K (nº clusters)", gridcolor="#2A2A3A", dtick=1),
        yaxis=dict(title="Silhouette", gridcolor="#2A2A3A"),
        margin=dict(l=20, r=20, t=20, b=40),
        showlegend=False,
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════
def render_driver_summary(session) -> None:
    st.header("Resumen del Piloto · Clustering de Estilos")
    st.caption("K-MEANS DETERMINISTA  ·  RANDOM_STATE=42  ·  VALIDACIÓN: SILHOUETTE + SPEARMAN")

    # ── 1. Extracción de features ────────────────────────────────────────────
    with st.spinner("Extrayendo telemetría de cada piloto..."):
        df = _extract_driver_features(session)

    if df.empty or len(df) < 4:
        st.error("No hay suficientes pilotos con telemetría válida para hacer clustering.")
        return

    feature_cols = [
        "speed_avg", "speed_max", "speed_std",
        "throttle_full_pct", "throttle_avg",
        "brake_pct", "coast_pct",
        "gear_change_rate", "drs_usage_pct",
        "lap_consistency",
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]

    # ── 2. Selector de K ─────────────────────────────────────────────────────
    cfg_col1, cfg_col2 = st.columns([1, 3])
    with cfg_col1:
        k = st.slider("Nº de clusters (K)", min_value=2, max_value=min(5, len(df)-1),
                      value=3, step=1)
    with cfg_col2:
        st.markdown(
            f"<div style='padding-top:24px; font-family:\"Share Tech Mono\",monospace; "
            f"font-size:11px; letter-spacing:2px; color:{DIM}; text-transform:uppercase;'>"
            f"PILOTOS ANALIZADOS · {len(df)}  ·  FEATURES · {len(feature_cols)}  ·  "
            f"ALGORITMO · K-MEANS (LLOYD) + SCALER STD"
            f"</div>",
            unsafe_allow_html=True,
        )

    # ── 3. K-Means ───────────────────────────────────────────────────────────
    labels, Xp, centroids_unscaled, sil, scaler, km = _run_kmeans(df, k, feature_cols)
    cluster_names = _label_clusters(centroids_unscaled, feature_cols)
    validation = _validate_with_results(df, labels)

    # ── 4. Métricas de validación ────────────────────────────────────────────
    st.divider()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Silhouette Score", f"{sil:.3f}",
              delta="bueno (>0.4)" if sil > 0.4 else "débil (<0.4)",
              delta_color="normal" if sil > 0.4 else "inverse")
    m2.metric("Pilotos", f"{len(df)}")
    m3.metric("Features", f"{len(feature_cols)}")
    if validation["available"]:
        rho = validation["spearman"]
        m4.metric("Spearman ρ (cluster vs pos.)", f"{rho:+.2f}",
                  delta=f"n = {validation['n']}", delta_color="off")
    else:
        m4.metric("Validación deportiva", "N/A",
                  delta=validation.get("note") or "Sin posiciones finales",
                  delta_color="off")

    # ── 5. Visualizaciones ──────────────────────────────────────────────────
    st.divider()
    tab_pca, tab_radar, tab_curve, tab_table = st.tabs(
        ["◉ PCA 2D", "◬ Perfil por Cluster", "◷ Curva Silhouette", "▤ Tabla"]
    )

    with tab_pca:
        st.markdown(
            "<div style='font-family:Titillium Web; color:#aaa; font-size:12px;'>"
            "Proyección bidimensional de los 10 features mediante PCA. Cada punto es "
            "un piloto; el color indica su cluster asignado por K-Means."
            "</div>", unsafe_allow_html=True,
        )
        st.plotly_chart(_fig_pca_scatter(df, Xp, labels, cluster_names),
                        use_container_width=True, config={"displayModeBar": False})

    with tab_radar:
        st.markdown(
            "<div style='font-family:Titillium Web; color:#aaa; font-size:12px;'>"
            "Perfil promedio (centroide) de cada cluster, normalizado al rango "
            "[0, 100] sobre todos los pilotos analizados."
            "</div>", unsafe_allow_html=True,
        )
        st.plotly_chart(_fig_radar_clusters(df, labels, feature_cols, cluster_names),
                        use_container_width=True, config={"displayModeBar": False})

    with tab_curve:
        st.markdown(
            "<div style='font-family:Titillium Web; color:#aaa; font-size:12px;'>"
            "Score de Silhouette para diferentes valores de K. La línea verde marca "
            "el K que maximiza la separación entre clusters."
            "</div>", unsafe_allow_html=True,
        )
        st.plotly_chart(_fig_silhouette_curve(df, feature_cols),
                        use_container_width=True, config={"displayModeBar": False})

    with tab_table:
        # Tabla con piloto, equipo, cluster y top feature
        df_show = df.copy()
        df_show["cluster_id"]   = labels
        df_show["cluster_name"] = [cluster_names[l] for l in labels]

        # Determinar feature dominante de cada piloto (z-score más alto)
        Xn = (df_show[feature_cols] - df_show[feature_cols].mean()) / (df_show[feature_cols].std() + 1e-9)
        df_show["feature_destacada"] = Xn.idxmax(axis=1).map(lambda c: FEATURE_LABELS.get(c, c))

        cols_show = ["abbr", "full_name", "team", "cluster_id", "cluster_name",
                     "feature_destacada", "speed_max", "speed_avg",
                     "throttle_full_pct", "brake_pct", "lap_consistency"]
        cols_show = [c for c in cols_show if c in df_show.columns]
        df_show = df_show[cols_show].rename(columns={
            "abbr": "PIL", "full_name": "PILOTO", "team": "EQUIPO",
            "cluster_id": "CL", "cluster_name": "ESTILO",
            "feature_destacada": "DESTACA EN",
            "speed_max": "VEL.MAX", "speed_avg": "VEL.AVG",
            "throttle_full_pct": "%THROTTLE", "brake_pct": "%FRENO",
            "lap_consistency": "CONSIST.",
        })
        st.dataframe(df_show, use_container_width=True, hide_index=True)

    # ── 6. Insights deportivos ──────────────────────────────────────────────
    st.divider()
    st.subheader("Insights deportivos")

    cluster_summary = []
    for cl in sorted(set(labels)):
        mask = labels == cl
        members = df.loc[mask, "abbr"].tolist()
        teams   = df.loc[mask, "team"].tolist()
        cluster_summary.append({
            "cluster": cluster_names[cl],
            "n":       int(mask.sum()),
            "pilotos": ", ".join(members),
            "equipos": ", ".join(sorted(set(teams))),
        })

    for cs in cluster_summary:
        st.markdown(f"""
        <div style='background:#12121A; border-left:3px solid {F1_RED};
                    padding:12px 16px; margin-bottom:8px; border-radius:2px;'>
            <div style='font-family:"Share Tech Mono",monospace; font-size:11px;
                        letter-spacing:3px; color:{F1_RED}; text-transform:uppercase;'>
                {cs['cluster']}  ·  {cs['n']} PILOTOS
            </div>
            <div style='font-family:"Titillium Web",sans-serif; color:white;
                        font-size:14px; margin-top:6px;'>
                <b>Pilotos:</b> {cs['pilotos']}
            </div>
            <div style='font-family:"Titillium Web",sans-serif; color:#aaa;
                        font-size:12px; margin-top:2px;'>
                <b>Equipos representados:</b> {cs['equipos']}
            </div>
        </div>
        """, unsafe_allow_html=True)

    if validation["available"]:
        rho = validation["spearman"]
        if abs(rho) > 0.5:
            interpretacion = (
                "Existe **correlación fuerte** entre el cluster asignado y la posición final. "
                "El clustering captura aspectos relevantes del rendimiento deportivo."
            )
        elif abs(rho) > 0.3:
            interpretacion = (
                "Existe **correlación moderada** entre cluster y posición. "
                "El clustering refleja parcialmente la jerarquía deportiva."
            )
        else:
            interpretacion = (
                "Correlación **débil** entre cluster y posición — los clusters capturan "
                "estilo de conducción más que velocidad pura."
            )
        st.info(
            f"**Validación deportiva** · Spearman ρ = {rho:+.3f} (n = {validation['n']}). "
            f"{interpretacion}"
        )

    # ── 7. Footer técnico ───────────────────────────────────────────────────
    with st.expander("◷  Detalles técnicos del modelo"):
        st.markdown(f"""
        - **Algoritmo**: K-Means (Lloyd's algorithm), `random_state=42`, `n_init=10`.
        - **Preprocesamiento**: `StandardScaler` (media 0, std 1).
        - **Reducción dimensional para visualización**: PCA 2 componentes.
        - **Features ({len(feature_cols)})**: {", ".join(feature_cols)}.
        - **Determinismo**: cualquier ejecución sobre la misma sesión devuelve los
          mismos clusters (semilla fija, sin elementos aleatorios).
        - **Validación interna**: Silhouette score = `{sil:.4f}`.
        - **Validación deportiva**: correlación de Spearman entre rank del cluster
          (ordenado por velocidad media descendente) y posición final del piloto.
        """)
