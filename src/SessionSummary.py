"""
SessionSummary.py — TALOS · Race Summary
==========================================

Vista panorámica de la carrera: hero, mapa del circuito, clima, estadísticas, 
y ADN del circuito (clasificación K-means).
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np

from ui_assets import CSS_SESSION_SUMMARY, CSS_CIRCUIT_DNA
from CircuitClassifier import render_circuit_dna


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: render HTML sin que streamlit lo procese como markdown
# ─────────────────────────────────────────────────────────────────────────────

def _render_html(html: str) -> None:
    """Usa st.html (≥1.33) si existe; si no, markdown."""
    if hasattr(st, "html"):
        st.html(html)
    else:
        st.markdown(html, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# UTILIDADES DE FORMATO
# ─────────────────────────────────────────────────────────────────────────────

def _section(title: str, eyebrow: str = "") -> None:
    eb = f'<span class="ss-sect-eyebrow">{eyebrow}</span>' if eyebrow else ""
    _render_html(
        f'<div class="ss-sect">'
        f'<div class="ss-sect-pip"></div>'
        f'<div class="ss-sect-title">{title}</div>'
        f'{eb}'
        f'</div>'
    )


# ─────────────────────────────────────────────────────────────────────────────
# BLOQUES
# ─────────────────────────────────────────────────────────────────────────────

def _render_hero(session) -> None:
    ev = session.event
    name = ev.get("EventName", "—")
    country = ev.get("Country", "")
    location = ev.get("Location", "")
    try:
        date_str = pd.Timestamp(ev.get("EventDate")).strftime("%d %b %Y").upper()
    except Exception:
        date_str = ""
    round_n = ev.get("RoundNumber", "")

    bits = []
    if round_n:  bits.append(f"<b>RONDA {round_n}</b>")
    if date_str: bits.append(date_str)
    if country:  bits.append(country)
    if location and location != country:
        bits.append(location)
    meta = '<span class="sep">·</span>'.join(bits)

    _render_html(
        f'<div class="ss-hero">'
        f'<div class="ss-hero-pill">RESUMEN DE CARRERA</div>'
        f'<div class="ss-hero-name">{name}</div>'
        f'<div class="ss-hero-meta">{meta}</div>'
        f'</div>'
    )


def _render_circuit_map(ref_tel, fastest_lap, circuit_info=None) -> None:
    _section("CIRCUITO", "SECTORES Y CURVAS")

    fig = go.Figure()
    sector_colors = {"S1": "#E8002D", "S2": "#005AFF", "S3": "#FFD24A"}
    sectors_ok = False
    drs_ok     = False

    # ── 1. Asfalto gris (base, más ancho, añadido primero → debajo) ───────────
    fig.add_trace(go.Scatter(
        x=ref_tel["X"], y=ref_tel["Y"],
        mode="lines",
        line=dict(color="#1A1A1A", width=14),
        hoverinfo="skip", showlegend=False,
    ))

    # ── 2. Sectores coloreados (encima del gris) ──────────────────────────────
    try:
        s1_time = fastest_lap["Sector1SessionTime"]
        s2_time = fastest_lap["Sector2SessionTime"]
        idx_s1 = ref_tel[ref_tel["SessionTime"] <= s1_time].index[-1]
        idx_s2 = ref_tel[ref_tel["SessionTime"] <= s2_time].index[-1]
        for seg_name, seg, color in [
            ("Sector 1", ref_tel.loc[:idx_s1],       sector_colors["S1"]),
            ("Sector 2", ref_tel.loc[idx_s1:idx_s2], sector_colors["S2"]),
            ("Sector 3", ref_tel.loc[idx_s2:],       sector_colors["S3"]),
        ]:
            fig.add_trace(go.Scatter(
                x=seg["X"], y=seg["Y"],
                mode="lines",
                line=dict(color=color, width=3),
                hoverinfo="skip", name=seg_name, showlegend=False,
            ))
        sectors_ok = True
    except Exception:
        fig.add_trace(go.Scatter(
            x=ref_tel["X"], y=ref_tel["Y"],
            mode="lines",
            line=dict(color="#E8E9EF", width=3),
            hoverinfo="skip", showlegend=False,
        ))

    # ── 3. Apertura DRS: marcador en el punto donde DRS se abre ──────────────
    if "DRS" in ref_tel.columns:
        try:
            drs_col  = ref_tel["DRS"].fillna(0)
            prev_drs = drs_col.shift(1).fillna(0)
            # Transición: de inactivo (<10) a activo (≥10) → punto de apertura
            apertura = ref_tel[(drs_col >= 10) & (prev_drs < 10)]
            for _, pt in apertura.iterrows():
                fig.add_trace(go.Scatter(
                    x=[float(pt["X"])], y=[float(pt["Y"])],
                    mode="markers+text",
                    marker=dict(size=9, color="#39FF14", symbol="circle",
                                line=dict(color="#0D0D12", width=1.5)),
                    text=[" DRS"],
                    textposition="middle right",
                    textfont=dict(size=8, color="#39FF14", family="JetBrains Mono"),
                    hovertemplate="APERTURA DRS<extra></extra>",
                    showlegend=False, name="DRS",
                ))
            drs_ok = len(apertura) > 0
        except Exception:
            pass

    # ── 4. Trampa de velocidad (punto de velocidad máxima) ────────────────────
    if "Speed" in ref_tel.columns:
        try:
            max_idx = ref_tel["Speed"].idxmax()
            st_x    = float(ref_tel.loc[max_idx, "X"])
            st_y    = float(ref_tel.loc[max_idx, "Y"])
            st_v    = float(ref_tel.loc[max_idx, "Speed"])
            fig.add_trace(go.Scatter(
                x=[st_x], y=[st_y],
                mode="markers+text",
                marker=dict(size=10, color="#FFD24A", symbol="circle",
                            line=dict(color="#0D0D12", width=1.5)),
                text=[f"  {st_v:.0f} km/h"],
                textposition="middle right",
                textfont=dict(size=9, color="#FFD24A", family="JetBrains Mono"),
                hovertemplate=f"TRAMPA DE VELOCIDAD · <b>{st_v:.0f} km/h</b><extra></extra>",
                showlegend=False, name="Trampa de velocidad",
            ))
        except Exception:
            pass

    # ── 5. Salida / Meta + flecha de dirección ────────────────────────────────
    sf_x = float(ref_tel["X"].iloc[0])
    sf_y = float(ref_tel["Y"].iloc[0])

    # Calcular ángulo de la pista en la meta (dirección de marcha)
    dx = float(ref_tel["X"].iloc[8]) - float(ref_tel["X"].iloc[0])
    dy = float(ref_tel["Y"].iloc[8]) - float(ref_tel["Y"].iloc[0])
    angle_deg = float(np.degrees(np.arctan2(dy, dx)))  # ángulo matemático (CCW desde eje X)

    fig.add_trace(go.Scatter(
        x=[sf_x], y=[sf_y],
        mode="markers",
        marker=dict(color="#fff", size=12, symbol="diamond",
                    line=dict(color="#0D0D12", width=2)),
        hovertemplate="SALIDA / META<extra></extra>",
        showlegend=False,
    ))

    # Flecha ▶▶▶ paralela al trazado, desplazada al exterior (perpendicular derecha)
    # Perpendicular derecha respecto a la dirección de marcha: (sin θ, -cos θ)
    # Escalamos el offset al 3% del ancho del circuito para que sea proporcional
    track_w   = float(ref_tel["X"].max() - ref_tel["X"].min())
    track_h   = float(ref_tel["Y"].max() - ref_tel["Y"].min())
    offset_x    = max(track_w, track_h) * -0.085         
    offset_y    = max(track_w, track_h) * -0.08          
    perp_x    = float(np.sin(np.radians(angle_deg)))   # perpendicular derecha X
    perp_y    = -float(np.cos(np.radians(angle_deg)))  # perpendicular derecha Y

    fig.add_annotation(
        x=sf_x + perp_x * offset_x,
        y=sf_y + perp_y * offset_y,
        text=">>>>",
        showarrow=False,
        textangle=-angle_deg,
        font=dict(color="rgba(255,255,255,0.80)", size=11, family="Arial"),
        xanchor="center", yanchor="middle",
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=460, margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False, scaleanchor="x", scaleratio=1),
    )

    _render_html('<div class="ss-circuit-frame">')
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    _render_html('</div>')

    # ── Leyenda dinámica ──────────────────────────────────────────────────────
    legend = ['<div class="ss-sector-legend">']
    if sectors_ok:
        for k, lbl in [("S1","SECTOR 1"),("S2","SECTOR 2"),("S3","SECTOR 3")]:
            legend.append(
                f'<span class="ss-sector-leg">'
                f'<span class="ss-sector-dot" style="background:{sector_colors[k]};"></span>'
                f'{lbl}</span>'
            )
    if drs_ok:
        legend.append(
            '<span class="ss-sector-leg">'
            '<span class="ss-sector-dot" style="background:#39FF14;"></span>'
            'APERTURA DRS</span>'
        )
    legend.append(
        '<span class="ss-sector-leg">'
        '<span class="ss-sector-dot" style="background:#FFD24A;"></span>'
        'VEL. MÁXIMA</span>'
    )
    legend.append('</div>')
    _render_html("".join(legend))


def _render_weather(session) -> None:
    _section("METEOROLOGÍA", "MEDIA DURANTE LA SESIÓN")

    try:
        weather = session.weather_data
        if weather is None or weather.empty:
            _render_html('<div class="ss-empty">Datos meteorológicos no disponibles para esta sesión.</div>')
            return

        air   = float(weather["AirTemp"].mean())
        track = float(weather["TrackTemp"].mean())
        humid = float(weather["Humidity"].mean())

        bars = [
            ("TEMP. AIRE",    air,   "°C", 50,  "#00D7B5"),
            ("TEMP. ASFALTO", track, "°C", 70,  "#FF6B35"),
            ("HUMEDAD",       humid, "%",  100, "#3B82F6"),
        ]
        cards = ['<div class="ss-weather-grid">']
        for name, val, unit, vmax, color in bars:
            pct = float(np.clip(val / vmax * 100, 0, 100))
            cards.append(
                f'<div class="ss-wbar" style="--wbar-color:{color};">'
                f'<div class="ss-wbar-row">'
                f'<span class="ss-wbar-name">{name}</span>'
                f'<span class="ss-wbar-value">{val:.1f}<span class="ss-wbar-unit">{unit}</span></span>'
                f'</div>'
                f'<div class="ss-wbar-track"><div class="ss-wbar-fill" style="width:{pct:.1f}%;"></div></div>'
                f'<div class="ss-wbar-meta"><span>0</span><span>{vmax}{unit}</span></div>'
                f'</div>'
            )
        cards.append('</div>')
        _render_html("".join(cards))
    except Exception:
        _render_html('<div class="ss-empty">Datos meteorológicos no disponibles.</div>')


def _render_stats(session, ref_tel, circuit_info) -> None:
    _section("ESTADÍSTICAS")

    circuit_km = float(ref_tel["Distance"].max() / 1000) if "Distance" in ref_tel.columns else 0.0
    try:
        sc_laps = session.laps[session.laps["TrackStatus"].astype(str).str.contains("4|6", na=False)]
        sc_pct = (len(sc_laps) / len(session.laps)) * 100 if len(session.laps) > 0 else 0
    except Exception:
        sc_pct = 0.0
    num_corners = len(circuit_info.corners) if circuit_info is not None else None
    total_laps = session.total_laps if hasattr(session, "total_laps") else None

    stats = [
        ("CURVAS",         f"{num_corners}" if num_corners else "—", ""),
        ("VUELTAS",        f"{total_laps}"  if total_laps  else "—", ""),
        ("LONGITUD",       f"{circuit_km:.3f}",                       "km"),
        ("BAJO SC / VSC",  f"{sc_pct:.1f}",                           "%"),
    ]

    parts = ['<div class="ss-stats">']
    for label, val, unit in stats:
        u = f'<span class="ss-stat-unit">{unit}</span>' if unit else ""
        parts.append(
            f'<div class="ss-stat">'
            f'<div class="ss-stat-label">{label}</div>'
            f'<div class="ss-stat-value">{val}{u}</div>'
            f'</div>'
        )
    parts.append('</div>')
    _render_html("".join(parts))


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def render_session_summary(session):
    """Orquesta hero + mapa + clima + estadísticas + ADN."""
    # Inyectar CSS (vive en ui_assets)
    st.markdown(CSS_SESSION_SUMMARY, unsafe_allow_html=True)
    st.markdown(CSS_CIRCUIT_DNA,     unsafe_allow_html=True)

    # 0. Datos base
    try:
        fastest_lap = session.laps.pick_fastest()
        ref_tel = fastest_lap.get_telemetry().add_distance()
    except Exception:
        _render_html(
            '<div class="ss-empty"><b>Telemetría insuficiente</b> — esta sesión no contiene '
            'datos suficientes para generar el resumen.</div>'
        )
        return
    try:
        circuit_info = session.get_circuit_info()
    except Exception:
        circuit_info = None

    # 1. Hero
    _render_hero(session)

    # 2. Mapa del circuito (El protagonista visual, subido de jerarquía)
    _render_circuit_map(ref_tel, fastest_lap, circuit_info)

    # 3. Clima y Estadísticas
    _render_weather(session)
    _render_stats(session, ref_tel, circuit_info)

    # 4. ADN del circuito (clasificación K-means + insight)
    _section("ADN DEL CIRCUITO", "K-MEANS · 5 CLUSTERS · 2022–2025")
    render_circuit_dna(session)