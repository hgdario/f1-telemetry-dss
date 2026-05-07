"""
SessionCompare.py — TALOS F1 Comparativa de Sesión
====================================================
Módulo completo de análisis comparativo. Cubre desde la comparativa de
vueltas individuales hasta el contexto de campeonato en la fecha de la sesión.
"""

from __future__ import annotations

import ui_assets
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import fastf1
from fastf1.core import Session
from typing import Optional
import Head2head as h2h

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────────────────────────────────────────

BG_DARK    = "#0E0E0F"
BG_MAP     = "#13131A"
BG_PANEL   = "#111115"
BG_SURFACE = "#1A1A2E"
F1_WHITE   = "#FFFFFF"
F1_RED     = "#E8002D"
MONO_FONT  = "'JetBrains Mono', 'Courier New', monospace"

TYRE_COLORS = {
    "SOFT": "#E8002D", "MEDIUM": "#FFF200", "HARD": "#EBEBEB",
    "INTER": "#43B02A", "WET": "#0067FF",
    "SUPERSOFT": "#E8002D", "ULTRASOFT": "#C77DFF", "HYPERSOFT": "#FF66B2",
    "UNKNOWN": "#888888",
}
TYRE_ICONS = {
    "SOFT": "🔴", "MEDIUM": "🟡", "HARD": "⚪",
    "INTER": "🟢", "WET": "🔵", "UNKNOWN": "⚫",
}

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_time(seconds: float) -> str:
    if np.isnan(seconds) or seconds <= 0:
        return "—"
    m  = int(seconds // 60)
    s  = int(seconds % 60)
    ms = int(round((seconds % 1) * 1000))
    return f"{m}:{s:02d}.{ms:03d}"


def _td_to_s(td) -> float:
    try:
        return float(td.total_seconds())
    except Exception:
        return np.nan


def _get_driver_color(sesion: Session, drv: str) -> str:
    try:
        color = sesion.get_driver(drv)['TeamColor']
        if pd.isna(color) or not color:
            return "#888888"
        return f"#{color}"
    except Exception:
        return "#888888"


def _get_team_name(sesion: Session, drv: str) -> str:
    try:
        info = sesion.get_driver(drv)
        return str(info.get("TeamName", info.get("Team", drv)))
    except Exception:
        return drv


def _get_full_name(sesion: Session, drv: str) -> str:
    try:
        info = sesion.get_driver(drv)
        return f"{info.get('FirstName','')} {info.get('LastName', drv)}".strip()
    except Exception:
        return drv


def _apply_dark(fig: go.Figure, height: int = 420) -> None:
    fig.update_layout(
        height=height,
        paper_bgcolor=BG_DARK, plot_bgcolor=BG_PANEL,
        font=dict(family=MONO_FONT, color=F1_WHITE, size=10),
        hoverlabel=dict(bgcolor=BG_SURFACE, bordercolor="rgba(255,255,255,0.15)",
                        font=dict(family=MONO_FONT, color=F1_WHITE, size=10)),
    )


def _grid_axis(suffix="") -> dict:
    return dict(
        showgrid=True, gridcolor="rgba(255,255,255,0.06)",
        zeroline=False, ticksuffix=suffix,
        tickfont=dict(size=9, color="rgba(255,255,255,0.45)", family=MONO_FONT),
    )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — COMPARATIVA DE VUELTAS
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_brake(b: np.ndarray) -> np.ndarray:
    if b.dtype == bool or set(np.unique(b[~np.isnan(b)])).issubset({0,1}):
        return b.astype(float) * 100
    return b * 100 if b.max() <= 1.0 else b.copy()


def _render_tab_laps(sesion: Session) -> None:
    st.markdown(
        "<p style='font-family:JetBrains Mono,monospace;font-size:10px;"
        "letter-spacing:2px;color:rgba(255,255,255,0.35);margin-bottom:8px;'>"
        "SELECCIONA HASTA 4 VUELTAS PARA SUPERPONER</p>",
        unsafe_allow_html=True,
    )

    drivers_list = list(sesion.drivers)
    driver_names = {d: _get_full_name(sesion, d) for d in drivers_list}

    # Configuración de N vueltas (1 a 4)
    n_laps = st.radio("Número de vueltas a comparar", [2, 3, 4],
                      horizontal=True, key="sc_nlaps")

    lap_configs: list[tuple[str, int]] = []
    cols = st.columns(n_laps)

    for i, col in enumerate(cols):
        with col:
            drv = st.selectbox(
                f"Piloto {i+1}", drivers_list,
                index=min(i, len(drivers_list)-1),
                format_func=lambda x: driver_names.get(x, x),
                key=f"sc_lap_drv_{i}",
            )
            try:
                laps = sesion.laps.pick_drivers(drv).pick_accurate()
                fastest_n = laps.pick_fastest()["LapNumber"]
                lap_list  = sorted(laps["LapNumber"].tolist())
                def_idx   = lap_list.index(fastest_n) if fastest_n in lap_list else 0
                lap_n = st.selectbox(
                    "Vuelta", lap_list, index=def_idx,
                    format_func=lambda x: f"V{int(x)} ⏱" if x == fastest_n else f"V{int(x)}",
                    key=f"sc_lap_n_{i}",
                )
            except Exception:
                lap_n = 1
            lap_configs.append((str(drv), int(lap_n)))

    st.divider()

    # Carga de telemetrías
    tels: list[tuple[str, int, pd.DataFrame, str, float]] = []
    with st.spinner("Cargando telemetrías..."):
        for drv, lap_n in lap_configs:
            try:
                laps = sesion.laps.pick_drivers(drv)
                lap  = laps[laps["LapNumber"] == lap_n].iloc[0]
                tel  = lap.get_telemetry().add_distance()
                color = _get_driver_color(sesion, drv)
                lt    = _td_to_s(lap.get("LapTime"))
                tels.append((drv, lap_n, tel, color, lt))
            except Exception:
                st.warning(f"⚠️ No se pudo cargar {drv} V{lap_n}")

    if len(tels) < 2:
        st.error("Se necesitan al menos 2 vueltas con telemetría válida.")
        return

    # Rejilla de distancia común
    dist_max  = min(float(t[2]["Distance"].max()) for t in tels)
    dist_grid = np.linspace(0, dist_max, 2000)

    # Tabla de tiempos en cabecera
    _render_lap_time_table(sesion, tels)

    st.divider()

    # Subplots de telemetría
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=["VELOCIDAD (km/h)", "ACELERADOR / FRENO (%)",
                        "MARCHAS", "RPM"],
        row_heights=[0.35, 0.25, 0.20, 0.20],
    )

    channels = [
        ("Speed",    1, " km/h"),
        ("Throttle", 2, "%"),
        ("nGear",    3, ""),
        ("RPM",      4, " rpm"),
    ]

    dashes = ["solid", "dot", "dash", "dashdot"]

    for i, (drv, lap_n, tel, color, lt) in enumerate(tels):
        dist  = tel["Distance"].values.astype(float)
        label = f"{drv} V{lap_n}  {_fmt_time(lt)}" if lt else f"{drv} V{lap_n}"
        dash  = dashes[i % len(dashes)]

        for chan, row, sfx in channels:
            if chan not in tel.columns:
                continue
            vals = tel[chan].values.astype(float)
            y    = np.interp(dist_grid, dist, vals)
            fig.add_trace(go.Scatter(
                x=dist_grid, y=y,
                mode="lines",
                line=dict(color=color, width=1.6, dash=dash),
                name=label, legendgroup=label,
                showlegend=(row == 1),
                hovertemplate=f"<b>{label}</b>  %{{y:.1f}}{sfx}<extra></extra>",
            ), row=row, col=1)

        # Freno en row 2
        if "Brake" in tel.columns:
            brk = _normalize_brake(tel["Brake"].values)
            y_b = np.interp(dist_grid, dist, brk.astype(float))
            fig.add_trace(go.Scatter(
                x=dist_grid, y=y_b,
                mode="lines",
                line=dict(color=f"rgba({ui_assets.hex_rgb(color)},0.45)", width=1.0, dash=dash),
                showlegend=False, hoverinfo="skip", name=f"_brk_{i}",
            ), row=2, col=1)

    # Etiquetas curvas
    try:
        ci = sesion.get_circuit_info()
        for _, row in ci.corners.iterrows():
            d_c = float(row["Distance"])
            if d_c > dist_max:
                continue
            for r in range(1, 5):
                fig.add_vline(x=d_c, line_width=0.5, line_dash="dash",
                              line_color="rgba(255,255,255,0.08)", row=r, col=1)
            fig.add_annotation(
                x=d_c, y=1.02, xref="x", yref="paper",
                text=f"T{int(row['Number'])}",
                showarrow=False,
                font=dict(size=7, color="rgba(255,255,255,0.30)", family=MONO_FONT),
            )
    except Exception:
        pass

    _apply_dark(fig, 560)
    fig.update_layout(
        hovermode="x unified",
        legend=dict(orientation="h", y=1.06, x=0.5, xanchor="center",
                    bgcolor="rgba(0,0,0,0)", font=dict(size=9)),
        margin=dict(l=54, r=20, t=55, b=40),
        xaxis4=dict(**_grid_axis(" m"), title="Distancia (m)"),
    )
    for i in range(1, 5):
        key = "xaxis" if i == 1 else f"xaxis{i}"
        fig.update_layout(**{key: _grid_axis()})
        key = "yaxis" if i == 1 else f"yaxis{i}"
        fig.update_layout(**{key: _grid_axis()})
    for ann in fig.layout.annotations:
        ann.update(font=dict(family=MONO_FONT, size=9,
                             color="rgba(255,255,255,0.38)"),
                   x=0, xanchor="left")

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def _render_lap_time_table(
    sesion: Session,
    tels: list[tuple[str, int, pd.DataFrame, str, float]],
) -> None:
    rows_html = ""
    min_time  = min((lt for _, _, _, _, lt in tels if not np.isnan(lt)), default=0)

    for drv, lap_n, _, color, lt in tels:
        full_name  = _get_full_name(sesion, drv)
        team_name  = _get_team_name(sesion, drv)
        gap        = lt - min_time if not np.isnan(lt) else np.nan
        gap_str    = "POLE" if gap < 0.001 else (f"+{gap:.3f}s" if not np.isnan(gap) else "—")
        gap_color  = "#39FF14" if gap < 0.001 else "rgba(255,255,255,0.6)"

        # SIN SANGRÍA para evitar el bloque de código Markdown
        rows_html += f"""
<tr style="border-bottom:1px solid rgba(255,255,255,0.05);">
<td style="padding:7px 12px;">
<span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:{color};margin-right:8px;"></span>
<b style="color:{color};">{drv}</b>
<span style="color:rgba(255,255,255,0.5);font-size:10px;"> {full_name}</span>
</td>
<td style="padding:7px 12px;color:rgba(255,255,255,0.5);font-size:10px;">{team_name}</td>
<td style="padding:7px 12px;color:rgba(255,255,255,0.4);font-size:10px;">V{lap_n}</td>
<td style="padding:7px 12px;font-weight:700;color:{F1_WHITE};">{_fmt_time(lt)}</td>
<td style="padding:7px 12px;font-weight:700;color:{gap_color};">{gap_str}</td>
</tr>
"""

    html = f"""
<table style="width:100%;border-collapse:collapse;font-family:{MONO_FONT};font-size:12px;color:{F1_WHITE};background:{BG_MAP};margin-bottom:12px;">
<thead>
<tr style="border-bottom:2px solid rgba(255,255,255,0.12);color:rgba(255,255,255,0.35);font-size:9px;letter-spacing:1px;">
<th style="padding:8px 12px;text-align:left;">PILOTO</th>
<th style="padding:8px 12px;text-align:left;">EQUIPO</th>
<th style="padding:8px 12px;text-align:left;">VUELTA</th>
<th style="padding:8px 12px;text-align:left;">TIEMPO</th>
<th style="padding:8px 12px;text-align:left;">GAP</th>
</tr>
</thead>
<tbody>{rows_html}</tbody>
</table>
"""
    st.markdown(html, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — ANÁLISIS DE STINTS
# ─────────────────────────────────────────────────────────────────────────────

def _render_tab_stints(sesion: Session) -> None:
    drivers_list = list(sesion.drivers)
    driver_names = {d: _get_full_name(sesion, d) for d in drivers_list}

    selected_drivers = st.multiselect(
        "Pilotos a comparar",
        options=drivers_list,
        default=drivers_list[:4],
        format_func=lambda x: driver_names.get(x, x),
        key="sc_stint_drivers",
    )
    if not selected_drivers:
        st.info("Selecciona al menos un piloto.")
        return

    st.divider()

    # ── Ritmo por vuelta (pace evolution) ─────────────────────────────────
    st.markdown(
        "<p style='font-family:JetBrains Mono,monospace;font-size:10px;"
        "letter-spacing:2px;color:rgba(255,255,255,0.35);margin-bottom:4px;'>"
        "EVOLUCIÓN DE RITMO POR VUELTA · tamaño = vida del compuesto</p>",
        unsafe_allow_html=True,
    )

    fig_pace = go.Figure()

    for drv in selected_drivers:
        color = _get_driver_color(sesion, drv)
        try:
            laps = sesion.laps.pick_drivers(drv).pick_accurate()
            laps = laps[laps["LapTime"].notna()].copy()
            laps["LapTime_s"] = laps["LapTime"].apply(_td_to_s)
            laps["Compound"]  = laps["Compound"].fillna("UNKNOWN").str.upper()
            laps["TyreLife"]  = laps["TyreLife"].fillna(1)

            for compound in laps["Compound"].unique():
                mask = laps["Compound"] == compound
                sub  = laps[mask]
                tc   = TYRE_COLORS.get(compound, "#888888")

                fig_pace.add_trace(go.Scatter(
                    x=sub["LapNumber"],
                    y=sub["LapTime_s"],
                    mode="markers+lines",
                    marker=dict(
                        size=np.clip(sub["TyreLife"].values / 2 + 6, 6, 18),
                        color=tc,
                        line=dict(color=color, width=1.5),
                        symbol="circle",
                    ),
                    line=dict(color=color, width=1.2, dash="solid"),
                    name=f"{drv} {TYRE_ICONS.get(compound,'⚫')} {compound}",
                    legendgroup=drv,
                    hovertemplate=(
                        f"<b>{drv}</b>  V%{{x}}<br>"
                        f"Tiempo: %{{y:.3f}}s<br>"
                        f"Compuesto: {compound}<extra></extra>"
                    ),
                ))
        except Exception as e:
            st.caption(f"⚠️ {drv}: {e}")

    _apply_dark(fig_pace, 400)
    fig_pace.update_layout(
        hovermode="x unified",
        legend=dict(orientation="h", y=1.04, x=0.5, xanchor="center",
                    bgcolor="rgba(0,0,0,0)", font=dict(size=9)),
        margin=dict(l=54, r=20, t=30, b=40),
        xaxis=dict(**_grid_axis(), title="Vuelta"),
        yaxis=dict(**_grid_axis("s"), title="Tiempo (s)", autorange="reversed"),
    )
    st.plotly_chart(fig_pace, use_container_width=True, config={"displayModeBar": False})

    st.divider()

    # ── Degradación por compuesto ─────────────────────────────────────────
    st.markdown(
        "<p style='font-family:JetBrains Mono,monospace;font-size:10px;"
        "letter-spacing:2px;color:rgba(255,255,255,0.35);margin-bottom:4px;'>"
        "DEGRADACIÓN · tiempo vs vida del neumático (regresión lineal por stint)</p>",
        unsafe_allow_html=True,
    )

    fig_deg = go.Figure()

    for drv in selected_drivers:
        color = _get_driver_color(sesion, drv)
        try:
            laps = sesion.laps.pick_drivers(drv).pick_accurate()
            laps = laps[laps["LapTime"].notna()].copy()
            laps["LapTime_s"] = laps["LapTime"].apply(_td_to_s)
            laps["TyreLife"]  = laps["TyreLife"].fillna(1).astype(float)
            laps["Compound"]  = laps["Compound"].fillna("UNKNOWN").str.upper()

            for compound in laps["Compound"].unique():
                mask = laps["Compound"] == compound
                sub  = laps[mask].sort_values("TyreLife")
                if len(sub) < 3:
                    continue

                tl  = sub["TyreLife"].values.astype(float)
                lt  = sub["LapTime_s"].values.astype(float)
                tc  = TYRE_COLORS.get(compound, "#888888")

                fig_deg.add_trace(go.Scatter(
                    x=tl, y=lt, mode="markers",
                    marker=dict(size=7, color=tc, line=dict(color=color, width=1)),
                    name=f"{drv} {compound}",
                    legendgroup=f"{drv}_{compound}",
                    hovertemplate=f"<b>{drv}</b>  Vida: %{{x}} · %{{y:.3f}}s<extra>{compound}</extra>",
                ))

                coeffs = np.polyfit(tl, lt, 1)
                x_reg  = np.linspace(tl.min(), tl.max(), 50)
                y_reg  = np.polyval(coeffs, x_reg)
                deg_per_lap = coeffs[0]

                fig_deg.add_trace(go.Scatter(
                    x=x_reg, y=y_reg,
                    mode="lines",
                    line=dict(color=color, width=1.5, dash="dot"),
                    showlegend=False,
                    hovertemplate=(
                        f"<b>{drv} {compound}</b><br>"
                        f"Degradación: {deg_per_lap:+.3f}s/vuelta<extra></extra>"
                    ),
                ))
        except Exception:
            continue

    _apply_dark(fig_deg, 360)
    fig_deg.update_layout(
        hovermode="closest",
        legend=dict(orientation="h", y=1.04, x=0.5, xanchor="center",
                    bgcolor="rgba(0,0,0,0)", font=dict(size=9)),
        margin=dict(l=54, r=20, t=20, b=40),
        xaxis=dict(**_grid_axis(), title="Vida del neumático (vueltas)"),
        yaxis=dict(**_grid_axis("s"), title="Tiempo de vuelta (s)", autorange="reversed"),
    )
    st.plotly_chart(fig_deg, use_container_width=True, config={"displayModeBar": False})

    st.divider()

    # ── Tabla de stints ───────────────────────────────────────────────────
    st.markdown(
        "<p style='font-family:JetBrains Mono,monospace;font-size:10px;"
        "letter-spacing:2px;color:rgba(255,255,255,0.35);margin-bottom:8px;'>"
        "RESUMEN DE STINTS</p>",
        unsafe_allow_html=True,
    )

    stint_rows = []
    for drv in selected_drivers:
        try:
            laps = sesion.laps.pick_drivers(drv).pick_accurate()
            laps = laps[laps["LapTime"].notna()].copy()
            laps["LapTime_s"] = laps["LapTime"].apply(_td_to_s)
            laps["Stint"]     = laps["Stint"].fillna(0).astype(int)
            laps["Compound"]  = laps["Compound"].fillna("UNKNOWN").str.upper()
            laps["TyreLife"]  = laps["TyreLife"].fillna(0).astype(int)

            for stint_n, grp in laps.groupby("Stint"):
                if grp.empty:
                    continue
                compound   = str(grp["Compound"].iloc[0])
                life_start = int(grp["TyreLife"].iloc[0])
                life_end   = int(grp["TyreLife"].iloc[-1])
                n_laps     = len(grp)
                best_time  = float(grp["LapTime_s"].min())
                mean_time  = float(grp["LapTime_s"].mean())
                color      = _get_driver_color(sesion, drv)
                tc         = TYRE_COLORS.get(compound, "#888888")

                stint_rows.append({
                    "driver"    : drv,
                    "full_name" : _get_full_name(sesion, drv),
                    "team"      : _get_team_name(sesion, drv),
                    "color"     : color,
                    "stint"     : stint_n,
                    "compound"  : compound,
                    "tyre_color": tc,
                    "life_range": f"{life_start}–{life_end}",
                    "n_laps"    : n_laps,
                    "best"      : best_time,
                    "mean"      : mean_time,
                })
        except Exception:
            continue

    if stint_rows:
        rows_html = ""
        for r in sorted(stint_rows, key=lambda x: (x["driver"], x["stint"])):
            # SIN SANGRÍA para la fila
            rows_html += f"""
<tr style="border-bottom:1px solid rgba(255,255,255,0.04);">
<td style="padding:6px 10px;">
<span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:{r['color']};margin-right:6px;"></span>
<b style="color:{r['color']};">{r['driver']}</b>
</td>
<td style="padding:6px 10px;color:rgba(255,255,255,0.45);font-size:10px;">{r['team']}</td>
<td style="padding:6px 10px;color:rgba(255,255,255,0.5);">{r['stint']}</td>
<td style="padding:6px 10px;">
<span style="background:{r['tyre_color']};color:#000;padding:2px 8px;border-radius:3px;font-size:10px;font-weight:700;">
{TYRE_ICONS.get(r['compound'],'⚫')} {r['compound']}
</span>
</td>
<td style="padding:6px 10px;color:rgba(255,255,255,0.5);">{r['life_range']} vueltas</td>
<td style="padding:6px 10px;">{r['n_laps']} vueltas</td>
<td style="padding:6px 10px;color:#39FF14;font-weight:700;">{_fmt_time(r['best'])}</td>
<td style="padding:6px 10px;color:#00D2FF;">{_fmt_time(r['mean'])}</td>
</tr>
"""
        # SIN SANGRÍA para la tabla global
        html = f"""
<div style="overflow-x:auto;max-height:400px;overflow-y:auto;">
<table style="width:100%;border-collapse:collapse;font-family:{MONO_FONT};font-size:12px;color:{F1_WHITE};background:{BG_MAP};">
<thead>
<tr style="border-bottom:2px solid rgba(255,255,255,0.12);color:rgba(255,255,255,0.35);font-size:9px;letter-spacing:1px;position:sticky;top:0;background:{BG_MAP};z-index:5;">
<th style="padding:8px 10px;text-align:left;">PILOTO</th>
<th style="padding:8px 10px;text-align:left;">EQUIPO</th>
<th style="padding:8px 10px;text-align:left;">STINT</th>
<th style="padding:8px 10px;text-align:left;">COMPUESTO</th>
<th style="padding:8px 10px;text-align:left;">VIDA</th>
<th style="padding:8px 10px;text-align:left;">VUELTAS</th>
<th style="padding:8px 10px;text-align:left;">MEJOR</th>
<th style="padding:8px 10px;text-align:left;">MEDIA</th>
</tr>
</thead>
<tbody>{rows_html}</tbody>
</table>
</div>
"""
        st.markdown(html, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PUNTO DE ENTRADA PÚBLICO
# ─────────────────────────────────────────────────────────────────────────────

def render_session_compare(sesion: Session, corners) -> None:
    tab_laps,tab_h2h, tab_stints  = st.tabs(["Vueltas", "H2H","Stints & Degradación"])

    with tab_laps:
        _render_tab_laps(sesion)

    with tab_h2h:
        h2h.render_lap_overlay(sesion, corners)


    with tab_stints:
        _render_tab_stints(sesion)

    
