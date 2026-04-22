"""
SessionCompare.py — TALOS F1 Comparativa de Sesión
====================================================
Módulo completo de análisis comparativo. Cubre desde la comparativa de
vueltas individuales hasta el contexto de campeonato en la fecha de la sesión.

Estructura de pestañas
───────────────────────
  Tab 1 · VUELTAS         Superposición de telemetría de N vueltas del mismo
                          piloto o de pilotos distintos. Canales: Speed, Throttle,
                          Brake, Gear, RPM. Comparativa de tiempos y sectores.

  Tab 2 · STINTS          Análisis por stint teniendo en cuenta el compuesto y
                          la vida del neumático. Evolución del pace por vuelta,
                          degradación, comparativa entre pilotos en el mismo stint.

  Tab 3 · EQUIPOS         Heatmap de tiempos de vuelta por equipo, distribución
                          de V punta y velocidad media. Radar chart de métricas
                          de desempeño normalizadas entre equipos.

  Tab 4 · CAMPEONATO      Tabla de puntos del mundial hasta la carrera de la
                          sesión. Gráfico de evolución de puntos a lo largo de
                          la temporada. Solo disponible si se carga con datos de
                          carrera (Race) o si FastF1 tiene el histórico.

Conexión al router:
  import SessionCompare as sc
  elif active == "Comparativa de Sesión":
      if not require_session(): st.stop()
      st.header("Comparativa de Sesión")
      st.divider()
      sc.render_session_compare(st.session_state["f1_session"])
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import fastf1
from fastf1.core import Session
from typing import Optional

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

# Puntos F1 estándar por posición (top 10)
POINTS_TABLE = {1:25, 2:18, 3:15, 4:12, 5:10, 6:8, 7:6, 8:4, 9:2, 10:1}


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


def _hex_rgb(hex_color: str) -> str:
    h = hex_color.lstrip("#")
    return f"{int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)}"


def _get_driver_color(sesion: Session, drv: str) -> str:
    try:
        return f"#{sesion.get_driver(drv)['TeamColor']}"
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
                line=dict(color=f"rgba({_hex_rgb(color)},0.45)", width=1.0, dash=dash),
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

        rows_html += f"""
        <tr style="border-bottom:1px solid rgba(255,255,255,0.05);">
            <td style="padding:7px 12px;">
                <span style="display:inline-block;width:10px;height:10px;border-radius:50%;
                    background:{color};margin-right:8px;"></span>
                <b style="color:{color};">{drv}</b>
                <span style="color:rgba(255,255,255,0.5);font-size:10px;"> {full_name}</span>
            </td>
            <td style="padding:7px 12px;color:rgba(255,255,255,0.5);font-size:10px;">
                {team_name}
            </td>
            <td style="padding:7px 12px;color:rgba(255,255,255,0.4);font-size:10px;">
                V{lap_n}
            </td>
            <td style="padding:7px 12px;font-weight:700;color:{F1_WHITE};">
                {_fmt_time(lt)}
            </td>
            <td style="padding:7px 12px;font-weight:700;color:{gap_color};">
                {gap_str}
            </td>
        </tr>
        """

    html = f"""
    <table style="width:100%;border-collapse:collapse;font-family:{MONO_FONT};
        font-size:12px;color:{F1_WHITE};background:{BG_MAP};margin-bottom:12px;">
        <thead>
        <tr style="border-bottom:2px solid rgba(255,255,255,0.12);
                   color:rgba(255,255,255,0.35);font-size:9px;letter-spacing:1px;">
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

            # Un punto por vuelta, coloreado por compuesto, tamaño por vida del neumático
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
        yaxis=dict(**_grid_axis("s"), title="Tiempo (s)",
                   autorange="reversed"),  # menor tiempo = arriba
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

                # Scatter puntos
                fig_deg.add_trace(go.Scatter(
                    x=tl, y=lt, mode="markers",
                    marker=dict(size=7, color=tc, line=dict(color=color, width=1)),
                    name=f"{drv} {compound}",
                    legendgroup=f"{drv}_{compound}",
                    hovertemplate=f"<b>{drv}</b>  Vida: %{{x}} · %{{y:.3f}}s<extra>{compound}</extra>",
                ))

                # Línea de regresión
                coeffs = np.polyfit(tl, lt, 1)
                x_reg  = np.linspace(tl.min(), tl.max(), 50)
                y_reg  = np.polyval(coeffs, x_reg)
                deg_per_lap = coeffs[0]   # segundos perdidos por vuelta

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
            rows_html += f"""
            <tr style="border-bottom:1px solid rgba(255,255,255,0.04);">
                <td style="padding:6px 10px;">
                    <span style="display:inline-block;width:8px;height:8px;border-radius:50%;
                        background:{r['color']};margin-right:6px;"></span>
                    <b style="color:{r['color']};">{r['driver']}</b>
                </td>
                <td style="padding:6px 10px;color:rgba(255,255,255,0.45);font-size:10px;">
                    {r['team']}
                </td>
                <td style="padding:6px 10px;color:rgba(255,255,255,0.5);">
                    {r['stint']}
                </td>
                <td style="padding:6px 10px;">
                    <span style="background:{r['tyre_color']};color:#000;
                        padding:2px 8px;border-radius:3px;font-size:10px;font-weight:700;">
                        {TYRE_ICONS.get(r['compound'],'⚫')} {r['compound']}
                    </span>
                </td>
                <td style="padding:6px 10px;color:rgba(255,255,255,0.5);">{r['life_range']} vueltas</td>
                <td style="padding:6px 10px;">{r['n_laps']} vueltas</td>
                <td style="padding:6px 10px;color:#39FF14;font-weight:700;">{_fmt_time(r['best'])}</td>
                <td style="padding:6px 10px;color:#00D2FF;">{_fmt_time(r['mean'])}</td>
            </tr>
            """
        html = f"""
        <div style="overflow-x:auto;max-height:400px;overflow-y:auto;">
        <table style="width:100%;border-collapse:collapse;font-family:{MONO_FONT};
            font-size:12px;color:{F1_WHITE};background:{BG_MAP};">
            <thead>
            <tr style="border-bottom:2px solid rgba(255,255,255,0.12);
                       color:rgba(255,255,255,0.35);font-size:9px;letter-spacing:1px;
                       position:sticky;top:0;background:{BG_MAP};z-index:5;">
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
        </table></div>
        """
        st.markdown(html, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — COMPARATIVA DE EQUIPOS
# ─────────────────────────────────────────────────────────────────────────────

def _render_tab_teams(sesion: Session) -> None:
    # Construir DataFrame de métricas por equipo
    records = []
    for drv in sesion.drivers:
        try:
            laps  = sesion.laps.pick_drivers(drv).pick_accurate()
            if laps.empty:
                continue
            best  = laps.pick_fastest()
            tel   = best.get_telemetry().add_distance()
            team  = _get_team_name(sesion, drv)
            color = _get_driver_color(sesion, drv)

            records.append({
                "driver"    : drv,
                "full_name" : _get_full_name(sesion, drv),
                "team"      : team,
                "color"     : color,
                "best_lap_s": _td_to_s(best["LapTime"]),
                "vmax"      : float(tel["Speed"].max()) if "Speed" in tel else np.nan,
                "vmean"     : float(tel["Speed"].mean()) if "Speed" in tel else np.nan,
                "thr_full_pct": float((tel["Throttle"]>=99).mean()*100) if "Throttle" in tel else np.nan,
                "compound"  : str(best.get("Compound","UNKNOWN")).upper(),
            })
        except Exception:
            continue

    if not records:
        st.warning("No hay datos de equipo disponibles.")
        return

    df = pd.DataFrame(records).sort_values("best_lap_s")

    # ── Heatmap de tiempos ────────────────────────────────────────────────
    st.markdown(
        "<p style='font-family:JetBrains Mono,monospace;font-size:10px;"
        "letter-spacing:2px;color:rgba(255,255,255,0.35);margin-bottom:4px;'>"
        "MEJOR VUELTA POR PILOTO</p>",
        unsafe_allow_html=True,
    )

    # Calcular todos los tiempos de vuelta para el violin
    all_lap_times: dict[str, list[float]] = {}
    for drv in sesion.drivers:
        try:
            laps = sesion.laps.pick_drivers(drv).pick_accurate()
            laps = laps[laps["LapTime"].notna()]
            team = _get_team_name(sesion, drv)
            times = laps["LapTime"].apply(_td_to_s).dropna().tolist()
            if team not in all_lap_times:
                all_lap_times[team] = []
            all_lap_times[team].extend(times)
        except Exception:
            continue

    # ── Gráfico de barras: mejor vuelta por piloto ─────────────────────────
    fig_bars = go.Figure()
    pole_s   = float(df["best_lap_s"].min())

    for _, row in df.iterrows():
        gap = row["best_lap_s"] - pole_s
        fig_bars.add_trace(go.Bar(
            x=[row["driver"]],
            y=[row["best_lap_s"]],
            name=row["team"],
            legendgroup=row["team"],
            marker=dict(color=row["color"],
                        line=dict(color="rgba(0,0,0,0.3)", width=1)),
            text=[f"+{gap:.3f}s" if gap > 0 else "POLE"],
            textposition="outside",
            textfont=dict(size=9, color=F1_WHITE),
            hovertemplate=(
                f"<b>{row['driver']}</b>  {row['team']}<br>"
                f"Mejor: {_fmt_time(row['best_lap_s'])}<br>"
                f"Gap: +{gap:.3f}s<extra></extra>"
            ),
        ))

    _apply_dark(fig_bars, 320)
    fig_bars.update_layout(
        barmode="group",
        showlegend=False,
        margin=dict(l=20, r=20, t=20, b=40),
        xaxis=dict(showgrid=False, tickfont=dict(size=10)),
        yaxis=dict(**_grid_axis("s"), title="Tiempo (s)", autorange="reversed"),
    )
    st.plotly_chart(fig_bars, use_container_width=True, config={"displayModeBar": False})

    st.divider()

    # ── Radar chart de métricas normalizadas ──────────────────────────────
    st.markdown(
        "<p style='font-family:JetBrains Mono,monospace;font-size:10px;"
        "letter-spacing:2px;color:rgba(255,255,255,0.35);margin-bottom:4px;'>"
        "RADAR DE DESEMPEÑO · métricas normalizadas por equipo</p>",
        unsafe_allow_html=True,
    )

    # Agrupar por equipo (media de sus 2 pilotos)
    team_df = df.groupby("team").agg({
        "best_lap_s"   : "min",
        "vmax"         : "max",
        "vmean"        : "mean",
        "thr_full_pct" : "mean",
        "color"        : "first",
    }).reset_index()

    # Normalizar 0-1 (mayor = mejor para todos)
    def norm(series: pd.Series, invert: bool = False) -> pd.Series:
        mn, mx = series.min(), series.max()
        n      = (series - mn) / (mx - mn + 1e-9)
        return 1 - n if invert else n

    team_df["n_pace"]   = norm(team_df["best_lap_s"], invert=True)   # menor tiempo = mejor
    team_df["n_vmax"]   = norm(team_df["vmax"])
    team_df["n_vmean"]  = norm(team_df["vmean"])
    team_df["n_thr"]    = norm(team_df["thr_full_pct"])

    categories = ["Velocidad Punta", "Vel. Media", "% A Fondo", "Pace (Inv.)"]

    fig_radar = go.Figure()
    for _, row in team_df.iterrows():
        vals = [row["n_vmax"], row["n_vmean"], row["n_thr"], row["n_pace"]]
        vals_closed = vals + [vals[0]]
        cats_closed = categories + [categories[0]]

        fig_radar.add_trace(go.Scatterpolar(
            r=vals_closed,
            theta=cats_closed,
            fill="toself",
            fillcolor=f"rgba({_hex_rgb(row['color'])},0.15)",
            line=dict(color=row["color"], width=2),
            name=row["team"],
            hovertemplate=f"<b>{row['team']}</b><br>%{{theta}}: %{{r:.2f}}<extra></extra>",
        ))

    fig_radar.update_layout(
        height=400,
        paper_bgcolor=BG_DARK, plot_bgcolor=BG_DARK,
        font=dict(family=MONO_FONT, color=F1_WHITE, size=10),
        polar=dict(
            bgcolor=BG_PANEL,
            radialaxis=dict(visible=True, range=[0, 1], showticklabels=False,
                            gridcolor="rgba(255,255,255,0.1)"),
            angularaxis=dict(tickfont=dict(size=10, color=F1_WHITE)),
        ),
        legend=dict(orientation="h", y=-0.1, x=0.5, xanchor="center",
                    bgcolor="rgba(0,0,0,0)", font=dict(size=9)),
        margin=dict(l=40, r=40, t=20, b=60),
    )
    st.plotly_chart(fig_radar, use_container_width=True, config={"displayModeBar": False})

    st.divider()

    # ── Distribución de velocidades por equipo (box plot) ──────────────────
    st.markdown(
        "<p style='font-family:JetBrains Mono,monospace;font-size:10px;"
        "letter-spacing:2px;color:rgba(255,255,255,0.35);margin-bottom:4px;'>"
        "DISTRIBUCIÓN DE TIEMPOS DE VUELTA POR EQUIPO</p>",
        unsafe_allow_html=True,
    )

    fig_box = go.Figure()
    for team, times in all_lap_times.items():
        if not times:
            continue
        color_match = team_df[team_df["team"] == team]["color"]
        tc = color_match.iloc[0] if not color_match.empty else "#888"

        fig_box.add_trace(go.Box(
            y=times,
            name=team,
            marker=dict(color=tc, size=4),
            line=dict(color=tc, width=1.5),
            boxmean="sd",
            hovertemplate=f"<b>{team}</b><br>%{{y:.3f}}s<extra></extra>",
        ))

    _apply_dark(fig_box, 360)
    fig_box.update_layout(
        showlegend=False,
        margin=dict(l=54, r=20, t=10, b=40),
        yaxis=dict(**_grid_axis("s"), title="Tiempo de vuelta (s)", autorange="reversed"),
        xaxis=dict(showgrid=False, tickfont=dict(size=10)),
    )
    st.plotly_chart(fig_box, use_container_width=True, config={"displayModeBar": False})


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — CAMPEONATO
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_standings(year: int, round_n: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Descarga clasificaciones de pilotos y constructores hasta el round dado
    usando la API pública de Jolpi/Ergast.
    Retorna (df_drivers, df_constructors).
    """
    base = "https://api.jolpi.ca/ergast/f1"
    df_d = pd.DataFrame()
    df_c = pd.DataFrame()

    try:
        r = __import__("requests").get(
            f"{base}/{year}/{round_n}/driverStandings.json", timeout=8
        )
        if r.ok:
            standings = r.json()["MRData"]["StandingsTable"]["StandingsLists"]
            if standings:
                rows = []
                for s in standings[0]["DriverStandings"]:
                    rows.append({
                        "pos"    : int(s["position"]),
                        "driver" : f"{s['Driver']['givenName']} {s['Driver']['familyName']}",
                        "abbr"   : s["Driver"].get("code", "?"),
                        "team"   : s["Constructors"][0]["name"] if s["Constructors"] else "?",
                        "points" : float(s["points"]),
                        "wins"   : int(s["wins"]),
                    })
                df_d = pd.DataFrame(rows)
    except Exception:
        pass

    try:
        r = __import__("requests").get(
            f"{base}/{year}/{round_n}/constructorStandings.json", timeout=8
        )
        if r.ok:
            standings = r.json()["MRData"]["StandingsTable"]["StandingsLists"]
            if standings:
                rows = []
                for s in standings[0]["ConstructorStandings"]:
                    rows.append({
                        "pos"    : int(s["position"]),
                        "team"   : s["Constructor"]["name"],
                        "points" : float(s["points"]),
                        "wins"   : int(s["wins"]),
                    })
                df_c = pd.DataFrame(rows)
    except Exception:
        pass

    return df_d, df_c


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_season_results(year: int, round_n: int) -> pd.DataFrame:
    """
    Descarga los resultados de todas las carreras de la temporada hasta round_n.
    Retorna DataFrame con columnas: round, driver, points.
    """
    base = "https://api.jolpi.ca/ergast/f1"
    rows = []
    try:
        import requests
        for rnd in range(1, round_n + 1):
            r = requests.get(f"{base}/{year}/{rnd}/results.json", timeout=6)
            if not r.ok:
                continue
            races = r.json()["MRData"]["RaceTable"]["Races"]
            if not races:
                continue
            race_name = races[0].get("raceName", f"R{rnd}")
            for res in races[0].get("Results", []):
                pos = int(res.get("position", 99))
                pts = POINTS_TABLE.get(pos, 0)
                # Fastest lap bonus
                if res.get("FastestLap", {}).get("rank") == "1":
                    pts += 1
                rows.append({
                    "round"  : rnd,
                    "race"   : race_name,
                    "driver" : f"{res['Driver']['givenName']} {res['Driver']['familyName']}",
                    "points" : pts,
                })
    except Exception:
        pass

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _render_tab_championship(sesion: Session) -> None:
    # Extraer año y ronda de la sesión
    try:
        year  = int(sesion.event["EventDate"].year)
        round_n = int(sesion.event["RoundNumber"])
    except Exception:
        st.info("No se pudo determinar la fecha/ronda de la sesión. "
                "Introduce los datos manualmente.")
        col1, col2 = st.columns(2)
        year    = col1.number_input("Año", 2018, 2026, 2024, key="sc_champ_year")
        round_n = col2.number_input("Ronda", 1, 24, 1, key="sc_champ_round")

    st.markdown(
        f"<p style='font-family:JetBrains Mono,monospace;font-size:10px;"
        f"letter-spacing:2px;color:rgba(255,255,255,0.35);margin-bottom:8px;'>"
        f"MUNDIAL {year} · CLASIFICACIÓN HASTA LA RONDA {round_n}</p>",
        unsafe_allow_html=True,
    )

    with st.spinner("Cargando clasificaciones..."):
        df_drivers, df_constructors = _fetch_standings(year, round_n)

    # ── Clasificación de pilotos ──────────────────────────────────────────
    if not df_drivers.empty:
        rows_html = ""
        for _, row in df_drivers.iterrows():
            # Intentar mapear abreviatura al color del equipo de la sesión
            color = "#888888"
            try:
                for drv in sesion.drivers:
                    if sesion.get_driver(drv).get("Abbreviation","").upper() == row["abbr"].upper():
                        color = _get_driver_color(sesion, drv)
                        break
            except Exception:
                pass

            rows_html += f"""
            <tr style="border-bottom:1px solid rgba(255,255,255,0.04);">
                <td style="padding:7px 10px;font-weight:800;color:{color};">{int(row['pos'])}</td>
                <td style="padding:7px 10px;">
                    <span style="display:inline-block;width:8px;height:8px;border-radius:50%;
                        background:{color};margin-right:8px;"></span>
                    <b>{row['driver']}</b>
                </td>
                <td style="padding:7px 10px;color:rgba(255,255,255,0.5);font-size:10px;">
                    {row['team']}
                </td>
                <td style="padding:7px 10px;font-weight:700;color:#FFD600;font-size:14px;">
                    {int(row['points'])}
                </td>
                <td style="padding:7px 10px;color:rgba(255,255,255,0.5);">{int(row['wins'])} ✓</td>
            </tr>
            """

        html = f"""
        <div style="overflow-y:auto;max-height:500px;margin-bottom:16px;">
        <table style="width:100%;border-collapse:collapse;font-family:{MONO_FONT};
            font-size:12px;color:{F1_WHITE};background:{BG_MAP};">
            <thead>
            <tr style="border-bottom:2px solid rgba(255,255,255,0.12);
                       color:rgba(255,255,255,0.35);font-size:9px;letter-spacing:1px;
                       position:sticky;top:0;background:{BG_MAP};z-index:5;">
                <th style="padding:8px 10px;text-align:left;">POS</th>
                <th style="padding:8px 10px;text-align:left;">PILOTO</th>
                <th style="padding:8px 10px;text-align:left;">EQUIPO</th>
                <th style="padding:8px 10px;text-align:left;">PTS</th>
                <th style="padding:8px 10px;text-align:left;">VICTORIAS</th>
            </tr>
            </thead>
            <tbody>{rows_html}</tbody>
        </table></div>
        """
        col_drv, col_con = st.columns([3, 2])

        with col_drv:
            st.markdown(
                "<p style='font-family:JetBrains Mono,monospace;font-size:10px;"
                "letter-spacing:2px;color:rgba(255,255,255,0.35);'>PILOTOS</p>",
                unsafe_allow_html=True,
            )
            st.markdown(html, unsafe_allow_html=True)

        with col_con:
            st.markdown(
                "<p style='font-family:JetBrains Mono,monospace;font-size:10px;"
                "letter-spacing:2px;color:rgba(255,255,255,0.35);'>CONSTRUCTORES</p>",
                unsafe_allow_html=True,
            )
            if not df_constructors.empty:
                fig_con = go.Figure(go.Bar(
                    x=df_constructors["points"],
                    y=df_constructors["team"],
                    orientation="h",
                    marker=dict(
                        color=[
                            next(
                                (_get_driver_color(sesion, drv)
                                 for drv in sesion.drivers
                                 if _get_team_name(sesion, drv) == row["team"]),
                                "#888"
                            )
                            for _, row in df_constructors.iterrows()
                        ],
                        line=dict(color="rgba(0,0,0,0.3)", width=1),
                    ),
                    text=df_constructors["points"].astype(int).astype(str),
                    textposition="outside",
                    textfont=dict(size=10, color=F1_WHITE),
                    hovertemplate="<b>%{y}</b><br>%{x} pts<extra></extra>",
                ))
                _apply_dark(fig_con, max(160, len(df_constructors) * 38 + 40))
                fig_con.update_layout(
                    margin=dict(l=10, r=60, t=10, b=10),
                    showlegend=False,
                    xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                               zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False,
                               tickfont=dict(size=10, color=F1_WHITE)),
                )
                st.plotly_chart(fig_con, use_container_width=True,
                                config={"displayModeBar": False})

    st.divider()

    # ── Evolución de puntos a lo largo de la temporada ────────────────────
    st.markdown(
        "<p style='font-family:JetBrains Mono,monospace;font-size:10px;"
        "letter-spacing:2px;color:rgba(255,255,255,0.35);margin-bottom:4px;'>"
        "EVOLUCIÓN DE PUNTOS · temporada acumulada</p>",
        unsafe_allow_html=True,
    )

    with st.spinner("Cargando resultados de temporada..."):
        df_season = _fetch_season_results(year, round_n)

    if df_season.empty:
        st.info("No se pudieron cargar los resultados de temporada.")
        return

    # Pilotos top 8 por puntos totales
    top_drivers = (
        df_season.groupby("driver")["points"].sum()
        .sort_values(ascending=False).head(8).index.tolist()
    )

    # Puntos acumulados
    fig_evo = go.Figure()
    races   = df_season["race"].unique().tolist()

    for drv_name in top_drivers:
        drv_data = df_season[df_season["driver"] == drv_name].sort_values("round")
        cum_pts  = []
        acc      = 0.0
        race_labels = []
        for _, row in drv_data.iterrows():
            acc += row["points"]
            cum_pts.append(acc)
            race_labels.append(str(row["race"])[:12])

        # Color: intentar mapear por apellido
        color = "#888888"
        for drv_code in sesion.drivers:
            info = sesion.get_driver(drv_code)
            if info.get("LastName","").lower() in drv_name.lower():
                color = _get_driver_color(sesion, drv_code)
                break

        fig_evo.add_trace(go.Scatter(
            x=drv_data["round"].tolist(),
            y=cum_pts,
            mode="lines+markers",
            name=drv_name,
            line=dict(color=color, width=2),
            marker=dict(size=6, color=color),
            hovertemplate=f"<b>{drv_name}</b><br>R%{{x}}: %{{y}} pts<extra></extra>",
        ))

    _apply_dark(fig_evo, 400)
    fig_evo.update_layout(
        hovermode="x unified",
        legend=dict(orientation="h", y=1.04, x=0.5, xanchor="center",
                    bgcolor="rgba(0,0,0,0)", font=dict(size=9)),
        margin=dict(l=54, r=20, t=30, b=40),
        xaxis=dict(**_grid_axis(), title="Ronda", dtick=1),
        yaxis=dict(**_grid_axis(" pts"), title="Puntos acumulados"),
    )
    st.plotly_chart(fig_evo, use_container_width=True, config={"displayModeBar": False})


# ─────────────────────────────────────────────────────────────────────────────
# PUNTO DE ENTRADA PÚBLICO
# ─────────────────────────────────────────────────────────────────────────────

def render_session_compare(sesion: Session) -> None:
    """
    Punto de entrada principal del módulo SessionCompare.

    Organiza el análisis en 4 pestañas:
      · Vueltas: superposición de hasta 4 vueltas de telemetría.
      · Stints: evolución de pace, degradación, tabla de stints.
      · Equipos: radar de métricas, box plot de tiempos, ranking.
      · Campeonato: tabla de puntos, evolución de temporada vía Ergast API.

    Parámetros
    ──────────
    sesion : fastf1.core.Session ya cargada (con load_telemetry=True).
    """
    tab_laps, tab_stints, tab_teams, tab_champ = st.tabs([
        "📈 Vueltas",
        "🔄 Stints & Degradación",
        "🏎 Equipos",
        "🏆 Campeonato",
    ])

    with tab_laps:
        _render_tab_laps(sesion)

    with tab_stints:
        _render_tab_stints(sesion)

    with tab_teams:
        _render_tab_teams(sesion)

    with tab_champ:
        _render_tab_championship(sesion)