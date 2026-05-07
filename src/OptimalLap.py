"""
IdealLap.py — TALOS F1 Vuelta Ideal por Microsectores
=======================================================
Divide el circuito en N microsectores y colorea cada uno con el color del
equipo más veloz en él, construyendo la vuelta teórica ideal.

Conceptos clave
───────────────

  MICROSECTOR
    Segmento de distancia [d_inicio, d_fin] de longitud fija.
    Se calcula la velocidad media de cada piloto dentro del microsector
    usando su mejor vuelta (pick_fastest por piloto).
    El piloto con mayor velocidad media gana el microsector.

  ¿POR QUÉ VELOCIDAD MEDIA Y NO TIEMPO MÍNIMO?
    El tiempo en un microsector depende de cuántas muestras caen dentro
    del segmento, que varía según la frecuencia de muestreo. La velocidad
    media es robusta: si un piloto tiene pocas muestras en un segmento
    estrecho, la media sigue siendo representativa de su paso por ahí.
    Además, velocidad media ≈ longitud_sector / tiempo_sector, así que
    el piloto con mayor velocidad media es exactamente el que tarda menos.

  VUELTA IDEAL (TIEMPO TEÓRICO)
    Para cada microsector se toma el tiempo mínimo real registrado por
    cualquier piloto (filtrado). La suma de todos esos mínimos es el tiempo
    de la vuelta ideal teórica. Este tiempo es siempre inalcanzable en la
    práctica porque:
      · Los microsectores no son independientes (las curvas se encadenan).
      · No existe físicamente un coche que sea el más rápido en todos los
        sectores simultáneamente.
    Es un lower bound técnico, no una predicción realista.

  FILTRADO POR EQUIPOS
    Permite comparar solo un subconjunto de equipos. Útil para:
      · Comparativa intra-equipo (compañeros de equipo).
      · Batallas del midfield aisladas del top-3.
      · Análisis de fortalezas/debilidades relativas en distintas zonas.

Uso desde el enrutador:
    import IdealLap as il
    il.render_ideal_lap(st.session_state["f1_session"], corners=corners)

Conexión al router (appResearch.py):
    1. Import: import IdealLap as il
    2. Sustituir el placeholder "La Vuelta Ideal":
         elif active == "La Vuelta Ideal":
             if not require_session(): st.stop()
             st.header("La Vuelta Ideal · Microsectores")
             st.divider()
             il.render_ideal_lap(st.session_state["f1_session"], corners=corners)
"""


from __future__ import annotations
import ui_assets

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional
from fastf1.core import Session

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────────────────────────────────────────

BG_DARK    = "#0E0E0F"
BG_MAP     = "#13131A"
BG_PANEL   = "#111115"
BG_SURFACE = "#1A1A1F"
F1_WHITE   = "#FFFFFF"
F1_RED     = "#E8002D"
GREY_TRACK = "rgba(80,80,106,0.4)"
MONO_FONT  = "'JetBrains Mono', 'Courier New', monospace"

DEFAULT_N_SECTORS = 25
MAP_POINTS_PER_M  = 0.5


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


def _get_team_color(sesion: Session, driver_code: str) -> str:
    try:
        return f"#{sesion.get_driver(driver_code)['TeamColor']}"
    except Exception:
        return "#888888"


def _get_team_name(sesion: Session, driver_code: str) -> str:
    try:
        info = sesion.get_driver(driver_code)
        return str(info.get("TeamName", info.get("Team", "?")))
    except Exception:
        return "?"


def _get_full_name(sesion: Session, driver_code: str) -> str:
    try:
        info = sesion.get_driver(driver_code)
        return f"{info.get('FirstName','')} {info.get('LastName', driver_code)}".strip()
    except Exception:
        return driver_code


def _exterior_coords(
    x: np.ndarray, y: np.ndarray,
    mid_x: float, mid_y: float,
    idx: int, offset: float,
) -> tuple[float, float]:
    i_n = min(idx + 5, len(x) - 1)
    i_p = max(idx - 5, 0)
    dx, dy = x[i_n] - x[i_p], y[i_n] - y[i_p]
    nx, ny = -dy, dx
    mag = np.sqrt(nx**2 + ny**2) or 1.0
    nx /= mag; ny /= mag
    if (np.sqrt((x[idx]+nx*offset-mid_x)**2+(y[idx]+ny*offset-mid_y)**2) <
            np.sqrt((x[idx]-mid_x)**2+(y[idx]-mid_y)**2)):
        nx, ny = -nx, -ny
    return float(x[idx]+nx*offset), float(y[idx]+ny*offset)


# ─────────────────────────────────────────────────────────────────────────────
# EXTRACCIÓN DE TELEMETRÍA POR PILOTO
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=600, show_spinner=False)
def _load_best_laps_telemetry(
    _sesion: Session,
    driver_codes: tuple[str, ...],
) -> dict[str, pd.DataFrame]:
    result: dict[str, pd.DataFrame] = {}
    for drv in driver_codes:
        try:
            laps     = _sesion.laps.pick_drivers(drv).pick_accurate()
            if laps.empty:
                continue
            best_lap = laps.pick_fastest()
            tel      = best_lap.get_telemetry().add_distance()
            if tel.empty or "X" not in tel.columns:
                continue
            result[drv] = tel
        except Exception:
            continue
    return result


# ─────────────────────────────────────────────────────────────────────────────
# CÁLCULO DE MICROSECTORES
# ─────────────────────────────────────────────────────────────────────────────

def _compute_microsectors(
    tel_map: dict[str, pd.DataFrame],
    n_sectors: int,
) -> pd.DataFrame:
    if not tel_map:
        return pd.DataFrame()

    dist_max = min(float(tel["Distance"].max()) for tel in tel_map.values())
    edges    = np.linspace(0, dist_max, n_sectors + 1)

    rows = []
    for drv, tel in tel_map.items():
        dist  = tel["Distance"].values.astype(float)
        speed = tel["Speed"].values.astype(float)
        time  = tel["Time"].dt.total_seconds().values.astype(float)

        for s_idx in range(n_sectors):
            d0, d1 = edges[s_idx], edges[s_idx + 1]
            mask   = (dist >= d0) & (dist < d1)
            if mask.sum() < 2:
                continue

            spd_seg  = speed[mask]
            time_seg = time[mask]

            t_sector = float(time_seg[-1] - time_seg[0])
            if t_sector <= 0:
                continue

            rows.append({
                "sector"  : s_idx,
                "driver"  : drv,
                "v_mean"  : float(np.mean(spd_seg)),
                "t_sector": t_sector,
                "v_entry" : float(spd_seg[0]),
                "v_exit"  : float(spd_seg[-1]),
                "v_min"   : float(np.min(spd_seg)),
                "v_max"   : float(np.max(spd_seg)),
                "d_start" : d0,
                "d_end"   : d1,
            })

    return pd.DataFrame(rows)


def _find_sector_winners(df_sectors: pd.DataFrame) -> pd.DataFrame:
    if df_sectors.empty:
        return pd.DataFrame()
    idx_max = df_sectors.groupby("sector")["v_mean"].idxmax()
    winners = df_sectors.loc[idx_max].rename(columns={"driver": "winner_driver"})
    return winners.reset_index(drop=True).sort_values("sector")


# ─────────────────────────────────────────────────────────────────────────────
# MAPA DEL CIRCUITO COLOREADO POR MICROSECTORES
# ─────────────────────────────────────────────────────────────────────────────

def _build_microsector_map(
    ref_tel: pd.DataFrame,
    winners: pd.DataFrame,
    sesion: Session,
    circuit_info,
    show_corners: bool,
    n_sectors: int,
) -> go.Figure:
    x    = ref_tel["X"].values.astype(float)
    y    = ref_tel["Y"].values.astype(float)
    dist = ref_tel["Distance"].values.astype(float)

    mid_x = float(np.mean(x))
    mid_y = float(np.mean(y))

    range_xy = max(x.max()-x.min(), y.max()-y.min()) / 1.24
    x_lim    = [mid_x-range_xy, mid_x+range_xy]
    y_lim    = [mid_y-range_xy*1.4, mid_y+range_xy*0.7]

    fig = go.Figure()

    fig.add_trace(go.Scattergl(
        x=x, y=y,
        mode="lines",
        line=dict(color="rgba(80,80,106,0.35)", width=10),
        hoverinfo="skip",
        showlegend=False,
        name="_base",
    ))

    dist_max = float(dist.max())
    edges    = np.linspace(0, dist_max, n_sectors + 1)

    for _, row in winners.iterrows():
        s_idx  = int(row["sector"])
        d0, d1 = edges[s_idx], edges[s_idx + 1]
        mask   = (dist >= d0) & (dist <= d1)

        if mask.sum() < 2:
            continue

        winner_drv  = str(row["winner_driver"])
        team_color  = _get_team_color(sesion, winner_drv)
        team_name   = _get_team_name(sesion, winner_drv)
        full_name   = _get_full_name(sesion, winner_drv)
        v_mean      = float(row["v_mean"])
        t_sector    = float(row["t_sector"])

        hover = (
            f"<b>Microsector {s_idx + 1}</b><br>"
            f"Ganador: <b>{full_name}</b><br>"
            f"Equipo: {team_name}<br>"
            f"V media: {v_mean:.1f} km/h<br>"
            f"Tiempo: {t_sector:.3f}s<br>"
            f"Dist: {d0:.0f}–{d1:.0f} m"
        )

        next_mask = np.zeros_like(mask)
        idxs      = np.where(mask)[0]
        if len(idxs) and idxs[-1] + 1 < len(mask):
            next_mask[idxs[-1] + 1] = True
        paint_mask = mask | next_mask

        fig.add_trace(go.Scattergl(
            x=x[paint_mask],
            y=y[paint_mask],
            mode="lines",
            line=dict(color=team_color, width=6),
            hovertemplate=hover + "<extra></extra>",
            showlegend=False,
            name=f"_s{s_idx}",
        ))

    dx_m  = float(x[1]-x[0]); dy_m = float(y[1]-y[0])
    mag_m = np.sqrt(dx_m**2+dy_m**2) or 1.0
    nmx, nmy = -dy_m/mag_m, dx_m/mag_m
    meta_w = 200.0
    fig.add_trace(go.Scatter(
        x=[x[0]-nmx*meta_w, x[0]+nmx*meta_w],
        y=[y[0]-nmy*meta_w, y[0]+nmy*meta_w],
        mode="lines",
        line=dict(color=F1_WHITE, width=3),
        hoverinfo="skip", showlegend=False, name="_meta",
    ))

    if show_corners and circuit_info is not None:
        for _, row in circuit_info.corners.iterrows():
            d_c   = float(row["Distance"])
            if d_c > dist_max:
                continue
            idx_c = int(np.argmin(np.abs(dist - d_c)))
            cx, cy = _exterior_coords(x, y, mid_x, mid_y, idx_c, 240)
            fig.add_trace(go.Scatter(
                x=[cx], y=[cy],
                mode="markers+text",
                marker=dict(size=18, color=F1_WHITE, symbol="circle",
                            line=dict(color=F1_RED, width=1.5)),
                text=[str(int(row["Number"]))],
                textfont=dict(size=9, color="#0E0E0F", family=MONO_FONT),
                textposition="middle center",
                hoverinfo="skip", showlegend=False, name=f"_T{int(row['Number'])}",
            ))

    for s_idx in range(0, n_sectors, max(1, n_sectors // 10)):
        if s_idx >= len(winners):
            continue
        d_center = (edges[s_idx] + edges[s_idx + 1]) / 2
        idx_c    = int(np.argmin(np.abs(dist - d_center)))
        cx = float(x[idx_c]) + (mid_x - float(x[idx_c])) * 0.08
        cy = float(y[idx_c]) + (mid_y - float(y[idx_c])) * 0.08
        fig.add_annotation(
            x=cx, y=cy,
            text=f"<b>S{s_idx+1}</b>",
            showarrow=False,
            font=dict(size=8, color="rgba(255,255,255,0.45)", family=MONO_FONT),
            bgcolor="rgba(14,14,15,0.6)",
            borderpad=2,
        )

    fig.update_layout(
        height=600,
        paper_bgcolor=BG_DARK,
        plot_bgcolor=BG_MAP,
        font=dict(family=MONO_FONT, color=F1_WHITE, size=11),
        showlegend=False,
        margin=dict(l=10, r=10, t=20, b=10),
        hoverlabel=dict(
            bgcolor=BG_SURFACE, bordercolor="rgba(255,255,255,0.15)",
            font=dict(family=MONO_FONT, color=F1_WHITE, size=11),
        ),
        xaxis=dict(range=x_lim, scaleanchor="y", scaleratio=1,
                   showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(range=y_lim,
                   showgrid=False, zeroline=False, showticklabels=False),
    )

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# LEYENDA DE EQUIPOS
# ─────────────────────────────────────────────────────────────────────────────

def _build_team_legend(
    winners: pd.DataFrame,
    sesion: Session,
    n_sectors: int,
) -> go.Figure:
    if winners.empty:
        return go.Figure()

    team_wins: dict[str, dict] = {}
    for _, row in winners.iterrows():
        drv  = str(row["winner_driver"])
        team = _get_team_name(sesion, drv)
        col  = _get_team_color(sesion, drv)
        if team not in team_wins:
            team_wins[team] = {"count": 0, "color": col, "driver": drv}
        team_wins[team]["count"] += 1

    teams  = sorted(team_wins.keys(), key=lambda t: team_wins[t]["count"], reverse=True)
    counts = [team_wins[t]["count"] for t in teams]
    colors = [team_wins[t]["color"] for t in teams]
    pcts   = [f"{c/n_sectors*100:.0f}%" for c in counts]

    fig = go.Figure(go.Bar(
        x=counts,
        y=teams,
        orientation="h",
        marker=dict(color=colors, line=dict(color="rgba(0,0,0,0.3)", width=1)),
        text=[f"{c} sec  ({p})" for c, p in zip(counts, pcts)],
        textposition="outside",
        textfont=dict(size=10, color=F1_WHITE, family=MONO_FONT),
        hovertemplate="<b>%{y}</b><br>%{x} microsectores<extra></extra>",
    ))

    fig.update_layout(
        height=max(120, len(teams) * 36 + 40),
        paper_bgcolor=BG_DARK,
        plot_bgcolor=BG_PANEL,
        font=dict(family=MONO_FONT, color=F1_WHITE, size=10),
        margin=dict(l=10, r=80, t=10, b=10),
        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                   zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False,
                   tickfont=dict(size=10, color=F1_WHITE)),
        showlegend=False,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# TABLA DE MICROSECTORES (SIN IDENTACIÓN EN HTML)
# ─────────────────────────────────────────────────────────────────────────────

def _render_sector_table(
    winners: pd.DataFrame,
    sesion: Session,
    ideal_time: float,
    pole_time: float,
) -> None:
    st.markdown(
        "<p style='font-family: JetBrains Mono, monospace; font-size: 11px; "
        "letter-spacing: 2px; color: rgba(255,255,255,0.35); margin: 16px 0 8px;'>"
        "TABLA DE MICROSECTORES · VUELTA IDEAL</p>",
        unsafe_allow_html=True,
    )

    rows_html = ""
    for _, row in winners.iterrows():
        s_idx      = int(row["sector"]) + 1
        drv        = str(row["winner_driver"])
        team_color = _get_team_color(sesion, drv)
        team_name  = _get_team_name(sesion, drv)
        full_name  = _get_full_name(sesion, drv)
        v_mean     = float(row["v_mean"])
        t_sector   = float(row["t_sector"])
        v_entry    = float(row.get("v_entry", 0))
        v_min      = float(row.get("v_min", 0))
        v_exit     = float(row.get("v_exit", 0))
        d_start    = float(row.get("d_start", 0))
        d_end      = float(row.get("d_end", 0))

        # Importante: Pegado a la izquierda
        rows_html += f"""
<tr style="border-bottom:1px solid rgba(255,255,255,0.05);">
<td style="padding:6px 10px;color:rgba(255,255,255,0.4);font-size:10px;">S{s_idx}</td>
<td style="padding:6px 10px;">
<span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:{team_color};margin-right:8px;"></span>
<b style="color:{team_color};">{drv}</b>
<span style="color:rgba(255,255,255,0.5);font-size:10px;"> {full_name}</span>
</td>
<td style="padding:6px 10px;color:rgba(255,255,255,0.55);font-size:10px;">{team_name}</td>
<td style="padding:6px 10px;color:rgba(255,255,255,0.6);font-size:11px;text-align:right;">{v_entry:.0f}</td>
<td style="padding:6px 10px;color:#C77DFF;font-size:11px;text-align:right;">{v_min:.0f}</td>
<td style="padding:6px 10px;color:rgba(255,255,255,0.6);font-size:11px;text-align:right;">{v_exit:.0f}</td>
<td style="padding:6px 10px;color:#00D2FF;font-weight:700;text-align:right;">{v_mean:.1f}</td>
<td style="padding:6px 10px;color:#39FF14;font-weight:700;text-align:right;">{t_sector:.3f}s</td>
<td style="padding:6px 10px;color:rgba(255,255,255,0.3);font-size:9px;text-align:right;">{d_start:.0f}–{d_end:.0f} m</td>
</tr>
"""

    delta_vs_pole = ideal_time - pole_time if pole_time > 0 else 0
    delta_str     = f"−{abs(delta_vs_pole):.3f}s" if delta_vs_pole < 0 else f"+{delta_vs_pole:.3f}s"
    delta_color   = "#39FF14" if delta_vs_pole <= 0 else F1_RED

    # Importante: Pegado a la izquierda
    html_table = f"""
<div style="overflow-x:auto;overflow-y:auto;max-height:520px;">
<table style="width:100%;border-collapse:collapse;font-family:{MONO_FONT};font-size:12px;color:{F1_WHITE};background:{BG_MAP};">
<thead>
<tr style="border-bottom:2px solid rgba(255,255,255,0.12);color:rgba(255,255,255,0.35);font-size:9px;letter-spacing:1px;position:sticky;top:0;background:{BG_MAP};z-index:10;">
<th style="padding:8px 10px;text-align:left;">SEC</th>
<th style="padding:8px 10px;text-align:left;">PILOTO</th>
<th style="padding:8px 10px;text-align:left;">EQUIPO</th>
<th style="padding:8px 10px;text-align:right;">V ENTRADA</th>
<th style="padding:8px 10px;text-align:right;color:#C77DFF;">V ÁPEX</th>
<th style="padding:8px 10px;text-align:right;">V SALIDA</th>
<th style="padding:8px 10px;text-align:right;color:#00D2FF;">V MEDIA</th>
<th style="padding:8px 10px;text-align:right;color:#39FF14;">TIEMPO</th>
<th style="padding:8px 10px;text-align:right;">DISTANCIA</th>
</tr>
</thead>
<tbody>{rows_html}</tbody>
<tfoot>
<tr style="border-top:2px solid rgba(255,255,255,0.15);background:{BG_PANEL};">
<td colspan="7" style="padding:10px 10px;font-size:13px;font-weight:800;letter-spacing:2px;">VUELTA IDEAL TEÓRICA</td>
<td style="padding:10px 10px;color:#39FF14;font-size:14px;font-weight:800;text-align:right;">{_fmt_time(ideal_time)}</td>
<td style="padding:10px 10px;color:{delta_color};font-size:11px;text-align:right;font-weight:700;">{delta_str} vs pole</td>
</tr>
</tfoot>
</table>
</div>
"""
    st.markdown(html_table, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# GRÁFICO DE VELOCIDADES POR SECTOR 
# ─────────────────────────────────────────────────────────────────────────────

def _build_speed_profile(
    tel_map: dict[str, pd.DataFrame],
    winners: pd.DataFrame,
    sesion: Session,
    n_sectors: int,
) -> go.Figure:
    if tel_map is None or winners.empty:
        return go.Figure()

    dist_max_ref = min(float(tel["Distance"].max()) for tel in tel_map.values())
    edges        = np.linspace(0, dist_max_ref, n_sectors + 1)
    d_centers    = [(edges[i]+edges[i+1])/2 for i in range(n_sectors)]

    fig = go.Figure()

    for drv, tel in tel_map.items():
        dist  = tel["Distance"].values.astype(float)
        speed = tel["Speed"].values.astype(float)
        color = _get_team_color(sesion, drv)
        name  = _get_full_name(sesion, drv)

        v_by_sector = []
        for s_idx in range(n_sectors):
            d0, d1 = edges[s_idx], edges[s_idx+1]
            mask   = (dist >= d0) & (dist < d1)
            v_by_sector.append(float(np.mean(speed[mask])) if mask.sum() > 0 else np.nan)

        fig.add_trace(go.Scatter(
            x=list(range(1, n_sectors+1)),
            y=v_by_sector,
            mode="lines",
            line=dict(color=color, width=1.2),
            opacity=0.6,
            name=f"{drv}  {name}",
            hovertemplate=f"<b>{drv}</b>  S%{{x}}: %{{y:.1f}} km/h<extra></extra>",
        ))

    winner_x, winner_y, winner_colors, winner_hover = [], [], [], []
    for _, row in winners.iterrows():
        s_idx     = int(row["sector"])
        drv       = str(row["winner_driver"])
        winner_x.append(s_idx + 1)
        winner_y.append(float(row["v_mean"]))
        winner_colors.append(_get_team_color(sesion, drv))
        winner_hover.append(f"<b>S{s_idx+1} — {drv}</b><br>{row['v_mean']:.1f} km/h")

    fig.add_trace(go.Scatter(
        x=winner_x, y=winner_y,
        mode="markers",
        marker=dict(
            size=10, color=winner_colors, symbol="star",
            line=dict(color=F1_WHITE, width=1),
        ),
        text=winner_hover,
        hovertemplate="%{text}<extra>Ganador sector</extra>",
        showlegend=False,
        name="_winners",
    ))

    fig.update_layout(
        height=280,
        paper_bgcolor=BG_DARK,
        plot_bgcolor=BG_PANEL,
        font=dict(family=MONO_FONT, color=F1_WHITE, size=10),
        margin=dict(l=54, r=20, t=10, b=40),
        hovermode="x unified",
        hoverlabel=dict(bgcolor=BG_SURFACE, bordercolor="rgba(255,255,255,0.15)",
                        font=dict(family=MONO_FONT, color=F1_WHITE, size=10)),
        legend=dict(
            orientation="h", y=1.08, x=0.5, xanchor="center",
            bgcolor="rgba(0,0,0,0)", font=dict(size=9),
        ),
        xaxis=dict(
            title="Microsector",
            showgrid=True, gridcolor="rgba(255,255,255,0.05)",
            zeroline=False, dtick=1 if n_sectors <= 30 else 5,
            tickfont=dict(size=9, color="rgba(255,255,255,0.4)"),
        ),
        yaxis=dict(
            title="V media (km/h)",
            showgrid=True, gridcolor="rgba(255,255,255,0.05)",
            zeroline=False,
            tickfont=dict(size=9, color="rgba(255,255,255,0.4)"),
            ticksuffix=" km/h",
        ),
    )

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# PUNTO DE ENTRADA PÚBLICO
# ─────────────────────────────────────────────────────────────────────────────

def render_ideal_lap(sesion: Session, corners: bool = True) -> None:
    st.markdown(
        "<p style='font-family: JetBrains Mono, monospace; font-size: 11px; "
        "letter-spacing: 2px; color: rgba(255,255,255,0.35); margin-bottom: 8px;'>"
        "VUELTA IDEAL · MICROSECTORES POR EQUIPO</p>",
        unsafe_allow_html=True,
    )

    # ── 1. Controles ──────────────────────────────────────────────────────
    ctrl_left, ctrl_right = st.columns([2, 5])

    with ctrl_left:
        n_sectors = st.slider(
            "Microsectores",
            min_value=10, max_value=60,
            value=DEFAULT_N_SECTORS, step=5,
            help="Más sectores = granularidad mayor, pero más ruido en sectores cortos.",
            key="il_nsectors",
        )

    all_teams: dict[str, list[str]] = {}
    driver_names: dict[str, str]    = {}
    for drv in sesion.drivers:
        try:
            info = sesion.get_driver(drv)
            team = str(info.get("TeamName", info.get("Team", "?")))
            if team not in all_teams:
                all_teams[team] = []
            all_teams[team].append(str(drv))
            driver_names[str(drv)] = _get_full_name(sesion, drv)
        except Exception:
            continue

    with ctrl_right:
        selected_teams = st.multiselect(
            "Filtrar equipos (vacío = todos)",
            options=sorted(all_teams.keys()),
            default=[],
            key="il_teams",
            help="Selecciona equipos para comparar solo ese subconjunto.",
        )

    if selected_teams:
        active_drivers = [drv for team in selected_teams for drv in all_teams.get(team, [])]
    else:
        active_drivers = list(sesion.drivers)

    if not active_drivers:
        st.warning("No hay pilotos en el filtro seleccionado.")
        return

    st.divider()

    # ── 2. Carga y Cálculo ─────────────────────────────────────────────
    with st.spinner("Cargando telemetrías y calculando sectores..."):
        tel_map = _load_best_laps_telemetry(sesion, tuple(active_drivers))
        if not tel_map:
            st.error("❌ No se pudo cargar telemetría.")
            return

        df_sectors = _compute_microsectors(tel_map, n_sectors)
        winners    = _find_sector_winners(df_sectors)

        if winners.empty:
            st.error("❌ No hay datos suficientes.")
            return

    ideal_time = float(winners["t_sector"].sum())
    try:
        pole_time = float(sesion.laps.pick_accurate().pick_fastest()["LapTime"].total_seconds())
    except Exception:
        pole_time = 0.0

    try:
        circuit_info = sesion.get_circuit_info()
    except Exception:
        circuit_info = None

    ref_driver = list(tel_map.keys())[0]
    ref_tel    = tel_map[ref_driver]

    # ── 3. Panel de Insights (Subido aquí) ──────────────────────────────
    team_counts: dict[str, int] = {}
    for _, row in winners.iterrows():
        drv  = str(row["winner_driver"])
        team = _get_team_name(sesion, drv)
        team_counts[team] = team_counts.get(team, 0) + 1

    top_team  = max(team_counts, key=lambda t: team_counts[t])
    top_count = team_counts[top_team]
    pct_dom   = top_count / len(winners) * 100

    idx_slow = winners["v_min"].idxmin()
    s_slow   = int(winners.loc[idx_slow, "sector"]) + 1
    

    idx_fast = winners["v_mean"].idxmax()
    s_fast   = int(winners.loc[idx_fast, "sector"]) + 1

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Equipo dominante", top_team, f"{top_count} sec · {pct_dom:.0f}%")
    k2.metric("Sector más veloz", f"S{s_fast}", f"{winners.loc[idx_fast,'v_mean']:.0f} km/h")
    k3.metric("Sector más lento (Vértice)", f"S{s_slow}", f"{winners.loc[idx_slow,'v_min']:.0f} km/h")
    k4.metric("Vuelta ideal teórica", _fmt_time(ideal_time))

    st.divider()

    # ── 4. Mapa + leyenda ─────────────────────────────────────────────────
    col_map, col_legend = st.columns([3, 1])

    with col_map:
        st.markdown(
            "<p style='font-family: JetBrains Mono, monospace; font-size: 11px; "
            "letter-spacing: 2px; color: rgba(255,255,255,0.35); margin-bottom: 4px;'>"
            "CIRCUIT IDEAL MAP · COLOR = EQUIPO MÁS VELOZ</p>",
            unsafe_allow_html=True,
        )
        fig_map = _build_microsector_map(ref_tel, winners, sesion, circuit_info, corners, n_sectors)
        st.plotly_chart(fig_map, use_container_width=True, config={"displayModeBar": False})

    with col_legend:
        st.markdown(
            "<p style='font-family: JetBrains Mono, monospace; font-size: 11px; "
            "letter-spacing: 2px; color: rgba(255,255,255,0.35); margin-bottom: 4px;'>"
            "DOMINIO POR EQUIPO</p>",
            unsafe_allow_html=True,
        )
        fig_legend = _build_team_legend(winners, sesion, n_sectors)
        st.plotly_chart(fig_legend, use_container_width=True, config={"displayModeBar": False})

    # ── 5. Perfil de velocidad ──────────────────────────────────────────
    st.markdown(
        "<p style='font-family: JetBrains Mono, monospace; font-size: 11px; "
        "letter-spacing: 2px; color: rgba(255,255,255,0.35); margin: 8px 0 4px;'>"
        "PERFIL DE VELOCIDAD POR MICROSECTOR ★ = ganador del sector</p>",
        unsafe_allow_html=True,
    )
    fig_speed = _build_speed_profile(tel_map, winners, sesion, n_sectors)
    st.plotly_chart(fig_speed, use_container_width=True, config={"displayModeBar": False})

    # ── 6. Tabla (Con el Gap vs Pole en el footer) ────────────────────────
    _render_sector_table(winners, sesion, ideal_time, pole_time)
