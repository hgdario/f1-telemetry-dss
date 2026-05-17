from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from fastf1.core import Session


BG_DARK   = "#0E0E0F"
BG_PANEL  = "#111115"
F1_RED    = "#E8002D"
F1_WHITE  = "#FFFFFF"
MONO_FONT = "'JetBrains Mono', monospace"

TYRE_COLORS = {
    "SOFT": "#E8002D", "MEDIUM": "#FFF200", "HARD": "#EBEBEB",
    "INTER": "#43B02A", "WET": "#0067FF", "UNKNOWN": "#888888",
}


def _fmt_time(td) -> str:
    if td is None or pd.isna(td):
        return "—"
    try:
        s = td.total_seconds()
        return f"{int(s // 60)}:{s % 60:06.3f}"
    except Exception:
        return "—"


def _fmt_delta(td) -> str:
    if td is None or pd.isna(td):
        return "—"
    try:
        s = td.total_seconds()
        sign = "+" if s >= 0 else "−"
        return f"{sign}{abs(s):.3f}s"
    except Exception:
        return "—"


def _team_color(session: Session, driver: str) -> str:
    try:
        c = session.get_driver(driver).get("TeamColor", "")
        return f"#{c}" if c else "#888888"
    except Exception:
        return "#888888"


def render_driver_report(session: Session, driver: str) -> None:
    laps = session.laps.pick_drivers(driver)
    if laps.empty:
        st.warning("Sin datos de vueltas para este piloto.")
        return

    info    = session.get_driver(driver)
    first   = info.get("FirstName", "")
    last    = info.get("LastName", driver)
    team    = info.get("TeamName", "—")
    number  = info.get("DriverNumber", driver)
    abbr    = info.get("Abbreviation", driver)
    headshot = info.get("HeadshotUrl", "")
    color   = _team_color(session, driver)

    # ── Header ────────────────────────────────────────────────────────────
    col_photo, col_info = st.columns([1, 4], gap="medium")
    with col_photo:
        if headshot:
            st.image(headshot, width=160)
        else:
            st.markdown(
                f"<div style='font-size:48px;text-align:center;color:{color};"
                f"font-weight:900;padding:20px 0;'>{abbr}</div>",
                unsafe_allow_html=True,
            )
    with col_info:
        st.markdown(
            f"""
            <div style="border-left:4px solid {color};padding:10px 16px;
                        background:rgba(255,255,255,0.02);margin-bottom:18px;height:100%;">
              <div style="display:flex;align-items:center;gap:14px;">
                <span style="background:{color};color:#fff;padding:4px 10px;
                             font-family:{MONO_FONT};font-size:14px;font-weight:800;">
                  {number}
                </span>
                <span style="font-size:22px;font-weight:800;color:{F1_WHITE};">
                  {first} <span style="color:{color};">{last.upper()}</span>
                </span>
                <span style="font-size:12px;color:rgba(255,255,255,0.5);
                             letter-spacing:2px;text-transform:uppercase;
                             margin-left:auto;">{team}</span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── KPIs cabecera ─────────────────────────────────────────────────────
    pos = pts = gap_to_leader = "—"
    try:
        res = session.results
        if res is not None and not res.empty:
            row = None
            if "Abbreviation" in res.columns:
                mask = res["Abbreviation"] == abbr
                if mask.any():
                    row = res[mask].iloc[0]
            if row is None and "DriverNumber" in res.columns:
                mask = res["DriverNumber"].astype(str) == str(number)
                if mask.any():
                    row = res[mask].iloc[0]
            if row is not None:
                pos_val = row.get("Position")
                pos = int(pos_val) if pos_val is not None and pd.notna(pos_val) else "—"
                pts_val = row.get("Points")
                pts = int(pts_val) if pts_val is not None and pd.notna(pts_val) else 0
                my_time = row.get("Time")
                if my_time is not None and pd.notna(my_time):
                    gap_to_leader = _fmt_delta(my_time)
    except Exception:
        pass

    best_lap = laps.pick_fastest()
    best_lt  = best_lap["LapTime"] if best_lap is not None else None

    try:
        tel = best_lap.get_telemetry() if best_lap is not None else None
        top_speed = float(tel["Speed"].max()) if tel is not None and not tel.empty else np.nan
    except Exception:
        top_speed = np.nan

    n_laps = int(laps["LapNumber"].max())

    cluster_label = "—"
    cluster_color = "#888888"
    try:
        from DriverStyleClassifier import classify_driver, CLUSTER_COLORS as _CC
        cls = classify_driver(abbr)
        if cls is not None:
            cluster_label = cls["label"]
            cluster_color = _CC.get(cls["cluster_id"], "#888888")
    except Exception:
        pass

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Posición", f"P{pos}" if pos != "—" else "—")
    c2.metric("Puntos", pts)
    c3.metric("Mejor vuelta", _fmt_time(best_lt))
    c4.metric("Vmax", f"{top_speed:.0f} km/h" if not np.isnan(top_speed) else "—")
    c5.metric("Vueltas", n_laps)
    c6.metric("Estilo", cluster_label)

    if gap_to_leader != "—" and pos != 1:
        st.caption(f"Gap al líder: {gap_to_leader}")

    st.divider()

    # ── Mejores sectores: piloto vs absoluto del field ────────────────────
    st.markdown(
        "<p style='font-size:11px;letter-spacing:2px;color:rgba(255,255,255,0.5);"
        "margin:0 0 8px;'>MEJORES SECTORES</p>",
        unsafe_allow_html=True,
    )

    all_laps = session.laps.pick_accurate()
    sec_cols = st.columns(3)
    for i, sec in enumerate(["Sector1Time", "Sector2Time", "Sector3Time"]):
        if sec not in laps.columns:
            continue
        my_best = laps[sec].min()
        field_best = all_laps[sec].min() if sec in all_laps.columns else None
        with sec_cols[i]:
            if pd.notna(my_best) and field_best is not None and pd.notna(field_best):
                delta = (my_best - field_best).total_seconds()
                label = "Mejor del field" if delta == 0 else f"+{delta:.3f}s vs mejor"
            else:
                label = "—"
            st.metric(f"S{i + 1}", _fmt_time(my_best), delta=label, delta_color="off")

    st.divider()

    # ── Stints ────────────────────────────────────────────────────────────
    st.markdown(
        "<p style='font-size:11px;letter-spacing:2px;color:rgba(255,255,255,0.5);"
        "margin:0 0 8px;'>STINTS</p>",
        unsafe_allow_html=True,
    )

    if "Stint" in laps.columns and "Compound" in laps.columns:
        stints = (
            laps.groupby(["Stint", "Compound"])
                .agg(Inicio=("LapNumber", "min"),
                     Final=("LapNumber", "max"),
                     Vueltas=("LapNumber", "count"),
                     Mejor=("LapTime", "min"),
                     Media=("LapTime", "mean"))
                .reset_index()
                .sort_values("Stint")
        )
        for _, s in stints.iterrows():
            comp = str(s["Compound"]) if pd.notna(s["Compound"]) else "UNKNOWN"
            ccol = TYRE_COLORS.get(comp, TYRE_COLORS["UNKNOWN"])
            st.markdown(
                f"""
                <div style="display:flex;align-items:center;gap:14px;
                            background:{BG_PANEL};border-left:3px solid {ccol};
                            padding:10px 14px;margin-bottom:6px;border-radius:2px;
                            font-family:{MONO_FONT};">
                  <span style="background:{ccol};color:#15151E;padding:2px 8px;
                               font-weight:800;font-size:11px;letter-spacing:1px;">
                    {comp}
                  </span>
                  <span style="color:#aaa;font-size:11px;">STINT {int(s['Stint'])}</span>
                  <span style="color:#fff;font-size:12px;">
                    L{int(s['Inicio'])}–L{int(s['Final'])} · {int(s['Vueltas'])} vueltas
                  </span>
                  <span style="color:#aaa;font-size:11px;margin-left:auto;">MEJOR</span>
                  <span style="color:#fff;font-size:12px;font-weight:700;">
                    {_fmt_time(s['Mejor'])}
                  </span>
                  <span style="color:#aaa;font-size:11px;">MEDIA</span>
                  <span style="color:#fff;font-size:12px;">
                    {_fmt_time(s['Media'])}
                  </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.divider()

    # ── Distribución de lap times por compuesto ──────────────────────────
    st.markdown(
        "<p style='font-size:11px;letter-spacing:2px;color:rgba(255,255,255,0.5);"
        "margin:0 0 8px;'>DISTRIBUCIÓN DE TIEMPOS POR COMPUESTO</p>",
        unsafe_allow_html=True,
    )

    valid = laps.pick_accurate()
    if not valid.empty and "Compound" in valid.columns:
        fig = go.Figure()
        for comp in valid["Compound"].dropna().unique():
            sub = valid[valid["Compound"] == comp]
            times_s = sub["LapTime"].dt.total_seconds().dropna()
            if times_s.empty:
                continue
            fig.add_trace(go.Box(
                y=times_s,
                name=str(comp),
                marker_color=TYRE_COLORS.get(str(comp), "#888"),
                line=dict(color=TYRE_COLORS.get(str(comp), "#888")),
                boxmean=True,
            ))
        fig.update_layout(
            height=320,
            paper_bgcolor=BG_DARK, plot_bgcolor=BG_PANEL,
            font=dict(family=MONO_FONT, color=F1_WHITE, size=10),
            yaxis=dict(title="Tiempo (s)", gridcolor="rgba(255,255,255,0.06)"),
            margin=dict(l=50, r=20, t=20, b=30),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    else:
        st.caption("Sin vueltas válidas para construir la distribución.")

    # ── Posición lap-by-lap (sólo en carreras) ────────────────────────────
    ses_name = getattr(session, "name", "") or ""
    is_race = "Race" in ses_name and "Sprint" not in ses_name
    if is_race and "Position" in laps.columns:
        st.divider()
        st.markdown(
            "<p style='font-size:11px;letter-spacing:2px;color:rgba(255,255,255,0.5);"
            "margin:0 0 8px;'>EVOLUCIÓN DE POSICIÓN</p>",
            unsafe_allow_html=True,
        )
        pos_data = (
            laps[["LapNumber", "Position"]]
                .dropna()
                .sort_values("LapNumber")
        )
        if not pos_data.empty:
            fig = go.Figure(go.Scatter(
                x=pos_data["LapNumber"], y=pos_data["Position"],
                mode="lines+markers",
                line=dict(color=color, width=2),
                marker=dict(size=5, color=color),
                hovertemplate="L%{x} · P%{y}<extra></extra>",
            ))
            fig.update_layout(
                height=300,
                paper_bgcolor=BG_DARK, plot_bgcolor=BG_PANEL,
                font=dict(family=MONO_FONT, color=F1_WHITE, size=10),
                xaxis=dict(title="Vuelta", gridcolor="rgba(255,255,255,0.06)"),
                yaxis=dict(title="Posición", autorange="reversed",
                           dtick=1, gridcolor="rgba(255,255,255,0.06)"),
                margin=dict(l=50, r=20, t=20, b=40),
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
