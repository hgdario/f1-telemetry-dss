"""
DemandMap.py — TALOS F1 Circuit Demand Map
"""


from __future__ import annotations
import ui_assets
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Literal
from fastf1.core import Session
from scipy.signal import savgol_filter


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES DE DISEÑO — misma paleta que TelemetryTrace
# ─────────────────────────────────────────────────────────────────────────────
F1_WHITE      = "#FFFFFF"
F1_RED        = "#E8002D"
ACCENT_CYAN   = "#00D2FF"
ACCENT_AMBER  = "#FFA500"
ACCENT_GREEN  = "#39FF14"
ACCENT_PURPLE = "#C77DFF"
ACCENT_YELLOW = "#FFD600"

BG_DARK    = "#0E0E0F"
BG_SURFACE = "#1A1A1F"
BG_PANEL   = "#111115"
GRID_COLOR = "rgba(255,255,255,0.06)"

PLOTLY_FONT = dict(
    family="'JetBrains Mono', 'Courier New', monospace",
    color=F1_WHITE,
    size=11,
)

# Escalas de color para cada modo
# ── Lateral: divergente azul-blanco-rojo (izquierda ← neutro → derecha)
COLORSCALE_LATERAL = [
    [0.00, "#0044FF"],   # máx. izquierda  → azul
    [0.25, "#5599FF"],
    [0.50, "#FFFFFF"],   # neutro          → blanco
    [0.75, "#FF5533"],
    [1.00, "#E8002D"],   # máx. derecha    → rojo F1
]

# ── Longitudinal: divergente verde-blanco-ámbar (aceleración ← neutro → frenada)
COLORSCALE_LONGITUDINAL = [
    [0.00, "#39FF14"],   # máx. aceleración → verde neón
    [0.25, "#A8FF80"],
    [0.50, "#FFFFFF"],   # neutro
    [0.75, "#FFB347"],
    [1.00, "#FFA500"],   # máx. frenada     → ámbar
]

# ── Total: secuencial negro-púrpura-cian (magnitud de carga)
COLORSCALE_TOTAL = [
    [0.00, "#0E0E0F"],   # cero G           → negro
    [0.30, "#4B0082"],
    [0.60, "#C77DFF"],   # G media
    [0.85, "#00D2FF"],
    [1.00, "#FFFFFF"],   # máxima carga     → blanco
]

GMode = Literal["lateral", "longitudinal", "total"]


# ─────────────────────────────────────────────────────────────────────────────
# CÁLCULO DE FUERZAS G
# ─────────────────────────────────────────────────────────────────────────────

def _estimate_sampling_sigma(tel: pd.DataFrame) -> float:
    """
    Estima el sigma óptimo para el filtro Savitzky según la frecuencia
    de muestreo de la telemetría.

    FastF1 combina canales de distinta frecuencia (algunos a 10 Hz, otros a
    240 Hz) y los interpola. Calculamos la frecuencia media real y elegimos
    sigma para suavizar ~0.15 segundos de señal, lo que elimina el ruido de
    cuantización sin borrar los eventos reales de frenada o cambio de marcha.
    """
    try:
        dt_series = tel["Time"].dt.total_seconds().diff().dropna()
        dt_median = float(dt_series.median())
        if dt_median <= 0:
            return 5.0
        hz = 1.0 / dt_median
        # sigma en muestras para cubrir 0.15 s
        sigma = max(2.0, hz * 0.15)
        return round(sigma, 1)
    except Exception:
        return 5.0


def _calculate_g_forces(tel: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula las fuerzas G laterales, longitudinales y totales desde la telemetría.

    Requiere: Time, Speed y (para la G lateral) X, Y.
    Añade: g_lat (+ izquierda, − derecha), g_lon (+ acelera, − frena), g_total.

    Aplica directamente las fórmulas físicas:
        G_lon = (dv/dt) / g                  — derivada de la velocidad
        G_lat = (v · ω) / g,  ω = dθ/dt      — θ = arctan2(dY, dX)

    Las fórmulas son directas. Lo único que se añade es:
      · un suavizado Savitzky-Golay ligero (una derivada numérica de datos
        muestreados siempre necesita algo de suavizado),
      · anular la G lateral por debajo de 30 km/h, donde la dirección de marcha
        no es fiable (boxes, in/out laps) y v·ω se dispara,
      · un clip al límite físico (±6 G) como red de seguridad.
    """
    tel = tel.copy()
    n = len(tel)
    if n < 7 or not {"Speed", "Time"}.issubset(tel.columns):
        tel["g_lat"] = tel["g_lon"] = tel["g_total"] = np.nan
        return tel

    speed_ms = tel["Speed"].to_numpy(dtype=float) / 3.6
    time_s   = tel["Time"].dt.total_seconds().to_numpy(dtype=float)

    # Ventana de suavizado ~0.5 s (impar, > polyorder, acotada a la longitud)
    dt  = np.diff(time_s, prepend=time_s[0])
    win = int((1.0 / np.median(dt[dt > 0])) * 0.5) if np.any(dt > 0) else 11
    win = win + 1 if win % 2 == 0 else win
    win = max(5, min(win, n if n % 2 == 1 else n - 1))
    poly = 3 if win > 3 else 2

    # ── G longitudinal: dv/dt ───────────────────────────────────────────────
    g_lon = savgol_filter(np.gradient(speed_ms, time_s) / 9.81, win, poly)

    # ── G lateral: v · ω ────────────────────────────────────────────────────
    if "X" in tel.columns and "Y" in tel.columns:
        x = savgol_filter(tel["X"].to_numpy(dtype=float), win, poly)
        y = savgol_filter(tel["Y"].to_numpy(dtype=float), win, poly)
        heading = np.unwrap(np.arctan2(np.gradient(y, time_s),
                                       np.gradient(x, time_s)))
        omega = np.gradient(heading, time_s)
        g_lat = savgol_filter(speed_ms * omega / 9.81, win, poly)
        g_lat[speed_ms < 30.0 / 3.6] = 0.0      # heading no fiable a baja velocidad
    else:
        g_lat = np.zeros(n)

    g_lat = np.clip(np.nan_to_num(g_lat, nan=0.0), -6.0, 6.0)
    g_lon = np.clip(np.nan_to_num(g_lon, nan=0.0), -6.0, 6.0)

    tel["g_lat"]   = g_lat
    tel["g_lon"]   = g_lon
    tel["g_total"] = np.sqrt(g_lat**2 + g_lon**2)
    return tel

def _g_to_color(val: float, cmin: float, cmax: float, colorscale: list) -> str:
    """Interpola un valor dentro del colorscale de Plotly."""
    t = (val - cmin) / (cmax - cmin) if cmax > cmin else 0
    t = np.clip(t, 0, 1)
    
    for i in range(len(colorscale) - 1):
        t0, c0 = colorscale[i]
        t1, c1 = colorscale[i + 1]
        if t0 <= t <= t1:
            f = (t - t0) / (t1 - t0) if t1 > t0 else 0
            r0, g0, b0 = ui_assets.hex_to_rgb_tuple(c0)
            r1, g1, b1 = ui_assets.hex_to_rgb_tuple(c1)
            
            r = int(r0 + f * (r1 - r0))
            g = int(g0 + f * (g1 - g0))
            b = int(b0 + f * (b1 - b0))
            return f'rgb({r},{g},{b})'
            
    # Fallback al último color
    r, g, b = ui_assets.hex_to_rgb_tuple(colorscale[-1][1])
    return f'rgb({r},{g},{b})'

# ─────────────────────────────────────────────────────────────────────────────
# MAPA DE TRAZADO COLOREADO
# ─────────────────────────────────────────────────────────────────────────────

def _build_circuit_demand_map(
    tel: pd.DataFrame,
    mode: GMode = "total",
    team_color: str = F1_RED,
) -> go.Figure:
    
    if "X" not in tel.columns or "Y" not in tel.columns:
        fig = go.Figure()
        fig.add_annotation(
            text="❌ Coordenadas XY no disponibles en la telemetría",
            x=0.5, y=0.5, xref="paper", yref="paper",
            showarrow=False, font=dict(color=F1_WHITE, size=14)
        )
        _apply_dark_layout(fig, height=500)
        return fig

    x = tel["X"].values
    y = tel["Y"].values

    if mode == "lateral":
        g_values   = tel["g_lat"].values
        colorscale = COLORSCALE_LATERAL
        cbar_title = "G Lateral"
        g_abs_max  = max(abs(np.nanpercentile(g_values, 2)), abs(np.nanpercentile(g_values, 98)))
        cmin, cmax = -g_abs_max, g_abs_max
        hover_unit = "G lat"

    elif mode == "longitudinal":
        g_values   = tel["g_lon"].values
        colorscale = COLORSCALE_LONGITUDINAL
        cbar_title = "G Longitudinal"
        g_abs_max  = max(abs(np.nanpercentile(g_values, 2)), abs(np.nanpercentile(g_values, 98)))
        cmin, cmax = -g_abs_max, g_abs_max
        hover_unit = "G lon"

    else:  # "total"
        g_values   = tel["g_total"].values
        colorscale = COLORSCALE_TOTAL
        cbar_title = "G Total"
        cmin       = 0.0
        cmax       = float(np.nanpercentile(g_values, 98))
        hover_unit = "G total"

    fig = go.Figure()

    # ── Capa 1: contorno base del trazado (gris oscuro grueso, como SpeedHeatMap)
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode="lines",
        line=dict(color="#2A2A3A", width=10),
        hoverinfo="skip",
        showlegend=False,
    ))

    # ── Capa 2: Trazos por bandas de color (Líneas en vez de puntos)
    N_BANDAS = 30
    bins = np.linspace(cmin, cmax, N_BANDAS + 1)

    for i in range(N_BANDAS):
        v_mid = (bins[i] + bins[i + 1]) / 2
        color = _g_to_color(v_mid, cmin, cmax, colorscale)
        mask  = (g_values[:-1] >= bins[i]) & (g_values[:-1] <= bins[i + 1])
        
        if not mask.any():
            continue
            
        xs, ys, hover = [], [], []
        for idx in np.where(mask)[0]:
            xs    += [x[idx], x[idx + 1], None]
            ys    += [y[idx], y[idx + 1], None]
            hover += [f"<b>{g_values[idx]:.2f} {hover_unit}</b>", 
                      f"<b>{g_values[idx+1]:.2f} {hover_unit}</b>", None]
            
        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            mode="lines",
            line=dict(color=color, width=4),
            hovertext=hover,
            hoverinfo="text",
            showlegend=False,
        ))

    # ── Capa 3: Colorbar (Scatter invisible solo para mostrar la escala)
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode="markers",
        marker=dict(
            size=0,
            color=[cmin, cmax],
            colorscale=colorscale,
            cmin=cmin,
            cmax=cmax,
            showscale=True,
            colorbar=dict(
                title=dict(text=cbar_title, font=dict(size=10, color="rgba(255,255,255,0.6)")),
                tickfont=dict(size=9, color="rgba(255,255,255,0.5)", family="JetBrains Mono, monospace"),
                bgcolor="rgba(0,0,0,0)",
                bordercolor="rgba(255,255,255,0.1)",
                borderwidth=1,
                thickness=12,
                len=0.7,
                ticksuffix=" G",
            ),
        ),
        hoverinfo="skip",
        showlegend=False,
    ))

    # ── Añadir marcador de inicio/meta
    fig.add_trace(go.Scatter(
        x=[x[0]], y=[y[0]],
        mode="markers+text",
        marker=dict(size=12, color=team_color, symbol="diamond",
                    line=dict(color=F1_WHITE, width=1.5)),
        text=["S/F"],
        textposition="top center",
        textfont=dict(size=9, color=F1_WHITE, family="JetBrains Mono, monospace"),
        hoverinfo="skip",
        showlegend=False,
        name="start",
    ))

    # ── Añadir curvas
    try:
        circuit_info = tel.attrs.get("circuit_info", None)
        if circuit_info is not None:
            _annotate_corners_on_map(fig, tel, circuit_info)
    except Exception:
        pass

    _apply_dark_layout(fig, height=560)
    fig.update_layout(
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, scaleanchor="y", scaleratio=1),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=20, r=20, t=30, b=20),
    )

    return fig


def _annotate_corners_on_map(
    fig: go.Figure,
    tel: pd.DataFrame,
    circuit_info,
) -> None:
    """
    Añade etiquetas de curva (T1, T2…) al mapa.
    """
    if "Distance" not in tel.columns:
        return

    dist = tel["Distance"].values
    x    = tel["X"].values
    y    = tel["Y"].values

    for _, corner in circuit_info.corners.iterrows():
        d_corner = corner["Distance"]
        if d_corner < dist.min() or d_corner > dist.max():
            continue

        cx = float(np.interp(d_corner, dist, x))
        cy = float(np.interp(d_corner, dist, y))

        fig.add_annotation(
            x=cx, y=cy,
            text=f"T{corner['Number']}",
            showarrow=False,
            font=dict(
                family="JetBrains Mono, monospace",
                size=8,
                color="rgba(255,255,255,0.45)",
            ),
            bgcolor="rgba(14,14,15,0.7)",
            borderpad=2,
        )


# ─────────────────────────────────────────────────────────────────────────────
# GRÁFICO DE DISTRIBUCIÓN DE FUERZAS G
# ─────────────────────────────────────────────────────────────────────────────

def _build_g_distribution(tel: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    if "g_lat" not in tel.columns or "g_lon" not in tel.columns:
        return fig

    g_lat = tel["g_lat"].values
    g_lon = tel["g_lon"].values
    
    # Aseguramos que tenemos g_total
    g_total = tel["g_total"].values if "g_total" in tel.columns else np.sqrt(g_lat**2 + g_lon**2)

    # ── 1. FILTRADO GLOBAL DE OUTLIERS (La "Roomba" de la FIA) ─────────────
    # Descartamos los glitches del GPS: el 1% más extremo de la sesión y
    # cualquier punto que toque el límite del clip (>= 5.9 G), que son ruido
    # de la doble derivada de posición, no cargas reales.
    p99_global = np.nanpercentile(g_total, 99.0)
    valid_mask = (g_total < p99_global) & (g_total < 5.9)

    g_lat_clean = g_lat[valid_mask]
    g_lon_clean = g_lon[valid_mask]
    g_total_clean = g_total[valid_mask]

    # ── 2. NUBE DE PUNTOS DE TELEMETRÍA (El piloto) ────────────────────────
    fig.add_trace(go.Scattergl(
        x=g_lat_clean, 
        y=g_lon_clean, 
        mode="markers",
        marker=dict(
            size=4.5,
            color=g_total_clean,
            colorscale=COLORSCALE_TOTAL, # Asume que está definida en tu config
            opacity=0.85,
            showscale=False,
        ),
        hovertemplate="<b>G lat: %{x:.2f}</b><br>G lon: %{y:.2f}<extra></extra>",
        showlegend=False,
        name="Telemetría",
    ))

    # ── 3. LÍMITE DE ADHERENCIA — ENVOLVENTE CONVEXA ───────────────────────
    # La envolvente convexa (convex hull) encierra TODOS los puntos limpios con
    # el polígono convexo de menor área. Representa el límite de agarre que el
    # piloto llegó a exprimir; ningún punto queda fuera.
    pts = np.column_stack((g_lat_clean, g_lon_clean))
    if len(pts) > 10:
        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(pts)
            verts = np.append(hull.vertices, hull.vertices[0])  # cerrar el polígono
            fig.add_trace(go.Scattergl(
                x=pts[verts, 0],
                y=pts[verts, 1],
                mode="lines",
                line=dict(color="rgba(255,255,255,0.8)", width=2),
                fill="toself",
                fillcolor="rgba(255,255,255,0.04)",
                name="Límite de adherencia",
                hoverinfo="skip",
                showlegend=False,
            ))
        except Exception:
            pass

    # ── 4. FORMATO DEL GRÁFICO ─────────────────────────────────────────────
    # Rango fijo a la nube real (con un pequeño margen) para que los pocos
    # puntos extremos no compriman la visualización.
    if len(g_lat_clean) > 0:
        lim = max(np.nanpercentile(np.abs(g_lat_clean), 99.5),
                  np.nanpercentile(np.abs(g_lon_clean), 99.5)) * 1.15
        lim = float(np.clip(lim, 3.0, 5.0))
    else:
        lim = 4.0

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            title=dict(text="G Lateral", font=dict(color="rgba(255,255,255,0.6)")),
            range=[-lim, lim],
            zeroline=True, zerolinecolor="rgba(255,255,255,0.3)", zerolinewidth=1.5,
            showgrid=True, gridcolor="rgba(255,255,255,0.05)",
            scaleanchor="y", scaleratio=1
        ),
        yaxis=dict(
            title=dict(text="G Longitudinal", font=dict(color="rgba(255,255,255,0.6)")),
            range=[-lim, lim],
            zeroline=True, zerolinecolor="rgba(255,255,255,0.3)", zerolinewidth=1.5,
            showgrid=True, gridcolor="rgba(255,255,255,0.05)"
        ),
        margin=dict(l=20, r=20, t=30, b=20)
    )

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# GRÁFICO DE G VS DISTANCIA
# ─────────────────────────────────────────────────────────────────────────────

def _build_g_vs_distance(tel: pd.DataFrame, team_color: str = F1_RED) -> go.Figure:
    """
    Gráfico de línea de G lateral, longitudinal y total a lo largo de la vuelta,
    compartiendo eje X de distancia con el TelemetryTrace para comparación directa.

    Permite identificar exactamente en qué metro del circuito se produce cada
    pico de carga, relacionándolo con los puntos de frenada, ápex y salida de curva.
    """
    dist = tel["Distance"].values

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=["G LATERAL", "G LONGITUDINAL", "G TOTAL"],
        row_heights=[0.35, 0.35, 0.30],
    )

    # ── G Lateral ─────────────────────────────────────────────────────────
    g_lat = tel["g_lat"].values
    fig.add_trace(go.Scatter(
        x=dist, y=g_lat,
        mode="lines",
        line=dict(color=ACCENT_CYAN, width=1.5),
        fill="tozeroy",
        fillcolor="rgba(0,210,255,0.06)",
        hovertemplate="<b>%{y:.2f} G lat</b><extra></extra>",
        showlegend=False,
    ), row=1, col=1)

    # Línea de cero referencia
    fig.add_hline(y=0, line_width=1, line_color="rgba(255,255,255,0.15)", row=1, col=1)

    # ── G Longitudinal ────────────────────────────────────────────────────
    g_lon = tel["g_lon"].values
    # Área positiva (aceleración) en verde, negativa (frenada) en ámbar
    g_lon_pos = np.where(g_lon >= 0, g_lon, 0)
    g_lon_neg = np.where(g_lon <  0, g_lon, 0)

    fig.add_trace(go.Scatter(
        x=dist, y=g_lon_pos,
        mode="lines",
        line=dict(color=ACCENT_GREEN, width=1),
        fill="tozeroy",
        fillcolor="rgba(57,255,20,0.10)",
        name="Aceleración",
        hovertemplate="<b>+%{y:.2f} G lon</b><extra>Aceleración</extra>",
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=dist, y=g_lon_neg,
        mode="lines",
        line=dict(color=ACCENT_AMBER, width=1),
        fill="tozeroy",
        fillcolor="rgba(255,165,0,0.12)",
        name="Frenada",
        hovertemplate="<b>%{y:.2f} G lon</b><extra>Frenada</extra>",
    ), row=2, col=1)

    fig.add_hline(y=0, line_width=1, line_color="rgba(255,255,255,0.15)", row=2, col=1)

    # ── G Total ───────────────────────────────────────────────────────────
    g_tot = tel["g_total"].values
    fig.add_trace(go.Scatter(
        x=dist, y=g_tot,
        mode="lines",
        line=dict(color=team_color, width=1.5),
        fill="tozeroy",
        fillcolor=f"rgba({ui_assets.hex_rgb(team_color)},0.10)",
        hovertemplate="<b>%{y:.2f} G total</b><extra></extra>",
        showlegend=False,
    ), row=3, col=1)

    # Marcar picos de G total (top 3)
    _add_peak_annotations(fig, dist, g_tot, row=3)

    # ── Líneas verticales marcando las curvas (sin etiquetas) ─────────────
    try:
        circuit_info = tel.attrs.get("circuit_info", None)
        if circuit_info is not None and hasattr(circuit_info, "corners"):
            for _, corner in circuit_info.corners.iterrows():
                fig.add_vline(
                    x=corner['Distance'],
                    line_width=1,
                    line_dash="dash",
                    line_color="rgba(255, 255, 255, 0.2)",
                    layer="below",
                )
    except Exception:
        pass

    _apply_dark_layout(fig, height=480)
    fig.update_layout(
        hovermode="x unified",
        legend=dict(
            orientation="h", y=1.02, x=1, xanchor="right",
            bgcolor="rgba(0,0,0,0)", font=dict(size=9),
        ),
        margin=dict(l=54, r=24, t=40, b=40),
    )

    axis_common = dict(
        showgrid=True, gridcolor=GRID_COLOR, zeroline=False,
        tickfont=dict(family="JetBrains Mono, monospace", size=9,
                      color="rgba(255,255,255,0.5)"),
    )

    for i in range(1, 4):
        suffix = "G" if i < 4 else ""
        ykey   = "yaxis" if i == 1 else f"yaxis{i}"
        xkey   = "xaxis" if i == 1 else f"xaxis{i}"
        fig.update_layout(**{
            ykey: {**axis_common, "ticksuffix": " G"},
            xkey: {
                **axis_common,
                "title": "Distancia (m)" if i == 3 else "",
                "ticksuffix": " m" if i == 3 else "",
            },
        })

    for annotation in fig.layout.annotations:
        annotation.update(
            font=dict(family="JetBrains Mono, monospace", size=10,
                      color="rgba(255,255,255,0.45)"),
            x=0, xanchor="left",
        )

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# MÉTRICAS KPI
# ─────────────────────────────────────────────────────────────────────────────

def _render_g_metrics(tel: pd.DataFrame, team_color: str) -> None:
    """Panel de KPIs de fuerzas G en 4 columnas."""
    g_lat   = tel["g_lat"].dropna()
    g_lon   = tel["g_lon"].dropna()
    g_total = tel["g_total"].dropna()

    lat_max = g_lat.abs().max() if not g_lat.empty else 0
    brk_max = (-g_lon).clip(0).max() if not g_lon.empty else 0
    acc_max = g_lon.clip(0).max() if not g_lon.empty else 0
    tot_max = g_total.max() if not g_total.empty else 0

    st.markdown(
        f"""
        <div style="
            border-bottom: 2px solid {team_color};
            margin-bottom: 16px;
            padding-bottom: 6px;
        ">
            <span style="
                font-family: 'JetBrains Mono', monospace;
                font-size: 13px;
                color: rgba(255,255,255,0.45);
                letter-spacing: 3px;
            ">DEMAND MAP · FUERZAS G</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("↔ G LAT MÁX",  f"{lat_max:.2f} G", help="Pico de G lateral absoluto de la vuelta")
    c2.metric("⬇ G FRENADA",  f"{brk_max:.2f} G", help="Pico de deceleración (G longitudinal negativo)")
    c3.metric("⬆ G ACELERAC", f"{acc_max:.2f} G", help="Pico de aceleración (G longitudinal positivo)")
    c4.metric("⭕ G TOTAL",    f"{tot_max:.2f} G", help="Pico de fuerza resultante (vector lat+lon)")


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS DE LAYOUT
# ─────────────────────────────────────────────────────────────────────────────

def _apply_dark_layout(fig: go.Figure, height: int = 500) -> None:
    """Aplica el tema oscuro F1 a cualquier figura Plotly."""
    fig.update_layout(
        height=height,
        paper_bgcolor=BG_DARK,
        plot_bgcolor=BG_PANEL,
        font=PLOTLY_FONT,
        hoverlabel=dict(
            bgcolor=BG_SURFACE,
            bordercolor="rgba(255,255,255,0.15)",
            font=dict(family="JetBrains Mono, monospace", color=F1_WHITE, size=11),
        ),
    )


def _add_peak_annotations(
    fig: go.Figure,
    dist: np.ndarray,
    values: np.ndarray,
    row: int,
    n_peaks: int = 3,
) -> None:
    """Marca los n_peaks picos más altos de una serie en el gráfico."""
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(values, distance=50, prominence=0.3)
    if len(peaks) == 0:
        return
    top_peaks = sorted(peaks, key=lambda i: values[i], reverse=True)[:n_peaks]
    for idx in top_peaks:
        fig.add_annotation(
            x=float(dist[idx]),
            y=float(values[idx]),
            text=f"{values[idx]:.1f}G",
            showarrow=True,
            arrowhead=2,
            arrowcolor=F1_RED,
            font=dict(color=F1_RED, size=9, family="JetBrains Mono, monospace"),
            bgcolor=BG_SURFACE,
            bordercolor=F1_RED,
            borderwidth=1,
            row=row, col=1,
            ax=0, ay=-25,
        )


def _attach_circuit_info(tel: pd.DataFrame, session: Session) -> pd.DataFrame:
    """
    Intenta adjuntar el circuit_info al DataFrame como atributo,
    para que _build_circuit_demand_map pueda anotar las curvas.
    """
    try:
        ci = session.get_circuit_info()
        tel.attrs["circuit_info"] = ci
    except Exception:
        pass
    return tel


# ─────────────────────────────────────────────────────────────────────────────
# PUNTO DE ENTRADA PÚBLICO
# ─────────────────────────────────────────────────────────────────────────────

def render_demand_map(
    session: Session,
    driver: str,
    lap_number: int,
) -> None:

    with st.spinner(f"Calculando fuerzas G — {driver} · Vuelta {lap_number}"):
        # ── 1. Extraer telemetría ─────────────────────────────────────────
        try:
            laps = session.laps.pick_drivers(driver)
            lap  = laps[laps["LapNumber"] == lap_number].iloc[0]
            tel  = lap.get_telemetry().add_distance()
        except Exception as e:
            st.error(f"❌ No se pudo cargar la telemetría: {e}")
            return

        if tel.empty or "X" not in tel.columns:
            st.error(
                "❌ La telemetría no contiene coordenadas XY. "
                "Verifica que la sesión fue cargada con `load_telemetry=True`."
            )
            return

        # ── 2. Calcular fuerzas G ─────────────────────────────────────────
        tel = _calculate_g_forces(tel)
        tel = _attach_circuit_info(tel, session)

        # ── 3. Color de equipo ─────────────────────────────────────────────
        try:
            team_color = "#" + session.results.loc[
                session.results["Abbreviation"] == driver, "TeamColor"
            ].iloc[0]
        except Exception:
            team_color = F1_RED

    # ── 4. KPIs ───────────────────────────────────────────────────────────
    _render_g_metrics(tel, team_color)

    st.divider()

    # ── 5. Selector de modo del mapa ──────────────────────────────────────
    st.markdown(
        "<p style='font-family: JetBrains Mono, monospace; font-size: 11px; "
        "letter-spacing: 2px; color: rgba(255,255,255,0.35); margin-bottom: 8px;'>"
        "MAPA DE EXIGENCIA · MODO DE VISUALIZACIÓN</p>",
        unsafe_allow_html=True,
    )

    mode_labels = {
        "G Total (magnitud)"   : "total",
        "G Lateral (firmado)"  : "lateral",
        "G Longitudinal (firmado)": "longitudinal",
    }
    selected_label = st.radio(
        label="Modo",
        options=list(mode_labels.keys()),
        horizontal=True,
        label_visibility="collapsed",
    )
    mode: GMode = mode_labels[selected_label]

    # ── 6. Mapa del trazado ───────────────────────────────────────────────
    st.markdown(
        "<p style='font-family: JetBrains Mono, monospace; font-size: 11px; "
        "letter-spacing: 2px; color: rgba(255,255,255,0.35); margin-bottom: 4px;'>"
        "CIRCUIT DEMAND MAP</p>",
        unsafe_allow_html=True,
    )
    fig_map = _build_circuit_demand_map(tel, mode=mode, team_color=team_color)
    st.plotly_chart(fig_map, use_container_width=True, config={
        "displayModeBar"         : True,
        "displaylogo"            : False,
        "modeBarButtonsToRemove" : ["select2d", "lasso2d"],
        "toImageButtonOptions"   : {
            "format"  : "png",
            "filename": f"talos_demand_{driver}_lap{lap_number}",
            "scale"   : 2,
        },
    })

    st.divider()

    # ── 7. G vs Distancia ─────────────────────────────────────────────────
    st.markdown(
        "<p style='font-family: JetBrains Mono, monospace; font-size: 11px; "
        "letter-spacing: 2px; color: rgba(255,255,255,0.35); margin-bottom: 4px;'>"
        "FUERZAS G · DISTANCIA vs CANALES</p>",
        unsafe_allow_html=True,
    )
    fig_dist = _build_g_vs_distance(tel, team_color=team_color)
    st.plotly_chart(fig_dist, use_container_width=True, config={"displayModeBar": False})

    st.divider()

    # ── 8. Diagrama G-G ───────────────────────────────────────────────────
    st.markdown(
        "<p style='font-family: JetBrains Mono, monospace; font-size: 11px; "
        "letter-spacing: 2px; color: rgba(255,255,255,0.35); margin-bottom: 4px;'>"
        "DIAGRAMA G-G · HUELLA DE USO DEL GRIP</p>",
        unsafe_allow_html=True,
    )
    fig_gg = _build_g_distribution(tel)
    st.plotly_chart(fig_gg, use_container_width=True, config={"displayModeBar": False})
