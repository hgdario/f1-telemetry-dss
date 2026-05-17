"""
Aero.py — TALOS F1 · Estimación de Carga Aerodinámica
======================================================

Estima la influencia del downforce sobre el grip lateral usando:

    G_lat_max(v) = α + β · v²

donde α captura el grip mecánico base y β la ganancia por carga
aerodinámica.  Si β > 0 con p < 0.05, hay evidencia estadística
de que el grip crece con la velocidad (firma del downforce).

Pipeline:
  1. Calcular fuerzas G (mismo método que GGDiagram.py).
  2. Filtrar puntos de recta (|G_lat| < umbral adaptativo) para
     aislar datos donde el coche realmente usa grip lateral.
  3. Agrupar por velocidad (bins Freedman-Diaconis, mín. 15 pts/bin).
  4. Percentil k de |G_lat| por bin → envelope de grip.
  5. Regresión OLS: envelope vs v².
  6. Análisis de sensibilidad al percentil (valida robustez).

Limitaciones declaradas:
  · α y β son parámetros compuestos — no separan Cl, μ, m ni A.
  · Si el piloto no está al límite, α y β quedan subestimados.
  · μ varía con temperatura y desgaste → α no es constante real.
  · β solo es interpretable de forma relativa entre pilotos/sesiones.

Constantes: g = 9.81 m/s² (BIPM) — única constante física usada.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import savgol_filter
from scipy.stats import linregress, t as t_dist
from typing import Optional

import ui_assets
from fastf1.core import Session

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES FÍSICAS UNIVERSALES
# ─────────────────────────────────────────────────────────────────────────────
G_CONST = 9.81   # m/s²  — BIPM. La única constante que entra en los cálculos.

BG_DARK    = "#0E0E0F"
BG_PANEL   = "#111115"
BG_SURFACE = "#1A1A2E"
F1_WHITE   = "#FFFFFF"
F1_RED     = "#E8002D"
ACCENT_CYAN   = "#00D2FF"
ACCENT_GREEN  = "#39FF14"
ACCENT_AMBER  = "#FFA500"
ACCENT_PURPLE = "#C77DFF"
MONO_FONT  = "'JetBrains Mono', 'Courier New', monospace"


# ─────────────────────────────────────────────────────────────────────────────
# PASO 1 — CÁLCULO DE FUERZAS G (consistente con GGDiagram)
# ─────────────────────────────────────────────────────────────────────────────

def _calculate_g_forces(tel: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula g_lat [G], g_lon [G], g_total [G] y speed_ms [m/s].
    Mismo pipeline que GGDiagram._calculate_g_forces() para consistencia.
    """
    tel = tel.copy()
    if "Speed" not in tel.columns:
        for col in ("g_lat", "g_lon", "g_total", "speed_ms"):
            tel[col] = np.nan
        return tel

    speed_ms = tel["Speed"].values / 3.6
    time_s   = tel["Time"].dt.total_seconds().values
    dt       = np.diff(time_s, prepend=time_s[0])
    dt       = np.where(dt <= 0, 1e-3, dt)

    if "X" in tel.columns and "Y" in tel.columns:
        x, y     = tel["X"].values.astype(float), tel["Y"].values.astype(float)
        vx, vy   = np.gradient(x, time_s), np.gradient(y, time_s)
        heading  = np.arctan2(vy, vx)
        d_heading = np.diff(np.unwrap(heading), prepend=heading[0])   # BUG FIX
        omega     = d_heading / dt
        g_lat_raw = (speed_ms * omega) / G_CONST
    else:
        g_lat_raw = np.zeros(len(tel))

    g_lon_raw = np.gradient(speed_ms, time_s) / G_CONST

    # Ventana Savitzky-Golay: hz derivado de la mediana real del muestreo
    # Ventana de 0.5 s — consistente con GGDiagram._calculate_g_forces()
    hz     = 1.0 / float(np.median(dt[1:]))
    window = int(hz * 0.5)
    if window % 2 == 0:
        window += 1
    if window <= 3:
        window = 5

    g_lat_s = savgol_filter(g_lat_raw, window_length=window, polyorder=3)
    g_lon_s = savgol_filter(g_lon_raw, window_length=window, polyorder=3)
    g_lat_s = np.clip(g_lat_s, -6.0, 6.0)
    g_lon_s = np.clip(g_lon_s, -6.0, 6.0)

    tel["g_lat"]   = g_lat_s
    tel["g_lon"]   = g_lon_s
    tel["g_total"] = np.sqrt(g_lat_s**2 + g_lon_s**2)
    tel["speed_ms"] = speed_ms
    return tel


# ─────────────────────────────────────────────────────────────────────────────
# PASO 2 — FILTRADO DE PUNTOS DE RECTA
# ─────────────────────────────────────────────────────────────────────────────

def _filter_cornering(
    speed_kmh: np.ndarray,
    g_lat_abs: np.ndarray,
    min_g: float = 0.3,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Filtra puntos de recta donde |G_lat| ≈ 0.

    En las rectas el coche no gira → |G_lat| ≈ 0.  Incluir estos puntos
    en los bins de velocidad contamina los percentiles de alta velocidad
    (donde dominan las rectas) y aplasta la pendiente → R² ≈ 0.

    Umbral adaptativo:
      threshold = max(min_g, P25(|G_lat| donde |G_lat| > 0.1G))

    Justificación del suelo 0.3 G:
      0.3 G ≈ 2.9 m/s² de aceleración lateral → curvatura muy suave.
      Por debajo de este valor no hay curva significativa en F1.

    Retorna (speed_filtrado, g_lat_filtrado, threshold_usado).
    """
    above_noise = g_lat_abs > 0.1
    if above_noise.sum() > 50:
        threshold = max(min_g, float(np.percentile(g_lat_abs[above_noise], 25)))
    else:
        threshold = min_g

    mask = g_lat_abs >= threshold
    return speed_kmh[mask], g_lat_abs[mask], threshold


# ─────────────────────────────────────────────────────────────────────────────
# PASO 3 — BINNING DE VELOCIDAD (Freedman-Diaconis)
# ─────────────────────────────────────────────────────────────────────────────

def _fd_bins(data: np.ndarray, min_pts_per_bin: int = 15) -> np.ndarray:
    """
    Calcula los bordes de bins usando la regla de Freedman-Diaconis (1981):

        h = 2 · IQR(data) · N^(-1/3)

    Esta regla minimiza el MSE del estimador de densidad para distribuciones
    arbitrarias, sin asumir normalidad. Es estándar en estadística descriptiva
    (Freedman & Diaconis, Z. Wahrsch. verw. Gebiete, 57:453-476, 1981).

    Se impone además que cada bin tenga al menos `min_pts_per_bin` puntos,
    lo que garantiza que el percentil calculado dentro del bin sea estable.
    Con 15 puntos, el error estándar del percentil 95 es ~σ/√15 ≈ 26% de σ,
    aceptable para análisis exploratorio. Para publicación se recomendarían
    ≥30 puntos (Harrell, 2015). Aquí elegimos 15 como compromiso entre
    granularidad y estabilidad.

    Si los bins FD resultan demasiado estrechos (pocos puntos), se amplían
    hasta que cada bin alcanza el mínimo requerido.
    """
    n   = len(data)
    iqr = float(np.percentile(data, 75) - np.percentile(data, 25))

    if iqr <= 0:
        # distribución degenerada: usar sqrt(n) bins
        h = (data.max() - data.min()) / max(1, np.sqrt(n))
    else:
        h = 2.0 * iqr * n**(-1/3)

    d_min, d_max = float(data.min()), float(data.max())
    n_bins = max(2, int(np.ceil((d_max - d_min) / h)))

    # Ajustar hasta que todos los bins tengan suficientes puntos
    while True:
        edges  = np.linspace(d_min, d_max, n_bins + 1)
        counts = np.array([
            int(np.sum((data >= edges[i]) & (data < edges[i+1])))
            for i in range(n_bins)
        ])
        if n_bins <= 3 or counts.min() >= min_pts_per_bin:
            break
        n_bins -= 1

    return np.linspace(d_min, d_max, n_bins + 1)


# ─────────────────────────────────────────────────────────────────────────────
# PASO 4 — ENVELOPE DE GRIP: |G_lat|_k por bin de velocidad
# ─────────────────────────────────────────────────────────────────────────────

def _compute_envelope(
    speed_kmh: np.ndarray,
    g_lat_abs: np.ndarray,
    percentile: float,
) -> dict:
    """
    Para cada bin de velocidad (bordes derivados por Freedman-Diaconis),
    calcula el percentil `percentile` de |G_lat| dentro del bin.

    Usamos |G_lat| y no G_total porque:
      · G_total en una recta incluye G_lon (aceleración motriz), que no
        está relacionado con el grip lateral ni con el downforce de curva.
      · Usar G_total inflaría el envelope a altas velocidades donde hay
        más rectas, creando una pendiente espuria.

    El percentil es controlado por el usuario (slider), no fijado.
    La función se llama múltiples veces en el análisis de sensibilidad.

    Retorna dict con:
      v_centers  — velocidades centrales de cada bin [km/h]
      v_sq_ms    — (v_center en m/s)² para la regresión [m²/s²]
      g_pk       — percentil k de |G_lat| en cada bin [G]
      counts     — nº de muestras en cada bin
      bin_edges  — bordes de los bins [km/h]
    """
    edges    = _fd_bins(speed_kmh)
    centers  = (edges[:-1] + edges[1:]) / 2
    n_bins   = len(centers)

    g_pk   = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=int)

    for i in range(n_bins):
        mask = (speed_kmh >= edges[i]) & (speed_kmh < edges[i+1])
        cnt  = int(mask.sum())
        if cnt > 0:
            g_pk[i]   = float(np.percentile(g_lat_abs[mask], percentile))
            counts[i] = cnt

    # Velocidad central en m/s² para la regresión
    v_sq_ms = (centers / 3.6) ** 2

    return {
        "v_centers" : centers,
        "v_sq_ms"   : v_sq_ms,
        "g_pk"      : g_pk,
        "counts"    : counts,
        "bin_edges" : edges,
        "percentile": percentile,
    }


# ─────────────────────────────────────────────────────────────────────────────
# PASO 5 — REGRESIÓN OLS: G_lat_Pk = α + β·v²
# ─────────────────────────────────────────────────────────────────────────────

def _fit_regression(envelope: dict) -> Optional[dict]:
    """
    Regresión OLS (mínimos cuadrados ordinarios) de G_lat_Pk contra v²:

        G_lat_Pk = α + β · v²

    Requiere mínimo 4 bins válidos (con datos). Por debajo de 4, el sistema
    está subdeterminado o el R² no es interpretable.

    Retorna:
      alpha     — intercepto de la recta [G]
      beta      — pendiente [G/(m/s)²]
      r_sq      — coeficiente de determinación R²
      p_value   — p-valor de β (H0: β=0, contraste bilateral)
      ci95_beta — intervalo de confianza 95% de β (±)
      se_beta   — error estándar de β
      y_fit     — valores ajustados en los centros de bin

    Interpretación honesta de α y β:
      α  absorbe el grip mecánico pero no es igual a μ porque:
         (a) el piloto puede no estar al límite en todas las curvas
         (b) el modelo usa un percentil, no el máximo absoluto
         (c) hay carga aerodinámica incluso a bajas velocidades en F1
      β  es proporcional a μ·Cl_eff·A/(2mg). Sin conocer m, A, μ o Cl
         de forma independiente, β es un parámetro compuesto. Solo es
         interpretable de forma relativa (comparaciones entre sesiones
         o entre pilotos en la misma sesión donde m, A son constantes).

    El R² y el p-valor se reportan siempre para que el usuario pueda
    juzgar la calidad del ajuste de forma independiente.
    """
    v_sq  = envelope["v_sq_ms"]
    g_pk  = envelope["g_pk"]
    valid = ~np.isnan(g_pk)

    if valid.sum() < 4:
        return None

    x, y = v_sq[valid], g_pk[valid]
    slope, intercept, r, p_val, se_slope = linregress(x, y)
    r_sq = float(r**2)
    n    = int(valid.sum())

    # IC 95% usando distribución t con n-2 grados de libertad
    t95  = float(t_dist.ppf(0.975, df=n - 2))
    ci95 = float(t95 * se_slope)

    y_fit = intercept + slope * v_sq   # predicción en todos los bins

    return {
        "alpha"    : float(intercept),
        "beta"     : float(slope),
        "r_sq"     : r_sq,
        "p_value"  : float(p_val),
        "se_beta"  : float(se_slope),
        "ci95_beta": ci95,
        "n_bins"   : n,
        "y_fit"    : y_fit,
    }


# ─────────────────────────────────────────────────────────────────────────────
# PASO 6 — ANÁLISIS DE SENSIBILIDAD AL PERCENTIL
# ─────────────────────────────────────────────────────────────────────────────

def _sensitivity_analysis(
    speed_kmh: np.ndarray,
    g_lat_abs: np.ndarray,
    percentiles: list[float] | None = None,
) -> pd.DataFrame:
    """
    Calcula la regresión para múltiples percentiles y retorna una tabla
    con α, β, R² y p-valor para cada uno.

    Esto transforma el percentil de un supuesto arbitrario en un resultado
    validado: si β es estable en el rango [k_min, k_max], la elección
    específica de k no introduce sesgo sistemático en la conclusión.

    El CV de β (coeficiente de variación = σ/|μ|) cuantifica la robustez:
      CV < 5%  → muy estable, elección de k no crítica
      CV < 10% → estable, resultado defendible
      CV > 15% → sensible al percentil, interpretar con precaución
    """
    if percentiles is None:
        percentiles = [75, 80, 85, 90, 95, 98]

    rows = []
    for k in percentiles:
        env  = _compute_envelope(speed_kmh, g_lat_abs, k)
        reg  = _fit_regression(env)
        if reg is None:
            rows.append({"percentile": k, "alpha": np.nan, "beta": np.nan,
                         "r_sq": np.nan, "p_value": np.nan, "ci95_beta": np.nan})
        else:
            rows.append({
                "percentile": k,
                "alpha"     : reg["alpha"],
                "beta"      : reg["beta"],
                "r_sq"      : reg["r_sq"],
                "p_value"   : reg["p_value"],
                "ci95_beta" : reg["ci95_beta"],
            })

    df  = pd.DataFrame(rows)
    betas = df["beta"].dropna()
    if len(betas) > 1:
        df.attrs["cv_beta"] = float(betas.std() / (abs(betas.mean()) + 1e-9))
    else:
        df.attrs["cv_beta"] = np.nan
    return df


# ─────────────────────────────────────────────────────────────────────────────
# PASO 7 — RADIO DE CURVATURA (cinemático puro)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_curvature(tel: pd.DataFrame) -> np.ndarray:
    """
    Radio de curvatura instantáneo:

        R(t) = v(t)² / a_lat(t) = v(t)² / (|G_lat(t)| · g)

    Esta es una identidad cinemática exacta. No requiere ningún parámetro
    de vehículo. Solo usa g = 9.81 m/s² (constante física universal).

    Se protege contra división por cero asignando R = 2000 m cuando
    |G_lat| < 0.05G. Esta protección es numérica, no física:
    cualquier valor de a_lat < 0.05·9.81 ≈ 0.49 m/s² corresponde a
    radio > 2000 m, que es equivalente a "recta" a efectos prácticos
    (el circuito más largo de F1 tiene curvas con R ≈ 300 m en ápex).
    La referencia es el radio de curvatura de Eau Rouge (~80 m) como
    extremo inferior y Kemmel Straight como extremo superior (R → ∞).
    """
    speed_ms  = tel["speed_ms"].values if "speed_ms" in tel.columns \
                else tel["Speed"].values / 3.6
    g_lat_abs = np.abs(tel["g_lat"].values)
    a_lat     = g_lat_abs * G_CONST   # [m/s²]

    with np.errstate(divide="ignore", invalid="ignore"):
        radius = np.where(a_lat > 0.05 * G_CONST,
                          speed_ms**2 / a_lat,
                          2000.0)

    return np.clip(radius, 0.0, 2000.0)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS DE LAYOUT
# ─────────────────────────────────────────────────────────────────────────────

def _apply_dark(fig: go.Figure, height: int = 420) -> None:
    fig.update_layout(
        height=height,
        paper_bgcolor=BG_DARK, plot_bgcolor=BG_PANEL,
        font=dict(family=MONO_FONT, color=F1_WHITE, size=10),
        hoverlabel=dict(bgcolor=BG_SURFACE, bordercolor="rgba(255,255,255,0.15)",
                        font=dict(family=MONO_FONT, color=F1_WHITE, size=10)),
    )


def _ax(**kw) -> dict:
    base = dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)",
                zeroline=False,
                tickfont=dict(size=9, color="rgba(255,255,255,0.4)", family=MONO_FONT))
    base.update(kw)
    return base


# ─────────────────────────────────────────────────────────────────────────────
# VISUALIZACIONES
# ─────────────────────────────────────────────────────────────────────────────

def _fig_envelope_regression(
    envelope: dict,
    reg: dict,
    team_color: str,
) -> go.Figure:
    """
    Tab principal: |G_lat|_Pk por bin de velocidad + recta OLS ajustada.

    Qué muestra:
      · Puntos azules: el valor del percentil k de |G_lat| en cada bin.
        El tamaño del punto es proporcional al número de muestras en el bin,
        dando más peso visual a los bins más poblados.
      · Recta de regresión con banda de confianza al 95%.
      · Pendiente β con su intervalo de confianza e interpretación.

    Lo que NO se afirma:
      · Los puntos no son "el límite del grip" — son el percentil k de las
        G laterales registradas, que es un estimador del límite si el piloto
        estuvo frecuentemente cerca del máximo.
      · α no es μ — es el intercepto de la recta ajustada, que puede estar
        sesgado a la baja si el piloto no alcanzó el límite en baja velocidad.
    """
    v_c  = envelope["v_centers"]          # km/h
    v_sq = envelope["v_sq_ms"]            # (m/s)²
    g_pk = envelope["g_pk"]              # G
    cnt  = envelope["counts"]
    k    = int(envelope["percentile"])

    valid = ~np.isnan(g_pk)

    fig = go.Figure()

    # Banda de confianza 95%
    v_fit    = np.linspace(0, float(v_sq[valid].max()) * 1.05, 300)
    y_fit    = reg["alpha"] + reg["beta"] * v_fit
    y_fit_hi = reg["alpha"] + (reg["beta"] + reg["ci95_beta"]) * v_fit
    y_fit_lo = reg["alpha"] + (reg["beta"] - reg["ci95_beta"]) * v_fit
    v_fit_hi = np.concatenate([v_fit, v_fit[::-1]])
    g_fit_band = np.concatenate([y_fit_hi, y_fit_lo[::-1]])

    fig.add_trace(go.Scatter(
        x=v_fit_hi, y=g_fit_band,
        fill="toself",
        fillcolor=f"rgba({ui_assets.hex_rgb(team_color)},0.10)",
        line=dict(width=0),
        hoverinfo="skip", showlegend=False, name="_ci",
    ))

    # Recta de regresión
    label_reg = (
        f"G_lat = {reg['alpha']:.3f} + {reg['beta']*1e3:.4f}×10⁻³·v²"
        f"  (R²={reg['r_sq']:.3f}, p={reg['p_value']:.3f})"
    )
    fig.add_trace(go.Scatter(
        x=v_fit, y=y_fit,
        mode="lines",
        line=dict(color=team_color, width=2.5, dash="dash"),
        name=label_reg,
        hovertemplate="v²=%{x:.0f}m²/s²<br>Ajuste=%{y:.3f}G<extra></extra>",
    ))

    # Puntos del envelope
    sizes = [max(7, min(20, c // 4)) for c in cnt[valid]]
    hover = [
        f"v≈{vc:.0f} km/h<br>P{k}(|G_lat|)={gp:.3f}G<br>n={cn} muestras"
        for vc, gp, cn in zip(v_c[valid], g_pk[valid], cnt[valid])
    ]
    fig.add_trace(go.Scatter(
        x=v_sq[valid], y=g_pk[valid],
        mode="markers",
        marker=dict(size=sizes, color=ACCENT_CYAN,
                    line=dict(color=F1_WHITE, width=0.8)),
        text=hover,
        hovertemplate="%{text}<extra></extra>",
        name=f"P{k}(|G_lat|) por bin (Freedman-Diaconis)",
    ))

    # Anotación del intercepto
    fig.add_annotation(
        x=0, y=reg["alpha"],
        text=f"α = {reg['alpha']:.3f} G<br><span style='font-size:9px;'>(intercepto en v=0)</span>",
        showarrow=True, arrowhead=2, arrowcolor=ACCENT_CYAN, ax=90, ay=-40,
        font=dict(color=ACCENT_CYAN, size=9, family=MONO_FONT),
        bgcolor=BG_SURFACE, bordercolor=ACCENT_CYAN, borderwidth=1,
    )

    _apply_dark(fig, 440)
    fig.update_layout(
        margin=dict(l=60, r=20, t=20, b=60),
        legend=dict(y=1.04, x=0.0, bgcolor="rgba(0,0,0,0)", font=dict(size=9)),
        hovermode="closest",
        xaxis=dict(**_ax(title="v²  [m²/s²]", ticksuffix=" m²/s²")),
        yaxis=dict(**_ax(title=f"P{k}(|G_lat|)  [G]", ticksuffix=" G")),
    )
    return fig


def _fig_envelope_vs_speed(
    envelope: dict,
    reg: dict,
    team_color: str,
) -> go.Figure:
    """
    Envolvente |G_lat|_Pk vs velocidad en km/h (eje más intuitivo que v²).

    La curva creciente es la evidencia visual directa de que el grip
    lateral aumenta con la velocidad — consecuencia del downforce.
    Una curva plana indicaría ausencia de carga aerodinámica.

    La curva ajustada se superpone solo si R² ≥ 0.40 y p < 0.10, para
    no inducir a interpretar un ajuste de baja calidad como significativo.
    """
    v_c  = envelope["v_centers"]
    g_pk = envelope["g_pk"]
    cnt  = envelope["counts"]
    k    = int(envelope["percentile"])
    valid = ~np.isnan(g_pk)

    fig = go.Figure()

    # Barras de error (simetría: no tenemos asimetría del percentil → solo visual)
    fig.add_trace(go.Scatter(
        x=v_c[valid], y=g_pk[valid],
        mode="lines+markers",
        line=dict(color=team_color, width=2),
        marker=dict(
            size=[max(5, min(14, c // 5)) for c in cnt[valid]],
            color=team_color, line=dict(color=F1_WHITE, width=0.8),
        ),
        name=f"P{k}(|G_lat|)",
        hovertemplate="v=%{x:.0f}km/h<br>P" + str(k) + "(|G_lat|)=%{y:.3f}G<extra></extra>",
    ))

    # Curva ajustada en escala km/h
    if reg["r_sq"] >= 0.40 and reg["p_value"] < 0.10:
        v_fit_kmh = np.linspace(float(v_c.min()), float(v_c.max()), 300)
        v_fit_sq  = (v_fit_kmh / 3.6) ** 2
        g_fit     = reg["alpha"] + reg["beta"] * v_fit_sq
        fig.add_trace(go.Scatter(
            x=v_fit_kmh, y=g_fit,
            mode="lines",
            line=dict(color=ACCENT_CYAN, width=1.5, dash="dot"),
            name=f"Ajuste OLS (R²={reg['r_sq']:.2f})",
            hovertemplate="v=%{x:.0f}km/h<br>Ajuste=%{y:.3f}G<extra></extra>",
        ))
    elif reg["r_sq"] < 0.40:
        fig.add_annotation(
            x=0.5, y=0.5, xref="paper", yref="paper",
            text=f"R²={reg['r_sq']:.2f} — ajuste de baja calidad,<br>curva no superpuesta",
            showarrow=False,
            font=dict(color=ACCENT_AMBER, size=10, family=MONO_FONT),
            bgcolor="rgba(14,14,15,0.7)",
        )

    _apply_dark(fig, 360)
    fig.update_layout(
        margin=dict(l=60, r=20, t=10, b=50),
        legend=dict(orientation="h", y=1.04, x=0.5, xanchor="center",
                    bgcolor="rgba(0,0,0,0)", font=dict(size=9)),
        hovermode="x unified",
        xaxis=dict(**_ax(title="Velocidad [km/h]", ticksuffix=" km/h")),
        yaxis=dict(**_ax(title=f"P{k}(|G_lat|) [G]", ticksuffix=" G")),
    )
    return fig


def _fig_sensitivity(sens_df: pd.DataFrame, team_color: str) -> go.Figure:
    """
    Análisis de sensibilidad: cómo cambian α y β con el percentil k.

    Si la línea de β es horizontal (plana) entre k=80 y k=98, la elección
    de percentil no afecta a la conclusión principal (β > 0, pendiente
    positiva). En ese caso el resultado es robusto.

    El CV de β (coeficiente de variación) se muestra en el título:
      CV < 5%  → muy estable
      CV < 10% → estable, resultado defendible
      CV > 15% → sensible, interpretar con precaución

    Esta gráfica es la que convierte el percentil de "supuesto arbitrario"
    en "resultado validado".
    """
    valid = sens_df.dropna(subset=["alpha", "beta"])
    cv    = sens_df.attrs.get("cv_beta", np.nan)
    cv_str = f"CV(β) = {cv*100:.1f}%" if not np.isnan(cv) else "CV(β) = —"

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Pendiente β vs percentil", "Intercepto α vs percentil"],
        horizontal_spacing=0.12,
    )

    # β con barras de error ±1.96·SE
    fig.add_trace(go.Scatter(
        x=valid["percentile"], y=valid["beta"],
        mode="lines+markers",
        line=dict(color=team_color, width=2),
        marker=dict(size=9, color=team_color, line=dict(color=F1_WHITE, width=1)),
        error_y=dict(
            type="data",
            array=valid["ci95_beta"].tolist(),
            visible=True,
            color=f"rgba({ui_assets.hex_rgb(team_color)},0.4)",
            thickness=1.2,
        ),
        name="β",
        hovertemplate="k=%{x}<br>β=%{y:.5f}G/(m/s)²<extra></extra>",
    ), row=1, col=1)

    # α
    fig.add_trace(go.Scatter(
        x=valid["percentile"], y=valid["alpha"],
        mode="lines+markers",
        line=dict(color=ACCENT_CYAN, width=2),
        marker=dict(size=9, color=ACCENT_CYAN, line=dict(color=F1_WHITE, width=1)),
        name="α",
        hovertemplate="k=%{x}<br>α=%{y:.3f}G<extra></extra>",
    ), row=1, col=2)

    # Franja de R²: colorear fondo según calidad del ajuste
    for _, row in valid.iterrows():
        color_r2 = ACCENT_GREEN if row["r_sq"] >= 0.6 else \
                   ACCENT_AMBER  if row["r_sq"] >= 0.4 else F1_RED
        fig.add_annotation(
            x=row["percentile"], y=valid["beta"].max() * 1.02,
            text=f"R²={row['r_sq']:.2f}",
            showarrow=False,
            font=dict(size=7, color=color_r2, family=MONO_FONT),
            row=1, col=1,
        )

    _apply_dark(fig, 360)
    fig.update_layout(
        margin=dict(l=54, r=20, t=55, b=50),
        showlegend=False,
        title=dict(
            text=f"Análisis de sensibilidad al percentil — {cv_str}",
            font=dict(size=11, color="rgba(255,255,255,0.5)", family=MONO_FONT),
            x=0.5, xanchor="center",
        ),
        xaxis=dict(**_ax(title="Percentil k", dtick=5)),
        xaxis2=dict(**_ax(title="Percentil k", dtick=5)),
        yaxis=dict(**_ax(title="β [G/(m/s)²]")),
        yaxis2=dict(**_ax(title="α [G]")),
    )
    for ann in fig.layout.annotations[:2]:   # subplot titles
        ann.update(font=dict(family=MONO_FONT, size=10,
                             color="rgba(255,255,255,0.40)"))
    return fig




def _fig_curvature_map(tel: pd.DataFrame, team_color: str) -> go.Figure:
    """
    Mapa del circuito coloreado por el radio de curvatura R = v²/(G_lat·g).

    Esta es la única pestaña 100% determinista sin ningún supuesto.
    La escala de color es logarítmica porque los radios van de ~50m (chicane)
    a 2000m (recta larga), un rango de 40:1 que en escala lineal aplanaría
    los colores en las curvas rápidas.

    Referencia de escala:
      R < 80 m  → curva muy lenta (ej. La Source, Loews)
      R ~ 150m  → curva técnica (ej. Zandvoort ola)
      R ~ 400m  → curva rápida (ej. Copse, Maggotts)
      R > 800m  → curva de alta velocidad / recta de curvatura gradual
    """
    x      = tel["X"].values
    y      = tel["Y"].values
    radius = _compute_curvature(tel)
    speed  = tel["Speed"].values

    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=x, y=y, mode="lines",
        line=dict(color="rgba(80,80,106,0.25)", width=9),
        hoverinfo="skip", showlegend=False,
    ))
    fig.add_trace(go.Scattergl(
        x=x, y=y, mode="markers",
        marker=dict(
            size=4,
            color=np.log1p(radius),
            colorscale=[
                [0.00, "#E8002D"], [0.20, "#FFA500"],
                [0.50, "#FFF200"], [0.80, "#00D2FF"],
                [1.00, "#0044FF"],
            ],
            colorbar=dict(
                title=dict(text="log(R+1) m", font=dict(size=9, color="rgba(255,255,255,0.5)")),
                tickvals=[np.log1p(r) for r in [50, 100, 200, 500, 1000, 2000]],
                ticktext=["50m", "100m", "200m", "500m", "1km", "recta"],
                thickness=10, len=0.6,
                tickfont=dict(size=8, color="rgba(255,255,255,0.4)", family=MONO_FONT),
                bgcolor="rgba(0,0,0,0)", bordercolor="rgba(255,255,255,0.1)", borderwidth=1,
            ),
            showscale=True,
        ),
        text=[f"R ≈ {r:.0f} m · v = {v:.0f} km/h"
              for r, v in zip(radius, speed)],
        hovertemplate="%{text}<extra></extra>",
        showlegend=False,
    ))

    mid_x, mid_y = float(np.mean(x)), float(np.mean(y))
    rng = max(x.max()-x.min(), y.max()-y.min()) / 1.24
    _apply_dark(fig, 520)
    fig.update_layout(
        showlegend=False, margin=dict(l=10, r=10, t=20, b=10),
        xaxis=dict(range=[mid_x-rng, mid_x+rng], scaleanchor="y", scaleratio=1,
                   showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(range=[mid_y-rng*1.35, mid_y+rng*0.65],
                   showgrid=False, zeroline=False, showticklabels=False),
    )
    return fig


def _render_model_card(reg: dict, cv_beta: float, k: int, team_color: str) -> None:
    """Panel de resultados del modelo — desglosado en parámetros y calidad."""
    quality_color = (
        ACCENT_GREEN if reg["r_sq"] >= 0.65 else
        ACCENT_AMBER if reg["r_sq"] >= 0.40 else F1_RED
    )
    quality_label = (
        "Bueno" if reg["r_sq"] >= 0.65 else
        "Moderado" if reg["r_sq"] >= 0.40 else
        "Débil"
    )
    cv_color = (
        ACCENT_GREEN if not np.isnan(cv_beta) and cv_beta < 0.05 else
        ACCENT_AMBER if not np.isnan(cv_beta) and cv_beta < 0.10 else
        F1_RED
    )
    cv_label = (
        "Muy estable" if not np.isnan(cv_beta) and cv_beta < 0.05 else
        "Estable"     if not np.isnan(cv_beta) and cv_beta < 0.10 else
        "Sensible"
    )
    sig_color = ACCENT_GREEN if reg["p_value"] < 0.05 else F1_RED
    sig_label = "Significativa" if reg["p_value"] < 0.05 else "No significativa"

    # ── Cabecera: ecuación del modelo ──────────────────────────────────────
    st.markdown(
        f"""
        <div style="border-left:3px solid {team_color};padding:8px 14px;
                    background:rgba(255,255,255,0.02);margin-bottom:14px;">
          <div style="font-size:9px;letter-spacing:2px;color:rgba(255,255,255,0.4);">
            ECUACIÓN DEL MODELO · OLS sobre {reg['n_bins']} bins
          </div>
          <div style="font-family:{MONO_FONT};font-size:16px;color:#FFF;margin-top:2px;">
            P<sub>{k}</sub>(|G<sub>lat</sub>|) = α + β · v²
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Bloque 1: Parámetros del modelo ───────────────────────────────────
    st.markdown(
        "<p style='font-size:10px;letter-spacing:2px;color:rgba(255,255,255,0.45);"
        "margin:8px 0 6px;'>PARÁMETROS</p>",
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            f"""
            <div style="background:{BG_PANEL};padding:14px 16px;border-radius:3px;
                        border-top:2px solid {ACCENT_CYAN};">
              <div style="font-size:10px;color:rgba(255,255,255,0.5);letter-spacing:1px;">
                α  ·  INTERCEPTO
              </div>
              <div style="font-size:24px;font-weight:800;color:{ACCENT_CYAN};
                          font-family:{MONO_FONT};margin-top:4px;">
                {reg['alpha']:.3f} <span style="font-size:13px;color:rgba(255,255,255,0.4);">G</span>
              </div>
              <div style="font-size:10px;color:rgba(255,255,255,0.35);margin-top:4px;">
                Grip extrapolado a v = 0
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div style="background:{BG_PANEL};padding:14px 16px;border-radius:3px;
                        border-top:2px solid {team_color};">
              <div style="font-size:10px;color:rgba(255,255,255,0.5);letter-spacing:1px;">
                β  ·  PENDIENTE
              </div>
              <div style="font-size:24px;font-weight:800;color:{team_color};
                          font-family:{MONO_FONT};margin-top:4px;">
                {reg['beta']*1e3:.4f} <span style="font-size:13px;color:rgba(255,255,255,0.4);">×10⁻³</span>
              </div>
              <div style="font-size:10px;color:rgba(255,255,255,0.35);margin-top:4px;">
                ± {reg['ci95_beta']*1e3:.4f}  (IC 95%) · G/(m/s)²
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Bloque 2: Calidad del ajuste ──────────────────────────────────────
    st.markdown(
        "<p style='font-size:10px;letter-spacing:2px;color:rgba(255,255,255,0.45);"
        "margin:18px 0 6px;'>CALIDAD Y ROBUSTEZ</p>",
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f"""
            <div style="background:{BG_PANEL};padding:12px 14px;border-radius:3px;">
              <div style="font-size:10px;color:rgba(255,255,255,0.5);letter-spacing:1px;">R²</div>
              <div style="font-size:22px;font-weight:800;color:{quality_color};
                          font-family:{MONO_FONT};margin-top:2px;">{reg['r_sq']:.3f}</div>
              <div style="font-size:10px;color:{quality_color};margin-top:2px;">{quality_label}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div style="background:{BG_PANEL};padding:12px 14px;border-radius:3px;">
              <div style="font-size:10px;color:rgba(255,255,255,0.5);letter-spacing:1px;">P-VALOR β</div>
              <div style="font-size:22px;font-weight:800;color:{sig_color};
                          font-family:{MONO_FONT};margin-top:2px;">{reg['p_value']:.4f}</div>
              <div style="font-size:10px;color:{sig_color};margin-top:2px;">{sig_label}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""
            <div style="background:{BG_PANEL};padding:12px 14px;border-radius:3px;">
              <div style="font-size:10px;color:rgba(255,255,255,0.5);letter-spacing:1px;">CV(β)</div>
              <div style="font-size:22px;font-weight:800;color:{cv_color};
                          font-family:{MONO_FONT};margin-top:2px;">{cv_beta*100:.1f}%</div>
              <div style="font-size:10px;color:{cv_color};margin-top:2px;">{cv_label}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# PUNTO DE ENTRADA PÚBLICO
# ─────────────────────────────────────────────────────────────────────────────

def render_aero_analysis(session: Session, driver: str, lap_number: int) -> None:
    """
    Renders:
      Card   — resultados del modelo OLS con interpretación honesta.
      Tab 1  — Regresión G_lat_Pk vs v² (scatter + recta + IC 95%).
      Tab 2  — Envolvente vs velocidad en km/h.
      Tab 3  — Análisis de sensibilidad al percentil (valida robustez).
      Tab 4  — Mapa de curvatura R = v²/(G_lat·g) [100% determinista].
    """
    with st.spinner(f"Calculando análisis aerodinámico — {driver} · V{lap_number}"):
        try:
            lap = session.laps.pick_drivers(driver)[
                session.laps.pick_drivers(driver)["LapNumber"] == lap_number
            ].iloc[0]
            tel = lap.get_telemetry().add_distance()
        except Exception as e:
            st.error(f"❌ No se pudo cargar la telemetría: {e}")
            return

        if tel.empty or "X" not in tel.columns:
            st.error("❌ Telemetría sin coordenadas XY (requiere load_telemetry=True).")
            return

        tel = _calculate_g_forces(tel)
        try:
            team_color = f"#{session.results.loc[session.results['Abbreviation']==driver,'TeamColor'].iloc[0]}"
        except Exception:
            team_color = F1_RED

    # ── Datos base ────────────────────────────────────────────────────────
    speed_kmh_all = tel["Speed"].values
    g_lat_abs_all = np.abs(tel["g_lat"].values)

    # ── Filtrado de puntos de recta ───────────────────────────────────────
    # Sin este filtro, los bins de alta velocidad están dominados por rectas
    # donde |G_lat| ≈ 0, lo que aplasta el percentil y mata la regresión.
    speed_kmh, g_lat_abs, g_threshold = _filter_cornering(speed_kmh_all, g_lat_abs_all)
    n_total   = len(speed_kmh_all)
    n_corners = len(speed_kmh)

    # ── Filtros y datos procesados ──────────────────────
    col_sl, col_data1, col_data2 = st.columns([3, 1, 1])
    with col_sl:
        k = st.slider(
            "Percentil k del envelope de grip lateral",
            min_value=75, max_value=99, value=90, step=1,
            key="aero_percentile",
        )

    # Cálculos
    envelope  = _compute_envelope(speed_kmh, g_lat_abs, float(k))
    reg       = _fit_regression(envelope)
    sens_df   = _sensitivity_analysis(speed_kmh, g_lat_abs)
    cv_beta   = sens_df.attrs.get("cv_beta", np.nan)
    n_bins_valid = int((~np.isnan(envelope["g_pk"])).sum())

    with col_data1:
        st.metric("Datos en curva", f"{n_corners:,}",
                  delta=f"{100*n_corners/max(n_total,1):.0f}% del total")
    with col_data2:
        st.metric("Bins válidos", n_bins_valid)

    st.divider()

    # ── Card del modelo ───────────────────────────────
    if reg:
        _render_model_card(reg, cv_beta if not np.isnan(cv_beta) else 0.0, k, team_color)
    else:
        st.warning(
            f"No hay suficientes bins válidos para la regresión "
            f"(se necesitan >=4, hay {n_bins_valid})."
        )

    st.divider()

    # ── Tabs (4: sin G-G, que ya está en GGDiagram) ──────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        f"Regresión P{k} vs v²",
        "Envelope vs velocidad",
        "Sensibilidad al percentil",
        "Mapa de curvatura",
    ])

    with tab1:
        if reg:
            st.plotly_chart(_fig_envelope_regression(envelope, reg, team_color),
                            use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("Regresión no disponible.")

    with tab2:
        if reg:
            st.plotly_chart(_fig_envelope_vs_speed(envelope, reg, team_color),
                            use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("Envolvente no disponible.")

    with tab3:
        if len(sens_df.dropna(subset=["beta"])) >= 3:
            st.plotly_chart(_fig_sensitivity(sens_df, team_color),
                            use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("Análisis de sensibilidad no disponible (pocos bins válidos).")

    with tab4:
        st.plotly_chart(_fig_curvature_map(tel, team_color),
                        use_container_width=True, config={"displayModeBar": False})