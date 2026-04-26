"""
TyreLoad.py — TALOS F1 · Mapa de Carga por Rueda
=================================================

Módulo que visualiza la distribución cualitativa de carga en los
4 neumáticos de un monoplaza a lo largo de una vuelta, usando
únicamente la aceleración lateral y longitudinal de la telemetría.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import savgol_filter
from typing import Optional
from fastf1.core import Session

from GGDiagram import _calculate_g_forces
from Aero import _filter_cornering, _compute_envelope, _fit_regression

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES FÍSICAS Y DE DISEÑO
# ─────────────────────────────────────────────────────────────────────────────
G_CONST = 9.81   # m/s²

BG_DARK    = "#0E0E0F"
BG_PANEL   = "#111115"
BG_SURFACE = "#1A1A2E"
F1_WHITE   = "#FFFFFF"
F1_RED     = "#E8002D"
GREY_TRACK = "#50506A"

ACCENT_CYAN   = "#00D2FF"  # FL
ACCENT_AMBER  = "#FFB347"  # FR
ACCENT_GREEN  = "#39FF14"  # RL
ACCENT_PURPLE = "#C77DFF"  # RR

MONO_FONT  = "'JetBrains Mono', 'Courier New', monospace"

TARGET_FRAMES = 1000

# ─────────────────────────────────────────────────────────────────────────────
# PASO 1 — MODELO DE DISTRIBUCIÓN DE CARGA Y EXIGENCIA
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# PASO 2 — MODELO DE DISTRIBUCIÓN DE CARGA
# ─────────────────────────────────────────────────────────────────────────────

def _compute_wheel_loads(tel: pd.DataFrame, alpha: float, beta: float, brake_bias: float) -> pd.DataFrame:
    """
    Modelo Físico de Exigencia del Neumático (Grip Utilization).
    
    1. Calcula el Downforce a partir de alpha (mecánico) y beta (aero).
    2. Calcula F_z (carga vertical) estática + transferencias (h/L=0.10, h/t=0.15).
    3. Calcula F_x (fuerza longitudinal demandada) basada en G_lon y Brake Bias.
    4. Calcula F_y (fuerza lateral demandada) basada en G_lat.
    5. Retorna la Exigencia = sqrt(F_x^2 + F_y^2) / (alpha * F_z).
    """
    if "g_lat" not in tel.columns or "g_lon" not in tel.columns:
        for col in ["load_fl", "load_fr", "load_rl", "load_rr", "Fx_fl", "Fx_fr", "Fx_rl", "Fx_rr"]:
            tel[col] = np.nan
        return tel

    g_lat = tel["g_lat"].values
    g_lon = tel["g_lon"].values
    speed_ms = tel["Speed"].values / 3.6
    
    # ── Parámetros del Vehículo ────────────────────────────
    W_f = 0.45  # 45% peso delante
    W_r = 0.55  # 55% peso detrás
    h_L = 0.10  # CG height / Wheelbase
    h_t = 0.15  # CG height / Track width
    
    # ── 1. Carga Vertical (F_z) con efecto DRS ──────────────
    # El DRS reduce el downforce total del coche (aprox -20% de carga total en F1 moderna)
    has_drs = "DRS" in tel.columns
    # En FastF1, DRS >= 10 suele indicar abierto (10, 12, 14...)
    drs_active = (tel["DRS"].values >= 10) if has_drs else np.zeros_like(speed_ms, dtype=bool)
    
    # Aplicamos reducción de beta si el DRS está abierto
    beta_inst = np.where(drs_active, beta * 0.80, beta)
    
    # DF = (beta / alpha) * v^2  (múltiplos del peso)
    df_ratio = (beta_inst / alpha) * (speed_ms**2) if alpha > 0 else np.zeros_like(speed_ms)
    Fz_total = 1.0 + df_ratio
    
    # Transferencia Longitudinal y Lateral
    delta_Fz_lon = g_lon * h_L
    delta_Fz_lat = g_lat * h_t
    
    # Fz por neumático (en Gs relativos)
    # Al frenar (g_lon < 0), delante gana carga -> restamos g_lon
    # Curva izquierda (g_lat > 0), derecha gana carga -> sumamos g_lat a derecha
    Fz_fl = np.clip((Fz_total * W_f / 2) - delta_Fz_lon - delta_Fz_lat, 0.01, None)
    Fz_fr = np.clip((Fz_total * W_f / 2) - delta_Fz_lon + delta_Fz_lat, 0.01, None)
    Fz_rl = np.clip((Fz_total * W_r / 2) + delta_Fz_lon - delta_Fz_lat, 0.01, None)
    Fz_rr = np.clip((Fz_total * W_r / 2) + delta_Fz_lon + delta_Fz_lat, 0.01, None)
    
    # ── 2. Fuerza Demandada (F_x, F_y) ──────────────────────
    # F_x:
    # Usamos el canal Brake para distinguir frenada mecánica de coasting/freno motor
    has_brake = "Brake" in tel.columns
    # Consideramos frenada activa si el canal Brake es > 0 (funciona para booleanos 0/1 y continuos)
    brake_active = (tel["Brake"].values > 0) if has_brake else (g_lon < -0.2)
    
    Fx_fl = np.zeros_like(g_lon)
    Fx_fr = np.zeros_like(g_lon)
    Fx_rl = np.zeros_like(g_lon)
    Fx_rr = np.zeros_like(g_lon)
    
    # Aceleración (g_lon > 0) -> Solo eje trasero (RWD)
    mask_accel = g_lon > 0
    Fx_rl[mask_accel] = g_lon[mask_accel] / 2
    Fx_rr[mask_accel] = g_lon[mask_accel] / 2
    
    # Frenada mecánica (brake_active y decelerando) -> Repartido por Brake Bias
    mask_brake = brake_active & (g_lon < 0)
    Fx_front_total = abs(g_lon[mask_brake]) * brake_bias
    Fx_rear_total  = abs(g_lon[mask_brake]) * (1 - brake_bias)
    Fx_fl[mask_brake] = Fx_front_total / 2
    Fx_fr[mask_brake] = Fx_front_total / 2
    Fx_rl[mask_brake] = Fx_rear_total / 2
    Fx_rr[mask_brake] = Fx_rear_total / 2
    
    # Coasting / Freno Motor (decelerando pero sin pisar el freno)
    # En F1 el freno motor actúa en el eje trasero. El drag es global pero lo simplificamos a eje trasero
    # para no inventar un reparto de drag 50/50 que no conocemos.
    mask_coast = (~brake_active) & (g_lon < 0)
    Fx_rl[mask_coast] = abs(g_lon[mask_coast]) / 2
    Fx_rr[mask_coast] = abs(g_lon[mask_coast]) / 2
    
    # F_y: repartido proporcional a la carga vertical de cada rueda respecto al total
    Fy_fl = abs(g_lat) * (Fz_fl / Fz_total)
    Fy_fr = abs(g_lat) * (Fz_fr / Fz_total)
    Fy_rl = abs(g_lat) * (Fz_rl / Fz_total)
    Fy_rr = abs(g_lat) * (Fz_rr / Fz_total)
    
    # ── 3. Exigencia (Grip Utilization) ────────────────────
    # Exigencia = Fuerza Resultante / Grip Máximo Disponible
    # Grip Máximo = alpha * Fz (asumiendo alpha como coeficiente de fricción base)
    exig_fl = np.sqrt(Fx_fl**2 + Fy_fl**2) / (alpha * Fz_fl)
    exig_fr = np.sqrt(Fx_fr**2 + Fy_fr**2) / (alpha * Fz_fr)
    exig_rl = np.sqrt(Fx_rl**2 + Fy_rl**2) / (alpha * Fz_rl)
    exig_rr = np.sqrt(Fx_rr**2 + Fy_rr**2) / (alpha * Fz_rr)
    
    tel["load_fl"] = np.clip(exig_fl, 0, 1)
    tel["load_fr"] = np.clip(exig_fr, 0, 1)
    tel["load_rl"] = np.clip(exig_rl, 0, 1)
    tel["load_rr"] = np.clip(exig_rr, 0, 1)
    
    # Exportamos Fx para el modelo térmico de frenos
    tel["Fx_fl"] = Fx_fl
    tel["Fx_fr"] = Fx_fr
    tel["Fx_rl"] = Fx_rl
    tel["Fx_rr"] = Fx_rr

    return tel

# ─────────────────────────────────────────────────────────────────────────────
# PASO 3 — MODELO TÉRMICO Y BALANCE DINÁMICO
# ─────────────────────────────────────────────────────────────────────────────

def _temp_to_color(temp: float) -> str:
    """Interpola temperatura a un mapa distinto (Azul -> Magenta -> Blanco) para diferenciarlo de los neumáticos"""
    t = np.clip(temp, 100, 1200)
    if t < 500:
        # Azul oscuro a Azul vivo
        f = (t - 100) / 400
        r, g, b = 0, int(f*100), 255
    elif t < 900:
        # Azul a Rosa/Magenta
        f = (t - 500) / 400
        r, g, b = int(f*255), 0, 255
    else:
        # Magenta a Blanco
        f = (t - 900) / 300
        r, g, b = 255, int(f*255), 255
    return f"rgb({r},{g},{b})"

def _compute_brake_thermals(tel: pd.DataFrame) -> pd.DataFrame:
    """
    Modelo termodinámico de primer orden para los frenos.
    Integra la potencia de frenado (calentamiento) y el flujo de aire (enfriamiento).
    """
    time_s = tel["Time"].dt.total_seconds().values
    dt = np.diff(time_s, prepend=time_s[0])
    dt = np.where(dt <= 0, 1e-3, dt)
    
    speed_ms = tel["Speed"].values / 3.6
    
    if "Brake" in tel.columns:
        brake_col = pd.to_numeric(tel["Brake"], errors='coerce').fillna(0)
        brake_active = brake_col.values > 0
    else:
        brake_active = tel["g_lon"].values < -0.2
    
    # Constantes térmicas (aproximadas para F1)
    # Fx está normalizado por G. Fuerza real = Fx * m_car * g
    M_CAR = 800.0
    G = 9.81
    M_DISC = 1.5
    C_CARBON = 800.0  # J/kgK
    K_HEAT = (M_CAR * G) / (M_DISC * C_CARBON)
    
    # Constante de enfriamiento convectivo sintonizada
    K_CONV = 0.001
    T_AMB = 30.0
    
    for ax in ["fl", "fr", "rl", "rr"]:
        fx = tel[f"Fx_{ax}"].values
        temps = np.zeros(len(tel))
        temps[0] = 100.0  # temp inicial cálida
        
        for i in range(1, len(tel)):
            heating = 0.0
            if brake_active[i] and fx[i] > 0:
                power = fx[i] * speed_ms[i]
                heating = power * K_HEAT * dt[i]
                
            cooling = K_CONV * speed_ms[i] * (temps[i-1] - T_AMB) * dt[i]
            
            temps[i] = temps[i-1] + heating - cooling
            
            if temps[i] < T_AMB:
                temps[i] = T_AMB
            elif temps[i] > 1200.0:
                temps[i] = 1200.0
                
        tel[f"temp_{ax}"] = temps
        
    return tel

def _compute_dynamic_balance(tel: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula el balance dinámico basado en la diferencia de utilización de grip.
    > 0 : Limitado por tren delantero (Subviraje)
    < 0 : Limitado por tren trasero (Sobreviraje)
    """
    front_stress = (tel["load_fl"] + tel["load_fr"]) / 2.0
    rear_stress = (tel["load_rl"] + tel["load_rr"]) / 2.0
    
    tel["dyn_balance"] = (front_stress - rear_stress) * 100.0 # en %
    return tel

# ─────────────────────────────────────────────────────────────────────────────
# PASO 4 — COLOR MAP (Verde -> Amarillo -> Rojo)
# ─────────────────────────────────────────────────────────────────────────────

def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)

COLOR_GREEN  = _hex_to_rgb("#39FF14")
COLOR_YELLOW = _hex_to_rgb("#FFD600")
COLOR_RED    = _hex_to_rgb("#E8002D")

def _load_to_color(val: float) -> str:
    """Interpola un valor 0-1 a un color verde -> amarillo -> rojo."""
    v = np.clip(val, 0, 1)
    if v < 0.5:
        # Interpolate Green to Yellow
        f = v / 0.5
        r = int(COLOR_GREEN[0] + f * (COLOR_YELLOW[0] - COLOR_GREEN[0]))
        g = int(COLOR_GREEN[1] + f * (COLOR_YELLOW[1] - COLOR_GREEN[1]))
        b = int(COLOR_GREEN[2] + f * (COLOR_YELLOW[2] - COLOR_GREEN[2]))
    else:
        # Interpolate Yellow to Red
        f = (v - 0.5) / 0.5
        r = int(COLOR_YELLOW[0] + f * (COLOR_RED[0] - COLOR_YELLOW[0]))
        g = int(COLOR_YELLOW[1] + f * (COLOR_RED[1] - COLOR_YELLOW[1]))
        b = int(COLOR_YELLOW[2] + f * (COLOR_RED[2] - COLOR_YELLOW[2]))
    return f"rgb({r},{g},{b})"

# ─────────────────────────────────────────────────────────────────────────────
# PASO 4 — SILUETA F1 Y TRACK MAP ANIMADO
# ─────────────────────────────────────────────────────────────────────────────

import base64, os as _os

def _img_to_uri(path: str) -> str:
    """Convierte un archivo PNG a data URI base64 para Plotly."""
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f"data:image/png;base64,{b64}"

def _arc_path(cx: float, cy: float, r: float, pct: float) -> str:
    """
    Genera un path SVG de arco (sentido horario desde las 12h) para un porcentaje 0-1.
    Se aproxima mediante una polyline para evitar los graves bugs de Plotly 
    al renderizar el comando 'A' en coordenadas de datos.
    """
    import math
    if pct <= 0.01:
        return ""
    if pct > 1:
        pct = 1.0
        
    angle_start = math.pi / 2          # 12 en punto
    angle_end   = angle_start - 2 * math.pi * pct  # horario
    
    # 40 segmentos para un círculo completo garantizan suavidad perfecta
    n_segments = max(2, int(40 * pct))
    
    cmds = []
    for i in range(n_segments + 1):
        f = i / n_segments
        angle = angle_start * (1 - f) + angle_end * f
        x = cx + r * math.cos(angle)
        y = cy + r * math.sin(angle)
        if i == 0:
            cmds.append(f"M {x:.4f},{y:.4f}")
        else:
            cmds.append(f"L {x:.4f},{y:.4f}")
            
    return " ".join(cmds)

def _rounded_rect_path(cx: float, cy: float, w: float, h: float, r: float) -> str:
    """Genera un path SVG de un rectángulo redondeado con centro en cx, cy."""
    x0, x1 = cx - w/2, cx + w/2
    y0, y1 = cy - h/2, cy + h/2
    return (f"M {cx},{y1} "
            f"L {x1-r},{y1} Q {x1},{y1} {x1},{y1-r} "
            f"L {x1},{y0+r} Q {x1},{y0} {x1-r},{y0} "
            f"L {x0+r},{y0} Q {x0},{y0} {x0},{y0+r} "
            f"L {x0},{y1-r} Q {x0},{y1} {x0+r},{y1} Z")

def _build_animated_figure(tel: pd.DataFrame, lap_time_seg: float) -> go.Figure:
    # ── Datos ─────────────────────────────────────────────────────────────────
    x     = tel["X"].values
    y     = tel["Y"].values
    fl    = tel["load_fl"].values
    fr    = tel["load_fr"].values
    rl    = tel["load_rl"].values
    rr    = tel["load_rr"].values
    g_tot = tel["g_total"].values

    N            = len(x)
    step         = max(1, N // TARGET_FRAMES)
    frame_indices = list(range(0, N, step))
    n_frames     = len(frame_indices)

    # ── Imagen silueta ────────────────────────────────────────────────────────
    _here     = _os.path.dirname(_os.path.abspath(__file__))
    img_path  = _os.path.join(_here, "assets", "f1silueta.png")
    img_uri   = _img_to_uri(img_path)

    # ── Subplots: mapa izq (65%) + silueta der (35%) ─────────────────────────
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.65, 0.35],
        specs=[[{"type": "xy"}, {"type": "xy"}]],
        horizontal_spacing=0.04,
    )

    # ── Traza 0: trazado base ─────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="lines",
        line=dict(color=GREY_TRACK, width=4),
        hoverinfo="skip", showlegend=False,
    ), row=1, col=1)

    # ── Traza 1: trail animado ────────────────────────────────────────────────
    fig.add_trace(go.Scattergl(
        x=[], y=[], mode="lines",
        line=dict(color=F1_WHITE, width=3),
        hoverinfo="skip", showlegend=False,
    ), row=1, col=1)

    # ── Traza 2: cursor ───────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=[], y=[], mode="markers",
        marker=dict(size=13, color=F1_WHITE, symbol="circle",
                    line=dict(color=F1_RED, width=2)),
        hoverinfo="skip", showlegend=False,
    ), row=1, col=1)

    # ── Traza 3: scatter invisible para fijar el eje del panel de silueta ─────
    # Coordenadas: x en [-4, 4], y en [-5.5, 5.5] — espacio para los gauges
    fig.add_trace(go.Scatter(
        x=[-4, 4], y=[-5.5, 5.5],
        mode="markers", marker=dict(color="rgba(0,0,0,0)", size=1),
        hoverinfo="skip", showlegend=False,
    ), row=1, col=2)
    
    # ── Traza 4: hover de frenos (marcadores invisibles sobre los discos) ────
    fig.add_trace(go.Scatter(
        x=[], y=[], mode="markers",
        marker=dict(size=18, opacity=0),
        hoverinfo="text", text=[], showlegend=False,
    ), row=1, col=2)

    # ── Posiciones de las ruedas en el espacio del eje x2/y2 ─────────────────
    # La imagen es 894×1264 px, con el coche centrado.
    # Mapeamos a x∈[-4,4], y∈[-5.5,5.5] (0,0 = centro de la imagen).
    # Ruedas delanteras (arriba en imagen): y ≈ +2.1
    # Ruedas traseras   (abajo en imagen):  y ≈ -2.3
    # Ruedas izquierdas (izq vista cenital): x ≈ -1.6
    # Ruedas derechas:                       x ≈ +2.7
    TH = 1.25   # alto del neumático
    TYRES = {
        "fl": (-2.2,  2.5),
        "fr": ( 0.7,  2.5),
        "rl": (-2.29, -3.1),
        "rr": ( 0.74, -3.1),
    }
    # Posiciones de los círculos de progreso (más separados que la rueda)
    GAUGE_R  = 0.72   # radio del arco
    GAUGE_TRACK = 0.18  # grosor del track del arco (círculo gris base)
    GAUGE_OFFSET_X = 1.6  # offset lateral para separarlo de la rueda
    
    # Disco de freno: rectángulo adyacente al borde interior de cada neumático, separado unos píxeles
    DISC_W = 0.15  # ancho del disco
    DISC_H = 0.5    # misma altura que el neumático
    DISC_GAP = 0.12 # píxeles/unidades de separación respecto a la rueda
    BRAKES = {}
    for _k, (_cx, _cy) in TYRES.items():
        _tw = 0.65 if _k.startswith("r") else 0.55
        # Left-side tyres: disco en el borde derecho (interior) + gap
        # Right-side tyres: disco en el borde izquierdo (interior) - gap
        if _k.endswith("l"):
            _bx = _cx + _tw / 2 + DISC_GAP + DISC_W / 2
        else:
            _bx = _cx - _tw / 2 - DISC_GAP - DISC_W / 2
        BRAKES[_k] = (_bx, _cy)

    # ── Construcción de frames ────────────────────────────────────────────────
    frames = []

    for i, idx in enumerate(frame_indices):
        idx = min(idx, N - 1)

        trail_x = x[:idx].tolist()
        trail_y = y[:idx].tolist()
        c_cursor = _load_to_color(float(np.clip(g_tot[idx] / 4.0, 0, 1)))

        loads = {
            "fl": float(fl[idx]),
            "fr": float(fr[idx]),
            "rl": float(rl[idx]),
            "rr": float(rr[idx]),
        }
        colors = {k: _load_to_color(v) for k, v in loads.items()}

        t_loads = {
            "fl": float(tel["temp_fl"].values[idx]),
            "fr": float(tel["temp_fr"].values[idx]),
            "rl": float(tel["temp_rl"].values[idx]),
            "rr": float(tel["temp_rr"].values[idx]),
        }
        
        # ── Shapes del frame: rectángulos de color sobre las ruedas +
        #    pistas base grises de los arcos + discos de freno ─────────
        frame_shapes = []
        for key, (cx, cy) in TYRES.items():
            col = colors[key]
            # Rectángulo de color sobre la rueda (sin borde, esquinas redondeadas)
            tw = 0.65 if key.startswith("r") else 0.55
            tire_path = _rounded_rect_path(cx, cy, tw, TH, r=0.15)
            frame_shapes.append(dict(
                type="path",
                path=tire_path,
                line=dict(width=0, color="rgba(0,0,0,0)"),
                fillcolor=col,
                opacity=0.82,
                xref="x2", yref="y2",
                layer="above",
            ))
            
            # Disco de Freno (rectángulo adyacente al neumático)
            bx, by = BRAKES[key]
            disc_col = _temp_to_color(t_loads[key])
            disc_path = _rounded_rect_path(bx, by, DISC_W, DISC_H, r=0.04)
            frame_shapes.append(dict(
                type="path",
                path=disc_path,
                line=dict(width=0.5, color="rgba(0,0,0,0.4)"),
                fillcolor=disc_col,
                xref="x2", yref="y2",
                layer="above",
            ))
            
            # Centro del arco (al lado de la rueda)
            gx = cx - GAUGE_OFFSET_X if cx < 0 else cx + GAUGE_OFFSET_X
            gy = cy

            # Pista base del arco (círculo completo gris oscuro)
            r = GAUGE_R
            frame_shapes.append(dict(
                type="circle",
                x0=gx - r, y0=gy - r,
                x1=gx + r, y1=gy + r,
                line=dict(width=8, color="rgba(60,60,70,0.9)"),
                fillcolor="rgba(0,0,0,0)",
                xref="x2", yref="y2",
                layer="above",
            ))
            # Arco de progreso SVG
            pct = loads[key]
            arc = _arc_path(gx, gy, r, pct)
            if arc:
                frame_shapes.append(dict(
                    type="path",
                    path=arc,
                    line=dict(width=8, color=col),
                    fillcolor="rgba(0,0,0,0)",
                    xref="x2", yref="y2",
                    layer="above",
                ))

        # ── Anotaciones: solo el % en blanco, muy separado del arco ──────────
        frame_annotations = []
        for key, (cx, cy) in TYRES.items():
            gx = cx - GAUGE_OFFSET_X if cx < 0 else cx + GAUGE_OFFSET_X
            gy = cy
            pct_val = loads[key] * 100
            # Texto centrado dentro del círculo
            frame_annotations.append(dict(
                x=gx, y=gy,
                text=f"<b>{pct_val:.0f}%</b>",
                xref="x2", yref="y2",
                showarrow=False,
                font=dict(color=F1_WHITE, family=MONO_FONT, size=16),
            ))

        # Scatter de hover de frenos (4 puntos, uno por disco)
        brake_x_pts = [BRAKES[k][0] for k in ["fl", "fr", "rl", "rr"]]
        brake_y_pts = [BRAKES[k][1] for k in ["fl", "fr", "rl", "rr"]]
        brake_texts = [f"<b>{t_loads[k]:.0f} °C</b>" for k in ["fl", "fr", "rl", "rr"]]
        brake_colors_list = [_temp_to_color(t_loads[k]) for k in ["fl", "fr", "rl", "rr"]]

        frame_data = [
            go.Scattergl(x=trail_x, y=trail_y, line=dict(color=c_cursor)),
            go.Scatter(x=[float(x[idx])], y=[float(y[idx])],
                       marker=dict(color=c_cursor)),
            go.Scatter(
                x=brake_x_pts, y=brake_y_pts,
                text=brake_texts,
                hoverinfo="text",
                marker=dict(size=18, color=brake_colors_list, opacity=0),
            ),
        ]

        frames.append(go.Frame(
            data=frame_data,
            layout={"shapes": frame_shapes, "annotations": frame_annotations},
            traces=[1, 2, 4],
            name=str(i),
        ))

    fig.frames = frames

    # ── Slider y play/pause ───────────────────────────────────────────────────
    frame_duration_ms = max(16, int((lap_time_seg * 1000) / n_frames))
    dist_arr  = tel["Distance"].values
    step_every = max(1, n_frames // 80)

    slider_steps = []
    for i in range(0, n_frames, step_every):
        idx_i  = frame_indices[min(i, n_frames - 1)]
        dist_i = float(dist_arr[min(idx_i, N - 1)])
        slider_steps.append(dict(
            args=[[str(i)], {"frame": {"duration": frame_duration_ms, "redraw": True},
                             "mode": "immediate", "transition": {"duration": 0}}],
            label=f"{dist_i/1000:.1f}km", method="animate",
        ))

    updatemenus = [dict(
        type="buttons", showactive=False, y=0.0, x=-0.05,
        xanchor="right", yanchor="bottom",
        buttons=[
            dict(label="▶ PLAY", method="animate",
                 args=[None, {"frame": {"duration": frame_duration_ms, "redraw": True},
                              "fromcurrent": True, "transition": {"duration": 0}}]),
            dict(label="⏸ PAUSE", method="animate",
                 args=[[None], {"frame": {"duration": 0, "redraw": False},
                                "mode": "immediate", "transition": {"duration": 0}}]),
        ],
    )]

    sliders = [dict(
        active=0, transition=dict(duration=0),
        pad=dict(b=10, t=50), len=0.9, x=0.1, y=0.0,
        steps=slider_steps,
        bgcolor="rgba(255,255,255,0.05)",
        bordercolor="rgba(255,255,255,0.1)",
        tickcolor="rgba(255,255,255,0.3)",
        font=dict(size=8, color="rgba(255,255,255,0.4)", family=MONO_FONT),
    )]

    # ── Layout global ─────────────────────────────────────────────────────────
    x_min, x_max = float(np.min(x)), float(np.max(x))
    y_min, y_max = float(np.min(y)), float(np.max(y))
    mx = (x_max - x_min) * 0.05
    my = (y_max - y_min) * 0.05

    # Imagen de silueta como fondo del subgráfico derecho.
    # sizing="contain" con xref/yref="paper" requiere dominio del subplot.
    # Usamos xref="x2 domain", yref="y2 domain" para anclarla al panel.
    layout_image = [dict(
        source=img_uri,
        xref="x2 domain", yref="y2 domain",
        x=0.045, y=1,
        sizex=1, sizey=1,
        xanchor="left", yanchor="top",
        sizing="contain",
        opacity=1.0,
        layer="below",
    )]

    # Shapes iniciales (estado frame 0)
    init_loads = {"fl": float(fl[0]), "fr": float(fr[0]),
                  "rl": float(rl[0]), "rr": float(rr[0])}
    
    t_loads_init = {
        "fl": float(tel["temp_fl"].values[0]),
        "fr": float(tel["temp_fr"].values[0]),
        "rl": float(tel["temp_rl"].values[0]),
        "rr": float(tel["temp_rr"].values[0]),
    }
                  
    init_colors = {k: _load_to_color(v) for k, v in init_loads.items()}
    init_shapes = []
    for key, (cx, cy) in TYRES.items():
        col = init_colors[key]
        r   = GAUGE_R
        tw = 0.65 if key.startswith("r") else 0.55
        tire_path = _rounded_rect_path(cx, cy, tw, TH, r=0.15)
        init_shapes.append(dict(
            type="path",
            path=tire_path,
            line=dict(width=0, color="rgba(0,0,0,0)"),
            fillcolor=col, opacity=0.82,
            xref="x2", yref="y2", layer="above",
        ))
        
        # Disco de freno inicial (rectángulo adyacente)
        bx, by = BRAKES[key]
        disc_col = _temp_to_color(t_loads_init[key])
        disc_path = _rounded_rect_path(bx, by, DISC_W, DISC_H, r=0.04)
        init_shapes.append(dict(
            type="path",
            path=disc_path,
            line=dict(width=0.5, color="rgba(0,0,0,0.4)"),
            fillcolor=disc_col,
            xref="x2", yref="y2", layer="above",
        ))
        
        gx = cx - GAUGE_OFFSET_X if cx < 0 else cx + GAUGE_OFFSET_X
        gy = cy

        init_shapes.append(dict(
            type="circle",
            x0=gx - r, y0=gy - r, x1=gx + r, y1=gy + r,
            line=dict(width=8, color="rgba(60,60,70,0.9)"),
            fillcolor="rgba(0,0,0,0)",
            xref="x2", yref="y2", layer="above",
        ))
        arc = _arc_path(gx, gy, r, init_loads[key])
        if arc:
            init_shapes.append(dict(
                type="path", path=arc,
                line=dict(width=8, color=col),
                fillcolor="rgba(0,0,0,0)",
                xref="x2", yref="y2", layer="above",
            ))

    init_annotations = []
    for key, (cx, cy) in TYRES.items():
        gx = cx - GAUGE_OFFSET_X if cx < 0 else cx + GAUGE_OFFSET_X
        gy = cy
        init_annotations.append(dict(
            x=gx, y=gy,
            text=f"<b>{init_loads[key]*100:.0f}%</b>",
            xref="x2", yref="y2",
            showarrow=False,
            font=dict(color=F1_WHITE, family=MONO_FONT, size=16),
        ))

    fig.update_layout(
        height=650,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=MONO_FONT, color=F1_WHITE, size=11),
        updatemenus=updatemenus,
        sliders=sliders,
        showlegend=False,
        margin=dict(l=40, r=10, t=10, b=40),
        images=layout_image,
        shapes=init_shapes,
        annotations=init_annotations,
        xaxis=dict(
            range=[x_min - mx, x_max + mx],
            showgrid=False, zeroline=False, visible=False,
            scaleanchor="y", scaleratio=1,
        ),
        yaxis=dict(
            range=[y_min - my, y_max + my],
            showgrid=False, zeroline=False, visible=False,
        ),
        xaxis2=dict(range=[-4, 4], showgrid=False, zeroline=False, visible=False, fixedrange=True),
        yaxis2=dict(range=[-5.5, 5.5], showgrid=False, zeroline=False, visible=False,
                    scaleanchor="x2", scaleratio=1, fixedrange=True),
    )

    return fig



# ─────────────────────────────────────────────────────────────────────────────
# PASO 5 — GRÁFICO DE CARGA VS DISTANCIA
# ─────────────────────────────────────────────────────────────────────────────

def _build_load_vs_distance(tel: pd.DataFrame, circuit_info=None) -> go.Figure:
    dist = tel["Distance"].values
    
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=["FL LOAD", "FR LOAD", "RL LOAD", "RR LOAD"]
    )

    channels = [
        ("load_fl", "FL Load", ACCENT_CYAN, 1),
        ("load_fr", "FR Load", ACCENT_AMBER, 2),
        ("load_rl", "RL Load", ACCENT_GREEN, 3),
        ("load_rr", "RR Load", ACCENT_PURPLE, 4),
    ]

    for col, name, color, row in channels:
        fig.add_trace(go.Scatter(
            x=dist, y=tel[col].values * 100, # %
            mode="lines",
            line=dict(color=color, width=1.5),
            fill="tozeroy",
            fillcolor=f"rgba({','.join(map(str, _hex_to_rgb(color)))}, 0.05)",
            name=name,
            hovertemplate="<b>%{y:.1f}%</b> " + name + "<extra></extra>"
        ), row=row, col=1)

    for ann in fig.layout.annotations:
        ann.update(font=dict(family=MONO_FONT, size=9, color="rgba(255,255,255,0.40)"), x=0, xanchor="left")

    if circuit_info is not None:
        dist_max = float(dist[-1])
        for _, row in circuit_info.corners.iterrows():
            d_c = float(row["Distance"])
            if d_c <= dist_max:
                for r in range(1, 5):
                    fig.add_vline(x=d_c, line_width=0.6, line_dash="dash",
                                  line_color="rgba(255,255,255,0.1)", row=r, col=1)
                fig.add_annotation(
                    x=d_c, y=1.02, xref="x", yref="paper",
                    text=f"T{int(row['Number'])}",
                    showarrow=False,
                    font=dict(size=8, color="rgba(255,255,255,0.35)", family=MONO_FONT),
                )

    ax_common = dict(
        showgrid=True, gridcolor="rgba(255,255,255,0.05)",
        zeroline=False, range=[0, 105], ticksuffix=" %",
        tickfont=dict(size=8, color="rgba(255,255,255,0.4)", family=MONO_FONT)
    )

    fig.update_layout(
        height=650,
        paper_bgcolor=BG_DARK,
        plot_bgcolor=BG_PANEL,
        font=dict(family=MONO_FONT, color=F1_WHITE, size=10),
        hoverlabel=dict(bgcolor=BG_SURFACE, bordercolor="rgba(255,255,255,0.15)", font=dict(family=MONO_FONT, color=F1_WHITE, size=10)),
        hovermode="x unified",
        margin=dict(l=54, r=20, t=50, b=40),
        showlegend=False,
        xaxis4=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", zeroline=False, title="Distancia (m)", ticksuffix=" m", tickfont=dict(size=8, color="rgba(255,255,255,0.4)", family=MONO_FONT)),
    )
    
    for i in range(1, 5):
        fig.update_layout(**{f"xaxis{i if i>1 else ''}": dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", zeroline=False, tickfont=dict(size=8, color="rgba(255,255,255,0.4)", family=MONO_FONT))})
        fig.update_layout(**{f"yaxis{i if i>1 else ''}": ax_common})

    return fig

# ─────────────────────────────────────────────────────────────────────────────
# PASO 6.5 — GRÁFICO DE DINÁMICA VEHICULAR
# ─────────────────────────────────────────────────────────────────────────────

def _build_dynamics_figure(tel: pd.DataFrame, circuit_info=None) -> go.Figure:
    """Gráfico de temperaturas de freno y balance dinámico vs distancia."""
    dist = tel["Distance"].values

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=["TEMPERATURA DE FRENOS", "BALANCE DINÁMICO (SUBVIRAJE / SOBREVIRAJE)"],
        row_heights=[0.55, 0.45],
    )

    # ── Subplot 1: Temperaturas ─────────────────────────
    temp_channels = [
        ("temp_fl", "FL Brake", ACCENT_CYAN),
        ("temp_fr", "FR Brake", ACCENT_AMBER),
        ("temp_rl", "RL Brake", ACCENT_GREEN),
        ("temp_rr", "RR Brake", ACCENT_PURPLE),
    ]
    for col_name, name, color in temp_channels:
        fig.add_trace(go.Scatter(
            x=dist, y=tel[col_name].values,
            mode="lines",
            line=dict(color=color, width=1.5),
            name=name,
            hovertemplate="<b>%{y:.0f} °C</b> " + name + "<extra></extra>",
        ), row=1, col=1)

    # ── Subplot 2: Balance Dinámico ─────────────────────
    balance = tel["dyn_balance"].values
    bal_pos = np.clip(balance, 0, None)
    bal_neg = np.clip(balance, None, 0)

    fig.add_trace(go.Scatter(
        x=dist, y=bal_pos,
        mode="lines", line=dict(color="rgba(0,150,255,0.7)", width=0.5),
        fill="tozeroy", fillcolor="rgba(0,150,255,0.15)",
        name="Subviraje",
        hovertemplate="<b>Subviraje: %{y:.1f}%</b><extra></extra>",
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=dist, y=bal_neg,
        mode="lines", line=dict(color="rgba(255,100,50,0.7)", width=0.5),
        fill="tozeroy", fillcolor="rgba(255,100,50,0.15)",
        name="Sobreviraje",
        hovertemplate="<b>Sobreviraje: %{y:.1f}%</b><extra></extra>",
    ), row=2, col=1)

    # ── Curvas de circuito ──────────────────────────────
    if circuit_info is not None:
        dist_max = float(dist[-1])
        for _, row_data in circuit_info.corners.iterrows():
            d_c = float(row_data["Distance"])
            if d_c <= dist_max:
                for r in range(1, 3):
                    fig.add_vline(x=d_c, line_width=0.6, line_dash="dash",
                                  line_color="rgba(255,255,255,0.1)", row=r, col=1)
                fig.add_annotation(
                    x=d_c, y=1.02, xref="x", yref="paper",
                    text=f"T{int(row_data['Number'])}",
                    showarrow=False,
                    font=dict(size=8, color="rgba(255,255,255,0.35)", family=MONO_FONT),
                )

    for ann in fig.layout.annotations:
        ann.update(font=dict(family=MONO_FONT, size=9, color="rgba(255,255,255,0.40)"), x=0, xanchor="left")

    fig.update_layout(
        height=500,
        paper_bgcolor=BG_DARK,
        plot_bgcolor=BG_PANEL,
        font=dict(family=MONO_FONT, color=F1_WHITE, size=10),
        hoverlabel=dict(bgcolor=BG_SURFACE, bordercolor="rgba(255,255,255,0.15)",
                        font=dict(family=MONO_FONT, color=F1_WHITE, size=10)),
        hovermode="x unified",
        margin=dict(l=54, r=20, t=50, b=40),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    font=dict(size=9, color="rgba(255,255,255,0.5)")),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", zeroline=False,
                   title="°C", ticksuffix=" °C",
                   tickfont=dict(size=8, color="rgba(255,255,255,0.4)", family=MONO_FONT)),
        yaxis2=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", zeroline=True,
                    zerolinecolor="rgba(255,255,255,0.2)", zerolinewidth=1,
                    title="Balance %", ticksuffix=" %",
                    tickfont=dict(size=8, color="rgba(255,255,255,0.4)", family=MONO_FONT)),
        xaxis2=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", zeroline=False,
                    title="Distancia (m)", ticksuffix=" m",
                    tickfont=dict(size=8, color="rgba(255,255,255,0.4)", family=MONO_FONT)),
    )

    return fig

# ─────────────────────────────────────────────────────────────────────────────
# PASO 7 — MÉTRICAS KPI
# ─────────────────────────────────────────────────────────────────────────────

def _render_load_metrics(tel: pd.DataFrame) -> None:
    m_fl = tel["load_fl"].max() * 100
    m_fr = tel["load_fr"].max() * 100
    m_rl = tel["load_rl"].max() * 100
    m_rr = tel["load_rr"].max() * 100
    avg_load = tel[["load_fl", "load_fr", "load_rl", "load_rr"]].mean().mean() * 100
    p_gtot = tel["g_total"].max()

    cols = st.columns(6)
    cols[0].metric("FL Max Exig.", f"{m_fl:.0f} %")
    cols[1].metric("FR Max Exig.", f"{m_fr:.0f} %")
    cols[2].metric("RL Max Exig.", f"{m_rl:.0f} %")
    cols[3].metric("RR Max Exig.", f"{m_rr:.0f} %")
    cols[4].metric("Avg Exigencia", f"{avg_load:.0f} %")
    cols[5].metric("Peak G Total", f"{p_gtot:.2f} G")

# ─────────────────────────────────────────────────────────────────────────────
# PUNTO DE ENTRADA PÚBLICO
# ─────────────────────────────────────────────────────────────────────────────

def render_tyre_load(session: Session, driver: str, lap_number: int) -> None:
    with st.spinner(f"Calculando modelo físico avanzado — {driver} · V{lap_number}"):
        try:
            laps = session.laps.pick_drivers(driver)
            lap = laps[laps["LapNumber"] == lap_number].iloc[0]
            tel = lap.get_telemetry().add_distance()
            lap_time_seg = lap.get("LapTime").total_seconds()
            compound = lap.get("Compound", "UNKNOWN")
        except Exception as e:
            st.error(f"❌ No se pudo cargar la telemetría: {e}")
            return

        if tel.empty or "X" not in tel.columns:
            st.error("❌ Telemetría sin coordenadas XY.")
            return

        # Computations iniciales para determinar capacidad de grip
        tel = _calculate_g_forces(tel)
        
        # Aero extraction
        speed_kmh_all = tel["Speed"].values
        g_lat_abs_all = np.abs(tel["g_lat"].values)
        try:
            speed_kmh, g_lat_abs, _ = _filter_cornering(speed_kmh_all, g_lat_abs_all)
            envelope = _compute_envelope(speed_kmh, g_lat_abs, 90.0)
            reg = _fit_regression(envelope)
            alpha = reg["alpha"] if reg else float(np.nanmax(g_lat_abs_all))
            beta = reg["beta"] if reg and reg["beta"] > 0 else 0.0
            if np.isnan(alpha) or alpha <= 0:
                alpha = 1.5
        except Exception:
            alpha = 1.5
            beta = 0.0

    # 1. Header (Ahora con variables definidas)
    st.header(f"Exigencia de Neumáticos ({compound})")
    st.caption(f"Grip Utilization: Nivel de estrés respecto al límite de fricción dinámico. El modelo extrae automáticamente el agarre del compuesto {compound} a partir de los datos reales de esta vuelta.")
    st.divider()

    col1, col2 = st.columns([2, 5])
    with col1:
        brake_bias_pct = st.slider("Brake Bias (%)", min_value=50.0, max_value=70.0, value=58.0, step=0.5,
                                   help="Reparto de fuerza de frenada hacia el tren delantero.")
    brake_bias = brake_bias_pct / 100.0
    
    with col2:
        st.caption(f"Basado en el análisis de telemetría, se ha detectado un grip mecánico base (α) de {alpha:.2f} G para el compuesto {compound}. El efecto del DRS y el Brake Bias se calculan sobre este límite real demostrado.")

    # 2. Computación final de cargas usando el Brake Bias del slider
    tel = _compute_wheel_loads(tel, alpha, beta, brake_bias)
    
    # 2.5 Cálculos de Dinámica Vehicular y Termodinámica
    tel = _compute_brake_thermals(tel)
    tel = _compute_dynamic_balance(tel)

    try:
        circuit_info = session.get_circuit_info()
    except Exception:
        circuit_info = None

    # 3. KPIs
    _render_load_metrics(tel)
    st.divider()

    # 4. Animation
    st.markdown(
        "<p style='font-family: JetBrains Mono, monospace; font-size: 11px; letter-spacing: 2px; color: rgba(255,255,255,0.35); margin-bottom: 8px;'>ANIMACIÓN DEL TRAZADO Y SILUETA F1</p>",
        unsafe_allow_html=True
    )
    fig_anim = _build_animated_figure(tel, lap_time_seg)
    st.plotly_chart(fig_anim, use_container_width=True, config={"displayModeBar": False})

    st.divider()

    # 5. Gráfico vs Distancia
    st.markdown(
        "<p style='font-family: JetBrains Mono, monospace; font-size: 11px; letter-spacing: 2px; color: rgba(255,255,255,0.35); margin-bottom: 8px;'>EXIGENCIA RELATIVA vs DISTANCIA</p>",
        unsafe_allow_html=True
    )
    fig_line = _build_load_vs_distance(tel, circuit_info)
    st.plotly_chart(fig_line, use_container_width=True, config={"displayModeBar": False})

    st.divider()

    # 6. Gráficos de Dinámica (Frenos y Balance)
    st.markdown(
        "<p style='font-family: JetBrains Mono, monospace; font-size: 11px; letter-spacing: 2px; color: rgba(255,255,255,0.35); margin-bottom: 8px;'>DINÁMICA VEHICULAR: FRENOS Y BALANCE (SUBVIRAJE/SOBREVIRAJE)</p>",
        unsafe_allow_html=True
    )
    fig_dyn = _build_dynamics_figure(tel, circuit_info)
    st.plotly_chart(fig_dyn, use_container_width=True, config={"displayModeBar": False})
