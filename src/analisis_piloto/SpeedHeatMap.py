import fastf1
import fastf1.plotting
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


# ─── CARGA DE DATOS ───────────────────────────────────────────────────────────
# Separamos carga y render porque @st.cache_data no puede decorar funciones
# que contienen llamadas a st.* (métricas, gráficos, etc.)
# El guión bajo en _sesion evita que Streamlit intente hashear el objeto FastF1
@st.cache_data(show_spinner=True)
def _load_data(_sesion, driver_code, lap_number):
    vueltas_piloto = _sesion.laps.pick_drivers(driver_code)
    Pole           = vueltas_piloto[vueltas_piloto['LapNumber'] == lap_number].iloc[0]
    telemetria     = Pole.get_telemetry()
    driver_info    = _sesion.get_driver(Pole['Driver'])
    return Pole, telemetria, driver_info


# ─── INTERPOLADOR DE COLOR PLASMA ─────────────────────────────────────────────
# Plotly no expone la escala Plasma como función directa, así que la
# reimplementamos manualmente con 5 paradas clave de la paleta oficial.
# Entrada: valor numérico + rango [vmin, vmax] → Salida: string 'rgb(r,g,b)'
def _speed_to_color(val, vmin, vmax):
    t = (val - vmin) / (vmax - vmin) if vmax > vmin else 0
    t = np.clip(t, 0, 1)
    colors = [
        (0.0,  (13,  8,   135)),   # azul oscuro  — velocidad mínima
        (0.25, (126, 3,   168)),   # violeta       — lento
        (0.5,  (204, 71,  120)),   # rosa-rojo     — medio
        (0.75, (248, 149, 64)),    # naranja       — rápido
        (1.0,  (240, 249, 33)),    # amarillo      — velocidad máxima
    ]
    for i in range(len(colors) - 1):
        t0, c0 = colors[i]
        t1, c1 = colors[i + 1]
        if t0 <= t <= t1:
            f = (t - t0) / (t1 - t0)
            r = int(c0[0] + f * (c1[0] - c0[0]))
            g = int(c0[1] + f * (c1[1] - c0[1]))
            b = int(c0[2] + f * (c1[2] - c0[2]))
            return f'rgb({r},{g},{b})'
    return 'rgb(240,249,33)'


# ─── HELPER: OFFSET EXTERIOR ──────────────────────────────────────────────────
# Dado un índice en el trazado, calcula un punto desplazado hacia el EXTERIOR
# del circuito a una distancia `offset` en unidades del sistema de coordenadas.
# Usado para meta, DRS y frenadas — cualquier anotación que no deba pisar el trazado.
def _offset_exterior(idx, x, y, mid_x, mid_y, offset):
    idx_next = min(idx + 5, len(x) - 1)
    idx_prev = max(idx - 5, 0)
    dx = x[idx_next] - x[idx_prev]
    dy = y[idx_next] - y[idx_prev]
    # Vector perpendicular al trazado
    nx, ny = -dy, dx
    mag = np.sqrt(nx**2 + ny**2)
    if mag != 0:
        nx /= mag; ny /= mag
    # Test: si el punto offset se acerca al centro, invertimos el vector
    pos_test_x = x[idx] + nx * offset
    pos_test_y = y[idx] + ny * offset
    dc_actual  = np.sqrt((x[idx]    - mid_x)**2 + (y[idx]    - mid_y)**2)
    dc_test    = np.sqrt((pos_test_x - mid_x)**2 + (pos_test_y - mid_y)**2)
    if dc_test < dc_actual:
        nx, ny = -nx, -ny
    return x[idx] + nx * offset, y[idx] + ny * offset


# ─── RENDER PRINCIPAL ─────────────────────────────────────────────────────────
def render_speed_heatmap(sesion, driver_code, show_corners, lap_number):

    Pole, telemetria, driver_info = _load_data(sesion, driver_code, lap_number)

    # Metadatos de la vuelta
    Driver       = Pole['Driver']
    Team         = Pole['Team']
    TiempoVuelta = Pole['LapTime']
    compound     = Pole['Compound']
    tyre_age     = Pole['TyreLife']

    # Arrays de telemetría
    x          = np.array(telemetria['X'].values)
    y          = np.array(telemetria['Y'].values)
    speed      = np.array(telemetria['Speed'].values)
    dist_array = np.array(telemetria['Distance'].values)
    brake      = np.array(telemetria['Brake'].values)
    drs        = np.array(telemetria['DRS'].values)

    # Centro geométrico del circuito — usado para calcular offsets exteriores
    mid_x, mid_y = np.mean(x), np.mean(y)

    circuit_info = sesion.get_circuit_info()
    event_name   = sesion.event['EventName']
    ses_name     = sesion.name

    vmin, vmax = speed.min(), speed.max()
    idx_max    = int(np.argmax(speed))

    # ── Métricas ──────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Velocidad máxima", f"{vmax:.0f} km/h")
    c2.metric("Velocidad mínima", f"{vmin:.0f} km/h")
    c3.metric("Velocidad media",  f"{speed.mean():.0f} km/h")
    c4.metric("Tiempo de vuelta", str(TiempoVuelta)[7:15])

    # ── Figura ────────────────────────────────────────────────────────────────
    fig = go.Figure()

    # Trazado base — capa oscura de fondo que da grosor visual al circuito
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode='lines',
        line=dict(color='#2A2A3A', width=10),
        hoverinfo='skip',
        showlegend=False,
    ))

    # ── Gradiente de velocidad ────────────────────────────────────────────────
    N_BANDAS = 30
    bins = np.linspace(vmin, vmax, N_BANDAS + 1)

    for i in range(N_BANDAS):
        v_mid = (bins[i] + bins[i + 1]) / 2
        color = _speed_to_color(v_mid, vmin, vmax)
        mask  = (speed[:-1] >= bins[i]) & (speed[:-1] < bins[i + 1])
        if not mask.any():
            continue
        xs, ys, hover = [], [], []
        for idx in np.where(mask)[0]:
            xs    += [x[idx], x[idx + 1], None]
            ys    += [y[idx], y[idx + 1], None]
            hover += [f"{speed[idx]:.0f} km/h", f"{speed[idx+1]:.0f} km/h", None]
        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            mode='lines',
            line=dict(color=color, width=4),
            hovertext=hover,
            hoverinfo='text',
            showlegend=False,
        ))

    # Colorbar — scatter invisible que solo sirve para mostrar la escala de color
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='markers',
        marker=dict(
            size=0,
            color=[vmin, vmax],
            colorscale='Plasma',
            showscale=True,
            colorbar=dict(
                title=dict(text="km/h", font=dict(color='#D0D2DC', family='Titillium Web, sans-serif')),
                tickfont=dict(color='#D0D2DC', family='Titillium Web, sans-serif', size=10),
                thickness=20, len=0.6, outlinewidth=0,
            ),
        ),
        hoverinfo='skip',
        showlegend=False,
    ))

    # ── Velocidad máxima ──────────────────────────────────────────────────────
    # Estrella amarilla en el punto de velocidad punta de la vuelta
    fig.add_trace(go.Scatter(
        x=[x[idx_max]], y=[y[idx_max]],
        mode='markers+text',
        marker=dict(color='#FFE81E', size=10, symbol='star'),
        text=[f"MÁX {vmax:.0f}"],
        textposition='top center',
        textfont=dict(color='#FFE81E', size=10, family='Titillium Web, sans-serif'),
        hoverinfo='skip',
        showlegend=False,
    ))

    # ── Línea de meta con offset ──────────────────────────────────────────────
    # x[0]/y[0] es siempre el primer punto de telemetría = línea de meta.
    # Calculamos el offset exterior para que el marcador no pise el trazado.
    mx, my = _offset_exterior(0, x, y, mid_x, mid_y, offset=0)
    fig.add_trace(go.Scatter(
        x=[mx], y=[my],
        mode='markers+text',
        marker=dict(
            color='#FFFFFF', size=14,
            symbol='line-ew',
            line=dict(width=3, color='#FFFFFF'),
        ),
        hovertemplate='<b>Línea de meta</b><extra></extra>',
        showlegend=True,
        name='Meta',
    ))

    # ── Puntos de frenada y DRS — mismo bloque, mismo offset ─────────────────
    # Detectamos flancos de subida (0→1) en brake y en drs para localizar
    # el momento exacto en que el piloto empieza a frenar o activa el DRS.
    # Ambos usan _offset_exterior con distancia diferente para separarse visualmente.

    # Flancos de frenada: brake pasa de 0 a >0
    brake_bin    = (brake > 0).astype(int)
    brake_starts = np.where(np.diff(brake_bin) > 0)[0]

    # Flancos DRS: drs pasa de ≤8 a >8 (activación del alerón)
    drs_bin      = (drs > 8).astype(int)
    drs_starts   = np.where(np.diff(drs_bin) > 0)[0]

    # Puntos de frenada — triángulos rojos hacia abajo, offset 160
    if len(brake_starts) > 0:
        bx = [_offset_exterior(i, x, y, mid_x, mid_y, 160)[0] for i in brake_starts]
        by = [_offset_exterior(i, x, y, mid_x, mid_y, 160)[1] for i in brake_starts]
        fig.add_trace(go.Scatter(
            x=bx, y=by,
            mode='markers',
            marker=dict(color='#E8002D', size=8, symbol='triangle-down'),
            name='Frenada',
            hovertemplate='<b>Punto de frenada</b><extra></extra>',
            showlegend=True,
        ))

    # Puntos DRS — círculos teal, offset 120 (más cerca del trazado que las frenadas)
    if len(drs_starts) > 0:
        dx_list = [_offset_exterior(i, x, y, mid_x, mid_y, 120)[0] for i in drs_starts]
        dy_list = [_offset_exterior(i, x, y, mid_x, mid_y, 120)[1] for i in drs_starts]
        fig.add_trace(go.Scatter(
            x=dx_list, y=dy_list,
            mode='markers',
            marker=dict(color='#00D2BE', size=8, symbol='circle'),
            name='DRS activado',
            hovertemplate='<b>DRS activado</b><extra></extra>',
            showlegend=True,
        ))

    # ── Números de curva — lollypops ──────────────────────────────────────────
    if show_corners:
        cx_list, cy_list, num_list = [], [], []
        for _, row in circuit_info.corners.iterrows():
            dist_curva = row['Distance']
            idx        = (np.abs(dist_array - dist_curva)).argmin()
            ox, oy     = _offset_exterior(idx, x, y, mid_x, mid_y, 200)
            cx_list.append(ox)
            cy_list.append(oy)
            num_list.append(str(row['Number']))

        fig.add_trace(go.Scatter(
            x=cx_list, y=cy_list,
            mode='markers+text',
            marker=dict(
                size=20, color='white',
                line=dict(color='#E8002D', width=2),
                symbol='circle',
            ),
            text=num_list,
            textfont=dict(color='#15151E', size=9, family='Titillium Web, sans-serif'),
            textposition='middle center',
            hoverinfo='skip',
            showlegend=False,
        ))

    # ── Layout ────────────────────────────────────────────────────────────────
    fig.update_layout(
        title=dict(
            text=f"{Driver}  ·  {Team}  ·  {event_name}  ·  {ses_name}",
            x=0.475,xanchor='center',
            font=dict(color='#D0D2DC', size=14, family='Titillium Web, sans-serif'),
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False,
                   scaleanchor='x', scaleratio=1),
        margin=dict(l=0, r=60, t=60, b=0),
        height=600,
        dragmode='pan',
        legend=dict(
            orientation='h',
            yanchor='bottom', y=-0.05,
            xanchor='center', x=0.5,
            font=dict(family='Titillium Web, sans-serif', size=11, color='#D0D2DC'),
            bgcolor='rgba(0,0,0,0)',
        ),
    )

    st.plotly_chart(fig, use_container_width=True,
                    config={'displayModeBar': True, 'scrollZoom': True})