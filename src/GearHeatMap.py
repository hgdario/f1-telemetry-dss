import fastf1
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ─── CARGA DE DATOS ───────────────────────────────────────────────────────────
@st.cache_data(show_spinner=True)
def _load_data(_sesion, driver_code, lap_number):
    vueltas_piloto = _sesion.laps.pick_drivers(driver_code)
    Pole           = vueltas_piloto[vueltas_piloto['LapNumber'] == lap_number].iloc[0]
    telemetria     = Pole.get_telemetry()
    driver_info    = _sesion.get_driver(Pole['Driver'])
    return Pole, telemetria, driver_info

# ─── MAPA DE COLORES DISCRETO PARA MARCHAS ────────────────────────────────────
# A diferencia de la velocidad, las marchas son valores fijos. 
# Asignamos un color vibrante y único a cada marcha (1 a 8).
GEAR_COLORS = {
    1: '#ff4b4b',  # Rojo
    2: '#ff9000',  # Naranja
    3: '#fce83a',  # Amarillo
    4: '#57df50',  # Verde claro
    5: '#00cbf4',  # Cyan
    6: '#3b5af1',  # Azul
    7: '#9c31fa',  # Morado
    8: '#f05df4'   # Magenta
}

def _gear_to_color(g):
    return GEAR_COLORS.get(g, '#ffffff') # Blanco por defecto si hay fallos (ej. neutral)

# ─── HELPER: OFFSET EXTERIOR ──────────────────────────────────────────────────
def _offset_exterior(idx, x, y, mid_x, mid_y, offset):
    idx_next = min(idx + 5, len(x) - 1)
    idx_prev = max(idx - 5, 0)
    dx = x[idx_next] - x[idx_prev]
    dy = y[idx_next] - y[idx_prev]
    
    nx, ny = -dy, dx
    mag = np.sqrt(nx**2 + ny**2)
    if mag != 0:
        nx /= mag; ny /= mag
        
    pos_test_x = x[idx] + nx * offset
    pos_test_y = y[idx] + ny * offset
    dc_actual  = np.sqrt((x[idx]    - mid_x)**2 + (y[idx]    - mid_y)**2)
    dc_test    = np.sqrt((pos_test_x - mid_x)**2 + (pos_test_y - mid_y)**2)
    
    if dc_test < dc_actual:
        nx, ny = -nx, -ny
    return x[idx] + nx * offset, y[idx] + ny * offset

# ─── RENDER PRINCIPAL ─────────────────────────────────────────────────────────
def render_gear_heatmap(sesion, driver_code, show_corners, lap_number):

    Pole, telemetria, driver_info = _load_data(sesion, driver_code, lap_number)

    # Metadatos
    Driver       = Pole['Driver']
    Team         = Pole['Team']
    TiempoVuelta = Pole['LapTime']
    event_name   = sesion.event['EventName']
    ses_name     = sesion.name

    # Arrays
    x          = np.array(telemetria['X'].values)
    y          = np.array(telemetria['Y'].values)
    gear       = np.array(telemetria['nGear'].values)
    dist_array = np.array(telemetria['Distance'].values)
    brake      = np.array(telemetria['Brake'].values)

    mid_x, mid_y = np.mean(x), np.mean(y)
    circuit_info = sesion.get_circuit_info()

    # Cálculos específicos de marchas para los KPIs
    cambios_totales = np.sum(np.diff(gear) != 0) # Cuenta cada vez que la marcha cambia
    marcha_moda = pd.Series(gear[gear > 0]).mode()[0] # Marcha más frecuente (ignorando N)

    # ── Métricas ──────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Marcha más usada", f"{int(marcha_moda)}ª")
    c2.metric("Cambios de marcha", f"{cambios_totales}")
    c3.metric("Marcha máxima", f"{int(gear.max())}ª")
    c4.metric("Tiempo de vuelta", str(TiempoVuelta)[7:15])

    # ── Figura ────────────────────────────────────────────────────────────────
    fig = go.Figure()

    # Trazado base oscuro
    fig.add_trace(go.Scatter(
        x=x, y=y, mode='lines',
        line=dict(color='#2A2A3A', width=10),
        hoverinfo='skip', showlegend=False,
    ))

    # ── Trazado por Marchas (Colores discretos) ───────────────────────────────
    # Iteramos del 1 al 8. Esto es mucho más rápido y limpio que el gradiente.
    for g in range(1, 9):
        mask = (gear[:-1] == g)
        if not mask.any():
            continue
            
        xs, ys, hover = [], [], []
        # Encontramos todos los puntos donde iba en esta marcha y los unimos
        for idx in np.where(mask)[0]:
            xs    += [x[idx], x[idx + 1], None]
            ys    += [y[idx], y[idx + 1], None]
            hover += [f"{g}ª Marcha", f"{g}ª Marcha", None]
            
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode='lines',
            line=dict(color=_gear_to_color(g), width=4),
            hovertext=hover, hoverinfo='text',
            name=f"Marcha {g}", showlegend=False,
        ))

    # ── Leyenda de color (Custom Colorbar para variables discretas) ───────────
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode='markers',
        marker=dict(
            size=0,
            color=[1, 8],
            # Simulamos el array de colores para la barra lateral
            colorscale=[[ (i-1)/7, _gear_to_color(i) ] for i in range(1, 9)],
            showscale=True,
            colorbar=dict(
                title=dict(text="Marcha", font=dict(color='#D0D2DC', family='Titillium Web, sans-serif')),
                tickvals=list(range(1, 9)),
                ticktext=[f"{i}ª" for i in range(1, 9)],
                tickfont=dict(color='#D0D2DC', family='Titillium Web, sans-serif', size=11),
                thickness=20, len=0.5, outlinewidth=0,
            ),
        ),
        hoverinfo='skip', showlegend=False,
    ))

    # ── Meta (Igual que SpeedHeatMap) ─────────────────────────────────────────
    mx, my = _offset_exterior(0, x, y, mid_x, mid_y, offset=0)
    fig.add_trace(go.Scatter(
        x=[mx], y=[my], mode='markers+text',
        marker=dict(color='#FFFFFF', size=14, symbol='line-ew', line=dict(width=3, color='#FFFFFF')),
        hovertemplate='<b>Línea de meta</b><extra></extra>', name='Meta',
    ))

    # ── Puntos de Frenada ─────────────────────────────────────────────────────
    brake_bin    = (brake > 0).astype(int)
    brake_starts = np.where(np.diff(brake_bin) > 0)[0]

    if len(brake_starts) > 0:
        bx = [_offset_exterior(i, x, y, mid_x, mid_y, 160)[0] for i in brake_starts]
        by = [_offset_exterior(i, x, y, mid_x, mid_y, 160)[1] for i in brake_starts]
        fig.add_trace(go.Scatter(
            x=bx, y=by, mode='markers',
            marker=dict(color='#E8002D', size=8, symbol='triangle-down'),
            hovertemplate='<b>Punto de frenada</b><extra></extra>', name='Frenada',
        ))

    # ── Números de Curva ──────────────────────────────────────────────────────
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
            x=cx_list, y=cy_list, mode='markers+text',
            marker=dict(size=20, color='white', line=dict(color='#E8002D', width=2), symbol='circle'),
            text=num_list, textfont=dict(color='#15151E', size=9, family='Titillium Web, sans-serif'),
            textposition='middle center', hoverinfo='skip', showlegend=False,
        ))

    # ── Layout final ──────────────────────────────────────────────────────────
    fig.update_layout(
        title=dict(
            text=f"{Driver}  ·  {Team}  ·  {event_name}  ·  {ses_name}",
            x=0.475,xanchor='center', font=dict(family='Titillium Web, sans-serif', color='#D0D2DC'),
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False, scaleanchor='x', scaleratio=1),
        margin=dict(l=0, r=60, t=60, b=0),
        height=600, dragmode='pan',
        legend=dict(
            orientation='h', yanchor='bottom', y=-0.05, xanchor='center', x=0.5,
            font=dict(family='Titillium Web, sans-serif', color='#D0D2DC'),
            bgcolor='rgba(0,0,0,0)',
        ),
    )

    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True, 'scrollZoom': True})