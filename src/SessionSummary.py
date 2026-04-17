import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import fastf1

def render_session_summary(session):
    st.markdown("<h1 style='text-align: center; color: #E8002D;'> RACE SUMMARY: " + session.event['EventName'].upper() + "</h1>", unsafe_allow_html=True)
    # Obtenemos info del circuito (Corners, DRS, etc.)
    circuit_info = session.get_circuit_info()
    ref_tel = session.laps.pick_fastest().get_telemetry().add_distance()

    # --- 1. CLIMA ---
    st.subheader("☁️ Condiciones Meteorológicas")
    weather = session.weather_data
    col_w1, col_w2, col_w3 = st.columns(3)

    def create_radial(val, title, unit, max_val, color):
        return go.Figure(go.Indicator(
            mode="gauge+number",
            value=val,
            title={'text': title, 'font': {'size': 18, 'color': "white"}},
            number={'suffix': unit, 'font': {'size': 24, 'color': "white"}},
            gauge={
                'axis': {'range': [0, max_val], 'tickwidth': 1, 'tickcolor': "white"},
                'bar': {'color': color},
                'bgcolor': "rgba(0,0,0,0)",
                'borderwidth': 2,
                'bordercolor': "gray",
            }
        )).update_layout(height=250, margin=dict(l=30, r=30, t=50, b=20), paper_bgcolor='rgba(0,0,0,0)')

    with col_w1:
        st.plotly_chart(create_radial(weather['AirTemp'].mean(), "Temp. Aire", "°C", 50, "#00D2FF"), use_container_width=True)
    with col_w2:
        st.plotly_chart(create_radial(weather['TrackTemp'].mean(), "Temp. Pista", "°C", 70, "#FF3E00"), use_container_width=True)
    with col_w3:
        st.plotly_chart(create_radial(weather['Humidity'].mean(), "Humedad", "%", 100, "#00FF87"), use_container_width=True)

    st.divider()

     # --- 2. ESTADÍSTICAS RÁPIDAS (SC, CURVAS) ---
    st.subheader("Datos de Carrera")
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    #Distancia
    fastest_lap = session.laps.pick_fastest()
    telemetry = fastest_lap.get_telemetry().add_distance()
    circuit_length_meters = telemetry['Distance'].max()
    circuit_length_km = circuit_length_meters / 1000
    # Cálculo de Safety Car (basado en TrackStatus)
    # Status 4 = SC, Status 6 = VSC
    sc_laps = session.laps[session.laps['TrackStatus'].str.contains('4|6')]
    sc_percent = (len(sc_laps) / len(session.laps)) * 100 if len(session.laps) > 0 else 0
    
    with col_s1:
        st.metric("Número de Curvas", f"{len(circuit_info.corners)}")
    with col_s2:
        st.metric("Porcentaje bajo SC/VSC", f"{sc_percent:.1f} %")
    with col_s3:
        st.metric("Vueltas Totales", f"{session.total_laps}")
    with col_s4:
        st.metric("Longitud del trazado", f"{circuit_length_km:.3f} km")

    st.divider()

    # --- 3. MAPA DEL CIRCUITO CON DRS Y SPEED TRAPS ---
    st.subheader("Trazado y Puntos Clave")
     
    fig_map = go.Figure()

    # --- LÓGICA DE COLORES POR SECTOR ---
    # Usamos los tiempos de la vuelta rápida para saber dónde corta cada sector
    s1_time = fastest_lap['Sector1SessionTime']
    s2_time = fastest_lap['Sector2SessionTime']

    # Encontrar los índices exactos donde corta cada sector
    idx_s1_end = ref_tel[ref_tel['SessionTime'] <= s1_time].index[-1]
    idx_s2_end = ref_tel[ref_tel['SessionTime'] <= s2_time].index[-1]

    # Cortar los datos (solapando el índice final para que no haya huecos en la línea)
    tel_s1 = ref_tel.loc[:idx_s1_end]
    tel_s2 = ref_tel.loc[idx_s1_end:idx_s2_end]
    tel_s3 = ref_tel.loc[idx_s2_end:]

    #Linea sobre la que pintaremos todo.
    fig_map.add_trace(go.Scatter(
        x=ref_tel['X'], y=ref_tel['Y'],
        mode='lines', line=dict(color='rgba(255,255,255,0.1)', width=4),
        hoverinfo='skip', showlegend=False
    ))

    # Trazado Sector 1 (Rojo)
    fig_map.add_trace(go.Scatter(
        x=tel_s1['X'], y=tel_s1['Y'],
        mode='lines', line=dict(color='#00D2FF', width=6),
        hoverinfo='skip', name="Sector 1"
    ))

    # Trazado Sector 2 (Azul/Cyan)
    fig_map.add_trace(go.Scatter(
        x=tel_s2['X'], y=tel_s2['Y'],
        mode='lines', line=dict(color='#E8002D', width=6),
        hoverinfo='skip', name="Sector 2"
    ))

    # Trazado Sector 3 (Amarillo)
    fig_map.add_trace(go.Scatter(
        x=tel_s3['X'], y=tel_s3['Y'],
        mode='lines', line=dict(color='#FFFB00', width=6),
        hoverinfo='skip', name="Sector 3"
    ))

    # Zonas DRS (Puntitos verdes)
    drs_zones = circuit_info.marshal_sectors
    for i, region in circuit_info.corners.iterrows():
         pass
    
    # Añadir marcadores de DRS
    fig_map.add_trace(go.Scatter(
        x=[ref_tel['X'].iloc[100], ref_tel['X'].iloc[500]], 
        y=[ref_tel['Y'].iloc[100], ref_tel['Y'].iloc[500]],
        mode='markers', marker=dict(color='#00FF00', size=12, symbol='circle'),
        name="Zona DRS"
    ))

    # Speed Trap (Icono de radar)
    fig_map.add_trace(go.Scatter(
        x=[ref_tel['X'].iloc[len(ref_tel)//2]], 
        y=[ref_tel['Y'].iloc[len(ref_tel)//2]],
        mode='markers', marker=dict(color='#FFFFFF', size=15, symbol='star', line=dict(color='#E8002D', width=2)),
        name="Speed Trap"
    ))

    fig_map.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        height=500, showlegend=True,
        xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x", scaleratio=1),
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, font=dict(color="white"))
    )
    st.plotly_chart(fig_map, use_container_width=True)

# --- 4. CLASIFICACIÓN (ESTILO PODIO) ---
    st.subheader("🏁 Clasificación Final")
    res = session.results
    
    summary_df = pd.DataFrame({
        "Pos": res['Position'].astype(int),
        "Piloto": res['Abbreviation'],
        "Equipo": res['TeamName'],
        "Salida": res['GridPosition'].astype(int),
        "Estado": res.apply(lambda r: f"+ {r['Status']}" if r['Status'] == 'Lapped'else (f"🔴 {r['Status']}" if r['Status'] not in ['Finished', 'Completed'] and not r['Status'].startswith('+') else "Finished"),
    axis=1
),
        "Puntos": res['Points'].astype(int)
    })

    # LÓGICA DE COLORES DEL PODIO
    def style_podium(row):
        if row['Pos'] == 1:
            return ['background-color: rgba(255, 215, 0, 0.2); color: #FFD700; font-weight: bold'] * len(row)
        elif row['Pos'] == 2:
            return ['background-color: rgba(192, 192, 192, 0.2); color: #C0C0C0; font-weight: bold'] * len(row)
        elif row['Pos'] == 3:
            return ['background-color: rgba(205, 127, 50, 0.2); color: #CD7F32; font-weight: bold'] * len(row)
        return [''] * len(row)

    st.dataframe(
        summary_df.style.apply(style_podium, axis=1),
        use_container_width=True, 
        hide_index=True
    )