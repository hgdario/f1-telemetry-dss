import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import fastf1

def render_session_summary(session):
    st.markdown(f"<h1 style='text-align: center; color: #E8002D;'> RACE SUMMARY: {session.event['EventName'].upper()}</h1>", unsafe_allow_html=True)
    
    # ── 0. EXTRACCIÓN SEGURA DE DATOS BÁSICOS ──
    try:
        fastest_lap = session.laps.pick_fastest()
        ref_tel = fastest_lap.get_telemetry().add_distance()
    except Exception:
        st.error("❌ No hay telemetría suficiente en esta sesión para generar el mapa.")
        return

    # Info del circuito (Falla en años antiguos, la API de curvas se introdujo más tarde)
    try:
        circuit_info = session.get_circuit_info()
    except Exception:
        circuit_info = None

    # --- 1. CLIMA ---
    st.subheader("☁️ Condiciones Meteorológicas")
    try:
        weather = session.weather_data
        if weather is not None and not weather.empty:
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
        else:
            st.warning("⚠️ Datos meteorológicos no disponibles para este año.")
    except Exception:
        st.warning("⚠️ Datos meteorológicos no disponibles para este año.")

    st.divider()

    # --- 2. ESTADÍSTICAS RÁPIDAS ---
    st.subheader("Datos de Carrera")
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    
    circuit_length_meters = ref_tel['Distance'].max() if 'Distance' in ref_tel.columns else 0
    circuit_length_km = circuit_length_meters / 1000
    
    try:
        sc_laps = session.laps[session.laps['TrackStatus'].str.contains('4|6', na=False)]
        sc_percent = (len(sc_laps) / len(session.laps)) * 100 if len(session.laps) > 0 else 0
    except Exception:
        sc_percent = 0
    
    num_corners = len(circuit_info.corners) if circuit_info is not None else "N/A"

    with col_s1:
        st.metric("Número de Curvas", f"{num_corners}")
    with col_s2:
        st.metric("Porcentaje bajo SC/VSC", f"{sc_percent:.1f} %")
    with col_s3:
        st.metric("Vueltas Totales", f"{session.total_laps}")
    with col_s4:
        st.metric("Longitud del trazado", f"{circuit_length_km:.3f} km")

    st.divider()

    # --- 3. MAPA DEL CIRCUITO ---
    st.subheader("Trazado y Puntos Clave")
    fig_map = go.Figure()

    # Lógica de sectores envuelta en Try/Except por si faltan tiempos
    try:
        s1_time = fastest_lap['Sector1SessionTime']
        s2_time = fastest_lap['Sector2SessionTime']

        idx_s1_end = ref_tel[ref_tel['SessionTime'] <= s1_time].index[-1]
        idx_s2_end = ref_tel[ref_tel['SessionTime'] <= s2_time].index[-1]

        tel_s1 = ref_tel.loc[:idx_s1_end]
        tel_s2 = ref_tel.loc[idx_s1_end:idx_s2_end]
        tel_s3 = ref_tel.loc[idx_s2_end:]
        
        fig_map.add_trace(go.Scatter(x=tel_s1['X'], y=tel_s1['Y'], mode='lines', line=dict(color='#00D2FF', width=6), hoverinfo='skip', name="Sector 1"))
        fig_map.add_trace(go.Scatter(x=tel_s2['X'], y=tel_s2['Y'], mode='lines', line=dict(color='#E8002D', width=6), hoverinfo='skip', name="Sector 2"))
        fig_map.add_trace(go.Scatter(x=tel_s3['X'], y=tel_s3['Y'], mode='lines', line=dict(color='#FFFB00', width=6), hoverinfo='skip', name="Sector 3"))
    except Exception:
        # Fallback si falla el corte de sectores
        fig_map.add_trace(go.Scatter(x=ref_tel['X'], y=ref_tel['Y'], mode='lines', line=dict(color='#FFFFFF', width=6), hoverinfo='skip', name="Trazado"))

    # Marcador fijo genérico para evitar usar marshal_sectors que peta en 2019
    fig_map.add_trace(go.Scatter(
        x=[ref_tel['X'].iloc[len(ref_tel)//2]], 
        y=[ref_tel['Y'].iloc[len(ref_tel)//2]],
        mode='markers', marker=dict(color='#FFFFFF', size=15, symbol='star', line=dict(color='#E8002D', width=2)),
        name="Speed Trap Aprox"
    ))

    fig_map.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=500, showlegend=True,
        xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x", scaleratio=1),
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, font=dict(color="white"))
    )
    st.plotly_chart(fig_map, use_container_width=True)

    # --- 4. CLASIFICACIÓN (ESTILO PODIO) ---
    st.subheader("🏁 Clasificación Final")
    try:
        res = session.results
        
        # ──> EL ESCUDO ANTI-NULOS <──
        def safe_int(val):
            if pd.isna(val):
                return "-"
            try:
                return int(float(val))
            except (ValueError, TypeError):
                return "-"

        summary_df = pd.DataFrame({
            "Pos": res['Position'].apply(safe_int),
            "Piloto": res['Abbreviation'],
            "Equipo": res['TeamName'],
            "Salida": res['GridPosition'].apply(safe_int),
            "Estado": res.apply(lambda r: f"+ {r['Status']}" if r['Status'] == 'Lapped' else (f"🔴 {r['Status']}" if r['Status'] not in ['Finished', 'Completed'] and not str(r['Status']).startswith('+') else "Finished"), axis=1),
            "Puntos": res['Points'].apply(safe_int)
        })

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
    except Exception as e:
        st.error(f"❌ No se pudieron cargar los resultados de la sesión.")