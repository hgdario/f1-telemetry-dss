import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import fastf1


def render_strategy_dashboard(session):
    st.title(f"Informe de Sesión: {session.event['EventName']} {session.event.year}")

    laps = session.laps.copy()

    # =====================
    # COLORES OFICIALES
    # =====================
    TIRE_COLORS = {
        'HYPERSOFT': '#FFB3C1', 'ULTRASOFT': '#B34FB3', 'SUPERSOFT': '#F20000',
        'SOFT': '#FF3333' if session.event.year > 2018 else '#FFD300',
        'MEDIUM': '#FFD300' if session.event.year > 2018 else '#F0F0F0',
        'HARD': '#F0F0F0' if session.event.year > 2018 else '#00D2FF',
        'INTERMEDIATE': '#43B02A', 'WET': '#0067B9'
    }

    COMPOUND_ORDER = [
        'HYPERSOFT', 'ULTRASOFT', 'SUPERSOFT',
        'SOFT', 'MEDIUM', 'HARD',
        'INTERMEDIATE', 'WET'
    ]

    # =====================
    # TOP POR COMPUESTO
    # =====================
    st.subheader("Mejor rendimiento por compuesto")

    best_compounds = laps.loc[
        laps.groupby('Compound')['LapTime'].idxmin()
    ][['Compound', 'Driver', 'LapTime']]

    cols = st.columns(len(best_compounds))

    for i, (_, row) in enumerate(best_compounds.iterrows()):
        cols[i].metric(
            f"{row['Compound']}",
            f"{row['Driver']}",
            f"{row['LapTime'].total_seconds():.3f}s"
        )

    st.divider()

    # =====================
    # RESUMEN (DONUT + ESTRATEGIA)
    # =====================
    col1, col2 = st.columns(2)

    # -------- DONUT --------
    with col1:
        st.subheader("Uso de compuestos")

        usage = laps.groupby('Compound').agg(
            Laps=('LapNumber', 'count')
        ).reset_index()

        fig_donut = go.Figure(data=[go.Pie(
            labels=usage['Compound'],
            values=usage['Laps'],
            hole=0.5,
            marker=dict(
                colors=[TIRE_COLORS.get(str(c).upper(), '#AAA') for c in usage['Compound']]
            )
        )])

        fig_donut.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color="#EEE"),
            height=350
        )

        st.plotly_chart(fig_donut, use_container_width=True)

    # -------- ESTRATEGIA --------
    with col2:
        st.subheader("Estrategia más frecuente")

        stints = (
            laps.groupby(['Driver', 'Stint', 'Compound'])
            .agg(Start=('LapNumber', 'min'))
            .reset_index()
            .sort_values(['Driver', 'Start'])
        )

        strategy = (
            stints.groupby('Driver')['Compound']
            .apply(lambda x: '-'.join(x.astype(str)))
            .reset_index(name='Strategy')
        )

        strategy_counts = (
            strategy['Strategy']
            .value_counts()
            .reset_index()
        )

        strategy_counts.columns = ['Strategy', 'Count']

        if not strategy_counts.empty:
            most_common = strategy_counts.iloc[0]

            comps = most_common['Strategy'].split('-')
            cols_strat = st.columns(len(comps))

            # Dibujar los neumáticos
            for i, comp in enumerate(comps):
                comp = comp.upper()
                cols_strat[i].markdown(
                    f"<div style='text-align:center; padding:10px; border-radius:8px; "
                    f"background-color:{TIRE_COLORS.get(comp, '#444')}; color:black;'>"
                    f"<b>{comp}</b></div>",
                    unsafe_allow_html=True
                )

            # --- NUEVA SECCIÓN: FOTOS DE LOS PILOTOS ---
            drivers_most_common = strategy[strategy['Strategy'] == most_common['Strategy']]['Driver'].tolist()
            
            st.markdown(f"<p style='color:#ccc; font-size:14px; margin-top:15px; margin-bottom:5px;'>"
                        f"<b>{most_common['Count']} pilotos</b> siguieron esta estrategia:</p>", 
                        unsafe_allow_html=True)
            
            # Obtener fotos y agruparlas en filas de 5 para que no se vean enanas
            pics = []
            for d in drivers_most_common:
                try:
                    drv_info = session.get_driver(d)
                    url = drv_info.get('HeadshotUrl')
                    pics.append((d, url if pd.notna(url) and isinstance(url, str) else None))
                except:
                    pics.append((d, None))
            
            n_cols = 5
            for i in range(0, len(pics), n_cols):
                row_pics = pics[i:i+n_cols]
                cols_pics = st.columns(n_cols)
                for j, (drv, url) in enumerate(row_pics):
                    with cols_pics[j]:
                        if url:
                            st.image(url, use_container_width=True)
                            st.markdown(f"<div style='text-align:center; font-size:12px;'>{drv}</div>", unsafe_allow_html=True)
                        else:
                            st.markdown(f"<div style='text-align:center; font-size:12px; font-weight:bold; padding-top:20px;'>{drv}</div>", unsafe_allow_html=True)

        else:
            st.write("No hay datos suficientes.")

    # =====================
    # MATRIZ DE ESTRATEGIA
    # =====================
    st.subheader("Estrategia de carrera por piloto")

    stints_full = (
        laps.groupby(['Driver', 'Stint', 'Compound'])
        .agg(Start=('LapNumber', 'min'), End=('LapNumber', 'max'), Count=('LapNumber', 'count'))
        .reset_index()
    )

    driver_order = (
        session.results.sort_values('ClassifiedPosition', ascending=False)['Abbreviation']
        .fillna('DNF')
        .tolist()
    )

    fig = go.Figure()

    for driver in driver_order:
        d_stints = stints_full[stints_full['Driver'] == driver]

        if d_stints.empty:
            fig.add_trace(go.Bar(
                y=[driver], x=[1],
                orientation='h',
                marker=dict(color='#555'),
                showlegend=False
            ))
            continue

        for _, row in d_stints.iterrows():
            comp = str(row['Compound']).upper()

            fig.add_trace(go.Bar(
                y=[driver],
                x=[row['Count']],
                base=row['Start'] - 1,
                orientation='h',
                marker=dict(
                    color=TIRE_COLORS.get(comp, '#888'),
                    line=dict(color='#111', width=1)
                ),
                showlegend=False
            ))

    # Leyenda dinámica 
    used_compounds = [
        c for c in COMPOUND_ORDER
        if c in laps['Compound'].str.upper().unique()
    ]

    for comp in used_compounds:
        fig.add_trace(go.Bar(
            y=[None], x=[None],
            marker=dict(color=TIRE_COLORS.get(comp, '#AAA')),
            name=comp,
            showlegend=True
        ))

    fig.update_layout(
        height=650,
        barmode='overlay',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color="#EEE"),
        xaxis=dict(title="Vueltas", gridcolor="#333", side='top'),
        yaxis=dict(categoryorder='array', categoryarray=driver_order)
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # =====================
    # DEGRADACIÓN
    # =====================
    st.subheader("Evolución del rendimiento por compuesto")

    valid_laps = laps.dropna(subset=['LapTime', 'Compound', 'TyreLife'])

    fig_deg = go.Figure()

    for comp in used_compounds:
        comp_data = valid_laps[valid_laps['Compound'].str.upper() == comp]

        if comp_data.empty:
            continue

        grouped = comp_data.groupby('TyreLife')['LapTime'].median().reset_index()

        fig_deg.add_trace(go.Scatter(
            x=grouped['TyreLife'],
            y=grouped['LapTime'].dt.total_seconds(),
            mode='lines',
            name=comp,
            line=dict(color=TIRE_COLORS.get(comp, '#AAA'), width=2)
        ))

    fig_deg.update_layout(
        height=400,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color="#EEE"),
        xaxis=dict(title="Vida del neumático (vueltas)", gridcolor="#333"),
        yaxis=dict(title="Tiempo medio por vuelta (s)", gridcolor="#333")
    )

    st.plotly_chart(fig_deg, use_container_width=True)