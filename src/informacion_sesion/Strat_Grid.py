import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import fastf1

import ui_assets

# Corrección de carga de combustible (constantes públicas).
FUEL_KG_INICIAL = 110.0   # depósito máximo al arrancar la carrera (kg)
SEG_POR_KG      = 0.03     # penalización de tiempo por kg a bordo (s/vuelta)


def fuel_corrected_laptime(lap_seconds, lap_number, total_laps,
                           fuel_kg=FUEL_KG_INICIAL, sec_per_kg=SEG_POR_KG):
    """
    Corrige un tiempo por vuelta descontando el efecto de la carga de combustible.

    Un F1 arranca con hasta ``fuel_kg`` de combustible que quema de forma
    aproximadamente lineal, y cada kg a bordo penaliza el tiempo en ``sec_per_kg``
    segundos. Esta función estima el combustible restante en cada vuelta y resta
    su penalización, dejando los tiempos referidos a una misma carga (depósito
    vacío) para que sean comparables entre fases de carrera.

    Acepta escalares o series/arrays (la aritmética es vectorizada).
    """
    kg_restante = fuel_kg * (1.0 - lap_number / total_laps)
    return lap_seconds - kg_restante * sec_per_kg


def render_strategy_dashboard(session):
    st.title(f"Informe de Sesión: {session.event['EventName']} {session.event.year}")

    laps = session.laps.copy()

    # Usar paleta centralizada de colores de neumáticos
    tire_colors = ui_assets.get_tire_colors(session.event.year)

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
                colors=[tire_colors.get(str(c).upper(), '#AAA') for c in usage['Compound']]
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
                    f"background-color:{tire_colors.get(comp, '#444')}; color:black;'>"
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
                    color=tire_colors.get(comp, '#888'),
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
            marker=dict(color=tire_colors.get(comp, '#AAA')),
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

    valid_laps = valid_laps.copy()

    # Descartar vueltas de entrada y salida de boxes (in/out laps): la vuelta de
    # instalación sale con neumático frío y rueda lenta para calentarlo, por lo
    # que no refleja el rendimiento del compuesto y distorsiona la curva.
    if 'PitOutTime' in valid_laps.columns:
        valid_laps = valid_laps[valid_laps['PitOutTime'].isna()]
    if 'PitInTime' in valid_laps.columns:
        valid_laps = valid_laps[valid_laps['PitInTime'].isna()]

    # Corrección de carga de combustible (ver fuel_corrected_laptime). Sin ella un
    # compuesto usado al inicio (coche pesado) parecería más lento que otro usado
    # al final (coche ligero), enmascarando su rendimiento real.
    total_laps = float(valid_laps['LapNumber'].max())
    valid_laps['LapTimeCorr'] = fuel_corrected_laptime(
        valid_laps['LapTime'].dt.total_seconds(),
        valid_laps['LapNumber'],
        total_laps,
    )

    # Descartar vueltas anómalas (Safety Car, tráfico, errores): por encima del
    # percentil 95 dejan de representar el ritmo real del compuesto.
    umbral = valid_laps['LapTimeCorr'].quantile(0.95)
    valid_laps = valid_laps[valid_laps['LapTimeCorr'] <= umbral]

    fig_deg = go.Figure()

    # Para cada compuesto mostramos el dato real (mediana por vida de neumático,
    # en puntos tenues) y, encima, una recta de tendencia por regresión lineal.
    # La pendiente de esa recta es el ritmo de degradación (s/vuelta) y el punto
    # donde dos rectas se cruzan es el "crossover" entre compuestos.
    deg_fits = {}  # compuesto -> (slope, intercept) para el crossover

    for comp in used_compounds:
        comp_data = valid_laps[valid_laps['Compound'].str.upper() == comp]

        if comp_data.empty:
            continue

        color = tire_colors.get(comp, '#AAA')

        # Dato real: mediana del tiempo corregido por vida de neumático
        grouped = comp_data.groupby('TyreLife')['LapTimeCorr'].median().reset_index()
        fig_deg.add_trace(go.Scatter(
            x=grouped['TyreLife'],
            y=grouped['LapTimeCorr'],
            mode='markers',
            name=comp,
            marker=dict(color=color, size=5, opacity=0.4),
            hovertemplate=f"{comp}<br>Vida: %{{x}} vueltas<br>%{{y:.2f}} s<extra></extra>",
        ))

        # Tendencia: regresión lineal sobre todas las vueltas del compuesto
        x = comp_data['TyreLife'].to_numpy(dtype=float)
        y = comp_data['LapTimeCorr'].to_numpy(dtype=float)
        if len(np.unique(x)) >= 2:
            slope, intercept = np.polyfit(x, y, 1)
            deg_fits[comp] = (slope, intercept)
            xs = np.array([x.min(), x.max()])
            fig_deg.add_trace(go.Scatter(
                x=xs,
                y=slope * xs + intercept,
                mode='lines',
                name=f"{comp} · {slope:+.3f} s/vuelta",
                line=dict(color=color, width=3),
            ))

    # Crossover entre los dos compuestos más usados: vida de neumático en la que
    # sus rectas de tendencia se igualan (a partir de ahí, el más rápido en seco
    # pasa a ser el más lento). Solo se marca si cae dentro del rango observado.
    if len(deg_fits) >= 2:
        top2 = sorted(deg_fits, key=lambda c: -len(
            valid_laps[valid_laps['Compound'].str.upper() == c]))[:2]
        (s1, i1), (s2, i2) = deg_fits[top2[0]], deg_fits[top2[1]]
        if not np.isclose(s1, s2):
            x_cross = (i2 - i1) / (s1 - s2)
            vidas = valid_laps['TyreLife']
            if vidas.min() <= x_cross <= vidas.max():
                y_cross = s1 * x_cross + i1
                fig_deg.add_trace(go.Scatter(
                    x=[x_cross], y=[y_cross],
                    mode='markers+text',
                    marker=dict(color='#FFFFFF', size=10, symbol='x'),
                    text=[f"  cruce ~v{x_cross:.0f}"],
                    textposition='top right',
                    textfont=dict(color='#FFFFFF', size=11),
                    name=f"Cruce {top2[0]}/{top2[1]}",
                    hovertemplate=f"Cruce {top2[0]}/{top2[1]}<br>~vuelta %{{x:.1f}}<extra></extra>",
                ))

    fig_deg.update_layout(
        height=400,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color="#EEE"),
        xaxis=dict(title="Vida del neumático (vueltas)", gridcolor="#333"),
        yaxis=dict(title="Tiempo por vuelta corregido por combustible (s)", gridcolor="#333")
    )

    st.plotly_chart(fig_deg, use_container_width=True)