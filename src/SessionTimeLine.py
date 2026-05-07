import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import fastf1

import ui_assets


# ─── PALETA SEMÁNTICA DE TRACK STATUS ─────────────────────────────────────────
# Colores alineados con el tema F1 broadcast (acento rojo + amarillos cálidos +
# purple sector + blue).
STATUS_MAP = {
    '1': {'label': 'PISTA LIMPIA',     'color': '#3FBF7F', 'desc': 'Sesión reanudada.'},
    '2': {'label': 'BANDERA AMARILLA', 'color': '#F2C14E', 'desc': 'Peligro en pista.'},
    '3': {'label': 'AMARILLA SECTOR',  'color': '#E89B3C', 'desc': 'Peligro localizado.'},
    '4': {'label': 'SAFETY CAR',       'color': '#FF7A29', 'desc': 'Coche de Seguridad.'},
    '5': {'label': 'BANDERA ROJA',     'color': '#E8002D', 'desc': 'Sesión suspendida.'},
    '6': {'label': 'VSC',              'color': '#B138DD', 'desc': 'Virtual Safety Car.'},
    '7': {'label': 'VSC TERMINANDO',   'color': '#7FD8B8', 'desc': 'Pista lista para verde.'},
    '8': {'label': 'BANDERA AZUL',     'color': '#3B82F6', 'desc': 'Aviso de doblado.'},
}


def _get_involved_drivers(session, timestamp, event_type):
    """Radio FIA completa, mostrando todo el historial del incidente sin recortes."""
    try:
        if not hasattr(session, 'race_control_messages') or session.race_control_messages.empty:
            return "Canal FIA no disponible."

        rcm = session.race_control_messages.copy()

        # 1. Alinear relojes
        if pd.api.types.is_datetime64_any_dtype(rcm['Time']) and hasattr(session, 't0_date'):
            rcm['Time'] = rcm['Time'] - session.t0_date

        # 2. Ventana de tiempo
        mask = (rcm['Time'] >= timestamp - pd.Timedelta(seconds=60)) & (rcm['Time'] <= timestamp + pd.Timedelta(seconds=120))
        mensajes = rcm[mask]

        if mensajes.empty:
            return "Sin mensajes oficiales."

        # 3. Limpiar mensajes genéricos
        relevantes = mensajes[~mensajes['Message'].str.contains("DRS|CLEAR|GREEN|LIMITS", case=False, na=False)]

        # 4. Coger TODOS los mensajes y unirlos
        if relevantes.empty:
            textos = mensajes['Message'].dropna().unique().tolist()
        else:
            textos = relevantes['Message'].dropna().unique().tolist()

        mensaje_final = " · ".join(textos)
        return mensaje_final.replace('"', "'")

    except Exception:
        return "Error de lectura."


def render_timeline(session: fastf1.core.Session):
    st.markdown(ui_assets.CSS_TIMELINE, unsafe_allow_html=True)

    st.markdown("""
    <div class="tlo-tl-header">
      <span class="tlo-tl-pip"></span>
      <span class="tlo-tl-title">Panel de Control de Carrera</span>
      <span class="tlo-tl-subtitle">Race Control</span>
    </div>
    """, unsafe_allow_html=True)

    try:
        # ── ALGORITMO "LIGHTS OUT" GLOBAL ──
        # 1. Calculamos el inicio real (aprox 1.5 mins antes de que el líder acabe la Vuelta 1)
        try:
            start_time = session.laps[session.laps['LapNumber'] == 1]['Time'].min() - pd.Timedelta(minutes=1.5)
            if pd.isna(start_time) or start_time.total_seconds() < 0:
                start_time = pd.Timedelta(seconds=0)
        except:
            start_time = pd.Timedelta(seconds=0)

        ts = session.track_status.copy()

        # 2. Filtramos para borrar toda la hora previa de mecánicos en parrilla
        ts = ts[ts['Time'] >= start_time].copy()

        # 3. Recalculamos los minutos para que arranquen en 0
        ts['Minuto'] = (ts['Time'] - start_time).dt.total_seconds() / 60.0

        # 4. Aseguramos que el primer registro empiece exactamente en 0.0
        if not ts.empty and ts.iloc[0]['Minuto'] > 0:
            first = ts.iloc[0].copy()
            first['Minuto'], first['Status'] = 0.0, '1'
            ts = pd.concat([pd.DataFrame([first]), ts]).reset_index(drop=True)

        # 5. Calculamos el final y la duración con el nuevo tiempo base
        max_time = (session.laps['Time'].max() - start_time).total_seconds() / 60.0
        ts['Minuto_Fin'] = ts['Minuto'].shift(-1).fillna(max_time)
        ts['Duracion'] = ts['Minuto_Fin'] - ts['Minuto']

    except Exception as e:
        st.error(f"Error al procesar la telemetría: {e}")
        return

    # ── 1. BARRA DE TIEMPO ──
    fig = go.Figure()

    # Track rail (fondo) · da peso visual y delimita la sesión
    fig.add_trace(go.Bar(
        x=[max_time], y=["STATUS"], orientation='h',
        marker=dict(
            color='rgba(255,255,255,0.025)',
            line=dict(color='rgba(255,255,255,0.06)', width=1),
        ),
        showlegend=False, hoverinfo='skip',
    ))

    estados_presentes = sorted(ts['Status'].unique(), key=int)
    for s_code in estados_presentes:
        df_group = ts[ts['Status'] == s_code]
        info = STATUS_MAP.get(str(s_code), {'label': f'ST {s_code}', 'color': '#4A4D5E'})
        fig.add_trace(go.Bar(
            name=info['label'],
            x=df_group['Duracion'], y=["STATUS"] * len(df_group),
            base=df_group['Minuto'], orientation='h',
            marker=dict(
                color=info['color'],
                line=dict(width=0),
                opacity=0.92,
            ),
            showlegend=True,
            hovertemplate=(
                f"<b style='color:{info['color']};'>{info['label']}</b><br>"
                "<span style='color:#9497A8;'>Inicio:</span> %{base:.1f} min<br>"
                "<span style='color:#9497A8;'>Duración:</span> %{x:.1f} min"
                "<extra></extra>"
            ),
        ))

    # Marcadores de inicio y fin (LIGHTS OUT / CHEQUERED)
    fig.add_shape(
        type="line", x0=0, x1=0, y0=-0.45, y1=0.45,
        line=dict(color="#E8002D", width=2),
        layer="above",
    )
    fig.add_shape(
        type="line", x0=max_time, x1=max_time, y0=-0.45, y1=0.45,
        line=dict(color="rgba(255,255,255,0.4)", width=2, dash="dot"),
        layer="above",
    )
    fig.add_annotation(
        x=0, y=-0.62, yref="paper", xanchor="left",
        text="<b>LIGHTS OUT</b>",
        showarrow=False,
        font=dict(size=8, family="JetBrains Mono, monospace", color="#E8002D"),
    )
    fig.add_annotation(
        x=max_time, y=-0.62, yref="paper", xanchor="right",
        text="<b>CHEQUERED</b>",
        showarrow=False,
        font=dict(size=8, family="JetBrains Mono, monospace", color="#9497A8"),
    )

    fig.update_layout(
        height=210,
        margin=dict(t=64, b=64, l=14, r=14),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        barmode='overlay',
        bargap=0.55,
        font=dict(family="Inter, system-ui, sans-serif", color="#9497A8"),
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.18,
            xanchor="center",  x=0.5,
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            font=dict(size=10, color="#B8BACA",
                      family="JetBrains Mono, monospace"),
            itemsizing="constant",
            itemwidth=30,
            tracegroupgap=8,
        ),
        xaxis=dict(
            title=dict(
                text="MINUTOS DE CARRERA",
                standoff=16,
                font=dict(size=9, color="#555870",
                          family="JetBrains Mono, monospace"),
            ),
            range=[-max_time * 0.005, max_time * 1.005],
            color="#9497A8",
            gridcolor="rgba(255,255,255,0.04)",
            linecolor="rgba(255,255,255,0.10)",
            tickfont=dict(size=9, family="JetBrains Mono, monospace", color="#8A8D9E"),
            dtick=5,
            tick0=0,
            ticks="outside",
            ticklen=4,
            tickcolor="rgba(255,255,255,0.15)",
            zeroline=False,
            showspikes=False,
        ),
        yaxis=dict(showgrid=False, showticklabels=False, fixedrange=True,
                   range=[-0.6, 0.6]),
        hoverlabel=dict(
            bgcolor="#15151E",
            bordercolor="rgba(232,0,45,0.5)",
            font=dict(family="Inter, system-ui, sans-serif",
                      color="#E8E9EF", size=11),
            align="left",
        ),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # ── 2. PESTAÑAS DE ANÁLISIS DE CARRERA ──
    tab_inc, tab_pit, tab_ovt = st.tabs([
        "Incidentes y Banderas",
        "Estrategia y Pit Stops",
        "Adelantamientos",
    ])

    # --- PESTAÑA 1: INCIDENTES ---
    with tab_inc:
        eventos = ts[ts['Status'] != '1'].copy()
        if eventos.empty:
            st.markdown(
                '<div class="tlo-note">Sesión limpia · sin incidentes registrados.</div>',
                unsafe_allow_html=True,
            )
        else:
            rows = ""
            for _, row in eventos.iterrows():
                info = STATUS_MAP.get(str(row['Status']), {'label': 'EVENTO', 'color': '#7A7D8F', 'desc': '-'})
                sospechosos = _get_involved_drivers(session, row['Time'], str(row['Status']))
                color = info["color"]

                rows += "<tr>"
                rows += f'<td style="width:10%;"><span class="tlo-mono">{row["Minuto"]:.1f} min</span></td>'
                rows += (
                    f'<td style="width:15%;">'
                    f'<span class="tlo-pill" style="color:{color};border-color:{color}59;background:{color}1A;">'
                    f'{info["label"]}</span></td>'
                )
                rows += f'<td style="width:20%;">{info["desc"]}</td>'
                rows += f'<td style="width:55%;color:#7A7D8F;font-size:0.78rem;line-height:1.5;">{sospechosos}</td>'
                rows += "</tr>"

            st.markdown(
                f'<table class="tlo-table">'
                f'<thead><tr>'
                f'<th>Tiempo</th><th>Estado</th><th>Descripción</th>'
                f'<th>Radio Dirección de Carrera</th>'
                f'</tr></thead><tbody>{rows}</tbody></table>',
                unsafe_allow_html=True,
            )

    # --- PESTAÑA 2: ESTRATEGIA Y BOXES ---
    with tab_pit:
        laps = session.laps.copy()

        # Traemos el 'PitInTime' de la vuelta anterior
        laps['PitIn_Prev'] = laps.groupby('Driver')['PitInTime'].shift(1)

        # Filtramos solo cuando salen de boxes
        pits = laps[pd.notnull(laps['PitOutTime'])].copy()

        if pits.empty:
            st.markdown(
                '<div class="tlo-note">No hay datos de Pit Stops registrados.</div>',
                unsafe_allow_html=True,
            )
        else:
            try:
                start_time = laps[laps['LapNumber'] == 1]['Time'].min() - pd.Timedelta(seconds=90)
            except:
                start_time = pd.Timedelta(seconds=0)

            # Calculamos la duración real (Salida actual - Entrada de la vuelta anterior)
            pits['Duration'] = (pits['PitOutTime'] - pits['PitIn_Prev']).dt.total_seconds()

            # Limpiamos: F1 pitlane dura entre 15 y 40 segundos. Fuera de [10, 100] = no válido.
            pits['Duration'] = pits['Duration'].apply(
                lambda x: x if pd.notnull(x) and 10 < x < 100 else None
            )

            fastest_pit = pits['Duration'].min()

            pits['Minuto_Relativo'] = (pits['Time'] - start_time).dt.total_seconds() / 60.0
            pits = pits.sort_values(by='Minuto_Relativo')

            # Colores de neumático desaturados
            TIRE_COLORS = {
                'HYPERSOFT':    '#FF80B5', # Rosa (solo 2018)
                'ULTRASOFT':    '#B138DD', # Morado (solo 2018)
                'SUPERSOFT':    '#E8002D', # Rojo (solo 2018)
                'SOFT':         '#E8002D' if session.event.year > 2018 else '#F2C14E', # Rojo (>2018) | Amarillo (2018)
                'MEDIUM':       '#F2C14E' if session.event.year > 2018 else '#E8E9EF', # Amarillo (>2018) | Blanco (2018)
                'HARD':         '#E8E9EF' if session.event.year > 2018 else '#4DB8D6', # Blanco (>2018) | Azul Hielo (2018)
                'INTERMEDIATE': '#3FBF7F', # Verde
                'WET':          '#3B82F6', # Azul
            }

            rows = ""
            for _, p in pits.iterrows():
                comp = str(p['Compound']).upper()
                color = TIRE_COLORS.get(comp, '#4A4D5E')
                pill = (
                    f'<span class="tlo-tyre" '
                    f'style="color:{color};border-color:{color}66;background:{color}1A;">{comp}</span>'
                )

                dur_val = p['Duration']
                if pd.notnull(dur_val):
                    if dur_val == fastest_pit:
                        dur_str = f'<span class="tlo-fastest">{dur_val:.1f} s · best</span>'
                    else:
                        dur_str = f'<span class="tlo-mono" style="color:#E8E9EF;">{dur_val:.1f} s</span>'
                else:
                    dur_str = '<span class="tlo-mono" style="color:#4A4D5E;">N/A</span>'

                rows += "<tr>"
                rows += f'<td><b>{p["Driver"]}</b></td>'
                rows += f'<td><span class="tlo-mono">{int(p["LapNumber"])}</span></td>'
                rows += f'<td>{pill}</td>'
                rows += f'<td>{dur_str}</td>'
                rows += f'<td><span class="tlo-mono" style="color:#4A4D5E;">{p["Minuto_Relativo"]:.1f} min</span></td>'
                rows += "</tr>"

            st.markdown(
                f'<table class="tlo-table">'
                f'<thead><tr>'
                f'<th>Piloto</th><th>Vuelta</th><th>Neumático</th>'
                f'<th>Tiempo Pitlane</th><th>Reloj</th>'
                f'</tr></thead><tbody>{rows}</tbody></table>',
                unsafe_allow_html=True,
            )

            st.markdown(
                '<div class="tlo-note">'
                'Los tiempos <b>N/A</b> corresponden a casos sin delta válido de tránsito: '
                'salidas desde pit lane (vueltas iniciales sin entrada previa), '
                'periodos de bandera roja (tiempo de permanencia fuera de los límites de competición), '
                'o retiradas al finalizar la sesión. '
                'El marcador <b>best</b> destaca la parada más rápida de la sesión '
                '(tiempo entrada → salida).'
                '</div>',
                unsafe_allow_html=True,
            )

    # --- PESTAÑA 3: ADELANTAMIENTOS Y POSICIONES ---
    with tab_ovt:
        st.markdown(
            '<div class="tlo-tl-header" style="margin-top:0.4rem;">'
            '<span class="tlo-tl-pip"></span>'
            '<span class="tlo-tl-title" style="font-size:1rem;">Evolución de Posiciones</span>'
            '<span class="tlo-tl-subtitle">Grid → Resultado</span>'
            '</div>',
            unsafe_allow_html=True,
        )

        try:
            # 1. Obtenemos los resultados y forzamos la limpieza de datos
            res = session.results[['Abbreviation', 'GridPosition', 'ClassifiedPosition', 'Status']].copy()

            # Convertimos a número (los DNF/R se vuelven NaN)
            res['PosFinal'] = pd.to_numeric(res['ClassifiedPosition'], errors='coerce')

            # Si es un abandono, le asignamos la última posición disponible
            res['PosFinal'] = res['PosFinal'].fillna(len(res)).astype(int)

            # GridPosition 0 es Pit Lane → 20
            res['GridFixed'] = res['GridPosition'].replace(0, 20).astype(int)

            res['Diff'] = res['GridFixed'] - res['PosFinal']
            remontadas = res.sort_values(by='Diff', ascending=False)

            rows = ""
            for _, r in remontadas.iterrows():
                nombre  = str(r['Abbreviation'])
                grid    = int(r['GridFixed'])
                p_final = int(r['PosFinal'])
                diff    = int(r['Diff'])
                status  = str(r['Status'])

                if diff > 0:
                    diff_html = f'<span class="tlo-diff-up">▲ {diff}</span>'
                elif diff < 0:
                    diff_html = f'<span class="tlo-diff-down">▼ {abs(diff)}</span>'
                else:
                    diff_html = '<span class="tlo-diff-eq">=</span>'

                pos_cls = "tlo-pos p1" if p_final == 1 else "tlo-pos"

                rows += "<tr>"
                rows += f'<td style="font-weight:600;font-size:0.95rem;">{nombre}</td>'
                rows += f'<td><span class="tlo-mono" style="color:#7A7D8F;">Salida</span> <b>{grid}º</b></td>'
                rows += f'<td><span class="tlo-mono" style="color:#7A7D8F;">Final</span> <span class="{pos_cls}">{p_final}º</span></td>'
                rows += f'<td>{diff_html}</td>'
                rows += f'<td><span class="tlo-mono" style="color:#4A4D5E;">{status}</span></td>'
                rows += "</tr>"

            st.markdown(
                f'<table class="tlo-table">'
                f'<thead><tr>'
                f'<th>Piloto</th><th>Parrilla</th><th>Resultado</th>'
                f'<th>Balance</th><th>Estado</th>'
                f'</tr></thead><tbody>{rows}</tbody></table>',
                unsafe_allow_html=True,
            )

            st.markdown(
                '<div class="tlo-note">'
                '<b>Balance:</b> diferencia entre la posición de salida (Grid) '
                'y la posición final en la clasificación oficial.'
                '</div>',
                unsafe_allow_html=True,
            )

        except Exception as e:
            st.error(f"Error al procesar posiciones: {e}")