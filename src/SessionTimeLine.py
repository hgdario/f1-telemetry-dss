import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import fastf1

# ─── CONFIGURACIÓN MAESTRA DE ESTADOS ───
STATUS_MAP = {
    '1': {'label': 'PISTA LIMPIA', 'color': '#2ECC71', 'desc': 'Sesión reanudada.'},
    '2': {'label': 'BANDERA AMARILLA', 'color': '#F1C40F', 'desc': 'Peligro en pista.'},
    '3': {'label': 'AMARILLA SECTOR', 'color': '#F39C12', 'desc': 'Peligro localizado.'},
    '4': {'label': 'SAFETY CAR', 'color': '#E67E22', 'desc': 'Coche de Seguridad.'},
    '5': {'label': 'BANDERA ROJA', 'color': '#E74C3C', 'desc': 'SESIÓN SUSPENDIDA.'},
    '6': {'label': 'VSC', 'color': '#9B59B6', 'desc': 'Virtual Safety Car.'},
    '7': {'label': 'VSC TERMINANDO', 'color': '#00A86B', 'desc': 'Pista lista para verde.'},
    '8': {'label': 'BANDERA AZUL', 'color': '#3498DB', 'desc': 'Aviso de doblado.'},
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
        
        # Usamos comillas simples de escape por si la FIA usa dobles comillas en su texto
        mensaje_final = "📻 " + " | ".join(textos)
        # Limpiamos las comillas dobles para que no rompan el HTML del atributo title luego
        return mensaje_final.replace('"', "'")
        
    except Exception:
        return "Error de lectura."


def render_timeline(session: fastf1.core.Session):
    st.subheader("🏁 Panel de Control de Carrera (Race Control)")
    
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
    fig.add_trace(go.Bar(
        x=[max_time], y=["STATUS"], orientation='h',
        marker=dict(color='rgba(255,255,255,0.05)'),
        showlegend=False, hoverinfo='skip'
    ))

    estados_presentes = sorted(ts['Status'].unique(), key=int)
    for s_code in estados_presentes:
        df_group = ts[ts['Status'] == s_code]
        info = STATUS_MAP.get(str(s_code), {'label': f'ST {s_code}', 'color': '#555'})
        fig.add_trace(go.Bar(
            name=info['label'], x=df_group['Duracion'], y=["STATUS"] * len(df_group),
            base=df_group['Minuto'], orientation='h',
            marker=dict(color=info['color'], line=dict(width=0)), showlegend=True
        ))

    fig.update_layout(
        height=180, margin=dict(t=10, b=80, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", barmode='overlay',
        legend=dict(orientation="h", yanchor="bottom", y=1.1, xanchor="center", x=0.5, font=dict(size=10, color="#888")),
        xaxis=dict(title=dict(text="Minutos de Carrera", standoff=25, font=dict(size=12, color="#555")), range=[0, max_time], color="#888", dtick=5),
        yaxis=dict(showgrid=False, showticklabels=False, fixedrange=True)
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # ── 2. PESTAÑAS DE ANÁLISIS DE CARRERA ──
    tab_inc, tab_pit, tab_ovt = st.tabs(["🚦 Incidentes y Banderas", "🔧 Estrategia y Pit Stops", "⚔️ Adelantamientos"])

    # --- PESTAÑA 1: INCIDENTES ---
    with tab_inc:
        eventos = ts[ts['Status'] != '1'].copy()
        if eventos.empty:
            st.success("Sesión limpia.")
        else:
            # CSS ARREGLADO: Fuera los recortes del 'td', blindaje 'nowrap' solo en el 'status-pill'
            style = """<style>
                .pit-table { width: 100%; border-collapse: collapse; font-family: 'Titillium Web', sans-serif; color: #E0E0E0; margin-top: 10px; } 
                .pit-table th { text-align: left; padding: 12px; border-bottom: 2px solid #333; color: #888; font-size: 12px; } 
                .pit-table td { padding: 12px; border-bottom: 1px solid #222; font-size: 14px; line-height: 1.5; } 
                .status-pill { padding: 4px 8px; border-radius: 3px; font-weight: bold; font-size: 11px; white-space: nowrap; display: inline-block; }
            </style>"""
            
            rows = ""
            for _, row in eventos.iterrows():
                info = STATUS_MAP.get(str(row['Status']), {'label': 'EVENTO', 'color': '#FFF', 'desc': '-'})
                sospechosos = _get_involved_drivers(session, row['Time'], str(row['Status']))
                
                rows += f'<tr>'
                rows += f'<td style="width: 10%;"><span style="color:#888;">{row["Minuto"]:.1f} min</span></td>'
                rows += f'<td style="width: 15%;"><span class="status-pill" style="background:{info["color"]}33; color:{info["color"]}; border:1px solid {info["color"]}66;">{info["label"]}</span></td>'
                rows += f'<td style="width: 20%;">{info["desc"]}</td>'
                # Aquí entra el mensaje crudo y completo de la FIA. Al no tener límites, saltará de línea naturalmente.
                rows += f'<td style="width: 55%; font-style:italic; color:#777;">{sospechosos}</td>'
                rows += f'</tr>'
            
            st.markdown(f'{style}<table class="pit-table"><thead><tr><th>TIEMPO</th><th>ESTADO</th><th>DESCRIPCIÓN</th><th>RADIO DIRECCIÓN DE CARRERA</th></tr></thead><tbody>{rows}</tbody></table>', unsafe_allow_html=True)
    
# --- PESTAÑA 2: ESTRATEGIA Y BOXES ---
    with tab_pit:
        laps = session.laps.copy()
        
        # 1. LA MAGIA MATEMÁTICA: Traemos el 'PitInTime' de la vuelta anterior
        laps['PitIn_Prev'] = laps.groupby('Driver')['PitInTime'].shift(1)
        
        # Filtramos solo cuando salen de boxes
        pits = laps[pd.notnull(laps['PitOutTime'])].copy()
        
        if pits.empty:
            st.info("No hay datos de Pit Stops registrados.")
        else:
            try:
                start_time = laps[laps['LapNumber'] == 1]['Time'].min() - pd.Timedelta(seconds=90)
            except:
                start_time = pd.Timedelta(seconds=0)

            # Calculamos la duración real (Salida actual - Entrada de la vuelta anterior)
            pits['Duration'] = (pits['PitOutTime'] - pits['PitIn_Prev']).dt.total_seconds()
            
            # Limpiamos locuras: Un tiempo de pitlane en F1 dura entre 15 y 40 segundos. 
            # Si dura más (ej. bandera roja) o es NaN (salida desde el pitlane en la V1), lo ignoramos.
            pits['Duration'] = pits['Duration'].apply(lambda x: x if pd.notnull(x) and 10 < x < 100 else None)
            
            # Buscamos al campeón de los mecánicos
            fastest_pit = pits['Duration'].min()
            
            # Ordenamos Cronológicamente
            pits['Minuto_Relativo'] = (pits['Time'] - start_time).dt.total_seconds() / 60.0
            pits = pits.sort_values(by='Minuto_Relativo')

            TIRE_COLORS = {
                'HYPERSOFT': '#FFB3C1', 'ULTRASOFT': '#B34FB3', 'SUPERSOFT': '#F20000',
                'SOFT': '#FF3333' if session.event.year > 2018 else '#FFD300',
                'MEDIUM': '#FFD300' if session.event.year > 2018 else '#F0F0F0',
                'HARD': '#F0F0F0' if session.event.year > 2018 else '#00D2FF',
                'INTERMEDIATE': '#43B02A', 'WET': '#0067B9'
            }

            style = """<style>
                .f1-table { width: 100%; border-collapse: collapse; font-family: 'Titillium Web', sans-serif; }
                .f1-table th { text-align: left; padding: 12px; border-bottom: 2px solid #333; color: #888; font-size: 11px; text-transform: uppercase; }
                .f1-table td { padding: 12px; border-bottom: 1px solid #222; font-size: 14px; color: #EEE; }
                .status-pill { padding: 4px 10px; border-radius: 3px; font-weight: bold; font-size: 11px; white-space: nowrap; display: inline-block; }
            </style>"""

            rows = ""
            for _, p in pits.iterrows():
                comp = str(p['Compound']).upper()
                color = TIRE_COLORS.get(comp, '#555')
                pill = f'<span class="status-pill" style="background:{color}33; color:{color}; border:1px solid {color}66;">{comp}</span>'
                
                # Lógica del tiempo: Destacamos el récord y ocultamos los inválidos (N/A)
                dur_val = p['Duration']
                if pd.notnull(dur_val):
                    if dur_val == fastest_pit:
                        # Récord absoluto: Corona y dorado
                        dur_str = f"👑 <span style='color:#F1C40F;'>{dur_val:.1f} s</span>"
                    else:
                        dur_str = f"{dur_val:.1f} s"
                else:
                    dur_str = "<span style='color:#666;'>N/A</span>"

                rows += f"<tr>"
                rows += f"<td><b>{p['Driver']}</b></td>"
                rows += f"<td>{int(p['LapNumber'])}</td>"
                rows += f"<td>{pill}</td>"
                rows += f"<td style='color:#2ECC71; font-family:monospace; font-weight:bold;'>{dur_str}</td>"
                rows += f"<td style='color:#666;'>{p['Minuto_Relativo']:.1f} min</td>"
                rows += f"</tr>"

            st.markdown(f'{style}<table class="f1-table"><thead><tr><th>Piloto</th><th>Vuelta</th><th>Neumático</th><th>Tiempo Pitlane</th><th>Reloj</th></tr></thead><tbody>{rows}</tbody></table>', unsafe_allow_html=True)
            st.caption("""
            ⏱️ **Nota técnica sobre telemetría de Pit Lane:** Los tiempos **N/A** corresponden a situaciones donde no existe un delta válido de tránsito: 
            (1) Salidas desde el **Pit Lane** (vueltas iniciales sin entrada previa), 
            (2) Periodos de **Bandera Roja** (donde el tiempo de permanencia excede los límites lógicos de competición), 
            o (3) **Retiradas** al finalizar la sesión.  
            La corona 👑 destaca el **La parada en boxes más rápida contando el tiempo desde que entra hasta que sale.** (récord de la sesión).
        """)

    # --- PESTAÑA 3: ADELANTAMIENTOS Y POSICIONES ---
    with tab_ovt:
        st.markdown("#### 📈 Evolución de Posiciones")
        
        try:
            # 1. Obtenemos los resultados y forzamos la limpieza de datos
            res = session.results[['Abbreviation', 'GridPosition', 'ClassifiedPosition', 'Status']].copy()
            
            # Convertimos a número (los DNF/R se vuelven NaN)
            res['PosFinal'] = pd.to_numeric(res['ClassifiedPosition'], errors='coerce')
            
            # Si es un abandono, le asignamos la última posición disponible para que el cálculo sea lógico
            res['PosFinal'] = res['PosFinal'].fillna(len(res)).astype(int)
            
            # GridPosition 0 es Pit Lane, lo ponemos como 20
            res['GridFixed'] = res['GridPosition'].replace(0, 20).astype(int)
            
            # Calculamos la diferencia
            res['Diff'] = res['GridFixed'] - res['PosFinal']
            
            # Ordenamos por los que más han remontado
            remontadas = res.sort_values(by='Diff', ascending=False)

            style_ovt = """<style>
                .ovt-table { width: 100%; border-collapse: collapse; font-family: 'Titillium Web', sans-serif; }
                .ovt-table th { text-align: left; padding: 12px; border-bottom: 2px solid #333; color: #888; font-size: 11px; }
                .ovt-table td { padding: 12px; border-bottom: 1px solid #222; font-size: 14px; }
                .pos-badge { padding: 2px 8px; border-radius: 3px; font-weight: bold; font-size: 12px; display: inline-block; min-width: 35px; text-align: center; }
            </style>"""

            rows = ""
            for _, r in remontadas.iterrows():
                # Forzamos conversión a tipos básicos para evitar errores de concatenación
                nombre = str(r['Abbreviation'])
                grid = int(r['GridFixed'])
                p_final = int(r['PosFinal'])
                diff = int(r['Diff'])
                status = str(r['Status'])
                
                # Definimos el balance visual
                if diff > 0:
                    diff_html = f'<span style="color:#2ECC71;">▲ {diff}</span>'
                    bg_color = "#2ECC711A" # Verde muy suave para los que remontan
                elif diff < 0:
                    diff_html = f'<span style="color:#E74C3C;">▼ {abs(diff)}</span>'
                    bg_color = "transparent"
                else:
                    diff_html = '<span style="color:#888;">=</span>'
                    bg_color = "transparent"

                # Color para el podio o resto
                color_f = "#F1C40F" if p_final == 1 else "#EEE"
                
                # Construcción de la fila con f-strings (evita el error de concatenación)
                rows += f'<tr style="background:{bg_color};">'
                rows += f'<td style="font-weight:bold; font-size:16px;">{nombre}</td>'
                rows += f'<td>Salida: <b>{grid}º</b></td>'
                rows += f'<td>Final: <span class="pos-badge" style="border:1px solid {color_f}66; color:{color_f};">{p_final}º</span></td>'
                rows += f'<td style="font-size:16px; font-weight:bold;">{diff_html}</td>'
                rows += f'<td style="font-size:12px; color:#666;">{status}</td>'
                rows += f'</tr>'

            st.markdown(f'{style_ovt}<table class="ovt-table"><thead><tr><th>PILOTO</th><th>PARRILLA</th><th>RESULTADO</th><th>BALANCE</th><th>ESTADO</th></tr></thead><tbody>{rows}</tbody></table>', unsafe_allow_html=True)
            
            st.caption("🔄 **Balance:** Diferencia entre la posición de salida (Grid) y la posición final en la clasificación oficial.")

        except Exception as e:
            st.error(f"Error al procesar posiciones: {e}")