import streamlit as st
import fastf1
# Importamos tus scripts como módulos
import Circuit2d as c2d


# 1. Configuración de la SPA (Single Page Application)
st.set_page_config(page_title="F1 Telemetry Pro", layout="wide")

# Estilo CSS rápido para el look & feel de producto
st.markdown("""
    <style>
    .main { background-color: #0b0d10; }
    .stSidebar { background-color: #161b22; }
    .block-container{max-width:100%!important;padding:2rem 2.5rem!important;}
    #MainMenu,footer,[data-testid='stHeader']{visibility:hidden;}
    [data-testid="stMetric"]{border-top:2px solid #E8002D;background:#1E1E2D;padding:12px 16px;}
    </style>
    """, unsafe_allow_html=True)

# 2. Sidebar: Configuración Global (Fase de Ingesta)
st.sidebar.title("🏁 Control de Telemetría")
menu = st.sidebar.radio(
    "Selecciona Módulo:",
    ["Dashboard Principal", "Simulación 3D", "Delta Time", "Lap Time", "Pole Heatmap"]
)

st.sidebar.divider()
year = st.sidebar.number_input("Año", value=2024, min_value=2018)
gp = st.sidebar.text_input("Gran Premio", value="Monaco")
session_type = st.sidebar.selectbox("Sesión", ["Q", "R", "FP1", "FP2", "FP3"])
show_corners = st.checkbox("Mostrar números de curvas", value=False)
# 3. Lógica de Carga con Caché (Crucial para el rendimiento)
@st.cache_data
def get_f1_session(yr, event, stype):
    session = fastf1.get_session(yr, event, stype)
    session.load()
    return session

if st.sidebar.button("🚀 Cargar Datos"):
    with st.spinner("Sincronizando con FastF1..."):
        st.session_state['f1_session'] = get_f1_session(year, gp, session_type)
        st.success(f"Datos de {gp} cargados correctamente.")

# 4. Renderizado de Módulos (Fase de Interfaz Gráfica)
if 'f1_session' not in st.session_state:
    st.info("👈 Configura la sesión y pulsa 'Cargar Datos' en el menú lateral.")
else:
    f1_session = st.session_state['f1_session']
    
    if menu == "Dashboard Principal":
        st.header(f"Resumen: {gp} {year}")
        st.write("Selecciona una métrica en el menú para empezar el análisis profundo.")
        
    elif menu == "Simulación 3D":
       driver_to_test = st.selectbox("Selecciona Piloto para Render", f1_session.drivers)
       c2d.render_interactive_sim(f1_session, driver_to_test,show_corners)
        

    elif menu == "Delta Time":
        st.subheader("Comparativa de Tiempos Delta")
        # dt.run_module(f1_session)
        st.info("Módulo Delta Time listo para conectar.")