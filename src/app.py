"""
TALOS — Telemetry and Lap Optimization System
Esqueleto principal · Streamlit nativo + config.toml
"""

import streamlit as st
import Circuit2d as c2d
import requests
import SpeedHeatMap as sm
import GearHeatMap as gm
import Telemetrytrace as trace
# ─── CONSTANTES DE NAVEGACIÓN ─────────────────────────────────────────────────

# ─── CONSTANTES DE NAVEGACIÓN ─────────────────────────────────────────────────

NAV = {
    "Información de Sesión": [
        "Cronología de Eventos",
        "Estrategia de Neumáticos",
        "Exigencia del Trazado",
    ],
    "Análisis de Piloto (Micro)": [
        "Circuito Animado",
        "Mapa de Velocidad",
        "Mapa de Marchas",
        "Telemetry Trace",
    ],
    "Comparativas Competitivas": [
        "Superposición de Deltas",
        "La Vuelta Ideal",
        "Coche Fantasma (Ghost)",
    ],
    "Dinámica Vehicular": [
        "Diagrama G-G",
        "Carga Aerodinámica",
        "Modelo Térmico de Frenos",
    ],
}

# Clave única para cada módulo (para session_state)
MODULE_KEYS = [m for mods in NAV.values() for m in mods]

# ─── CONFIG ───────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="TALOS · F1 Telemetry",
    page_icon="🏎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Única inyección permitida: eliminar el max-width nativo de Streamlit e importar Titillium Web
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Titillium+Web:wght@400;600;700;900&display=swap');
    
    .block-container{max-width:100%!important;padding:2rem 2.5rem!important;}
    #MainMenu,footer,[data-testid='stHeader']{visibility:hidden;}
    [data-testid="stMetric"]{border-top:2px solid #E8002D;background:#1E1E2D;padding:12px 16px;}
    
    /* Aplicar la fuente también a toda la interfaz general de la app */
    html, body, [class*="css"] {
        font-family: 'Titillium Web', sans-serif;
    }
    </style>
""", unsafe_allow_html=True)

# ─── ESTADO INICIAL ───────────────────────────────────────────────────────────

if "active_module" not in st.session_state:
    st.session_state["active_module"] = None   # None = mostrar Hub

if "session_loaded" not in st.session_state:
    st.session_state["session_loaded"] = False

if "f1_session" not in st.session_state:
    st.session_state["f1_session"] = None


# ─── SIDEBAR ──────────────────────────────────────────────────────────────────

with st.sidebar:

    # Cabecera
    st.markdown("# TALOS")
    st.caption("Telemetry and Lap Optimization System")
    st.divider()

    # Botón Hub
    if st.button("⬡  Hub Principal", use_container_width=True):
        st.session_state["active_module"] = None

    st.divider()

    # Navegación por categorías
    for category, modules in NAV.items():
        st.caption(category.upper())
        for module in modules:
            # Resaltar el módulo activo
            is_active = st.session_state["active_module"] == module
            label = f"▸  {module}" if is_active else f"   {module}"
            if st.button(label, key=f"nav_{module}", use_container_width=True):
                st.session_state["active_module"] = module

        st.markdown("")   # espaciado entre categorías

    # ── Configuración de sesión — fijada al fondo ──────────────────────────
    st.divider()
    st.caption("CONFIGURACIÓN DE SESIÓN")

    year     = st.number_input("Año",          min_value=2018, max_value=2025, value=2024)
    gp       = st.text_input ("Gran Premio",   value="Monaco")
    ses_type = st.selectbox  ("Sesión",        ["Q", "R", "FP1", "FP2", "FP3"])
    corners  = st.checkbox   ("Mostrar curvas", value=False)

    st.divider()

    load = st.button("CARGAR SESIÓN", use_container_width=True, type="primary")

    if load:
        import fastf1

        @st.cache_data(show_spinner=False)
        def _load(yr: int, gp_name: str, stype: str):
            s = fastf1.get_session(yr, gp_name, stype)
            s.load()
            return s

        with st.spinner("Sincronizando con FastF1..."):
            try:
                st.session_state["f1_session"]    = _load(year, gp, ses_type)
                st.session_state["session_loaded"] = True
                st.session_state["session_label"]  = f"{gp}  {year} · {ses_type}"
                st.success("Sesión cargada.")
            except Exception as e:
                st.error(f"Error: {e}")

    # Indicador de estado de sesión
    if st.session_state["session_loaded"]:
        st.success(f"✓ {st.session_state.get('session_label', '')}")
    else:
        st.warning("Sin sesión activa")


# ─── HELPERS ──────────────────────────────────────────────────────────────────

def require_session() -> bool:
    """Muestra aviso y devuelve False si no hay sesión cargada."""
    if not st.session_state["session_loaded"]:
        st.warning("Carga una sesión desde el panel lateral para continuar.")
        return False
    return True


def placeholder(module_name: str, description: str, channels: list[str] | None = None):
    """Vista placeholder estándar para módulos no implementados aún."""
    st.header(module_name)
    if channels:
        st.caption("Canales:  " + "  ·  ".join(channels))
    st.divider()
    st.info(f"Módulo **{module_name}** listo para conectar.")
    with st.expander("Vista previa del contenedor de datos"):
        cols = st.columns(4)
        for i, col in enumerate(cols):
            col.metric(label=f"Canal {i+1}", value="—", delta=None)
        st.area_chart({"Esperando datos": []})


# ─── ENRUTADOR PRINCIPAL ──────────────────────────────────────────────────────

active = st.session_state["active_module"]

# ── HUB ───────────────────────────────────────────────────────────────────────
# ── HUB ───────────────────────────────────────────────────────────────────────
if active is None:
    st.markdown("---")

    # Título centrado con columnas nativas
    _, center, _ = st.columns([1, 2, 1])
    with center:
        st.markdown("# T  A  L  O  S")
        st.markdown("#### Telemetry and Lap Optimization System")
        st.divider()
        st.markdown(
            "Selecciona un módulo en el panel lateral para comenzar el análisis.\n\n"
            "Configura primero la sesión en la parte inferior del menú y pulsa **CARGAR SESIÓN**."
        )

    st.divider()

    st.subheader("Módulos disponibles")

    # Creamos 4 columnas dinámicamente basadas en nuestro nuevo diccionario NAV
    cols = st.columns(len(NAV), gap="medium")

    for col, (category, modules) in zip(cols, NAV.items()):
        with col:
            st.markdown(f"##### {category}")
            for m in modules:
                # El botón dibuja el módulo y si se clica, cambia el estado
                st.button(m, key=f"hub_{m}", use_container_width=True)

    # Detectar click desde el hub
    for m in MODULE_KEYS:
        if st.session_state.get(f"hub_{m}"):
            st.session_state["active_module"] = m
            st.rerun()

# ── INFORMACIÓN DE SESIÓN ─────────────────────────────────────────────────────

elif active == "Cronología de Eventos":
    if not require_session(): st.stop()
    placeholder("Cronología de Eventos", "Línea de tiempo de Banderas, Safety Cars y VSC.", ["TrackStatus", "Time"])

elif active == "Estrategia de Neumáticos":
    if not require_session(): st.stop()
    placeholder("Estrategia de Neumáticos", "Gantt de paradas en boxes y compuestos usados.", ["Stints", "Compound", "PitOutTime"])

elif active == "Exigencia del Trazado":
    if not require_session(): st.stop()
    placeholder("Exigencia del Trazado", "Mapa del circuito coloreado por fuerzas G medias.", ["X", "Y", "gLat", "gLon"])


# ── ANÁLISIS DE PILOTO (MICRO) ────────────────────────────────────────────────

elif active == "Circuito Animado":
    if not require_session(): st.stop()


    st.header("Circuito Animado")
    st.divider()
    f1s    = st.session_state["f1_session"]

    #Diccionario para mapear el id con el nombre
    driver_names= { d: f1s.get_driver(d).get("FullName", str(d))
                   for d in f1s.drivers}
    driver = st.selectbox("Piloto", options= f1s.drivers,format_func=lambda x: driver_names.get(x,x))

    vuelta_limpia = f1s.laps.pick_drivers(driver).pick_accurate()
    num_fastest = vuelta_limpia.pick_fastest()['LapNumber']
    lista_vueltas = vuelta_limpia['LapNumber'].tolist()
    
    lap_n = st.selectbox(
        "Vuelta a analizar", 
        options=lista_vueltas,
        index=lista_vueltas.index(num_fastest),
        format_func=lambda x: f"Vuelta {int(x)} ⏱️ (Best Lap)" if x == num_fastest else f"Vuelta {int(x)}"
    )

# ── Tarjeta del piloto ───────────────────────────────────────────────────
    @st.cache_data(show_spinner=False)
    def fetch_driver_stats(ergast_id: str) -> dict:
        base  = "https://api.jolpi.ca/ergast/f1"
        stats = {"nationality": "—", "dob": "—",
                 "wins": "—", "podiums": "—", "championships": "—"}
        
        # Si el ID está vacío, no se llama a la api
        if not ergast_id:
            return stats
            
        try:
            # 1. Nacionalidad y Nacimiento
            r = requests.get(f"{base}/drivers/{ergast_id}.json", timeout=5)
            if r.ok:
                d = r.json()["MRData"]["DriverTable"]["Drivers"]
                if d:
                    stats["nationality"] = d[0].get("nationality", "—")
                    stats["dob"]         = d[0].get("dateOfBirth", "—")

            # 2. Victorias totales (1er puesto)
            r = requests.get(f"{base}/drivers/{ergast_id}/results/1.json?limit=1000", timeout=5)
            if r.ok:
                stats["wins"] = r.json()["MRData"]["total"]

            # 3. Podios totales (Filtrando posiciones 1, 2 y 3)
            r = requests.get(f"{base}/drivers/{ergast_id}/results.json?limit=1000", timeout=5)
            if r.ok:
                races = r.json()["MRData"]["RaceTable"]["Races"]
                stats["podiums"] = sum(
                    1 for race in races for res in race["Results"]
                    if int(res.get("position", 99)) <= 3
                )

            # 4. Campeonatos del mundo
            CAMPEONES = {
                    "hamilton": "7",
                    "max_verstappen": "4",
                    "alonso": "2",
                    "norris": "1",
                    "vettel": "4",
                    "raikkonen":"1"
                }
            stats["championships"] = CAMPEONES.get(ergast_id, "0")
        except Exception: pass
        return stats

    # ─── EXTRAER DATOS 100% NATIVOS DE FASTF1 ───
    driver_info = f1s.get_driver(driver)
    
    # Diccionario de banderas para la parrilla actual
    BANDERAS = {
            "Spanish": "🇪🇸", "Monegasque": "🇲🇨", "Dutch": "🇳🇱", "British": "🇬🇧",
            "Mexican": "🇲🇽", "Australian": "🇦🇺", "French": "🇫🇷", "Japanese": "🇯🇵",
            "Canadian": "🇨🇦", "Thai": "🇹🇭", "American": "🇺🇸", "Finnish": "🇫🇮",
            "Chinese": "🇨🇳", "Danish": "🇩🇰", "German": "🇩🇪", "New Zealander": "🇳🇿",
            "Italian": "🇮🇹", "Brazilian": "🇧🇷", "Argentine": "🇦🇷", "Colombian": "🇨🇴"
    }
    # y la URL de la foto oficial que FastF1 ya nos provee
    ergast_id   = driver_info.get("DriverId", "")
    headshot    = driver_info.get("HeadshotUrl", None)
    
    first_name  = driver_info.get("FirstName", "")
    last_name   = driver_info.get("LastName", driver)
    team        = driver_info.get("TeamName", "—")

    number      = driver_info.get("DriverNumber",driver)

    # Llamamos a la API 
    stats = fetch_driver_stats(ergast_id)


    # ─── RENDERIZAR LA TARJETA ───
    col_img, col_data = st.columns([1, 3], gap="large")
    with col_img:
        try:
            # Si FastF1 tiene la foto, la pintamos directamente
            if headshot:
                st.image(headshot, width=200)
            else:
                raise ValueError
        except Exception:
            st.markdown(
                f"<div style='font-size:56px;text-align:center;padding:16px;'>"
                f"{driver[:3].upper()}</div>",
                unsafe_allow_html=True,
            )

    with col_data:
        st.markdown(f"### {number} {first_name} {last_name}")
        st.caption(team)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Victorias",    stats["wins"])
        m2.metric("Podios",       stats["podiums"])
        m3.metric("Campeonatos",  stats["championships"])
        m4.metric("Nacimiento",   stats["dob"])
        bandera = BANDERAS.get(stats['nationality'],"🏁")
        st.caption(f"Nacionalidad: {stats['nationality']} {bandera}")

    st.divider()
    
    # Lanzamos el mapa 2D
    c2d.render_interactive_sim(f1s, driver, corners, lap_n)

elif active == "Mapa de Velocidad":
    if not require_session(): st.stop()
    st.header("Mapa de Calor: Velocidad (Km/h)")
    st.divider()
    f1s = st.session_state["f1_session"]
    driver_names = {d: f1s.get_driver(d).get("FullName", str(d)) for d in f1s.drivers}
    driver = st.selectbox("Piloto", options=f1s.drivers, format_func=lambda x: driver_names.get(x, x), key="spd_drv")
    vuelta_limpia = f1s.laps.pick_drivers(driver).pick_accurate()
    num_fastest = vuelta_limpia.pick_fastest()['LapNumber']
    lista_vueltas = vuelta_limpia['LapNumber'].tolist()
    lap_n = st.selectbox("Vuelta a analizar", options=lista_vueltas, index=lista_vueltas.index(num_fastest), format_func=lambda x: f"Vuelta {int(x)} ⏱️ (Best Lap)" if x == num_fastest else f"Vuelta {int(x)}", key="spd_lap")
    
    sm.render_speed_heatmap(f1s, driver, st.session_state.get("corners", False), lap_n)

elif active == "Mapa de Marchas":
    if not require_session(): st.stop()
    st.header("Mapa de Calor: Marchas (1-8)")
    st.divider()
    f1s = st.session_state["f1_session"]
    driver_names = {d: f1s.get_driver(d).get("FullName", str(d)) for d in f1s.drivers}
    driver = st.selectbox("Piloto", options=f1s.drivers, format_func=lambda x: driver_names.get(x, x), key="gear_drv")
    vuelta_limpia = f1s.laps.pick_drivers(driver).pick_accurate()
    num_fastest = vuelta_limpia.pick_fastest()['LapNumber']
    lista_vueltas = vuelta_limpia['LapNumber'].tolist()
    lap_n = st.selectbox("Vuelta a analizar", options=lista_vueltas, index=lista_vueltas.index(num_fastest), format_func=lambda x: f"Vuelta {int(x)} ⏱️ (Best Lap)" if x == num_fastest else f"Vuelta {int(x)}", key="gear_lap")
    
    gm.render_gear_heatmap(f1s, driver, st.session_state.get("corners", False), lap_n)

elif active == "Telemetry Trace":
    if not require_session(): st.stop()
    st.header("Análisis de Telemetría (Trace)")
    st.divider()
    f1s = st.session_state["f1_session"]
    driver_names = {d: f1s.get_driver(d).get("FullName", str(d)) for d in f1s.drivers}
    driver = st.selectbox("Piloto", options=f1s.drivers, format_func=lambda x: driver_names.get(x, x), key="trace_drv")
    vuelta_limpia = f1s.laps.pick_drivers(driver).pick_accurate()
    num_fastest = vuelta_limpia.pick_fastest()['LapNumber']
    lista_vueltas = vuelta_limpia['LapNumber'].tolist()
    lap_n = st.selectbox("Vuelta a analizar", options=lista_vueltas, index=lista_vueltas.index(num_fastest), format_func=lambda x: f"Vuelta {int(x)} ⏱️ (Best Lap)" if x == num_fastest else f"Vuelta {int(x)}", key="trace_lap")
    
    trace.render_telemetry_trace(f1s, driver, int(lap_n))


# ── COMPARATIVAS COMPETITIVAS ─────────────────────────────────────────────────

elif active == "Superposición de Deltas":
    if not require_session(): st.stop()
    placeholder("Superposición de Deltas", "Comparativa de velocidad y delta de tiempo entre dos pilotos.", ["Speed", "Distance", "Delta Time"])

elif active == "La Vuelta Ideal":
    if not require_session(): st.stop()
    placeholder("La Vuelta Ideal", "El circuito pintado con los colores del piloto más rápido en cada micro-sector.", ["Sectors", "MinTime", "DriverColor"])

elif active == "Coche Fantasma (Ghost)":
    if not require_session(): st.stop()
    placeholder("Coche Fantasma (Ghost)", "Animación 2D de persecución entre dos pilotos interpolada por spline.", ["X", "Y", "Time", "Interpolation"])


# ── DINÁMICA VEHICULAR ────────────────────────────────────────────────────────

elif active == "Diagrama G-G":
    if not require_session(): st.stop()
    placeholder("Diagrama G-G (Círculo de Tracción)", "Envolvente de aceleraciones laterales y longitudinales para evaluar el uso del neumático.", ["gLat", "gLon", "vCar"])

elif active == "Carga Aerodinámica":
    if not require_session(): st.stop()
    placeholder("Mapa de Carga Aerodinámica", "Relación matemática entre velocidad y la capacidad de generar fuerzas laterales.", ["vCar", "gLat", "Downforce Estimate"])

elif active == "Modelo Térmico de Frenos":
    if not require_session(): st.stop()
    placeholder("Modelo Térmico de Frenos", "Simulación del calentamiento de discos basada en presión de freno y enfriamiento por velocidad.", ["Brake", "vCar", "Time"])