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
import SessionTimeLine as stl
import Strat_Grid as sgrid
import SessionSummary as ss
import Ghostcar as ghost
import OptimalLap as ol
import Head2head as h2h
import OverallStandings as os
import GGDiagram as gg
import Aero as aero
import TyreLoad as tload
# ─── CONSTANTES DE NAVEGACIÓN ─────────────────────────────────────────────────

NAV = {
    "Información de Sesión": [
        "Resumen de la sesión",
        "Cronología de Eventos",
        "Estrategia de Neumáticos",
    ],
    "Análisis de Piloto (Micro)": [
        "Mapa de Velocidad",
        "Mapa de Marchas",
        "Telemetry Trace",
    ],
    "Comparativas Competitivas": [
        "Panel de Equipo",
        "Comparativa de Sesión",
        "La Vuelta Ideal",
        "Coche Fantasma",
    ],
    "Dinámica Vehicular": [
        "Diagrama G-G",
        "Carga Aerodinámica",
        "Mapa de Carga por Rueda",
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

    year     = st.number_input("Año",          min_value=2018, max_value=2026, value=2024)
    gp       = st.text_input ("Gran Premio",   value="Monaco")
    ses_type = st.selectbox  ("Sesión",        ["Q", "R", "FP1", "FP2", "FP3"])
    corners  = st.checkbox   ("Mostrar curvas", value=False)
    # Lo guardamos en session_state para poder pasarlo a los módulos
    st.session_state["corners"] = corners 

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

def select_driver_and_lap(f1_session, key_prefix: str):
    """Genera los selectores de Piloto y Vuelta en 2 columnas y devuelve las selecciones."""
    col1, col2 = st.columns(2)
    driver_names = {d: f1_session.get_driver(d).get("FullName", str(d)) for d in f1_session.drivers}
    
    with col1:
        drv = st.selectbox("Piloto", options=f1_session.drivers, format_func=lambda x: driver_names.get(x, x), key=f"{key_prefix}_drv")
    
    vuelta_limpia = f1_session.laps.pick_drivers(drv).pick_accurate()
    
    if vuelta_limpia.empty:
        st.warning(f"⚠️ No hay datos de telemetría válidos para {drv} en esta sesión.")
        return drv, None

    num_fastest = vuelta_limpia.pick_fastest()['LapNumber']
    lista_vueltas = vuelta_limpia['LapNumber'].tolist()
    def_idx = lista_vueltas.index(num_fastest) if num_fastest in lista_vueltas else 0
    
    with col2:
        lap = st.selectbox("Vuelta a analizar", options=lista_vueltas, index=def_idx, 
                           format_func=lambda x: f"Vuelta {int(x)} ⏱️ (Best Lap)" if x == num_fastest else f"Vuelta {int(x)}", 
                           key=f"{key_prefix}_lap")
        
    return drv, int(lap)

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
f1s = st.session_state["f1_session"]
show_corners = st.session_state.get("corners", False)

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

    # Creamos columnas dinámicamente basadas en nuestro diccionario NAV
    cols = st.columns(len(NAV), gap="medium")
    for col, (category, modules) in zip(cols, NAV.items()):
        with col:
            st.markdown(f"##### {category}")
            for m in modules:
                if st.button(m, key=f"hub_{m}", use_container_width=True):
                    st.session_state["active_module"] = m
                    st.rerun()

# ── INFORMACIÓN DE SESIÓN ─────────────────────────────────────────────────────
elif active == "Resumen de la sesión":
    if not require_session(): st.stop()
    ss.render_session_summary(f1s)
    
elif active == "Cronología de Eventos":
    if not require_session(): st.stop()
    stl.render_timeline(f1s)

elif active == "Estrategia de Neumáticos":
    if not require_session(): st.stop()
    sgrid.render_strategy_dashboard(f1s)

# ── ANÁLISIS DE PILOTO (MICRO) ────────────────────────────────────────────────
elif active == "Mapa de Velocidad":
    if not require_session(): st.stop()
    st.header("Mapa de Calor: Velocidad (Km/h)")
    st.divider()
    driver, lap_n = select_driver_and_lap(f1s, "spd")
    if lap_n:
        sm.render_speed_heatmap(f1s, driver, show_corners, lap_n)

elif active == "Mapa de Marchas":
    if not require_session(): st.stop()
    st.header("Mapa de Calor: Marchas (1-8)")
    st.divider()
    driver, lap_n = select_driver_and_lap(f1s, "gear")
    if lap_n:
        gm.render_gear_heatmap(f1s, driver, show_corners, lap_n)

elif active == "Telemetry Trace":
    if not require_session(): st.stop()
    st.header("Análisis de Telemetría (Trace)")
    st.divider()
    
    driver, lap_n = select_driver_and_lap(f1s, "trace")
    if not lap_n: st.stop()
    
    # ── Tarjeta del piloto ───────────────────────────────────────────────────
    @st.cache_data(show_spinner=False)
    def fetch_driver_stats(ergast_id: str) -> dict:
        base  = "https://api.jolpi.ca/ergast/f1"
        stats = {"nationality": "—", "dob": "—", "wins": "—", "podiums": "—", "championships": "—"}
        if not ergast_id: return stats
        try:
            r = requests.get(f"{base}/drivers/{ergast_id}.json", timeout=5)
            if r.ok and r.json()["MRData"]["DriverTable"]["Drivers"]:
                d = r.json()["MRData"]["DriverTable"]["Drivers"][0]
                stats["nationality"] = d.get("nationality", "—")
                stats["dob"]         = d.get("dateOfBirth", "—")

            r = requests.get(f"{base}/drivers/{ergast_id}/results/1.json?limit=1000", timeout=5)
            if r.ok: stats["wins"] = r.json()["MRData"]["total"]

            r = requests.get(f"{base}/drivers/{ergast_id}/results.json?limit=1000", timeout=5)
            if r.ok:
                races = r.json()["MRData"]["RaceTable"]["Races"]
                stats["podiums"] = sum(1 for race in races for res in race["Results"] if int(res.get("position", 99)) <= 3)

            CAMPEONES = {"hamilton": "7", "max_verstappen": "4", "alonso": "2", "norris": "1", "vettel": "4", "raikkonen":"1"}
            stats["championships"] = CAMPEONES.get(ergast_id, "0")
        except Exception: pass
        return stats

    driver_info = f1s.get_driver(driver)
    BANDERAS = {
            "Spanish": "🇪🇸", "Monegasque": "🇲🇨", "Dutch": "🇳🇱", "British": "🇬🇧",
            "Mexican": "🇲🇽", "Australian": "🇦🇺", "French": "🇫🇷", "Japanese": "🇯🇵",
            "Canadian": "🇨🇦", "Thai": "🇹🇭", "American": "🇺🇸", "Finnish": "🇫🇮",
            "Chinese": "🇨🇳", "Danish": "🇩🇰", "German": "🇩🇪", "New Zealander": "🇳🇿",
            "Italian": "🇮🇹", "Brazilian": "🇧🇷", "Argentine": "🇦🇷", "Colombian": "🇨🇴"
    }
    ergast_id   = driver_info.get("DriverId", "")
    headshot    = driver_info.get("HeadshotUrl", None)
    first_name  = driver_info.get("FirstName", "")
    last_name   = driver_info.get("LastName", driver)
    team        = driver_info.get("TeamName", "—")
    number      = driver_info.get("DriverNumber", driver)

    stats = fetch_driver_stats(ergast_id)

    col_img, col_data = st.columns([1, 3], gap="large")
    with col_img:
        try:
            if headshot: st.image(headshot, width=200)
            else: raise ValueError
        except Exception:
            st.markdown(f"<div style='font-size:56px;text-align:center;padding:16px;'>{driver[:3].upper()}</div>", unsafe_allow_html=True)

    with col_data:
        st.markdown(f"### {number} {first_name} {last_name}")
        st.caption(team)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Victorias",    stats["wins"])
        m2.metric("Podios",       stats["podiums"])
        m3.metric("Campeonatos",  stats["championships"])
        m4.metric("Nacimiento",   stats["dob"])
        bandera = BANDERAS.get(stats['nationality'], "🏁")
        st.caption(f"Nacionalidad: {stats['nationality']} {bandera}")
        
    trace.render_telemetry_trace(f1s, driver, lap_n, show_corners)

# ── COMPARATIVAS COMPETITIVAS ─────────────────────────────────────────────────
elif active == "Panel de Equipo":
    if not require_session(): st.stop()
    st.header("Panel de Equipo (SAP)")
    st.divider()
    import TeamTelemetry as tteam
    tteam.render_team_telemetry(f1s, corners=show_corners)

elif active == "Comparativa de Sesión":
    if not require_session(): st.stop()
    st.header("Comparativa de Sesión")
    st.divider()
    os.render_session_compare(f1s, corners=show_corners)

elif active == "La Vuelta Ideal":
    if not require_session(): st.stop()
    st.header("La Vuelta Ideal · Microsectores")
    st.divider()
    ol.render_ideal_lap(f1s, corners=show_corners)

elif active == "Coche Fantasma":
    if not require_session(): st.stop()
    st.header("Coche Fantasma")
    st.divider()
    ghost.render_ghost_car(sesion=f1s, corners=show_corners)

# ── DINÁMICA VEHICULAR ────────────────────────────────────────────────────────
elif active == "Diagrama G-G":
    if not require_session(): st.stop()
    st.header("Diagrama de Fricción (G-G)")
    st.divider()
    driver, lap_n = select_driver_and_lap(f1s, "gg")
    if lap_n:
        gg.render_demand_map(f1s, driver, lap_n)

elif active == "Carga Aerodinámica":
    if not require_session(): st.stop()
    st.header("Carga Aerodinámica (Estimación G-Lat vs Velocidad)")
    st.divider()
    
    import Aero as aero
    
    # Reutilizamos tu helper para no repetir código
    driver, lap_n = select_driver_and_lap(f1s, "aero")
    if lap_n:
        aero.render_aero_analysis(f1s, driver, lap_n)

elif active == "Mapa de Carga por Rueda":
    if not require_session(): st.stop()
    
    driver, lap_n = select_driver_and_lap(f1s, "tyre")
    if lap_n:
        tload.render_tyre_load(f1s, driver, lap_n)