"""
TALOS — Telemetry and Lap Optimization System
Buscador moderno de Grand Prix · Streamlit + fastf1
"""

import streamlit as st
import pandas as pd
import requests
import fastf1

import Circuit2d as c2d
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
import session_loader as sl
import ui_assets


# ─── NAVEGACIÓN ───────────────────────────────────────────────────────────────

NAV = {
    "Información de Sesión": [
        ("Resumen de la sesión",    "◉", "Posiciones, tiempos y resumen completo"),
        ("Cronología de Eventos",   "▷", "Banderas, SC y VSC sobre la línea de tiempo"),
        ("Estrategia de Neumáticos","◆", "Gantt de stints y compuestos usados"),
    ],
    "Análisis de Piloto": [
        ("Mapa de Velocidad",       "△", "Circuito coloreado por velocidad (km/h)"),
        ("Mapa de Marchas",         "◈", "Circuito coloreado por marcha seleccionada"),
        ("Telemetry Trace",         "⌇", "Canales de telemetría sincronizados"),
    ],
    "Comparativas": [
        ("Panel de Equipo",         "⊞", "Análisis por equipos y comparativa interna"),
        ("Comparativa de Sesión",   "⇌", "Overlays multi-piloto sobre la misma vuelta"),
        ("La Vuelta Ideal",         "★", "Microsectores óptimos de toda la parrilla"),
        ("Coche Fantasma",          "◯", "Animación 2D de persecución entre pilotos"),
    ],
    "Dinámica Vehicular": [
        ("Diagrama G-G",            "◕", "Círculo de tracción · envolvente de Gs"),
        ("Carga Aerodinámica",      "▲", "G-Lat vs Velocidad · estimación downforce"),
        ("Mapa de Carga por Rueda", "◎", "Distribución de carga por eje y vuelta"),
    ],
}

MODULE_KEYS = [m for cat in NAV.values() for m, _, _ in cat]


# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="F1 · Grand Prix Search",
    page_icon="🏎",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ─── CSS + JS GLOBALES ────────────────────────────────────────────────────────

st.markdown(ui_assets.CSS_THEME, unsafe_allow_html=True)
st.markdown(ui_assets.JS_TWEAKS, unsafe_allow_html=True)


# ─── ESTADO ───────────────────────────────────────────────────────────────────

sl.init_session_state()

# ─── BANDERAS ─────────────────────────────────────────────────────────────────

COUNTRY_FLAGS = {
    "Australia": "🇦🇺", "Austria": "🇦🇹", "Azerbaijan": "🇦🇿",
    "Bahrain": "🇧🇭", "Belgium": "🇧🇪", "Brazil": "🇧🇷",
    "Canada": "🇨🇦", "China": "🇨🇳", "France": "🇫🇷",
    "Germany": "🇩🇪", "Great Britain": "🇬🇧", "Hungary": "🇭🇺",
    "Italy": "🇮🇹", "Japan": "🇯🇵", "Mexico": "🇲🇽", "Monaco": "🇲🇨",
    "Netherlands": "🇳🇱", "Portugal": "🇵🇹", "Qatar": "🇶🇦",
    "Russia": "🇷🇺", "Saudi Arabia": "🇸🇦", "Singapore": "🇸🇬",
    "Spain": "🇪🇸", "Turkey": "🇹🇷", "United Arab Emirates": "🇦🇪",
    "United Kingdom": "🇬🇧", "United States": "🇺🇸", "USA": "🇺🇸",
}

COUNTRY_CODES = {
    "Australia": "au", "Austria": "at", "Azerbaijan": "az",
    "Bahrain": "bh", "Belgium": "be", "Brazil": "br",
    "Canada": "ca", "China": "cn", "France": "fr",
    "Germany": "de", "Great Britain": "gb", "Hungary": "hu",
    "Italy": "it", "Japan": "jp", "Mexico": "mx", "Monaco": "mc",
    "Netherlands": "nl", "Portugal": "pt", "Qatar": "qa",
    "Russia": "ru", "Saudi Arabia": "sa", "Singapore": "sg",
    "Spain": "es", "Turkey": "tr", "United Arab Emirates": "ae",
    "United Kingdom": "gb", "United States": "us", "USA": "us",
}


def flag_html(country: str, size: int = 28) -> str:
    """Devuelve un span con la bandera del país usando flag-icons (jsDelivr CDN)."""
    code = COUNTRY_CODES.get(country, "")
    if code:
        h = max(12, size // 2)
        w = int(h * 4 / 3)
        return (
            f'<span class="fi fi-{code}" '
            f'style="width:{w}px;height:{h}px;border-radius:3px;'
            f'display:inline-block;vertical-align:middle;'
            f'background-size:cover;"></span>'
        )
    return "🏁"


DRIVER_FLAGS = {
    "Spanish": "🇪🇸", "Monegasque": "🇲🇨", "Dutch": "🇳🇱", "British": "🇬🇧",
    "Mexican": "🇲🇽", "Australian": "🇦🇺", "French": "🇫🇷", "Japanese": "🇯🇵",
    "Canadian": "🇨🇦", "Thai": "🇹🇭", "American": "🇺🇸", "Finnish": "🇫🇮",
    "Chinese": "🇨🇳", "Danish": "🇩🇰", "German": "🇩🇪", "New Zealander": "🇳🇿",
    "Italian": "🇮🇹", "Brazilian": "🇧🇷", "Argentine": "🇦🇷", "Colombian": "🇨🇴",
}

SESSION_SHORT = {
    "Practice 1": "FP1", "Practice 2": "FP2", "Practice 3": "FP3",
    "Qualifying": "Q",   "Race": "R",
    "Sprint": "S",       "Sprint Qualifying": "SQ", "Sprint Shootout": "SS",
}

# ── Metadata visual por tipo de sesión ───────────────────────────────────────
SESSION_META = {
    "Practice 1":        {"cls": "scard-fp",  "color": "#8A8D9E", "short": "FP1", "full": "Free Practice 1"},
    "Practice 2":        {"cls": "scard-fp",  "color": "#8A8D9E", "short": "FP2", "full": "Free Practice 2"},
    "Practice 3":        {"cls": "scard-fp",  "color": "#8A8D9E", "short": "FP3", "full": "Free Practice 3"},
    "Qualifying":        {"cls": "scard-q",   "color": "#C40026", "short": "Q",   "full": "Qualifying"},
    "Sprint":            {"cls": "scard-spr", "color": "#B138DD", "short": "S",   "full": "Sprint Race"},
    "Sprint Qualifying": {"cls": "scard-spr", "color": "#B138DD", "short": "SQ",  "full": "Sprint Qualifying"},
    "Sprint Shootout":   {"cls": "scard-spr", "color": "#B138DD", "short": "SS",  "full": "Sprint Shootout"},
    "Race":              {"cls": "scard-r",   "color": "#E8002D", "short": "R",   "full": "Formula 1 Grand Prix"},
}

# ── Color de acento por categoría del hub (Escala de Rojos) ─────────────────
CAT_COLORS = {
    "Información de Sesión": ("mc-red-1", "#E8002D"), # Rojo F1 (Puro)
    "Análisis de Piloto":    ("mc-red-2", "#C40026"), # Rojo un poco más oscuro
    "Comparativas":          ("mc-red-3", "#99001D"), # Rojo vino
    "Dinámica Vehicular":    ("mc-red-4", "#6E0015"), # Granate oscuro
}


# ─── HELPERS ──────────────────────────────────────────────────────────────────

def _nat(d) -> bool:
    if d is None:
        return True
    try:
        return bool(pd.isna(d))
    except Exception:
        return False


def fmt_date(d) -> str:
    if _nat(d):
        return "—"
    try:
        return pd.Timestamp(d).strftime("%d %b").upper()
    except Exception:
        return "—"


def fmt_dt(d) -> str:
    if _nat(d):
        return "—"
    try:
        return pd.Timestamp(d).strftime("%d %b · %H:%M")
    except Exception:
        return "—"


# ─── CABECERA (selector de año + título) ──────────────────────────────────────
# haciendo clic en el lollipop del mapamundi interactivo.

def _back_to_grid_button():
    """Botón prominente para volver a seleccionar otro circuito."""
    st.markdown("""
    <style>
    .change-circuit-row [data-testid="stButton"] > button {
        background: rgba(255,255,255,0.04) !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
        border-left: 3px solid #00D7B5 !important;
        border-radius: 2px !important;
        color: #E8E9EF !important;
        font-family: 'Titillium Web','Inter',sans-serif !important;
        font-style: italic !important;
        font-weight: 800 !important;
        letter-spacing: 0.16em !important;
        text-transform: uppercase !important;
        font-size: 0.72rem !important;
        padding: 9px 16px !important;
        transition: all 0.18s ease !important;
    }
    .change-circuit-row [data-testid="stButton"] > button:hover {
        background: rgba(0,215,181,0.10) !important;
        border-color: rgba(0,215,181,0.55) !important;
        color: #00D7B5 !important;
        transform: translateX(-2px) !important;
        box-shadow: 0 0 18px rgba(0,215,181,0.18) !important;
    }
    </style>
    <div class="change-circuit-row"></div>
    """, unsafe_allow_html=True)
    c1, _ = st.columns([0.22, 0.78])
    with c1:
        if st.button("◂ Cambiar circuito",
                     key=f"chg_circ_{st.session_state.get('view','x')}_{st.session_state.get('active_module','x')}",
                     use_container_width=True):
            st.session_state["session_loaded"] = False
            st.session_state["active_module"] = None
            sl.go("grid")


def _minimal_sidebar():
    """Sidebar slim para vistas grid/sessions: marca + navegaci\u00f3n m\u00ednima.
    El sidebar arranca colapsado; el chevron de Streamlit lo expande."""
    with st.sidebar:
        st.markdown(
            "<div style='font-family:Titillium Web,Inter,sans-serif;"
            "font-style:italic;font-weight:900;font-size:1.6rem;"
            "letter-spacing:-0.02em;color:#fff;margin:0 0 4px 0;'>TALOS</div>",
            unsafe_allow_html=True,
        )
        st.caption("Telemetry & Lap Optimization")
        st.divider()
        if st.button("\u25c0  Buscador de GP", use_container_width=True, key="ms_grid"):
            sl.go("grid")
        if st.session_state.get("session_loaded"):
            if st.button("\u22a1  Hub de m\u00f3dulos", use_container_width=True, key="ms_hub"):
                st.session_state["active_module"] = None
                sl.go("analysis")


def _back_to_hub_button():
    """Bot\u00f3n compacto para volver al hub desde cualquier m\u00f3dulo."""
    st.markdown("""
    <style>
    .back-hub-row [data-testid="stButton"] > button {
        background: rgba(255,255,255,0.04) !important;
        border: 1px solid rgba(255,255,255,0.10) !important;
        border-left: 3px solid #E8002D !important;
        border-radius: 2px !important;
        color: #E8E9EF !important;
        font-style: italic !important;
        font-weight: 700 !important;
        letter-spacing: 0.04em !important;
        text-transform: uppercase !important;
        font-size: 0.78rem !important;
        padding: 8px 16px !important;
        transition: all 0.15s ease !important;
    }
    .back-hub-row [data-testid="stButton"] > button:hover {
        background: rgba(232,0,45,0.10) !important;
        border-color: rgba(232,0,45,0.5) !important;
        transform: translateX(-2px) !important;
    }
    </style>
    <div class="back-hub-row"></div>
    """, unsafe_allow_html=True)
    c1, _ = st.columns([0.22, 0.78])
    with c1:
        if st.button("◂ Volver al hub", key=f"back_hub_{st.session_state.get('active_module','x')}",
                     use_container_width=True):
            st.session_state["active_module"] = None
            st.rerun()


def _header():
    c_yr, _ = st.columns([0.9, 9.1])

    with c_yr:
        with st.popover(
            f"  {st.session_state['selected_year']}",
            use_container_width=True,
        ):
            st.caption("Año")
            for yr in range(2026, 2017, -1):
                if st.button(str(yr), key=f"yr_{yr}", use_container_width=True):
                    st.session_state["selected_year"] = yr
                    st.session_state["selected_event"] = None
                    sl.go("grid")

    st.markdown(
        '<h1 class="gp-title">FORMULA 1 <span>GRAND PRIX</span> SEARCH</h1>',
        unsafe_allow_html=True,
    )


# ─── VISTA GRID — Mapamundi interactivo con lollipops ─────────────────────────

def view_grid():
    _minimal_sidebar()
    _header()

    # ── Cargar calendario ──────────────────────────────────────────────────
    try:
        sched = sl.load_schedule(st.session_state["selected_year"])
    except Exception as e:
        st.error(f"No se pudo cargar el calendario: {e}")
        return

    if sched.empty:
        st.info("Sin eventos para el año seleccionado.")
        return

    # ── Instrucción de uso ─────────────────────────────────────────────────
    st.markdown(
        '<p style="color:#555870;font-size:0.8rem;font-weight:500;'
        'margin:-1rem 0 0.6rem 0;letter-spacing:0.03em;">'
        '▸ Haz clic en un lollipop para acceder al circuito</p>',
        unsafe_allow_html=True,
    )

    # ── Construir y renderizar el mapa ─────────────────────────────────────
    fig = ui_assets.build_world_map(sched)

    map_event = st.plotly_chart(
        fig,
        use_container_width=True,
        on_select="rerun",
        selection_mode=["points"],
        key="world_map",
        config={
            "scrollZoom": True,
            "displayModeBar": True,
            "modeBarButtonsToRemove": [
                "select2d", "lasso2d", "toImage",
                "sendDataToCloud", "toggleSpikelines",
            ],
            "displaylogo": False,
        },
    )

    # ── Gestionar selección de punto ───────────────────────────────────────
    if (
        map_event
        and hasattr(map_event, "selection")
        and map_event.selection
        and map_event.selection.points
    ):
        point = map_event.selection.points[0]
        customdata = point.get("customdata")

        # customdata = [event_name, country, location, round_n, date_str, city]
        if customdata and len(customdata) >= 1:
            ev_name = customdata[0]

            # Buscar la fila del evento en el calendario
            mask   = sched["EventName"] == ev_name
            ev_row = sched[mask]

            if not ev_row.empty:
                ev_dict = ev_row.iloc[0].to_dict()
                st.session_state["selected_event"] = ev_dict
                sl.go("sessions")
            else:
                st.warning(f"No se encontró el evento '{ev_name}' en el calendario.")


# ─── VISTA SESIONES ───────────────────────────────────────────────────────────

def view_sessions():
    _minimal_sidebar()
    st.markdown(ui_assets.CSS_SESSIONS, unsafe_allow_html=True)
    _back_to_grid_button()

    ev = st.session_state.get("selected_event")
    if not ev:
        sl.go("grid")
        return


    # ── Hero del circuito ─────────────────────────────────────────────────
    country  = str(ev.get("Country",   "—"))
    flag     = flag_html(country, size=56)
    raw_name = str(ev.get("EventName", "—"))
    location = str(ev.get("Location",  "—"))
    round_n  = ev.get("RoundNumber",   "?")
    date_s   = fmt_date(ev.get("EventDate"))
    yr       = st.session_state["selected_year"]

    # Partir nombre: "Bahrain Grand Prix" → "BAHRAIN" + "GRAND PRIX"
    words = raw_name.split()
    if len(words) >= 3:
        name_top  = " ".join(words[:-2]).upper()
        name_bot  = " ".join(words[-2:]).upper()
    else:
        name_top  = raw_name.upper()
        name_bot  = ""

    st.markdown(f"""
    <div class="circuit-hero">
        <div class="circuit-hero-pill">◉&nbsp; ROUND {round_n} &nbsp;·&nbsp; {yr}</div>
        <div style="display:flex;align-items:center;gap:22px;margin-bottom:14px;">
            <div style="flex-shrink:0;">{flag}</div>
            <div class="circuit-hero-name">
                {name_top}<br><span>{name_bot}</span>
            </div>
        </div>
        <div class="circuit-hero-meta">
            <span>{location}</span>
            <span class="sep">·</span>
            <span>{country}</span>
            <span class="sep">·</span>
            <span>{date_s}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Sesiones ──────────────────────────────────────────────────────────
    sessions = []
    for i in range(1, 6):
        sn = ev.get(f"Session{i}")
        sd = ev.get(f"Session{i}Date")
        if sn and not (isinstance(sn, float) and _nat(sn)):
            sessions.append((str(sn), sd))
    if not sessions:
        sessions = [
            ("Practice 1", None), ("Practice 2", None), ("Practice 3", None),
            ("Qualifying", None), ("Race", None),
        ]

    cols = st.columns(len(sessions), gap="medium")
    for idx, (sn, sd) in enumerate(sessions):
        with cols[idx]:
            _session_card(sn, sd, ev)


def _session_card(session_name: str, session_date, ev):
    gp_name = str(ev.get("EventName", "—"))
    meta    = SESSION_META.get(session_name.strip(), {
        "cls": "scard-fp", "color": "#555870",
        "short": session_name[:3].upper(), "full": session_name,
    })
    css_cls = meta["cls"]
    color   = meta["color"]
    short   = meta["short"]
    full    = meta["full"]
    is_race = short == "R"
    when    = fmt_dt(session_date)
    date_s  = fmt_date(session_date)

    code_size = "3.6rem" if is_race else "2.8rem"
    pad_top   = "28px"   if is_race else "22px"

    st.markdown(f"""
    <div class="scard {css_cls}" style="padding:{pad_top} 20px 20px;">
        <div style="display:flex;justify-content:space-between;
             align-items:center;margin-bottom:20px;position:relative;z-index:1;">
            <span style="background:{color}1A;color:{color};
                 font-size:0.6rem;font-weight:800;letter-spacing:0.14em;
                 text-transform:uppercase;border-radius:2px;padding:3px 9px;
                 border:1px solid {color}33;font-style:italic;">{short}</span>
            <span class="mono" style="color:#555870;font-size:0.7rem;">{date_s}</span>
        </div>
        <div style="font-size:{code_size};font-weight:900;color:{color};
             font-style:italic;letter-spacing:-0.05em;line-height:0.88;
             margin-bottom:12px;position:relative;z-index:1;">
            {short}
        </div>
        <div style="color:#E8E9EF;font-size:0.81rem;
             font-weight:{"700" if is_race else "500"};
             line-height:1.4;margin-bottom:20px;position:relative;z-index:1;">{full}</div>
        <div style="height:1px;background:rgba(255,255,255,0.06);
             margin-bottom:13px;position:relative;z-index:1;"></div>
        <div class="mono" style="color:#555870;font-size:0.69rem;position:relative;z-index:1;">{when}</div>
    </div>
    """, unsafe_allow_html=True)

    if st.button(
        f"▸  Cargar {full}",
        key=f"load_{gp_name}_{session_name}",
        use_container_width=True,
    ):
        with st.spinner(f"Sincronizando {session_name} con FastF1…"):
            try:
                yr = st.session_state["selected_year"]
                s  = sl.load_f1_session(yr, gp_name, session_name)
                st.session_state["f1_session"]     = s
                st.session_state["session_loaded"] = True
                st.session_state["session_label"]  = f"{gp_name}  {yr} · {short}"
                st.session_state["active_module"]  = None
                sl.go("analysis")
            except Exception as e:
                st.error(f"Error: {e}")


# ─── VISTA ANÁLISIS (sidebar + módulos con tabs) ──────────────────────────────

def _analysis_sidebar():
    with st.sidebar:
        st.markdown("# TALOS")
        st.caption("Telemetry & Lap Optimization")
        st.divider()

        if st.button("◀  Buscador de GP", use_container_width=True):
            sl.go("grid")

        if st.button("⬡  Hub de módulos", use_container_width=True):
            st.session_state["active_module"] = None
            st.rerun()

        st.divider()

        for cat, mods in NAV.items():
            st.caption(cat.upper())
            for m, icon, _ in mods:
                active = st.session_state["active_module"] == m
                lbl = f"▸  {m}" if active else f"    {m}"
                if st.button(lbl, key=f"nav_{m}", use_container_width=True):
                    st.session_state["active_module"] = m
            st.markdown("")

        st.divider()
        c = st.checkbox(
            "Mostrar curvas",
            value=st.session_state.get("corners", False),
        )
        st.session_state["corners"] = c
        st.divider()

        if st.session_state["session_loaded"]:
            st.success(f"✓ {st.session_state.get('session_label', '')}")
        else:
            st.warning("Sin sesión activa")


def _hub_stats(f1s):
    """Panel derecho del hub: métricas rápidas + clasificación de sesión."""
    if f1s is None:
        return

    try:
        ev      = f1s.event
        country = str(ev.get("Country", ""))
        flag    = flag_html(country, size=40)
        gp_name = str(ev.get("EventName", ""))
        loc     = str(ev.get("Location", ""))
        fmt     = str(ev.get("EventFormat", "")).replace("_", " ").title()
    except Exception:
        flag, gp_name, loc, fmt = "🏁", "", "", ""

    st.markdown(f"""
    <div style="background:var(--surface2);border-radius:10px;
         padding:14px 16px;margin-bottom:1.2rem;">
        <div style="font-size:1.6rem;margin-bottom:6px;">{flag}</div>
        <div style="color:var(--white);font-weight:800;font-size:1rem;
             text-transform:uppercase;letter-spacing:-0.01em;">{gp_name}</div>
        <div style="color:var(--t2);font-size:0.78rem;margin-top:2px;">{loc} · {fmt}</div>
    </div>
    """, unsafe_allow_html=True)

    try:
        fastest  = f1s.laps.pick_fastest()
        lt       = fastest["LapTime"]
        ts       = lt.total_seconds()
        lt_str   = f"{int(ts // 60)}:{ts % 60:06.3f}"
        fast_drv = str(fastest["Driver"])
    except Exception:
        lt_str, fast_drv = "—", "—"

    try:
        n_vueltas = int(f1s.laps["LapNumber"].max()) if not f1s.laps.empty else "—"
    except Exception:
        n_vueltas = "—"

    m1, m2 = st.columns(2)
    m1.metric("Vuelta rápida", lt_str)
    m2.metric("Piloto",        fast_drv)
    m3, m4 = st.columns(2)
    m3.metric("Pilotos",  len(f1s.drivers))
    m4.metric("Vueltas",  n_vueltas)

    st.divider()
    st.caption("CLASIFICACIÓN FINAL")

    try:
        res = f1s.results
        if res is None or res.empty:
            raise ValueError("sin resultados")

        for _, row in res.iterrows():
            pos     = str(int(row.get("Position", "—")))
            abbr    = str(row.get("Abbreviation", ""))
            team    = str(row.get("TeamName", ""))
            pts_raw = row.get("Points", None)
            pts     = (
                f"{int(pts_raw)} pts"
                if (pts_raw is not None and pd.notna(pts_raw) and int(pts_raw) > 0)
                else ""
            )

            oro = "#FFC800"
            plata = "#E2E8F0"
            bronce = "#D97706"
            pos_col = oro if pos == "1" else plata if pos =="2" else bronce if pos =="3" else "var(--t2)"
            weight  = "1000" if pos in {"1","2","3"} else "600"

            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:9px;
                 padding:5px 0;border-bottom:1px solid var(--border);">
                <span style="color:{pos_col};font-weight:{weight};width:20px;
                     font-size:0.8rem;flex-shrink:0;text-align:right;">{pos}</span>
                <span style="color:var(--white);font-weight:700;font-size:0.82rem;
                     width:32px;flex-shrink:0;">{abbr}</span>
                <span style="color:var(--t2);font-size:0.74rem;flex:1;
                     overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">{team}</span>
                <span style="color:var(--t1);font-size:0.76rem;
                     flex-shrink:0;">{pts}</span>
            </div>
            """, unsafe_allow_html=True)

    except Exception:
        st.caption("(Sin resultados — vueltas más rápidas por piloto)")
        try:
            best = (
                f1s.laps.pick_accurate()
                .groupby("Driver")["LapTime"]
                .min()
                .sort_values()
                .reset_index()
            )
            for i, row in best.iterrows():
                drv = str(row["Driver"])
                ts  = row["LapTime"].total_seconds()
                t   = f"{int(ts // 60)}:{ts % 60:06.3f}"
                pos_col = "var(--red)" if i == 0 else "var(--t2)"
                st.markdown(f"""
                <div style="display:flex;align-items:center;gap:9px;
                     padding:5px 0;border-bottom:1px solid var(--border);">
                    <span style="color:{pos_col};font-weight:700;width:20px;
                         font-size:0.8rem;flex-shrink:0;">{i+1}</span>
                    <span style="color:var(--white);font-weight:700;
                         font-size:0.82rem;flex:1;">{drv}</span>
                    <span style="color:var(--t1);font-size:0.76rem;">{t}</span>
                </div>
                """, unsafe_allow_html=True)
        except Exception:
            st.caption("Sin datos disponibles.")


def _hub_podium(f1s):
    """Visualización de podio para el hub — P1/P2/P3 con colores de equipo."""

    TEAM_COLORS = {
        "Mercedes":         "#00D2BE",
        "Red Bull Racing":  "#3671C6",
        "Ferrari":          "#E8002D",
        "McLaren":          "#FF8000",
        "Aston Martin":     "#358C75",
        "Alpine":           "#2293D1",
        "Williams":         "#64C4FF",
        "RB":               "#6692FF",
        "AlphaTauri":       "#5E8FAA",
        "Haas F1 Team":     "#B6BABD",
        "Kick Sauber":      "#52E252",
        "Alfa Romeo":       "#C92D4B",
    }

    try:
        top3   = []
        is_race = False

        try:
            res = f1s.results
            if res is not None and not res.empty:
                session_name = getattr(f1s, "name", "") or ""
                is_race = "Race" in session_name and "Sprint" not in session_name
                for _, row in res.head(3).iterrows():
                    abbr = str(row.get("Abbreviation", "—"))
                    name = str(row.get("FullName", abbr))
                    team = str(row.get("TeamName", ""))
                    pts  = row.get("Points", None)
                    if is_race and pts is not None and pd.notna(pts):
                        stat = f"{int(pts)} PTS"
                    else:
                        try:
                            drv_laps = f1s.laps.pick_drivers(abbr).pick_accurate()
                            if not drv_laps.empty:
                                lt = drv_laps.pick_fastest()["LapTime"]
                                ts = lt.total_seconds()
                                stat = f"{int(ts//60)}:{ts%60:06.3f}"
                            else:
                                stat = ""
                        except Exception:
                            stat = ""
                    top3.append({"abbr": abbr, "name": name,
                                 "team": team, "stat": stat})
        except Exception:
            pass

        if not top3:
            best = (
                f1s.laps.pick_accurate()
                .groupby("Driver")["LapTime"]
                .min()
                .sort_values()
                .head(3)
                .reset_index()
            )
            for _, row in best.iterrows():
                drv = str(row["Driver"])
                ts  = row["LapTime"].total_seconds()
                top3.append({
                    "abbr": drv, "name": drv, "team": "",
                    "stat": f"{int(ts//60)}:{ts%60:06.3f}",
                })

        while len(top3) < 3:
            top3.append({"abbr": "—", "name": "—", "team": "", "stat": ""})

        p1, p2, p3 = top3[0], top3[1], top3[2]

        def block(d, pos, height):
            color = TEAM_COLORS.get(d["team"], "#9497A8")
            medal = ["🥇", "🥈", "🥉"][pos - 1]
            return (
                f'<div style="display:flex;flex-direction:column;align-items:center;flex:1;min-width:0;">'
                f'<div style="font-size:0.65rem;color:#9497A8;font-weight:600;text-align:center;'
                f'margin-bottom:8px;padding:0 4px;white-space:nowrap;overflow:hidden;'
                f'text-overflow:ellipsis;max-width:100%;">{medal}&nbsp;{d["name"]}</div>'
                f'<div style="width:100%;height:{height}px;'
                f'background:linear-gradient(160deg,{color}28 0%,{color}0a 100%);'
                f'border:1px solid {color}50;border-bottom:3px solid {color};'
                f'border-radius:10px 10px 0 0;'
                f'display:flex;flex-direction:column;align-items:center;'
                f'justify-content:center;gap:6px;padding:14px 6px;">'
                f'<div style="font-size:0.58rem;color:{color};font-weight:800;'
                f'letter-spacing:0.14em;text-transform:uppercase;">P{pos}</div>'
                f'<div style="font-size:2.1rem;font-weight:900;color:#ffffff;'
                f'letter-spacing:-0.03em;line-height:1;">{d["abbr"]}</div>'
                f'<div style="font-size:0.68rem;color:{color};font-weight:700;'
                f'text-align:center;letter-spacing:0.02em;">{d["stat"]}</div>'
                f'<div style="font-size:0.62rem;color:#555870;text-align:center;'
                f'margin-top:2px;">{d["team"]}</div>'
                f'</div></div>'
            )

        st.markdown(
            '<p style="color:#555870;font-size:0.68rem;font-weight:700;'
            'text-transform:uppercase;letter-spacing:0.1em;'
            'margin:0.4rem 0 0.8rem 0;">PODIO DE SESIÓN</p>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div style="display:flex;align-items:flex-end;gap:8px;height:240px;">'
            f'{block(p2, 2, 155)}'
            f'{block(p1, 1, 215)}'
            f'{block(p3, 3, 120)}'
            f'</div>',
            unsafe_allow_html=True,
        )

    except Exception:
        pass


def _hub():
    """Hub de módulos — organizado por pestañas (tabs)."""
    label = st.session_state.get("session_label", "")
    f1s   = st.session_state.get("f1_session")

    st.markdown(ui_assets.CSS_HUB, unsafe_allow_html=True)

    # ── Banner sesión activa ──
    st.markdown(f"""
    <div class="hub-banner">
        <div class="hub-banner-dot"></div>
        <div>
            <div class="hub-banner-label">Sesión activa</div>
            <div class="hub-banner-name">{label}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    left, right = st.columns([3, 2], gap="large")

    with left:
        # ── Pestañas de categoría (st.tabs) ──
        # Obtenemos los nombres de las categorías directamente de las claves de NAV
        nombres_categorias = list(NAV.keys())
        pestañas = st.tabs(nombres_categorias)

        # Iteramos sobre las pestañas y el contenido de NAV al mismo tiempo
        for tab, (cat, mods) in zip(pestañas, NAV.items()):
            with tab:
                cls_name, color = CAT_COLORS.get(cat, ("mc-blue", "#3B82F6"))

                # Opcional: Espaciado superior dentro de la pestaña
                st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)

                # Tarjetas de módulo en columnas
                cols = st.columns(len(mods), gap="small")
                for col, (m, icon, desc) in zip(cols, mods):
                    with col:
                        st.markdown(f"""
                        <div class="mcard {cls_name}">
                            <span class="mcard-icon">{icon}</span>
                            <div class="mcard-name">{m}</div>
                            <div class="mcard-desc">{desc}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        if st.button(f"▸  Abrir", key=f"hub_{m}", use_container_width=True):
                            st.session_state["active_module"] = m
                            st.rerun()

        # ── Podio ──
        if f1s is not None:
            st.divider()
            _hub_podium(f1s)

    with right:
        _hub_stats(f1s)


def view_analysis():
    if not st.session_state["session_loaded"]:
        sl.go("grid")
        return

    _analysis_sidebar()

    f1s          = st.session_state["f1_session"]
    show_corners = st.session_state.get("corners", False)
    active       = st.session_state["active_module"]

    if active is None:
        _back_to_grid_button()
        _hub()
        return

    # ── Botón volver al hub (en TODOS los módulos) ──
    _back_to_hub_button()

    if active == "Resumen de la sesión":
        ss.render_session_summary(f1s)

    elif active == "Cronología de Eventos":
        stl.render_timeline(f1s)

    elif active == "Estrategia de Neumáticos":
        sgrid.render_strategy_dashboard(f1s)

    elif active == "Mapa de Velocidad":
        st.header("Mapa de Calor · Velocidad (km/h)")
        st.divider()
        drv, lap_n = _driver_lap(f1s, "spd")
        if lap_n:
            sm.render_speed_heatmap(f1s, drv, show_corners, lap_n)

    elif active == "Mapa de Marchas":
        st.header("Mapa de Calor · Marchas (1-8)")
        st.divider()
        drv, lap_n = _driver_lap(f1s, "gear")
        if lap_n:
            gm.render_gear_heatmap(f1s, drv, show_corners, lap_n)

    elif active == "Telemetry Trace":
        st.header("Análisis de Telemetría · Trace")
        st.divider()
        drv, lap_n = _driver_lap(f1s, "trace")
        if not lap_n:
            return
        _driver_card(f1s, drv)
        trace.render_telemetry_trace(f1s, drv, lap_n, show_corners)

    elif active == "Panel de Equipo":
        st.header("Panel de Equipo")
        st.divider()
        import TeamTelemetry as tteam
        tteam.render_team_telemetry(f1s, corners=show_corners)

    elif active == "Comparativa de Sesión":
        st.header("Comparativa de Sesión")
        st.divider()
        os.render_session_compare(f1s, corners=show_corners)

    elif active == "La Vuelta Ideal":
        st.header("La Vuelta Ideal · Microsectores")
        st.divider()
        ol.render_ideal_lap(f1s, corners=show_corners)

    elif active == "Coche Fantasma":
        st.header("Coche Fantasma")
        st.divider()
        ghost.render_ghost_car(sesion=f1s, corners=show_corners)

    elif active == "Diagrama G-G":
        st.header("Diagrama de Fricción (G-G)")
        st.divider()
        drv, lap_n = _driver_lap(f1s, "gg")
        if lap_n:
            gg.render_demand_map(f1s, drv, lap_n)

    elif active == "Carga Aerodinámica":
        st.header("Carga Aerodinámica · G-Lat vs Velocidad")
        st.divider()
        drv, lap_n = _driver_lap(f1s, "aero")
        if lap_n:
            aero.render_aero_analysis(f1s, drv, lap_n)

    elif active == "Mapa de Carga por Rueda":
        drv, lap_n = _driver_lap(f1s, "tyre")
        if lap_n:
            tload.render_tyre_load(f1s, drv, lap_n)


# ─── HELPERS DE MÓDULO ────────────────────────────────────────────────────────

def _driver_lap(f1s, prefix: str):
    c1, c2 = st.columns(2)
    names = {d: f1s.get_driver(d).get("FullName", str(d)) for d in f1s.drivers}
    with c1:
        drv = st.selectbox(
            "Piloto",
            options=f1s.drivers,
            format_func=lambda x: names.get(x, x),
            key=f"{prefix}_drv",
        )
    laps = f1s.laps.pick_drivers(drv).pick_accurate()
    if laps.empty:
        st.warning(f"Sin datos de telemetría válidos para {drv}.")
        return drv, None
    best = laps.pick_fastest()["LapNumber"]
    lst  = laps["LapNumber"].tolist()
    didx = lst.index(best) if best in lst else 0
    with c2:
        lap = st.selectbox(
            "Vuelta",
            options=lst,
            index=didx,
            format_func=lambda x: (
                f"Vuelta {int(x)} (Best)" if x == best else f"Vuelta {int(x)}"
            ),
            key=f"{prefix}_lap",
        )
    return drv, int(lap)


@st.cache_data(show_spinner=False)
def _ergast(eid: str) -> dict:
    base  = "https://api.jolpi.ca/ergast/f1"
    stats = {
        "nationality": "—", "dob": "—",
        "wins": "—", "podiums": "—", "championships": "—",
    }
    if not eid:
        return stats
    try:
        r = requests.get(f"{base}/drivers/{eid}.json", timeout=5)
        if r.ok:
            dl = r.json()["MRData"]["DriverTable"]["Drivers"]
            if dl:
                stats["nationality"] = dl[0].get("nationality", "—")
                stats["dob"]         = dl[0].get("dateOfBirth", "—")
        r = requests.get(
            f"{base}/drivers/{eid}/results/1.json?limit=1000", timeout=5
        )
        if r.ok:
            stats["wins"] = r.json()["MRData"]["total"]
        r = requests.get(
            f"{base}/drivers/{eid}/results.json?limit=1000", timeout=5
        )
        if r.ok:
            races = r.json()["MRData"]["RaceTable"]["Races"]
            stats["podiums"] = sum(
                1 for race in races for res in race["Results"]
                if int(res.get("position", 99)) <= 3
            )
        CHAMPS = {
            "hamilton": "7", "max_verstappen": "4", "alonso": "2",
            "norris": "1", "vettel": "4", "raikkonen": "1",
        }
        stats["championships"] = CHAMPS.get(eid, "0")
    except Exception:
        pass
    return stats


def _driver_card(f1s, drv):
    info = f1s.get_driver(drv)
    eid  = info.get("DriverId", "")
    shot = info.get("HeadshotUrl")
    fn   = info.get("FirstName", "")
    ln   = info.get("LastName", drv)
    team = info.get("TeamName", "—")
    num  = info.get("DriverNumber", drv)
    st_  = _ergast(eid)

    ci, cd = st.columns([1, 3], gap="large")
    with ci:
        try:
            if shot:
                st.image(shot, width=200)
            else:
                raise ValueError
        except Exception:
            st.markdown(
                f"<div style='font-size:52px;text-align:center;padding:12px;"
                f"color:var(--red);font-weight:900;'>{drv[:3].upper()}</div>",
                unsafe_allow_html=True,
            )
    with cd:
        st.markdown(f"### {num}  {fn} {ln}")
        st.caption(team)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Victorias",   st_["wins"])
        m2.metric("Podios",      st_["podiums"])
        m3.metric("Campeonatos", st_["championships"])
        m4.metric("Nacimiento",  st_["dob"])
        bnd = DRIVER_FLAGS.get(st_["nationality"], "")
        st.caption(f"Nacionalidad: {st_['nationality']} {bnd}")


# ─── ROUTER PRINCIPAL ─────────────────────────────────────────────────────────

_view = st.session_state["view"]

if _view == "grid":
    view_grid()
elif _view == "sessions":
    view_sessions()
elif _view == "analysis":
    view_analysis()
else:
    sl.go("grid")
