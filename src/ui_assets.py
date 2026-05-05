"""
TALOS — UI Assets
Bloques HTML/CSS/JS y helpers de visualización Plotly.
Importar desde appResearch.py para mantener el router limpio.
"""

import pandas as pd
import plotly.graph_objects as go


# ─────────────────────────────────────────────────────────────────────────────
# CSS · TEMA F1
# ─────────────────────────────────────────────────────────────────────────────

CSS_THEME = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:ital,wght@0,400;0,500;0,600;0,700;0,800;0,900;1,700;1,800;1,900&family=Titillium+Web:ital,wght@0,400;0,600;0,700;0,900;1,700;1,900&family=JetBrains+Mono:wght@400;500;600;700&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200');
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined');
@import url('https://fonts.googleapis.com/icon?family=Material+Icons');
@import url('https://cdn.jsdelivr.net/npm/flag-icons@7.2.3/css/flag-icons.min.css');

/* ── Variables ── */
:root {
    --bg:       #0D0D12;
    --surface:  rgba(255, 255, 255, 0.03);
    --surface2: rgba(255, 255, 255, 0.03);
    --red:      #E8002D;
    --red-hi:   #FF1F45;
    --cyan:     #00D7B5;
    --white:    #FFFFFF;
    --t1:       #E8E9EF;
    --t2:       #8A8D9E;
    --t3:       #555870;
    --border:   rgba(255,255,255,0.08);
    --border2:  rgba(255,255,255,0.11);
}

/* ── Reset base · Carbon-fiber backdrop ── */
.stApp, .main, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background-color: var(--bg) !important;
    background-image:
        repeating-linear-gradient( 45deg, transparent 0 2px, rgba(255,255,255,0.012) 2px 3px),
        repeating-linear-gradient(-45deg, transparent 0 2px, rgba(255,255,255,0.010) 2px 3px),
        radial-gradient(ellipse at 20% 0%, rgba(232,0,45,0.05) 0%, transparent 50%),
        radial-gradient(ellipse at 100% 100%, rgba(59,130,246,0.025) 0%, transparent 55%) !important;
    background-attachment: fixed !important;
}
html, body, [class*="css"], [class*="st-"] {
    font-family: 'Inter', 'Titillium Web', system-ui, sans-serif;
}
.mono, .num { font-family: 'JetBrains Mono', ui-monospace, monospace !important; font-feature-settings: 'tnum' 1, 'cv11' 1; letter-spacing: -0.02em; }
.block-container {
    max-width: 1520px !important;
    padding: 1.6rem 2.8rem 5rem 2.8rem !important;
    background: transparent !important;
}

/* ── Ocultar elementos de Streamlit no deseados ── */
#MainMenu, footer,
[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"],
[data-testid="stMainMenuButton"],
[data-testid="stToolbarActions"],
[data-testid="stKeyboardShortcutsModal"],
[data-baseweb="notification"],
[data-baseweb="tooltip"] { display: none !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0E0E16 !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] button[title*="keyboard" i],
[data-testid="stSidebar"] button[title*="shortcut" i],
[data-testid="stSidebar"] [aria-label*="keyboard" i],
[data-testid="stSidebar"] [data-testid="stToolbarActions"] { display: none !important; }

/* ── Títulos ── */
/* ── Streamlit sidebar collapse/expand control · F1 broadcast HUD ── */
[data-testid="stSidebarCollapsedControl"],
[data-testid="stSidebarCollapseButton"],
[data-testid="collapsedControl"] {
    background: rgba(232,0,45,0.16) !important;
    border: 1px solid rgba(232,0,45,0.50) !important;
    border-radius: 4px !important;
    padding: 6px 9px !important;
    box-shadow: 0 0 0 1px rgba(0,0,0,0.4),
                0 4px 14px rgba(232,0,45,0.18) !important;
    transition: background 0.18s ease, box-shadow 0.18s ease,
                transform 0.18s ease !important;
    z-index: 999990 !important;
}
[data-testid="stSidebarCollapsedControl"]:hover,
[data-testid="stSidebarCollapseButton"]:hover,
[data-testid="collapsedControl"]:hover {
    background: rgba(232,0,45,0.32) !important;
    box-shadow: 0 0 24px rgba(232,0,45,0.42),
                0 0 0 1px rgba(232,0,45,0.7) !important;
    transform: translateX(2px) !important;
}
[data-testid="stSidebarCollapsedControl"] svg,
[data-testid="stSidebarCollapseButton"] svg,
[data-testid="collapsedControl"] svg {
    color: #fff !important;
    fill: #fff !important;
    width: 22px !important;
    height: 22px !important;
}
[data-testid="stSidebarCollapsedControl"]::after {
    content: 'MENU';
    display: block;
    font-family: 'Titillium Web', sans-serif;
    font-style: italic;
    font-weight: 900;
    font-size: 0.55rem;
    letter-spacing: 0.18em;
    color: #fff;
    margin-top: 1px;
    text-align: center;
}

.gp-title {
    font-family: 'Titillium Web', 'Inter', sans-serif;
    font-size: 3.4rem;
    font-weight: 900;
    font-style: italic;
    color: var(--white);
    letter-spacing: -0.025em;
    line-height: 0.95;
    margin: 1.2rem 0 2.2rem 0;
    text-transform: uppercase;
    text-shadow: 0 2px 24px rgba(232,0,45,0.12);
}
.gp-title span { color: var(--red); }
.gp-title::before {
    content:''; display:inline-block;
    width: 4px; height: 0.85em;
    background: linear-gradient(180deg, var(--red) 0%, transparent 100%);
    margin-right: 18px;
    transform: skewX(-18deg);
    vertical-align: -0.06em;
}

/* ── Pill buttons (popovers) ── */
div[data-testid="stPopover"] > div > button {
    background: transparent !important;
    border: 1.5px solid var(--border2) !important;
    color: var(--t1) !important;
    border-radius: 999px !important;
    padding: 5px 16px 5px 14px !important;
    font-weight: 600 !important;
    font-size: 0.84rem !important;
    letter-spacing: 0.01em !important;
    box-shadow: none !important;
    height: 34px !important;
    min-height: 34px !important;
    transition: border-color 0.15s, background 0.15s;
}
div[data-testid="stPopover"] > div > button:hover {
    background: var(--surface) !important;
    border-color: var(--border2) !important;
    color: var(--white) !important;
}
div[data-testid="stPopover"] > div > button::after { display: none !important; }

/* ── Tarjeta flotante ── */
div[data-testid="stVerticalBlockBorderWrapper"]:has(.card-anchor) {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-top: 3px solid var(--red) !important;
    border-radius: 12px !important;
    padding: 20px 20px 18px 20px !important;
    box-shadow: 0 6px 28px rgba(0,0,0,0.45) !important;
    transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
    position: relative;
}
div[data-testid="stVerticalBlockBorderWrapper"]:has(.card-anchor):hover {
    transform: translateY(-4px);
    box-shadow: 0 14px 40px rgba(0,0,0,0.6),
                0 0 0 1px rgba(232,0,45,0.25) !important;
    border-color: rgba(232,0,45,0.4) !important;
}

/* ── Layout interior de tarjeta ── */
.card-header {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    margin-bottom: 16px;
    gap: 10px;
}
.card-flag-wrap {
    display: flex;
    align-items: center;
    gap: 10px;
    min-width: 0;
}
.card-flag {
    font-size: 16px;
    background: rgba(255,255,255,0.05);
    border-radius: 50%;
    width: 30px; height: 30px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
}
.card-name {
    color: var(--white);
    font-weight: 700;
    font-size: 1rem;
    line-height: 1.2;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
.round-badge {
    background: var(--red);
    color: #fff;
    font-size: 0.66rem;
    font-weight: 800;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    border-radius: 4px;
    padding: 3px 7px;
    white-space: nowrap;
    flex-shrink: 0;
}
.card-date {
    color: var(--t2);
    font-size: 0.8rem;
    font-weight: 400;
    flex-shrink: 0;
    white-space: nowrap;
}

/* ── Botón primario (Analizar / Cargar) ── */
.stButton > button {
    background: rgba(232, 0, 45, 0.1) !important;
    color: var(--red) !important;
    border: 1px solid var(--red) !important;
    border-radius: 8px !important;
    padding: 9px 14px !important;
    font-weight: 700 !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    width: 100% !important;
    box-shadow: 0 4px 16px rgba(232,0,45,0.1) !important;
    transition: background 0.12s, transform 0.1s, box-shadow 0.12s, color 0.12s !important;
}
.stButton > button:hover {
    background: rgba(232, 0, 45, 0.2) !important;
    transform: translateY(-1px);
    box-shadow: 0 0 12px rgba(232,0,45,0.4) !important;
    color: var(--white) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ── Botón Secundario (rojito) ── */
.stButton > button[kind="secondary"],
[data-testid="stBaseButton-secondary"] {
    background: rgba(232, 0, 45, 0.04) !important; /* Un hilito de fondo para que no se pierda */
    border: 1px solid rgba(232, 0, 45, 0.6) !important; /* Borde al 60% en vez del 30% */
    color: #E8002D !important; /* Rojo F1 puro, cero transparencias */
    text-transform: uppercase !important;
    font-weight: 800 !important; /* Letra un punto más gorda */
    box-shadow: none !important;
    letter-spacing: 0.06em !important;
}

.stButton > button[kind="secondary"]:hover,
[data-testid="stBaseButton-secondary"]:hover {
    background: rgba(232, 0, 45, 0.15) !important;
    border-color: #E8002D !important;
    color: #fff !important;
    box-shadow: 0 0 16px rgba(232,0,45,0.25) !important;
    transform: translateY(-1px);
}
/* ── Separador tarjeta ── */
.card-sep {
    height: 1px;
    background: var(--border);
    margin: 15px 0 14px;
}

/* ── Grid inferior de datos ── */
.card-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px 16px;
}
.kv-label {
    color: var(--t3);
    font-size: 0.67rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-weight: 600;
    margin-bottom: 2px;
}
.kv-value {
    color: var(--t1);
    font-size: 0.88rem;
    font-weight: 600;
}

/* ── Hero de sesiones ── */
.sessions-hero {
    display: flex;
    align-items: center;
    gap: 16px;
    margin: 1rem 0 2rem 0;
}
.sessions-hero .flag-xl {
    font-size: 22px;
    background: var(--surface);
    border-radius: 50%;
    width: 52px; height: 52px;
    display: inline-flex; align-items: center; justify-content: center;
    border: 2px solid var(--red);
    flex-shrink: 0;
}
.sessions-hero h2 {
    color: var(--white);
    font-weight: 800;
    font-size: 1.9rem;
    margin: 0;
    text-transform: uppercase;
    letter-spacing: -0.01em;
}
.sessions-hero .hero-meta {
    color: var(--t2); font-size: 0.88rem; margin-top: 3px;
}

.session-code {
    color: var(--white);
    font-size: 2.2rem;
    font-weight: 900;
    letter-spacing: -0.03em;
    line-height: 1;
    text-align: center;
    padding: 12px 0 8px;
    text-transform: uppercase;
}
.session-fullname {
    color: var(--t2);
    font-size: 0.78rem;
    text-align: center;
    margin-bottom: 4px;
    font-weight: 400;
}

/* ── Tabs del hub de análisis ── */
[data-testid="stTabs"] [data-testid="stTab"] {
    background: transparent !important;
    color: var(--t2) !important;
    font-weight: 600 !important;
    font-size: 0.84rem !important;
    border-radius: 0 !important;
    border-bottom: 2px solid transparent !important;
    padding: 8px 18px !important;
    letter-spacing: 0.03em;
    text-transform: uppercase;
    transition: color 0.15s;
}
[data-testid="stTabs"] [data-testid="stTab"]:hover {
    color: var(--white) !important;
}
[data-testid="stTabs"] [data-testid="stTab"][aria-selected="true"] {
    color: var(--white) !important;
    border-bottom-color: var(--red) !important;
    background: transparent !important;
}
[data-testid="stTabsContent"] { padding-top: 1.4rem !important; }
[data-testid="stTabs"] > div:first-child {
    border-bottom: 1px solid var(--border) !important;
}

div[data-testid="stVerticalBlockBorderWrapper"]:has(.module-card-anchor) {
    background: var(--surface2) !important;
    border: 1px solid var(--border2) !important;
    border-radius: 10px !important;
    padding: 18px 16px 16px !important;
    box-shadow: none !important;
    transition: border-color 0.15s, background 0.15s, box-shadow 0.15s;
    cursor: pointer;
}
div[data-testid="stVerticalBlockBorderWrapper"]:has(.module-card-anchor):hover {
    background: #2e2e45 !important;
    border-color: rgba(232,0,45,0.35) !important;
    box-shadow: 0 4px 18px rgba(0,0,0,0.4) !important;
}
.module-icon {
    font-size: 1.4rem;
    color: var(--red);
    margin-bottom: 8px;
    line-height: 1;
}
.module-name {
    color: var(--white);
    font-weight: 700;
    font-size: 0.9rem;
    margin-bottom: 5px;
    line-height: 1.2;
}
.module-desc {
    color: var(--t2);
    font-size: 0.74rem;
    font-weight: 400;
    line-height: 1.4;
}

/* ── Sidebar de análisis ── */
[data-testid="stSidebar"] .stMarkdown h1 {
    color: var(--white) !important;
    font-weight: 900;
    letter-spacing: -0.01em;
}
[data-testid="stSidebar"] [data-testid="stCaption"] {
    color: var(--t3) !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    font-weight: 600 !important;
}

/* Métricas F1 */
[data-testid="stMetric"] {
    border-top: 2px solid var(--red);
    background: var(--surface);
    padding: 12px 16px;
    border-radius: 6px;
}
[data-testid="stMetricLabel"] { color: var(--t2) !important; font-size: 0.75rem !important; }
[data-testid="stMetricValue"] { color: var(--white) !important; }

/* Inputs */
.stSelectbox > div > div,
.stTextInput > div > div > input,
.stNumberInput input {
    background: var(--surface) !important;
    color: var(--white) !important;
    border-color: var(--border2) !important;
    border-radius: 8px !important;
}

[data-testid="stIconMaterial"],
span.material-icons,
span.material-symbols-rounded,
span.material-symbols-outlined {
    font-family: 'Material Symbols Rounded', 'Material Symbols Outlined', 'Material Icons' !important;
    font-weight: normal !important;
    font-style: normal !important;
    font-variant: normal !important;
    text-transform: none !important;
    line-height: 1 !important;
    white-space: nowrap !important;
    word-wrap: normal !important;
    direction: ltr !important;
    -webkit-font-feature-settings: 'liga' !important;
    -webkit-font-smoothing: antialiased !important;
    font-feature-settings: 'liga' !important;
}

img.emoji {
    height: 1.15em;
    width:  1.15em;
    vertical-align: -0.15em;
    display: inline-block;
}
.card-flag img.emoji  { height: 18px; width: 18px; }
.flag-xl   img.emoji  { height: 26px; width: 26px; }
</style>
"""


# ─────────────────────────────────────────────────────────────────────────────
# JS · TWEAKS (Twemoji + keyboard titles + card backgrounds)
# ─────────────────────────────────────────────────────────────────────────────

JS_TWEAKS = """
<script>
(function() {
    function stripKbTitles() {
        document.querySelectorAll('[title]').forEach(function(el) {
            var t = el.getAttribute('title').toLowerCase();
            if (t.includes('keyboard') || t.includes('shortcut')) {
                el.removeAttribute('title');
            }
        });
    }

    var twemojReady = false;
    var debounce;
    function applyTwemoji() {
        if (!twemojReady) return;
        clearTimeout(debounce);
        debounce = setTimeout(function() {
            twemoji.parse(document.body, {
                folder: 'svg', ext: '.svg',
                base: 'https://cdn.jsdelivr.net/npm/twemoji@14.0.2/dist/'
            });
        }, 120);
    }

    function loadTwemoji() {
        var s = document.createElement('script');
        s.src = 'https://cdn.jsdelivr.net/npm/twemoji@14.0.2/dist/twemoji.min.js';
        s.crossOrigin = 'anonymous';
        s.onload = function() { twemojReady = true; applyTwemoji(); };
        document.head.appendChild(s);
    }

    var cardDebounce;
    function applyCardBg() {
        clearTimeout(cardDebounce);
        cardDebounce = setTimeout(function() {
            document.querySelectorAll('.card-anchor').forEach(function(el) {
                var w = el.closest('[data-testid="stVerticalBlockBorderWrapper"]');
                if (!w) return;
                w.style.setProperty('background',       'rgba(255,255,255,0.03)', 'important');
                w.style.setProperty('background-color', 'rgba(255,255,255,0.03)', 'important');
                w.style.setProperty('border-top',       '3px solid #E8002D', 'important');
                w.style.setProperty('border-radius',    '12px', 'important');
                w.style.setProperty('border',           '1px solid rgba(255,255,255,0.08)', 'important');
            });
            document.querySelectorAll('.module-card-anchor').forEach(function(el) {
                var w = el.closest('[data-testid="stVerticalBlockBorderWrapper"]');
                if (!w) return;
                w.style.setProperty('background',       'rgba(255,255,255,0.03)', 'important');
                w.style.setProperty('background-color', 'rgba(255,255,255,0.03)', 'important');
                w.style.setProperty('border',           '1px solid rgba(255,255,255,0.08)', 'important');
                w.style.setProperty('border-radius',    '10px', 'important');
            });
        }, 80);
    }

    function run() {
        stripKbTitles();
        loadTwemoji();
        applyCardBg();
        new MutationObserver(function() {
            stripKbTitles();
            applyTwemoji();
            applyCardBg();
        }).observe(document.documentElement, {
            subtree: true, childList: true,
            attributes: true, attributeFilter: ['title']
        });
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', run);
    } else {
        run();
    }
})();
</script>
"""


# ─────────────────────────────────────────────────────────────────────────────
# CSS · OCULTAR SIDEBAR en vistas grid / sessions
# ─────────────────────────────────────────────────────────────────────────────

HIDE_SIDEBAR_CSS = """
<style>
section[data-testid="stSidebar"],
[data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"],
button[aria-label="open sidebar"],
button[aria-label="close sidebar"],
button[aria-label="Open sidebar"],
button[aria-label="Close sidebar"] {
    display: none !important;
    visibility: hidden !important;
    width: 0 !important;
    height: 0 !important;
    overflow: hidden !important;
    pointer-events: none !important;
}
</style>
"""


# ─────────────────────────────────────────────────────────────────────────────
# COORDENADAS GPS DE CIRCUITOS F1
# Clave: fragmento del nombre del evento o del país (case-insensitive match)
# Valor: (latitud, longitud, nombre_ciudad)
# Orden de búsqueda: primero match por EventName, luego por Country/Location
# ─────────────────────────────────────────────────────────────────────────────

CIRCUIT_COORDS: dict[str, tuple[float, float, str]] = {
    # ── Oriente Medio ──────────────────────────────────────────────────────
    "Bahrain":          (26.0325,   50.5106,  "Sakhir"),
    "Saudi":            (21.6319,   39.1044,  "Jeddah"),
    "Qatar":            (25.4900,   51.4542,  "Lusail"),
    "Abu Dhabi":        (24.4672,   54.6031,  "Yas Marina"),
    "Azerbaijan":       (40.3725,   49.8533,  "Baku"),
    # ── Asia-Pacífico ──────────────────────────────────────────────────────
    "Australia":        (-37.8497,  144.9680, "Melbourne"),
    "Japan":            (34.8431,   136.5411, "Suzuka"),
    "China":            (31.3389,   121.2200, "Shanghai"),
    "Singapore":        (1.2914,    103.8640, "Marina Bay"),
    # ── Europa ─────────────────────────────────────────────────────────────
    "Monaco":           (43.7347,    7.4206,  "Monaco"),
    "Spain":            (41.5700,    2.2611,  "Barcelona"),
    "Austria":          (47.2197,   14.7647,  "Spielberg"),
    "Great Britain":    (52.0786,   -1.0169,  "Silverstone"),
    "Hungary":          (47.5789,   19.2486,  "Budapest"),
    "Belgium":          (50.4372,    5.9714,  "Spa-Francorchamps"),
    "Netherlands":      (52.3888,    4.5409,  "Zandvoort"),
    "Italy":            (45.6156,    9.2811,  "Monza"),
    "Emilia Romagna":   (44.3439,   11.7167,  "Imola"),
    "France":           (43.2506,    5.7914,  "Le Castellet"),
    "Portugal":         (38.4000,   -8.9900,  "Portimão"),
    "Turkey":           (40.9517,   29.4050,  "Istanbul"),
    "Russia":           (43.4057,   39.9519,  "Sochi"),
    "Germany":          (49.3288,    8.5661,  "Hockenheim"),
    # ── América ────────────────────────────────────────────────────────────
    "Canada":           (45.5000,  -73.5228,  "Montréal"),
    "Miami":            (25.9581,  -80.2389,  "Miami Gardens"),
    "United States":    (30.1328,  -97.6411,  "Austin"),
    "Las Vegas":        (36.1147, -115.1728,  "Las Vegas Strip"),
    "Mexico":           (19.4042,  -99.0907,  "Mexico City"),
    "Brazil":           (-23.7036, -46.6997,  "Interlagos"),
}


def _find_coords(event_name: str, country: str, location: str):
    """
    Devuelve (lat, lon, city) haciendo match parcial case-insensitive contra
    los fragmentos de CIRCUIT_COORDS. Primero intenta el nombre del evento,
    luego el país, luego la ubicación.
    """
    haystack = f"{event_name} | {country} | {location}".lower()
    for key, (lat, lon, city) in CIRCUIT_COORDS.items():
        if key.lower() in haystack:
            return lat, lon, city
    return None


# ─────────────────────────────────────────────────────────────────────────────
# BUILDER DEL MAPA MUNDIAL CON LOLLIPOPS
# ─────────────────────────────────────────────────────────────────────────────

# Paleta para colorear lollipops por número de ronda (opcional, añade lectura
# visual al calendario): todos rojos pero intensidad varía con el mes.
_RED_BASE  = "#E8002D"
_RED_LIGHT = "#FF4D6A"
_STEM_DEG  = 4.5          # longitud del palo en grados de latitud


# ─────────────────────────────────────────────────────────────────────────────
# CSS · SELECCIÓN DE SESIÓN
# ─────────────────────────────────────────────────────────────────────────────

CSS_SESSIONS = """
<style>
@keyframes scard-in {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ── Tarjeta base · estilo HUD/broadcast ── */
.scard {
    border-radius: 4px 4px 0 0;
    border: 1px solid rgba(255,255,255,0.08);
    border-bottom: none;
    overflow: hidden;
    animation: scard-in 0.28s ease forwards;
    transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
    position: relative;
    background-image:
        repeating-linear-gradient(45deg,  transparent 0 2px, rgba(255,255,255,0.014) 2px 3px),
        repeating-linear-gradient(-45deg, transparent 0 2px, rgba(255,255,255,0.012) 2px 3px);
}
.scard::before {
    content: '';
    position: absolute;
    top: 0; right: 0;
    width: 28px; height: 28px;
    background: linear-gradient(225deg, currentColor 0%, currentColor 50%, transparent 50%);
    opacity: 0.18;
    pointer-events: none;
}
.scard::after {
    content: '';
    position: absolute;
    inset: 0;
    background: repeating-linear-gradient(0deg, transparent 0 3px, rgba(255,255,255,0.008) 3px 4px);
    pointer-events: none;
    border-radius: inherit;
}

/* ── CTA bar (Streamlit button) integrado al pie de cada card ── */
[data-testid="stColumn"]:has(.scard) [data-testid="stButton"] {
    margin-top: -1px !important;
}
[data-testid="stColumn"]:has(.scard) [data-testid="stButton"] > button {
    width: 100% !important;
    border-radius: 0 0 4px 4px !important;
    font-family: 'Titillium Web','Inter',sans-serif !important;
    font-style: italic !important;
    font-weight: 800 !important;
    letter-spacing: 0.16em !important;
    text-transform: uppercase !important;
    font-size: 0.72rem !important;
    padding: 14px 12px !important;
    transition: background 0.18s ease, border-color 0.18s ease, box-shadow 0.18s ease, transform 0.18s ease !important;
}
[data-testid="stColumn"]:has(.scard) [data-testid="stButton"] > button:hover {
    transform: translateY(-2px) !important;
}

/* ── Fondo base para todas las tarjetas ── */
.scard { background-color: var(--surface); }

/* ── Free Practice (Gris Técnico) ── */
.scard-fp { border-top: 3px solid #8A8D9E; }
[data-testid="stColumn"]:has(.scard-fp) [data-testid="stButton"] > button {
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    color: #8A8D9E !important;
}
[data-testid="stColumn"]:has(.scard-fp) [data-testid="stButton"] > button:hover {
    background: rgba(255,255,255,0.08) !important;
    border-color: #8A8D9E !important;
    color: #fff !important;
    box-shadow: 0 0 15px rgba(255,255,255,0.1) !important;
    transform: translateY(-2px) !important;
}

/* ── Qualifying (Rojo Oscuro) ── */
.scard-q { border-top: 3px solid #C40026; }
[data-testid="stColumn"]:has(.scard-q) [data-testid="stButton"] > button {
    background: rgba(196,0,38,0.05) !important;
    border: 1px solid rgba(196,0,38,0.3) !important;
    color: #C40026 !important;
}
[data-testid="stColumn"]:has(.scard-q) [data-testid="stButton"] > button:hover {
    background: rgba(196,0,38,0.2) !important;
    border-color: #C40026 !important;
    color: #fff !important;
    box-shadow: 0 0 15px rgba(196,0,38,0.3) !important;
    transform: translateY(-2px) !important;
}

/* ── Sprints (Morado Purple Sector) ── */
.scard-spr { border-top: 3px solid #B138DD; }
[data-testid="stColumn"]:has(.scard-spr) [data-testid="stButton"] > button {
    background: rgba(177,56,221,0.05) !important;
    border: 1px solid rgba(177,56,221,0.3) !important;
    color: #B138DD !important;
}
[data-testid="stColumn"]:has(.scard-spr) [data-testid="stButton"] > button:hover {
    background: rgba(177,56,221,0.2) !important;
    border-color: #B138DD !important;
    color: #fff !important;
    box-shadow: 0 0 15px rgba(177,56,221,0.3) !important;
    transform: translateY(-2px) !important;
}

/* ── Race (Rojo F1 Brillante - Destaca más) ── */
.scard-r {
    border-top: 3px solid #E8002D;
    background-color: rgba(232,0,45,0.03); /* Ligerísimo tinte rojo de fondo */
    box-shadow: 0 0 25px rgba(232,0,45,0.08);
}
[data-testid="stColumn"]:has(.scard-r) [data-testid="stButton"] > button {
    background: rgba(232,0,45,0.12) !important;
    border: 1px solid rgba(232,0,45,0.5) !important;
    color: #E8002D !important;
    font-weight: 900 !important;
}
[data-testid="stColumn"]:has(.scard-r) [data-testid="stButton"] > button:hover {
    background: rgba(232,0,45,0.25) !important;
    border-color: #E8002D !important;
    color: #fff !important;
    box-shadow: 0 0 20px rgba(232,0,45,0.35) !important;
    transform: translateY(-2px) !important;
}

/* ── Circuit hero ── */
.circuit-hero {
    background: linear-gradient(135deg, #1a1a28 0%, #15151E 100%);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 4px;
    padding: 34px 40px;
    margin-bottom: 2.2rem;
    position: relative;
    overflow: hidden;
    /* Diagonal cut esquina inferior-izq como visor HUD */
    clip-path: polygon(0 0, 100% 0, 100% 100%, 36px 100%, 0 calc(100% - 36px));
}
.circuit-hero-name { font-style: italic; }
.circuit-hero::before {
    content: '';
    position: absolute;
    inset: 0;
    background: repeating-linear-gradient(
        -55deg,
        transparent,
        transparent 38px,
        rgba(255,255,255,0.013) 38px,
        rgba(255,255,255,0.013) 39px
    );
    pointer-events: none;
}
.circuit-hero::after {
    content: '';
    position: absolute;
    right: -60px; top: -60px;
    width: 260px; height: 260px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(232,0,45,0.06) 0%, transparent 70%);
    pointer-events: none;
}
.circuit-hero-pill {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    background: rgba(232,0,45,0.1);
    border: 1px solid rgba(232,0,45,0.22);
    border-radius: 999px;
    padding: 5px 14px;
    color: #E8002D;
    font-size: 0.68rem;
    font-weight: 800;
    letter-spacing: 0.13em;
    text-transform: uppercase;
    margin-bottom: 18px;
}
.circuit-hero-name {
    font-size: 2.8rem;
    font-weight: 900;
    color: #fff;
    letter-spacing: -0.03em;
    line-height: 0.95;
    text-transform: uppercase;
    margin-bottom: 12px;
}
.circuit-hero-name span { color: #E8002D; }
.circuit-hero-meta {
    color: #9497A8;
    font-size: 0.83rem;
    display: flex;
    align-items: center;
    gap: 10px;
    flex-wrap: wrap;
}
.circuit-hero-meta .sep {
    color: #2D2D44;
    font-size: 1rem;
}
</style>
"""


# ─────────────────────────────────────────────────────────────────────────────
# CSS · HUB DE MÓDULOS
# ─────────────────────────────────────────────────────────────────────────────

CSS_HUB = """
<style>
@keyframes mcard-in {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ── CTA bar integrado en mcard ── */
[data-testid="stColumn"]:has(.mcard) [data-testid="stButton"] {
    margin-top: -1px !important;
}
[data-testid="stColumn"]:has(.mcard) [data-testid="stButton"] > button {
    width: 100% !important;
    border-radius: 0 0 4px 4px !important;
    font-family: 'Titillium Web','Inter',sans-serif !important;
    font-style: italic !important;
    font-weight: 800 !important;
    letter-spacing: 0.16em !important;
    text-transform: uppercase !important;
    font-size: 0.66rem !important;
    padding: 9px 8px !important;
    background: rgba(255,255,255,0.025) !important;
    transition: all 0.18s ease !important;
}
[data-testid="stColumn"]:has(.mcard.mc-blue) [data-testid="stButton"] > button {
    border: 1px solid rgba(59,130,246,0.30) !important;
    color: #3B82F6 !important;
    background: rgba(59,130,246,0.08) !important;
}
[data-testid="stColumn"]:has(.mcard.mc-blue) [data-testid="stButton"] > button:hover {
    background: rgba(59,130,246,0.20) !important;
    border-color: rgba(59,130,246,0.65) !important;
    box-shadow: 0 0 18px rgba(59,130,246,0.22) !important;
}
[data-testid="stColumn"]:has(.mcard.mc-green) [data-testid="stButton"] > button {
    border: 1px solid rgba(16,185,129,0.30) !important;
    color: #10B981 !important;
    background: rgba(16,185,129,0.08) !important;
}
[data-testid="stColumn"]:has(.mcard.mc-green) [data-testid="stButton"] > button:hover {
    background: rgba(16,185,129,0.20) !important;
    border-color: rgba(16,185,129,0.65) !important;
    box-shadow: 0 0 18px rgba(16,185,129,0.22) !important;
}
[data-testid="stColumn"]:has(.mcard.mc-amber) [data-testid="stButton"] > button {
    border: 1px solid rgba(245,158,11,0.30) !important;
    color: #F59E0B !important;
    background: rgba(245,158,11,0.08) !important;
}
[data-testid="stColumn"]:has(.mcard.mc-amber) [data-testid="stButton"] > button:hover {
    background: rgba(245,158,11,0.20) !important;
    border-color: rgba(245,158,11,0.65) !important;
    box-shadow: 0 0 18px rgba(245,158,11,0.22) !important;
}
[data-testid="stColumn"]:has(.mcard.mc-purple) [data-testid="stButton"] > button {
    border: 1px solid rgba(139,92,246,0.30) !important;
    color: #8B5CF6 !important;
    background: rgba(139,92,246,0.08) !important;
}
[data-testid="stColumn"]:has(.mcard.mc-purple) [data-testid="stButton"] > button:hover {
    background: rgba(139,92,246,0.20) !important;
    border-color: rgba(139,92,246,0.65) !important;
    box-shadow: 0 0 18px rgba(139,92,246,0.22) !important;
}
[data-testid="stColumn"]:has(.mcard) [data-testid="stButton"] > button:hover {
    transform: translateY(-2px) !important;
}

/* ── Category header ── */
.mcat-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 0.85rem;
    padding-bottom: 8px;
}
.mcat-pip {
    width: 4px;
    height: 18px;
    border-radius: 3px;
    flex-shrink: 0;
}
.mcat-label {
    font-size: 0.66rem;
    font-weight: 800;
    letter-spacing: 0.14em;
    text-transform: uppercase;
}
.mcat-rule {
    flex: 1;
    height: 1px;
    background: rgba(255,255,255,0.05);
}


/* ── Module card ── */
.mcard {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-left-width: 3px;
    border-bottom: none;
    border-radius: 4px 4px 0 0;
    padding: 18px 15px 14px;
    transition: transform 0.18s ease, box-shadow 0.18s ease,
                border-color 0.18s ease, background 0.18s ease;
    animation: mcard-in 0.25s ease forwards;
    height: 100%;
    box-sizing: border-box;
    position: relative;
    background-image:
        repeating-linear-gradient(45deg,  transparent 0 2px, rgba(255,255,255,0.012) 2px 3px),
        repeating-linear-gradient(-45deg, transparent 0 2px, rgba(255,255,255,0.010) 2px 3px);
}
.mcard::before {
    content:''; position:absolute; top:0; right:0;
    width:18px; height:18px;
    background: linear-gradient(225deg, currentColor 0%, currentColor 50%, transparent 50%);
    opacity: 0.22;
    pointer-events:none;
}
.mcard:hover {
    transform: translateY(-5px);
    background-color: rgba(255, 255, 255, 0.06);
    border-color: rgba(255, 255, 255, 0.15);
}

.mcard-icon {
    font-size: 1.55rem;
    line-height: 1;
    display: block;
    margin-bottom: 10px;
}
.mcard-name {
    color: #fff;
    font-weight: 800;
    font-style: italic;
    font-size: 0.88rem;
    line-height: 1.22;
    margin-bottom: 6px;
    letter-spacing: -0.01em;
}
.mcard-desc {
    color: #9497A8;
    font-size: 0.7rem;
    line-height: 1.5;
}

/* ── Variantes de Rojo (Escala Monocromática) ── */
.mc-red-1 { border-left-color: #E8002D; }
.mc-red-1 .mcard-icon { color: #E8002D; }

.mc-red-2 { border-left-color: #C40026; }
.mc-red-2 .mcard-icon { color: #C40026; }

.mc-red-3 { border-left-color: #99001D; }
.mc-red-3 .mcard-icon { color: #99001D; }

.mc-red-4 { border-left-color: #6E0015; }
.mc-red-4 .mcard-icon { color: #6E0015; }

/* ── Botones de las tarjetas ── */
/* Estado normal: Gris ahumado sutil para que no molesten visualmente */
[data-testid="stColumn"]:has(.mcard[class*="mc-red-"]) [data-testid="stButton"] > button {
    background: rgba(255, 255, 255, 0.03) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    color: var(--t2) !important;
}

/* Hover: Al pasar el ratón, todos se iluminan con el rojo F1 principal */
[data-testid="stColumn"]:has(.mcard[class*="mc-red-"]) [data-testid="stButton"] > button:hover {
    background: rgba(232, 0, 45, 0.15) !important;
    border-color: #E8002D !important;
    box-shadow: 0 0 16px rgba(232, 0, 45, 0.3) !important;
    color: #fff !important;
    transform: translateY(-2px) !important;
}

/* ── Hub active-session banner ── */
.hub-banner {
    display: flex;
    align-items: center;
    gap: 14px;
    background: linear-gradient(90deg, #1a1a2e 0%, #1E1E2D 100%);
    border: 1px solid rgba(255,255,255,0.06);
    border-left: 3px solid #E8002D;
    border-radius: 0 10px 10px 0;
    padding: 12px 18px;
    margin-bottom: 2rem;
}
.hub-banner-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #E8002D;
    box-shadow: 0 0 8px rgba(232,0,45,0.6);
    flex-shrink: 0;
    animation: pulse-dot 2s ease-in-out infinite;
}
@keyframes pulse-dot {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.6; transform: scale(0.8); }
}
.hub-banner-label {
    color: #9497A8;
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-weight: 700;
    margin-bottom: 2px;
}
.hub-banner-name {
    color: #fff;
    font-weight: 700;
    font-size: 0.92rem;
    letter-spacing: -0.01em;
}
</style>
"""


def build_world_map(events_df: pd.DataFrame) -> go.Figure:
    """
    Construye un go.Figure con un scatter_geo de lollipops F1.

    Cada lollipop:
      · Palo  — línea vertical de STEM_DEG grados hacia el norte
      · Cabeza — círculo rojo con el número de ronda en blanco
      · Hover  — nombre del GP, país, ciudad, ronda y fecha

    customdata de cada cabeza: [event_name, country, location, round_n, date_str]
    Uso: pasar figure a st.plotly_chart(..., on_select='rerun')
         y leer event.selection.points[0]['customdata'][0] → event_name
    """

    # ── Recopilar datos con coordenadas ────────────────────────────────────
    lats, lons, names, countries, locations_list = [], [], [], [], []
    rounds, dates, cities = [], [], []

    for _, ev in events_df.iterrows():
        ev_name  = str(ev.get("EventName",  ""))
        country  = str(ev.get("Country",    ""))
        location = str(ev.get("Location",   ""))
        round_n  = ev.get("RoundNumber", "?")
        ev_date  = ev.get("EventDate",   None)

        try:
            date_str = pd.Timestamp(ev_date).strftime("%d %b %Y").upper()
        except Exception:
            date_str = "—"

        result = _find_coords(ev_name, country, location)
        if result is None:
            continue  # circuito sin coordenadas conocidas → omitir

        lat, lon, city = result
        lats.append(lat)
        lons.append(lon)
        names.append(ev_name)
        countries.append(country)
        locations_list.append(location)
        rounds.append(round_n)
        dates.append(date_str)
        cities.append(city)

    # ── Traza 1: palos (stems) ─────────────────────────────────────────────
    # Segmentos separados por None: [base, cabeza, None, base, cabeza, None, …]
    stem_lats, stem_lons = [], []
    for lat, lon in zip(lats, lons):
        stem_lats.extend([lat, lat + _STEM_DEG, None])
        stem_lons.extend([lon, lon, None])

    trace_stems = go.Scattergeo(
        lat=stem_lats,
        lon=stem_lons,
        mode="lines",
        line=dict(color=_RED_BASE, width=2.5),
        opacity=0.7,
        hoverinfo="skip",
        showlegend=False,
        name="stems",
    )

    # ── Traza 2: puntos de anclaje (base del palo = ubicación exacta) ──────
    # Círculo pequeño en el suelo para que el palo "toque" el circuito
    trace_base = go.Scattergeo(
        lat=lats,
        lon=lons,
        mode="markers",
        marker=dict(
            size=6,
            color=_RED_BASE,
            symbol="circle",
            line=dict(color="rgba(255,255,255,0.3)", width=1),
        ),
        hoverinfo="skip",
        showlegend=False,
        name="base_dots",
    )

    # ── Traza 3: cabezas de lollipop (círculos grandes + número) ──────────
    # La cabeza está en lat + STEM_DEG  (extremo norte del palo)
    head_lats = [lat + _STEM_DEG for lat in lats]
    head_lons = list(lons)

    custom = list(zip(names, countries, locations_list, rounds, dates, cities))

    hover_tpl = (
        "<span style='font-size:13px;font-weight:700;color:#fff;'>%{customdata[0]}</span>"
        "<br>"
        "<span style='color:#9497A8;font-size:11px;'>%{customdata[5]} · %{customdata[1]}</span>"
        "<br>"
        "<span style='color:#E8002D;font-size:11px;font-weight:700;'>"
        "Ronda %{customdata[3]}  ·  %{customdata[4]}"
        "</span>"
        "<extra></extra>"
    )

    trace_heads = go.Scattergeo(
        lat=head_lats,
        lon=head_lons,
        mode="markers+text",
        marker=dict(
            size=26,
            color=_RED_BASE,
            symbol="circle",
            line=dict(color="white", width=2),
            opacity=1.0,
        ),
        text=[str(r) for r in rounds],
        textfont=dict(
            color="white",
            size=10,
            family="Inter, Titillium Web, sans-serif",
        ),
        textposition="middle center",
        customdata=custom,
        hovertemplate=hover_tpl,
        hoverlabel=dict(
            bgcolor="#1E1E2D",
            bordercolor="#E8002D",
            font=dict(color="white", size=12,
                      family="Inter, Titillium Web, sans-serif"),
            align="left",
        ),
        showlegend=False,
        name="circuits",
    )

    # ── Figura ─────────────────────────────────────────────────────────────
    fig = go.Figure(data=[trace_stems, trace_base, trace_heads])

    fig.update_geos(
        showframe=False,
        showcoastlines=True,
        coastlinecolor="#2D2D44",
        coastlinewidth=0.8,
        showland=True,
        landcolor="#1E1E2D",
        showocean=True,
        oceancolor="#13131B",
        showlakes=True,
        lakecolor="#13131B",
        showcountries=True,
        countrycolor="#2A2A3E",
        countrywidth=0.5,
        showsubunits=False,
        bgcolor="#15151E",
        projection_type="natural earth",
        # Zoom inicial: mundo completo
        lataxis_range=[-60, 75],
        lonaxis_range=[-160, 170],
    )

    fig.update_layout(
        paper_bgcolor="#15151E",
        plot_bgcolor="#15151E",
        margin=dict(l=0, r=0, t=0, b=0),
        height=520,
        dragmode="pan",
        geo=dict(bgcolor="#15151E"),
        font=dict(family="Inter, sans-serif", color="white"),
        # Barra de herramientas mínima: solo zoom + reset
        modebar=dict(
            bgcolor="rgba(21,21,30,0.8)",
            color="#555870",
            activecolor="#E8002D",
            orientation="v",
        ),
        newselection=dict(mode="immediate"),
    )

    return fig
