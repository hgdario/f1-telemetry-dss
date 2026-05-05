"""
session_loader.py
─────────────────
Núcleo de estado, caché y carga de sesiones FastF1 para TALOS.

Responsabilidades:
  · Inicialización del session_state (vista, año, evento, sesión, módulo…).
  · Caché de calendario (`load_schedule`) y de sesión (`load_f1_session`).
  · Helper `require_session()` para los módulos.
  · Helper `go(view)` para navegar entre vistas.

Uso en el enrutador (appResearch.py):
    from session_loader import (
        init_session_state, load_schedule, load_f1_session,
        require_session, go,
    )
    init_session_state()
"""

import streamlit as st
import pandas as pd
import fastf1


# ─── ESTADO ───────────────────────────────────────────────────────────────────

def init_session_state() -> None:
    """Inicializa todas las claves de session_state que usa la app."""
    st.session_state.setdefault("view",            "grid")   # grid | sessions | analysis
    st.session_state.setdefault("selected_year",   2024)
    st.session_state.setdefault("selected_event",  None)
    st.session_state.setdefault("f1_session",      None)
    st.session_state.setdefault("session_loaded",  False)
    st.session_state.setdefault("session_label",   "")
    st.session_state.setdefault("active_module",   None)     # None = Hub
    st.session_state.setdefault("corners",         False)


# ─── CACHÉ ────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False, ttl=3600)
def load_schedule(year: int) -> pd.DataFrame:
    """Calendario FastF1 del año (sin testing)."""
    return fastf1.get_event_schedule(year, include_testing=False)


@st.cache_data(show_spinner=False)
def load_f1_session(year: int, gp: str, ses: str):
    """Carga (y cachea) una sesión FastF1. Cachea por (year, gp, ses)."""
    s = fastf1.get_session(year, gp, ses)
    s.load()
    return s


# ─── HELPERS ──────────────────────────────────────────────────────────────────

def require_session() -> bool:
    """Devuelve False y muestra aviso si no hay sesión cargada."""
    if not st.session_state.get("session_loaded", False):
        st.warning("Carga una sesión desde el panel lateral para continuar.")
        return False
    return True


def go(view: str) -> None:
    """Cambia la vista activa y fuerza rerun."""
    st.session_state["view"] = view
    st.rerun()
