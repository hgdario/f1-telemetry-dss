"""
F1 24 UDP Telemetry Analyzer — ATLAS/SAP-style engineering UI
Fixes:
  - Queues live in @st.cache_resource (no longer reset on rerun → data flows).
  - Fragment defined at module level (Streamlit can track it correctly).
  - Track map: only shown when FastF1 overlay loaded; uses overlay X/Y as
    circuit outline and interpolates current position from lap_distance.
"""
from __future__ import annotations

import io
import queue
import socket
import struct
import threading
import time
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# ============================================================================
# CONFIG
# ============================================================================
UDP_HOST       = "0.0.0.0"
UDP_PORT       = 20777
MAX_ROWS       = 18000
DATA_Q_SIZE    = 20000
DIAG_Q_SIZE    = 500
REFRESH_MS     = 150  # Live update period

# ATLAS / Mercedes SAP palette
BG_PRIMARY     = "#0A0A0B"
BG_SECONDARY   = "#111115"
BG_TERTIARY    = "#1A1A2E"
GRID_LINE      = "rgba(255,255,255,0.05)"
ACCENT_RED     = "#E8002D"
ACCENT_GREEN   = "#39FF14"
ACCENT_CYAN    = "#00D2FF"
ACCENT_YELLOW  = "#FFD700"
ACCENT_PURPLE  = "#C77DFF"
TEXT_PRIMARY   = "#FFFFFF"
TEXT_MUTED     = "#888888"
FONT_MONO      = "'JetBrains Mono', 'Courier New', monospace"

TYRES = {
    16: ("SOFT",   ACCENT_RED),
    17: ("MEDIUM", "#FFF200"),
    18: ("HARD",   "#EBEBEB"),
    7:  ("INTER",  "#43B02A"),
    8:  ("WET",    "#0067FF"),
}

# F1 24 binary formats
HDR_FMT    = "<HBBBBBQfIIBB"
HDR_SZ     = struct.calcsize(HDR_FMT)
MOT_FMT    = "<ffffffhhhhhffff"
MOT_SZ     = struct.calcsize(MOT_FMT)
TEL_FMT    = "<HfffBbHBBHHHHHBBBBBBBBHffffBBBB"
TEL_SZ     = struct.calcsize(TEL_FMT)
STA_FMT    = "<BBBBBfffHHBBHBBBbBBffffB"
STA_SZ     = struct.calcsize(STA_FMT)
LAP_PKT_SZ = 57

# Session-state keys
SK_LISTENER = "_f124_listener"
SK_ROWS     = "_f124_rows"
SK_LAST     = "_f124_last"
SK_DIAG     = "_f124_diag"
SK_RAW      = "_f124_raw"
SK_PKT      = "_f124_pkt"
SK_ERR      = "_f124_err"
SK_FRAMES   = "_f124_frames"
SK_OVERLAY  = "_f124_overlay"


# ============================================================================
# SHARED RESOURCES (persist across reruns)
# ============================================================================
@st.cache_resource
def _shared_queues():
    """Persistent queues across Streamlit reruns. THIS IS THE KEY FIX."""
    return {
        "data": queue.Queue(maxsize=DATA_Q_SIZE),
        "diag": queue.Queue(maxsize=DIAG_Q_SIZE),
    }


def data_q() -> queue.Queue:
    return _shared_queues()["data"]


def diag_q() -> queue.Queue:
    return _shared_queues()["diag"]


# ============================================================================
# PACKET PARSERS
# ============================================================================
def _parse_header(buf: bytes) -> Optional[dict]:
    if len(buf) < HDR_SZ:
        return None
    try:
        f = struct.unpack_from(HDR_FMT, buf, 0)
        return {"pid": int(f[5]), "pidx": int(f[10])}
    except Exception:
        return None


def _parse_motion(buf: bytes, pidx: int) -> Optional[dict]:
    off = HDR_SZ + pidx * MOT_SZ
    if len(buf) < off + MOT_SZ:
        return None
    try:
        f = struct.unpack_from(MOT_FMT, buf, off)
        return {"world_x": float(f[0]), "world_z": float(f[2]),
                "g_lat": float(f[12]), "g_lon": float(f[13])}
    except Exception:
        return None


def _parse_telemetry(buf: bytes, pidx: int) -> Optional[dict]:
    off = HDR_SZ + pidx * TEL_SZ
    if len(buf) < off + TEL_SZ:
        return None
    try:
        f = struct.unpack_from(TEL_FMT, buf, off)
        return {"speed": float(f[0]), "throttle": float(f[1]) * 100,
                "brake": float(f[3]) * 100, "gear": int(f[5]),
                "rpm": float(f[6]), "drs": int(f[7])}
    except Exception:
        return None


def _parse_lap(buf: bytes, pidx: int) -> Optional[dict]:
    off = HDR_SZ + pidx * LAP_PKT_SZ
    if len(buf) < off + LAP_PKT_SZ:
        return None
    try:
        ld = struct.unpack_from("<f", buf, off + 18)[0]
        ln = struct.unpack_from("<B", buf, off + 43)[0]
        return {"lap_distance": float(ld), "lap_number": int(ln)}
    except Exception:
        return None


def _parse_status(buf: bytes, pidx: int) -> Optional[dict]:
    off = HDR_SZ + pidx * STA_SZ
    if len(buf) < off + STA_SZ:
        return None
    try:
        f = struct.unpack_from(STA_FMT, buf, off)
        cid = int(f[18]) if len(f) > 18 else 0
        name, color = TYRES.get(cid, ("?", TEXT_MUTED))
        age = int(f[19]) if len(f) > 19 else 0
        return {"compound": name, "tyre_color": color, "tyre_age": age}
    except Exception:
        return None


# ============================================================================
# UDP LISTENER THREAD
# ============================================================================
class UDPListener(threading.Thread):
    def __init__(self, port: int, data_q_: queue.Queue, diag_q_: queue.Queue):
        super().__init__(daemon=True, name="F124-UDP")
        self.port = port
        self.data_q = data_q_
        self.diag_q = diag_q_
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    def _diag(self, msg: str):
        try:
            self.diag_q.put_nowait(f"[{time.strftime('%H:%M:%S')}] {msg}")
        except queue.Full:
            pass

    def run(self):
        self._diag(f"Hilo iniciado. Bind 0.0.0.0:{self.port}")
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        except AttributeError:
            pass
        sock.settimeout(1.0)
        try:
            sock.bind((UDP_HOST, self.port))
            self._diag("Bind OK. Escuchando...")
        except OSError as e:
            self._diag(f"ERROR bind: {e}")
            try:
                self.data_q.put_nowait({"_error": str(e)})
            except queue.Full:
                pass
            sock.close()
            return

        raw = 0
        pkts: dict = {}
        motion: dict = {}
        lap: dict = {}
        status: dict = {}

        while not self._stop.is_set():
            try:
                data, addr = sock.recvfrom(4096)
            except socket.timeout:
                continue
            except Exception as e:
                self._diag(f"recvfrom error: {e}")
                break

            raw += 1
            if raw == 1:
                self._diag(f"Primer datagrama de {addr[0]}:{addr[1]} ({len(data)}B)")

            hdr = _parse_header(data)
            if hdr is None:
                continue
            pid, pidx = hdr["pid"], hdr["pidx"]
            pkts[pid] = pkts.get(pid, 0) + 1

            if pid == 0:
                m = _parse_motion(data, pidx)
                if m: motion = m
            elif pid == 2:
                l = _parse_lap(data, pidx)
                if l: lap = l
            elif pid == 7:
                s = _parse_status(data, pidx)
                if s: status = s
            elif pid == 6:
                t = _parse_telemetry(data, pidx)
                if t:
                    frame = {
                        **t, **motion, **lap,
                        **(status or {"compound": "?", "tyre_color": TEXT_MUTED, "tyre_age": 0}),
                        "_raw": raw,
                        "_pkts": dict(pkts),
                        "ts": time.time(),
                    }
                    pkts["frames"] = pkts.get("frames", 0) + 1
                    try:
                        self.data_q.put_nowait(frame)
                    except queue.Full:
                        try:
                            self.data_q.get_nowait()
                            self.data_q.put_nowait(frame)
                        except queue.Empty:
                            pass

        sock.close()
        self._diag("Hilo terminado.")


# ============================================================================
# STATE MANAGEMENT
# ============================================================================
def init_state():
    defaults = {
        SK_LISTENER: None, SK_ROWS: [], SK_LAST: {}, SK_DIAG: [],
        SK_RAW: 0, SK_PKT: {}, SK_ERR: "", SK_FRAMES: 0,
        SK_OVERLAY: None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def drain_queues():
    """Drain shared queues into session state. Called from inside the fragment."""
    dq = diag_q()
    diag = st.session_state[SK_DIAG]
    while True:
        try:
            diag.append(dq.get_nowait())
        except queue.Empty:
            break
    if len(diag) > 50:
        st.session_state[SK_DIAG] = diag[-50:]

    rq = data_q()
    rows = st.session_state[SK_ROWS]
    while True:
        try:
            row = rq.get_nowait()
        except queue.Empty:
            break
        if "_error" in row:
            st.session_state[SK_ERR] = row["_error"]
            continue
        st.session_state[SK_LAST] = row
        st.session_state[SK_RAW] = row.get("_raw", 0)
        st.session_state[SK_PKT] = row.get("_pkts", {})
        st.session_state[SK_FRAMES] = row.get("_pkts", {}).get("frames", 0)
        rows.append(row)
    if len(rows) > MAX_ROWS:
        st.session_state[SK_ROWS] = rows[-MAX_ROWS:]


def start_listener():
    existing = st.session_state.get(SK_LISTENER)
    if existing and existing.is_alive():
        return
    # Drain stale data from cached queues
    for q in (data_q(), diag_q()):
        while not q.empty():
            try:
                q.get_nowait()
            except queue.Empty:
                break
    # Reset session state
    st.session_state[SK_ROWS] = []
    st.session_state[SK_LAST] = {}
    st.session_state[SK_DIAG] = []
    st.session_state[SK_RAW] = 0
    st.session_state[SK_PKT] = {}
    st.session_state[SK_ERR] = ""
    st.session_state[SK_FRAMES] = 0
    listener = UDPListener(UDP_PORT, data_q(), diag_q())
    listener.start()
    st.session_state[SK_LISTENER] = listener


def stop_listener():
    listener = st.session_state.get(SK_LISTENER)
    if listener:
        listener.stop()
        st.session_state[SK_LISTENER] = None


def is_listening() -> bool:
    listener = st.session_state.get(SK_LISTENER)
    return listener is not None and listener.is_alive()


# ============================================================================
# DATA PROCESSING
# ============================================================================
def rows_to_df(rows: list) -> pd.DataFrame:
    """Convert UDP rows -> FastF1-compatible DataFrame (Distance-aware)."""
    if not rows:
        return pd.DataFrame()
    clean = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]
    df = pd.DataFrame(clean)
    t0 = float(df["ts"].iloc[0])
    df["Time"] = pd.to_timedelta(df["ts"] - t0, unit="s")
    df = df.rename(columns={
        "speed": "Speed", "throttle": "Throttle", "brake": "Brake",
        "world_x": "X", "world_z": "Y",
        "gear": "nGear", "rpm": "RPM",
        "lap_distance": "Distance", "lap_number": "LapNumber",
        "compound": "Compound", "tyre_age": "TyreLife",
    })
    df["DRS"] = df.get("drs", pd.Series([0] * len(df))).apply(lambda x: 12 if x == 1 else 0)
    df["g_total"] = np.sqrt(
        df.get("g_lat", pd.Series([0.0] * len(df))) ** 2 +
        df.get("g_lon", pd.Series([0.0] * len(df))) ** 2
    )
    cols = ["Time", "Speed", "Throttle", "Brake", "nGear", "RPM", "DRS",
            "Distance", "X", "Y", "g_lat", "g_lon", "g_total",
            "Compound", "TyreLife", "LapNumber"]
    return df[[c for c in cols if c in df.columns]].reset_index(drop=True)


def load_overlay(file) -> Optional[pd.DataFrame]:
    """Load FastF1-exported lap (CSV/Parquet) for overlay."""
    if file is None:
        return None
    try:
        df = pd.read_parquet(file) if file.name.endswith(".parquet") else pd.read_csv(file)
        if "Distance" not in df.columns:
            st.error("El archivo de overlay debe contener la columna 'Distance' (export FastF1).")
            return None
        if "Time" in df.columns and df["Time"].dtype == object:
            df["Time"] = pd.to_timedelta(df["Time"])
        return df.sort_values("Distance").reset_index(drop=True)
    except Exception as e:
        st.error(f"Error cargando overlay: {e}")
        return None


def overlay_has_xy(overlay: Optional[pd.DataFrame]) -> bool:
    return (overlay is not None and "X" in overlay.columns
            and "Y" in overlay.columns and "Distance" in overlay.columns)


# ============================================================================
# VISUALIZATIONS
# ============================================================================
def _frame_atlas(fig: go.Figure, height: int = 220) -> go.Figure:
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=BG_SECONDARY,
        font=dict(family=FONT_MONO, color=TEXT_PRIMARY, size=9),
        margin=dict(l=6, r=6, t=6, b=6),
    )
    return fig


def make_gauge(val: float, max_val: float, color: str, suffix: str,
               threshold: Optional[float] = None) -> go.Figure:
    g = dict(
        axis=dict(range=[0, max_val], tickfont=dict(size=7)),
        bar=dict(color=color, thickness=0.25),
        bgcolor=BG_SECONDARY,
        bordercolor="rgba(255,255,255,0.08)",
    )
    if threshold:
        g["threshold"] = dict(line=dict(color=ACCENT_RED, width=2),
                              thickness=0.65, value=threshold)
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=val,
        number=dict(font=dict(size=22, color=color, family=FONT_MONO), suffix=suffix),
        gauge=g,
    ))
    return _frame_atlas(fig, 180)


def make_gg(rows: list) -> go.Figure:
    if len(rows) < 5:
        return _frame_atlas(go.Figure(), 220)
    sub = rows[-500:]
    g_lat = np.array([r.get("g_lat", 0) for r in sub], dtype=float)
    g_lon = np.array([r.get("g_lon", 0) for r in sub], dtype=float)
    g_tot = np.sqrt(g_lat ** 2 + g_lon ** 2)
    fig = go.Figure(go.Scattergl(
        x=g_lat, y=g_lon, mode="markers",
        marker=dict(size=3, color=g_tot,
                    colorscale=[[0, BG_PRIMARY], [0.4, "#4B0082"],
                                [0.8, ACCENT_PURPLE], [1, "#FFF"]],
                    opacity=0.75, showscale=False),
        hoverinfo="skip",
    ))
    p98 = float(np.percentile(g_tot, 98)) if len(g_tot) else 2.0
    rng = max(2.0, p98 * 1.15) if not np.isnan(p98) else 2.0
    fig.update_layout(
        showlegend=False,
        xaxis=dict(range=[-rng, rng], showgrid=True, zeroline=True,
                   gridcolor=GRID_LINE, zerolinecolor="rgba(255,255,255,0.2)",
                   title=dict(text="G Lat", font=dict(size=8)),
                   tickfont=dict(size=7)),
        yaxis=dict(range=[-rng, rng], showgrid=True, zeroline=True,
                   gridcolor=GRID_LINE, zerolinecolor="rgba(255,255,255,0.2)",
                   title=dict(text="G Lon", font=dict(size=8)),
                   scaleanchor="x", scaleratio=1, tickfont=dict(size=7)),
    )
    return _frame_atlas(fig, 220)


def make_track_from_overlay(overlay: pd.DataFrame, current_distance: Optional[float],
                            color: str, height: int = 220) -> Optional[go.Figure]:
    """Draw FastF1 circuit outline; place driver via Distance interpolation."""
    if not overlay_has_xy(overlay):
        return None
    d = overlay["Distance"].to_numpy(dtype=float)
    x = overlay["X"].to_numpy(dtype=float)
    y = overlay["Y"].to_numpy(dtype=float)

    fig = go.Figure()
    # Circuit outline
    fig.add_trace(go.Scattergl(
        x=x, y=y, mode="lines",
        line=dict(color="rgba(255,255,255,0.45)", width=2),
        hoverinfo="skip", showlegend=False,
    ))
    # Start/Finish marker
    fig.add_trace(go.Scatter(
        x=[x[0]], y=[y[0]], mode="markers",
        marker=dict(size=8, color=ACCENT_GREEN, symbol="square",
                    line=dict(color="#FFF", width=1)),
        hoverinfo="skip", showlegend=False,
    ))
    # Current position via interpolation against Distance
    if current_distance is not None and len(d) > 1:
        d_max = float(d[-1])
        cd = float(current_distance) % d_max if d_max > 0 else 0.0
        cx = float(np.interp(cd, d, x))
        cy = float(np.interp(cd, d, y))
        fig.add_trace(go.Scatter(
            x=[cx], y=[cy], mode="markers",
            marker=dict(size=14, color=color, symbol="circle",
                        line=dict(color="#FFF", width=2)),
            hoverinfo="skip", showlegend=False,
        ))
    fig.update_layout(
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   scaleanchor="y", scaleratio=1),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    )
    return _frame_atlas(fig, height)


def make_telemetry_trace(df: pd.DataFrame, color: str,
                         overlay: Optional[pd.DataFrame] = None,
                         x_axis: str = "Distance",
                         height_per_row: int = 110) -> go.Figure:
    """Stacked telemetry channels vs Distance (or Time). Optional FastF1 overlay."""
    spec = [
        ("Speed",    "km/h", color,         [0, 380]),
        ("Throttle", "%",    ACCENT_GREEN,  [-2, 105]),
        ("Brake",    "%",    ACCENT_RED,    [-2, 105]),
        ("nGear",    "",     ACCENT_CYAN,   [0, 9]),
        ("RPM",      "",     ACCENT_YELLOW, [0, 14000]),
        ("DRS",      "",     ACCENT_PURPLE, [-2, 14]),
    ]
    avail = [s for s in spec if s[0] in df.columns]
    if not avail or x_axis not in df.columns or len(df) < 2:
        return _frame_atlas(go.Figure(), 220)

    fig = make_subplots(
        rows=len(avail), cols=1, shared_xaxes=True,
        vertical_spacing=0.012,
        subplot_titles=[s[0] for s in avail],
    )
    for i, (ch, unit, ch_color, yrange) in enumerate(avail, start=1):
        fig.add_trace(go.Scattergl(
            x=df[x_axis], y=df[ch],
            mode="lines", line=dict(color=ch_color, width=1.6),
            name=ch,
            hovertemplate=f"{ch}: %{{y:.1f}} {unit}<extra></extra>",
        ), row=i, col=1)
        if overlay is not None and ch in overlay.columns and x_axis in overlay.columns:
            fig.add_trace(go.Scattergl(
                x=overlay[x_axis], y=overlay[ch],
                mode="lines",
                line=dict(color="rgba(255,255,255,0.55)", width=1, dash="dot"),
                name=f"{ch} REF",
                hovertemplate=f"REF {ch}: %{{y:.1f}} {unit}<extra></extra>",
            ), row=i, col=1)
        fig.update_yaxes(range=yrange, gridcolor=GRID_LINE,
                         zerolinecolor="rgba(255,255,255,0.1)",
                         tickfont=dict(size=8), row=i, col=1)

    fig.update_xaxes(gridcolor=GRID_LINE, tickfont=dict(size=8),
                     title=dict(text=x_axis, font=dict(size=10)),
                     row=len(avail), col=1)
    fig.update_layout(
        height=height_per_row * len(avail),
        showlegend=False,
        paper_bgcolor=BG_PRIMARY,
        plot_bgcolor=BG_SECONDARY,
        font=dict(family=FONT_MONO, color=TEXT_PRIMARY, size=9),
        margin=dict(l=40, r=10, t=30, b=40),
    )
    fig.update_annotations(font_size=10, font_color=TEXT_MUTED)
    return fig


# ============================================================================
# UI BLOCKS
# ============================================================================
ATLAS_CSS = f"""
<style>
  .stApp {{ background: linear-gradient(180deg, {BG_PRIMARY} 0%, #0a0a14 100%); }}
  [data-testid="stMetric"] {{
    background: {BG_SECONDARY}; padding: 8px 12px;
    border-left: 2px solid {ACCENT_RED}; border-radius: 2px;
  }}
  [data-testid="stMetricLabel"] {{
    font-family: {FONT_MONO}; font-size: 9px !important;
    letter-spacing: 1.2px; color: rgba(255,255,255,0.5) !important;
    text-transform: uppercase;
  }}
  [data-testid="stMetricValue"] {{
    font-family: {FONT_MONO}; font-size: 18px !important; font-weight: 700;
  }}
  .stButton > button {{
    font-family: {FONT_MONO}; letter-spacing: 1px;
    text-transform: uppercase; font-size: 11px; border-radius: 2px;
  }}
  .stTabs [data-baseweb="tab-list"] {{ gap: 2px; }}
  .stTabs [data-baseweb="tab"] {{
    font-family: {FONT_MONO}; font-size: 11px;
    letter-spacing: 1.2px; text-transform: uppercase;
  }}
  h1, h2, h3 {{ font-family: {FONT_MONO} !important; letter-spacing: 1.5px; }}
  hr {{ border-color: rgba(255,255,255,0.06) !important; }}
</style>
"""


def render_header():
    st.markdown(ATLAS_CSS, unsafe_allow_html=True)
    st.markdown(
        f"""
        <div style="display:flex;align-items:center;justify-content:space-between;
                    border-bottom:1px solid rgba(255,255,255,0.08);padding-bottom:10px;
                    margin-bottom:14px;">
          <div>
            <div style="font-family:{FONT_MONO};font-size:10px;letter-spacing:3px;
                        color:{ACCENT_RED};font-weight:800;">▌ TELEMETRY ENGINEERING</div>
            <div style="font-family:{FONT_MONO};font-size:22px;font-weight:800;
                        color:{TEXT_PRIMARY};letter-spacing:2px;">F1·24 UDP ANALYZER</div>
          </div>
          <div style="text-align:right;font-family:{FONT_MONO};font-size:10px;
                      color:rgba(255,255,255,0.45);letter-spacing:1.5px;">
            ATLAS-STYLE · LIVE+POST · DISTANCE-INDEXED
          </div>
        </div>
        """, unsafe_allow_html=True
    )


def render_status_strip(frame: dict, color: str):
    speed   = frame.get("speed", 0)
    gear    = frame.get("gear", 0)
    rpm     = frame.get("rpm", 0)
    drs     = frame.get("drs", 0)
    lap     = frame.get("lap_number", 0)
    dist    = frame.get("lap_distance", 0)
    comp    = frame.get("compound", "?")
    age     = frame.get("tyre_age", 0)
    t_color = frame.get("tyre_color", "#888")
    g_lat   = frame.get("g_lat", 0.0)
    g_lon   = frame.get("g_lon", 0.0)
    g_tot   = float(np.sqrt(g_lat ** 2 + g_lon ** 2))
    drs_html = (f"<span style='color:{ACCENT_GREEN};font-weight:800;'>OPEN</span>"
                if drs == 1
                else "<span style='color:rgba(255,255,255,0.25);'>—</span>")

    st.markdown(f"""
    <div style="background:{BG_TERTIARY};border-left:4px solid {color};
        padding:10px 18px;border-radius:3px;font-family:{FONT_MONO};
        display:grid;grid-template-columns:repeat(6,1fr);gap:10px;">
      <div>
        <div style="font-size:9px;color:rgba(255,255,255,0.4);">SPEED</div>
        <div style="font-size:18px;font-weight:800;color:{color};">{speed:.0f}<span style="font-size:10px;color:rgba(255,255,255,0.4);"> km/h</span></div></div>
      <div>
        <div style="font-size:9px;color:rgba(255,255,255,0.4);">LAP / DIST</div>
        <div style="font-size:15px;font-weight:800;color:{TEXT_PRIMARY};">L{int(lap)} · {dist:.0f}m</div></div>
      <div>
        <div style="font-size:9px;color:rgba(255,255,255,0.4);">GEAR / RPM</div>
        <div style="font-size:15px;font-weight:800;color:{TEXT_PRIMARY};">{int(gear)}&nbsp;<span style='font-size:11px;'>{rpm:.0f}</span></div></div>
      <div>
        <div style="font-size:9px;color:rgba(255,255,255,0.4);">G LAT / LON / TOT</div>
        <div style="font-size:12px;font-weight:700;color:{ACCENT_CYAN};">{g_lat:+.2f} / {g_lon:+.2f} / {g_tot:.2f}</div></div>
      <div>
        <div style="font-size:9px;color:rgba(255,255,255,0.4);">TYRE</div>
        <div style="font-size:13px;font-weight:700;">
          <span style="background:{t_color};color:#000;padding:2px 6px;border-radius:3px;">{comp}</span>
          <span style="color:rgba(255,255,255,0.5);font-size:10px;margin-left:5px;">{int(age)} L</span></div></div>
      <div>
        <div style="font-size:9px;color:rgba(255,255,255,0.4);">DRS</div>
        <div style="font-size:14px;">{drs_html}</div></div>
    </div>""", unsafe_allow_html=True)


def render_metrics_strip():
    raw    = st.session_state[SK_RAW]
    pkts   = st.session_state[SK_PKT]
    frames = st.session_state[SK_FRAMES]
    err    = st.session_state[SK_ERR]
    diag   = st.session_state[SK_DIAG]
    alive  = is_listening()
    last   = st.session_state.get(SK_LAST, {})
    age    = time.time() - last.get("ts", 0) if last.get("ts") else 9999

    cols = st.columns(6)
    cols[0].metric("Hilo UDP", "🔴 LIVE" if alive else "⚫ OFF")
    cols[1].metric("RAW pkts", f"{raw:,}")
    cols[2].metric("Motion(0)", f"{pkts.get(0, 0):,}")
    cols[3].metric("Telem(6)", f"{pkts.get(6, 0):,}")
    cols[4].metric("Frames", f"{frames:,}")
    cols[5].metric("Δt last", f"{age:.1f}s" if age < 9999 else "—")

    if err:
        st.error(f"❌ {err}")
    if diag:
        with st.expander("📋 Log UDP", expanded=raw == 0 and alive):
            st.code("\n".join(diag[-20:]))
    if alive and raw == 0:
        st.warning("🔌 Socket abierto, sin datagramas. Usa **🧪 Test síncrono**.")
    elif alive and raw > 0 and pkts.get(6, 0) == 0:
        st.warning(f"📦 RAW={raw:,} pero packet_id=6 (Car Telemetry) = 0. Verifica **UDP Format: 2024**.")


def sync_test(seconds: float = 5.0):
    was_alive = is_listening()
    if was_alive:
        stop_listener()
        time.sleep(0.3)
    st.info(f"🔌 Test síncrono {seconds:.0f}s en 0.0.0.0:{UDP_PORT}.")
    bar = st.progress(0.0)
    box = st.empty()
    log = []
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.settimeout(0.25)
    try:
        sock.bind((UDP_HOST, UDP_PORT))
        log.append(f"✅ Socket abierto en 0.0.0.0:{UDP_PORT}")
        box.code("\n".join(log))
    except OSError as e:
        st.error(f"❌ {e}")
        if was_alive:
            start_listener()
        return
    t0 = time.time()
    n = 0
    while time.time() - t0 < seconds:
        elapsed = time.time() - t0
        bar.progress(min(elapsed / seconds, 1.0))
        try:
            data, addr = sock.recvfrom(4096)
            n += 1
            hdr = _parse_header(data)
            pid = hdr["pid"] if hdr else "?"
            log.append(f"✅ #{n} {len(data):4d}B {addr[0]}:{addr[1]} id={pid}")
            box.code("\n".join(log[-15:]))
        except socket.timeout:
            if not n:
                box.caption(f"Esperando... {elapsed:.1f}/{seconds:.0f}s")
    sock.close()
    bar.empty()
    if n:
        st.success(f"✅ {n} paquetes recibidos.")
    else:
        st.error("❌ Cero paquetes. Revisa F1 24 → UDP On, 127.0.0.1, 20777, Format 2024.")
    if was_alive:
        start_listener()


# ============================================================================
# LIVE FRAGMENT — module level, fixed run_every. THIS IS THE LIVE LOOP.
# ============================================================================
@st.fragment(run_every=f"{REFRESH_MS}ms")
def live_dashboard():
    drain_queues()
    last    = st.session_state.get(SK_LAST, {})
    rows    = st.session_state.get(SK_ROWS, [])
    overlay = st.session_state.get(SK_OVERLAY, None)
    color   = st.session_state.get("f124_color", ACCENT_RED)

    render_metrics_strip()
    st.divider()

    if not last:
        st.info("Hilo activo. Esperando primer frame de telemetría (packet_id=6)...")
        return

    render_status_strip(last, color)
    st.markdown("<br>", unsafe_allow_html=True)

    has_track = overlay_has_xy(overlay)
    if has_track:
        gc = st.columns([1.4, 1, 1, 1.8, 1.8])
    else:
        gc = st.columns([1.4, 1, 1, 2.6])

    with gc[0]:
        st.caption("VELOCIDAD")
        st.plotly_chart(make_gauge(last.get("speed", 0), 380, color, " km/h", 320),
                        use_container_width=True, config={"displayModeBar": False},
                        key="lv_g_spd")
    with gc[1]:
        st.caption("ACELERADOR")
        st.plotly_chart(make_gauge(last.get("throttle", 0), 100, ACCENT_GREEN, "%"),
                        use_container_width=True, config={"displayModeBar": False},
                        key="lv_g_thr")
    with gc[2]:
        st.caption("FRENO")
        st.plotly_chart(make_gauge(last.get("brake", 0), 100, ACCENT_RED, "%"),
                        use_container_width=True, config={"displayModeBar": False},
                        key="lv_g_brk")
    with gc[3]:
        st.caption("G-G (last 500)")
        st.plotly_chart(make_gg(rows),
                        use_container_width=True, config={"displayModeBar": False},
                        key="lv_gg")
    if has_track:
        with gc[4]:
            st.caption("TRAZADO (FastF1)")
            track_fig = make_track_from_overlay(
                overlay, last.get("lap_distance"), color, height=220
            )
            st.plotly_chart(track_fig,
                            use_container_width=True, config={"displayModeBar": False},
                            key="lv_track")

    st.markdown("<br>", unsafe_allow_html=True)
    if len(rows) >= 2:
        df = rows_to_df(rows)
        if not df.empty and "Distance" in df.columns:
            if "LapNumber" in df.columns and not df["LapNumber"].empty:
                cur_lap = df["LapNumber"].iloc[-1]
                df_live = df[df["LapNumber"] == cur_lap]
                if len(df_live) < 2:
                    df_live = df
            else:
                df_live = df
            fig = make_telemetry_trace(df_live, color, overlay=overlay,
                                       x_axis="Distance", height_per_row=95)
            st.plotly_chart(fig, use_container_width=True,
                            config={"displayModeBar": False},
                            key="lv_telem")


# ============================================================================
# POST-SESSION VIEW
# ============================================================================
def render_post_session(df: pd.DataFrame, color: str,
                        overlay: Optional[pd.DataFrame] = None):
    if df.empty:
        st.info("Sin datos.")
        return

    laps = sorted(df["LapNumber"].unique().tolist()) if "LapNumber" in df.columns else [1]
    c1, c2 = st.columns([1, 3])
    with c1:
        sel_lap = st.selectbox("Vuelta", laps,
                               format_func=lambda x: f"Lap {int(x)}",
                               key="post_lap")
    sub = df[df["LapNumber"] == sel_lap].copy() if "LapNumber" in df.columns else df
    with c2:
        st.metric("Muestras", f"{len(sub):,}")

    has_track = overlay_has_xy(overlay)
    tab_labels = ["📊 TELEMETRÍA", "⭕ G-G"]
    if has_track:
        tab_labels.append("📍 TRAZADO (FastF1)")
    tabs = st.tabs(tab_labels)

    with tabs[0]:
        if "Distance" in sub.columns:
            fig = make_telemetry_trace(sub, color, overlay=overlay,
                                       x_axis="Distance", height_per_row=110)
            st.plotly_chart(fig, use_container_width=True,
                            config={"displayModeBar": False},
                            key="post_telem")
        else:
            st.info("Falta columna Distance.")

    with tabs[1]:
        if "g_lat" in sub.columns:
            fig = go.Figure(go.Scattergl(
                x=sub["g_lat"], y=sub["g_lon"], mode="markers",
                marker=dict(size=3,
                            color=sub.get("g_total", sub["g_lat"].abs()),
                            colorscale=[[0, BG_PRIMARY], [0.6, ACCENT_PURPLE], [1, "#FFF"]],
                            showscale=False, opacity=0.7),
            ))
            fig.update_layout(
                height=480, paper_bgcolor=BG_PRIMARY, plot_bgcolor=BG_SECONDARY,
                font=dict(family=FONT_MONO, color=TEXT_PRIMARY),
                xaxis=dict(title="G Lat", scaleanchor="y", scaleratio=1, gridcolor=GRID_LINE),
                yaxis=dict(title="G Lon", gridcolor=GRID_LINE),
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True,
                            config={"displayModeBar": False},
                            key="post_gg")

    if has_track:
        with tabs[2]:
            d_ref = overlay["Distance"].to_numpy(dtype=float)
            x_ref = overlay["X"].to_numpy(dtype=float)
            y_ref = overlay["Y"].to_numpy(dtype=float)
            d_max = float(d_ref[-1]) if len(d_ref) else 0.0

            sub_d = sub["Distance"].to_numpy(dtype=float)
            sub_d = sub_d % d_max if d_max > 0 else sub_d
            sx = np.interp(sub_d, d_ref, x_ref)
            sy = np.interp(sub_d, d_ref, y_ref)

            fig = go.Figure()
            fig.add_trace(go.Scattergl(
                x=x_ref, y=y_ref, mode="lines",
                line=dict(color="rgba(255,255,255,0.35)", width=2),
                hoverinfo="skip", showlegend=False,
            ))
            fig.add_trace(go.Scattergl(
                x=sx, y=sy, mode="markers",
                marker=dict(size=4, color=sub.get("Speed", 0),
                            colorscale=[[0, "#0044FF"], [0.5, ACCENT_GREEN], [1, ACCENT_RED]],
                            showscale=True,
                            colorbar=dict(title="km/h", tickfont=dict(size=8))),
                hoverinfo="skip", showlegend=False,
            ))
            fig.update_layout(
                height=600, paper_bgcolor=BG_PRIMARY, plot_bgcolor=BG_SECONDARY,
                font=dict(family=FONT_MONO, color=TEXT_PRIMARY),
                xaxis=dict(scaleanchor="y", scaleratio=1, showgrid=False,
                           zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True,
                            config={"displayModeBar": False},
                            key="post_track")


def render_downloads(rows: list):
    df = rows_to_df(rows)
    if df.empty:
        return
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv = df.to_csv(index=False).encode()
    c1, c2 = st.columns(2)
    c1.download_button("⬇ CSV", csv, f"f124_{ts}.csv", "text/csv", key="dl_csv")
    try:
        buf = io.BytesIO()
        df.to_parquet(buf, index=False)
        c2.download_button("⬇ Parquet", buf.getvalue(),
                           f"f124_{ts}.parquet", key="dl_pq")
    except Exception:
        pass


def render_uploaded():
    f = st.file_uploader("Sesión grabada (.csv / .parquet)",
                         type=["csv", "parquet"], key="up_session")
    if f is None:
        return None
    try:
        df = pd.read_parquet(f) if f.name.endswith(".parquet") else pd.read_csv(f)
        if "Time" in df.columns and df["Time"].dtype == object:
            df["Time"] = pd.to_timedelta(df["Time"])
        st.success(f"✅ {len(df):,} muestras cargadas")
        return df
    except Exception as e:
        st.error(str(e))
        return None


# ============================================================================
# MAIN
# ============================================================================
def render_f1_live():
    st.set_page_config(page_title="F1·24 ATLAS", layout="wide",
                       initial_sidebar_state="expanded")
    init_state()
    render_header()

    # ----- Sidebar: control + overlay -----
    with st.sidebar:
        st.markdown(f"<div style='font-family:{FONT_MONO};font-size:10px;"
                    f"letter-spacing:2px;color:{ACCENT_RED};'>▌ CONTROL</div>",
                    unsafe_allow_html=True)
        st.color_picker("Driver color", ACCENT_RED, key="f124_color")
        st.caption(f"Live refresh: {REFRESH_MS} ms")

        st.divider()
        st.markdown(f"<div style='font-family:{FONT_MONO};font-size:10px;"
                    f"letter-spacing:2px;color:{ACCENT_CYAN};'>▌ OVERLAY (FastF1)</div>",
                    unsafe_allow_html=True)
        ov_file = st.file_uploader("Reference lap (.csv / .parquet)",
                                   type=["csv", "parquet"], key="ov_file")
        if ov_file is not None:
            ov = load_overlay(ov_file)
            if ov is not None:
                st.session_state[SK_OVERLAY] = ov
                xy_ok = overlay_has_xy(ov)
                st.success(f"✅ Overlay: {len(ov):,} pts" +
                           (" · X/Y OK (trazado activo)" if xy_ok
                            else " · sin X/Y (sólo telemetría)"))
        if st.session_state[SK_OVERLAY] is not None:
            if st.button("🗑 Quitar overlay", key="clear_ov", use_container_width=True):
                st.session_state[SK_OVERLAY] = None
                st.rerun()

    # ----- Top action bar -----
    bc = st.columns([1, 1, 1, 1, 6])
    with bc[0]:
        if st.button("▶ START", type="primary", key="btn_start", use_container_width=True):
            start_listener()
            st.rerun()
    with bc[1]:
        if st.button("⏹ STOP", key="btn_stop", use_container_width=True):
            stop_listener()
            st.rerun()
    with bc[2]:
        rows = st.session_state.get(SK_ROWS, [])
        if rows:
            render_downloads(rows)
    with bc[3]:
        if st.button("🧪 TEST", key="btn_test", use_container_width=True,
                     help="Test síncrono 5s sin hilo"):
            sync_test(5.0)

    st.divider()

    # ----- View selector -----
    view = st.radio("VIEW", ["📡 LIVE", "📊 POST-SESSION", "📂 LOAD"],
                    horizontal=True, key="view_sel", label_visibility="collapsed")

    if view == "📡 LIVE":
        if not is_listening():
            st.info("Pulsa **▶ START** para comenzar la captura.")
            return
        live_dashboard()

    elif view == "📊 POST-SESSION":
        rows = st.session_state.get(SK_ROWS, [])
        if rows:
            render_post_session(rows_to_df(rows), st.session_state["f124_color"],
                                overlay=st.session_state.get(SK_OVERLAY))
        else:
            st.info("Sin datos en esta sesión.")

    else:  # LOAD
        df = render_uploaded()
        if df is not None:
            render_post_session(df, st.session_state["f124_color"],
                                overlay=st.session_state.get(SK_OVERLAY))


