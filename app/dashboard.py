import os
import json
from datetime import datetime
import pandas as pd
import streamlit as st
import requests
from PIL import Image
import io

# =========================================================
# Configuration — API backend connection
# =========================================================
# In Docker Compose, API_URL is injected as http://api:8000 (the
# service name, resolved via Docker's internal DNS). Falls back to
# localhost for running the dashboard outside Docker during dev.
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Fabric QC Platform",
    layout="wide",
    initial_sidebar_state="collapsed"
)


@st.cache_data(ttl=10)
def check_api_health():
    """
    Ping the backend's /health endpoint.

    Cached for 10s (st.cache_data, not cache_resource — this is a
    small JSON value, not an expensive resource) so we don't fire
    an HTTP request on every single widget interaction/rerun, while
    still picking up a backend restart within a few seconds.
    """
    try:
        resp = requests.get(f"{API_URL}/health", timeout=10)
        if resp.status_code == 200:
            return True, resp.json()
        return False, None
    except requests.exceptions.RequestException:
        return False, None


def call_predict(file_bytes: bytes, filename: str, content_type: str) -> dict:
    """
    Call the backend's /predict endpoint — JSON detections, with
    the seam-merge postprocessing already applied server-side (see
    api/postprocess.py). This is the single source of truth for
    detection results; the dashboard never re-implements inference
    or merge logic locally.
    """
    files = {"file": (filename, file_bytes, content_type)}
    resp = requests.post(f"{API_URL}/predict", files=files, timeout=120)
    resp.raise_for_status()
    return resp.json()


def call_predict_annotated(file_bytes: bytes, filename: str, content_type: str) -> bytes:
    """
    Call the backend's /predict/annotated endpoint — PNG bytes with
    boxes already drawn server-side using DEFECT_METADATA colors
    (see api/main.py draw_merged_boxes()), so the dashboard doesn't
    need its own drawing logic and can never drift out of sync with
    the backend's box colors.
    """
    files = {"file": (filename, file_bytes, content_type)}
    resp = requests.post(f"{API_URL}/predict/annotated", files=files, timeout=120)
    resp.raise_for_status()
    return resp.content


def call_predict_batch(files_payload: list) -> dict:
    """Call the backend's /predict/batch endpoint for factory QC mode."""
    resp = requests.post(f"{API_URL}/predict/batch", files=files_payload, timeout=120)
    resp.raise_for_status()
    return resp.json()

# =========================================================
# Defect Classes & Colors
# =========================================================
# These hex values are the exact RGB equivalent of the BGR tuples
# used by the backend's draw_merged_boxes() (api/main.py), which is
# what actually paints the annotated PNG returned by
# /predict/annotated. The two must always match — if you change a
# color here, change the matching BGR tuple in api/main.py's
# class_colors dict too, or the legend will lie about what the
# boxes look like (this is exactly the bug that was just fixed:
# the two palettes had drifted apart with no connection between
# them).
#
#   Class        Backend BGR (api/main.py)   Hex (this file)
#   -----------  --------------------------  ---------------
#   Stain        (0, 165, 255)                #FFA500  orange
#   Thread       (255, 0, 0)                  #0000FF  blue
#   Warp_Weft    (0, 255, 255)                #FFFF00  yellow
#   hole         (0, 0, 255)                  #FF0000  red
#   seam         (255, 0, 255)                #FF00FF  magenta
DEFECT_METADATA = {
    "Stain": {"color": "#FFA500"},
    "Thread": {"color": "#0000FF"},
    "Warp_Weft": {"color": "#FFFF00"},
    "hole": {"color": "#FF0000"},
    "seam": {"color": "#FF00FF"}
}

# =========================================================
# Session State & Theme Management
# =========================================================

if "history" not in st.session_state:
    st.session_state.history = []

if "theme_mode" not in st.session_state:
    st.session_state.theme_mode = "Dark"

if "single_result" not in st.session_state:
    st.session_state.single_result = None

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

if "batch_files_list" not in st.session_state:
    st.session_state.batch_files_list = []

if "batch_key" not in st.session_state:
    st.session_state.batch_key = 200

if "batch_result" not in st.session_state:
    st.session_state.batch_result = None


def start_new_single_scan():
    """
    Clear the current single-scan image + result and remount the
    uploader with a fresh widget instance.

    st.file_uploader has no public "clear" method — the standard
    Streamlit pattern is to change the widget's `key`, which forces
    a brand-new (empty) uploader on the next render. We reuse the
    same uploader_key counter the existing "✕ remove image" button
    already increments, so both paths stay consistent.
    """
    st.session_state.uploader_key += 1
    st.session_state.single_result = None


def clear_all_batch():
    """
    Clear every staged batch image, the last batch result, and
    remount the batch uploader (same key-bump pattern as
    start_new_single_scan — see its docstring for why).
    """
    st.session_state.batch_files_list = []
    st.session_state.batch_result = None
    st.session_state.batch_key += 1


effective_theme = st.session_state.theme_mode.lower()

# =========================================================
# Professional Theme Palette
# =========================================================

THEMES = {
    "light": {
        "bg_primary": "#FFFFFF",
        "bg_secondary": "#F8FAFC",
        "bg_tertiary": "#F1F5F9",
        "text_primary": "#0F172A",
        "text_secondary": "#334155",
        "text_muted": "#64748B",
        "border": "#CBD5E1",
        "accent": "#2563EB",
        "accent_hover": "#1D4ED8",
        "accent_soft": "rgba(37, 99, 235, 0.10)",
        "success": "#059669",
        "success_bg": "#ECFDF5",
        "error": "#DC2626",
        "error_bg": "#FEF2F2",
        "shadow": "0 1px 3px rgba(0,0,0,0.1)",
        "shadow_hover": "0 8px 20px rgba(15, 23, 42, 0.08)"
    },
    "dark": {
        "bg_primary": "#0B0F1A",
        "bg_secondary": "#161B2B",
        "bg_tertiary": "#1E2536",
        "text_primary": "#F8FAFC",
        "text_secondary": "#CBD5E1",
        "text_muted": "#94A3B8",
        "border": "#2D3748",
        "accent": "#3B82F6",
        "accent_hover": "#60A5FA",
        "accent_soft": "rgba(59, 130, 246, 0.12)",
        "success": "#10B981",
        "success_bg": "#064E3B",
        "error": "#EF4444",
        "error_bg": "#7F1D1D",
        "shadow": "0 4px 6px rgba(0,0,0,0.3)",
        "shadow_hover": "0 12px 28px rgba(0,0,0,0.45)"
    }
}

theme = THEMES[effective_theme]

# =========================================================
# CSS Injection (FORCED OVERRIDES)
# =========================================================

def inject_css():
    css = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    .stApp, .stApp > header, .stApp [data-testid="stHeader"] {{
        background-color: {theme['bg_primary']} !important;
    }}

    html, body, [data-testid="stAppViewContainer"], [data-testid="stMainViewContainer"] {{
        background-color: {theme['bg_primary']} !important;
        color: {theme['text_primary']} !important;
        font-family: 'Inter', sans-serif;
    }}

    h1, h2, h3, h4, h5, h6, p, span, label, .stMarkdown, div, li, ul {{
        color: {theme['text_primary']} !important;
    }}

    .brand-sub, .kpi-label, .text-muted, .footer-text {{
        color: {theme['text_muted']} !important;
    }}

    /* Subtle global scrollbar polish */
    ::-webkit-scrollbar {{ width: 10px; height: 10px; }}
    ::-webkit-scrollbar-track {{ background: {theme['bg_primary']}; }}
    ::-webkit-scrollbar-thumb {{
        background: {theme['border']};
        border-radius: 8px;
    }}
    ::-webkit-scrollbar-thumb:hover {{ background: {theme['accent']}; }}

    .nav-container, .legend-bar, .kpi-card {{
        background-color: {theme['bg_secondary']} !important;
        border: 1px solid {theme['border']} !important;
        border-radius: 12px;
        box-shadow: {theme['shadow']};
    }}

    .legend-bar {{
        display: flex;
        justify-content: center;
        align-items: center;
        flex-wrap: wrap;
        gap: 24px;
        padding: 12px 20px;
        margin-bottom: 24px;
    }}

    .legend-item {{
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 4px 10px;
        border-radius: 999px;
        transition: background-color 0.15s ease;
    }}

    .legend-item:hover {{
        background-color: {theme['bg_tertiary']};
    }}

    .legend-dot {{
        width: 10px;
        height: 10px;
        border-radius: 50%;
        box-shadow: 0 0 0 3px transparent;
        transition: box-shadow 0.15s ease;
    }}

    .legend-item:hover .legend-dot {{
        box-shadow: 0 0 0 3px var(--dot-glow, transparent);
    }}

    .legend-text {{
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}

    /* ---- KPI Cards ---- */
    .kpi-card {{
        padding: 22px 24px 20px 24px;
        text-align: center;
        position: relative;
        overflow: hidden;
        transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
    }}

    .kpi-card::before {{
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, {theme['accent']}, transparent);
        opacity: 0.85;
    }}

    .kpi-card:hover {{
        transform: translateY(-2px);
        box-shadow: {theme['shadow_hover']};
        border-color: {theme['accent']} !important;
    }}

    .kpi-label {{
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 6px;
    }}

    .kpi-value {{
        font-size: 2.5rem;
        font-weight: 800;
        letter-spacing: -1.5px;
        line-height: 1.1;
    }}

    /* Section headers used throughout (Upload, Detection Result, etc.) */
    .section-heading {{
        font-size: 1.05rem;
        font-weight: 700;
        letter-spacing: -0.01em;
        margin-bottom: 4px;
        display: flex;
        align-items: center;
        gap: 8px;
    }}

    .section-heading .dot {{
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: {theme['accent']};
        display: inline-block;
    }}

    /* ---- Table Styling ----
       Redesigned away from the solid-accent-fill header (felt heavy
       / less "industry dashboard") toward a flat, bordered look:
       muted uppercase header on a tertiary tint with just a bottom
       rule, subtle zebra striping for row scanning, and a single
       outer border + rounded corners so the whole table reads as
       one contained card in both themes — no theme-specific
       overrides needed since every color comes from {theme}. */
    table {{
        width: 100%;
        background-color: {theme['bg_secondary']} !important;
        color: {theme['text_primary']} !important;
        border-collapse: separate;
        border-spacing: 0;
        border: 1px solid {theme['border']} !important;
        border-radius: 10px;
        overflow: hidden;
    }}

    th {{
        /* accent_soft (a light wash of the accent blue) instead of
           bg_tertiary — bg_tertiary sat too close to the body row
           color to read as "this is the header" at a glance,
           especially in light theme. A light accent tint is a
           standard data-table convention and stays distinct in
           both themes since it's derived from the theme accent,
           not a fixed gray. */
        background-color: {theme['accent_soft']} !important;
        color: {theme['accent']} !important;
        text-align: left;
        padding: 11px 14px;
        font-weight: 700;
        text-transform: uppercase;
        font-size: 0.7rem;
        letter-spacing: 0.06em;
        border-bottom: 1px solid {theme['border']} !important;
    }}

    td {{
        padding: 11px 14px;
        border-bottom: 1px solid {theme['border']} !important;
        color: {theme['text_primary']} !important;
        font-size: 0.85rem;
    }}

    tr:last-child td {{
        border-bottom: none !important;
    }}

    tbody tr:nth-child(even) td {{
        background-color: {theme['bg_tertiary']}66;
    }}

    tbody tr {{
        transition: background-color 0.12s ease;
    }}

    tbody tr:hover td {{
        background-color: {theme['accent_soft']} !important;
    }}

    /* Theme Selector Fix */
    div[data-baseweb="select"] > div {{
        background-color: {theme['bg_secondary']} !important;
        border: 1px solid {theme['border']} !important;
        color: {theme['text_primary']} !important;
        border-radius: 8px !important;
    }}

    div[data-baseweb="select"] > div:hover {{
        border-color: {theme['accent']} !important;
    }}

    div[data-baseweb="popover"] ul {{
        background-color: {theme['bg_primary']} !important;
        border: 1px solid {theme['border']} !important;
    }}

    div[data-baseweb="popover"] li:hover {{
        background-color: {theme['bg_secondary']} !important;
    }}

    /* ---- File Uploader ---- */
    [data-testid="stFileUploaderDropzone"] {{
        background-color: {theme['bg_secondary']} !important;
        border: 2px dashed {theme['border']} !important;
        outline: none !important;
        box-shadow: none !important;
        border-radius: 12px !important;
        transition: border-color 0.18s ease, background-color 0.18s ease;
    }}

    [data-testid="stFileUploaderDropzone"]:hover {{
        border-color: {theme['accent']} !important;
        background-color: {theme['accent_soft']} !important;
    }}

    /* Kill the blue focus ring/outline entirely — dropzone, button, children */
    [data-testid="stFileUploaderDropzone"]:focus,
    [data-testid="stFileUploaderDropzone"]:focus-visible,
    [data-testid="stFileUploaderDropzone"]:focus-within,
    [data-testid="stFileUploaderDropzone"] *:focus,
    [data-testid="stFileUploaderDropzone"] *:focus-visible,
    [data-testid="stFileUploader"] *:focus,
    [data-testid="stFileUploader"] *:focus-visible {{
        outline: none !important;
        box-shadow: none !important;
    }}

    [data-testid="stFileUploaderDropzone"]:focus-within {{
        border: 2px dashed {theme['border']} !important;
    }}

    /* Target the upload button to remove its background */
    [data-testid="stFileUploader"] button {{
        background-color: transparent !important;
        border: 1px solid {theme['border']} !important;
        color: {theme['text_primary']} !important;
        box-shadow: none !important;
        border-radius: 8px !important;
        transition: border-color 0.15s ease, color 0.15s ease;
    }}

    [data-testid="stFileUploader"] button:hover {{
        border-color: {theme['accent']} !important;
        color: {theme['accent']} !important;
    }}

    [data-testid="stFileUploader"] button:focus,
    [data-testid="stFileUploader"] button:focus-visible,
    [data-testid="stFileUploader"] button:active {{
        outline: none !important;
        box-shadow: none !important;
        border: 1px solid {theme['border']} !important;
    }}

    [data-testid="stFileUploader"] section {{
        background-color: transparent !important;
    }}

    [data-testid="stFileUploaderFileName"],
    [data-testid="stFileUploaderFileData"],
    [data-testid="stFileUploaderFileSize"],
    [data-testid="stFileUploaderDropzone"] div,
    [data-testid="stFileUploaderDropzone"] p,
    [data-testid="stFileUploaderDropzone"] span,
    [data-testid="stUploadedFile"],
    [data-testid="stUploadedFile"] > div {{
        color: {theme['text_primary']} !important;
        background-color: {theme['bg_secondary']} !important;
    }}

    [data-testid="stUploadedFile"] {{
        border: 1px solid {theme['border']} !important;
        border-radius: 8px !important;
        margin-bottom: 4px !important;
        transition: border-color 0.15s ease;
    }}

    [data-testid="stUploadedFile"]:hover {{
        border-color: {theme['accent']} !important;
    }}

    /* Native Streamlit upload-cloud icon kept as-is — no custom
       arrow overlay (a previous ::before '↑' duplicated it,
       showing two icons side by side). */

    /* ---- Main Action Buttons ----
       Lighter/thinner by design: a soft-tinted fill with an accent
       border (not a heavy solid block), slimmer padding, and a
       normal (not bold) weight. Hover deepens to the full accent
       color so the button still reads clearly as the primary
       action, it just doesn't shout at rest.

       :not([data-testid="stFileUploader"] *) excludes the uploader's
       internal "Browse files" button, which Streamlit also renders
       as data-testid="stBaseButton-secondary" — without this
       exclusion, the upload button gets the same fill as Run
       Inspection / Run Batch. */
    button[data-testid="stBaseButton-secondary"]:not([data-testid="stFileUploader"] button),
    button[data-testid="stBaseButton-primary"]:not([data-testid="stFileUploader"] button) {{
        background-color: {theme['accent_soft']} !important;
        color: {theme['accent']} !important;
        border: 1px solid {theme['accent']}55 !important;
        border-radius: 7px !important;
        font-weight: 500 !important;
        padding: 0.35rem 0.9rem !important;
        transition: background-color 0.15s ease, color 0.15s ease, transform 0.1s ease, box-shadow 0.15s ease;
        box-shadow: none !important;
    }}

    button[data-testid="stBaseButton-secondary"]:not([data-testid="stFileUploader"] button):hover,
    button[data-testid="stBaseButton-primary"]:not([data-testid="stFileUploader"] button):hover {{
        background-color: {theme['accent']} !important;
        color: white !important;
        border-color: {theme['accent']} !important;
        box-shadow: 0 2px 8px {theme['accent_soft']} !important;
    }}

    button[data-testid="stBaseButton-secondary"]:not([data-testid="stFileUploader"] button):active,
    button[data-testid="stBaseButton-primary"]:not([data-testid="stFileUploader"] button):active {{
        transform: translateY(1px);
    }}

    /* Disabled state should look inert, not styled as a clickable accent button */
    button[data-testid="stBaseButton-secondary"]:not([data-testid="stFileUploader"] button):disabled,
    button[data-testid="stBaseButton-primary"]:not([data-testid="stFileUploader"] button):disabled {{
        background-color: transparent !important;
        color: {theme['text_muted']} !important;
        border-color: {theme['border']} !important;
        box-shadow: none !important;
        cursor: not-allowed !important;
    }}

    /* File uploader's "Browse files" button — explicitly neutral,
       never accent-filled, so it never shows the blue rectangle */
    [data-testid="stFileUploader"] button[data-testid^="stBaseButton"] {{
        background-color: transparent !important;
        color: {theme['text_primary']} !important;
        border: 1px solid {theme['border']} !important;
        box-shadow: none !important;
    }}

    [data-testid="stFileUploader"] button[data-testid^="stBaseButton"]:hover {{
        border-color: {theme['accent']} !important;
        color: {theme['accent']} !important;
        background-color: transparent !important;
        box-shadow: none !important;
    }}

    [data-testid="stFileUploader"] button[data-testid^="stBaseButton"]:focus,
    [data-testid="stFileUploader"] button[data-testid^="stBaseButton"]:focus-visible,
    [data-testid="stFileUploader"] button[data-testid^="stBaseButton"]:active {{
        outline: none !important;
        box-shadow: none !important;
        border-color: {theme['border']} !important;
    }}

    /* ---- Remove Button (✕) ---- */
    .remove-btn-container {{
        position: relative;
        display: inline-block;
        width: 100%;
    }}

    .stButton > button[key^="remove_"] {{
        position: absolute !important;
        top: 5px !important;
        right: 5px !important;
        z-index: 1000 !important;
        background-color: rgba(239, 68, 68, 0.8) !important;
        color: white !important;
        border-radius: 50% !important;
        width: 22px !important;
        height: 22px !important;
        min-width: 22px !important;
        min-height: 22px !important;
        padding: 0 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        border: none !important;
        font-size: 12px !important;
        font-weight: 300 !important;
        line-height: 1 !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.2) !important;
        transition: background-color 0.15s ease, transform 0.15s ease;
    }}

    .stButton > button[key^="remove_"]:hover {{
        background-color: #ef4444 !important;
        transform: scale(1.08);
    }}

    /* Streamlit's built-in image hover overlay (the zoom-icon toolbar
       that appears over st.image() on hover) sits on top of our
       caption div with a light/semi-transparent background in light
       theme — making the muted-gray caption text blend into it
       (white-on-near-white). Giving the caption its own solid
       background + higher stacking order keeps it legible
       regardless of what Streamlit renders behind it. */
    .image-caption {{
        background-color: {theme['bg_secondary']} !important;
        color: {theme['text_muted']} !important;
        font-size: 0.7rem !important;
        text-align: center;
        margin-top: 4px;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
        position: relative;
        z-index: 5;
        border-radius: 4px;
        padding: 2px 6px;
    }}

    /* ---- Verdict Box ---- */
    .verdict-box {{
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: 800;
        margin: 16px 0;
        letter-spacing: -0.01em;
        position: relative;
    }}

    .verdict-pass {{
        background: {theme['success_bg']} !important;
        border: 2px solid {theme['success']} !important;
        color: {theme['success']} !important;
        box-shadow: 0 0 0 4px rgba(16, 185, 129, 0.08);
    }}

    .verdict-fail {{
        background: {theme['error_bg']} !important;
        border: 2px solid {theme['error']} !important;
        color: {theme['error']} !important;
        box-shadow: 0 0 0 4px rgba(239, 68, 68, 0.08);
    }}

    /* ---- Tabs ---- */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 4px;
        border-bottom: 1px solid {theme['border']};
    }}

    .stTabs [data-baseweb="tab"] {{
        color: {theme['text_muted']} !important;
        font-weight: 600 !important;
        padding: 8px 4px !important;
    }}

    .stTabs [aria-selected="true"] {{
        color: {theme['text_primary']} !important;
    }}

    .stTabs [data-baseweb="tab-highlight"] {{
        background-color: {theme['accent']} !important;
        height: 2.5px !important;
    }}

    /* ---- Progress bar ---- */
    [data-testid="stProgress"] > div > div {{
        background-color: {theme['accent']} !important;
    }}

    /* ---- Image hover / fullscreen overlay ----
       st.image() shows a toolbar (fullscreen-expand icon) on hover
       that Streamlit hardcodes with a near-black backdrop regardless
       of app theme. In light theme this reads as a solid black
       block with black-on-black icon/label — illegible. Recoloring
       the toolbar's own background + icon + tooltip label from
       {{theme}} makes it correctly invert per theme instead of
       always being dark. */
    [data-testid="stElementToolbar"] {{
        background-color: transparent !important;
    }}

    [data-testid="stElementToolbarButtonContainer"] {{
        background-color: transparent !important;
        background: transparent !important;
        box-shadow: none !important;
        border: none !important;
    }}

    [data-testid="stElementToolbar"] button[data-testid^="stBaseButton"],
    [data-testid="stElementToolbar"] button[data-testid^="stBaseButton"]:hover,
    [data-testid="stElementToolbar"] button[data-testid^="stBaseButton"]:focus,
    [data-testid="stElementToolbar"] button[data-testid^="stBaseButton"]:focus-visible,
    [data-testid="stElementToolbar"] button[data-testid^="stBaseButton"]:active {{
        background-color: {theme['bg_secondary']} !important;
        border: 1px solid {theme['border']} !important;
        border-radius: 50% !important;
        box-shadow: none !important;
        outline: none !important;
    }}

    [data-testid="stElementToolbarButtonIcon"] svg {{
        fill: {theme['text_primary']} !important;
    }}

    /* The "Fullscreen" tooltip label that appears next to the icon
       — Streamlit renders this via baseweb's tooltip popover, which
       also ships a hardcoded dark backdrop. Make the backdrop
       transparent and give the text itself a theme-colored outline
       (text-shadow trick) so it stays readable over the image in
       both themes without needing a solid box behind it. */
    div[data-baseweb="tooltip"],
    div[data-baseweb="tooltip"] > div {{
        background-color: transparent !important;
        background: transparent !important;
        box-shadow: none !important;
        border: none !important;
    }}

    div[data-baseweb="tooltip"] {{
        color: {theme['text_primary']} !important;
        font-weight: 700 !important;
        text-shadow:
            0 0 3px {theme['bg_primary']}, 0 0 3px {theme['bg_primary']},
            0 0 5px {theme['bg_primary']}, 0 0 5px {theme['bg_primary']};
    }}

    /* ---- Footer ---- */
    .footer-container {{
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 40px 0;
        margin-top: 60px;
        border-top: 1px solid {theme['border']};
        text-align: center;
    }}

    .footer-name {{
        font-size: 1.1rem;
        font-weight: 800;
        color: {theme['text_primary']} !important;
        margin: 4px 0;
    }}

    header {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

inject_css()

# =========================================================
# Backend Connectivity Check
# =========================================================
api_ok, health_data = check_api_health()

# =========================================================
# UI Layout
# =========================================================

# 1. Navbar
col_brand, col_theme = st.columns([4, 1])
with col_brand:
    st.markdown("""
    <div class="brand-container">
        <div class="brand-title" style="font-size: 1.75rem; font-weight: 800;">Fabric QC Platform</div>
        <div class="brand-sub">Industrial Grade Automated Defect Detection</div>
    </div>
    """, unsafe_allow_html=True)

with col_theme:
    selected_theme = st.selectbox(
        "Theme", ["Dark", "Light"],
        index=["Dark", "Light"].index(st.session_state.theme_mode),
        label_visibility="collapsed"
    )
    if selected_theme != st.session_state.theme_mode:
        st.session_state.theme_mode = selected_theme
        st.rerun()

if not api_ok:
    st.error(
        f"⚠️ Cannot reach the detection API at `{API_URL}`. "
        "Start the backend first — `uvicorn api.main:app --reload` "
        "(or, in Docker Compose, make sure the `api` service is healthy) — then refresh."
    )
    st.stop()

# 2. Horizontal Legend
legend_items = "".join([
    f'<div class="legend-item" style="--dot-glow:{meta["color"]}55;">'
    f'<div class="legend-dot" style="background:{meta["color"]};"></div>'
    f'<div class="legend-text">{name}</div></div>'
    for name, meta in DEFECT_METADATA.items()
])
legend_html = f'<div class="legend-bar">{legend_items}</div>'
st.markdown(legend_html, unsafe_allow_html=True)

st.write("---")

# 3. KPI Row
history = st.session_state.history
total = len(history)
passed = sum(1 for x in history if x["verdict"] == "PASS")
rejected = total - passed
rate = f"{(passed/total*100):.1f}%" if total > 0 else "0%"

k1, k2, k3, k4 = st.columns(4)
for col, label, val in zip([k1, k2, k3, k4], ["Checked", "Passed", "Rejected", "Pass Rate"], [total, passed, rejected, rate]):
    with col:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{val}</div>
        </div>
        """, unsafe_allow_html=True)

st.write("")

# 4. Main Tabs
tab1, tab2, tab3 = st.tabs(["Single Scan", "Batch QC", "History"])

# =========================
# TAB 1 — SINGLE SCAN
# =========================
with tab1:
    l, r = st.columns([1, 1.2], gap="large")

    with l:
        st.markdown('<div class="section-heading"><span class="dot"></span>Upload Fabric Image</div>', unsafe_allow_html=True)

        uploaded = st.file_uploader(
            "Upload",
            type=["jpg", "png", "jpeg"],
            label_visibility="collapsed",
            key=f"uploader_{st.session_state.uploader_key}"
        )

        if uploaded:
            input_image = Image.open(uploaded)

            # Clean wrapper (prevents focus styling artifacts)
            st.markdown('<div style="position:relative;">', unsafe_allow_html=True)

            if st.button("✕", key="remove_single_img", help="Remove this image"):
                st.session_state.uploader_key += 1
                st.session_state.single_result = None
                st.rerun()

            st.image(input_image, use_container_width=True)

            st.markdown('</div>', unsafe_allow_html=True)

            if st.button("Run Inspection", type="primary", key="run_single", use_container_width=True):
                with st.spinner("Analyzing with YOLO..."):
                    try:
                        file_bytes = uploaded.getvalue()
                        # Two calls: one for JSON detections, one for the
                        # pre-rendered annotated PNG. The /predict/annotated
                        # stream is consumed by the first call, so we re-send
                        # the same bytes for /predict (cheap — bytes are
                        # already in memory, no re-upload from disk).
                        annotated_bytes = call_predict_annotated(file_bytes, uploaded.name, uploaded.type)
                        result = call_predict(file_bytes, uploaded.name, uploaded.type)
                    except requests.exceptions.RequestException as e:
                        st.error(f"Request to API failed: {e}")
                        st.stop()

                    defects = result["defects"]
                    verdict = result["verdict"]

                    st.session_state.single_result = {
                        "filename": result["filename"],
                        "verdict": verdict,
                        "count": result["count"],
                        "defects": defects,
                        "annotated": annotated_bytes
                    }

                    st.session_state.history.append({
                        "time": datetime.now().strftime("%H:%M"),
                        "filename": result["filename"],
                        "verdict": verdict,
                        "count": result["count"]
                    })

    with r:
        st.markdown('<div class="section-heading"><span class="dot"></span>Detection Result</div>', unsafe_allow_html=True)

        if st.session_state.single_result:
            res = st.session_state.single_result

            st.image(res["annotated"], use_container_width=True)

            cls = "verdict-pass" if res["verdict"] == "PASS" else "verdict-fail"
            label = "✓ QUALITY PASSED" if res["verdict"] == "PASS" else f"✗ REJECTED ({res['count']} DEFECTS)"

            st.markdown(
                f'<div class="verdict-box {cls}">{label}</div>',
                unsafe_allow_html=True
            )

            if res["count"] > 0:
                st.markdown("#### Defect Breakdown")
                df = pd.DataFrame(res["defects"])
                st.table(df)
                st.download_button(
                    "Download CSV",
                    df.to_csv(index=False),
                    "defects.csv",
                    "text/csv"
                )

            # Standard post-result action: clear the current image +
            # result and reset the uploader (bumping uploader_key
            # forces Streamlit to remount file_uploader with a fresh,
            # empty widget instance — st.file_uploader has no public
            # "clear" API, so a new widget key is the standard pattern).
            st.button(
                "↻ New Scan",
                key="new_scan_single",
                use_container_width=True,
                on_click=start_new_single_scan,
            )
        else:
            st.info("Upload an image to start the automated QC process.")


# =========================
# TAB 2 — BATCH QC
# =========================
with tab2:
    st.markdown('<div class="section-heading"><span class="dot"></span>Batch Quality Control</div>', unsafe_allow_html=True)

    new_batch_files = st.file_uploader(
        "Upload Multiple Images",
        accept_multiple_files=True,
        type=["jpg", "png", "jpeg"],
        label_visibility="collapsed",
        key=f"batch_up_{st.session_state.batch_key}"
    )

    if new_batch_files:
        for f in new_batch_files:
            if f.name not in [x.name for x in st.session_state.batch_files_list]:
                st.session_state.batch_files_list.append(f)

        st.session_state.batch_key += 1
        st.rerun()

    if st.session_state.batch_files_list:
        head_l, head_r = st.columns([3, 1])
        with head_l:
            st.markdown(f"#### Uploaded Images ({len(st.session_state.batch_files_list)})")
        with head_r:
            st.button(
                "🗑 Clear All",
                key="clear_all_batch",
                use_container_width=True,
                on_click=clear_all_batch,
            )

        cols = st.columns(4)
        files_to_remove = []

        for idx, file in enumerate(st.session_state.batch_files_list):
            with cols[idx % 4]:

                if st.button("✕", key=f"remove_batch_{idx}", help=f"Remove {file.name}"):
                    files_to_remove.append(idx)

                img = Image.open(file)
                st.image(img, use_container_width=True)

                short_name = file.name[:15] + "..." if len(file.name) > 15 else file.name
                st.markdown(
                    f'<div class="image-caption">{short_name}</div>',
                    unsafe_allow_html=True
                )

        if files_to_remove:
            for idx in sorted(files_to_remove, reverse=True):
                st.session_state.batch_files_list.pop(idx)
            st.rerun()

        if st.button("Run Batch Inspection", type="primary", use_container_width=True):
            with st.spinner(f"Analyzing {len(st.session_state.batch_files_list)} images..."):
                try:
                    files_payload = [
                        ("files", (f.name, f.getvalue(), f.type))
                        for f in st.session_state.batch_files_list
                    ]
                    st.session_state.batch_result = call_predict_batch(files_payload)
                except requests.exceptions.RequestException as e:
                    st.error(f"Batch request to API failed: {e}")
                    st.stop()

            for r in st.session_state.batch_result["results"]:
                st.session_state.history.append({
                    "time": datetime.now().strftime("%H:%M"),
                    "filename": r["filename"],
                    "verdict": r["verdict"],
                    "count": r["count"]
                })

        # Rendered from session_state (not the just-computed local
        # var) so the results table + actions survive reruns
        # triggered by other widgets on this tab (e.g. removing one
        # image), instead of vanishing until "Run Batch Inspection"
        # is clicked again.
        if st.session_state.batch_result:
            batch_result = st.session_state.batch_result
            summary = batch_result["summary"]
            st.success(f"Successfully processed {summary['total_checked']} images.")

            df_batch = pd.DataFrame([
                {
                    "filename": r["filename"],
                    "verdict": r["verdict"],
                    "defects_found": r["count"],
                }
                for r in batch_result["results"]
            ])
            st.table(df_batch)

            dl_col, new_col = st.columns(2)
            with dl_col:
                st.download_button(
                    "Download Batch Results",
                    df_batch.to_csv(index=False),
                    "batch_results.csv",
                    "text/csv",
                    use_container_width=True,
                )
            with new_col:
                st.button(
                    "↻ New Batch",
                    key="new_batch",
                    use_container_width=True,
                    on_click=clear_all_batch,
                )


# =========================
# TAB 3 — HISTORY
# =========================
with tab3:
    st.markdown('<div class="section-heading"><span class="dot"></span>Session History</div>', unsafe_allow_html=True)

    if history:
        df_history = pd.DataFrame(history)
        st.table(df_history)

        c1, c2 = st.columns(2)

        with c1:
            st.download_button(
                "Download History",
                df_history.to_csv(index=False),
                "history.csv",
                "text/csv"
            )

        with c2:
            if st.button("Clear Logs"):
                st.session_state.history = []
                st.rerun()
    else:
        st.info("No inspection logs available.")

# 5. Professional Footer (Centered Bottom Middle)
st.markdown("""
<div class="footer-container">
    <div class="footer-text">Developed by</div>
    <div class="footer-name">Ashikur Rahman</div>
    <div class="footer-text" style="font-size: 0.75rem; margin-top: 8px;">Fabric QC Platform v4.6 • Industrial Textile Solutions</div>
</div>
""", unsafe_allow_html=True)