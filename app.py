
from __future__ import annotations

import hashlib
import os
import re
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from india_comparable import estimate_india_comparable

# Optional interactive charts
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    px = None
    go = None
    PLOTLY_AVAILABLE = False


# ============================================================
# 1. APPLICATION CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="EstateIQ | Real Estate Intelligence",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent

PAGE_NAMES = [
    "Overview",
    "US Home Estimate",
    "India Property Estimate",
    "Room Photo Checker",
    "Demand Forecast",
    "Review Sentiment",
    "Explore Datasets",
]

PAGE_ICONS = {
    "Overview": "🏠",
    "US Home Estimate": "💵",
    "India Property Estimate": "📍",
    "Room Photo Checker": "🖼️",
    "Demand Forecast": "📈",
    "Review Sentiment": "💬",
    "Explore Datasets": "🗂️",
}

# ============================================================
# 2. CONSISTENT LIGHT THEME
# ============================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Manrope:wght@500;600;700;800&display=swap');

    :root {
        --ink: #15213B;
        --muted: #68758D;
        --line: #E4EAF3;
        --primary: #5865E8;
        --primary-dark: #414CC8;
        --teal: #0F9D9A;
        --surface: #FFFFFF;
        --canvas: #F3F6FB;
        --navy: #101B34;
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif !important;
    }

    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    [data-testid="stMainBlockContainer"] {
        background: linear-gradient(180deg, #F1F5FB 0%, #F7F9FC 55%, #F3F6FB 100%) !important;
        color: var(--ink) !important;
    }

    .block-container {
        max-width: 1440px;
        padding-top: 1.55rem;
        padding-bottom: 3.2rem;
    }

    h1, h2, h3, h4, h5, h6 {
        color: var(--ink) !important;
        font-family: 'Manrope', sans-serif !important;
        font-weight: 800 !important;
        letter-spacing: -0.035em;
    }
    h1 { font-size: clamp(2rem, 3vw, 2.75rem) !important; }
    h2 { font-size: clamp(1.4rem, 2vw, 1.8rem) !important; margin-top: 1.4rem !important; }
    h3 { font-size: 1.16rem !important; }

    p, li, label,
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stCaptionContainer"],
    [data-testid="stWidgetLabel"] p,
    [data-testid="stText"] {
        color: #35435B;
    }
    [data-testid="stWidgetLabel"] p {
        font-weight: 700 !important;
        color: #293750 !important;
    }
    [data-testid="stCaptionContainer"] { color: #7B879B !important; }

    /* Sidebar: dark, high-contrast navigation */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #101A33 0%, #172B4D 54%, #133B52 100%) !important;
        border-right: 1px solid rgba(255,255,255,.08) !important;
    }
    [data-testid="stSidebar"] > div:first-child { background: transparent !important; }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: #F5F8FF !important;
    }
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
        color: #B8C7DE !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] > label {
        color: #DCE6F5 !important;
        font-size: .76rem !important;
        font-weight: 800 !important;
        letter-spacing: .12em;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        border-radius: 12px;
        padding: 9px 11px;
        transition: background .18s ease, transform .18s ease;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
        background: rgba(255,255,255,.10) !important;
        transform: translateX(2px);
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
        background: linear-gradient(100deg, rgba(99,102,241,.34), rgba(20,184,166,.20)) !important;
        border: 1px solid rgba(165,180,252,.25);
    }
    [data-testid="stSidebar"] hr { border-color: rgba(255,255,255,.13) !important; }

    /* Page introduction */
    .page-heading {
        padding: 4px 2px 20px 2px;
        margin-bottom: 12px;
    }
    .page-heading h1 {
        margin: 7px 0 5px 0 !important;
        line-height: 1.13;
    }
    .page-heading p {
        color: var(--muted) !important;
        font-size: 1rem;
        max-width: 860px;
        line-height: 1.65;
        margin: 0;
    }
    .eyebrow {
        color: #5865D9 !important;
        font-weight: 800;
        font-size: .72rem;
        letter-spacing: .17em;
        text-transform: uppercase;
    }

    /* Overview hero */
    .hero {
        position: relative;
        overflow: hidden;
        background: linear-gradient(118deg, #101A33 0%, #263F79 52%, #087F83 100%);
        border-radius: 26px;
        padding: clamp(28px, 4vw, 48px);
        margin: 2px 0 28px 0;
        box-shadow: 0 18px 44px rgba(20, 38, 76, .18);
    }
    .hero:after {
        content: "";
        position: absolute;
        width: 340px;
        height: 340px;
        border-radius: 50%;
        right: -90px;
        top: -150px;
        border: 1px solid rgba(255,255,255,.17);
        box-shadow: 0 0 0 35px rgba(255,255,255,.035), 0 0 0 72px rgba(255,255,255,.025);
        pointer-events: none;
    }
    .hero h1 {
        color: #FFFFFF !important;
        font-size: clamp(2.25rem, 4.4vw, 3.45rem) !important;
        line-height: 1.09;
        margin: 12px 0 12px 0 !important;
        max-width: 800px;
    }
    .hero p {
        color: #DDE9FA !important;
        font-size: 1.03rem;
        line-height: 1.75;
        max-width: 760px;
        margin-bottom: 0;
    }
    .hero-kicker {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        color: #D7E7FF !important;
        background: rgba(255,255,255,.11);
        border: 1px solid rgba(255,255,255,.16);
        border-radius: 999px;
        padding: 7px 12px;
        font-size: .72rem;
        font-weight: 800;
        letter-spacing: .13em;
    }

    /* Feature cards */
    .module-card, .soft-card {
        background: rgba(255,255,255,.96);
        border: 1px solid #E2E9F3;
        border-radius: 19px;
        padding: 21px;
        box-shadow: 0 7px 24px rgba(24, 42, 76, .055);
    }
    .module-card {
        min-height: 190px;
        margin-bottom: 10px;
        transition: transform .2s ease, box-shadow .2s ease, border-color .2s ease;
    }
    .module-card:hover {
        transform: translateY(-4px);
        border-color: #B9C6F5;
        box-shadow: 0 14px 32px rgba(30, 52, 92, .11);
    }
    .module-card h3 { margin: 7px 0 8px 0 !important; font-size: 1.08rem !important; }
    .module-card p { color: #6A7890 !important; line-height: 1.58; font-size: .91rem; }
    .module-icon {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 46px;
        height: 46px;
        border-radius: 14px;
        background: linear-gradient(135deg, #EEF1FF, #E4F8F5);
        font-size: 1.55rem;
    }
    .soft-card { padding: 23px 25px; }
    .result-label {
        color: #718098 !important;
        text-transform: uppercase;
        letter-spacing: .13em;
        font-size: .69rem;
        font-weight: 800;
    }
    .result-value {
        color: #172744 !important;
        font-family: 'Manrope', sans-serif;
        font-size: clamp(1.8rem, 3vw, 2.5rem);
        font-weight: 800;
        letter-spacing: -.04em;
        margin-top: 7px;
        line-height: 1.2;
    }

    /* Callouts */
    .callout, .warning-callout, .success-callout {
        border-radius: 14px;
        padding: 14px 17px;
        line-height: 1.65;
        margin: 12px 0 18px 0;
        font-size: .94rem;
    }
    .callout { background: #EEF3FF; border: 1px solid #D4DFFF; color: #2D4B8F; }
    .warning-callout { background: #FFF8E9; border: 1px solid #F6DDA6; color: #795018; }
    .success-callout { background: #EAF8F3; border: 1px solid #BCE8D7; color: #17634D; }

    /* Metrics and charts */
    [data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #E3EAF3;
        border-radius: 16px;
        padding: 16px 17px;
        box-shadow: 0 5px 18px rgba(24, 42, 76, .045);
    }
    [data-testid="stMetricLabel"] { color: #748198 !important; font-weight: 700 !important; }
    [data-testid="stMetricValue"] { color: #182744 !important; font-family: 'Manrope',sans-serif !important; font-weight: 800 !important; }
    [data-testid="stMetricDelta"] { font-weight: 700 !important; }
    [data-testid="stPlotlyChart"], [data-testid="stVegaLiteChart"] {
        background: #FFFFFF;
        border: 1px solid #E4EAF3;
        border-radius: 17px;
        padding: 7px;
        box-shadow: 0 5px 18px rgba(24, 42, 76, .04);
    }

    /* Inputs */
    [data-testid="stTextInput"] input,
    [data-testid="stNumberInput"] input,
    [data-testid="stTextArea"] textarea,
    [data-testid="stDateInput"] input {
        background: #FFFFFF !important;
        color: #17233A !important;
        -webkit-text-fill-color: #17233A !important;
        border: 1px solid #CDD7E5 !important;
        border-radius: 11px !important;
        caret-color: #17233A !important;
    }
    [data-testid="stTextInput"] input:focus,
    [data-testid="stNumberInput"] input:focus,
    [data-testid="stTextArea"] textarea:focus {
        border-color: #7885EE !important;
        box-shadow: 0 0 0 3px rgba(88,101,232,.12) !important;
    }
    [data-testid="stSelectbox"] [data-baseweb="select"] > div,
    [data-testid="stMultiSelect"] [data-baseweb="select"] > div {
        background: #FFFFFF !important;
        color: #17233A !important;
        border: 1px solid #CDD7E5 !important;
        border-radius: 11px !important;
        min-height: 42px;
    }
    [data-testid="stSelectbox"] *, [data-testid="stMultiSelect"] * { color: #17233A !important; }
    [data-baseweb="popover"], [data-baseweb="menu"], [role="listbox"], [role="option"] {
        background: #FFFFFF !important;
        color: #17233A !important;
    }
    [data-testid="stNumberInput"] button {
        background: #EEF1FF !important;
        color: #263A68 !important;
        border-color: #D3DBEA !important;
    }
    [data-testid="stFileUploader"] {
        background: #FFFFFF;
        border: 1.5px dashed #9EADD0;
        border-radius: 16px;
        padding: 12px;
    }
    [data-testid="stFileUploader"] section { background: #F8FAFE !important; border-radius: 12px; }
    [data-testid="stForm"] {
        background: rgba(255,255,255,.94);
        border: 1px solid #E3EAF3;
        border-radius: 19px;
        padding: 21px;
        box-shadow: 0 6px 22px rgba(24,42,76,.045);
    }

    /* Buttons, tabs, tables and expanders */
    .stButton > button, .stFormSubmitButton > button, .stDownloadButton > button {
        background: linear-gradient(100deg, #5865E8 0%, #4776D0 55%, #0D9D9A 100%) !important;
        color: #FFFFFF !important;
        border: 0 !important;
        border-radius: 11px !important;
        font-weight: 800 !important;
        min-height: 43px;
        box-shadow: 0 6px 15px rgba(73, 94, 202, .16);
        transition: transform .18s ease, filter .18s ease, box-shadow .18s ease;
    }
    .stButton > button *, .stFormSubmitButton > button *, .stDownloadButton > button * { color: #FFFFFF !important; }
    .stButton > button:hover, .stFormSubmitButton > button:hover, .stDownloadButton > button:hover {
        filter: brightness(1.06);
        transform: translateY(-1px);
        box-shadow: 0 9px 19px rgba(73, 94, 202, .23);
    }
    .stButton > button:disabled { opacity: .48; transform: none; box-shadow: none; }
    [data-testid="stTabs"] [role="tablist"] { gap: 8px; border-bottom: 1px solid #DFE6F0; }
    [data-testid="stTabs"] button[role="tab"] {
        border-radius: 10px 10px 0 0;
        color: #65728A !important;
        font-weight: 750 !important;
        padding: 10px 15px;
    }
    [data-testid="stTabs"] button[aria-selected="true"] {
        color: #4E5BD4 !important;
        background: #EEF1FF !important;
    }
    [data-testid="stDataFrame"], [data-testid="stTable"] {
        border: 1px solid #E2E9F3;
        border-radius: 14px;
        overflow: hidden;
    }
    [data-testid="stExpander"] {
        background: rgba(255,255,255,.88);
        border: 1px solid #E2E9F3;
        border-radius: 13px;
    }
    [data-testid="stAlert"] { border-radius: 13px !important; }
    [data-testid="stAlert"] p { color: #25344D !important; }
    .muted { color: var(--muted) !important; line-height: 1.7; }
    hr { border-color: #E3EAF3 !important; }
    #MainMenu, footer { visibility: hidden; }

    @media (max-width: 760px) {
        .block-container { padding-top: 1rem; padding-left: 1rem; padding-right: 1rem; }
        .hero { border-radius: 20px; padding: 26px 22px; }
        .hero h1 { font-size: 2.15rem !important; }
        .module-card { min-height: 0; }
        [data-testid="stForm"] { padding: 15px; }
    }
    </style>

    """,
    unsafe_allow_html=True,
)


# ============================================================
# 3. PROJECT FILE LOCATION
# ============================================================

def project_roots() -> list[Path]:
    """Support both the original project folder and a bundled folder."""
    candidates = [
        ROOT,
        ROOT / "EstateIQ_Real_Estate_Intelligence",
        ROOT.parent,
    ]

    roots = []
    seen = set()

    for path in candidates:
        if path.exists():
            key = str(path.resolve()).casefold()
            if key not in seen:
                roots.append(path)
                seen.add(key)

    return roots


def normalized_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.casefold())


DIRECT_FILE_PATHS = {
    "house_prices.csv": Path("Property prize prediction/data/House Prices.csv"),
    "location_house_data.csv": Path("Location Based Price Prediction using ANN/data/house_data.csv"),
    "property_demand_forecasting_synthetic.csv": Path("Property demand forecast/data/property_demand_forecasting_synthetic.csv"),
    "house_price_ann.keras": Path("Property prize prediction/house_price_ann.keras"),
    "house_price_preprocessor.joblib": Path("Property prize prediction/models/house_price_preprocessor.joblib"),
    "location_based_price_ann.keras": Path("Location Based Price Prediction using ANN/location_based_price_ann.keras"),
    "location_price_preprocessor.joblib": Path("Location Based Price Prediction using ANN/models/location_price_preprocessor.joblib"),
    "property_cnn_model.keras": Path("Property Image Classification Using CNN/property_cnn_model.keras"),
    "property_demand_lstm.keras": Path("Property demand forecast/property_demand_lstm.keras"),
    "customer_review_rnn.keras": Path("Customer_Review_Sentiment_Analysis_RNN/customer_review_rnn.keras"),
    "customer_review_tokenizer.json": Path("Customer_Review_Sentiment_Analysis_RNN/customer_review_tokenizer.json"),
}


def find_file(filename: str, aliases: list[str] | None = None) -> Path:
    """Resolve known assets directly, then fall back to a case-insensitive search."""
    direct = DIRECT_FILE_PATHS.get(filename)
    if direct is not None:
        candidate = ROOT / direct
        if candidate.is_file():
            return candidate

    names = {normalized_name(filename)}
    names.update(normalized_name(alias) for alias in (aliases or []))

    skip_dirs = {
        ".venv", "venv", "env", ".git", "__pycache__",
        ".ipynb_checkpoints", "node_modules", "house_room_dataset",
    }

    for root in project_roots():
        for current, dirs, files in os.walk(root):
            dirs[:] = [
                d for d in dirs
                if d.casefold() not in skip_dirs and not d.startswith(".")
            ]
            for file in files:
                if normalized_name(file) in names:
                    return Path(current) / file

    raise FileNotFoundError(
        f"Could not find {filename}. Keep the original model and data folders inside the project folder."
    )

DATA_ALIASES = {
    "house_prices.csv": [
        "House Prices.csv",
        "house_prices.csv",
    ],
    "location_house_data.csv": [
        "house_data.csv",
        "location_house_data.csv",
    ],
    "property_demand_forecasting_synthetic.csv": [
        "property_demand_forecasting_synthetic.csv",
    ],
}

MODEL_ALIASES = {
    "house_price_ann.keras": [
        "house_price_ann.keras",
    ],
    "house_price_preprocessor.joblib": [
        "house_price_preprocessor.joblib",
    ],
    "location_based_price_ann.keras": [
        "location_based_price_ann.keras",
    ],
    "location_price_preprocessor.joblib": [
        "location_price_preprocessor.joblib",
    ],
    "property_cnn_model.keras": [
        "property_cnn_model.keras",
    ],
    "property_demand_lstm.keras": [
        "property_demand_lstm.keras",
    ],
    "customer_review_rnn.keras": [
        "customer_review_rnn.keras",
    ],
    "customer_review_tokenizer.json": [
        "customer_review_tokenizer.json",
    ],
}


def data_file(filename: str) -> Path:
    return find_file(filename, DATA_ALIASES.get(filename, []))


def model_file(filename: str) -> Path:
    return find_file(filename, MODEL_ALIASES.get(filename, []))


# ============================================================
# 4. MODEL AND DATASET LOADERS
# ============================================================

@st.cache_data(show_spinner=False)
def load_house_data() -> pd.DataFrame:
    df = pd.read_csv(data_file("house_prices.csv"))

    required = [
        "Date", "Price", "zipcode", "Yr_built", "Yr_renovated",
        "Lat", "Long",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"House dataset missing columns: {missing}")

    dates = pd.to_datetime(
        df["Date"].astype(str),
        format="%Y%m%dT%H%M%S",
        errors="coerce",
    )

    # Fallback for datasets whose date is already in a standard format.
    if dates.isna().all():
        dates = pd.to_datetime(df["Date"], errors="coerce")

    df["Date"] = dates
    df = df.dropna(subset=["Date", "Price"]).copy()
    df["zipcode"] = df["zipcode"].astype(str)
    df["sale_year"] = df["Date"].dt.year
    df["sale_month"] = df["Date"].dt.month
    df["house_age_at_sale"] = (
        df["sale_year"] - df["Yr_built"]
    ).clip(lower=0)

    df["has_renovation"] = (df["Yr_renovated"] > 0).astype(int)

    df["years_since_renovation"] = np.where(
        df["Yr_renovated"] > 0,
        (df["sale_year"] - df["Yr_renovated"]).clip(lower=0),
        0,
    )

    return df


@st.cache_data(show_spinner=False)
def load_location_data() -> pd.DataFrame:
    df = pd.read_csv(data_file("location_house_data.csv"))

    for col in ["City", "Location"]:
        df[col] = df[col].astype("string").str.strip()

    df["City"] = df["City"].replace("", pd.NA)
    df["Location"] = df["Location"].replace("", pd.NA)

    for col in ["Price", "Area", "No. of Bedrooms"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(
        subset=["City", "Location", "Price", "Area", "No. of Bedrooms"]
    ).copy()

    df["City_Location"] = (
        df["City"].astype(str) + " | " + df["Location"].astype(str)
    )

    return df


@st.cache_data(show_spinner=False)
def load_demand_data() -> pd.DataFrame:
    df = pd.read_csv(data_file("property_demand_forecasting_synthetic.csv"))

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Sales_Count"] = pd.to_numeric(
        df["Sales_Count"], errors="coerce"
    )

    return df.dropna(
        subset=["Date", "Sales_Count"]
    ).sort_values("Date")


def load_keras_model(filename: str):
    """Load once per model file version; replacing a model invalidates the cache."""
    path = model_file(filename)
    stat = path.stat()
    return _load_keras_model_cached(str(path), stat.st_mtime_ns, stat.st_size)


@st.cache_resource(show_spinner="Loading trained AI model...")
def _load_keras_model_cached(path_text: str, modified_ns: int, file_size: int):
    import tensorflow as tf

    return tf.keras.models.load_model(path_text, compile=False)


def load_preprocessor(filename: str):
    path = model_file(filename)
    stat = path.stat()
    return _load_preprocessor_cached(str(path), stat.st_mtime_ns, stat.st_size)


@st.cache_resource(show_spinner="Loading fitted preprocessing pipeline...")
def _load_preprocessor_cached(path_text: str, modified_ns: int, file_size: int):
    import joblib

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return joblib.load(path_text)


def predict_array(model, values: Any) -> np.ndarray:
    """Run a small inference batch directly, avoiding predict()'s tf.data overhead."""
    batch = np.asarray(values)
    output = model(batch, training=False)
    if hasattr(output, "numpy"):
        output = output.numpy()
    return np.asarray(output)


# ============================================================
# 5. COMMON HELPER FUNCTIONS
# ============================================================

def money_inr(value: float) -> str:
    if not np.isfinite(value):
        return "Not available"

    if abs(value) >= 10_000_000:
        return f"₹{value / 10_000_000:.2f} Cr"

    if abs(value) >= 100_000:
        return f"₹{value / 100_000:.2f} L"

    return f"₹{value:,.0f}"


def money_usd(value: float) -> str:
    return f"${value:,.0f}" if np.isfinite(value) else "Not available"


def clean_review(text: Any) -> str:
    text = str(text).lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def amenity_label(value: str) -> str:
    """Turn dataset feature names into readable labels for the form."""
    special = {
        "24X7Security": "24×7 Security",
        "Children'splayarea": "Children's Play Area",
        "Gasconnection": "Gas Connection",
        "Wifi": "Wi-Fi",
        "AC": "Air Conditioning",
        "BED": "Bed",
        "TV": "Television",
    }
    if value in special:
        return special[value]
    value = value.replace("_", " ")
    value = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", value)
    return value.strip()


def page_title(eyebrow: str, title: str, description: str):
    st.markdown(
        f"""
        <div class="page-heading">
            <div class="eyebrow">{eyebrow}</div>
            <h1>{title}</h1>
            <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_strip(items):
    columns = st.columns(len(items))

    for col, (label, value, delta) in zip(columns, items):
        col.metric(label, value, delta)


def show_model_error(action: str, exc: Exception):
    st.error(f"Unable to load {action}.")

    st.info(
        "Check that the complete project is extracted and that its Python "
        "requirements are installed in the same environment used to launch Streamlit."
    )

    with st.expander("Technical details"):
        st.code(f"{type(exc).__name__}: {exc}")


PLOT_LAYOUT = {
    "template": "plotly_white",
    "paper_bgcolor": "rgba(255,255,255,0)",
    "plot_bgcolor": "rgba(255,255,255,0)",
    "font": {"family": "Arial", "color": "#17233A"},
    "margin": {"l": 15, "r": 15, "t": 55, "b": 15},
}


def show_plotly(fig):
    if PLOTLY_AVAILABLE:
        fig.update_layout(**PLOT_LAYOUT)
        st.plotly_chart(
            fig,
            use_container_width=True,
            config={"displaylogo": False},
        )


# ============================================================
# 6. PAGE NAVIGATION WITH BACK AND HOME
# ============================================================

if "navigation" not in st.session_state:
    st.session_state["navigation"] = "Overview"

if "page_history" not in st.session_state:
    st.session_state["page_history"] = []

if "_last_page" not in st.session_state:
    st.session_state["_last_page"] = "Overview"


def radio_navigation_changed():
    previous = st.session_state.get("_last_page")
    current = st.session_state.get("navigation")

    if previous and current and previous != current:
        history = st.session_state.get("page_history", [])
        history.append(previous)
        st.session_state["page_history"] = history[-20:]

    st.session_state["_last_page"] = current


def navigate_to(page: str, save_history: bool = True):
    current = st.session_state.get("navigation", "Overview")

    if save_history and current != page:
        history = st.session_state.get("page_history", [])
        history.append(current)
        st.session_state["page_history"] = history[-20:]

    st.session_state["_pending_page"] = page
    st.rerun()


# Apply navigation before creating the radio widget.
pending_page = st.session_state.pop("_pending_page", None)

if pending_page in PAGE_NAMES:
    st.session_state["navigation"] = pending_page
    st.session_state["_last_page"] = pending_page


# ============================================================
# 7. SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown(
        """
        <div style="padding:14px 4px 20px 4px;">
            <div style="display:flex;align-items:center;gap:11px;">
                <div style="width:44px;height:44px;border-radius:14px;display:flex;align-items:center;justify-content:center;background:linear-gradient(135deg,#818CF8,#2DD4BF);font-size:23px;box-shadow:0 8px 20px rgba(0,0,0,.18);">🏙️</div>
                <div>
                    <div style="font-size:25px;font-weight:850;letter-spacing:-1px;color:#FFFFFF;line-height:1.05;">EstateIQ</div>
                    <div style="font-size:10px;letter-spacing:1.8px;color:#B8C7E8;margin-top:5px;">PROPERTY INTELLIGENCE</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Workspace")

    selected_page = st.radio(
        "Navigate",
        PAGE_NAMES,
        format_func=lambda x: f"{PAGE_ICONS[x]}  {x}",
        key="navigation",
        label_visibility="collapsed",
        on_change=radio_navigation_changed,
    )

    st.divider()

    st.markdown("**PROJECT INFORMATION**")
    st.caption("AI-powered student project")
    st.caption("Local trained models")
    st.caption("No external API key required")

    st.divider()

    if not PLOTLY_AVAILABLE:
        st.warning(
            "Plotly is not installed. Basic Streamlit charts are active."
        )
        st.caption(
            "Install Plotly in the same Python environment to enable "
            "interactive charts."
        )

    st.caption(
        "EstateIQ provides educational model outputs, "
        "not professional property valuation advice."
    )


# ============================================================
# 8. INNER PAGE NAVIGATION BAR
# ============================================================

if selected_page != "Overview":
    back_col, home_col, empty_col = st.columns([1, 1, 5])

    with back_col:
        if st.button("← Back", use_container_width=True):
            history = st.session_state.get("page_history", [])

            if history:
                target = history.pop()
                st.session_state["page_history"] = history
            else:
                target = "Overview"

            navigate_to(target, save_history=False)

    with home_col:
        if st.button("⌂ Home", use_container_width=True):
            navigate_to("Overview")


# ============================================================
# 9. OVERVIEW DASHBOARD
# ============================================================

if selected_page == "Overview":

    st.markdown(
        """
        <div class="hero">
            <div style="color:#C7D2FE;font-weight:800;
                        letter-spacing:2px;font-size:12px;">
                WELCOME TO ESTATEIQ
            </div>
            <h1>Make property data<br>easier to understand.</h1>
            <p>
                Explore property prices, analyse room images,
                forecast demand and understand customer feedback
                using machine learning and deep learning.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("## Your AI workspace")

    modules = [
        (
            "💵",
            "US Home Estimate",
            "Estimate a historical US home sale price using property details.",
        ),
        (
            "📍",
            "India Property Estimate",
            "Explore property listings by city, locality, size and amenities.",
        ),
        (
            "🖼️",
            "Room Photo Checker",
            "Upload an interior photograph to identify its room category.",
        ),
        (
            "📈",
            "Demand Forecast",
            "View six modelled months beyond the latest sample date, with a clear historical-data warning.",
        ),
        (
            "💬",
            "Review Sentiment",
            "Classify customer reviews into positive, neutral or negative.",
        ),
        (
            "🗂️",
            "Explore Datasets",
            "Preview datasets, inspect missing values and create charts.",
        ),
    ]

    cols = st.columns(3, gap="medium")

    for i, (icon, title, description) in enumerate(modules):
        with cols[i % 3]:
            st.markdown(
                f"""
                <div class="module-card">
                    <div class="module-icon">{icon}</div>
                    <h3>{title}</h3>
                    <p>{description}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if st.button(
                f"Open {title} →",
                key=f"open_{i}",
                use_container_width=True,
            ):
                navigate_to(title)

    st.markdown("## Getting started")

    steps = st.columns(3)

    with steps[0]:
        st.markdown(
            "**01 · Choose a module**\n\n"
            "Select a tool from the cards or the sidebar."
        )

    with steps[1]:
        st.markdown(
            "**02 · Enter your information**\n\n"
            "Fill in the form, upload a room image or select a dataset."
        )

    with steps[2]:
        st.markdown(
            "**03 · View the result**\n\n"
            "Read the prediction, confidence score, chart or data summary."
        )

    # Dashboard dataset summary
    st.markdown("## Project data snapshot")

    try:
        hdf = load_house_data()
        ldf = load_location_data()
        ddf = load_demand_data()

        city_count = ldf["City"].nunique()
        locality_count = ldf["Location"].nunique()

        metric_strip([
            ("US property records", f"{len(hdf):,}", "Historical sales"),
            ("India listings", f"{len(ldf):,}", "Sample listings"),
            ("Cities", f"{city_count:,}", "India dataset"),
            ("Localities", f"{locality_count:,}", "India dataset"),
        ])

        city_summary = (
            ldf.groupby("City", as_index=False)["Price"]
            .median()
            .sort_values("Price", ascending=False)
        )

        demand_summary = (
            ddf.groupby("Date", as_index=False)["Sales_Count"]
            .sum()
            .sort_values("Date")
        )

        left, right = st.columns(2)

        with left:
            st.markdown("### Median listing price by city")

            if PLOTLY_AVAILABLE:
                fig = px.bar(
                    city_summary,
                    x="City",
                    y="Price",
                    color="Price",
                    title="Sample median property prices",
                )
                show_plotly(fig)
            else:
                st.bar_chart(city_summary.set_index("City")["Price"])

        with right:
            st.markdown("### Monthly sample demand")

            if PLOTLY_AVAILABLE:
                fig = px.line(
                    demand_summary.tail(36),
                    x="Date",
                    y="Sales_Count",
                    markers=True,
                    title="Latest 36 monthly observations",
                )
                show_plotly(fig)
            else:
                st.line_chart(
                    demand_summary.tail(36).set_index("Date")["Sales_Count"]
                )

    except Exception as exc:
        st.warning(
            "Some overview statistics could not be loaded. "
            "You can still open the individual modules."
        )
        with st.expander("Technical details"):
            st.code(str(exc))

    st.markdown(
        """
        <div class="warning-callout">
            <b>Important:</b> This is an academic AI project, not a live
            property marketplace. The US model uses historical 2014–2015
            data, the demand dataset is synthetic, and the Indian
            price model has limited predictive performance.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# 10. US HOME PRICE PREDICTION - ANN
# ============================================================

elif selected_page == "US Home Estimate":

    page_title(
        "PROPERTY PRICE MODULE",
        "Estimate a US Home",
        "Enter the property details to estimate a historical sale price.",
    )

    st.markdown(
        """
        <div class="warning-callout">
            This model uses historical King County, Washington sales
            from 2014–2015. The estimate is in USD and is not a current
            market valuation or an Indian property estimate.
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        house_df = load_house_data()
        model = load_keras_model("house_price_ann.keras")
        preprocessor = load_preprocessor(
            "house_price_preprocessor.joblib"
        )

        zipcode_values = sorted(
            house_df["zipcode"].dropna().astype(str).unique()
        )

        with st.form("us_price_form"):

            st.markdown("### Tell us about the home")

            c1, c2, c3 = st.columns(3)

            with c1:
                bedrooms = st.number_input(
                    "Bedrooms", 1, 10, 3, step=1
                )
                bathrooms = st.number_input(
                    "Bathrooms", 0.0, 8.0, 2.25, step=0.25
                )
                floors = st.selectbox(
                    "Floors", [1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
                    index=2,
                )
                sqft_living = st.number_input(
                    "Interior living area (sq ft)",
                    290, 13540, 1910, step=50,
                )

            with c2:
                sqft_lot = st.number_input(
                    "Lot area (sq ft)", 520, 250000, 7600, step=100
                )
                sqft_above = st.number_input(
                    "Above-ground area (sq ft)",
                    290, 9410, 1560, step=50,
                )
                sqft_basement = st.number_input(
                    "Basement area (sq ft)", 0, 4820, 0, step=50
                )
                condition = st.select_slider(
                    "Property condition",
                    options=[1, 2, 3, 4, 5],
                    value=3,
                )

            with c3:
                grade = st.select_slider(
                    "Construction / grade",
                    options=list(range(1, 14)),
                    value=7,
                )
                waterfront = st.selectbox(
                    "Waterfront", [0, 1],
                    format_func=lambda x: "Yes" if x else "No",
                )
                view = st.selectbox("View score", [0, 1, 2, 3, 4])
                zip_index = (
                    zipcode_values.index("98002")
                    if "98002" in zipcode_values else 0
                )
                zipcode = st.selectbox(
                    "ZIP code", zipcode_values, index=zip_index
                )

            st.markdown("### Additional details")

            c4, c5, c6 = st.columns(3)

            with c4:
                yr_built = st.number_input(
                    "Year built", 1900, 2015, 1975
                )
                renovation_year_options = [0] + sorted(
                    house_df.loc[
                        house_df["Yr_renovated"] > 0, "Yr_renovated"
                    ].astype(int).unique().tolist()
                )
                yr_renovated = st.selectbox(
                    "Year renovated",
                    renovation_year_options,
                    index=0,
                    format_func=lambda value: (
                        "Never renovated" if value == 0 else str(value)
                    ),
                    help="Choose a real calendar year. Never renovated is stored as 0.",
                )
                st.caption("Choose a renovation year between construction and sale, or select Never renovated.")

            with c5:
                sale_year = st.selectbox("Sale year", [2014, 2015], index=1)
                sale_month = st.selectbox(
                    "Sale month",
                    list(range(1, 13)),
                    index=2,
                    format_func=lambda x: pd.Timestamp(
                        year=2000, month=x, day=1
                    ).strftime("%B"),
                )
                sqft_living15 = st.number_input(
                    "Nearby median living area (sq ft)",
                    399, 6210, 1840, step=50,
                )

            with c6:
                sqft_lot15 = st.number_input(
                    "Nearby median lot area (sq ft)",
                    651, 871200, 7620, step=100,
                )

                zip_rows = house_df[
                    house_df["zipcode"] == str(zipcode)
                ]

                lat = float(
                    zip_rows["Lat"].median()
                    if not zip_rows.empty
                    else house_df["Lat"].median()
                )
                long = float(
                    zip_rows["Long"].median()
                    if not zip_rows.empty
                    else house_df["Long"].median()
                )

                st.info(f"Approximate ZIP coordinates: {lat:.4f}, {long:.4f}")

            submitted = st.form_submit_button(
                "Estimate historical property price",
                use_container_width=True,
            )

        if submitted:
            renovation_year = int(yr_renovated)
            built_year = int(yr_built)
            transaction_year = int(sale_year)

            renovation_is_valid = (
                renovation_year == 0
                or (
                    renovation_year >= max(1934, built_year)
                    and renovation_year <= transaction_year
                )
            )

            if not renovation_is_valid:
                st.error(
                    "Invalid renovation year. Enter 0 if the home was never renovated. "
                    f"Otherwise enter a real year from {max(1934, built_year)} "
                    f"to {transaction_year}. A value such as 1 is not a Yes/No flag."
                )
            else:
                row = pd.DataFrame([{
                    "Bedrooms": bedrooms,
                    "Bathrooms": bathrooms,
                    "Sqft_living": sqft_living,
                    "Sqft_lot": sqft_lot,
                    "Floors": floors,
                    "Waterfront": waterfront,
                    "View": view,
                    "Condition": condition,
                    "Grade": grade,
                    "Sqft_above": sqft_above,
                    "Sqft_basement": sqft_basement,
                    "Yr_built": built_year,
                    "Yr_renovated": renovation_year,
                    "zipcode": str(zipcode),
                    "Lat": lat,
                    "Long": long,
                    "Sqft_living15": sqft_living15,
                    "Sqft_lot15": sqft_lot15,
                    "sale_year": transaction_year,
                    "sale_month": sale_month,
                    "house_age_at_sale": max(0, transaction_year - built_year),
                    "has_renovation": int(renovation_year > 0),
                    "years_since_renovation": (
                        max(0, transaction_year - renovation_year)
                        if renovation_year > 0 else 0
                    ),
                }])

                expected_columns = list(preprocessor.feature_names_in_)
                row = row.reindex(columns=expected_columns)
                prepared = preprocessor.transform(row).astype(np.float32)

                expected_features = int(model.input_shape[-1])
                if prepared.shape[1] != expected_features:
                    raise ValueError(
                        f"US preprocessor generated {prepared.shape[1]} features, "
                        f"but the saved ANN expects {expected_features}. "
                        "Use the matching model and preprocessor files."
                    )

                raw_log_prediction = float(
                    predict_array(model, prepared).reshape(-1)[0]
                )
                prediction = float(np.expm1(raw_log_prediction))

                # A large output usually means an invalid/out-of-distribution input.
                # Keep the app from displaying astronomical or nonsensical amounts.
                lower_bound = float(house_df["Price"].quantile(0.001)) * 0.5
                upper_bound = float(house_df["Price"].quantile(0.999)) * 2.0
                if (
                    not np.isfinite(raw_log_prediction)
                    or not np.isfinite(prediction)
                    or prediction < lower_bound
                    or prediction > upper_bound
                ):
                    st.error(
                        "The model output is outside the price range represented in its training data. "
                        "Check the property inputs; no estimate has been displayed."
                    )
                    with st.expander("Technical details"):
                        st.code(
                            f"Raw model output (log price): {raw_log_prediction:.4f}\n"
                            f"Allowed approximate range: ${lower_bound:,.0f}–${upper_bound:,.0f}"
                        )
                else:
                    st.markdown("---")
                    st.markdown(
                        f"""
                        <div class="soft-card">
                            <div class="result-label">
                                Estimated historical sale price
                            </div>
                            <div class="result-value">
                                {money_usd(prediction)}
                            </div>
                            <p class="muted">
                                ANN prediction · USD · {sale_year}-{sale_month:02d}
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    metric_strip([
                        ("Model test R²", "0.841", "Saved holdout score"),
                        ("Typical test MAE", "$79.9K", "Historical test split"),
                    ])

                    with st.expander("View model input details"):
                        st.dataframe(row, use_container_width=True)

    except Exception as exc:
        show_model_error("US home price model", exc)


# ============================================================
# 11. INDIA PROPERTY PRICE PREDICTION - ANN
# ============================================================

elif selected_page == "India Property Estimate":

    page_title(
        "INDIA PROPERTY MODULE",
        "Explore an Indian Property Price",
        "Compare the selected property with observed listings in the same market.",
    )

    st.markdown(
        """
        <div class="warning-callout">
            The original ANN has a negative held-out R² (about -0.065), so it is not
            reliable enough to provide the headline price. This page now uses a
            transparent comparable-listing reference and shows when the selected
            property falls outside the dataset's observed range.
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        loc_df = load_location_data()

        excluded = {
            "Price", "Area", "Location", "No. of Bedrooms",
            "Resale", "City", "City_Location", "Price_per_Area",
            "Amenity_Yes_Count", "Amenity_No_Count", "Amenity_Unknown_Count",
        }
        amenity_cols = [col for col in loc_df.columns if col not in excluded]
        city_options = sorted(loc_df["City"].dropna().astype(str).unique())

        with st.form("india_price_form"):
            c1, c2 = st.columns(2)

            with c1:
                st.markdown("### Property location")
                city_index = city_options.index("Bangalore") if "Bangalore" in city_options else 0
                city = st.selectbox("City", city_options, index=city_index)
                city_locations = sorted(
                    loc_df.loc[
                        loc_df["City"].astype(str) == city,
                        "Location",
                    ].dropna().astype(str).unique()
                )
                location = st.selectbox("Locality", city_locations)
                area = st.number_input(
                    "Built-up area (sq ft)",
                    min_value=100,
                    max_value=20000,
                    value=1200,
                    step=50,
                )
                bedrooms = st.selectbox("Bedrooms", [1, 2, 3, 4, 5, 6], index=1)
                resale = st.selectbox(
                    "Resale property?",
                    [0, 1],
                    format_func=lambda value: "Yes" if value else "No",
                )

            with c2:
                st.markdown("### Property amenities")
                default_amenities = [
                    item for item in [
                        "CarParking", "Gymnasium", "LiftAvailable", "24X7Security",
                    ] if item in amenity_cols
                ]
                selected_amenities = st.multiselect(
                    "Amenities available (Yes)",
                    amenity_cols,
                    default=default_amenities,
                    format_func=amenity_label,
                    key="india_amenities_yes",
                )
                no_options = [item for item in amenity_cols if item not in selected_amenities]
                unavailable_amenities = st.multiselect(
                    "Confirmed not available (No)",
                    no_options,
                    format_func=amenity_label,
                    key="india_amenities_no",
                )
                unknown_amenities = [
                    item for item in amenity_cols
                    if item not in selected_amenities and item not in unavailable_amenities
                ]
                st.caption(
                    f"{len(selected_amenities)} marked Yes · "
                    f"{len(unavailable_amenities)} marked No · "
                    f"{len(unknown_amenities)} left as not specified."
                )
                amenity_unknown_share = (
                    float((loc_df[amenity_cols] == 9).to_numpy().mean() * 100)
                    if amenity_cols else 0.0
                )
                st.caption(
                    f"Source data has about {amenity_unknown_share:.1f}% unreported amenity values. "
                    "Amenities are recorded here but are not used to add or subtract price: "
                    "the source does not support reliable amenity premiums."
                )

            submitted = st.form_submit_button(
                "Run property price estimate",
                use_container_width=True,
            )

        if submitted:
            result = estimate_india_comparable(
                loc_df,
                city=city,
                location=location,
                area_sqft=float(area),
                bedrooms=int(bedrooms),
                resale=int(resale),
            )

            st.markdown(
                f"""
                <div class="soft-card">
                    <div class="result-label">COMPARABLE-BASED REFERENCE · NOT A MARKET VALUATION</div>
                    <div class="result-value">{money_inr(result['estimate'])}</div>
                    <p class="muted">
                        Indicative range: {money_inr(result['lower_estimate'])} – {money_inr(result['upper_estimate'])}
                    </p>
                    <p class="muted">
                        {city} · {location} · {area:,.0f} sq ft · {bedrooms} bedroom(s)
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if result["area_out_of_support"]:
                if result["bedroom_area_support_low"] is not None:
                    observed_range = (
                        f"{result['bedroom_area_support_low']:,.0f}–"
                        f"{result['bedroom_area_support_high']:,.0f} sq ft"
                    )
                else:
                    observed_range = "not enough comparable rows to establish a reliable range"
                st.warning(
                    f"Low-confidence reference: the selected {bedrooms}-bedroom / {area:,.0f} sq ft "
                    f"combination is outside the observed {city} training range for this bedroom count "
                    f"({observed_range}; {result['city_bedroom_count']} listings). "
                    f"The displayed reference uses {result['sample_count']} nearby-area listings "
                    f"and {result['bedroom_match_count']} of them match the selected bedroom count. "
                    "Do not treat this as a price estimate for the exact property."
                )
            elif result["confidence"] == "Low":
                st.warning(
                    f"Limited comparable data: only {result['sample_count']} listings were available "
                    f"using '{result['method']}'. Treat this as a rough reference."
                )
            else:
                st.success(
                    f"Reference based on {result['sample_count']} comparable listings using "
                    f"'{result['method']}'."
                )

            st.info(
                "Important: the source dataset leaves about "
                f"{amenity_unknown_share:.1f}% of amenity values unreported. "
                "Amenity and resale selections are captured in the form but are not used as "
                "automatic price adjustments. This prevents unsupported Yes/No inputs from "
                "creating misleading price jumps. Bedroom count is used only when the data "
                "contains enough comparable listings; otherwise the page explicitly reports the mismatch."
            )

            c1, c2, c3 = st.columns(3)
            c1.metric("Comparable listings", f"{result['sample_count']:,}")
            c2.metric("Median price per sq ft", money_inr(result["median_ppsf"]))
            c3.metric(
                "Same-bedroom matches",
                f"{result['bedroom_match_count']:,} / {result['sample_count']:,}",
            )

            with st.expander("View comparable-data details"):
                st.write({
                    "Comparable group": result["method"],
                    "Locality listings available": result["locality_count"],
                    "Locality + selected bedroom + area matches": result["locality_bedroom_area_count"],
                    "City + selected bedroom + area matches": result["city_bedroom_area_count"],
                    "Selected comparable area range (10th–90th percentile)": (
                        f"{result['selected_comp_areas_low']:,.0f}–"
                        f"{result['selected_comp_areas_high']:,.0f} sq ft"
                    ),
                    "Confidence": result["confidence"],
                    "Resale adjustment applied": "No",
                    "Amenity adjustment applied": "No",
                })

            with st.expander("Why the original ANN output is not used"):
                st.write(
                    "The saved ANN has a held-out R² of approximately -0.065 and an MAE of "
                    "about ₹82.3 lakh. Its R² is worse than a simple baseline, so it can produce "
                    "implausible outputs for uncommon city / locality / bedroom / area combinations. "
                    "It remains in the submodule for academic study, but it is not promoted as a reliable estimator."
                )

    except Exception as exc:
        show_model_error("India comparable estimate", exc)


# ============================================================
# 12. PROPERTY IMAGE CLASSIFICATION - CNN
# ============================================================

elif selected_page == "Room Photo Checker":

    page_title(
        "IMAGE CLASSIFICATION MODULE",
        "What Room Is in This Photo?",
        "Upload a clear room image to identify its category.",
    )

    class_names = [
        "Bathroom",
        "Bedroom",
        "Dinning",
        "Kitchen",
        "Livingroom",
    ]

    display_names = {
        "Bathroom": "Bathroom",
        "Bedroom": "Bedroom",
        "Dinning": "Dining room",
        "Kitchen": "Kitchen",
        "Livingroom": "Living room",
    }

    left, right = st.columns([1.05, 1])

    with left:
        uploaded_image = st.file_uploader(
            "Upload a room image",
            type=["jpg", "jpeg", "png", "webp"],
        )

        img = None

        if uploaded_image is not None:
            try:
                img = Image.open(uploaded_image).convert("RGB")
                st.image(
                    img,
                    caption=f"Uploaded image · {img.width} × {img.height}",
                    use_container_width=True,
                )
            except Exception as exc:
                st.error(f"Unable to read image: {exc}")

    with right:
        st.markdown("### Your photo result")

        st.info(
            "The model classifies the room category only. Its test accuracy is about 54%, "
            "so ambiguous photos may be marked uncertain. Scores are model scores, not "
            "calibrated probabilities. It does not evaluate property quality or price."
        )

        run_image = st.button(
            "Classify uploaded image",
            type="primary",
            use_container_width=True,
            disabled=img is None,
        )

        if run_image and img is not None:
            try:
                model = load_keras_model("property_cnn_model.keras")

                resized = img.resize((96, 96), Image.Resampling.LANCZOS)
                image_array = np.asarray(
                    resized, dtype=np.float32
                )

                probabilities = predict_array(
                    model, np.expand_dims(image_array, axis=0)
                ).reshape(-1)

                if len(probabilities) != len(class_names):
                    raise ValueError(
                        "Model output does not match the five expected classes."
                    )

                best_idx = int(np.argmax(probabilities))
                best_class_name = display_names[class_names[best_idx]]
                best_score = float(probabilities[best_idx])
                sorted_scores = np.sort(probabilities)
                second_score = float(sorted_scores[-2]) if len(sorted_scores) > 1 else 0.0
                score_margin = best_score - second_score
                # A weak top score or close second choice should not be presented as certain.
                confident_enough = best_score >= 0.62 and score_margin >= 0.10
                best_name = (
                    best_class_name
                    if confident_enough
                    else f"Uncertain · possible {best_class_name}"
                )

                st.markdown(
                    f"""
                    <div class="soft-card">
                        <div class="result-label">Predicted room type</div>
                        <div class="result-value">{best_name}</div>
                        <p class="muted">
                            Top-class model score: {best_score:.1%}
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if not confident_enough:
                    st.warning(
                        f"The model is not confident enough to confirm this room. "
                        f"Top match: {best_class_name} ({best_score:.1%}); "
                        f"next match: {second_score:.1%}. Try a clearer photo showing the main room features."
                    )

                probability_df = pd.DataFrame({
                    "Room type": [
                        display_names[x] for x in class_names
                    ],
                    "Probability": probabilities,
                })

                if PLOTLY_AVAILABLE:
                    fig = px.bar(
                        probability_df,
                        x="Room type",
                        y="Probability",
                        color="Room type",
                        title="Model score by room category",
                    )
                    fig.update_yaxes(tickformat=".0%")
                    show_plotly(fig)
                else:
                    st.bar_chart(
                        probability_df.set_index("Room type")["Probability"]
                    )

                st.download_button(
                    "Download prediction result",
                    probability_df.to_csv(index=False).encode("utf-8"),
                    file_name="estateiq_room_prediction.csv",
                    mime="text/csv",
                )

            except Exception as exc:
                show_model_error("CNN image classification model", exc)


# ============================================================
# 13. PROPERTY DEMAND FORECASTING - LSTM
# ============================================================

elif selected_page == "Demand Forecast":

    page_title(
        "DEMAND FORECAST MODULE",
        "Six-Month Demand Forecast",
        "Explore six modelled monthly steps beyond the latest available sample record.",
    )

    st.info(
        "This is a learning/demo forecast built from synthetic sample data. "
        "It forecasts aggregated monthly demand—not individual cities—and "
        "does not represent live property-market activity."
    )

    try:
        from sklearn.preprocessing import MinMaxScaler

        demand_df = load_demand_data()

        series = (
            demand_df.groupby("Date", as_index=False)["Sales_Count"]
            .sum()
            .sort_values("Date")
            .reset_index(drop=True)
        )

        if len(series) < 17:
            raise ValueError(
                "At least 17 monthly records are needed for forecasting."
            )

        model = load_keras_model("property_demand_lstm.keras")

        values = series["Sales_Count"].to_numpy(
            dtype=float
        ).reshape(-1, 1)

        lookback = 12
        train_end = max(1, int(len(values) * 0.70))

        scaler = MinMaxScaler()
        scaler.fit(values[:train_end])

        current_sequence = scaler.transform(
            values[-lookback:]
        ).flatten()

        forecasts = []

        for _ in range(6):
            model_input = current_sequence.reshape(
                1, lookback, 1
            ).astype(np.float32)

            next_scaled = float(predict_array(model, model_input).reshape(-1)[0])

            next_value = float(
                scaler.inverse_transform(
                    np.array([[next_scaled]])
                )[0, 0]
            )

            forecasts.append(max(0.0, next_value))

            current_sequence = np.append(
                current_sequence[1:], next_scaled
            )

        future_dates = pd.date_range(
            start=series["Date"].max() + pd.offsets.MonthBegin(1),
            periods=6,
            freq="MS",
        )

        forecast_start = future_dates.min()
        forecast_end = future_dates.max()
        current_month = pd.Timestamp.now().to_period("M")
        if forecast_start.to_period("M") < current_month:
            window_status = (
                "is historical now, not a fully forward-looking forecast."
                if forecast_end.to_period("M") < current_month
                else "has already started, so it is not a fully forward-looking forecast."
            )
            st.warning(
                f"The supplied sample data ends in {series['Date'].max():%B %Y}. "
                f"This model's six-step window ({forecast_start:%b %Y}–{forecast_end:%b %Y}) "
                f"{window_status} For a current outlook, add recent real data and retrain the model."
            )
        else:
            st.caption(
                f"Forecast window: {forecast_start:%B %Y} to {forecast_end:%B %Y} "
                "(six steps after the last available record)."
            )

        forecast_df = pd.DataFrame({
            "Date": future_dates,
            "Predicted_Sales_Count": np.round(forecasts, 2),
        })

        latest = float(series["Sales_Count"].iloc[-1])
        average = float(np.mean(forecasts))

        change = (
            (forecasts[-1] - latest) / latest * 100
            if latest else 0
        )

        metric_strip([
            ("Latest monthly demand", f"{latest:,.0f}", "Sample data"),
            ("Forecast six-month average", f"{average:,.0f}", "LSTM output"),
            ("Month 6 change", f"{change:+.1f}%", "Compared with latest"),
        ])

        actual = series.tail(36).rename(
            columns={"Sales_Count": "Actual demand"}
        )

        predicted = forecast_df.rename(
            columns={"Predicted_Sales_Count": "Forecast demand"}
        )

        if PLOTLY_AVAILABLE:
            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=actual["Date"],
                    y=actual["Actual demand"],
                    mode="lines+markers",
                    name="Historical sample",
                    line={"color": "#4F46E5", "width": 3},
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=predicted["Date"],
                    y=predicted["Forecast demand"],
                    mode="lines+markers",
                    name="LSTM forecast",
                    line={"color": "#0891B2", "width": 3, "dash": "dash"},
                )
            )

            fig.update_layout(
                title="Historical demand and six-month forecast",
                xaxis_title="Month",
                yaxis_title="Sales count",
            )

            show_plotly(fig)
        else:
            actual_line = actual.set_index("Date")["Actual demand"]
            future_line = predicted.set_index("Date")["Forecast demand"]

            comparison = pd.concat(
                [actual_line, future_line], axis=1
            )

            st.line_chart(comparison)

        st.markdown("### Forecast details")

        shown = forecast_df.copy()
        shown["Date"] = shown["Date"].dt.strftime("%B %Y")

        st.dataframe(
            shown,
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Download forecast CSV",
            forecast_df.to_csv(index=False).encode("utf-8"),
            file_name="estateiq_demand_forecast.csv",
            mime="text/csv",
        )

        with st.expander("View historical demand data"):
            st.dataframe(
                series.tail(36),
                use_container_width=True,
                hide_index=True,
            )

    except Exception as exc:
        show_model_error("LSTM demand forecasting model", exc)


# ============================================================
# 14. CUSTOMER REVIEW SENTIMENT - RNN
# ============================================================

elif selected_page == "Review Sentiment":

    page_title(
        "CUSTOMER FEEDBACK MODULE",
        "Understand Customer Reviews",
        "Analyse one review or a CSV containing multiple reviews.",
    )

    st.info(
        "The RNN model predicts Negative, Neutral or Positive sentiment. "
        "Sarcasm, short text and mixed opinions may be difficult to classify."
    )

    tab_one, tab_batch = st.tabs([
        "Single review",
        "Batch CSV analysis",
    ])

    try:
        import tensorflow as tf

        model = load_keras_model("customer_review_rnn.keras")

        tokenizer_path = model_file(
            "customer_review_tokenizer.json"
        )

        tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(
            tokenizer_path.read_text(encoding="utf-8")
        )

        sentiment_names = [
            "Negative", "Neutral", "Positive"
        ]

        def predict_reviews(reviews: list[str]) -> np.ndarray:
            cleaned = [clean_review(x) for x in reviews]
            sequences = tokenizer.texts_to_sequences(cleaned)

            padded = tf.keras.preprocessing.sequence.pad_sequences(
                sequences,
                maxlen=120,
                padding="post",
                truncating="post",
            )

            return predict_array(model, padded)

        with tab_one:
            review_text = st.text_area(
                "Write or paste a customer review",
                value=(
                    "The product quality is excellent and "
                    "delivery was faster than expected."
                ),
                height=140,
                placeholder="Enter a customer review...",
            )

            if st.button(
                "Analyse sentiment",
                key="single_sentiment",
                type="primary",
                use_container_width=True,
            ):

                if not review_text.strip():
                    st.warning("Please enter a review first.")
                else:
                    probabilities = predict_reviews([review_text])[0]

                    if len(probabilities) != 3:
                        raise ValueError(
                            "RNN model output is not three sentiment classes."
                        )

                    index = int(np.argmax(probabilities))
                    sentiment = sentiment_names[index]

                    st.markdown(
                        f"""
                        <div class="soft-card">
                            <div class="result-label">Predicted sentiment</div>
                            <div class="result-value">{sentiment}</div>
                            <p class="muted">
                                Confidence: {probabilities[index]:.1%}
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    probability_df = pd.DataFrame({
                        "Sentiment": sentiment_names,
                        "Probability": probabilities,
                    })

                    if PLOTLY_AVAILABLE:
                        fig = px.bar(
                            probability_df,
                            x="Sentiment",
                            y="Probability",
                            color="Sentiment",
                            title="Sentiment probability",
                        )
                        fig.update_yaxes(tickformat=".0%")
                        show_plotly(fig)
                    else:
                        st.bar_chart(
                            probability_df.set_index("Sentiment")
                            ["Probability"]
                        )

        with tab_batch:
            st.markdown(
                "Upload a CSV with a column containing customer reviews."
            )

            batch_file = st.file_uploader(
                "Choose review CSV",
                type=["csv"],
                key="sentiment_csv",
            )

            if batch_file is not None:
                upload_signature = hashlib.sha1(batch_file.getbuffer()).hexdigest()
                if st.session_state.get("estateiq_batch_upload_signature") != upload_signature:
                    st.session_state.pop("estateiq_batch_result", None)
                    st.session_state["estateiq_batch_upload_signature"] = upload_signature

                # One extra row is enough to tell the user when the 2,000-row cap applies.
                batch_file.seek(0)
                batch_df = pd.read_csv(batch_file, nrows=2001)

                if batch_df.empty:
                    st.warning("The uploaded CSV is empty.")
                else:
                    text_column = st.selectbox(
                        "Select the review text column",
                        batch_df.columns.tolist(),
                    )
                    if st.session_state.get("estateiq_batch_column") != text_column:
                        st.session_state.pop("estateiq_batch_result", None)

                    if st.button(
                        "Analyse CSV reviews",
                        key="batch_sentiment",
                        type="primary",
                        use_container_width=True,
                    ):
                        reviews = (
                            batch_df[text_column]
                            .fillna("")
                            .astype(str)
                            .tolist()[:2000]
                        )

                        nonempty = [
                            i for i, value in enumerate(reviews)
                            if value.strip()
                        ]

                        if not nonempty:
                            st.warning(
                                "No non-empty reviews were found."
                            )
                        else:
                            with st.spinner(
                                f"Analysing {len(nonempty):,} reviews..."
                            ):
                                probabilities = predict_reviews(
                                    [reviews[i] for i in nonempty]
                                )

                            result_df = batch_df.iloc[
                                :len(reviews)
                            ].copy()

                            result_df["Predicted_Sentiment"] = "No text"
                            result_df["Sentiment_Confidence"] = np.nan

                            for idx, prob in zip(nonempty, probabilities):
                                result_df.loc[
                                    result_df.index[idx],
                                    "Predicted_Sentiment",
                                ] = sentiment_names[int(np.argmax(prob))]

                                result_df.loc[
                                    result_df.index[idx],
                                    "Sentiment_Confidence",
                                ] = float(np.max(prob))

                            st.session_state["estateiq_batch_result"] = result_df
                            st.session_state["estateiq_batch_column"] = text_column

                    if "estateiq_batch_result" in st.session_state:
                        result_df = st.session_state[
                            "estateiq_batch_result"
                        ]

                        st.markdown("### Analysis results")

                        st.dataframe(
                            result_df.head(100),
                            use_container_width=True,
                            hide_index=True,
                        )

                        counts = (
                            result_df["Predicted_Sentiment"]
                            .value_counts()
                            .rename_axis("Sentiment")
                            .reset_index(name="Reviews")
                        )

                        if PLOTLY_AVAILABLE:
                            fig = px.bar(
                                counts,
                                x="Sentiment",
                                y="Reviews",
                                color="Sentiment",
                                title="Sentiment distribution",
                            )
                            show_plotly(fig)
                        else:
                            st.bar_chart(
                                counts.set_index("Sentiment")["Reviews"]
                            )

                        st.download_button(
                            "Download analysed reviews",
                            result_df.to_csv(index=False).encode("utf-8"),
                            file_name="estateiq_sentiment_results.csv",
                            mime="text/csv",
                        )

                        if len(batch_df) > 2000:
                            st.warning(
                                "Only the first 2,000 rows were analysed."
                            )

    except Exception as exc:
        show_model_error("RNN sentiment analysis model", exc)


# ============================================================
# 15. DATA EXPLORER
# ============================================================

elif selected_page == "Explore Datasets":

    page_title(
        "DATA WORKSPACE",
        "Explore Your Dataset",
        "Preview records, check data quality and create charts.",
    )

    data_options = {
        "US house sales": "house",
        "Indian property listings": "location",
        "Synthetic demand history": "demand",
        "Upload my own CSV": "upload",
    }

    choice = st.selectbox(
        "Choose a data source",
        list(data_options.keys()),
    )

    try:
        source = data_options[choice]

        if source == "house":
            df = load_house_data()

        elif source == "location":
            df = load_location_data()

        elif source == "demand":
            df = load_demand_data()

        else:
            upload = st.file_uploader(
                "Upload a CSV dataset",
                type=["csv"],
                key="explorer_upload",
            )

            if upload is None:
                st.info("Upload a CSV file to begin exploring.")
                st.stop()

            df = pd.read_csv(upload)

        if df.empty:
            st.warning("This dataset has no data rows.")
            st.stop()

        metric_strip([
            ("Rows", f"{len(df):,}", None),
            ("Columns", f"{df.shape[1]:,}", None),
            ("Duplicates", f"{df.duplicated().sum():,}", None),
            ("Missing cells", f"{df.isna().sum().sum():,}", None),
        ])

        st.markdown("### Data preview")

        st.dataframe(
            df.head(100),
            use_container_width=True,
            hide_index=True,
        )

        left, right = st.columns(2)

        with left:
            st.markdown("### Column quality")

            quality = pd.DataFrame({
                "Column": df.columns,
                "Data type": [str(x) for x in df.dtypes],
                "Missing": df.isna().sum().values,
                "Missing %": (
                    df.isna().mean().values * 100
                ).round(2),
            })

            st.dataframe(
                quality,
                use_container_width=True,
                height=380,
                hide_index=True,
            )

        with right:
            st.markdown("### Quick chart builder")

            numeric_cols = df.select_dtypes(
                include=np.number
            ).columns.tolist()

            if len(df.columns) >= 2 and numeric_cols:

                x_col = st.selectbox(
                    "X axis",
                    df.columns.tolist(),
                )

                y_col = st.selectbox(
                    "Numeric Y axis",
                    numeric_cols,
                )

                chart_type = st.selectbox(
                    "Chart type",
                    ["Bar", "Line", "Scatter", "Histogram"],
                )

                plot_df = df.head(10000).copy()

                try:
                    if chart_type == "Bar":
                        grouped = (
                            plot_df.groupby(
                                x_col,
                                dropna=False,
                            )[y_col]
                            .mean()
                            .reset_index()
                            .sort_values(y_col, ascending=False)
                            .head(30)
                        )

                        if PLOTLY_AVAILABLE:
                            fig = px.bar(
                                grouped,
                                x=x_col,
                                y=y_col,
                                title=f"Average {y_col} by {x_col}",
                            )
                            show_plotly(fig)
                        else:
                            st.bar_chart(
                                grouped.set_index(x_col)[y_col]
                            )

                    elif chart_type == "Line":
                        sorted_df = plot_df.sort_values(x_col).head(1500)

                        if PLOTLY_AVAILABLE:
                            fig = px.line(
                                sorted_df,
                                x=x_col,
                                y=y_col,
                                title=f"{y_col} over {x_col}",
                            )
                            show_plotly(fig)
                        else:
                            st.line_chart(
                                sorted_df.set_index(x_col)[y_col]
                            )

                    elif chart_type == "Scatter":
                        if PLOTLY_AVAILABLE:
                            fig = px.scatter(
                                plot_df,
                                x=x_col,
                                y=y_col,
                                title=f"{y_col} vs {x_col}",
                            )
                            show_plotly(fig)
                        elif pd.api.types.is_numeric_dtype(plot_df[x_col]):
                            st.scatter_chart(
                                plot_df,
                                x=x_col,
                                y=y_col,
                            )
                        else:
                            st.info(
                                "Install Plotly for category-based scatter charts."
                            )

                    else:
                        if PLOTLY_AVAILABLE:
                            fig = px.histogram(
                                plot_df,
                                x=y_col,
                                nbins=40,
                                title=f"Distribution of {y_col}",
                            )
                            show_plotly(fig)
                        else:
                            st.bar_chart(
                                plot_df[y_col].value_counts().sort_index()
                            )

                except Exception as chart_error:
                    st.warning(
                        f"Unable to create this chart: {chart_error}"
                    )

            else:
                st.info(
                    "This dataset does not have enough numeric columns "
                    "to build a chart."
                )

        st.download_button(
            "Download current dataset",
            df.to_csv(index=False).encode("utf-8"),
            file_name="estateiq_dataset_export.csv",
            mime="text/csv",
        )

    except Exception as exc:
        show_model_error("selected dataset", exc)


# ============================================================
# 16. FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div style="
        display:flex;
        justify-content:space-between;
        flex-wrap:wrap;
        gap:10px;
        color:#64748B;
        font-size:12px;
    ">
        <span>EstateIQ · AI-Powered Real Estate Intelligence Platform</span>
        <span>Streamlit · Python · TensorFlow/Keras · Pandas · Plotly</span>
    </div>
    """,
    unsafe_allow_html=True,
)
