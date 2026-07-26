"""
Breast Cancer Diagnosis — Streamlit demo (v3, simplified)

Wired to the user's own trained pipeline: a StandardScaler + StackingClassifier
(XGBoost + LightGBM + RandomForest, meta-learner: Logistic Regression), saved as
a single sklearn Pipeline object. No separate scaler.pkl is needed — scaling is
the first step inside the pipeline itself.
"""

import json
from pathlib import Path

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# --------------------------------------------------------------------------
# Page setup
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Breast Cancer Diagnosis · ML Demo",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"

# --------------------------------------------------------------------------
# Unified SaaS Design System & CSS
# --------------------------------------------------------------------------
INK = "#0F172A"
BODY = "#334155"
BG = "#F8FAFC"
PANEL = "#FFFFFF"
BORDER = "#E2E8F0"
PRIMARY = "#0F766E"
PRIMARY_HOVER = "#115E59"
ACCENT = "#EA580C"
ACCENT_HOVER = "#C2410C"
SUCCESS = "#16A34A"
WARNING = "#D97706"
ERROR = "#DC2626"
INFO = "#2563EB"

st.markdown(
    f"""
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">

    <style>
    /* Global Styles & Reset */
    html, body, [class*="css"], [data-testid="stAppViewContainer"] {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        color: {INK} !important;
        background-color: {BG} !important;
        -webkit-font-smoothing: antialiased;
    }}
    .stApp {{
        background-color: {BG} !important;
    }}
    
    .block-container {{
        padding-top: 2rem !important;
        padding-bottom: 3rem !important;
        max-width: 1200px;
    }}
    
    /* Typography */
    h1, h2, h3, h4, h5, h6 {{
        font-family: 'Inter', sans-serif !important;
        letter-spacing: -0.02em !important;
        color: {INK} !important;
    }}
    
    .hero-title {{
        font-size: 32px;
        line-height: 1.2;
        margin: 0 0 8px 0;
        color: {INK} !important;
        font-weight: 700;
        letter-spacing: -0.025em;
    }}
    
    .hero-lede {{
        color: {BODY} !important;
        font-size: 15px;
        line-height: 1.6;
        max-width: 68ch;
        margin: 0 0 24px 0;
        font-weight: 400;
    }}
    
    .section-label {{
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: {BODY} !important;
        margin: 0 0 12px 0;
    }}

    /* Sensor Info Box */
    .sensor-info-box {{
        background-color: #F1F5F9;
        border: 1px solid #CBD5E1;
        border-radius: 10px;
        padding: 12px 16px;
        font-size: 13px;
        line-height: 1.5;
        color: {INK};
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        gap: 10px;
    }}

    .sensor-info-box strong {{
        color: {PRIMARY};
        font-weight: 600;
    }}

    /* Verdict Cards */
    .verdict-card {{
        margin-top: 16px;
        width: 100%;
        text-align: left;
        padding: 16px 20px;
        border-radius: 12px;
        display: flex;
        flex-direction: column;
        gap: 4px;
        transition: all 0.25s ease;
    }}
    
    .verdict-card.benign {{
        background: rgba(22, 163, 74, 0.08) !important;
        border: 1px solid rgba(22, 163, 74, 0.3) !important;
    }}
    
    .verdict-card.malignant {{
        background: rgba(234, 88, 12, 0.08) !important;
        border: 1px solid rgba(234, 88, 12, 0.3) !important;
    }}
    
    .verdict-card.idle {{
        background: {BG} !important;
        border: 1px dashed {BORDER} !important;
        color: {BODY} !important;
        text-align: center;
        padding: 20px;
    }}
    
    .verdict-header {{
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 16px;
        font-weight: 700;
    }}
    
    .verdict-card.benign .verdict-header {{ color: {SUCCESS} !important; }}
    .verdict-card.malignant .verdict-header {{ color: {ACCENT} !important; }}
    
    .verdict-subtext {{
        font-size: 12px;
        color: {BODY} !important;
        font-weight: 400;
        line-height: 1.4;
    }}

    /* Note Container */
    .note {{
        font-size: 13px;
        color: {BODY} !important;
        line-height: 1.6;
        background: {PANEL} !important;
        border: 1px solid {BORDER} !important;
        border-radius: 12px;
        padding: 14px 18px;
        margin-top: 24px;
        box-shadow: 0 1px 2px 0 rgba(15, 23, 42, 0.02);
    }}

    /* Buttons */
    .stButton > button {{
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        border: 1px solid {BORDER} !important;
        background-color: {PANEL} !important;
        color: {INK} !important;
        font-family: 'Inter', sans-serif !important;
        transition: all 0.25s ease !important;
        box-shadow: 0 1px 2px 0 rgba(0,0,0,0.05) !important;
        padding: 0.5rem 1rem !important;
    }}
    
    .stButton > button:hover {{
        background-color: {BG} !important;
        border-color: #CBD5E1 !important;
        color: {INK} !important;
        transform: translateY(-1px);
    }}
    
    .stButton > button[kind="primary"] {{
        background: linear-gradient(180deg, {PRIMARY} 0%, {PRIMARY_HOVER} 100%) !important;
        color: #FFFFFF !important;
        border: 1px solid {PRIMARY_HOVER} !important;
        min-height: 46px !important;
        box-shadow: 0 1px 3px 0 rgba(15, 118, 110, 0.3), 0 1px 2px -1px rgba(15, 118, 110, 0.2) !important;
    }}
    
    .stButton > button[kind="primary"]:hover {{
        background: linear-gradient(180deg, {PRIMARY_HOVER} 0%, #0F4C46 100%) !important;
        box-shadow: 0 4px 12px 0 rgba(15, 118, 110, 0.35) !important;
        transform: translateY(-1px);
    }}

    /* Input Controls */
    div[data-testid="stNumberInput"] label {{
        font-size: 13px !important;
        font-weight: 600 !important;
        color: {INK} !important;
        margin-bottom: 4px !important;
    }}

    div[data-baseweb="input"] {{
        border-radius: 8px !important;
        background-color: #FFFFFF !important;
        border: 1px solid {BORDER} !important;
        transition: all 0.2s ease !important;
    }}
    
    div[data-baseweb="input"] input {{
        color: {INK} !important;
        background-color: #FFFFFF !important;
    }}
    
    div[data-baseweb="input"]:hover {{
        border-color: #94A3B8 !important;
    }}

    /* ---------------------------------------------------- */
    /* ULTRA-STRONG TABS STYLING (FORCE FULL BLACK VISIBILITY) */
    /* ---------------------------------------------------- */
    div[data-testid="stTabs"] {{
        border: 1px solid #CBD5E1 !important;
        background-color: #E2E8F0 !important;
        padding: 4px !important;
        border-radius: 12px !important;
        margin-bottom: 20px !important;
    }}

    /* Target every single child element inside Streamlit tabs */
    div[data-testid="stTabs"] button,
    div[data-testid="stTabs"] button *,
    div[data-testid="stTabs"] [role="tab"],
    div[data-testid="stTabs"] [role="tab"] *,
    div[data-testid="stTabs"] p,
    div[data-testid="stTabs"] span {{
        color: #000000 !important;
        -webkit-text-fill-color: #000000 !important;
        opacity: 1 !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 14px !important;
        font-weight: 700 !important;
    }}

    div[data-testid="stTabs"] button {{
        flex: 1 !important;
        padding: 9px 16px !important;
        border: 1px solid transparent !important;
        background: transparent !important;
        border-radius: 8px !important;
        transition: all 0.2s ease !important;
    }}

    div[data-testid="stTabs"] button:hover {{
        background-color: rgba(255, 255, 255, 0.6) !important;
    }}

    div[data-testid="stTabs"] button[aria-selected="true"] {{
        background-color: #FFFFFF !important;
        border-color: #94A3B8 !important;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1) !important;
    }}

    div[data-testid="stTabs"] button[aria-selected="true"] * {{
        color: {PRIMARY} !important;
        -webkit-text-fill-color: {PRIMARY} !important;
    }}

    /* Expander Contrast Fixes */
    div[data-testid="stExpander"] {{
        border: 1px solid {BORDER} !important;
        border-radius: 12px !important;
        background-color: {PANEL} !important;
        box-shadow: none !important;
        overflow: hidden;
    }}

    div[data-testid="stExpander"] summary {{
        font-size: 13.5px !important;
        font-weight: 600 !important;
        color: {INK} !important;
        background-color: #F1F5F9 !important;
        padding: 12px 16px !important;
    }}

    div[data-testid="stExpander"] summary * {{
        color: {INK} !important;
    }}

    div[data-testid="stExpanderDetails"] {{
        background-color: {PANEL} !important;
        color: {INK} !important;
        padding: 16px !important;
    }}

    div[data-testid="stExpanderDetails"] * {{
        color: {INK} !important;
    }}

    /* Hide Streamlit Chrome */
    footer {{visibility: hidden;}}
    #MainMenu {{visibility: hidden;}}
    header {{visibility: hidden;}}
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Load model artifacts (cached so this only runs once per session)
# --------------------------------------------------------------------------
FALLBACK_SUMMARY = {
    "best_model_name": "Stacking Ensemble (XGBoost + LightGBM + RandomForest → Logistic Regression)",
    "selected_threshold": 0.5,
}


@st.cache_resource
def load_artifacts():
    pipeline = joblib.load(ARTIFACTS_DIR / "best_breast_cancer_pipeline.pkl")
    label_encoder = joblib.load(ARTIFACTS_DIR / "label_encoder.pkl")
    summary_path = ARTIFACTS_DIR / "model_summary.json"
    if summary_path.exists():
        with open(summary_path, encoding="utf-8") as f:
            summary = json.load(f)
    else:
        summary = FALLBACK_SUMMARY
    return pipeline, label_encoder, summary


pipeline, label_encoder, summary = load_artifacts()
FEATURE_NAMES = list(pipeline.feature_names_in_)
THRESHOLD = summary.get("selected_threshold", 0.5)

GROUPS = [("mean", "Mean"), ("se", "Standard Error"), ("worst", "Worst")]

# --------------------------------------------------------------------------
# Sample presets — real WDBC cases, trimmed to the features this pipeline
# was trained on.
# --------------------------------------------------------------------------
_MALIGNANT_FULL = {
    "radius_mean": 20.57, "texture_mean": 17.77, "perimeter_mean": 132.9,
    "area_mean": 1326.0, "smoothness_mean": 0.08474, "compactness_mean": 0.07864,
    "concavity_mean": 0.0869, "concave points_mean": 0.07017, "symmetry_mean": 0.1812,
    "fractal_dimension_mean": 0.05667, "radius_se": 0.5435, "texture_se": 0.7339,
    "perimeter_se": 3.398, "area_se": 74.08, "smoothness_se": 0.005225,
    "compactness_se": 0.01308, "concavity_se": 0.0186, "concave points_se": 0.0134,
    "symmetry_se": 0.01389, "fractal_dimension_se": 0.003532, "radius_worst": 24.99,
    "texture_worst": 23.41, "perimeter_worst": 158.8, "area_worst": 1956.0,
    "smoothness_worst": 0.1238, "compactness_worst": 0.1866, "concavity_worst": 0.2416,
    "concave points_worst": 0.186, "symmetry_worst": 0.275, "fractal_dimension_worst": 0.08902,
}
_BENIGN_FULL = {
    "radius_mean": 13.54, "texture_mean": 14.36, "perimeter_mean": 87.46, "area_mean": 566.3,
    "smoothness_mean": 0.09779, "compactness_mean": 0.08129, "concavity_mean": 0.06664,
    "concave points_mean": 0.04781, "symmetry_mean": 0.1885, "fractal_dimension_mean": 0.05766,
    "radius_se": 0.2699, "texture_se": 0.7886, "perimeter_se": 2.058, "area_se": 23.56,
    "smoothness_se": 0.008462, "compactness_se": 0.0146, "concavity_se": 0.02387,
    "concave points_se": 0.01315, "symmetry_se": 0.0198, "fractal_dimension_se": 0.0023,
    "radius_worst": 15.11, "texture_worst": 19.26, "perimeter_worst": 99.7, "area_worst": 711.2,
    "smoothness_worst": 0.144, "compactness_worst": 0.1773, "concavity_worst": 0.239,
    "concave points_worst": 0.1288, "symmetry_worst": 0.2977, "fractal_dimension_worst": 0.07259,
}
MALIGNANT_SAMPLE = {k: v for k, v in _MALIGNANT_FULL.items() if k in FEATURE_NAMES}
BENIGN_SAMPLE = {k: v for k, v in _BENIGN_FULL.items() if k in FEATURE_NAMES}

STEP_BY_GROUP = {"mean": 0.5, "se": 0.01, "worst": 0.5}


def input_key(name: str) -> str:
    return f"inp_{name}"


for _name in FEATURE_NAMES:
    _key = input_key(_name)
    if _key not in st.session_state:
        st.session_state[_key] = float(MALIGNANT_SAMPLE.get(_name, 0.0))

if "result" not in st.session_state:
    st.session_state.result = None


def apply_preset(preset: dict):
    for feat in FEATURE_NAMES:
        st.session_state[input_key(feat)] = float(preset.get(feat, 0.0))
    st.session_state.result = None


def run_prediction():
    row = {feat: st.session_state[input_key(feat)] for feat in FEATURE_NAMES}
    X_new = pd.DataFrame([row])[FEATURE_NAMES]
    proba_malignant = float(pipeline.predict_proba(X_new)[0, 1])
    pred = int(proba_malignant >= THRESHOLD)
    label = label_encoder.inverse_transform([pred])[0]
    st.session_state.result = {"diagnosis": label, "probability": proba_malignant}


# --------------------------------------------------------------------------
# Header — SaaS Title & Subtitle Context
# --------------------------------------------------------------------------
st.markdown(
    '<h1 class="hero-title">Breast Cancer Diagnosis</h1>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="hero-lede">A stacked ensemble (XGBoost, LightGBM, Random Forest) trained on the '
    'Wisconsin Diagnostic Breast Cancer dataset. Enter cell-nucleus measurements below, or load a '
    'real sample, to get a prediction.</p>',
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Layout: Input Form (Left) + SaaS Prediction Console (Right)
# --------------------------------------------------------------------------
left, right = st.columns([1.65, 1], gap="large")

with left:
    st.markdown('<div class="section-label">Presets & Feature Groups</div>', unsafe_allow_html=True)
    p1, p2 = st.columns(2)
    with p1:
        st.button(
            "Load benign example",
            use_container_width=True,
            on_click=apply_preset,
            args=(BENIGN_SAMPLE,),
        )
    with p2:
        st.button(
            "Load malignant example",
            use_container_width=True,
            on_click=apply_preset,
            args=(MALIGNANT_SAMPLE,),
        )

    st.write("")
    
    # Sensor note inserted cleanly
    st.markdown(
        '<div class="sensor-info-box">'
        '💡 <span><strong>Automated Data Feed:</strong> In clinical deployment, these cell-nucleus feature sets '
        'are computed automatically via direct integration with imaging sensors and quantitative cytology pipelines.</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    tabs = st.tabs([title for _, title in GROUPS])
    for (suffix, title), tab in zip(GROUPS, tabs):
        with tab:
            group_features = [f for f in FEATURE_NAMES if f.endswith(f"_{suffix}")]
            cols = st.columns(3)
            for i, feat in enumerate(group_features):
                label = feat.replace(f"_{suffix}", "").replace("_", " ").title()
                with cols[i % 3]:
                    st.number_input(
                        label,
                        key=input_key(feat),
                        step=STEP_BY_GROUP[suffix],
                        format="%.5f",
                    )

    st.write("")
    st.button("Run Prediction", type="primary", use_container_width=True, on_click=run_prediction)

with right:
    st.markdown('<div class="section-label">Diagnostic Output</div>', unsafe_allow_html=True)
    
    result = st.session_state.result

    prob = result["probability"] if result else 0.0
    diagnosis = result["diagnosis"] if result else None
    
    # Dynamic Gauge Bar Color
    bar_color = ACCENT if diagnosis == "M" else PRIMARY

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number={"suffix": "%", "font": {"size": 36, "color": INK, "family": "Inter", "weight": 700}},
            gauge={
                "axis": {
                    "range": [0, 100], 
                    "tickcolor": BORDER, 
                    "tickfont": {"color": BODY, "size": 11, "family": "Inter"},
                    "tickwidth": 1,
                },
                "bar": {"color": bar_color, "thickness": 0.22},
                "bgcolor": BG,
                "borderwidth": 1,
                "bordercolor": BORDER,
                "threshold": {
                    "line": {"color": ERROR, "width": 2},
                    "thickness": 0.8,
                    "value": THRESHOLD * 100,
                },
            },
        )
    )
    fig.update_layout(
        height=190,
        margin=dict(l=20, r=20, t=15, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": INK, "family": "Inter"},
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    
    st.markdown(
        f'<div style="text-align:center;color:{BODY};font-size:12px;font-weight:500;margin-top:-10px;margin-bottom:16px;">'
        f'Probability of Malignancy</div>',
        unsafe_allow_html=True,
    )

    if result is None:
        st.markdown(
            '<div class="verdict-card idle">'
            '<div style="font-size: 13px; font-weight: 500;">Select a sample preset or adjust parameters, then click <strong>Run Prediction</strong>.</div>'
            '</div>',
            unsafe_allow_html=True,
        )
    elif diagnosis == "M":
        st.markdown(
            f'<div class="verdict-card malignant">'
            f'<div class="verdict-header"><span>Malignant Diagnosis</span></div>'
            f'<div class="verdict-subtext">Classified via {summary.get("best_model_name", "stacked pipeline")} at decision threshold {THRESHOLD}.</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="verdict-card benign">'
            f'<div class="verdict-header"><span>Benign Diagnosis</span></div>'
            f'<div class="verdict-subtext">Classified via {summary.get("best_model_name", "stacked pipeline")} at decision threshold {THRESHOLD}.</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.write("")
    with st.expander("Model Specifications"):
        st.write(f"**Architecture:** {summary.get('best_model_name')}")
        st.write(f"**Decision Threshold:** {THRESHOLD}")
        st.write(f"**Evaluated Features:** {len(FEATURE_NAMES)} parameters")
        if "test_precision_malignant_selected_threshold" in summary:
            st.write(f"**Test Precision (Malignant):** {summary.get('test_precision_malignant_selected_threshold', 0) * 100:.1f}%")
        if "test_f1_selected_threshold" in summary:
            st.write(f"**Test F1 Score:** {summary.get('test_f1_selected_threshold', 0) * 100:.1f}%")
        if "cv_roc_auc" in summary:
            st.write(f"**CV ROC-AUC:** {summary.get('cv_roc_auc', 0) * 100:.2f}%")

st.markdown(
    '<div class="note"><strong>Disclaimer:</strong> In a clinical pipeline, these measurements are extracted automatically '
    'from digitized cell image analyses. This application serves as a demonstration of machine learning model '
    'performance for educational purposes only — not intended for clinical decision support.</div>',
    unsafe_allow_html=True,
)