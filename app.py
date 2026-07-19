"""
app.py
------
Entry point. Streamlit executes this file on every rerun; it configures the
page, loads global CSS, defines the sidebar navigation, and hands off to
whichever page the user selected.

Run with:  streamlit run app.py
"""

import sys
from pathlib import Path

import streamlit as st

# Make `config` and `utils` importable from pages/*.py regardless of
# Streamlit's working directory quirks across versions/platforms.
sys.path.append(str(Path(__file__).resolve().parent))

from config.settings import APP_ICON, APP_NAME
from utils.theme import inject_css

st.set_page_config(
    page_title=f"{APP_NAME} — Placement Predictor",
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_css()

pages = [
    st.Page("pages/home.py", title="Home", icon="🏠", default=True),
    st.Page("pages/resume_analyzer.py", title="Resume Analyzer", icon="📄"),
    st.Page("pages/direct_prediction.py", title="Direct Prediction", icon="🎯"),
    st.Page("pages/model_performance.py", title="Model Performance", icon="📊"),
    st.Page("pages/about.py", title="About", icon="ℹ️"),
]

with st.sidebar:
    st.markdown(
        f"""
        <div style="padding: 0.4rem 0 1rem 0;">
            <div style="font-family:'Sora',sans-serif; font-weight:700; font-size:1.25rem; color:#e5e9f5;">
                {APP_ICON} {APP_NAME}
            </div>
            <div style="color:#8b94ad; font-size:0.8rem; margin-top:0.15rem;">
                AI Resume Analyzer &amp; Placement Predictor
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

navigation = st.navigation(pages, position="sidebar")

with st.sidebar:
    st.markdown('<hr class="pm-divider">', unsafe_allow_html=True)
    st.caption("Built with Streamlit + scikit-learn")

navigation.run()
