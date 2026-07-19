"""
pages/home.py
-------------
Landing page: project overview, feature highlights, tech stack, and a short
explanation of how the prediction pipeline works end to end.
"""

import streamlit as st

from config.settings import APP_NAME
from utils.model_utils import artifacts_ready, load_model_metrics
from utils.theme import placement_gauge, render_card, render_hero, render_section_label

render_hero(
    eyebrow="AI Resume Analyzer & Placement Prediction System",
    title=f"🎓 {APP_NAME}",
    subtitle=(
        "Predict your campus placement chances from your resume or your own inputs, "
        "understand exactly which factors matter, and get personalised, data-driven "
        "advice on how to improve them."
    ),
)

if not artifacts_ready():
    st.warning(
        "⚠️ No trained model found yet. Run **notebooks/Placement_Model_Training.ipynb** "
        "top to bottom first — it saves `models/placement_model.pkl` and the supporting "
        "files this app reads. Every page below will work once that's done.",
        icon="⚠️",
    )

# ── Quick stats row ──────────────────────────────────────────────────────────
metrics = load_model_metrics()
col1, col2, col3, col4 = st.columns(4)
with col1:
    render_card("Best Model", metrics.get("best_model_name", "—"), "Chosen by F1 Score", "🏆")
with col2:
    test_metrics = metrics.get("test_metrics", {})
    render_card("Test Accuracy", f"{test_metrics.get('Accuracy', 0):.1%}" if test_metrics else "—",
                "On held-out 20% split", "📈")
with col3:
    render_card("F1 Score", f"{test_metrics.get('F1 Score', 0):.3f}" if test_metrics else "—",
                "Primary selection metric", "⚖️")
with col4:
    split = metrics.get("train_test_split", {})
    total = split.get("train_size", 0) + split.get("test_size", 0)
    render_card("Students in Dataset", f"{total:,}" if total else "—", "Used to train the model", "🗂️")

st.write("")

# ── Two workflows ────────────────────────────────────────────────────────────
render_section_label("Two ways to get your prediction")
left, right = st.columns(2)

with left:
    st.markdown(
        """
        <div class="pm-card" style="min-height:190px;">
            <div class="pm-card-title" style="font-size:1.1rem;">📄 Upload or Build a Resume</div>
            <p class="pm-muted">
                Upload an existing PDF resume (parsed automatically — with AI-powered
                extraction if a Groq API key is configured, or a built-in rule-based
                parser otherwise), or fill a guided form to build one from scratch.
                Review and correct the extracted details, then predict.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.page_link("pages/resume_analyzer.py", label="Go to Resume Analyzer", icon="📄", width="stretch")

with right:
    st.markdown(
        """
        <div class="pm-card" style="min-height:190px;">
            <div class="pm-card-title" style="font-size:1.1rem;">🎯 Skip the Resume — Enter Details Directly</div>
            <p class="pm-muted">
                Don't have a resume ready? No problem. Enter your CGPA, skills, projects,
                internships, and a few other details directly and get an instant
                prediction — no upload required.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.page_link("pages/direct_prediction.py", label="Go to Direct Prediction", icon="🎯", width="stretch")

st.write("")

# ── How it works ─────────────────────────────────────────────────────────────
render_section_label("How the prediction works")
steps = st.columns(4)
step_content = [
    ("1️⃣", "Collect Profile", "From your resume or direct input: academics, skills, projects, internships, and more."),
    ("2️⃣", "Engineer & Encode", "The same pipeline used in training scales numbers and encodes categories — automatically."),
    ("3️⃣", "Predict", "A tuned Random-Forest-family model (chosen from 7 candidates) estimates your placement probability."),
    ("4️⃣", "Get Advice", "Your values are compared to placed students in the training data for targeted, actionable tips."),
]
for col, (icon, title, desc) in zip(steps, step_content):
    with col:
        st.markdown(
            f"""
            <div class="pm-card" style="min-height:170px;">
                <div style="font-size:1.4rem;">{icon}</div>
                <div class="pm-card-title" style="margin-top:0.3rem;">{title}</div>
                <p class="pm-muted" style="font-size:0.85rem;">{desc}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.write("")

# ── Illustrative gauge + tech stack ──────────────────────────────────────────
gauge_col, stack_col = st.columns([1, 1.3])
with gauge_col:
    render_section_label("Example output")
    st.plotly_chart(placement_gauge(0.68, "Sample Placement Probability"), width="stretch", config={"displayModeBar": False})
    st.caption("Illustrative example — your real result appears on the Resume Analyzer or Direct Prediction page.")

with stack_col:
    render_section_label("Technology stack")
    badges = [
        "Python", "Streamlit", "scikit-learn", "Pandas", "NumPy",
        "Matplotlib", "Seaborn", "Plotly", "Joblib", "pdfplumber / PyPDF2",
        "Jinja2", "Groq LLM (optional)",
    ]
    st.markdown(
        " ".join(
            f'<span class="pm-badge pm-badge-info" style="margin:3px;">{b}</span>' for b in badges
        ),
        unsafe_allow_html=True,
    )
    st.write("")
    st.markdown(
        """
        <div class="pm-card">
            <div class="pm-card-title">🧠 Why this matters</div>
            <p class="pm-muted">
                This system merges two ideas into one product: an AI resume analyzer
                and a placement prediction model, trained on a real dataset of Indian
                engineering students' academic, skill, and lifestyle profiles — so the
                advice you get is grounded in actual placement outcomes, not guesswork.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
