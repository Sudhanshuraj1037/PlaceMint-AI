"""
pages/about.py
---------------
Student/project placeholders, a short "how this was built" story, and the
full technology stack. Edit the PLACEHOLDER_* constants below with your
own details before submitting.
"""

import streamlit as st

from utils.theme import render_hero, render_section_label

# ── Edit these before submitting your project ────────────────────────────────
PLACEHOLDER_STUDENT_NAME = "Hari Om, Preet Bana, Mridul Vatsal and Sudhanshu Raj"
PLACEHOLDER_ROLL_NUMBER = "5000 students from Lovely Professional University"
PLACEHOLDER_COLLEGE = "Lovely Professional University, Phagwara, Punjab"
PLACEHOLDER_DEPARTMENT = "Department of Computer Science & Engineering"
PLACEHOLDER_GUIDE = " Sir Anzar Hussain Lone"
PLACEHOLDER_YEAR = "2025 - 2026"

render_hero(
    eyebrow="About This Project",
    title="ℹ️ Project Information",
    subtitle="AI Resume Analyzer & Placement Prediction System — a university machine learning project.",
)

col1, col2 = st.columns([1, 1.2])

with col1:
    render_section_label("Submitted by")
    st.markdown(
        f"""
        <div class="pm-card">
            <p style="line-height:2.0; margin:0;">
                <b>Student:</b> {PLACEHOLDER_STUDENT_NAME}<br>
                <b>Dataset:</b> {PLACEHOLDER_ROLL_NUMBER}<br>
                <b>College:</b> {PLACEHOLDER_COLLEGE}<br>
                <b>Department:</b> {PLACEHOLDER_DEPARTMENT}<br>
                <b>Project Guide:</b> {PLACEHOLDER_GUIDE}<br>
                <b>Academic Year:</b> {PLACEHOLDER_YEAR}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("✏️ Edit these placeholders in `pages/about.py` before submitting your report.")

with col2:
    render_section_label("What this project does")
    st.markdown(
        """
        <div class="pm-card">
            <p class="pm-muted" style="line-height:1.7;">
                This system predicts whether an engineering student is likely to be
                placed during campus recruitment, using a machine learning model
                trained on academic performance, skills, projects, internships, and
                lifestyle factors. It merges two ideas into one product:
            </p>
            <ul class="pm-muted" style="line-height:1.8;">
                <li>An <b>AI Resume Analyzer</b> that reads an uploaded resume (or a
                resume built from scratch inside the app) and extracts a structured profile.</li>
                <li>A <b>Placement Prediction Model</b>, compared across 7 algorithms and
                tuned for the best F1 Score, that turns that profile into a probability
                and a set of personalised, data-driven improvement suggestions.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")
render_section_label("Technology stack")

stack_groups = [
    ("Machine Learning", ["scikit-learn", "Pandas", "NumPy", "Joblib"], "🧠"),
    ("Visualization", ["Matplotlib", "Seaborn", "Plotly"], "📊"),
    ("Application", ["Streamlit", "Jinja2"], "🖥️"),
    ("Resume Processing", ["pdfplumber", "PyPDF2", "Regex-based extraction"], "📄"),
    ("Optional AI Layer", ["Groq API (Llama 3.3 70B)"], "✨"),
]
cols = st.columns(len(stack_groups))
for col, (group, items, icon) in zip(cols, stack_groups):
    with col:
        st.markdown(
            f"""
            <div class="pm-card" style="min-height:150px;">
                <div class="pm-card-title">{icon} {group}</div>
                <p class="pm-muted" style="font-size:0.85rem;">{"<br>".join(items)}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.write("")
render_section_label("How this project was built")
st.markdown(
    """
    <div class="pm-card">
        <p class="pm-muted" style="line-height:1.75;">
            This project merges two earlier prototypes into one product: an
            <b>AI Resume Builder</b> (PDF parsing, LLM-based extraction, and resume
            export) and a <b>Student Placement Predictor</b> (a dataset of engineering
            student profiles with placement outcomes). Rather than running side by
            side, they now share one trained model, one feature pipeline, and one
            dashboard — a resume (uploaded or built in-app) and manually entered
            details both flow through the exact same prediction and suggestion engine.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# with st.expander("📌 Scope decisions (for your viva)"):
#     st.markdown(
#         """
# - **ATS-score / skill-gap / interview-prep features** from the original resume
#   builder were left out of this merged app — they compare a resume against a
#   *specific job description*, which this placement-prediction workflow doesn't
#   collect. The resume *parsing* and *export* pieces were kept and reused.
# - **PDF export uses a graceful fallback.** WeasyPrint needs system-level
#   libraries (Pango/Cairo) that aren't guaranteed on every machine (especially
#   Windows), so resume export always produces a styled HTML file, and adds a
#   one-click PDF download only when a working PDF engine is detected.
# - **Feature engineering lives inside the saved model pipeline itself**
#   (`utils/feature_engineering.py`), not duplicated between the notebook and
#   the app, so training and serving can never drift out of sync.
# - **Suggestions are grounded in the training data**, not invented: each tip
#   compares your value for a feature to the *Placed* cohort's statistics,
#   weighted by that feature's permutation importance from the notebook.
#         """
#     )

st.write("")
st.caption(
    "Built as a university machine learning project. Dataset: engineering student "
    "placement records (academics, skills, internships, projects, lifestyle factors)."
)
