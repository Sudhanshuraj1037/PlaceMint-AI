"""
pages/direct_prediction.py
---------------------------
Workflow 2: the student skips the resume entirely and enters their profile
directly. Core spec-named fields are shown prominently; every other model
feature is available in an "Additional factors" expander, defaulted from
the training data.
"""

import time

import streamlit as st

from config.settings import FEATURE_DISPLAY_NAMES
from utils import llm_engine
from utils.model_utils import load_feature_config, predict_placement
from utils.suggestions import generate_suggestions
from utils.theme import placement_gauge, render_badge, render_hero, render_section_label, render_suggestions

render_hero(
    eyebrow="Workflow 2 — No Resume Needed",
    title="🎯 Direct Placement Prediction",
    subtitle="Enter your profile directly and get an instant, data-driven placement prediction.",
)

config = load_feature_config()
if not config:
    st.error(
        "No trained model found. Run **notebooks/Placement_Model_Training.ipynb** first, "
        "then reload this page."
    )
    st.stop()

ranges = config["numeric_ranges"]
defaults = config["numeric_defaults"]
cat_options = config["categorical_options"]
cat_defaults = config["categorical_defaults"]


def dname(key: str) -> str:
    return FEATURE_DISPLAY_NAMES.get(key, key.replace("_", " ").title())


with st.form("direct_prediction_form"):
    render_section_label("Core profile")
    c1, c2 = st.columns(2)
    with c1:
        cgpa = st.slider(dname("cgpa"), *ranges["cgpa"], value=defaults["cgpa"], step=0.05)
        coding = st.slider(
            "Programming / Technical Skills (1-5)", 1, 5, int(defaults["coding_skill_rating"]),
            help="Combines your programming ability and general technical skill level.",
        )
        communication = st.slider(
            "Communication / Soft Skills (1-5)", 1, 5, int(defaults["communication_skill_rating"]),
            help="Your communication ability and general interpersonal / soft skills.",
        )
        aptitude = st.slider(
            "Aptitude / Logical Reasoning (1-5)", 1, 5, int(defaults["aptitude_skill_rating"]),
            help="Quantitative aptitude and logical-reasoning ability (as in placement aptitude tests).",
        )
    with c2:
        projects = st.number_input(dname("projects_completed"), *[int(v) for v in ranges["projects_completed"]],
                                    value=int(defaults["projects_completed"]))
        internships = st.number_input(dname("internships_completed"), *[int(v) for v in ranges["internships_completed"]],
                                       value=int(defaults["internships_completed"]))
        certifications = st.number_input(dname("certifications_count"), *[int(v) for v in ranges["certifications_count"]],
                                          value=int(defaults["certifications_count"]))
        extracurricular = st.selectbox(
            dname("extracurricular_involvement"),
            [o for o in cat_options["extracurricular_involvement"] if o != "Unknown"],
            index=[o for o in cat_options["extracurricular_involvement"] if o != "Unknown"].index(
                cat_defaults["extracurricular_involvement"]
            ) if cat_defaults["extracurricular_involvement"] != "Unknown" else 0,
        )

    with st.expander("➕ Additional factors (optional — improves prediction accuracy)"):
        a1, a2, a3 = st.columns(3)
        with a1:
            st.markdown("**Academics**")
            tenth = st.slider(dname("tenth_percentage"), *ranges["tenth_percentage"], value=defaults["tenth_percentage"])
            twelfth = st.slider(dname("twelfth_percentage"), *ranges["twelfth_percentage"], value=defaults["twelfth_percentage"])
            backlogs = st.number_input(dname("backlogs"), *[int(v) for v in ranges["backlogs"]], value=int(defaults["backlogs"]))
            attendance = st.slider(dname("attendance_percentage"), *ranges["attendance_percentage"], value=defaults["attendance_percentage"])
        with a2:
            st.markdown("**Activity & Lifestyle**")
            study_hours = st.slider(dname("study_hours_per_day"), *ranges["study_hours_per_day"], value=defaults["study_hours_per_day"])
            hackathons = st.number_input(dname("hackathons_participated"), *[int(v) for v in ranges["hackathons_participated"]],
                                          value=int(defaults["hackathons_participated"]))
            sleep_hours = st.slider(dname("sleep_hours"), *ranges["sleep_hours"], value=defaults["sleep_hours"])
            stress = st.slider(dname("stress_level"), *[int(v) for v in ranges["stress_level"]], value=int(defaults["stress_level"]))
        with a3:
            st.markdown("**Background**")
            gender = st.selectbox(dname("gender"), cat_options["gender"], index=cat_options["gender"].index(cat_defaults["gender"]))
            branch = st.selectbox(dname("branch"), cat_options["branch"], index=cat_options["branch"].index(cat_defaults["branch"]))
            part_time = st.selectbox(dname("part_time_job"), cat_options["part_time_job"],
                                      index=cat_options["part_time_job"].index(cat_defaults["part_time_job"]))
            family_income = st.selectbox(dname("family_income_level"), cat_options["family_income_level"],
                                          index=cat_options["family_income_level"].index(cat_defaults["family_income_level"]))
            city_tier = st.selectbox(dname("city_tier"), cat_options["city_tier"],
                                      index=cat_options["city_tier"].index(cat_defaults["city_tier"]))
            internet = st.selectbox(dname("internet_access"), cat_options["internet_access"],
                                     index=cat_options["internet_access"].index(cat_defaults["internet_access"]))

    ai_advice_requested = st.checkbox(
        "✨ Also generate AI coaching advice (requires a GROQ_API_KEY in .env)",
        value=llm_engine.is_available(), disabled=not llm_engine.is_available(),
        help="Enable this in .env to unlock a personalised coaching paragraph on top of the data-driven tips below."
             if not llm_engine.is_available() else None,
    )
    submitted = st.form_submit_button("🚀 Predict My Placement Chance", width="stretch", type="primary")

if submitted:
    features = {
        "cgpa": cgpa, "coding_skill_rating": coding, "communication_skill_rating": communication,
        "aptitude_skill_rating": aptitude, "projects_completed": projects, "internships_completed": internships,
        "certifications_count": certifications, "extracurricular_involvement": extracurricular,
        "tenth_percentage": tenth, "twelfth_percentage": twelfth, "backlogs": backlogs,
        "attendance_percentage": attendance, "study_hours_per_day": study_hours,
        "hackathons_participated": hackathons, "sleep_hours": sleep_hours, "stress_level": stress,
        "gender": gender, "branch": branch, "part_time_job": part_time,
        "family_income_level": family_income, "city_tier": city_tier, "internet_access": internet,
    }
    with st.spinner("Analyzing your profile against 5,000 real student outcomes..."):
        time.sleep(0.6)
        result = predict_placement(features)
    st.session_state["direct_prediction_result"] = result
    st.session_state["direct_prediction_features"] = features
    st.session_state["direct_prediction_want_ai"] = ai_advice_requested

# ── Results ──────────────────────────────────────────────────────────────────
if "direct_prediction_result" in st.session_state:
    result = st.session_state["direct_prediction_result"]
    features = st.session_state["direct_prediction_features"]

    st.write("")
    render_section_label("Your result")
    res_col1, res_col2 = st.columns([1, 1.4])

    with res_col1:
        st.plotly_chart(placement_gauge(result["placed_probability"]), width="stretch",
                         config={"displayModeBar": False})

    with res_col2:
        if result["prediction"] == "Placed":
            st.markdown(render_badge("Predicted: Placed", "success"), unsafe_allow_html=True)
        else:
            st.markdown(render_badge("Predicted: Not Placed (yet)", "warning"), unsafe_allow_html=True)
        st.write("")
        st.metric("Confidence", f"{result['confidence']:.1%}")
        st.progress(result["placed_probability"])
        st.caption(
            f"Placed probability: {result['placed_probability']:.1%} · "
            f"Not placed probability: {result['not_placed_probability']:.1%}"
        )
        if result["prediction"] == "Placed" and result["confidence"] > 0.75:
            st.balloons()

    st.write("")
    render_section_label("Personalised suggestions")
    suggestions = generate_suggestions(features)
    if suggestions:
        render_suggestions(suggestions)
    else:
        st.info("No specific suggestions to show — your profile looks well-rounded across the board!")

    if st.session_state.get("direct_prediction_want_ai") and llm_engine.is_available():
        with st.spinner("Asking the AI coach for a personalised note..."):
            try:
                advice = llm_engine.generate_improvement_advice(features, result, suggestions)
                st.markdown(
                    f'<div class="pm-card"><div class="pm-card-title">✨ AI Coach</div>'
                    f'<p style="line-height:1.6;">{advice}</p></div>',
                    unsafe_allow_html=True,
                )
            except Exception as exc:
                st.info(f"AI coaching note unavailable right now ({exc}). The suggestions above still apply.")
