"""
pages/resume_analyzer.py
--------------------------
Workflow 1: "the student uploads OR creates a resume."

Tab 1 (Upload): parse an existing PDF resume (AI-powered via Groq if a
key is configured, otherwise a dependency-free regex/heuristic parser),
let the student review and correct the extracted values, then predict.

Tab 2 (Build): a guided form builds a resume from scratch. The same data
feeds the placement model AND can be exported as a polished resume
(HTML always; PDF too if a PDF engine is available on this machine).

Both tabs converge on the same underlying feature dict and the same
prediction + suggestions rendering, via utils.model_utils / utils.suggestions.
"""

import time

import pandas as pd
import streamlit as st

from config.settings import FEATURE_DISPLAY_NAMES
from utils import llm_engine
from utils.model_utils import load_feature_config, predict_placement
from utils.pdf_generator import generate_resume
from utils.resume_parser import extract_text_from_pdf, map_resume_to_features, parse_resume_text_heuristic
from utils.resume_schema import get_default_resume_data
from utils.suggestions import generate_suggestions
from utils.theme import placement_gauge, render_badge, render_hero, render_section_label, render_suggestions

render_hero(
    eyebrow="Workflow 1 — Upload or Create",
    title="📄 AI Resume Analyzer",
    subtitle="Upload an existing resume, or build one from scratch — either way, "
             "we'll extract your profile, let you fine-tune it, and predict your placement chances.",
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


def render_review_form_and_predict(extracted: dict, key_prefix: str) -> None:
    """Shared editable review form: pre-filled from `extracted`, backed by
    dataset defaults for anything not provided. Used by both tabs."""
    st.info(
        "✏️ **Review and correct the values below** — automatic extraction is a helpful "
        "starting point, not a guarantee. A few fields (like Aptitude Score) usually "
        "can't be read from a resume at all, so please fill those in yourself.",
        icon="✏️",
    )

    def val(key, is_int=False):
        v = extracted.get(key, defaults.get(key, cat_defaults.get(key)))
        if v is not None and key in ranges:  # clamp -- extraction can overshoot (e.g. bullet-heavy resumes)
            lo, hi = ranges[key]
            v = max(lo, min(hi, v))
        return int(v) if is_int and v is not None else v

    with st.form(f"{key_prefix}_review_form"):
        render_section_label("Core profile")
        c1, c2 = st.columns(2)
        with c1:
            cgpa = st.slider(dname("cgpa"), *ranges["cgpa"], value=float(val("cgpa")), step=0.05, key=f"{key_prefix}_cgpa")
            coding = st.slider("Programming / Technical Skills (1-5)", 1, 5, val("coding_skill_rating", True),
                               key=f"{key_prefix}_coding")
            communication = st.slider("Communication / Soft Skills (1-5)", 1, 5, val("communication_skill_rating", True),
                                       key=f"{key_prefix}_comm")
            aptitude = st.slider("Aptitude / Logical Reasoning (1-5)", 1, 5, val("aptitude_skill_rating", True),
                                  key=f"{key_prefix}_apt",
                                  help="Rarely stated on a resume — set this from any aptitude test scores you have.")
        with c2:
            projects = st.number_input(dname("projects_completed"), int(ranges["projects_completed"][0]),
                                        int(ranges["projects_completed"][1]), val("projects_completed", True),
                                        key=f"{key_prefix}_proj")
            internships = st.number_input(dname("internships_completed"), int(ranges["internships_completed"][0]),
                                           int(ranges["internships_completed"][1]), val("internships_completed", True),
                                           key=f"{key_prefix}_intern")
            certifications = st.number_input(dname("certifications_count"), int(ranges["certifications_count"][0]),
                                              int(ranges["certifications_count"][1]), val("certifications_count", True),
                                              key=f"{key_prefix}_cert")
            ec_options = [o for o in cat_options["extracurricular_involvement"] if o != "Unknown"]
            ec_value = extracted.get("extracurricular_involvement", cat_defaults["extracurricular_involvement"])
            if ec_value not in ec_options:
                ec_value = ec_options[0]
            extracurricular = st.selectbox(dname("extracurricular_involvement"), ec_options,
                                            index=ec_options.index(ec_value), key=f"{key_prefix}_ec")

        with st.expander("➕ Additional factors (optional — improves prediction accuracy)"):
            a1, a2, a3 = st.columns(3)
            with a1:
                st.markdown("**Academics**")
                tenth = st.slider(dname("tenth_percentage"), *ranges["tenth_percentage"], value=float(val("tenth_percentage")),
                                   key=f"{key_prefix}_10th")
                twelfth = st.slider(dname("twelfth_percentage"), *ranges["twelfth_percentage"], value=float(val("twelfth_percentage")),
                                     key=f"{key_prefix}_12th")
                backlogs = st.number_input(dname("backlogs"), int(ranges["backlogs"][0]), int(ranges["backlogs"][1]),
                                            val("backlogs", True), key=f"{key_prefix}_backlogs")
                attendance = st.slider(dname("attendance_percentage"), *ranges["attendance_percentage"],
                                        value=float(val("attendance_percentage")), key=f"{key_prefix}_att")
            with a2:
                st.markdown("**Activity & Lifestyle**")
                study_hours = st.slider(dname("study_hours_per_day"), *ranges["study_hours_per_day"],
                                         value=float(val("study_hours_per_day")), key=f"{key_prefix}_study")
                hackathons = st.number_input(dname("hackathons_participated"), int(ranges["hackathons_participated"][0]),
                                              int(ranges["hackathons_participated"][1]), val("hackathons_participated", True),
                                              key=f"{key_prefix}_hack")
                sleep_hours = st.slider(dname("sleep_hours"), *ranges["sleep_hours"], value=float(val("sleep_hours")),
                                         key=f"{key_prefix}_sleep")
                stress = st.slider(dname("stress_level"), int(ranges["stress_level"][0]), int(ranges["stress_level"][1]),
                                    val("stress_level", True), key=f"{key_prefix}_stress")
            with a3:
                st.markdown("**Background**")
                gender = st.selectbox(dname("gender"), cat_options["gender"],
                                       index=cat_options["gender"].index(extracted.get("gender", cat_defaults["gender"])
                                                                          if extracted.get("gender") in cat_options["gender"]
                                                                          else cat_defaults["gender"]),
                                       key=f"{key_prefix}_gender")
                branch_val = extracted.get("branch", cat_defaults["branch"])
                if branch_val not in cat_options["branch"]:
                    branch_val = cat_defaults["branch"]
                branch = st.selectbox(dname("branch"), cat_options["branch"], index=cat_options["branch"].index(branch_val),
                                       key=f"{key_prefix}_branch")
                part_time = st.selectbox(dname("part_time_job"), cat_options["part_time_job"],
                                          index=cat_options["part_time_job"].index(cat_defaults["part_time_job"]),
                                          key=f"{key_prefix}_pt")
                family_income = st.selectbox(dname("family_income_level"), cat_options["family_income_level"],
                                              index=cat_options["family_income_level"].index(cat_defaults["family_income_level"]),
                                              key=f"{key_prefix}_income")
                city_tier = st.selectbox(dname("city_tier"), cat_options["city_tier"],
                                          index=cat_options["city_tier"].index(cat_defaults["city_tier"]),
                                          key=f"{key_prefix}_city")
                internet = st.selectbox(dname("internet_access"), cat_options["internet_access"],
                                         index=cat_options["internet_access"].index(cat_defaults["internet_access"]),
                                         key=f"{key_prefix}_net")

        ai_advice_requested = st.checkbox(
            "✨ Also generate AI coaching advice (requires a GROQ_API_KEY in .env)",
            value=llm_engine.is_available(), disabled=not llm_engine.is_available(),
            key=f"{key_prefix}_ai_toggle",
        )
        go = st.form_submit_button("🚀 Predict My Placement Chance", width="stretch", type="primary")

    if go:
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
        st.session_state[f"{key_prefix}_result"] = result
        st.session_state[f"{key_prefix}_features"] = features
        st.session_state[f"{key_prefix}_want_ai"] = ai_advice_requested

    if f"{key_prefix}_result" in st.session_state:
        result = st.session_state[f"{key_prefix}_result"]
        features = st.session_state[f"{key_prefix}_features"]

        st.write("")
        render_section_label("Your result")
        rc1, rc2 = st.columns([1, 1.4])
        with rc1:
            st.plotly_chart(placement_gauge(result["placed_probability"]), width="stretch",
                             config={"displayModeBar": False}, key=f"{key_prefix}_gauge")
        with rc2:
            if result["prediction"] == "Placed":
                st.markdown(render_badge("Predicted: Placed", "success"), unsafe_allow_html=True)
            else:
                st.markdown(render_badge("Predicted: Not Placed (yet)", "warning"), unsafe_allow_html=True)
            st.write("")
            st.metric("Confidence", f"{result['confidence']:.1%}")
            st.progress(result["placed_probability"])
            if result["prediction"] == "Placed" and result["confidence"] > 0.75:
                st.balloons()

        st.write("")
        render_section_label("Personalised suggestions")
        suggestions = generate_suggestions(features)
        if suggestions:
            render_suggestions(suggestions)
        else:
            st.info("No specific suggestions to show — your profile looks well-rounded across the board!")

        if st.session_state.get(f"{key_prefix}_want_ai") and llm_engine.is_available():
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


tab_upload, tab_build = st.tabs(["📤 Upload an Existing Resume", "✍️ Build a Resume From Scratch"])

# ═════════════════════════════════════════════════════════════════════════
# TAB 1 — UPLOAD
# ═════════════════════════════════════════════════════════════════════════
with tab_upload:
    st.markdown("Upload a **PDF resume** and we'll extract your profile automatically.")
    if llm_engine.is_available():
        st.markdown(render_badge("AI-powered extraction active (Groq)", "success"), unsafe_allow_html=True)
    else:
        st.markdown(
            render_badge("Using built-in rule-based extraction (add a GROQ_API_KEY in .env for AI-powered extraction)", "info"),
            unsafe_allow_html=True,
        )

    uploaded_file = st.file_uploader("Upload resume (PDF)", type=["pdf"], key="resume_pdf_uploader")

    if uploaded_file is not None and st.button("🔍 Extract Information", type="primary"):
        with st.spinner("Reading your resume..."):
            try:
                raw_text = extract_text_from_pdf(uploaded_file)
            except Exception as exc:
                st.error(f"Couldn't read that PDF: {exc}")
                raw_text = None

            if raw_text:
                resume_data = None
                extraction_method = "rule-based"
                if llm_engine.is_available():
                    try:
                        resume_data = llm_engine.parse_resume_to_json(raw_text)
                        extraction_method = "AI (Groq)"
                    except Exception as exc:
                        st.warning(f"AI extraction failed ({exc}) — falling back to rule-based extraction.")
                if resume_data is None:
                    resume_data = parse_resume_text_heuristic(raw_text)

                mapped_features = map_resume_to_features(resume_data)
                st.session_state["upload_resume_data"] = resume_data
                st.session_state["upload_mapped_features"] = mapped_features
                st.session_state["upload_extraction_method"] = extraction_method
                st.session_state["upload_parse_version"] = st.session_state.get("upload_parse_version", 0) + 1
                # Clear any previous prediction so stale results don't linger.
                st.session_state.pop("resume_upload_result", None)

    if "upload_mapped_features" in st.session_state:
        st.success(f"✅ Extraction complete (method: {st.session_state['upload_extraction_method']}).", icon="✅")

        resume_data = st.session_state["upload_resume_data"]
        with st.expander("📋 See everything we extracted from your resume", expanded=False):
            ec1, ec2 = st.columns(2)
            with ec1:
                st.markdown(f"**Name:** {resume_data.get('full_name') or '—'}")
                st.markdown(f"**Email:** {resume_data.get('email') or '—'}")
                st.markdown(f"**Phone:** {resume_data.get('phone') or '—'}")
                st.markdown(f"**Education entries found:** {len(resume_data.get('education', []))}")
                st.markdown(f"**Projects found:** {len(resume_data.get('projects', []))}")
            with ec2:
                st.markdown(f"**Experience/Internship entries found:** {len(resume_data.get('experience', []))}")
                st.markdown(f"**Certifications found:** {len(resume_data.get('certifications', []))}")
                st.markdown(f"**Technical skills found:** {', '.join(resume_data.get('skills_technical', [])) or '—'}")
                st.markdown(f"**Soft skills found:** {', '.join(resume_data.get('skills_soft', [])) or '—'}")

        render_review_form_and_predict(
            st.session_state["upload_mapped_features"],
            key_prefix=f"resume_upload_v{st.session_state.get('upload_parse_version', 0)}",
        )

# ═════════════════════════════════════════════════════════════════════════
# TAB 2 — BUILD
# ═════════════════════════════════════════════════════════════════════════
with tab_build:
    st.markdown("Don't have a resume yet? Fill this in and we'll build one **and** predict your placement chances.")

    if "build_resume_data" not in st.session_state:
        st.session_state["build_resume_data"] = get_default_resume_data()

    render_section_label("Personal information")
    p1, p2, p3 = st.columns(3)
    with p1:
        full_name = st.text_input("Full name", key="build_name")
        email = st.text_input("Email", key="build_email")
    with p2:
        phone = st.text_input("Phone", key="build_phone")
        city = st.text_input("City", key="build_city")
    with p3:
        linkedin = st.text_input("LinkedIn URL", key="build_linkedin")
        github = st.text_input("GitHub URL", key="build_github")

    render_section_label("Education")
    education_df = st.data_editor(
        pd.DataFrame([{"level": "UG", "institution": "", "board_university": "", "stream": "",
                        "percentage_cgpa": "", "year_of_passing": ""}]),
        num_rows="dynamic", key="build_education_editor", width="stretch",
        column_config={
            "level": st.column_config.SelectboxColumn(options=["10th", "12th", "Diploma", "UG", "PG"]),
        },
    )

    render_section_label("Projects")
    projects_df = st.data_editor(
        pd.DataFrame([{"title": "", "tech_stack": "", "description": ""}]),
        num_rows="dynamic", key="build_projects_editor", width="stretch",
    )

    render_section_label("Internships / Work Experience")
    experience_df = st.data_editor(
        pd.DataFrame([{"role": "", "company": "", "start_date": "", "end_date": "", "description": ""}]),
        num_rows="dynamic", key="build_experience_editor", width="stretch",
    )

    render_section_label("Certifications")
    certifications_df = st.data_editor(
        pd.DataFrame([{"name": "", "issuer": "", "year": ""}]),
        num_rows="dynamic", key="build_certifications_editor", width="stretch",
    )

    render_section_label("Skills & Activities")
    s1, s2, s3 = st.columns(3)
    with s1:
        technical_skills = st.text_area("Technical skills (comma-separated)", key="build_tech_skills",
                                         placeholder="Python, SQL, React, AWS")
    with s2:
        soft_skills = st.text_area("Soft skills (comma-separated)", key="build_soft_skills",
                                    placeholder="Communication, Leadership, Teamwork")
    with s3:
        cocurricular = st.text_area("Co-curricular / achievements (one per line)", key="build_cocurricular",
                                     placeholder="Won inter-college hackathon\nClub secretary")

    summary_col1, summary_col2 = st.columns([3, 1])
    with summary_col1:
        summary = st.text_area("Professional summary", key="build_summary", height=90)
    with summary_col2:
        st.write("")
        st.write("")
        if st.button("✨ AI-generate summary", disabled=not llm_engine.is_available(), width="stretch"):
            draft = {
                "full_name": full_name,
                "skills_technical": [s.strip() for s in technical_skills.split(",") if s.strip()],
                "projects": projects_df.to_dict(orient="records"),
                "experience": experience_df.to_dict(orient="records"),
            }
            try:
                with st.spinner("Writing your summary..."):
                    st.session_state["build_summary"] = llm_engine.generate_resume_summary(draft)
                st.rerun()
            except Exception as exc:
                st.warning(f"Couldn't generate a summary right now ({exc}).")
        if not llm_engine.is_available():
            st.caption("Requires a GROQ_API_KEY in .env")

    build_clicked = st.button("🚀 Build Resume & Predict Placement", type="primary", width="stretch",
                               key="build_and_predict_btn")

    if build_clicked:
        resume_data = {
            "full_name": full_name, "email": email, "phone": phone, "city": city, "state": "",
            "linkedin": linkedin, "github": github, "portfolio": "", "dob": "", "gender": "",
            "languages_known": [],
            "education": [r for r in education_df.to_dict(orient="records") if any(str(v).strip() for v in r.values())],
            "cocurricular": [ln.strip() for ln in cocurricular.split("\n") if ln.strip()],
            "summary": summary,
            "experience": [r for r in experience_df.to_dict(orient="records") if any(str(v).strip() for v in r.values())],
            "projects": [r for r in projects_df.to_dict(orient="records") if any(str(v).strip() for v in r.values())],
            "skills_technical": [s.strip() for s in technical_skills.split(",") if s.strip()],
            "skills_soft": [s.strip() for s in soft_skills.split(",") if s.strip()],
            "skills_tools": [],
            "certifications": [r for r in certifications_df.to_dict(orient="records") if any(str(v).strip() for v in r.values())],
        }
        st.session_state["build_resume_data"] = resume_data
        st.session_state["build_mapped_features"] = map_resume_to_features(resume_data)
        st.session_state["build_parse_version"] = st.session_state.get("build_parse_version", 0) + 1
        st.session_state.pop("resume_build_result", None)

        with st.spinner("Rendering your resume..."):
            try:
                export = generate_resume(resume_data)
                st.session_state["build_resume_export"] = export
            except Exception as exc:
                st.warning(f"Resume rendering failed ({exc}); you can still get your prediction below.")
                st.session_state["build_resume_export"] = None

    if "build_mapped_features" in st.session_state:
        render_review_form_and_predict(
            st.session_state["build_mapped_features"],
            key_prefix=f"resume_build_v{st.session_state.get('build_parse_version', 0)}",
        )

        export = st.session_state.get("build_resume_export")
        if export:
            st.write("")
            render_section_label("Your generated resume")
            with st.expander("📄 Preview resume", expanded=False):
                st.iframe(export["html"], height=700)

            dl1, dl2 = st.columns(2)
            with dl1:
                st.download_button("⬇️ Download as HTML", export["html"], file_name="resume.html",
                                    mime="text/html", width="stretch")
            with dl2:
                if export["pdf_engine_available"]:
                    st.download_button("⬇️ Download as PDF", export["pdf"], file_name="resume.pdf",
                                        mime="application/pdf", width="stretch")
                else:
                    st.caption(
                        "PDF export needs WeasyPrint's system libraries, which aren't installed here. "
                        "Download the HTML above and print to PDF from your browser (Ctrl+P), or "
                        "`pip install weasyprint` with its system dependencies (see README)."
                    )
