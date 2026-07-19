"""
resume_schema.py
----------------
The canonical resume data schema shared by:
  - the "Upload Resume" flow (populated from a parsed PDF),
  - the "Build a Resume" flow (populated directly from form input), and
  - templates/ats_friendly.html (the Jinja2 export template).

This is a trimmed-down version of the schema from the original AI Resume
Builder project: the Indian-resume-specific fields (education, experience,
projects, skills, certifications, co-curricular) are preserved as-is since
they map directly onto the resume export template and onto the placement
model's input features. Job-description-matching fields (ATS score, target
JD, domain/sub-domain) were dropped since this merged app has no job
description to match against -- see README "Scope decisions" for why.
"""

from __future__ import annotations

from typing import Any


def get_default_resume_data() -> dict[str, Any]:
    """Return the canonical empty resume-data dictionary. This is the single
    source of truth for every field collected across the Resume Analyzer
    page, whichever of its two flows (Upload / Build) populates it."""
    return {
        # ── Personal info ────────────────────────────────────────────────
        "full_name": "",
        "email": "",
        "phone": "",
        "city": "",
        "state": "",
        "linkedin": "",
        "github": "",
        "portfolio": "",
        "dob": "",
        "gender": "",
        "languages_known": [],

        # ── Education ─────────────────────────────────────────────────────
        # Each entry: {level, institution, board_university, year_of_passing,
        #              percentage_cgpa, stream}
        "education": [],

        # ── Co-curricular / achievements ──────────────────────────────────
        "cocurricular": [],

        # ── Summary ───────────────────────────────────────────────────────
        "summary": "",

        # ── Experience / internships ──────────────────────────────────────
        # Each entry: {company, role, location, start_date, end_date, description}
        "experience": [],

        # ── Projects ──────────────────────────────────────────────────────
        # Each entry: {title, tech_stack, github_link, live_link, description}
        "projects": [],

        # ── Skills ────────────────────────────────────────────────────────
        "skills_technical": [],
        "skills_soft": [],
        "skills_tools": [],

        # ── Certifications ────────────────────────────────────────────────
        # Each entry: {name, issuer, year, url}
        "certifications": [],
    }


LIST_FIELDS = {
    "education", "experience", "projects", "certifications",
    "skills_technical", "skills_soft", "skills_tools",
    "cocurricular", "languages_known",
}


def sanitise_for_template(data: dict[str, Any]) -> dict[str, Any]:
    """Jinja2-safe copy: None -> "" for strings, None -> [] for list fields.
    (Adapted from the original AI Resume Builder's pdf_generator.py.)"""
    ctx: dict[str, Any] = {}
    for key, value in data.items():
        if key in LIST_FIELDS:
            ctx[key] = value if isinstance(value, list) else []
        elif value is None:
            ctx[key] = ""
        else:
            ctx[key] = value
    return ctx
