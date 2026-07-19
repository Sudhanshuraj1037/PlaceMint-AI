"""
config/settings.py
-------------------
Single source of truth for filesystem paths, app-wide constants, and small
display metadata used across every page. Keeping these here (instead of
scattering literal paths through the codebase) means the app can be moved
or renamed without hunting through every file.
"""

# from __future__ import annotations

# from pathlib import Path

# import os
# from dotenv import load_dotenv

from __future__ import annotations

import os
import pathlib
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ── Filesystem layout ────────────────────────────────────────────────────────

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
ASSETS_DIR = BASE_DIR / "assets"
TEMPLATES_DIR = BASE_DIR / "templates"

MODEL_PATH = MODELS_DIR / "placement_model.pkl"
FEATURE_CONFIG_PATH = MODELS_DIR / "feature_config.json"
MODEL_METRICS_PATH = MODELS_DIR / "model_metrics.json"
DATASET_PATH = DATA_DIR / "placement_data.csv"
STYLE_CSS_PATH = ASSETS_DIR / "style.css"

# ── App metadata ─────────────────────────────────────────────────────────────

APP_NAME = "PlaceMint AI"
APP_TAGLINE = "AI Resume Analyzer & Placement Prediction System"
APP_ICON = "🎓"

# ── Branch / label display names ─────────────────────────────────────────────

BRANCH_LABELS = {
    "CSE": "Computer Science (CSE)",
    "IT": "Information Technology (IT)",
    "ECE": "Electronics & Communication (ECE)",
    "ME": "Mechanical Engineering (ME)",
    "CE": "Civil Engineering (CE)",
}

FEATURE_DISPLAY_NAMES = {
    "cgpa": "CGPA",
    "tenth_percentage": "10th Percentage",
    "twelfth_percentage": "12th Percentage",
    "backlogs": "Active Backlogs",
    "study_hours_per_day": "Study Hours / Day",
    "attendance_percentage": "Attendance %",
    "projects_completed": "Projects Completed",
    "internships_completed": "Internships Completed",
    "coding_skill_rating": "Programming / Technical Skill",
    "communication_skill_rating": "Communication / Soft Skills",
    "aptitude_skill_rating": "Aptitude / Logical Reasoning",
    "hackathons_participated": "Hackathons Participated",
    "certifications_count": "Certifications",
    "sleep_hours": "Sleep Hours / Night",
    "stress_level": "Stress Level",
    "gender": "Gender",
    "branch": "Branch",
    "part_time_job": "Part-time Job",
    "family_income_level": "Family Income Level",
    "city_tier": "City Tier",
    "internet_access": "Internet Access",
    "extracurricular_involvement": "Extracurricular Involvement",
    "overall_academic_score": "Overall Academic Score",
    "practical_experience_score": "Practical Experience Score",
    "soft_skill_index": "Soft Skill Index",
}

# Core fields shown prominently on the Direct Prediction / Resume Analyzer
# review forms. Every other model feature still exists in an "Additional
# factors" expander, defaulted from the training data (see feature_config.json).
CORE_FIELDS = [
    "cgpa",
    "communication_skill_rating",
    "aptitude_skill_rating",
    "coding_skill_rating",
    "internships_completed",
    "projects_completed",
    "certifications_count",
    "extracurricular_involvement",
]

GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_FALLBACK_MODEL = "llama-3.1-8b-instant"
