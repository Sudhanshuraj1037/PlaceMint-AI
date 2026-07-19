"""
resume_parser.py
----------------
Three responsibilities:

1. `extract_text_from_pdf`   -- pull plain text out of an uploaded PDF
   (pdfplumber preferred, PyPDF2 fallback). Adapted from the original AI
   Resume Builder project's utils.py -- this function needed no changes,
   it was already well-tested and dependency-free of Streamlit specifics.

2. `parse_resume_text_heuristic` -- a regex/keyword-based extractor that
   turns raw resume text into a `resume_data` dict (see resume_schema.py)
   WITHOUT needing any API key. This is what runs when no Groq API key is
   configured, or when the AI parse fails for any reason -- the app always
   produces *something* to review and edit, never a hard failure.

3. `map_resume_to_features` -- converts a resume_data dict (from either the
   heuristic parser, the LLM parser in llm_engine.py, or the "Build a
   Resume" form) into the placement model's raw feature dict. Because the
   Direct Prediction page's review form always lets the student see and
   correct every value before predicting, this mapping only needs to be a
   reasonable starting point, not a perfect one.
"""

from __future__ import annotations

import io
import re
from typing import Any

try:
    import pdfplumber

    _HAS_PDFPLUMBER = True
except ImportError:  # pragma: no cover
    _HAS_PDFPLUMBER = False

try:
    import PyPDF2

    _HAS_PYPDF2 = True
except ImportError:  # pragma: no cover
    _HAS_PYPDF2 = False


# ─────────────────────────────────────────────────────────────────────────────
# 1. PDF TEXT EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────


def extract_text_from_pdf(file_obj: Any) -> str:
    """Extract plain text from an uploaded PDF (BytesIO or Streamlit UploadedFile).
    Tries pdfplumber first (best layout fidelity), falls back to PyPDF2."""
    if hasattr(file_obj, "seek"):
        file_obj.seek(0)
    raw_bytes = file_obj.read()
    buffer = io.BytesIO(raw_bytes)

    if _HAS_PDFPLUMBER:
        try:
            with pdfplumber.open(buffer) as pdf:
                pages_text = [
                    page.extract_text(x_tolerance=3, y_tolerance=3) or "" for page in pdf.pages
                ]
            text = "\n\n".join(t.strip() for t in pages_text if t)
            if text.strip():
                return text
        except Exception:
            buffer.seek(0)

    if _HAS_PYPDF2:
        try:
            reader = PyPDF2.PdfReader(buffer)
            pages_text = [page.extract_text() or "" for page in reader.pages]
            return "\n\n".join(t.strip() for t in pages_text if t)
        except Exception as exc:
            raise RuntimeError(f"Could not parse the uploaded PDF: {exc}") from exc

    raise RuntimeError(
        "No PDF text extraction library is available. Install pdfplumber or PyPDF2."
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2. HEURISTIC (NO-API-KEY) RESUME TEXT PARSER
# ─────────────────────────────────────────────────────────────────────────────

_SECTION_KEYWORDS = {
    "education": ["education", "academic background", "academics", "qualification"],
    "experience": ["experience", "internship", "work experience", "internships"],
    "projects": ["projects", "academic projects", "personal projects"],
    "skills": ["skills", "technical skills", "programming skills", "core competencies"],
    "soft_skills": ["soft skills", "interpersonal skills"],
    "certifications": ["certifications", "certificates", "licenses & certifications"],
    "cocurricular": ["co-curricular", "extracurricular", "extra-curricular", "achievements", "activities"],
    "summary": ["summary", "objective", "professional summary", "career objective"],
}

_KNOWN_TECH_SKILLS = [
    "python", "java", "c++", "c", "javascript", "typescript", "sql", "html", "css",
    "react", "node.js", "node", "django", "flask", "spring", "angular", "vue",
    "machine learning", "deep learning", "tensorflow", "pytorch", "pandas", "numpy",
    "aws", "azure", "gcp", "docker", "kubernetes", "git", "github", "linux",
    "mongodb", "mysql", "postgresql", "excel", "power bi", "tableau", "r",
    "matlab", "autocad", "solidworks", "figma", "photoshop", "android", "kotlin", "swift",
]

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE_RE = re.compile(r"(?:\+91[\s-]?)?[6-9]\d{9}\b")
_CGPA_RE = re.compile(r"(?:cgpa|gpa)[\s:]*([0-9]\.[0-9]{1,2}|[0-9]{1,2})\b", re.IGNORECASE)
_PERCENT_RE = re.compile(r"([0-9]{2,3}(?:\.[0-9]{1,2})?)\s*%")


def _split_into_sections(text: str) -> dict[str, str]:
    """Split resume text into sections keyed by canonical section name, based
    on lines that look like a standalone header (short line, matches a known
    keyword). Anything before the first recognised header is treated as the
    header-less top block (often name/contact/summary)."""
    lines = [ln.strip() for ln in text.splitlines()]
    sections: dict[str, list[str]] = {}
    current = "_header"
    sections[current] = []

    for line in lines:
        stripped = line.strip(" \t:•-")
        lower = stripped.lower()
        matched_key = None
        if 0 < len(stripped) <= 40:
            for key, keywords in _SECTION_KEYWORDS.items():
                if any(lower == kw or lower.startswith(kw) for kw in keywords):
                    matched_key = key
                    break
        if matched_key:
            current = matched_key
            sections.setdefault(current, [])
            continue
        sections.setdefault(current, []).append(line)

    return {k: "\n".join(v).strip() for k, v in sections.items()}


def _extract_bullet_items(section_text: str) -> list[str]:
    """Split a section's text into distinct entries, using blank lines and
    bullet markers as separators."""
    if not section_text:
        return []
    # Normalise common bullet characters to newlines-with-marker.
    text = re.sub(r"[•●▪‣·]", "\n", section_text)
    raw_lines = [ln.strip(" -\t") for ln in text.split("\n")]
    lines = [ln for ln in raw_lines if ln]
    return lines


def _guess_full_name(header_block: str) -> str:
    for line in header_block.splitlines():
        candidate = line.strip()
        if not candidate or _EMAIL_RE.search(candidate) or _PHONE_RE.search(candidate):
            continue
        words = candidate.split()
        if 1 <= len(words) <= 4 and all(w.replace(".", "").isalpha() for w in words):
            return candidate.title()
    return ""


def parse_resume_text_heuristic(raw_text: str) -> dict[str, Any]:
    """Best-effort, dependency-free resume parser. Returns a resume_data dict
    (see resume_schema.get_default_resume_data). Always returns *something*
    usable -- fields it can't confidently extract are left blank/empty for
    the student to fill in on the review form."""
    from utils.resume_schema import get_default_resume_data

    data = get_default_resume_data()
    sections = _split_into_sections(raw_text)
    header_block = sections.get("_header", "")

    data["full_name"] = _guess_full_name(header_block)
    email_match = _EMAIL_RE.search(raw_text)
    if email_match:
        data["email"] = email_match.group(0)
    phone_match = _PHONE_RE.search(raw_text)
    if phone_match:
        data["phone"] = phone_match.group(0)

    if sections.get("summary"):
        data["summary"] = sections["summary"][:600]

    # ---- Skills ----
    text_lower = raw_text.lower()
    found_tech = [s for s in _KNOWN_TECH_SKILLS if s in text_lower]
    # De-duplicate near-variants (e.g. "node" vs "node.js") by keeping the longer form.
    found_tech = sorted(set(found_tech), key=len, reverse=True)
    deduped: list[str] = []
    for skill in found_tech:
        if not any(skill != other and skill in other for other in deduped):
            deduped.append(skill)
    data["skills_technical"] = [s.title() if s.islower() else s for s in deduped][:20]

    if sections.get("soft_skills"):
        data["skills_soft"] = _extract_bullet_items(sections["soft_skills"])[:10]

    # ---- Projects ----
    if sections.get("projects"):
        items = _extract_bullet_items(sections["projects"])
        # Group consecutive short "title-like" lines as separate project entries.
        data["projects"] = [{"title": item[:120], "tech_stack": "", "github_link": "",
                              "live_link": "", "description": item} for item in items[:15]]

    # ---- Experience / Internships ----
    if sections.get("experience"):
        items = _extract_bullet_items(sections["experience"])
        data["experience"] = [{"company": "", "role": item[:120], "location": "",
                                "start_date": "", "end_date": "", "description": item}
                               for item in items[:10]]

    # ---- Certifications ----
    if sections.get("certifications"):
        items = _extract_bullet_items(sections["certifications"])
        data["certifications"] = [{"name": item[:150], "issuer": "", "year": "", "url": ""}
                                   for item in items[:15]]

    # ---- Co-curricular ----
    if sections.get("cocurricular"):
        data["cocurricular"] = _extract_bullet_items(sections["cocurricular"])[:15]

    # ---- Education (best effort: look for a level keyword, then read the
    #      CGPA/percentage from THAT SAME LINE first, falling back to the
    #      next line only if the current line has no number -- searching a
    #      wide character window here would risk picking up a neighbouring
    #      education entry's score instead of this one's). ----
    edu_text = sections.get("education", "") or raw_text
    edu_lines = [ln for ln in edu_text.splitlines() if ln.strip()]
    education_entries = []
    level_patterns = [
        ("10th", [r"\b10th\b", r"\bssc\b", r"class\s*x\b"]),
        ("12th", [r"\b12th\b", r"\bhsc\b", r"class\s*xii\b", r"\bdiploma\b"]),
        ("UG", [r"b\.?\s*tech", r"b\.?\s*e\b", r"bachelor"]),
    ]
    matched_levels = set()
    for i, line in enumerate(edu_lines):
        for level, patterns in level_patterns:
            if level in matched_levels:
                continue
            if any(re.search(pattern, line, re.IGNORECASE) for pattern in patterns):
                same_line_cgpa = _CGPA_RE.search(line)
                same_line_pct = _PERCENT_RE.search(line)
                next_line = edu_lines[i + 1] if i + 1 < len(edu_lines) else ""
                next_line_cgpa = _CGPA_RE.search(next_line) if not (same_line_cgpa or same_line_pct) else None
                next_line_pct = _PERCENT_RE.search(next_line) if not (same_line_cgpa or same_line_pct) else None

                score_match = same_line_cgpa or same_line_pct or next_line_cgpa or next_line_pct
                education_entries.append({
                    "level": level,
                    "institution": "",
                    "board_university": "",
                    "year_of_passing": "",
                    "percentage_cgpa": score_match.group(1) if score_match else "",
                    "stream": "",
                })
                matched_levels.add(level)
                break
    data["education"] = education_entries

    return data


# ─────────────────────────────────────────────────────────────────────────────
# 3. RESUME DATA  →  MODEL FEATURE MAPPING
# ─────────────────────────────────────────────────────────────────────────────


def _cgpa_from_education(education: list[dict], level: str) -> float | None:
    for entry in education:
        if entry.get("level") == level and entry.get("percentage_cgpa"):
            raw = str(entry["percentage_cgpa"]).strip()
            try:
                value = float(re.sub(r"[^\d.]", "", raw))
            except ValueError:
                continue
            if level == "UG":
                # A value > 10 is almost certainly a percentage, not a 10-point
                # CGPA -- convert it down so it's on the scale the model was
                # trained on. A value <= 10 is already a native CGPA.
                return value / 9.5 if value > 10 else value
            # 10th/12th are stored as percentages; convert if it looks like a CGPA.
            return value * 9.5 if value <= 10 else value
    return None


def map_resume_to_features(resume_data: dict[str, Any]) -> dict[str, Any]:
    """
    Convert a resume_data dict into a partial model-feature dict. Only keys
    that could be confidently derived are included -- everything else is
    left for `model_utils.fill_defaults()` to back-fill with dataset medians
    (and for the student to adjust on the review form).
    """
    features: dict[str, Any] = {}

    ug_cgpa = _cgpa_from_education(resume_data.get("education", []), "UG")
    if ug_cgpa is not None:
        features["cgpa"] = round(min(ug_cgpa, 10.0), 2)

    tenth = _cgpa_from_education(resume_data.get("education", []), "10th")
    if tenth is not None:
        features["tenth_percentage"] = round(min(tenth, 100.0), 1)

    twelfth = _cgpa_from_education(resume_data.get("education", []), "12th")
    if twelfth is not None:
        features["twelfth_percentage"] = round(min(twelfth, 100.0), 1)

    features["projects_completed"] = len(resume_data.get("projects", []))
    features["certifications_count"] = len(resume_data.get("certifications", []))

    experience = resume_data.get("experience", [])
    internship_like = [e for e in experience if "intern" in (e.get("role", "") + e.get("description", "")).lower()]
    features["internships_completed"] = len(internship_like) if internship_like else len(experience)

    cocurricular_count = len(resume_data.get("cocurricular", []))
    if cocurricular_count == 0:
        features["extracurricular_involvement"] = "Low"
    elif cocurricular_count <= 2:
        features["extracurricular_involvement"] = "Medium"
    else:
        features["extracurricular_involvement"] = "High"

    n_tech = len(resume_data.get("skills_technical", []))
    if n_tech:
        features["coding_skill_rating"] = max(1, min(5, 1 + n_tech // 3))

    n_soft = len(resume_data.get("skills_soft", []))
    all_text = " ".join(resume_data.get("skills_soft", [])).lower()
    if n_soft:
        base = max(1, min(5, 1 + n_soft // 2))
        if any(k in all_text for k in ["communicat", "leadership", "public speak"]):
            base = min(5, base + 1)
        features["communication_skill_rating"] = base

    hackathon_mentions = sum(
        1 for p in resume_data.get("projects", []) + [
            {"description": c} for c in resume_data.get("cocurricular", [])
        ]
        if "hackathon" in (p.get("description", "") or "").lower()
    )
    if hackathon_mentions:
        features["hackathons_participated"] = min(hackathon_mentions, 6)

    # Branch, from the highest-level education entry's stream text, if present.
    for entry in resume_data.get("education", []):
        stream = (entry.get("stream") or "").lower()
        if not stream:
            continue
        if "computer" in stream or "cse" in stream:
            features["branch"] = "CSE"
        elif "information technology" in stream or " it" in f" {stream}":
            features["branch"] = "IT"
        elif "electronic" in stream or "ece" in stream:
            features["branch"] = "ECE"
        elif "mechanical" in stream:
            features["branch"] = "ME"
        elif "civil" in stream:
            features["branch"] = "CE"
        if "branch" in features:
            break

    return {k: v for k, v in features.items() if v is not None}
