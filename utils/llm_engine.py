"""
llm_engine.py
-------------
Optional AI layer, adapted from the original AI Resume Builder project's
llm_engine.py. Everything here is used ONLY if a GROQ_API_KEY is configured
(a free key from https://console.groq.com) -- the app is fully functional
without it, falling back to utils/resume_parser.py's regex-based extraction
and utils/suggestions.py's rule-based advice.

Two features are kept from the original project (both work well without a
job description, unlike the original's ATS-score/skill-gap/interview-prep
features, which compared a resume against a specific job posting that this
app doesn't collect):

  1. parse_resume_to_json   -- structured PDF -> JSON extraction (much more
     accurate than regex on real-world resume formats/layouts).
  2. generate_improvement_advice -- turns the rule-based suggestions from
     utils/suggestions.py into a short, warm, actionable coaching paragraph.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from config.settings import GROQ_FALLBACK_MODEL, GROQ_MODEL

logger = logging.getLogger(__name__)

try:
    from groq import Groq

    _GROQ_INSTALLED = True
except ImportError:  # pragma: no cover
    _GROQ_INSTALLED = False


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
_client = None


def is_available() -> bool:
    """True if the groq package is installed AND an API key is configured."""
    return _GROQ_INSTALLED and bool(os.getenv("GROQ_API_KEY"))


def _get_client():
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GROQ_API_KEY is not set. Copy .env.example to .env and add your "
                "free key from https://console.groq.com to enable AI-powered features."
            )
        _client = Groq(api_key=api_key)
    return _client


def _strip_fences(text: str) -> str:
    match = _FENCE_RE.search(text)
    return match.group(1).strip() if match else text.strip()


def _safe_parse_json(raw: str) -> dict[str, Any]:
    """Parse LLM JSON output, tolerating markdown fences and preamble text."""
    cleaned = _strip_fences(raw)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(cleaned[start: end + 1])
        except json.JSONDecodeError as exc:
            raise ValueError(f"LLM returned malformed JSON: {exc}") from exc
    raise ValueError(f"No JSON object found in LLM response. Raw (first 300 chars): {raw[:300]}")


def _call_groq(system_prompt: str, user_prompt: str, temperature: float = 0.4, max_tokens: int = 2000) -> str:
    client = _get_client()

    def _request(model: str) -> str:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()

    try:
        return _request(GROQ_MODEL)
    except Exception as primary_exc:
        logger.warning("Primary model %s failed (%s); retrying with %s.", GROQ_MODEL, primary_exc, GROQ_FALLBACK_MODEL)
        return _request(GROQ_FALLBACK_MODEL)


def parse_resume_to_json(raw_text: str) -> dict[str, Any]:
    """Use an LLM to extract structured resume data. Raises on failure --
    callers should catch and fall back to resume_parser.parse_resume_text_heuristic."""
    safe_text = raw_text[:6000]

    system_prompt = """
You are an expert HR data-extraction engine specialised in Indian engineering
student resumes. Read the resume text and extract information into a strict
JSON schema.

OUTPUT RULES (MANDATORY):
- Respond ONLY with a single valid JSON object.
- No explanatory text, no preamble, no markdown code fences.
- If a field is missing, use "" for strings and [] for arrays.
- For education, identify 10th, 12th/Diploma, and UG (undergraduate) entries.

JSON SCHEMA:
{
  "full_name": "string", "email": "string", "phone": "string",
  "city": "string", "state": "string", "linkedin": "string", "github": "string",
  "summary": "string",
  "education": [{"level": "10th|12th|Diploma|UG|PG", "institution": "string",
                 "board_university": "string", "year_of_passing": "string",
                 "percentage_cgpa": "string", "stream": "string"}],
  "experience": [{"company": "string", "role": "string", "location": "string",
                  "start_date": "string", "end_date": "string", "description": "string"}],
  "projects": [{"title": "string", "tech_stack": "string", "github_link": "string",
                "live_link": "string", "description": "string"}],
  "skills_technical": ["string"], "skills_soft": ["string"], "skills_tools": ["string"],
  "certifications": [{"name": "string", "issuer": "string", "year": "string", "url": "string"}],
  "cocurricular": ["string"]
}
""".strip()

    user_prompt = f"RESUME TEXT:\n---\n{safe_text}\n---\nExtract the data now."
    raw_response = _call_groq(system_prompt, user_prompt, temperature=0.2, max_tokens=3000)
    return _safe_parse_json(raw_response)


def generate_improvement_advice(
    user_features: dict[str, Any],
    prediction: dict[str, Any],
    rule_based_suggestions: list[dict[str, str]],
) -> str:
    """Turn the rule-based suggestions into a short, warm, actionable coaching
    paragraph tailored to Indian campus placements. Raises on failure --
    callers should fall back to displaying the rule-based bullets as-is."""
    bullets = "\n".join(f"- {s['text']}" for s in rule_based_suggestions)

    system_prompt = """
You are a supportive, experienced campus placement coach at an Indian
engineering college. You speak encouragingly but honestly, and you give
concrete, specific next steps -- never vague platitudes like "work hard".

RULES:
- 4-6 sentences, plain paragraph (no bullet points, no markdown, no headers).
- Reference the student's actual numbers naturally where useful.
- Keep it realistic and specific to Indian campus placement culture (DSA
  practice, aptitude tests, mock interviews, resume projects, etc.).
- Do not invent facts about the student that weren't given to you.
""".strip()

    user_prompt = f"""
Placement prediction: {prediction['prediction']} ({prediction['placed_probability']:.0%} probability of being placed)

Student profile (key values): {json.dumps(user_features, default=str)}

Data-driven observations already computed:
{bullets}

Write the coaching paragraph now.
""".strip()

    return _call_groq(system_prompt, user_prompt, temperature=0.6, max_tokens=350)


def generate_resume_summary(resume_data: dict[str, Any]) -> str:
    """Auto-write a 3-4 sentence professional summary for the 'Build a
    Resume' flow, from whatever the student has filled in so far. Unlike the
    original project's version, this needs no target job description."""
    name = resume_data.get("full_name") or "The candidate"
    skills = ", ".join((resume_data.get("skills_technical") or [])[:8])
    projects = "; ".join(p.get("title", "") for p in (resume_data.get("projects") or [])[:3])
    experience = "; ".join(
        f"{e.get('role', '')} at {e.get('company', '')}" for e in (resume_data.get("experience") or [])[:2]
    ) or "no formal work experience yet"

    system_prompt = """
You are a professional resume copywriter for Indian engineering students
seeking their first campus placement.

RULES:
- Write exactly 3-4 sentences, third person.
- Sentence 1: field of study + standout strength.
- Sentence 2: key technical skills and/or flagship project.
- Sentence 3: internship/experience if any, otherwise enthusiasm + core values.
- Sentence 4 (optional): career aspiration.
- Return ONLY the paragraph -- no labels, no markdown, no quotation marks.
""".strip()

    user_prompt = f"""
Name: {name}
Key skills: {skills or "not specified"}
Notable projects: {projects or "not specified"}
Experience: {experience}

Write the professional summary now.
""".strip()

    return _call_groq(system_prompt, user_prompt, temperature=0.6, max_tokens=250)
