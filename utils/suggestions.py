"""
suggestions.py
--------------
Generates personalised "how to improve your placement chances" feedback by
comparing a student's own values against the *Placed* cohort's statistics
from the training data, weighted by how important the model actually found
each feature (permutation importance from Section 19 of the notebook).

This is deliberately rule-based and needs no external API -- it always
works. `utils/llm_engine.py` can optionally rephrase/expand these into a
warmer, more narrative form when a Groq API key is configured, but the
underlying facts always come from here.
"""

from __future__ import annotations

from typing import Any

from config.settings import FEATURE_DISPLAY_NAMES
from utils.model_utils import load_feature_config, load_model_metrics

# Features where a HIGHER value is better (the default assumption).
# `stress_level` and `backlogs` are the exceptions -- lower is better there.
LOWER_IS_BETTER = {"backlogs", "stress_level"}

# Only give actionable advice on things a student can realistically act on
# (skip things like gender/city_tier/family_income which aren't "advice").
ACTIONABLE_NUMERIC_FEATURES = [
    "cgpa",
    "projects_completed",
    "internships_completed",
    "certifications_count",
    "coding_skill_rating",
    "communication_skill_rating",
    "aptitude_skill_rating",
    "hackathons_participated",
    "backlogs",
    "study_hours_per_day",
    "attendance_percentage",
    "stress_level",
]


def _feature_importance_rank() -> dict[str, int]:
    metrics = load_model_metrics()
    importances = metrics.get("feature_importance", [])
    ranked = sorted(importances, key=lambda r: r["Importance"], reverse=True)
    return {row["Feature"]: i for i, row in enumerate(ranked)}


def generate_suggestions(user_features: dict[str, Any], max_suggestions: int = 5) -> list[dict[str, str]]:
    """
    Compare the student's values to the Placed cohort and return a ranked
    list of suggestions.

    Returns a list of dicts: {"type": "improve" | "strength", "text": str, "feature": str}
    ordered by how influential the model found that feature (most influential
    first), so the student sees the highest-leverage advice at the top.
    """
    config = load_feature_config()
    if not config:
        return []

    placed_stats = config.get("placed_cohort_stats", {})
    importance_rank = _feature_importance_rank()

    candidates: list[dict[str, Any]] = []

    for feature in ACTIONABLE_NUMERIC_FEATURES:
        if feature not in placed_stats or feature not in user_features:
            continue

        stats = placed_stats[feature]
        user_val = user_features[feature]
        if user_val is None:
            continue
        try:
            user_val = float(user_val)
        except (TypeError, ValueError):
            continue

        median = stats["median"]
        p25, p75 = stats["p25"], stats["p75"]
        lower_is_better = feature in LOWER_IS_BETTER
        label = FEATURE_DISPLAY_NAMES.get(feature, feature.replace("_", " ").title())
        rank = importance_rank.get(feature, len(importance_rank))

        below_typical = (user_val < p25) if not lower_is_better else (user_val > p75)
        above_typical = (user_val > p75) if not lower_is_better else (user_val < p25)

        if below_typical:
            if lower_is_better:
                text = (
                    f"Your {label} ({user_val:g}) is higher than most placed students "
                    f"(typical: {median:g} or below). Bringing this down could meaningfully "
                    f"improve your chances."
                )
            else:
                text = (
                    f"Your {label} ({user_val:g}) is below the typical placed student "
                    f"(median: {median:g}). Focusing here could meaningfully improve your chances."
                )
            candidates.append({"type": "improve", "text": text, "feature": feature, "rank": rank})
        elif above_typical:
            text = f"Your {label} ({user_val:g}) is at or above the placed-student median ({median:g}) — nice work."
            candidates.append({"type": "strength", "text": text, "feature": feature, "rank": rank})

    # Most model-influential features first; within a tie, "improve" tips
    # before "strength" call-outs so actionable advice surfaces first.
    candidates.sort(key=lambda c: (c["rank"], c["type"] != "improve"))

    # Keep a healthy mix: prioritise "improve" items, but always show at
    # least one "strength" if one exists, so feedback doesn't read as 100% negative.
    improves = [c for c in candidates if c["type"] == "improve"][:max_suggestions - 1]
    strengths = [c for c in candidates if c["type"] == "strength"][:1]
    combined = improves + strengths
    if not combined:
        combined = candidates[:max_suggestions]

    return combined[:max_suggestions]


def suggestions_to_bullet_text(suggestions: list[dict[str, str]]) -> str:
    """Flatten suggestion dicts into a plain bullet-point string, e.g. for
    handing to the (optional) LLM coach as context."""
    return "\n".join(f"- {s['text']}" for s in suggestions)
