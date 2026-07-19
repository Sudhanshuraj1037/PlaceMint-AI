"""
model_utils.py
--------------
Loads the trained pipeline and its supporting JSON artifacts (produced by
notebooks/Placement_Model_Training.ipynb) and exposes a single clean
`predict_placement()` function used by both the Resume Analyzer and Direct
Prediction pages.

Everything here is read-only with respect to the model: this module never
re-implements encoding/scaling/feature-engineering -- that all lives inside
the pickled pipeline itself (see utils/feature_engineering.py) so the app
can never drift out of sync with how the model was trained.
"""

from __future__ import annotations

import json
from typing import Any

import joblib
import pandas as pd
import streamlit as st

from config.settings import FEATURE_CONFIG_PATH, MODEL_METRICS_PATH, MODEL_PATH

# `PlacementFeatureEngineer` is never called directly in this module, but it
# MUST be imported somewhere before `joblib.load` runs, so that Python can
# resolve the class the pickle refers to.
from utils.feature_engineering import PlacementFeatureEngineer  # noqa: F401


@st.cache_resource(show_spinner=False)
def load_pipeline():
    """Load the trained scikit-learn pipeline (cached for the app's lifetime)."""
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


@st.cache_data(show_spinner=False)
def load_feature_config() -> dict[str, Any]:
    """Load feature lists, dropdown options, ranges, defaults, and cohort stats."""
    if not FEATURE_CONFIG_PATH.exists():
        return {}
    with open(FEATURE_CONFIG_PATH) as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_model_metrics() -> dict[str, Any]:
    """Load the saved comparison table / confusion matrix / ROC curve / etc."""
    if not MODEL_METRICS_PATH.exists():
        return {}
    with open(MODEL_METRICS_PATH) as f:
        return json.load(f)


def artifacts_ready() -> bool:
    """True once the notebook has been run and all three artifacts exist."""
    return MODEL_PATH.exists() and FEATURE_CONFIG_PATH.exists() and MODEL_METRICS_PATH.exists()


def fill_defaults(user_values: dict[str, Any]) -> dict[str, Any]:
    """Merge user-supplied feature values with dataset defaults for anything
    the user didn't provide, so a prediction can always be made even from a
    partially-filled form or a partially-parsed resume."""
    config = load_feature_config()
    full = {}

    for col in config.get("numeric_features", []):
        full[col] = user_values.get(col, config["numeric_defaults"].get(col))
    for col in config.get("categorical_features", []):
        full[col] = user_values.get(col, config["categorical_defaults"].get(col))

    return full


def predict_placement(feature_values: dict[str, Any]) -> dict[str, Any]:
    """
    Run a placement prediction for one student.

    Parameters
    ----------
    feature_values : dict
        Raw feature values (numeric_features + categorical_features from
        feature_config.json). Any missing keys are back-filled with dataset
        defaults via `fill_defaults`.

    Returns
    -------
    dict with keys:
        prediction        : "Placed" | "Not Placed"
        placed_probability : float in [0, 1]
        not_placed_probability : float in [0, 1]
        confidence         : float in [0, 1]  (probability of the predicted class)
    """
    pipeline = load_pipeline()
    if pipeline is None:
        raise FileNotFoundError(
            "No trained model found at models/placement_model.pkl. "
            "Run notebooks/Placement_Model_Training.ipynb first."
        )

    full_values = fill_defaults(feature_values)
    row = pd.DataFrame([full_values])

    proba = pipeline.predict_proba(row)[0]
    not_placed_p, placed_p = float(proba[0]), float(proba[1])
    prediction = "Placed" if placed_p >= 0.5 else "Not Placed"

    return {
        "prediction": prediction,
        "placed_probability": placed_p,
        "not_placed_probability": not_placed_p,
        "confidence": max(placed_p, not_placed_p),
        "input_features": full_values,
    }
