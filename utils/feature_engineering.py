"""
feature_engineering.py
-----------------------
Defines `PlacementFeatureEngineer`, a custom scikit-learn transformer that
derives three composite features from the raw student profile columns:

    overall_academic_score      -- blended academic strength (0-100 scale)
    practical_experience_score  -- hands-on exposure (projects + internships + hackathons)
    soft_skill_index            -- interpersonal / aptitude readiness (1-5 scale)

WHY THIS LIVES IN ITS OWN MODULE (AND NOT INLINE IN THE NOTEBOOK)
------------------------------------------------------------------
This class is the FIRST step inside the saved `placement_model.pkl`
pipeline (see notebooks/Placement_Model_Training.ipynb, Section 10).
Because it is part of the pickled pipeline, the Streamlit app never has
to re-implement these formulas -- it just calls `pipeline.predict()`
with the 22 raw student-profile columns, and the pipeline itself derives
the 3 engineered columns internally before scaling/encoding/classifying.

This is what keeps training and serving perfectly in sync: one formula,
defined once, used everywhere. It also has to live in an importable
module path (rather than a notebook cell or `__main__`) because
`joblib`/`pickle` need to re-import the exact class when the app loads
the model later -- a class defined only inside a notebook cell cannot be
unpickled from a different process.
"""

from __future__ import annotations

from sklearn.base import BaseEstimator, TransformerMixin


class PlacementFeatureEngineer(BaseEstimator, TransformerMixin):
    """Adds composite academic / experience / soft-skill scores to a
    student-profile DataFrame. Pure, stateless (fit is a no-op) so it
    behaves identically on training data and on a single live prediction.
    """

    # Columns this transformer reads (must already exist on the input).
    REQUIRED_COLUMNS = (
        "cgpa",
        "tenth_percentage",
        "twelfth_percentage",
        "projects_completed",
        "internships_completed",
        "hackathons_participated",
        "communication_skill_rating",
        "aptitude_skill_rating",
    )

    # Columns this transformer adds.
    OUTPUT_COLUMNS = (
        "overall_academic_score",
        "practical_experience_score",
        "soft_skill_index",
    )

    def fit(self, X, y=None):  # noqa: D102 - trivial, stateless transformer
        return self

    def transform(self, X):
        missing = [c for c in self.REQUIRED_COLUMNS if c not in X.columns]
        if missing:
            raise KeyError(
                f"PlacementFeatureEngineer is missing expected column(s): {missing}. "
                "The input to the pipeline must contain the raw student-profile "
                "columns (see models/feature_config.json -> 'numeric_features' / "
                "'categorical_features' for the exact list)."
            )

        X = X.copy()

        # Blended 0-100 academic strength score. CGPA is scaled to a
        # percentage-equivalent (assuming a 10-point scale) so all three
        # inputs share the same 0-100 range before averaging.
        X["overall_academic_score"] = (
            (X["cgpa"].astype(float) / 10 * 100)
            + X["tenth_percentage"].astype(float)
            + X["twelfth_percentage"].astype(float)
        ) / 3

        # Total hands-on exposure. Internships are weighted 2x since they
        # typically reflect industry-vetted experience rather than
        # self-directed practice.
        X["practical_experience_score"] = (
            X["projects_completed"].astype(float)
            + 2 * X["internships_completed"].astype(float)
            + X["hackathons_participated"].astype(float)
        )

        # Non-technical readiness: average of communication and aptitude
        # ratings (both already on a 1-5 scale in the source dataset).
        X["soft_skill_index"] = (
            X["communication_skill_rating"].astype(float)
            + X["aptitude_skill_rating"].astype(float)
        ) / 2

        return X

    def get_feature_names_out(self, input_features=None):
        base = list(input_features) if input_features is not None else []
        return base + list(self.OUTPUT_COLUMNS)
