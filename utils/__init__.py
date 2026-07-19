"""
utils/
------
Shared logic used by both the Streamlit application (pages/*.py) and,
via `feature_engineering.py`, the model-training notebook.

Keeping this code in one importable package (instead of copy-pasting
the same formulas into the notebook and the app separately) is what
guarantees the model always sees features engineered exactly the same
way at training time and at prediction time.
"""
