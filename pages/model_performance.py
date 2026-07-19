"""
pages/model_performance.py
----------------------------
Displays everything saved by the notebook into models/model_metrics.json:
the 7-model comparison table, confusion matrix, ROC curve, classification
report, cross-validation results, and feature importance — plus the
original report-ready PNG charts (handy for screenshotting into a project
report).
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config.settings import FEATURE_DISPLAY_NAMES, REPORTS_DIR
from utils.model_utils import artifacts_ready, load_model_metrics
from utils.theme import COLORS, placement_gauge, plotly_dark_layout, render_card, render_hero, render_section_label

render_hero(
    eyebrow="Model Performance",
    title="📊 How Good Is the Model, Really?",
    subtitle="Every metric below comes straight from the training notebook — nothing here is recomputed on the fly.",
)

if not artifacts_ready():
    st.error(
        "No trained model found. Run **notebooks/Placement_Model_Training.ipynb** first, "
        "then reload this page."
    )
    st.stop()

metrics = load_model_metrics()
best_model = metrics["best_model_name"]
test_metrics = metrics["test_metrics"]
comparison = pd.DataFrame(metrics["comparison_table"])

# ── KPI row ──────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    render_card("Model Used", best_model, metrics["final_model_source"].title() + " hyperparameters", "🏆")
with k2:
    render_card("Accuracy", f"{test_metrics['Accuracy']:.1%}", "Held-out test set", "🎯")
with k3:
    render_card("Precision", f"{test_metrics['Precision']:.1%}", "Of predicted 'Placed'", "🔎")
with k4:
    render_card("Recall", f"{test_metrics['Recall']:.1%}", "Of actual 'Placed'", "📡")
with k5:
    render_card("F1 / ROC AUC", f"{test_metrics['F1 Score']:.3f} / {test_metrics['ROC AUC']:.3f}", "Primary selection metrics", "⚖️")

st.write("")

# ── Model comparison ─────────────────────────────────────────────────────────
render_section_label("Model comparison — all 7 candidates")
tbl_col, chart_col = st.columns([1, 1.1])

with tbl_col:
    display_df = comparison.copy()
    for c in ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"]:
        display_df[c] = display_df[c].map(lambda v: f"{v:.4f}")
    st.dataframe(display_df, width="stretch", hide_index=True, height=320)
    st.caption(f"Ranked by F1 Score — the dataset is imbalanced "
               f"({metrics.get('class_balance', {}).get('placed_pct', '~86')}% Placed), so F1 is a more "
               f"reliable selection metric than raw Accuracy.")

with chart_col:
    melted = comparison.melt(id_vars="Model", value_vars=["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"],
                              var_name="Metric", value_name="Score")
    fig = px.bar(melted, x="Model", y="Score", color="Metric", barmode="group",
                 color_discrete_sequence=["#93c5fd", "#60a5fa", "#3b82f6", "#2563eb", "#1e3a8a"])
    fig.update_layout(height=340, yaxis_range=[0, 1.05], legend=dict(orientation="h", y=1.15))
    plotly_dark_layout(fig)
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

st.write("")

# ── Confusion matrix + ROC curve ─────────────────────────────────────────────
render_section_label("Confusion matrix & ROC curve — final model")
cm_col, roc_col = st.columns(2)

with cm_col:
    cm = metrics["confusion_matrix"]
    z = [[cm[0][0], cm[0][1]], [cm[1][0], cm[1][1]]]
    fig_cm = go.Figure(data=go.Heatmap(
        z=z, x=["Not Placed", "Placed"], y=["Not Placed", "Placed"],
        colorscale=[[0, "#0d1424"], [1, "#3b82f6"]], showscale=False,
        text=z, texttemplate="%{text}", textfont={"size": 22, "color": "white"},
    ))
    fig_cm.update_layout(height=340, xaxis_title="Predicted", yaxis_title="Actual",
                         yaxis=dict(autorange="reversed"))
    plotly_dark_layout(fig_cm)
    st.plotly_chart(fig_cm, width="stretch", config={"displayModeBar": False})
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    st.caption(f"TN={tn} · FP={fp} · FN={fn} · TP={tp}")

with roc_col:
    roc = metrics["roc_curve"]
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=roc["fpr"], y=roc["tpr"], mode="lines",
                                  name=f"{best_model} (AUC={roc['auc']:.3f})",
                                  line=dict(color=COLORS["accent_light"], width=3)))
    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random guess",
                                  line=dict(color=COLORS["text_muted"], dash="dash")))
    fig_roc.update_layout(height=340, xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
                           legend=dict(orientation="h", y=-0.2))
    plotly_dark_layout(fig_roc)
    st.plotly_chart(fig_roc, width="stretch", config={"displayModeBar": False})

st.write("")

# ── Classification report + Cross-validation ─────────────────────────────────
report_col, cv_col = st.columns(2)
with report_col:
    render_section_label("Classification report")
    report = metrics["classification_report"]
    rows = []
    for label in ["Not Placed", "Placed"]:
        r = report[label]
        rows.append({"Class": label, "Precision": f"{r['precision']:.3f}", "Recall": f"{r['recall']:.3f}",
                      "F1": f"{r['f1-score']:.3f}", "Support": int(r["support"])})
    rows.append({"Class": "Accuracy", "Precision": "", "Recall": "", "F1": f"{report['accuracy']:.3f}",
                 "Support": int(report["macro avg"]["support"])})
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

with cv_col:
    render_section_label("5-fold cross-validation (whole dataset)")
    cv = metrics["cross_validation"]
    st.plotly_chart(placement_gauge(cv["f1_mean"], "Mean CV F1 Score"), width="stretch",
                     config={"displayModeBar": False})
    st.caption(f"F1 across folds: {[round(s, 3) for s in cv['f1_scores']]}  →  "
               f"mean {cv['f1_mean']:.4f} ± {cv['f1_std']:.4f}")

st.write("")

# ── Feature importance ───────────────────────────────────────────────────────
render_section_label("Feature importance (permutation importance)")
imp_df = pd.DataFrame(metrics["feature_importance"]).head(12)
imp_df["Label"] = imp_df["Feature"].map(lambda f: FEATURE_DISPLAY_NAMES.get(f, f.replace("_", " ").title()))
fig_imp = px.bar(imp_df.sort_values("Importance"), x="Importance", y="Label", orientation="h",
                  color_discrete_sequence=[COLORS["accent"]])
fig_imp.update_layout(height=420, xaxis_title="Mean decrease in F1 when shuffled", yaxis_title="")
plotly_dark_layout(fig_imp)
st.plotly_chart(fig_imp, width="stretch", config={"displayModeBar": False})

st.write("")

# ── Dataset & tuning info ────────────────────────────────────────────────────
render_section_label("Dataset & tuning details")
d1, d2 = st.columns(2)
with d1:
    split = metrics["train_test_split"]
    st.markdown(
        f"""
        <div class="pm-card">
            <div class="pm-card-title">🗂️ Dataset</div>
            <p class="pm-muted">
                Training samples: <b>{split['train_size']:,}</b><br>
                Test samples: <b>{split['test_size']:,}</b> ({split['test_fraction']:.0%} held out)<br>
                Class balance: <b>{metrics.get('class_balance', {}).get('placed_pct', '—')}% Placed</b>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with d2:
    tuning = metrics["tuning"]
    params_str = "<br>".join(f"{k.replace('clf__', '')}: <b>{v}</b>" for k, v in tuning["best_params"].items())
    st.markdown(
        f"""
        <div class="pm-card">
            <div class="pm-card-title">🔧 Best Hyperparameters (GridSearchCV)</div>
            <p class="pm-muted">{params_str}<br>Best CV F1 during tuning: <b>{tuning['best_cv_f1']:.4f}</b></p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── Report-ready charts ───────────────────────────────────────────────────────
with st.expander("🖼️ View report-ready charts (from the training notebook — great for screenshots)"):
    chart_files = [
        ("target_distribution.png", "Target Distribution"),
        ("correlation_heatmap.png", "Correlation Heatmap"),
        ("feature_distributions.png", "Feature Distributions"),
        ("pairplot.png", "Pairplot"),
        ("countplot_branch.png", "Placement by Branch"),
        ("placement_rate_heatmap.png", "Placement Rate — Branch × City Tier"),
    ]
    grid = st.columns(2)
    for i, (filename, caption) in enumerate(chart_files):
        path = REPORTS_DIR / filename
        if path.exists():
            with grid[i % 2]:
                st.image(str(path), caption=caption, width="stretch")
