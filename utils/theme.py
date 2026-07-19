"""
theme.py
--------
CSS injection and small reusable UI-building-block functions shared by
every page, so the dashboard has one consistent visual language instead of
each page reinventing headers/cards/gauges independently.
"""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from config.settings import STYLE_CSS_PATH

# Shared color tokens (must match assets/style.css :root values).
COLORS = {
    "bg": "#0a0e1a",
    "surface": "#121a2e",
    "border": "#223052",
    "accent": "#3b82f6",
    "accent_light": "#60a5fa",
    "cyan": "#22d3ee",
    "success": "#34d399",
    "warning": "#fbbf24",
    "text": "#e5e9f5",
    "text_muted": "#8b94ad",
}


def inject_css() -> None:
    """Load assets/style.css into the page. Call once at the top of every page."""
    if STYLE_CSS_PATH.exists():
        css = STYLE_CSS_PATH.read_text()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def render_hero(eyebrow: str, title: str, subtitle: str) -> None:
    """The consistent page-top banner used on every page."""
    st.markdown(
        f"""
        <div class="pm-hero">
            <div class="pm-hero-eyebrow">{eyebrow}</div>
            <div class="pm-hero-title">{title}</div>
            <p class="pm-hero-subtitle">{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_card(title: str, value: str, caption: str = "", icon: str = "") -> None:
    """A single stat card (used for KPI-style summaries)."""
    st.markdown(
        f"""
        <div class="pm-card">
            <div class="pm-card-title">{icon} {title}</div>
            <div class="pm-card-value">{value}</div>
            <div class="pm-card-caption">{caption}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_label(text: str) -> None:
    st.markdown(f'<div class="pm-section-label">{text}</div>', unsafe_allow_html=True)


def render_badge(text: str, kind: str = "info") -> str:
    """Returns badge HTML (kind: 'success' | 'warning' | 'info') -- embed
    inline inside other markdown, or call st.markdown() directly on it."""
    icon = {"success": "✅", "warning": "⚠️", "info": "ℹ️"}.get(kind, "")
    return f'<span class="pm-badge pm-badge-{kind}">{icon} {text}</span>'


def render_suggestions(suggestions: list[dict[str, str]]) -> None:
    """Render the suggestion list produced by utils/suggestions.py."""
    for s in suggestions:
        icon = "🎯" if s["type"] == "improve" else "✅"
        st.markdown(
            f'<div class="pm-suggestion {s["type"]}">'
            f'<span class="pm-suggestion-icon">{icon}</span><span>{s["text"]}</span></div>',
            unsafe_allow_html=True,
        )


def placement_gauge(probability: float, label: str = "Placement Probability") -> go.Figure:
    """The app's signature visual: a circular gauge showing placement
    probability, reused on Home (illustrative), Direct Prediction, Resume
    Analyzer (actual results), and Model Performance (metric gauges)."""
    pct = probability * 100
    if pct >= 70:
        bar_color = COLORS["success"]
    elif pct >= 45:
        bar_color = COLORS["accent_light"]
    else:
        bar_color = COLORS["warning"]

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=pct,
            number={"suffix": "%", "font": {"size": 42, "color": COLORS["text"], "family": "Sora, sans-serif"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": COLORS["text_muted"], "tickfont": {"color": COLORS["text_muted"]}},
                "bar": {"color": bar_color, "thickness": 0.28},
                "bgcolor": COLORS["surface"],
                "borderwidth": 1,
                "bordercolor": COLORS["border"],
                "steps": [
                    {"range": [0, 45], "color": "rgba(251,191,36,0.10)"},
                    {"range": [45, 70], "color": "rgba(96,165,250,0.10)"},
                    {"range": [70, 100], "color": "rgba(52,211,153,0.10)"},
                ],
            },
            title={"text": label, "font": {"size": 15, "color": COLORS["text_muted"], "family": "Inter, sans-serif"}},
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=25, r=25, t=55, b=15),
        height=280,
        font=dict(color=COLORS["text"]),
    )
    return fig


def plotly_dark_layout(fig: go.Figure, **kwargs) -> go.Figure:
    """Apply the app's consistent dark chart styling to any Plotly figure."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLORS["text"], family="Inter, sans-serif"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=10, r=10, t=45, b=10),
        **kwargs,
    )
    fig.update_xaxes(gridcolor=COLORS["border"], zerolinecolor=COLORS["border"])
    fig.update_yaxes(gridcolor=COLORS["border"], zerolinecolor=COLORS["border"])
    return fig
