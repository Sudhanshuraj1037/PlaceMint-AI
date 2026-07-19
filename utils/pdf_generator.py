"""
pdf_generator.py
----------------
Renders a resume_data dict (see utils/resume_schema.py) into the ATS-friendly
HTML template, and optionally converts it to a PDF.

Adapted from the original AI Resume Builder project's pdf_generator.py.
The rendering logic (Jinja2 + template) is unchanged; the PDF conversion
step is made more defensive here because WeasyPrint needs system-level
libraries (Pango/Cairo) that are not guaranteed to be present on every
machine a student runs this on -- especially Windows. If no PDF engine is
available, `generate_resume()` still returns the fully-styled HTML (which
renders correctly in the app and can be saved/printed to PDF from any
browser with Ctrl+P), so the "Build a Resume" flow never hard-fails just
because a system library is missing.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

from config.settings import TEMPLATES_DIR
from utils.resume_schema import sanitise_for_template

logger = logging.getLogger(__name__)

_jinja_env = Environment(
    loader=FileSystemLoader(str(TEMPLATES_DIR)),
    autoescape=select_autoescape(["html"]),
    trim_blocks=True,
    lstrip_blocks=True,
)
_jinja_env.filters.setdefault(
    "truncate",
    lambda s, length=255, killwords=False, end="...", leeway=0: (
        s if len(s) <= length else s[: length - len(end)] + end
    ),
)

TEMPLATE_FILE = "ats_friendly.html"

try:
    from weasyprint import HTML as WeasyHTML  # type: ignore

    _HAS_WEASYPRINT = True
except Exception:  # pragma: no cover - depends on system libraries
    _HAS_WEASYPRINT = False

try:
    import pdfkit  # type: ignore

    _HAS_PDFKIT = True
except Exception:  # pragma: no cover
    _HAS_PDFKIT = False


def render_html(resume_data: dict[str, Any]) -> str:
    """Render the resume_data dict into the ATS-friendly HTML template."""
    template = _jinja_env.get_template(TEMPLATE_FILE)
    ctx = sanitise_for_template(resume_data)
    return template.render(**ctx)


def html_to_pdf_bytes(html_string: str) -> bytes | None:
    """
    Convert HTML to PDF bytes using whichever engine is available.
    Returns None (rather than raising) if no PDF engine is installed --
    callers should fall back to offering the HTML file instead.
    """
    if _HAS_WEASYPRINT:
        try:
            return WeasyHTML(string=html_string).write_pdf()
        except Exception as exc:  # pragma: no cover
            logger.warning("WeasyPrint failed (%s); trying pdfkit.", exc)

    if _HAS_PDFKIT:
        try:
            options = {
                "page-size": "A4", "margin-top": "0mm", "margin-right": "0mm",
                "margin-bottom": "0mm", "margin-left": "0mm", "encoding": "UTF-8",
                "enable-local-file-access": None, "quiet": "",
            }
            return pdfkit.from_string(html_string, False, options=options)
        except Exception as exc:  # pragma: no cover
            logger.warning("pdfkit also failed (%s).", exc)

    return None


def generate_resume(resume_data: dict[str, Any]) -> dict[str, Any]:
    """
    Full pipeline: data -> HTML -> (optionally) PDF.

    Returns
    -------
    dict with keys:
        html : str            -- always present, safe to preview in an iframe
        pdf  : bytes | None    -- present only if a PDF engine was available
        pdf_engine_available : bool
    """
    html = render_html(resume_data)
    pdf = html_to_pdf_bytes(html)
    return {
        "html": html,
        "pdf": pdf,
        "pdf_engine_available": pdf is not None,
    }
