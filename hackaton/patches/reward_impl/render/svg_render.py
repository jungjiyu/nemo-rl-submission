"""SVG renderer using resvg-py (Rust-backed, no native deps).

Two responsibilities:
1. Extract <svg>...</svg> block from LLM-generated text (handle code fences, junk text).
2. Render to PNG bytes.

Returns a structured result so callers can compute R_fmt easily:
- ok=True  → svg_text + png_bytes are populated
- ok=False → reason explains failure (no_svg_tag / parse_error / render_error)
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import resvg_py

# Match the LAST <svg ...>...</svg> block. Greedy on inner content, single-line search
# disabled so DOTALL is required.
_SVG_PATTERN = re.compile(r"<svg\b[^>]*>.*?</svg>", re.DOTALL | re.IGNORECASE)
# Also accept code-fenced ```svg ... ``` blocks
_FENCE_PATTERN = re.compile(r"```(?:svg|xml)?\s*(.*?)```", re.DOTALL)


@dataclass
class RenderResult:
    ok: bool
    reason: str  # "ok" | "no_svg_tag" | "render_error: <msg>"
    svg_text: str | None
    png_bytes: bytes | None


def extract_svg(text: str) -> str | None:
    """Pull the last well-formed <svg>...</svg> block from arbitrary LLM output.

    Handles: code fences, surrounding prose, multiple SVGs (returns last).
    Returns None if no <svg>...</svg> pair found.
    """
    if not text:
        return None

    candidates: list[str] = []
    for fence_body in _FENCE_PATTERN.findall(text):
        candidates.extend(_SVG_PATTERN.findall(fence_body))
    candidates.extend(_SVG_PATTERN.findall(text))

    return candidates[-1] if candidates else None


def render(text: str) -> RenderResult:
    """Extract SVG from text and render to PNG bytes."""
    svg = extract_svg(text)
    if not svg:
        return RenderResult(ok=False, reason="no_svg_tag", svg_text=None, png_bytes=None)

    try:
        png = resvg_py.svg_to_bytes(svg_string=svg)
    except Exception as e:
        return RenderResult(ok=False, reason=f"render_error: {e!r}", svg_text=svg, png_bytes=None)

    return RenderResult(ok=True, reason="ok", svg_text=svg, png_bytes=bytes(png))
