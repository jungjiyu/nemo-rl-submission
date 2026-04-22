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
# Also accept code-fenced ```svg/xml/html ... ``` blocks.
_FENCE_PATTERN = re.compile(
    r"```(?:svg|xml|html)?\s*(.*?)```", re.DOTALL | re.IGNORECASE
)
# Unterminated fence — opening ``` with <svg inside but no closing fence.
_OPEN_FENCE_PATTERN = re.compile(
    r"```(?:svg|xml|html)?\s*(<svg\b.*)", re.DOTALL | re.IGNORECASE
)
# Open <svg> tag without a closing </svg> — pure truncation, no fence.
_OPEN_SVG_PATTERN = re.compile(r"(<svg\b[^>]*>.*)", re.DOTALL | re.IGNORECASE)


@dataclass
class RenderResult:
    ok: bool
    reason: str  # "ok" | "no_svg_tag" | "render_error: <msg>"
    svg_text: str | None
    png_bytes: bytes | None


def _close_svg(svg_open: str) -> str:
    """Append `</svg>` if missing — resvg-py rejects truncated documents."""
    return svg_open if "</svg>" in svg_open.lower() else svg_open + "</svg>"


def extract_svg(text: str) -> str | None:
    """Best-effort pull of an SVG string from arbitrary LLM output.

    Handles code fences (complete and truncated), surrounding prose, and
    truncated outputs missing the closing `</svg>`. Returns None if nothing
    plausible is found.
    """
    if not text:
        return None

    # Prefer complete <svg>...</svg>, scanning inside fences first.
    candidates: list[str] = []
    for fence_body in _FENCE_PATTERN.findall(text):
        candidates.extend(_SVG_PATTERN.findall(fence_body))
    candidates.extend(_SVG_PATTERN.findall(text))
    if candidates:
        return candidates[-1]

    # Truncated fenced <svg> — recover up to EOF with synthetic close.
    m = _OPEN_FENCE_PATTERN.search(text)
    if m:
        return _close_svg(m.group(1).rstrip("`").rstrip())

    # Truncated bare <svg> — same recovery.
    m = _OPEN_SVG_PATTERN.search(text)
    if m:
        return _close_svg(m.group(1).rstrip("`").rstrip())

    return None


def render(text: str) -> RenderResult:
    """Extract SVG from text and render to PNG bytes on a WHITE background.

    The mathcanvas SVG style draws black strokes on a transparent canvas.
    resvg-py emits RGBA, and the downstream judge/solver API receives PNG
    bytes — if we don't composite onto white first, black strokes are
    indistinguishable from the transparent background once flattened.
    """
    svg = extract_svg(text)
    if not svg:
        return RenderResult(ok=False, reason="no_svg_tag", svg_text=None, png_bytes=None)

    try:
        raw_png = resvg_py.svg_to_bytes(svg_string=svg)
    except Exception as e:
        return RenderResult(ok=False, reason=f"render_error: {e!r}", svg_text=svg, png_bytes=None)

    try:
        import io as _io
        from PIL import Image as _Image
        rgba = _Image.open(_io.BytesIO(bytes(raw_png))).convert("RGBA")
        bg = _Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        composed = _Image.alpha_composite(bg, rgba).convert("RGB")
        buf = _io.BytesIO()
        composed.save(buf, format="PNG")
        png_bytes = buf.getvalue()
    except Exception as e:
        # If compositing fails, fall back to raw RGBA bytes so the caller
        # still sees a renderable PNG (alpha handling deferred to caller).
        png_bytes = bytes(raw_png)

    return RenderResult(ok=True, reason="ok", svg_text=svg, png_bytes=png_bytes)
