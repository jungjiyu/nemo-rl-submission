"""R_fmt — Format-Validity Gate (SGP Eq.4.5 / RLRF R_valid).

Binary 0/1 reward. Returns 1 iff the LLM output contains an extractable
<svg>...</svg> block that resvg-py renders to non-empty PNG.

This is the gating reward that multiplies all others — if R_fmt = 0, the
total reward is 0 regardless of pixel/semantic similarity.
"""

from __future__ import annotations

from dataclasses import dataclass

from hackathon.render.svg_render import RenderResult, render


@dataclass
class FmtResult:
    score: float            # 0.0 or 1.0
    reason: str             # "ok" | "no_svg_tag" | "render_error: ..."
    svg_text: str | None
    png_bytes: bytes | None


def r_fmt(text: str) -> FmtResult:
    """Compute R_fmt for one model output."""
    rr: RenderResult = render(text)
    return FmtResult(
        score=1.0 if rr.ok else 0.0,
        reason=rr.reason,
        svg_text=rr.svg_text,
        png_bytes=rr.png_bytes,
    )
