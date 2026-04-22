"""R_geometric — SVG-level structural validity (no GT auxiliary lines required).

Geometry3K has no GT auxiliary-line annotations, so we score the *plausibility*
of generated red-dashed auxiliary lines instead of fidelity to a gold:

    1. has_aux_line:        ≥1 SVG element with stroke="red" AND dasharray
    2. aux_in_viewbox:      every aux endpoint lies within viewBox bounds
    3. aux_anchored:        every aux endpoint within `tol` px of either
                            (a) a non-aux endpoint or (b) a labeled <text>
                            position — i.e. the aux line connects to existing
                            points rather than floating in space

Score = mean of the three sub-scores, in [0, 1].

This is a heuristic. It rewards SVGs where the red lines look like meaningful
constructions, not random doodles.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Match a generic <svg ... viewBox="x y w h" ...> opening
_VIEWBOX = re.compile(r'viewBox\s*=\s*"\s*([\-\d.]+)\s+([\-\d.]+)\s+([\-\d.]+)\s+([\-\d.]+)\s*"')

# Lines:    <line x1=... y1=... x2=... y2=... stroke="red" stroke-dasharray="..." ...>
_LINE = re.compile(
    r'<line\b([^>]*)/?>',
    re.IGNORECASE,
)
_ATTR = re.compile(r'(\w[\w\-]*)\s*=\s*"([^"]*)"')

# Text element with x, y attributes
_TEXT = re.compile(r'<text\b([^>]*)>(.*?)</text>', re.IGNORECASE | re.DOTALL)


@dataclass
class GeomResult:
    score: float
    has_aux: bool
    aux_in_viewbox_ratio: float
    aux_anchored_ratio: float
    n_aux: int
    n_solid_endpoints: int
    n_text_anchors: int


def _parse_attrs(attr_str: str) -> dict[str, str]:
    return {k.lower(): v for k, v in _ATTR.findall(attr_str)}


def _is_aux(attrs: dict[str, str]) -> bool:
    """Aux = red stroke (any case) AND has dasharray."""
    stroke = (attrs.get("stroke") or "").strip().lower()
    dash = (attrs.get("stroke-dasharray") or "").strip()
    return ("red" in stroke or "#f" in stroke or "rgb(255" in stroke) and bool(dash)


def _line_endpoints(attrs: dict[str, str]) -> list[tuple[float, float]] | None:
    try:
        x1 = float(attrs["x1"]); y1 = float(attrs["y1"])
        x2 = float(attrs["x2"]); y2 = float(attrs["y2"])
        return [(x1, y1), (x2, y2)]
    except (KeyError, ValueError):
        return None


def r_geometric(svg_text: str, *, tol_px: float = 20.0) -> GeomResult:
    """Score auxiliary-line plausibility on a single SVG."""
    if not svg_text:
        return GeomResult(0.0, False, 0.0, 0.0, 0, 0, 0)

    # viewBox
    m = _VIEWBOX.search(svg_text)
    if m:
        vx, vy, vw, vh = (float(g) for g in m.groups())
    else:
        vx, vy, vw, vh = 0.0, 0.0, 1000.0, 1000.0  # permissive default

    aux_endpoints: list[tuple[float, float]] = []
    solid_endpoints: list[tuple[float, float]] = []
    for line_m in _LINE.finditer(svg_text):
        attrs = _parse_attrs(line_m.group(1))
        eps = _line_endpoints(attrs)
        if not eps:
            continue
        (aux_endpoints if _is_aux(attrs) else solid_endpoints).extend(eps)

    text_anchors: list[tuple[float, float]] = []
    for tm in _TEXT.finditer(svg_text):
        attrs = _parse_attrs(tm.group(1))
        try:
            text_anchors.append((float(attrs.get("x", 0)), float(attrs.get("y", 0))))
        except ValueError:
            pass

    n_aux = len(aux_endpoints)
    has_aux = n_aux >= 2  # at least one full aux line (2 endpoints)

    if not has_aux:
        return GeomResult(0.0, False, 0.0, 0.0, n_aux, len(solid_endpoints), len(text_anchors))

    # In viewBox?
    in_box = sum(
        1 for x, y in aux_endpoints
        if vx - tol_px <= x <= vx + vw + tol_px and vy - tol_px <= y <= vy + vh + tol_px
    )
    aux_in_viewbox_ratio = in_box / n_aux

    # Anchored to a non-aux endpoint or labeled text?
    anchors = solid_endpoints + text_anchors
    if not anchors:
        aux_anchored_ratio = 0.0
    else:
        def _near(p):
            for ax, ay in anchors:
                if (p[0] - ax) ** 2 + (p[1] - ay) ** 2 <= tol_px * tol_px:
                    return True
            return False
        anchored = sum(1 for p in aux_endpoints if _near(p))
        aux_anchored_ratio = anchored / n_aux

    score = (1.0 + aux_in_viewbox_ratio + aux_anchored_ratio) / 3.0  # has_aux=1 if we got here
    return GeomResult(
        score=round(score, 4),
        has_aux=True,
        aux_in_viewbox_ratio=round(aux_in_viewbox_ratio, 4),
        aux_anchored_ratio=round(aux_anchored_ratio, 4),
        n_aux=n_aux,
        n_solid_endpoints=len(solid_endpoints),
        n_text_anchors=len(text_anchors),
    )
