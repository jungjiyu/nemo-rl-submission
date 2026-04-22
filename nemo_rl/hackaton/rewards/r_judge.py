"""Group-aware judge reward — VLM judge scores N candidate aux-line overlays jointly.

Paradigm A (overlay): the model's red-dashed SVG is composited ONTO the
original problem image before being sent to the judge. Our Nemotron-Nano SFT
(mix_v1) was trained to produce additive red-dashed edits over a pre-existing
base diagram, NOT to reproduce full diagrams — so overlay is the in-distribution
choice. The upstream bundle's Paradigm B (send rendered SVG alone) would be
OOD for this warm-start model.

Score contract: len(responses) floats in [0, 1]. Render-failed candidates get
0 deterministically (no API call wasted). If ALL candidates render-fail, skip
API and return [0.0]*n. On any API/parse failure, return [0.0]*n and log.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import re
import time
import uuid
from pathlib import Path
from typing import Optional

import httpx
from PIL import Image

from nemo_rl.hackaton._env import openrouter_headers
from nemo_rl.hackaton.render.svg_render import render as svg_render


# --- Image compositing (sync, CPU only) -----------------------------------


def _composite_overlay(orig_png_bytes: bytes, overlay_png_bytes: bytes) -> bytes:
    """Alpha-composite overlay PNG onto original PNG. Overlay is resized to match."""
    orig = Image.open(io.BytesIO(orig_png_bytes)).convert("RGBA")
    overlay = Image.open(io.BytesIO(overlay_png_bytes)).convert("RGBA")
    overlay = overlay.resize(orig.size, Image.Resampling.LANCZOS)
    composed = Image.alpha_composite(orig, overlay)
    buf = io.BytesIO()
    composed.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


def _b64(b: bytes) -> str:
    return base64.b64encode(b).decode()


# --- Judge prompt (Paradigm A / overlay) ----------------------------------
#
# Under overlay, the original diagram is already visible in the rendered image —
# the judge sees the problem figure PLUS the model's red-dashed additions drawn
# on top. So the judging criteria drop "faithful reproduction" and focus only
# on the quality of the auxiliary construction.

_JUDGE_PROMPT = """You are judging student-drawn auxiliary construction lines \
overlaid on a geometry problem diagram.

Problem: {q}

You will see {n} images. In each image, the black diagram IS the problem figure \
(provided unchanged); the RED DASHED lines are the student's proposed auxiliary \
construction that would help solve the problem.

Score each candidate on a 0-10 scale. Use the full range — compare candidates \
against each other even if all are weak. Criteria:
  - Are the red dashed aux lines anchored to meaningful existing points or \
features in the diagram (not floating / random)?
  - Do they create a useful sub-structure (sub-triangle, parallel, perpendicular, \
altitude, midline, angle bisector, etc.) for the stated problem?
  - Is the count minimal-but-sufficient (not cluttered with irrelevant lines)?
  - Do the endpoints land precisely rather than drifting off-anchor?

Respond in strict JSON only, no prose, no markdown fences:
{{"scores": [s0, s1, ..., s{nm1}], "ranking": [best_idx, ..., worst_idx], "rationale": "<one sentence>"}}"""


# --- Core async group call ------------------------------------------------


async def r_judge_group(
    client: httpx.AsyncClient,
    question: str,
    problem_image_png_bytes: bytes,
    responses: list[str],
    judge_model: str = "qwen/qwen3.5-122b-a10b-20260224",
    max_tokens: int = 32768,
    temperature: float = 0.0,
    ledger_dir: Optional[Path] = None,
    timeout_s: float = 120.0,
) -> list[float]:
    """Score a GRPO group of SVG responses. Returns len(responses) scores in [0, 1]."""
    base_url, headers = openrouter_headers("judge")
    n = len(responses)
    t0 = time.time()
    ledger: dict = {
        "ts": t0,
        "judge_model": judge_model,
        "question": question,
        "responses": [r[:2000] for r in responses],
        "n": n,
    }

    # 1. Extract + render + composite per response (sync, CPU)
    composites: list[Optional[bytes]] = []
    render_ok: list[bool] = []
    for r in responses:
        rr = svg_render(r)
        if not rr.ok:
            composites.append(None)
            render_ok.append(False)
            continue
        try:
            composed = _composite_overlay(problem_image_png_bytes, rr.png_bytes)
        except Exception as e:
            print(f"[r_judge_group] composite failed: {e!r}")
            composites.append(None)
            render_ok.append(False)
            continue
        composites.append(composed)
        render_ok.append(True)
    ledger["render_ok"] = render_ok

    if not any(render_ok):
        ledger["skip_reason"] = "all_render_fail"
        _dump_ledger(ledger, ledger_dir)
        return [0.0] * n

    # 2. Build judge message with composited images (grey placeholder for fails
    #    so index alignment is preserved in the API call)
    grey_png = _grey_placeholder(orig=problem_image_png_bytes)
    content: list[dict] = [
        {"type": "text", "text": _JUDGE_PROMPT.format(q=question, n=n, nm1=n - 1)}
    ]
    for c in composites:
        png = c if c is not None else grey_png
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{_b64(png)}"},
            }
        )

    # 3. Single API call
    try:
        resp = await client.post(
            f"{base_url}/chat/completions",
            headers=headers,
            json={
                "model": judge_model,
                "messages": [{"role": "user", "content": content}],
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
            timeout=timeout_s,
        )
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"] if data.get("choices") else ""
        ledger["usage"] = data.get("usage", {})
        ledger["raw_head"] = (text or "")[:600]
    except (httpx.HTTPError, asyncio.TimeoutError, KeyError) as e:
        ledger["api_error"] = f"{type(e).__name__}: {e}"
        ledger["elapsed_s"] = time.time() - t0
        _dump_ledger(ledger, ledger_dir)
        return [0.0] * n

    # 4. Parse
    try:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        parsed = json.loads(m.group()) if m else None
        if parsed is None:
            raise ValueError("no JSON object found in response")
        raw_scores = parsed.get("scores", [])
        if not isinstance(raw_scores, list) or len(raw_scores) != n:
            raise ValueError(
                f"scores list length mismatch: got {raw_scores!r}, expected {n}"
            )
    except Exception as e:
        ledger["parse_error"] = f"{type(e).__name__}: {e}"
        ledger["elapsed_s"] = time.time() - t0
        _dump_ledger(ledger, ledger_dir)
        return [0.0] * n

    # 5. 0-10 → 0-1, zero render-failed slots, clip
    scores_01: list[float] = []
    for i, s in enumerate(raw_scores):
        try:
            v = float(s) / 10.0
        except Exception:
            v = 0.0
        if not render_ok[i]:
            v = 0.0
        scores_01.append(max(0.0, min(1.0, v)))

    ledger["parsed"] = parsed
    ledger["scores_01"] = scores_01
    ledger["elapsed_s"] = time.time() - t0
    _dump_ledger(ledger, ledger_dir)
    return scores_01


# --- Helpers ---------------------------------------------------------------


_GREY_CACHE: dict[tuple[int, int], bytes] = {}


def _grey_placeholder(orig: bytes) -> bytes:
    """Grey image matching original size — visual marker for render-fail slots."""
    o = Image.open(io.BytesIO(orig))
    key = o.size
    if key in _GREY_CACHE:
        return _GREY_CACHE[key]
    grey = Image.new("RGB", o.size, (160, 160, 160))
    buf = io.BytesIO()
    grey.save(buf, format="PNG")
    b = buf.getvalue()
    _GREY_CACHE[key] = b
    return b


def _dump_ledger(ledger: dict, ledger_dir: Optional[Path]) -> None:
    if ledger_dir is None:
        return
    try:
        ledger_dir = Path(ledger_dir)
        ledger_dir.mkdir(parents=True, exist_ok=True)
        fname = f"judge_{int(ledger['ts'] * 1000)}_{uuid.uuid4().hex[:8]}.json"
        (ledger_dir / fname).write_text(json.dumps(ledger, indent=2, default=str))
    except Exception:
        pass  # never let ledger I/O kill training
