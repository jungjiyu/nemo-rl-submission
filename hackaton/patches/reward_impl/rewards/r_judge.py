"""Sprint-6 production reward — judge-based structural/geometric quality scoring.

Group-aware: takes a list of SVG responses for the SAME question/image and scores
them jointly so the judge can compare candidates. Scheme: each response gets a
score in [0, 1] (0 = render_fail or worst-in-group, 1 = best-in-group).

Design notes (from PLAN C.1, C.4, C.9):
- Fire-all-gather: all composite renders happen sync (cheap, no GPU), then the
  single judge API call per group is the only async point.
- Scheme B alignment: render_fail candidates get 0 (consistent with r_solve).
- Ledger dump: per-call, everything needed to replay the decision offline.
- Input to `r_judge_group` is a list of SVG response strings + one image; caller
  handles pre-grouping by (question, image).
- A shared httpx.AsyncClient must be passed in by the wrapper (one per training run).
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

import httpx

# Import path hygiene: this module is imported from nemo_rl env worker processes.
# Uses the running interpreter's site-packages (nemo_rl_venv py3.12 has PIL).
REPO = Path("/mnt/cmlssd004/public/chanwoo/hackaton/nemo-RL")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from PIL import Image  # noqa: E402

from hackathon.render.svg_render import render as svg_render  # noqa: E402


# --- OpenRouter config ---

def _load_env():
    env_file = REPO / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k, v)


_load_env()
_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
_BASE = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
_HEADERS = {
    "Authorization": f"Bearer {_API_KEY}",
    "Content-Type": "application/json",
    "HTTP-Referer": os.environ.get("OPENROUTER_HTTP_REFERER", ""),
    "X-Title": os.environ.get("OPENROUTER_APP_NAME", "sprint6-judge"),
}


# --- Image compositing (sync, CPU only) ---

def _composite_overlay(orig_png_bytes: bytes, overlay_png_bytes: bytes) -> bytes:
    orig = Image.open(io.BytesIO(orig_png_bytes)).convert("RGBA")
    overlay = Image.open(io.BytesIO(overlay_png_bytes)).convert("RGBA")
    overlay = overlay.resize(orig.size, Image.Resampling.LANCZOS)
    composed = Image.alpha_composite(orig, overlay)
    buf = io.BytesIO()
    composed.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


def _b64(b: bytes) -> str:
    return base64.b64encode(b).decode()


# --- Judge prompt (validated-v1 2026-04-20) ---

_JUDGE_PROMPT = """You are judging student-generated diagrams for a geometry problem.

Problem: {q}

You will see {n} images. Each image is a student's rendered SVG diagram attempting to:
  (1) reproduce the original figure accurately, AND
  (2) add red dashed auxiliary construction lines that would help solve the problem.

Score each candidate on a 0-10 scale. USE THE FULL RANGE — compare candidates to each other even if all are weak:
  - Is the diagram a faithful reproduction of what the problem describes (correct shape types, labels, proportions)?
  - Are the red dashed aux lines anchored to meaningful existing points (not floating)?
  - Do they create useful sub-structure (sub-triangle / parallel / perpendicular / altitude / midline)?
  - Is the count minimal-but-sufficient (not cluttered)?

Respond in strict JSON only, no prose, no markdown fences:
{{"scores": [s0, s1, ..., s{nm1}], "ranking": [best_idx, ..., worst_idx], "rationale": "<one sentence>"}}"""


# --- Core async group call ---

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
    """Score a GRPO group of SVG responses. Returns len(responses) scores in [0, 1].

    Render-failed candidates get 0.0 deterministically (no API call wasted on them).
    If ALL candidates render-fail, skip API and return [0.0]*n.
    If API fails (timeout/parse/http), return [0.0]*n and log.
    """
    n = len(responses)
    t0 = time.time()
    ledger: dict = {
        "ts": t0,
        "judge_model": judge_model,
        "question": question,
        "responses": [r[:2000] for r in responses],  # truncate in ledger for size
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
        # Paradigm B (2026-04-20 user directive): send SVG render DIRECTLY, no composite.
        # Model is expected to reproduce the full diagram in SVG; judge scores the
        # reproduction+aux as a single image. SFT pre-train will align model behavior.
        composites.append(rr.png_bytes)
        render_ok.append(True)
    ledger["render_ok"] = render_ok

    if not any(render_ok):
        ledger["skip_reason"] = "all_render_fail"
        _dump_ledger(ledger, ledger_dir)
        return [0.0] * n

    # 2. Build judge message with composited images (use grey placeholder for fails
    #    so index alignment is preserved)
    grey_png = _grey_placeholder(orig=problem_image_png_bytes)
    content = [
        {"type": "text", "text": _JUDGE_PROMPT.format(q=question, n=n, nm1=n - 1)}
    ]
    for i, c in enumerate(composites):
        png = c if c is not None else grey_png
        content.append(
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_b64(png)}"}}
        )

    # 3. Single API call
    try:
        resp = await client.post(
            f"{_BASE}/chat/completions",
            headers=_HEADERS,
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
        usage = data.get("usage", {})
        ledger["usage"] = usage
        ledger["raw_head"] = (text or "")[:600]
    except (httpx.HTTPError, asyncio.TimeoutError, KeyError) as e:
        ledger["api_error"] = f"{type(e).__name__}: {e}"
        ledger["elapsed_s"] = time.time() - t0
        _dump_ledger(ledger, ledger_dir)
        return [0.0] * n

    # 4. Parse
    try:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        parsed = json.loads(m.group())
        raw_scores = parsed.get("scores", [])
        if not isinstance(raw_scores, list) or len(raw_scores) != n:
            raise ValueError(f"scores list length mismatch: got {raw_scores!r}, expected {n}")
    except Exception as e:
        ledger["parse_error"] = f"{type(e).__name__}: {e}"
        ledger["elapsed_s"] = time.time() - t0
        _dump_ledger(ledger, ledger_dir)
        return [0.0] * n

    # 5. Scale 0-10 → 0-1, zero-out render-failed slots
    scores_01: list[float] = []
    for i, s in enumerate(raw_scores):
        try:
            v = float(s) / 10.0
        except Exception:
            v = 0.0
        if not render_ok[i]:
            v = 0.0
        # clip
        v = max(0.0, min(1.0, v))
        scores_01.append(v)

    ledger["parsed"] = parsed
    ledger["scores_01"] = scores_01
    ledger["elapsed_s"] = time.time() - t0
    _dump_ledger(ledger, ledger_dir)
    return scores_01


# --- Helpers ---

_GREY_CACHE: dict[tuple[int, int], bytes] = {}


def _grey_placeholder(orig: bytes) -> bytes:
    """Grey image matching original size — used as visual marker for render-fail slots."""
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


# --- Self-test against cached fixtures (no real API) ---

def _selftest_composite():
    """Validate compositing + render pipeline end-to-end WITHOUT API."""
    from hackathon.render.svg_render import render as r
    img_path = REPO / "logs/sprint1/problem_2590.png"
    if not img_path.exists():
        print(f"[selftest] skip: {img_path} missing")
        return
    orig = img_path.read_bytes()
    fake_svg = (
        'Some reasoning here. '
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200">'
        '<line x1="10" y1="10" x2="190" y2="190" stroke="red" stroke-dasharray="5,5"/>'
        '</svg>'
    )
    rr = r(fake_svg)
    assert rr.ok, f"render failed: {rr.reason}"
    composed = _composite_overlay(orig, rr.png_bytes)
    out = Path("/tmp/_r_judge_selftest_composed.png")
    out.write_bytes(composed)
    print(f"[selftest] composite OK -> {out} ({len(composed)} bytes)")


async def _selftest_api():
    """Hit real API with 2 candidate SVGs on the cached problem image."""
    img_path = REPO / "logs/sprint1/problem_2590.png"
    if not img_path.exists() or not _API_KEY:
        print("[selftest_api] skip: need cached image + OPENROUTER_API_KEY")
        return
    orig = img_path.read_bytes()
    responses = [
        (
            'Thinking... '
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200">'
            '<line x1="50" y1="10" x2="50" y2="190" stroke="red" stroke-dasharray="5,5"/>'
            '</svg>'
        ),
        'NO SVG HERE — should score 0',
        (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200">'
            '<line x1="10" y1="100" x2="190" y2="100" stroke="red" stroke-dasharray="5,5"/>'
            '<line x1="100" y1="10" x2="100" y2="190" stroke="red" stroke-dasharray="5,5"/>'
            '</svg>'
        ),
    ]
    async with httpx.AsyncClient(
        limits=httpx.Limits(max_connections=128, max_keepalive_connections=64)
    ) as client:
        scores = await r_judge_group(
            client,
            question="A problem about a triangle",
            problem_image_png_bytes=orig,
            responses=responses,
            ledger_dir=REPO / "hackathon/rewards/_validation",
        )
    print(f"[selftest_api] scores={scores}")
    assert len(scores) == 3
    assert all(0.0 <= s <= 1.0 for s in scores)
    print("[selftest_api] OK — scores in [0,1]")


if __name__ == "__main__":
    _selftest_composite()
    if "--api" in sys.argv:
        asyncio.run(_selftest_api())
