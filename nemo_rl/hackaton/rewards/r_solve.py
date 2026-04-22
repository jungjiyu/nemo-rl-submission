"""Group-aware solver-pairwise reward (scheme B: 0 hurt / 1 neutral / 2 helped).

Paradigm A (overlay): each candidate's red-dashed SVG is composited ONTO the
original problem image, so the solver sees "problem diagram + optional red
dashed aux hints" — matching our Nemotron-Nano SFT distribution (additive edits
over a base). The upstream bundle's Paradigm B (send rendered SVG alone) would
present the solver with an image missing the problem geometry.

For one group of N candidates against a shared question/image, fires (N+1)
solver calls simultaneously via one asyncio.gather:
  - 1 shared no-aux call on the original problem image (baseline)
  - N with-aux calls on each candidate's overlaid composite

Scoring per candidate:
  0.0 — render_fail OR api_err OR aux hurt (correct_noaux, not correct_with)
  1.0 — neutral (same correctness outcome)
  2.0 — helped (not correct_noaux, correct_with)
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


# --- Image compositing -----------------------------------------------------


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


# --- Answer parsing --------------------------------------------------------


_BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")
_FINAL_RE = re.compile(r"(?:final answer|answer)\s*[:=]\s*([^\n]+)", re.IGNORECASE)


def extract_answer(text: str) -> Optional[str]:
    if not text:
        return None
    for m in _BOXED_RE.findall(text):
        return m.strip()
    for m in _FINAL_RE.findall(text):
        s = m.strip().strip(".,:")
        if s:
            return s
    last_line = text.strip().splitlines()[-1] if text.strip().splitlines() else ""
    nums = re.findall(r"-?\d+(?:\.\d+)?", last_line)
    if nums:
        return nums[-1]
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    return nums[-1] if nums else None


def to_float(s: Optional[str]) -> Optional[float]:
    if s is None:
        return None
    s = s.strip().replace(",", "")
    s = re.sub(r"[°\s]", "", s)
    m = re.match(r"\\?frac\{(-?\d+(?:\.\d+)?)\}\{(-?\d+(?:\.\d+)?)\}", s)
    if m:
        try:
            return float(m.group(1)) / float(m.group(2))
        except ZeroDivisionError:
            return None
    try:
        return float(s)
    except ValueError:
        m = re.match(r"(-?\d+)\s+(\d+)/(\d+)", s)
        if m:
            try:
                return float(m.group(1)) + float(m.group(2)) / float(m.group(3))
            except ZeroDivisionError:
                return None
        m = re.match(r"(\d+)/(\d+)", s)
        if m:
            try:
                return float(m.group(1)) / float(m.group(2))
            except ZeroDivisionError:
                return None
    return None


def answers_match(pred: Optional[str], gold: Optional[str], tol: float = 1e-2) -> bool:
    if pred is None or gold is None:
        return False
    p, g = to_float(pred), to_float(gold)
    if p is not None and g is not None:
        return abs(p - g) <= tol * max(1.0, abs(g))
    return str(pred).strip().lower() == str(gold).strip().lower()


# --- Solver prompt (overlay-compatible; hints-are-optional framing) -------
#
# Matches Paradigm A because it tells the solver the red dashed lines are
# optional aux hints layered over the problem diagram. Keeping the v2 framing
# shown to minimize solver-disruption in upstream experiments.

_SOLVER_PROMPT = """You are solving a geometry problem. The image shows the \
problem diagram; any red dashed lines are optional auxiliary construction hints \
that may or may not be useful. Use them only if you find them genuinely helpful.

Problem: {q}

Think step by step, then give a single numeric final answer in \\boxed{{...}}. \
No units, no explanation after the box."""


async def _solver_call(
    client: httpx.AsyncClient,
    solver_model: str,
    question: str,
    image_png: bytes,
    base_url: str,
    headers: dict[str, str],
    max_tokens: int = 2048,
    temperature: float = 0.0,
    timeout_s: float = 180.0,
) -> dict:
    """Single solver call. Returns {"text", "elapsed", "usage"?, "error"?}."""
    content = [
        {"type": "text", "text": _SOLVER_PROMPT.format(q=question)},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{_b64(image_png)}"},
        },
    ]
    payload = {
        "model": solver_model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    t0 = time.time()
    try:
        r = await client.post(
            f"{base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=timeout_s,
        )
        elapsed = time.time() - t0
        if r.status_code != 200:
            return {"text": "", "elapsed": elapsed, "error": f"status_{r.status_code}"}
        d = r.json()
        if "choices" not in d:
            return {
                "text": "",
                "elapsed": elapsed,
                "error": f"no_choices:{str(d)[:200]}",
            }
        return {
            "text": d["choices"][0]["message"]["content"],
            "elapsed": elapsed,
            "usage": d.get("usage", {}),
        }
    except (httpx.TimeoutException, httpx.HTTPError, KeyError) as e:
        return {"text": "", "elapsed": time.time() - t0, "error": f"{type(e).__name__}:{e}"}


# --- Core group scoring ----------------------------------------------------


async def r_solve_group(
    client: httpx.AsyncClient,
    question: str,
    problem_image_png_bytes: bytes,
    responses: list[str],
    gold_answer: str,
    solver_model: str = "qwen/qwen3.5-9b-20260310",
    max_tokens: int = 2048,
    ledger_dir: Optional[Path] = None,
) -> list[float]:
    """Score a GRPO group under scheme B."""
    base_url, headers = openrouter_headers("solve")
    n = len(responses)
    t0 = time.time()
    ledger: dict = {
        "ts": t0,
        "solver_model": solver_model,
        "question": question,
        "gold": gold_answer,
        "responses": [r[:2000] for r in responses],
        "n": n,
    }

    # 1. Render + composite per response (CPU, sync)
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
            print(f"[r_solve_group] composite failed: {e!r}")
            composites.append(None)
            render_ok.append(False)
            continue
        composites.append(composed)
        render_ok.append(True)
    ledger["render_ok"] = render_ok

    # 2. Fire all (N+1) solver calls simultaneously:
    #    - 1 no-aux (shared baseline for the group)
    #    - N with-aux (render-failed slots still call on the original image; their
    #      score is forced to 0 regardless, but gathering with the same arity
    #      keeps result indexing simple).
    tasks = [
        _solver_call(
            client, solver_model, question, problem_image_png_bytes, base_url, headers,
            max_tokens,
        )
    ]
    for c in composites:
        img = c if c is not None else problem_image_png_bytes
        tasks.append(
            _solver_call(
                client, solver_model, question, img, base_url, headers, max_tokens
            )
        )
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # 3. Parse no-aux baseline
    noaux_result = results[0]
    noaux_pred = None
    noaux_err: Optional[str] = None
    if isinstance(noaux_result, Exception):
        noaux_err = f"{type(noaux_result).__name__}:{noaux_result}"
    else:
        noaux_err = noaux_result.get("error")
        noaux_pred = extract_answer(noaux_result.get("text", ""))
    correct_noaux = answers_match(noaux_pred, gold_answer)
    ledger["noaux_pred"] = noaux_pred
    ledger["noaux_correct"] = correct_noaux
    ledger["noaux_err"] = noaux_err

    # 4. Score each candidate (scheme B)
    scores: list[float] = []
    per_cand = []
    for i, (res, ok) in enumerate(zip(results[1:], render_ok)):
        cand: dict = {"render_ok": ok}
        if not ok:
            scores.append(0.0)
            cand["score"] = 0.0
            cand["reason"] = "render_fail"
            per_cand.append(cand)
            continue
        if isinstance(res, Exception) or (isinstance(res, dict) and res.get("error")):
            scores.append(0.0)
            cand["score"] = 0.0
            cand["reason"] = (
                f"api_err:{res if isinstance(res, Exception) else res.get('error')}"
            )
            per_cand.append(cand)
            continue

        pred_i = extract_answer(res.get("text", ""))
        correct_with = answers_match(pred_i, gold_answer)
        cand["pred"] = pred_i
        cand["correct_with"] = correct_with

        if correct_noaux and not correct_with:
            score, reason = 0.0, "hurt"
        elif correct_noaux == correct_with:
            score, reason = 1.0, "neutral"
        else:
            score, reason = 2.0, "helped"
        cand["score"] = score
        cand["reason"] = reason
        scores.append(score)
        per_cand.append(cand)

    ledger["candidates"] = per_cand
    ledger["scores"] = scores
    ledger["elapsed_s"] = time.time() - t0
    _dump_ledger(ledger, ledger_dir)
    return scores


def _dump_ledger(ledger: dict, ledger_dir: Optional[Path]) -> None:
    if ledger_dir is None:
        return
    try:
        ledger_dir = Path(ledger_dir)
        ledger_dir.mkdir(parents=True, exist_ok=True)
        fname = f"solve_{int(ledger['ts'] * 1000)}_{uuid.uuid4().hex[:8]}.json"
        (ledger_dir / fname).write_text(json.dumps(ledger, indent=2, default=str))
    except Exception:
        pass
