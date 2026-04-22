"""Sprint-6 production reward — solver-pairwise downstream-accuracy scoring.

Group-aware pairwise-Δ: for a single question + image with N candidate SVG responses,
fire (N+1) solver calls (1 shared no-aux baseline + N with-aux composites) via a
single asyncio.gather. Score each response on scheme B (PLAN C.1):

  0.0 — render_fail OR aux hurt (right→wrong)
  1.0 — neutral (same outcome with_aux vs no_aux)
  2.0 — helped (wrong→right)

This is NOT scaled to [0,1] — GRPO group-norm handles variance, and the
integer scale is more robust to downstream loss term composition (see PLAN C.1).

Prompt: v2 (hints-are-optional) — pre-PoC shows Δ = -6.7pp vs v1's Δ = -23.3pp
(significantly less solver-disruption). Validated 2026-04-20.
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
    "X-Title": os.environ.get("OPENROUTER_APP_NAME", "sprint6-solve"),
}


# --- Composite (same as r_judge) ---

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


# --- Answer parsing (copy of pre_poc_solve_delta_promptv2.py, kept in-file for stability) ---

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


# --- Solver prompt (validated-v1 2026-04-20, v2 framing, Δ=-6.7pp vs v1 -23.3pp) ---

_SOLVER_PROMPT = """You are solving a geometry problem. The image shows the problem diagram; any red dashed lines are optional auxiliary construction hints that may or may not be useful. Use them only if you find them genuinely helpful.

Problem: {q}

Think step by step, then give a single numeric final answer in \\boxed{{...}}. No units, no explanation after the box."""


async def _solver_call(
    client: httpx.AsyncClient,
    solver_model: str,
    question: str,
    image_png: bytes,
    max_tokens: int = 2048,
    temperature: float = 0.0,
    timeout_s: float = 180.0,
) -> dict:
    """Single solver call. Returns {"text", "elapsed", "usage"?, "error"?}."""
    content = [
        {"type": "text", "text": _SOLVER_PROMPT.format(q=question)},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_b64(image_png)}"}},
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
            f"{_BASE}/chat/completions", headers=_HEADERS, json=payload, timeout=timeout_s
        )
        elapsed = time.time() - t0
        if r.status_code != 200:
            return {"text": "", "elapsed": elapsed, "error": f"status_{r.status_code}"}
        d = r.json()
        if "choices" not in d:
            return {"text": "", "elapsed": elapsed, "error": f"no_choices:{str(d)[:200]}"}
        return {
            "text": d["choices"][0]["message"]["content"],
            "elapsed": elapsed,
            "usage": d.get("usage", {}),
        }
    except (httpx.TimeoutException, httpx.HTTPError, KeyError) as e:
        return {"text": "", "elapsed": time.time() - t0, "error": f"{type(e).__name__}:{e}"}


# --- Core group scoring ---

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
    """Score a GRPO group under scheme B (0 hurt/fail, 1 neutral, 2 helped).

    Fires (N+1) solver calls simultaneously via one asyncio.gather:
      - 1 shared no-aux call (on the original image)
      - N with-aux calls (one per composited image)
    """
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
        # Paradigm B (2026-04-20 user directive): send SVG render DIRECTLY as the
        # "with-aux" image to solver — NOT overlay on original. Model generates
        # reproduction+aux as single SVG; solver treats it as the problem diagram.
        composites.append(rr.png_bytes)
        render_ok.append(True)
    ledger["render_ok"] = render_ok

    # 2. Fire all (N+1) calls simultaneously
    #    - one no-aux (shared baseline for group)
    #    - one with-aux per candidate; render-failed slots still call on ORIG
    #      (their score is forced to 0 regardless, but the call is cheap and
    #       keeps request-tagging simple). Alternatively, skip — but gathering
    #       with None placeholders complicates exception handling.
    tasks = [
        _solver_call(client, solver_model, question, problem_image_png_bytes, max_tokens),
    ]
    for c in composites:
        img = c if c is not None else problem_image_png_bytes  # dummy for failed
        tasks.append(_solver_call(client, solver_model, question, img, max_tokens))
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # 3. Parse answers
    noaux_result = results[0]
    withaux_results = results[1:]
    noaux_pred = None
    noaux_err = None
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
    for i, (res, ok) in enumerate(zip(withaux_results, render_ok)):
        cand = {"render_ok": ok}
        if not ok:
            scores.append(0.0)
            cand["score"] = 0.0
            cand["reason"] = "render_fail"
            per_cand.append(cand)
            continue
        if isinstance(res, Exception) or (isinstance(res, dict) and res.get("error")):
            scores.append(0.0)
            cand["score"] = 0.0
            cand["reason"] = f"api_err:{res if isinstance(res, Exception) else res.get('error')}"
            per_cand.append(cand)
            continue

        pred_i = extract_answer(res.get("text", ""))
        correct_with = answers_match(pred_i, gold_answer)
        cand["pred"] = pred_i
        cand["correct_with"] = correct_with

        # Scheme B
        if correct_noaux and not correct_with:
            score = 0.0
            reason = "hurt"
        elif correct_noaux == correct_with:
            score = 1.0
            reason = "neutral"
        elif not correct_noaux and correct_with:
            score = 2.0
            reason = "helped"
        else:
            score = 0.0
            reason = "?"  # unreachable
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


# --- Self-test ---

async def _selftest_api():
    """Run one group on real Geometry3K problem with 3 candidate SVGs."""
    if not _API_KEY:
        print("[selftest] skip: no OPENROUTER_API_KEY")
        return

    from datasets import load_dataset
    os.environ.setdefault("HF_HOME", "/mnt/cmlssd004/public/chanwoo/hf")
    os.environ.setdefault("HF_DATASETS_CACHE", "/mnt/cmlssd004/public/chanwoo/hf/datasets")
    ds = load_dataset("hiyouga/geometry3k", split="test")
    ex = ds[0]

    # Convert PIL image to PNG bytes
    buf = io.BytesIO()
    ex["images"][0].convert("RGB").save(buf, format="PNG")
    image_bytes = buf.getvalue()
    question = str(ex["problem"]).replace("<image>", "")
    gold = str(ex["answer"])

    responses = [
        'no svg at all, policy failure',
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200">'
        f'<line x1="10" y1="100" x2="190" y2="100" stroke="red" stroke-dasharray="5,5"/></svg>',
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200">'
        f'<line x1="50" y1="10" x2="50" y2="190" stroke="red" stroke-dasharray="5,5"/>'
        f'<line x1="150" y1="10" x2="150" y2="190" stroke="red" stroke-dasharray="5,5"/></svg>',
    ]
    async with httpx.AsyncClient(
        limits=httpx.Limits(max_connections=128, max_keepalive_connections=64)
    ) as client:
        scores = await r_solve_group(
            client,
            question=question,
            problem_image_png_bytes=image_bytes,
            responses=responses,
            gold_answer=gold,
            ledger_dir=REPO / "hackathon/rewards/_validation",
        )
    print(f"[selftest_api] gold={gold!r} scores={scores}")
    assert len(scores) == 3
    assert all(s in (0.0, 1.0, 2.0) for s in scores)
    print("[selftest_api] OK — scores in {0,1,2}")


if __name__ == "__main__":
    if "--api" in sys.argv:
        asyncio.run(_selftest_api())
    else:
        # Pure parsing selftest — no API
        assert answers_match("42", "42.0")
        assert answers_match(extract_answer("the answer is \\boxed{42}"), "42")
        assert not answers_match("42", "43")
        assert extract_answer("the answer is \\boxed{3.14}") == "3.14"
        assert extract_answer("final answer: 42") == "42"
        print("[selftest] parse/match OK")
