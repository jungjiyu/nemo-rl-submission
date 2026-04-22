"""Open-ended solver Δ evaluation — does the aux SVG actually help a solver
answer the real Geometry3K problem?

Given:
  - `--gen-input`: eval JSON with per-sample `gen` (the trained model's SVG aux)
                  produced by hackathon/eval_step10.py
  - `--no-aux-baseline`: a precomputed JSON from hackathon/tools/no_aux_baseline.py
                         with per-idx {correct_no_aux, solver_answer_no_aux}

Does:
  For each of the first N samples in the input:
    1. Render SVG → composite onto original Geometry3K image
    2. Send (problem text, composite image) to solver
    3. Extract boxed answer; compare to gold with numeric/string normalization
    4. Record per-sample correctness WITH aux
  Merge with the no-aux baseline → compute Δ = rate_with_aux − rate_no_aux.

Supports two solvers in parallel to cross-check (default: qwen3.5-9b
training-family + glm-4.6v cross-family).

Output JSON:
    {
      "gen_input": ..., "n": ..., "solvers": [...],
      "per_solver": {
        "qwen3.5-9b": {
          "rate_with_aux": 0.43,
          "rate_no_aux":   0.39,
          "delta":         0.04,
          "n_with_aux":    100, "n_no_aux": 100,
        },
        ...
      },
      "per_sample": [
        {"idx": 0, "gold": "48", "render_ok": true,
         "qwen3.5-9b": {"with_aux": {"answer":"48","correct":true}, "no_aux": {"answer":"50","correct":false}},
         ...
        },
        ...
      ],
    }

Usage:
    /mnt/tmp/nemo_rl_venv/bin/python hackathon/eval_solve_open.py \\
        --gen-input hackathon/reports/eval_rl_s6a_step25_n100.json \\
        --no-aux-baseline hackathon/reports/no_aux_solve_baseline.json \\
        --n 100 \\
        --out hackathon/reports/solve_open_s6a_step25_n100.json
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import os
import re
import sys
import time
from pathlib import Path

import httpx

REPO = Path("/mnt/cmlssd004/public/chanwoo/hackaton/nemo-RL")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from PIL import Image  # noqa: E402

from hackathon.render.svg_render import render as svg_render  # noqa: E402

# env
for line in (REPO / ".env").read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        os.environ.setdefault(k, v)
API_KEY = os.environ["OPENROUTER_API_KEY"]
BASE = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
    "HTTP-Referer": os.environ.get("OPENROUTER_HTTP_REFERER", ""),
    "X-Title": os.environ.get("OPENROUTER_APP_NAME", "sprint6-solve-open"),
}

# Solvers: (label, model_id, max_tokens). Default = training-family qwen9b + cross-family glm.
# qwen3.5-9b is a reasoning model — emits hidden CoT tokens that can exhaust short max_tokens
# (n=3 sanity showed 2/3 returned empty at max_tok=2048). Bumped to 8192 to accommodate.
SOLVERS = [
    ("qwen3.5-9b", "qwen/qwen3.5-9b-20260310", 16384),
]

SOLVE_PROMPT = """You are solving a geometry problem shown in the image.

Problem: {problem}

Think step by step. Then give your final numeric or symbolic answer inside \\boxed{{}}.

Example format: "... therefore the answer is \\boxed{{12}}."

If the answer is a square root, write it as \\boxed{{\\sqrt{{3}}}} or \\boxed{{2\\sqrt{{3}}}}.
If the answer has units, include them inside the box.
"""

_BOXED = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")
_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def composite_overlay(orig_png: bytes, overlay_png: bytes) -> bytes:
    orig = Image.open(io.BytesIO(orig_png)).convert("RGBA")
    overlay = Image.open(io.BytesIO(overlay_png)).convert("RGBA")
    overlay = overlay.resize(orig.size, Image.Resampling.LANCZOS)
    composed = Image.alpha_composite(orig, overlay)
    buf = io.BytesIO()
    composed.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


def b64(b: bytes) -> str:
    return base64.b64encode(b).decode()


def extract_boxed(text: str) -> str | None:
    """Extract innermost \\boxed{...} content. Returns None if not found."""
    if not text:
        return None
    matches = _BOXED.findall(text)
    if not matches:
        # fallback: last number in last 200 chars
        m = _NUM.findall(text[-200:])
        return m[-1] if m else None
    return matches[-1].strip()


def normalize_answer(s: str) -> str:
    """Light normalization for string match."""
    if s is None:
        return ""
    t = s.strip()
    # strip latex wrappers
    t = t.replace("$", "").replace("\\,", " ").strip()
    # strip common units
    for u in ["degrees", "degree", "°", "cm", "m"]:
        if t.lower().endswith(u):
            t = t[: -len(u)].strip()
    return t


def numeric_match(a: str, b: str, tol: float = 1e-3) -> bool:
    """True if both parse as numbers and within tolerance."""
    try:
        fa = float(a)
        fb = float(b)
        return abs(fa - fb) < tol * max(1.0, abs(fa), abs(fb))
    except (TypeError, ValueError):
        return False


def answers_match(pred: str | None, gold: str) -> bool:
    if pred is None:
        return False
    p = normalize_answer(pred)
    g = normalize_answer(gold)
    if not p or not g:
        return False
    if p == g:
        return True
    if numeric_match(p, g):
        return True
    # try extracting a bare number from pred; match against gold number
    m = _NUM.findall(p)
    if m and numeric_match(m[-1], g):
        return True
    return False


async def solve_one(client, solver_label, solver_model, max_tok, problem, png_bytes):
    content = [
        {"type": "text", "text": SOLVE_PROMPT.format(problem=problem)},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64(png_bytes)}"}},
    ]
    payload = {
        "model": solver_model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tok,
        "temperature": 0.0,
    }
    t0 = time.time()
    try:
        r = await client.post(f"{BASE}/chat/completions", headers=HEADERS, json=payload, timeout=180)
        elapsed = time.time() - t0
        if r.status_code != 200:
            return {
                "solver": solver_label,
                "answer": None,
                "raw": "",
                "elapsed": elapsed,
                "error": f"status_{r.status_code}",
            }
        d = r.json()
        text = d["choices"][0]["message"]["content"] if "choices" in d else ""
        ans = extract_boxed(text)
        return {
            "solver": solver_label,
            "answer": ans,
            "raw": text[:400],
            "elapsed": elapsed,
            "tokens": d.get("usage", {}),
        }
    except Exception as e:
        return {
            "solver": solver_label,
            "answer": None,
            "raw": "",
            "elapsed": time.time() - t0,
            "error": f"{type(e).__name__}:{e}",
        }


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gen-input", required=True, help="eval JSON with per-sample 'gen'")
    p.add_argument("--no-aux-baseline", required=True, help="precomputed no-aux solver results JSON")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--out", required=True)
    p.add_argument(
        "--solvers", default="qwen3.5-9b", help="comma-separated solver labels to run (from SOLVERS)"
    )
    args = p.parse_args()

    wanted = set(s.strip() for s in args.solvers.split(","))
    active_solvers = [s for s in SOLVERS if s[0] in wanted]
    assert active_solvers, f"no solvers match --solvers={args.solvers}"

    from datasets import load_dataset

    os.environ.setdefault("HF_HOME", "/mnt/cmlssd004/public/chanwoo/hf")
    os.environ.setdefault("HF_DATASETS_CACHE", "/mnt/cmlssd004/public/chanwoo/hf/datasets")
    ds = load_dataset("hiyouga/geometry3k", split="test")

    gen_data = json.loads(Path(args.gen_input).read_text())
    samples = gen_data.get("results", gen_data.get("samples", []))[: args.n]
    baseline = json.loads(Path(args.no_aux_baseline).read_text())
    baseline_by_idx = {row["idx"]: row for row in baseline["per_sample"]}

    # Composite each sample's aux (or skip if render failed)
    composites: list = []  # (idx, problem, composed_png OR None, gold)
    for s in samples:
        idx = s["idx"]
        gen = s.get("gen", s.get("gen_tail", ""))
        problem = str(ds[idx]["problem"]).replace("<image>", "").strip()
        gold = str(ds[idx]["answer"])
        buf = io.BytesIO()
        ds[idx]["images"][0].convert("RGB").save(buf, format="PNG")
        orig_png = buf.getvalue()
        # Paradigm B: send SVG render directly (no composite on original).
        rr = svg_render(gen)
        svg_png = rr.png_bytes if rr.ok else None
        composites.append((idx, problem, svg_png, gold, rr.ok))

    # Fire all {sample, solver} calls concurrently
    t0 = time.time()
    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=64, max_keepalive_connections=32)) as client:
        tasks = []
        tag_list = []  # (sample_idx_in_list, solver_label)
        for si, (idx, problem, composed, gold, ok) in enumerate(composites):
            if not ok:
                continue  # render-fail: skip API calls entirely
            for label, model, mxt in active_solvers:
                tasks.append(solve_one(client, label, model, mxt, problem, composed))
                tag_list.append((si, label))
        print(f"[solve-open] {len(tasks)} API calls concurrent across {len(active_solvers)} solvers", flush=True)
        results = await asyncio.gather(*tasks, return_exceptions=True)

    # Assemble per-sample
    per_sample: list[dict] = []
    for si, (idx, problem, composed, gold, ok) in enumerate(composites):
        row = {
            "idx": idx,
            "gold": gold,
            "render_ok": ok,
        }
        for label, _, _ in active_solvers:
            row[label] = {"with_aux": None, "no_aux": None}
        per_sample.append(row)

    for (si, label), res in zip(tag_list, results):
        if isinstance(res, Exception):
            per_sample[si][label]["with_aux"] = {"answer": None, "correct": False, "error": str(res)[:200]}
        else:
            ans = res.get("answer")
            correct = answers_match(ans, per_sample[si]["gold"])
            per_sample[si][label]["with_aux"] = {"answer": ans, "correct": correct, "raw": res.get("raw", "")}

    # Merge in no-aux baseline for each idx
    for row in per_sample:
        bl = baseline_by_idx.get(row["idx"], {})
        for label, _, _ in active_solvers:
            if label in bl:
                row[label]["no_aux"] = bl[label]
            else:
                # baseline didn't run this solver — set None
                row[label]["no_aux"] = None

    # Aggregate per solver
    per_solver: dict = {}
    for label, _, _ in active_solvers:
        w_total, w_correct, n_total, n_correct = 0, 0, 0, 0
        for row in per_sample:
            w = row[label]["with_aux"]
            n = row[label]["no_aux"]
            if w and "correct" in w:
                w_total += 1
                w_correct += int(bool(w["correct"]))
            if n and "correct" in n:
                n_total += 1
                n_correct += int(bool(n["correct"]))
        rate_with = w_correct / w_total if w_total else 0.0
        rate_no = n_correct / n_total if n_total else 0.0
        per_solver[label] = {
            "rate_with_aux": rate_with,
            "rate_no_aux": rate_no,
            "delta": rate_with - rate_no,
            "n_with_aux": w_total,
            "n_correct_with_aux": w_correct,
            "n_no_aux": n_total,
            "n_correct_no_aux": n_correct,
        }

    elapsed = time.time() - t0
    out = {
        "gen_input": args.gen_input,
        "no_aux_baseline": args.no_aux_baseline,
        "n": len(samples),
        "solvers": [s[0] for s in active_solvers],
        "wallclock_s": elapsed,
        "per_solver": per_solver,
        "per_sample": per_sample,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, default=str))
    print(f"[solve-open] {elapsed:.1f}s — per-solver: {per_solver}", flush=True)
    print(f"[solve-open] wrote {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
