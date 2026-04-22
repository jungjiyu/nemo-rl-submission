"""Phase 5.2 — judge-ensemble offline eval.

Given a completed eval JSON (from hackathon/eval_step10.py, which has per-sample
'gen' SVG generations), score each generation's SVG with 3 different VL judges:

  - qwen3.5-122b-a10b-20260224 (self: same family + scale as training judge)
  - qwen3.5-9b-20260310        (same-family scale-down — cheaper, tests robustness)
  - glm-4.6v                   (cross-family — catches reward-hacking)

Output: enriched JSON with per-sample per-judge scores + aggregate.

Usage:
    /mnt/tmp/nemo_rl_venv/bin/python hackathon/eval_judge_ensemble.py \\
        --input hackathon/reports/eval_rl_s6a_step25_n100.json \\
        --n 100 \\
        --out hackathon/reports/eval_rl_s6a_step25_n100_judge_ensemble.json
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
    "X-Title": os.environ.get("OPENROUTER_APP_NAME", "sprint6-judge-ensemble"),
}

JUDGES = [
    ("qwen3.5-122b-a10b", "qwen/qwen3.5-122b-a10b-20260224", 16384),
    ("qwen3.5-9b",        "qwen/qwen3.5-9b-20260310",        16384),
    ("glm-4.6v",          "z-ai/glm-4.6v",                   16384),
]

JUDGE_PROMPT = """You are judging a student-generated diagram for a geometry problem.

Problem: {q}

The image is a student's rendered SVG attempting to: (1) reproduce the original figure, AND (2) add red dashed auxiliary construction lines. If the image is blank or empty, the SVG failed to render.

Score this SINGLE candidate on a 0-10 scale.
  - Is the diagram a faithful reproduction of what the problem describes (correct shape types, labels, proportions)?
  - Are the red dashed aux lines anchored to meaningful existing points (not floating)?
  - Do they create useful sub-structure (sub-triangle / parallel / perpendicular / altitude / midline)?
  - Is the count minimal-but-sufficient (not cluttered)?

Respond in strict JSON only, no prose:
{{"score": <0-10>, "rationale": "<one sentence>"}}"""


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


async def judge_one(client, judge_name, judge_model, max_tok, q, png_bytes):
    content = [
        {"type": "text", "text": JUDGE_PROMPT.format(q=q)},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64(png_bytes)}"}},
    ]
    payload = {
        "model": judge_model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tok,
        "temperature": 0.0,
    }
    t0 = time.time()
    try:
        r = await client.post(f"{BASE}/chat/completions", headers=HEADERS, json=payload, timeout=180)
        elapsed = time.time() - t0
        if r.status_code != 200:
            return {"judge": judge_name, "score": 0, "elapsed": elapsed, "error": f"status_{r.status_code}"}
        d = r.json()
        text = d["choices"][0]["message"]["content"] if "choices" in d else ""
        m = re.search(r"\{.*?\}", text, re.DOTALL)
        parsed = json.loads(m.group()) if m else {}
        score = float(parsed.get("score", 0))
        return {
            "judge": judge_name, "score": score,
            "rationale": parsed.get("rationale", ""),
            "elapsed": elapsed, "tokens": d.get("usage", {}),
        }
    except Exception as e:
        return {"judge": judge_name, "score": 0, "elapsed": time.time()-t0, "error": f"{type(e).__name__}:{e}"}


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="JSON from hackathon/eval_step10.py")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    from datasets import load_dataset
    os.environ.setdefault("HF_HOME", "/mnt/cmlssd004/public/chanwoo/hf")
    os.environ.setdefault("HF_DATASETS_CACHE", "/mnt/cmlssd004/public/chanwoo/hf/datasets")
    ds = load_dataset("hiyouga/geometry3k", split="test")

    eval_data = json.loads(Path(args.input).read_text())
    samples = eval_data.get("results", eval_data.get("samples", []))[: args.n]
    print(f"[ensemble] {len(samples)} samples, {len(JUDGES)} judges → {len(samples)*len(JUDGES)} API calls", flush=True)

    # Pre-composite all images (sync, CPU)
    per_sample_composite: list = []  # (idx, question, composed_png OR None)
    for s in samples:
        idx = s["idx"]
        gen = s.get("gen", s.get("gen_tail", ""))
        problem = str(ds[idx]["problem"]).replace("<image>", "")
        buf = io.BytesIO()
        ds[idx]["images"][0].convert("RGB").save(buf, format="PNG")
        orig_png = buf.getvalue()
        # Paradigm B: judge sees SVG render directly (no composite on original).
        rr = svg_render(gen)
        svg_png = rr.png_bytes if rr.ok else None
        per_sample_composite.append((idx, problem, svg_png, rr.ok))

    t0 = time.time()
    async with httpx.AsyncClient(
        limits=httpx.Limits(max_connections=128, max_keepalive_connections=64)
    ) as client:
        # Fire all samples × all judges concurrently (single gather).
        tasks = []
        tag_list = []
        for s_i, (idx, q, composed, ok) in enumerate(per_sample_composite):
            if not ok:
                continue  # render-fail: skip API calls, score=0 all judges
            for judge_name, judge_model, max_tok in JUDGES:
                tasks.append(judge_one(client, judge_name, judge_model, max_tok, q, composed))
                tag_list.append((s_i, judge_name))
        print(f"[ensemble] firing {len(tasks)} API calls concurrently...", flush=True)
        results = await asyncio.gather(*tasks, return_exceptions=True)

    # Assemble per-sample per-judge
    per_sample: list[dict] = []
    for s_i, (idx, q, composed, ok) in enumerate(per_sample_composite):
        row = {"idx": idx, "render_ok": ok, "scores": {}}
        for judge_name, _, _ in JUDGES:
            row["scores"][judge_name] = 0.0 if not ok else None
        per_sample.append(row)

    for (s_i, jname), res in zip(tag_list, results):
        if isinstance(res, Exception):
            per_sample[s_i]["scores"][jname] = {"score": 0.0, "error": str(res)[:200]}
        else:
            per_sample[s_i]["scores"][jname] = res

    elapsed = time.time() - t0
    # Aggregate
    per_judge_mean = {}
    for judge_name, _, _ in JUDGES:
        scores = []
        for row in per_sample:
            v = row["scores"][judge_name]
            if isinstance(v, dict):
                scores.append(float(v.get("score", 0)))
            elif isinstance(v, (int, float)):
                scores.append(float(v))
        per_judge_mean[judge_name] = sum(scores) / max(1, len(scores))

    out = {
        "input": args.input,
        "n": len(samples),
        "judges": [j[0] for j in JUDGES],
        "wallclock_s": elapsed,
        "per_judge_mean_0_10": per_judge_mean,
        "per_sample": per_sample,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, default=str))
    print(f"[ensemble] done in {elapsed:.1f}s — per-judge mean: {per_judge_mean}", flush=True)
    print(f"[ensemble] wrote {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
