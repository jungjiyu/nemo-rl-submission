"""Precompute "no-aux" solver baseline once; reused by all ablation eval runs.

For each of N test samples in Geometry3K:
  - Send (problem text, bare original image, NO aux overlay) to each solver
  - Extract boxed answer; compare to gold

Output JSON:
    {
      "n": ...,
      "solvers": ["qwen3.5-9b", "glm-4.6v"],
      "per_sample": [
        {"idx": 0, "gold": "48",
         "qwen3.5-9b": {"answer": "48", "correct": true, "raw": "..."},
         "glm-4.6v":   {"answer": "50", "correct": false, "raw": "..."}},
        ...
      ],
      "per_solver": {
        "qwen3.5-9b": {"rate": 0.39, "n": 100, "n_correct": 39},
        ...
      },
    }

Idempotent: if output file exists and --force not passed, skips.

Usage:
    /mnt/tmp/nemo_rl_venv/bin/python hackathon/tools/no_aux_baseline.py \\
        --n 100 \\
        --out hackathon/reports/no_aux_solve_baseline.json
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import os
import sys
import time
from pathlib import Path

import httpx

REPO = Path("/mnt/cmlssd004/public/chanwoo/hackaton/nemo-RL")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from PIL import Image  # noqa: E402

# import shared helpers from sibling eval_solve_open.py
from hackathon.eval_solve_open import (  # noqa: E402
    HEADERS, BASE, SOLVERS, SOLVE_PROMPT, solve_one, answers_match,
)


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--out", required=True)
    p.add_argument("--solvers", default="qwen3.5-9b")
    p.add_argument("--force", action="store_true", help="re-run even if output exists")
    args = p.parse_args()

    out_path = Path(args.out)
    if out_path.exists() and not args.force:
        existing = json.loads(out_path.read_text())
        if existing.get("n", 0) >= args.n:
            print(f"[no-aux-baseline] {out_path} already has n={existing['n']} (≥{args.n}); skipping. Use --force to re-run.")
            return

    wanted = set(s.strip() for s in args.solvers.split(","))
    active_solvers = [s for s in SOLVERS if s[0] in wanted]
    assert active_solvers, f"no solvers match --solvers={args.solvers}"

    from datasets import load_dataset
    os.environ.setdefault("HF_HOME", "/mnt/cmlssd004/public/chanwoo/hf")
    os.environ.setdefault("HF_DATASETS_CACHE", "/mnt/cmlssd004/public/chanwoo/hf/datasets")
    ds = load_dataset("hiyouga/geometry3k", split="test")

    # Prepare bare images (no overlay) for first N test samples
    samples = []
    for idx in range(min(args.n, len(ds))):
        problem = str(ds[idx]["problem"]).replace("<image>", "").strip()
        gold = str(ds[idx]["answer"])
        buf = io.BytesIO()
        ds[idx]["images"][0].convert("RGB").save(buf, format="PNG")
        samples.append((idx, problem, buf.getvalue(), gold))

    t0 = time.time()
    async with httpx.AsyncClient(
        limits=httpx.Limits(max_connections=64, max_keepalive_connections=32)
    ) as client:
        tasks = []
        tag_list = []
        for si, (idx, problem, png, gold) in enumerate(samples):
            for (label, model, mxt) in active_solvers:
                tasks.append(solve_one(client, label, model, mxt, problem, png))
                tag_list.append((si, label))
        print(f"[no-aux-baseline] {len(tasks)} API calls concurrent across {len(active_solvers)} solvers", flush=True)
        results = await asyncio.gather(*tasks, return_exceptions=True)

    per_sample = []
    for si, (idx, problem, png, gold) in enumerate(samples):
        row = {"idx": idx, "gold": gold}
        for (label, _, _) in active_solvers:
            row[label] = None
        per_sample.append(row)

    for (si, label), res in zip(tag_list, results):
        if isinstance(res, Exception):
            per_sample[si][label] = {"answer": None, "correct": False, "error": str(res)[:200]}
        else:
            ans = res.get("answer")
            correct = answers_match(ans, per_sample[si]["gold"])
            per_sample[si][label] = {"answer": ans, "correct": correct, "raw": res.get("raw", "")}

    per_solver = {}
    for (label, _, _) in active_solvers:
        n_total, n_correct = 0, 0
        for row in per_sample:
            v = row[label]
            if v and "correct" in v:
                n_total += 1
                n_correct += int(bool(v["correct"]))
        per_solver[label] = {
            "rate": n_correct / n_total if n_total else 0.0,
            "n": n_total,
            "n_correct": n_correct,
        }

    out = {
        "n": len(samples),
        "solvers": [s[0] for s in active_solvers],
        "wallclock_s": time.time() - t0,
        "per_solver": per_solver,
        "per_sample": per_sample,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"[no-aux-baseline] done in {time.time()-t0:.1f}s — per-solver: {per_solver}")
    print(f"[no-aux-baseline] wrote {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
