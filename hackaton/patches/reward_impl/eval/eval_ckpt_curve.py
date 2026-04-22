"""Evaluate a single run at multiple checkpoint steps, producing a curve.

For each step in --steps (default 25,50,75,100):
  1. ckpt prep (run hackathon/prep_ckpt_for_eval.sh on <run_dir>/step_N)
  2. gen SVGs via hackathon/eval_step10.py on n=100 (produces fmt+geom scores too)
  3. judge-ensemble eval (3 judges × 100 samples, existing script)
  4. open-ended solve Δ vs no-aux baseline
  5. aggregate into <run_dir>/curve.json

Budget per step: ~2min ckpt prep + ~10min generation + ~20min judge + ~5min solve = ~35min/step.
With --steps 25,50,75,100 = ~2.5h per run (serial).

Usage:
    /mnt/tmp/nemo_rl_venv/bin/python hackathon/eval_ckpt_curve.py \\
        --run-dir /mnt/ddn/vuvlm/mmpt/chanwoo/sprint6/j_only_0p2_0p8_0p0 \\
        --run-id j_only \\
        --steps 25,50,75,100 \\
        --n 100

Writes:
    hackathon/reports/eval_<run_id>_step<N>_n100.json        (fmt+geom + gen)
    hackathon/reports/eval_<run_id>_step<N>_n100_judge.json  (judge ensemble)
    hackathon/reports/solve_open_<run_id>_step<N>_n100.json  (solver Δ)
    <run_dir>/curve.json                                      (aggregated summary)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path("/mnt/cmlssd004/public/chanwoo/hackaton/nemo-RL")
REPORTS = REPO / "hackathon" / "reports"
V2 = "/mnt/tmp/ray_venvs/nemo_rl.models.policy.workers.dtensor_policy_worker_v2.DTensorPolicyWorkerV2"
VENV = "/mnt/tmp/nemo_rl_venv"


def run(cmd: list[str], env=None, check=True) -> int:
    print(f"[curve] $ {' '.join(cmd)}", flush=True)
    t0 = time.time()
    rc = subprocess.run(cmd, env=env, check=False).returncode
    print(f"[curve]   → rc={rc} in {time.time()-t0:.1f}s", flush=True)
    if check and rc != 0:
        raise SystemExit(rc)
    return rc


def eval_one_step(run_dir: Path, run_id: str, step: int, n: int, no_aux_baseline: Path):
    ckpt_src = run_dir / f"step_{step}"
    if not (ckpt_src / "policy" / "weights" / "model" / "consolidated").is_dir():
        print(f"[curve] SKIP step {step}: {ckpt_src}/policy/weights/model/consolidated missing")
        return None

    hf_name = f"{run_id}_step{step}_hf"
    hf_dst = Path(f"/mnt/tmp/{hf_name}")

    out_gen = REPORTS / f"eval_{run_id}_step{step}_n{n}.json"
    out_judge = REPORTS / f"eval_{run_id}_step{step}_n{n}_judge.json"
    out_solve = REPORTS / f"solve_open_{run_id}_step{step}_n{n}.json"

    # 1. ckpt prep
    if not hf_dst.is_dir():
        run(["bash", str(REPO / "hackathon" / "prep_ckpt_for_eval.sh"),
             str(ckpt_src), hf_name])
    else:
        print(f"[curve] skip ckpt prep ({hf_dst} exists)")

    # 2. gen SVGs via parallel 4-GPU eval (node_B's eval_step10_parallel.sh, 4× speedup)
    # Single-GPU fallback kept commented below for reference.
    if not out_gen.exists():
        run(["bash", str(REPO / "hackathon" / "eval_step10_parallel.sh"),
             "rl", str(n), str(hf_dst), str(out_gen)])
        # Fallback single-GPU path (kept for debug; SUPERSEDED by parallel):
        # env = {**os.environ,
        #        "LD_LIBRARY_PATH": f"{V2}/lib/python3.12/site-packages/torch/lib:" + os.environ.get("LD_LIBRARY_PATH", "")}
        # run([f"{V2}/bin/python", str(REPO / "hackathon" / "eval_step10.py"),
        #      "--which", "rl", "--n", str(n), "--max-new", "1536",
        #      "--rl-path", str(hf_dst), "--out", str(out_gen)], env=env)
    else:
        print(f"[curve] skip gen eval ({out_gen} exists)")

    # 3. judge ensemble
    if not out_judge.exists():
        run([f"{VENV}/bin/python", str(REPO / "hackathon" / "eval_judge_ensemble.py"),
             "--input", str(out_gen), "--n", str(n), "--out", str(out_judge)])
    else:
        print(f"[curve] skip judge ensemble ({out_judge} exists)")

    # 4. open-ended solve Δ
    if not out_solve.exists():
        run([f"{VENV}/bin/python", str(REPO / "hackathon" / "eval_solve_open.py"),
             "--gen-input", str(out_gen),
             "--no-aux-baseline", str(no_aux_baseline),
             "--n", str(n),
             "--out", str(out_solve)])
    else:
        print(f"[curve] skip solve Δ ({out_solve} exists)")

    # read outputs, pack summary
    summary = {"step": step}
    try:
        gen_d = json.loads(out_gen.read_text())
        summary["svg_format_rate"] = gen_d.get("svg_format_rate")
        summary["svg_geometric_rate"] = gen_d.get("svg_geometric_rate")
    except Exception as e:
        summary["gen_error"] = str(e)[:100]
    try:
        j_d = json.loads(out_judge.read_text())
        summary["judge_mean"] = j_d.get("per_judge_mean_0_10", {})
        vals = [v for v in summary["judge_mean"].values() if isinstance(v, (int, float))]
        summary["judge_spread"] = (max(vals) - min(vals)) if vals else None
    except Exception as e:
        summary["judge_error"] = str(e)[:100]
    try:
        s_d = json.loads(out_solve.read_text())
        summary["solve_per_solver"] = s_d.get("per_solver", {})
    except Exception as e:
        summary["solve_error"] = str(e)[:100]
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True, help="root dir of the GRPO run (contains step_N subdirs)")
    p.add_argument("--run-id", required=True, help="short id used in output filenames")
    p.add_argument("--steps", default="25,50,75,100")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--no-aux-baseline", default=str(REPORTS / "no_aux_solve_baseline.json"))
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    assert run_dir.is_dir(), f"run_dir not found: {run_dir}"
    steps = [int(s) for s in args.steps.split(",")]
    no_aux_baseline = Path(args.no_aux_baseline)
    if not no_aux_baseline.exists():
        raise SystemExit(f"no_aux_baseline not found: {no_aux_baseline}. Run hackathon/tools/no_aux_baseline.py first.")

    t0 = time.time()
    curve = {"run_id": args.run_id, "run_dir": str(run_dir), "n": args.n, "steps": []}
    for step in steps:
        print(f"\n[curve] ========== step {step} ==========")
        summ = eval_one_step(run_dir, args.run_id, step, args.n, no_aux_baseline)
        if summ is not None:
            curve["steps"].append(summ)

    curve["wallclock_s"] = time.time() - t0
    out = run_dir / "curve.json"
    out.write_text(json.dumps(curve, indent=2, default=str))
    print(f"\n[curve] done in {curve['wallclock_s']:.1f}s")
    print(f"[curve] wrote {out}")
    for s in curve["steps"]:
        jm = s.get("judge_mean", {})
        sps = s.get("solve_per_solver", {})
        print(f"  step={s['step']:>3} fmt={s.get('svg_format_rate',''):.3f} geom={s.get('svg_geometric_rate',''):.3f} "
              f"judge_mean={jm} judge_spread={s.get('judge_spread',None)} solve={ {k: v.get('delta') for k,v in sps.items()} }")


if __name__ == "__main__":
    main()
