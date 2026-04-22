"""Parse C.10 monitor + train log and plot RL reward trajectory.

Usage:
    /mnt/tmp/nemo_rl_venv/bin/python hackathon/tools/plot_reward_trajectory.py \\
        --log logs/train/retry_20260421T141857Z_attempt1.log \\
        --out hackathon/reports/reward_s6a_sftwarm.png

Parses:
    [C.10 monitor] step=N loss_mult_mean=M n_rollouts=R reward_mean=X reward_var=Y
                    pct_allzero_groups=Z% min_group_std=S

Plots 4 panels: reward_mean, reward_var, pct_allzero_groups, min_group_std.
Since per-component (fmt/judge/solve) not logged individually, only aggregate.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_C10 = re.compile(
    r"\[C\.10 monitor\] step=(?P<step>\d+) loss_mult_mean=(?P<lm>[\d.]+) "
    r"n_rollouts=(?P<n>\d+) reward_mean=(?P<rm>[\d.-]+) reward_var=(?P<rv>[\d.-]+) "
    r"pct_allzero_groups=(?P<pz>[\d.]+)% min_group_std=(?P<ms>[\d.-]+)"
)


def parse_log(log_path: Path) -> dict[str, list]:
    data = {"step": [], "reward_mean": [], "reward_var": [], "pct_allzero": [], "min_std": []}
    for line in log_path.read_text(errors="ignore").splitlines():
        m = _C10.search(line)
        if m:
            data["step"].append(int(m["step"]))
            data["reward_mean"].append(float(m["rm"]))
            data["reward_var"].append(float(m["rv"]))
            data["pct_allzero"].append(float(m["pz"]))
            data["min_std"].append(float(m["ms"]))
    return data


def plot(data: dict, out: Path, title: str = "") -> None:
    if not data["step"]:
        print("[plot] no C.10 monitor lines in log")
        return
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    s = data["step"]
    axes[0, 0].plot(s, data["reward_mean"], "b.-")
    axes[0, 0].set_title("reward_mean"); axes[0, 0].set_xlabel("step"); axes[0, 0].grid(True, alpha=0.3)
    axes[0, 1].plot(s, data["reward_var"], "g.-")
    axes[0, 1].set_title("reward_var"); axes[0, 1].set_xlabel("step"); axes[0, 1].grid(True, alpha=0.3)
    axes[1, 0].plot(s, data["pct_allzero"], "r.-")
    axes[1, 0].set_title("pct_allzero_groups (%)"); axes[1, 0].set_xlabel("step"); axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axhline(95, color="red", linestyle="--", alpha=0.5, label="C.10 threshold")
    axes[1, 1].plot(s, data["min_std"], "m.-")
    axes[1, 1].set_title("min_group_std"); axes[1, 1].set_xlabel("step"); axes[1, 1].grid(True, alpha=0.3)
    fig.suptitle(title or f"RL trajectory ({len(s)} steps)")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=100)
    print(f"[plot] wrote {out} ({len(s)} points)")
    # Also print text summary
    print(f"  reward_mean: first={data['reward_mean'][0]:.3f}  last={data['reward_mean'][-1]:.3f}  "
          f"mean={sum(data['reward_mean'])/len(s):.3f}  max={max(data['reward_mean']):.3f}")
    print(f"  allzero≥25%: {sum(1 for p in data['pct_allzero'] if p >= 25)} / {len(s)} steps")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--title", default="")
    args = p.parse_args()
    data = parse_log(Path(args.log))
    plot(data, Path(args.out), title=args.title)


if __name__ == "__main__":
    main()
