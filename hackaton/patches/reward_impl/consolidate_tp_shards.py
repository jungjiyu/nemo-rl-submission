"""Consolidate NeMo-RL TP=4 sharded safetensors into HF-loadable format.

The training checkpoints under tmp_step_N/policy/weights/model/ look like:
    shard-00001-model-00001-of-00001.safetensors  (~6.2 GB, 759 tensors, TP rank 0)
    shard-00002-model-00001-of-00001.safetensors  (TP rank 1)
    shard-00003-model-00001-of-00001.safetensors  (TP rank 2)
    shard-00004-model-00001-of-00001.safetensors  (TP rank 3)

Each shard contains all 759 tensor names, but each tensor is a 1/4 slice
along the TP axis. This script:

1. Opens all 4 shards and builds a tensor-name → [shape_per_rank] map.
2. For each tensor, detects the TP axis as the dim that varies across ranks.
3. If no dim varies (replicated tensor — e.g. layernorm weights), uses rank-0.
4. Otherwise concatenates along the TP axis.
5. Writes an HF-style sharded output:
   - model-00001-of-00002.safetensors, model-00002-of-00002.safetensors
     (two ~12 GB shards, safetensors's 50 GB per-shard limit is generous but
     smaller shards load faster)
   - model.safetensors.index.json

Usage:
    V2=/mnt/tmp/ray_venvs/nemo_rl.models.policy.workers.dtensor_policy_worker_v2.DTensorPolicyWorkerV2
    TORCH_LIB=$V2/lib/python3.12/site-packages/torch/lib
    LD_LIBRARY_PATH=$TORCH_LIB:$LD_LIBRARY_PATH $V2/bin/python hackathon/consolidate_tp_shards.py \\
        --in-dir results/vlm-grpo-geometry3k-100step/tmp_step_10/policy/weights/model \\
        --out-dir /mnt/tmp/rl_step10_hf_full \\
        --shard-bytes 12000000000
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from typing import Dict, List, Tuple

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def _collect_shard_paths(in_dir: str) -> List[str]:
    paths = sorted(glob.glob(os.path.join(in_dir, "shard-*.safetensors")))
    if not paths:
        raise SystemExit(f"no shard-*.safetensors found in {in_dir}")
    return paths


def _open_shards(paths: List[str]):
    return [safe_open(p, framework="pt", device="cpu") for p in paths]


def _gather_tensor_shapes(readers) -> Dict[str, List[Tuple[int, ...]]]:
    """name -> list of shape tuples, indexed by TP rank."""
    shapes: Dict[str, List[Tuple[int, ...]]] = {}
    for rank, reader in enumerate(readers):
        for name in reader.keys():
            shapes.setdefault(name, [None] * len(readers))
            t = reader.get_slice(name)
            shapes[name][rank] = tuple(t.get_shape())
    return shapes


def _detect_tp_axis(shapes_per_rank: List[Tuple[int, ...]]) -> int:
    """Return the dim index that varies across ranks, or -1 if all identical.

    Skips ranks with None shape (tensor absent on that rank).
    """
    present = [s for s in shapes_per_rank if s is not None]
    if len(present) <= 1:
        return -1
    s0 = present[0]
    for d in range(len(s0)):
        if any(shape[d] != s0[d] for shape in present[1:]):
            return d
    return -1


def _load_and_concat(
    readers, name: str, tp_axis: int, shapes_per_rank: List[Tuple[int, ...]],
) -> torch.Tensor:
    """Load and concatenate along TP axis. Handles ranks where tensor is absent."""
    parts = []
    for rank, reader in enumerate(readers):
        if shapes_per_rank[rank] is None:
            continue
        parts.append(reader.get_tensor(name))
    if tp_axis < 0:
        # Replicated: return the first present copy (all are identical)
        return parts[0]
    return torch.cat(parts, dim=tp_axis)


def _plan_output_shards(
    name_to_size: Dict[str, int], shard_bytes: int
) -> Tuple[Dict[str, int], int]:
    """Pack tensors greedily into output shards of ~shard_bytes each.

    Returns (name -> output-shard-idx (1-based), num_output_shards).
    """
    # Order: iterate names in fqn_to_file_index_mapping order if available,
    # otherwise sorted alphabetically. Use simple greedy bin-packing.
    plan: Dict[str, int] = {}
    current_shard = 1
    current_bytes = 0
    for name, sz in sorted(name_to_size.items()):
        if current_bytes + sz > shard_bytes and current_bytes > 0:
            current_shard += 1
            current_bytes = 0
        plan[name] = current_shard
        current_bytes += sz
    return plan, current_shard


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--shard-bytes", type=int, default=12_000_000_000)
    ap.add_argument("--copy-metadata-from", default=None,
                    help="dir to copy non-weight files (config.json, modeling*.py, tokenizer, etc.) from. "
                         "If not given, tries <in-dir>/.hf_metadata then <parent_of_in_dir_parent>/tokenizer")
    ap.add_argument("--dry-run", action="store_true", help="report plan only, don't write")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    shard_paths = _collect_shard_paths(args.in_dir)
    print(f"[consolidate] found {len(shard_paths)} input shards:")
    for p in shard_paths:
        print(f"  {p} ({os.path.getsize(p)/1e9:.2f} GB)")

    readers = _open_shards(shard_paths)
    t0 = time.time()
    shapes = _gather_tensor_shapes(readers)
    print(f"[consolidate] collected shapes for {len(shapes)} tensors in {time.time()-t0:.1f}s")

    # Detect TP axis and full shape for each tensor
    full_info: Dict[str, Tuple[int, Tuple[int, ...]]] = {}  # name -> (tp_axis, full_shape)
    total_bytes = 0
    axis_counts = {-1: 0, 0: 0, 1: 0, 2: 0, 3: 0}
    for name, shape_list in shapes.items():
        axis = _detect_tp_axis(shape_list)
        present = [s for s in shape_list if s is not None]
        if axis == -1:
            full_shape = present[0]
        else:
            s0 = list(present[0])
            s0[axis] = sum(shape[axis] for shape in present)
            full_shape = tuple(s0)
        full_info[name] = (axis, full_shape)
        axis_counts[axis] = axis_counts.get(axis, 0) + 1
        # Estimate bytes (bf16 = 2 bytes)
        numel = 1
        for d in full_shape:
            numel *= d
        total_bytes += numel * 2

    print(f"[consolidate] TP axis distribution: {axis_counts}")
    print(f"[consolidate] full model total bytes (bf16): {total_bytes/1e9:.2f} GB")

    # Plan output shards
    name_to_size = {n: 1 for n in full_info}  # placeholder; size by numel*dtype
    for name, (_, shape) in full_info.items():
        numel = 1
        for d in shape:
            numel *= d
        name_to_size[name] = numel * 2  # bf16
    plan, num_out = _plan_output_shards(name_to_size, args.shard_bytes)
    print(f"[consolidate] output plan: {num_out} shards, ~{args.shard_bytes/1e9:.1f} GB each")

    if args.dry_run:
        print("[consolidate] DRY RUN — exiting before write")
        return

    # Write output shards
    weight_map: Dict[str, str] = {}
    out_filename = lambda idx: f"model-{idx:05d}-of-{num_out:05d}.safetensors"

    for target_shard in range(1, num_out + 1):
        tensors_for_shard = {}
        for name, shard_idx in plan.items():
            if shard_idx != target_shard:
                continue
            tp_axis, _ = full_info[name]
            tensors_for_shard[name] = _load_and_concat(readers, name, tp_axis, shapes[name])
            weight_map[name] = out_filename(target_shard)
        out_path = os.path.join(args.out_dir, out_filename(target_shard))
        print(f"[consolidate] writing {out_path} with {len(tensors_for_shard)} tensors...")
        save_file(tensors_for_shard, out_path, metadata={"format": "pt"})
        del tensors_for_shard  # free memory before next shard

    # Write index
    index = {
        "metadata": {"total_size": sum(os.path.getsize(os.path.join(args.out_dir, out_filename(i))) for i in range(1, num_out + 1))},
        "weight_map": weight_map,
    }
    with open(os.path.join(args.out_dir, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=2)
    print(f"[consolidate] wrote model.safetensors.index.json ({len(weight_map)} entries)")

    # Copy metadata (config.json etc.) so HF can load
    import shutil
    meta_src = args.copy_metadata_from
    if meta_src is None:
        # default: look for .hf_metadata sibling (step_N format)
        # in-dir is e.g. tmp_step_N/policy/weights/model — the .hf_metadata is a sibling under weights/model/
        candidates = [
            os.path.join(args.in_dir, ".hf_metadata"),
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(args.in_dir))), "policy/weights/model/.hf_metadata"),
        ]
        meta_src = next((c for c in candidates if os.path.isdir(c)), None)
    if meta_src and os.path.isdir(meta_src):
        print(f"[consolidate] copying metadata from {meta_src}")
        for fn in os.listdir(meta_src):
            shutil.copy(os.path.join(meta_src, fn), args.out_dir)
    else:
        print(f"[consolidate] WARNING: no metadata source found; you'll need to hand-copy config.json, modeling*.py, tokenizer files")

    print(f"[consolidate] DONE. Output: {args.out_dir}")


if __name__ == "__main__":
    main()
