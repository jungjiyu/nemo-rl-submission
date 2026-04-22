"""vLLM DP generation on mathcanvas samples — step 1 of checkpoint eval.

Writes `<out-dir>/predictions.jsonl` with one row per sample. Pair with
`score_lpips.py` to compute LPIPS afterwards. Keeping the two phases
separate means:
  - metric iteration (LPIPS net, image size) doesn't re-generate.
  - batch generation across many checkpoints once, score later.

Fast-inference design:
- DATA-PARALLEL via Ray actors: `dp` actors, each with tensor_parallel_size=1
  and its own vLLM instance on one GPU. 12B bf16 fits on one GPU, so TP has
  no memory benefit and would pay all-reduce cost per generate() for small
  batches. DP splits the N prompts round-robin across actors.
- Greedy decoding (temperature=0) — single-valued metric per checkpoint.

Example:
    uv run python tools/inference/generate_mathcanvas.py \\
        --model-dir /ephemeral/vuvlm/nemo-RL-v0.5.0/results/sft_lora/step_275 \\
        --n 32 --dp 8 \\
        --out-dir /ephemeral/vuvlm/nemo-RL-render-reward/evals/step_275

For LoRA checkpoints, merge into an HF dir first via
tools/inference/merge_lora_nemotron_vl.py — vLLM loads full weights only.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import ray
from PIL import Image
from transformers import AutoTokenizer


MATHCANVAS_IMG_MARKER = "sft_llava_prod_foundational"  # path marker from mix_v1


def normalize_row(r: dict, jsonl_parent: Path) -> tuple[str, str, str]:
    """Extract (image_path, instruction_text, gt_svg_text) from either:

    - openai format (mix_v1/train_*.jsonl):
        {"messages": [{"role":"user", "content":[{"type":"image","image":"/abs/..."},
                                                 {"type":"text","text":"..."}]},
                      {"role":"assistant", "content":[{"type":"text","text":"<svg>"}]}]}
    - LLaVA format (sft_llava_prod_foundational_jsonl/train_*.jsonl):
        {"id": "...", "image": "images/bin_.../x.png",
         "conversations": [{"from":"human", "value":"<image>\\n..."},
                           {"from":"gpt", "value":"<svg>"}]}
        — image paths are relative to the JSONL's parent dir.
    """
    if "messages" in r:
        img = text = gt = None
        for m in r["messages"]:
            if m["role"] == "user":
                for c in m["content"]:
                    if c["type"] == "image":
                        img = c["image"]
                    elif c["type"] == "text":
                        text = c["text"]
            elif m["role"] == "assistant":
                for c in m["content"]:
                    if c["type"] == "text":
                        gt = c["text"]
        if img is None or text is None or gt is None:
            raise ValueError("openai-format row missing required fields")
        return img, text, gt

    if "conversations" in r:
        img_rel = r.get("image")
        if img_rel is None:
            raise ValueError("llava-format row missing top-level 'image'")
        img = str(jsonl_parent / img_rel)
        text = gt = None
        for conv in r["conversations"]:
            if conv.get("from") == "human":
                text = conv["value"].replace("<image>", "").strip()
            elif conv.get("from") == "gpt":
                gt = conv["value"]
        if text is None or gt is None:
            raise ValueError("llava-format row missing human/gpt turn")
        return img, text, gt

    raise ValueError(f"unknown row format (keys: {sorted(r.keys())})")


def load_mathcanvas_samples(
    data_jsonl: Path, n: int, seed: int
) -> list[tuple[str, str, str]]:
    """Return up to n normalized (image_path, instruction, gt_svg) tuples.

    For openai-format sources we filter rows to mathcanvas via path marker
    because mix_v1 interleaves geo3k. For LLaVA-format sources (the raw
    mathcanvas foundational dump) every row is already mathcanvas, so we
    keep them all.
    """
    parent = data_jsonl.parent
    out: list[tuple[str, str, str]] = []
    with open(data_jsonl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            is_openai = "messages" in r
            is_llava = "conversations" in r
            try:
                img, text, gt = normalize_row(r, parent)
            except ValueError:
                continue
            if is_openai and MATHCANVAS_IMG_MARKER not in img:
                continue
            out.append((img, text, gt))
            if is_llava and len(out) >= 10 * n:
                # LLaVA files are big (~5k rows); stop early to save a shuffle
                # over the whole file. 10x headroom is enough for the shuffle.
                break
    rng = random.Random(seed)
    rng.shuffle(out)
    return out[:n]


@ray.remote(num_gpus=1)
class VLLMWorker:
    def __init__(
        self,
        model_dir: str,
        max_model_len: int,
        max_new_tokens: int,
        gpu_memory_utilization: float,
        enforce_eager: bool,
        seed: int,
    ) -> None:
        from vllm import LLM, SamplingParams

        self.llm = LLM(
            model=model_dir,
            trust_remote_code=True,
            tensor_parallel_size=1,
            dtype="bfloat16",
            max_model_len=max_model_len,
            limit_mm_per_prompt={"image": 1},
            enforce_eager=enforce_eager,
            gpu_memory_utilization=gpu_memory_utilization,
            seed=seed,
        )
        self.sp = SamplingParams(
            temperature=0.0, top_p=1.0, max_tokens=max_new_tokens
        )

    def generate(self, prompts: list[dict]) -> list[dict]:
        if not prompts:
            return []
        outs = self.llm.generate(prompts, self.sp)
        return [
            {
                "text": o.outputs[0].text,
                "finish_reason": o.outputs[0].finish_reason,
                "output_tokens": len(o.outputs[0].token_ids),
            }
            for o in outs
        ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True,
                    help="HF-format model directory vLLM can load.")
    ap.add_argument("--data-jsonl", type=Path,
                    default=Path("/ephemeral/vuvlm/sft-data/sft_llava_prod_foundational_jsonl/train_00500.jsonl"),
                    help="Default is a held-out foundational (LLaVA-format) file "
                         "to minimize train/val overlap with mix_v1.")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--dp", type=int, default=8,
                    help="Data-parallel actor count. Each gets 1 GPU via Ray.")
    ap.add_argument("--max-new-tokens", type=int, default=4096)
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--enforce-eager", action="store_true", default=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = load_mathcanvas_samples(args.data_jsonl, args.n, args.seed)
    print(
        f"[gen] loaded {len(samples)} mathcanvas samples from {args.data_jsonl}",
        flush=True,
    )
    if len(samples) < args.n:
        print(f"[gen] WARNING: wanted {args.n}, got {len(samples)}", flush=True)

    print(f"[gen] loading tokenizer from {args.model_dir}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)

    prompts: list[dict] = []
    meta: list[dict] = []
    for img_path, instr, gt_text in samples:
        chat_msgs = [{
            "role": "user",
            "content": [
                {"type": "image", "image": ""},
                {"type": "text", "text": instr},
            ],
        }]
        prompt_text = tokenizer.apply_chat_template(
            chat_msgs, tokenize=False, add_generation_prompt=True,
        )
        base_img = Image.open(img_path).convert("RGB")
        prompts.append(
            {"prompt": prompt_text, "multi_modal_data": {"image": base_img}}
        )
        meta.append({
            "image_path": img_path,
            "instruction": instr,
            "gt_svg": gt_text,
        })

    dp = min(args.dp, len(prompts)) or 1
    if not ray.is_initialized():
        # Explicitly cap resources. Without this, Ray auto-detects the full
        # container CPU count (e.g. 252) and pre-starts that many Python
        # workers — each importing vLLM — which hangs forever.
        ray.init(
            ignore_reinit_error=True,
            num_cpus=max(16, dp * 2),
            num_gpus=dp,
        )
    print(f"[gen] spawning {dp} vLLM workers (1 GPU each)", flush=True)
    workers = [
        VLLMWorker.remote(
            model_dir=args.model_dir,
            max_model_len=args.max_model_len,
            max_new_tokens=args.max_new_tokens,
            gpu_memory_utilization=args.gpu_memory_utilization,
            enforce_eager=args.enforce_eager,
            seed=args.seed,
        )
        for _ in range(dp)
    ]

    # Round-robin split: preserves global index recovery via stride.
    shards_prompts: list[list[dict]] = [prompts[i::dp] for i in range(dp)]
    shard_global_idx: list[list[int]] = [
        list(range(i, len(prompts), dp)) for i in range(dp)
    ]

    t0 = time.time()
    shard_results = ray.get(
        [w.generate.remote(sp) for w, sp in zip(workers, shards_prompts)]
    )
    t_gen = time.time() - t0

    # Scatter back into original order.
    gen_by_idx: list[dict | None] = [None] * len(prompts)
    for gidx_list, shard_out in zip(shard_global_idx, shard_results):
        for gidx, row in zip(gidx_list, shard_out):
            gen_by_idx[gidx] = row
    assert all(r is not None for r in gen_by_idx), "missing generation output"

    print(
        f"[gen] generated {len(prompts)} in {t_gen:.1f}s "
        f"({t_gen / max(1, len(prompts)):.2f}s / sample, dp={dp})",
        flush=True,
    )

    # Persist — score_lpips.py will pick this up.
    with (out_dir / "predictions.jsonl").open("w") as f:
        for idx, (m, g) in enumerate(zip(meta, gen_by_idx)):
            f.write(json.dumps({
                "idx": idx,
                "image_path": m["image_path"],
                "instruction": m["instruction"],
                "gt_svg": m["gt_svg"],
                "pred_text": g["text"],
                "finish_reason": g["finish_reason"],
                "output_tokens": g["output_tokens"],
            }, ensure_ascii=False) + "\n")
    with (out_dir / "generation_info.json").open("w") as f:
        json.dump({
            "model_dir": args.model_dir,
            "data_jsonl": str(args.data_jsonl),
            "n": len(prompts),
            "dp": dp,
            "max_new_tokens": args.max_new_tokens,
            "max_model_len": args.max_model_len,
            "seed": args.seed,
            "generation_seconds": t_gen,
        }, f, indent=2)

    print(
        f"[gen] wrote {out_dir / 'predictions.jsonl'} "
        f"(+ generation_info.json). Run score_lpips.py next.",
        flush=True,
    )

    # Clean up Ray actors — they're holding GPU weights.
    for w in workers:
        ray.kill(w)
    ray.shutdown()


if __name__ == "__main__":
    main()
