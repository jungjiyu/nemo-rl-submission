"""
Run greedy inference with vLLM on the first 5 validation samples of
llava_smoke, using the merged Nemotron-Nano-12B-v2-VL LoRA checkpoint.

For each val sample we take only the first user turn (image + text) and let
the model generate the assistant response, then dump predictions + GT to
jsonl for quick inspection.

Invoke via dtensor_v2 venv (ships vllm 0.11.2):
  CUDA_VISIBLE_DEVICES=0,1 \
    /opt/ray_venvs/.../DTensorPolicyWorkerV2/bin/python \
    tools/inference/vllm_infer_val.py \
    --model-dir  /ephemeral/vuvlm/nemo-RL-v0.5.0/results/sft_nemotron_vl_lora/step_10_merged \
    --val-jsonl  /ephemeral/vuvlm/sft-data/llava_smoke/llava_smoke_val.jsonl \
    --n 5 \
    --out-jsonl  /ephemeral/vuvlm/nemo-RL-v0.5.0/results/sft_nemotron_vl_lora/inference/val_step10.jsonl
"""

import argparse
import json
from pathlib import Path

from PIL import Image
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def first_user_turn(messages):
    """Return (image_path, text) from the first user turn. Assumes exactly one image."""
    for m in messages:
        if m["role"] != "user":
            continue
        img_path, text = None, None
        for c in m["content"]:
            if c["type"] == "image":
                img_path = c["image"]
            elif c["type"] == "text":
                text = c["text"]
        return img_path, text
    raise ValueError("no user turn found")


def first_assistant_turn(messages):
    for m in messages:
        if m["role"] == "assistant":
            for c in m["content"]:
                if c["type"] == "text":
                    return c["text"]
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--val-jsonl", required=True)
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--tp", type=int, default=2,
                    help="tensor_parallel_size (keep within one NVLink pair; default 2)")
    ap.add_argument("--max-new-tokens", type=int, default=8192)
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--enforce-eager", action="store_true", default=True,
                    help="Mamba hybrid + CUDA graph sometimes misbehaves; eager first.")
    args = ap.parse_args()

    model_dir = args.model_dir
    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load first N samples from val jsonl.
    samples = []
    with open(args.val_jsonl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
            if len(samples) >= args.n:
                break
    print(f"[infer] loaded {len(samples)} val samples")

    print(f"[infer] loading tokenizer from {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

    print(f"[infer] initialising vLLM (tp={args.tp})")
    llm = LLM(
        model=model_dir,
        trust_remote_code=True,
        tensor_parallel_size=args.tp,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        limit_mm_per_prompt={"image": 1},
        enforce_eager=args.enforce_eager,
        gpu_memory_utilization=0.85,
    )

    sp = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.max_new_tokens,
    )

    # Build prompts.
    prompts = []
    meta = []  # per sample: dict with image_path, user_text, gt
    for s in samples:
        img_path, user_text = first_user_turn(s["messages"])
        gt = first_assistant_turn(s["messages"])
        chat_msgs = [{
            "role": "user",
            "content": [
                {"type": "image", "image": ""},
                {"type": "text", "text": user_text},
            ],
        }]
        prompt_text = tokenizer.apply_chat_template(
            chat_msgs, tokenize=False, add_generation_prompt=True,
        )
        img = Image.open(img_path).convert("RGB")
        prompts.append({
            "prompt": prompt_text,
            "multi_modal_data": {"image": img},
        })
        meta.append({
            "image_path": img_path,
            "user_text": user_text,
            "gt": gt,
            "prompt_text": prompt_text,
        })

    print(f"[infer] generating {len(prompts)} prompts (greedy, max_new={args.max_new_tokens})")
    outputs = llm.generate(prompts, sp)

    with out_path.open("w") as fout:
        for m, o in zip(meta, outputs):
            pred = o.outputs[0].text
            row = {
                "image_path": m["image_path"],
                "user_text": m["user_text"],
                "gt": m["gt"],
                "prediction": pred,
                "prompt_text": m["prompt_text"],
                "finish_reason": o.outputs[0].finish_reason,
                "prompt_token_ids_len": len(o.prompt_token_ids),
                "output_token_ids_len": len(o.outputs[0].token_ids),
            }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            print("=" * 60)
            print("IMG:", m["image_path"])
            print("Q  :", m["user_text"])
            print("GT :", m["gt"])
            print("PRED (first 300 chars):", pred[:300].replace("\n", " "))

    print(f"\n[infer] wrote {len(outputs)} rows -> {out_path}")


if __name__ == "__main__":
    main()
