"""
Merge a LoRA adapter trained by NeMo-RL (PEFT-format output) back into the base
`nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16` model and save the full merged
checkpoint so that vLLM can serve it directly.
"""

import argparse
import json
import shutil
from pathlib import Path

# peft >=0.19 insists torchao >= 0.16 via `is_torchao_available()` and raises
# ImportError when that call is exercised during LoRA dispatch. Our adapter is
# plain Linear LoRA (no torchao quant), so patch the check out at every
# site that has already imported the original reference.
import peft.import_utils as _peft_imp
_peft_imp.is_torchao_available = lambda: False
from peft.tuners.lora import torchao as _lora_torchao  # noqa: E402
_lora_torchao.is_torchao_available = lambda: False

import torch  # noqa: E402
from peft import PeftModel  # noqa: E402
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--base-model", default=None,
                    help="Override base model name; defaults to adapter_config.json base_model_name_or_path")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    adapter_dir = Path(args.adapter_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with (adapter_dir / "adapter_config.json").open() as f:
        adapter_cfg = json.load(f)
    base_name = args.base_model or adapter_cfg["base_model_name_or_path"]
    print(f"[merge] base model: {base_name}")
    print(f"[merge] adapter:    {adapter_dir}")
    print(f"[merge] out:        {out_dir}")

    print("[merge] loading base model (bf16, untied embeddings)...")
    # tie_word_embeddings=False is critical when the LoRA adapter includes
    # lm_head: the base model normally shares one tensor between
    # embed_tokens and lm_head, so merge_and_unload() would write the LoRA
    # delta into that shared tensor and corrupt embed_tokens too. Loading
    # untied gives lm_head its own tensor (initially a clone of
    # embed_tokens) that can absorb the merge cleanly.
    base = AutoModelForCausalLM.from_pretrained(
        base_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        tie_word_embeddings=False,
    )
    base.eval()

    # Some checkpoints store the shared tensor only under embed_tokens; make
    # sure lm_head actually holds a copy so LoRA merge doesn't land on a
    # meta/empty tensor.
    try:
        lm_head = getattr(base, "lm_head", None) or getattr(
            base.model, "lm_head", None
        )
        embed = None
        for m in (base, getattr(base, "model", None), getattr(base, "language_model", None)):
            if m is None:
                continue
            e = getattr(m, "embed_tokens", None)
            if e is not None:
                embed = e
                break
            be = getattr(getattr(m, "backbone", None), "embeddings", None) if getattr(m, "backbone", None) else None
            if be is not None:
                embed = be
                break
        if lm_head is not None and embed is not None and lm_head.weight.data_ptr() == embed.weight.data_ptr():
            print("[merge] cloning tied lm_head weight into independent tensor...")
            lm_head.weight = torch.nn.Parameter(embed.weight.data.clone())
    except Exception as e:
        print(f"[merge] WARN: lm_head untie probe failed: {e!r}")

    print("[merge] attaching LoRA adapter...")
    peft_model = PeftModel.from_pretrained(base, str(adapter_dir))

    print("[merge] merge_and_unload...")
    merged = peft_model.merge_and_unload()

    # Persist untied state so vLLM doesn't retie at load time.
    try:
        merged.config.tie_word_embeddings = False
    except Exception:
        pass

    print("[merge] save_pretrained (safetensors)...")
    merged.save_pretrained(str(out_dir), safe_serialization=True)

    print("[merge] saving tokenizer & processor...")
    tokenizer = AutoTokenizer.from_pretrained(base_name, trust_remote_code=True)
    tokenizer.save_pretrained(str(out_dir))
    try:
        processor = AutoProcessor.from_pretrained(base_name, trust_remote_code=True)
        processor.save_pretrained(str(out_dir))
    except Exception as e:
        print(f"[merge] WARN: processor save failed: {e}")

    print("[merge] copying custom code files from adapter dir...")
    for py in adapter_dir.glob("*.py"):
        dst = out_dir / py.name
        if not dst.exists():
            shutil.copy2(py, dst)
            print(f"         + {py.name}")

    print(f"[merge] DONE -> {out_dir}")


if __name__ == "__main__":
    main()
