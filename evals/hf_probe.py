"""HF-native + PEFT probe (fixed): mirrors quick_test_image.py API.

Loads base + applies LoRA adapter (no merge), runs inference on 1 val sample.
If output is coherent → adapter is fine, merge is the bug.
If garbage → adapter (or training) is broken.
"""
import torch, json
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
import peft.import_utils as _peft_imp
_peft_imp.is_torchao_available = lambda: False
from peft.tuners.lora import torchao as _lora_torchao
_lora_torchao.is_torchao_available = lambda: False
from peft import PeftModel
from PIL import Image

import sys
BASE = "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16"
# Usage: hf_probe.py [adapter_path]. Default points at step_200 for
# backward compat with the initial probe run.
ADAPTER = sys.argv[1] if len(sys.argv) > 1 else \
    "/ephemeral/vuvlm/nemo-RL-v0.5.0/results/sft_lora/step_200/policy/weights/model"
PROBE = "/ephemeral/vuvlm/nemo-RL-render-reward/evals/val_probe.jsonl"

print("[probe] loading base...", flush=True)
base = AutoModelForCausalLM.from_pretrained(
    BASE, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="cuda:0",
).eval()
print("[probe] attaching full LoRA (lm_head included)...", flush=True)
mdl = PeftModel.from_pretrained(base, ADAPTER).eval()

tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
proc = AutoProcessor.from_pretrained(BASE, trust_remote_code=True)

row = json.loads(open(PROBE).readline())
img_path = instr = None
for c in row["messages"][0]["content"]:
    if c["type"] == "image": img_path = c["image"]
    elif c["type"] == "text": instr = c["text"]
print("img:", img_path, flush=True)

img = Image.open(img_path).convert("RGB")
# Match quick_test_image.py message shape — includes /no_think system hint.
messages = [
    {"role": "system", "content": "/no_think"},
    {"role": "user", "content": [
        {"type": "image", "image": ""},
        {"type": "text", "text": instr},
    ]},
]
prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = proc(text=[prompt], images=[img], return_tensors="pt").to("cuda:0")

print("[probe] generating (match quick_test API)...", flush=True)
with torch.no_grad():
    out = mdl.generate(
        pixel_values=inputs.pixel_values,
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=1024,
        do_sample=True,
        temperature=1.0,
        top_k=20,
        eos_token_id=tok.eos_token_id,
    )
gen = tok.decode(out[0, inputs.input_ids.shape[-1]:], skip_special_tokens=False)
print("=== GEN (first 800 chars) ===", flush=True)
print(gen[:800], flush=True)

# Persist full output for later comparison across ckpts.
import os, datetime, re
out_dir = "/ephemeral/vuvlm/nemo-RL-render-reward/evals/hf_probes"
os.makedirs(out_dir, exist_ok=True)
m = re.search(r"step_(\d+)", ADAPTER)
tag = f"step_{m.group(1)}" if m else os.path.basename(ADAPTER.rstrip("/"))
ts = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
path = f"{out_dir}/{tag}_{ts}.txt"
with open(path, "w") as f:
    f.write(f"# adapter: {ADAPTER}\n# probe_row_img: {img_path}\n")
    f.write(f"# sampling: do_sample=True temp=1.0 top_k=20 max_new_tokens=1024\n")
    f.write(f"# ==== full generation ====\n{gen}\n")
print(f"[probe] saved full gen to {path}", flush=True)
