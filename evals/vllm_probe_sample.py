"""vLLM probe with temperature=1.0 top_k=20 — test if sampling rescues the output."""
import json
from PIL import Image
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

MODEL = "/ephemeral/vuvlm/nemo-RL-v0.5.0/results/sft_lora/step_200_merged"
PROBE = "/ephemeral/vuvlm/nemo-RL-render-reward/evals/val_probe.jsonl"

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
llm = LLM(
    model=MODEL, trust_remote_code=True,
    tensor_parallel_size=1, dtype="bfloat16",
    max_model_len=8192, limit_mm_per_prompt={"image": 1},
    enforce_eager=True, gpu_memory_utilization=0.85,
)
sp = SamplingParams(temperature=1.0, top_k=20, max_tokens=400)

row = json.loads(open(PROBE).readline())
img_path = instr = None
for c in row["messages"][0]["content"]:
    if c["type"] == "image": img_path = c["image"]
    elif c["type"] == "text": instr = c["text"]

messages = [
    {"role": "system", "content": "/no_think"},
    {"role": "user", "content": [
        {"type": "image", "image": ""},
        {"type": "text", "text": instr},
    ]},
]
prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
img = Image.open(img_path).convert("RGB")

outs = llm.generate([{"prompt": prompt, "multi_modal_data": {"image": img}}], sp)
gen = outs[0].outputs[0].text
print("=== vLLM GEN (first 800 chars) ===", flush=True)
print(gen[:800], flush=True)
