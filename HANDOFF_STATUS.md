# Autonomous handoff — status

**Started:** 2026-04-21 ~22:15 UTC (user stepped away).

## Plan

1. Monitor `results/sft_lora/step_300` until it lands.
2. Kill `run_vlm_sft.py` once step_300 is saved.
3. Merge each of {step_200, step_225, step_250, step_275, step_300} via
   `tools/inference/merge_lora_nemotron_vl.py`.
4. For each merged ckpt: `generate_mathcanvas.py` (32 samples from
   held-out foundational `train_00500.jsonl`) → `score_lpips.py`.
5. Pick best by lowest `mean_lpips_all` (tie-break: higher
   `render_ok_rate`, then lower `mean_lpips_render_ok_only`).
6. Run RL debug: 2 steps on best ckpt with `grpo_geometry3k_judge.yaml`.
7. Launch RL full: `save_period=10`, `max_num_steps=300`.

## Status log

- 22:15 UTC: SFT observed at step_275 (loss 0.296), running. Eval scripts
  committed (`cc7c1776`). Monitoring scheduled.
- 22:32 UTC: SFT still alive (tfevents fresh, 8 GPU at 100% inside
  container vuvlm-v050). step_300 ETA ~22:37. NOTE: SFT process runs
  inside Docker container, not visible to host `pgrep`.
- 22:37 UTC: step_300 saved, loss 0.279. Sent `pkill -f run_vlm_sft.py`
  via `docker exec`. Ray workers + raylet went defunct; 8 GPUs freed
  (0 MiB) by 22:39.
- 22:39 UTC: Starting Phase E — merge 5 SFT LoRA ckpts to HF dirs.
- 22:40 UTC: First merge attempt failed — used `/opt/nemo_rl_venv/bin/python` which
  lacks `peft`. Relaunched with dtensor_v2 venv python
  (`/opt/ray_venvs/nemo_rl.models.policy.workers.dtensor_policy_worker_v2.DTensorPolicyWorkerV2/bin/python`).
- 22:41 UTC: 5 merges running in parallel on GPU 0-4 inside vuvlm-v050.
  Logs: `logs_sft_lora/merge_step_{200,225,250,275,300}.log`.
  Completion markers: `merge_step_N.done` (rc=X).
- 22:46 UTC: All 5 merges rc=0. Each `step_N_merged/` has 6 safetensors
  shards. GPUs fully freed.
- 22:47 UTC: Installed `resvg-py` + `lpips` into the dtensor_v2 venv
  inside the container (pyproject updates haven't been uv-synced yet).
  Launched sequential eval driver `evals/run_all.sh` via
  `docker exec -d`; runs generate_mathcanvas.py + score_lpips.py for
  step_{200,225,250,275,300} back-to-back. Log: `evals/eval.log`,
  completion marker: `evals/eval.done`.
- 22:53 UTC: step_200 still in vLLM DP worker init (~5min elapsed, 0
  GPU memory observed). 8 workers × ~24GB disk reads of base weights —
  cold-cache takes a few minutes. Polling again at 22:58.
- 22:58 UTC: Still stuck, 10min in, 0 GPU memory. Raylet log shows 252
  pre-started Python workers never registering — Ray auto-detected
  CPU=252 on the container and spawned that many worker processes,
  each trying to import vLLM. Killed the stuck driver + patched
  `generate_mathcanvas.py` to `ray.init(num_cpus=max(16,dp*2), num_gpus=dp)`.
- 22:59 UTC: Eval driver relaunched. Monitoring again at 23:04.
- 23:04 UTC: step_200 eval completed but **n_render_ok=0** (all 32 outputs
  garbage: `<svg /svg <svg .0100.0100...`). Root cause identified from the
  merge log — the model has `tie_word_embeddings=True`, so the LoRA's
  lm_head adapter got merged into the shared embed_tokens tensor too,
  corrupting both. Confirmed broken via a probe using the known-good
  `vllm_infer_val.py` against step_200_merged (same garbage).
- 23:10 UTC: Patched `merge_lora_nemotron_vl.py`:
  - Load base with `tie_word_embeddings=False` so lm_head is independent.
  - After load, clone `embed_tokens.weight` into `lm_head.weight` if still
    tied by pointer.
  - After merge_and_unload(), set `merged.config.tie_word_embeddings=False`
    so vLLM loads the model untied.
  Relaunched all 5 merges on GPU 0-4.
- 23:17 UTC: All re-merges rc=0, but probe on step_200_merged STILL
  garbage. Inspection: lm_head.weight and backbone.embeddings.weight are
  now different (LoRA delta applied only to lm_head, good), BUT
  `config.json` lacked a `tie_word_embeddings` key — default True would
  re-tie at vLLM load. Patched `config.json` in all 5 merged dirs to
  explicitly set `tie_word_embeddings: false` at both top-level and in
  `llm_config`.
- 23:22 UTC: Re-probed after config patch — still garbage. vLLM custom
  loader appears to tie regardless of config.
- 23:23 UTC: Pivot — drop `lm_head` from the LoRA adapter entirely so
  merge never touches the tied tensor. Wrote filtered adapters at
  `results/sft_lora/step_N_adapter_nolm/` (target_modules minus
  lm_head, adapter_model.safetensors without lm_head lora_A/B).
  Relaunched step_200 merge using the nolm adapter.
- 23:31 UTC: Probe on step_200_merged (nolm) — still garbage ("G G G G …").
- 23:32 UTC: Base-model probe — coherent output ("To extend the given
  geometric diagram according to the edit instruction..."). Rules out
  vLLM/prompt bug. Bug is in the SFT-trained adapter, not the template.
- 23:34 UTC: HF-native + PEFT probe (greedy and temp=1.0/top_k=20) —
  still garbage. Output distribution has right tokens (`fill="red"`,
  `<svg`, numeric attrs) but no valid XML nesting. Greedy mode-collapse
  ruled out.
- 23:39 UTC: Per user suggestion, testing the overfitting hypothesis:
  merging step_{25,50,75,100,125} (earliest ckpts) with the nolm fix
  in parallel on GPU 0-4.
- 23:41 UTC: Per user, bumped HF-probe `max_new_tokens` 400 → 1024 and
  re-probed step_200_merged with sampling (temp=1.0, top_k=20). Still
  garbage — extends longer but stays structurally broken. Output:
  `evals/hf_probes/step_200_sample1024_20260421_234100.txt`.
- 23:45 UTC: Probed full SFT LoRA adapters (not merged) against base in
  parallel HF-native + PEFT runs for step_{25,50,75,100,125}. Outputs:
  `evals/hf_probes/step_<N>_<ts>.txt`. **Result: SFT collapsed between
  step_25 and step_50.**
    - step_25: coherent English explanation ("To extend the given
      geometric diagram according to the edit instruction..."),
      essentially still the base model (loss was 9.68 — barely trained).
    - step_50/75/100/125/200: all garbage in different shapes.

**DIAGNOSIS.** The SFT run from `sft_vlm_nemotron_nano_v2_llava_smoke.yaml`
is unusable: the model catastrophically degraded very early and every
checkpoint ≥ step_50 produces unstructured token salad. Training loss
continued to drop (11.79 → 0.28 by step_275) because the model learned
the *token distribution* of SVG text (lots of `fill=`, `x=`, `y=`,
numeric attributes, `<g`/`<line`) even while destroying sequence
coherence. Loss on memorised per-token statistics descended; generation
quality did not.

Possible root causes worth investigating before the next SFT run:
  1. `policy.optimizer.lr = 1e-4` is aggressive for LoRA-dim=8; common
     safe range is 1e-5–5e-5. Step_25 → step_50 is ~17 optimizer steps;
     if the update size was excessive, divergence lands right there.
  2. `data.add_generation_prompt: false` with the multimodal
     image-placeholder tokens — verify the loss mask only covers
     assistant tokens, not user+image placeholders. If the image
     placeholder tokens end up in the loss, the model trains to
     predict placeholder IDs interleaved with SVG, which would explain
     the structural collapse.
  3. `NRL_VISION_AC` (vision activation checkpointing) plus the
     `input_conditioner.norm_mean/std` missing-weight warning (seen in
     every merge log) together may be corrupting image embeddings
     during SFT's backward pass. A SFT probe using HF-native on the
     BASE model (same prompt and image) is coherent — so vLLM / prompt
     are not the cause.

**ACTION: autonomous handoff paused here.** All 5 RL-candidate
checkpoints (200/225/250/275/300) are broken. Launching RL warm-start
from any of them would waste many GPU hours. Best-ckpt selection
(task #18), RL debug (#19), and RL full (#20) are now **blocked on a
user decision** — either (a) redo SFT with revised settings, or (b)
fall back to a different base ckpt. HANDOFF_STATUS + memory updated;
GPUs free and ready for whatever the user chooses.
- 00:03 UTC (04-22): User requested (1) parser robustness for ``` fenced
  outputs and (2) a step_25 vs base comparison. (1) landed —
  `extract_svg` in `tools/inference/score_lpips.py` and
  `nemo_rl/hackaton/render/svg_render.py` now handles complete and
  truncated fences (svg/xml/html langs), and recovers truncated `<svg>`
  with a synthetic `</svg>` close. Smoke test passes on 7 shapes.
  (2) launched — `evals/run_compare.sh` runs both step_25_merged and
  base via the same DP-8 generate + score pipeline.
- 00:11 UTC: both evals completed. Black-background initial render made
  the comparison bogus (GT's black base strokes invisible on black bg) —
  fixed `render_svg_to_pil` + `svg_render.render` to composite onto
  WHITE before flattening RGBA → RGB. Re-scored with --save-renders.
- Side-by-side PNGs under `evals/compare_samples/`. Corrected numbers
  (white bg):
  - step_25: render_ok=0.875, mean_lpips_all=0.573, ok_only=0.512
  - base:    render_ok=0.875, mean_lpips_all=0.517, ok_only=0.448
  - **step_25 +0.056 LPIPS worse than base** — even after only 25
    optimizer steps, the adapter has moved visibly AWAY from the GT
    style. idx=0 visual: base still attempts a circle-plus-polygons
    composition near the GT shape; step_25 produces a chunky generic
    trapezoid unrelated to the problem. Confirms the catastrophic-drift
    finding: training went wrong direction from the first ~25 steps.
