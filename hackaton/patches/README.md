# Sprint-6 Reward System — Integration Guide

Recipe for integrating our SVG-aux reward system (fmt + judge + solve) into a
vanilla NeMo-RL checkout for Geometry3K + Nemotron Nano 12B VL.

**Scope**: reward functions + framework integration points. LoRA/training
infrastructure is **out of scope** (assumed to work already in target setup).

---

## 1. Architecture (what we built)

```
┌─────────────────────────────────────────────────────────────────┐
│  GRPO rollout: N prompts × K gens per prompt                    │
│                                                                 │
│  For each sample i:                                             │
│    prompt (image + question text)                               │
│    ground_truth = JSON({question, image_b64, answer, _format})  │ ← dataset envelope
│    response = model-generated SVG                               │
└────────────────────────────┬────────────────────────────────────┘
                             │
                 ┌───────────▼────────────┐
                 │ VLMVerifyWorker.verify │
                 └─────────────┬──────────┘
                               │ 2-tier dispatch
        ┌──────────────────────┼───────────────────────┐
        │                                              │
┌───────▼────────────┐                   ┌─────────────▼──────────┐
│ per-sample rewards │                   │  batch (group) rewards │
│ (CPU, no API)      │                   │  (async OpenRouter)    │
│                    │                   │                        │
│ svg_format  (r_fmt)│                   │ svg_judge_group        │
│                    │                   │   → 1 API / group      │
│                    │                   │                        │
│                    │                   │ svg_solve_aux          │
│                    │                   │   → (N+1) API / group  │
└────────────────────┘                   └────────────────────────┘
        │                                              │
        └──────────────────────┬───────────────────────┘
                               ▼
              weighted_sum + renormalize → per-sample reward
                               │
                               ▼
                       GRPO advantage calc
```

**Key ideas:**
- **Group-aware rewards** exploit GRPO's N-gen-per-prompt structure: judge compares
  N candidates in one call; solver gets N+1 calls (1 shared no-aux baseline + N
  with-aux) fired concurrently via `asyncio.gather`.
- **Paradigm B**: model output SVG is rendered directly as the judged/solved
  image — NO overlay on original. Model is expected to reproduce the diagram AND
  add red dashed aux in the same SVG. Requires SFT bootstrap for the reproduction.
- **Scheme B solver scoring**: `{0,1,2}` for {hurt, neutral, helped} rather than
  `{-1,0,+1}`, to play well with GRPO's group-normalization.
- **JSON envelope in ground_truth**: instead of a separate image channel, the
  dataset packs everything the reward needs (image, question, gold) into one
  string that survives NeMo-RL's metadata serialization.

---

## 2. File layout in this patch bundle

```
hackathon/patches/
├── README.md                           # you are here
├── reward_impl/                        # NEW files — drop into target repo
│   ├── prompts.py                      # → hackathon/prompts.py
│   ├── consolidate_tp_shards.py        # → hackathon/consolidate_tp_shards.py
│   ├── rewards/                        # → hackathon/rewards/
│   │   ├── r_fmt.py                    #   per-sample SVG format reward (CPU)
│   │   ├── r_geom.py                   #   per-sample structural reward (CPU)
│   │   ├── r_judge.py                  #   group-async judge reward (API)
│   │   └── r_solve.py                  #   group-async solver-Δ reward (API)
│   ├── render/                         # → hackathon/render/
│   │   ├── __init__.py
│   │   └── svg_render.py               #   SVG → PNG via cairosvg
│   ├── eval/                           # → hackathon/ (flat), tools/ for baseline
│   │   ├── eval_judge_ensemble.py      #   n=100 judge eval (3 judges)
│   │   ├── eval_solve_open.py          #   n=100 open-ended solve Δ
│   │   ├── no_aux_baseline.py          #   precomputed no-aux solver baseline
│   │   ├── eval_ckpt_curve.py          #   driver: eval at step_25/50/75/100
│   │   └── plot_reward_trajectory.py   #   parse C.10 log → 4-panel plot
│   └── config/
│       └── grpo_geometry3k_judge.yaml  # → examples/configs/ (reward weights cfg)
│
└── framework_diffs/                    # unified diffs against upstream NeMo-RL
    ├── 01-rewards-batch-dispatcher.patch
    ├── 02-vlm-environment-2tier-verify.patch
    ├── 03-geometry3k-json-envelope.patch
    └── 04-grpo-c10-silent-masking-assertions.patch
```

---

## 3. Apply patches (integration steps)

From the root of a fresh NeMo-RL checkout:

### Step 1 — Copy new files

```bash
# Place our reward implementation
mkdir -p hackathon/rewards hackathon/render hackathon/tools
cp reward_impl/prompts.py       hackathon/
cp reward_impl/consolidate_tp_shards.py hackathon/
cp reward_impl/rewards/*.py     hackathon/rewards/
cp reward_impl/render/*.py      hackathon/render/
cp reward_impl/eval/eval_judge_ensemble.py hackathon/
cp reward_impl/eval/eval_solve_open.py     hackathon/
cp reward_impl/eval/eval_ckpt_curve.py     hackathon/
cp reward_impl/eval/no_aux_baseline.py     hackathon/tools/
cp reward_impl/eval/plot_reward_trajectory.py hackathon/tools/
cp reward_impl/config/grpo_geometry3k_judge.yaml examples/configs/
```

### Step 2 — Apply framework diffs

Each diff adds hooks into NeMo-RL. **Order matters** (03 depends on 01+02):

```bash
cd <NeMo-RL root>
git apply framework_diffs/01-rewards-batch-dispatcher.patch
git apply framework_diffs/02-vlm-environment-2tier-verify.patch
git apply framework_diffs/03-geometry3k-json-envelope.patch
git apply framework_diffs/04-grpo-c10-silent-masking-assertions.patch
```

If diffs don't apply cleanly (NeMo-RL version skew), see §5 for what each change
does so you can port manually.

### Step 3 — Python deps

Beyond NeMo-RL's own environment, rewards need:
```
httpx          # async OpenRouter client (probably already present)
cairosvg       # SVG → PNG rendering  ← the one you may need to add
Pillow         # already required by NeMo-RL VLM path
```

Install cairosvg in **every actor venv** that runs the env worker (typically
`nemo_rl_venv` + `ray_venvs/*/` depending on your setup).

### Step 4 — Configure API keys

Copy `.env` file or export:
```bash
export OPENROUTER_API_KEY=sk-or-...
export OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
export OPENROUTER_HTTP_REFERER=<optional>
export OPENROUTER_APP_NAME=<optional>
```

Reward modules auto-load `.env` from repo root if present.

---

## 4. YAML configuration

To use the rewards in your own GRPO run, in your yaml:

```yaml
env:
  geometry3k:
    num_workers: 4
    reward_functions:
      - { name: svg_format,       weight: 0.2 }   # per-sample, CPU
      - { name: svg_judge_group,  weight: 0.4 }   # batch, judge API
      - { name: svg_solve_aux,    weight: 0.4 }   # batch, solver API
```

The dispatcher (in `nemo_rl/environments/rewards.py` after patch 01) knows which
are per-sample vs batch via the `_BATCH_REWARD_NAMES` frozenset.

Weights can be anything; they're renormalized in `verify()`.

---

## 5. What each framework diff does

### `01-rewards-batch-dispatcher.patch` → `nemo_rl/environments/rewards.py`

**Adds:**
- `_BATCH_REWARD_NAMES = frozenset({"svg_judge_group", "svg_solve_aux"})`
- `_parse_gt_payload(gt: str)` — parses the JSON envelope
- `_group_by_gt(gts, responses)` — groups indices by identical ground_truth
- `_run_group_calls(gts, responses, async_fn)` — runs one async task per group
  inside a single `asyncio.gather` (fire-all-gather pattern)
- `svg_judge_group_scores(gts, responses) -> list[float]`
- `svg_solve_aux_scores(gts, responses) -> list[float]`

The dispatcher introspects `async_fn.__signature__` to detect if it accepts
`gold_answer` (solver does, judge doesn't) — so one helper serves both.

### `02-vlm-environment-2tier-verify.patch` → `nemo_rl/environments/vlm_environment.py`

**Modifies `VLMVerifyWorker`:**
- `__init__` splits `reward_functions` into `per_sample_pairs` vs `batch_pairs`
  by looking up `_BATCH_REWARD_NAMES`.
- `verify(responses, gts)`:
  1. For each `(response, gt)`: call each per-sample reward; accumulate weighted.
  2. For each batch reward: call once on the full list; accumulate weighted.
  3. Fuse + normalize by total weight.

Output shape is unchanged (`list[float]`) — caller never sees the split.

### `03-geometry3k-json-envelope.patch` → `nemo_rl/data/datasets/response_datasets/geometry3k.py`

**Changes `format_geometry3k_dataset`:**
- Sets assistant content to `json.dumps({"question", "image_b64", "answer",
  "_format": "sprint6_v1"})`.
- This string becomes `ground_truth` in the GRPO rollout metadata.
- Reward modules parse it via `_parse_gt_payload` in patch 01.

**Why**: NeMo-RL's VLM env only passes a single `ground_truth` string to
`verify()` — no separate image channel. This envelope is our workaround.

### `04-grpo-c10-silent-masking-assertions.patch` → `nemo_rl/algorithms/grpo.py`

**Adds defensive assertions** in both the vanilla and async paths before
`advantage_calculation`:

```python
# Computes: loss_mult_mean, reward_mean, reward_var,
#           pct_allzero_groups, min_group_std
assert _lm_mean > 0.5, "silent masking triggered"
assert _pct_allzero_groups < 0.95, "catastrophic group collapse"
```

Plus a `[C.10 monitor]` print line per step — parsed by
`plot_reward_trajectory.py`. Skippable via `NRL_DISABLE_C10_ASSERTS=1`.

---

## 6. How a reward call flows (end-to-end)

**Tracing one group of 4 rollouts:**

1. **Dataset** builds 4 samples with the same `ground_truth` JSON (one per prompt).

2. **Rollout** generates 4 candidate SVGs → `responses = [svg1, svg2, svg3, svg4]`.

3. **Env** calls `VLMVerifyWorker.verify(responses, ground_truths)`.

4. **Per-sample tier**: for each `(svg_i, gt_i)`:
   - `svg_format` parses `<svg>...</svg>` tag + calls `svg_render` → 1.0 or 0.0
   - Accumulated weighted into `per_sample_accum[i]`

5. **Batch tier** (for `svg_judge_group`):
   - `_group_by_gt` → 1 group of 4 indices (same ground_truth)
   - Call `r_judge_group(client, question, image_bytes, [svg1..4])`:
     - Render each SVG → PNG via `svg_render`
     - Build ONE OpenRouter request with 4 images + prompt asking judge to rank
     - Parse JSON response `{scores: [s0,s1,s2,s3]}`
     - Return `[s0/10, s1/10, s2/10, s3/10]` (0 for render-failed)
   - Scatter 4 scores back into `batch_accum` indices

6. **Batch tier** (for `svg_solve_aux`, same group):
   - Call `r_solve_group(client, question, image_bytes, [svg1..4], gold_answer)`:
     - Render each SVG → PNG
     - Fire 5 solver API calls in one `asyncio.gather`:
       - 1× no-aux (original image) — the shared baseline
       - 4× with-aux (rendered SVG)
     - Extract `\boxed{...}` from each response
     - Scheme B: for each candidate,
       - correct_noaux=True, correct_with=False → **0** (hurt)
       - same outcome → **1** (neutral)
       - correct_noaux=False, correct_with=True → **2** (helped)
   - Scatter 4 scores back

7. **Fuse**:
   ```
   final[i] = (w_fmt·sf[i] + w_judge·sj[i] + w_solve·ss[i]) / (w_fmt+w_judge+w_solve)
   ```

8. Return to GRPO; group-normalized advantages computed.

---

## 7. Eval pipeline (after training)

### One-time setup: precompute no-aux baseline
```bash
/mnt/tmp/nemo_rl_venv/bin/python hackathon/tools/no_aux_baseline.py \
    --n 100 --out hackathon/reports/no_aux_solve_baseline.json
# ~5-10 min, $1-2 API. Cache reused across all run evals.
```

### Per-ckpt eval curve
```bash
/mnt/tmp/nemo_rl_venv/bin/python hackathon/eval_ckpt_curve.py \
    --run-dir /path/to/sprint6/<run_id> \
    --run-id <run_id> \
    --steps 25,50,75,100 \
    --n 100 \
    --no-aux-baseline hackathon/reports/no_aux_solve_baseline.json
```

For each step this runs:
1. `prep_ckpt_for_eval.sh <ckpt> <hf_name>` — consolidate TP shards to HF format
2. `eval_step10.py` (or `eval_step10_parallel.sh` for 4-GPU) — gen SVGs, compute fmt+geom
3. `eval_judge_ensemble.py` — 3 judges (qwen122b + qwen9b + glm)
4. `eval_solve_open.py` — open-ended solver Δ vs no-aux baseline

Produces `<run_dir>/curve.json` with metric-vs-step map.

### Live training plot
```bash
/mnt/tmp/nemo_rl_venv/bin/python hackathon/tools/plot_reward_trajectory.py \
    --log logs/train/retry_*.log \
    --out hackathon/reports/reward_trajectory.png
```
Parses `[C.10 monitor]` lines into 4-panel plot (reward_mean, var, allzero%, min_std).

---

## 8. Known issues + workarounds

### 8.1 save-on-consolidation bug

**Symptom**: `FileNotFoundError: model-00006-of-00007.safetensors` during ckpt save.
`step_N/` ends up with metadata only, no safetensors. But `tmp_step_N/` has 4
raw TP shards (~6.5GB each).

**Cause**: Automodel `_HuggingFaceStorageWriter.finish()` race when
`single_rank_consolidation=True`.

**Workaround**: After training, manually consolidate:
```bash
V2=/path/to/dtensor_v2_venv
$V2/bin/python hackathon/consolidate_tp_shards.py \
    --in-dir <run_dir>/tmp_step_N/policy/weights/model \
    --out-dir <run_dir>/step_N/policy/weights/model/consolidated \
    --shard-bytes 6000000000
```

Produces HF-loadable `model-{001,002}-of-002.safetensors` + index.json.

### 8.2 silent masking (long seqlen)

**Symptom**: Training proceeds but reward signal degrades. Cause: Geometry3K
samples with text+image >2048 tokens were silently masked by
`loss_multiplier=0.0` in the sample processor.

**Detection**: Patch 04's C.10 assertions catch this — `loss_mult_mean < 0.5`
trips immediately at step 1.

**Fix**: Set `data.max_input_seq_length: 4096` (covers p99 of Geometry3K
text+image token count).

### 8.3 group collapse (all-zero)

**Symptom**: `pct_allzero_groups=25%` spikes — one GRPO group has identical
reward across all 4 rollouts, so GRPO advantage = 0 for that group.

**Cause**: reward signal too coarse (e.g., all 4 rollouts render-fail → all get
svg_format=0 → fused reward identical), or SFT prior too strong → low diversity.

**Mitigation**: C.10 assertion threshold at 95% (not every group, some OK).
If sustained >50% for many steps, consider:
- Increasing temperature in generation
- Revisiting reward scale — solver Δ in {0,1,2} gives more granularity than binary

---

## 9. Minimum viable checklist

For a new target setup to use this reward system:

- [ ] Copy `hackathon/rewards/`, `hackathon/render/`, `hackathon/prompts.py`
- [ ] Apply 4 framework patches (or port manually — §5)
- [ ] Install `cairosvg` in actor venv(s)
- [ ] Set `OPENROUTER_API_KEY` env or `.env` file
- [ ] Verify Geometry3K dataset loads with new JSON envelope (run 1 smoke step)
- [ ] Add `reward_functions` block to yaml with 3 names + weights
- [ ] Confirm C.10 monitor line appears in log at step 1
- [ ] Launch full run

If any step fails, compare against the origin commit hashes (see `git log
--grep=sprint6` in the source repo) to find what upstream change we were
integrating against.
