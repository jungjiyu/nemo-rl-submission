#!/bin/bash
set -u
PY=/opt/ray_venvs/nemo_rl.models.policy.workers.dtensor_policy_worker_v2.DTensorPolicyWorkerV2/bin/python
REPO=/ephemeral/vuvlm/nemo-RL-render-reward
EVAL_BASE=$REPO/evals
LOG=$EVAL_BASE/compare.log

echo "[$(date +%H:%M:%S)] === BEGIN step_25 vs base comparison ===" > $LOG

run_one() {
    local name="$1" model="$2"
    local OUT=$EVAL_BASE/$name
    mkdir -p "$OUT"
    echo "[$(date +%H:%M:%S)] --- $name: generate (model=$model) ---" >> $LOG
    $PY $REPO/tools/inference/generate_mathcanvas.py \
        --model-dir "$model" --n 32 --dp 8 --out-dir "$OUT" \
        >> $LOG 2>&1
    rc=$?
    if [ $rc -ne 0 ]; then echo "[$(date +%H:%M:%S)] $name GEN FAIL rc=$rc" >> $LOG; return $rc; fi
    echo "[$(date +%H:%M:%S)] --- $name: score ---" >> $LOG
    $PY $REPO/tools/inference/score_lpips.py --in-dir "$OUT" >> $LOG 2>&1
}

run_one step_25 /ephemeral/vuvlm/nemo-RL-v0.5.0/results/sft_lora/step_25_merged
run_one base   nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16

echo "[$(date +%H:%M:%S)] === DONE ===" >> $LOG
touch $EVAL_BASE/compare.done
