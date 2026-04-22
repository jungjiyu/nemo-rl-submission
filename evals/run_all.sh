#!/bin/bash
set -u
PY=/opt/ray_venvs/nemo_rl.models.policy.workers.dtensor_policy_worker_v2.DTensorPolicyWorkerV2/bin/python
REPO=/ephemeral/vuvlm/nemo-RL-render-reward
EVAL_BASE=$REPO/evals
LOG=$EVAL_BASE/eval.log

echo "[$(date +%H:%M:%S)] === BEGIN eval driver ===" > $LOG
cd $REPO

for i in 200 225 250 275 300; do
  OUT=$EVAL_BASE/step_$i
  mkdir -p "$OUT"
  MERGED=/ephemeral/vuvlm/nemo-RL-v0.5.0/results/sft_lora/step_${i}_merged

  echo "[$(date +%H:%M:%S)] --- step_$i: generate ---" >> $LOG
  $PY tools/inference/generate_mathcanvas.py \
    --model-dir "$MERGED" \
    --n 32 --dp 8 \
    --out-dir "$OUT" \
    >> $LOG 2>&1
  gen_rc=$?
  if [ "$gen_rc" -ne 0 ]; then
    echo "[$(date +%H:%M:%S)] step_$i GENERATE FAILED rc=$gen_rc" >> $LOG
    touch "$OUT/_generate_failed"
    continue
  fi

  echo "[$(date +%H:%M:%S)] --- step_$i: score ---" >> $LOG
  $PY tools/inference/score_lpips.py --in-dir "$OUT" >> $LOG 2>&1
  sc_rc=$?
  if [ "$sc_rc" -ne 0 ]; then
    echo "[$(date +%H:%M:%S)] step_$i SCORE FAILED rc=$sc_rc" >> $LOG
    touch "$OUT/_score_failed"
  fi
done

echo "[$(date +%H:%M:%S)] === END eval driver ===" >> $LOG
touch $EVAL_BASE/eval.done
