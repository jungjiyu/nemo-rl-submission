#!/usr/bin/env bash
# Run inside container (vuvlm-v050) from /ephemeral/vuvlm/nemo-RL-v0.5.0/
# Foreground: output goes to current tmux pane.

set -euo pipefail
cd /ephemeral/vuvlm/nemo-RL-v0.5.0

NCCL_P2P_DISABLE=1 \
NCCL_SHM_DISABLE=1 \
NCCL_DEBUG=WARN \
PYTORCH_ALLOC_CONF=expandable_segments:True \
uv run python examples/run_vlm_sft.py \
    --config examples/configs/sft_vlm_nemotron_nano_v2_llava_smoke.yaml
