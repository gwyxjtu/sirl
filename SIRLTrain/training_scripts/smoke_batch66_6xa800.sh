#!/bin/bash
# Smoke: 1–2 steps at TRAIN_BATCH_SIZE=66 to check VRAM before full train.
# Need 6 GPUs. Run in tmux; watch nvidia-smi in another pane.
#
#   bash training_scripts/smoke_batch66_6xa800.sh

set -euo pipefail
cd /root/llm/sirl/SIRLTrain

export http_proxy= https_proxy= HTTP_PROXY= HTTPS_PROXY= ALL_PROXY= all_proxy=
export NO_PROXY='*' no_proxy='*'

# Use a separate exp dir so smoke does not wipe full-run ckpts accidentally.
# Override exp by editing script or: we pass via env if supported — currently
# exp_name is hardcoded in run script, so smoke shares dir; FRESH_START clears it.
# Prefer SKIP wipe of full later: use FRESH_START=1 only for smoke when intentional.

FRESH_START=1 \
EXP_NAME=smoke_batch66_qwen3_8b \
TRAIN_MAX_SAMPLES=132 \
VAL_MAX_SAMPLES=12 \
TOTAL_EPOCHS=1 \
TRAIN_BATCH_SIZE=66 \
SAVE_FREQ=999999 \
MAX_ACTOR_CKPT_TO_KEEP=1 \
  bash training_scripts/run_partialKL_6xa800_8b.sh
