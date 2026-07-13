#!/bin/bash
# Full Partial-KL train on 6×A800. Run inside tmux/screen.
#
#   tmux new -s sirl
#   bash training_scripts/start_full_6xa800.sh
#
# Default: 1 epoch (2 often unnecessary for first full run).
# Batch default 66 (nearest to 68 that divides by 6).
#
# Monitor:
#   tail -f /root/autodl-tmp/checkpoints/SIRL/partialKL_6xa800_qwen3_8b/train.log
#   watch -n 5 nvidia-smi

set -euo pipefail
cd /root/llm/sirl/SIRLTrain

export http_proxy= https_proxy= HTTP_PROXY= HTTPS_PROXY= ALL_PROXY= all_proxy=
export NO_PROXY='*' no_proxy='*'

FRESH_START=1 \
TRAIN_MAX_SAMPLES=-1 \
VAL_MAX_SAMPLES=-1 \
TOTAL_EPOCHS=1 \
TRAIN_BATCH_SIZE=66 \
SAVE_FREQ=15 \
MAX_ACTOR_CKPT_TO_KEEP=1 \
  bash training_scripts/run_partialKL_6xa800_8b.sh
