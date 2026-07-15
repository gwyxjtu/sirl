#!/bin/bash
# Resume full train from last checkpoint (step 30).
# Only use when checkpoint exists and hasn't been deleted by FRESH_START.
#
# Monitor:
#   tail -f /root/autodl-tmp/checkpoints/SIRL/partialKL_6xa800_qwen3_8b/train.log

set -euo pipefail
cd /root/llm/sirl/SIRLTrain

export http_proxy= https_proxy= HTTP_PROXY= HTTPS_PROXY= ALL_PROXY= all_proxy=
export NO_PROXY='*' no_proxy='*'

SKIP_CLEANUP=1 \
RESUME_MODE=auto \
FRESH_START=0 \
TRAIN_MAX_SAMPLES=-1 \
VAL_MAX_SAMPLES=-1 \
TOTAL_EPOCHS=1 \
TRAIN_BATCH_SIZE=66 \
SAVE_FREQ=15 \
MAX_ACTOR_CKPT_TO_KEEP=1 \
  bash training_scripts/run_partialKL_6xa800_8b.sh
