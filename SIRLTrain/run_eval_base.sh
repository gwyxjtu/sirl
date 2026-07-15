#!/bin/bash
set -euo pipefail
cd /root/llm/sirl/SIRLTrain

OUTDIR=/root/autodl-tmp/eval_base
rm -rf "$OUTDIR"
mkdir -p "$OUTDIR"

exec > "$OUTDIR/eval.log" 2>&1

CUDA_VISIBLE_DEVICES=0 python3 tools/eval_test_data_vllm.py \
  --model_path /root/autodl-tmp/models/Qwen3-8B \
  --out_dir "$OUTDIR" \
  --prompt shot \
  --tp 1 \
  --gpu_memory_utilization 0.85 \
  --temperature 0 \
  --top_p 1.0 \
  --max_tokens 3072

echo DONE
