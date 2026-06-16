#!/bin/bash
# Download Qwen/Qwen3.5-2B to local dir via hf-mirror.com
set -euo pipefail

PYTHON="${PYTHON:-/home/guo/anaconda3/bin/python}"
LOCAL_DIR="${LOCAL_DIR:-/home/guo/LLM/SIRL/Qwen3.5-2B}"
REPO_ID="${REPO_ID:-Qwen/Qwen3.5-2B}"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export no_proxy="${no_proxy:-127.0.0.1,::1,localhost,hf-mirror.com,huggingface.co}"

# Cursor/sandbox proxy breaks large xet downloads; unset for this script.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy

mkdir -p "$LOCAL_DIR"
echo "Downloading ${REPO_ID} -> ${LOCAL_DIR} (mirror: ${HF_ENDPOINT})"

"$PYTHON" -m huggingface_hub.cli download "$REPO_ID" --local-dir "$LOCAL_DIR"

echo "Done. Model size:"
du -sh "$LOCAL_DIR"
ls -lh "$LOCAL_DIR"/*.safetensors 2>/dev/null || true
