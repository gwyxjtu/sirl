#!/bin/bash
# Partial-KL PPO training on 6× A800 (80GB each) with Qwen3-8B
# Architecture Qwen3ForCausalLM — supported by vLLM 0.10.2 + transformers 4.56.
#
# Smoke test:
#   TRAIN_MAX_SAMPLES=10 VAL_MAX_SAMPLES=5 TOTAL_EPOCHS=1 \
#     bash training_scripts/run_partialKL_6xa800_8b.sh
#
# Full run (recommended; ~3.5–4.5 days). Use tmux/screen so SSH drop is safe:
#   FRESH_START=1 TRAIN_MAX_SAMPLES=-1 VAL_MAX_SAMPLES=-1 TOTAL_EPOCHS=2 \
#     bash training_scripts/run_partialKL_6xa800_8b.sh
#
# Do NOT use auto_train.sh (it calls shutdown). Clear proxies before train.

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5"

# ── Proxies off (Gurobi WLS / downloads must not go through proxy) ──
export http_proxy= https_proxy= HTTP_PROXY= HTTPS_PROXY= ALL_PROXY= all_proxy=
export NO_PROXY='*' no_proxy='*'

# ── NCCL / OMP (avoid init hang in containerized envs without P2P/NVLS/IB) ──
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_NVLS_ENABLE=0
export NCCL_CUMEM_ENABLE=0
# export NCCL_DEBUG=INFO   # uncomment to debug NCCL init

# ── Paths ──
WORKING_DIR="/root/llm/sirl"
SIRL_DIR="/root/llm/sirl/SIRLTrain"
CKPTS_DIR="/root/autodl-tmp/checkpoints"
PYTHON="/root/miniconda3/bin/python3"

project_name='SIRL'
exp_name=${EXP_NAME:-partialKL_6xa800_qwen3_8b}
MODEL_PATH=${MODEL_PATH:-/root/autodl-tmp/models/Qwen3-8B}

# ── 训练参数 ──
TRAIN_MAX_SAMPLES=${TRAIN_MAX_SAMPLES:-10}
VAL_MAX_SAMPLES=${VAL_MAX_SAMPLES:-5}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
# Checkpoint 策略:
# - save_freq: 每 N step 存一次（全量 ~step/epoch 随 TRAIN_BATCH_SIZE 变；100 ≈ 数小时级）。
# - 只存 model（~16G），不存 optimizer；max_actor_ckpt_to_keep=1 存新删旧。
# - 写入瞬间可能新旧短暂共存（峰值 ~32G），当前盘空闲需 ≥35G。
SAVE_FREQ=${SAVE_FREQ:-100}
MAX_ACTOR_CKPT_TO_KEEP=${MAX_ACTOR_CKPT_TO_KEEP:-1}
# train_batch_size 必须能被 n_gpus(=6) 整除。68 不行 → 默认 66（最接近）。
# 也可覆盖: TRAIN_BATCH_SIZE=72 bash ...
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-66}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-6}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-4096}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-3072}
MAX_MODEL_LEN=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))

output_dir=$CKPTS_DIR/$project_name/$exp_name
mkdir -p "$output_dir"

# ── FRESH_START ──
if [[ "${FRESH_START:-0}" == "1" ]]; then
  echo "=== FRESH_START: removing checkpoints and rollout dumps in $output_dir ==="
  rm -rf "$output_dir"/global_step_* \
         "$output_dir"/rollouts \
         "$output_dir"/val_rollouts
  rm -f "$output_dir"/latest_checkpointed_iteration.txt
  echo "=== FRESH_START done ==="
fi

# ── 清理残留进程 ──
if [[ "${SKIP_CLEANUP:-0}" != "1" ]]; then
  echo "=== Cleaning up previous training processes ==="
  ray stop --force 2>/dev/null || true
  pkill -f "verl.trainer.partialKL_ppo" 2>/dev/null || true
  pkill -f "ray::WorkerDict" 2>/dev/null || true
  pkill -f "ray::vLLMHttpServer" 2>/dev/null || true
  pkill -f "ray::AgentLoopWorker" 2>/dev/null || true
  pkill -f "ray::RewardLoopWorker" 2>/dev/null || true
  pkill -f "VLLM::Worker" 2>/dev/null || true
  pkill -9 -f "VLLM::EngineCore" 2>/dev/null || true
  pkill -9 -f "EngineCore_DP" 2>/dev/null || true
  # Hard cleanup: kill any process still holding GPU memory (orphan vLLM engines
  # that survive pkill due to setproctitle). One run at a time on this box.
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
    kill -9 "$p" 2>/dev/null || true
  done
  sleep 3
  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "=== GPU memory after cleanup ==="
    nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
  fi
  echo "=== Cleanup done ==="
fi

# ── Preflight: 6 GPUs + disk ──
NGPU=$(nvidia-smi -L 2>/dev/null | wc -l)
if [[ "$NGPU" -lt 6 ]]; then
  echo "ERROR: need 6 GPUs, nvidia-smi sees $NGPU. Mount 6 cards before full train."
  exit 1
fi
AVAIL_G=$(df -BG /root/autodl-tmp | awk 'NR==2{gsub(/G/,"",$4); print $4}')
if [[ "${AVAIL_G:-0}" -lt 35 ]]; then
  echo "ERROR: /root/autodl-tmp free ${AVAIL_G}G < 35G (need headroom for ~16G ckpt peak)."
  exit 1
fi
echo "=== Preflight OK: ${NGPU} GPUs, ${AVAIL_G}G free, SAVE_FREQ=${SAVE_FREQ}, keep=${MAX_ACTOR_CKPT_TO_KEEP} ==="
echo "=== Model: ${MODEL_PATH} ==="
echo "=== Output: ${output_dir} ==="
echo "=== Samples: train=${TRAIN_MAX_SAMPLES} val=${VAL_MAX_SAMPLES} epochs=${TOTAL_EPOCHS} ==="
echo "=== Batch: train_batch=${TRAIN_BATCH_SIZE} ppo_mini=${PPO_MINI_BATCH_SIZE} rollout.n=16 ==="

if (( TRAIN_BATCH_SIZE % 6 != 0 )); then
  echo "ERROR: TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE} not divisible by 6 (DP ranks)."
  exit 1
fi
if (( TRAIN_BATCH_SIZE % PPO_MINI_BATCH_SIZE != 0 )); then
  echo "ERROR: TRAIN_BATCH_SIZE must be divisible by PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE}."
  exit 1
fi

# ── 启动训练 ──
# 6 GPU, TP=1 → 6 DP ranks; batch sizes divisible by 6.
$PYTHON -m verl.trainer.partialKL_ppo \
 algorithm.adv_estimator=reinforce_plus_plus \
 algorithm.use_kl_in_reward=True \
 algorithm.kl_ctrl.kl_coef=0.0005 \
 data.train_files=$SIRL_DIR/trainset/gurobi_examples_OR_train_shot.parquet \
 data.val_files=$SIRL_DIR/trainset/gurobi_examples_OR_test_shot.parquet \
 data.train_max_samples=$TRAIN_MAX_SAMPLES \
 data.val_max_samples=$VAL_MAX_SAMPLES \
 data.train_batch_size=$TRAIN_BATCH_SIZE \
 data.max_prompt_length=$MAX_PROMPT_LENGTH \
 data.max_response_length=$MAX_RESPONSE_LENGTH \
 actor_rollout_ref.rollout.max_model_len=$MAX_MODEL_LEN \
 data.filter_overlong_prompts=False \
 data.prompt_key=prompt \
 data.truncation=left \
 data.custom_cls.path="$SIRL_DIR/dataset/gurobi_rl_dataset.py" \
 data.custom_cls.name=GurobiRLHFDataset \
 reward_model.reward_manager=naive \
 reward_model.num_workers=2 \
 custom_reward_function.path="$SIRL_DIR/reward_func/batch_score_gurobi.py" \
 actor_rollout_ref.model.path="${MODEL_PATH}" \
 actor_rollout_ref.model.use_remove_padding=True \
 actor_rollout_ref.model.enable_gradient_checkpointing=True \
 actor_rollout_ref.actor.fsdp_config.dtype=bf16 \
 actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
 actor_rollout_ref.ref.fsdp_config.dtype=bf16 \
 actor_rollout_ref.ref.fsdp_config.model_dtype=bf16 \
 actor_rollout_ref.actor.optim.lr=1e-6 \
 actor_rollout_ref.actor.clip_ratio_low=0.20 \
 actor_rollout_ref.actor.clip_ratio_high=0.30 \
 actor_rollout_ref.actor.entropy_coeff=0 \
 actor_rollout_ref.actor.calculate_entropy=false \
 actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
 actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$MAX_MODEL_LEN \
 actor_rollout_ref.actor.fsdp_config.param_offload=False \
 actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
 actor_rollout_ref.actor.fsdp_config.use_torch_compile=False \
 actor_rollout_ref.actor.checkpoint.save_contents='[model]' \
 actor_rollout_ref.actor.checkpoint.load_contents='[model]' \
 actor_rollout_ref.rollout.max_num_batched_tokens=$MAX_MODEL_LEN \
 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
 actor_rollout_ref.rollout.dtype=bfloat16 \
 actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
 actor_rollout_ref.rollout.enforce_eager=True \
 actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
 actor_rollout_ref.rollout.n=16 \
 actor_rollout_ref.rollout.agent.num_workers=2 \
 actor_rollout_ref.rollout.name=vllm \
 actor_rollout_ref.rollout.load_format=auto \
 custom_reward_function.name=compute_score \
 actor_rollout_ref.ref.fsdp_config.param_offload=True \
 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
 actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$MAX_MODEL_LEN \
 trainer.project_name="${project_name}" \
 trainer.experiment_name="${exp_name}" \
 trainer.logger=['console'] \
 trainer.n_gpus_per_node=6 \
 trainer.nnodes=1 \
 trainer.save_freq=$SAVE_FREQ \
 trainer.max_actor_ckpt_to_keep=$MAX_ACTOR_CKPT_TO_KEEP \
 trainer.test_freq=500 \
 trainer.val_before_train=False \
 trainer.default_local_dir=$output_dir \
 trainer.rollout_data_dir=$output_dir/rollouts \
 trainer.validation_data_dir=$output_dir/val_rollouts \
 trainer.total_epochs=$TOTAL_EPOCHS 2>&1 | tee "$output_dir/train.log"
