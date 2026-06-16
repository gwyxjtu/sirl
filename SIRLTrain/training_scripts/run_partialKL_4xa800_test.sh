#!/bin/bash
# Partial-KL PPO training on 4× A800 (80GB each)
# Smoke test script — 10 train / 5 val samples, 1 epoch
#
# Usage:
#   bash training_scripts/run_partialKL_4xa800_test.sh
#
# Full run:
#   FRESH_START=1 TRAIN_MAX_SAMPLES=-1 VAL_MAX_SAMPLES=-1 TOTAL_EPOCHS=2 \
#     bash training_scripts/run_partialKL_4xa800_test.sh

export CUDA_VISIBLE_DEVICES="0,1,2,3"

# ── Paths (修改为 A800 服务器上的实际路径) ──
WORKING_DIR="/home/guo/LLM/Verl/verl"
SIRL_DIR="/home/guo/LLM/SIRL/SIRLTrain"
CKPTS_DIR="/home/guo/LLM/SIRL/checkpoints"
PYTHON="/home/guo/anaconda3/bin/python"

project_name='SIRL'
exp_name='partialKL_4xa800_qwen3_4b_test'
MODEL_PATH=${MODEL_PATH:-/home/guo/LLM/SIRL/Qwen3-4B-Instruct-2507}

# ── 训练参数 ──
TRAIN_MAX_SAMPLES=${TRAIN_MAX_SAMPLES:-10}
VAL_MAX_SAMPLES=${VAL_MAX_SAMPLES:-5}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-2048}
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
  sleep 3
  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "=== GPU memory after cleanup ==="
    nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
  fi
  echo "=== Cleanup done ==="
fi

# ── 启动训练 ──
$PYTHON -m verl.trainer.partialKL_ppo \
 algorithm.adv_estimator=reinforce_plus_plus \
 algorithm.use_kl_in_reward=True \
 algorithm.kl_ctrl.kl_coef=0.0005 \
 data.train_files=$WORKING_DIR/trainset/gurobi_examples_OR_train_fixed.parquet \
 data.val_files=$WORKING_DIR/trainset/gurobi_examples_OR_test_fixed.parquet \
 data.train_max_samples=$TRAIN_MAX_SAMPLES \
 data.val_max_samples=$VAL_MAX_SAMPLES \
 data.train_batch_size=8 \
 data.max_prompt_length=$MAX_PROMPT_LENGTH \
 data.max_response_length=$MAX_RESPONSE_LENGTH \
 actor_rollout_ref.rollout.max_model_len=$MAX_MODEL_LEN \
 data.filter_overlong_prompts=True \
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
 actor_rollout_ref.actor.ppo_mini_batch_size=8 \
 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
 actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$MAX_MODEL_LEN \
 actor_rollout_ref.actor.fsdp_config.param_offload=False \
 actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
 actor_rollout_ref.actor.fsdp_config.use_torch_compile=False \
 actor_rollout_ref.rollout.max_num_batched_tokens=$MAX_MODEL_LEN \
 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
 actor_rollout_ref.rollout.dtype=bf16 \
 actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
 actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
 actor_rollout_ref.rollout.n=2 \
 actor_rollout_ref.rollout.agent.num_workers=2 \
 actor_rollout_ref.rollout.name=vllm \
 actor_rollout_ref.rollout.load_format=auto \
 custom_reward_function.name=compute_score \
 actor_rollout_ref.ref.fsdp_config.param_offload=True \
 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
 actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$MAX_MODEL_LEN \
 trainer.project_name="${project_name}" \
 trainer.experiment_name="${exp_name}" \
 trainer.logger=['console'] \
 trainer.n_gpus_per_node=4 \
 trainer.nnodes=1 \
 trainer.save_freq=25 \
 trainer.test_freq=500 \
 trainer.val_before_train=False \
 trainer.default_local_dir=$output_dir \
 trainer.rollout_data_dir=$output_dir/rollouts \
 trainer.validation_data_dir=$output_dir/val_rollouts \
 trainer.total_epochs=$TOTAL_EPOCHS 2>&1 | tee "$output_dir/train.log"
