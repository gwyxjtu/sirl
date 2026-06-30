export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
# vllm 0.12 + torch 2.9: do not set VLLM_ATTENTION_BACKEND=XFORMERS (xformers incompatible)

WORKING_DIR="/root/llm/sirl"
SIRL_DIR="/root/llm/sirl/SIRLTrain"
CKPTS_DIR="/root/autodl-tmp/checkpoints"
PYTHON="/root/miniconda3/bin/python3"
project_name='SIRL'
exp_name='partialKL'
# Checkpoint 策略:
# - save_freq 必须 >0 才会存盘；设为很大值表示只在最后一步 (is_last_step) 存一次。
# - 默认只存 model 权重，不存 optimizer/extra（VeRL 默认含 optimizer，体积约为 model 的 2–3 倍）。
#   optimizer 仅断点续训需要；若要完整 checkpoint 可覆盖:
#   actor_rollout_ref.actor.checkpoint.save_contents='[model,optimizer,extra]'
SAVE_FREQ=${SAVE_FREQ:-999999}

output_dir=$CKPTS_DIR/$project_name/$exp_name
$PYTHON -m verl.trainer.partialKL_ppo \
 algorithm.adv_estimator=reinforce_plus_plus \
 algorithm.use_kl_in_reward=True \
 algorithm.kl_ctrl.kl_coef=0.0005 \
 data.train_files=$SIRL_DIR/trainset/gurobi_examples_OR_train.parquet \
 data.val_files=$SIRL_DIR/trainset/gurobi_examples_OR_test.parquet \
 data.train_batch_size=32 \
 data.max_prompt_length=2048 \
 data.max_response_length=8192 \
 data.filter_overlong_prompts=True \
 data.prompt_key=prompt \
 data.truncation=left \
 data.custom_cls.path="$SIRL_DIR/dataset/gurobi_rl_dataset.py" \
 data.custom_cls.name=GurobiRLHFDataset \
 reward_model.reward_manager=naive \
 custom_reward_function.path="$SIRL_DIR/reward_func/batch_score_gurobi.py" \
 actor_rollout_ref.model.path='/root/autodl-tmp/models/Qwen3-4B-Instruct-2507' \
 actor_rollout_ref.model.use_remove_padding=True \
 actor_rollout_ref.model.enable_gradient_checkpointing=True \
 actor_rollout_ref.actor.optim.lr=1e-6 \
 actor_rollout_ref.actor.clip_ratio_low=0.20 \
 actor_rollout_ref.actor.clip_ratio_high=0.30 \
 actor_rollout_ref.actor.ppo_mini_batch_size=32 \
 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
 actor_rollout_ref.actor.fsdp_config.param_offload=True \
 actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
 actor_rollout_ref.actor.checkpoint.save_contents='[model]' \
 actor_rollout_ref.actor.checkpoint.load_contents='[model]' \
 actor_rollout_ref.rollout.max_num_batched_tokens=12232 \
 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
 actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
 actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
 actor_rollout_ref.rollout.n=16 \
 actor_rollout_ref.rollout.name=vllm \
 actor_rollout_ref.rollout.load_format=auto \
 custom_reward_function.name=compute_score \
 actor_rollout_ref.ref.fsdp_config.param_offload=True \
 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
 critic.optim.lr=1e-5 \
 critic.ppo_micro_batch_size_per_gpu=4 \
 trainer.project_name="${project_name}" \
 trainer.experiment_name="${exp_name}" \
 trainer.logger=['console','wandb'] \
 trainer.n_gpus_per_node=8 \
 trainer.nnodes=1 \
 trainer.save_freq=$SAVE_FREQ \
 trainer.test_freq=1000 \
 trainer.default_local_dir=$output_dir \
 trainer.total_epochs=2 2>&1 | tee withKL.log
