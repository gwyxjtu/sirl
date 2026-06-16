# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Partial-KL PPO entry point for VeRL 0.8+.
Uses official main_ppo TaskRunner/workers and only patches apply_kl_penalty via kl_ppo.ray_trainer.
"""

import os
import socket

from verl.utils.distributed import configure_cross_numa_nccl_env, _CROSS_NUMA_NCCL_ENV

configure_cross_numa_nccl_env()

import hydra
import ray
from omegaconf import OmegaConf

from verl.experimental.reward_loop import migrate_legacy_reward_impl
from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
from verl.trainer.distillation import is_distillation_enabled
from verl.trainer.kl_ppo.ray_trainer import RayPPOTrainer
from verl.trainer.ppo.utils import need_critic, need_reference_policy
from verl.utils.config import validate_config
from verl.utils.device import auto_set_device, is_cuda_available


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    auto_set_device(config)
    config = migrate_legacy_reward_impl(config)
    run_ppo(config)


def run_ppo(config, task_runner_class=None) -> None:
    if not ray.is_initialized():
        default_runtime_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})

        if config.transfer_queue.enable:
            runtime_env_vars = runtime_env_kwargs.get("env_vars", {})
            runtime_env_vars["TRANSFER_QUEUE_ENABLE"] = "1"
            runtime_env_kwargs["env_vars"] = runtime_env_vars

        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        env_vars = dict(OmegaConf.to_container(runtime_env.get("env_vars", {}), resolve=True) or {})
        env_vars.update(_CROSS_NUMA_NCCL_ENV)
        runtime_env["env_vars"] = {k: str(v) for k, v in env_vars.items()}
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        print(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    if task_runner_class is None:
        task_runner_class = ray.remote(num_cpus=1)(TaskRunner)

    if (
        is_cuda_available
        and config.global_profiler.tool == "nsys"
        and config.global_profiler.get("steps") is not None
        and len(config.global_profiler.get("steps", [])) > 0
    ):
        from verl.utils.import_utils import is_nvtx_available

        assert is_nvtx_available(), "nvtx is not available in CUDA platform. Please 'pip3 install nvtx'"
        nsight_options = OmegaConf.to_container(
            config.global_profiler.global_tool_config.nsys.controller_nsight_options
        )
        runner = task_runner_class.options(runtime_env={"nsight": nsight_options}).remote()
    else:
        runner = task_runner_class.remote()
    ray.get(runner.run.remote(config))

    timeline_json_file = config.ray_kwargs.get("timeline_json_file", None)
    if timeline_json_file:
        ray.timeline(filename=timeline_json_file)


class TaskRunner:
    def __init__(self):
        self.role_worker_mapping = {}
        self.mapping = {}

    def add_actor_rollout_worker(self, config):
        from verl.single_controller.ray import RayWorkerGroup
        from verl.trainer.ppo.ray_trainer import Role
        from verl.workers.engine_workers import ActorRolloutRefWorker

        actor_rollout_cls = ActorRolloutRefWorker
        ray_worker_group_cls = RayWorkerGroup

        lora_rank = config.actor_rollout_ref.model.get("lora", {}).get("rank", 0)
        if lora_rank <= 0:
            lora_rank = config.actor_rollout_ref.model.get("lora_rank", 0)
        ref_in_actor = lora_rank > 0 or config.actor_rollout_ref.model.get("lora_adapter_path") is not None
        if need_reference_policy(config) and not ref_in_actor:
            role = Role.ActorRolloutRef
        else:
            role = Role.ActorRollout
        self.role_worker_mapping[role] = ray.remote(actor_rollout_cls)
        self.mapping[role] = "global_pool"
        return actor_rollout_cls, ray_worker_group_cls

    def add_critic_worker(self, config):
        from verl.trainer.ppo.ray_trainer import Role
        from verl.workers.engine_workers import TrainingWorker

        self.role_worker_mapping[Role.Critic] = ray.remote(TrainingWorker)
        self.mapping[Role.Critic] = "global_pool"

    def init_resource_pool_mgr(self, config):
        from verl.trainer.ppo.ray_trainer import ResourcePoolManager

        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }

        if config.reward.reward_model.enable_resource_pool:
            reward_pool = [config.reward.reward_model.n_gpus_per_node] * config.reward.reward_model.nnodes
            resource_pool_spec["reward_pool"] = reward_pool
        else:
            config.reward.reward_model.nnodes = config.trainer.nnodes
            config.reward.reward_model.n_gpus_per_node = config.trainer.n_gpus_per_node

        distillation_config = config.get("distillation")
        if is_distillation_enabled(distillation_config):
            teacher_pool = [distillation_config.n_gpus_per_node] * distillation_config.nnodes
            resource_pool_spec["teacher_pool"] = teacher_pool

        return ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=self.mapping)

    def add_reward_model_resource_pool(self, config):
        from verl.trainer.ppo.ray_trainer import Role

        if config.reward.reward_model.enable:
            if config.reward.reward_model.enable_resource_pool:
                self.mapping[Role.RewardModel] = "reward_pool"
            else:
                self.mapping[Role.RewardModel] = "global_pool"

    def add_teacher_model_resource_pool(self, config):
        from verl.trainer.ppo.ray_trainer import Role

        if is_distillation_enabled(config.get("distillation")):
            self.mapping[Role.TeacherModel] = "teacher_pool"

    def add_ref_policy_worker(self, config, ref_policy_cls):
        return

    def run(self, config):
        from pprint import pprint

        from omegaconf import OmegaConf

        from verl.utils.fs import copy_to_local

        print(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        actor_rollout_cls, ray_worker_group_cls = self.add_actor_rollout_worker(config)
        self.add_critic_worker(config)
        self.add_reward_model_resource_pool(config)
        self.add_teacher_model_resource_pool(config)
        self.add_ref_policy_worker(config, actor_rollout_cls)

        validate_config(
            config=config,
            use_reference_policy=need_reference_policy(config),
            use_critic=need_critic(config),
        )

        local_path = copy_to_local(
            config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False)
        )

        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        resource_pool_manager = self.init_resource_pool_mgr(config)

        from verl.utils.dataset.rl_dataset import collate_fn

        train_dataset = create_rl_dataset(
            config.data.train_files,
            config.data,
            tokenizer,
            processor,
            is_train=True,
            max_samples=config.data.get("train_max_samples", -1),
        )
        val_dataset = create_rl_dataset(
            config.data.val_files,
            config.data,
            tokenizer,
            processor,
            is_train=False,
            max_samples=config.data.get("val_max_samples", -1),
        )
        train_sampler = create_rl_sampler(config.data, train_dataset)

        trainer = RayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=self.role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
        )
        trainer.init_workers()
        trainer.fit()


def create_rl_dataset(data_paths, data_config, tokenizer, processor, is_train=True, max_samples: int = -1):
    from verl.utils.dataset.rl_dataset import get_dataset_class

    dataset_cls = get_dataset_class(data_config)
    return dataset_cls(
        data_files=data_paths,
        tokenizer=tokenizer,
        processor=processor,
        config=data_config,
        max_samples=max_samples,
    )


def create_rl_sampler(data_config, dataset):
    import torch
    from torch.utils.data import SequentialSampler
    from torchdata.stateful_dataloader.sampler import RandomSampler

    if data_config.shuffle:
        train_dataloader_generator = torch.Generator()
        seed = data_config.get("seed")
        if seed is not None:
            train_dataloader_generator.manual_seed(seed)
        sampler = RandomSampler(data_source=dataset, generator=train_dataloader_generator)
    else:
        sampler = SequentialSampler(data_source=dataset)
    return sampler


if __name__ == "__main__":
    main()
