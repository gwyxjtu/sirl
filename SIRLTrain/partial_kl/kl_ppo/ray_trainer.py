"""VeRL 0.8 compatible Partial-KL trainer: patches apply_kl_penalty on the official RayPPOTrainer."""

from verl.trainer.ppo import ray_trainer as _ppo_ray_trainer
from verl.trainer.ppo.ray_trainer import *  # noqa: F401,F403
from verl.trainer.ppo.ray_trainer import RayPPOTrainer as _BaseRayPPOTrainer

from verl.trainer.kl_ppo.partial_kl import apply_partial_kl_penalty, init_partial_kl_context

# Patch module-level function used inside RayPPOTrainer.fit()
_ppo_ray_trainer.apply_kl_penalty = apply_partial_kl_penalty


class RayPPOTrainer(_BaseRayPPOTrainer):
    """Official VeRL 0.8 trainer with Partial-KL context initialized from tokenizer tags."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        init_partial_kl_context(self.tokenizer)
