#!/bin/bash
# Quick VeRL + SIRL environment check
PY=${PY:-/home/guo/anaconda3/bin/python}

echo "=== Python ==="
$PY --version

echo "=== Core packages ==="
$PY -c "
import verl, torch, vllm, ray
print('verl:', verl.__file__)
print('torch:', torch.__version__)
print('vllm:', vllm.__version__)
print('ray:', ray.__version__)
print('cuda:', torch.cuda.is_available(), 'gpus:', torch.cuda.device_count())
from torch.distributed.tensor import DTensor
print('DTensor: OK')
"

echo "=== SIRL modules ==="
$PY -c "
import importlib
for m in ['verl.trainer.main_ppo','verl.trainer.partialKL_ppo','verl.trainer.kl_ppo.ray_trainer']:
    importlib.import_module(m)
    print(m, 'OK')
from verl.experimental.reward_loop.reward_manager import get_reward_manager_cls
print('NaiveRewardManager:', get_reward_manager_cls('naive'))
import sys; sys.path.insert(0,'/home/guo/LLM/Verl/verl')
from batch_score_gurobi import compute_score
print('batch_score_gurobi (naive+batch API): OK')
"

echo "=== CLI ==="
$PY -m verl.trainer.partialKL_ppo --help >/dev/null && echo "partialKL_ppo --help: OK"

echo "=== Done ==="
