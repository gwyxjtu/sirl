# https://github.com/volcengine/verl/blob/main/verl/utils/reward_score/math_batch.py
"""
Stage-2 reward: python-only output with LP validation.

Reward composition:
    answer_reward  × 1.0   — objVal vs ground truth
    code_reward    × 1.0   — code execution success (Done)
    format_reward  × 0.5   — <python> tag presence (simplified, no think/model required)
    lp_reward      × 0.5   — LP file structure validation
"""

import os
import sys

# 确保 VeRL load_extern_object 时能找到同目录的 executor / content_utils
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import re
import uuid
import numpy as np
import requests
import json
from collections import Counter
from executor import PythonExecutor
from content_utils import extract_code_block, extract_obj, lp_structure_reward

# LP 文件输出目录（executor 子进程内可写）
LP_OUTPUT_DIR = "/tmp/sirl_lp_outputs"


def code_reward(code_excu_result):
    """代码是否执行成功"""
    return code_excu_result == 'Done'


def answer_reward(solver_result, ans, code_excu_result, cri=1e-6):
    """求解目标值 vs ground_truth（相对误差 < 1e-6）"""
    if ans is None:
        abs_err = 1
    else:
        abs_err = np.abs(ans) if ans else 1
    if ans is None and solver_result is None and code_excu_result == 'Done':
        abs_err = 0
    if ans and solver_result:
        abs_err = np.abs(ans - solver_result) / (np.abs(ans) + 1)
    if ans is None:
        ans = 1
    return abs_err < cri


def format_reward(processed_str: str) -> float:
    """
    检查 <python> 标签是否完整出现。
    不要求 thinking / <model> 标签。
    满分 1.0。
    """
    has_open = '<python>' in processed_str
    has_close = '</python>' in processed_str

    if has_open and has_close:
        return 1.0
    elif has_open or has_close:
        return 0.5   # 只有一半
    else:
        return 0.0


# ── 核心评分函数 ──

def _compute_scores_batch(data_sources, solution_strs, ground_truths, extra_infos):
    """
    批量计算每条样本的 reward。

    新版流程:
        1. extract_code_block + insert_lp_generation → 执行代码
        2. 代码执行中产生 .lp 文件 + print objVal
        3. answer_reward: 从 stdout 提取 objVal
        4. code_reward: 代码是否跑通
        5. format_reward: <python> 标签是否完整
        6. lp_reward: LP 文件结构验证
    """
    ans_score = 1.0
    code_score = 1.0
    format_score = 0.5
    lp_score_weight = 0.5

    os.makedirs(LP_OUTPUT_DIR, exist_ok=True)

    # 为每条样本分配 LP 文件名
    lp_paths = []
    codes = []
    for _ in solution_strs:
        lp_name = f"gurobi_model_{uuid.uuid4().hex[:8]}.lp"
        lp_paths.append(os.path.join(LP_OUTPUT_DIR, lp_name))

    # 提取并注入 LP 生成的代码
    for sol_str, lp_path in zip(solution_strs, lp_paths):
        code = extract_code_block(sol_str, 'gurobi', lp_output_path=lp_path)
        codes.append(code)

    # 执行代码
    executor = PythonExecutor()
    response = executor.batch_apply(codes)

    # 解析结果
    obj_result = [response[0][i] for i in range(len(solution_strs))]
    code_excu_result = [response[2][i] for i in range(len(solution_strs))]

    # 各项打分
    format_scores = [format_reward(solution_strs[i]) for i in range(len(solution_strs))]
    code_scores = [code_reward(code_excu_result[i]) for i in range(len(code_excu_result))]
    ans = [answer_reward(obj_result[i], ground_truths[i], code_excu_result[i]) for i in range(len(ground_truths))]
    lp_scores = [lp_structure_reward(lp_paths[i]) for i in range(len(lp_paths))]

    # 清理 LP 文件
    for lp_path in lp_paths:
        try:
            os.remove(lp_path)
        except OSError:
            pass

    # 合成总分
    scores = []
    for i in range(len(solution_strs)):
        total = (
            ans[i] * ans_score +
            format_scores[i] * format_score +
            code_scores[i] * code_score +
            lp_scores[i] * lp_score_weight
        )
        scores.append(total)

    return scores


def compute_score(
    data_sources=None,
    solution_strs=None,
    ground_truths=None,
    extra_infos=None,
    *,
    data_source=None,
    solution_str=None,
    ground_truth=None,
    extra_info=None,
    **kwargs,
):
    """VeRL 0.8 naive: single-item kwargs; legacy batch: list arguments."""
    if solution_str is not None:
        rewards = _compute_scores_batch(
            [data_source],
            [solution_str],
            [ground_truth],
            [extra_info or {}],
        )
        return rewards[0]
    return _compute_scores_batch(data_sources, solution_strs, ground_truths, extra_infos)