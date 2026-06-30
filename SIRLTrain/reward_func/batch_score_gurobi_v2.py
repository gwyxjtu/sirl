"""
Stage-2 reward V2: Enhanced reward with code-correctness diagnostics.

Key improvements over V1:
  1. code_correctness_reward — static analysis of raw LLM output BEFORE execution.
     Penalises known API mistakes (objVal, var.x, lb=-GRB.INFINITY) and
     rewards correct patterns (vtype=GRB.BINARY, model.status check).
     This gives gradient even when code fails to execute.

  2. Enhanced lp_structure_reward — checks not just LP structure existence,
     but also lower-bound sanity (flow variables should be >= 0).

  3. Better weight balance — code correctness provides a -0.3..+0.3 signal
     on top of the existing 0-3 scale, creating finer-grained differentiation
     for the otherwise identical 0.5-scored failures.

Reward composition (unchanged from V1 except added correctness):
    answer_reward         × 1.0   — objVal vs ground truth (binary)
    code_reward           × 1.0   — code execution success (binary)
    format_reward         × 0.5   — <python> tag presence
    lp_structure_reward   × 0.5   — LP file structure validation
    code_correctness      × 0.4   — static code quality analysis (NEW, range -0.3..+0.3)

Max total: 1.0 + 1.0 + 0.5 + 0.5 + 0.4*0.75 ≈ 3.3
"""

import os
import sys

# Ensure VeRL load_extern_object can find sibling modules
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import re
import uuid
import numpy as np
from executor import PythonExecutor
from content_utils import (
    extract_code_block,
    lp_structure_reward as _base_lp_reward,
)

LP_OUTPUT_DIR = "/tmp/sirl_lp_outputs"


# ── Core reward functions (unchanged from V1) ──

def code_reward(code_excu_result):
    """Whether code executed successfully."""
    return code_excu_result == 'Done'


def answer_reward(solver_result, ans, code_excu_result, cri=1e-6):
    """objVal vs ground truth (relative error < 1e-6)."""
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
    """Check <python>...</python> tag completeness."""
    has_open = '<python>' in processed_str
    has_close = '</python>' in processed_str
    if has_open and has_close:
        return 1.0
    elif has_open or has_close:
        return 0.5
    else:
        return 0.0


# ── NEW: Static code correctness diagnostics ──

# Patterns that indicate an API mistake (penalised)
_API_MISTAKE_PATTERNS = [
    (r'\bmodel\.objVal\b', 0.15, "model.objVal → should be model.ObjVal"),
    (r'(?<!\w)\.x\b',      0.10, ".x → should be .X for Gurobi variable attributes"),
    (r'lb=-GRB\.INFINITY', 0.15, "flow variables should be lb=0, not lb=-GRB.INFINITY"),
]

# Patterns that indicate good practice (rewarded)
_API_CORRECT_PATTERNS = [
    (r'vtype\s*=\s*GRB\.(BINARY|INTEGER)', 0.10, "explicit variable type"),
    (r'model\.status\s*==\s*GRB\.OPTIMAL', 0.05, "checks optimal status before printing"),
    (r'model\.ObjVal',     0.05, "correct ObjVal (capital O, V)"),
    (r'\.X\b',             0.05, "correct .X (capital X) for variable value"),
]


def code_correctness_reward(raw_output: str) -> float:
    """
    Static analysis of raw LLM output (BEFORE execution).

    Penalises known Gurobi API mistakes and rewards correct patterns.
    Provides a gradient signal even when code fails to execute,
    differentiating between "almost right" and "completely wrong".

    Returns: float in [-0.3, +0.3]
    """
    # Extract just the code portion if wrapped in <python> tags
    code_match = re.search(r'<python>(.*?)</python>', raw_output, re.DOTALL)
    code_str = code_match.group(1) if code_match else raw_output

    score = 0.0

    # Penalise API mistakes
    for pattern, penalty, _msg in _API_MISTAKE_PATTERNS:
        if re.search(pattern, code_str):
            score -= penalty

    # Reward correct patterns
    for pattern, bonus, _msg in _API_CORRECT_PATTERNS:
        if re.search(pattern, code_str):
            score += bonus

    # Clamp to [-0.3, 0.3]
    return max(-0.3, min(0.3, score))


# ── Enhanced LP structure reward ──

def enhanced_lp_reward(lp_path: str, raw_output: str) -> float:
    """
    Enhanced LP reward: base structure check + lower-bound sanity.

    The base reward (from content_utils) checks: objective, constraints,
    variables, integer/binary types, and quadratic terms.

    This wrapper adds: if the problem mentions "flow" or "capacity" but the
    LP bounds section is empty/missing, apply a small penalty (-0.10).
    """
    base_score = _base_lp_reward(lp_path)
    if base_score == 0.0 or not os.path.exists(lp_path):
        return 0.0

    # Quick check: does the LP file have a non-empty Bounds section?
    with open(lp_path) as f:
        lp_text = f.read()

    # If problem mentions flow/capacity but Bounds section is empty
    code_match = re.search(r'<python>(.*?)</python>', raw_output, re.DOTALL)
    code_str = code_match.group(1) if code_match else raw_output
    has_flow_keyword = any(kw in code_str.lower() for kw in ['flow', 'capacity', 'demand'])
    has_bounds_section = 'Bounds' in lp_text

    # Count bound lines after "Bounds" header
    bounds_lines = 0
    in_bounds = False
    for line in lp_text.split('\n'):
        stripped = line.strip()
        if stripped == 'Bounds':
            in_bounds = True
            continue
        if in_bounds and (stripped == 'Binaries' or stripped == 'Generals' or
                          stripped == 'End' or stripped == 'Subject To'):
            break
        if in_bounds and stripped:
            bounds_lines += 1

    if has_flow_keyword and bounds_lines == 0:
        base_score -= 0.10  # penalty for missing bounds on flow-like problems

    return max(0.0, base_score)


# ── Batch scoring ──

def _compute_scores_batch(data_sources, solution_strs, ground_truths, extra_infos):
    """Batch compute rewards with enhanced V2 scoring."""

    # Weights
    ans_score       = 1.0
    code_score      = 1.0
    format_score    = 0.5
    lp_score_weight = 0.5
    correctness_weight = 0.4   # NEW: code-correctness weight

    os.makedirs(LP_OUTPUT_DIR, exist_ok=True)

    # Assign LP file names
    lp_paths = []
    codes = []
    for _ in solution_strs:
        lp_name = f"gurobi_model_{uuid.uuid4().hex[:8]}.lp"
        lp_paths.append(os.path.join(LP_OUTPUT_DIR, lp_name))

    # Extract code + inject LP generation
    for sol_str, lp_path in zip(solution_strs, lp_paths):
        code = extract_code_block(sol_str, 'gurobi', lp_output_path=lp_path)
        codes.append(code)

    # ── Static analysis (BEFORE execution) ──
    correctness_scores = [code_correctness_reward(solution_strs[i])
                          for i in range(len(solution_strs))]

    # Execute code
    executor = PythonExecutor()
    response = executor.batch_apply(codes)

    # Parse results
    obj_result = [response[0][i] for i in range(len(solution_strs))]
    code_excu_result = [response[2][i] for i in range(len(solution_strs))]

    # Score components
    format_scores = [format_reward(solution_strs[i]) for i in range(len(solution_strs))]
    code_scores = [code_reward(code_excu_result[i]) for i in range(len(code_excu_result))]
    ans = [answer_reward(obj_result[i], ground_truths[i], code_excu_result[i])
           for i in range(len(ground_truths))]
    lp_scores = [enhanced_lp_reward(lp_paths[i], solution_strs[i])
                 for i in range(len(lp_paths))]

    # Clean LP files
    for lp_path in lp_paths:
        try:
            os.remove(lp_path)
        except OSError:
            pass

    # Compose total
    scores = []
    for i in range(len(solution_strs)):
        total = (
            ans[i] * ans_score +
            format_scores[i] * format_score +
            code_scores[i] * code_score +
            lp_scores[i] * lp_score_weight +
            correctness_scores[i] * correctness_weight   # NEW
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
