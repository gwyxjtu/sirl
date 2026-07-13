import re
import subprocess
import textwrap
import os
import uuid

# 全局静音 Gurobi 默认环境的日志输出（"Read LP format model" / solve log 等）。
# RewardLoopWorker 的 stdout 走 Ray 管道，Gurobi 巨量 C 层输出会写满管道
# 导致 write() 永久阻塞（fork 子进程继承该设置，同样静音）。
try:
    import gurobipy as _gp
    _gp.setParam('OutputFlag', 0)
except Exception:
    pass

# ---------------------------------------
# 提取代码块的函数
# ---------------------------------------

def insert_print(code: str, solver_name: str) -> str:
    """插入打印最优目标值和解的代码（保留兼容，不再使用）"""
    model_pattern = r'^(\s*)(\w+)\.(optimize|solve)\(\)'
    model_match = re.search(model_pattern, code, re.M)
    if model_match:
        indent = model_match.group(1)
        model_name = model_match.group(2)
        if solver_name == "gurobi":
            pattern = r'^(\s*)(' + model_name + r'\.optimize\(\))'
            status_check = (
                f"{indent}if {model_name}.status == GRB.OPTIMAL:\n"
                f"{indent}    print(f'Just print the best obj: {{{model_name}.ObjVal}}')\n"
                f"{indent}    print('Just print the best sol:[', end = '')\n"
                f"{indent}    for var in {model_name}.getVars():\n"
                f"{indent}        print(f'{{var.X}}', end = ',')\n"
                f"{indent}    print(']')\n"
                f"{indent}else:\n"
                f"{indent}    print('No optimal solution found, status:', {model_name}.status)"
            )
        elif solver_name == "copt":
            pattern = r'^(\s*)(' + model_name + r'\.solve\(\))'
            status_check = (
                f"{indent}if {model_name}.status == COPT.OPTIMAL:\n"
                f"{indent}    print(f'Just print the best obj: {{{model_name}.ObjVal}}')\n"
                f"{indent}    print('Just print the best sol:[', end = '')\n"
                f"{indent}    for var in {model_name}.getVars():\n"
                f"{indent}        print(f'{{var.X}}', end = ',')\n"
                f"{indent}    print(']')\n"
                f"{indent}else:\n"
                f"{indent}    print('No optimal solution found, status:', {model_name}.status)"
            )
        code = re.sub(pattern, rf'\1\2\n{status_check}', code, flags=re.M)
    return code

def insert_lp_generation(code: str, output_name: str, skip_optimize: bool = False) -> str:
    """在 optimize() 前注入 model.write('xxx.lp')，让 Gurobi 写出 LP 文件；
    同时在 optimize 后打印最优目标值。
    
    当 skip_optimize=True 时，只写 LP 文件不运行求解（用于离线预处理）。"""
    model_pattern = r'^(\s*)(\w+)\.(optimize|solve)\(\)'
    try:
        code = str(code)
    except:
        return None
    model_match = re.search(model_pattern, code, re.M)
    if model_match:
        indent = model_match.group(1)
        model_name = model_match.group(2)
        pattern = r'^(\s*)(' + model_name + r'\.optimize\(\))'
        if skip_optimize:
            # 只写 LP 不求解（离线预处理专用）：替换 optimize() 为 write()
            replacement = f"{indent}{model_name}.write('{output_name}')"
            code = re.sub(pattern, replacement, code, flags=re.M)
        else:
            status_check = (
                f"{indent}{model_name}.write('{output_name}')\n"
                f"{indent}{model_name}.optimize()\n"
                f"{indent}if {model_name}.status == GRB.OPTIMAL:\n"
                f"{indent}    print(f'Just print the best obj: {{{model_name}.ObjVal}}')\n"
                f"{indent}else:\n"
                f"{indent}    print('No optimal solution found, status:', {model_name}.status)"
            )
            # setParam 放在原始 optimize() 之前，两次求解的日志都被静音
            quiet_line = f"{indent}{model_name}.setParam('OutputFlag', 0)"
            code = re.sub(pattern, rf'{quiet_line}\n\1\2\n{status_check}', code, flags=re.M)
    return code

def extract_code_block(llm_output: str, solver_name, lp_output_path: str = None) -> str:
    """
    从 LLM 输出中提取代码块，并为 Gurobi 注入 LP 文件生成 + print obj。
    若未匹配到则返回 None。

    Args:
        llm_output: 模型原始输出文本
        solver_name: 'gurobi' 或 'copt'
        lp_output_path: LP 文件输出路径（如 /tmp/gurobi_xxx.lp）
    """
    code = None
    pattern = r'<python>(.*?)</python>'
    match = re.search(pattern, llm_output, re.DOTALL)
    if match:
        code = match.group(1).strip()
        if '```' in code:
            pattern = r'```python(.*?)```'
            match = re.search(pattern, code, re.DOTALL)
            if match:
                code = match.group(1).strip()
    else:
        # 没有 <python> 标签，尝试裸 ```python```
        pattern = r'```python(.*?)```'
        match = re.search(pattern, llm_output, re.DOTALL)
        if match:
            code = match.group(1).strip()

    if code is None:
        return None

    # 注入 LP 生成和 print obj
    if solver_name == "gurobi" and lp_output_path:
        code = insert_lp_generation(code, lp_output_path)
    else:
        code = insert_print(code, solver_name)

    return code

def extract_block(llm_output, part_name):
    """提取指定标签内的内容块"""
    pattern = rf'<{part_name}>(.*?)</{part_name}>'
    block = None
    match = re.search(pattern, llm_output, re.DOTALL)
    if match:
        block = match.group(1).strip()
    return block

def extract_obj(str_log):
    """从执行输出中提取最优目标值"""
    if 'Just print the best obj:' in str_log:
        item = next(i for i in str_log.split('\n') if 'Just print the best obj:' in i)
        result = re.findall(r'-?\d+\.?\d*', item)
        return float(result[0]) if result else None
    return None

def extract_sol(str_log):
    """从执行输出中提取最优解"""
    if 'Just print the best sol:' in str_log:
        sol_match = re.search(r'Just print the best sol:\s*\[([-\d.,\s]*)\]', str_log)
        best_sol = [float(x) for x in sol_match.group(1).split(',') if x.strip()] if sol_match else None
        if best_sol:
            best_sol.sort()
            return best_sol
        else:
            print(str_log)
            return [None]
    return [None]

def parse_lp_structure(lp_path: str) -> dict:
    """
    解析 Gurobi LP 文件的基本结构。

    Returns:
        dict with keys:
            objective_type: 'Minimize' / 'Maximize' / ''
            num_variables: int (从 Bounds 段估算)
            num_constraints: int (从 Subject To 段估算)
            num_binary: int (Binaries 段变量数)
            num_integer: int (Generals 段变量数，不含 Binary)
            has_quadratic: bool (目标中有 ^2 或 * 等二次项)
            has_integers: bool (General 或 Binary 段存在)
            raw_lines: list of lines
    """
    result = {
        "objective_type": "",
        "num_variables": 0,
        "num_constraints": 0,
        "num_binary": 0,
        "num_integer": 0,
        "has_quadratic": False,
        "has_integers": False,
        "raw_lines": [],
    }

    if not os.path.exists(lp_path):
        return result

    # errors='replace'：LP 文件可能含非 UTF-8 字节（模型生成的变量名等），
    # 严格解码曾在 2026-07-13 step14 直接炸掉整个训练
    with open(lp_path, encoding='utf-8', errors='replace') as f:
        lines = f.readlines()
    result["raw_lines"] = lines

    in_section = None
    constraint_count = 0
    var_count = 0
    binary_count = 0
    integer_count = 0

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("Minimize") or stripped.startswith("Maximize"):
            result["objective_type"] = stripped.split()[0]
            in_section = "objective"
            continue
        elif stripped == "Subject To":
            in_section = "constraints"
            continue
        elif stripped == "Bounds":
            in_section = "bounds"
            continue
        elif stripped == "Binaries":
            result["has_integers"] = True
            in_section = "binaries"
            continue
        elif stripped == "Generals":
            result["has_integers"] = True
            in_section = "generals"
            continue
        elif stripped == "End":
            break

        if in_section == "objective":
            if "^2" in stripped or " * " in stripped:
                result["has_quadratic"] = True

        elif in_section == "constraints":
            if stripped and not stripped.startswith("\\") and ":" in stripped:
                constraint_count += 1

        elif in_section == "bounds":
            if stripped:
                var_count += 1

        elif in_section == "binaries":
            if stripped:
                # 每行可能有多个空格分隔的变量名
                binary_count += len(stripped.split())

        elif in_section == "generals":
            if stripped:
                integer_count += len(stripped.split())

    # 总变量数 = Bounds 段 + Binaries/Generals 段（后者可能不重复出现在 Bounds）
    result["num_variables"] = var_count + binary_count + integer_count
    result["num_constraints"] = constraint_count
    result["num_binary"] = binary_count
    result["num_integer"] = integer_count

    return result


def extract_lp_stats_from_code(reference_code: str, lp_path: str | None = None, timeout: int = 30) -> dict | None:
    """
    执行 reference 代码，写出 LP 文件，解析其结构统计。

    用于离线预处理：对每条训练样本，用 reference/output 代码跑一遍，
    提取 GT 的 LP 结构统计，存入 extra_info.lp_ref。

    Args:
        reference_code: 参考代码（字符串）
        lp_path: 输出 LP 文件路径（None 则用临时文件）
        timeout: 执行超时秒数

    Returns:
        dict with keys: objective_type, num_variables, num_constraints,
                        num_binary, num_integer, has_quadratic
        若执行失败返回 None
    """
    import tempfile
    if lp_path is None:
        tmp_fd, lp_path = tempfile.mkstemp(suffix=".lp", prefix="gt_lp_")
        os.close(tmp_fd)
        cleanup = True
    else:
        os.makedirs(os.path.dirname(lp_path) or ".", exist_ok=True)
        cleanup = False

    try:
        # 注入 LP 生成（跳过求解，只写 LP 文件）
        code = insert_lp_generation(reference_code, lp_path, skip_optimize=True)
        if code is None:
            return None

        # 在子进程中执行（隔离环境，避免污染当前进程）
        result = subprocess.run(
            ["python3", "-c", code],
            capture_output=True, text=True, timeout=timeout,
            env={
                **os.environ,
                "PYTHONDONTWRITEBYTECODE": "1",
                # 绕过代理，避免 Gurobi WLS TLS 错误
                "https_proxy": "",
                "HTTPS_PROXY": "",
                "http_proxy": "",
                "HTTP_PROXY": "",
                "ALL_PROXY": "",
                "all_proxy": "",
            },
        )

        if not os.path.exists(lp_path) or os.path.getsize(lp_path) == 0:
            return None

        stats = parse_lp_structure(lp_path)
        extended = {
            "objective_type": stats["objective_type"],
            "num_variables": stats["num_variables"],
            "num_constraints": stats["num_constraints"],
            "num_binary": stats["num_binary"],
            "num_integer": stats["num_integer"],
            "has_quadratic": stats["has_quadratic"],
        }

        # ── 提取系数矩阵 (L2) ──
        mat = extract_coefficient_matrix(lp_path)
        if mat is not None:
            # 稀疏矩阵序列化: 存 (row, col, data) 三元组
            A_coo = mat["A"].tocoo()
            extended["coeff_A"] = {
                "shape": [int(A_coo.shape[0]), int(A_coo.shape[1])],
                "rows": [int(r) for r in A_coo.row],
                "cols": [int(c) for c in A_coo.col],
                "data": [float(d) for d in A_coo.data],
            }
            extended["coeff_b"] = [float(x) for x in mat["b"]]

        # ── 计算 GT 扰动响应 (L4) ──
        try:
            pert = _compute_gt_perturbation(lp_path)
            if pert is not None:
                extended["perturbation"] = [float(x) for x in pert]
        except Exception:
            pass

        return extended
    except Exception:
        return None
    finally:
        if cleanup:
            try:
                os.remove(lp_path)
            except OSError:
                pass


def _coerce_int(value, default: int = 0) -> int:
    """Coerce None / missing stats to a safe int for L1 comparisons."""
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _within_tolerance(pred: int, gt: int, rel_tol: float = 0.3, abs_tol: int = 2) -> bool:
    """检查 pred 是否在 gt 的容差范围内: |pred - gt| <= max(abs_tol, rel_tol * gt)"""
    pred = _coerce_int(pred)
    gt = _coerce_int(gt)
    threshold = max(abs_tol, int(rel_tol * gt))
    return abs(pred - gt) <= threshold


def _tolerance_score(pred: int, gt: int, rel_tol: float = 0.3, abs_tol: int = 2) -> float:
    """根据偏离程度给分 (0.0~1.0)：
    - 在容差内: 1.0
    - 偏离 1 倍容差: 0.5
    - 偏离 3 倍容差以上: 0.0
    线性插值介于之间。

    特殊处理: 当 GT=0 时（不需要该类型变量），pred=0 给满分，pred>0 线性递减。
    """
    pred = _coerce_int(pred)
    gt = _coerce_int(gt)
    deviation = abs(pred - gt)

    # GT 为 0: 期望 pred 也为 0
    if gt == 0:
        if pred == 0:
            return 1.0
        # pred > 0, penalty proportional to deviation
        threshold = abs_tol or 1
        ratio = deviation / threshold
        if ratio >= 3.0:
            return 0.0
        return max(0.0, 1.0 - (ratio - 1.0) / 2.0)

    threshold = max(abs_tol, int(rel_tol * gt))
    # 当 threshold 为 0 但 GT > 0 时（小数值），使用 exact match
    if threshold == 0:
        threshold = 1

    if deviation <= threshold:
        return 1.0
    ratio = deviation / threshold
    if ratio >= 3.0:
        return 0.0
    return max(0.0, 1.0 - (ratio - 1.0) / 2.0)


def lp_structure_reward(lp_path: str, gt_stats: dict | None = None) -> float:
    """
    基于 LP 文件结构给出分数（0.0 ~ 0.5）。

    当提供 gt_stats 时，与 ground truth 结构做对比打分：
        - 目标函数类型一致 (Min/Max):                           +0.05
        - 变量数接近 (容差 max(2, 30%)):                        +0.10
        - 约束数接近 (容差 max(2, 30%)):                        +0.10
        - Binary 变量数接近 (容差 max(1, 20%)):                 +0.15
        - Integer 变量数接近 (容差 max(1, 20%)):                +0.10
        - 二次项匹配:                                           +0.05

    未提供 gt_stats 时，使用启发式检查：
        - 有目标函数:                                           +0.10
        - 有至少 2 个约束:                                      +0.15
        - 有至少 2 个变量:                                      +0.10
        - 变量是整数/二进制:                                    +0.10
        - 有二次项:                                             +0.05

    Args:
        lp_path: 模型生成的 LP 文件路径
        gt_stats: ground truth 结构统计（dict, 来自 extra_info.lp_ref）

    Returns:
        float: 0.0 ~ 0.5
    """
    if not os.path.exists(lp_path):
        return 0.0

    info = parse_lp_structure(lp_path)
    # 无法解析出任何结构（坏文件/空文件）时直接 0，避免 has_quadratic 等默认匹配误给分
    if not info.get("objective_type") and info.get("num_variables", 0) == 0 and info.get("num_constraints", 0) == 0:
        return 0.0

    if gt_stats is None:
        # ── 旧版启发式打分（向后兼容）──
        score = 0.0
        if info["objective_type"]:
            score += 0.10
        if info["num_constraints"] >= 2:
            score += 0.15
        if info["num_variables"] >= 2:
            score += 0.10
        if info["has_integers"]:
            score += 0.10
        if info["has_quadratic"]:
            score += 0.05
        return score

    # ── GT 对比打分 ──
    score = 0.0

    # 1. 目标类型一致
    gt_obj = gt_stats.get("objective_type", "")
    if info["objective_type"] and gt_obj and info["objective_type"] == gt_obj:
        score += 0.05

    # 2. 变量数接近
    gt_vars = _coerce_int(gt_stats.get("num_variables"))
    score += _tolerance_score(info["num_variables"], gt_vars, rel_tol=0.3, abs_tol=2) * 0.10

    # 3. 约束数接近
    gt_cons = _coerce_int(gt_stats.get("num_constraints"))
    score += _tolerance_score(info["num_constraints"], gt_cons, rel_tol=0.3, abs_tol=2) * 0.10

    # 4. Binary 变量数接近（严格：若一方为 0 则最多 0.3 分）
    # Mini ablation: binary sub-weight 0.10 → 0.15 (MIP / IndustryOR-style signal).
    gt_bin = _coerce_int(gt_stats.get("num_binary"))
    pred_bin = _coerce_int(info.get("num_binary"))
    if gt_bin > 0 or pred_bin > 0:
        bin_score = _tolerance_score(pred_bin, gt_bin, rel_tol=0.3, abs_tol=0)
        if (pred_bin == 0) != (gt_bin == 0):
            bin_score = min(bin_score, 0.3)
        score += bin_score * 0.15

    # 5. Integer 变量数接近
    gt_int = _coerce_int(gt_stats.get("num_integer"))
    pred_int = _coerce_int(info.get("num_integer"))
    if gt_int > 0 or pred_int > 0:
        int_score = _tolerance_score(pred_int, gt_int, rel_tol=0.3, abs_tol=0)
        if (pred_int == 0) != (gt_int == 0):
            int_score = min(int_score, 0.3)
        score += int_score * 0.10

    # 6. 二次项匹配
    if info["has_quadratic"] == bool(gt_stats.get("has_quadratic", False)):
        score += 0.05

    # 7. 增广主角 (L2)
    # GT 系数矩阵在 lp_ref 中是序列化 dict，需要反序列化
    try:
        mat = extract_coefficient_matrix(lp_path)
        gt_A_raw = gt_stats.get("coeff_A")
        gt_b_raw = gt_stats.get("coeff_b")
        if mat is not None and gt_A_raw is not None and gt_b_raw is not None:
            gt_A = _deserialize_coeff_matrix(gt_A_raw) if isinstance(gt_A_raw, dict) else gt_A_raw
            if gt_A is not None:
                pa_score = _principal_angle_score(mat["A"], mat["b"], gt_A, gt_b_raw)
                score += pa_score * 0.15
    except Exception:
        pass   # GP LP 格式不标准或含 None 值，跳过 L2

    # 8. 扰动响应 (L4)
    gt_pert = gt_stats.get("perturbation")
    if gt_pert is not None and len(gt_pert) > 0:
        try:
            pert_score = _perturbation_score(lp_path, gt_pert)
            score += pert_score * 0.10
        except Exception:
            pass   # 扰动求解失败，跳过 L4

    return min(score, 0.5)


# ── 系数矩阵提取（使用 Gurobi API）──

def extract_coefficient_matrix(lp_path: str):
    """
    使用 gurobipy.read() 从 LP 文件中提取系数矩阵和相关向量。

    返回 None（若 LP 无效或 Gurobi 不可用），否则返回 dict:
        A: sparse CSR matrix (n_constrs × n_vars)
        b: RHS vector (n_constrs,)
        c: objective coefficient vector (n_vars,)
        sense: constraint sense list ('<', '>', '=')
        vtype: variable type list ('C'=continuous, 'B'=binary, 'I'=integer)
        n_vars: int
        n_constrs: int
    """
    try:
        from gurobipy import read as grb_read
        m = grb_read(lp_path)
    except Exception:
        return None

    try:
        A = m.getA()                                                    # sparse CSR
        n_constrs, n_vars = A.shape
        b = m.getAttr("RHS", m.getConstrs())
        c = m.getAttr("Obj", m.getVars())
        sense = m.getAttr("Sense", m.getConstrs())
        vtype = m.getAttr("VType", m.getVars())

        return {
            "A": A,
            "b": b,
            "c": c,
            "sense": sense,
            "vtype": vtype,
            "n_vars": n_vars,
            "n_constrs": n_constrs,
        }
    except Exception:
        return None


def _normalize_rows(A_sparse, b):
    """对增广矩阵 [A, -b] 的每一行做 L2 归一化，消除约束缩放歧义。"""
    import numpy as np
    from scipy.sparse import csr_matrix

    n_constrs = A_sparse.shape[0]
    A_dense = np.array(A_sparse.todense())
    b_arr = np.array(b).reshape(-1, 1)
    aug = np.hstack([A_dense, -b_arr])

    norms = np.linalg.norm(aug, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1.0   # 避免除零
    aug_normalized = aug / norms

    return aug_normalized


def _principal_angle_score(A_pred, b_pred, A_gt, b_gt):
    """
    计算增广主角得分。

    A_pred, A_gt: sparse CSR matrices
    b_pred, b_gt: RHS 向量

    返回 float in [0,1]，1表示行空间完全对齐
    """
    import numpy as np

    # 归一化增广矩阵
    aug_pred = _normalize_rows(A_pred, b_pred)
    aug_gt   = _normalize_rows(A_gt, b_gt)

    # 确保列数一致（变量数不同时用零列补齐）
    max_cols = max(aug_pred.shape[1], aug_gt.shape[1])
    if aug_pred.shape[1] < max_cols:
        aug_pred = np.hstack([aug_pred, np.zeros((aug_pred.shape[0], max_cols - aug_pred.shape[1]))])
    if aug_gt.shape[1] < max_cols:
        aug_gt = np.hstack([aug_gt, np.zeros((aug_gt.shape[0], max_cols - aug_gt.shape[1]))])

    # SVD 截断取行空间基
    k = min(aug_pred.shape[0], aug_pred.shape[1], 10)
    try:
        _, _, Vt_pred = np.linalg.svd(aug_pred, full_matrices=False)
        _, _, Vt_gt   = np.linalg.svd(aug_gt, full_matrices=False)
    except np.linalg.LinAlgError:
        return 0.0

    U_pred = Vt_pred[:k, :].T    # 列是行空间标准正交基
    U_gt   = Vt_gt[:k, :].T

    # 跨投影矩阵的奇异值 = cos(θ_k)
    cross = U_pred.T @ U_gt
    try:
        singular_vals = np.linalg.svd(cross, compute_uv=False)
    except np.linalg.LinAlgError:
        return 0.0

    cos2_sum = np.sum(np.clip(np.square(singular_vals), 0, 1))
    score = cos2_sum / min(k, len(singular_vals))

    return max(0.0, min(1.0, score))


# ── 扰动响应 ──

def _perturbation_score(lp_path: str, gt_perturbation: list | None) -> float:
    """
    通过对目标函数系数做微小扰动，比较灵敏度。

    gt_perturbation: GT 的灵敏度向量 [Δ₁, Δ₂, Δ₃]（离线预计算存入 lp_ref）。
    若未提供，返回 0.0。

    返回 float in [0,1]
    """
    if gt_perturbation is None:
        return 0.0

    try:
        from gurobipy import read as grb_read, GRB
        m = grb_read(lp_path)
        # 必须限时：此处求解跑在 RewardLoopWorker 主进程，没有 executor 的
        # 超时保护，病态 MIP 无限求解会卡死整个训练流水线
        m.setParam('TimeLimit', 10)
        m.setParam('OutputFlag', 0)
    except Exception:
        return 0.0

    n_vars = m.NumVars
    if n_vars == 0:
        return 0.0

    # 获取当前目标系数
    obj_orig = m.getAttr("Obj", m.getVars())

    # 记录原始 objVal（如果可以求解）
    try:
        m.optimize()
        if m.Status == GRB.OPTIMAL:
            obj_orig_val = m.ObjVal
        else:
            return 0.0
    except Exception:
        return 0.0

    # 3 个正交扰动方向
    delta = 0.02  # 2% 相对扰动
    pred_deltas = []
    for dim in range(min(3, n_vars)):
        try:
            mc = m.copy()
            mc.setParam('TimeLimit', 10)
            mc.setParam('OutputFlag', 0)
            # 对第 dim 个变量的目标系数加扰动
            mc.getVars()[dim].Obj = obj_orig[dim] * (1.0 + delta)
            mc.optimize()
            if mc.Status == GRB.OPTIMAL:
                delta_val = (mc.ObjVal - obj_orig_val) / (delta * abs(obj_orig[dim]) + 1e-8)
                pred_deltas.append(delta_val)
            else:
                pred_deltas.append(0.0)
            mc.dispose()
        except Exception:
            pred_deltas.append(0.0)

    if len(pred_deltas) == 0:
        return 0.0

    # 与 GT 灵敏度比较
    gt_deltas = gt_perturbation[:len(pred_deltas)]
    errors = [abs(p - g) / (abs(g) + 1e-8) for p, g in zip(pred_deltas, gt_deltas)]
    score = max(0.0, 1.0 - sum(errors) / len(errors))
    return min(1.0, score)


def _compute_gt_perturbation(lp_path: str):
    """计算 GT 模型的扰动响应向量（离线预处理用）。"""
    try:
        from gurobipy import read as grb_read, GRB
        m = grb_read(lp_path)
    except Exception:
        return None

    n_vars = m.NumVars
    if n_vars == 0:
        return None

    obj_coeffs = m.getAttr("Obj", m.getVars())
    try:
        m.optimize()
        if m.Status != GRB.OPTIMAL:
            return None
        obj_orig = m.ObjVal
    except Exception:
        return None

    delta = 0.02
    deltas = []
    for dim in range(min(3, n_vars)):
        if abs(obj_coeffs[dim]) < 1e-8:
            deltas.append(0.0)
            continue
        try:
            mc = m.copy()
            mc.getVars()[dim].Obj = obj_coeffs[dim] * (1.0 + delta)
            mc.optimize()
            if mc.Status == GRB.OPTIMAL:
                d_val = (mc.ObjVal - obj_orig) / (delta * abs(obj_coeffs[dim]))
                deltas.append(d_val)
            else:
                deltas.append(0.0)
            mc.dispose()
        except Exception:
            deltas.append(0.0)
    m.dispose()
    return deltas


def _deserialize_coeff_matrix(coeff_A_dict):
    """从序列化的 dict 恢复 scipy.sparse.csr_matrix。"""
    try:
        from scipy.sparse import csr_matrix
        import numpy as np
        shape = tuple(coeff_A_dict["shape"])
        data = np.array(coeff_A_dict["data"], dtype=float)
        row = np.array(coeff_A_dict["rows"], dtype=int)
        col = np.array(coeff_A_dict["cols"], dtype=int)
        return csr_matrix((data, (row, col)), shape=shape)
    except Exception:
        return None


# ── 以下为兼容旧代码的函数 ──

def extract_integer_binary(str_log):
    return 'Integer Variables Exists' in str_log or 'Binary Variables Exists' in str_log

def enforce_integer_variables(code):
    """在 Gurobi 的 addVar/addVars 中插入 vtype=GRB.INTEGER"""
    pattern = r'(\w+\s*=\s*\w+\.addVar[s]?)\(([\s\S]*?)(\)\n)'
    def replacer(match):
        var_assignment = match.group(1)
        params = match.group(2).rstrip()
        closing = match.group(3)
        if re.search(r'\bvtype\s*=', params):
            return match.group(0)
        if params:
            if not params.endswith(','):
                params += ','
            new_params = f"{params} vtype=GRB.INTEGER"
        else:
            new_params = "vtype=GRB.INTEGER"
        return f"{var_assignment}({new_params}{closing}"
    return re.sub(pattern, replacer, code, flags=re.MULTILINE)

def change_variable_types(str_log):
    if "Vtype" in str_log or "vtype" in str_log:
        if 'INTEGER' in str_log:
            return str_log.replace('INTEGER', 'CONTINUOUS')
        elif 'CONTINUOUS' in str_log:
            return str_log.replace('CONTINUOUS', 'INTEGER')
    else:
        return enforce_integer_variables(str_log)

if __name__ == "__main__":
    """
    用数据集样本测试 content_utils 的核心函数:
      1. extract_code_block  — 从 LLM 输出提取 <python> 代码
      2. extract_obj         — 从执行 stdout 提取 objVal
      3. parse_lp_structure  — 解析 LP 文件结构
      4. lp_structure_reward — LP 结构打分
      5. insert_lp_generation — 注入 LP 生成代码并执行
    """
    import pandas as pd
    from executor import PythonExecutor

    # ── 加载数据集 ──
    DATA_PATH = os.path.join(
        os.path.dirname(__file__), "..", "..",
        "trainset", "gurobi_examples_OR_train_fixed.parquet"
    )
    if not os.path.exists(DATA_PATH):
        DATA_PATH = "/home/guo/LLM/Verl/verl/trainset/gurobi_examples_OR_train_fixed.parquet"

    print(f"Loading dataset from: {DATA_PATH}")
    df = pd.read_parquet(DATA_PATH)
    print(f"Dataset shape: {df.shape}\n")

    # ── 取 3 条样本测试 ──
    test_indices = [0, 1, 2]
    for idx in test_indices:
        row = df.iloc[idx]
        msgs = row["prompt"]
        if hasattr(msgs, "tolist"):
            msgs = msgs.tolist()

        # 提取 user prompt
        user_msg = next((m["content"] for m in msgs if m["role"] == "user"), "")
        ground_truth = row.get("reward_model", {}).get("ground_truth", None)
        en_answer = row.get("en_answer", None)

        print(f"{'='*60}")
        print(f"Sample {idx}")
        print(f"  data_source : {row.get('data_source', '?')}")
        print(f"  ground_truth: {ground_truth}")
        print(f"  en_answer   : {en_answer}")
        print(f"  user prompt : {user_msg[:100]}...")
        print()

        # ── 测试 1: extract_code_block（用一段模拟的 LLM 输出）──
        mock_output_with_tags = f"""<think>
This is a simple LP problem.
</think>
<model>
Maximize x + y subject to constraints.
</model>
<python>
from gurobipy import *

m = Model("test")
x = m.addVar(name="x", lb=0)
y = m.addVar(name="y", lb=0)
m.setObjective(x + y, GRB.MAXIMIZE)
m.addConstr(x + 2 * y <= 4, "c1")
m.addConstr(3 * x + y <= 6, "c2")
m.optimize()
print(f"Obj: {{m.objVal}}")
</python>"""

        lp_path = f"/tmp/test_lp_{uuid.uuid4().hex[:8]}.lp"
        code = extract_code_block(mock_output_with_tags, "gurobi", lp_output_path=lp_path)
        print(f"  [extract_code_block]")
        print(f"    extracted code ({len(code) if code else 0} chars):")
        if code:
            for line in code.split("\n")[:8]:
                print(f"      {line}")
            print(f"      ...")
        print(f"    lp_path: {lp_path}")
        print()

        # ── 测试 2: 执行代码 + extract_obj ──
        if code:
            executor = PythonExecutor(timeout_length=30)
            result, report = executor.execute(code.split("\n"), runtime=executor.runtime, timeout_length=30)
            print(f"  [executor.execute]")
            print(f"    report: {report}")
            print(f"    stdout: {str(result)[:200]}")
            obj_val = extract_obj(str(result))
            print(f"    extract_obj: {obj_val}")
            print()

        # ── 测试 3: parse_lp_structure ──
        if os.path.exists(lp_path):
            info = parse_lp_structure(lp_path)
            print(f"  [parse_lp_structure]")
            print(f"    objective_type : {info['objective_type']}")
            print(f"    num_variables  : {info['num_variables']}")
            print(f"    num_constraints: {info['num_constraints']}")
            print(f"    has_quadratic  : {info['has_quadratic']}")
            print(f"    has_integers   : {info['has_integers']}")
            print()

            # ── 测试 4: lp_structure_reward ──
            lp_score = lp_structure_reward(lp_path)
            print(f"  [lp_structure_reward] score = {lp_score}")
            print()

            # 清理
            os.remove(lp_path)
        else:
            print(f"  [WARN] LP file not generated: {lp_path}")
            print()

    # ── 测试 5: 不带 <python> 标签的 fallback ──
    print(f"{'='*60}")
    print("Test 5: fallback to ```python``` block")
    mock_output_fallback = """Here is the solution:

```python
from gurobipy import *
m = Model()
x = m.addVar()
m.setObjective(x, GRB.MAXIMIZE)
m.addConstr(x <= 5)
m.optimize()
print(f"Just print the best obj: {m.objVal}")
```
"""
    code2 = extract_code_block(mock_output_fallback, "gurobi")
    print(f"  extracted ({len(code2) if code2 else 0} chars): {code2[:100] if code2 else 'None'}...")
    print()

    print("All tests done.")