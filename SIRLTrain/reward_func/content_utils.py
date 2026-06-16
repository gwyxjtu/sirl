import re
import subprocess
import textwrap
import os
import uuid

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

def insert_lp_generation(code: str, output_name: str) -> str:
    """在 optimize() 前注入 model.write('xxx.lp')，让 Gurobi 写出 LP 文件；
    同时在 optimize 后打印最优目标值。"""
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
        status_check = (
            f"{indent}{model_name}.write('{output_name}')\n"
            f"{indent}{model_name}.optimize()\n"
            f"{indent}if {model_name}.status == GRB.OPTIMAL:\n"
            f"{indent}    print(f'Just print the best obj: {{{model_name}.ObjVal}}')\n"
            f"{indent}else:\n"
            f"{indent}    print('No optimal solution found, status:', {model_name}.status)"
        )
        code = re.sub(pattern, rf'\1\2\n{status_check}', code, flags=re.M)
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
            has_quadratic: bool (目标中有 ^2 或 * 等二次项)
            has_integers: bool (General 或 Binary 段存在)
            raw_lines: list of lines
    """
    result = {
        "objective_type": "",
        "num_variables": 0,
        "num_constraints": 0,
        "has_quadratic": False,
        "has_integers": False,
        "raw_lines": [],
    }

    if not os.path.exists(lp_path):
        return result

    with open(lp_path) as f:
        lines = f.readlines()
    result["raw_lines"] = lines

    in_section = None
    constraint_count = 0
    var_count = 0

    for line in lines:
        stripped = line.strip()

        # 识别段落
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
        elif stripped == "Generals" or stripped == "Binaries":
            result["has_integers"] = True
            in_section = "integers"
            continue
        elif stripped == "End":
            break

        # 计数
        if in_section == "objective":
            if "^2" in stripped or " * " in stripped:
                result["has_quadratic"] = True

        elif in_section == "constraints":
            # 每个非空、非注释行算一个约束名（Gurobi LP 格式： name: expr）
            if stripped and not stripped.startswith("\\") and ":" in stripped:
                constraint_count += 1

        elif in_section == "bounds":
            # 每个独立变量一行： x[A] <= 0.3  或 0 <= x_1 <= 1
            if stripped:
                var_count += 1

    result["num_variables"] = var_count
    result["num_constraints"] = constraint_count

    return result

def lp_structure_reward(lp_path: str) -> float:
    """
    基于 LP 文件结构给出分数（0.0 ~ 0.5）。

    检查项：
        - 有目标函数:                                         +0.10
        - 有至少 2 个约束:                                    +0.15
        - 有至少 2 个变量:                                    +0.10
        - 变量是整数/二进制 (如果有 Generals/Binaries 段):     +0.10
        - 有二次项 (说明模型复杂度不弱):                        +0.05

    Returns:
        float: 0.0 ~ 0.5
    """
    if not os.path.exists(lp_path):
        return 0.0

    info = parse_lp_structure(lp_path)
    score = 0.0

    if info["objective_type"]:
        score += 0.10                     # 有目标函数
    if info["num_constraints"] >= 2:
        score += 0.15                     # 有足够约束
    if info["num_variables"] >= 2:
        score += 0.10                     # 有足够变量
    if info["has_integers"]:
        score += 0.10                     # 有整数/二进制变量
    if info["has_quadratic"]:
        score += 0.05                     # 有二次项

    return score

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