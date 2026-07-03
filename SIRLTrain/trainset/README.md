# Trainset 目录说明

## 文件清单

| 文件 | 大小 | 说明 | 当前用途 |
|------|------|------|----------|
| `gurobi_examples_OR_train.parquet` | 23M | 原始训练集 | 构建 shot/fixed 的源数据，训练不直接用 |
| `gurobi_examples_OR_test.parquet` | 497K | 原始测试集 | 同上 |
| `gurobi_examples_OR_train_fixed.parquet` | 23M | 修正 prompt + lp_ref | 上一轮训练使用 |
| `gurobi_examples_OR_test_fixed.parquet` | 498K | 修正 prompt + lp_ref | 上一轮验证使用 |
| `gurobi_examples_OR_train_shot.parquet` | 24M | **fix prompt + few-shot + lp_ref** | **当前训练使用** |
| `gurobi_examples_OR_test_shot.parquet` | 500K | **fix prompt + few-shot + lp_ref** | **当前验证使用** |

## 所有文件统一的 Schema

| 列名 | 类型 | 说明 |
|------|------|------|
| `en_question` | str | 英文题目正文 |
| `en_answer` | float | 标准答案（目标函数值） |
| `output` | str | reference 代码（含 `<python>` 标签） |
| `sol` | list[float] | reference 最优解向量 |
| `data_source` | str | 题目来源，全部为 `"OptMATH"` |
| `prompt` | list[dict] | `[{"role":"system","content":...}, {"role":"user","content":...}]` |
| `ability` | str | 能力标签，全部为 `"OR"` |
| `reward_model` | dict | 训练时 VeRL 读取的 reward 信息，格式 `{"ground_truth": <float>}` |
| `extra_info` | dict | 训练时 VeRL 传入 reward 函数的额外信息 |
| `extra_info.lp_ref` | dict | GT 模型 LP 结构统计，由 `tools/build_lp_ref_stats.py` 离线预计算 |
| `extra_info.sol` | list[float] | reference 最优解向量（从 `sol` 列复制） |

## extra_info.lp_ref 结构

```python
{
    "objective_type": "Minimize",   # 目标类型: "Minimize" / "Maximize" / ""
    "num_variables": 12,            # 变量总数
    "num_constraints": 8,           # 约束总数
    "num_binary": 5,                # 0-1 变量数
    "num_integer": 0,               # 整数变量数 (不含 Binary)
    "has_quadratic": False          # 是否有二次项
}
```

## 三个版本的系统 prompt 差异

### 1. `_fixed` 版本（基础指令，531 字符）

```
You are a helpful Assistant with expertise in mathematical modeling and the Gurobi solver.
Given an optimization problem, provide the Gurobi Python code to solve it.

Your response MUST follow this structure exactly:

<python>
Provide the complete Gurobi Python code to implement the model.
IMPORTANT: Always include `from gurobipy import *` at the top.
Print the optimal objective value and optimal solution clearly.
</python>

Do NOT include any thinking steps or mathematical model sections.
Only output the <python> code block.
```

### 2. `_shot` 版本（基础指令 + few-shot 示例 + API 规则，3041 字符）

在基础指令后追加：

```
--- EXAMPLE (study this before writing your code) ---

Problem: A factory must decide which of 2 machines (M1, M2) to operate to produce 2 products (A, B).
- Demand: A=100, B=150 units
- Machine capacities: M1=200, M2=180 hours
- Production time per unit: M1→A:2h, M1→B:3h, M2→A:4h, M2→B:2h
- Operating cost per machine: M1=$500, M2=$400 (only when used)
- If total production < 200 units, pay a one-time penalty of $1000. Otherwise no penalty.

<python>
from gurobipy import *

# Data
demand  = {'A': 100, 'B': 150}
cap     = {'M1': 200, 'M2': 180}
hours   = {('M1','A'):2, ('M1','B'):3, ('M2','A'):4, ('M2','B'):2}
op_cost = {'M1': 500, 'M2': 400}
penalty = 1000
penalty_threshold = 200

model = Model("Factory")
machines, products = ['M1','M2'], ['A','B']

x = model.addVars(machines, products, lb=0, vtype=GRB.CONTINUOUS, name="x")
y = model.addVars(machines, vtype=GRB.BINARY, name="y")
z = model.addVar(vtype=GRB.BINARY, name="z")

total = sum(x[m,p] for m in machines for p in products)
model.setObjective(
    sum(op_cost[m] * y[m] for m in machines) + penalty * (1 - z),
    GRB.MINIMIZE
)

for p in products:
    model.addConstr(sum(x[m,p] for m in machines) == demand[p], f"Demand_{p}")
for m in machines:
    model.addConstr(sum(hours[m,p] * x[m,p] for p in products)
                    <= cap[m] * y[m], f"Capacity_{m}")

model.addConstr(total >= penalty_threshold * z, "Penalty_Indicator")
model.optimize()

if model.status == GRB.OPTIMAL:
    print("Optimal Objective Value:", model.ObjVal)
    for m in machines:
        if y[m].X > 0.5:
            print(f"{m} selected")
            for p in products:
                if x[m,p].X > 0:
                    print(f"  {p}: {x[m,p].X}")
else:
    print("No optimal solution, status:", model.status)
</python>

--- KEY GUROBI RULES (must follow) ---
1. Use model.ObjVal (capital O,V) — NOT objVal; use var.X (capital X) — NOT var.x.
2. Binary: vtype=GRB.BINARY; Integer: vtype=GRB.INTEGER; Continuous: vtype=GRB.CONTINUOUS.
3. Conditional penalties → binary indicator var + Big-M (see z above). Do NOT use addConstr(>= threshold) + addConstr(<= threshold).
4. Flow variables must be non-negative (lb=0). lb=-GRB.INFINITY is wrong.
5. Always check model.status == GRB.OPTIMAL before printing results.
---
```

### 3. 原始版本（`_train.parquet` / `_test.parquet`）

系统 prompt 和用户 prompt 来自数据构建阶段，**没有上述两种修正**。`output` 中部分 reference 代码的 API 拼写与 Gurobi 标准不一致（如 `objVal` 而非 `ObjVal`）。

## lp_ref 覆盖率

| 文件 | 总数 | 有 lp_ref | 覆盖率 | 说明 |
|------|------|----------|--------|------|
| `_train_fixed` | 4097 | 4091 | 99.85% | 6 条纯文本解释无代码 |
| `_test_fixed` | 84 | 84 | 100% | |
| `_train_shot` | 4097 | 3999 | 97.6% | 98 条 reference 执行失败 |
| `_test_shot` | 84 | 83 | 98.8% | 1 条 reference 执行失败 |

## lp_ref 生成方法

```bash
cd tools/
python build_lp_ref_stats.py --trainset-path ../trainset/gurobi_examples_OR_train_shot.parquet \
                              --testset-path  ../trainset/gurobi_examples_OR_test_shot.parquet \
                              --overwrite-src --workers 4
```

流程：读取 parquet → 提取 reference 代码 → 注入 `model.write("xxx.lp")` → 子进程执行 → 解析 LP 文件 → 写入 `extra_info.lp_ref`

## 数据构建流水线

```
原始数据 (train.parquet / test.parquet)
  │
  ├─→ tools/fix_prompts.py                     → 修正系统/用户 prompt     → _fixed.parquet
  ├─→ tools/fix_prompts.py --with-example       → 修正 + few-shot 示例      → _shot.parquet
  └─→ tools/build_lp_ref_stats.py               → 离线提取 GT LP 结构统计
                                                    └→ 写入 extra_info.lp_ref
```
