# SIRL: Structural-Informed RL for Optimization Modeling

## 1. 核心主张

传统的 LLM-for-OR 评估只比较最优目标值（objVal），但两个结构截然不同的模型可以产生相同的数值答案。我们主张：**ground truth 不应只是目标函数，还应包括模型的样子**——决策变量的类型、约束系统的几何结构、变量-约束二分图的拓扑特征。

论文围绕两个核心贡献展开：

- **贡献 1（信号层）**：从约束系统的代数几何中提取多层级结构信号，设计超越 objVal 的 reward 函数
- **贡献 2（算法层）**：求解器引导的自改进训练——求解器不仅是打分器，还是诊断器 + 修正器，构建反馈驱动的自改进循环

---

## 2. 贡献 1：结构感知 Reward 体系

### 2.1 总公式

```
score = ans_ok × 1.0 + code_ok × 1.0 + format × 0.5 + lp_score × 0.5
        ───────────   ───────────   ──────────   ──────────────
        数值正确性      代码可执行性     输出格式       结构忠实度
```

### 2.2 lp_score 的多层级结构

#### Level 1: 基数统计（✅ 已实现）

| 维度 | 权重 | 容差 | 说明 |
|------|------|------|------|
| 目标类型 (Min/Max) | 0.05 | 精确匹配 | |
| 变量数 | 0.10 | max(2, 30%) | `_tolerance_score` 连续打分 |
| 约束数 | 0.10 | max(2, 30%) | 同上 |
| Binary 变量数 | 0.10 | max(0, 30%) | 一方为0则cap 0.03 |
| Integer 变量数 | 0.10 | max(0, 30%) | 一方为0则cap 0.03 |
| 二次项匹配 | 0.05 | 精确 | 布尔 |

#### Level 2: 增广主角（⏳ 待实现）

**理论**：约束系统 `A x ≤ b` 定义了可行域多面体。升维到齐次坐标：增广矩阵 `Ã = [A, -b]`，每行 L2 归一化以消除约束缩放歧义。对 Ã 做 SVD，比较行空间之间的主角：

```
cos(θ_k) = σ_k (Ũ_pred^T · Ũ_gt)
principal_score = (1/K) · Σ_k cos²(θ_k)
```

- 权重：**0.15**
- 捕捉：约束方向 + 截距（完整的仿射几何）
- 性质：排列不变、缩放不变（每行归一化）
- 理论保证：score=1 ⇔ 两个可行域仿射等价

#### Level 3: 拉普拉斯谱（⏳ 待实现）

**理论**：系数矩阵的 sparsity pattern 定义变量-约束二分图。不同类型的 OR 问题有特征性的二分图结构（指派类型的每个变量恰好连2条约束，网络流的连通性模式，背包的单约束中心）。归一化拉普拉斯的谱编码了这些拓扑特征。

```
λ = eigsh(L_norm) 的前 K=10 个最小特征值
spectral_score = 1 - W₁(λ_pred, λ_gt)   (Wasserstein-1 距离)
```

- 权重：**0.05**
- 捕捉：二分图拓扑（连通性、代数连通度、谱间隙）

#### Level 4: 扰动响应与绑定秩（⏳ 待实现）

**扰动响应**：对目标函数系数做微小扰动，比较 pred 和 reference 的灵敏度。两个碰巧产出相同 objVal 但结构不同的模型，对系数的微小变化会产生不同的响应——这是 stress test 的思路，不依赖约束对齐。

**绑定秩**：提取 binding 约束（dual ≠ 0）的系数矩阵，计算其秩。反映最优解处的有效约束维度，是布局不变的聚合指标。

- 权重：**0.10**
- 捕捉：结构灵敏度 + 最优解的有效约束维度

### 2.3 GT 结构统计的离线提取

`tools/build_lp_ref_stats.py`：parquet → 子进程执行 reference 代码 → 写出 ref.lp → parse → 写入 `extra_info.lp_ref`

当前覆盖率：train `_shot` 97.6%，test `_shot` 98.8%

---

## 3. 贡献 2：求解器引导的自改进训练

### 3.1 核心思想

标准 GRPO 中，代码执行失败的 sample 被无情丢弃——无论它是"差一步就对了"还是"完全胡写"。求解器内部的错误信息（AttributeError、INFEASIBLE、UNBOUNDED）被压缩成一个标量 `code_ok=0`，丢失了所有诊断信息。

我们主张：**求解器不仅是打分器，还是诊断器 + 修正器**。

```
标准 GRPO:
  rollout → 求解器执行 → Done/Error → 标量 reward → 失败样本丢弃

我们的方法:
  rollout → 求解器执行
    ├── Done → reward 评分 → 参与 PPO
    └── Error → 求解器诊断（IIS/Unbounded/API错误）
         ├── 可自动修复 → 修正 → 重新求解 → 成功
         │    → reward 评分 → 标记 "corrected" → 参与同一步 PPO
         └── 无法修复 → 丢弃
```

### 3.2 求解器诊断与自动修复

所有类型的执行失败统一走一个诊断-修复流水线：

```
                    求解器执行失败
                         │
        ┌────────────────┼────────────────┐
        ▼                ▼                ▼
   INFEASIBLE       UNBOUNDED        API Error
   computeIIS()    getUnbdRay()    字符串匹配
        │                │                │
   提取矛盾约束     提取无界方向     objVal→ObjVal
        │                │          Binary→GRB.BINARY
   删矛盾约束      补缺失上界          等自动替换
        │                │                │
        └────────────────┼────────────────┘
                         ▼
                    修正代码
                         │
                    重新求解
                    ┌───┴───┐
                    ▼       ▼
                  成功     仍失败
                    │        │
              参与PPO    丢弃
```

**为什么是一个统一组件**：不同错误类型的诊断接口不同（IIS、getUnbdRay、字符串匹配），但**修正后的处理完全一致**——重新求解，成功则参与 PPO，失败则丢弃。把诊断阶段按错误类型分叉，把修正阶段统一处理。

### 3.3 自动对比对：Corrected-vs-Original

修正样本参与 PPO 时，和原始 Done 样本不加区分。但每个修正样本都携带一个特殊的元信息：它的"前身"是一个失败样本。

这自然构造了一个对比学习场景：

```
original (模型自己的草稿):   失败 — 求解器报错
corrected (求解器修正版):    成功 — 与 reference 结构一致
```

训练过程中，模型的参数更新同时作用于两个样本：
- 修正样本的正向 advantage 告诉模型"这样的代码是好的"
- 原始样本的负向 advantage 告诉模型"你自己的写法有问题"

**和 DPO 的关系**：DPO 需要人类标注 preferred vs dispreferred 对。我们的 corrected-vs-original 对是求解器自动生成的——**Solver-Automated Contrastive Pairs**，无需人类参与。

### 3.4 涌现的课程效应

随着训练推进，诊断-修正率自然下降：

```
训练初期: correction_rate ≈ 30%  （大量 API 拼写错误 + 不可行约束）
训练后期: correction_rate ≈ 3%   （模型自己写对了）
```

模型学会的是"避免触发诊断"。这产生了和显式课程学习相同的效果——先学会代码可执行性，再学会数值正确性，最后优化结构忠实度——但这个顺序是**涌现的**，不需要人工设计阶段切换逻辑。

实验可以展示：correction_rate 的下降曲线和 code_ok 的提升曲线高度负相关，证明自改进循环的有效性。

### 3.5 论文叙事

> Standard GRPO discards failed rollouts, compressing rich solver diagnostics into a binary code_ok signal. We propose Solver-Guided Self-Improvement, which treats the optimization solver as an active diagnostic oracle. When a generated model fails, the solver's internal interfaces (IIS for infeasibility, Unbounded Ray for missing constraints, error-type classification for API mistakes) pinpoint the failure cause. An automated repair module produces a corrected version, which is re-evaluated and fed back into the same PPO update as a positive training sample. This creates a tight feedback loop where the model learns to avoid failure patterns—producing emergent curriculum effects without explicit phase scheduling. The corrected-vs-original sample pairs also serve as solver-automated contrastive pairs, bypassing the need for human preference labeling.

---

## 4. 实验设计

### 4.1 基线对比

| 方法 | Reward | 算法 | 预期 |
|------|--------|------|------|
| Baseline | 仅 objVal | 标准 GRPO | 答案对但结构差 |
| Ours-Signal | objVal + lp_score (Level1-4) | 标准 GRPO | 结构 reward 的价值 |
| Ours-Full | objVal + lp_score (Level1-4) | + Solver-Guided Self-Improvement | 全方案 |

### 4.2 消融实验

| 消融 | 控制变量 | 验证结论 |
|------|----------|----------|
| A. 结构信号消融 | Baseline vs Ours-Signal | 基数统计是否已足够？主角和谱是否有 beyond-counting 的价值？ |
| B. 自改进消融 | Ours-Signal vs Ours-Full | 求解器诊断反馈是否有额外收益？ |
| C. 修正样本贡献 | 仅 Done样本 vs Done+修正样本 | 修正样本是否加速了收敛？ |
| D. 扰动响应消融 | lp_score 含/不含 perturbation | 灵敏度检验是否提供了额外信号？ |
| E. correction_rate 下降曲线 | 训练过程中监控 | 涌现的课程效应是否确实发生？ |

### 4.3 评估指标

| 指标 | 含义 | 测量方式 |
|------|------|----------|
| answer_accuracy | ans_ok 率 | 验证集 |
| code_viability | code_ok 率 | 验证集 |
| structural_fidelity | structure_score | 验证集 |
| correction_rate | 需要求解器诊断修正的比例 | 训练 log（应随训练递减） |
| corrected_sample_reward | 修正样本的 reward 分布 | 与原始 Done 样本分布的比较 |

---

## 5. 论文结构

```
1. Introduction
   - 仅比较 objVal 的局限性
   - 两层贡献: structural reward + solver-guided self-improvement
   - 核心创新: 求解器不仅是 evaluator，还是 diagnostic oracle

2. Related Work
   - LLM for OR modeling
   - RLHF / GRPO for code generation
   - Solver-assisted optimization
   - Self-improvement and contrastive learning for LLMs

3. Method
   3.1 Structural Reward Design
       3.1.1 Cardinality-based lp_score
       3.1.2 Augmented Principal Angles
       3.1.3 Laplacian Spectral Distance
       3.1.4 Perturbation Response & Binding Rank
   3.2 Solver-Guided Self-Improvement
       3.2.1 Failure Diagnosis via Solver APIs
       3.2.2 Automated Repair & Corrected Sample Re-injection
       3.2.3 Solver-Automated Contrastive Pairs
       3.2.4 Emergent Curriculum via Repair Decay
   3.3 Training Pipeline

4. Experiments
   4.1 Setup: GRPO on OptMATH
   4.2 Main Results
   4.3 Ablation Studies (5组消融)
   4.4 Qualitative Analysis

5. Conclusion
```

---

## 6. 当前进度 vs 待完成

| 组件 | 状态 | 位置 |
|------|------|------|
| lp_score 6维基数对比 | ✅ | `reward_func/content_utils.py` |
| Dict返回 + jsonl 日志 | ✅ | `reward_func/batch_score_gurobi.py` |
| lp_ref 离线预处理 | ✅ | `tools/build_lp_ref_stats.py` |
| Few-shot prompt | ✅ | `*_shot.parquet` |
| -------------------------- | --- | --- |
| 增广主角 (Signal L2) | ⏳ | `content_utils.py` |
| 拉普拉斯谱 (Signal L3) | ⏳ | `content_utils.py` |
| 扰动响应 + 绑定秩 (Signal L4) | ⏳ | `content_utils.py` |
| -------------------------- | --- | --- |
| 求解器引导自改进 (贡献2) | ⏳ | `batch_score_gurobi.py` + VeRL 端改动 |
|   ├─ IIS 诊断 + 矛盾约束修复 | ⏳ | |
|   ├─ Unbounded 诊断 + 上界补全 | ⏳ | |
|   ├─ API 错误自动修正 | ⏳ | |
|   ├─ 修正样本注入 PPO | ⏳ | |
|   └─ correction_rate 监控 | ⏳ | |
