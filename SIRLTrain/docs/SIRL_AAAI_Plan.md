# SIRL: Structural-Informed RL for Optimization Modeling

## 1. 核心主张

传统的 LLM-for-OR 评估只比较最优目标值（objVal），但两个结构截然不同的模型可以产生相同的数值答案。我们主张：**ground truth 不应只是目标函数，还应包括模型的样子**——决策变量的类型、约束基数、以及可写入记忆的结构指纹。

同时，评测表明同一模型在不同 prompt 格式下分数可大幅塌方（如 OptMATH shot ≈ 7% vs reproduce 更好）。这说明策略 \(\pi(y \mid x, \text{prompt})\) **过拟合 prompt 表面形式**；成功轨迹稀缺时，失败样本没有可复用的经验。

论文围绕两个核心贡献展开：

- **贡献 1（信号层，✅ 已完成并冻结）**：结构感知 reward（`ans_ok + code_ok + format + lp_score`），用超越 objVal 的信号衡量建模质量
- **贡献 2（算法层，⏳ 主攻）**：**Solver-Guided Episodic Memory (SG-Memory)**——用冻结的结构 reward 门控写入；用求解器诊断做检索键；用记忆条件化策略 + prompt 不变性正则，降低格式脆性、提高稀疏成功下的样本效率

> **设计约束**：Reward 权重与训练 checkpoint（step 62）已固定，后续创新只改算法与推理管线，不再重训 reward。

---

## 2. 贡献 1：结构感知 Reward 体系（已冻结）

### 2.1 总公式（与当前训练一致）

```
score = ans_ok × 1.0 + code_ok × 1.0 + format × 0.5 + lp_score × 0.75
        ───────────   ───────────   ──────────   ───────────────
        数值正确性      代码可执行性     输出格式       结构忠实度
满分 = 3.25
```

该 \(R(y;x)\) 同时服务于：在线 GRPO 打分，以及 **SG-Memory 的写入门控**（见 §3）。

### 2.2 lp_score：Level 1 基数统计（✅ 已实现，主用）

| 维度 | 权重 | 容差 | 说明 |
|------|------|------|------|
| 目标类型 (Min/Max) | 0.05 | 精确匹配 | |
| 变量数 | 0.10 | max(2, 30%) | `_tolerance_score` 连续打分 |
| 约束数 | 0.10 | max(2, 30%) | 同上 |
| Binary 变量数 | 0.10 | max(0, 30%) | 一方为0则cap 0.03 |
| Integer 变量数 | 0.10 | max(0, 30%) | 一方为0则cap 0.03 |
| 二次项匹配 | 0.05 | 精确 | 布尔 |

从 Level 1 可导出结构指纹 \(s\)（变量/约束/整数/二进制计数 + 目标类型），供记忆检索使用。

### 2.3 更高阶几何信号（附录 / Future Work，不阻塞主线）

下列 Level 2–4 **不纳入主消融**，避免高 GPU 成本；若时间允许可作为附录增强：

| Level | 内容 | 状态 |
|-------|------|------|
| L2 增广主角 | \(\tilde A=[A,-b]\) 行归一化后的 SVD 主角 | ⏳ 可选 |
| L3 拉普拉斯谱 | 变量-约束二分图 \(L_{\mathrm{norm}}\) 特征值 + \(W_1\) | ⏳ 可选 |
| L4 扰动响应 / 绑定秩 | 目标扰动灵敏度 + binding 约束秩 | ⏳ 可选 |

### 2.4 GT 结构统计的离线提取

`tools/build_lp_ref_stats.py`：parquet → 子进程执行 reference 代码 → 写出 ref.lp → parse → 写入 `extra_info.lp_ref`

当前覆盖率：train `_shot` 97.6%，test `_shot` 98.8%

### 2.5 已有训练结果（支撑后续算法设计）

| 发现 | 含义 |
|------|------|
| MAMO Complex shot：Base 26.6% → Step62 **35.0%**（+8.4%） | 结构 reward + RL 在复杂题上有效 |
| OptMATH-166 shot ≈ 7.8%，reproduce 略好 | **prompt 格式脆性**是算法问题，不是再调 reward |
| RL 后 reproduce 全面下滑 | 策略过拟合 shot 表面形式；需要 format-invariant 算法 |
| IndustryOR / OptiBench 基本持平 | 仅改信号难再推天花板；需记忆复用成功轨迹 |

---

## 3. 贡献 2：Solver-Guided Episodic Memory（SG-Memory）

### 3.1 核心思想

普通 RAG / agent 记忆按文本相似取 few-shot；规则修代码再灌 PPO 又偏工程。我们提出：

> **用冻结的结构 reward 筛选可写入经验；用求解器诊断签名做检索键；策略在记忆条件化下生成，并显式正则掉 prompt 格式依赖。**

```
标准 GRPO:
  rollout → 求解器 → 标量 R → 更新；失败样本信息丢弃

SG-Memory:
  rollout → 求解器 → R + 诊断 d
    ├── R ≥ τ_write → 写入记忆 M = {(φ(x), y, s, d)}
    ├── 失败 → Retrieve(M; φ(x), s, d) → 记忆条件再生成 → 同组算 advantage
    └── 训练额外项 L_inv：同一题、不同 prompt 下策略分布对齐
```

### 3.2 形式化

#### 3.2.1 记忆条件策略

\[
\pi_\theta(y \mid x, \mathcal{M})
= \pi_\theta\big(y \mid x, \tau\big),\quad
\tau = \mathrm{Retrieve}(\mathcal{M}, \phi(x), s, d)
\]

- \(\phi(x)\)：题目**语义**嵌入（**不含** system / few-shot 包装），逼迫表示对 prompt 格式不变
- \(s\)：Level-1 结构指纹（可来自草稿解析或检索候选的历史结构）
- \(d\)：求解器诊断签名（API 错误类 / IIS 约束子集 hash / Unbounded 类型）；失败回合使用
- \(\tau\)：检索得到的 \(k\) 条情景（成功代码 + 结构摘要 + 可选诊断对照）

#### 3.2.2 Reward-filtered 写入（挂钩冻结的 \(R\)）

\[
\mathcal{M} \leftarrow \mathcal{M} \cup \{(\phi(x), y, s, d)\}
\quad\text{iff}\quad
R(y;x) \ge \tau_{\mathrm{write}}
\]

**数学点 1**：记忆是 **reward-filtered nonparametric buffer**，不是普通文档库；写入质量由贡献 1 的结构信号保证（优于仅 objVal）。

#### 3.2.3 诊断感知检索

\[
\mathrm{sim}(q, m)
= \alpha\cdot\cos(\phi_q,\phi_m)
+ \beta\cdot \mathrm{overlap}(s_q, s_m)
+ \gamma\cdot \mathbb{1}[d_q = d_m]
\]

- 成功路径：偏 \(\alpha,\beta\)（找结构相近的成功建模）
- 失败路径：抬高 \(\gamma\)（优先找「同类诊断、已修好」的轨迹）

**数学点 2**：把 IIS / 错误类型嵌入检索度量，求解器从打分器升级为**诊断索引器**。

#### 3.2.4 训练组构成（算法更新，reward 不动）

对每个问题 \(x\)，同一 GRPO 组内可含：

| 样本 | 含义 |
|------|------|
| \(y_0\) | 无记忆 rollout |
| \(y_{\mathcal{M}}\) | 记忆条件 rollout |
| \(y_{\mathrm{rep}}\)（可选） | 失败后按 \(d\) 再检索的 repair rollout |

均用冻结的 \(R\) 计算 advantage，做 Partial KL + REINFORCE++ / GRPO。

#### 3.2.5 Prompt 不变性正则（针对 OptMATH 落差）

\[
\mathcal{L}_{\mathrm{inv}}
= \mathbb{E}_{x}\Big[
D_{\mathrm{KL}}\big(
  \pi_\theta(\cdot\mid x, p_1, \mathcal{M})
  \,\big\|\,
  \pi_\theta(\cdot\mid x, p_2, \mathcal{M})
\big)
\Big]
\]

其中 \(p_1=\) shot，\(p_2=\) reproduce（或其它包装）。记忆提供共享条件 \(\tau\)，迫使「换皮 prompt」行为一致。

**数学点 3**：**format-invariant policy + memory conditioning**，直接对应已观测的 OptMATH prompt gap。

总目标示意：

\[
\mathcal{L}
= \mathcal{L}_{\mathrm{RL}}(R)
+ \lambda_{\mathrm{inv}}\,\mathcal{L}_{\mathrm{inv}}
\]

### 3.3 与普通 Agent Memory / 规则修复的区别

| 普通 Agent Memory / RAG | 规则修代码 + 灌 PPO | **SG-Memory（本文）** |
|-------------------------|---------------------|------------------------|
| 按文本相似取 few-shot | 字符串替换 / 删约束 | **语义 + 结构指纹 + 诊断** 检索 |
| 写入不筛选 | 修正成功即正样本 | **冻结结构 \(R\)** 做写入门控 |
| 多只在推理用 | 偏工程自动 patch | **训练期**与 online rollout 同组算 advantage |
| 无形式化挂钩 | 难写数学贡献 | nonparametric 方差削减 + \(\mathcal{L}_{\mathrm{inv}}\) |

### 3.4 推理期管线（可先零训练出主表）

已有 `tools/eval_test_data_vllm.py` 的 `repair_rounds` 可扩展为记忆条件生成：

```
1. 用 φ(x) 检索 M → 构造记忆增强 prompt
2. 生成 y → 求解 → 得到 R 与 d
3. 若失败且仍有预算：用 d 再检索 → 再生成（诊断进入检索，而非仅拼 traceback）
4. 高 R 轨迹可写回 M（test-time 可选）
```

**成本策略**：先做 **Memory @ test-time**（几乎零训练），再决定是否 **Memory in training + \(\mathcal{L}_{\mathrm{inv}}\)** 续训。

### 3.5 论文叙事（英文草稿）

> Outcome-only rewards and prompt-tied policies leave OR modeling brittle: the same model can collapse across prompt formats, and failed rollouts discard solver diagnostics. We keep a frozen structural reward \(R\) (cardinality-aligned \(lp\_score\)) as a quality oracle, and propose **Solver-Guided Episodic Memory**. High-reward trajectories are written into a nonparametric buffer keyed by prompt-invariant semantic embeddings, Level-1 structure fingerprints, and solver diagnostic signatures (API / IIS / unbounded). Generation is conditioned on retrieved episodes; failed attempts re-retrieve by diagnosis. A KL invariance regularizer aligns policies across prompt wrappers (e.g., shot vs. think), directly targeting observed OptMATH format gaps—without retuning the reward.

---

## 4. 实验设计

### 4.0 评测协议（所有主表共用）

**Benchmark（6 套）**：NL4OPT / MAMO-Easy / MAMO-Complex / IndustryOR / OptMATH-166 / OptiBench  

**Checkpoint**：Qwen3-8B Base；Ours-Signal = step62（冻结 \(R\)）；后续 +Mem-*  

**Prompt**：默认 **shot**（与训练一致）；另报 **reproduce** 用于 prompt_gap  

**解码**：
| 指标 | temperature | n | 说明 |
|------|:---:|:---:|------|
| pass@1 / 结构主分 | 0 | 1 | 确定性；与已有结果对齐 |
| **pass@5** | 0.6（可调） | 5 | 同题 5 条独立样本 |
| 伪 GT `lp_ref`（majority） | 同 pass@5 的 5 条 | — | 结构指纹多数票，见 §4.0.2 |

**指标定义**：

| 符号 | 定义 |
|------|------|
| pass@1 | 单次 ObjVal 相对误差 \(<10^{-6}\) 的准确率 |
| **pass@5** | 5 次采样中 **至少一次** ObjVal 正确的题目比例 |
| code_ok | 代码可执行（含 FAIL：跑通但答案错） |
| ans_ok | 与 pass@1 同判定（单样本） |
| lp_score (heuristic) | 无参考时的结构合法性分（现脚本默认） |
| **lp_score (maj-GT)** | 相对 **majority 伪 `lp_ref`** 的 L1 对比分（test_data 无真 GT 时的主结构指标） |
| lp_score (true-GT) | 仅训练/验证 parquet（有 reference 代码）可用 |
| prompt_gap | 同模型下 \(\mathrm{metric}(shot)-\mathrm{metric}(reproduce)\) |

#### 4.0.1 pass@5 与多数伪 GT 的协同（同一批 5 样本）

外部 benchmark **没有** reference 代码 → 无法跑 `build_lp_ref_stats.py` 那条真 GT 路径。约定：

```
对每题采样 y1..y5（与 pass@5 共用）
  ├── pass@5:  是否存在 i 使 ObjVal(yi) 正确
  └── 伪 lp_ref:
        对每个 yi 写 LP → 指纹 s_i = (obj_type, n_vars, n_cons, n_bin, n_int, has_quad)
        离散字段取众数；计数取中位数（或众数 bin）
        → ŝ 写入 sidecar lp_ref/{dataset}.jsonl
        之后单样本评测用 ŝ 算 lp_score (maj-GT)
```

**论文表述**：maj-GT 是 self-consistency 伪标签，不是人类建模 GT；主表同时报 pass@1 / pass@5 / lp_score(maj-GT)。真 GT 结构分放在 OptMATH train/val 或附录。

**实现注意**：伪 `lp_ref` 应用 **固定 teacher**（建议 Base 或独立采样池）生成，避免「自己给自己当 GT」虚高；主模型换 Base/Signal/Mem 时 **sidecar 不变**。

---

### 4.1 实施顺序（控 GPU）

| 阶段 | 内容 | 训练 | 产出表 |
|------|------|:----:|--------|
| **E0** | 补指标：对已有 `eval_*` 离线结构分；跑 **pass@5**；Base 造 maj-GT sidecar | 无 | 表 A / B / C 的 Base+Signal 列 |
| **P0** | 高 \(R\) 轨迹 → \(\mathcal{M}_0\) | 无 | — |
| **P1** | Mem-Test（固定 ckpt + 检索） | 无/低 | 表 A/B 的 Mem-Test 列 |
| **P2** | Mem-Train 短程续训 | 中 | 表 A 的 Mem-Train 列 |
| **P3** | + \(\mathcal{L}_{\mathrm{inv}}\) | 中 | 表 A + 表 C Full 列 |
| **Abl** | 检索键 / 写入门控 / \(L_{inv}\) | 低–中 | 表 D |

---

### 4.2 方法列（主表横轴）

| 列名 | 含义 | 状态 |
|------|------|------|
| Base | Qwen3-8B，无 RL | ✅ pass@1 已有；⏳ pass@5 / 结构 |
| Signal | step62，冻结 \(R\)，无记忆 | ✅ pass@1 已有；⏳ pass@5 / 结构 |
| Mem-Test | Signal + test-time SG-Memory | ⏳ 本篇必做 |
| Mem-Train | 记忆条件续训 | ⏳ |
| Full | Mem-Train + \(\mathcal{L}_{\mathrm{inv}}\) | ⏳ |

---

### 4.3 论文表格模板（直接填数）

#### 表 A — 主结果：pass@1 / pass@5（shot prompt）

> 单位：%；括号内为 pass@5。每一格：`pass@1 (pass@5)`。

| Benchmark | Base | Signal | Mem-Test | Mem-Train | Full |
|-----------|:----:|:------:|:--------:|:---------:|:----:|
| NL4OPT | 94.7 (—) | 95.1 (—) | | | |
| MAMO Easy | 88.2 (—) | 88.9 (—) | | | |
| MAMO Complex | 26.6 (—) | **35.0** (—) | | | |
| IndustryOR | 31.0 (—) | 31.0 (—) | | | |
| OptMATH-166 | 7.8 (—) | 7.8 (—) | | | |
| OptiBench | 62.2 (—) | 61.7 (—) | | | |
| **Macro-avg** | | | | | |

已填：现有 pass@1；`(—)` = pass@5 待测。

#### 表 B — 结构忠实度：lp_score (maj-GT) + code_ok（shot）

> 同一批 maj-GT sidecar；`lp` = mean lp_score(maj-GT)，范围与训练 L1 一致（约 0–0.5）；`code` = code_ok rate %。

| Benchmark | Base lp / code | Signal lp / code | Mem-Test lp / code | Full lp / code |
|-----------|:--------------:|:----------------:|:------------------:|:--------------:|
| NL4OPT | — / — | — / — | | |
| MAMO Easy | — / — | — / — | | |
| MAMO Complex | — / — | — / — | | |
| IndustryOR | — / — | — / — | | |
| OptMATH-166 | — / — | — / — | | |
| OptiBench | — / — | — / — | | |

可选附录列：lp_score (heuristic)，证明 maj-GT 与启发式趋势一致。

#### 表 C — Prompt 鲁棒性（prompt_gap）

> 同 checkpoint；\(\Delta = \mathrm{shot}-\mathrm{reproduce}\)（百分点）。主看 pass@1；附录可加 pass@5 / lp。

| Benchmark | Base Δpass@1 | Signal Δpass@1 | Full Δpass@1 |
|-----------|:------------:|:--------------:|:------------:|
| NL4OPT | +6.9 (94.7−87.8) | +9.8 (95.1−85.3) | |
| MAMO Complex | −3.9 (26.6−30.5) | +6.4 (35.0−28.6) | |
| OptMATH-166 | −1.8 (7.8−9.6) | +1.2 (7.8−6.6) | |
| … | | | |

预期：Full（+\(\mathcal{L}_{\mathrm{inv}}\)）使 |\Delta| 缩小，尤其 NL4OPT / OptMATH。

#### 表 D — 消融（建议在 MAMO-C + OptMATH-166 上报）

| 设置 | pass@1 | pass@5 | lp (maj-GT) |
|------|:------:|:------:|:-----------:|
| Mem-Test（全文检索键） | | | |
| − 诊断键 \(d\) | | | |
| − 结构指纹 \(s\)（仅语义） | | | |
| − \(R\) 写入门控（任意写入） | | | |
| + \(\mathcal{L}_{\mathrm{inv}}\)（→ Full） | | | |

#### 表 E — 记忆过程指标（训练/推理 log）

| 指标 | Signal | Mem-Train 初 | Mem-Train 末 | Full |
|------|:------:|:------------:|:------------:|:----:|
| memory_write_rate | — | | | |
| retrieval_hit（与 maj-GT overlap） | — | | | |
| repair_via_memory 成功率 | — | | | |
| correction / re-retrieve rate | — | | | |

---

### 4.4 已有结果如何填进表（避免重复实验）

| 来源 | 可直接填 | 仍需补 |
|------|----------|--------|
| `eval_results_full.md` | 表 A 的 Base/Signal pass@1；表 C 部分 Δ | — |
| 现有 `eval_*` 单次 output | 离线 heuristic 结构、code_ok | 非 maj-GT |
| 新跑 pass@5（K=5） | 表 A 的 pass@5；同时产出 maj-GT sidecar | **必做** |
| Mem-* | — | 表 A/B/D/E 对应列 |

---

### 4.5 预期故事线

1. **表 A**：Signal 已在 MAMO-C pass@1 +8.4%；Mem-* 主攻 OptMATH / Complex 的 pass@1 **与 pass@5**
2. **表 B**：结构分随 Mem 上升，说明不是只刷 ObjVal
3. **表 C**：Full 缩小 prompt_gap，回应格式脆性
4. **表 D**：诊断键 + \(R\) 门控必要；去掉则记忆污染或检索变噪
5. **pass@5 ↔ maj-GT**：同一管线同时服务答案上界与无真 GT 时的结构评测

---

## 5. 论文结构

```
1. Introduction
   - objVal-only 局限 + prompt 脆性
   - 贡献: frozen structural R + SG-Memory
2. Related Work
3. Method
   3.1 Structural Reward R (frozen, L1)
   3.2 SG-Memory (write / retrieve / L_inv)
   3.3 Eval protocol: pass@1, pass@5, majority pseudo-GT
4. Experiments
   4.1 Setup & protocol (§4.0)
   4.2 Main: Table A (pass@1/5) + Table B (structure)
   4.3 Robustness: Table C (prompt_gap)
   4.4 Ablations: Table D + process Table E
5. Conclusion
```

---

## 6. 当前进度 vs 待完成

| 组件 | 状态 | 位置 |
|------|------|------|
| lp_score L1 + 训练日志字段 | ✅ 冻结 | `reward_func/` |
| 训练 parquet 真 `lp_ref` | ✅ | `build_lp_ref_stats.py` → `*_shot.parquet` |
| Benchmark pass@1 矩阵 | ✅ | `docs/eval_results_full.md` |
| 评测落盘结构字段 + 离线重打 | ✅ 代码 | `structure_eval_utils.py` / `rescore_eval_structure.py` |
| -------------------------- | --- | --- |
| **pass@5 评测**（与表 A） | ⏳ | 扩展 `eval_test_data_vllm.py` |
| **majority 伪 `lp_ref` sidecar**（与表 B） | ⏳ | 新 script：5 样本 → `lp_ref/{dataset}.jsonl` |
| 用 maj-GT 重打 Base/Signal 结构表 | ⏳ | `rescore_eval_structure.py --lp_ref_dir` |
| 记忆库 \(\mathcal{M}_0\) | ⏳ | |
| Mem-Test / Mem-Train / \(\mathcal{L}_{\mathrm{inv}}\) | ⏳ | 表 A–E 对应列 |
| -------------------------- | --- | --- |
| L2–L4 几何 reward | ⏸ 附录 | |
| 规则自动改代码灌 PPO | ⏸ 降级 | |
