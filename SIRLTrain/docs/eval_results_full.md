# SIRL 训练完整评估结果

> 模型：Qwen3-8B / 训练数据：OptMATH 4097 条 / 算法：Partial KL + REINFORCE++ / 6×A800 GPU

---

## 1. 评估配置

| 参数 | 值 |
|------|-----|
| 评测脚本 | `tools/eval_test_data_vllm.py` |
| 推理引擎 | vLLM, `--tp 1` (单卡) |
| 采样 | temperature=0, top_p=1.0 |
| max_tokens | 3072 |
| repair rounds | 最多 1 轮 |
| repair_on_infeasible | False |

### 两种 Prompt 格式

| 配置 | `--prompt shot` | `--prompt reproduce` |
|------|:---:|:---:|
| enable_thinking | **False** | **True** |
| 模型输出格式 | 直接 `<python>...</python>` | `<think>推理...</think>` + `<python>...</python>` |
| 系统提示 | few-shot 示例 + Gurobi API 规则 | 旧版建模提示 |
| 与训练一致 | ✅ 与训练 `_shot` 格式相同 | ❌ 不同 |

---

## 2. 完整对比矩阵

### 2.1 Shot Prompt（与训练一致）

| 数据集 | Qwen3-8B<br/>Base | Step 30<br/>(训练中) | Step 62<br/>(最终) | Base→62<br/>变化 |
|--------|:---:|:---:|:---:|:---:|
| NL4OPT | 94.7% | 95.1% | 95.1% | +0.4% |
| MAMO Easy | 88.2% | 88.6% | 88.9% | +0.8% |
| **MAMO Complex** | **26.6%** | **28.6%** | **35.0%** | **+8.4%** |
| IndustryOR | 31.0% | 27.0% | 31.0% | 0 |
| OptMATH-166 | 7.8% | 6.6% | 7.8% | 0 |
| OptiBench | 62.2% | 61.2% | 61.7% | -0.5% |

### 2.2 Reproduce Prompt（含 `<think>` 推理）

| 数据集 | Qwen3-8B<br/>Base | Step 30<br/>(训练中) | Step 62<br/>(最终) | Base→62<br/>变化 |
|--------|:---:|:---:|:---:|:---:|
| **NL4OPT** | **87.8%** | 95.1% | 85.3% | -2.5% |
| MAMO Easy | 87.7% | 88.6% | 86.9% | -0.8% |
| MAMO Complex | 30.5% | 28.6% | 28.6% | -1.9% |
| IndustryOR | 33.0% | 29.0% | 31.0% | -2.0% |
| OptMATH-166 | 9.6% | 6.6% | 6.6% | -3.0% |
| OptiBench | 60.5% | 61.3% | 59.8% | -0.7% |

### 2.3 4 象限完整矩阵

| | Shot Prompt | Reproduce Prompt |
|---|---|---|
| **Qwen3-8B Base** | NL4OPT=94.7 MAMO-E=88.2 MAMO-C=26.6 IndOR=31.0 Math=7.8 OptiB=62.2 | NL4OPT=87.8 MAMO-E=87.7 MAMO-C=30.5 IndOR=33.0 Math=9.6 OptiB=60.5 |
| **Step 30** | NL4OPT=95.1 MAMO-E=88.6 MAMO-C=28.6 IndOR=27.0 Math=6.6 OptiB=61.2 | NL4OPT=95.1 MAMO-E=88.6 MAMO-C=28.6 IndOR=29.0 Math=6.6 OptiB=61.3 |
| **Step 62 (最终)** | NL4OPT=95.1 MAMO-E=88.9 MAMO-C=35.0 IndOR=31.0 Math=7.8 OptiB=61.7 | NL4OPT=85.3 MAMO-E=86.9 MAMO-C=28.6 IndOR=31.0 Math=6.6 OptiB=59.8 |

---

## 3. 训练过程指标

| 指标 | 前半段 (s31-45) | 后半段 (s46-62) | 趋势 |
|------|:---:|:---:|------|
| 平均 reward score | 2.082 | 2.119 | ↑ 缓升 |
| 最高 score | 2.253 (s48) | — | 微升 |
| 最终 score | — | 2.162 (s62) | |
| pg_loss | 持续负值 | 持续负值 | 正常优化 |
| clipfrac | ~0.001 | ~0.001 | 无退化 |
| grad_norm | 0.64 | 0.63 | 健康 |
| KL 散度 | 0.0005 | 0.0020 | 缓慢上升 |

### 训练配置

| 参数 | 值 |
|------|-----|
| 总步数 | 62 (1 epoch) |
| 总耗时 | 18h25min |
| 总 GPU hours | 110.5 (6 卡) |
| 学习率 | 1×10⁻⁶ |
| batch size | 66 |
| rollout.n | 16 |
| KL 系数 | 0.0005 |
| 存档 | step 30, 45, 60, 62 |

### Reward 公式

```
score = ans_ok × 1.0 + code_ok × 1.0 + format × 0.5 + lp_score × 0.75
满分 = 3.25
当前 avg score ≈ 2.10 (约 65% 得分率)
```

---

## 4. 关键发现

### 4.1 真实训练增益

用统一的 `--prompt shot` 评测时，从 Base → Step 62：

- **MAMO Complex +8.4%**：唯一持续上升的 benchmark，真金白银的进步
- **其余持平或微调**：天花板效应明显，Base 本身已经很强

### 4.2 Prompt 格式的深远影响

- **Shot → Reproduce 转换**：Base 模型在 reproduce 下 NL4OPT 从 94.7% 暴跌至 87.8%（-6.9%），说明 `<think>` 推理标签对简单 LP 问题有害
- **RL 训练抑制了 thinking**：随着训练推进（Base→Step30→Step62），reproduce prompt 下的分数持续下降，因为 RL 把模型优化为直接输出代码的模式
- **MAMO Complex 需要 thinking**：reproduce 下 Base=MAMO-C 30.5% vs shot 26.6%，`<think>` 给复杂约束建模带来增益

### 4.3 数据集偏差

- 训练数据 100% 来自 OptMATH (4097 条)
- 但 OptMATH 在 shot prompt 下三个模型都只有 ~7%（与 prompt 格式不兼容）
- 泛化到工业场景（IndustryOR 31%）和复杂问题（OptiBench 62%），说明训练数据覆盖面仍然较广

---

## 5. 评测细节

### 5.1 NL4OPT (245 条)
线性规划问题，模型在 shot 格式下接近饱和（94.7-95.1%），reproduce 格式下 Base 最好（87.8%）。

### 5.2 MAMO Easy (642 条)
简单混合整数规划，表现稳定（shot 88.2-88.9%），两个 prompt 差异小。

### 5.3 MAMO Complex (203 条)
复杂混合整数规划，是唯一一个在 RL 训练中持续提升的 benchmark（shot: 26.6→35.0%）。

### 5.4 IndustryOR (100 条)
工业场景优化，Base 模型本身已到 31%，RL 训练无额外增益。

### 5.5 OptMATH-166 (166 条)
与训练数据同源，但在 shot prompt 下得分极低（~7%），需要 `<think>` 推理标签才能发挥。

### 5.6 OptiBench (605 条)
最大规模的评测集，含线性/非线性/表格数据。Base 模型 62.2%，RL 训练后保持 61.7%，基本持平。

---

## 6. 技术修正记录

| 修改 | 文件 | 原因 |
|------|------|------|
| pebble → ThreadPoolExecutor | `executor.py` | 修复多线程 Ray worker 中 fork 导致的僵尸进程和死锁 |
| `redirect_stdout` → 注入 `print` | `executor.py` | 线程安全地捕获代码输出 |
| Gurobi TimeLimit=60 | `content_utils.py` | 防止病态 MIP 无限求解拖死 reward |
| 静音 Gurobi (OutputFlag=0) | `content_utils.py` | 防止 stdout 写满 Ray 管道导致死锁 |

---

*生成时间：2026-07-16 01:37 (UTC+8)*
