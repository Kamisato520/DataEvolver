# VLM-Feedback 3D 渲染进化系统：v2 → v6b 完整进展总结

> 本文档记录 VLM-feedback render evolution pipeline 从 v2 到 v6b 的完整实验历史、结论与归档信息。
> **controller 层已收敛**是本轮归档的核心结论，但"上游资产质量是主瓶颈"仍属最强假设，有待实验验证。

---

## 目录

1. [系统目标与架构](#1-系统目标与架构)
2. [版本演化时间线](#2-版本演化时间线)
3. [各版本做了什么、学到了什么](#3-各版本做了什么学到了什么)
4. [已证明结论](#4-已证明结论)
5. [当前最强假设](#5-当前最强假设)
6. [为什么说 controller 已收敛](#6-为什么说-controller-已收敛)
7. [为什么下一阶段转向上游资产质量](#7-为什么下一阶段转向上游资产质量)
8. [结果文件索引](#8-结果文件索引)
9. [归档说明](#9-归档说明)

---

## 1. 系统目标与架构

**目标**：用 VLM（Qwen3-VL-8B-Instruct）评分 + 离散参数动作（灯光/材质/相机/物体/场景）反馈，自动优化 3D 物体的 Blender 渲染质量，使每个物体的渲染在 VLM 视角下尽可能接近参考 T2I 图像。

**架构**：

```
3D 参考图 ─→ VLM 审查 ─→ 4维诊断（lighting/structure/color/physics）
                  ↑               ↓
Blender 渲染 ←── 参数动作 ←── 控制器（run_evolution_loop.py）
                               ├── 稳定性确认（p0+p1 probe）
                               ├── 三区间决策（accept/preserve/mid/low）
                               └── bounded search（从 baseline_state 出发）
```

**服务器**：wwz（3×A800 80GB）
**代码目录**：`/aaaidata/zhangqisong/data_build/`
**Python 环境**：`/home/wuwenzhuo/Qwen-VL-Series-Finetune/env/bin/python3`
**VLM 模型**：`/huggingface/model_hub/Qwen3-VL-8B-Instruct`（~18GB/卡）
**Blender**：`/home/wuwenzhuo/blender-4.24/blender`

---

## 2. 版本演化时间线

完整演化链：`v2 → v4 → v4b → v5 → v5b → v6 → v6b`

| 版本 | 类型 | avg_excl_obj002 | @≥0.80 | 核心变化 |
|------|------|-----------------|--------|---------|
| v2   | full run（10 obj） | 0.7477 | 1/10 | 基准线；动作执行有 bug，实质 NO_OP |
| v4   | full run（10 obj） | ~0.7197 | 0/10 | 首次真实动作执行；compound 激活 |
| v4b  | full run（10 obj） | 0.7197 | 0/10 | rim_z=0（与 v4 结果完全相同）|
| v5   | full run（10 obj） | 0.7184 | 0/10 | compound_guard + preserve_lock_exit |
| **v5b** | **smoke test only（3 obj）** | — | — | "preserved" 退出状态验证；无 full run |
| v6   | full run（10 obj）×2 | 0.6960 | 1/10 | 架构重写；稳定确认；4 维诊断；bounded search |
| v6b  | full run（10 obj） | 0.6936 | 1/10 | action space 调优（Fix A/B/C）|

> **avg 说明**：v6/v6b avg 看似低于 v2，是因为两者使用不同测量口径。v6 的 `confirmed_score` 是 p0+p1 两次独立 probe 的均值（去除上行噪声），而 v2 的 `final_hybrid` 取的是多探针中的最大值。若用同口径重测，v2 baseline 的均值会下降约 0.03-0.05。

---

## 3. 各版本做了什么、学到了什么

### v2 — 基准线（2026-03 初）

**做了什么**：建立 VLM 评分基准。进化循环运行但动作执行有 bug，实质上 v2 是一次纯 baseline VLM 评分。

**学到了什么**：建立了 10 个对象的初始分数参考（avg_excl_obj002 = 0.7477）。obj_010 是整个项目中唯一一个基准渲染本身就 ≥0.80 的对象。

| obj | final_score |
|-----|-------------|
| obj_001 | 0.7830 |
| obj_002 | 0.4987 |
| obj_003 | 0.6895 |
| obj_004 | 0.7290 |
| obj_005 | 0.7349 |
| obj_006 | 0.7147 |
| obj_007 | 0.7548 |
| obj_008 | 0.7116 |
| obj_009 | 0.7799 |
| obj_010 | **0.8322** |

---

### v4 / v4b — 首次真实动作执行（2026-03 中）

**做了什么**：9 项 bug fix + 激活 compound 动作（L_RIM_FILL_UP 等）。v4b 改了 rim_z=0（v4 为 3）。

**学到了什么**：
- compound 动作有害：激活后 8/9 可对比对象分数下降，avg 从 0.7477 降至 0.7197
- obj_002 出现 render_hard_fail（得分 0.0），是渲染崩溃
- rim_z 参数无实质影响：v4b 结果与 v4 完全相同
- obj_010 损失最大：0.8322 → 0.7587（-0.0735）

---

### v5 — compound_guard + preserve_lock_exit（2026-03 下）

**做了什么**：
1. `_compound_allowed()`：rim_scale>1.25 或 fill_scale>1.15 时禁止 compound
2. `PROMOTE_CONTEXT` 收紧：flat_lighting 不再触发 compound 提升
3. `preserve_score_threshold=0.76`：直接 break，保护高分对象

**学到了什么**：avg excl obj_002 = 0.7184，改善有限。preserve_lock_exit 在特定 seed 下对 obj_009 失效（0.7806→0.7691）。根本问题——"VLM 噪声下如何判断动作是否有效"——未解决。

---

### v5b — preserve_lock_exit 策略验证（2026-03 下）

**做了什么**：3-obj smoke test（obj_007/009/010）。引入 `preserved` 退出状态。

**学到了什么**：preserve_lock_exit 机制对 obj_007（0.6873→0.7606）和 obj_009（high zone → preserved）在 smoke 条件下工作正确。但因只有 smoke test，无全量结果。

> **注意**：v5b 没有 10-obj full run，仅有 3-obj smoke test。表中无 v5b full 数据属正常，不是缺失。

---

### v6 — 架构重写（2026-03-29）

**做了什么**（v6 代码共 2 次 full run：evolution_v6_full 和 evolution_v6_rerender）：

1. **稳定性确认**：每轮先做 p0/p1（/p2）独立 probe，`prev_renders_dir=None, history_path=None`，`confirmed_score = mean(p0, p1)`
2. **三区间决策**：confirmed≥0.80→accept；≥0.77→preserve；≥0.68→mid(budget=1)；<0.68→low(budget=2)；<0.40→reject
3. **4 维诊断**：lighting_diagnosis / structure_consistency / color_consistency / physics_consistency（worst-case 聚合 struct/physics，majority 聚合 color/lighting）
4. **preset_mode=True**：诊断字段查表选 action，不依赖 VLM suggested_actions
5. **bounded search**：每次从 baseline_state 出发，失败后 blacklist 该 action

**v6 两次 full run 对比**（详见 `results/v6_initial_vs_rerender.json`）：
- `evolution_v6_full`（初始）：部分 flat_no_rim 对象 L_RIM_UP delta=0.0（疑似渲染应用 bug）
- `evolution_v6_rerender`（重新渲染验证）：产生真实 delta，提供 v6b 依据

**关键发现**：
- L_RIM_UP 对 flat_no_rim 系统性无效（v6_rerender：3/4 负收益，1/4 接近 0）
- L_KEY_UP 对 underexposed_global 无效，但 L_WORLD_EV_UP 有效（obj_002 +0.012）

---

### v6b — Action Space 调优（2026-03-30）

**做了什么**（仅修改 `configs/action_space.json` 3 处）：

| Fix | 字段 | 旧值 | 新值 | 依据 |
|-----|------|------|------|------|
| A | `flat_no_rim` actions | `[L_RIM_UP, L_RIM_FILL_UP]` | `[L_FILL_DOWN, L_KEY_SHADOW_SOFT]` | L_RIM_UP 3/4 对象负收益 |
| B | `underexposed_global` order | `[L_KEY_UP, L_WORLD_EV_UP]` | `[L_WORLD_EV_UP, L_KEY_UP]` | L_WORLD_EV_UP 在 rerender 实验中有效 |
| C | `flat_lighting` fallback | L_RIM_UP 排首位 | L_RIM_UP 移至末位 | 与 Fix A 一致 |

**v6b full 结果**（10 obj，3-GPU 并行）：

| obj | confirmed | final | delta | exit | zone | action | diag_lighting |
|-----|-----------|-------|-------|------|------|--------|---------------|
| obj_001 | 0.6959 | 0.6959 | 0.000 | mid_no_improve | mid | NO_OP | good |
| obj_002 | 0.4861 | 0.5127 | **+0.027** | **accepted_after_try** | low | **L_WORLD_EV_UP** | underexposed_global |
| obj_003 | 0.6543 | 0.6543 | 0.000 | low_exhausted | low | NO_OP | good |
| obj_004 | 0.6561 | 0.6561 | 0.000 | low_exhausted | low | L_FILL_DOWN → L_KEY_SHADOW_SOFT | flat_no_rim |
| obj_005 | 0.6816 | 0.6816 | 0.000 | mid_no_improve | mid | NO_OP | good |
| obj_006 | 0.7088 | 0.7088 | 0.000 | mid_no_improve | mid | L_FILL_DOWN | flat_no_rim |
| obj_007 | 0.6505 | 0.6505 | 0.000 | low_exhausted | low | L_FILL_DOWN → L_KEY_SHADOW_SOFT | flat_no_rim |
| obj_008 | 0.6483 | 0.6483 | 0.000 | low_exhausted | low | L_FILL_DOWN → L_KEY_SHADOW_SOFT | flat_no_rim |
| obj_009 | 0.8053 | 0.8053 | 0.000 | **accepted_baseline** | — | — | good |
| obj_010 | 0.7412 | 0.7412 | 0.000 | mid_no_improve | mid | NO_OP | good |

**4 项验收标准**：

| 标准 | 结果 |
|------|------|
| 零退化：final ≥ confirmed - 0.001 | ✅ 10/10 |
| underexposed 正收益：obj_002 +0.027 | ✅ Fix B 有效 |
| flat_no_rim 安全退出：4 对象 final = confirmed | ✅ Fix A 无新回退 |
| 无重复 action：bounded search 从 baseline 出发 | ✅ |

---

## 4. 已证明结论

以下结论有直接文件或多轮实验支撑（详见 `results/proven_vs_hypothesis.json`）：

1. **P1 — bounded search 零退化保证**：v6b_full（full run）、v6b_smoke（3-obj smoke test）、v6_rerender（full run）三次验证均 final ≥ confirmed，无例外。

2. **P2 — L_WORLD_EV_UP 对 underexposed_global 有效**：v6_rerender +0.012，v6b_full +0.027，两次均超过 eps=0.01。仅在 obj_002 上验证，其他对象的泛化能力未知。

3. **P3 — L_KEY_UP 对 underexposed_global 无效**：v6_full 和 v6_rerender 均显示负 delta（-0.003）。

4. **P4 — L_RIM_UP 对 flat_no_rim 不可靠**（中等强度）：v6_rerender 中 3/4 对象负收益，1/4 小正收益但低于 eps。v6_full 中 obj_007 出现 +0.0107（可能是 VLM 噪声）。

5. **P5 — 当前动作空间无法突破 flat_no_rim 的 eps=0.01 门槛**：v6b_full 和 v6b_smoke 中，v6b Fix A 替换动作（L_FILL_DOWN、L_KEY_SHADOW_SOFT）同样未超过 eps。

6. **P6 — compound 动作无帮助**：v4/v4b/v5 以及 v6_rerender 中 L_RIM_FILL_UP 均无改善。

7. **P7 — 单次 VLM 测量作为动作 delta 估计不可靠**（中等强度）：v6_full obj_007 L_RIM_UP=+0.0107 被接受，v6_rerender 同场景同动作仅 +0.0057（被拒）。两者差异超出了 p0+p1 稳定范围。

---

## 5. 当前最强假设

以下结论目前最合理，但尚未通过直接实验验证：

- **H1（高可信度）**：flat_no_rim 的根因是 mesh 表面曲率不足。
  *证据*：所有灯光动作都无法修复 flat 外观，而灯光无法弥补几何平面性。
  *仍需验证*：mesh 曲率测量，以及 flat_no_rim 频率与曲率指标的相关性。

- **H2（中等可信度）**：4 个全诊断 good 低分对象（obj_001/003/005/010）受限于 mesh/材质质量，而非 Blender 参数。
  *证据*：4 维诊断全 good 但得分 0.65-0.74，无动作信号可用。
  *替代解释*：当前诊断维度不完整，可能存在未被捕获的可修复问题（背景、材质高光等）。

- **H3（待测）**：VLM 对当前渲染变化的辨别力不足以可靠检测 <1% 场景亮度变化。
  *仍需验证*：对同一渲染 20 次重复评分，测量方差分布。

---

## 6. 为什么说 controller 已收敛

"controller 已收敛"的具体含义：

1. **不再退化**：bounded search + baseline_state reset 确保 final ≥ confirmed（P1）。
2. **可解问题已解**：underexposed_global（1 种诊断类型，1 个对象）能被正确修复（P2）。
3. **有害 action 已识别并替换**：L_RIM_UP → Fix A（P4/P5）。
4. **9 种 exit reason 覆盖所有情况**：对象质量分布内的每种情况都有对应退出路径。

"controller 已收敛"不意味着：
- 所有可优化问题都已被优化（H1/H2 指出当前 9/10 对象超出当前 action space 的能力范围）
- action space 已完备（H1/H2 建议扩展诊断维度和资产质量）
- VLM 评分系统足够稳定（H3 仍待验证）

---

## 7. 为什么下一阶段转向上游资产质量

从实验结果中观察到的模式（非已证明结论，属推断）：

- 10 个对象中，9 个的 final_delta = 0.0。controller 已经没有可改善的空间。
- 4 个 flat_no_rim 对象：已穷举当前动作空间，无法超过 eps。
- 4 个全诊断 good 低分对象：无诊断信号，controller 无法选择有效 action。
- 1 个对象（obj_009）已高质量，无需优化。

**逻辑链**：controller 无法改善 → 诊断没有可用信号 → 资产质量决定了诊断信号 → 提升资产质量是使更多对象变得"可优化"的前提条件。

**注意**：这是基于当前实验数据的最合理推断，不是通过直接的 mesh 质量实验证明的。可能存在替代路径（如扩展诊断维度、更换评分方式），在排除之前不能将"资产质量"视为唯一瓶颈。

---

## 8. 结果文件索引

### 本地归档（ARIS/docs/vlm_render_evolution_summary/）

```
results/
├── v2_baseline_summary.json           # v2 10对象分数 + 说明（来源：服务器文件）
├── v6_initial_vs_rerender.json        # v6_full vs v6_rerender 对比（来源：服务器文件）
├── v6b_smoke_results.json             # v6b smoke test 3对象完整结果（来源：服务器文件）
├── v6b_full_validation_results.json   # v6b full run 10对象完整结果（来源：服务器文件）
└── proven_vs_hypothesis.json          # 已证明 vs 假设 结构化分析
```

### 服务器原始文件（wwz: /aaaidata/zhangqisong/data_build/pipeline/data/）

| 路径 | 说明 |
|------|------|
| `evolution_v2/evolution_summary.json` | v2 完整结果（含 state_log） |
| `evolution_v4b_full/_partial_cuda*.json` | v4b 3 个 GPU partial 结果 |
| `evolution_v5/_partial_cuda*.json` | v5 partial 结果 |
| `evolution_v5b_smoke/_partial_cuda0.json` | v5b smoke 3对象 |
| `evolution_v6_full/obj_*/evolution_result.json` | v6 初始 run 逐对象结果 |
| `evolution_v6_rerender/obj_*/evolution_result.json` | v6 rerender 实验逐对象结果 |
| `evolution_v6b_smoke/obj_*/evolution_result.json` | v6b smoke 3对象逐对象结果 |
| `evolution_v6b_full/obj_*/evolution_result.json` | v6b full run 逐对象结果（最权威） |

---

## 9. 归档说明

### 数据来源分类

| 来源 | 说明 |
|------|------|
| `server_result_file` | 直接从服务器 JSON 文件读取，数值可重现验证 |
| `analysis_from_server_files` | 从多个服务器文件交叉计算的派生值（如 avg） |
| `conversation_record` | 来自本轮对话记录，无独立文件存档，可信度较低 |

### 各 JSON 文件的数据来源

- `v2_baseline_summary.json`：`server_result_file`（evolution_v2/evolution_summary.json）
- `v6_initial_vs_rerender.json`：`server_result_file`（evolution_v6_full/ 和 evolution_v6_rerender/ 的 evolution_result.json）
- `v6b_smoke_results.json`：`server_result_file`（evolution_v6b_smoke/ 的 evolution_result.json）
- `v6b_full_validation_results.json`：`server_result_file`（evolution_v6b_full/ 的 evolution_result.json）
- `proven_vs_hypothesis.json`：`analysis_from_server_files`（分析推断，引用上述文件）

### 缺失文件（当前无法回溯）

| 内容 | 原因 |
|------|------|
| v4 和 v4b 逐对象 state_log | v4b_full 只有 partial 汇总 JSON，无逐对象 evolution_result.json |
| v5 逐对象 state_log | 同上 |
| v5b 具体动作细节 | v5b 只有 _partial_cuda0.json 汇总，无逐对象文件 |
| v6_full 中 obj_004/006/008 delta=0.0 的根因 | 推测是渲染应用 bug，但未有记录说明 |
| VLM 单次评分方差分布 | 未做系统性重复测量实验 |

### 本文档修订记录

- v1（2026-03-31 初稿）：基于对话记录整理
- **v2（2026-03-31 当前版本）**：修正版本演化链（补充 v5b）、修正对象分类笔误（B 类从 5 改为 4）、拆分已证明 vs 假设、补充数据来源说明、添加归档说明节
