# 阶段工作汇报：VLM-Feedback 3D 渲染进化系统（v2 → v6b）

**汇报时间**：2026-03-31
**代码路径**：wwz 服务器 `/aaaidata/zhangqisong/data_build/`
**归档文档**：`ARIS/docs/vlm_render_evolution_summary/`

---

## 一、背景与目标

本阶段工作依托一个 3D 物体渲染质量优化系统展开。系统的完整数据流为：

1. **Stage 2**：文本描述 → T2I 生成参考图（目标外观）
2. **Stage 3**：image-to-3D 生成 mesh
3. **Stage 4**：Blender 渲染（默认参数）
4. **Stage 5**（本阶段重点）：VLM 评分 + 参数反馈 → 迭代渲染优化

**系统目标**：以 VLM（Qwen3-VL-8B-Instruct）为评分器，通过自动调节 Blender 的灯光/材质/相机等离散参数，使每个对象的渲染在 VLM 视角下尽可能接近对应的 T2I 参考图，实现"闭环自动渲染优化"。

**本阶段具体目标**：
- 构建一个**稳健、不退化**的进化控制器（controller）
- 识别当前渲染质量的**可改善项**与**不可改善项**
- 为下一阶段（上游资产质量提升）提供实验依据

---

## 二、本阶段完成的关键工作

本阶段从 2026 年 3 月初延续至 3 月末，历经 **v2 → v4 → v4b → v5 → v5b → v6 → v6b** 七个版本迭代（其中 v5b 为 smoke-only 验证版本），完成以下关键工作：

| 工作内容 | 完成状态 |
|---------|--------|
| 建立 VLM 基准评分（10 对象）| ✅ v2 |
| 修复 action 执行 bug，实现真实参数更新 | ✅ v4 |
| 识别 compound 动作系统性有害 | ✅ v4/v4b |
| 建立 baseline-first + preserve_lock_exit 策略 | ✅ v5/v5b |
| 架构重写：4 维 VLM 诊断（光照/结构/色差/物理）| ✅ v6 |
| 实现 p0+p1 稳定性确认（去噪均值替代单次读数）| ✅ v6 |
| 实现 bounded search（从 baseline_state 出发）| ✅ v6 |
| 设计并验证 9 种 exit reason 决策路径 | ✅ v6 |
| action space 调优（Fix A/B/C，基于 rerender 实验）| ✅ v6b |
| 10 对象 full run 最终验收（4 项标准全 PASS）| ✅ v6b |
| 完整归档：proven/hypothesis 分层文档 | ✅ 2026-03-31 |

---

## 三、版本演化与关键结果

### v2 — 基准线（2026-03 初）

**core**：首次建立 VLM 评分基准，但 action 执行存在 bug（实质无参数更新）。
**结果**：avg_excl_obj002 = 0.7477，@0.80 = 1/10（obj_010）。
**重要说明**：v2 高分是基准渲染的本底，**不是**"不执行 action 最优"的证据。

### v4 / v4b — 首次真实动作执行（2026-03 中）

**core**：9 项 bug fix，激活 compound 动作（L_RIM_FILL_UP 等）。
**结果**：avg_excl_obj002 = 0.7197，0/10 达到 0.80 阈值。**8/10 对象分数下降**。
**学到**：compound 动作对此数据集系统性有害；obj_002 出现渲染崩溃（score=0.0）。

### v5 / v5b — compound_guard + preserve_lock_exit（2026-03 下）

**core**：引入 compound_guard（禁止在高 scale 时触发 compound），preserve_lock_exit 保护高分对象。
**结果**：avg_excl_obj002 = 0.7184，改善有限。
**v5b（smoke only）**：验证 preserve 机制对 obj_007 和 obj_009 在小测试集上工作正确。

### v6 — 架构重写（2026-03-29）

**core 变化**（5 项）：
1. p0+p1 双 probe 稳定性确认（`confirmed_score = mean(p0, p1)`）
2. 三区间决策：accept(≥0.80) / preserve(≥0.77) / mid(≥0.68) / low(<0.68) / reject(<0.40)
3. 4 维 VLM 诊断：lighting / structure_consistency / color_consistency / physics_consistency
4. preset_mode=True：按诊断确定性选 action，不依赖 VLM suggested_actions
5. bounded search：每次从 baseline_state 出发，失败 action 加入 blacklist

**v6 两次关键实验**（均为 10-obj full run）：
- `evolution_v6_full`：发现 flat_no_rim 对象的 delta 可疑地均为 0.0（疑似渲染应用 bug）
- `evolution_v6_rerender`：重新运行产生真实 delta，提供 v6b Fix A/B 的实验依据
  - L_RIM_UP 对 flat_no_rim：3/4 对象负收益（-0.008, -0.001, -0.009）
  - L_WORLD_EV_UP 对 underexposed_global：obj_002 +0.012（有效）

### v6b — Action Space 调优（2026-03-30）

**core**：仅修改 `configs/action_space.json` 3 处（Fix A/B/C）：

| Fix | 修改内容 | 依据 |
|-----|---------|------|
| A | `flat_no_rim` 动作：`L_RIM_UP → L_FILL_DOWN, L_KEY_SHADOW_SOFT` | v6_rerender：L_RIM_UP 3/4 负收益 |
| B | `underexposed_global` 顺序：`L_WORLD_EV_UP` 移至首位 | v6_rerender：L_WORLD_EV_UP +0.012，L_KEY_UP -0.003 |
| C | `flat_lighting` fallback：`L_RIM_UP` 移至末位 | 与 Fix A 一致 |

**v6b full run 最终结果（10 对象，3-GPU 并行，2026-03-31）**：

| 对象 | confirmed | final | delta | exit | 诊断 |
|------|-----------|-------|-------|------|------|
| obj_001 | 0.6959 | 0.6959 | 0.000 | mid_no_improve | good（全） |
| **obj_002** | 0.4861 | **0.5127** | **+0.027** | accepted_after_try | underexposed_global |
| obj_003 | 0.6543 | 0.6543 | 0.000 | low_exhausted | good（全） |
| obj_004 | 0.6561 | 0.6561 | 0.000 | low_exhausted | flat_no_rim |
| obj_005 | 0.6816 | 0.6816 | 0.000 | mid_no_improve | good（全） |
| obj_006 | 0.7088 | 0.7088 | 0.000 | mid_no_improve | flat_no_rim |
| obj_007 | 0.6505 | 0.6505 | 0.000 | low_exhausted | flat_no_rim |
| obj_008 | 0.6483 | 0.6483 | 0.000 | low_exhausted | flat_no_rim |
| **obj_009** | **0.8053** | **0.8053** | 0.000 | accepted_baseline | good（全） |
| obj_010 | 0.7412 | 0.7412 | 0.000 | mid_no_improve | good（全） |

**avg_all_10 = 0.6755，avg_excl_obj002 = 0.6936，@0.80 = 1/10**

> **avg 口径说明**：v6b 使用 `confirmed_score = mean(p0, p1)`（去噪均值），而 v2 使用多次探针的最大值。二者不可直接比较。若换成相同口径，v2 baseline 均值将下降约 0.03–0.05。

**4 项验收标准均 PASS**：

| 验收标准 | 结果 |
|---------|------|
| 零退化（final ≥ confirmed − 0.001）| ✅ 10/10 |
| underexposed 正收益（obj_002 +0.027）| ✅ Fix B 有效 |
| flat_no_rim 安全退出（4 对象 final = confirmed）| ✅ Fix A 无新回退 |
| 无重复 action（bounded search 从 baseline 出发）| ✅ |

---

## 四、已解决的问题

1. **action 执行可靠性**：v4 完成 9 项 bug fix，action 执行后 Blender 参数确实变化。
2. **compound 动作危害**：识别并从主流程移除 compound 动作（D1 已证伪）。
3. **VLM 噪声导致的误判**：p0+p1 双 probe 确认机制（v6）使稳定性从"不保证"变为"可量化"，span 通常在 0.000–0.012 之间。
4. **action 退化问题**：bounded search（v6）保证 final ≥ confirmed，100% 不让对象变差（v6b_full、v6b_smoke、v6_rerender 三次验证，共 23 对象次）。
5. **局部可解问题已修复**：underexposed_global 类型的对象（obj_002）通过 L_WORLD_EV_UP 成功改善 +0.027，说明系统确实能改善可改善的问题。
6. **诊断覆盖**：从 1 维（lighting_diagnosis）扩展至 4 维（+ structure_consistency / color_consistency / physics_consistency），为更全面的诊断打下基础。

---

## 五、已证伪的假设

| 假设 | 证伪证据 |
|------|---------|
| D1：compound 动作改善光照质量 | v4/v4b/v5/v6_rerender 均显示系统性负收益 |
| D2：v2 证明"不执行 action 是最优策略" | v2 的 action 执行存在 bug，是测量基准而非策略结果；v6b obj_002 +0.027 直接反驳 |
| D3：VLM 单次高分即代表渲染真实改善 | v6_full obj_007 单测 +0.011 被接受，v6_rerender 同场景同动作仅 +0.006（被拒），二者矛盾 |

---

## 六、当前结论

### 已证明（有直接实验文件支撑）

- **bounded search 零退化保证**：final ≥ confirmed，三次验证（P1）
- **L_WORLD_EV_UP 对 underexposed_global 有效**：两次独立实验均超 eps=0.01（P2，仅在 1 个对象上验证）
- **L_KEY_UP 对 underexposed_global 无效**：两次均为负 delta（P3）
- **当前动作空间无法突破 flat_no_rim 瓶颈**：穷举全部动作后，4 个对象 max_delta < 0.01（P4/P5）
- **compound 动作无改善**：多版本实验一致（P6）
- **VLM 单次测量不可靠**：稳定确认机制是必要条件（P7，中等强度）

### 当前最强假设（尚未通过直接实验验证）

- **H1**：flat_no_rim 4 对象的根本原因是 mesh 表面曲率不足（高可信，待 mesh 几何测量验证）
- **H2**：全诊断 good 低分 4 对象（obj_001/003/005/010）受限于 mesh/材质质量（中等可信，存在替代解释）
- **H3**：VLM 对微小亮度变化的辨别力不足（部分有证据，待 20 次重复测量确认）

### 为什么说 controller 已收敛

controller 已收敛的**具体含义**：

1. 不再退化：bounded search 零退化，已证明
2. 可解问题已解：underexposed_global 类型能被正确修复，已证明
3. 有害动作已识别并替换：L_RIM_UP 对 flat_no_rim 系统性有害，已证明
4. 9 种 exit reason 覆盖所有情况：controller 对所有对象类型都有清晰的处理路径

**不意味着**：
- 所有对象都被优化（9/10 delta=0.0）
- action space 已完备（当前空间对 flat_no_rim 和全诊断 good 低分两类均无效）
- VLM 评分系统足够稳定（H3 待验证）

**结论**：继续优化 controller 的边际回报已接近于零。下一阶段的主要增量来源来自上游。

---

## 七、下一阶段计划

**核心任务**：**验证 H1/H2 假设**，而非直接投入 mesh 改善工程。

### 实验 1（优先）：Mesh 几何质量分析

**目标**：测量 10 个 mesh 的曲率分布，验证 flat_no_rim 对象（4/10）是否在几何上明显不同于其他对象。
**方法**：用 `trimesh` 计算顶点数、面数密度、表面曲率分布、法向量方差。
**验收**：flat_no_rim 4 对象与曲率低分对象重叠率 ≥ 70%。

### 实验 2（1 周内）：Mesh 细分实验

**目标**：对 flat_no_rim 4 对象应用 Blender Subdivision Surface modifier（level=2），重新渲染并 VLM 评分。
**验收**：至少 1 对象脱离 flat_no_rim 诊断，或分数提升 ≥ 0.03。

### 实验 3（中期）：VLM 评分方差标定

**目标**：对同一渲染重复 20 次 VLM 评分，测量方差分布，为"最小可信 delta"的设定提供数据依据。
**验收**：明确 improve_eps 的合理设定范围，以及是否需要增加 probe 数量。

---

## 附：关键文件索引

| 文件 | 用途 |
|------|------|
| `results/v6b_full_validation_results.json` | v6b 最终权威数据（最高优先级） |
| `results/proven_vs_hypothesis.json` | 已证明 vs 假设 vs 已证伪 分类 |
| `SUMMARY.md` | v2→v6b 完整演化历史 |
| `NEXT_STEPS.md` | 下阶段实验方案（含代码模板）|
| `results/v6_initial_vs_rerender.json` | v6b Fix A/B 的实验依据 |
