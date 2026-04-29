# SESSION BRIEF — VLM-Feedback 3D 渲染进化系统

> **用途**：供新开的 Claude Code session 在 3 分钟内接手当前项目。阅读完本文件后，应能直接进入下一阶段实验设计。

---

## 1. 项目一句话

用 VLM（Qwen3-VL-8B-Instruct）对 Blender 渲染打分，再通过离散参数动作（灯光/材质/相机/物体/场景）迭代优化，使 image-to-3D mesh 的渲染质量尽量接近对应的 T2I 参考图。

---

## 2. 当前状态一句话

**controller 层已收敛**（v6b，2026-03-31）。10 个对象中 9 个 delta=0.0，1 个被成功改善（+0.027）。下一阶段目标：验证"上游资产质量是主瓶颈"这一假设，而非继续调整 controller。

---

## 3. v2 → v6b 极简演化时间线

| 版本 | 类型 | avg_excl_obj002 | 核心结论 |
|------|------|-----------------|---------|
| v2 | full run（10 obj）| 0.7477 | **基准线**；action 执行存在 bug，实质未改变参数 |
| v4 / v4b | full run | 0.7197 | 首次真实执行 action；compound 动作系统性有害 |
| v5 | full run | 0.7184 | compound_guard + preserve_lock_exit；改善有限 |
| v5b | **smoke test（3 obj）** | — | preserve 退出机制验证；无 full run |
| v6 | full run × 2 | ~0.6960 | 架构重写：4 维诊断、bounded search、稳定确认 |
| **v6b** | **full run（10 obj）** | **0.6936** | action space 调优（Fix A/B/C）；4 项验收全 PASS |

> **关于 avg 的口径差异**：v6b 使用 `confirmed_score = mean(p0, p1)`（去噪均值），v2 使用多次探针的最大值。直接数字比较会低估 v6b 的实际水平。详见 `SUMMARY.md §2`。

---

## 4. 已证明结论（有直接文件支撑）

- **P1 — bounded search 零退化**：v6b_full、v6b_smoke、v6_rerender 三次验证，全部 final ≥ confirmed，无例外。
- **P2 — L_WORLD_EV_UP 对 underexposed_global 有效**：v6_rerender +0.012，v6b_full +0.027，均超 eps=0.01。（仅在 obj_002 上验证，泛化能力未知）
- **P3 — L_KEY_UP 对 underexposed_global 无效**：v6_full 和 v6_rerender 均为负 delta（约 -0.003）。
- **P4 — L_RIM_UP 对 flat_no_rim 不可靠**（中等强度）：v6_rerender 中 3/4 对象负收益，1/4 低于 eps。
- **P5 — 当前动作空间无法突破 flat_no_rim 的 eps=0.01 门槛**：v6b_full 和 v6b_smoke 穷举后，4 个 flat_no_rim 对象 delta 全为 0。
- **P6 — compound 动作无帮助**：v4/v4b/v5 + v6_rerender 均无改善，已从 action space 移除。
- **P7 — 单次 VLM 测量不可靠作为 action delta 估计**（中等强度）：v6_full obj_007 单测 +0.011 被接受，v6_rerender 相同场景重测仅 +0.006（被拒）。

---

## 5. 当前最强假设（尚未直接实验验证）

- **H1（高可信）**：4 个 flat_no_rim 对象（obj_004/006/007/008）的根因是 mesh 表面曲率不足，灯光参数无法弥补。→ *待验证：测量 mesh 曲率分布*
- **H2（中等可信）**：4 个全诊断 good 低分对象（obj_001/003/005/010）受限于 mesh/材质质量，而非 Blender 参数。→ *替代解释存在*：可能是 VLM prompt 维度不完整。
- **H3（部分支持）**：VLM 对 <1% 亮度变化的辨别力不足，signal-to-noise 接近 1。→ *待验证：同一渲染 20 次重复评分*

> ⚠️ H1/H2 是目前最合理的推断，但**尚未通过直接测量验证**。不要把它们当作已证结论来设计后续实验。

---

## 6. 下一阶段实验目标

**首要任务**：验证或证伪 H1/H2（mesh 质量假设）

**优先级 1（立即可做）**：
```python
# 在服务器上用 trimesh 测量 10 个 mesh 的几何特征
import trimesh
m = trimesh.load(meshf)
# 输出：顶点数、面数、表面曲率分布、法向量方差
# 目标：flat_no_rim 对象（4/10）的曲率是否显著低于 obj_009？
```
验收：flat_no_rim 4 对象与 VLM 分类的重叠率 ≥ 70%。

**优先级 2（1 周内）**：
- 对 flat_no_rim 4 对象做 Blender mesh 细分（Subdivision Surface mod level=2），重渲染后 VLM 重新评分
- 验收：至少 1 个对象脱离 flat_no_rim 诊断，或 VLM 分数提升 ≥ 0.03

**优先级 3（中期）**：VLM 评分方差分析（同一渲染 20 次重复测量）

> 服务器：wwz | 代码目录：`/aaaidata/zhangqisong/data_build/` | Python：`/home/wuwenzhuo/Qwen-VL-Series-Finetune/env/bin/python3`

---

## 7. 必读文件（按优先级）

| 顺序 | 文件 | 目的 |
|------|------|------|
| 1 | `SESSION_BRIEF.md`（本文件）| 快速接手 |
| 2 | `results/proven_vs_hypothesis.json` | 哪些是事实，哪些是假设，哪些已被证伪 |
| 3 | `results/v6b_full_validation_results.json` | 最终权威数据，含每个对象的详细诊断 |
| 4 | `NEXT_STEPS.md` | 下一阶段实验计划（含代码模板） |
| 5 | `SUMMARY.md` | 完整演化历史（遇到具体版本问题时查阅）|

> **不推荐直接使用** `experiment_results.json`（v1 遗留文件，含已知错误，顶部有 `_status: LEGACY_DO_NOT_USE_AS_AUTHORITATIVE` 标注）
