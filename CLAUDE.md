# CLAUDE.md

保持使用中文回复、使用中文写文档，使用英文思考和搜索

> 本文件是主线索引。详细内容已抽到 `docs/` 子文件，按需点击对应链接查阅。

---

## 终极目标

`/data-build <数据集描述>` → 自动完成 prompt 生成 → 3D 渲染 → VLM 质检，产出训练就绪数据集。

**数据集构建是核心，LoRA 训练仅为验证手段。**

---

## 当前主线（2026-04-23）

| 项目 | 服务器 | 代码目录 |
|------|--------|---------|
| Scene-Aware 数据合成 | `wwz` | `/aaaidata/zhangqisong/data_build_runs/scene_full20_loop_20260404_code` |
| LoRA 训练 + 评测（验证） | `68` | `$WORKDIR/DiffSynth-Studio` |

> `$WORKDIR=/gemini/platform/public/aigc/aigc_image/zhanghy56_intern/zhangqisong/data-build`

### 当前任务：GPT 成品技术报告审阅与修复（2026-04-24）

详见 [`arxiv_report/CURRENT_WORK.md`](arxiv_report/CURRENT_WORK.md) — 已完成 Top-5 修复、verdict 升级到 `accept-with-edits`，远程拉取清单 [`arxiv_report/LIST_TO_FETCH.md`](arxiv_report/LIST_TO_FETCH.md) 等待执行授权。

### 历史任务：反馈 Loop v2 — Scaling R1 已完成，准备 R2

**目标**：通过增加 20 个新物体在弱角度（90°/180°/270°）上扩充训练集，验证 scaling 是否能提升弱角度指标。

**R1 进度**：11 阶段全部完成，verdict = `inspect`。详细分阶段记录、故障复盘、隐性风险、wwz/68 路径、监控命令、执行分工见 [`docs/v2_scaling_r1_runtime_notes.md`](docs/v2_scaling_r1_runtime_notes.md)。

### R1 评测结果（SpatialEdit-Bench 488 pairs）

| 指标 | exp5 (baseline) | v2 R1 | delta |
|------|:-:|:-:|:-:|
| PSNR ↑ | 16.63 | **16.68** | +0.05 |
| SSIM ↑ | 0.7296 | **0.7310** | +0.001 |
| LPIPS ↓ | 0.2564 | **0.2546** | -0.002 |
| CLIP-I ↑ | 0.9050 | **0.9499** | **+0.045** |
| DINO ↑ | **0.8895** | 0.8837 | -0.006 |
| FID ↓ | **50.83** | 55.93 | +5.10 |
| Score_view ↑ | 0.7705 | **0.7828** | **+0.012** |
| Score_cons ↑ | **0.9709** | 0.9676 | -0.003 |
| VIE Overall ↑ | 0.8649 | **0.8703** | +0.005 |

**Verdict: `inspect`** — CLIP-I 与 Score_view 同步提升（180° +0.049、135° +0.049、315° +0.033），但 DINO 在 4 个角度退化（45°/90°/180°/270°），触发 strong-angle regression guard。根因：20 个新物体渲染质量低（hybrid_score 全部 < 0.6）。

**结论**：Scaling 方向正确（语义泛化提升），但渲染质量是瓶颈。R2 需加质量门控。

详细 per-angle 对比见 [`docs/R1_vs_exp5_spatialedit_bench.md`](docs/R1_vs_exp5_spatialedit_bench.md)。

---

## 反馈 Loop 架构

简要：baseline eval → 识别弱角度 → wwz 生成新物体 → 合并 augmented dataset → 训练 → 评测 → compare verdict → 下一轮。

详细闭环流程图、v1/v2 脚本清单、Prompt 版本、状态持久化、Codex 调用注意事项、基线实验数据、远程服务器配置、开发工作流见 [`docs/feedback_loop_internals.md`](docs/feedback_loop_internals.md)。

---

## 关键约束

- `source_image / target_image` 必须指向渲染原图，不是 `bbox_views/`
- Stage 1 T2I 输出必须白底
- 渲染用物体旋转（`stage4_scene_render.py` 的 `yaw_deg`），不用相机轨道
- `stage4_blender_render.py` 是废弃的相机轨道脚本，不要修改
- `full20` frozen roots 只读
- Checkpoint 规则：所有轮次统一 epoch 29
- 训练集 7 角度（45°~315°）vs SpatialEdit 8 角度（含 360°），360° 在训练集不存在

---

## 工作规则

- **非平凡任务**先进 plan mode，简单改动直接执行
- 本地改完代码 push/scp 到服务器再测，不在服务器上直接改
- 每次 edit 前重读文件，edit 后验证结果
- **写 markdown 文档使用 sonnet subagent**：所有 markdown 文档（`*.md`）的创建与编辑必须委托给 sonnet 模型的 subagent（`Agent` 工具，`model="sonnet"`），Opus 只负责规划、编排、决策。除非是单行/极小改动（≤ 3 行）可由 Opus 直接 Edit。

---

## 文档索引

| 文档 | 内容 |
|------|------|
| [`docs/v2_scaling_r1_runtime_notes.md`](docs/v2_scaling_r1_runtime_notes.md) | R1 详细阶段、Stage 3 故障、VLM bootstrap 报告、隐性风险、wwz/68 路径、监控命令、执行分工 |
| [`docs/R1_vs_exp5_spatialedit_bench.md`](docs/R1_vs_exp5_spatialedit_bench.md) | R1 vs exp5 在 SpatialEdit-Bench 下的 Overall + per-angle 对比 |
| [`docs/feedback_loop_internals.md`](docs/feedback_loop_internals.md) | 反馈 Loop 架构图、v1/v2 脚本清单、Prompt 版本、远程服务器、开发工作流 |
| [`docs/FEEDBACK_LOOP_FRAMEWORK.md`](docs/FEEDBACK_LOOP_FRAMEWORK.md) | 反馈 Loop v2 框架详细说明（接手文档） |
| [`docs/HANDOVER_20260421.md`](docs/HANDOVER_20260421.md) | 接手文档（服务器路径、数据集、实验历史） |
| [`docs/PPT_OUTLINE_FEEDBACK_LOOP.md`](docs/PPT_OUTLINE_FEEDBACK_LOOP.md) | 6 页汇报 PPT 大纲 |
| [`docs/server_reference.md`](docs/server_reference.md) | 服务器配置、脚本列表、路径详细 |
| [`PLAN.md`](PLAN.md) | 反馈 Loop v2 设计文档 |
| [`scripts/feedback_loop/V2_SCALING_TASK.md`](scripts/feedback_loop/V2_SCALING_TASK.md) | v2 Scaling R1 执行步骤 |
| [`notion-weekly/lora_experiments_full_record.md`](notion-weekly/lora_experiments_full_record.md) | 全部实验记录（exp1-5）、per-angle 分析 |
| [`arxiv_report/assets/pipeline_figures/PIPELINE_ARCHITECTURE.md`](arxiv_report/assets/pipeline_figures/PIPELINE_ARCHITECTURE.md) | 框架图结构化描述（3 张图：Pipeline Overview / Inner Loop / Outer Loop） |
| 周报 | [week1](notion-weekly/week1_0325-0403.md) / [week2](notion-weekly/week2_0404-0407.md) / [week3](notion-weekly/week3_0408-0414.md) / [week5](notion-weekly/week5_0421-0425.md) |
