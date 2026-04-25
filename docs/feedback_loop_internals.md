# 反馈 Loop 架构 / 脚本清单 / Prompt 版本（内部参考）

> 从 CLAUDE.md 抽出，存放反馈 Loop 的架构图、脚本清单、Prompt 版本与状态持久化机制。CLAUDE.md 主体只保留链接。

## v2 闭环流程（评测驱动数据集扩容）

```
baseline eval（SpatialEdit-Bench per-angle metrics）
  ↓
compare.py → 识别弱角度（PSNR 最低 3 个非 360° 角度）
  ↓
analyze_feedback_for_dataset.py → dataset_feedback_plan.json
  ↓
wwz: 新物体 Stage 1-3 → VLM loop bootstrap → rotation export → trainready
  ↓
传输到 68（本地中继 ssh→ssh tar pipe）
  ↓
build_augmented_rotation_dataset.py → 合并 baseline + 新物体弱角度 pairs
  ↓
validate_dataset.py → 确保可训练（pinned split 校验）
  ↓
训练（68 服务器，epoch 29）
  ↓
评测（SpatialEdit-Bench 传统指标 + VIEScore）
  ↓
compare.py → verdict + strong-angle regression guard
  ↓
下一轮（增加更多物体或调整策略）
```

## v2 关键文件

| 文件 | 用途 |
|------|------|
| `scripts/feedback_loop/compare.py` | Delta + verdict + per-angle/per-object 分析 |
| `scripts/feedback_loop/analyze_feedback_for_dataset.py` | 弱角度识别 + dataset plan |
| `scripts/feedback_loop/build_pinned_split.py` | 冻结 split 生成 |
| `scripts/feedback_loop/build_feedback_expansion_objects.py` | 新物体概念生成 |
| `scripts/feedback_loop/build_render_prior_library.py` | Golden config 聚合 |
| `scripts/feedback_loop/build_augmented_rotation_dataset.py` | Augmented dataset 合并 |
| `scripts/feedback_loop/validate_dataset.py` | 数据集可训练性检查（含 pinned split） |
| `scripts/feedback_loop/V2_SCALING_TASK.md` | v2 scaling codex exec 任务 prompt |
| `scripts/feedback_loop/STAGE3_DIAGNOSE_TASK.md` | Stage 3 诊断任务 prompt |
| `scripts/feedback_loop/HINT_FOR_CODEX_20260421.md` | Stage 3 故障提示 |

## v1 关键文件（仍可用）

| 文件 | 用途 |
|------|------|
| `scripts/feedback_loop/run_round.sh` | 统一 round runner（两阶段） |
| `scripts/feedback_loop/build_plan_prompt.py` | 自动生成 plan_prompt.md |
| `scripts/feedback_loop/update_state.py` | 更新 FEEDBACK_STATE.json |
| `scripts/feedback_loop/train_feedback_round.sh` | 训练 wrapper |
| `scripts/feedback_loop/eval_feedback_round.sh` | 评测 wrapper |

## 框架验证状态

- v1 核心 4 脚本已在 68 服务器通过灰度验证（2026-04-20）
- v2 augmented dataset 合并逻辑已通过首次端到端验证（2026-04-21）：305 train / 49 val / 56 test, 0 errors
- v2 VIEScore 6 卡并行评测已通过端到端验证（2026-04-23）：488/488 pair, 0 errors

## Prompt 版本

| 版本 | 格式 | 用于 |
|------|------|------|
| v3 | `Rotate this {obj} clockwise from front view to {view}.` | exp5 baseline / v2 scaling |
| v4_dual_degree | `Rotate this {obj} clockwise from front view to {view} ({deg} degrees).` | v1 Round 1 P1 干预 |

通过 `--prompt-version v3|v4_dual_degree` 参数控制（`scripts/build_rotation8_trainready_dataset.py`）。

## 状态持久化

- 远端 `$WORKDIR/feedback_loop_runs/<experiment_id>/` 存储所有状态和产物。本地仅存轻量索引。
- 每轮数据集/训练输出不可变——新建目录，不覆盖不回滚。

## Codex 自动化调用注意事项

Codex exec 在 Windows 本地环境中存在可靠性问题：
- stdin 传参不稳定（"Reading additional input from stdin" 挂起）
- API 重连失败（"Reconnecting... 5/5"）
- codex MCP 服务可能断连

当前做法：Claude Code 直接通过 SSH 执行服务器操作，不依赖 codex exec。如需子 agent，优先使用 Claude Code 的 Agent 工具。

## 数据集有效性验证（基线参考）

### 当前最优：实验 5 Object-Info Prompt LoRA

- Prompt v3（含物体描述）、LoRA rank=32、lr=1e-4、epoch 29
- Checkpoint：`output/rotation8_bright_objinfo_rank32/epoch_0029/lora.safetensors`

**SpatialEdit-Bench 传统指标（488 对）**：

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ | CLIP-I ↑ | DINO ↑ | FID ↓ |
|--------|--------|--------|---------|----------|--------|-------|
| base | 15.66 | 0.6623 | 0.3304 | 0.8807 | 0.8517 | 65.47 |
| fal | 15.76 | 0.6545 | 0.3443 | 0.8747 | 0.8405 | 68.35 |
| **ours_objinfo** | **16.63** | **0.7296** | **0.2564** | **0.9050** | **0.8895** | **50.83** |

**SpatialEdit-Bench VIEScore（Qwen3.5-VL-8B）**：

| Method | Score_view ↑ | Score_cons ↑ | Overall ↑ |
|--------|-------------|-------------|-----------|
| base | 0.7746 | 0.9020 | 0.7415 |
| fal | 0.7234 | 0.8658 | 0.6782 |
| **ours_objinfo** | 0.7705 | **0.9709** | 0.8649 |

详细实验记录、per-angle 分析见 [`../notion-weekly/lora_experiments_full_record.md`](../notion-weekly/lora_experiments_full_record.md)。

## 远程服务器（详尽配置见 server_reference.md）

| 服务器 | SSH | GPU | 用途 |
|--------|-----|-----|------|
| 68 | `zhanghy56_68` | 8×H100 | 训练 + 评测（主力） |
| intern | `zhanghy56_intern` | 8×H100 | 备用（与 68 共享磁盘） |
| wwz | `wwz` | 3×A800 | 数据构建 |

- 68 Python 环境：`source .venv/bin/activate`，依赖用 `uv pip install`
- wwz Python 环境：`/home/wuwenzhuo/Qwen-VL-Series-Finetune/env/bin/python3`
- wwz 必须用 tmux（screen 不可用）
- 本地是 PowerShell，服务器是 bash

## 开发工作流

本地修改 → push/scp 到服务器 → 运行 → 根据反馈修改本地 → 再同步

代码全部在本地 repo 中维护：

| 类别 | 本地路径 |
|------|---------|
| 数据集构建 | `scripts/build_*.py` |
| 渲染 | `pipeline/stage*.py` |
| 训练 | `DiffSynth-Studio/train_clockwise.py/.sh` |
| 评测 | `scripts/68server/eval_*.py` |
| 反馈 Loop | `scripts/feedback_loop/` |

服务器上的代码是本地的部署副本，不在服务器上直接改代码。
