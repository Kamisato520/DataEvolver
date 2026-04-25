# Feedback Loop v1 — Round 1 P1 Dual Degree Prompt

You are the main orchestrator agent for the rotation LoRA feedback loop. Your task is to execute Round 1 of the feedback loop using the P1 (prompt template rewrite) intervention.

**IMPORTANT**: This machine is Windows (PowerShell). When running SSH commands to Linux servers, be careful with line endings. Use `-Command` with single-line strings or here-strings. Do NOT use `head -20\r` — strip carriage returns from numbers.

---

## CLAUDE.md — Project Context

保持使用中文回复、使用中文写文档，使用英文思考和搜索

### 终极目标

`/data-build <数据集描述>` → 自动完成 prompt 生成 → 3D 渲染 → VLM 质检，产出训练就绪数据集。

### 当前主线（2026-04-20）

**数据集构建是核心，LoRA 训练仅为验证手段。**

| 项目 | 服务器 | 代码目录 |
|------|--------|---------|
| **Scene-Aware 数据合成** | `wwz` | `/aaaidata/zhangqisong/data_build_runs/scene_full20_loop_20260404_code` |
| **LoRA 训练 + 评测**（验证） | `68` | `$WORKDIR/DiffSynth-Studio` |

> `$WORKDIR=/gemini/platform/public/aigc/aigc_image/zhanghy56_intern/zhangqisong/data-build`

### 反馈 Loop v1 架构

```
baseline snapshot → Plan Agent (codex exec, read-only) → INTERVENTION_PLAN.md
  → 主 agent 审阅 → Execution Agent (codex exec, limited write) → 派生新 dataset
  → validate_dataset.py → 训练 (68 server, epoch 29)
  → 评测 (Test Set + SpatialEdit) → compare.py → verdict → FEEDBACK_STATE.json → next round
```

### v1 关键文件

| 文件 | 用途 |
|------|------|
| `scripts/feedback_loop/run_round.sh` | 统一 round runner（两阶段） |
| `scripts/feedback_loop/validate_dataset.py` | 数据集可训练性检查 |
| `scripts/feedback_loop/compare.py` | 简单 delta + verdict |
| `scripts/feedback_loop/update_state.py` | 更新 FEEDBACK_STATE.json |
| `scripts/feedback_loop/train_feedback_round.sh` | 训练 wrapper |
| `scripts/feedback_loop/eval_feedback_round.sh` | 评测 wrapper |

### 数据集有效性验证：当前最优（实验 5 Object-Info Prompt LoRA）

- Prompt v3（含物体描述）、LoRA rank=32、lr=1e-4、epoch 29
- Checkpoint：`output/rotation8_bright_objinfo_rank32/epoch_0029/lora.safetensors`

SpatialEdit-Bench 传统指标（488 对）：

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ | CLIP-I ↑ | DINO ↑ | FID ↓ |
|--------|--------|--------|---------|----------|--------|-------|
| base | 15.66 | 0.6623 | 0.3304 | 0.8807 | 0.8517 | 65.47 |
| fal | 15.76 | 0.6545 | 0.3443 | 0.8747 | 0.8405 | 68.35 |
| **ours_objinfo** | **16.63** | **0.7296** | **0.2564** | **0.9050** | **0.8895** | **50.83** |

### 开发工作流

**本地修改 → push/scp 到服务器 → 运行 → 根据反馈修改本地 → 再同步**

### 远程服务器

| 服务器 | SSH | GPU | 用途 |
|--------|-----|-----|------|
| 68 | `zhanghy56_68` | 8×H100 | 训练 + 评测（主力） |
| intern | `zhanghy56_intern` | 8×H100 | 备用（与 68 共享磁盘） |
| wwz | `wwz` | 3×A800 | 数据构建 |

- 68 Python 环境：`source .venv/bin/activate`，依赖用 `uv pip install`
- 本地是 PowerShell，服务器是 bash

### 关键约束

- `source_image / target_image` 必须指向渲染原图，不是 `bbox_views/`
- 渲染用物体旋转（`stage4_scene_render.py` 的 `yaw_deg`），不用相机轨道
- `full20` frozen roots 只读
- Checkpoint 规则：所有轮次统一 epoch 29
- 训练集 7 角度（45°~315°）vs SpatialEdit 8 角度（含 360°）

---

## AGENTS.md — Dataset Pipeline & Server Details

### 数据集构建 Pipeline

高层三段：`Assets Prep（T2I + 分割 + 3D 建模） -> Blender Scene Render + VLM Refinement -> Dataset Build（配对 + 拆分 + prompt 注入）`

### Stage 9：Train-Ready / Object-Disjoint Split

- source 固定为 `yaw000 / front view`，target 是其它 7 个角度
- 50 个 object 共 350 个 pair
- split 规则：seed=42，train 35 obj / 245 pairs，val 7 obj / 49 pairs，test 8 obj / 56 pairs

关键脚本：
- `scripts/build_rotation8_trainready_dataset.py`
- `scripts/build_object_split_for_rotation_dataset.py`

Prompt v3 格式：`Rotate this {object_description} clockwise from front view to {view}.`

### 68 / intern 服务器

- SSH：`zhanghy56_68` / `zhanghy56_intern`
- 共享工作目录：`/gemini/platform/public/aigc/aigc_image/zhanghy56_intern/zhangqisong/data-build`
- Python 环境：`source .venv/bin/activate`
- 依赖安装：`uv pip install`
- Qwen-Image-Edit-2511：`/gemini/platform/public/aigc/aigc_image/zhanghy56_intern/zhangqisong/Qwen-Image-Edit-2511`

### 数据集 Lineage

```
20260410 full50 standard train-ready + object-disjoint split
  -> 20260416 full50 bright（亮度增强 + clockwise prompt）
  -> 20260418 full50 bright objinfo（物体描述 prompt v3，当前推荐训练验证入口）
```

当前推荐训练验证入口位于 `$WORKDIR` 下：
- `dataset_scene_v7_full50_rotation8_trainready_front2others_splitobj_seed42_bboxmask_bright_objinfo_20260418`

### 工作规则

1. 不要覆盖已有数据集根，任何新构建都用并列新目录
2. `rotation` 的定义是旋转物体，不是旋转相机
3. 最终 consistent 数据必须来自 `yaw000 canonical best state`
4. 当前 LoRA 主训练默认用 raw `views/`，不要误把 `bbox_views/` 当默认输入
5. 68 和 intern 共享磁盘，不要并发写同一输出路径
6. 所有远端长任务统一使用 `tmux`
7. 不要在数据集根目录内写训练日志、cache 或临时中间结果
8. `full20` frozen roots 只读，禁止原地修补

---

## Round 1 Task: P1 Dual Degree Prompt

### Goal

Build a new dataset with `v4_dual_degree` prompt template, validate it, set up training on the 68 server, and kick off training.

The prompt change adds degree numbers to instructions:
- v4: `"Rotate this {obj} clockwise from front view to {view} ({deg} degrees)."`
- v3 (current): `"Rotate this {obj} clockwise from front view to {view}."`

### Existing Baseline (exp5 objinfo)

- Dataset: `$WORKDIR/dataset_scene_v7_full50_rotation8_trainready_front2others_splitobj_seed42_bboxmask_bright_objinfo_20260418/`
  - train: 245, val: 49, test: 56 pairs. Object-disjoint, seed=42
- Checkpoint: `$WORKDIR/DiffSynth-Studio/output/rotation8_bright_objinfo_rank32/epoch_0029/lora.safetensors`
- Eval (SpatialEdit): `$WORKDIR/DiffSynth-Studio/output/eval_spatialedit_metrics/ours_objinfo_metrics.csv`
- Eval (TestSet): `$WORKDIR/DiffSynth-Studio/output/eval_metrics/combined_summary.json`
- Scripts deployed: `$WORKDIR/feedback_loop_scripts/`

### Step-by-Step Plan

#### Step 1: Set up Round 0 baseline
SSH to 68, create `$WORKDIR/feedback_loop_runs/feedback_r1_v4dual/round_0/eval_results/` and symlink existing eval CSVs.

#### Step 2: Build v4_dual_degree dataset

**Option A (preferred)**: Use existing source root (the one with manifest.json containing `objects` dict with rotation views). The baseline dataset `bboxmask_bright_objinfo_20260418` itself has a manifest.json — it IS a valid source root for `build_rotation8_trainready_dataset.py`.

```bash
# On the 68 server (via SSH):
cd $WORKDIR && source .venv/bin/activate
python feedback_loop_scripts/build_rotation8_trainready_dataset.py \
  --source-root $WORKDIR/dataset_scene_v7_full50_rotation8_trainready_front2others_splitobj_seed42_bboxmask_bright_objinfo_20260418 \
  --output-dir $WORKDIR/feedback_loop_runs/feedback_r1_v4dual/round_1/dataset_trainready \
  --prompts-json prompts.json \
  --prompt-version v4_dual_degree \
  --copy-mode symlink
```

NOTE: You need to push `build_rotation8_trainready_dataset.py` and `build_object_split_for_rotation_dataset.py` from local to `$WORKDIR/feedback_loop_scripts/` first.

Also push `prompts.json` from local `pipeline/data/prompts.json` to server.

**Option B (fallback)**: If the source root manifest structure doesn't match what `build_rotation8_trainready_dataset.py` expects, just copy the baseline dataset's JSONL files and patch the `instruction` field in-place.

#### Step 3: Apply object-disjoint split

```bash
python feedback_loop_scripts/build_object_split_for_rotation_dataset.py \
  --source-root $WORKDIR/feedback_loop_runs/feedback_r1_v4dual/round_1/dataset_trainready \
  --output-dir $WORKDIR/feedback_loop_runs/feedback_r1_v4dual/round_1/dataset \
  --seed 42 --train-objects 35
```

#### Step 4: Validate dataset

```bash
python feedback_loop_scripts/validate_dataset.py \
  $WORKDIR/feedback_loop_runs/feedback_r1_v4dual/round_1/dataset \
  --output $WORKDIR/feedback_loop_runs/feedback_r1_v4dual/round_1/validation_report.json
```

Must pass: 0 errors, train=245, val=49, test=56.

#### Step 5: Generate training command

```bash
bash feedback_loop_scripts/train_feedback_round.sh \
  --dataset-root $WORKDIR/feedback_loop_runs/feedback_r1_v4dual/round_1/dataset \
  --output-dir $WORKDIR/feedback_loop_runs/feedback_r1_v4dual/round_1/output \
  --generate-only > $WORKDIR/feedback_loop_runs/feedback_r1_v4dual/round_1/train_command.sh
chmod +x $WORKDIR/feedback_loop_runs/feedback_r1_v4dual/round_1/train_command.sh
```

#### Step 6: Kick off training (tmux)

```bash
tmux new-session -d -s r1_train \
  "bash $WORKDIR/feedback_loop_runs/feedback_r1_v4dual/round_1/train_command.sh 2>&1 | tee $WORKDIR/feedback_loop_runs/feedback_r1_v4dual/round_1/train.log"
```

Training takes ~13 hours for 30 epochs. Take epoch 29.

#### Step 7: Update state + generate post-training commands

```bash
python feedback_loop_scripts/update_state.py \
  --experiment-id feedback_r1_v4dual --round 1 \
  --run-root $WORKDIR/feedback_loop_runs/feedback_r1_v4dual/round_1 \
  --step training_started
```

Generate eval_command.sh and compare_command.sh for post-training use.

### Critical Constraints

- **Never modify existing datasets** — create new directories only
- **source_image / target_image must point to views/ not bbox_views/**
- **Object-disjoint split**: seed=42, train 35 / val 7 / test 8 objects
- **Checkpoint rule**: all rounds use epoch 29
- **Python env on 68**: `source .venv/bin/activate`, deps via `uv pip install`
- **Do NOT modify**: old datasets, existing checkpoints, pipeline render scripts
- **Windows → Linux SSH**: Be careful with line endings. Do NOT pass `\r` in numeric args.

### What to report when done

1. Dataset path and validation status
2. Training tmux session name and how to check progress
3. Post-training commands (eval + compare) saved to files on server
4. FEEDBACK_STATE.json status
