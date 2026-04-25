# Feedback Loop v2 — 评测驱动数据集扩容（Scaling Law Round 1）

你是反馈 loop v2 的主执行 agent。目标：利用现有评测结果，识别弱角度，在 wwz 服务器上构建新物体数据，传输到 68 服务器，合并为 augmented 数据集，启动训练。

**IMPORTANT**: 本机是 Windows (PowerShell)。所有服务器操作通过 SSH。注意 `\r` 换行问题。

---

## 核心设计思想

- **Scaling Law 驱动**：不修改训练策略，只通过增加新物体来扩充数据集规模
- **聚焦弱角度**：新物体只在弱角度方向贡献 train pairs
- **Golden Config Warm-Start**：从已有高质量渲染中提取 render prior，减少 VLM loop 迭代次数
- **Pinned Split**：冻结原 50 物体的 train/val/test 分配，新物体只进 train

---

## 服务器信息

| 服务器 | SSH | GPU | Python 环境 | 用途 |
|--------|-----|-----|-------------|------|
| 68 | `zhanghy56_68` | 8×H100 | `source .venv/bin/activate` (uv) | 训练 + 评测 |
| wwz | `wwz` | 3×A800 | `/home/wuwenzhuo/Qwen-VL-Series-Finetune/env/bin/python3` | 数据构建 |

- 68 工作目录 ($WORKDIR): `/gemini/platform/public/aigc/aigc_image/zhanghy56_intern/zhangqisong/data-build`
- wwz 代码根 ($WWZ_CODE): `/aaaidata/zhangqisong/data_build_runs/scene_full20_loop_20260404_code`
- wwz Blender: `/home/wuwenzhuo/blender-4.24/blender`
- wwz 必须用 tmux（screen 不可用）
- 本地是 PowerShell，服务器是 bash

---

## 关键路径

### 68 服务器
- 基线数据集: `$WORKDIR/dataset_scene_v7_full50_rotation8_trainready_front2others_splitobj_seed42_bboxmask_bright_objinfo_20260418`
- 基线 checkpoint: `$WORKDIR/DiffSynth-Studio/output/rotation8_bright_objinfo_rank32/epoch_0029/lora.safetensors`
- SpatialEdit 评测 CSV: `$WORKDIR/DiffSynth-Studio/output/eval_spatialedit_metrics/ours_objinfo_metrics.csv`
- TestSet 评测: `$WORKDIR/DiffSynth-Studio/output/eval_metrics/combined_summary.json`
- 反馈 loop 脚本（已部署）: `$WORKDIR/feedback_loop_scripts/`
- Qwen-Image-Edit-2511: `/gemini/platform/public/aigc/aigc_image/zhanghy56_intern/zhangqisong/Qwen-Image-Edit-2511`
- accelerate 配置: `$WORKDIR/DiffSynth-Studio/accelerate_config_6gpu.yaml`
- 训练脚本: `$WORKDIR/DiffSynth-Studio/train_clockwise.py`

### wwz 服务器
- 代码根: `$WWZ_CODE` = `/aaaidata/zhangqisong/data_build_runs/scene_full20_loop_20260404_code`
- evolution roots (golden config 来源):
  - `$WWZ_CODE/pipeline/data/evolution_scene_v7_full20_rotation4_agent_round0_20260404`
  - `$WWZ_CODE/pipeline/data/evolution_scene_v7_obj021_obj050_yaw000_bootstrap_20260408`
- meshes: `$WWZ_CODE/pipeline/data/meshes`
- 现有物体列表: `$WWZ_CODE/configs/seed_concepts/scene_full50_objects.json`
- prompts.json: `$WWZ_CODE/pipeline/data/prompts.json`

### 本地 repo 路径（用于同步到服务器）
- 反馈 loop 脚本: `scripts/feedback_loop/`
- bootstrap 脚本: `scripts/bootstrap_scene_yaw000_objects.py`
- rotation export: `scripts/export_rotation8_from_best_object_state.py`
- trainready builder: `scripts/build_rotation8_trainready_dataset.py`
- 物体列表: `configs/seed_concepts/scene_full50_objects.json`

---

## 现有评测指标（exp5 objinfo baseline）

SpatialEdit-Bench 传统指标（488 对）：

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ | CLIP-I ↑ | DINO ↑ | FID ↓ |
|--------|--------|--------|---------|----------|--------|-------|
| base | 15.66 | 0.6623 | 0.3304 | 0.8807 | 0.8517 | 65.47 |
| fal | 15.76 | 0.6545 | 0.3443 | 0.8747 | 0.8405 | 68.35 |
| **ours_objinfo** | **16.63** | **0.7296** | **0.2564** | **0.9050** | **0.8895** | **50.83** |

评测 CSV 格式: `image_name,psnr,ssim,lpips,dino_similarity,clip_similarity,fid`
- 角度通过 image_name 中的 `angle00`~`angle07` 提取（对应 45°~360°）

---

## 执行步骤

### Phase 1: 脚本同步（~2 min）

1. 将本地 `scripts/feedback_loop/` 下所有 `.py` 和 `.sh` 文件同步到 68 服务器 `$WORKDIR/feedback_loop_scripts/`
2. 将本地 `scripts/feedback_loop/` 下所有 `.py` 文件同步到 wwz 服务器 `$WWZ_CODE/scripts/feedback_loop/`
3. 将本地 `scripts/bootstrap_scene_yaw000_objects.py` 同步到 `$WWZ_CODE/scripts/`

```powershell
# 同步到 68
scp scripts/feedback_loop/*.py scripts/feedback_loop/*.sh zhanghy56_68:$WORKDIR/feedback_loop_scripts/
# 同步到 wwz
scp scripts/feedback_loop/*.py wwz:$WWZ_CODE/scripts/feedback_loop/
scp scripts/bootstrap_scene_yaw000_objects.py wwz:$WWZ_CODE/scripts/
```

### Phase 2: 评测分析 + 弱角度识别（~5 min）

1. SSH 到 68，创建 round 目录

```bash
ssh zhanghy56_68 "mkdir -p $WORKDIR/feedback_loop_runs/v2_scaling_r1/round_0/eval_results"
```

2. 创建 baseline eval symlinks（round 0 = 现有 exp5 结果）

```bash
ssh zhanghy56_68 "ln -sf $WORKDIR/DiffSynth-Studio/output/eval_spatialedit_metrics $WORKDIR/feedback_loop_runs/v2_scaling_r1/round_0/eval_results/spatialedit"
ssh zhanghy56_68 "ln -sf $WORKDIR/DiffSynth-Studio/output/eval_metrics $WORKDIR/feedback_loop_runs/v2_scaling_r1/round_0/eval_results/testset"
```

3. 运行 compare.py 生成 baseline compare_report（自己和自己比较以提取 per-angle metrics 和 weak angles）

```bash
ssh zhanghy56_68 "cd $WORKDIR && source .venv/bin/activate && python feedback_loop_scripts/compare.py \
  --current $WORKDIR/feedback_loop_runs/v2_scaling_r1/round_0/eval_results \
  --baseline $WORKDIR/feedback_loop_runs/v2_scaling_r1/round_0/eval_results \
  --output $WORKDIR/feedback_loop_runs/v2_scaling_r1/round_0/compare_report.json \
  --current-mode ours_objinfo \
  --baseline-mode ours_objinfo"
```

4. 读取 compare_report.json，获取 weak_angles

```bash
ssh zhanghy56_68 "cat $WORKDIR/feedback_loop_runs/v2_scaling_r1/round_0/compare_report.json"
```

5. 运行 analyze_feedback_for_dataset.py 生成 dataset_feedback_plan.json

```bash
ssh zhanghy56_68 "cd $WORKDIR && source .venv/bin/activate && python feedback_loop_scripts/analyze_feedback_for_dataset.py \
  --compare-report $WORKDIR/feedback_loop_runs/v2_scaling_r1/round_0/compare_report.json \
  --output $WORKDIR/feedback_loop_runs/v2_scaling_r1/round_1/dataset_feedback_plan.json \
  --new-object-budget 20 \
  --max-weak-angles 4 \
  --cost-cap-hours 12"
```

**注意**: compare.py 自比较 verdict 会是 `no_signal`，但我们需要的是 `weak_angles` 列表（per-angle PSNR 最低的 3 个角度）。如果 analyze 因为 "自比较没有 weak angles" 而失败（比较的 delta 都是 0），需要手动处理：
- 直接从 baseline per-angle metrics 中找 PSNR 最低的 3 个非 360° 角度作为 weak_angles
- 手工创建 dataset_feedback_plan.json

### Phase 3: Pinned Split + Render Prior（~5 min）

1. 在 68 上创建 pinned_split.json（冻结现有 50 物体的 split）

```bash
ssh zhanghy56_68 "cd $WORKDIR && source .venv/bin/activate && python feedback_loop_scripts/build_pinned_split.py \
  --dataset-root $WORKDIR/dataset_scene_v7_full50_rotation8_trainready_front2others_splitobj_seed42_bboxmask_bright_objinfo_20260418 \
  --output $WORKDIR/feedback_loop_runs/v2_scaling_r1/pinned_split.json \
  --source-note 'exp5 objinfo baseline, 50 objects seed=42 split'"
```

2. 在 wwz 上构建 render prior library（golden config）

```bash
ssh wwz "cd $WWZ_CODE && python scripts/feedback_loop/build_render_prior_library.py \
  --source-root $WWZ_CODE/pipeline/data/evolution_scene_v7_full20_rotation4_agent_round0_20260404 \
  --source-root $WWZ_CODE/pipeline/data/evolution_scene_v7_obj021_obj050_yaw000_bootstrap_20260408 \
  --objects-file $WWZ_CODE/configs/seed_concepts/scene_full50_objects.json \
  --min-hybrid-score 0.78 \
  --output $WWZ_CODE/pipeline/data/golden_config_library.json"
```

**注意**: wwz 上的 Python 需要用 `/home/wuwenzhuo/Qwen-VL-Series-Finetune/env/bin/python3`，但 feedback_loop 脚本是纯 Python（json/csv/pathlib），系统自带 python3 就够了。

### Phase 4: 生成新物体概念（~2 min）

在 wwz 上运行：

```bash
ssh wwz "cd $WWZ_CODE && python scripts/feedback_loop/build_feedback_expansion_objects.py \
  --existing-objects-file $WWZ_CODE/configs/seed_concepts/scene_full50_objects.json \
  --existing-prompts-json $WWZ_CODE/pipeline/data/prompts.json \
  --count 20 \
  --start-id-number 51 \
  --output $WWZ_CODE/configs/seed_concepts/feedback_expansion_r1_objects.json"
```

验证输出：20 个新物体，ID 从 obj_051 开始，无重复名称。

### Phase 5: 在 wwz 上执行数据构建 pipeline（~5-12 小时 GPU 任务）

这是最耗时的步骤。需要在 wwz 上依次执行：

**Stage 1: Text Expansion（生成 prompt 描述）**
- 读取 `feedback_expansion_r1_objects.json`，为每个新物体生成详细文字描述
- 输出扩展后的 prompts

**Stage 2: T2I 生成白底图**
- 用 Qwen-Image-Edit-2511 或 T2I 模型为每个新物体生成白底参考图

**Stage 2.5: SAM2 分割**
- 对白底图做 SAM2 分割获取 mask

**Stage 3: Hunyuan3D 3D 建模**
- 从白底图+mask 生成 3D mesh

**Stage 4: VLM Loop Bootstrap（yaw000 场景优化）**
- 运行 `bootstrap_scene_yaw000_objects.py` 并传入 golden config
- 使用 render prior library warm-start 减少迭代

**Stage 5: Rotation Export（导出旋转视图）**
- 只导出 yaw000 + weak angles 的旋转视图

先检查 wwz 上是否有现成的端到端 pipeline 脚本：

```bash
ssh wwz "ls $WWZ_CODE/scripts/build_scene_full50_expansion_pipeline.py 2>/dev/null && echo EXISTS || echo NOT_FOUND"
ssh wwz "ls $WWZ_CODE/scripts/build_bright_full_pipeline.sh 2>/dev/null && echo EXISTS || echo NOT_FOUND"
```

如果有端到端 pipeline 脚本，阅读理解后使用它。

如果没有或不适用，分步骤在 tmux 中启动：

```bash
# 所有 GPU 任务都在 tmux 中运行
ssh wwz "tmux new-session -d -s v2_data_build 'cd $WWZ_CODE && bash -c \"echo Phase5 started && <具体命令>\"'"
```

**重要约束**：
- Stage 1 T2I 输出必须白底
- 渲染用物体旋转（`yaw_deg`），不是相机轨道
- wwz 有 3 个 GPU (0,1,2)，可以并行处理

### Phase 5B: Bootstrap VLM Loop

如果 Stage 1-3 的产物（meshes）已经准备好，运行 bootstrap：

```bash
ssh wwz "cd $WWZ_CODE && tmux new-session -d -s v2_bootstrap 'python scripts/bootstrap_scene_yaw000_objects.py \
  --objects-file configs/seed_concepts/feedback_expansion_r1_objects.json \
  --output-root pipeline/data/evolution_v2_scaling_r1_feedback_objects_$(date +%Y%m%d) \
  --gpus 0,1,2 \
  --meshes-dir pipeline/data/meshes \
  --render-prior-library pipeline/data/golden_config_library.json \
  --max-rounds 10 \
  --plateau-window 2 \
  --plateau-eps 0.01 2>&1 | tee pipeline/data/v2_bootstrap.log'"
```

### Phase 5C: Rotation Export（只导出弱角度）

等 bootstrap 完成后，导出弱角度旋转：

```bash
# 从 dataset_feedback_plan.json 中获取 weak_target_rotations
# 假设 weak angles = [90, 180, 270]（实际由 Phase 2 的分析结果决定）
ssh wwz "cd $WWZ_CODE && python scripts/export_rotation8_from_best_object_state.py \
  --evolution-root pipeline/data/evolution_v2_scaling_r1_feedback_objects_<DATE> \
  --objects-file configs/seed_concepts/feedback_expansion_r1_objects.json \
  --output-dir pipeline/data/v2_scaling_r1_rotation_export \
  --rotations 0,<weak_angle_1>,<weak_angle_2>,<weak_angle_3>"
```

### Phase 5D: Build Trainready Dataset（wwz 上构建 pairs）

```bash
ssh wwz "cd $WWZ_CODE && python scripts/build_rotation8_trainready_dataset.py \
  --source-root pipeline/data/v2_scaling_r1_rotation_export \
  --output-dir pipeline/data/v2_scaling_r1_trainready \
  --prompts-json pipeline/data/prompts.json \
  --prompt-version v3 \
  --target-rotations <weak_angle_1>,<weak_angle_2>,<weak_angle_3>"
```

### Phase 6: 从 wwz 传输到 68（~10-30 min）

使用 tar+ssh pipe 传输（wwz 和 68 不能直接通信，需要本地中继）：

```powershell
# 先确认 wwz 上的输出目录
ssh wwz "ls pipeline/data/v2_scaling_r1_trainready/pairs/ 2>/dev/null"

# 传输 trainready 数据
ssh wwz "tar czf - -C /aaaidata/zhangqisong/data_build_runs/scene_full20_loop_20260404_code/pipeline/data/v2_scaling_r1_trainready ." | ssh zhanghy56_68 "mkdir -p /gemini/platform/public/aigc/aigc_image/zhanghy56_intern/zhangqisong/data-build/feedback_loop_runs/v2_scaling_r1/round_1/new_trainready && tar xzf - -C /gemini/platform/public/aigc/aigc_image/zhanghy56_intern/zhangqisong/data-build/feedback_loop_runs/v2_scaling_r1/round_1/new_trainready"

# 传输 views 和 objects 目录（渲染图像，较大）
ssh wwz "tar czf - -C /aaaidata/zhangqisong/data_build_runs/scene_full20_loop_20260404_code/pipeline/data/v2_scaling_r1_rotation_export views objects" | ssh zhanghy56_68 "mkdir -p /gemini/platform/public/aigc/aigc_image/zhanghy56_intern/zhangqisong/data-build/feedback_loop_runs/v2_scaling_r1/round_1/new_trainready && tar xzf - -C /gemini/platform/public/aigc/aigc_image/zhanghy56_intern/zhangqisong/data-build/feedback_loop_runs/v2_scaling_r1/round_1/new_trainready"
```

### Phase 7: 在 68 上构建 Augmented Dataset（~5 min）

```bash
ssh zhanghy56_68 "cd $WORKDIR && source .venv/bin/activate && python feedback_loop_scripts/build_augmented_rotation_dataset.py \
  --baseline-split-root $WORKDIR/dataset_scene_v7_full50_rotation8_trainready_front2others_splitobj_seed42_bboxmask_bright_objinfo_20260418 \
  --new-trainready-root $WORKDIR/feedback_loop_runs/v2_scaling_r1/round_1/new_trainready \
  --output-dir $WORKDIR/feedback_loop_runs/v2_scaling_r1/round_1/augmented_dataset \
  --pinned-split-path $WORKDIR/feedback_loop_runs/v2_scaling_r1/pinned_split.json \
  --target-rotations <weak_angle_1>,<weak_angle_2>,<weak_angle_3> \
  --asset-mode copy"
```

### Phase 8: 验证 Augmented Dataset（~1 min）

```bash
ssh zhanghy56_68 "cd $WORKDIR && source .venv/bin/activate && python feedback_loop_scripts/validate_dataset.py \
  $WORKDIR/feedback_loop_runs/v2_scaling_r1/round_1/augmented_dataset \
  --output $WORKDIR/feedback_loop_runs/v2_scaling_r1/round_1/validation_report.json \
  --pinned-split-path $WORKDIR/feedback_loop_runs/v2_scaling_r1/pinned_split.json"
```

验证要求：
- 0 errors
- train pairs 数量 = 245 (原) + 20×len(weak_angles) (新)
- val = 49, test = 56 (不变)
- 新物体只在 train 中出现

### Phase 9: 启动训练（~13 小时）

```bash
# 查看 GPU 占用
ssh zhanghy56_68 "nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv"

# 写训练脚本
ssh zhanghy56_68 "cat > $WORKDIR/feedback_loop_runs/v2_scaling_r1/round_1/train_command.sh << 'TRAINEOF'
#!/bin/bash
set -e
cd $WORKDIR/DiffSynth-Studio
source ../.venv/bin/activate

DATASET=$WORKDIR/feedback_loop_runs/v2_scaling_r1/round_1/augmented_dataset
OUTPUT=$WORKDIR/DiffSynth-Studio/output/v2_scaling_r1_augmented

accelerate launch --config_file accelerate_config_6gpu.yaml train_clockwise.py \
  --pretrained_model_name_or_path /gemini/platform/public/aigc/aigc_image/zhanghy56_intern/zhangqisong/Qwen-Image-Edit-2511 \
  --train_data_dir ${DATASET}/pairs/train_pairs.jsonl \
  --validation_data_dir ${DATASET}/pairs/val_pairs.jsonl \
  --train_image_root ${DATASET} \
  --output_dir ${OUTPUT} \
  --num_train_epochs 30 \
  --lora_rank 32 \
  --learning_rate 1e-4 \
  --mixed_precision bf16 \
  --dataloader_num_workers 4 \
  --gradient_accumulation_steps 1 \
  --save_epochs 1

echo "Training completed at $(date)"
TRAINEOF
chmod +x $WORKDIR/feedback_loop_runs/v2_scaling_r1/round_1/train_command.sh"

# 在 tmux 中启动训练
ssh zhanghy56_68 "tmux new-session -d -s v2_train 'bash $WORKDIR/feedback_loop_runs/v2_scaling_r1/round_1/train_command.sh 2>&1 | tee $WORKDIR/feedback_loop_runs/v2_scaling_r1/round_1/train.log'"
```

### Phase 10: 写训练后评测脚本（准备好，训练完成后执行）

```bash
ssh zhanghy56_68 "cat > $WORKDIR/feedback_loop_runs/v2_scaling_r1/round_1/eval_command.sh << 'EVALEOF'
#!/bin/bash
set -e
cd $WORKDIR/DiffSynth-Studio
source ../.venv/bin/activate

CHECKPOINT=$WORKDIR/DiffSynth-Studio/output/v2_scaling_r1_augmented/epoch_0029/lora.safetensors
EVAL_OUTPUT=$WORKDIR/feedback_loop_runs/v2_scaling_r1/round_1/eval_results

# SpatialEdit-Bench 评测
CUDA_VISIBLE_DEVICES=0 python eval_spatialedit_inference.py \
  --mode ours_feedback \
  --lora-path ${CHECKPOINT} \
  --output-dir ${EVAL_OUTPUT}/spatialedit

CUDA_VISIBLE_DEVICES=0 python eval_spatialedit_metrics.py \
  --pred-dir ${EVAL_OUTPUT}/spatialedit/ours_feedback \
  --output ${EVAL_OUTPUT}/spatialedit/ours_feedback_metrics.csv

echo "Eval completed at $(date)"
EVALEOF
chmod +x $WORKDIR/feedback_loop_runs/v2_scaling_r1/round_1/eval_command.sh"

# 写 compare 脚本
ssh zhanghy56_68 "cat > $WORKDIR/feedback_loop_runs/v2_scaling_r1/round_1/compare_command.sh << 'CMPEOF'
#!/bin/bash
set -e
cd $WORKDIR
source .venv/bin/activate

python feedback_loop_scripts/compare.py \
  --current $WORKDIR/feedback_loop_runs/v2_scaling_r1/round_1/eval_results \
  --baseline $WORKDIR/feedback_loop_runs/v2_scaling_r1/round_0/eval_results \
  --output $WORKDIR/feedback_loop_runs/v2_scaling_r1/round_1/compare_report.json \
  --current-mode ours_feedback \
  --baseline-mode ours_objinfo

echo "Compare completed at $(date)"
cat $WORKDIR/feedback_loop_runs/v2_scaling_r1/round_1/compare_report.json | python -m json.tool
CMPEOF
chmod +x $WORKDIR/feedback_loop_runs/v2_scaling_r1/round_1/compare_command.sh"
```

---

## 关键约束

- **Never modify existing datasets** — 创建新目录，不覆盖
- **source_image / target_image 必须指向 views/ 不是 bbox_views/**
- **Pinned Split**: 原 50 物体 seed=42 分配不变
- **Checkpoint 规则**: 统一 epoch 29
- **Python env on 68**: `source .venv/bin/activate`, deps via `uv pip install`
- **wwz 用 tmux**, 不用 screen
- **Windows → Linux SSH**: 注意 `\r` 换行问题
- **cross-server 传输必须经过本地中继**: `ssh wwz "tar..." | ssh zhanghy56_68 "tar..."`
- 训练集 7 角度（45°~315°）vs SpatialEdit 8 角度（含 360°）
- 360° 在训练集不存在，weak angles 分析时排除 360°

## 执行优先级

1. **Phase 1-4 立即执行**（脚本同步、评测分析、pinned split、golden config、新物体概念）
2. **Phase 5 评估可行性后启动**（检查 wwz GPU 是否空闲、pipeline 脚本是否就绪）
   - 如果 wwz 上已有端到端 pipeline 脚本，直接使用
   - 如果需要手动分步执行，在 tmux 中启动并记录
3. **Phase 6-10 在 Phase 5 完成后执行**（或写成脚本等待执行）

## 遇到问题时的回退策略

1. **compare.py 自比较 no_signal**: 直接从 per-angle metrics 中手动提取 weak angles（PSNR 最低的 3 个非 360° 角度），手工构建 dataset_feedback_plan.json
2. **analyze_feedback_for_dataset.py 失败（no weak angles from self-compare）**: 使用 fallback 方案 — 读取 ours_objinfo_metrics.csv，按角度分组计算平均 PSNR，取最低 3 个
3. **wwz GPU 被占用**: `nvidia-smi` 检查，如果全占用则等待或只执行 Phase 1-4
4. **Stage 1-3 pipeline 脚本不可用**: 只执行到 Phase 4，保存 golden config 和 expansion objects list，输出手动执行指南
5. **bootstrap 脚本依赖缺失**: 检查 `run_scene_agent_monitor.py` 和 `export_scene_multiview_from_pair_evolution.py` 是否在 wwz 上

## 输出报告

完成后报告以下内容：

1. **弱角度分析结果**: 哪些角度被识别为弱角度，per-angle PSNR 数据
2. **Golden Config**: render prior library 包含多少条高质量记录
3. **新物体列表**: 20 个新物体的 ID、名称、类别
4. **数据构建状态**: pipeline 各阶段完成情况，tmux session 名称
5. **传输状态**: 是否已传输到 68
6. **Augmented Dataset**: pair counts、validation 结果
7. **训练状态**: tmux session 名称、预计完成时间
8. **训练后脚本**: eval_command.sh 和 compare_command.sh 的路径
