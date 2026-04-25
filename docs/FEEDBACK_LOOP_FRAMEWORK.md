# 评测驱动数据集扩容框架（Feedback Loop v2）

**日期**：2026-04-22
**状态**：v2 Scaling R1 首轮运行中（训练阶段）

---

## 一、框架目标

**核心思想**：不修改训练策略，仅通过评测反馈来识别弱角度，自动扩充对应方向的训练数据，验证 Scaling Law 是否能持续提升指标。

**一句话概括**：评测 → 找弱点 → 造数据 → 再训练 → 再评测 → 闭环迭代。

### 1.1 背景

我们训练 LoRA 使 Qwen-Image-Edit-2511 实现单轴旋转编辑（azimuth 45°~315°，7 个角度）。实验发现不同角度的 PSNR 差异显著：

| 类型 | 角度 | PSNR（baseline） |
|------|------|------------------|
| 弱角度 | 270° | 15.73 |
| 弱角度 | 180° | 16.08 |
| 弱角度 | 90° | 16.40 |
| 强角度 | 45° / 135° / 225° / 315° | 16.8~17.2 |

框架的目标是：**自动识别弱角度 → 在弱角度方向增加新物体数据 → 训练后验证弱角度是否提升，同时强角度不退化**。

### 1.2 设计约束

- **Pinned Split**：原 50 物体的 train/val/test 分配冻结不变（seed=42），新物体只进 train
- **Checkpoint 规则**：所有轮次统一取 epoch 29
- **数据集不可变**：每轮新建目录，不覆盖不回滚
- **不修改训练代码**：纯数据驱动，只增加训练 pairs

---

## 二、闭环流程

```
┌─────────────────────────────────────────────────────────────────┐
│                    68 服务器（训练 + 评测）                        │
│                                                                  │
│  ① Baseline Eval                                                │
│     SpatialEdit-Bench per-angle metrics CSV                     │
│           │                                                      │
│  ② compare.py                                                   │
│     识别弱角度（PSNR 最低 3 个非 360° 角度）                        │
│           │                                                      │
│  ③ analyze_feedback_for_dataset.py                              │
│     生成 dataset_feedback_plan.json                              │
│     （弱角度列表 + 新物体预算 + 时间上限）                           │
│           │                                                      │
│  ④ build_feedback_expansion_objects.py                          │
│     生成新物体概念列表                                             │
│           │                                                      │
└───────────┼──────────────────────────────────────────────────────┘
            │ 新物体概念 JSON
            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    wwz 服务器（数据构建）                          │
│                                                                  │
│  ⑤ Stage 1: Text Expansion                                     │
│     种子概念 → 详细文字描述 prompt                                 │
│           │                                                      │
│  ⑥ Stage 2: T2I Generate                                       │
│     prompt → 白底参考图 (1024×1024 PNG)                           │
│           │                                                      │
│  ⑦ Stage 2.5: SAM3 Segment                                     │
│     白底图 → RGBA mask（透明背景）                                 │
│           │                                                      │
│  ⑧ Stage 3: Hunyuan3D                                          │
│     白底图 + mask → 3D mesh（GLB with PBR texture）              │
│     ⚠️ 必须单 GPU 或验证过的多 GPU 运行                            │
│           │                                                      │
│  ⑨ VLM Loop Bootstrap                                          │
│     mesh → Blender 渲染 → VLM 评审 → 调参 → 再渲染               │
│     使用 Golden Config Prior 做 warm-start                       │
│     输出：每个物体的最佳 control_state                             │
│           │                                                      │
│  ⑩ Rotation Export                                              │
│     最佳 state → 多角度 Blender 渲染                              │
│     只导出 yaw000 + 弱角度 (90°/180°/270°)                       │
│           │                                                      │
│  ⑪ Trainready Build                                            │
│     渲染图 → JSONL pairs + views/ 目录结构                        │
│                                                                  │
└───────────┼──────────────────────────────────────────────────────┘
            │ tar+ssh 本地中继传输
            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    68 服务器（合并 + 训练）                        │
│                                                                  │
│  ⑫ build_augmented_rotation_dataset.py                          │
│     baseline 245 pairs + 新物体 60 pairs = 305 pairs            │
│     val/test 保持不变（pinned split 校验）                        │
│           │                                                      │
│  ⑬ validate_dataset.py                                         │
│     检查：JSONL 完整性 / 图像存在 / split 无泄漏 / pair 数量       │
│           │                                                      │
│  ⑭ 训练                                                        │
│     accelerate launch (6×H100) → 30 epoch → 取 epoch 29         │
│           │                                                      │
│  ⑮ 评测                                                        │
│     SpatialEdit-Bench 传统指标 (PSNR/SSIM/LPIPS/CLIP-I/DINO)    │
│           │                                                      │
│  ⑯ compare.py                                                   │
│     当前 vs baseline → verdict:                                  │
│     continue / inspect / stop_or_revert / no_signal             │
│     + strong-angle regression guard                              │
│           │                                                      │
│     如果 continue → 回到 ① 开始下一轮                             │
│     如果 stop_or_revert → 排查原因，调整策略                       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 三、各步骤详细说明

### 3.1 弱角度识别（compare.py）

- 输入：per-pair metrics CSV（SpatialEdit-Bench 评测输出）
- 逻辑：按角度分组计算平均 PSNR，排除 360°，取最低的 3 个角度
- 输出：`compare_report.json`，包含 `weak_angles`、`strong_angles`、`overall_delta`、`per_angle_delta`、`verdict`

**verdict 决策逻辑**：
| 条件 | verdict |
|------|---------|
| 有指标超过退化阈值 | `stop_or_revert` |
| 有改善但同时有退化 | `inspect` |
| 有改善且无退化 | `continue` |
| 无显著变化 | `no_signal` |

### 3.2 数据集计划生成（analyze_feedback_for_dataset.py）

- 输入：`compare_report.json`
- 输出：`dataset_feedback_plan.json`
- 内容：弱角度列表、新物体预算数量、时间上限、每角度预期新增 pair 数

### 3.3 新物体概念生成（build_feedback_expansion_objects.py）

- 输入：现有 50 物体列表 + prompts.json
- 输出：`feedback_expansion_r1_objects.json`（20 个新物体定义）
- 规则：ID 从 obj_051 开始，名称不与现有物体重复

### 3.4 数据构建 Pipeline（wwz 服务器，Stage 1-3）

**Stage 1** (`stage1_text_expansion.py`)：种子概念 → 详细文字描述
**Stage 2** (`stage2_t2i_generate.py`)：文字 → T2I 白底图（必须白底）
**Stage 2.5** (`stage2_5_sam2_segment.py`)：白底图 → SAM3 mask（文件名含 sam2 但内部用 SAM3）
**Stage 3** (`stage3_image_to_3d.py`)：白底图 + mask → Hunyuan3D 3D mesh

**⚠️ Stage 3 已知问题**：Hunyuan3D 的 Paint 管线依赖 CUDA extension（custom_rasterizer / DifferentiableRenderer），多 GPU 并行时可能在非 GPU0 上静默失败。建议单 GPU 运行或先做 smoke test。

### 3.5 VLM Loop Bootstrap（bootstrap_scene_yaw000_objects.py）

- 多 GPU 并行（3×A800），每个 GPU 分配一批物体
- 使用 Golden Config Prior（`golden_config_library.json`，58 条高质量渲染配置的中位数聚合）做 warm-start
- 每轮：Blender 渲染 → VLM (Qwen3.5-VL) 评审 → agent 决策调参 → 再渲染
- 收敛条件：`plateau_window=2`, `plateau_eps=0.01`（连续 2 轮 hybrid_score 提升 < 0.01 则停止）
- 最大轮数：10
- 输出：每个物体的最佳 `control_state`（lighting / camera / object / scene / material 参数）

### 3.6 Rotation Export（export_rotation8_from_best_object_state.py）

- 读取 bootstrap 的最佳 control_state
- 用 Blender 按指定角度列表渲染：`--rotations 0,90,180,270`
- 3 GPU 并行，每个物体 4 张渲染 + 4 张 mask = 8 张 PNG
- 输出：`objects/obj_XXX/yaw{000,090,180,270}.png` + `_mask.png` + `_control.json` + `_render_metadata.json`

### 3.7 Trainready Build（build_rotation8_trainready_dataset.py）

- 将 export 的 `objects/` 结构重组为训练用 `views/` + `pairs/` 结构
- 生成 JSONL pairs：每个物体 × 每个弱角度 = 1 pair（source=yaw000, target=yaw{angle}）
- 支持 `--prompts-json` 加载物体描述（v3 prompt 格式）
- 输出：`pairs/train_pairs.jsonl` + `views/obj_XXX/` + `objects/obj_XXX/`

### 3.8 Augmented Dataset（build_augmented_rotation_dataset.py）

- 合并 baseline split（245 train / 49 val / 56 test）+ 新物体弱角度 pairs（60 train）
- Pinned split 校验：确保原 50 物体的分配不变
- 新物体只进 train，不影响 val/test
- 资产复制模式（`--asset-mode copy`）：复制渲染图到新目录

### 3.9 验证（validate_dataset.py）

检查项：
1. JSONL 文件存在且可解析
2. source_image / target_image 路径存在
3. 图像不指向 bbox_views/
4. Object-disjoint split 无泄漏
5. 必要字段完整（instruction, target_rotation_deg）
6. Pinned split 一致性（如提供）
7. Pair 数量与预期匹配

### 3.10 训练 + 评测 + 比较

- 训练：`accelerate launch` (6×H100)，30 epoch，LoRA rank=32，lr=1e-4
- 评测：`eval_spatialedit_inference.py` → `eval_spatialedit_metrics.py`
- 比较：`compare.py --current round_1/eval_results --baseline round_0/eval_results`

---

## 四、wwz 服务器文件结构

```
$WWZ_CODE = /aaaidata/zhangqisong/data_build_runs/scene_full20_loop_20260404_code

$WWZ_CODE/
├── configs/
│   ├── seed_concepts/
│   │   ├── scene_full50_objects.json          # 原 50 物体列表
│   │   ├── scene_full20_objects.json          # 前 20 物体
│   │   ├── scene_obj021_050_objects.json      # 后 30 物体
│   │   └── feedback_expansion_r1_objects.json # ★ v2 R1 新增 20 物体 (obj_051~070)
│   ├── scene_template.json                    # Blender 场景模板（光照/相机/材质参数）
│   ├── scene_action_space.json                # VLM agent 可用动作空间
│   ├── vlm_review_schema.json                 # VLM 评审输出 schema
│   └── dataset_profiles/
│       └── scene_v7_full50_loop.json          # 数据集构建 profile
│
├── pipeline/
│   ├── stage1_text_expansion.py               # Stage 1: 种子概念 → 详细 prompt
│   ├── stage2_t2i_generate.py                 # Stage 2: prompt → 白底参考图
│   ├── stage2_5_sam2_segment.py               # Stage 2.5: 白底图 → SAM3 mask
│   ├── stage3_image_to_3d.py                  # Stage 3: 白底图+mask → Hunyuan3D mesh
│   ├── stage3_5_mesh_sanitize.py              # Stage 3.5: mesh 清理
│   ├── stage4_scene_render.py                 # Stage 4: Blender 场景渲染（当前使用）
│   ├── stage4_blender_render.py               # ⚠️ 废弃脚本，不要修改
│   ├── stage5_5_vlm_review.py                 # VLM 评审（Qwen3.5-VL via API）
│   ├── stage5_6_feedback_apply.py             # 反馈应用（agent 调参逻辑）
│   └── data/                                  # ★ 所有数据产物
│       ├── prompts.json                       # 原 50 物体描述
│       ├── golden_config_library.json         # ★ 58 条高质量渲染 prior
│       ├── meshes/                            # 原 50 物体 3D mesh
│       │
│       │  ===== v2 Scaling R1 产物 =====
│       │
│       ├── dataset_v2_scaling_r1_feedback_stage1_assets_20260421/
│       │   ├── prompts.json                   # 新 20 物体描述
│       │   ├── images/                        # Stage 2 白底图
│       │   ├── images_rgba/                   # Stage 2.5 RGBA mask
│       │   ├── meshes/                        # Stage 3 GLB（修复后，20/20 有贴图）
│       │   ├── meshes_raw/                    # Stage 3 原始输出（含 GPU1/2 失败的）
│       │   └── summary.json
│       │
│       ├── evolution_v2_scaling_r1_feedback_objects_20260421_v2/
│       │   ├── bootstrap_request.json         # bootstrap 请求参数
│       │   ├── _logs/                         # worker 日志
│       │   │   ├── worker_gpu_0.json
│       │   │   ├── worker_gpu_0_manifest.json
│       │   │   ├── worker_gpu_1.json
│       │   │   ├── worker_gpu_1_manifest.json
│       │   │   ├── worker_gpu_2.json
│       │   │   └── worker_gpu_2_manifest.json
│       │   └── obj_XXX_yaw000/               # 每个物体的 VLM loop 产物
│       │       ├── render_prior_selection.json # golden config 选择记录
│       │       ├── sharpness_history.json     # 清晰度追踪
│       │       ├── states/                    # 每轮 control_state
│       │       │   ├── round00_render_prior_input.json
│       │       │   ├── round00.json
│       │       │   ├── round01.json
│       │       │   └── ...
│       │       ├── reviews/                   # VLM 评审结果
│       │       │   ├── obj_XXX_r00_agg.json   # 聚合评分（hybrid_score）
│       │       │   ├── obj_XXX_r00_az000_el+00.json  # 单视角评分
│       │       │   └── obj_XXX_r00_az000_el+00_trace.json  # 决策 trace
│       │       ├── decisions/                 # agent 决策记录
│       │       ├── round00_renders/           # 每轮渲染结果
│       │       └── agent_round00.json         # agent 动作日志
│       │
│       ├── v2_scaling_r1_rotation_export_20260421_v2/
│       │   ├── manifest.json                  # 导出汇总（success=true, 80 renders）
│       │   ├── export_request.json            # 导出请求参数
│       │   ├── scene_template_fixed_camera.json
│       │   ├── objects/                       # 每个物体的多角度渲染
│       │   │   └── obj_XXX/
│       │   │       ├── yaw000.png             # 正面渲染（source）
│       │   │       ├── yaw000_mask.png        # 正面 mask
│       │   │       ├── yaw000_control.json    # 渲染参数
│       │   │       ├── yaw000_render_metadata.json
│       │   │       ├── yaw090.png             # 90° 旋转渲染
│       │   │       ├── yaw180.png             # 180° 旋转渲染
│       │   │       └── yaw270.png             # 270° 旋转渲染
│       │   ├── _shards/                       # GPU 分片临时目录
│       │   └── _logs/                         # worker 日志
���       │
│       ├── v2_scaling_r1_trainready_20260421_v2/
│       │   ├── manifest.json                  # 构建汇总（60 pairs）
│       │   ├── summary.json
│       │   ├── pairs/
│       │   │   ├── train_pairs.jsonl          # ★ 60 条训练 pair
│       │   │   └── train_pairs.csv            # CSV 格式副本
│       │   ├── views/                         # 训练用渲染图（按物体分目录）
│       │   │   └── obj_XXX/
│       │   │       ├── yaw000.png
│       │   │       ├── yaw090.png
│       │   │       └── ...
│       │   └── objects/                       # 原始渲染（含 metadata）
│       │
│       └── v2_scaling_r1_logs_20260421/       # 所有日志
│           ├── bootstrap_v2.log
│           ├── export_v2.log
│           ├── trainready_v2.log
│           ├── stage1_3.log
│           └── stage3_repaint.log
│
├── scripts/
│   ├── bootstrap_scene_yaw000_objects.py      # ★ VLM loop bootstrap 主脚本
│   ├── export_rotation8_from_best_object_state.py  # ★ 多角度旋转导出
│   ├── build_rotation8_trainready_dataset.py  # ★ trainready 数据集构建
│   ├── build_scene_assets_from_stage1.py      # Stage 1-3 端到端
│   ├── feedback_loop/                         # ★ 反馈 loop 脚本（见下方详细说明）
│   └── run_scene_agent_monitor.py             # VLM agent 监控
│
└── 外部依赖
    ├── /home/wuwenzhuo/blender-4.24/blender   # Blender 渲染引擎
    ├── /huggingface/model_hub/sam3/sam3.pt     # SAM3 模型
    ├── /aaaidata/zhangqisong/data_build/sam3   # SAM3 库
    └── Hunyuan3D                               # 3D 重建（pip 安装）
```

### 4.1 反馈 Loop 脚本详细说明

```
$WWZ_CODE/scripts/feedback_loop/

分析与规划（在 68 服务器运行）：
├── compare.py                          # ★ 评测对比 + verdict
│   输入: --current <eval_dir> --baseline <eval_dir>
│   输出: compare_report.json (overall_delta, per_angle_delta, weak_angles, verdict)
│
├── analyze_feedback_for_dataset.py     # 弱角度 → 数据集扩容计划
│   输入: --compare-report <json> --new-object-budget 20
│   输出: dataset_feedback_plan.json
│
├── build_pinned_split.py               # 冻结现有物体的 train/val/test 分配
│   输入: --dataset-root <baseline_dataset>
│   输出: pinned_split.json
│
├── build_feedback_expansion_objects.py  # 生成新物体概念
│   输入: --existing-objects-file <json> --count 20 --start-id-number 51
│   输出: feedback_expansion_r1_objects.json

构建（在 wwz 服务器运行）：
├── build_render_prior_library.py       # 从已有高质量渲染聚合 golden config
│   输入: --source-root <evolution_dirs> --min-hybrid-score 0.78
│   输出: golden_config_library.json
│
├── build_augmented_rotation_dataset.py # ★ 合并 baseline + 新物体 pairs
│   输入: --baseline-split-root <dir> --new-trainready-root <dir> --pinned-split-path <json>
│   输出: augmented_dataset/ (merged train/val/test JSONL + views + objects)

验证（在 68 服务器运行）：
├── validate_dataset.py                 # ★ 数据集可训练性检查
│   输入: <dataset_root> --pinned-split-path <json>
│   输出: validation_report.json (status: passed/failed, errors, warnings)

Pipeline 脚本（在 wwz 服务器运行）：
├── run_v2_scaling_r1_pipeline.sh       # v2 R1 全流程 pipeline（包含 bootstrap → export → trainready）

状态管理（v1 遗留，仍可用）：
├── build_plan_prompt.py                # 自动生成 plan agent prompt
├── update_state.py                     # 更新 FEEDBACK_STATE.json
└── run_round.sh / run_plan_agent.sh / run_exec_agent.sh  # v1 round runner（Codex agent 调用）
```

---

## 五、68 服务器文件结构

```
$WORKDIR = /gemini/platform/public/aigc/aigc_image/zhanghy56_intern/zhangqisong/data-build

$WORKDIR/
├── feedback_loop_scripts/                     # ★ 已部署的反馈 loop 脚本（从本地 scp）
│   ├── compare.py
│   ├── analyze_feedback_for_dataset.py
│   ├── build_pinned_split.py
│   ├── build_feedback_expansion_objects.py
│   ├── build_augmented_rotation_dataset.py
│   ├── validate_dataset.py
│   ├── build_plan_prompt.py
│   ├── update_state.py
│   ├── run_round.sh
│   ├── run_plan_agent.sh / run_exec_agent.sh
│   ├── train_feedback_round.sh
│   └── eval_feedback_round.sh
│
├── feedback_loop_runs/
│   └── v2_scaling_r1/                         # ★ v2 Scaling R1 运行产物
│       ├── pinned_split.json                  # 冻结的 50 物体 split
│       ├── round_0/
│       │   ├── eval_results/ → symlink        # baseline eval 结果
│       │   └── compare_report.json            # baseline 自比较（提取 weak_angles）
│       └── round_1/
│           ├── dataset_feedback_plan.json      # 弱角度=[90°,180°,270°], budget=20
│           ├── new_trainready/                 # wwz 传输过来的新物体数据
│           │   ├── pairs/train_pairs.jsonl     # 60 pairs
│           │   ├── views/obj_051~070/
│           │   └── objects/obj_051~070/
│           ├── augmented_dataset/              # ★ 合并后的训练数据集
│           │   ├── pairs/
│           │   │   ├── train_pairs.jsonl       # 305 pairs (245+60)
│           │   │   ├── val_pairs.jsonl         # 49 pairs (不变)
│           │   │   └── test_pairs.jsonl        # 56 pairs (不变)
│           │   ├── views/                      # 所有物体渲染图
│           │   └── objects/
│           ├── validation_report.json          # ✅ passed
│           ├── train_command.sh                # 训练命令
│           ├── eval_command.sh                 # 评测命令
│           ├── compare_command.sh              # 比较命令
│           ├── build_augmented_dataset.sh      # 数据集构建命令
│           ├── validate_augmented_dataset.sh   # 验证命令
│           └── train.log                       # 训练日志
│
├── DiffSynth-Studio/
│   ├── train_clockwise.py                     # 训练脚本
│   ├── accelerate_config_6gpu.yaml            # 6 卡配置
│   ├── eval_spatialedit_inference.py          # SpatialEdit 推理
│   ├── eval_spatialedit_metrics.py            # SpatialEdit 指标计算
│   └── output/
│       ├── rotation8_bright_objinfo_rank32/    # baseline checkpoint (exp5)
│       │   └── epoch_0029/lora.safetensors
│       ├── v2_scaling_r1_augmented/            # ★ R1 训练输出（进行中）
│       │   └── epoch_0029/lora.safetensors     # （训练完成后生成）
│       └── eval_spatialedit_metrics/          # 评测结果 CSV
│
└── dataset_scene_v7_full50_..._objinfo_20260418/  # baseline 数据集
    ├── pairs/ (train 245 / val 49 / test 56)
    ├── views/ → symlink
    └── objects/ → symlink
```

---

## 六、本地 Repo 文件结构

```
DataEvolver/DataEvolver/
├── scripts/
│   ├── feedback_loop/                         # ★ 反馈 loop 脚本（本地是 source of truth）
│   │   ├── compare.py
│   │   ├── analyze_feedback_for_dataset.py
│   │   ├── build_pinned_split.py
│   │   ├── build_feedback_expansion_objects.py
│   │   ├── build_render_prior_library.py
│   │   ├── build_augmented_rotation_dataset.py
│   │   ├── validate_dataset.py
│   │   ├── V2_SCALING_TASK.md                 # Codex exec 任务 prompt
│   │   ├── STAGE3_DIAGNOSE_TASK.md            # Stage 3 诊断任务
│   │   └── HINT_FOR_CODEX_20260421.md         # Stage 3 故障提示
│   ├── bootstrap_scene_yaw000_objects.py      # VLM loop bootstrap
│   ├── export_rotation8_from_best_object_state.py  # rotation export
│   ├── build_rotation8_trainready_dataset.py  # trainready 构建
│   └── 68server/                              # 68 服务器评测脚本
│       ├── eval_spatialedit_inference.py
│       ├── eval_spatialedit_metrics.py
│       └── run_spatialedit_eval.sh
├── configs/seed_concepts/                     # 物体概念列表
├── pipeline/                                  # 数据构建 pipeline 脚本
└── docs/
    ├── FEEDBACK_LOOP_FRAMEWORK.md             # 本文档
    └── HANDOVER_20260421.md                   # 接手文档
```

---

## 七、跨服务器数据流

wwz 和 68 不能直接通信，必须通过本地 Windows 中继：

```bash
# wwz → 68 传输
ssh wwz "tar czf - -C <wwz_source> ." | \
  ssh zhanghy56_68 "mkdir -p <68_dest> && tar xzf - -C <68_dest>"
```

**传输内容**：trainready 目录（pairs/ + views/ + objects/），约 500MB~2GB。

**脚本同步**：本地 → 服务器
```bash
scp scripts/feedback_loop/*.py zhanghy56_68:$WORKDIR/feedback_loop_scripts/
scp scripts/feedback_loop/*.py wwz:$WWZ_CODE/scripts/feedback_loop/
scp scripts/build_rotation8_trainready_dataset.py wwz:$WWZ_CODE/scripts/
```

---

## 八、已知问题与改进方向

### 8.1 已验证的问题

| 问题 | 严重度 | 状态 | 说明 |
|------|--------|------|------|
| Stage 3 多 GPU Paint 失败 | 高 | 已修复 | GPU1/2 的 CUDA extension 不兼容，需单 GPU 或预验证 |
| wwz 脚本版本过旧 | 中 | 已修复 | `build_rotation8_trainready_dataset.py` 缺少 `--prompts-json` 参数，需手动 scp 同步 |
| Export 后 pipeline 脚本挂起 | 中 | 绕过 | export worker 完成后主进程未退出，需手动运行 trainready |
| VLM bootstrap 质量低 | 高 | 待解决 | 20 个新物体 hybrid_score 全部 < 0.6（门槛 0.78），根因待查 |

### 8.2 建议改进

1. **质量门控**（优先级高）
   - 在 rotation export 前加 hybrid_score 过滤（建议 ≥ 0.5）
   - 低于阈值的物体不进入训练集，避免噪声数据拉低指标

2. **自动化程度**
   - 当前 wwz→68 传输需要手动执行，可以写一个端到端脚本
   - training 完成后自动触发 eval + compare

3. **多轮迭代**
   - R1 完成后，compare.py 的 verdict 决定是否进入 R2
   - R2 可以：增加更多物体、调整弱角度定义、对 R1 中质量差的物体做替换

4. **VLM bootstrap 优化**
   - 当前 golden config prior 可能不适配新类型物体
   - 可以尝试：增大 max_rounds（当前 10）、调整 plateau 参数、用更强的 VLM 模型

5. **Stage 3 修复**
   - 修改 `stage3_image_to_3d.py` 的 Paint 失败处理：不再静默 fallback，而是抛出异常
   - 添加多 GPU Paint smoke test 步骤

---

## 九、如何执行下一轮（R2）

### 前提：R1 训练完成 + 评测 + 比较

```bash
# 1. 训练完成后评测
ssh zhanghy56_68 "bash $WORKDIR/feedback_loop_runs/v2_scaling_r1/round_1/eval_command.sh"

# 2. 与 baseline 比较
ssh zhanghy56_68 "bash $WORKDIR/feedback_loop_runs/v2_scaling_r1/round_1/compare_command.sh"

# 3. 查看结果
ssh zhanghy56_68 "cat $WORKDIR/feedback_loop_runs/v2_scaling_r1/round_1/compare_report.json | python3 -m json.tool"
```

### 如果 verdict = "continue"（弱角度改善，强角度未退化）

1. 以 R1 的 checkpoint 作为新 baseline
2. 重新运行 compare.py 识别新的弱角度
3. 生成新一批物体（obj_071~obj_090）
4. 重复 Stage 1-3 → bootstrap → export → trainready → 传输 → 合并 → 训练

### 如果 verdict = "stop_or_revert"（出现退化）

1. 检查退化来源：是强角度退化还是整体退化？
2. 排查新物体质量（hybrid_score 分布）
3. 可能的修复：过滤低质量物体、减少新增 pair 比例、调整训练超参

### 如果 verdict = "no_signal"（无变化）

1. 当前策略无效，需换方向
2. 备选策略：D1 弱角度过采样、T1 Loss weighting、更多物体数量

---

## 十、环境信息

| 项目 | 值 |
|------|-----|
| wwz SSH | `ssh wwz` |
| 68 SSH | `ssh zhanghy56_68` |
| wwz Python | `/home/wuwenzhuo/Qwen-VL-Series-Finetune/env/bin/python3` |
| wwz Blender | `/home/wuwenzhuo/blender-4.24/blender` |
| 68 Python | `source $WORKDIR/.venv/bin/activate` |
| wwz 后台 | tmux（screen 不可用） |
| 本地系统 | Windows 11 (PowerShell) |
| 基模 | Qwen-Image-Edit-2511 |
| 训练框架 | DiffSynth-Studio + accelerate |
| 评测基准 | SpatialEdit-Bench (488 pairs, 8 angles) |
