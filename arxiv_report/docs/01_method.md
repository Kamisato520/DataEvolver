# 方法

## 1. 总体架构

本系统由两个核心组件构成：**数据合成 Pipeline** 与 **评测驱动反馈 Loop**。数据合成 Pipeline 负责从文本概念出发，经多阶段自动化流程生成可训练的多角度旋转编辑数据集；反馈 Loop 则以 SpatialEdit-Bench 评测结果为信号，识别模型弱角度，自动扩充对应方向的训练数据，并在下一轮训练后再次评测，形成闭环迭代。

```
文本种子概念
    │
    ▼
数据合成 Pipeline（Stage 1 → Stage 5）
    │ 训练就绪数据集（JSONL pairs + 渲染图）
    ▼
LoRA 训练（Qwen-Image-Edit-2511）
    │
    ▼
SpatialEdit-Bench 评测（传统指标 + VIEScore）
    │
    ▼
compare.py → 弱角度识别 → verdict
    │ 弱角度列表 + 新物体概念
    ▼
反馈 Loop（扩充弱角度数据 → 合并 → 再训练 → 再评测）
```

---

## 2. 数据合成 Pipeline

### Stage 1：文本扩展（Text Expansion）

以物体名称（种子概念）为输入，通过语言模型生成详细的文字描述（prompt），描述涵盖外观、材质、颜色等细节。输出写入 `prompts.json`，格式为 `{id: ..., name: ..., prompt: ...}`。相关脚本：`pipeline/stage1_text_expansion.py`。

### Stage 2：白底参考图生成（T2I）

以 Stage 1 生成的 prompt 为输入，调用 Qwen Image local 模型（T2I）生成 1024×1024 的白底参考图（PNG）。白底是后续 SAM3 分割和 Hunyuan3D 重建的前置条件，必须严格保证。输出写入 `images/` 目录。相关脚本：`pipeline/stage2_t2i_generate.py`。

### Stage 2.5：SAM3 分割（Segmentation）

以白底参考图为输入，调用 SAM3 模型提取物体 RGBA mask（透明背景），用于后续 3D 重建时区分前景与背景。脚本文件名为 `stage2_5_sam2_segment.py`（内部实际调用 SAM3）。输出写入 `images_rgba/` 目录。

### Stage 3：Hunyuan3D 网格重建（Image-to-3D）

以白底参考图和 SAM3 mask 为输入，调用 Hunyuan3D 模型重建带 PBR 贴图的 GLB 网格。该阶段包含 texture baking（GPU paint 管线）。**已知风险**：Paint 管线依赖 CUDA custom extension（DifferentiableRenderer），多 GPU 并行时可能在非 GPU0 上静默 fallback 为无贴图的 shape-only GLB，因此建议单 GPU 或事先做 smoke test。相关脚本：`pipeline/stage3_image_to_3d.py`。

### Stage 4：Blender 场景渲染（Scene Render）

以 Stage 3 输出的 GLB 网格为输入，使用 Blender 进行场景渲染。**关键设计**：通过旋转物体（`yaw_deg` 参数）实现多角度渲染，而非旋转相机轨道。这样可以保持灯光和场景构图稳定，减少角度变换引入的光照伪影。训练集覆盖 7 个角度（45°、90°、135°、180°、225°、270°、315°），每个角度导出一张渲染图及对应的 mask。废弃脚本 `stage4_blender_render.py`（相机轨道方式）不再使用。

### Stage 4.5：VLM Bootstrap（渲染参数调优）

渲染参数（光照、材质、相机位置等）对最终图像质量影响显著。本阶段引入 VLM 自动调参循环：以 `golden_config_library.json`（58 条已验证的高质量渲染配置聚合中位数）作为 warm-start prior，每轮由 Blender 渲染当前参数下的图像，再由 Qwen3.5-VL 评审并给出 hybrid_score（综合清晰度、视角正确性、外观一致性），agent 根据反馈调整参数，迭代至收敛（连续 2 轮提升 < 0.01 或达到 10 轮上限）。相关脚本：`scripts/bootstrap_scene_yaw000_objects.py`。

### Stage 5：元数据合并与 Trainready 构建

将多角度渲染图和 VLM bootstrap 输出的最佳 control_state 重组为训练用格式：`views/obj_XXX/` 存放各角度渲染图，`pairs/train_pairs.jsonl` 中每行为一个训练 pair，字段包括 `source_image`（yaw000 正面图）、`target_image`（目标角度图）、`instruction`（prompt v3 格式）、`target_rotation_deg`。merge 脚本：`scripts/feedback_loop/build_augmented_rotation_dataset.py`。

---

## 3. 反馈 Loop v2 架构

反馈 Loop 的核心思想是：**不修改训练策略，仅通过评测反馈识别弱角度，自动扩充对应方向的训练数据**，验证 Scaling Law 是否能持续提升弱角度指标。

```
┌──────────────────────────────────────────────────────┐
│ 68 服务器（训练 + 评测）                               │
│ ① SpatialEdit-Bench per-angle 评测                   │
│ ② compare.py → 识别弱角度（PSNR 最低 3 个非 360°）   │
│ ③ analyze_feedback_for_dataset.py → 数据集扩容计划   │
│ ④ build_feedback_expansion_objects.py → 新物体概念   │
└──────────────────┬───────────────────────────────────┘
                   │ 新物体概念 JSON
                   ▼
┌──────────────────────────────────────────────────────┐
│ wwz 服务器（数据构建）                                 │
│ ⑤ Stage 1-3：T2I + SAM3 + Hunyuan3D               │
│ ⑥ VLM Loop Bootstrap（golden config warm-start）    │
│ ⑦ Rotation Export（弱角度 90°/180°/270°）            │
│ ⑧ Trainready Build → JSONL pairs                    │
└──────────────────┬───────────────────────────────────┘
                   │ tar+ssh 本地中继传输
                   ▼
┌──────────────────────────────────────────────────────┐
│ 68 服务器（合并 + 训练）                               │
│ ⑨ build_augmented_rotation_dataset.py → 合并数据集  │
│ ⑩ validate_dataset.py → 可训练性校验                 │
│ ⑪ 训练（epoch 29）→ 评测 → compare.py              │
│    verdict: PASS / INSPECT / FAIL → 下一轮           │
└──────────────────────────────────────────────────────┘
```

verdict 决策逻辑（`compare.py`）：

| 条件 | verdict |
|------|---------|
| 弱角度改善 + 强角度无退化 | `continue` |
| 有改善但同时触发 strong-angle regression guard | `inspect` |
| 关键指标超过退化阈值 | `stop_or_revert` |
| 无显著变化 | `no_signal` |

反馈 Loop 相关脚本位于 [`scripts/feedback_loop/`](../../scripts/feedback_loop/) 和 [`docs/FEEDBACK_LOOP_FRAMEWORK.md`](../FEEDBACK_LOOP_FRAMEWORK.md)。

---

## 4. 关键设计决策

**（1）物体旋转而非相机轨道**
渲染时通过旋转物体（`yaw_deg`）而非移动相机实现视角变化。这样灯光方向、背景构图保持不变，避免相机轨道带来的光照伪影，提升数据集内的光照一致性。废弃脚本 `stage4_blender_render.py` 不再维护。

**（2）训练集 7 角度 vs 评测 8 角度**
训练集覆盖 7 个非零角度（45°~315°），SpatialEdit-Bench 评测包含 8 个角度（含 360°）。360° 在训练集中不存在，评测时该角度为零样本泛化测试。

**（3）统一 Checkpoint epoch 29**
所有实验轮次统一取 epoch 29 的 checkpoint，排除 checkpoint 选取差异对横向对比的干扰。

**（4）VLM Bootstrap 用 golden\_config\_library 作 prior**
直接从零调参（冷启动）收敛慢且不稳定。引入 golden\_config\_library（58 条已验证高质量渲染配置的统计聚合）作为 warm-start prior，显著加速收敛、减少渲染轮次。

**（5）Pinned Split 保证数据无泄漏**
原始 50 个物体的 train/val/test 分配以 seed=42 冻结（`pinned_split.json`），新增物体只进 train。`validate_dataset.py` 在每次 merge 后自动校验 split 一致性，确保 val/test 集不被新数据污染。
