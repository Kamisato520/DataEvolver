# 数据集

## 1. 总览

本工作围绕单轴 azimuth 旋转编辑任务构建了两个版本的训练数据集：

| 版本 | 物体数 | 角度覆盖 | 训练 pairs | val pairs | test pairs |
|------|--------|----------|------------|-----------|------------|
| 基线 exp5（rotation8） | 35 | 7 个（45°~315°） | 245 | 49 | 56 |
| v2 R1 扩展（augmented） | 35 + 20 新 | 基线 7 角度 + 新物体 3 弱角度 | 305（+60） | 49 | 56 |

val/test split 在 R1 中保持冻结不变（pinned split），新物体只进 train。

---

## 2. 基线数据集（rotation8）

基线数据集（exp5）包含 35 个原始物体，每个物体在 7 个 azimuth 角度（45°、90°、135°、180°、225°、270°、315°）各生成一对 source→target 渲染图，共 245 个训练 pairs。source\_image 为 yaw000（正面图），target\_image 为目标角度渲染图。

数据集通过完整 Stage 1-5 pipeline 构建：种子概念 → Qwen T2I 白底图 → SAM3 mask → Hunyuan3D mesh → Blender 物体旋转渲染 → VLM bootstrap 调优 → trainready JSONL。

SpatialEdit-Bench 评测基准包含 488 对（61 物体 × 8 角度，含 360°），其中 360° 角度在训练集中不存在，属于零样本泛化测试。数据集训练集按物体维度划分（object-disjoint split），seed=42 冻结，确保评测物体与训练物体无重叠。

---

## 3. v2 R1 扩展数据集

### 3.1 扩容策略

基于 exp5 的 per-angle 评测分析，PSNR 最低的 3 个角度（270°、180°、90°）被识别为弱角度。v2 R1 新增 20 个物体（obj\_051 ~ obj\_070），每个物体仅在弱角度（90°/180°/270°）生成 3 个训练 pair，总计新增 60 pairs，合并后训练集扩至 305 pairs。

### 3.2 新增物体清单

新物体概念由 `scripts/feedback_loop/build_feedback_expansion_objects.py` 自动生成，类别分布如下：

| 类别 | 物体列表 |
|------|---------|
| 日常用品（daily\_item） | floor\_lamp, laundry\_basket, rolling\_office\_chair, small\_bookshelf, tool\_cart, garden\_hose\_reel, portable\_air\_compressor, folding\_table, plastic\_storage\_bin, standing\_fan |
| 运动器材（sports\_item） | golf\_bag, hockey\_stick, snowboard |
| 街道设施（street\_object） | bus\_stop\_sign, construction\_barrel, newspaper\_box, bike\_rack |
| 交通工具（vehicle） | cargo\_bike, electric\_scooter, mini\_excavator |

物体定义文件：[`eval/v2_scaling_r1/feedback_expansion_r1_objects.json`](../eval/v2_scaling_r1/feedback_expansion_r1_objects.json)

### 3.3 Trainready manifest 结构示例

以 obj\_051（floor\_lamp）为例，trainready manifest 中每个物体记录如下：

```json
{
  "obj_id": "obj_051",
  "source_rotation_deg": 0,
  "source_view_name": "front view",
  "target_rotations": [90, 180, 270],
  "views": {
    "yaw000": {
      "rgb_path": "views/obj_051/yaw000.png",
      "mask_path": "views/obj_051/yaw000_mask.png",
      "base_hybrid_score": 0.4891
    },
    "yaw090": {"rgb_path": "views/obj_051/yaw090.png", ...},
    "yaw180": {"rgb_path": "views/obj_051/yaw180.png", ...},
    "yaw270": {"rgb_path": "views/obj_051/yaw270.png", ...}
  }
}
```

manifest 文件：[`eval/v2_scaling_r1/trainready/trainready_manifest.json`](../eval/v2_scaling_r1/trainready/trainready_manifest.json)

---

## 4. 数据质量指标（关键发现）

VLM bootstrap 对 20 个新物体的渲染质量评估（hybrid\_score）结果如下：

| 指标 | 值 |
|------|----|
| 最高分 | 0.582（obj\_052，needs\_fix） |
| 中位数 | ~0.46 |
| 最低分 | 0.368（obj\_059，reject） |
| golden config 门槛 | 0.78 |
| 达到门槛的物体数 | 0 |

**全部 20 个新物体 hybrid\_score < 0.6**，均被标记为低质量（route = hybrid\_reject）。这一发现是 R1 最终 verdict = `inspect` 的直接根因（见第 5 节）：低质量渲染图引入噪声，导致 DINO/FID 退化，触发 strong-angle regression guard。

质量低的根本原因推测为：Hunyuan3D Stage 3 的 texture baking 虽已修复（重跑 GPU0 单卡），但贴图精度仍偏低；golden config prior 对新类别物体（如 bike\_rack、mini\_excavator）的适配性不足。

---

## 5. 数据组织格式

训练数据以 JSONL 格式存储，每行为一个训练 pair，核心字段如下：

```jsonl
{
  "source_image": "views/obj_051/yaw000.png",
  "target_image": "views/obj_051/yaw090.png",
  "instruction": "Rotate this floor lamp clockwise from front view to right side view.",
  "target_rotation_deg": 90,
  "obj_id": "obj_051",
  "obj_name": "floor_lamp",
  "prompt_version": "v3"
}
```

Prompt 格式为 v3：`Rotate this {obj} clockwise from front view to {view}.`，其中 `{view}` 为视角名称（right side view / back view / left side view 等）。

目录结构：

```
trainready/
├── pairs/
│   ├── train_pairs.jsonl    # 训练 pairs（305 行）
│   ├── val_pairs.jsonl      # 验证 pairs（49 行，冻结）
│   └── test_pairs.jsonl     # 测试 pairs（56 行，冻结）
└── views/
    └── obj_XXX/
        ├── yaw000.png       # source 图像
        ├── yaw090.png       # target 图像（90°）
        ├── yaw180.png       # target 图像（180°）
        └── yaw270.png       # target 图像（270°）
```

---

## 6. 复现入口

| 步骤 | 脚本 |
|------|------|
| 新物体概念生成 | `scripts/feedback_loop/build_feedback_expansion_objects.py` |
| Stage 1-3 数据构建 | `pipeline/stage1_text_expansion.py` → `stage2_t2i_generate.py` → `stage3_image_to_3d.py` |
| VLM bootstrap | `scripts/bootstrap_scene_yaw000_objects.py` |
| 旋转导出 | `scripts/export_rotation8_from_best_object_state.py` |
| Trainready 构建 | `scripts/build_rotation8_trainready_dataset.py` |
| 数据集合并 | `scripts/feedback_loop/build_augmented_rotation_dataset.py` |
| 可训练性校验 | `scripts/feedback_loop/validate_dataset.py` |
| manifest 文件 | [`eval/v2_scaling_r1/trainready/trainready_manifest.json`](../eval/v2_scaling_r1/trainready/trainready_manifest.json) |
