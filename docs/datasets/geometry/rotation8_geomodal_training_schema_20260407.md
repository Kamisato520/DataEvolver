# Rotation8 Geomodal 训练侧 Schema 与 Loader 说明

## 目标

这份文档描述当前几何增强版 `rotation8` 数据集在训练侧的读取约定，覆盖：

- 数据根目录
- pair 级 schema
- geometry metadata schema
- 推荐 loader
- split 版本的使用方式

对应实现模块：

- [rotation_geomodal_dataset.py](/D:/新建文件夹/用户目录/桌面/autoresearch_vlm/Auto-claude-code-research-in-sleep/DataEvolver/DataEvolver/pipeline/rotation_geomodal_dataset.py)
- [inspect_rotation_geomodal_loader.py](/D:/新建文件夹/用户目录/桌面/autoresearch_vlm/Auto-claude-code-research-in-sleep/DataEvolver/DataEvolver/scripts/inspect_rotation_geomodal_loader.py)

## 支持的数据根

### geomodal train-ready 根

- `/aaaidata/zhangqisong/data_build_runs/scene_full20_loop_20260404_code/pipeline/data/dataset_scene_v7_full20_rotation8_geomodal_trainready_front2others_20260407`

特点：

- `20` 个 object
- `160` 个视图
- `140` 个训练对
- 每个视图都有：
  - RGB
  - mask
  - `geometry_metadata`
  - `depth.exr`
  - `depth_vis.png`
  - `normal.exr`
  - `normal.png`

### geomodal split 根

- `/aaaidata/zhangqisong/data_build_runs/scene_full20_loop_20260404_code/pipeline/data/dataset_scene_v7_full20_rotation8_geomodal_trainready_front2others_splitobj_seed42_20260407`

特点：

- object-disjoint split
- `train / val / test = 98 / 21 / 21`
- `views/` 和 `objects/` 为共享引用

## pair 级 schema

每一行 `pairs/*.jsonl` 至少包含以下字段：

- `pair_id`
- `obj_id`
- `task_type`
- `split`
- `source_rotation_deg`
- `target_rotation_deg`
- `source_view_name`
- `target_view_name`
- `instruction`

### RGB / mask / 原始 metadata

- `source_image`
- `target_image`
- `source_mask`
- `target_mask`
- `source_render_metadata`
- `target_render_metadata`
- `source_control_state`
- `target_control_state`

### 几何增强字段

- `source_geometry_metadata`
- `target_geometry_metadata`
- `source_depth`
- `target_depth`
- `source_normal`
- `target_normal`
- `source_normal_vis`
- `target_normal_vis`
- `source_depth_vis`
- `target_depth_vis`

这些路径全部是**相对数据根目录**的相对路径。

## geometry metadata schema

每个视角新增：

- `views/obj_xxx/yawYYY_geometry_metadata.json`

核心字段：

### Camera

- `camera_model`
- `image_width`
- `image_height`
- `fx`
- `fy`
- `cx`
- `cy`
- `camera_to_world_4x4`
- `world_to_camera_4x4`
- `camera_position_xyz`
- `camera_forward_xyz`
- `camera_up_xyz`

### Object Pose

- `object_to_world_4x4`
- `world_to_object_4x4`
- `object_position_xyz`
- `object_rotation_euler_xyz`
- `object_scale_xyz`
- `object_yaw_deg`
- `support_plane_z`

### Placement / Derived Geometry

- `camera_mode`
- `placement_anchor_xy`
- `ground_contact_epsilon`
- `bbox_3d_world`
- `bbox_3d_object`
- `bbox_2d_xyxy`
- `object_center_world`
- `object_center_camera`

## 推荐 loader

推荐使用：

- [RotationGeomodalDataset](/D:/新建文件夹/用户目录/桌面/autoresearch_vlm/Auto-claude-code-research-in-sleep/DataEvolver/DataEvolver/pipeline/rotation_geomodal_dataset.py)

特点：

- 兼容 train-ready 根与 split 根
- 默认返回：
  - pair 行字段
  - 绝对路径
  - 已解析的 `source_geometry` / `target_geometry`
- 其他重模态按需懒加载

### 默认行为

默认：

- `load_geometry=True`
- 其他图像/EXR 不主动加载

所以最轻量的训练前处理可以直接只读：

- instruction
- source / target 图像路径
- source / target geometry metadata

### 可选模态

可按需启用：

- `load_rgb=True`
- `load_mask=True`
- `load_render_metadata=True`
- `load_control_state=True`
- `load_depth_exr=True`
- `load_normal_exr=True`
- `load_depth_vis=True`
- `load_normal_vis=True`

## 最小使用示例

```python
from pipeline.rotation_geomodal_dataset import RotationGeomodalDataset

root = "/aaaidata/zhangqisong/data_build_runs/scene_full20_loop_20260404_code/pipeline/data/dataset_scene_v7_full20_rotation8_geomodal_trainready_front2others_20260407"

dataset = RotationGeomodalDataset(
    root,
    split="train",
    load_geometry=True,
    load_rgb=False,
    load_depth_exr=False,
    load_normal_exr=False,
)

sample = dataset[0]
print(sample["pair_id"])
print(sample["instruction"])
print(sample["source_image_path_abs"])
print(sample["target_image_path_abs"])
print(sample["source_geometry"]["bbox_2d_xyxy"])
```

### split 根示例

```python
from pipeline.rotation_geomodal_dataset import RotationGeomodalDataset

root = "/aaaidata/zhangqisong/data_build_runs/scene_full20_loop_20260404_code/pipeline/data/dataset_scene_v7_full20_rotation8_geomodal_trainready_front2others_splitobj_seed42_20260407"

train_ds = RotationGeomodalDataset(root, split="train")
val_ds = RotationGeomodalDataset(root, split="val")
test_ds = RotationGeomodalDataset(root, split="test")
```

## 返回样本结构

`dataset[i]` 返回一个 `dict`，包含：

- 原始 pair 行字段
- `dataset_root`
- `paths`
- 各类绝对路径字段

默认还包含：

- `source_geometry`
- `target_geometry`

如果启用了额外懒加载，还会增加：

- `source_image_data`
- `target_image_data`
- `source_mask_data`
- `target_mask_data`
- `source_depth_data`
- `target_depth_data`
- `source_normal_data`
- `target_normal_data`

## EXR 读取说明

loader 内部的 EXR 读取顺序是：

1. `imageio.v3`
2. `cv2.imread(..., IMREAD_UNCHANGED)`

因此如果训练环境要直接读 `depth.exr / normal.exr`，推荐至少保证以下之一可用：

- `imageio`
- 带 EXR 支持的 `opencv-python`

如果只做快速检查，也可以先使用：

- `depth_vis.png`
- `normal.png`

## 训练时的推荐分层

### 最轻量方案

训练 loop 只消费：

- `instruction`
- `source_image`
- `target_image`

geometry 仅用于分析或过滤。

### 几何增强方案

训练 loop 同时消费：

- `source / target RGB`
- `source / target depth`
- `source / target normal`
- `source / target geometry metadata`

适合做：

- multi-view editing
- geometry-aware consistency
- pose-aware conditioning

## 检查脚本

快速检查 loader 可用性：

```bash
python scripts/inspect_rotation_geomodal_loader.py ^
  --root /aaaidata/zhangqisong/data_build_runs/scene_full20_loop_20260404_code/pipeline/data/dataset_scene_v7_full20_rotation8_geomodal_trainready_front2others_20260407 ^
  --split train ^
  --index 0
```

如果要同时测试 RGB / EXR 读取：

```bash
python scripts/inspect_rotation_geomodal_loader.py ^
  --root /aaaidata/zhangqisong/data_build_runs/scene_full20_loop_20260404_code/pipeline/data/dataset_scene_v7_full20_rotation8_geomodal_trainready_front2others_20260407 ^
  --split train ^
  --index 0 ^
  --load-rgb ^
  --load-depth-exr ^
  --load-normal-exr
```

## 约束

- loader 是只读的，不修改任何数据集文件
- split 根和 train-ready 根的路径解析逻辑一致
- 几何增强字段全部假定指向新建的 geomodal 数据集根，不回指旧 train-ready 根
