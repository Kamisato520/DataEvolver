# Rotation8 几何模态规划修订版（不污染现有数据集）

## 摘要

目标是在现有 `rotation8` 数据集体系之上，补齐：

- `camera`
- `pose`
- `depth`
- `normal`

新增硬约束如下：

- **绝不改写、覆盖、补写现有数据集目录**
- 所有几何模态工作都必须写入**新的并列目录**
- 现有 3 套数据集目录全部视为只读输入源

现有冻结目录：

- `/aaaidata/zhangqisong/data_build_runs/scene_full20_loop_20260404_code/pipeline/data/dataset_scene_v7_full20_rotation8_consistent_yaw000_20260407`
- `/aaaidata/zhangqisong/data_build_runs/scene_full20_loop_20260404_code/pipeline/data/dataset_scene_v7_full20_rotation8_trainready_front2others_20260407`
- `/aaaidata/zhangqisong/data_build_runs/scene_full20_loop_20260404_code/pipeline/data/dataset_scene_v7_full20_rotation8_trainready_front2others_splitobj_seed42_20260407`

这些目录只作为输入源，后续所有 `geometry / depth / normal` 工作一律新建并列输出根。

## 当前现状

当前 `rotation8` 数据集已经具备：

- `RGB`
- `mask`
- `render_metadata`
- `control_state`

其中 `render_metadata` 中已经可复用的信息包括：

- `camera_mode`
- `camera_name`
- `control_state.object.yaw_deg`
- `control_state.object.scale`
- `control_state.object.offset_z`
- `support_plane_z`
- `placement_anchor_xy`
- `ground_contact_epsilon`
- `frames`

但现有数据集还缺少训练友好的显式几何模态：

- 标准化 `camera intrinsics`
- 标准化 `camera extrinsics`
- 标准化 `object pose`
- `depth`
- `normal`
- 稳定的 `2D bbox / 3D bbox`

结论：

- 现有数据集已经足够做图像编辑训练。
- 但还不够做更强的几何一致性训练、几何监督或几何评估。

## 新目录命名规则

后续实现固定使用以下三个并列目录，不允许另起命名风格。

### A. geometry-only 回填版

- `dataset_scene_v7_full20_rotation8_geommeta_from_consistent_20260407`

用途：

- 从 `rotation8 consistent` 回填 `camera / pose / bbox`
- 不新增 `depth / normal`
- 若不复制旧 RGB/mask，可使用 `symlink` 或 manifest 引用，但根目录必须新建

### B. geometry-complete 版

- `dataset_scene_v7_full20_rotation8_geomodal_trainready_front2others_20260407`

用途：

- 在 train-ready 语义层上挂接：
  - `geometry metadata`
  - `depth`
  - `normal`
- 作为真正可训练的几何增强版数据集

### C. split 版

- `dataset_scene_v7_full20_rotation8_geomodal_trainready_front2others_splitobj_seed42_20260407`

用途：

- 保持 object-disjoint split
- 继续只引用新的 geomodal train-ready 根
- 不碰旧 split 根

## 总体原则

### 1. 旧目录只读

实现阶段禁止向旧目录写入任何新文件，包括但不限于：

- `yawYYY_geometry_metadata.json`
- `yawYYY_depth.exr`
- `yawYYY_depth_vis.png`
- `yawYYY_normal.exr`
- `yawYYY_normal.png`

### 2. 新目录并列输出

所有新增模态都只能出现在新根中，例如：

- `...geommeta_from_consistent.../views/obj_xxx/yawYYY_geometry_metadata.json`
- `...geomodal_trainready.../views/obj_xxx/yawYYY_depth.exr`
- `...geomodal_trainready.../views/obj_xxx/yawYYY_normal.exr`

### 3. consistent 根是真源

几何真源统一以：

- `dataset_scene_v7_full20_rotation8_consistent_yaw000_20260407`

作为唯一 source of truth，不从旧的 `best-of-each-pair` 版本回填。

## 几何 metadata schema

每个视角新增一份：

- `yawYYY_geometry_metadata.json`

字段标准固定如下。

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

### Scene / Placement

- `camera_mode`
- `placement_anchor_xy`
- `ground_contact_epsilon`
- `base_pair_name`
- `base_best_round_idx`
- `base_hybrid_score`

### Derived Geometry

- `bbox_3d_world`
- `bbox_3d_object`
- `bbox_2d_xyxy`
- `object_center_world`
- `object_center_camera`

### 矩阵与坐标约定

- 使用 Blender 世界坐标系
- 所有矩阵统一保存为 `4x4 row-major nested list`
- 不允许在实现时更换坐标定义

## Phase 1：只读源 + 新根输出的 geometry metadata 回填

### 输入

- `rotation8 consistent` 作为只读源

### 输出

- `dataset_scene_v7_full20_rotation8_geommeta_from_consistent_20260407`

### 实现策略

新增回填脚本负责：

- 读取旧 consistent 数据集中的：
  - `manifest`
  - `object_manifest`
  - `render_metadata`
  - `control_state`
- 重新进入 Blender 计算几何信息
- 将 `geometry_metadata.json` 写入新根
- 如需要复制或链接原 RGB/mask/control/render_metadata，只能写到新根
- 绝不能向旧 consistent 根写任何新文件

### 实现要求

- 不重渲 RGB
- 直接复用现有 canonical render state
- 相机内参由 Blender 相机参数与分辨率显式推导
- 物体位姿使用插入后 root object 的 `matrix_world`
- `2D bbox` 使用 `3D bbox` 顶点投影得到
- mask 外接框只可用于辅助校验，不能替代主版本 `bbox_2d_xyxy`

## Phase 2：只读源 + 新根输出的 depth / normal 补全

### 输入

- `rotation8 consistent`
- 或 `geommeta_from_consistent`

### 输出

- `dataset_scene_v7_full20_rotation8_geomodal_trainready_front2others_20260407`

### 实现策略

`depth / normal` 不能仅靠 metadata 回填，必须重新跑 Blender pass。

但这一阶段：

- 不重新跑 VLM loop
- 不重新做 best-state selection
- 不重新生成旧版 RGB 数据集

它只是从现有 canonical `rotation8 consistent` 状态出发，补几何模态。

### 文件命名

每个视角允许出现：

- `yawYYY.png`
- `yawYYY_mask.png`
- `yawYYY_geometry_metadata.json`
- `yawYYY_depth.exr`
- `yawYYY_depth_vis.png`
- `yawYYY_normal.exr`
- `yawYYY_normal.png`

这些文件全部必须属于**新目录**。

### 格式约定

Depth：

- 原始训练文件：`EXR`
- 预览检查文件：`PNG`

Normal：

- 原始训练文件：`EXR`
- 预览/轻量兼容文件：`PNG`

默认采用 `EXR + PNG` 双格式。

## Phase 3：新 train-ready 与新 split

后续 pair 清单接线时，不修改旧的：

- `dataset_scene_v7_full20_rotation8_trainready_front2others_20260407`
- `dataset_scene_v7_full20_rotation8_trainready_front2others_splitobj_seed42_20260407`

而是新建：

- `dataset_scene_v7_full20_rotation8_geomodal_trainready_front2others_20260407`
- `dataset_scene_v7_full20_rotation8_geomodal_trainready_front2others_splitobj_seed42_20260407`

### pair manifest 新增字段

- `source_geometry_metadata`
- `target_geometry_metadata`
- `source_depth`
- `target_depth`
- `source_normal`
- `target_normal`

这些字段全部只允许指向新根内路径。

### split 规则

- 仍沿用 `seed = 42`
- 仍保持 `14 / 3 / 3`
- split 根只引用新的 geomodal train-ready 根
- 不回指旧 train-ready 根中新生成的几何资产

## 实施顺序

1. 先写正式规划文档，并将“旧目录只读、新目录并列输出”写成硬约束
2. Phase 1：
   - 从 `rotation8 consistent` 只读回填 geometry metadata
   - 输出到 `geommeta_from_consistent` 新根
3. Phase 2：
   - 基于只读源补 `depth / normal`
   - 输出到 `geomodal_trainready` 新根
4. Phase 3：
   - 从 `geomodal_trainready` 生成新的 object-disjoint split 根
5. 所有验收与抽查只针对新目录进行

## 测试与验收

### 旧目录保护检查

实现前后，旧三套数据集目录中的：

- 文件数
- 关键文件修改时间

都必须保持不变。

### 新目录完整性检查

新根中必须存在预期文件：

- geometry-only 根：每视角 `geometry_metadata`
- geomodal 根：每视角 `depth / normal / geometry_metadata`
- split 根：新的 split pair 清单与 manifest

### 接线检查

- 新 train-ready / split manifest 必须全部指向新根
- 不存在路径回指到旧目录中的新生成文件

### 数值与几何检查

- `camera_to_world_4x4` 与 `world_to_camera_4x4` 互逆
- `object_to_world_4x4` 与 `world_to_object_4x4` 互逆
- `fx / fy / cx / cy` 与分辨率一致
- `bbox_2d_xyxy` 与 mask 大致对齐
- `object_center_camera.z` 应为正
- 物体不应落在相机后方

### 模态检查

- `depth` 不能全黑或全常数
- `normal` 不能全零或全噪声
- `depth / normal` 与 mask 空间对齐合理

### 抽查对象

- `obj_001`
- `obj_009`
- `obj_020`

## 假设与默认值

- 默认所有旧目录都冻结，不做原地补写
- 默认新目录允许通过 `symlink` 或 `copy` 复用旧 RGB/mask，但必须在新根下体现
- 默认 split 仍沿用 `seed=42, 14/3/3`
- 默认 `depth / normal` 原始格式为 `EXR`
- 默认预览/检查格式为 `PNG`
- 默认 source view 仍为 `yaw000 / front view`
- 默认不重新跑 VLM loop，不重新做 selection

## 实现备注

当前推荐的实现路径是：

- geometry-only 回填脚本：从 consistent 根只读回填 `camera / pose / bbox`
- modal 导出脚本：在 Blender 中补 `depth / normal`
- geomodal train-ready 构建脚本：挂接几何模态与 pair metadata
- 复用现有 split 构建脚本，生成新的 object-disjoint split 根

这套流程的目标不是替换现有数据集，而是**在不污染现有资产的前提下，平行构建一套更适合几何训练和评估的新版本数据集**。
