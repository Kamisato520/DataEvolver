# BBoxMask 数据集快速说明

## 结果位置

远端已经生成好的 bbox-conditioned 数据集在这里：

### 1. 全量 train-ready

`/aaaidata/zhangqisong/data_build_runs/scene_full20_loop_20260404_code/pipeline/data/dataset_scene_v7_full50_rotation8_trainready_front2others_bboxmask_final_20260414`

### 2. 按 object 划分的 train/val/test

`/aaaidata/zhangqisong/data_build_runs/scene_full20_loop_20260404_code/pipeline/data/dataset_scene_v7_full50_rotation8_trainready_front2others_splitobj_seed42_bboxmask_final_20260414`

正式训练默认建议直接用第 2 个，也就是 split 根。

## split 配置与规模

当前默认训练根：

`/aaaidata/zhangqisong/data_build_runs/scene_full20_loop_20260404_code/pipeline/data/dataset_scene_v7_full50_rotation8_trainready_front2others_splitobj_seed42_bboxmask_final_20260414`

这版 split 是严格按 **object-disjoint** 划分的，也就是同一个物体不会同时出现在 train 和 val/test 中。

具体规模是：

- `train = 35` 个物体，`245` 个 pair
- `val = 7` 个物体，`49` 个 pair
- `test = 8` 个物体，`56` 个 pair

总计：

- `50` 个物体
- `350` 个 pair

## 这版数据集做了什么

这版是在 `20260410 final` 标准数据集基础上补出来的 bbox 版本：

- 根据每张 `mask` 自动反推出主物体 `bbox`
- 在图像上画出 **红色矩形框**
- 生成新的带框图
- 在 pair 行里补充 bbox 坐标和 bbox 标注文件路径

## pair 是怎么构成的

pair 的构成方式是固定的：

- 每个物体都以 `yaw000` 作为 source view
- 再分别配对到 7 个 target 角度：
  - `45`
  - `90`
  - `135`
  - `180`
  - `225`
  - `270`
  - `315`

所以：

- 每个物体固定产生 `7` 个 pair
- 整个任务始终是：
  - `front view -> target rotated view`

例如：

- `obj_001_yaw000_to_yaw045`

表示：

- 输入 `obj_001` 的 `yaw000/front view`
- 目标是生成 `yaw045/front-right view`

## 训练时默认该读哪个字段

pair 文件在：

- `pairs/train_pairs.jsonl`
- `pairs/val_pairs.jsonl`
- `pairs/test_pairs.jsonl`

每一行里现在最重要的字段是：

- `source_image`
- `target_image`
- `instruction`

注意：

- 这里的 `source_image` / `target_image` **已经是带红色 bbox 的图**
- 如果你想读原始无框图，用：
  - `source_image_raw`
  - `target_image_raw`

## CSV / JSONL 里是什么信息

`pairs/train_pairs.csv`、`pairs/val_pairs.csv`、`pairs/test_pairs.csv` 和对应的 `.jsonl` 存的是同一套样本信息，只是一个是表格形式，一个是逐行 JSON。

这些文件包含三类主要信息：

### 1. 基本任务信息

- `pair_id`
- `obj_id`
- `task_type`
- `split`
- `source_rotation_deg`
- `target_rotation_deg`
- `source_view_name`
- `target_view_name`
- `instruction`

### 2. 图像与控制文件路径

- `source_image`
- `target_image`
- `source_mask`
- `target_mask`
- `source_render_metadata`
- `target_render_metadata`
- `source_control_state`
- `target_control_state`

### 3. bbox 增强信息

- `source_image_raw`
- `target_image_raw`
- `source_image_bbox`
- `target_image_bbox`
- `source_bbox_json`
- `target_bbox_json`
- `source_bbox_xyxy`
- `target_bbox_xyxy`
- `source_bbox_xywh`
- `target_bbox_xywh`
- `source_bbox_xyxy_norm`
- `target_bbox_xyxy_norm`
- `source_bbox_xywh_norm`
- `target_bbox_xywh_norm`

其中最关键的是：

- `source_image / target_image`：默认已经是**带红色矩形框**的图
- `source_image_raw / target_image_raw`：原始无框 RGB
- `source_bbox_json / target_bbox_json`：每张图对应的 bbox 标注文件

## bbox 相关字段

每一行还新增了这些字段：

- `source_bbox_json`
- `target_bbox_json`
- `source_bbox_xyxy`
- `target_bbox_xyxy`
- `source_bbox_xywh`
- `target_bbox_xywh`
- `source_bbox_xyxy_norm`
- `target_bbox_xyxy_norm`
- `source_bbox_xywh_norm`
- `target_bbox_xywh_norm`

## bbox 标注文件位置

每个视图对应两类新文件：

- 带框图：
  - `bbox_views/obj_xxx/yawYYY.png`
- bbox 标注：
  - `bbox_annotations/obj_xxx/yawYYY_bbox.json`

## 最小训练建议

第一版训练，直接这样用就够了：

- 输入：
  - `source_image`
  - `instruction`
- 监督目标：
  - `target_image`

也就是：

- 直接训练“带 bbox 的源图 -> 带 bbox 的目标图”

如果后面想做对照实验，再切到：

- `source_image_raw`
- `target_image_raw`

### 一条样本的实际含义

例如 CSV 里会看到这种组合：

- `source_image = bbox_views/obj_001/yaw000.png`
- `target_image = bbox_views/obj_001/yaw045.png`
- `source_image_raw = views/obj_001/yaw000.png`
- `target_image_raw = views/obj_001/yaw045.png`
- `source_bbox_json = bbox_annotations/obj_001/yaw000_bbox.json`

这表示：

- 训练默认使用的是带框图
- 原图和 bbox 标注也都保留了，方便做 ablation 或额外条件控制

## 最小检查命令

```bash
ssh wwz "python3 - <<'PY'
import json
from pathlib import Path
root = Path('/aaaidata/zhangqisong/data_build_runs/scene_full20_loop_20260404_code/pipeline/data/dataset_scene_v7_full50_rotation8_trainready_front2others_splitobj_seed42_bboxmask_final_20260414')
row = json.loads(next(open(root / 'pairs' / 'train_pairs.jsonl', 'r', encoding='utf-8')))
print(json.dumps(row, ensure_ascii=False, indent=2))
print((root / row['source_image']).exists())
print((root / row['source_bbox_json']).exists())
PY"
```

## 一句话

如果你现在要直接开训，就用：

`/aaaidata/zhangqisong/data_build_runs/scene_full20_loop_20260404_code/pipeline/data/dataset_scene_v7_full50_rotation8_trainready_front2others_splitobj_seed42_bboxmask_final_20260414`

然后把 `source_image`、`target_image`、`instruction` 送进训练即可。
