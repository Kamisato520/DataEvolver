# Full50 BBoxMask 训练说明

## 可直接使用的数据根

### 全量 bbox-conditioned train-ready

- `/aaaidata/zhangqisong/data_build_runs/scene_full20_loop_20260404_code/pipeline/data/dataset_scene_v7_full50_rotation8_trainready_front2others_bboxmask_final_20260414`

### object-disjoint split 版

- `/aaaidata/zhangqisong/data_build_runs/scene_full20_loop_20260404_code/pipeline/data/dataset_scene_v7_full50_rotation8_trainready_front2others_splitobj_seed42_bboxmask_final_20260414`

建议训练默认使用第二个 split 根。

## 这版数据集做了什么

它是在 `20260410 final train-ready/split` 的基础上新增了一层 bbox conditioning：

- 根据每张 `*_mask.png` 自动反推出主要物体的 2D bbox
- 为每个视图生成一张画好 bbox 的 overlay 图
- 在 pair 行里保留原图路径，同时新增 bbox 路径和 bbox 坐标
- 默认把 `source_image / target_image` 改成 bbox overlay 图，便于训练侧直接使用

## pair 行新增字段

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

其中：

- `source_image` 和 `target_image` 现在默认指向 `bbox_views/.../*.png`
- `source_image_raw` 和 `target_image_raw` 才是原始无框 RGB

## bbox 标注文件

每个视图对应：

- `bbox_annotations/obj_xxx/yawYYY_bbox.json`
- `bbox_views/obj_xxx/yawYYY.png`

`bbox.json` 中包含：

- `bbox_xyxy`
- `bbox_xywh`
- `bbox_xyxy_norm`
- `bbox_xywh_norm`
- `area_pixels`
- `area_ratio`

## 推荐训练口径

第一版如果要做 bbox-conditioned 训练，建议直接用：

- 输入：
  - `source_image`
  - `instruction`
- 监督：
  - `target_image`

因为这两个字段已经是带框版本。

如果后面想做 ablation，可以切换为：

- source 用 `source_image_raw`
- target 用 `target_image_raw`

## 最小检查

```bash
ssh wwz "python3 - <<'PY'
import json
from pathlib import Path
root=Path('/aaaidata/zhangqisong/data_build_runs/scene_full20_loop_20260404_code/pipeline/data/dataset_scene_v7_full50_rotation8_trainready_front2others_splitobj_seed42_bboxmask_final_20260414')
row=json.loads(next(open(root/'pairs'/'train_pairs.jsonl','r',encoding='utf-8')))
print(row['source_image'])
print(row['source_image_raw'])
print(row['source_bbox_xyxy'])
print((root/row['source_image']).exists())
print((root/row['source_bbox_json']).exists())
PY"
```

## 注意

- 这版 bbox 是由 mask 自动反推的，不是人工框
- 当前只补了 bbox-conditioned standard dataset
- 还没有同步刷新 `geomodal + bbox` 版本
