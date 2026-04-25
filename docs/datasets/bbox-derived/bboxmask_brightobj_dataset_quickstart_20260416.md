# BBoxMask BrightObj Dataset Quickstart

这版数据集是在 `bboxmask final 20260414` 的基础上派生出来的**并列亮度增强版**。  
核心目标是：**不改原数据集、不改原 bbox 配置，只把 mask 内的物体区域提亮到接近参考图的亮度水平。**

## 新数据集根

全量 bbox train-ready：

- `/aaaidata/zhangqisong/data_build_runs/scene_full20_loop_20260404_code/pipeline/data/dataset_scene_v7_full50_rotation8_trainready_front2others_bboxmask_brightobj_refstage35_final_20260416`

split 版：

- `/aaaidata/zhangqisong/data_build_runs/scene_full20_loop_20260404_code/pipeline/data/dataset_scene_v7_full50_rotation8_trainready_front2others_splitobj_seed42_bboxmask_brightobj_refstage35_final_20260416`

## 来源与约束

来源根分别是：

- `/aaaidata/zhangqisong/data_build_runs/scene_full20_loop_20260404_code/pipeline/data/dataset_scene_v7_full50_rotation8_trainready_front2others_bboxmask_final_20260414`
- `/aaaidata/zhangqisong/data_build_runs/scene_full20_loop_20260404_code/pipeline/data/dataset_scene_v7_full50_rotation8_trainready_front2others_splitobj_seed42_bboxmask_final_20260414`

原 `20260414` 数据集没有被覆盖。

## 参考亮度来源

参考图：

- `/aaaidata/zhangqisong/data_build_runs/scene_full20_loop_20260404_code/pipeline/data/evolution_scene_v7_full20_rotation4_agent_round0_stage35_aborted_20260404/_shards/shard_gpu_0/obj_001_yaw270/round00_renders/obj_001/az000_el+00.png`

参考 mask：

- `/aaaidata/zhangqisong/data_build_runs/scene_full20_loop_20260404_code/pipeline/data/evolution_scene_v7_full20_rotation4_agent_round0_stage35_aborted_20260404/_shards/shard_gpu_0/obj_001_yaw270/round00_renders/obj_001/az000_el+00_mask.png`

从参考图测得的物体区域 HSV `V` 均值为：

- `0.414893`

构建时就是按这个目标亮度去做匹配。

## 构建参数

使用脚本：

- [build_rotation_object_brightened_dataset.py](/D:/新建文件夹/用户目录/桌面/autoresearch_vlm/Auto-claude-code-research-in-sleep/DataEvolver/DataEvolver/scripts/build_rotation_object_brightened_dataset.py)

关键参数：

- `asset_mode = symlink`
- `mask_threshold = 127`
- `mask_feather_radius = 2.0`
- `max_gain = 3.0`
- `overwrite_pair_image_fields = true`

## 输出结构

新根里会新增：

- `bright_views/`
- `bright_bbox_views/`
- `brightness_annotations/`

其中：

- `bright_views/` 是提亮后的 raw 图
- `bright_bbox_views/` 是在提亮后的 raw 图上重新绘制红色矩形 bbox 的版本

## pair 字段

训练默认读取到的：

- `source_image`
- `target_image`

已经被改成：

- `bright_bbox_views/...`

同时保留：

- `source_image_before_brighten`
- `target_image_before_brighten`
- `source_image_bright`
- `target_image_bright`
- `source_image_bright_raw`
- `target_image_bright_raw`
- `source_brightness_json`
- `target_brightness_json`

## 抽查结果

以 `obj_001 / yaw000` 为例：

- 旧 bbox 根物体区域 `V` 均值：`0.132785`
- 新 bright bbox 根物体区域 `V` 均值：`0.380575`
- 参考图物体区域 `V` 均值：`0.414893`

也就是说，这版已经明显接近参考亮度，而不是原来那种整体偏暗。

全局统计（400 张 bbox 视图）：

- 旧根 `mean object value = 0.269751`
- 新根 `mean object value = 0.378909`

## 默认使用建议

如果你现在要继续训练，优先使用：

- `/aaaidata/zhangqisong/data_build_runs/scene_full20_loop_20260404_code/pipeline/data/dataset_scene_v7_full50_rotation8_trainready_front2others_splitobj_seed42_bboxmask_brightobj_refstage35_final_20260416`

这样默认读到的 `source_image / target_image` 就已经是：

- 亮起来的物体
- 红色矩形 bbox
- 其他 split / pair 结构不变
