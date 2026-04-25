# Object Brightened Dataset Quickstart

这份说明对应脚本：

- [build_rotation_object_brightened_dataset.py](/D:/新建文件夹/用户目录/桌面/autoresearch_vlm/Auto-claude-code-research-in-sleep/DataEvolver/DataEvolver/scripts/build_rotation_object_brightened_dataset.py)

目标是：**不改原数据集，不改原配置，只基于已有 `image + mask` 生成一套“物体更亮”的并列数据集根。**

## 适用输入

支持这些已有数据集根：

- `rotation8 train-ready`
- `rotation8 split`
- `bbox-conditioned train-ready`
- `bbox-conditioned split`

脚本会保留原来的 `views/`、`objects/`、`bbox_views/`、`bbox_annotations/`，并新增：

- `bright_views/`
- `brightness_annotations/`

如果加了 `--overwrite-pair-image-fields`，新的 `pairs/*.jsonl` / `*.csv` 里：

- `source_image`
- `target_image`

会直接改成指向提亮后的图。

同时会新增保留字段：

- `source_image_before_brighten`
- `target_image_before_brighten`
- `source_image_bright`
- `target_image_bright`
- `source_brightness_json`
- `target_brightness_json`

## 提亮策略

脚本只处理 **mask 内的物体区域**：

1. 用 `mask_threshold` 得到前景物体区域
2. 计算该区域的 HSV `V` 通道平均亮度
3. 根据目标亮度计算 brightness gain
4. 只在 mask 内做提亮
5. 用 feathered mask 做软混合，避免边缘硬切

默认不会动背景。

## 两种目标亮度来源

### 1. 显式指定目标亮度

适合已经知道目标值时使用：

```bash
python scripts/build_rotation_object_brightened_dataset.py \
  --source-root /path/to/source_dataset \
  --output-dir /path/to/output_dataset \
  --target-object-value-mean 0.62 \
  --overwrite-pair-image-fields
```

### 2. 用参考图自动估计目标亮度

这更适合当前需求。你给了一张更亮的参考图，所以推荐直接用：

```bash
python scripts/build_rotation_object_brightened_dataset.py \
  --source-root /aaaidata/zhangqisong/data_build_runs/scene_full20_loop_20260404_code/pipeline/data/dataset_scene_v7_full50_rotation8_trainready_front2others_splitobj_seed42_bboxmask_final_20260414 \
  --output-dir /aaaidata/zhangqisong/data_build_runs/scene_full20_loop_20260404_code/pipeline/data/dataset_scene_v7_full50_rotation8_trainready_front2others_splitobj_seed42_bboxmask_brightobj_final_20260416 \
  --reference-image /aaaidata/zhangqisong/data_build_runs/scene_full20_loop_20260404_code/pipeline/data/evolution_scene_v7_full20_rotation4_agent_round0_stage35_aborted_20260404/_shards/shard_gpu_0/obj_001_yaw270/round00_renders/obj_001/az000_el+00.png \
  --reference-mask /aaaidata/zhangqisong/data_build_runs/scene_full20_loop_20260404_code/pipeline/data/evolution_scene_v7_full20_rotation4_agent_round0_stage35_aborted_20260404/_shards/shard_gpu_0/obj_001_yaw270/round00_renders/obj_001/az000_el+00_mask.png \
  --max-gain 1.35 \
  --mask-feather-radius 2.0 \
  --overwrite-pair-image-fields
```

## 常用参数

- `--asset-mode symlink|copy`
  - 默认 `symlink`
  - 推荐保持默认，避免重复复制大资产

- `--mask-threshold 127`
  - 用于从 mask 提取物体区域

- `--mask-feather-radius 2.0`
  - 边缘软混合半径
  - 值越大，过渡越柔和

- `--min-gain 1.0`
  - 最小亮度增益

- `--max-gain 1.35`
  - 最大亮度增益
  - 推荐先保守，不要一开始就把物体洗白

- `--gain-scale 1.0`
  - 在目标亮度比值基础上再乘一个全局系数
  - 如果觉得还不够亮，可以先试 `1.05`

- `--overwrite-pair-image-fields`
  - 打开后，训练默认读取到的就是提亮后的图

## 输出结构

新根中会出现：

- `bright_views/obj_xxx/yaw000.png`
- `brightness_annotations/obj_xxx/yaw000_brightness.json`
- `pairs/train_pairs.jsonl`
- `pairs/train_pairs.csv`
- `summary.json`
- `manifest.json`

其中 `brightness_annotations/*.json` 会记录：

- 提亮前物体平均亮度
- 提亮后物体平均亮度
- 目标亮度
- 原始 gain
- 实际应用的 gain

## 建议使用顺序

当前最稳的做法是：

1. 先从 `bboxmask split final 20260414` 派生一版 `brightobj`
2. 抽查几个对象，看是否接近你想要的“更亮但不洗白”
3. 如果视觉上合适，再直接用这版 `brightobj split` 开训

## 核心约束

- 这是**并列新根**
- 不覆盖原 `20260410` / `20260414` 数据集
- 不改原 `views/`
- 不改原 `pairs/`
- 不改原 `bbox` 配置
