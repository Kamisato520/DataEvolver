# Full50 Final 数据集训练启动 README（给 Claude）

## 1. 这份 README 的目的

这份文档是给后续接手训练的 Claude/Codex 用的。

目标只有一个：

- **基于最新版 `full50 final` 旋转数据集，正确启动首轮训练，不混用旧数据根，不误用还没刷新的 geomodal 版本。**

## 2. 先说结论：默认该用哪套数据

### 默认训练入口

优先使用这两个最新目录：

- **标准 train-ready 根**
  - `/aaaidata/zhangqisong/data_build_runs/scene_full20_loop_20260404_code/pipeline/data/dataset_scene_v7_full50_rotation8_trainready_front2others_final_20260410`
- **标准 object-disjoint split 根**
  - `/aaaidata/zhangqisong/data_build_runs/scene_full20_loop_20260404_code/pipeline/data/dataset_scene_v7_full50_rotation8_trainready_front2others_splitobj_seed42_final_20260410`

### 默认训练策略

如果要正式训练，**默认从 split 根开始**，不要直接从全量 train-ready 根开始。

原因：

- split 根已经做了 `object-disjoint` 划分
- 可以避免同一物体出现在 train 和 val/test 中
- 更适合作为第一版可靠实验口径

### 当前不要默认使用的目录

下面这套几何增强数据虽然可用，但**不是最新 best-state 刷新的 final 版**：

- `/aaaidata/zhangqisong/data_build_runs/scene_full20_loop_20260404_code/pipeline/data/dataset_scene_v7_full50_rotation8_geommeta_from_consistent_20260408`
- `/aaaidata/zhangqisong/data_build_runs/scene_full20_loop_20260404_code/pipeline/data/dataset_scene_v7_full50_rotation8_geomodal_trainready_front2others_20260408`
- `/aaaidata/zhangqisong/data_build_runs/scene_full20_loop_20260404_code/pipeline/data/dataset_scene_v7_full50_rotation8_geomodal_trainready_front2others_splitobj_seed42_20260408`

结论：

- **首轮训练默认先用 `20260410 final` 的标准 RGB 版**
- 如果后面明确要加 geometry/depth/normal，再单独刷新一版 `geomodal final`

## 3. 数据集定义

这版数据集是一个固定任务：

- 输入：`source image (yaw000/front view)` + `instruction`
- 输出：目标角度对应的 `target image`

目标角度一共 7 个：

- `45`
- `90`
- `135`
- `180`
- `225`
- `270`
- `315`

也就是：

- 每个 object 产生 `7` 个训练对
- `50` 个 object 共 `350` 个训练对

## 4. 最新 final 数据集规模

### full50 final train-ready

- 根目录：
  - `/aaaidata/zhangqisong/data_build_runs/scene_full20_loop_20260404_code/pipeline/data/dataset_scene_v7_full50_rotation8_trainready_front2others_final_20260410`
- 规模：
  - `50` 个 object
  - `400` 个视图
  - `400` 个 mask
  - `350` 个训练对

### full50 final split

- 根目录：
  - `/aaaidata/zhangqisong/data_build_runs/scene_full20_loop_20260404_code/pipeline/data/dataset_scene_v7_full50_rotation8_trainready_front2others_splitobj_seed42_final_20260410`
- 划分：
  - `train = 35 objects / 245 pairs`
  - `val = 7 objects / 49 pairs`
  - `test = 8 objects / 56 pairs`

## 5. 目录结构

### train-ready 根

目录结构核心是：

```text
dataset_scene_v7_full50_rotation8_trainready_front2others_final_20260410/
  summary.json
  manifest.json
  views/
    obj_001/
      yaw000.png
      yaw000_mask.png
      yaw000_render_metadata.json
      yaw000_control.json
      ...
  objects/
    obj_001/
      object_manifest.json
  pairs/
    train_pairs.jsonl
    train_pairs.csv
```

### split 根

目录结构核心是：

```text
dataset_scene_v7_full50_rotation8_trainready_front2others_splitobj_seed42_final_20260410/
  summary.json
  manifest.json
  object_splits.json
  pairs/
    train_pairs.jsonl
    val_pairs.jsonl
    test_pairs.jsonl
    all_pairs.jsonl
```

注意：

- split 根里的 `views/` 和 `objects/` 是共享资产引用
- 训练代码把它们当普通文件路径使用即可

## 6. pair 行里有哪些字段

训练时最重要的是 `pairs/*.jsonl`。

每一行至少包含这些字段：

- `pair_id`
- `obj_id`
- `task_type`
- `split`
- `source_rotation_deg`
- `target_rotation_deg`
- `source_view_name`
- `target_view_name`
- `instruction`
- `source_image`
- `target_image`
- `source_mask`
- `target_mask`
- `source_render_metadata`
- `target_render_metadata`
- `source_control_state`
- `target_control_state`

默认第一版训练最少只需要：

- `instruction`
- `source_image`
- `target_image`

如果后面要加辅助损失，再接：

- `source_mask`
- `target_mask`

## 7. 推荐的第一版训练口径

### 默认任务定义

先做最朴素、最稳的 baseline：

- 输入：
  - `source_image`
  - `instruction`
- 监督目标：
  - `target_image`

### 不建议第一版就做的事情

第一版不要同时做下面这些：

- 不要一上来就混 geometry/depth/normal
- 不要一上来就混多种 loss 设计
- 不要一上来就用全量 root 自己手拆 split
- 不要混用旧 `20260408` 的 geomodal 和新 `20260410` 的 standard final

### 推荐顺序

1. 先跑 RGB-only baseline
2. 验证 train / val loss 和可视化推理是否正常
3. 再决定是否引入：
   - mask-aware loss
   - 几何模态
   - pose/camera conditioning

## 8. 训练前必须做的 smoke test

在正式训练前，先做三个检查。

### 检查 1：summary

```bash
ssh wwz "python3 - <<'PY'
import json
path='/aaaidata/zhangqisong/data_build_runs/scene_full20_loop_20260404_code/pipeline/data/dataset_scene_v7_full50_rotation8_trainready_front2others_splitobj_seed42_final_20260410/summary.json'
print(json.dumps(json.load(open(path,'r',encoding='utf-8')), ensure_ascii=False, indent=2))
PY"
```

应该能看到：

- `train = 245`
- `val = 49`
- `test = 56`

### 检查 2：读取一条 pair

```bash
ssh wwz "python3 - <<'PY'
import json
path='/aaaidata/zhangqisong/data_build_runs/scene_full20_loop_20260404_code/pipeline/data/dataset_scene_v7_full50_rotation8_trainready_front2others_splitobj_seed42_final_20260410/pairs/train_pairs.jsonl'
with open(path,'r',encoding='utf-8') as f:
    row=json.loads(next(f))
print(json.dumps(row, ensure_ascii=False, indent=2))
PY"
```

### 检查 3：验证路径确实存在

```bash
ssh wwz "python3 - <<'PY'
import json
from pathlib import Path
root=Path('/aaaidata/zhangqisong/data_build_runs/scene_full20_loop_20260404_code/pipeline/data/dataset_scene_v7_full50_rotation8_trainready_front2others_splitobj_seed42_final_20260410')
path=root/'pairs'/'train_pairs.jsonl'
with open(path,'r',encoding='utf-8') as f:
    row=json.loads(next(f))
for key in ['source_image','target_image','source_mask','target_mask']:
    p=root/row[key]
    print(key, p.exists(), p)
PY"
```

如果这三步不通过，不要开始正式训练。

## 9. 最小 Python 读取示例

标准 `20260410 final` 数据集目前没有专门的 `Dataset` 类，所以第一版可以直接按 JSONL 读。

最小示例如下：

```python
import json
from pathlib import Path
from PIL import Image

root = Path(
    "/aaaidata/zhangqisong/data_build_runs/scene_full20_loop_20260404_code/"
    "pipeline/data/"
    "dataset_scene_v7_full50_rotation8_trainready_front2others_splitobj_seed42_final_20260410"
)

rows = []
with (root / "pairs" / "train_pairs.jsonl").open("r", encoding="utf-8") as f:
    for line in f:
        rows.append(json.loads(line))

row = rows[0]
source = Image.open(root / row["source_image"]).convert("RGB")
target = Image.open(root / row["target_image"]).convert("RGB")

instruction = row["instruction"]
print(row["pair_id"])
print(instruction)
print(source.size, target.size)
```

## 10. 如果要做 PyTorch 训练

最简单的起步方式是：

1. 自己写一个很薄的 `torch.utils.data.Dataset`
2. 内部读取 `pairs/train_pairs.jsonl`
3. 返回：
   - `instruction`
   - `source_image`
   - `target_image`
   - 可选 `source_mask` / `target_mask`

### 推荐最小返回格式

```python
{
    "pair_id": str,
    "obj_id": str,
    "instruction": str,
    "source_image": tensor,
    "target_image": tensor,
    "source_rotation_deg": int,
    "target_rotation_deg": int,
}
```

### 第一版 collate_fn 建议

先不要做复杂 packing。

第一版只需要：

- 文本字段保留为 `list[str]`
- 图像字段堆成 batch tensor

## 11. 如果后面要切到 geomodal

仓库里现成的 loader 是：

- [rotation_geomodal_dataset.py](/D:/新建文件夹/用户目录/桌面/autoresearch_vlm/Auto-claude-code-research-in-sleep/ARIS/ARIS/pipeline/rotation_geomodal_dataset.py)

检查脚本是：

- [inspect_rotation_geomodal_loader.py](/D:/新建文件夹/用户目录/桌面/autoresearch_vlm/Auto-claude-code-research-in-sleep/ARIS/ARIS/scripts/inspect_rotation_geomodal_loader.py)

但要注意：

- 当前 loader 对接的是 **geomodal 数据根**
- 而最新刷新的 `20260410 final` 目前是 **standard train-ready/split**
- 所以如果想做“最新 final + geomodal”，应该先重建一版 `geomodal final`，再把训练切过去

## 12. 训练时的工作目录和环境

远端主目录：

- `/aaaidata/zhangqisong/data_build_runs/scene_full20_loop_20260404_code`

默认 Python：

- `/home/wuwenzhuo/Qwen-VL-Series-Finetune/env/bin/python3`

建议 Claude 在远端启动训练时：

1. 先进入代码根
2. 不要修改数据集目录
3. 把实验输出写到新的并列目录，例如：
   - `runtime/training_logs/...`
   - `pipeline/data/experiments/...`
   - 或你自己新建的 `runs/...`

## 13. 推荐的首轮训练执行策略

### 第一轮

- 数据：`split/train_pairs.jsonl`
- 验证：`split/val_pairs.jsonl`
- 模态：RGB-only
- 目标：先证明训练管线、loss、验证可视化都正常

### 第二轮

在第一轮稳定后，再决定是否加：

- mask loss
- target angle embedding
- source/target control metadata
- geometry/depth/normal

## 14. 不要踩的坑

1. 不要把 `20260408` 的旧 `train-ready` 当成最新版
2. 不要把 `20260408` 的 geomodal 和 `20260410` 的 standard final 混成一套实验
3. 不要直接在 dataset 根里写缓存、中间文件、训练输出
4. 不要跳过 `object-disjoint split` 直接报结果
5. 第一轮不要同时引入太多辅助模态，否则定位问题会变困难

## 15. Claude 的默认启动建议

如果 Claude 接手，请按下面顺序做：

1. 用 `splitobj_seed42_final_20260410` 作为唯一训练入口
2. 先做 JSONL + PIL 的最小读取 smoke test
3. 先写一个最小 PyTorch `Dataset`
4. 先跑 RGB-only baseline
5. 跑通后再决定是否需要 `geomodal final`

一句话：

- **先用最新版 `20260410 final split` 跑通最小 RGB baseline，再讨论多模态增强。**
