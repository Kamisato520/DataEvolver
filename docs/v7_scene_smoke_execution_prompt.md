# v7 Scene Smoke Execution Prompt

这份文档用于在代码已同步到服务器后，直接发给 Claude 执行 `v7` 场景化 Blender 自动调优的最小 smoke 验证。

## 任务目标

不要继续做设计，也不要再改代码。直接去远端服务器执行 `v7` scene loop 的最小 smoke 验证，并把结果整理返回。

工作目录：

```bash
/aaaidata/zhangqisong/data_build
```

使用环境：

- Python:
  `/home/wuwenzhuo/Qwen-VL-Series-Finetune/env/bin/python3`
- Blender:
  `/home/wuwenzhuo/blender-4.24/blender`

这次只做执行与检查，不做新设计。

---

## 一、先做前置检查

请先确认以下资源存在：

1. 场景资产：

```bash
/home/wuwenzhuo/blender/data/sence/4.blend
```

2. mesh：

- `pipeline/data/meshes/obj_009.glb`
- `pipeline/data/meshes/obj_001.glb`
- `pipeline/data/meshes/obj_004.glb`

3. 配置文件：

- `configs/scene_template.json`
- `configs/dataset_profiles/scene_v7.json`

如果缺任何一个，请停止并明确说明缺失项，不要继续执行。

---

## 二、先跑单对象 scene render 检查

先只跑 `obj_009`，验证 scene 渲染器本身能否正常工作。

输出目录：

```bash
pipeline/data/scene_smoke_render_obj009
```

如果目录已存在，可以先删除再跑。

执行命令：

```bash
/home/wuwenzhuo/blender-4.24/blender -b -P pipeline/stage4_scene_render.py -- \
  --input-dir pipeline/data/meshes \
  --output-dir pipeline/data/scene_smoke_render_obj009 \
  --obj-id obj_009 \
  --resolution 512 \
  --engine EEVEE \
  --scene-template configs/scene_template.json
```

跑完后检查：

1. `pipeline/data/scene_smoke_render_obj009/obj_009/` 下是否有：
   - 4 张 RGB 图
   - 4 张 `_mask.png`
   - `metadata.json`

2. 读取 `metadata.json`，重点看：
   - `support_plane_name`
   - `contact_gap`
   - `penetration_depth`
   - `is_floating`
   - `is_intersecting_support`
   - `is_out_of_support_bounds`

3. 如果单对象渲染失败，请停止，不要继续 smoke，并把 stderr 或关键报错整理出来。

---

## 三、如果单对象通过，再跑 3-object smoke

输出目录：

```bash
pipeline/data/evolution_scene_v7_smoke
```

如果目录已存在，可以先删除再跑。

执行命令：

```bash
/home/wuwenzhuo/Qwen-VL-Series-Finetune/env/bin/python3 run_scene_evolution_loop.py \
  --meshes-dir pipeline/data/meshes \
  --output-dir pipeline/data/evolution_scene_v7_smoke \
  --obj-ids obj_009 obj_001 obj_004 \
  --device cuda:0 \
  --blender /home/wuwenzhuo/blender-4.24/blender \
  --profile configs/dataset_profiles/scene_v7.json \
  --scene-template configs/scene_template.json
```

---

## 四、跑完后请检查并汇报

请读取并总结这些文件：

1. 总 summary：

```bash
pipeline/data/evolution_scene_v7_smoke/scene_validation_summary.json
```

2. 每对象结果：

- `pipeline/data/evolution_scene_v7_smoke/obj_009/scene_evolution_result.json`
- `pipeline/data/evolution_scene_v7_smoke/obj_001/scene_evolution_result.json`
- `pipeline/data/evolution_scene_v7_smoke/obj_004/scene_evolution_result.json`

3. 同时检查每个对象目录下是否真的产出了：

- baseline/probe render
- attempt render（如有）
- reviews
- best_state

---

## 五、输出要求

请不要只贴原始日志。请按下面结构汇报：

### 1. 前置检查结果

- 哪些文件存在
- 是否可以开始跑

### 2. 单对象 scene render 结果（obj_009）

- 成功 / 失败
- 产物是否齐全
- metadata 里关键 physics 指标是什么
- support plane 是否看起来合理

### 3. 3-object smoke 结果

对 `obj_009 / obj_001 / obj_004` 分别说明：

- confirmed / final
- baseline zone
- exit_reason
- 是否有 attempt
- 是否出现 render hard fail / unstable / reject

### 4. 关键结论

明确回答：

- scene renderer 能不能跑通
- scene loop 能不能跑通
- 当前 smoke 是“实现已可用”还是“仍有 blocker”
- 如果有 blocker，请指出最关键的那个，不要泛泛而谈

---

## 六、注意事项

- 这一步不要再改代码，除非你发现一个极小且明确的运行时 bug，并且不修就完全无法继续
- 如果你确实修改了代码，请明确说明改了什么、为什么改
- 重点是执行验证，不是重新设计方案
- 如果单对象 scene render 都过不了，请不要继续跑 3-object smoke，先停在单对象阶段并汇报
