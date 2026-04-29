# Pipeline 修改报告

日期：2026-03-29

## 背景

本轮修改针对本地 `pipeline/` 五阶段脚本做稳定性修复。目标不是扩展功能，而是先解决几个会直接导致“流水线看起来成功、但产物实际上不完整”的问题，并补齐一份可复用的本地说明。

当前本地版本仍然是基础五阶段：

1. `stage1_text_expansion.py`
2. `stage2_t2i_generate.py`
3. `stage3_image_to_3d.py`
4. `stage4_blender_render.py` / `stage4_batch_render.sh`
5. `stage5_merge_metadata.py`

## 修改文件

- `pipeline/run_all.sh`
- `pipeline/stage2_t2i_generate.py`
- `pipeline/stage3_image_to_3d.py`
- `pipeline/stage4_batch_render.sh`
- `pipeline/stage4_blender_render.py`

## 修改内容

### 1. Stage 2 / Stage 3 补上失败退出

修改文件：

- `pipeline/stage2_t2i_generate.py`
- `pipeline/stage3_image_to_3d.py`

问题：

- 原逻辑虽然会在末尾调用 `verify_outputs(...)`，但只是打印 warning，不会让脚本返回非零退出码。
- `run_all.sh` 只根据子进程退出码判断阶段是否成功，因此一旦有图片或 mesh 缺失，Stage 2 / Stage 3 仍会被记成成功。

修改：

- 当 `verify_outputs(...)` 返回 `False` 时，显式执行 `sys.exit(1)`。

效果：

- 一旦阶段产物不完整，流水线会在对应阶段立刻失败，不会把坏状态继续传给后续 Stage 4 / Stage 5。

### 2. 调整 `run_all.sh` 对 Stage 1 的默认行为

修改文件：

- `pipeline/run_all.sh`

问题：

- `run_all.sh` 默认总是先执行 Stage 1。
- 但 `stage1_text_expansion.py` 明确依赖 `ANTHROPIC_API_KEY`，而且文档说明更接近“本地先生成 prompts，再上传/复用”。
- 这会导致远程或离线环境里，一键流水线入口并不自洽。

修改：

- 新增 `--skip-stage1`
- 新增 `--force-stage1`
- 默认逻辑改为：
  - 如果显式传了 `--skip-stage1`，直接跳过 Stage 1
  - 如果显式传了 `--force-stage1`，强制执行 Stage 1
  - 如果环境变量 `ANTHROPIC_API_KEY` 存在，执行 Stage 1
  - 如果没有 key，但已有 `pipeline/data/prompts.json`，则复用现有 prompts 并跳过 Stage 1
  - 如果既没有 key，也没有 `prompts.json`，则直接报错退出，并提示正确用法

效果：

- `run_all.sh` 现在更符合“本地可生成 prompts，远程可复用 prompts”的实际工作方式。
- 流水线默认入口不会再因为缺少外部 API 配置而无提示失败。

### 3. 放宽 Stage 4 默认机位，并支持参数化控制

修改文件：

- `pipeline/stage4_blender_render.py`
- `pipeline/stage4_batch_render.sh`

问题：

- 原始默认相机距离较近，而脚本又会把物体归一化到接近 `2.0` 的尺度，配合 50mm 镜头，容易在高俯仰角或细长物体上出现裁边。
- 这是系统性参数问题，不是个别样本问题。

修改：

- 将默认 `CAMERA_DISTANCE` 从 `2.5` 调整为 `3.5`
- 新增 `--camera-distance` 参数
- `stage4_batch_render.sh` 会把该参数传给 Blender 脚本
- `stage4_blender_render.py` 会把实际使用的 `camera_distance` 写入每个对象 metadata 以及 `render_summary.json`

效果：

- 默认构图更保守，降低裁边概率。
- 后续如果需要继续调机位，不需要改源码，只要改 shell 参数或环境变量即可。

### 4. Stage 4 渲染结果校验从“警告”升级为“失败”

修改文件：

- `pipeline/stage4_batch_render.sh`

问题：

- 原逻辑在某个对象少于 48 张渲染图时只打印 warning，不会让阶段失败。
- 这会让不完整的渲染结果继续流入 Stage 5 合并元数据。

修改：

- 统计 `INCOMPLETE_OBJS`
- 若任意对象渲染数量少于 48，则脚本退出码改为失败

效果：

- Stage 4 现在不仅检查“目录是否存在”，也检查“视角是否完整”。
- 数据集产出更接近“全量配对”要求。

## 本轮验证

已执行：

```powershell
python -m py_compile pipeline\stage1_text_expansion.py pipeline\stage2_t2i_generate.py pipeline\stage3_image_to_3d.py pipeline\stage3_5_mesh_sanitize.py pipeline\stage4_blender_render.py pipeline\stage5_merge_metadata.py
```

结果：

- 语法校验通过

## 当前仍未覆盖的部分

以下内容本轮没有直接改：

- 没有在本地完整跑通一次 Stage 1 到 Stage 5
- 没有在远程服务器环境上做端到端实跑验证
- 没有改动更高层的 ARIS skill 编排
- 没有处理远程扩展版流水线中额外出现的 VLM review / feedback apply 逻辑

## 剩余风险

### 1. Stage 4 机位参数仍然是经验值

这次把默认距离从 `2.5` 改到 `3.5`，主要是为了减少明显裁边，但它仍然不是基于物体真实投影自动适配的自适应策略。

如果后续遇到极端长条物体、超薄物体或带外伸结构的 mesh，仍可能需要：

- 继续加大 `camera_distance`
- 或改成基于包围盒/视锥的自适应相机距离

### 2. 当前本地 pipeline 仍是基础版

本地 `pipeline/` 和远程服务器上已经出现的扩展版不完全一致。也就是说，本次修改提高了当前本地基础版的稳定性，但不等价于已经覆盖远程完整实验链路的全部问题。

## 结论

本轮修改的核心作用是把几个“静默失败”点收紧：

- Stage 2 / Stage 3 产物不完整时不再假成功
- Stage 1 默认入口更符合实际环境约束
- Stage 4 默认构图更稳，并支持外部调参
- Stage 4 不完整渲染不再继续流入 Stage 5

如果继续往下推进，下一步更值得做的是：

1. 在本地或服务器上实际跑一轮 `--from-stage 2`
2. 检查 Stage 4 新机位下的边界样本
3. 再决定是否把机位从固定常数改成自适应计算
