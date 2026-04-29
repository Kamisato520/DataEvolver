# Scene Rotation Consistent Dataset Handoff (2026-04-07)

## TL;DR

当前 scene render 主线已经从“每个角各选各的 best round”改成了更合理的一致性方案：

- **正确方法**：对每个物体，固定使用 `yaw000` 的最佳 base state
- 然后**只旋转物体**
- **保持场景、相机、整体光影配置尽量不变**

目前状态：

1. **一致性的 `rotation4` 已完成**
2. **一致性的 `rotation8` 已启动并在远端运行**
3. 旧的 `best-of-each-pair` 版 `rotation4` 只能当参考快照，**不要当最终一致性数据集**

---

## 这段时间做了什么

### 1. 先整理了一版旧逻辑的 `rotation4` 快照

目录：

`/aaaidata/zhangqisong/data_build_runs/scene_full20_loop_20260404_code/pipeline/data/dataset_scene_v7_full20_rotation4_basic_final_20260407`

这个目录的特点是：

- 每个 pair 都从自己历史 round 里选一个当前最优轮次
- 所以**同一个 obj 的不同角度，可能来自不同 base state**

优点：

- 方便快速拿到 `80/80` pair 的 best-of-each-pair 快照
- 适合做“当前最好单图结果”的统计

缺点：

- **角度间不一致**
- 例如用户已经明确指出：
  - `obj_001` 的 `0/90/180` 材质看起来基本一致
  - 但 `270` 明显和其他三个角不一致

所以这个目录**不是最终一致性版本**。

它对应的压缩包是：

`/aaaidata/zhangqisong/data_build_runs/scene_full20_loop_20260404_code/pipeline/data/artifacts/dataset_scene_v7_full20_rotation4_basic_final_20260407.tar.gz`

---

### 2. 用户明确纠正了正确的数据集构建方式

用户要求的正确方式不是：

- 每个角都 independently 选最优图
- 也不是转相机做 multiview export

而是：

- **以 `yaw000` 的最佳渲染结果作为 canonical base state**
- 再对物体本身做 `0 / 90 / 180 / 270` 或 `0 / 45 / 90 / ... / 315` 的旋转
- **固定场景和相机**
- 同时尽量保持光影合理、一致

这意味着：

- `rotation4` 和 `rotation8` 都应该从**同一个 canonical base state** 派生
- 同一个 obj 的所有角度应该尽量共享：
  - 材质
  - 场景光影
  - 相机
  - 背景语义

---

### 3. 新增了正确的一致性导出脚本

本地脚本：

`D:\新建文件夹\用户目录\桌面\autoresearch_vlm\Auto-claude-code-research-in-sleep\ARIS\ARIS\scripts\export_rotation8_from_best_object_state.py`

远端脚本：

`/aaaidata/zhangqisong/data_build_runs/scene_full20_loop_20260404_code/scripts/export_rotation8_from_best_object_state.py`

这个脚本的逻辑是：

1. 扫描 pair evolution root：
   - `obj_xxx_yaw000`
   - `obj_xxx_yaw090`
   - `obj_xxx_yaw180`
   - `obj_xxx_yaw270`
2. **优先只选 `yaw000` 的最佳 pair state**
3. 取这个 pair 的 best round control state 作为 base
4. 对目标角度仅修改：
   - `control_state["object"]["yaw_deg"]`
5. 使用固定相机场景模板重新渲染

新增参数：

- `--base-rotation-deg 0`
  - 表示 canonical base 必须来自 `yaw000`
- `--fallback-to-best-any-angle`
  - 只有在缺失 canonical base 时才允许回退到任意角最优
  - 当前没有使用这个回退

---

## 关键代码变化

### A. 新脚本：一致性旋转导出

文件：

`D:\新建文件夹\用户目录\桌面\autoresearch_vlm\Auto-claude-code-research-in-sleep\ARIS\ARIS\scripts\export_rotation8_from_best_object_state.py`

用途：

- 统一用 canonical `yaw000` base state
- 支持重建一致性 `rotation4`
- 也支持后续扩到一致性 `rotation8`

### B. 修复旧 multiview export 脚本 bug

文件：

`D:\新建文件夹\用户目录\桌面\autoresearch_vlm\Auto-claude-code-research-in-sleep\ARIS\ARIS\scripts\export_scene_multiview_from_pair_evolution.py`

修复内容：

- `run_orchestrator()` 中引用了未定义的 `FULL48_QC_VIEWS`
- 已改成从实际模板读取 `qc_views`

说明：

- 这个脚本本身不是当前推荐的最终方案
- 但它之前会直接崩，已经顺手修掉

### C. 之前已经做过的重要修复

文件：

`D:\新建文件夹\用户目录\桌面\autoresearch_vlm\Auto-claude-code-research-in-sleep\ARIS\ARIS\pipeline\stage4_scene_render.py`

重要点：

- 显式 `force_reference_material: false` 时，不再被阈值逻辑偷偷覆盖
- 这对交通锥等“必须保留原贴图条纹”的 case 很关键

---

## 当前可用的数据目录

### 1. 旧快照版 `rotation4`，仅供参考

`/aaaidata/zhangqisong/data_build_runs/scene_full20_loop_20260404_code/pipeline/data/dataset_scene_v7_full20_rotation4_basic_final_20260407`

性质：

- best-of-each-pair
- 角度之间可能不一致
- 不应作为最终一致性数据集

### 2. 一致性 `rotation4`，当前推荐审阅版本

`/aaaidata/zhangqisong/data_build_runs/scene_full20_loop_20260404_code/pipeline/data/dataset_scene_v7_full20_rotation4_consistent_yaw000_20260407`

状态：

- 已完成

摘要：

- `20` 个物体
- `4` 个角度
- `80` 张 RGB
- `80` 张 mask

摘要文件：

- `.../summary.json`
- `.../manifest.json`

对象级示例：

- `.../objects/obj_001/object_manifest.json`
- `.../objects/obj_009/object_manifest.json`

### 3. 一致性 `rotation8`，当前正在跑

`/aaaidata/zhangqisong/data_build_runs/scene_full20_loop_20260404_code/pipeline/data/dataset_scene_v7_full20_rotation8_consistent_yaw000_20260407`

状态：

- **运行中**

远端 tmux：

- `export-rotation8-consistent`

日志：

- `/aaaidata/zhangqisong/data_build_runs/scene_full20_loop_20260404_code/runtime/full20_scene_20260404/logs/rotation8_consistent_yaw000.log`

---

## 一致性 `rotation4` 的确认结果

用户最关心的点是：

- 同一个 obj 的不同角，是否来自同一个 base state

我已经核过：

### `obj_001`

在一致性 `rotation4` 中：

- base pair = `obj_001_yaw000`
- base round = `0`
- 四个目标角：
  - `0 / 90 / 180 / 270`
  全部从这一个 canonical base 派生

### `obj_009`

在一致性 `rotation4` 中：

- base pair = `obj_009_yaw000`
- base round = `2`
- 四个目标角：
  - `0 / 90 / 180 / 270`
  也全部从这一个 canonical base 派生

这正是用户要求的行为。

---

## 为什么之前那版 4 角不行

根因不是渲染器本身坏掉，而是**选样策略错了**：

- 旧版 `rotation4_basic_final` 是：
  - 每个角各自选最优
- 这会让：
  - `yaw000` 可能来自一套材质
  - `yaw270` 可能来自另一套材质
  - 导致用户肉眼看到“前三个角基本一样，第四个角明显漂了”

所以当前标准已经改成：

**先固定 canonical base，再旋转物体。**

---

## 远端当前运行情况

### 一致性 `rotation8`

当前运行命令本质是：

- 使用远端 Python：
  - `/home/wuwenzhuo/Qwen-VL-Series-Finetune/env/bin/python`
- 调用脚本：
  - `scripts/export_rotation8_from_best_object_state.py`
- 参数：
  - `--rotations 0,45,90,135,180,225,270,315`
  - `--base-rotation-deg 0`

已经确认：

- 三张卡都进入了实际渲染
- worker 分别在 GPU0 / GPU1 / GPU2 跑
- 不会回写旧的 4 角目录

---

## Claude 下一步该做什么

### 如果用户先审 `rotation4`

优先看：

1. `obj_001`
2. `obj_009`
3. 再随机抽 `obj_002 / obj_006 / obj_020`

重点看：

- 四个角的材质是否一致
- 光影是否仍然合理
- 旋转后是否出现明显不自然的接地/阴影问题

### 如果用户确认 `rotation4` 可以

那就继续沿当前 `rotation8` 主线：

1. 等 `rotation8` 完成
2. 核 `summary.json`
3. 确认：
   - `20` objects
   - `8` rotations
   - `160` RGB
   - `160` mask
4. 再做抽样审阅

### 如果 `rotation8` 过程中出现问题

优先检查：

1. `tmux attach -t export-rotation8-consistent`
2. `rotation8_consistent_yaw000.log`
3. `.../_logs/worker_gpu_*.json`
4. `.../summary.json` 是否已生成

---

## 额外环境坑

本地 shell 有一个容易踩的坑：

- PowerShell profile：
  - `C:\Users\86159\Documents\PowerShell\OpenSpecCompletion.ps1`
- 它会产生 `ParserError`

所以当前在 shell tool 里执行远端命令时，最好：

- 使用 `login=false`

否则很容易出现：

- 本地 profile 先报错
- 但 ssh/scp 仍部分执行
- 让状态判断变得很混乱

---

## 最后的结论

现在这条主线的正确版本是：

1. **一致性 `rotation4` 已完成，可供用户审阅**
2. **一致性 `rotation8` 已启动并运行中**
3. **旧的 `rotation4_basic_final` 不应作为最终一致性数据集**

如果 Claude 接手，不要再回到：

- “每个角独立 best-of-each-pair”
- 或“转相机导出多视角”

当前唯一正确方向是：

**`yaw000 best state -> rotate object -> keep scene fixed`**
