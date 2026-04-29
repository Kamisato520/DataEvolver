# CLAUDE Scene Data Build Addendum (2026-04-07)

本文档是对根目录 `CLAUDE.md` 的**场景数据构建主线补充**。  
目的不是替代 `CLAUDE.md`，而是把最近这段时间已经验证过的方法、代码、路径、可行工作流、已踩坑和当前真实进度集中整理出来，让新的 Claude/Codex 能直接接住当前 scene-aware 数据集工作。

如果本文档和 `CLAUDE.md` 中与 scene render 相关的内容冲突，**以本文档为准**。

---

## 1. 当前真正主线是什么

当前主线不是旧的 `2026-04-03` 那条 `obj_001/003/007/008` 小规模 rotation4 loop，也不是 LoRA 训练项目。

当前 scene-aware 数据构建主线远端代码目录是：

`/aaaidata/zhangqisong/data_build_runs/scene_full20_loop_20260404_code`

当前最核心的上游优化根目录是：

`/aaaidata/zhangqisong/data_build_runs/scene_full20_loop_20260404_code/pipeline/data/evolution_scene_v7_full20_rotation4_agent_round0_nostage35_20260404`

这条主线已经用于：

- `20` 个物体
- scene-aware Blender 渲染
- `rotation4` 上游优化
- free-form VLM review
- 最终导出一致性的 `rotation4` / `rotation8` 数据集

注意：

- **stage3.5 继续视为废弃**，不要再把地板消除引回主线。
- 当前工作重点已经从“继续无止境地跑 per-pair VLM loop”转移到了：
  - 从已有 evolution root 选稳定 base state
  - 重建**一致性更强**的 rotation 数据集

---

## 2. 已经证明正确的方法

### 2.1 错误方法

已经明确证明不适合作为最终数据集的方法有两个：

1. **每个角度 independently 选 best round**
   - 问题：同一个物体不同角度可能来自不同材质/不同状态
   - 直接后果：用户肉眼能看出 `0/90/180` 一致，但 `270` 材质漂了

2. **转相机做 multiview export**
   - 问题：这不符合当前 rotation 数据集定义
   - 当前目标是**物体旋转**，不是相机 orbit

### 2.2 正确方法

当前已验证并采用的正确方法是：

1. 对每个物体，**固定使用 `yaw000` 的最佳 base state**
2. 只修改：
   - `control_state["object"]["yaw_deg"]`
3. 保持尽可能不变：
   - 场景
   - 相机
   - 光影总体配置
   - 材质风格
4. 用这同一个 canonical base state 分别导出：
   - `rotation4`
   - `rotation8`

一句话概括：

**`yaw000 best state -> rotate object -> keep scene fixed`**

这条方法现在是 scene rotation 数据集的标准做法。

---

## 3. 已完成与正在运行的数据目录

### 3.1 旧快照版 rotation4，仅供参考

目录：

`/aaaidata/zhangqisong/data_build_runs/scene_full20_loop_20260404_code/pipeline/data/dataset_scene_v7_full20_rotation4_basic_final_20260407`

性质：

- `best-of-each-pair`
- 同一物体不同角度可能来自不同 base state
- **不能作为最终一致性数据集**

配套压缩包：

`/aaaidata/zhangqisong/data_build_runs/scene_full20_loop_20260404_code/pipeline/data/artifacts/dataset_scene_v7_full20_rotation4_basic_final_20260407.tar.gz`

用途：

- 可作为“历史最优单图快照”的参考
- 不推荐作为正式训练集版本

### 3.2 当前推荐审阅版：一致性 rotation4

目录：

`/aaaidata/zhangqisong/data_build_runs/scene_full20_loop_20260404_code/pipeline/data/dataset_scene_v7_full20_rotation4_consistent_yaw000_20260407`

这版是按正确方法重建的：

- 对每个物体固定 `yaw000` canonical base
- 只转物体
- 4 个角度：`0/90/180/270`

当前状态：

- 已完成
- `20` 个物体
- `4` 个角度
- `80` 张 RGB
- `80` 张 mask

关键文件：

- `summary.json`
- `manifest.json`
- `objects/obj_001/object_manifest.json`
- `objects/obj_009/object_manifest.json`

### 3.3 当前正在跑：一致性 rotation8

目录：

`/aaaidata/zhangqisong/data_build_runs/scene_full20_loop_20260404_code/pipeline/data/dataset_scene_v7_full20_rotation8_consistent_yaw000_20260407`

方法与一致性 `rotation4` 相同，只是目标角度扩展为：

- `0`
- `45`
- `90`
- `135`
- `180`
- `225`
- `270`
- `315`

当前运行方式：

- 远端 tmux：`export-rotation8-consistent`
- 日志：
  `/aaaidata/zhangqisong/data_build_runs/scene_full20_loop_20260404_code/runtime/full20_scene_20260404/logs/rotation8_consistent_yaw000.log`

当前已确认：

- 三卡并行在跑
- 目标目录已经开始写入对象结果
- 现阶段不要切回旧的 multiview camera export 路线

---

## 4. 当前最关键的脚本

### 4.1 一致性 rotation 导出脚本

本地：

`D:\新建文件夹\用户目录\桌面\autoresearch_vlm\Auto-claude-code-research-in-sleep\ARIS\ARIS\scripts\export_rotation8_from_best_object_state.py`

远端：

`/aaaidata/zhangqisong/data_build_runs/scene_full20_loop_20260404_code/scripts/export_rotation8_from_best_object_state.py`

用途：

- 从 evolution root 中选 base pair
- 当前标准是优先只选 `yaw000`
- 读取该 pair 的 best round `control_state`
- 只改物体 `yaw_deg`
- 固定相机/场景重新渲染

重要参数：

- `--base-rotation-deg 0`
  - 指定 canonical base 必须来自 `yaw000`
- `--fallback-to-best-any-angle`
  - 只作为兜底，不是默认标准路径

### 4.2 旧 multiview export 脚本

文件：

`D:\新建文件夹\用户目录\桌面\autoresearch_vlm\Auto-claude-code-research-in-sleep\ARIS\ARIS\scripts\export_scene_multiview_from_pair_evolution.py`

说明：

- 这个脚本我已经顺手修过 bug
- 但它不是当前推荐的最终 rotation 数据集构建方式

只有在明确要做“相机视角导出”时才考虑它。  
当前 rotation 数据集不该再优先走这条路。

### 4.3 Blender 渲染核心

文件：

`D:\新建文件夹\用户目录\桌面\autoresearch_vlm\Auto-claude-code-research-in-sleep\ARIS\ARIS\pipeline\stage4_scene_render.py`

当前与这条主线最相关的一个关键修复：

- 显式 `force_reference_material: false` 不再被阈值逻辑偷偷覆盖

这对需要保留原贴图的对象很重要，尤其是交通锥这类带明显纹理/条纹的 case。

---

## 5. 这段时间已经验证过的可行方法

### 5.1 free-form VLM review 比 rigid score 更有用

这条结论仍然有效：

- `agg.json` / `hybrid_score` 只能辅助判断
- 真正对下一轮决策最有帮助的是：
  - `reviews/*_trace.json`
  - `attempts[-1].assistant_text`

已被验证有效的工作模式：

`render -> Qwen3.5 freeform review -> 读自由文本 -> 改控制状态/渲染逻辑 -> rerender`

### 5.2 “每角各选最优”不适合作为最终版本

已经在真实数据上验证失败：

- 虽然单张图分数可能更高
- 但跨角度一致性会坏掉

对训练数据集来说，这比单图分数更致命。

### 5.3 从 canonical yaw000 派生角度是一条更稳的路径

它的优势是：

- 材质更统一
- 背景和光影更连续
- object identity 更稳定
- 数据集整体可解释性更高

这比继续对每个角单独闭环要更符合“训练集构建”的目标。

### 5.4 failure-aware stage1 restart 已经打通

虽然当前重点不在继续扩这条线，但它已经是**可用机制**，不是概念。

它现在能做到：

- 从失败 attempt 中读 trace / agg / smoke
- 自动生成 `stage1_repair_spec.json`
- 基于失败信号重写 prompt
- 再重新进入 `stage2 -> stage3 -> smoke`

当前结论：

- 机制已落地
- 对改善 `structure_consistency` 有帮助
- 但它不能单独解决所有几何/材质问题

也就是说：

- 它是有效工具
- 不是万能解法

---

## 6. 已踩过的坑

### 6.1 最容易重复踩的坑

1. 不要把旧的 `best-of-each-pair rotation4` 当最终数据集
2. 不要把 `rotation8` 理解成相机 multiview export
3. 不要再回退到“每个角度自己挑最优”的策略
4. 不要把 LoRA 项目和当前 scene data build 主线混在一起
5. 不要重新引入 `stage3.5`

### 6.2 代码和执行上的坑

1. 本地 PowerShell profile 有问题  
   `OpenSpecCompletion.ps1` 会报 parser error。  
   本地 shell 命令最好统一 `login=false`。

2. 远端 Python 解释器不要乱用  
   当前 scene 项目远端应优先使用：
   `/home/wuwenzhuo/Qwen-VL-Series-Finetune/env/bin/python`

3. 错把旧目录当 active 目录  
   当前真正 active 的 scene 根不是早期 `20260402` 小规模目录，而是：
   `/aaaidata/zhangqisong/data_build_runs/scene_full20_loop_20260404_code`

4. 旧 multiview camera export 会把思路带偏  
   当前最终 rotation 数据集定义是**物体旋转**。

### 6.3 方法层面的坑

1. 单张图高分不代表数据集一致性高
2. reviewer 说“acceptable”不等于所有角度都适合直接拼成最终训练集
3. VLM loop 的价值更偏向：
   - 找到相对稳定的 canonical base state
   - 而不是让每个角都无限单独优化

---

## 7. 当前远端检查方法

### 7.1 看 rotation8 是否还在跑

```bash
ssh wwz "tmux ls | grep export-rotation8-consistent || true"
```

### 7.2 看日志

```bash
ssh wwz "tail -n 120 /aaaidata/zhangqisong/data_build_runs/scene_full20_loop_20260404_code/runtime/full20_scene_20260404/logs/rotation8_consistent_yaw000.log"
```

### 7.3 看当前导出进度

```bash
ssh wwz "python3 - <<'PY'
from pathlib import Path
root = Path('/aaaidata/zhangqisong/data_build_runs/scene_full20_loop_20260404_code/pipeline/data/dataset_scene_v7_full20_rotation8_consistent_yaw000_20260407')
rgb = sorted([p for p in root.rglob('*.png') if '_mask' not in p.name])
mask = sorted([p for p in root.rglob('*_mask.png')])
objs = sorted([p for p in (root/'objects').glob('obj_*')]) if (root/'objects').exists() else []
print('objects', len(objs))
print('rgb', len(rgb))
print('mask', len(mask))
PY"
```

最终期望值：

- `objects = 20`
- `rgb = 160`
- `mask = 160`

---

## 8. 给新 Claude 的接手建议

如果你是新的 Claude，建议按下面顺序接手：

1. 先把 `CLAUDE.md` 里的 scene 部分当作**历史背景**
2. 再以本文档为当前 scene 主线说明
3. 首先确认：
   - 一致性 `rotation4` 已存在
   - 一致性 `rotation8` 是否已跑完
4. 如果 `rotation8` 未完成：
   - 继续监控 tmux / 日志 / 输出目录
5. 如果 `rotation8` 已完成：
   - 生成 `summary.json / manifest.json`
   - 做对象级 spot check
   - 再交给用户审阅

重点不是再去回头优化旧的 per-pair 高轮 loop，  
而是把当前这条**一致性数据集构建路线稳定收口**。

---

## 9. 一句话结论

最近这段时间最重要的进展不是“又多跑了几轮 VLM loop”，而是：

**我们已经把 scene rotation 数据集的方法，从错误的“每角各选最优 / 转相机”，纠正成了正确的“固定 yaw000 canonical base -> 旋转物体 -> 保持 scene fixed”。**

这件事比再多几轮局部调参更重要，因为它直接决定最终训练数据集是不是一致、可用、可解释。
