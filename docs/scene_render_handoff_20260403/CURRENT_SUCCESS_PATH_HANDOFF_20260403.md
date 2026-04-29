# Current Success Path Handoff 2026-04-03

## 1. 这份文档的目的

这份文档只服务一个目标：

**让下一位 AI 能在不重新摸索、不重新踩坑的前提下，直接接住当前这条 scene-aware + freeform VLM loop 主线。**

这里强调的是“当前 active 主线”，但也会单独标出**用户已经明确认可、可作为兜底视觉基线**的其他结果路径。

当前应该继续推进的主线是：

```text
/aaaidata/zhangqisong/data_build/pipeline/data/evolution_scene_v7_qwen35_rotation4_agent_round0_obj001_obj003_obj007_obj008_20260402
```

### 1.1 新 session 优先阅读的补充 handoff

2026-04-03 这轮 loop 的最新进度、方法、保留结果路径、本地与服务器关键路径，已经单独整理到：

```text
memory/2026-04-03_scene_rotation4_loop_session.md
```

新开对话后，建议先读这份 memory，再读本文件。

这条主线的特点是：

- 使用现有 scene-aware Blender 渲染流程
- 有qwen image edit 2511模型根据RGBA图像（sam3模型处理后的结果图像）和渲染的背景图，完成了插入任务：将RGBA图像的物体插入到背景图，得到伪参考图（非gt），给后续claude和vlm作为“好的结果”的参考
- 使用 `Qwen3.5-35B-A3B`
- 使用 `thinking=True`
- 使用自由文本 reviewer
- 由 AI 自己读取 freeform review，再决定下一轮动作
- claude/codex（cli的模型主体）是loop的一部分，必须要实时接收来自vlm的反馈然后修改渲染的代码和参数，然后再进行渲染，再送给vlm，形成loop
- 不走“自动 controller 自己吃反馈”的旧路线
- 必须优先同时使用3张显卡一起完成任务，加快任务进程

另外，下面两条虽然不是当前正在补 backlog 的 active loop，但它们的静态结果用户已经明确说“可以接受/满意”，所以后续必须视为**可直接复用的兜底基线**：

```text
/aaaidata/zhangqisong/data_build/pipeline/data/evolution_scene_v7_pseudoref_multigpu_20260401_teachercam3_light1
/aaaidata/zhangqisong/data_build/pipeline/data/evolution_scene_v7_qwen35_freeform_full10_20260402
```

## 2. 当前最重要的结论

### 2.1 当前真正的 active 主线

当前不要再把重点放在以下这些分支上：

- `obj_002` 的 `stage1 white background -> mug -> force ceramic override` 实验线
- 更早的“完全自动 controller 自己吃反馈”的旧 freeform 线
- 任何“脚本自己自动解析 reviewer 文本，再自动选动作”的路线

这些都可以保留，但**不是当前要继续推进的 active 主线**。

当前主线是：

```text
rotation4 agent path
```

也就是：

```text
/aaaidata/zhangqisong/data_build/pipeline/data/evolution_scene_v7_qwen35_rotation4_agent_round0_obj001_obj003_obj007_obj008_20260402
```

### 2.2 用户已经认可、可作为兜底的视觉基线

除了 `rotation4 agent` 这条 active 主线，还必须把下面两条记成“用户已经认可”的静态/半静态基线。

#### A. `teachercam3_light1`：当前最稳的静态 scene render 兜底

```text
/aaaidata/zhangqisong/data_build/pipeline/data/evolution_scene_v7_pseudoref_multigpu_20260401_teachercam3_light1
```

这条路径的作用：

- 当前最稳的非旋转基础 scene render 结果
- 用户明确表示这条线“可以接受”
- 如果 rotation line 跑乱了，可以直接回退到它的渲染风格和 `best_state`

远端 summary 也支持它是稳定版本：

- `accepted = 3 / 3`
- `obj_001 final_hybrid = 0.8147`
- `obj_009 final_hybrid = 0.7910`
- `obj_010 final_hybrid = 0.8155`

关键文件：

- `.../multi_gpu_request.json`
- `.../multi_gpu_runtime.json`
- `.../scene_validation_summary.json`
- `.../obj_xxx/scene_evolution_result.json`
- `.../_shards/shard_cuda_*/obj_xxx/states/control_state_best.json`

#### B. `qwen35_freeform_full10_20260402`：视觉上可接受，但 controller 结论不能直接信

```text
/aaaidata/zhangqisong/data_build/pipeline/data/evolution_scene_v7_qwen35_freeform_full10_20260402
```

这条路径的作用：

- 用户明确说这条线的整体视觉效果“基本还可以”
- 它是 freeform + Qwen3.5 方向的重要视觉基线
- 后续完全可以从这里的 `best_state` 出发做旋转

但一定要写清楚：

- 这个目录里的 `scene_validation_summary.json` 写的是 `accepted = 0`
- 这反映的是 controller / action routing / summary 层面的问题
- **不等于最终图像全盘不可用**

这条线还暴露过旧回归：

- `select_action() got an unexpected keyword argument 'action_whitelist'`

所以对后续 AI 来说，正确理解是：

- controller-side 结论不稳定
- visual-side 结果用户仍然认可

关键文件：

- `.../multi_gpu_request.json`
- `.../multi_gpu_runtime.json`
- `.../scene_validation_summary.json`
- `.../obj_xxx/scene_evolution_result.json`
- `.../_shards/shard_cuda_*/obj_xxx/states/control_state_best.json`

### 2.3 当前的工作重点

当前重点不是再发明新的 pipeline，而是：

1. 把这条 `rotation4 agent` 主线补完整
2. 补齐 `obj_003 / obj_007 / obj_008` 的各个角度
3. 继续使用 freeform VLM review
4. 由 AI 自己读 reviewer 文本，手工决定 round1 / round2 / round3 的动作
5. 先把这条线稳定下来，再考虑别的实验

### 2.4 后续更推荐的推进方式

用户已经明确表达：

- `rotation4 agent` 这一轮整体还可以，只是有点乱
- `teachercam3_light1` 可以接受
- `qwen35_freeform_full10_20260402` 可以接受
- 后续更希望在这些已经满意的结果上继续做旋转

因此，对下一位 AI 来说，更推荐的推进方式是：

1. 把 `rotation4 agent` 当作当前 active 的旋转补完主线
2. 把 `teachercam3_light1` 当作最稳的静态 scene render 兜底
3. 把 `qwen35_freeform_full10_20260402` 当作 freeform 视觉兜底
4. 后续要扩更多角度时，优先从这两类“用户满意结果”的 `best_state` 出发
5. 尽量只改物体朝向和少量 scene 控制量，不要重新把整条静态结果链跑乱

### 2.4.1 退出原则必须收紧

用户已经明确补充了一条非常重要的经验：

- 不需要把每个 pair 都追到“特别高质量”
- **只要看起来已经自然、没有明显穿帮，就应该及时退出**
- 某些 pair 在十几轮后会开始退化，继续跑只会把结果跑坏

已确认的反例：

```text
/aaaidata/zhangqisong/data_build/pipeline/data/evolution_scene_v7_qwen35_rotation4_agent_round0_obj001_obj003_obj007_obj008_20260402/_shards/shard_gpu_2/obj_008_yaw270
```

这个 pair 到 `round48` 的结果已经明显不如 `round13_renders`。

因此后续所有 AI 必须遵守：

- **经验目标**：5 轮内没有明显提升就应该停
- **硬上限**：任何 pair 都不能超过 15 轮
- 不能再允许 40+ / 50+ round 的失控长跑

### 2.5 当前已经确认的一些事实

已经确认：

- `stage1 / stage2` 生成的参考图应该是**白底纯物体**
- 不能在 `stage1` 就把背景场景画进去
- `pseudo-reference` 不是 `stage1` 产物
- `pseudo-reference` 应该由：
  - `RGBA object`
  - `Blender background render`
  - `Qwen-Image-Edit-2511`
  来生成

也就是说：

```text
Stage1/2: 白底纯物体
Stage2.5: RGBA
Stage4 background render: 纯背景
Pseudo-reference: RGBA + background -> Qwen Image Edit 2511
Scene loop: Blender render + Qwen3.5 reviewer + AI 手动决策
```

### 2.6 这三条可接受路径的共通运行参数

当前用户认可的三条路径：

1. `evolution_scene_v7_qwen35_rotation4_agent_round0_obj001_obj003_obj007_obj008_20260402`
2. `evolution_scene_v7_pseudoref_multigpu_20260401_teachercam3_light1`
3. `evolution_scene_v7_qwen35_freeform_full10_20260402`

它们在运行层面共享一组很重要的稳定基线：

- `meshes_dir = /aaaidata/zhangqisong/data_build/pipeline/data/meshes`
- `python_bin = /home/wuwenzhuo/Qwen-VL-Series-Finetune/env/bin/python`
- `blender_bin = /home/wuwenzhuo/blender-4.24/blender`
- `profile = /aaaidata/zhangqisong/data_build/configs/dataset_profiles/scene_v7.json`
- `scene_template = /aaaidata/zhangqisong/data_build/configs/scene_template.json`
- `devices = cuda:0,cuda:1,cuda:2`
- `launch_stagger_seconds = 3.0`

这意味着：

- 后续要复现“当前满意风格”，优先保持这组底座不变
- 先不要随意切换 profile、scene template、Blender 版本、Python 环境
- 如果要扩旋转，优先在这组稳定配置上改物体 `yaw` 和少量 control state

## 3. 远端环境与工作目录

注意：这部分在@CLAUDE.md里有记录

### 3.1 远端服务器

- host: `10.160.4.185`
- port: `2223`
- user: `wuwenzhuo`
- key: `C:\Users\86159\.ssh\id_ed25519`

SSH 形式：

直接ssh wwz就能连接（推荐）

完整连接是：

```bash
ssh -i C:\Users\86159\.ssh\id_ed25519 -p 2223 wuwenzhuo@10.160.4.185
```

### 3.2 当前 scene 项目的真实远端工作目录

```text
/aaaidata/zhangqisong/data_build
```

注意：

**不要被 `AGENTS.md` 里另一个 LoRA 项目的 `/gemini/.../test/` 目录带偏。**

当前这条 scene render 主线不在那个目录里工作。

### 3.3 关键远端可执行路径

- Blender:
  ```text
  /home/wuwenzhuo/blender-4.24/blender
  ```
  
- Blender scene:
  ```text
  /home/wuwenzhuo/blender/data/sence/4.blend
  ```
  
- Python env:
  ```text
  /home/wuwenzhuo/Qwen-VL-Series-Finetune/env/bin/python
  ```
  
- Qwen3.5 reviewer model:
  ```text
  /data/wuwenzhuo/Qwen3.5-35B-A3B
  ```
  
- Qwen Image Edit 2511:

  ```
  /data/wuwenzhuo/Qwen-Image-Edit-2511
  ```

- Hunyuan3D-2.1：
  
  ```text
  /huggingface/model_hub/Hunyuan3D-2.1
  ```



## 4. 当前主线的本地代码入口

以下是当前主线真正相关的本地代码文件。

### 4.1 单步 agent render/review 入口

[scripts/run_scene_agent_step.py](/D:/新建文件夹/用户目录/桌面/autoresearch_vlm/Auto-claude-code-research-in-sleep/ARIS/ARIS/scripts/run_scene_agent_step.py)

作用：

- 对单个 `(obj, yaw)` 做一次完整 round
- 读取已有 control state
- 应用显式 actions
- 调 Blender 进行 scene-aware render
- 调 `Qwen3.5` 做 freeform review
- 写出 `agent_roundXX.json`
- 但**不会自动根据 review 再选下一轮动作**

这点非常重要：

**这个脚本只做一轮，不做自动闭环。**

### 4.2 主 loop 工具函数

[run_scene_evolution_loop.py](/D:/新建文件夹/用户目录/桌面/autoresearch_vlm/Auto-claude-code-research-in-sleep/ARIS/ARIS/run_scene_evolution_loop.py)

当前主线主要复用了这里面的：

- `load_profile(...)`
- `_resolve_reference_image(...)`
- `_resolve_pseudo_reference_image(...)`
- `_estimate_reference_rgb(...)`
- `render_scene_state(...)`
- `review_object(...)`
- `_apply_control_overrides(...)`

它不再被当成“自动 controller 主入口”，而是被 `run_scene_agent_step.py` 当作工具库调用。

### 4.3 Blender 渲染代码

[pipeline/stage4_scene_render.py](/D:/新建文件夹/用户目录/桌面/autoresearch_vlm/Auto-claude-code-research-in-sleep/ARIS/ARIS/pipeline/stage4_scene_render.py)

这是真正负责 scene-aware Blender 渲染的文件。

当前主线依赖它实现：

- `4.blend` 打开
- ground detection / ray cast placement
- object scale / offset / yaw
- preserve scene world / lights
- cycles render
- camera fill light
- mask render
- material sanitize / HSV adjust

注意：

这个文件里已经带有一些实验性材质逻辑，包括：

- reference RGB fallback material
- wood-like fallback material
- ceramic reference override 相关代码

但是：

**当前主线 `rotation4 agent` 并不依赖 `obj_002 ceramic override` 那条实验。**

### 4.4 VLM reviewer

[pipeline/stage5_5_vlm_review.py](/D:/新建文件夹/用户目录/桌面/autoresearch_vlm/Auto-claude-code-research-in-sleep/ARIS/ARIS/pipeline/stage5_5_vlm_review.py)

当前 reviewer 是这里：

- 模型：`Qwen3.5-35B-A3B`
- 支持 `thinking=True`
- 支持 freeform-first
- 支持 pseudo-reference
- 支持 previous render compare

### 4.5 动作空间

[configs/scene_action_space.json](/D:/新建文件夹/用户目录/桌面/autoresearch_vlm/Auto-claude-code-research-in-sleep/ARIS/ARIS/configs/scene_action_space.json)

当前主要在用这些动作：

- `S_CONTACT_SHADOW_UP`
- `ENV_STRENGTH_UP`
- `ENV_STRENGTH_DOWN`
- `O_SCALE_UP_10`
- `O_SCALE_DOWN_10`
- `O_LOWER_SMALL`
- `M_SATURATION_DOWN`
- `M_VALUE_DOWN`
- `M_VALUE_DOWN_STRONG`
- `M_ROUGHNESS_UP`

### 4.6 反馈动作应用

[pipeline/stage5_6_feedback_apply.py](/D:/新建文件夹/用户目录/桌面/autoresearch_vlm/Auto-claude-code-research-in-sleep/ARIS/ARIS/pipeline/stage5_6_feedback_apply.py)

当前主要复用：

- `apply_action(...)`
- `load_action_space(...)`
- `load_control_state(...)`
- `save_control_state(...)`

这里现在已经支持 `action_whitelist` 参数，之前那个回归坑已经修过。

补充说明：

上面这几份代码并不只服务 `rotation4 agent`，它们实际上也是下面两条“用户已认可静态基线”的共同代码底座：

- `evolution_scene_v7_pseudoref_multigpu_20260401_teachercam3_light1`
- `evolution_scene_v7_qwen35_freeform_full10_20260402`

所以如果后续是“从满意静态结果继续做旋转”，最优先复用/检查的仍然是这里这一套代码，而不是重新另起一条渲染栈。

## 5. 远端代码对应位置

下一位 AI 如果要在服务器上直接跑，真正执行的是远端这些文件：

- `/aaaidata/zhangqisong/data_build/scripts/run_scene_agent_step.py`
- `/aaaidata/zhangqisong/data_build/run_scene_evolution_loop.py`
- `/aaaidata/zhangqisong/data_build/pipeline/stage4_scene_render.py`
- `/aaaidata/zhangqisong/data_build/pipeline/stage5_5_vlm_review.py`
- `/aaaidata/zhangqisong/data_build/pipeline/stage5_6_feedback_apply.py`
- `/aaaidata/zhangqisong/data_build/configs/scene_action_space.json`
- `/aaaidata/zhangqisong/data_build/configs/scene_template.json`
- `/aaaidata/zhangqisong/data_build/configs/dataset_profiles/scene_v7.json`

重要提醒：

**本地改完代码，如果不同步到这些远端精确路径，远端运行根本不会生效。**

之前已经踩过一次坑：

- 错把文件 `scp` 到 `/aaaidata/zhangqisong/data_build/` 根目录
- 结果远端跑的还是旧文件

以后同步必须精确到目标文件路径。

### 5.1 三条可接受路径各自最值得直接看的配置文件

#### `rotation4 agent`

- `.../rotation_dataset_request.json`
- `.../rotation_dataset_summary.json`
- `.../_meta/roundN_actions_manifest.json`
- `.../_monitor_logs/monitor_gpu_*.json`

最核心的作用：

- 看当前旋转 backlog 是怎么分配到三张卡上的
- 看每个 round 实际下了哪些动作
- 看 monitor 是否还在继续推进

#### `teachercam3_light1`

- `.../multi_gpu_request.json`
- `.../multi_gpu_runtime.json`
- `.../scene_validation_summary.json`

最核心的作用：

- 看这条最稳静态基线到底用了什么运行底座
- 看 `accepted = 3/3` 的 summary
- 看每个对象的 `best_state_path`

#### `qwen35_freeform_full10_20260402`

- `.../multi_gpu_request.json`
- `.../multi_gpu_runtime.json`
- `.../scene_validation_summary.json`

最核心的作用：

- 看 freeform + Qwen3.5 full10 当时的运行参数
- 同时记住它的 summary 不能直接等价于人眼质量
- 从里面提取仍然可复用的 `best_state_path`

## 6. 当前主线的输出根目录

当前主线输出根目录：

[rotation4_agent_root](/aaaidata/zhangqisong/data_build/pipeline/data/evolution_scene_v7_qwen35_rotation4_agent_round0_obj001_obj003_obj007_obj008_20260402)

其结构大致是：

```text
_meta/
_logs/
_shards/
  shard_gpu_0/
  shard_gpu_1/
  shard_gpu_2/
```

每个 pair 目录结构：

```text
obj_xxx_yawYYY/
  agent_round00.json
  agent_round01.json
  ...
  round00_renders/
  round01_renders/
  ...
  reviews/
    obj_xxx_r00_agg.json
    obj_xxx_r00_az000_el+00_trace.json
    ...
  states/
    round00.json
    round00_input.json
    round01.json
    round01_input.json
    ...
```

### 6.1 最有用的文件

对你来说，最有用的是：

1. `agent_roundXX.json`
   - 这一轮实际上执行了什么
2. `reviews/obj_xxx_rXX_agg.json`
   - 聚合结论
3. `reviews/obj_xxx_rXX_az000_el+00_trace.json`
   - 真正的 freeform review 全文本
4. `states/roundXX.json`
   - 下一轮继续时要接的 control state
5. `roundXX_renders/obj_xxx/az000_el+00.png`
   - 这一轮主图

## 7. 当前主线的最新进度快照

以下进度快照对应我停笔时的远端状态。

### 7.1 当前 tmux 会话

当前还在运行的补跑会话：

- `rot4-resume-g0`
- `rot4-resume-g1`
- `rot4-resume-g2`

查看方式：

```bash
tmux attach -t rot4-resume-g0
tmux attach -t rot4-resume-g1
tmux attach -t rot4-resume-g2
```

如果只是看最后几行：

```bash
tmux capture-pane -pt rot4-resume-g0:0 | tail -n 60
tmux capture-pane -pt rot4-resume-g1:0 | tail -n 60
tmux capture-pane -pt rot4-resume-g2:0 | tail -n 60
```

### 7.2 各 pair 当前状态

截至我写文档时，远端的状态大致是：

- `obj_001_yaw000`: 已到 `r10`
- `obj_001_yaw090`: 已到 `r09`
- `obj_001_yaw180`: 已到 `r09`
- `obj_001_yaw270`: 已到 `r01`

- `obj_003_yaw000`: 已到 `r01`
- `obj_003_yaw090`: 已到 `r01`
- `obj_003_yaw180`: 已到 `r01`
- `obj_003_yaw270`: 已到 `r01`

- `obj_007_yaw000`: 已到 `r01`
- `obj_007_yaw090`: 至少有 `r00`
- `obj_007_yaw180`: 至少有 `r00`
- `obj_007_yaw270`: 至少有 `r00`

- `obj_008_yaw180`: 至少有 `r00`
- `obj_008_yaw000 / yaw090 / yaw270`: 正在补

### 7.3 已确认的几条代表性结果

#### `obj_001_yaw270`

`r00 -> r01`

- `0.5356 -> 0.5116`
- 说明这组动作没有带来收益

参考：

```text
/aaaidata/zhangqisong/data_build/pipeline/data/evolution_scene_v7_qwen35_rotation4_agent_round0_obj001_obj003_obj007_obj008_20260402/_shards/shard_gpu_0/obj_001_yaw270/reviews/obj_001_r00_agg.json
/aaaidata/zhangqisong/data_build/pipeline/data/evolution_scene_v7_qwen35_rotation4_agent_round0_obj001_obj003_obj007_obj008_20260402/_shards/shard_gpu_0/obj_001_yaw270/reviews/obj_001_r01_agg.json
```

#### `obj_003_yaw000`

`r00 -> r01`

- `0.4927 -> 0.5605`
- 这是一个真实正向例子
- 说明对 desk lamp 来说，`scale up + contact shadow up` 是有效的

参考：

```text
/aaaidata/zhangqisong/data_build/pipeline/data/evolution_scene_v7_qwen35_rotation4_agent_round0_obj001_obj003_obj007_obj008_20260402/_shards/shard_gpu_1/obj_003_yaw000/reviews/obj_003_r00_agg.json
/aaaidata/zhangqisong/data_build/pipeline/data/evolution_scene_v7_qwen35_rotation4_agent_round0_obj001_obj003_obj007_obj008_20260402/_shards/shard_gpu_1/obj_003_yaw000/reviews/obj_003_r01_agg.json
```

#### `obj_003_yaw090`

`r00 -> r01`

- `0.5551 -> 0.5113`
- 同样是 lamp，但这个角度反而变差

这说明：

**不能把一个 yaw 上有效的动作，原封不动套给另一个 yaw。**

#### `obj_007_yaw000`

`r00 -> r01`

- `0.5457 -> 0.5444`
- 几乎没有收益
- 当前主要问题仍然是：
  - `flat_lighting`
  - `shadow_missing`
  - `color_shift`

#### `obj_008_yaw180`

当前已有 `r00`

- `hybrid_score = 0.5109`
- 问题主要是：
  - `flat_lighting`
  - `weak_subject_separation`
  - `color_shift`

## 8. 当前主线真正的运行方式

### 8.1 不是自动 controller loop

当前主线不是这样：

```text
render -> review -> 自动 parse -> 自动 apply -> 下一轮
```

当前主线应该是这样：

```text
render -> freeform review -> AI 读 reviewer 文本 -> AI 自己决定动作 -> 再跑下一轮
```

也就是：

- `run_scene_agent_step.py` 只负责单轮
- 真正的闭环决策在 AI 手里

### 8.2 正确的工作节奏

对任意一个 pair，正确节奏是：

1. 先看：
   - `reviews/obj_xxx_rNN_agg.json`
   - `reviews/obj_xxx_rNN_az000_el+00_trace.json`
2. 判断：
   - 这一轮到底是在变好还是变差
   - 主问题到底是：
     - size
     - grounding
     - shadow
     - flat lighting
     - color shift
     - subject separation
3. 再AI手工决定下一轮动作
4. 再调用一次 `run_scene_agent_step.py`

## 9. VLM loop 阶段怎么接

这是最重要的一节。

### 9.1 一定要看 `trace.json`，不能只看 `agg.json`

`agg.json` 的作用：

- 给你一个结构化摘要
- 方便快速统计和排序

但真正能指导下一轮动作的是：

```text
*_trace.json
```

尤其是里面 `attempts[0].assistant_text`

这个字段里才有：

- reviewer 真正的自由文本分析
- 它怎么描述当前图像的问题
- 它为什么推荐某些动作
- 它是在说“太小”、“太暗”、“太平”、“太贴图感”，还是“位置不对”

### 9.2 不要机械照搬 `suggested_actions`

`agg.json` 里的：

- `suggested_actions`
- `advisor_actions`

只能当参考，不能机械照搬。

原因：

- 不同角度的问题可能不同
- 同一个动作在另一个 yaw 上可能是负收益
- 有时 reviewer 文本比离散 action 更准确

典型例子：

- `obj_003_yaw000` 的 round1 有效
- `obj_003_yaw090` 的 round1 反而变差

这说明：

**同一对象不同角度不能直接共用动作。**

### 9.3 正确读法

推荐顺序：

1. 先读 `assistant_text`
2. 再对照 `agg.json`
3. 再看当前 render 图

尤其注意 reviewer 是否在说：

- `object_too_small`
- `shadow_missing`
- `flat_lighting`
- `weak_subject_separation`
- `ground_intersection`
- `scene_light_mismatch`

### 9.4 当前主线里常见的动作模式

#### 对 desk lamp (`obj_003`)

常见有效动作：

- `O_SCALE_UP_10`
- `S_CONTACT_SHADOW_UP`
- 有时加 `ENV_STRENGTH_UP`

但不是每个 yaw 都有效。

#### 对 cat (`obj_007`)

当前 reviewer 常说：

- 贴上去
- 光影太平
- 阴影不自然
- 色彩略不协调

所以常见动作是：

- `M_VALUE_DOWN`
- `S_CONTACT_SHADOW_UP`
- 有时 `M_SATURATION_DOWN`

但如果阴影已经太硬，就不要继续无脑堆 `S_CONTACT_SHADOW_UP`。

#### 对 chair (`obj_001`)

当前已经跑很多轮，收益在下降。

对 `obj_001` 来说，当前最大的经验不是“继续加动作”，而是：

**收益已经明显进入递减区。**

所以继续做 `obj_001` 时，要非常谨慎，不要再无脑堆很多 round。

## 10. 继续当前主线时的推荐操作顺序

### 10.1 第一优先级

先把这几个 pair 看完：

- `obj_003_yaw180`
- `obj_003_yaw270`
- `obj_007_yaw000`

因为它们已经拿到了新的 `r01`。

先读：

```text
.../reviews/obj_xxx_r01_agg.json
.../reviews/obj_xxx_r01_az000_el+00_trace.json
```

判断 round1 是否值得继续到 round2。

### 10.2 第二优先级

等 `obj_007_yaw090 / 180 / 270` 的 `r00` 都出来后：

- 先读 trace
- 再决定它们各自的 round1

### 10.3 第三优先级

等 `obj_008` 的 `r00` 全部出来后：

- 先做每个 yaw 的 round0 诊断
- 再决定是否要统一 round1 策略，还是按 yaw 分开

## 11. 不要再踩的坑

这一节非常重要。

### 11.1 Stage1 不要带背景

这是已经被用户明确指出并纠正过的。

错误做法：

```text
stage1/stage2 直接生成“物体 + 场景背景”
```

正确做法：

```text
stage1/stage2 生成白底纯物体
```

### 11.2 Pseudo-reference 不能替代 stage1 reference

正确语义：

- `reference_image_path`：物体身份参考
- `pseudo_reference_path`：场景融合粗参考

不能混成一个东西。

### 11.3 不要继续推进 `obj_002` 分支

当前至少在这份文档对应时刻：

**不要把注意力放回 `obj_002 mug` 那条实验线上。**

那条线的问题包括：

- stage3 贴图本身偏成蓝黑色
- 后面衍生出材质 override 实验
- 会干扰当前主线节奏

当前任务是先把 `rotation4 agent` 主线补稳。

### 11.4 不要把本地改动忘了同步到远端

这是非常容易再犯的错误。

原则：

- 本地代码修改后
- 必须 `scp` 到远端**精确文件路径**
- 不能只改本地然后直接跑远端

### 11.5 `CUDA_VISIBLE_DEVICES` 和 `--device` 的关系

这是一个非常容易犯的坑。

如果你在 tmux 里这样跑：

```bash
CUDA_VISIBLE_DEVICES=1 python ...
```

那么进程内部看到的 GPU 索引会重映射。

因此 reviewer 脚本里一般要传：

```bash
--device cuda:0
```

不是 `cuda:1`。

因为对那个子进程来说，可见 GPU 只有一张，它就是 `cuda:0`。

### 11.6 不要依赖旧的自动 freeform loop

不要优先使用这类目录对应的旧自动线：

```text
evolution_scene_v7_qwen35_freeform_full10_20260402
```

原因不是它完全没价值，而是：

- 它混合了 controller 自动逻辑
- 历史上还有 `action_whitelist` 回归
- 现在这条 `rotation4 agent` 主线更清楚，也更符合用户当前要求

### 11.7 不要把 `accepted=0` 直接等同于“视觉完全失败”

这个坑现在很关键，尤其是：

```text
/aaaidata/zhangqisong/data_build/pipeline/data/evolution_scene_v7_qwen35_freeform_full10_20260402
```

这个目录在 controller / summary 层面确实有问题：

- `accepted = 0`
- 有 `action_whitelist` 旧回归

但用户已经明确说明：

- 它的整体视觉结果仍然是可以接受的

因此下一位 AI 必须明确区分：

- controller-side failure
- visual-side progress

不能因为 summary 很差，就把整条线一票否决。

### 11.8 不要忽略已经满意的静态结果可以直接作为旋转起点

当前很容易再次犯的错误是：

- 一想到“要做旋转”，就从更早、更差的静态版本重新起一条 pipeline

更合理的做法是：

- 优先从 `teachercam3_light1` 和 `qwen35_freeform_full10_20260402` 中挑用户已经满意的对象
- 直接复用它们的 `best_state_path` 或最终 control state
- 再做旋转扩展

因为用户已经明确表达：

- 后续更希望“在满意的静态结果上做旋转”
- 不想重新再证明一次静态基线能不能做对

### 11.9 不要把 active 主线和视觉兜底线混为一谈

现在要同时记住三件事：

1. `rotation4 agent`
   当前正在继续推进、会持续读 VLM 文本并补 round 的 active 旋转主线

2. `teachercam3_light1`
   当前最稳的静态 scene render 兜底

3. `qwen35_freeform_full10_20260402`
   当前视觉上也被用户认可，但 controller 叙事不可信的 freeform 静态线

如果把这三者混成一条，后续决策会再次混乱。

### 11.10 不要随意改掉这三条已认可路径的基础参数

至少在下一阶段，以下配置默认应视为锁定基线：

- Blender: `/home/wuwenzhuo/blender-4.24/blender`
- scene: `/home/wuwenzhuo/blender/data/sence/4.blend`
- Python: `/home/wuwenzhuo/Qwen-VL-Series-Finetune/env/bin/python`
- reviewer: `/data/wuwenzhuo/Qwen3.5-35B-A3B`
- scene profile: `/aaaidata/zhangqisong/data_build/configs/dataset_profiles/scene_v7.json`
- scene template: `/aaaidata/zhangqisong/data_build/configs/scene_template.json`
- GPU strategy: `cuda:0,1,2`

如果新 AI 一上来就改这些底座参数，很容易把当前已经满意的风格跑丢。

### 11.11 不要只看分数

用户已经明确要求：

**人眼判断优先。**

分数只作为辅助。

所以：

- `hybrid_score` 有用
- 但不能只看它
- 必须同时看图 + 看 trace

## 12. 推荐给下一位 AI 的最短接力流程

如果下一位 AI 只想最短时间接上当前工作，按下面做：

1. SSH 到远端
2. 查看：
   - `tmux ls | grep rot4-resume`
3. 对已经出新结果的 pair：
   - 先读 `r01_agg.json`
   - 再读 `r01_trace.json`
   - 再看主图
4. 决定下一轮动作
5. 用 `scripts/run_scene_agent_step.py` 跑单 pair 的下一轮
6. 所有结果继续写回：
   - `rotation4_agent_root`

## 13. 典型命令模板

### 13.1 查看某个 pair 的聚合 review

```bash
cat /aaaidata/zhangqisong/data_build/pipeline/data/evolution_scene_v7_qwen35_rotation4_agent_round0_obj001_obj003_obj007_obj008_20260402/_shards/shard_gpu_1/obj_003_yaw000/reviews/obj_003_r01_agg.json
```

### 13.2 查看 trace

```bash
cat /aaaidata/zhangqisong/data_build/pipeline/data/evolution_scene_v7_qwen35_rotation4_agent_round0_obj001_obj003_obj007_obj008_20260402/_shards/shard_gpu_1/obj_003_yaw000/reviews/obj_003_r01_az000_el+00_trace.json
```

### 13.3 再跑一轮

```bash
CUDA_VISIBLE_DEVICES=1 /home/wuwenzhuo/Qwen-VL-Series-Finetune/env/bin/python \
  /aaaidata/zhangqisong/data_build/scripts/run_scene_agent_step.py \
  --output-dir /aaaidata/zhangqisong/data_build/pipeline/data/evolution_scene_v7_qwen35_rotation4_agent_round0_obj001_obj003_obj007_obj008_20260402/_shards/shard_gpu_1/obj_003_yaw000 \
  --obj-id obj_003 \
  --rotation-deg 0 \
  --round-idx 2 \
  --actions O_SCALE_UP_10 S_CONTACT_SHADOW_UP \
  --control-state-in /aaaidata/zhangqisong/data_build/pipeline/data/evolution_scene_v7_qwen35_rotation4_agent_round0_obj001_obj003_obj007_obj008_20260402/_shards/shard_gpu_1/obj_003_yaw000/states/round01.json \
  --prev-renders-dir /aaaidata/zhangqisong/data_build/pipeline/data/evolution_scene_v7_qwen35_rotation4_agent_round0_obj001_obj003_obj007_obj008_20260402/_shards/shard_gpu_1/obj_003_yaw000/round01_renders \
  --device cuda:0 \
  --meshes-dir /aaaidata/zhangqisong/data_build/pipeline/data/meshes \
  --agent-note manual_round2_continue
```

## 14. 最后一句给下一位 AI

如果你要继续这条线，最重要的是记住：

**当前主线不是“自动化更强的脚本”，而是“AI 自己读 VLM 自由文本，再手动决策动作”的半自动闭环。**

不要再退回：

- 全自动 controller 盲跑
- 只看分数不看图
- stage1 带背景
- 把 pseudo-reference 当 GT
- 在别的分支上发散

先把：

```text
rotation4_agent_root
```

这条线稳稳补齐，再说别的。
