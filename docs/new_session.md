这是一个新的 session。你现在不要从零开始想，也不要重新发明流程。你要先快速跟上我们当前在 ARIS 里关于 scene-aware Blender 渲染 + Qwen3.5 自由文本 VLM loop 的实际进度，然后继续沿当前主线工作。

先强调几件事：

1. 你现在接的是“正在运行中的 scene render / VLM loop 主线”，不是 LoRA 训练项目。
2. 你不能主动退出 loop。
3. 在 VLM loop 阶段，你必须实时接收来自 VLM 的自由文本反馈，然后再去修改 Blender 渲染参数/配置。
4. 不能只看分数，更不能只看 `hybrid_score`。
5. 只有当 VLM 明确给出“可以了 / keep / acceptable”之类的结论时，某个 pair 才应该停止。
6. 旧的自动 controller 只能当参考，当前主线是“AI 自己读自由文本 -> AI 自己决定下一轮动作”。

--------------------------------
一、先看哪些文档
--------------------------------

本地仓库：
`D:\新建文件夹\用户目录\桌面\autoresearch_vlm\Auto-claude-code-research-in-sleep\ARIS\ARIS`

你先按顺序阅读：

1. `docs/scene_render_handoff_20260403/README.md`
2. `docs/scene_render_handoff_20260403/CURRENT_SUCCESS_PATH_HANDOFF_20260403.md`

如果还需要补上下文，再看：

3. `docs/scene_render_handoff_20260402/README.md`
4. `docs/scene_render_handoff_20260402/SCENE_RENDER_WORK_SUMMARY_20260402.md`
5. `docs/scene_render_handoff_20260402/VERSION_EVOLUTION_V2_TO_V7.md`
6. `docs/scene_render_handoff_20260402/WORKFLOW_SUMMARIES.md`
7. `docs/scene_render_handoff_20260402/REMOTE_RESULT_FOLDER_BRIEFS.md`

说明：
- v2-v6、v7 前期版本你只需要知道它们是历史演进，不是当前主线。
- 重点是理解“现在真正继续工作的路径”和“用户已经认可的视觉兜底路径”。

--------------------------------
二、当前最重要的结果路径
--------------------------------

当前真正持续写入、正在继续跑的主线目录是：

`/aaaidata/zhangqisong/data_build/pipeline/data/evolution_scene_v7_qwen35_rotation4_agent_round0_obj001_obj003_obj007_obj008_20260402`

这个目录是当前 active 主线。

注意：
下面这个目录不是当前活跃输出目录，只是旧目录/冻结目录，不要误以为它还在继续写：
`/aaaidata/zhangqisong/data_build/pipeline/data/evolution_scene_v7_qwen35_rotation4_freeform_obj001_obj003_obj007_obj008_20260402`

用户目前认为“可以接受/满意”的两条静态视觉基线是：

1. ` /aaaidata/zhangqisong/data_build/pipeline/data/evolution_scene_v7_pseudoref_multigpu_20260401_teachercam3_light1 `
2. ` /aaaidata/zhangqisong/data_build/pipeline/data/evolution_scene_v7_qwen35_freeform_full10_20260402 `

另外，当前 rotation4 主线目录：
`/aaaidata/zhangqisong/data_build/pipeline/data/evolution_scene_v7_qwen35_rotation4_agent_round0_obj001_obj003_obj007_obj008_20260402`
用户也认为“整体还是可以接受的，只是稍微有点乱”。

所以你要明确区分三类东西：

1. 当前继续推进中的 active loop：
   - `rotation4_agent_round0...`

2. 用户认可、可随时回退/抄参数的静态兜底：
   - `teachercam3_light1`
   - `qwen35_freeform_full10`

3. 旧的冻结目录：
   - `rotation4_freeform...`
   - 不要误判为当前没产出

--------------------------------
三、当前方法里，已经证明有效的核心思路
--------------------------------

你要重点继承的不是早期版本，而是下面这条方法链：

1. 使用 scene-aware Blender 渲染，而不是白底 studio 渲染
2. 使用 `Qwen3.5-35B-A3B`
3. 开启 thinking
4. 使用自由文本 reviewer，而不是强依赖 rigid JSON
5. AI 必须亲自读取 `trace.json` 里的 reviewer 自由文本
6. AI 根据 reviewer 文本，自己决定下一轮动作
7. 再次渲染
8. 继续读 reviewer 文本
9. 直到 reviewer 明确给出 keep / acceptable 才能停

也就是说，真正有效的方法是：

`render -> freeform VLM review -> AI 读 reviewer 文本 -> AI 决定动作 -> rerender -> repeat`

而不是：

`render -> 自动 controller 按分数选动作 -> repeat`

--------------------------------
四、你必须重点看的代码脚本
--------------------------------

本地代码先看这些：

1. `scripts/run_scene_agent_monitor.py`
   - 当前真正的持续 monitor / orchestrator
   - 它会轮询 pair
   - 读 reviewer 自由文本
   - 生成 decision
   - 发起下一轮
   - 你接这个 loop 时必须先读懂它

2. `scripts/run_scene_agent_step.py`
   - 单个 pair 的一步执行器
   - 一次只做一轮：
     - 载入/应用 control state
     - Blender render
     - Qwen3.5 review
   - 它本身不应该自动做最终决策，决策来自 AI/monitor

3. `pipeline/stage4_scene_render.py`
   - Blender scene-aware 渲染核心逻辑
   - 当前很多优化都在这里：
     - contact shadow
     - ground epsilon
     - scene light handling
     - material/reference material
     - hue/value/saturation / roughness / specular 等控制

4. `pipeline/stage5_5_vlm_review.py`
   - Qwen3.5 reviewer 入口
   - 当前 reviewer 单卡加载规则、自由文本 review、thinking 等都在这里
   - 最近已经改过，避免某些场景下错误走 `device_map=balanced`

5. `pipeline/stage5_6_feedback_apply.py`
   - action apply 逻辑

6. `configs/scene_action_space.json`
   - 当前动作空间定义
   - 包括 hue/value/saturation/roughness/specular/contact shadow/scale/offset 等

7. `configs/dataset_profiles/scene_v7.json`
   - 当前 scene profile

8. `configs/scene_template.json`
   - 当前 scene template

9. `run_scene_evolution_loop.py`
   - 仍然有参考价值，但当前 active 主线不是简单地依赖它自动闭环

--------------------------------
五、远端对应代码位置
--------------------------------

远端工作目录：
`/aaaidata/zhangqisong/data_build`

远端对应文件：

- `/aaaidata/zhangqisong/data_build/scripts/run_scene_agent_monitor.py`
- `/aaaidata/zhangqisong/data_build/scripts/run_scene_agent_step.py`
- `/aaaidata/zhangqisong/data_build/pipeline/stage4_scene_render.py`
- `/aaaidata/zhangqisong/data_build/pipeline/stage5_5_vlm_review.py`
- `/aaaidata/zhangqisong/data_build/pipeline/stage5_6_feedback_apply.py`
- `/aaaidata/zhangqisong/data_build/configs/scene_action_space.json`
- `/aaaidata/zhangqisong/data_build/configs/dataset_profiles/scene_v7.json`
- `/aaaidata/zhangqisong/data_build/configs/scene_template.json`
- `/aaaidata/zhangqisong/data_build/run_scene_evolution_loop.py`

--------------------------------
六、当前远端运行状态
--------------------------------

当前后台会话通常看这个：

`tmux attach -t scene-rot-monitor`

如果会话还在，这就是 active monitor。

你要优先检查：

1. `tmux capture-pane -pt scene-rot-monitor:0 | tail -n 200`
2. `_monitor_logs/monitor_gpu_*.json`
3. 每个 pair 的：
   - `reviews/*_trace.json`
   - `reviews/*_agg.json`
   - `decisions/roundXX_decision.json`
   - `states/roundXX.json`
   - `roundXX_renders/.../az000_el+00.png`

--------------------------------
七、在 loop 里到底应该怎么看 reviewer
--------------------------------

重点不是先看 `agg.json`，而是先看：

`reviews/*_trace.json`

特别是里面的：
- `attempts[-1].assistant_text`

因为真正要接住的是 reviewer 的自由文本。

`agg.json` 只能辅助你看：
- hybrid_score
- issue_tags
- lighting_diagnosis
- programmatic_physics

但**不能代替**自由文本判断。

你要优先从自由文本里识别：

1. 当前主要矛盾是不是：
   - color shift
   - flat lighting
   - weak grounding
   - shadow missing
   - ground intersection
   - material too plastic / too smooth
   - too dark / too bright
   - too small / too large

2. reviewer 的结论到底是：
   - keep
   - revise
   - reject

如果 reviewer 还没给 keep，就不能主动停。

--------------------------------
八、当前已经踩过、不要再踩的坑
--------------------------------

1. 不要再把旧的 `rotation4_freeform...` 当成当前活跃目录
2. 不要再回到“脚本自动吃分数自动选动作”的旧模式
3. 不要让 AI 主动退出 loop
4. 不要只看 score
5. 不要只看 `agg.json`，必须看 `trace.json`
6. 不要忘记用户明确要求：
   - Codex/Claude 必须实时接收 VLM 自由文本
   - 必须自己根据反馈调整 Blender 渲染参数和配置
7. 不要误把 scene 项目和 LoRA 训练项目混在一起
   - 这里当前主线是 `/aaaidata/zhangqisong/data_build`
   - 不是 `/gemini/.../test/`
8. 不要误以为结果“没有新增”
   - 先确认你看的是不是 active 主线目录
9. 不要忽略 GPU 资源干扰
   - 当前 gpu0 曾经因为外部进程占显存导致 reviewer OOM
   - 不是 Blender 没出图，而是 review 阶段卡住

--------------------------------
九、当前一个已知现实问题
--------------------------------

当前 `obj_001_yaw000` 这一路曾卡在 reviewer 阶段，不是渲染没出图。

已经确认：
- `round11_renders` 图像已经出来
- 但 review 没完成
- 原因是 reviewer 加载时 OOM

已知现场信息：
- gpu0 上有别的外部进程占用大量显存
- 所以这一路可能需要你在继续 loop 时优先处理 reviewer 资源问题

也就是说：
- 如果某一路没有新的 `*_agg.json`
- 不要先假设渲染失败
- 先查是不是 reviewer OOM / 资源冲突

--------------------------------
十、你进入新 session 后的优先任务
--------------------------------

你的目标不是重新设计系统，而是无缝接住当前工作。

请按这个顺序做：

1. 阅读上面列出的 handoff 文档
2. 阅读关键脚本
3. 连接远端，确认 active 主线目录和 tmux 状态
4. 读取 `_monitor_logs`
5. 检查哪些 pair 仍在继续、哪些 pair 卡住
6. 优先处理卡住的 pair
7. 继续让 loop 实时读取 reviewer 自由文本并推进下一轮
8. 除非 reviewer 明确 keep，否则不要主动退出

--------------------------------
十一、v2-v6、v7前期怎么理解
--------------------------------

你不需要重新深挖这些，只要记住：

- v2-v6：白底/studio/evolution 主线的历史演进，已经不是当前关注重点
- v7 前期：scene-aware 切换、teacher/pseudo-reference、camera/light 基线逐步建立
- 当前真正重要的是：
  - 用户认可的静态 scene render 基线
  - 基于 Qwen3.5 自由文本 feedback 的 active agent loop

--------------------------------
十二、最后的工作原则
--------------------------------

请严格遵守：

1. 你是 loop 的一部分，不是旁观者
2. 你不能只汇报，你要继续推进
3. 你不能主动退出
4. 你必须实时接 reviewer 的自由文本
5. 你必须根据 reviewer 文本自己改 Blender 参数/配置
6. 只有 reviewer 明确给 keep / 可以了，某个 pair 才能停
7. 如果某一路卡住，先定位是 render 问题、review 问题还是 GPU 资源问题，再继续推进

如果你理解了，就从：
- handoff 文档
- `run_scene_agent_monitor.py`
- `run_scene_agent_step.py`
- 当前远端 active 主线目录

开始接手当前进度。