# 2026-04-03 Scene Rotation4 Loop Session

## 1. Background & Goal

- 当前任务不是重做 pipeline，而是继续补 `rotation4 agent` 主线的 scene-aware render loop。
- active 主线远端结果根目录：
  - `/aaaidata/zhangqisong/data_build/pipeline/data/evolution_scene_v7_qwen35_rotation4_agent_round0_obj001_obj003_obj007_obj008_20260402`
- 本 session 的核心目标：
  - 清掉失控的高轮次自动 loop
  - 改成“看得过去就停”的手工/半手工 loop
  - 把退出规则写死到代码和文档，避免再出现 40+ / 50+ 轮
  - 跑一轮低风险补 round，并记录哪些 pair 应该保留哪一轮

## 2. Current Status

### 已完成

- 已确认“改渲染代码的是 Codex；VLM 只负责看图、给 freeform 反馈，不直接改仓库代码”。
- 已确认当前 active 路线应为 `rotation4 agent`，不是旧的自动 controller loop。
- 已发现并停止远端 runaway monitor。此前它把多个 pair 跑到了 40+ / 50+ 轮，明显违背当前目标。
- 已在本地和远端同步修复 `run_scene_agent_monitor.py`：
  - 加 `HARD_MAX_ROUNDS = 15`
  - 外部传更大的 `--max-rounds` 时会被硬截断
  - `max_rounds_reached` 现在会真正 `done = True`，不再无限死循环
- 已把“5 轮内没明显提升就停，15 轮绝对上限”写进 skill 和 handoff 文档。
- 已手工跑完 3 个低轮次 pair 的新一轮，并停止追加后续轮次。

### 本 session 手工 loop 结果

1. `obj_003_yaw180`
   - 原状态：`r01 = 0.4923`
   - 本 session：
     - `r02` 动作：`O_SCALE_UP_10 S_CONTACT_SHADOW_UP O_LIFT_SMALL`
     - `r02` 结果：`0.5609`
     - `r03` 动作：`O_SCALE_UP_10 ENV_STRENGTH_UP`
     - `r03` 结果：`0.5587`，轻微回退
   - 结论：停止；当前保留 `r02`

2. `obj_007_yaw090`
   - 原状态：`r00 = 0.5781`
   - 本 session：
     - `r01` 动作：`S_CONTACT_SHADOW_UP M_ROUGHNESS_UP O_SCALE_DOWN_10`
     - `r01` 结果：`0.5321`，变差
     - `r02` 动作：`O_SCALE_UP_10 M_HUE_WARM_POS`
     - `r02` 结果：`0.5768`，明显回升
   - 结论：停止；当前保留 `r02`

3. `obj_008_yaw000`
   - 原状态：`r00 = 0.4752`
   - 本 session：
     - `r01` 动作：`O_SCALE_UP_10 S_CONTACT_SHADOW_UP O_LOWER_SMALL`
     - `r01` 结果：`0.5149`
     - `r02` 动作：`O_SCALE_UP_10 O_LIFT_SMALL`
     - `r02` 结果：`0.4901`，回退
   - 结论：停止；当前保留 `r01`

### 当前最重要的下一步

1. 新 session 不要继续自动追这 3 个 pair 的更高轮次。
2. 若继续补 backlog，只选低轮次 pair，且一次只做 1 个手工 step。
3. 优先看图和 freeform feedback，不要只看 `hybrid_score`。
4. 任何 pair 超过 5 轮还没明显提升，就应停止；无论如何不得超过 15 轮。
5. 对于已经被旧 monitor 跑坏的高轮次 pair，优先回看较早轮次而不是继续向后跑。

## 3. Environment & Infra

### 本地

- 仓库根目录：
  - `D:\新建文件夹\用户目录\桌面\autoresearch_vlm\Auto-claude-code-research-in-sleep\ARIS\ARIS`
- shell：
  - `powershell`
- 当前日期：
  - `2026-04-03`

### 服务器

- SSH 别名：
  - `wwz`
- 远端工作根目录：
  - `/aaaidata/zhangqisong/data_build`
- Blender：
  - `/home/wuwenzhuo/blender-4.24/blender`
- Python：
  - `/home/wuwenzhuo/Qwen-VL-Series-Finetune/env/bin/python`
- reviewer 模型：
  - `/data/wuwenzhuo/Qwen3.5-35B-A3B`
- GPU：
  - 3 张卡用于当前 scene loop（逻辑 GPU 0/1/2）

### 当前运行状态

- 本 session 建的 `rot4-0403-g0/g1/g2` tmux 会话已经手动 kill。
- 最后一次检查时 3 张卡均空闲：
  - `0, 0 %, 0 MiB`
  - `1, 0 %, 0 MiB`
  - `2, 0 %, 0 MiB`

## 4. Key Paths

### 4.1 Local Paths

#### Repo / Code

- Repo root:
  - `D:\新建文件夹\用户目录\桌面\autoresearch_vlm\Auto-claude-code-research-in-sleep\ARIS\ARIS`
- 监控脚本（本地改过）：
  - `D:\新建文件夹\用户目录\桌面\autoresearch_vlm\Auto-claude-code-research-in-sleep\ARIS\ARIS\scripts\run_scene_agent_monitor.py`
- 单步 step 脚本：
  - `D:\新建文件夹\用户目录\桌面\autoresearch_vlm\Auto-claude-code-research-in-sleep\ARIS\ARIS\scripts\run_scene_agent_step.py`
- 旧自动 controller 主脚本（不要回去主用）：
  - `D:\新建文件夹\用户目录\桌面\autoresearch_vlm\Auto-claude-code-research-in-sleep\ARIS\ARIS\run_scene_evolution_loop.py`
- action apply 逻辑：
  - `D:\新建文件夹\用户目录\桌面\autoresearch_vlm\Auto-claude-code-research-in-sleep\ARIS\ARIS\pipeline\stage5_6_feedback_apply.py`

#### Local Docs / Skill

- 当前 scene handoff 主文档：
  - `D:\新建文件夹\用户目录\桌面\autoresearch_vlm\Auto-claude-code-research-in-sleep\ARIS\ARIS\docs\scene_render_handoff_20260403\CURRENT_SUCCESS_PATH_HANDOFF_20260403.md`
- 当前 skill（本地改过）：
  - `D:\新建文件夹\用户目录\桌面\autoresearch_vlm\Auto-claude-code-research-in-sleep\ARIS\ARIS\skills\scene-agent-loop\SKILL.md`
- 本次 session memory：
  - `D:\新建文件夹\用户目录\桌面\autoresearch_vlm\Auto-claude-code-research-in-sleep\ARIS\ARIS\memory\2026-04-03_scene_rotation4_loop_session.md`

#### Local Config

- action space：
  - `D:\新建文件夹\用户目录\桌面\autoresearch_vlm\Auto-claude-code-research-in-sleep\ARIS\ARIS\configs\scene_action_space.json`
- profile：
  - `D:\新建文件夹\用户目录\桌面\autoresearch_vlm\Auto-claude-code-research-in-sleep\ARIS\ARIS\configs\dataset_profiles\scene_v7.json`
- scene template：
  - `D:\新建文件夹\用户目录\桌面\autoresearch_vlm\Auto-claude-code-research-in-sleep\ARIS\ARIS\configs\scene_template.json`

#### Local Temp Outputs

- 本 session 拉回本地做目检的图片：
  - `D:\新建文件夹\用户目录\桌面\autoresearch_vlm\Auto-claude-code-research-in-sleep\ARIS\ARIS\_tmp\loop_review2`

### 4.2 Server Paths

#### Remote Code / Config

- 远端工作根目录：
  - `/aaaidata/zhangqisong/data_build`
- 远端 monitor（已同步本 session 改动）：
  - `/aaaidata/zhangqisong/data_build/scripts/run_scene_agent_monitor.py`
- 远端 step：
  - `/aaaidata/zhangqisong/data_build/scripts/run_scene_agent_step.py`
- 远端渲染脚本：
  - `/aaaidata/zhangqisong/data_build/pipeline/stage4_scene_render.py`
- 远端 action space：
  - `/aaaidata/zhangqisong/data_build/configs/scene_action_space.json`
- 远端 profile：
  - `/aaaidata/zhangqisong/data_build/configs/dataset_profiles/scene_v7.json`
- 远端 scene template：
  - `/aaaidata/zhangqisong/data_build/configs/scene_template.json`
- meshes：
  - `/aaaidata/zhangqisong/data_build/pipeline/data/meshes`

#### Models / Runtime

- Blender：
  - `/home/wuwenzhuo/blender-4.24/blender`
- Python：
  - `/home/wuwenzhuo/Qwen-VL-Series-Finetune/env/bin/python`
- reviewer 模型：
  - `/data/wuwenzhuo/Qwen3.5-35B-A3B`

#### Active Results

- active 主结果根：
  - `/aaaidata/zhangqisong/data_build/pipeline/data/evolution_scene_v7_qwen35_rotation4_agent_round0_obj001_obj003_obj007_obj008_20260402`
- shards 根：
  - `/aaaidata/zhangqisong/data_build/pipeline/data/evolution_scene_v7_qwen35_rotation4_agent_round0_obj001_obj003_obj007_obj008_20260402/_shards`
- 本 session 手工日志：
  - `/aaaidata/zhangqisong/data_build/pipeline/data/evolution_scene_v7_qwen35_rotation4_agent_round0_obj001_obj003_obj007_obj008_20260402/_manual_logs`

#### 当前建议保留的具体结果路径

- `obj_003_yaw180`
  - best render：
    - `/aaaidata/zhangqisong/data_build/pipeline/data/evolution_scene_v7_qwen35_rotation4_agent_round0_obj001_obj003_obj007_obj008_20260402/_shards/shard_gpu_0/obj_003_yaw180/round02_renders/obj_003/az000_el+00.png`
  - best review agg：
    - `/aaaidata/zhangqisong/data_build/pipeline/data/evolution_scene_v7_qwen35_rotation4_agent_round0_obj001_obj003_obj007_obj008_20260402/_shards/shard_gpu_0/obj_003_yaw180/reviews/obj_003_r02_agg.json`
  - state：
    - `/aaaidata/zhangqisong/data_build/pipeline/data/evolution_scene_v7_qwen35_rotation4_agent_round0_obj001_obj003_obj007_obj008_20260402/_shards/shard_gpu_0/obj_003_yaw180/states/round02.json`

- `obj_007_yaw090`
  - best render：
    - `/aaaidata/zhangqisong/data_build/pipeline/data/evolution_scene_v7_qwen35_rotation4_agent_round0_obj001_obj003_obj007_obj008_20260402/_shards/shard_gpu_0/obj_007_yaw090/round02_renders/obj_007/az000_el+00.png`
  - best review agg：
    - `/aaaidata/zhangqisong/data_build/pipeline/data/evolution_scene_v7_qwen35_rotation4_agent_round0_obj001_obj003_obj007_obj008_20260402/_shards/shard_gpu_0/obj_007_yaw090/reviews/obj_007_r02_agg.json`
  - state：
    - `/aaaidata/zhangqisong/data_build/pipeline/data/evolution_scene_v7_qwen35_rotation4_agent_round0_obj001_obj003_obj007_obj008_20260402/_shards/shard_gpu_0/obj_007_yaw090/states/round02.json`

- `obj_008_yaw000`
  - best render：
    - `/aaaidata/zhangqisong/data_build/pipeline/data/evolution_scene_v7_qwen35_rotation4_agent_round0_obj001_obj003_obj007_obj008_20260402/_shards/shard_gpu_0/obj_008_yaw000/round01_renders/obj_008/az000_el+00.png`
  - best review agg：
    - `/aaaidata/zhangqisong/data_build/pipeline/data/evolution_scene_v7_qwen35_rotation4_agent_round0_obj001_obj003_obj007_obj008_20260402/_shards/shard_gpu_0/obj_008_yaw000/reviews/obj_008_r01_agg.json`
  - state：
    - `/aaaidata/zhangqisong/data_build/pipeline/data/evolution_scene_v7_qwen35_rotation4_agent_round0_obj001_obj003_obj007_obj008_20260402/_shards/shard_gpu_0/obj_008_yaw000/states/round01.json`

#### 不要继续追的典型高轮次路径

- 用户明确指出该 pair 的高轮次效果不如较早轮次：
  - `/aaaidata/zhangqisong/data_build/pipeline/data/evolution_scene_v7_qwen35_rotation4_agent_round0_obj001_obj003_obj007_obj008_20260402/_shards/shard_gpu_2/obj_008_yaw270`
  - 应重点回看：
    - `/aaaidata/zhangqisong/data_build/pipeline/data/evolution_scene_v7_qwen35_rotation4_agent_round0_obj001_obj003_obj007_obj008_20260402/_shards/shard_gpu_2/obj_008_yaw270/round13_renders`
  - 本 session 检查时该目录已经至少存在到：
    - `round45_renders`

### 4.3 Other Baseline Roots

- 最稳静态兜底：
  - `/aaaidata/zhangqisong/data_build/pipeline/data/evolution_scene_v7_pseudoref_multigpu_20260401_teachercam3_light1`
- freeform 视觉兜底：
  - `/aaaidata/zhangqisong/data_build/pipeline/data/evolution_scene_v7_qwen35_freeform_full10_20260402`

## 5. Pipeline (Reproducible)

### 当前推荐方法：手工/半手工 scene loop

1. **先看状态，不先开跑**
   - 查 GPU 是否空闲
   - 查 tmux 是否已有历史 monitor / step 在后台
   - 如果发现旧的 runaway monitor，先停掉

2. **对每个 pair 先读 3 类信息**
   - `agg.json`
   - `freeform_feedback` / `trace.json`
   - 当前 render 图 + 上一轮 render 图 + 参考图

3. **只做最小动作集**
   - 每轮只选 1-2 个动作，最多 3 个
   - 不要一口气堆太多动作
   - 不跨 yaw 共享动作

4. **跑单步**
   - 用 `run_scene_agent_step.py`
   - 一次只推进一轮
   - 不让脚本自动继续追下一轮

5. **结果判定**
   - 人眼优先，`hybrid_score` 只作辅助
   - 如果变好但已“看得过去”，立即停
   - 如果变差或几乎不变，立即停

### 不推荐的方法

- 不要重启旧的“自动 controller 自己吃反馈”的老路线
- 不要把 `--max-rounds` 设很大再让 monitor 自己跑
- 不要对已到高轮次的 pair 再继续试图“优化”

## 6. Key Commands (Copy-paste)

### 查看服务器 GPU

```bash
ssh wwz "nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader"
```

### 查看 tmux

```bash
ssh wwz "tmux ls 2>/dev/null || true"
```

### 同步本地 monitor 到远端

```bash
scp scripts/run_scene_agent_monitor.py wwz:/aaaidata/zhangqisong/data_build/scripts/run_scene_agent_monitor.py
```

### 跑单个 pair 的单步 step 模板

```bash
ssh wwz "tmux new-session -d -s temp-step; tmux send-keys -t temp-step 'cd /aaaidata/zhangqisong/data_build && CUDA_VISIBLE_DEVICES=0 VLM_FORCE_SINGLE_GPU=1 /home/wuwenzhuo/Qwen-VL-Series-Finetune/env/bin/python /aaaidata/zhangqisong/data_build/scripts/run_scene_agent_step.py --output-dir <PAIR_DIR> --obj-id <OBJ_ID> --rotation-deg <YAW> --round-idx <NEXT_ROUND> --actions <ACTION1> <ACTION2> --agent-note <NOTE> --control-state-in <STATE_IN> --prev-renders-dir <PREV_RENDERS> --device cuda:0 --blender /home/wuwenzhuo/blender-4.24/blender --meshes-dir /aaaidata/zhangqisong/data_build/pipeline/data/meshes --profile /aaaidata/zhangqisong/data_build/configs/dataset_profiles/scene_v7.json --scene-template /aaaidata/zhangqisong/data_build/configs/scene_template.json > <LOG_PATH> 2>&1' Enter"
```

### 读取聚合结果

```bash
ssh wwz "tail -n 30 <LOG_PATH>"
```

### 拉图回本地目检

```bash
scp wwz:<REMOTE_PNG_PATH> _tmp/loop_review2/<LOCAL_NAME>.png
```

## 7. Folder Structure (Session-created/modified)

### 本地新增/修改

- `memory/`
  - 新建；用于保存可供新 session 继续的 handoff
- `memory/2026-04-03_scene_rotation4_loop_session.md`
  - 本次 session 总结
- `scripts/run_scene_agent_monitor.py`
  - 修改：硬上限 15、max round 到顶后真正终止
- `skills/scene-agent-loop/SKILL.md`
  - 修改：写死“5 轮软停、15 轮硬停、看得过去就停”
- `docs/scene_render_handoff_20260403/CURRENT_SUCCESS_PATH_HANDOFF_20260403.md`
  - 修改：补充退出原则和高轮次反例
- `_tmp/loop_review2/`
  - 本地临时下载图片目录，仅供本 session 目检

### 远端修改

- `/aaaidata/zhangqisong/data_build/scripts/run_scene_agent_monitor.py`
  - 已由本地同步过去
- `/aaaidata/zhangqisong/data_build/pipeline/data/evolution_scene_v7_qwen35_rotation4_agent_round0_obj001_obj003_obj007_obj008_20260402/_manual_logs`
  - 追加了本 session 的手工 step 日志

## 8. Key Code Notes (Brief)

- `scripts/run_scene_agent_monitor.py`
  - 远端 monitor 脚本；本 session 关键修复在这里
  - 当前要点：`HARD_MAX_ROUNDS = 15`，且 `max_rounds_reached -> done=True`

- `scripts/run_scene_agent_step.py`
  - 单次 round 的标准入口
  - 做一件事：加载控制状态、应用给定动作、渲染、跑 review、产出本轮结果

- `pipeline/stage5_6_feedback_apply.py`
  - action space 与 control state 的 apply 逻辑
  - 已支持 `action_whitelist`

- `run_scene_evolution_loop.py`
  - 老自动 loop 的主脚本
  - 仍保留旧逻辑；当前不应拿来当 active 主线

- `configs/scene_action_space.json`
  - 当前动作定义来源
  - 本 session 用到的动作包括：
    - `O_SCALE_UP_10`
    - `O_LIFT_SMALL`
    - `O_LOWER_SMALL`
    - `S_CONTACT_SHADOW_UP`
    - `ENV_STRENGTH_UP`
    - `M_HUE_WARM_POS`

- `docs/scene_render_handoff_20260403/CURRENT_SUCCESS_PATH_HANDOFF_20260403.md`
  - 当前 scene 路线最重要的总 handoff
  - 新 session 必读

## 9. Decisions & Gotchas

### 本 session 做出的关键选择

- 不再追求“特别高质量”，改为“看得过去就停”。
- 只对低轮次 pair 做最小动作步进。
- 旧 runaway monitor 先停，再手工控制下一轮。
- 只把 `run_scene_agent_monitor.py` 同步到远端，避免大范围不必要变动。

### 已确认的坑

- **最大坑**：旧 monitor 会把 pair 一直推到 40+ / 50+ 轮，后面经常变差。
- `obj_008_yaw270` 是明确反例：用户指出高轮次效果不如 `round13_renders`。
- `pairwise_vs_prev.available` 经常是 `False`，不要指望它稳定帮你做回归判断。
- `hybrid_score` 有参考价值，但不能替代看图。
- 同一对象不同 yaw 的问题类型可能完全不同，动作不能横向复制。
- 本地并没有 `scripts/github_sync.py`，所以这次只写了 repo 内文档，没有自动同步到外部 Memory 仓库。

## 10. Next Session Handoff

1. 先打开并阅读：
   - `D:\新建文件夹\用户目录\桌面\autoresearch_vlm\Auto-claude-code-research-in-sleep\ARIS\ARIS\memory\2026-04-03_scene_rotation4_loop_session.md`
   - `D:\新建文件夹\用户目录\桌面\autoresearch_vlm\Auto-claude-code-research-in-sleep\ARIS\ARIS\docs\scene_render_handoff_20260403\CURRENT_SUCCESS_PATH_HANDOFF_20260403.md`

2. 连服务器后先查：
   - `nvidia-smi`
   - `tmux ls`

3. 如果要继续补 pair：
   - 只从低轮次 pair 开始
   - 每次只跑一个单步 `run_scene_agent_step.py`
   - 每轮只给 1-2 个动作
   - 先看图，再决定是否继续

4. 当前保留结果优先看：
   - `obj_003_yaw180 -> round02`
   - `obj_007_yaw090 -> round02`
   - `obj_008_yaw000 -> round01`

5. 明确禁止：
   - 不要让任何 pair 超过 15 轮
   - 不要对已经 40+ / 50+ 轮的 pair 继续向后推
   - 不要重新启用旧自动 controller 主线作为默认方案
