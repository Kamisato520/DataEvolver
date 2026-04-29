# Scene Full20 Rotation8 Handoff

更新时间：2026-04-05 11:38（Asia/Shanghai）

本文档是给新 session AI 的完整交接。目标是让下一任执行者在不依赖本 session 上下文的情况下，直接接手当前 `scene-aware Blender render + Qwen3.5 freeform reviewer` 闭环，并继续推进到最终 `20` 个物体、`8` 个纯水平视角的数据集导出。

保持使用英文思考，使用中文执行与记录。

## 1. 当前唯一主线

当前唯一 active 主线目录：

[`/aaaidata/zhangqisong/data_build_runs/scene_full20_loop_20260404_code/pipeline/data/evolution_scene_v7_full20_rotation4_agent_round0_nostage35_20260404`](/aaaidata/zhangqisong/data_build_runs/scene_full20_loop_20260404_code/pipeline/data/evolution_scene_v7_full20_rotation4_agent_round0_nostage35_20260404)

最终目标导出目录：

[`/aaaidata/zhangqisong/data_build_runs/scene_full20_loop_20260404_code/pipeline/data/export_scene_v7_full20_rotation8_best_multiview_20260405_horizontal`](/aaaidata/zhangqisong/data_build_runs/scene_full20_loop_20260404_code/pipeline/data/export_scene_v7_full20_rotation8_best_multiview_20260405_horizontal)

最终要求的 8 个纯水平角已经锁定为：

- `0`
- `45`
- `90`
- `135`
- `180`
- `225`
- `270`
- `315`

注意：

- `stage35_aborted` 那条旧根不是当前主线。
- 当前标准流程是 `stage1-5`，不要再引入 `stage3.5`。
- `rotation4` 主根是上游优化根，不是最终 8 角数据集目录。

## 2. 当前是否完成

截至 `2026-04-05 11:30-11:38`，任务**没有完成**。

明确证据：

- 最终导出目录还不存在。
- `export_watch.log` 持续报 `queue-not-empty`。
- `render_tuner` 还在运行，不是收尾状态。
- `regeneration_queue.json` 还没有清空。

当时远端状态：

- `render_tuner_active.json`：
  - `status = running`
  - `iteration_idx = 7`
  - `updated_at = 2026-04-05 11:22:39`
- `regeneration_queue.json` 统计：
  - `total_jobs = 290`
  - `smoke_failed = 140`
  - `exhausted = 128`
  - `promoted = 9`
  - `running = 2`
  - `queued = 11`

结论：

- 系统还在运行，但没有完成最终 20 物体 × 8 视角数据集构建。

## 3. 远端当前现场

### 3.1 tmux 会话

截至最近一次核对，远端存在这些会话：

- `full20_export`
- `full20_regen`
- `full20_regen_gpu2`
- `full20_tuner`
- `full20_tuner_watchdog`
- `full20_watchdog`

### 3.2 GPU 占用

最近一次精确核对：

- `GPU0`：空闲
- `GPU1`：`PID 3081233`
- `GPU2`：`PID 3080548`

对应进程：

- `GPU1`
  - `python .../scripts/run_scene_agent_step.py`
  - 任务：`obj_013/attempt_16/smoke`
  - 显存约 `68802 MiB`
- `GPU2`
  - `python .../scripts/run_scene_agent_step.py`
  - 任务：`obj_014/attempt_16/smoke`
  - 显存约 `68802 MiB`

也就是说，最近一次现场里：

- `GPU1/GPU2` 被 replacement smoke 占着
- `GPU0` 是空闲的
- 当时没有看到 `run_render_feedback_tuner.py` 正在占 GPU

### 3.3 关键父进程

远端活着的关键父进程包括：

- `scripts/run_full20_watchdog.py`
- 两条 `scripts/run_asset_regeneration_queue.py`
- `full20_tuner_watchdog`

## 4. 我做过的核心代码改动

以下是本 session 里和当前 scene loop / replacement / export 最相关的改动。

### 4.1 废弃与补生机制

文件：

- [pipeline/asset_lifecycle.py](/D:/新建文件夹/用户目录/桌面/autoresearch_vlm/Auto-claude-code-research-in-sleep/ARIS/ARIS/pipeline/asset_lifecycle.py)
- [scripts/run_asset_regeneration_queue.py](/D:/新建文件夹/用户目录/桌面/autoresearch_vlm/Auto-claude-code-research-in-sleep/ARIS/ARIS/scripts/run_asset_regeneration_queue.py)
- [scripts/run_scene_agent_monitor.py](/D:/新建文件夹/用户目录/桌面/autoresearch_vlm/Auto-claude-code-research-in-sleep/ARIS/ARIS/scripts/run_scene_agent_monitor.py)
- [pipeline/stage5_5_vlm_review.py](/D:/新建文件夹/用户目录/桌面/autoresearch_vlm/Auto-claude-code-research-in-sleep/ARIS/ARIS/pipeline/stage5_5_vlm_review.py)

已做：

- `asset_viability / abandon_reason / abandon_confidence` 全链打通
- reviewer freeform prompt 要求输出 `Verdict` 之后的 `Asset viability`
- scene monitor 增加 stop-loss
- 引入 `asset_registry.json / regeneration_queue.json`
- 引入 replacement runner
- replacement 失败后支持回到 `stage1_restart`
- 之后又把 `stage1_restart` 升级成 failure-aware prompt 重写

### 4.2 failure-aware stage1 restart

文件：

- [pipeline/stage1_text_expansion.py](/D:/新建文件夹/用户目录/桌面/autoresearch_vlm/Auto-claude-code-research-in-sleep/ARIS/ARIS/pipeline/stage1_text_expansion.py)
- [scripts/run_asset_regeneration_queue.py](/D:/新建文件夹/用户目录/桌面/autoresearch_vlm/Auto-claude-code-research-in-sleep/ARIS/ARIS/scripts/run_asset_regeneration_queue.py)

已做：

- 新增 `--repair-spec-file`
- replacement 从失败 attempt 中抽取：
  - `major_issues`
  - `suggested_fixes`
  - `issue_tags`
  - `structure_consistency`
  - `hybrid_score`
- 生成 `stage1_repair_spec.json`
- `stage1_restart` 不再只是模板回退，而是基于失败 trace 自动重写 prompt

### 4.3 render tuner 闭环

核心文件：

- [scripts/run_render_feedback_tuner.py](/D:/新建文件夹/用户目录/桌面/autoresearch_vlm/Auto-claude-code-research-in-sleep/ARIS/ARIS/scripts/run_render_feedback_tuner.py)
- [scripts/run_render_feedback_tuner_daemon.sh](/D:/新建文件夹/用户目录/桌面/autoresearch_vlm/Auto-claude-code-research-in-sleep/ARIS/ARIS/scripts/run_render_feedback_tuner_daemon.sh)
- [scripts/run_render_feedback_tuner_watchdog.sh](/D:/新建文件夹/用户目录/桌面/autoresearch_vlm/Auto-claude-code-research-in-sleep/ARIS/ARIS/scripts/run_render_feedback_tuner_watchdog.sh)

`run_render_feedback_tuner.py` 现在已经不是只看分数，而是：

- 读取最新 `reviews/*_trace.json`
- 优先取 `attempts[-1].assistant_text`
- 依据 reviewer 自由文本做 pair 级控制修正
- 每个 pair 先写 `control_state.tuned.json`
- 再跑 `run_scene_agent_step.py`
- 结果写回 `agent_round00.json`
- 以 freeform 的 `keep / acceptable / good enough` 作为 acceptance 依据

当前 pair 级控制字段包括：

- `lighting.key_scale`
- `object.offset_z`
- `object.scale`
- `scene.env_strength_scale`
- `scene.contact_shadow_strength`
- `material.value_scale`
- `material.saturation_scale`
- `material.hue_offset`
- `material.specular_add`
- `material.roughness_add`

### 4.4 针对自由文本的关键启发式修正

这是后续最重要的上下文之一。

#### 已经补进 `run_render_feedback_tuner.py` 的自由文本响应

- `underexposed / too dark`
  - 提升 `env_strength_scale`
  - 提升 `value_scale`
  - 提升 `saturation_scale`
  - 提升 `key_scale`

- `flat lighting / weak subject separation`
  - 提升 `key_scale`
  - 提升 `specular_add`

- `cool / blue cast`
  - 提升 `hue_offset` 往暖色偏

- `floating / weak shadow`
  - 提升 `contact_shadow_strength`
  - 轻微下压 `offset_z`

- `ground intersection`
  - 轻微抬高 `offset_z`

- `too large / giant`
  - 缩 `object.scale`

- `too small`
  - 放大 `object.scale`

- `too matte / muddy / metallic reflection missing`
  - 提升 `saturation_scale`
  - 提升 `specular_add`
  - 降低 `roughness_add`

#### 本 session 后期新补的两类修正

这是新 session 必须知道的，因为是根据最新 reviewer 文本加上的：

- `god rays / rays are too strong / artificial volumetric light`
  - 现在会单独触发：
    - 略降 `scene.env_strength_scale`
    - 略降 `lighting.key_scale`
  - 同时在 template 级 summary 聚合里：
    - 略降 `scale_world_background`
    - 略降 `scale_existing_lights`

- `plastic / strange specular highlight / too sharp / too glossy`
  - 现在优先走“减塑料感”分支，而不是继续加高光：
    - `specular_add -= 0.05`
    - `roughness_add += 0.07`
  - 这是修复 `obj_002` 时加的，目的是防止 reviewer 明说“过塑料高光”却还继续加 specular

### 4.5 asset-blocked 检测扩展

`run_render_feedback_tuner.py` 中的 `is_asset_blocked_trace()` 已扩展。

原先就有：

- `fundamentally wrong`
- `different model`
- `object identity mismatch`
- `wrong object geometry`
- `asset viability: abandon`
- `should be regenerated`
- `completely wrong shape`

我后面又补了针对 `obj_014` 的硬词：

- `merged mesh`
- `glitchy mesh`
- `physically impossible`
- `two skateboards`
- `double truck`
- `6 wheels`
- `six wheels`

这是因为 `obj_014` 的 longboard reviewer 文本已经明显是资产级错误，不应该继续占 benchmark。

### 4.6 watchdog 修复

文件：

- [scripts/run_render_feedback_tuner_watchdog.sh](/D:/新建文件夹/用户目录/桌面/autoresearch_vlm/Auto-claude-code-research-in-sleep/ARIS/ARIS/scripts/run_render_feedback_tuner_watchdog.sh)
- [scripts/run_render_feedback_tuner_daemon.sh](/D:/新建文件夹/用户目录/桌面/autoresearch_vlm/Auto-claude-code-research-in-sleep/ARIS/ARIS/scripts/run_render_feedback_tuner_daemon.sh)
- [scripts/run_full20_watchdog.py](/D:/新建文件夹/用户目录/桌面/autoresearch_vlm/Auto-claude-code-research-in-sleep/ARIS/ARIS/scripts/run_full20_watchdog.py)

已做：

- watchdog 每 `120s` 轮询
- 默认 `STALL_SECONDS=1200`
- 监听：
  - `render_tuner_active.json`
  - 最新 `reviews/*_trace.json` 时间
- session 丢失或无活动时重拉 `full20_tuner`
- 新增对 `render_tuner_done.json` 的处理：
  - `good_enough` 时允许结束
  - 否则自动归档旧 `render_tuner` 并重开新 cycle

之前 watchdog 有一个问题：

- 日志会打印 `archive_and_restart`
- 但实际上旧目录没被移动
- `full20_tuner` 也没重新拉起

后来已改成用 Python `shutil.move()` 做归档，避免 shell move 在现场不稳。

### 4.7 2026-04-05 新补的“不停机 loop”托底

这是为了解决“session AI 只启动进程然后退出，loop 本体没有持续决策感”的问题。

已新增/修改：

- `run_render_feedback_tuner_daemon.sh`
  - 以前只要看到 `render_tuner_done.json` 就直接退出
  - 现在改成：
    - 如果 `status == good_enough`，才允许退出
    - 如果是 `max_iterations_reached` 或其他非 good_enough 状态：
      - 自动归档旧 `render_tuner/`
      - 重建空 runtime
      - 自动开始下一轮 tuner cycle

- `run_full20_watchdog.py`
  - 以前只托底：
    - scene workers
    - regen daemon
    - export watcher
  - 现在额外托底：
    - `run_render_feedback_tuner_watchdog.sh`
  - 也就是说，如果远端没有 `full20_tuner_watchdog`，它会自动重新拉起

这两处补丁的目的就是：

- 不让 tuner 停在 `done marker` 上假结束
- 不让系统进入“tmux 还活着，但没人继续推进 loop”的半停机状态

## 5. 远端运行中遇到过的关键问题

### 5.1 render_tuner 一度停在 `max_iterations_reached`

当时现场：

- `full20_tuner` 不存在
- `render_tuner_done.json` 存在，状态：
  - `max_iterations_reached`
  - `iteration_idx = 10`
  - `best_mean_hybrid_score = 0.4288`
  - `last_accepted_pairs = ['obj_002_yaw000']`

我做过的修复：

- 手动把旧 `render_tuner` 归档到：
  [`/aaaidata/zhangqisong/data_build_runs/scene_full20_loop_20260404_code/runtime/full20_scene_20260404/render_tuner_archive_manual_20260405_100245`](/aaaidata/zhangqisong/data_build_runs/scene_full20_loop_20260404_code/runtime/full20_scene_20260404/render_tuner_archive_manual_20260405_100245)
- 重新建空 `render_tuner/`
- 重启 `full20_tuner` 和 `full20_tuner_watchdog`

### 5.2 replacement queue 并发与 stale 问题

为此我在 `asset_lifecycle.py` 做过：

- 全局 state lock
- 原子写盘
- `claim_next_queued_job()`
- `recover_stale_running_jobs()`

并把 `run_asset_regeneration_queue.py` 改成 claim 模式，不再盲扫快照。

### 5.3 reviewer / smoke GPU 错配

之前有：

- `stage2.5` SAM3 设备错配
- reviewer 单卡抢显存

后来做过：

- GPU 组约束
- smoke 绑定可见 GPU
- replacement worker 按隔离的 `CUDA_VISIBLE_DEVICES` 跑逻辑 `cuda:0`

## 6. 真实灰度样本与结论

### 6.1 `obj_002`

这是最典型的非资产级但需要持续修 scene 的样本。

早期情况：

- 多轮 `attempt` 被判差资产
- 走过 replacement
- 走过 `stage1_restart`
- 走过 failure-aware prompt 重写

在 render tuner 里，最新一次关键反馈是：

- too cool / too dark / flat
- god rays too strong
- weak contact shadow
- ceramic highlight too plastic

我据此新增了：

- `god_rays_overdone` 分支
- `plastic specular` 分支

最近一次远端真实结果：

- `obj_002`：
  - `hybrid_score ≈ 0.4986`
  - `route = reject`
  - `issue_tags = ['underexposed', 'flat_lighting', 'color_shift']`
  - `structure_consistency = good`
  - `asset_viability = continue`

这说明：

- 塑料感和 grounding 问题已经比之前弱
- 主要残留问题变成偏暗、偏平、偏冷
- 当前这条调参方向是有效的，但还没收敛到 accept

### 6.2 `obj_014`

这是最典型的资产级错误样本。

reviewer 最近的自由文本实际在说：

- longboard geometry weird
- 像两块板 merged
- 像 double truck / 6 wheels
- physically impossible
- glitchy mesh

这不是 scene integration 问题，而是 asset 本身问题。

我已经把这类表述补进 `asset_blocked` 词表，但注意：

- 如果当前正在运行的 `full20_tuner` 进程是在补词表之前启动的，它本轮还不会吃到新规则
- 新 session 如果发现 `obj_014` 仍在 benchmark 里，不要怀疑方向，直接热重启 `full20_tuner`

## 7. 当前最重要的观察

### 7.1 系统没有完成的真正原因

不是 export 脚本坏了。

真正原因是：

- replacement queue 还没清空
- smoke gate 失败太多
- tuner 还没把 scene benchmark 收敛到 reviewer accept
- 所以 export watcher 一直不放行

### 7.2 `GPU0` 空闲是当前现场最明显的问题

最近一次核对中：

- `GPU1/GPU2` 在跑 replacement smoke
- `GPU0` 空闲

这意味着新 session 的 AI 最该做的一件事不是继续解释，而是：

- 把 `GPU0` 重新接回主线 scene tuner 或 pair-level scene loop
- 不要让它闲着

## 8. 新 session 接手后建议的第一批动作

按优先级执行。

### 8.1 先核对现场，不要假设

建议第一批命令：

```bash
ssh wwz "tmux ls | grep full20_ || true"
ssh wwz "nvidia-smi"
ssh wwz "python3 - <<'PY'
import os, json
for p in [
  '/aaaidata/zhangqisong/data_build_runs/scene_full20_loop_20260404_code/runtime/full20_scene_20260404/render_tuner/render_tuner_active.json',
  '/aaaidata/zhangqisong/data_build_runs/scene_full20_loop_20260404_code/runtime/full20_scene_20260404/render_tuner/render_tuner_done.json',
  '/aaaidata/zhangqisong/data_build_runs/scene_full20_loop_20260404_code/pipeline/data/regeneration_queue.json',
]:
    print('===', p, '===')
    print(open(p, 'r', encoding='utf-8').read() if os.path.exists(p) else 'MISSING')
PY"
```

### 8.2 如果 `GPU0` 还空闲，优先把它接回主线

目标：

- 不要只让 replacement 跑
- 把 `GPU0` 用于主线 `render_tuner` 或 scene loop

### 8.3 检查 `obj_014` 是否还在 tuner benchmark

如果还在：

- 说明当前 running 的 `full20_tuner` 进程还是旧逻辑
- 直接热重启 `full20_tuner`
- 不需要再改方向

### 8.4 持续优先读取 `reviews/*_trace.json`

尤其是：

- `attempts[-1].assistant_text`

不要只看：

- `hybrid_score`
- `issue_tags`
- `suggested_actions`

真正要接的是 reviewer 自由文本。

### 8.5 继续沿自由文本修 scene，不要回退成纯 controller

当前最常见自由文本问题仍然集中在：

- underexposed
- flat_lighting
- weak_subject_separation
- color_shift
- floating / weak contact shadow
- god rays too strong
- plastic specular

## 9. 本地代码与远端代码需要重点关注的文件

本地仓库中最关键的文件：

- [scripts/run_render_feedback_tuner.py](/D:/新建文件夹/用户目录/桌面/autoresearch_vlm/Auto-claude-code-research-in-sleep/ARIS/ARIS/scripts/run_render_feedback_tuner.py)
- [scripts/run_render_feedback_tuner_daemon.sh](/D:/新建文件夹/用户目录/桌面/autoresearch_vlm/Auto-claude-code-research-in-sleep/ARIS/ARIS/scripts/run_render_feedback_tuner_daemon.sh)
- [scripts/run_render_feedback_tuner_watchdog.sh](/D:/新建文件夹/用户目录/桌面/autoresearch_vlm/Auto-claude-code-research-in-sleep/ARIS/ARIS/scripts/run_render_feedback_tuner_watchdog.sh)
- [scripts/run_asset_regeneration_queue.py](/D:/新建文件夹/用户目录/桌面/autoresearch_vlm/Auto-claude-code-research-in-sleep/ARIS/ARIS/scripts/run_asset_regeneration_queue.py)
- [pipeline/asset_lifecycle.py](/D:/新建文件夹/用户目录/桌面/autoresearch_vlm/Auto-claude-code-research-in-sleep/ARIS/ARIS/pipeline/asset_lifecycle.py)
- [pipeline/stage1_text_expansion.py](/D:/新建文件夹/用户目录/桌面/autoresearch_vlm/Auto-claude-code-research-in-sleep/ARIS/ARIS/pipeline/stage1_text_expansion.py)
- [pipeline/stage5_5_vlm_review.py](/D:/新建文件夹/用户目录/桌面/autoresearch_vlm/Auto-claude-code-research-in-sleep/ARIS/ARIS/pipeline/stage5_5_vlm_review.py)
- [scripts/export_scene_multiview_from_pair_evolution.py](/D:/新建文件夹/用户目录/桌面/autoresearch_vlm/Auto-claude-code-research-in-sleep/ARIS/ARIS/scripts/export_scene_multiview_from_pair_evolution.py)
- [scripts/run_full20_export_watch.sh](/D:/新建文件夹/用户目录/桌面/autoresearch_vlm/Auto-claude-code-research-in-sleep/ARIS/ARIS/scripts/run_full20_export_watch.sh)

远端对应代码副本根：

[`/aaaidata/zhangqisong/data_build_runs/scene_full20_loop_20260404_code`](/aaaidata/zhangqisong/data_build_runs/scene_full20_loop_20260404_code)

## 10. 当前交接判断

当前系统状态可以概括为：

- 主线目录正确
- 8 角导出目标正确
- replacement / tuner / export watcher 都存在
- 闭环代码已经不是纯分数驱动，而是自由文本驱动
- 但系统没有完成收敛，也没有完成最终导出
- 当前最需要的是一个持续在线的新执行者，把 `GPU0` 接回主线，并继续用 reviewer 自由文本驱动 scene loop，不要再让系统变成“只有 tmux 活着、AI 本体退出”的状态

如果新 session 只看一段最短摘要，就看这一段：

> 当前主线是 `evolution_scene_v7_full20_rotation4_agent_round0_nostage35_20260404`，最终要导出到 `export_scene_v7_full20_rotation8_best_multiview_20260405_horizontal`，角度固定 `0,45,90,135,180,225,270,315`。任务还没完成；export 目录不存在；queue 没清空；`render_tuner` 还在跑。最近最重要的代码改动在 `run_render_feedback_tuner.py`：已经能按自由文本处理 `god rays too strong` 和 `plastic specular`，并把 `obj_014` 这类 `double truck / merged mesh / physically impossible` longboard 识别成 asset-blocked。新执行者接手后先核对 tmux / queue / GPU，再把空闲 GPU0 接回主线 scene loop。 
