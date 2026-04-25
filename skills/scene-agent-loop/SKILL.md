---
name: scene-agent-loop
description: "AI-in-the-loop scene render evolution. Reads VLM freeform reviews, analyzes trace text, decides actions, runs next round via SSH on 3-GPU. Use when continuing rotation4 agent line or starting new scene-aware render loop."
argument-hint: [continue | obj_list | status]
allowed-tools: Bash(*), Read, Write, Edit, Grep, Glob, Agent
---

# Scene Agent Loop: AI-in-the-Loop Scene Render Evolution

AI 驱动的场景渲染进化循环。读 VLM 自由文本 review → 分析问题 → 决定动作 → 3-GPU 并行跑下一轮。

## Context: $ARGUMENTS

## Constants

- SOFT_TARGET_ROUNDS_PER_PAIR = 5
- HARD_MAX_ROUNDS_PER_PAIR = 15
- PATIENCE = 2 (连续无改善轮数 → 停止该 pair)
- IMPROVE_EPS = 0.01
- ACCEPT_THRESHOLD = 0.78
- REJECT_THRESHOLD = 0.35
- STATE_FILE = `SCENE_AGENT_STATE.json` (project root)
- PYTHON_BIN = `/home/wuwenzhuo/Qwen-VL-Series-Finetune/env/bin/python`
- BLENDER_BIN = `/home/wuwenzhuo/blender-4.24/blender`
- MESHES_DIR = `/aaaidata/zhangqisong/data_build/pipeline/data/meshes`
- PROFILE = `/aaaidata/zhangqisong/data_build/configs/dataset_profiles/scene_v7.json`
- SCENE_TEMPLATE = `/aaaidata/zhangqisong/data_build/configs/scene_template.json`
- ACTION_SPACE = `/aaaidata/zhangqisong/data_build/configs/scene_action_space.json`
- AGENT_STEP_SCRIPT = `/aaaidata/zhangqisong/data_build/scripts/run_scene_agent_step.py`
- SSH_HOST = `wwz`

## 三条用户已认可的结果路径

必须同时记住这三条，不能混为一谈：

| # | 路径 | 角色 |
|---|------|------|
| 1 | `evolution_scene_v7_qwen35_rotation4_agent_round0_obj001_obj003_obj007_obj008_20260402` | **Active 主线** — 继续补 round |
| 2 | `evolution_scene_v7_pseudoref_multigpu_20260401_teachercam3_light1` | **最稳静态兜底** — accepted 3/3 |
| 3 | `evolution_scene_v7_qwen35_freeform_full10_20260402` | **视觉兜底** — 用户认可但 controller 结论不可信 |

后续扩旋转时，优先从 #2 和 #3 的 `best_state` 出发。

## State Persistence

长时间运行可能触发 context compaction。在每轮结束后写 `SCENE_AGENT_STATE.json`：

```json
{
  "status": "in_progress",
  "timestamp": "2026-04-03T...",
  "rotation4_root": "/aaaidata/zhangqisong/data_build/pipeline/data/evolution_scene_v7_qwen35_rotation4_agent_round0_obj001_obj003_obj007_obj008_20260402",
  "shard_map": {
    "shard_gpu_0": ["obj_001_yaw000", "obj_001_yaw270", "obj_003_yaw180", "obj_007_yaw090", "obj_008_yaw000"],
    "shard_gpu_1": ["obj_001_yaw090", "obj_003_yaw000", "obj_003_yaw270", "obj_007_yaw180", "obj_008_yaw090"],
    "shard_gpu_2": ["obj_001_yaw180", "obj_003_yaw090", "obj_007_yaw000", "obj_007_yaw270", "obj_008_yaw180", "obj_008_yaw270"]
  },
  "pairs": {
    "obj_003_yaw000": {
      "shard": "shard_gpu_1",
      "gpu_index": 1,
      "current_round": 2,
      "latest_hybrid": 0.5345,
      "best_hybrid": 0.5605,
      "best_round": 1,
      "no_improve_streak": 1,
      "status": "active",
      "history": []
    }
  }
}
```

### Recovery Logic

1. 无状态文件 → 全新启动
2. `status == "completed"` → 全新启动
3. `status == "in_progress"` + 时间戳 > 24h → 过期，删除重来
4. `status == "in_progress"` + 时间戳 < 24h → **恢复**，从保存的进度继续

## Workflow

### Phase 0: Initialize

1. **检查 `SCENE_AGENT_STATE.json`**，按 Recovery Logic 决定是恢复还是重新开始
2. **SSH 检查 GPU 状态**：
   ```bash
   ssh wwz "nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader"
   ```
3. **检查现有 tmux 会话**：
   ```bash
   ssh wwz "tmux ls 2>/dev/null"
   ```
4. **如果 `$ARGUMENTS` 是 `continue`**：读状态文件，继续所有 active pairs
5. **如果 `$ARGUMENTS` 是 `status`**：只报告当前状态，不执行
6. **如果 `$ARGUMENTS` 是对象列表**：初始化指定 pairs

### Phase 1: Batch Read（每轮开始）

对所有 active pairs，**并行**通过 SSH 读取最新结果。

对每个 pair (obj_XXX_yawYYY)，需要读取：

1. **agg.json** — 结构化摘要（score, diagnosis, suggested_actions）
   ```bash
   ssh wwz "cat {rotation4_root}/_shards/{shard}/{pair}/reviews/{obj}_r{round}_agg.json"
   ```

2. **trace.json** — **最重要**，含 VLM 自由文本分析
   ```bash
   ssh wwz "cat {rotation4_root}/_shards/{shard}/{pair}/reviews/{obj}_r{round}_az000_el+00_trace.json"
   ```

3. **渲染图** — 必须看图，不能只看分数
   ```bash
   ssh wwz "cat {rotation4_root}/_shards/{shard}/{pair}/round{round:02d}_renders/{obj}/az000_el+00.png" > /tmp/{pair}_r{round}.png
   ```
   然后用 Read 工具查看图片。

**关键原则**：一定要看 trace.json 中的 `attempts[0].assistant_text`，这才是 VLM 的真正分析。

### Phase 2: Analyze & Decide（AI 推理）

对每个 active pair，按以下流程推理：

#### 2.1 读 VLM Trace 文本

从 `assistant_text` 中提取：
- VLM 对当前图像的自然语言描述
- VLM 识别的主要问题
- VLM 推荐的动作及理由

#### 2.2 识别主要问题

常见问题分类：
- `object_too_small` → O_SCALE_UP_10
- `shadow_missing` → S_CONTACT_SHADOW_UP
- `flat_lighting` → ENV_STRENGTH_UP 或 L_KEY_UP
- `color_shift` → M_VALUE_DOWN, M_SATURATION_DOWN
- `weak_subject_separation` → M_VALUE_DOWN, S_CONTACT_SHADOW_UP
- `ground_intersection` → O_LIFT_SMALL
- `scene_light_mismatch` → ENV_ROTATE_30（但效果有限）
- `underexposed_global` → L_WORLD_EV_UP（唯一可靠的）

#### 2.3 对照已知经验

**obj_001 (wooden_chair)**：已到 r10，明显递减区。除非 trace 指出全新问题，否则不再继续。

**obj_003 (desk_lamp)**：O_SCALE_UP_10 + S_CONTACT_SHADOW_UP 在 yaw000 有效（+0.068），但 yaw090 反而变差。**不能跨 yaw 共享动作。**

**obj_007 (cat)**：常见问题是光影太平 + 色彩不协调。M_VALUE_DOWN + S_CONTACT_SHADOW_UP 常用，但阴影太硬时停止堆叠。

**obj_008 (dinosaur)**：多数 yaw 在 r00，先做诊断再统一/分开策略。

#### 2.4 决策规则

1. **不机械照搬** `suggested_actions`，只作参考
2. **同一对象不同角度独立决策**
3. **连续 2 轮 score < best - IMPROVE_EPS** → 停止该 pair
4. **选择 1-3 个动作**，不要堆太多
5. **人眼判断优先**，score 辅助
6. **不要追求“最优”**。看起来已经自然、没有明显穿帮时就可以停。
7. **5 轮内没明显提升就该停，15 轮是绝对硬上限**。任何 pair 都不能再出现 40+ / 50+ 轮。

### Phase 3: Batch Run（3-GPU 并行）

按 GPU 分组并行执行。每个 GPU 上串行处理其分配的 pairs。

#### 3.1 创建 tmux 会话

```bash
ssh wwz "tmux new-session -d -s scene-loop-g0 2>/dev/null || true"
ssh wwz "tmux new-session -d -s scene-loop-g1 2>/dev/null || true"
ssh wwz "tmux new-session -d -s scene-loop-g2 2>/dev/null || true"
```

#### 3.2 执行模板

对每个 pair，在对应 GPU 的 tmux 中执行：

```bash
ssh wwz "tmux send-keys -t scene-loop-g{GPU} 'CUDA_VISIBLE_DEVICES={GPU} {PYTHON_BIN} {AGENT_STEP_SCRIPT} \
  --output-dir {rotation4_root}/_shards/shard_gpu_{GPU}/{pair} \
  --obj-id {obj_id} \
  --rotation-deg {yaw_deg} \
  --round-idx {next_round} \
  --actions {action1} {action2} \
  --control-state-in {rotation4_root}/_shards/shard_gpu_{GPU}/{pair}/states/round{current_round:02d}.json \
  --prev-renders-dir {rotation4_root}/_shards/shard_gpu_{GPU}/{pair}/round{current_round:02d}_renders \
  --device cuda:0 \
  --meshes-dir {MESHES_DIR} \
  --agent-note scene_loop_r{next_round}' Enter"
```

**注意**：`--device cuda:0` 因为 `CUDA_VISIBLE_DEVICES` 已经做了重映射。

#### 3.3 等待完成

每个 agent step 约 3-5 分钟。监控方式：

```bash
ssh wwz "tmux capture-pane -pt scene-loop-g{GPU}:0 | tail -20"
```

检查是否出现 `"success": true` 或错误信息。确认进程正常运行后可以停止监测。

### Phase 4: Evaluate

读取新 round 结果，更新状态：

1. 读取新的 `agg.json`
2. 对比 `latest_hybrid` vs `best_hybrid`
3. 如果改善 > IMPROVE_EPS：更新 best，重置 no_improve_streak
4. 如果未改善：no_improve_streak += 1
5. 如果 no_improve_streak >= PATIENCE：标记 pair 为 `stopped`
6. 如果 latest_hybrid >= ACCEPT_THRESHOLD：标记 pair 为 `accepted`
7. 如果 current_round >= HARD_MAX_ROUNDS_PER_PAIR：标记 pair 为 `exhausted`
8. **写 `SCENE_AGENT_STATE.json`**

### Phase 5: Loop or Report

- 如果有 active pairs → 回到 Phase 1
- 如果全部 stopped/accepted/exhausted → 生成总结报告

#### 总结报告格式

```markdown
## Scene Agent Loop Summary

| Pair | Rounds | Best Score | Best Round | Final Score | Exit | Actions |
|------|--------|------------|------------|-------------|------|---------|
| obj_003_yaw000 | 3 | 0.5605 | 1 | 0.5345 | patience | O_SCALE_UP_10, S_CONTACT_SHADOW_UP |

### Key Findings
- [哪些动作有效，哪些无效]
- [哪些对象有改善空间，哪些已收敛]

### Recommendations
- [下一步建议]
```

## 绝对不要踩的坑

### 1. Stage1 不要带背景
Stage1/Stage2 生成的参考图必须是**白底纯物体**。pseudo-reference 由 Qwen-Image-Edit-2511 单独生成。

### 2. CUDA_VISIBLE_DEVICES + --device cuda:0
`CUDA_VISIBLE_DEVICES=1` 后进程内只看到一张卡，传 `--device cuda:0`，不是 `cuda:1`。

### 3. 本地改代码必须同步远端
```bash
scp local_file wwz:/aaaidata/zhangqisong/data_build/精确/目标/路径/文件名
```
不能只 scp 到根目录。

### 4. 不跨 yaw 共享动作
同一对象不同角度的问题可能完全不同，每个 (obj, yaw) 独立决策。

### 5. 不要只看分数
**人眼判断优先**。必须同时看渲染图 + trace 文本。`hybrid_score` 只作辅助。

### 6. 不要改基础参数
以下配置视为锁定基线，不要改：Blender路径、scene文件(4.blend)、Python环境、reviewer模型(Qwen3.5)、profile、scene_template、GPU策略。

### 7. accepted=0 不等于视觉失败
`qwen35_freeform_full10` 的 controller summary 有问题，但用户认可其视觉效果。区分 controller-side failure 和 visual-side progress。

### 8. 不要退回全自动 controller
当前主线是"AI 读 VLM 文本 → AI 自己决定动作"的半自动闭环，不是自动 controller 盲跑。

### 9. 做旋转从满意结果出发
优先从 teachercam3_light1 和 full10 的 best_state 出发扩旋转，不要重跑整条静态线。

## 可用动作速查

| 动作 | 效果 | 适用场景 |
|------|------|----------|
| O_SCALE_UP_10 | 物体放大 10% | object_too_small |
| O_SCALE_DOWN_10 | 物体缩小 10% | 物体过大 |
| O_LIFT_SMALL | 上移 0.02 | ground_intersection |
| O_LOWER_SMALL | 下移 0.02 | 浮空 |
| S_CONTACT_SHADOW_UP | 增强接触阴影 | shadow_missing, weak_separation |
| ENV_ROTATE_30 | 环境旋转 30° | scene_light_mismatch |
| ENV_STRENGTH_UP | 环境光增强 | 偏暗 |
| ENV_STRENGTH_DOWN | 环境光减弱 | 过亮 |
| M_VALUE_DOWN | 材质明度降低 | color_shift, 物体过亮 |
| M_VALUE_DOWN_STRONG | 材质明度大幅降低 | 严重过亮 |
| M_SATURATION_DOWN | 饱和度降低 | 色彩过鲜艳 |
| M_ROUGHNESS_UP | 粗糙度提高 | 过度光滑/塑料感 |
| M_HUE_WARM | 色相偏暖 | 冷色偏移 |
| M_SHEEN_UP | 光泽提高 | 缺少材质光泽 |
| L_KEY_UP | 主灯增强 | flat_lighting |
| L_KEY_DOWN | 主灯减弱 | 过曝 |
| L_WORLD_EV_UP | 世界光 EV 提高 | underexposed_global（唯一可靠） |

## 关键远端文件路径

```
代码根目录: /aaaidata/zhangqisong/data_build/
脚本入口:   /aaaidata/zhangqisong/data_build/scripts/run_scene_agent_step.py
工具库:     /aaaidata/zhangqisong/data_build/run_scene_evolution_loop.py
Blender渲染: /aaaidata/zhangqisong/data_build/pipeline/stage4_scene_render.py
VLM审查:    /aaaidata/zhangqisong/data_build/pipeline/stage5_5_vlm_review.py
动作应用:   /aaaidata/zhangqisong/data_build/pipeline/stage5_6_feedback_apply.py
动作空间:   /aaaidata/zhangqisong/data_build/configs/scene_action_space.json
场景模板:   /aaaidata/zhangqisong/data_build/configs/scene_template.json
数据集配置: /aaaidata/zhangqisong/data_build/configs/dataset_profiles/scene_v7.json
Mesh目录:   /aaaidata/zhangqisong/data_build/pipeline/data/meshes/
```

## 典型完整命令

### 查看 pair 状态
```bash
ssh wwz "cat {ROOT}/_shards/{SHARD}/{PAIR}/reviews/{OBJ}_r{NN}_agg.json | python3 -m json.tool"
```

### 查看 VLM 文本（最重要）
```bash
ssh wwz "cat {ROOT}/_shards/{SHARD}/{PAIR}/reviews/{OBJ}_r{NN}_az000_el+00_trace.json | python3 -c \"import json,sys; d=json.load(sys.stdin); print(d['attempts'][0]['assistant_text'][:3000])\""
```

### 跑下一轮
```bash
ssh wwz "CUDA_VISIBLE_DEVICES={GPU} {PYTHON_BIN} {AGENT_STEP_SCRIPT} \
  --output-dir {ROOT}/_shards/{SHARD}/{PAIR} \
  --obj-id {OBJ} --rotation-deg {YAW} \
  --round-idx {NEXT} \
  --actions {ACT1} {ACT2} \
  --control-state-in {ROOT}/_shards/{SHARD}/{PAIR}/states/round{CUR:02d}.json \
  --prev-renders-dir {ROOT}/_shards/{SHARD}/{PAIR}/round{CUR:02d}_renders \
  --device cuda:0 \
  --meshes-dir {MESHES_DIR} \
  --agent-note scene_loop_r{NEXT}"
```
