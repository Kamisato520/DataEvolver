# CLAUDE.md

保持使用中文回复、使用中文写文档，使用英文思考和搜索

---

## 终极目标

**一句话**：在 Claude Code / Codex 终端输入 `/data-build <数据集描述>` → 自动完成从 prompt 生成到 3D 渲染到 VLM 质检的全流程，产出训练就绪的高质量数据集。

### 里程碑

1. **`/data-build` 一键 Agent/Skill**
   - 将 stage1→2→2.5→3→4→5 整条 pipeline 封装为单个 skill 或 agent
   - 输入：自然语言数据集描述（如"100 个家具物体，4 个旋转角度，场景光照"）
   - 输出：训练就绪的数据集（图像 + metadata + QC 报告）
   - 支持多种数据集类型（不限于旋转：姿态变化、光照变化、材质变化等）

2. **Blender 操控能力增强**
   - 构建专用 skills，教会 Claude/Codex 理解和操作 Blender（场景搭建、灯光布置、材质调整、相机控制等）
   - 整合现有社区工具和工作流为 skills（如 BlenderGPT 风格的自然语言→Blender 操作）
   - 让 AI 能直接通过 Blender Python API 进行精细控制，而非只靠预设动作空间

3. **成品质量提升**
   - 丰富 3D 资产库（更多物体类型、更高质量的 mesh）
   - 丰富背景/场景资产（多种场景模板、HDRI 环境）
   - 引入资产废弃与补生机制（已在实现中），自动淘汰低质量资产
   - VLM loop 持续优化渲染参数直到 reviewer 满意

4. **降本：简单环节用国产便宜模型**
   - stage1（prompt 生成）：MiniMax M2.7 / GLM 5.1 / Qwen 3.6 Plus 等
   - 通过 `mcp-servers/llm-chat` MCP 接入任意 OpenAI 兼容 API
   - 只在关键环节（VLM review、复杂决策）保留 Qwen3.5-35B / Claude
   - 目标：在不降质的前提下大幅降低 API 成本

---

## 当前主线项目（2026-04-03）

**当前主线**：Scene-Aware Blender Render + Qwen3.5 VLM Loop，不是 LoRA 训练项目。

- 主线代码目录（远端）：`/aaaidata/zhangqisong/data_build/`
- LoRA 训练项目（另一独立项目）：`wwz:/gemini/.../test/`，不要混淆

### 核心方法链

```
T2I 生成物体图 → SAM3 抠图得 RGBA → 3D 重建（无地板，不需要 stage3.5）
→ Blender 插入场景（4.blend）→ Cycles 渲染
→ Qwen3.5-35B-A3B freeform review（thinking=True）
→ AI 亲自读 trace.json 的 assistant_text（自由文本）
→ AI 根据自由文本决定下一轮动作（不能脚本自动选）
→ 重新渲染 → 重复直到 reviewer 明确说 keep / acceptable
```

**stage3.5 已废弃**：SAM3 处理 T2I 结果得到的 RGBA 图像做 3D 重建，本身不包含地板，无需再做地板消除。

### 活跃目录与基线

| 类型 | 路径 |
|------|------|
| **Active 主线** | `pipeline/data/evolution_scene_v7_qwen35_rotation4_agent_round0_obj001_obj003_obj007_obj008_20260402/` |
| 静态基线1（最稳定）| `pipeline/data/evolution_scene_v7_pseudoref_multigpu_20260401_teachercam3_light1/` |
| 静态基线2（可接受）| `pipeline/data/evolution_scene_v7_qwen35_freeform_full10_20260402/` |
| 旧冻结目录（不要误判）| `...rotation4_freeform...`（非 active） |

Active 主线对象：obj_001/003/007/008，各 4 个 yaw（0/90/180/270），共 16 个 pair。

### 当前状态（2026-04-03）

- **循环已停止**，3 块 A800 全部空闲
- GPU0 shard：round 10 后 `launch_failed`
- GPU1/GPU2：pair 跑了 47-59 轮，分数卡在 0.47-0.52，反复选同样无效 action
- 主要诊断：`flat_low_contrast`、`color_shift`——当前 action space 能力有限
- 无活跃 tmux 窗口

### 关键脚本

| 脚本 | 作用 |
|------|------|
| `scripts/run_scene_agent_monitor.py` | 持续 monitor：读 VLM 文本 → 生成 decision → 发起下一轮 |
| `scripts/run_scene_agent_step.py` | 单 pair 单步：载入 state → 渲染 → VLM review（不做 decision） |
| `pipeline/stage4_scene_render.py` | Blender scene-aware 渲染核心 |
| `pipeline/stage5_5_vlm_review.py` | Qwen3.5 reviewer 入口，freeform + thinking |
| `pipeline/stage5_6_feedback_apply.py` | action apply，含 anti-oscillation 逻辑 |
| `configs/scene_action_space.json` | 29 个原子动作（lighting/object/scene/material） |
| `configs/dataset_profiles/scene_v7.json` | scene profile（accept=0.78, reject=0.35） |
| `configs/scene_template.json` | Blender 场景模板（4.blend, CYCLES 512 samples） |

### 关键配置（scene_template.json）

- `blend_path`: `/home/wuwenzhuo/blender/data/sence/4.blend`（注意是 `sence` 非 `scene`）
- `use_existing_world: true`——**不能**覆盖 world node tree
- `scale_existing_lights: 0.3`——**不能** disable 灯光，只能 scale
- `support_object_name: "Plane"`，`support_plane_mode: ground_object_raycast`
- `render_engine: CYCLES`，`cycles_samples: 512`，`render_resolution: 1024`

### 如何读 reviewer 结果

**优先看 `trace.json`，不是 `agg.json`**：

```bash
# 读自由文本
python3 -c "import json; d=json.load(open('<trace.json>')); print(d['attempts'][-1]['assistant_text'])"
```

从自由文本识别主要矛盾：color shift / flat lighting / weak grounding / shadow missing / material too plastic。`agg.json` 只作辅助（hybrid_score、issue_tags）。只有 reviewer 明确给出 keep 才能停。

### 继续 Loop 的入口

直接调用 skill 继续当前主线：

```
/scene-agent-loop continue
/scene-agent-loop status          # 只查状态，不发起新渲染
/scene-agent-loop obj_001,obj_003  # 指定对象
```

该 skill 会自动：SSH 连服务器 → 读 trace.json 自由文本 → 分析问题 → 决定动作 → 发起下一轮渲染（3 GPU 并行）。常量：SOFT_TARGET=5 轮/pair，HARD_MAX=15 轮/pair，ACCEPT_THRESHOLD=0.78。

### 已知坑

1. `rotation4_freeform...` 是旧冻结目录，不是 active
2. 不能只看 score / agg.json，必须读 trace.json 自由文本
3. 不能主动退出 loop（reviewer 未 keep 则继续）
4. GPU OOM ≠ 渲染失败——无新 agg.json 时先查 reviewer OOM
5. 不能让脚本自动按分数选 action，AI 必须亲自读文本后决策
6. stage3.5（地板消除）已废弃，不要再启用

---

## 远程服务器（wwz）

- SSH: `wwz`（密钥免密）
- GPU: 3×A800 80GB
- **Python 环境**：直接用全路径 `/home/wuwenzhuo/Qwen-VL-Series-Finetune/env/bin/python3`（不用 conda activate，conda 不可用）
- 代码目录: `/aaaidata/zhangqisong/data_build/`
- Blender binary: `/home/wuwenzhuo/blender-4.24/blender`
- 场景文件: `/home/wuwenzhuo/blender/data/sence/4.blend`
- Qwen3.5-35B 模型: `/data/wuwenzhuo/Qwen3.5-35B-A3B`
- 模型下载目录: `/huggingface/model_hub`（下载前 `export HF_ENDPOINT=https://hf-mirror.com`）
- **必须用 tmux**（screen 不可用），命名：`tmux new -s claudecode-research-X`
- 长任务确认正常运行后停止监测，训练完成用户会唤醒
- GPU 使用前先检查占用，有空闲卡优先用空闲卡，全占用则等待

### 快速命令

```bash
# 检查 GPU
ssh wwz "nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv"

# 查最新 decisions
ssh wwz "find <active-dir> -name '*_decision.json' | sort | tail -10"

# 读 trace 自由文本
ssh wwz "python3 -c \"import json; d=json.load(open('<path>')); print(d['attempts'][-1]['assistant_text'])\""
```

---

## 工作规则

### 任务执行

- **非平凡任务**先进 plan mode（多文件修改/架构决策），简单改动直接执行
- 复杂方案先让 Codex review（2-3 轮），再执行
- 子 agent 分配：Read/Explore 用 haiku（省成本），规划/review 用 opus
- 并行处理独立任务，>5 个文件必须用并行子 agent
- threadId 必须落盘到 PIPELINE_STATE.json，防 context compact 丢失

### 多模型协作（Opus 负责规划时）

使用 `/opus-coding` skill 触发完整的四阶段协作流程：

```
/opus-coding "实现 XXX 功能"
```

流程：
```
Phase 1: Opus 分解任务（3-8 个子任务）
Phase 2: mcp__codex__codex × N 并行编码（config: {"model_reasoning_effort": "xhigh"}）
Phase 3: Agent(model="haiku") × N 整理 Codex 输出 → CODE_SUMMARY.md
Phase 4: 新 Codex 线程 review → Opus 裁决（APPROVED / NEEDS_FIXES）→ 试运行 → REVIEW_VERDICT.md
```

Codex review 调用：`mcp__codex__codex(prompt="...", model="codex-5.4-high", cwd="...")`

### 安全规则

- 不要在本地测试，改完代码 push/scp 到服务器再测
- 编辑大文件（>300 LOC）前先清理死代码，再做实质改动
- 每次 edit 前重读文件，edit 后验证结果（Edit 工具在 old_string 不匹配时静默失败）
- 对文件超 500 行，用 offset+limit 分段读，不要假设单次读完整
- 遇到错误先分析根因，不要盲目重试相同操作
- Python 项目无 type-checker，明确说明而非假装通过验证

---

## 项目结构（参考）

本地仓库 `ARIS/` 包含：
- `pipeline/` — 数据合成主流程（stage1 文本扩展 → stage2 T2I → stage3 3D重建 → stage4 Blender渲染 → stage5 VLM review）
- `scripts/` — 运行脚本（agent monitor/step 等）
- `configs/` — 动作空间、场景模板、dataset profiles
- `refine-logs/` — 运行产物
- `docs/` — 文档（handoff 说明等）
- `feishu-claude-code/` — 飞书 bot（独立 git）
- `mcp-servers/` — MCP server 实现（llm-chat、minimax-chat、claude-review 等）
- `skills/` — Claude Code slash skills

**pipeline 阶段说明**（~~stage3.5~~ 已废弃）：

| Stage | 说明 |
|-------|------|
| stage1 | 文本扩展 |
| stage2 | T2I 生成 |
| stage3 | 图像→3D（SAM3 抠图 RGBA 输入，3D 结果不含地板） |
| ~~stage3.5~~ | ~~地板消除~~（**废弃**，SAM3 已解决此问题） |
| stage4 | Blender scene-aware 渲染 |
| stage5 | VLM review + feedback apply |

---

## 可用 Skills 速查

Skills 位于 `skills/` 目录，用 `/skill-name` 调用。

| Skill | 调用 | 适用场景 |
|-------|------|---------|
| `scene-agent-loop` | `/scene-agent-loop continue` | **当前主线**：继续 VLM loop，读 trace 自由文本 → 决策 → 多 GPU 渲染 |
| `opus-coding` | `/opus-coding "任务描述"` | 复杂编码任务：Opus 规划 → Codex 并行写代码 → Haiku 汇总 → 联合 review |
| `run-experiment` | `/run-experiment "运行描述"` | 在远端 GPU 服务器部署/运行训练或评估任务 |
| `research-pipeline` | `/research-pipeline "方向"` | 全流程 ARIS pipeline（idea → 合成 → 训练 → 分析） |
| `idea-discovery` | `/idea-discovery "方向"` | 从任务方向生成 FINAL_PROPOSAL.md 和数据需求 |
| `dataset-synthesis-gate` | `/dataset-synthesis-gate "plan路径"` | 双路数据合成（Blender+T2I）、QC 过滤、readiness gate |
| `experiment-bridge` | `/experiment-bridge "plan路径"` | 实现并运行 evaluator-VLM 训练实验 |
| `analyze-results` | `/analyze-results "结果路径"` | 分析 evaluator-VLM 结果，输出部署建议 |
| `dataset-eval-pipeline` | `/dataset-eval-pipeline "目标"` | 端到端数据集+评估器完整流水线 |

**当前主线只需关注**：`/scene-agent-loop`（渲染 loop）和 `/opus-coding`（代码修改）。
