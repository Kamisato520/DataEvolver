# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

保持使用英文思考，使用中文回答和撰写文档

## Project Overview

ARIS (Auto-Codex-Research-In-Sleep) is a collection of Codex slash skills for autonomous ML research. The system is a **pure-Markdown framework** — every skill is a single `SKILL.md` file with YAML frontmatter. No dependencies, no daemon, no framework.

The core design is **cross-model collaboration**: Codex (executor) + an external LLM via MCP (reviewer). Adversarial review beats self-play.

## Architecture

### Two-tier model architecture

```
Executor (Codex)          Reviewer (external LLM via MCP)
  ├── runs experiments              ├── Codex MCP → GPT-5.4 (default)
  ├── writes code                   └── llm-chat MCP → any OpenAI-compatible API
  ├── modifies papers                   (DeepSeek, MiniMax, Kimi, GLM, etc.)
  └── reads AGENTS.md for server info
```

The executor uses `ANTHROPIC_*` env vars. The reviewer uses either:
- **Codex MCP**: `mcp__codex__codex` / `mcp__codex__codex-reply` tools (stateful, conversation threads)
- **llm-chat MCP**: `mcp__llm-chat__chat` tool (stateless, `LLM_*` env vars in `~/.Codex/settings.json`)

### Skill file format

```markdown
---
name: skill-name
description: One-line description
argument-hint: [optional-arg]
allowed-tools: Bash(*), Read, Write, Edit, mcp__codex__codex
---
# Skill instructions...
```

Skills live in `skills/<name>/SKILL.md`. After development, they are installed to `~/.Codex/skills/` to be usable as `/skill-name` in Codex.

### Three research workflows

| Workflow | Orchestrator skill | What it does |
|----------|--------------------|--------------|
| 1: Idea Discovery | `/idea-discovery` | survey → brainstorm → novelty-check → pilot → refine → plan |
| 2: Auto Review Loop | `/auto-review-loop` | review → fix → experiment → repeat (max 4 rounds) |
| 3: Paper Writing | `/paper-writing` | narrative → outline → figures → LaTeX → PDF |
| Full pipeline | `/research-pipeline` | Workflow 1 → 2 → 3 end-to-end |

### 多模型编码协议 / Multi-Model Coding Protocol

当 Codex 负责编码任务时，按以下四阶段协作流程操作：

```
Codex (规划 & Review)
  ├── Phase 1: 分解任务 → 独立子任务列表
  ├── Phase 2: mcp__codex__codex × N (并行) → GPT-5.4 编码
  ├── Phase 3: 读取并汇总各 Codex 线程输出
  └── Phase 4: 新 mcp__codex__codex 线程 → 联合 review + 试运行
```

**Phase 1 — Plan（分解）**
- 将任务拆分为 3–8 个独立子任务
- 明确每个子任务的：描述、预期输出文件/函数、依赖关系
- 确定最终 trial_run_command（验证整合结果的命令）

**Phase 2 — Code（并行 Codex 线程）**
- 对无依赖关系的子任务：在**同一响应**中并行调用多个 `mcp__codex__codex`
- 每次调用必须带 `config: {"model_reasoning_effort": "xhigh"}`
- 保存所有 threadId（存入 `PIPELINE_STATE.json`）
- 有依赖的子任务等前置完成后再调用

**Phase 3 — Read（汇总各线程输出）**
- 对每个完成的 Codex 线程，提取：创建的文件、实现的函数、错误/TODO、关键代码片段
- 整合所有摘要写入 `CODE_SUMMARY.md`

**Phase 4 — Review（新 Codex review 线程）**
1. 在新 Codex 线程中发起联合 review 请求（携带 CODE_SUMMARY.md 内容）
2. GPT-5.4 返回 review 结果 → 输出追问或最终裁决
3. 如需追问，调用 `mcp__codex__codex-reply(threadId=..., content=[追问])`
4. 重复，最多 3 轮
5. 产出最终裁决：APPROVED 或 NEEDS_FIXES（含具体修改）
6. 若 NEEDS_FIXES：用 Write/Edit 应用修改
7. 运行 `trial_run_command`，记录结果
8. 写入 `REVIEW_VERDICT.md`（评分 1–10、问题列表、修改、试运行结果）

### State persistence

`/auto-review-loop` persists `REVIEW_STATE.json` after each round (round number, threadId, score, pending experiments). This survives context compaction — on resume, it reads this file and continues from where it left off. Set `"status": "completed"` when done.

### MCP servers in this repo

- `mcp-servers/llm-chat/server.py` — generic OpenAI-compatible chat MCP server. Configured via `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL` env vars.
- `mcp-servers/minimax-chat/server.py` — MiniMax-specific variant.

## Setup Commands

```bash
# Install all skills globally
cp -r skills/* ~/.Codex/skills/

# Install a single skill
cp -r skills/auto-review-loop ~/.Codex/skills/

# Test a skill after modifying it
cp -r skills/your-skill ~/.Codex/skills/
# Then in Codex: /your-skill test-argument
```

### Codex MCP (for review skills requiring GPT-5.4)

```bash
npm install -g @openai/codex
codex setup        # set model = "gpt-5.4" in ~/.codex/config.toml
Codex mcp add codex -s user -- codex mcp-server
```

### llm-chat MCP (for any OpenAI-compatible reviewer)

Add to `~/.Codex/settings.json`:
```json
{
  "mcpServers": {
    "llm-chat": {
      "command": "/usr/bin/python3",
      "args": ["/path/to/mcp-servers/llm-chat/server.py"],
      "env": {
        "LLM_API_KEY": "your-key",
        "LLM_BASE_URL": "https://api.deepseek.com/v1",
        "LLM_MODEL": "deepseek-chat"
      }
    }
  }
}
```

`llm-chat` requires `httpx` (`pip install httpx`).

### LaTeX (Workflow 3 only)

```bash
# macOS
brew install --cask mactex && brew install poppler

# Ubuntu
sudo apt install texlive-full latexmk poppler-utils

# Verify
latexmk --version && pdfinfo -v
```

### Auto-allow for overnight runs

Add to `.Codex/settings.local.json` in your research project:
```json
{
  "permissions": {
    "allow": [
      "mcp__codex__codex",
      "mcp__codex__codex-reply",
      "Write", "Edit",
      "Skill(auto-review-loop)"
    ]
  }
}
```

## GPU Server Setup 远程服务器

- 默认服务器：`wwz`（密钥免密）
- GPU：`3 x A800 80GB`
- 本地 shell：PowerShell
- 远端 shell：bash
- 长任务后台运行：**用 `tmux`，不要用 `screen`**
- 远端 Python：`/home/wuwenzhuo/Qwen-VL-Series-Finetune/env/bin/python3`
- Blender：`/home/wuwenzhuo/blender-4.24/blender`

## 当前主线项目：Scene-Aware Blender Rotation 数据集构建与训练准备

### 当前默认目标

当前默认主线已经不是旧的 `Qwen Image Edit 单轴旋转 LoRA` 项目，而是：

1. 用 `stage1 -> stage2 -> stage2.5 -> stage3 -> Blender stage4 -> VLM review` 构建 scene-aware synthetic rotation dataset
2. 产出一致性 `rotation4 / rotation8`
3. 产出标准 `train-ready / split`
4. 为后续训练准备 loader、schema、训练 README

### 当前活跃远端代码根

- `/aaaidata/zhangqisong/data_build_runs/scene_full20_loop_20260404_code`

后续所有 scene 数据集相关操作，默认都在这个根下进行。

### 当前正确的方法论

已经验证正确的方法是：

```text
yaw000 best state -> rotate object -> keep scene fixed
```

具体含义：

- 对每个 object，先只把 `yaw000` 调到最佳状态
- 之后导出 `rotation4 / rotation8` 时，只改 `object.yaw_deg`
- 不旋转相机
- 保持场景、相机、整体光影、材质策略尽量一致

明确不要回退到这些错误方法：

- 不要每个角度各自选最优再拼数据集
- 不要把 rotation 数据集做成相机 orbit multiview

### 当前上游优化根

旧 20 物体主线：

- `pipeline/data/evolution_scene_v7_full20_rotation4_agent_round0_nostage35_20260404`

新 30 物体 bootstrap 根：

- `pipeline/data/evolution_scene_v7_obj021_obj050_yaw000_bootstrap_20260408`

### 当前最新完成的数据集（final）

#### full50 standard final（最新，训练默认入口）

- 一致性 `rotation8 full50`：
  - `pipeline/data/dataset_scene_v7_full50_rotation8_consistent_yaw000_final_20260410`
- `train-ready final`：
  - `pipeline/data/dataset_scene_v7_full50_rotation8_trainready_front2others_final_20260410`
- `split final`：
  - `pipeline/data/dataset_scene_v7_full50_rotation8_trainready_front2others_splitobj_seed42_final_20260410`

其统计是：

- `50` 个 object
- `400` 个视图
- `400` 个 mask
- `350` 个训练对
- split 为：
  - `train = 35 objects / 245 pairs`
  - `val = 7 objects / 49 pairs`
  - `test = 8 objects / 56 pairs`

#### new30 refreshed consistent（最新刷新的新 30 物体）

- `pipeline/data/dataset_scene_v7_obj021_obj050_rotation8_consistent_yaw000_final_20260410`

#### 已存在但不是最新 final 的 geomodal 根

下面这些几何增强根是可用的，但它们是旧的 `20260408` 版本，不是最新 `best-state` 刷新的 final：

- `pipeline/data/dataset_scene_v7_full50_rotation8_geommeta_from_consistent_20260408`
- `pipeline/data/dataset_scene_v7_full50_rotation8_geomodal_trainready_front2others_20260408`
- `pipeline/data/dataset_scene_v7_full50_rotation8_geomodal_trainready_front2others_splitobj_seed42_20260408`

所以当前默认规则是：

- **标准训练先用 `20260410 final train-ready / split`**
- **不要把 `20260408 geomodal` 和 `20260410 final standard` 混成一套实验**

### 当前训练默认入口

如果 Claude/Codex 现在要启动训练，默认应使用：

- `pipeline/data/dataset_scene_v7_full50_rotation8_trainready_front2others_splitobj_seed42_final_20260410`

训练第一版建议：

- 先做 RGB-only baseline
- 先只读取：
  - `instruction`
  - `source_image`
  - `target_image`
- 跑通后再决定是否引入：
  - mask
  - control metadata
  - geometry / depth / normal

### 当前 loader / schema / README

#### 训练侧 loader

- `pipeline/rotation_geomodal_dataset.py`

注意：

- 这个 loader 当前主要对接 `geomodal` 数据根
- 它不是直接针对最新 `20260410 standard final` 写的
- 对 `20260410 standard final`，第一版训练更适合直接读 `pairs/*.jsonl`

#### 训练侧说明文档

- `docs/rotation8_geomodal_training_schema_20260407.md`
- `docs/full50_final_training_readme_for_claude_20260410.md`

其中：

- `full50_final_training_readme_for_claude_20260410.md` 是当前最该看的训练启动文档

### 当前关键脚本

#### 数据集导出 / 构建

- `scripts/export_rotation8_from_best_object_state.py`
- `scripts/merge_rotation_consistent_roots.py`
- `scripts/build_rotation8_trainready_dataset.py`
- `scripts/build_object_split_for_rotation_dataset.py`
- `scripts/build_rotation8_geommeta_from_consistent.py`
- `scripts/build_rotation8_geomodal_trainready_dataset.py`

#### 渲染与评审

- `pipeline/stage4_scene_render.py`
- `pipeline/stage5_5_vlm_review.py`
- `pipeline/stage5_6_feedback_apply.py`
- `scripts/run_scene_agent_step.py`
- `scripts/run_scene_agent_monitor.py`

#### 训练侧检查

- `scripts/inspect_rotation_geomodal_loader.py`

### 当前 VLM 预设规则入口

如果需要回看“Codex 正式接手前，VLM 评价维度和预设规则是怎么做的”，直接看本地打包好的 bundle：

- `deliverables/vlm_preset_rule_bundle_20260410/`
- `deliverables/vlm_preset_rule_bundle_20260410/README.md`

关键文件包括：

- `configs/vlm_review_schema.json`
- `configs/scene_action_space.json`
- `pipeline/stage5_5_vlm_review.py`
- `scripts/run_scene_agent_monitor.py`
- `scripts/run_scene_agent_step.py`

### 当前最重要的工作规则

1. 不要覆盖已有数据集根，新的导出一律并列新建目录
2. `20260408` 版本视为冻结资产，当前最新 standard final 是 `20260410`
3. 当前默认训练入口是 `full50 final split`，不是旧 full20，也不是旧 LoRA 项目
4. 读取 VLM 反馈时，不能只看 `agg.json`，优先读 `trace.json` 自由文本
5. 当自动规则明显 plateau 后，再由 Codex 自己接手难例
6. 第一版训练不要急着混 geometry/depth/normal，先跑通 RGB baseline

### 历史项目说明

旧的 `Qwen Image Edit 单轴旋转 LoRA` 训练项目不是当前默认主线。

如果用户明确切回那个项目，再去看它对应的远端目录和脚本；否则默认忽略那套旧说明，优先按照本节的 scene dataset 主线执行。

## Skill Development

- Each skill is a self-contained `skills/<name>/SKILL.md` with YAML frontmatter
- `allowed-tools` in frontmatter controls what the skill can call
- Skills that need the external reviewer use `mcp__codex__codex` and `mcp__codex__codex-reply` (Codex MCP) or `mcp__llm-chat__chat` (llm-chat MCP)
- Use `$ARGUMENTS` in the skill body to receive the slash command argument
- Parameter overrides use `— key: value` syntax appended to the command (e.g., `/auto-review-loop "topic" — human checkpoint: true`)
- To switch all skills from Codex MCP to llm-chat MCP, use `skills/auto-review-loop-llm/SKILL.md` as reference and ask Codex to rewrite all skills that use `mcp__codex__codex`

## Key Docs

- `docs/LLM_API_MIX_MATCH_GUIDE.md` — configure alternative executor/reviewer model combinations (GLM, Kimi, LongCat, DeepSeek, MiniMax)
- `docs/MODELSCOPE_GUIDE.md` — free tier via ModelScope (2000 calls/day)
- `docs/ALI_CODING_PLAN_GUIDE.md` — Alibaba Coding Plan setup (Kimi + Qwen + GLM + MiniMax)
- `docs/OPENCLAW_ADAPTATION.md` — running ARIS workflows in OpenClaw/OpenHands without Codex
- `docs/NARRATIVE_REPORT_EXAMPLE.md` — example input for Workflow 3 (`/paper-writing`)
- `skills/paper-write/templates/` — LaTeX templates for ICLR 2026, NeurIPS 2025, ICML 2025
