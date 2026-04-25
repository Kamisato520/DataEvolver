# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

保持使用英文思考，使用中文回答和撰写文档

## Project Overview

DataEvolver is a collection of Codex slash skills for autonomous ML research. The system is a **pure-Markdown framework** — every skill is a single `SKILL.md` file with YAML frontmatter. No dependencies, no daemon, no framework.

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

当前主线是 **Scene-Aware 数据合成**，训练只作为验证数据集有效性的下游实验，不是默认工作重点。

| 项目 | 服务器 | 代码目录 / 工作目录 | 用途 |
|------|--------|---------------------|------|
| **Scene-Aware 数据合成主线** | `wwz` | `/aaaidata/zhangqisong/data_build_runs/scene_full20_loop_20260404_code` | 数据构建、Blender 渲染、VLM 质检、数据集派生 |
| LoRA 训练 + 评测 | `zhanghy56_68` | `$WORKDIR/DiffSynth-Studio`，`$WORKDIR=/gemini/platform/public/aigc/aigc_image/zhanghy56_intern/zhangqisong/data-build` | 验证数据集有效性 |
| 共享备用机器 | `zhanghy56_intern` | 与 `68` 共享 `$WORKDIR` | 备用训练/评测，不要和 68 同时写同一文件 |

### wwz（数据构建主力）

- SSH：`wwz`
- GPU：`3 x A800 80GB`
- 本地 shell：PowerShell
- 远端 shell：bash
- 长任务后台运行：**用 `tmux`，不要用 `screen`**
- 场景代码根：`/aaaidata/zhangqisong/data_build_runs/scene_full20_loop_20260404_code`
- 远端 Python：`/home/wuwenzhuo/Qwen-VL-Series-Finetune/env/bin/python3`
- Blender：`/home/wuwenzhuo/blender-4.24/blender`
- 场景文件：`/home/wuwenzhuo/blender/data/sence/4.blend`（注意是 `sence`）
- Qwen3.5-35B：`/data/wuwenzhuo/Qwen3.5-35B-A3B`
- Qwen-Image-Edit-2511：`/data/wuwenzhuo/Qwen-Image-Edit-2511`

### 68 / intern（训练验证）

- SSH：`zhanghy56_68` / `zhanghy56_intern`
- GPU：`8 x H100`
- 共享工作目录：`/gemini/platform/public/aigc/aigc_image/zhanghy56_intern/zhangqisong/data-build`
- Python 环境：`source .venv/bin/activate`
- 依赖安装：优先 `uv pip install`，不要直接 `pip install`
- Qwen-Image-Edit-2511：`/gemini/platform/public/aigc/aigc_image/zhanghy56_intern/zhangqisong/Qwen-Image-Edit-2511`
- Blender：`/gemini/platform/public/aigc/aigc_image/zhanghy56_intern/transfer_part_2/blender/blender-4.2.4-linux-x64`

## 当前主线项目：Scene-Aware Rotation 数据集构建

### 终极目标

目标是把 `/data-build <数据集描述>` 做成自动数据集构建能力：

```text
用户描述数据集目标
  -> prompt / object concept 生成
  -> T2I 物体图
  -> 前景分割
  -> 图生 3D mesh
  -> Qwen Image Edit 2511 生成伪参考图
  -> Blender 场景插入与渲染
  -> VLM 质检与反馈
  -> yaw000 canonical best state
  -> 旋转物体导出 rotation4 / rotation8
  -> train-ready / split / bbox / geometry / brightness 派生
  -> 标准训练数据集
```

### 当前正确方法论

已经验证正确的数据集定义是：

```text
yaw000 best state -> rotate object -> keep scene fixed
```

具体含义：

- 每个物体先只调 `yaw000` 的 canonical 初始状态
- 后续导出 `rotation4 / rotation8` 时只改 `control_state.object.yaw_deg`
- 不旋转相机，不做 camera orbit multiview
- 场景、相机、接地、光影、材质策略尽量保持一致

已证明错误的方法：

- 每个角度 independently 选 best round：会造成跨角度材质和状态漂移
- 旋转相机做 multiview：不符合当前“物体旋转”的训练目标
- 把旧 `best-of-each-pair rotation4` 当 final：只能作历史参考

## 数据集构建 Pipeline（重点）

高层可以理解为三段：`Assets Prep（T2I + 分割 + 3D 建模） -> Blender Scene Render + VLM Refinement -> Dataset Build（配对 + 拆分 + prompt 注入）`。下面的 10 个 stage 是实际执行时需要追踪的细分步骤。

### Stage 1：Assets Prep（文本扩展 → T2I → 分割 → 3D 建模）

输入是用户指定的数据集目标和对象列表。系统先生成每个 object 的文本概念、名称、类别和 T2I prompt，然后一路得到可插入 Blender 的 mesh。Stage 1 内部四个子步骤中间没有人工决策点，通常作为一个 assets prep 阶段处理。

| 子步骤 | 脚本 | 模型 | 输出 |
|--------|------|------|------|
| 文本扩展 | `pipeline/stage1_text_expansion.py` | Claude / LLM，低温度确定性生成 | `prompts.json`，包含 T2I prompt 与物体描述 |
| Text-to-Image | `pipeline/stage2_t2i_generate.py` | Qwen-Image-2512，约 56GB 显存 | `images/{id}.png`，1024×1024 白底物体图 |
| 前景分割 | `pipeline/stage2_5_sam2_segment.py` | SAM2 | `images_rgba/{id}.png`，RGBA 前景 |
| Image-to-3D | `pipeline/stage3_image_to_3d.py` | Hunyuan3D-2.1 | `meshes_raw/{id}.glb`，带 UV/PBR 纹理的 3D mesh |

关键配置：

- `configs/seed_concepts/scene_full50_objects.json`
- `configs/seed_concepts/scene_obj021_050_objects.json`
- `configs/dataset_profiles/scene_v7_full50_loop.json`

关键约束：

- T2I 输出必须是白底，任何场景背景都会破坏 SAM 分割和 Hunyuan3D 建模质量。
- Hunyuan3D 需要 CUDA 扩展，如 `custom_rasterizer` 和 `DifferentiableRenderer`。
- Full50 扩容时，`obj_021` 到 `obj_050` 是从 Stage 1 重新开始，不是复用旧 mesh。
- 资源估计约为 50 min / 1×A800，具体取决于 T2I 和 Hunyuan3D 队列状态。

### Stage 2：T2I 物体图生成

使用 Qwen Image 生成白底物体图。白底是硬约束，因为后续 SAM 分割和图生 3D 对背景很敏感。

关键文件：

- `pipeline/stage2_t2i_generate.py`

输出示例：

- `pipeline/data/dataset_scene_v7_obj021_obj050_stage1_assets_20260408/images/`

### Stage 2.5：SAM2 前景分割

把 T2I 图像分割为 RGBA 前景，给图生 3D 使用。

关键文件：

- `pipeline/stage2_5_sam2_segment.py`

注意：

- 当前 RGBA 物体图通常不含地板，默认不需要做 floor removal
- `stage3_5_mesh_sanitize.py` 只在 Hunyuan3D 产出伪地板面时作为条件化清理步骤，不要无条件插回主流程

### Stage 3：Image-to-3D Mesh

从 RGBA 前景图生成 `.glb` / mesh 资产。

关键文件：

- `pipeline/stage3_image_to_3d.py`
- `pipeline/stage3_5_mesh_sanitize.py`

新 30 物体 mesh 来源：

- `pipeline/data/dataset_scene_v7_obj021_obj050_stage1_assets_20260408/meshes/`

这批 mesh 后续已 symlink 到 scene pipeline 可读取的位置。不要假设 `obj_021~050` 的 canonical mesh 一开始就在旧 `meshes/` 根下。

`stage3_5_mesh_sanitize.py` 只作为 Hunyuan3D 伪地板面的应急清理步骤：当 `floor_confidence > 0.78` 且确认 mesh 中存在伪地板时才允许在新根中启用。当前 final 数据集不依赖重新启用 stage3.5，不要把它作为默认步骤插回主流程。

### Stage 4：Pseudo-Reference 生成

使用 **Qwen Image Edit 2511** 把目标物体插入背景图，生成伪参考图。它不是最终训练 GT，而是给 VLM 和 agent 一个“什么样才算好”的视觉锚点。

作用：

- 让 VLM 对比 `current_render` 和 `pseudo_ref`
- 避免 VLM 只给出泛泛的“光线可以更好”
- 帮助定位材质、亮度、接地、构图、场景融合问题

如果伪参考图生成失败或明显异常，应在 manifest 中标注，并 fallback 到弱参考图，而不是静默使用坏参考。

### Stage 5：Blender Scene-Aware 渲染

把 3D 物体插入固定场景 `4.blend`，用 Blender 渲染 RGB、mask 和 metadata。

关键文件：

- `pipeline/stage4_scene_render.py`
- `configs/scene_template*.json`

关键约束：

- `blend_path` 指向 `/home/wuwenzhuo/blender/data/sence/4.blend`
- 使用 scene 里的既有世界和灯光，不要随意清空 world node tree
- 支持接地检测、support plane、mask、render_metadata、control_state
- `force_reference_material: false` 不能被阈值逻辑偷偷覆盖
- 渲染引擎默认是 Blender CYCLES，目标输出为 1024×1024，当前 rotation 数据集按 512 samples 规格记录

v7 修复过的关键问题：

| 问题 | 修复规则 |
|------|----------|
| `ensure_world_environment()` 覆写 `4.blend` 的 world node tree，导致场景灯光丢失 | 使用 `use_existing_world=True`，不要随意清空 world |
| 场景灯光被 `hide_render=True` 隐藏 | 改成 `scale_existing_lights(factor)` 调暗或增强 |
| 相机距离固定导致小物体出框 | 使用与物体 `max_span` 相关的自适应距离 |
| 地面检测误用 `Cube`，但实际地板是 `Plane` | 使用 `support_object_name="Plane"` |

### Stage 6：VLM Review + Agentic Refinement

渲染结果交给 VLM 评估。早期实现包含预设 schema / issue tag / action space，后续经验表明自由文本 `trace.json` 对真实修复更有价值。

关键文件：

- `pipeline/stage5_5_vlm_review.py`
- `pipeline/stage5_6_feedback_apply.py`
- `configs/vlm_review_schema.json`
- `configs/scene_action_space.json`
- `scripts/run_scene_agent_step.py`
- `scripts/run_scene_agent_monitor.py`
- `deliverables/vlm_preset_rule_bundle_20260410/README.md`

自动 loop 的基本形态：

```text
Round R = 1
while R <= MAX_ROUNDS:
    vlm_text = Qwen3.5(current_render, pseudo_ref, review_prompt)
    if "VERDICT: GOOD" in vlm_text: stop
    actions = parse_freeform(vlm_text)
    if actions repeats for 2 rounds: stop as plateau
    apply bounded render/config updates
    rerender
    R += 1
```

工作方式：

- VLM 评价光照、物体完整性、构图、材质/语义质量、整体可用性
- 自动规则先把常见问题映射到动作，如 `L_WORLD_EV_UP`、`S_CONTACT_SHADOW_UP`、`O_SCALE_UP_10`、`L_KEY_UP`、`ENV_STRENGTH_UP`、`M_SATURATION_DOWN`、`M_VALUE_DOWN`、`O_LIFT_SMALL`、`O_LOWER_SMALL`
- 退出条件是 VLM 明确 `VERDICT: GOOD`、连续动作重复导致 plateau、或达到 `MAX_ROUNDS=5`
- 预设规则阶段的修改范围应受限在 scene template 和灯光/材质函数，保持小 diff，避免把控制器改成不可追踪状态
- 当规则化动作 plateau 后，由 Codex 直接读 `trace.json` 自由文本接手难例
- 判断时不能只看 `agg.json` 分数，必须读 `reviews/*_trace.json` 的 `assistant_text`

常见问题：

- `underexposed`
- `flat_lighting`
- `weak grounding`
- `floating_visible`
- `color_shift`
- `material too plastic`
- `object too small`
- `missing_reference_rgb`

历史 pilot 结果显示，10 object pilot 的平均 confirmed score 从约 0.6043 提升到 0.6390，后续质量天花板主要受 HDRI/mesh 系统性不匹配限制。因此当前策略不是无限调 controller，而是先得到可用 canonical state，再进入一致性 rotation 导出。

### Stage 7：Canonical Yaw000 Best State

VLM loop 的最终目的不是每个角度无限调优，而是为每个物体得到一个稳定的 `yaw000` canonical base state。

上游优化根：

- 旧 20 物体：`pipeline/data/evolution_scene_v7_full20_rotation4_agent_round0_nostage35_20260404`
- 新 30 物体：`pipeline/data/evolution_scene_v7_obj021_obj050_yaw000_bootstrap_20260408`

新 30 物体里曾出现低分和白模/缺色问题，后续通过 Codex 接手修复，包括 `obj_050 fire_extinguisher`、`obj_028 wheelbarrow`、`obj_044 baseball_bat`、`obj_047 barbecue_grill` 等颜色恢复。

### Stage 8：Consistent Rotation Export

从 canonical `yaw000` best state 导出一致性旋转数据：

- `rotation4`: `0, 90, 180, 270`
- `rotation8`: `0, 45, 90, 135, 180, 225, 270, 315`

关键脚本：

- `scripts/export_rotation8_from_best_object_state.py`
- `scripts/merge_rotation_consistent_roots.py`

Full50 final 根：

- `pipeline/data/dataset_scene_v7_full50_rotation8_consistent_yaw000_final_20260410`
- `pipeline/data/dataset_scene_v7_obj021_obj050_rotation8_consistent_yaw000_final_20260410`

### Stage 9：Train-Ready / Object-Disjoint Split

把 consistent rotation8 整理成训练 pairs：

- source 固定为 `yaw000 / front view`
- target 是其它 7 个角度
- 每个 object 产生 7 个 pair
- 50 个 object 共 350 个 pair

split 规则：

- `seed = 42`
- train：35 objects / 245 pairs
- val：7 objects / 49 pairs
- test：8 objects / 56 pairs
- object-disjoint，同一物体不能跨 split

关键脚本：

- `scripts/build_rotation8_trainready_dataset.py`
- `scripts/build_object_split_for_rotation_dataset.py`

标准 full50 final：

- `pipeline/data/dataset_scene_v7_full50_rotation8_trainready_front2others_final_20260410`
- `pipeline/data/dataset_scene_v7_full50_rotation8_trainready_front2others_splitobj_seed42_final_20260410`

Pair 文件同时输出 CSV 和 JSONL。核心字段包括 source/target 图像路径、object id、source/target yaw、相对旋转角、instruction、物体名称/描述、bbox 或派生字段。训练前必须确认 `source_image / target_image` 指向期望视觉输入，不要只看目录名判断数据版本。

当前 prompt v3 通过 Stage 1 `prompts.json` 中的 `name` 字段指代物体，格式为：

```text
Rotate this {object_description} clockwise from front view to {view}.
```

角度到 view name 的默认映射：

| 角度 | view name |
|------|-----------|
| 45° | front-right view |
| 90° | right side view |
| 135° | back-right view |
| 180° | back view |
| 225° | back-left view |
| 270° | left side view |
| 315° | front-left view |

### Stage 10：BBox / Brightness / Geometry 派生

这些都是并列新根，不能原地污染已有数据集。

#### BBox 派生（20260414）

根据 mask 自动反推出主物体 bbox，并绘制红色矩形框：

- `pipeline/data/dataset_scene_v7_full50_rotation8_trainready_front2others_bboxmask_final_20260414`
- `pipeline/data/dataset_scene_v7_full50_rotation8_trainready_front2others_splitobj_seed42_bboxmask_final_20260414`

关键脚本：

- `scripts/build_rotation_bbox_condition_dataset.py`

注意：

- bbox 图在 `bbox_views/`
- bbox 标注在 `bbox_annotations/`
- bbox 相关字段包括 `source_bbox_xyxy`、`source_bbox_xywh`、归一化坐标等
- 当前 LoRA 主训练不应直接用 bbox overlay 做视觉输入，除非实验明确要求

#### Blender 重渲染 Bright 数据集（20260416）

发现原数据集物体偏暗后，创建 `scene_template_bright_camera.json` 并全量重渲 50×8：

```text
consistent_bright_20260416
  -> trainready_front2others_bright_20260416
  -> splitobj_seed42_bright_20260416
  -> bboxmask_bright_final_20260416
```

亮度关键参数：

| 参数 | 原值 | 新值 |
|------|------|------|
| `scale_existing_lights` | 0.16 | 0.54 |
| `scale_world_background` | 0.1252 | 0.42 |
| `key_light_energy` | 7.0 | 14.4 |
| `camera_fill_energy` | 316.3 | 600.0 |
| `target_brightness` | 0.398 | 0.66 |
| `adaptive_brightness_max_gain` | 1.18 | 1.9 |
| `filmic_gamma` | 0.50 | 0.58 |

#### 后处理 BrightObj 数据集（20260416，辅助）

这是在 `bboxmask final 20260414` 上只提亮 mask 内物体区域的后处理版本，不是 Blender 重渲染：

- `pipeline/data/dataset_scene_v7_full50_rotation8_trainready_front2others_bboxmask_brightobj_refstage35_final_20260416`
- `pipeline/data/dataset_scene_v7_full50_rotation8_trainready_front2others_splitobj_seed42_bboxmask_brightobj_refstage35_final_20260416`

关键脚本：

- `scripts/build_rotation_object_brightened_dataset.py`

不要把它和 Blender 重渲染的 `bboxmask_bright_final_20260416` 混成同一实验。

#### Geometry / Depth / Normal 派生（旧 20260408）

已有几何增强根可参考，但不是最新 bright final：

- `pipeline/data/dataset_scene_v7_full50_rotation8_geommeta_from_consistent_20260408`
- `pipeline/data/dataset_scene_v7_full50_rotation8_geomodal_trainready_front2others_20260408`
- `pipeline/data/dataset_scene_v7_full50_rotation8_geomodal_trainready_front2others_splitobj_seed42_20260408`

几何模态包含：

- camera intrinsics / extrinsics
- object pose
- 2D / 3D bbox
- depth
- normal

相关文档：

- `docs/rotation8_camera_pose_depth_normal_plan_20260407.md`
- `docs/rotation8_geomodal_training_schema_20260407.md`

## 数据集 Lineage（按时间线）

当前不要只按目录名判断数据集新旧，优先按 lineage 理解：

```text
20260407 full20 frozen roots（只读基线，20 obj）
  -> 20260408 full50 expansion（新增 obj_021-050，合并 full20）
  -> 20260410 full50 standard train-ready + object-disjoint split
  -> 20260416 full50 bright（亮度增强 + clockwise prompt）
  -> 20260418 full50 bright objinfo（物体描述 prompt v3，当前推荐训练验证入口）
```

当前推荐训练验证入口位于 68 / intern 共享 `$WORKDIR` 下：

- `dataset_scene_v7_full50_rotation8_trainready_front2others_splitobj_seed42_bboxmask_bright_final_20260416`
- `dataset_scene_v7_full50_rotation8_trainready_front2others_splitobj_seed42_bboxmask_bright_objinfo_20260418`

注意：这些训练验证入口和 wwz 上的 scene 构建根不是同一层概念。wwz 负责生产和派生数据；68 / intern 负责消费数据做 LoRA 训练与评测。

## 当前训练验证状态

训练不是默认主线，只用于证明数据集有效。

实验演进：

```text
实验 1 RGB baseline
  -> 实验 3 BBox
  -> 实验 4 Bright + Clockwise + raw input
  -> 实验 5 Object-Info Prompt
```

当前最优验证结果来自实验 5：

- 数据集：`bboxmask_bright_objinfo_20260418`
- Prompt：`Rotate this {object_description} clockwise from front view to {view}.`
- Checkpoint：`output/rotation8_bright_objinfo_rank32/epoch_0029/lora.safetensors`
- SpatialEdit-Bench 上 ours_objinfo 在传统指标整体领先，VIEScore 的 `Score_cons` 最优

SpatialEdit-Bench 关键结果快照：

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ | CLIP-I ↑ | DINO ↑ | FID ↓ |
|--------|--------|--------|---------|----------|--------|-------|
| base | 15.66 | 0.6623 | 0.3304 | 0.8807 | 0.8517 | 65.47 |
| fal | 15.76 | 0.6545 | 0.3443 | 0.8747 | 0.8405 | 68.35 |
| ours_objinfo | 16.63 | 0.7296 | 0.2564 | 0.9050 | 0.8895 | 50.83 |

VIEScore 快照：

| Method | Score_view ↑ | Score_cons ↑ | Overall ↑ |
|--------|--------------|---------------|-----------|
| base | 0.7746 | 0.9020 | 0.7415 |
| fal | 0.7234 | 0.8658 | 0.6782 |
| ours_objinfo | 0.7705 | 0.9709 | 0.7624 |

训练输入硬规则：

- `source_image / target_image` 是训练脚本直接读取字段
- 当前主训练应指向 `views/` 下无 bbox 原图
- bbox overlay 只作为辅助字段或特定 ablation，不是默认视觉输入

## 当前关键脚本索引

### 数据构建主链路

- `pipeline/stage1_text_expansion.py`
- `pipeline/stage2_t2i_generate.py`
- `pipeline/stage2_5_sam2_segment.py`
- `pipeline/stage3_image_to_3d.py`
- `pipeline/stage3_5_mesh_sanitize.py`
- `pipeline/stage4_scene_render.py`
- `pipeline/stage5_5_vlm_review.py`
- `pipeline/stage5_6_feedback_apply.py`
- `scripts/build_scene_assets_from_stage1.py`
- `scripts/bootstrap_scene_yaw000_objects.py`
- `scripts/build_scene_full50_expansion_pipeline.py`

### 数据集导出 / 派生

- `scripts/export_rotation8_from_best_object_state.py`
- `scripts/merge_rotation_consistent_roots.py`
- `scripts/build_rotation8_trainready_dataset.py`
- `scripts/build_object_split_for_rotation_dataset.py`
- `scripts/build_rotation_bbox_condition_dataset.py`
- `scripts/build_rotation_object_brightened_dataset.py`
- `scripts/build_rotation8_geommeta_from_consistent.py`
- `scripts/build_rotation8_geomodal_trainready_dataset.py`

### VLM loop / 规则配置

- `configs/vlm_review_schema.json`
- `configs/scene_action_space.json`
- `scripts/run_scene_agent_step.py`
- `scripts/run_scene_agent_monitor.py`
- `deliverables/vlm_preset_rule_bundle_20260410/`

## 当前最重要的工作规则

1. **数据集构建是主线**，LoRA 训练只是验证数据集质量，不要反过来主导数据构建文档。
2. 不要覆盖已有数据集根，任何新构建都用并列新目录。
3. `rotation` 的定义是旋转物体，不是旋转相机。
4. 最终 consistent 数据必须来自 `yaw000 canonical best state`，不要每角度独立选 best。
5. `stage3_5_mesh_sanitize.py` 只在确认 Hunyuan3D 伪地板时作为新根应急清理步骤，不是默认 final 流程。
6. `obj_021~050` 的 mesh 来源于 Stage1-3 sandbox，不要假设旧 `meshes/` 里天然存在。
7. 看到 `bright` 时必须区分 Blender 重渲染版和后处理 `brightobj_refstage35` 版。
8. 当前 LoRA 主训练默认用 raw `views/`，不要误把 `bbox_views/` 当默认输入。
9. 读 VLM 结果必须优先看 `trace.json` 自由文本，`agg.json` 分数只能辅助。
10. 68 和 intern 共享磁盘，不要并发写同一输出路径。
11. 所有远端长任务统一使用 `tmux`。
12. 不要跳过 object-disjoint split 直接用 full train-ready root 做训练验证。
13. 不要在数据集根目录内写训练日志、cache 或临时中间结果。
14. `full20` frozen roots 只读，禁止原地修补。

## 历史项目说明

旧的 `Qwen Image Edit 单轴旋转 LoRA` 单机说明不是当前默认主线。

如果用户明确切回旧项目，再按旧路径处理；否则默认优先按照：

- `CLAUDE.md`
- `docs/build-dataset-report.md`
- `docs/history/dataset_construction_history.md`
- `docs/server_reference.md`
- `notion-weekly/week2_0404-0407.md`
- `notion-weekly/week3_0408-0414.md`
- `notion-weekly/week4_0415-0418.md`
- `notion-weekly/lora_experiments_full_record.md`

来理解当前工作。

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
- `docs/OPENCLAW_ADAPTATION.md` — running DataEvolver workflows in OpenClaw/OpenHands without Codex
- `docs/NARRATIVE_REPORT_EXAMPLE.md` — example input for Workflow 3 (`/paper-writing`)
- `docs/server_reference.md` — 当前服务器、路径、脚本和常用命令
- `docs/history/dataset_construction_history.md` — 数据集构建历史、方法论和踩坑记录
- `notion-weekly/lora_experiments_full_record.md` — LoRA 验证实验的完整记录和评测结果
- `skills/paper-write/templates/` — LaTeX templates for ICLR 2026, NeurIPS 2025, ICML 2025
