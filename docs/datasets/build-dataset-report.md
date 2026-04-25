# Rotation8 数据集构建 Pipeline — 核心是 VLM-Driven 渲染优化 Loop

**日期**：2026-04-17
**对标产出**：`bboxmask_bright_final_20260416`（50 物体 × 8 视角，1024×1024）

---

## 总体流程一图

```
文本描述
  │
  ▼
┌──────────────────────┐
│ S1  Assets Prep      │  Claude 扩写 → Qwen-Image T2I → SAM2 分割 → Hunyuan3D 建模
└──────────────────────┘
  │  白底图 + 3D mesh
  ▼
┌──────────────────────┐
│ S2  Pseudo-Reference │  Qwen-Image-Edit 为每个 obj 生成"理想场景渲染"锚点图
└──────────────────────┘
  │  每 obj 一张参考图（VLM 对照的"好标准"）
  ▼
┌──────────────────────┐
│ S3  Baseline Render  │  以当前 bright camera 配置作为起点，跑一轮 50×8 渲染
└──────────────────────┘
  │  初版渲染图（不求完美，仅作 loop 入口）
  ▼
╔════════════════════════════════════════════════════════════════╗
║ S4  VLM Refinement Loop（★ 项目核心）                          ║
║                                                                ║
║    render  +  pseudo-ref                                       ║
║        │                                                       ║
║        ▼                                                       ║
║    Qwen3.5 对比打分 + 自由文本点评（非 JSON）                  ║
║        │                                                       ║
║        ▼                                                       ║
║    主 agent 解析自由文本 → 动作清单                            ║
║        │                                                       ║
║        ▼                                                       ║
║    codex-exec-worker（独立终端）改 scene_template /            ║
║    stage4_scene_render 的光照/材质/构图参数                    ║
║        │                                                       ║
║        ▼                                                       ║
║    主 agent 校验 git diff + validation → accept                ║
║        │                                                       ║
║        ▼                                                       ║
║    重新渲染 → 回到 Qwen3.5 → 下一轮                            ║
║                                                                ║
║    直到 VLM 输出 ✅VERDICT: GOOD  或  饱和  或  5 轮上限        ║
╚════════════════════════════════════════════════════════════════╝
  │  收敛后的渲染图（视觉质量达标）
  ▼
┌──────────────────────┐
│ S5/S6  Dataset Build │  一笔带过：rotation8 配对 + object-disjoint split + bbox 注入 + pairs 修复 + readiness gate
└──────────────────────┘
  │
  ▼
最终数据集（train 245 / val 49 / test 56）
```

---

## S1 — Assets Prep（合并 text+T2I+SAM2+3D）

四步合并为一个 stage，因为彼此之间**无决策点**。

| 子步 | 模型 | 产出 |
|------|------|------|
| 文本扩写 | Claude（T=0） | 50 条物体 prompt |
| 文生图 | Qwen-Image-2512 | 白底物体图 ×50 |
| 前景分割 | SAM2 | RGBA 前景 ×50 |
| 图生 3D | Hunyuan3D-2.1 | `.glb` mesh ×50 |

**硬约束**：Stage 2/3 必须白底，不能带场景背景，否则下游 3D 质量崩坏。

---

## S2 — Pseudo-Reference：给 VLM 一个"好标准"

**动机**：VLM 单独看一张渲染图，只会泛泛而谈（"光线可以更好"），无法指出具体差距。必须给它一个"理想该是什么样"的参考。

**做法**：用 **Qwen-Image-Edit-2511** 对每个 obj 的白底图做一次 scene-aware edit，prompt 示例：

> "Place this object in a natural well-lit indoor scene with soft shadow and balanced composition."

**产物**：`pipeline/data/pseudo_refs/obj_XXX.png`，每物体一张"参考构图 + 参考光照"的锚点图。

**后续作用**：S4 每轮 VLM 调用都以 **(current_render, pseudo_ref)** 配对输入，VLM 做**差异化点评**而非独立打分。

**降级**：若某 obj 的 pseudo-ref 生成失败 / 质量明显异常，fallback 到 T2I 原白底图作为弱参考，在 MANIFEST 标注。

---

## S3 — Baseline Pre-Render

不重新设计渲染参数，直接用**现有调优过的亮度配置** `scene_template_bright_camera.json` + `scene_v7_full50_loop.json`（50 obj × 8 yaw × seed=42）跑一次 Blender EEVEE。

**关键亮度参数（已调优，作为起点）**：

| 参数 | 值 |
|------|-----|
| `scale_existing_lights` | 0.54 |
| `scale_world_background` | 0.42 |
| `key_light_energy` | 14.4 |
| `target_brightness` | 0.66 |
| `filmic_gamma` | 0.58 |

这一步**不追求完美**，仅作为 S4 loop 的入口图。

---

## S4 — VLM Refinement Loop（项目核心）

**这是整个项目最有价值、也最有风险的环节。**其余 stage 是工程管道，S4 是让"数据集质量自我进化"的闭环。

### 4.1 Loop 总框架

```
R = 1
while R <= MAX_ROUNDS:
    # 视觉评估
    vlm_text = Qwen3.5(current_render, pseudo_ref, critique_prompt)

    if "✅VERDICT: GOOD" in vlm_text:
        break                           # 自然退出

    actions = parse_freeform(vlm_text)

    if actions == last_round_actions:
        break                           # 饱和退出（抖动）

    # 代码修改委托给独立 worker
    worker_run = spawn_codex_exec_worker(
        prompt = render_prompt_template(vlm_text, actions),
        scope  = ["scene_template_*.json", "stage4_scene_render.py"]
    )

    if not main_agent_approve(worker_run):
        retry_once_or_escalate()

    rerender(obj_subset)                # 重渲染采样子集
    R += 1
```

### 4.2 VLM 点评：自由文本，不是 JSON

**Prompt 给 Qwen3.5**：

> 对比参考图和当前渲染图，用自然语言描述渲染图存在的问题（光照、姿态、材质、阴影、构图等）。
> 若整体质量已可用于训练，请在回复末尾写 ✅VERDICT: GOOD。
> 否则说明 3 个以内最重要问题，每个问题给出建议调整方向。

**为什么不用 JSON**：过去强制 JSON schema 时，VLM 输出不稳定（字段缺失、格式错误、分数漂移）。自由文本反而让 VLM 能描述具体视觉现象，主 agent 用关键词映射反而更鲁棒。

### 4.3 动作解析

主 agent 读自由文本，匹配关键词 → 映射到预定义动作空间（复用 `scene-agent-loop`）：

| VLM 描述的问题 | 映射动作 |
|---------------|---------|
| "整体偏暗 / underexposed" | `L_WORLD_EV_UP` |
| "缺少接触阴影 / shadow missing" | `S_CONTACT_SHADOW_UP` |
| "物体在画面中太小" | `O_SCALE_UP_10` |
| "光照太平 / flat lighting" | `L_KEY_UP` / `ENV_STRENGTH_UP` |
| "颜色过艳 / oversaturated" | `M_SATURATION_DOWN` |
| "过曝 / too bright" | `M_VALUE_DOWN` |
| "浮空 / 穿地" | `O_LIFT_SMALL` / `O_LOWER_SMALL` |
| "场景光与物体不协调" | `ENV_ROTATE_30` |

每轮选 1–3 个动作，避免堆叠。

### 4.4 代码修改：交给独立终端 Worker

**核心架构决策**：主 agent 绝不直接 Edit 渲染代码，所有参数/代码修改委托给一个独立终端里的 `codex-exec-worker`（detached PowerShell 跑 `codex exec`）。

**每轮 worker 的上下文**：

```
Repo root: <abs path>

Task:
  本轮 VLM 反馈（自由文本原文）：
    --- VLM_OUTPUT_BEGIN ---
    <Qwen3.5 的输出> 
    --- VLM_OUTPUT_END ---

  主 agent 解析出的动作清单：
    · L_WORLD_EV_UP   —— 整体偏暗
    · S_CONTACT_SHADOW_UP —— 缺接触阴影

Allowed write scope:
  · configs/scene_template_bright_camera.json
  · pipeline/stage4_scene_render.py（仅修改已有光照/材质函数）

Do not touch:
  · pipeline/data/
  · pipeline/stage1_*, stage2_*, stage3_*, stage5_*

Requirements:
  · 每个改动注释写明对应 VLM 问题
  · 不新增依赖、不改函数签名
  · diff ≤ 50 行

Validation:
  · python -c "import json; json.load(open('<template>'))"
  · python -m py_compile pipeline/stage4_scene_render.py

Stop when:
  · validation 全部通过
  · last_message.txt 列出：改了哪些 key / 对应哪个 VLM 问题
```

### 4.5 主 Agent 把关 Checklist

Worker 返回后，主 agent 依次：

1. 读 `last_message.txt`
2. 验 `exit_status.json.exit_code == 0`
3. `git diff` 路径 ⊆ allowed write scope（自动断言，越权直接 reject）
4. 跑 validation 命令，存输出
5. 全过 → 接受改动，重渲染当前采样子集
6. 任一失败 → 重启 worker 一次；再失败 → escalate（停下等人）

**"主 agent 只调度和把关"**：它不写渲染代码，只负责把 VLM 的视觉语言翻译成 worker 能执行的工程任务，再审查 worker 的产出。

### 4.6 退出条件（三选一）

| 条件 | 含义 |
|------|------|
| ✅VERDICT: GOOD | VLM 判定质量可用，自然退出 |
| 连续 2 轮动作重复 | 进入抖动区（改 A 坏 B，改 B 坏 A），强制退出 |
| R ≥ 5（MAX_ROUNDS） | 硬上限，防止无限循环 |

### 4.7 每轮留痕

```
runs/<ts>/vlm_rounds/round_03/
├── vlm_text.txt           # Qwen3.5 原始自由文本
├── actions.json           # 解析后的动作清单
├── worker_run_id          # 指向 worker_runs/ 的对应目录
├── pre_render_sample.png  # 渲染前采样图
└── post_render_sample.png # 渲染后采样图

runs/<ts>/worker_runs/S4_round_03/
├── prompt.txt
├── last_message.txt
├── git_diff.patch
└── validation_output.txt
```

任何一轮都可复盘：VLM 说了什么 → 解析成什么动作 → worker 改了哪几行 → 修改前后渲染差异。

---

## S5/S6 — Dataset Build + Finalize（工程细节，一笔带过）

S4 收敛后，剩下全是**确定性**工程：

- **Rotation8 配对**：front → 其它 7 个视角，instruction 统一写 clockwise
- **Object-disjoint split**：seed=42，35/7/8 物体分到 train/val/test
- **BBox condition 注入**：带框图 + bbox 坐标作为元数据字段
- **Pairs raw 修复**：`source_image / target_image` 必须指向无框 `views/` 原图，带框路径仅作元数据
- **Readiness gate**：跑一次双模型评估确认数据质量门通过
- **MANIFEST**：冻结 git SHA、configs snapshot（sha256）、所有 seed、最终路径

这些步骤之间无决策点，全自动串跑，耗时 < 5 min。

---

## 可复现性保障

- **git SHA**：启动时记录，dirty 仓库拒绝启动
- **Configs snapshot**：所有 scene_template / profile / action_space 拷入 `runs/<ts>/configs_snapshot/`，每份记 sha256
- **Seed 固化**：global / t2i / split seed 全部存 STATE 文件，不从环境读
- **续跑**：`BUILD_DATASET_STATE.json` 记录当前 stage / loop round / worker runs；中断 24h 内可从断点续

---

## 资源与耗时（50 物体）

| Stage | 资源 | 耗时 |
|-------|------|------|
| S1 Assets Prep | 1× A800 | ~50 min |
| S2 Pseudo-Reference | 1× A800 | ~10 min |
| S3 Baseline Render | CPU/GPU | ~20 min |
| **S4 VLM Loop（平均 3 轮）** | **1× A800 + worker** | **~30 min** |
| S5/S6 Dataset + Finalize | CPU | < 5 min |
| **合计** | | **~2 小时** |

---

## 为什么这个 Loop 是关键

过去两周多次踩坑——亮度偏低、方向 prompt 错、pairs 字段错配——**全都是渲染质量问题**，靠人工肉眼发现、手工改配置、手工重跑。

S4 loop 把这个过程自动化：

- **VLM 代替人眼**：Qwen3.5 对照 pseudo-ref 找问题
- **Worker 代替人手**：独立 Codex 实例改配置，比主 agent 直接 edit 更安全、可审计
- **主 agent 只做翻译和把关**：视觉语言 → 工程动作，再审查 diff

这是"让数据集质量自己进化"的机制，也是 v1 最核心的交付物。
