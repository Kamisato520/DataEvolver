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

## 当前主线项目（2026-04-10）

**当前主线**：Scene-Aware Blender Render + 一致性 Rotation 数据集构建 + 训练准备，不是 LoRA 训练项目，也不是旧的小规模 per-pair VLM loop。

- 主线代码目录（远端）：`/aaaidata/zhangqisong/data_build_runs/scene_full20_loop_20260404_code`
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

### 活跃目录与数据集状态

所有路径均在远端目录 `/aaaidata/zhangqisong/data_build_runs/scene_full20_loop_20260404_code/` 下。

| 类型 | 路径 |
|------|------|
| **上游优化根（旧 20 物体）** | `pipeline/data/evolution_scene_v7_full20_rotation4_agent_round0_nostage35_20260404/` |
| **上游优化根（新 30 物体 bootstrap）** | `pipeline/data/evolution_scene_v7_obj021_obj050_yaw000_bootstrap_20260408/` |
| **一致性 rotation8 full50（最新 final）** | `pipeline/data/dataset_scene_v7_full50_rotation8_consistent_yaw000_final_20260410/` |
| **train-ready full50 final** | `pipeline/data/dataset_scene_v7_full50_rotation8_trainready_front2others_final_20260410/` |
| **split full50 final（训练默认入口）** | `pipeline/data/dataset_scene_v7_full50_rotation8_trainready_front2others_splitobj_seed42_final_20260410/` |
| new30 refreshed consistent | `pipeline/data/dataset_scene_v7_obj021_obj050_rotation8_consistent_yaw000_final_20260410/` |
| 旧 geomodal（20260408，非最新 final） | `pipeline/data/dataset_scene_v7_full50_rotation8_geommeta_from_consistent_20260408/` |
| 旧 geomodal train-ready（20260408） | `pipeline/data/dataset_scene_v7_full50_rotation8_geomodal_trainready_front2others_20260408/` |
| 旧 geomodal split（20260408） | `pipeline/data/dataset_scene_v7_full50_rotation8_geomodal_trainready_front2others_splitobj_seed42_20260408/` |
| 旧小规模 loop（已停止） | `pipeline/data/evolution_scene_v7_qwen35_rotation4_agent_round0_obj001_obj003_obj007_obj008_20260402/` |

### 已完成数据集清单（2026-04-10）

| 数据集 | 规模 | 说明 |
|--------|------|------|
| 一致性 rotation8 full50 | 400 RGB + 400 mask | canonical yaw000 base，50×8 角度 |
| train-ready full50 final | 350 训练对 | front → 7 target views（45°/90°/135°/180°/225°/270°/315°） |
| object-disjoint split | 245 / 49 / 56 对 | train 35 / val 7 / test 8 物体，seed=42 |
| new30 refreshed consistent | 240 RGB + 240 mask | 新 30 物体 × 8 角度，best-state 刷新 |
| 旧 geomodal（20260408，非 final） | 160 depth + 160 normal + 140 对 | 含几何增强字段，但基于旧 best-state |
| 训练侧 schema / loader | — | rotation_geomodal_dataset.py（对接 geomodal 根） |

### 当前状态（2026-04-10 完成态）

所有数据集构建任务已完成，服务器处于干净状态：

- **一致性 rotation8 full50 已完成**：50 物体 × 8 角度 = 400 RGB + 400 mask
- **train-ready full50 final 已完成**：front → 7 target views = 350 训练对
- **object-disjoint split 已完成**：train 35 / val 7 / test 8 物体（245 / 49 / 56 对，seed=42）
- **new30 refreshed consistent 已完成**：30 物体 × 8 角度，best-state 刷新
- **旧 geomodal（20260408）可用但非最新 final**：不要与 20260410 standard final 混用
- **训练侧 schema / loader 已完成**：rotation_geomodal_dataset.py + inspect 脚本
- 服务器：历史补生守护进程已清理，旧无用 tmux 已清理，无后台残留

### 当前训练默认入口

默认使用 `full50 final split`：

- `pipeline/data/dataset_scene_v7_full50_rotation8_trainready_front2others_splitobj_seed42_final_20260410`

训练第一版建议：

1. 先做 RGB-only baseline（只读 `instruction` / `source_image` / `target_image`）
2. 验证 train / val loss 和可视化推理正常
3. 跑通后再决定是否引入 mask / geometry / depth / normal

**不要**：
- 把 `20260408` geomodal 和 `20260410` standard final 混成一套实验
- 第一轮同时引入太多辅助模态
- 跳过 object-disjoint split 直接报结果

### Rotation 数据集方法论

**正确方法（已验证）**：

```
yaw000 best state → rotate object → keep scene fixed
```

具体做法：
1. 对每个物体，固定使用 `yaw000` 的最佳 base state
2. 只修改 `control_state["object"]["yaw_deg"]`
3. 保持场景、相机、光影、材质不变
4. 用同一个 canonical base state 分别导出 rotation4 和 rotation8

**已证明错误的方法**：
- ❌ 每个角度 independently 选 best round → 同一物体不同角度来自不同材质/状态，跨角度一致性崩坏
- ❌ 转相机做 multiview → 不符合 rotation 数据集定义（目标是物体旋转，不是相机 orbit）

### 关键脚本

| 脚本 | 作用 |
|------|------|
| `scripts/export_rotation8_from_best_object_state.py` | 一致性 rotation8 导出 |
| `scripts/merge_rotation_consistent_roots.py` | 合并多个 consistent 根 |
| `scripts/build_rotation8_trainready_dataset.py` | 构建 train-ready 数据集 |
| `scripts/build_object_split_for_rotation_dataset.py` | object-disjoint split |
| `scripts/build_rotation8_geommeta_from_consistent.py` | 几何 metadata 构建 |
| `scripts/build_rotation8_geomodal_trainready_dataset.py` | geomodal train-ready 构建 |
| `scripts/run_scene_agent_monitor.py` | 持续 monitor：读 VLM 文本 → 生成 decision → 发起下一轮 |
| `scripts/run_scene_agent_step.py` | 单 pair 单步：载入 state → 渲染 → VLM review（不做 decision） |
| `scripts/inspect_rotation_geomodal_loader.py` | 数据集检查脚本 |
| `pipeline/stage4_scene_render.py` | Blender scene-aware 渲染核心 |
| `pipeline/stage5_5_vlm_review.py` | Qwen3.5 reviewer 入口，freeform + thinking |
| `pipeline/stage5_6_feedback_apply.py` | action apply，含 anti-oscillation 逻辑 |
| `pipeline/rotation_geomodal_dataset.py` | **训练用**：统一 loader（RGB/mask/depth/normal/geometry metadata） |
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

### 当前训练实验状态（2026-04-11）

**Rotation8 RGB-only baseline LoRA 训练（rank=32）正在运行**：
- tmux session: `rotation8-train`（wwz 服务器）
- 训练代码: `/aaaidata/zhangqisong/DiffSynth-Studio/train_clockwise.py` + `train_clockwise.sh`
- 输出目录: `/aaaidata/zhangqisong/DiffSynth-Studio/output/rotation8_rgb_baseline_rank32/`
- TensorBoard: `.../rotation8_rgb_baseline_rank32/tensorboard/`
- 参数: lr=1e-4, epochs=30, LoRA rank=32, dataset_repeat=10, 3×A800 DDP, ~7.8s/step, ~65GB/卡
- Python 环境: `/home/wuwenzhuo/Qwen-VL-Series-Finetune/env/bin/python3`（PyTorch 2.8.0+cu128）
- PYTHONPATH: `/aaaidata/zhangqisong/DiffSynth-Studio`
- accelerate 配置: `/aaaidata/zhangqisong/DiffSynth-Studio/accelerate_config_3gpu.yaml`（MULTI_GPU bf16）
- 每 epoch ~817 步，预计每 epoch ~1.8h，30 epoch ~54h

**历史实验（已停止）**：
- rank=64 实验：`output/rotation8_rgb_baseline/`，跑了 1 完整 epoch + epoch 2 的 68%，epoch_0001 checkpoint 已保存
- rank=64 过大（900MB, 4.7 亿参数），对 245 训练对容易过拟合，已改为 rank=32

**参照 LoRA 对比**：
| | Ours (rank=32) | fal 社区 (rank=16) | 旧实验 (rank=64) |
|---|---|---|---|
| Rank | 32 | 16 | 64 |
| 预估参数量 | ~2.4 亿 | 1.5 亿 | 4.7 亿 |
| 路径 | `.../rotation8_rgb_baseline_rank32/` | `/data/wuwenzhuo/Qwen-Image-Edit-2511-Multiple-Angles-LoRA/qwen-image-edit-2511-multiple-angles-lora.safetensors` | `.../rotation8_rgb_baseline/` |

**评测推理脚本（已就绪，待训练完成后运行）**：
- 推理代码: `/aaaidata/zhangqisong/DiffSynth-Studio/eval_inference.py`
- 启动脚本: `/aaaidata/zhangqisong/DiffSynth-Studio/run_eval_inference.sh`
- 三组对比: base 模型 / fal 社区 LoRA / 我们训练的 LoRA
- 输出目录: `/aaaidata/zhangqisong/DiffSynth-Studio/output/eval_inference/{base,fal_lora,ours}/`

**训练代码关键修改**：
- `train_clockwise.py` 添加了 JSONL 字段重映射：`source_image→edit_image`, `target_image→image`, `target_rotation_deg→angle_bucket`（使用 ANGLE_TO_BUCKET 映射表）
- `train_clockwise.sh` 更新为 wwz 3×A800 配置，使用兼容 CUDA 12.8 的 Python 环境

**注意**：DiffSynth-Studio 的 uv venv（`.venv`）中 PyTorch 编译的 CUDA 13.0 与服务器 driver 12.2 不兼容，训练使用的是 `/home/wuwenzhuo/Qwen-VL-Series-Finetune/env/`。

### 继续任务的入口

**当前主线（训练实验 + 技术报告）**：
1. ~~基于 `full50 final split` 启动首轮 RGB-only baseline 训练~~ ✓ 已启动
2. 验证 train / val loss 和可视化推理正常
3. 跑通后决定是否引入 mask / geometry / depth / normal
4. 开始技术报告正文写作（agentic synthetic data construction 路线，rotation dataset 为首个验证实例）

**训练默认数据集**（远端 `scene_full20_loop_20260404_code` 下的 `pipeline/data/`）：
- split（默认入口）: `dataset_scene_v7_full50_rotation8_trainready_front2others_splitobj_seed42_final_20260410/`
- train-ready（全量）: `dataset_scene_v7_full50_rotation8_trainready_front2others_final_20260410/`

**Loader 使用**：

标准 `20260410 final` 数据集直接按 JSONL 读取：
```python
import json
from pathlib import Path
from PIL import Image

root = Path("<split-root>")
rows = []
with (root / "pairs" / "train_pairs.jsonl").open("r", encoding="utf-8") as f:
    for line in f:
        rows.append(json.loads(line))

row = rows[0]
source = Image.open(root / row["source_image"]).convert("RGB")
target = Image.open(root / row["target_image"]).convert("RGB")
instruction = row["instruction"]
```

如需 geomodal loader（对接旧 geomodal 根）：
```python
from pipeline.rotation_geomodal_dataset import RotationGeomodalDataset
ds = RotationGeomodalDataset(root="<geomodal-root>/train")
```

**训练侧说明文档**：
- `docs/full50_final_training_readme_for_claude_20260410.md`（当前最该看的训练启动文档）
- `docs/rotation8_geomodal_training_schema_20260407.md`（geomodal 版 schema 参考）

**VLM 预设规则入口**：
- `deliverables/vlm_preset_rule_bundle_20260410/`（含 README、configs、pipeline 脚本）

**历史 Loop 的入口**（如需重启 VLM loop）：
```
/scene-agent-loop continue
/scene-agent-loop status
```

### 已知坑

1. `rotation4_freeform...` 是旧冻结目录，不是 active
2. 不能只看 score / agg.json，必须读 trace.json 自由文本
3. 不能主动退出 loop（reviewer 未 keep 则继续）
4. GPU OOM ≠ 渲染失败——无新 agg.json 时先查 reviewer OOM
5. 不能让脚本自动按分数选 action，AI 必须亲自读文本后决策
6. stage3.5（地板消除）已废弃，不要再启用
7. **不要把旧的 `best-of-each-pair rotation4` 当最终数据集**（跨角度不一致）
8. **不要把 rotation8 理解成相机 multiview**（物体旋转，不是相机 orbit）
9. **不要回退到"每个角度各自选最优"策略**
10. 当前 active 的 scene 根是 `scene_full20_loop_20260404_code`，不是旧的 `20260402` 小规模目录
11. `force_reference_material: false` 可能被阈值逻辑偷偷覆盖，需显式设置
12. **不要把 `20260408` geomodal 和 `20260410` standard final 混成一套实验**
13. **不要跳过 object-disjoint split 直接报结果**
14. **第一轮训练不要同时引入太多辅助模态**，先跑通 RGB baseline
15. 不要直接在 dataset 根里写缓存、中间文件、训练输出

---

## 远程服务器（wwz）

- SSH: `wwz`（密钥免密）
- GPU: 3×A800 80GB
- **Python 环境**：直接用全路径 `/home/wuwenzhuo/Qwen-VL-Series-Finetune/env/bin/python3`（不用 conda activate，conda 不可用）
- 代码目录: `/aaaidata/zhangqisong/data_build_runs/scene_full20_loop_20260404_code`
- Blender binary: `/home/wuwenzhuo/blender-4.24/blender`
- 场景文件: `/home/wuwenzhuo/blender/data/sence/4.blend`
- Qwen3.5-35B 模型: `/data/wuwenzhuo/Qwen3.5-35B-A3B`
- 模型下载目录: `/huggingface/model_hub`（下载前 `export HF_ENDPOINT=https://hf-mirror.com`）
- **必须用 tmux**（screen 不可用），命名：`tmux new -s claudecode-research-X`
- 长任务确认正常运行后停止监测，训练完成用户会唤醒
- GPU 使用前先检查占用，有空闲卡优先用空闲卡，全占用则等待
- 服务器端的DiffSynth的路径是：/aaaidata/zhangqisong/DiffSynth-Studio，使用的是uv 环境，激活：source /aaaidata/zhangqisong/DiffSynth-Studio/.venv/bin/activate
- qwen image edit 2511 模型的路径是：/data/wuwenzhuo/Qwen-Image-Edit-2511，信息如下：

  ```
  (base) wuwenzhuo@goldenapple:/aaaidata/zhangqisong$ ls -R /data/wuwenzhuo/Qwen-Image-Edit-2511
  /data/wuwenzhuo/Qwen-Image-Edit-2511:
  model_index.json  processor  README.md  scheduler  text_encoder  tokenizer  transformer  vae
  
  /data/wuwenzhuo/Qwen-Image-Edit-2511/processor:
  added_tokens.json    merges.txt                special_tokens_map.json  tokenizer.json                  vocab.json
  chat_template.jinja  preprocessor_config.json  tokenizer_config.json    video_preprocessor_config.json
  
  /data/wuwenzhuo/Qwen-Image-Edit-2511/scheduler:
  scheduler_config.json
  
  /data/wuwenzhuo/Qwen-Image-Edit-2511/text_encoder:
  config.json             model-00001-of-00004.safetensors  model-00003-of-00004.safetensors  model.safetensors.index.json
  generation_config.json  model-00002-of-00004.safetensors  model-00004-of-00004.safetensors
  
  /data/wuwenzhuo/Qwen-Image-Edit-2511/tokenizer:
  added_tokens.json  chat_template.jinja  merges.txt  special_tokens_map.json  tokenizer_config.json  vocab.json
  
  /data/wuwenzhuo/Qwen-Image-Edit-2511/transformer:
  config.json                                         diffusion_pytorch_model-00004-of-00005.safetensors
  diffusion_pytorch_model-00001-of-00005.safetensors  diffusion_pytorch_model-00005-of-00005.safetensors
  diffusion_pytorch_model-00002-of-00005.safetensors  diffusion_pytorch_model.safetensors.index.json
  diffusion_pytorch_model-00003-of-00005.safetensors
  
  /data/wuwenzhuo/Qwen-Image-Edit-2511/vae:
  config.json  diffusion_pytorch_model.safetensors
  ```

- 参照使用的标准fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA路径（仅有lora）：/data/wuwenzhuo/Qwen-Image-Edit-2511-Multiple-Angles-LoRA/qwen-image-edit-2511-multiple-angles-lora.safetensors

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

> 周报详见 `notion-weekly/week1_0325-0403.md`、`notion-weekly/week2_0404-0407.md`
> 训练启动文档详见 `docs/full50_final_training_readme_for_claude_20260410.md`
