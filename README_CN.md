# ARIS

ARIS 现在是一个**数据集合成 + 评估 VLM 训练**流水线。

当前目标（精简后）只有 4 步：

1. 保留 **Workflow 1（Idea 发现 + 方案精炼）**，并把数据集检索/评估放在 Workflow 1 的中间阶段。
2. 若数据不足，走合成补齐：
   - Blender 路径
   - T2I 路径
   - 双路径（推荐）
3. 合成后做质检与过滤，生成门禁清单。
4. 在现有 VLM（使用Qwen3vl） 基础上微调，让 VLM 具备你定义指标的评估能力。

工作流结构：

- Workflow 1：`/idea-discovery -> （中间数据集评估）-> 精炼方案`
- Workflow 2：`/dataset-synthesis-gate`（正式合成 + 质检 + 门禁）
- Workflow 3：`/experiment-bridge`（评估型 VLM）
- Workflow 4：`/analyze-results`

## 核心 Skills

- `skills/dataset-synthesis-gate/`
- `skills/dataset-eval-pipeline/`
- `skills/experiment-bridge/`
- `skills/analyze-results/`
- `skills/research-pipeline/`（可选总编排）

## 关键产物

- `refine-logs/dataset_manifest.json`
- `refine-logs/DATASET_READINESS.md`
- `refine-logs/RENDER_QC_REPORT.md`
- `refine-logs/EVAL_MODEL_SPEC.md`
- `refine-logs/EVAL_MODEL_EXPERIMENT_PLAN.md`
- `refine-logs/EVAL_BENCHMARK_REPORT.md`

## 安装与 API 接入

请直接看：

- `refine-logs/MIGRATED_SETUP_API_GUIDE.md`

其中包含：

- 本地环境安装
- Blender/T2I 配置
- 双模型评估（Claude + GPT）API wrapper 接入
- 验证方法是否生效的检查清单

## 入口命令

优先用 skill：

- `/dataset-eval-pipeline "你的任务"`

脚本方式：

```bash
python skills/dataset-synthesis-gate/scripts/dataset_readiness_gate.py \
  --idea-path refine-logs/FINAL_PROPOSAL.md \
  --synthesis-mode dual \
  --blender-config-path skills/dataset-synthesis-gate/configs/blender_render.user.template.json \
  --t2i-config-path skills/dataset-synthesis-gate/configs/t2i_generation.user.template.json \
  --reports-dir refine-logs
```
