---
name: dataset-synthesis-gate
description: "Dataset readiness gate with dual-model evaluation, Blender/T2I synthesis fallback, preview-loop refinement, and training manifest output."
---

# Dataset Synthesis Gate

Use this skill for `dataset -> evaluate -> synthesize missing -> QC/filter -> training manifest`.

Workflow 1 in-between evaluation mode:

- run with `--render-fallback none` for retrieval/evaluation only
- refine proposal from `refine-logs/DATASET_READINESS.md`, then run formal synthesis/QC gate

Skill-local path:

- `skills/dataset-synthesis-gate/scripts/`
- `skills/dataset-synthesis-gate/configs/`

If dataset is insufficient, ask user first:

- synthesis path: `blender` / `t2i` / `dual`
- blender inputs: object folder + scene blend path
- t2i inputs: provider (qwen local / nano banana / gpt image) + config source

## Run

```bash
python skills/dataset-synthesis-gate/scripts/dataset_readiness_gate.py \
  --dataset-path <path> \
  --plan-path refine-logs/EXPERIMENT_PLAN.md \
  --idea-path refine-logs/FINAL_PROPOSAL.md \
  --thresholds-path skills/dataset-synthesis-gate/configs/dataset_thresholds.default.json \
  --llm-eval-config-path skills/dataset-synthesis-gate/configs/dataset_llm_eval.default.json \
  --render-fallback blender \
  --synthesis-mode dual \
  --t2i-config-path skills/dataset-synthesis-gate/configs/t2i_generation.default.json \
  --blender-config-path skills/dataset-synthesis-gate/configs/blender_render.default.json \
  --render-output-mode both \
  --data-merge-mode fill-gap \
  --gate-policy manual-override-on-fail \
  --reports-dir refine-logs
```

## Artifacts

- `refine-logs/dataset_manifest.json` (authoritative dataset entry)
- `refine-logs/DATASET_READINESS.md`
- `refine-logs/RENDER_QC_REPORT.md`
- `refine-logs/synthesis_manifest.json` (when synthesis runs)
