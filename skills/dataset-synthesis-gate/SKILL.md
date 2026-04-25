---
name: dataset-synthesis-gate
description: "Run dataset readiness gate with dual-model evaluation, dual-path synthesis (Blender + T2I), preview-loop refinement, QC filtering, and training-manifest output. Use when user asks to assess dataset-fit to idea and auto-supplement missing data."
argument-hint: [idea-or-plan-path]
allowed-tools: Bash(*), Read, Grep, Glob, Edit, Write, Agent, Skill
---

# Dataset Synthesis Gate

Run mandatory data readiness and synthetic fill pipeline for: **$ARGUMENTS**

## Skill-Local Layout

- Scripts: `skills/dataset-synthesis-gate/scripts/`
- Configs: `skills/dataset-synthesis-gate/configs/`

## Inputs

Required/typical files:

1. `refine-logs/FINAL_PROPOSAL.md` (idea text)
2. `refine-logs/EXPERIMENT_PLAN.md` (optional but recommended)
3. candidate real dataset paths (repeatable `--dataset-path`)
4. configs:
   - `skills/dataset-synthesis-gate/configs/dataset_thresholds.default.json`
   - `skills/dataset-synthesis-gate/configs/dataset_llm_eval.default.json`
   - `skills/dataset-synthesis-gate/configs/blender_render.default.json`
   - `skills/dataset-synthesis-gate/configs/t2i_generation.default.json`

When real dataset is missing/insufficient, collect user choices before running:

1. synthesis path: `blender` / `t2i` / `dual`
2. if `blender` or `dual`:
   - object `.blend` folder path
   - scene `.blend` file path
3. if `t2i` or `dual`:
   - provider choice: local qwen image / nano banana api / gpt image api
   - provider config source: command template or API env vars

## Command

```bash
python skills/dataset-synthesis-gate/scripts/dataset_readiness_gate.py \
  --dataset-path <pathA> \
  --dataset-path <pathB> \
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

Blender custom path mode (via render config):

```bash
python skills/dataset-synthesis-gate/scripts/dataset_readiness_gate.py \
  --idea-path refine-logs/FINAL_PROPOSAL.md \
  --synthesis-mode blender \
  --blender-config-path skills/dataset-synthesis-gate/configs/blender_render.user.template.json \
  --reports-dir refine-logs
```

`blender_render.user.template.json` should include at least:

- `object_blend_folder`
- `scene_blend_path`
- `command_images` (or phase-specific command templates)

T2I provider mode (via t2i config):

```bash
python skills/dataset-synthesis-gate/scripts/dataset_readiness_gate.py \
  --idea-path refine-logs/FINAL_PROPOSAL.md \
  --synthesis-mode t2i \
  --t2i-config-path skills/dataset-synthesis-gate/configs/t2i_generation.user.template.json \
  --reports-dir refine-logs
```

## Modes

- `--synthesis-mode blender`: only Blender fallback.
- `--synthesis-mode t2i`: only T2I fallback.
- `--synthesis-mode dual`: Blender + T2I then merge (default recommended).

Workflow 1 in-between evaluation mode (no synthesis):

- use `--render-fallback none` to run retrieval/evaluation only
- use output `refine-logs/DATASET_READINESS.md` to refine idea/proposal before formal gate run

## Preview Loop Behavior

`dataset_readiness_gate.py` calls `dataset_synthesis_loop.py` when missing data exists:

1. preview generation on small subset
2. dual-model evaluation of preview fit
3. if failed, refine params/prompts and retry
4. if passed, run full-scale synthesis
5. merge + QC + filter unsuitable synthetic data

## Outputs

- `refine-logs/dataset_manifest.json`
- `refine-logs/DATASET_READINESS.md`
- `refine-logs/RENDER_QC_REPORT.md`
- `refine-logs/synthesis_manifest.json` (if synthesis triggered)

Downstream training should only use `dataset_manifest.json` `training_dataset` section.

## Gate Rules

- Pass condition: `gate_passed == true` and `next_step == "allow_experiment"`.
- Fail condition: block experiments by default.
- Manual override path: allowed only when policy is `manual-override-on-fail`.

## Quick Validation

```bash
python3 -m py_compile \
  skills/dataset-synthesis-gate/scripts/dataset_readiness_gate.py \
  skills/dataset-synthesis-gate/scripts/dataset_synthesis_loop.py
```
