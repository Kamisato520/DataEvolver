---
name: experiment-bridge
description: "Implement and run evaluator-VLM experiments with mandatory dataset gate checks."
argument-hint: [experiment-plan-path]
allowed-tools: Bash(*), Read, Write, Edit, Grep, Glob, Agent, Skill
---

# Experiment Bridge (Evaluator VLM)

Implement and deploy experiments from: **$ARGUMENTS**

## Required Inputs

- `refine-logs/dataset_manifest.json` (mandatory)
- `refine-logs/EVAL_MODEL_EXPERIMENT_PLAN.md` (mandatory)
- `refine-logs/EVAL_MODEL_SPEC.md` (recommended)
- `refine-logs/METRIC_SPEC.md` (recommended)

## Phase 0: Gate Check (Mandatory)

Read `refine-logs/dataset_manifest.json` and require:

- `gate_passed == true`
- `next_step == "allow_experiment"`
- `dual_llm_eval.pre_render.consensus.pass == true`
- `dual_llm_eval.post_render_or_merge.consensus.pass == true`

If gate fails: block deployment.

## Phase 1: Implement Evaluator VLM Training

From plan/spec, implement:

- dataset loader from `training_dataset` only
- VLM fine-tuning pipeline (LoRA/QLoRA/full FT as specified)
- structured JSON output parser for evaluator predictions
- metric computation:
  - `decision_f1_macro`
  - `reject_reason_f1_macro`
  - `score_mae`
  - `ece`

## Phase 2: Run and Track

- start with small sanity split
- run full experiments (multi-seed if requested)
- save parseable results (`json/csv`) under `results/eval_vlm/`
- save checkpoints under `checkpoints/eval_vlm/`

## Phase 3: Handoff

Write:

- `refine-logs/EXPERIMENT_RESULTS.md`
- update `refine-logs/EVAL_BENCHMARK_REPORT.md`

## Rules

- Never use raw dataset paths outside `dataset_manifest.json`.
- Keep objective aligned to evaluation-metric realization and decision calibration.
