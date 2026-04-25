---
name: dataset-eval-pipeline
description: "Top-level pipeline for dataset retrieval/evaluation, dual-path synthesis (Blender + T2I), QC filtering, and evaluator-VLM development for custom metrics."
argument-hint: [task-goal-or-proposal-path]
allowed-tools: Bash(*), Read, Write, Edit, Grep, Glob, Agent, Skill, mcp__codex__codex, mcp__codex__codex-reply
---

# Dataset Eval Pipeline: Data -> Synthesis -> Metric -> Evaluator

Run the end-to-end dataset+evaluator workflow for: **$ARGUMENTS**

## Goal

Target deliverables:

1. a filtered training-ready dataset package (real + synthetic)
2. a metric definition suite aligned to your task acceptance criteria
3. an evaluator VLM that can output metric-aligned decisions/scores with validation report

## Composition

```
Workflow 1 (idea+refine): /idea-discovery -> dataset evaluation (in-between) -> refine
Workflow 2 (data gate): /dataset-synthesis-gate
Workflow 3 (metric+model): metric/evaluator plan -> /experiment-bridge
Workflow 4 (analysis): /analyze-results
```

## Constants

- **DATASET_GATE = true**
- **SYNTHESIS_MODE = dual**
- **RENDER_OUTPUT_MODE = both**
- **DATA_MERGE_MODE = fill-gap**
- **GATE_POLICY = manual-override-on-fail**
- **METRIC_MODEL_TARGET = vlm-evaluator**

## Workflow

### Stage 0: Input Triage

- If `$ARGUMENTS` is a path to proposal/plan, read it directly.
- If proposal is missing, run:

```
/idea-discovery "$ARGUMENTS"
```

Expected context artifacts:

- `refine-logs/FINAL_PROPOSAL.md`
- `refine-logs/EXPERIMENT_PLAN.md` (optional here; can be evaluator-focused later)

### Workflow 1: Idea Discovery + Plan Refinement

#### Stage 1.1: Initial Idea/Requirement Draft

Use `refine-logs/FINAL_PROPOSAL.md` as the initial draft after Stage 0.

#### Stage 1.2: In-Between Dataset Evaluation (No Synthesis)

Run dataset retrieval/evaluation as a middle stage in Workflow 1:

```bash
python skills/dataset-synthesis-gate/scripts/dataset_readiness_gate.py \
  --idea-path refine-logs/FINAL_PROPOSAL.md \
  --dataset-path <candidate_dataset_path> \
  --render-fallback none \
  --reports-dir refine-logs
```

Use `refine-logs/DATASET_READINESS.md` and dual-model output to extract:

- missing data dimensions
- unsuitable patterns
- required synthesis strategy

#### Stage 1.3: Plan Refinement

Refine `refine-logs/FINAL_PROPOSAL.md` by adding:

- explicit missing-data checklist
- synthesis path choice (`blender` / `t2i` / `dual`)
- source/provider requirements for chosen path

### Workflow 2: Dataset Gate + Synthesis/QC

Run unified gate skill:

```
/dataset-synthesis-gate "refine-logs/FINAL_PROPOSAL.md"
```

If no suitable real dataset is found, require a user decision before synthesis:

1. choose path: `blender` / `t2i` / `dual`
2. for `blender`:
   - collect object `.blend` folder path
   - collect scene `.blend` path
3. for `t2i`:
   - collect provider: local qwen image / nano banana api / gpt image api
   - collect provider config source (`--t2i-config-path`)
4. execute gate with corresponding configs

Required pass condition before any model work:

- `refine-logs/dataset_manifest.json` exists
- `gate_passed == true`
- `next_step == "allow_experiment"`
- `dual_llm_eval.pre_render.consensus.pass == true`
- `dual_llm_eval.post_render_or_merge.consensus.pass == true`

### Workflow 3: Metric Suite + Evaluator VLM

#### Stage 3.1: Define Metric Suite

Write `refine-logs/METRIC_SPEC.md` with:

- task objective and failure types
- metric groups:
  - fidelity (quality/realism)
  - coverage (missing-case coverage)
  - controllability/consistency (prompt/condition alignment)
  - utility (downstream task gain)
- each metric's:
  - formula or scoring rule
  - input data source
  - threshold/acceptance region
  - risk of metric gaming

#### Stage 3.2: Define Evaluator VLM

Write:

- `refine-logs/EVAL_MODEL_SPEC.md`
- `refine-logs/EVAL_MODEL_EXPERIMENT_PLAN.md`

You can start from template:

- `refine-logs/EVAL_MODEL_EXPERIMENT_PLAN.md` (this repository template version)
- `skills/dataset-synthesis-gate/configs/blender_render.user.template.json`
- `skills/dataset-synthesis-gate/configs/t2i_generation.user.template.json`

Scope the "training" task to evaluator construction only:

- model role: use existing VLM backbone to output metric-aligned structured evaluation
- labels: dual-model judgments + QC outcomes + optional human spot checks
- split protocol: train/val/test from `dataset_manifest.json` paths with real/blender/t2i stratification
- outputs: per-sample structured metric JSON + reject reason + calibrated threshold
- training strategy: VLM SFT first, optional preference optimization, then threshold calibration

#### Stage 3.3: Implement + Run Evaluator VLM Experiments

Use experiment bridge for implementation/deployment:

```
/experiment-bridge "refine-logs/EVAL_MODEL_EXPERIMENT_PLAN.md"
```

Requirements:

- consume dataset only from `refine-logs/dataset_manifest.json` `training_dataset`
- keep runs small/iterative first, then scale
- prioritize metric-realization quality on VLM (`decision/reason F1`, `score MAE`, `ECE`)

### Workflow 4: Analyze and Package

Run:

```
/analyze-results "results for evaluator runs"
```

Write final package report:

- `refine-logs/EVAL_BENCHMARK_REPORT.md`

Must include:

- final dataset snapshot and counts
- final metric suite and thresholds
- evaluator model performance summary
- recommended production filter policy (accept/reject band)

## Key Rules

- Prefer auditable, reproducible metric definitions over narrative claims.
- Dataset gate artifacts are mandatory input contracts for evaluator training.
