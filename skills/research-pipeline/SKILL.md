---
name: research-pipeline
description: "Current ARIS top-level pipeline: Workflow 1 (idea discovery + plan refinement with in-between dataset evaluation) -> synthesis/QC gate -> evaluator VLM experiments -> analysis."
argument-hint: [task-direction]
allowed-tools: Bash(*), Read, Write, Edit, Grep, Glob, Agent, Skill
---

# Current ARIS Pipeline

Run end-to-end for: **$ARGUMENTS**

## Composition

```
Workflow 1 (idea+refine): /idea-discovery -> dataset evaluation (in-between) -> refine
Workflow 2 (data gate): /dataset-synthesis-gate
Workflow 3 (model): /experiment-bridge
Workflow 4 (analysis): /analyze-results
```

## Workflow 1: Idea Discovery + Plan Refinement

### Stage 1.1: Initial Idea Scoping

```
/idea-discovery "$ARGUMENTS"
```

Expected artifact:

- `refine-logs/FINAL_PROPOSAL.md`

### Stage 1.2: In-Between Dataset Evaluation (No Synthesis)

Run dataset search/evaluation as the middle stage inside Workflow 1:

```bash
python skills/dataset-synthesis-gate/scripts/dataset_readiness_gate.py \
  --idea-path refine-logs/FINAL_PROPOSAL.md \
  --dataset-path <candidate_dataset_path> \
  --render-fallback none \
  --reports-dir refine-logs
```

Expected artifact:

- `refine-logs/DATASET_READINESS.md`

### Stage 1.3: Refine Proposal with Dataset Findings

Refine `refine-logs/FINAL_PROPOSAL.md` using:

- missing-data list from dual-model dataset evaluation
- synthesis path decision (`blender` / `t2i` / `dual`)
- user-provided source paths or provider configs

## Workflow 2: Dataset Gate + Synthesis/QC

```
/dataset-synthesis-gate "refine-logs/FINAL_PROPOSAL.md"
```

Required outputs:

- `refine-logs/dataset_manifest.json`
- `refine-logs/DATASET_READINESS.md`
- `refine-logs/RENDER_QC_REPORT.md`

Gate must pass before experiment deployment:

- `gate_passed == true`
- `next_step == "allow_experiment"`

## Workflow 3: Evaluator VLM Training Bridge

```
/experiment-bridge "refine-logs/EVAL_MODEL_EXPERIMENT_PLAN.md"
```

## Workflow 4: Analyze

```
/analyze-results "results/eval_vlm"
```

## Rules

- Dataset manifest is the only canonical data input contract.
- Keep training objective focused on metric-realization in VLM.
- Final package target is: synthesized/filtered dataset + metric suite + evaluator VLM report.
