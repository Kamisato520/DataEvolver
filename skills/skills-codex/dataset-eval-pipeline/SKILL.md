---
name: dataset-eval-pipeline
description: "Top-level pipeline for dataset retrieval/evaluation, synthesis fallback, metric definition, and evaluator-VLM construction."
---

# Dataset Eval Pipeline

Use this for: `Workflow 1 (idea+refine with in-between dataset eval) -> synthesis fallback -> metric suite -> evaluator VLM`.

If real dataset is insufficient, ask user to choose:

- `blender` path (needs object folder + scene blend path)
- `t2i` path (needs provider choice and config)
- `dual` path (both)

Use skill-local configs:

- `skills/dataset-synthesis-gate/configs/blender_render.user.template.json`
- `skills/dataset-synthesis-gate/configs/t2i_generation.user.template.json`

## Composition

```
Workflow 1: /idea-discovery -> dataset evaluation (in-between) -> refine
Workflow 2: /dataset-synthesis-gate
Workflow 3: /experiment-bridge
Workflow 4: /analyze-results
```

## Run

1. Dataset gate:

```
/dataset-synthesis-gate "refine-logs/FINAL_PROPOSAL.md"
```

2. Build evaluator-VLM artifacts:

- `refine-logs/METRIC_SPEC.md`
- `refine-logs/EVAL_MODEL_SPEC.md`
- `refine-logs/EVAL_MODEL_EXPERIMENT_PLAN.md`

3. Deploy evaluator experiments:

```
/experiment-bridge "refine-logs/EVAL_MODEL_EXPERIMENT_PLAN.md"
```

4. Analyze and finalize:

```
/analyze-results "results directory or description"
```

Output final report: `refine-logs/EVAL_BENCHMARK_REPORT.md`.
