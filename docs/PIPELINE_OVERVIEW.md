# ARIS Pipeline Overview (Current)

ARIS is now focused on one core objective:

- Workflow 1: idea discovery + plan refinement, with in-between dataset retrieval/evaluation
- synthetic completion (Blender / T2I / dual)
- post-synthesis filtering and QC
- VLM fine-tuning to realize custom evaluation metrics

Main entry skills:

1. `/dataset-eval-pipeline`
2. `/dataset-synthesis-gate`
3. `/experiment-bridge`

Pipeline structure:

1. Workflow 1: `/idea-discovery -> dataset evaluation (in-between) -> refine`
2. Workflow 2: `/dataset-synthesis-gate`
3. Workflow 3: `/experiment-bridge`
4. Workflow 4: `/analyze-results`

Canonical runtime artifacts:

- `refine-logs/dataset_manifest.json`
- `refine-logs/DATASET_READINESS.md`
- `refine-logs/RENDER_QC_REPORT.md`
- `refine-logs/EVAL_MODEL_SPEC.md`
- `refine-logs/EVAL_MODEL_EXPERIMENT_PLAN.md`
- `refine-logs/EVAL_BENCHMARK_REPORT.md`

For setup and API integration, read:

- `refine-logs/MIGRATED_SETUP_API_GUIDE.md`
