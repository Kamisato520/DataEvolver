---
name: idea-discovery
description: Lightweight idea scoping for dataset synthesis and evaluator-VLM training.
---

# Idea Discovery

Produce:

- `refine-logs/FINAL_PROPOSAL.md`
- optional `refine-logs/EXPERIMENT_PLAN.md`

In Workflow 1 middle stage, run dataset evaluation (no synthesis) with:

`python skills/dataset-synthesis-gate/scripts/dataset_readiness_gate.py --idea-path refine-logs/FINAL_PROPOSAL.md --render-fallback none --reports-dir refine-logs`

Then refine proposal using `refine-logs/DATASET_READINESS.md`.
