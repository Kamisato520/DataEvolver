---
name: experiment-bridge
description: Implement and run evaluator-VLM experiments after dataset gate pass.
---

# Experiment Bridge

Required:

- `refine-logs/dataset_manifest.json`
- `refine-logs/EVAL_MODEL_EXPERIMENT_PLAN.md`

Run only when:

- `gate_passed == true`
- `next_step == "allow_experiment"`

Use `training_dataset` paths from manifest only.
