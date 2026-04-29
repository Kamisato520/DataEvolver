---
name: idea-discovery
description: "Lightweight idea scoping for the current ARIS pipeline. Converts a task direction into FINAL_PROPOSAL.md and dataset requirements for synthesis/evaluation."
argument-hint: [task-direction]
allowed-tools: Bash(*), Read, Write, Edit, Grep, Glob, Agent, Skill
---

# Idea Discovery (Current Scope)

Use this skill to produce minimal, execution-ready idea artifacts for: **$ARGUMENTS**

## Output Targets

- `refine-logs/FINAL_PROPOSAL.md`
- `refine-logs/EXPERIMENT_PLAN.md` (optional high-level, evaluator-oriented)

## Workflow

1. Clarify task objective, constraints, and acceptance criteria.
2. Define required data coverage dimensions:
   - object/scene variation
   - lighting/viewpoint/quality constraints
   - edge/rare cases
3. Write concise proposal in `FINAL_PROPOSAL.md` including:
   - problem statement
   - target metric behavior
   - missing-data requirements list
4. Run in-between dataset evaluation hook (Workflow 1 middle stage):
   - `python skills/dataset-synthesis-gate/scripts/dataset_readiness_gate.py --idea-path refine-logs/FINAL_PROPOSAL.md --render-fallback none --reports-dir refine-logs`
5. Refine `FINAL_PROPOSAL.md` using evaluation output (`DATASET_READINESS.md`), then hand off to `/dataset-synthesis-gate` for formal synthesis/QC gate.

## Rules

- Keep scope narrow: dataset + evaluation-VLM capability.
- Make output directly consumable by `/dataset-synthesis-gate` (missing-data list must be explicit).
