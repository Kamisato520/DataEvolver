---
name: run-experiment
description: Deploy and run evaluator-VLM training/evaluation jobs on local or remote GPU servers.
argument-hint: [run-description]
allowed-tools: Bash(*), Read, Grep, Glob, Edit, Write, Agent
---

# Run Experiment

Deploy and run: **$ARGUMENTS**

## Workflow

1. Read environment settings from `AGENTS.md`.
2. Check available GPU.
3. Launch training/eval command (local or remote).
4. Save logs and result artifacts.

## Rules

- Prioritize evaluator-VLM jobs from `EVAL_MODEL_EXPERIMENT_PLAN.md`.
- Keep outputs machine-readable in `results/eval_vlm/`.
