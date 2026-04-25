---
name: analyze-results
description: Analyze evaluator-VLM results and generate metric-focused conclusions for deployment decisions.
argument-hint: [results-path-or-description]
allowed-tools: Bash(*), Read, Grep, Glob, Write, Edit, Agent
---

# Analyze Evaluator-VLM Results

Analyze: **$ARGUMENTS**

## Workflow

1. Load result files from `results/eval_vlm/` (json/csv).
2. Build comparison table across runs/seeds.
3. Report core metrics:
   - `decision_f1_macro`
   - `reject_reason_f1_macro`
   - `score_mae`
   - `ece`
4. Report subgroup metrics:
   - by source: real/blender/t2i
   - by provider: qwen/nano banana/gpt image
5. Output deployment recommendation:
   - accept/review/reject threshold suggestion

## Output

Write/refresh:

- `refine-logs/EVAL_BENCHMARK_REPORT.md`

Include:

- metric table
- key findings
- release recommendation
