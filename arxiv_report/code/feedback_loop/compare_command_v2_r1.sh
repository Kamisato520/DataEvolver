#!/bin/bash
set -euo pipefail

WORKDIR=/gemini/platform/public/aigc/aigc_image/zhanghy56_intern/zhangqisong/data-build
PY=$WORKDIR/.venv/bin/python

cd "$WORKDIR"
"$PY" feedback_loop_scripts/compare.py \
  --current "$WORKDIR/feedback_loop_runs/v2_scaling_r1/round_1/eval_results" \
  --baseline "$WORKDIR/DiffSynth-Studio/output" \
  --output "$WORKDIR/feedback_loop_runs/v2_scaling_r1/round_1/compare_report.json" \
  --current-mode ours_feedback \
  --baseline-mode ours_objinfo

python -m json.tool "$WORKDIR/feedback_loop_runs/v2_scaling_r1/round_1/compare_report.json"
