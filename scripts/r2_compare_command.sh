#!/bin/bash
# R2 vs exp5 comparison via feedback_loop_scripts/compare.py
set -euo pipefail

WORKDIR=/gemini/platform/public/aigc/aigc_image/zhanghy56_intern/zhangqisong/data-build
PY=$WORKDIR/.venv/bin/python

R2_EVAL=$WORKDIR/feedback_loop_runs/v2_scaling_r2_quality_gate/round_1/eval_results
OUT_REPORT=$WORKDIR/feedback_loop_runs/v2_scaling_r2_quality_gate/round_1/compare_report.json

cd "$WORKDIR"
"$PY" feedback_loop_scripts/compare.py \
  --current  "$R2_EVAL" \
  --baseline "$WORKDIR/DiffSynth-Studio/output" \
  --output   "$OUT_REPORT" \
  --current-mode  ours_feedback \
  --baseline-mode ours_objinfo

echo "===== R2 vs exp5 compare report ====="
"$PY" -m json.tool "$OUT_REPORT"
