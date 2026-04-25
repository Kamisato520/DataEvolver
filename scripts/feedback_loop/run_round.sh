#!/bin/bash
set -euo pipefail

if [ $# -lt 2 ]; then
  echo "Usage: $0 <experiment_id> <round_number> [--execute]" >&2
  exit 2
fi

EXPERIMENT_ID="$1"
ROUND="$2"
MODE="${3:-}"
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
WORKDIR="${WORKDIR:-/gemini/platform/public/aigc/aigc_image/zhanghy56_intern/zhangqisong/data-build}"
RUN_ROOT="${WORKDIR}/feedback_loop_runs/${EXPERIMENT_ID}/round_${ROUND}"
BASELINE_EVAL_DIR="${BASELINE_EVAL_DIR:-${WORKDIR}/feedback_loop_runs/${EXPERIMENT_ID}/round_0/eval_results}"

mkdir -p "$RUN_ROOT"

if [ "$MODE" != "--execute" ]; then
  echo "=== Building plan_prompt.md ==="
  python "$REPO_ROOT/scripts/feedback_loop/build_plan_prompt.py" \
    --experiment-id "$EXPERIMENT_ID" \
    --round "$ROUND" \
    --baseline-eval-dir "$BASELINE_EVAL_DIR" \
    --output "$RUN_ROOT/plan_prompt.md"

  echo "=== Running Plan Agent ==="
  bash "$REPO_ROOT/scripts/feedback_loop/run_plan_agent.sh" \
    --repo-root "$REPO_ROOT" \
    --run-root "$RUN_ROOT" \
    --model "${PLAN_AGENT_MODEL:-gpt-5.4}"

  python "$REPO_ROOT/scripts/feedback_loop/update_state.py" \
    --experiment-id "$EXPERIMENT_ID" \
    --round "$ROUND" \
    --run-root "$RUN_ROOT" \
    --step "plan_complete"

  echo "Plan Agent complete: $RUN_ROOT/INTERVENTION_PLAN.md"
  echo "Review it, then run: bash $REPO_ROOT/scripts/feedback_loop/run_round.sh $EXPERIMENT_ID $ROUND --execute"
  exit 0
fi

echo "=== Running Execution Agent ==="
bash "$REPO_ROOT/scripts/feedback_loop/run_exec_agent.sh" \
  --repo-root "$REPO_ROOT" \
  --run-root "$RUN_ROOT" \
  --model "${EXEC_AGENT_MODEL:-gpt-5.4}"

echo "=== Validating dataset ==="
VALIDATE_ARGS=(
  "$RUN_ROOT/dataset"
  --output "$RUN_ROOT/validation_report.json"
)
if [ -f "${PINNED_SPLIT_PATH:-$RUN_ROOT/pinned_split.json}" ]; then
  VALIDATE_ARGS+=(--pinned-split-path "${PINNED_SPLIT_PATH:-$RUN_ROOT/pinned_split.json}")
fi
if [ -f "$RUN_ROOT/dataset/augmented_manifest.json" ]; then
  VALIDATE_ARGS+=(--expected-manifest "$RUN_ROOT/dataset/augmented_manifest.json")
fi
if ! python "$REPO_ROOT/scripts/feedback_loop/validate_dataset.py" "${VALIDATE_ARGS[@]}"; then
  python "$REPO_ROOT/scripts/feedback_loop/update_state.py" \
    --experiment-id "$EXPERIMENT_ID" \
    --round "$ROUND" \
    --run-root "$RUN_ROOT" \
    --step "validation" \
    --status "failed" \
    --validation-report "$RUN_ROOT/validation_report.json" \
    --message "Dataset validation failed, see validation_report.json"
  echo "VALIDATION FAILED - see $RUN_ROOT/validation_report.json"
  exit 1
fi

echo "=== Generating training command ==="
bash "$REPO_ROOT/scripts/feedback_loop/train_feedback_round.sh" \
  --dataset-root "$RUN_ROOT/dataset" \
  --output-dir "$RUN_ROOT/output" \
  --generate-only \
  > "$RUN_ROOT/train_command.sh"
chmod +x "$RUN_ROOT/train_command.sh"

echo "=== Generating eval command ==="
bash "$REPO_ROOT/scripts/feedback_loop/eval_feedback_round.sh" \
  --checkpoint "$RUN_ROOT/output/epoch_0029/lora.safetensors" \
  --dataset-root "$RUN_ROOT/dataset" \
  --output-dir "$RUN_ROOT/eval_results" \
  --repo-root "$REPO_ROOT" \
  --generate-only \
  > "$RUN_ROOT/eval_command.sh"
chmod +x "$RUN_ROOT/eval_command.sh"

echo "=== Generating compare command ==="
cat > "$RUN_ROOT/compare_command.sh" <<EOF
#!/bin/bash
set -euo pipefail
python "$REPO_ROOT/scripts/feedback_loop/compare.py" \\
  --current "$RUN_ROOT/eval_results" \\
  --baseline "$BASELINE_EVAL_DIR" \\
  --current-mode ours \\
  --baseline-mode ours_objinfo \\
  --output "$RUN_ROOT/compare_report.json"
python "$REPO_ROOT/scripts/feedback_loop/analyze_feedback_for_dataset.py" \\
  --compare-report "$RUN_ROOT/compare_report.json" \\
  --output "$RUN_ROOT/dataset_feedback_plan.json"
python "$REPO_ROOT/scripts/feedback_loop/update_state.py" \\
  --experiment-id "$EXPERIMENT_ID" \\
  --round "$ROUND" \\
  --run-root "$RUN_ROOT" \\
  --step "compare_completed" \\
  --compare-report "$RUN_ROOT/compare_report.json" \\
  --dataset-feedback-plan "$RUN_ROOT/dataset_feedback_plan.json" \\
  --augmented-manifest "$RUN_ROOT/dataset/augmented_manifest.json"
EOF
chmod +x "$RUN_ROOT/compare_command.sh"

python "$REPO_ROOT/scripts/feedback_loop/update_state.py" \
  --experiment-id "$EXPERIMENT_ID" \
  --round "$ROUND" \
  --run-root "$RUN_ROOT" \
  --step "setup_complete" \
  --validation-report "$RUN_ROOT/validation_report.json" \
  --augmented-manifest "$RUN_ROOT/dataset/augmented_manifest.json"

echo "=== Round $ROUND setup complete ==="
echo "Run training:"
echo "  tmux new-session -d -s ${EXPERIMENT_ID}_r${ROUND}_train 'bash $RUN_ROOT/train_command.sh 2>&1 | tee $RUN_ROOT/train.log'"
echo "After training:"
echo "  bash $RUN_ROOT/eval_command.sh 2>&1 | tee $RUN_ROOT/eval.log"
echo "After eval:"
echo "  bash $RUN_ROOT/compare_command.sh"
