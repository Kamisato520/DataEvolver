#!/bin/bash
set -euo pipefail

REPO_ROOT=""
RUN_ROOT=""
MODEL="gpt-5.4"

while [ $# -gt 0 ]; do
  case "$1" in
    --repo-root) REPO_ROOT="$2"; shift 2 ;;
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [ -z "$REPO_ROOT" ] || [ -z "$RUN_ROOT" ]; then
  echo "Usage: $0 --repo-root <repo> --run-root <run> [--model gpt-5.4]" >&2
  exit 2
fi

PLAN_PROMPT="$RUN_ROOT/plan_prompt.md"
PLAN_OUT="$RUN_ROOT/INTERVENTION_PLAN.md"
EVENTS="$RUN_ROOT/plan_agent.events.jsonl"
STDERR_LOG="$RUN_ROOT/plan_agent.stderr.log"
COMMAND_LOG="$RUN_ROOT/plan_agent.command.txt"

if [ ! -f "$PLAN_PROMPT" ]; then
  echo "Missing plan prompt: $PLAN_PROMPT" >&2
  exit 1
fi
if ! command -v codex >/dev/null 2>&1; then
  echo "codex CLI not found in PATH" >&2
  exit 1
fi

cat > "$COMMAND_LOG" <<EOF
codex exec --cd "$REPO_ROOT" --model "$MODEL" -c 'model_reasoning_effort="xhigh"' -c 'shell_environment_policy.inherit="all"' --dangerously-bypass-approvals-and-sandbox --json -o "$PLAN_OUT" - < "$PLAN_PROMPT" > "$EVENTS" 2> "$STDERR_LOG"
EOF

codex exec \
  --cd "$REPO_ROOT" \
  --model "$MODEL" \
  -c 'model_reasoning_effort="xhigh"' \
  -c 'shell_environment_policy.inherit="all"' \
  --dangerously-bypass-approvals-and-sandbox \
  --json \
  -o "$PLAN_OUT" \
  - < "$PLAN_PROMPT" \
  > "$EVENTS" \
  2> "$STDERR_LOG"

if [ ! -s "$PLAN_OUT" ]; then
  echo "Plan Agent did not produce $PLAN_OUT" >&2
  exit 1
fi

echo "Plan written: $PLAN_OUT"
