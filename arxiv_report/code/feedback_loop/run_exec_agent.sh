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

PLAN_OUT="$RUN_ROOT/INTERVENTION_PLAN.md"
EXEC_PROMPT="$RUN_ROOT/exec_prompt.md"
EVENTS="$RUN_ROOT/exec_agent.events.jsonl"
STDERR_LOG="$RUN_ROOT/exec_agent.stderr.log"
COMMAND_LOG="$RUN_ROOT/exec_agent.command.txt"

if [ ! -f "$PLAN_OUT" ]; then
  echo "Missing approved intervention plan: $PLAN_OUT" >&2
  exit 1
fi
if ! command -v codex >/dev/null 2>&1; then
  echo "codex CLI not found in PATH" >&2
  exit 1
fi

cat > "$EXEC_PROMPT" <<EOF
# Feedback Loop Execution Agent

Repo root: $REPO_ROOT
Run root: $RUN_ROOT
Approved plan: $PLAN_OUT

You are the Execution Agent for the local-source/server-run feedback loop.

You may:
- Read repo files and remote/server-visible dataset roots.
- Write only under: $RUN_ROOT
- Create the new feedback dataset under: $RUN_ROOT/dataset
- Run the dataset build commands from the approved INTERVENTION_PLAN.md.
- Write concise logs under: $RUN_ROOT

You must not:
- Modify any repo-tracked code on the server.
- Modify old dataset roots, checkpoints, full20 frozen roots, or pipeline render code.
- Start training, inference, evaluation, tmux, or screen.
- Delete or overwrite existing shared datasets.

If a command in the approved plan would violate these rules, stop and explain the blocker in $RUN_ROOT/exec_blocker.md.

Approved INTERVENTION_PLAN.md:

$(cat "$PLAN_OUT")
EOF

cat > "$COMMAND_LOG" <<EOF
codex exec --cd "$REPO_ROOT" --model "$MODEL" -c 'model_reasoning_effort="xhigh"' --dangerously-bypass-approvals-and-sandbox --json - < "$EXEC_PROMPT" > "$EVENTS" 2> "$STDERR_LOG"
EOF

codex exec \
  --cd "$REPO_ROOT" \
  --model "$MODEL" \
  -c 'model_reasoning_effort="xhigh"' \
  --dangerously-bypass-approvals-and-sandbox \
  --json \
  - < "$EXEC_PROMPT" \
  > "$EVENTS" \
  2> "$STDERR_LOG"

echo "Execution Agent events: $EVENTS"
