#!/bin/bash
# Run eval metrics for rotation editing.
# Usage: bash run_eval_metrics.sh [--eval testset|benchmark|both]
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
PYTHON="/home/wuwenzhuo/Qwen-VL-Series-Finetune/env/bin/python3"
export PYTHONPATH="/aaaidata/zhangqisong/DiffSynth-Studio:${PYTHONPATH:-}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
$PYTHON "${SCRIPT_DIR}/eval_metrics.py" --device cuda "$@"
