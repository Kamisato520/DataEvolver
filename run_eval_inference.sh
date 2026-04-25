#!/bin/bash
# Run eval inference for all 3 modes sequentially on a single GPU.
# Usage: bash run_eval_inference.sh [ours|base|fal|all]

set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
PYTHON="/home/wuwenzhuo/Qwen-VL-Series-Finetune/env/bin/python3"
export PYTHONPATH="/aaaidata/zhangqisong/DiffSynth-Studio:${PYTHONPATH:-}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EVAL_SCRIPT="${SCRIPT_DIR}/eval_inference.py"

MODE="${1:-all}"

run_mode() {
    local mode="$1"
    echo ""
    echo "============================================"
    echo "  Running inference: ${mode}"
    echo "============================================"
    $PYTHON "$EVAL_SCRIPT" --mode "$mode" --num_steps 30 --seed 42
    echo "  ${mode} done."
}

case "$MODE" in
    ours)  run_mode ours ;;
    base)  run_mode base ;;
    fal)   run_mode fal ;;
    all)
        run_mode ours
        run_mode base
        run_mode fal
        ;;
    *)
        echo "Usage: $0 [ours|base|fal|all]"
        exit 1
        ;;
esac

echo ""
echo "All requested inference tasks completed."
