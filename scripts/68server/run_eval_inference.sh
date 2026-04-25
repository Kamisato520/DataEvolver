#!/bin/bash
# Run eval inference for rotation editing (68 server).
# Usage:
#   bash run_eval_inference.sh            # all available modes
#   bash run_eval_inference.sh ours       # specific mode
#   bash run_eval_inference.sh base       # base only
#   bash run_eval_inference.sh all        # all modes (skip missing LoRA)

set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

WORKDIR="/gemini/platform/public/aigc/aigc_image/zhanghy56_intern/zhangqisong/data-build"

# Python 环境（uv）
source "${WORKDIR}/.venv/bin/activate"
export PYTHONPATH="${WORKDIR}/DiffSynth-Studio:${PYTHONPATH:-}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EVAL_SCRIPT="${SCRIPT_DIR}/eval_inference.py"

# LoRA paths for availability check
OUR_LORA="${WORKDIR}/DiffSynth-Studio/output/rotation8_bbox_rank32/epoch_0030/lora.safetensors"
FAL_LORA="${WORKDIR}/data/fal_lora/qwen-image-edit-2511-multiple-angles-lora.safetensors"

MODE="${1:-all}"
EPOCH="${2:-30}"

run_mode() {
    local mode="$1"
    echo ""
    echo "============================================"
    echo "  Running inference: ${mode}"
    echo "============================================"
    python "$EVAL_SCRIPT" --mode "$mode" --num_steps 30 --seed 42 --epoch "$EPOCH"
    echo "  ${mode} done."
}

case "$MODE" in
    ours)
        if [ ! -f "${OUR_LORA/epoch_0030/epoch_$(printf '%04d' $EPOCH)}" ]; then
            echo "[WARN] Our LoRA epoch ${EPOCH} not found. Is training finished?"
            exit 1
        fi
        run_mode ours
        ;;
    base)
        run_mode base
        ;;
    fal)
        if [ ! -f "$FAL_LORA" ]; then
            echo "[WARN] fal LoRA not found at: $FAL_LORA"
            echo "Please transfer from wwz: /data/wuwenzhuo/Qwen-Image-Edit-2511-Multiple-Angles-LoRA/"
            exit 1
        fi
        run_mode fal
        ;;
    all)
        # Always run base
        run_mode base

        # Run ours if checkpoint exists
        EPOCH_LORA="${OUR_LORA/epoch_0030/epoch_$(printf '%04d' $EPOCH)}"
        if [ -f "$EPOCH_LORA" ]; then
            run_mode ours
        else
            echo "[SKIP] ours: LoRA epoch ${EPOCH} not found at ${EPOCH_LORA}"
        fi

        # Run fal if LoRA exists
        if [ -f "$FAL_LORA" ]; then
            run_mode fal
        else
            echo "[SKIP] fal: LoRA not found at ${FAL_LORA}"
        fi
        ;;
    *)
        echo "Usage: $0 [ours|base|fal|all] [epoch]"
        exit 1
        ;;
esac

echo ""
echo "All requested inference tasks completed."
