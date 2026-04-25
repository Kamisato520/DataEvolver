#!/bin/bash
# Full eval pipeline: inference → metrics for rotation editing (68 server).
# Usage:
#   bash run_full_eval.sh              # all modes, epoch 30
#   bash run_full_eval.sh 30           # specify epoch
#   bash run_full_eval.sh 30 base ours # specific modes only
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EPOCH="${1:-30}"
shift || true

if [ $# -gt 0 ]; then
    MODES=("$@")
else
    MODES=(all)
fi

echo "============================================"
echo "  Full Eval Pipeline"
echo "  Epoch: ${EPOCH}"
echo "  Modes: ${MODES[*]}"
echo "  GPU: ${CUDA_VISIBLE_DEVICES}"
echo "============================================"

# Step 1: Inference
echo ""
echo "[Step 1/2] Running inference..."
for mode in "${MODES[@]}"; do
    bash "${SCRIPT_DIR}/run_eval_inference.sh" "$mode" "$EPOCH"
done

# Step 2: Metrics
echo ""
echo "[Step 2/2] Computing metrics..."
# Determine which modes actually produced outputs
WORKDIR="/gemini/platform/public/aigc/aigc_image/zhanghy56_intern/zhangqisong/data-build"
EVAL_BASE="${WORKDIR}/DiffSynth-Studio/output/eval_inference"
AVAIL_MODES=()
for m in base ours fal; do
    if [ -f "${EVAL_BASE}/${m}/eval_meta.json" ]; then
        AVAIL_MODES+=("$m")
    fi
done

if [ ${#AVAIL_MODES[@]} -eq 0 ]; then
    echo "[ERROR] No inference results found. Check inference logs."
    exit 1
fi

echo "Computing metrics for: ${AVAIL_MODES[*]}"
bash "${SCRIPT_DIR}/run_eval_metrics.sh" --eval testset --modes "${AVAIL_MODES[@]}"

echo ""
echo "============================================"
echo "  Full eval pipeline completed!"
echo "  Results: ${WORKDIR}/DiffSynth-Studio/output/eval_metrics/"
echo "============================================"

