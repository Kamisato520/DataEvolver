#!/bin/bash
# Run SpatialEdit-Bench evaluation using eval_image_metrics.py (timm DINO + open_clip + torchmetrics FID)
# Uses GPUs 0-5, runs 3 modes in parallel (base on GPU0, ours on GPU2, fal on GPU4)

set -e

export HF_ENDPOINT="https://hf-mirror.com"
export HF_HUB_OFFLINE=0

WORKDIR="/gemini/platform/public/aigc/aigc_image/zhanghy56_intern/zhangqisong/data-build"
DSDIR="${WORKDIR}/DiffSynth-Studio"
METRICS_SCRIPT="${DSDIR}/metrics/eval_image_metrics.py"
PREPARE_SCRIPT="${DSDIR}/prepare_spatialedit_folders.py"
SYMLINK_BASE="${DSDIR}/output/eval_spatialedit_folders"
OUTPUT_CSV_DIR="${DSDIR}/output/eval_spatialedit_metrics"

source "${WORKDIR}/.venv/bin/activate"

# Step 1: Prepare symlink folders
echo "=== Preparing symlink folders ==="
python "${PREPARE_SCRIPT}" --all

# Step 2: Run eval_image_metrics.py for each mode in parallel
mkdir -p "${OUTPUT_CSV_DIR}"

echo ""
echo "=== Starting evaluation (3 modes in parallel) ==="

CUDA_VISIBLE_DEVICES=0 python "${METRICS_SCRIPT}" \
    --folder_a "${SYMLINK_BASE}/base/pred" \
    --folder_b "${SYMLINK_BASE}/base/gt" \
    --output_csv "${OUTPUT_CSV_DIR}/base_metrics.csv" \
    --device cuda \
    2>&1 | tee "${OUTPUT_CSV_DIR}/base_eval.log" &
PID_BASE=$!
echo "  base → GPU 0 (PID ${PID_BASE})"

CUDA_VISIBLE_DEVICES=2 python "${METRICS_SCRIPT}" \
    --folder_a "${SYMLINK_BASE}/ours/pred" \
    --folder_b "${SYMLINK_BASE}/ours/gt" \
    --output_csv "${OUTPUT_CSV_DIR}/ours_metrics.csv" \
    --device cuda \
    2>&1 | tee "${OUTPUT_CSV_DIR}/ours_eval.log" &
PID_OURS=$!
echo "  ours → GPU 2 (PID ${PID_OURS})"

CUDA_VISIBLE_DEVICES=4 python "${METRICS_SCRIPT}" \
    --folder_a "${SYMLINK_BASE}/fal/pred" \
    --folder_b "${SYMLINK_BASE}/fal/gt" \
    --output_csv "${OUTPUT_CSV_DIR}/fal_metrics.csv" \
    --device cuda \
    2>&1 | tee "${OUTPUT_CSV_DIR}/fal_eval.log" &
PID_FAL=$!
echo "  fal  → GPU 4 (PID ${PID_FAL})"

echo ""
echo "Waiting for all evaluations to complete..."
wait ${PID_BASE}
RC_BASE=$?
echo "  base done (exit ${RC_BASE})"

wait ${PID_OURS}
RC_OURS=$?
echo "  ours done (exit ${RC_OURS})"

wait ${PID_FAL}
RC_FAL=$?
echo "  fal  done (exit ${RC_FAL})"

# Step 3: Print summary
echo ""
echo "=========================================="
echo "SpatialEdit-Bench Evaluation Complete"
echo "=========================================="
for MODE in base ours fal; do
    CSV="${OUTPUT_CSV_DIR}/${MODE}_metrics.csv"
    if [ -f "${CSV}" ]; then
        echo ""
        echo "--- ${MODE} ---"
        tail -1 "${CSV}"
    fi
done

echo ""
echo "Results saved to: ${OUTPUT_CSV_DIR}/"
