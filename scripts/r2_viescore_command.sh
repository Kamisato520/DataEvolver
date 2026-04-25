#!/bin/bash
# R2 SpatialEdit-Bench VIEScore eval — single mode (ours_feedback) for R2.
# Uses Qwen3.5-VL-8B backbone. Default: single GPU sequential (488 pairs ~30-60min).
set -euo pipefail

WORKDIR=/gemini/platform/public/aigc/aigc_image/zhanghy56_intern/zhangqisong/data-build
DSDIR=${WORKDIR}/DiffSynth-Studio
BENCH_DIR=${WORKDIR}/SpatialEdit-Bench
BENCH_DATA_DIR=${BENCH_DIR}/images
META_FILE=${BENCH_DIR}/SpatialEdit_Bench_Meta_File.json
EVAL_CODE_DIR=${WORKDIR}/SpatialEdit-Bench-Eval
SAVE_DIR=${EVAL_CODE_DIR}/csv_results

R2_EVAL_OUT=${WORKDIR}/feedback_loop_runs/v2_scaling_r2_quality_gate/round_1/eval_results
MODE=ours_feedback
BACKBONE=qwen35vl
LANGUAGE=en
GPU=${VIESCORE_GPU:-0}

source ${WORKDIR}/.venv/bin/activate
export HF_ENDPOINT=https://hf-mirror.com

# Phase 1: Reorganize R2 predictions into VIEScore-expected layout
echo "Phase 1: Reorganize R2 predictions for VIEScore"
python3 - <<PY
import json, os
meta_file = "${META_FILE}"
pred_root = "${R2_EVAL_OUT}/spatialedit/${MODE}"
eval_meta = os.path.join(pred_root, "eval_meta.json")
target_base = "${EVAL_CODE_DIR}/${MODE}/fullset/rotate/en"
os.makedirs(target_base, exist_ok=True)

if not os.path.exists(eval_meta):
    raise SystemExit(f"[ERR] missing eval_meta.json at {eval_meta}")
with open(eval_meta) as f:
    pairs = json.load(f)

created = 0
for pair in pairs:
    obj_name = pair["obj_name"]
    edit_id = f"{pair['angle_idx']:02d}"
    target_dir = os.path.join(target_base, obj_name)
    os.makedirs(target_dir, exist_ok=True)
    target_link = os.path.join(target_dir, f"{edit_id}.png")
    if os.path.lexists(target_link):
        os.remove(target_link)
    os.symlink(os.path.abspath(pair["pred_path"]), target_link)
    created += 1
print(f"[{${MODE!r}}] {created} symlinks under {target_base}")
PY

# Phase 2: VIEScore via Qwen3.5-VL-8B (single GPU sequential)
echo "Phase 2: VIEScore eval (GPU ${GPU}, mode ${MODE})"
mkdir -p "${SAVE_DIR}"
cd ${EVAL_CODE_DIR}/object_level_eval

LOG=${SAVE_DIR}/${MODE}_viescore.log
CUDA_VISIBLE_DEVICES=${GPU} python3 calculate_score.py \
    --model_name "${MODE}" \
    --save_dir "${SAVE_DIR}" \
    --backbone "${BACKBONE}" \
    --edited_images_dir "${EVAL_CODE_DIR}" \
    --metadata_path "${META_FILE}" \
    --bench-data-dir="${BENCH_DATA_DIR}" \
    --instruction_language "${LANGUAGE}" \
    --type rotate \
    2>&1 | tee "${LOG}"

# Phase 3: Aggregate statistics
echo "Phase 3: Statistics"
python3 calculate_statistics.py \
    --model_name "${MODE}" \
    --save_path "${SAVE_DIR}" \
    --backbone "${BACKBONE}" \
    --language "${LANGUAGE}"

# Mirror VIEScore CSV into R2 eval_results for archival
mkdir -p "${R2_EVAL_OUT}/viescore"
SRC_CSV="${SAVE_DIR}/${MODE}/${BACKBONE}/${MODE}_rotate_${LANGUAGE}_vie_score.csv"
if [ -f "${SRC_CSV}" ]; then
    cp "${SRC_CSV}" "${R2_EVAL_OUT}/viescore/${MODE}_rotate_${LANGUAGE}_vie_score.csv"
    echo "Mirrored: ${R2_EVAL_OUT}/viescore/${MODE}_rotate_${LANGUAGE}_vie_score.csv"
fi
RESULT_TXT="${SAVE_DIR}/${MODE}/${BACKBONE}_results.txt"
[ -f "${RESULT_TXT}" ] && cp "${RESULT_TXT}" "${R2_EVAL_OUT}/viescore/${MODE}_${BACKBONE}_results.txt"

echo "=== VIEScore done at $(date '+%F %T') ==="
[ -f "${RESULT_TXT}" ] && cat "${RESULT_TXT}"
