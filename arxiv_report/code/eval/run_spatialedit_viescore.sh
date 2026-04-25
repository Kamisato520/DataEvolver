#!/bin/bash
# SpatialEdit-Bench official VIEScore evaluation (rotate only)
# Uses Qwen3.5-35B-A3B as VLM backbone
set -e

WORKDIR=/gemini/platform/public/aigc/aigc_image/zhanghy56_intern/zhangqisong/data-build
DSDIR=${WORKDIR}/DiffSynth-Studio
BENCH_DIR=${WORKDIR}/SpatialEdit-Bench
EVAL_DIR=${DSDIR}/output/eval_spatialedit
META_FILE=${BENCH_DIR}/SpatialEdit_Bench_Meta_File.json
EVAL_CODE_DIR=${WORKDIR}/SpatialEdit-Bench-Eval
SAVE_DIR=${EVAL_CODE_DIR}/csv_results

BACKBONE=qwen35vl
LANGUAGE=en
MODES="base ours fal ours_objinfo"

source ${WORKDIR}/.venv/bin/activate
export HF_ENDPOINT=https://hf-mirror.com

# ============================================
# Phase 1: Reorganize predictions into SpatialEdit directory structure
# ============================================
echo "============================================"
echo "Phase 1: Reorganize predictions"
echo "============================================"

python3 -c "
import os, json

meta_file = '${META_FILE}'
eval_dir = '${EVAL_DIR}'
eval_code_dir = '${EVAL_CODE_DIR}'
modes = '${MODES}'.split()

with open(meta_file) as f:
    meta = json.load(f)

rotate_items = [x for x in meta if x['type'] == 'rotate']
print(f'Metadata: {len(rotate_items)} rotate items')

lookup = {}
for item in rotate_items:
    lookup[(item['image_id'], item['edit_id'])] = item

for mode in modes:
    pred_dir = os.path.join(eval_dir, mode)
    meta_path = os.path.join(pred_dir, 'eval_meta.json')
    if not os.path.exists(meta_path):
        print(f'[SKIP] {mode}: no eval_meta.json')
        continue

    with open(meta_path) as f:
        pairs = json.load(f)

    target_base = os.path.join(eval_code_dir, mode, 'fullset', 'rotate', 'en')
    os.makedirs(target_base, exist_ok=True)

    created = 0
    for pair in pairs:
        obj_name = pair['obj_name']
        angle_idx = pair['angle_idx']
        edit_id = f'{angle_idx:02d}'
        pred_path = pair['pred_path']

        target_dir = os.path.join(target_base, obj_name)
        os.makedirs(target_dir, exist_ok=True)

        target_link = os.path.join(target_dir, f'{edit_id}.png')
        if os.path.exists(target_link):
            os.remove(target_link)
        os.symlink(os.path.abspath(pred_path), target_link)
        created += 1

    print(f'[{mode}] {created} symlinks created at {target_base}')
"

# ============================================
# Phase 2: Run VIEScore evaluation for each mode
# ============================================
echo ""
echo "============================================"
echo "Phase 2: VIEScore evaluation (${BACKBONE})"
echo "============================================"

cd ${EVAL_CODE_DIR}/object_level_eval

for MODE in ${MODES}; do
    echo ""
    echo "--- Evaluating ${MODE} ---"

    EDITED_IMAGES_DIR=${EVAL_CODE_DIR}
    MODEL_NAME=${MODE}

    python3 calculate_score.py \
        --model_name "${MODEL_NAME}" \
        --save_dir "${SAVE_DIR}" \
        --backbone "${BACKBONE}" \
        --edited_images_dir "${EDITED_IMAGES_DIR}" \
        --metadata_path "${META_FILE}" \
        --bench-data-dir="${BENCH_DIR}" \
        --instruction_language "${LANGUAGE}" \
        --type rotate

    echo "--- ${MODE} scoring done ---"
done

# ============================================
# Phase 3: Calculate statistics for each mode
# ============================================
echo ""
echo "============================================"
echo "Phase 3: Statistics"
echo "============================================"

for MODE in ${MODES}; do
    echo ""
    echo "--- Statistics for ${MODE} ---"

    python3 calculate_statistics.py \
        --model_name "${MODE}" \
        --save_path "${SAVE_DIR}" \
        --backbone "${BACKBONE}" \
        --language "${LANGUAGE}"
done

# ============================================
# Print all results
# ============================================
echo ""
echo "============================================"
echo "All Results (${BACKBONE})"
echo "============================================"
for MODE in ${MODES}; do
    RESULT_FILE=${SAVE_DIR}/${MODE}/${BACKBONE}_results.txt
    if [ -f "${RESULT_FILE}" ]; then
        echo ""
        echo "=== ${MODE} ==="
        cat "${RESULT_FILE}"
    fi
done

echo ""
echo "Done! Results in: ${SAVE_DIR}/"
