#!/bin/bash
# SpatialEdit-Bench VIEScore evaluation — sequential modes, single GPU
set -e

WORKDIR=/gemini/platform/public/aigc/aigc_image/zhanghy56_intern/zhangqisong/data-build
DSDIR=${WORKDIR}/DiffSynth-Studio
BENCH_DIR=${WORKDIR}/SpatialEdit-Bench
BENCH_DATA_DIR=${BENCH_DIR}/images
META_FILE=${BENCH_DIR}/SpatialEdit_Bench_Meta_File.json
EVAL_CODE_DIR=${WORKDIR}/SpatialEdit-Bench-Eval
SAVE_DIR=${EVAL_CODE_DIR}/csv_results

BACKBONE=qwen35vl
LANGUAGE=en

source ${WORKDIR}/.venv/bin/activate
export HF_ENDPOINT=https://hf-mirror.com

# Phase 1: Reorganize predictions
echo "Phase 1: Reorganize predictions"
python3 -c "
import os, json
meta_file = '${META_FILE}'
eval_dir = '${DSDIR}/output/eval_spatialedit'
eval_code_dir = '${EVAL_CODE_DIR}'
modes = ['base', 'ours', 'fal', 'ours_objinfo']
with open(meta_file) as f:
    meta = json.load(f)
for mode in modes:
    pred_dir = os.path.join(eval_dir, mode)
    meta_path = os.path.join(pred_dir, 'eval_meta.json')
    if not os.path.exists(meta_path):
        print(f'[SKIP] {mode}'); continue
    with open(meta_path) as f:
        pairs = json.load(f)
    target_base = os.path.join(eval_code_dir, mode, 'fullset', 'rotate', 'en')
    os.makedirs(target_base, exist_ok=True)
    created = 0
    for pair in pairs:
        obj_name = pair['obj_name']
        edit_id = f\"{pair['angle_idx']:02d}\"
        target_dir = os.path.join(target_base, obj_name)
        os.makedirs(target_dir, exist_ok=True)
        target_link = os.path.join(target_dir, f'{edit_id}.png')
        if os.path.exists(target_link): os.remove(target_link)
        os.symlink(os.path.abspath(pair['pred_path']), target_link)
        created += 1
    print(f'[{mode}] {created} symlinks')
"

# Phase 2: Evaluate each mode sequentially on GPU 0
echo ""
echo "Phase 2: VIEScore evaluation (sequential, GPU 0)"

cd ${EVAL_CODE_DIR}/object_level_eval
mkdir -p ${SAVE_DIR}

for MODE in base ours fal ours_objinfo; do
    echo ""
    echo "=== Evaluating ${MODE} ==="

    CUDA_VISIBLE_DEVICES=0 python3 calculate_score.py \
        --model_name "${MODE}" \
        --save_dir "${SAVE_DIR}" \
        --backbone "${BACKBONE}" \
        --edited_images_dir "${EVAL_CODE_DIR}" \
        --metadata_path "${META_FILE}" \
        --bench-data-dir="${BENCH_DATA_DIR}" \
        --instruction_language "${LANGUAGE}" \
        --type rotate \
        2>&1 | tee ${SAVE_DIR}/${MODE}_viescore.log

    echo "=== ${MODE} done ==="
done

# Phase 3: Statistics
echo ""
echo "Phase 3: Statistics"

for MODE in base ours fal ours_objinfo; do
    python3 calculate_statistics.py \
        --model_name "${MODE}" \
        --save_path "${SAVE_DIR}" \
        --backbone "${BACKBONE}" \
        --language "${LANGUAGE}"
done

# Print results
echo ""
echo "============================================"
echo "SpatialEdit-Bench VIEScore Results (${BACKBONE})"
echo "============================================"
for MODE in base ours fal ours_objinfo; do
    RESULT_FILE=${SAVE_DIR}/${MODE}/${BACKBONE}_results.txt
    if [ -f "${RESULT_FILE}" ]; then
        echo ""
        echo "=== ${MODE} ==="
        cat "${RESULT_FILE}"
    fi
done

echo ""
echo "Done! Results: ${SAVE_DIR}/"
