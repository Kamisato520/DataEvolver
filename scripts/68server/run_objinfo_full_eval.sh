#!/bin/bash
set -e

WORKDIR=/gemini/platform/public/aigc/aigc_image/zhanghy56_intern/zhangqisong/data-build
DSDIR=${WORKDIR}/DiffSynth-Studio
LORA_PATH=${DSDIR}/output/rotation8_bright_objinfo_rank32/epoch_0029/lora.safetensors
PRED_DIR=${DSDIR}/output/eval_spatialedit/ours_objinfo
SYMLINK_BASE=${DSDIR}/output/eval_spatialedit_folders
METRICS_DIR=${DSDIR}/output/eval_spatialedit_metrics

cd ${DSDIR}
source ${WORKDIR}/.venv/bin/activate
export HF_ENDPOINT=https://hf-mirror.com

echo '============================================'
echo 'Phase 1/3: Inference (objinfo LoRA epoch 29)'
echo '============================================'

CUDA_VISIBLE_DEVICES=0 python eval_spatialedit_inference.py \
    --mode ours \
    --lora_path ${LORA_PATH} \
    --output_dir ${PRED_DIR} \
    --device cuda

echo ''
echo '============================================'
echo 'Phase 2/3: Prepare symlink folders'
echo '============================================'

python -c "
import os, json
pred_dir = '${PRED_DIR}'
meta_path = os.path.join(pred_dir, 'eval_meta.json')
with open(meta_path) as f:
    pairs = json.load(f)
print(f'[ours_objinfo] {len(pairs)} pairs')
pred_folder = '${SYMLINK_BASE}/ours_objinfo/pred'
gt_folder = '${SYMLINK_BASE}/ours_objinfo/gt'
os.makedirs(pred_folder, exist_ok=True)
os.makedirs(gt_folder, exist_ok=True)
created = 0
for pair in pairs:
    fname = pair['pair_id'] + '.png'
    pred_link = os.path.join(pred_folder, fname)
    gt_link = os.path.join(gt_folder, fname)
    for lnk in [pred_link, gt_link]:
        if os.path.exists(lnk):
            os.remove(lnk)
    os.symlink(os.path.abspath(pair['pred_path']), pred_link)
    os.symlink(os.path.abspath(pair['gt_path']), gt_link)
    created += 1
print(f'  Created {created} symlink pairs')
print(f'  pred: {pred_folder}')
print(f'  gt:   {gt_folder}')
"

echo ''
echo '============================================'
echo 'Phase 3/3: Evaluation (eval_image_metrics.py)'
echo '============================================'

mkdir -p ${METRICS_DIR}

CUDA_VISIBLE_DEVICES=0 python ${DSDIR}/metrics/eval_image_metrics.py \
    --folder_a ${SYMLINK_BASE}/ours_objinfo/pred \
    --folder_b ${SYMLINK_BASE}/ours_objinfo/gt \
    --output_csv ${METRICS_DIR}/ours_objinfo_metrics.csv \
    --device cuda \
    2>&1 | tee ${METRICS_DIR}/ours_objinfo_eval.log

echo ''
echo '============================================'
echo 'All done! Comparing all methods:'
echo '============================================'
for MODE in base ours fal ours_objinfo; do
    CSV=${METRICS_DIR}/${MODE}_metrics.csv
    if [ -f "${CSV}" ]; then
        echo ""
        echo "--- ${MODE} ---"
        tail -1 "${CSV}"
    fi
done

echo ''
echo "Results: ${METRICS_DIR}/"
