#!/bin/bash
# R2 SpatialEdit-Bench inference + camera-level metrics (6 GPU parallel).
# Run after R2 training completes (epoch 29).
set -euo pipefail

WORKDIR=/gemini/platform/public/aigc/aigc_image/zhanghy56_intern/zhangqisong/data-build
source "$WORKDIR/.venv/bin/activate"
export PYTHONPATH="$WORKDIR/DiffSynth-Studio:${PYTHONPATH:-}"
cd "$WORKDIR/DiffSynth-Studio"

CHECKPOINT=$WORKDIR/DiffSynth-Studio/output/v2_scaling_r2_quality_gate/epoch_0029/lora.safetensors
EVAL_OUTPUT=$WORKDIR/feedback_loop_runs/v2_scaling_r2_quality_gate/round_1/eval_results
OUT_DIR=$EVAL_OUTPUT/spatialedit/ours_feedback
LOG_DIR=$EVAL_OUTPUT/inference_logs
mkdir -p "$OUT_DIR" "$LOG_DIR" "$EVAL_OUTPUT/spatialedit_metrics"

NUM_GPUS=6
echo "=== R2 6-GPU parallel inference ==="
echo "Checkpoint: $CHECKPOINT"
echo "Output: $OUT_DIR"

# Phase 1: 6-GPU sharded inference using eval_shard_inference.py
PIDS=()
for GPU_ID in $(seq 0 $((NUM_GPUS-1))); do
    LOG="$LOG_DIR/inference_gpu${GPU_ID}.log"
    CUDA_VISIBLE_DEVICES=$GPU_ID python eval_shard_inference.py \
        --shard-id $GPU_ID \
        --num-shards $NUM_GPUS \
        --lora-path "$CHECKPOINT" \
        --output-dir "$OUT_DIR" \
        > "$LOG" 2>&1 &
    PIDS+=($!)
    echo "  Launched GPU $GPU_ID (PID $!) → $LOG"
done

echo "Waiting for ${#PIDS[@]} shards..."
FAIL=0
for i in $(seq 0 $((NUM_GPUS-1))); do
    if ! wait ${PIDS[$i]}; then
        FAIL=$((FAIL+1))
        echo "  Shard $i FAILED — see $LOG_DIR/inference_gpu${i}.log"
    else
        echo "  Shard $i done"
    fi
done
[ $FAIL -gt 0 ] && { echo "FATAL: $FAIL shards failed"; exit 1; }

TOTAL=$(ls "$OUT_DIR"/*.png 2>/dev/null | wc -l)
echo "Total predicted images: $TOTAL"

# Phase 2: camera-level metrics (PSNR/SSIM/LPIPS/CLIP-I/DINO/FID)
echo "=== R2 camera-level metrics ==="
CUDA_VISIBLE_DEVICES=0 python - <<PY
import importlib.util
import json
import os
import shutil

EVAL_OUTPUT = "$EVAL_OUTPUT"
SCRIPT_PATH = "$WORKDIR/DiffSynth-Studio/eval_spatialedit_metrics.py"

spec = importlib.util.spec_from_file_location("esm", SCRIPT_PATH)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
mod.OUTPUT_BASE = os.path.join(EVAL_OUTPUT, "spatialedit")
mod.METRICS_DIR = os.path.join(EVAL_OUTPUT, "spatialedit_metrics")
os.makedirs(mod.METRICS_DIR, exist_ok=True)
summary = mod.run_metrics("ours_feedback", device="cuda:0")
per_pair = os.path.join(mod.METRICS_DIR, "ours_feedback_per_pair.csv")
metrics_csv = os.path.join(mod.METRICS_DIR, "ours_feedback_metrics.csv")
if os.path.exists(per_pair):
    shutil.copyfile(per_pair, metrics_csv)
with open(os.path.join(mod.METRICS_DIR, "ours_feedback_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print("Camera-level summary:", json.dumps(summary, indent=2, ensure_ascii=False))
PY

echo "=== R2 inference + camera-level metrics done at $(date '+%F %T') ==="
echo "Next: run viescore_command.sh, then compare_command.sh"
