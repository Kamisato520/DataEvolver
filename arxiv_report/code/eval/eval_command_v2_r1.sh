#!/bin/bash
set -euo pipefail

WORKDIR=/gemini/platform/public/aigc/aigc_image/zhanghy56_intern/zhangqisong/data-build
source "$WORKDIR/.venv/bin/activate"
export PYTHONPATH="$WORKDIR/DiffSynth-Studio:${PYTHONPATH:-}"
CHECKPOINT=$WORKDIR/DiffSynth-Studio/output/v2_scaling_r1_augmented/epoch_0029/lora.safetensors
EVAL_OUTPUT=$WORKDIR/feedback_loop_runs/v2_scaling_r1/round_1/eval_results

mkdir -p "$EVAL_OUTPUT/spatialedit" "$EVAL_OUTPUT/spatialedit_metrics"
cd "$WORKDIR/DiffSynth-Studio"

CUDA_VISIBLE_DEVICES=0 python eval_spatialedit_inference.py \
  --mode ours \
  --device cuda:0 \
  --lora_path "$CHECKPOINT" \
  --output_dir "$EVAL_OUTPUT/spatialedit/ours_feedback"

CUDA_VISIBLE_DEVICES=0 python - <<'PY'
import importlib.util
import json
import os
import shutil

WORKDIR = "/gemini/platform/public/aigc/aigc_image/zhanghy56_intern/zhangqisong/data-build"
EVAL_OUTPUT = f"{WORKDIR}/feedback_loop_runs/v2_scaling_r1/round_1/eval_results"
SCRIPT_PATH = f"{WORKDIR}/DiffSynth-Studio/eval_spatialedit_metrics.py"

spec = importlib.util.spec_from_file_location("eval_spatialedit_metrics_custom", SCRIPT_PATH)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
mod.OUTPUT_BASE = os.path.join(EVAL_OUTPUT, "spatialedit")
mod.METRICS_DIR = os.path.join(EVAL_OUTPUT, "spatialedit_metrics")
os.makedirs(mod.METRICS_DIR, exist_ok=True)
summary = mod.run_metrics("ours_feedback", device="cuda:0")
per_pair = os.path.join(mod.METRICS_DIR, "ours_feedback_per_pair.csv")
metrics_csv = os.path.join(mod.METRICS_DIR, "ours_feedback_metrics.csv")
shutil.copyfile(per_pair, metrics_csv)
with open(os.path.join(mod.METRICS_DIR, "ours_feedback_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print(metrics_csv)
PY

echo "Eval completed at $(date '+%F %T')"
