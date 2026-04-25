"""
Quantitative evaluation metrics for rotation editing (68 server).

Metrics:
  - Test set (has GT): PSNR, SSIM, LPIPS, CLIP-I, DINO, FID
  - Benchmark (no GT):  CLIP-T (text-image), DINO-src (source identity)

Usage:
    python eval_metrics.py --eval testset
    python eval_metrics.py --eval both
"""
import os, sys, json, argparse, csv
import numpy as np
from pathlib import Path
from PIL import Image

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# ── 68 服务器路径 ──
_WORKDIR = "/gemini/platform/public/aigc/aigc_image/zhanghy56_intern/zhangqisong/data-build"

SPLIT_ROOT = (
    f"{_WORKDIR}/dataset_scene_v7_full50_rotation8_trainready_front2others"
    "_splitobj_seed42_bboxmask_final_20260414"
)
EVAL_INFERENCE_BASE = f"{_WORKDIR}/DiffSynth-Studio/output/eval_inference"
OUTPUT_BASE = f"{_WORKDIR}/DiffSynth-Studio/output/eval_metrics"
MODES = ["base", "fal", "ours"]

# ── lazy-loaded model singletons ──
_lpips_model = None
_clip_model = None
_clip_preprocess = None
_di