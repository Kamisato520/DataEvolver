#!/usr/bin/env python3
"""
Prepare symlink folders for eval_image_metrics.py from SpatialEdit-Bench inference results.

Reads eval_meta.json (from eval_spatialedit_inference.py output) and creates
two flat folders with matching filenames:
  - {output}/pred/  → symlinks to prediction images
  - {output}/gt/    → symlinks to GT images

Then eval_image_metrics.py can directly compare them:
  python eval_image_metrics.py --folder_a {output}/pred --folder_b {output}/gt --output_csv ...

Usage:
    python prepare_spatialedit_folders.py --mode ours
    python prepare_spatialedit_folders.py --mode base
    python prepare_spatialedit_folders.py --mode fal
"""
import os, json, argparse
from pathlib import Path

_WORKDIR = "/gemini/platform/public/aigc/aigc_image/zhanghy56_intern/zhangqisong/data-build"
OUTPUT_BASE = f"{_WORKDIR}/DiffSynth-Studio/output/eval_spatialedit"
SYMLINK_BASE = f"{_WORKDIR}/DiffSynth-Studio/output/eval_spatialedit_folders"


def prepare(mode):
    pred_dir = os.path.join(OUTPUT_BASE, mode)
    meta_path = os.path.join(pred_dir, "eval_meta.json")
    if not os.path.exists(meta_path):
        print(f"[ERROR] {meta_path} not found")
        return

    with open(meta_path) as f:
        pairs = json.load(f)
    print(f"[{mode}] {len(pairs)} pairs")

    pred_folder = os.path.join(SYMLINK_BASE, mode, "pred")
    gt_folder = os.path.join(SYMLINK_BASE, mode, "gt")
    os.makedirs(pred_folder, exist_ok=True)
    os.makedirs(gt_folder, exist_ok=True)

    created = 0
    skipped = 0
    for pair in pairs:
        pair_id = pair["pair_id"]
        pred_path = pair["pred_path"]
        gt_path = pair["gt_path"]

        if not os.path.exists(pred_path):
            print(f"  [WARN] pred not found: {pred_path}")
            skipped += 1
            continue
        if not os.path.exists(gt_path):
            print(f"  [WARN] gt not found: {gt_path}")
            skipped += 1
            continue

        fname = f"{pair_id}.png"
        pred_link = os.path.join(pred_folder, fname)
        gt_link = os.path.join(gt_folder, fname)

        if os.path.exists(pred_link):
            os.remove(pred_link)
        if os.path.exists(gt_link):
            os.remove(gt_link)

        os.symlink(os.path.abspath(pred_path), pred_link)
        os.symlink(os.path.abspath(gt_path), gt_link)
        created += 1

    print(f"  Created {created} symlink pairs, skipped {skipped}")
    print(f"  pred: {pred_folder}")
    print(f"  gt:   {gt_folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["base", "ours", "fal"], default=None)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    if args.all:
        modes = ["base", "ours", "fal"]
    elif args.mode:
        modes = [args.mode]
    else:
        parser.error("Specify --mode or --all")

    for mode in modes:
        prepare(mode)
