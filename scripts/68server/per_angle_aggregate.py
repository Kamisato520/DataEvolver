import csv, os
from collections import defaultdict
import numpy as np

METRICS_DIR = "/gemini/platform/public/aigc/aigc_image/zhanghy56_intern/zhangqisong/data-build/DiffSynth-Studio/output/eval_spatialedit_metrics"

ANGLE_MAP = {0: 45, 1: 90, 2: 135, 3: 180, 4: 225, 5: 270, 6: 315, 7: 360}
VIEW_MAP = {45: "front-right", 90: "right side", 135: "back-right", 180: "back", 225: "back-left", 270: "left side", 315: "front-left", 360: "front(360)"}

for mode in ["base", "ours", "fal", "ours_objinfo"]:
    csv_path = os.path.join(METRICS_DIR, f"{mode}_metrics.csv")
    if not os.path.exists(csv_path):
        continue

    angle_data = defaultdict(list)
    with open(csv_path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = row["image_name"]
            angle_str = fname.split("_angle")[-1].replace(".png", "")
            if not angle_str.isdigit():
                continue
            angle_idx = int(angle_str)
            angle_deg = ANGLE_MAP.get(angle_idx, angle_idx)

            angle_data[angle_deg].append({
                "psnr": float(row["psnr"]),
                "ssim": float(row["ssim"]),
                "lpips": float(row["lpips"]),
                "dino": float(row["dino_similarity"]),
                "clip_i": float(row["clip_similarity"]),
            })

    print(f"\n=== {mode} ({sum(len(v) for v in angle_data.values())} pairs) ===")
    print(f"{'Angle':>6} {'View':<14} {'N':>3} {'PSNR':>7} {'SSIM':>7} {'LPIPS':>7} {'CLIP-I':>7} {'DINO':>7}")
    print("-" * 65)

    all_psnr, all_ssim, all_lpips, all_clip, all_dino = [], [], [], [], []
    for angle in sorted(angle_data.keys()):
        rows = angle_data[angle]
        psnr = np.mean([r["psnr"] for r in rows])
        ssim = np.mean([r["ssim"] for r in rows])
        lpips = np.mean([r["lpips"] for r in rows])
        clip_i = np.mean([r["clip_i"] for r in rows])
        dino = np.mean([r["dino"] for r in rows])
        view = VIEW_MAP.get(angle, f"yaw{angle}")
        print(f"{angle:>5} {view:<14} {len(rows):>3} {psnr:>7.2f} {ssim:>7.4f} {lpips:>7.4f} {clip_i:>7.4f} {dino:>7.4f}")
        all_psnr.extend([r["psnr"] for r in rows])
        all_ssim.extend([r["ssim"] for r in rows])
        all_lpips.extend([r["lpips"] for r in rows])
        all_clip.extend([r["clip_i"] for r in rows])
        all_dino.extend([r["dino"] for r in rows])

    print(f"{'AVG':>6} {'':<14} {len(all_psnr):>3} {np.mean(all_psnr):>7.2f} {np.mean(all_ssim):>7.4f} {np.mean(all_lpips):>7.4f} {np.mean(all_clip):>7.4f} {np.mean(all_dino):>7.4f}")
