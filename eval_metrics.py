"""
Quantitative evaluation metrics for rotation editing.

Eval 1 (test set, has GT): PSNR, SSIM, LPIPS, CLIP-I, DINO
Eval 2 (benchmark, no GT): CLIP-T (text-image), DINO-src (source identity)

Usage:
    python eval_metrics.py --eval testset
    python eval_metrics.py --eval benchmark
    python eval_metrics.py --eval both
"""
import os, sys, json, argparse, csv
import numpy as np
from pathlib import Path
from PIL import Image

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# ── paths ──
SPLIT_ROOT = (
    "/aaaidata/zhangqisong/data_build_runs/scene_full20_loop_20260404_code"
    "/pipeline/data/dataset_scene_v7_full50_rotation8_trainready_front2others"
    "_splitobj_seed42_final_20260410"
)
EVAL_INFERENCE_BASE = "/aaaidata/zhangqisong/DiffSynth-Studio/output/eval_inference"
EVAL_BENCHMARK_BASE = "/aaaidata/zhangqisong/DiffSynth-Studio/output/eval_benchmark"
EVAL_BENCHMARK_A_BASE = "/aaaidata/zhangqisong/DiffSynth-Studio/output/eval_benchmark_a"
OUTPUT_BASE = "/aaaidata/zhangqisong/DiffSynth-Studio/output/eval_metrics"
MODES = ["base", "fal", "ours"]

# ── lazy-loaded model singletons ──
_lpips_model = None
_clip_model = None
_clip_preprocess = None
_dino_model = None
_dino_processor = None


def load_rgba_as_rgb(path):
    """Load image as RGB, compositing RGBA onto white background."""
    img = Image.open(path)
    if img.mode == "RGBA":
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img).convert("RGB")
    else:
        img = img.convert("RGB")
    return img


def pil_to_tensor(img):
    """PIL RGB → [1,3,H,W] float32 [0,1] on CPU."""
    import torch
    arr = np.array(img).astype(np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return t


def get_lpips_model(device):
    global _lpips_model
    if _lpips_model is None:
        import lpips
        _lpips_model = lpips.LPIPS(net="alex").to(device)
        _lpips_model.eval()
    return _lpips_model


def get_clip_model(device):
    global _clip_model, _clip_preprocess
    if _clip_model is None:
        import clip
        _clip_model, _clip_preprocess = clip.load("ViT-B/32", device=device)
        _clip_model.eval()
    return _clip_model, _clip_preprocess


def get_dino_model(device):
    global _dino_model, _dino_processor
    if _dino_model is None:
        from transformers import AutoImageProcessor, AutoModel
        _dino_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        _dino_model = AutoModel.from_pretrained("facebook/dinov2-base").to(device)
        _dino_model.eval()
    return _dino_model, _dino_processor


# ── metric functions ──

def compute_psnr(pred_np, gt_np):
    from skimage.metrics import peak_signal_noise_ratio
    return float(peak_signal_noise_ratio(gt_np, pred_np, data_range=255))


def compute_ssim(pred_np, gt_np):
    from skimage.metrics import structural_similarity
    return float(structural_similarity(gt_np, pred_np, channel_axis=2, data_range=255))


def compute_lpips_score(pred_pil, gt_pil, device):
    import torch
    model = get_lpips_model(device)
    pred_t = pil_to_tensor(pred_pil).to(device) * 2 - 1  # [0,1] → [-1,1]
    gt_t = pil_to_tensor(gt_pil).to(device) * 2 - 1
    with torch.no_grad():
        d = model(pred_t, gt_t)
    return float(d.item())


def compute_clip_i(pred_pil, gt_pil, device):
    import torch
    model, preprocess = get_clip_model(device)
    with torch.no_grad():
        feat_p = model.encode_image(preprocess(pred_pil).unsqueeze(0).to(device))
        feat_g = model.encode_image(preprocess(gt_pil).unsqueeze(0).to(device))
        feat_p = feat_p / feat_p.norm(dim=-1, keepdim=True)
        feat_g = feat_g / feat_g.norm(dim=-1, keepdim=True)
    return float((feat_p * feat_g).sum().item())


def compute_clip_t(img_pil, text, device):
    import torch, clip
    model, preprocess = get_clip_model(device)
    with torch.no_grad():
        feat_i = model.encode_image(preprocess(img_pil).unsqueeze(0).to(device))
        tokens = clip.tokenize([text]).to(device)
        feat_t = model.encode_text(tokens)
        feat_i = feat_i / feat_i.norm(dim=-1, keepdim=True)
        feat_t = feat_t / feat_t.norm(dim=-1, keepdim=True)
    return float((feat_i * feat_t).sum().item())


def compute_dino_sim(a_pil, b_pil, device):
    import torch
    model, processor = get_dino_model(device)
    inputs_a = processor(images=a_pil, return_tensors="pt").to(device)
    inputs_b = processor(images=b_pil, return_tensors="pt").to(device)
    with torch.no_grad():
        feat_a = model(**inputs_a).last_hidden_state[:, 0]
        feat_b = model(**inputs_b).last_hidden_state[:, 0]
        feat_a = feat_a / feat_a.norm(dim=-1, keepdim=True)
        feat_b = feat_b / feat_b.norm(dim=-1, keepdim=True)
    return float((feat_a * feat_b).sum().item())


def resize_to_match(pred_pil, gt_pil):
    """Resize pred to match GT dimensions if different."""
    if pred_pil.size != gt_pil.size:
        pred_pil = pred_pil.resize(gt_pil.size, Image.LANCZOS)
    return pred_pil


# ── angle/view name mapping ──
ANGLE_VIEW_MAP = {
    45: "front-right view",
    90: "right side view",
    135: "back-right view",
    180: "back view",
    225: "back-left view",
    270: "left side view",
    315: "front-left view",
}


# ── eval 1: test set with GT ──

def eval_testset(modes, split_root, eval_base, device, output_dir):
    """Compute PSNR, SSIM, LPIPS, CLIP-I, DINO on test set predictions vs GT."""
    os.makedirs(output_dir, exist_ok=True)
    split_root = Path(split_root)
    all_rows = []

    for mode in modes:
        meta_path = os.path.join(eval_base, mode, "eval_meta.json")
        if not os.path.exists(meta_path):
            print(f"[WARN] {meta_path} not found, skipping {mode}")
            continue
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        print(f"\n=== Eval 1: {mode} ({len(meta)} pairs) ===")
        for i, entry in enumerate(meta):
            pair_id = entry["pair_id"]
            pred_path = os.path.join(eval_base, mode, entry["pred_image"])
            gt_path = split_root / entry["target_image"]

            pred_pil = Image.open(pred_path).convert("RGB")
            gt_pil = load_rgba_as_rgb(gt_path)
            pred_pil = resize_to_match(pred_pil, gt_pil)

            pred_np = np.array(pred_pil)
            gt_np = np.array(gt_pil)

            psnr = compute_psnr(pred_np, gt_np)
            ssim = compute_ssim(pred_np, gt_np)
            lpips_val = compute_lpips_score(pred_pil, gt_pil, device)
            clip_i = compute_clip_i(pred_pil, gt_pil, device)
            dino = compute_dino_sim(pred_pil, gt_pil, device)

            obj_id = pair_id.split("_yaw")[0]
            angle = entry.get("target_rotation_deg", 0)

            all_rows.append({
                "mode": mode, "pair_id": pair_id, "obj_id": obj_id,
                "angle": angle, "psnr": psnr, "ssim": ssim,
                "lpips": lpips_val, "clip_i": clip_i, "dino": dino,
            })

            if (i + 1) % 10 == 0 or i == len(meta) - 1:
                print(f"  [{i+1}/{len(meta)}] PSNR={psnr:.2f} SSIM={ssim:.4f} "
                      f"LPIPS={lpips_val:.4f} CLIP-I={clip_i:.4f} DINO={dino:.4f}")

    # write CSV
    csv_path = os.path.join(output_dir, "eval1_per_pair.csv")
    fields = ["mode", "pair_id", "obj_id", "angle", "psnr", "ssim", "lpips", "clip_i", "dino"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nPer-pair CSV: {csv_path} ({len(all_rows)} rows)")

    # aggregate
    summary = {}
    for mode in modes:
        rows = [r for r in all_rows if r["mode"] == mode]
        if not rows:
            continue
        overall = {k: float(np.mean([r[k] for r in rows]))
                   for k in ["psnr", "ssim", "lpips", "clip_i", "dino"]}

        per_angle = {}
        angles = sorted(set(r["angle"] for r in rows))
        for a in angles:
            a_rows = [r for r in rows if r["angle"] == a]
            per_angle[str(a)] = {k: float(np.mean([r[k] for r in a_rows]))
                                 for k in ["psnr", "ssim", "lpips", "clip_i", "dino"]}

        summary[mode] = {"overall": overall, "per_angle": per_angle, "n_pairs": len(rows)}

    json_path = os.path.join(output_dir, "eval1_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary JSON: {json_path}")

    # print table
    print("\n" + "=" * 72)
    print("Eval 1: Test Set — Overall Metrics")
    print("=" * 72)
    print(f"{'Method':<10} {'PSNR ↑':>8} {'SSIM ↑':>8} {'LPIPS ↓':>8} {'CLIP-I ↑':>9} {'DINO ↑':>8}")
    print("-" * 72)
    for mode in modes:
        if mode not in summary:
            continue
        o = summary[mode]["overall"]
        print(f"{mode:<10} {o['psnr']:>8.2f} {o['ssim']:>8.4f} {o['lpips']:>8.4f} "
              f"{o['clip_i']:>9.4f} {o['dino']:>8.4f}")

    # per-angle breakdown
    print("\n" + "=" * 72)
    print("Eval 1: Per-Angle PSNR Breakdown")
    print("=" * 72)
    angles = sorted(set(r["angle"] for r in all_rows))
    header = f"{'Angle':<8}"
    for mode in modes:
        if mode in summary:
            header += f" {mode:>10}"
    print(header)
    print("-" * 72)
    for a in angles:
        view = ANGLE_VIEW_MAP.get(a, f"yaw{a}")
        line = f"{a:<4} {view:<18}"
        for mode in modes:
            if mode in summary and str(a) in summary[mode]["per_angle"]:
                line += f" {summary[mode]['per_angle'][str(a)]['psnr']:>10.2f}"
        print(line)

    return summary


# ── eval 2: benchmark (no GT) ──

def eval_benchmark(modes, eval_base, device, output_dir, prefix="eval2"):
    """Compute CLIP-T and DINO-src on benchmark predictions."""
    os.makedirs(output_dir, exist_ok=True)
    all_rows = []

    for mode in modes:
        mode_dir = os.path.join(eval_base, mode)
        meta_path = os.path.join(mode_dir, "benchmark_meta.json")

        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        else:
            # scan directory structure if meta missing
            print(f"[WARN] {meta_path} not found, scanning directories for {mode}")
            meta = []
            obj_dirs = sorted([d for d in os.listdir(mode_dir)
                              if os.path.isdir(os.path.join(mode_dir, d))])
            for obj_id in obj_dirs:
                for angle, view_name in ANGLE_VIEW_MAP.items():
                    pred_file = f"yaw{angle:03d}.png"
                    pred_path = os.path.join(mode_dir, obj_id, pred_file)
                    if os.path.exists(pred_path):
                        meta.append({
                            "obj_id": obj_id,
                            "source_image": f"{obj_id}.png",
                            "angle_deg": angle,
                            "view_name": view_name,
                            "instruction": f"Rotate this object from front view to {view_name}.",
                            "pred_image": f"{obj_id}/{pred_file}",
                        })

        print(f"\n=== {prefix}: {mode} ({len(meta)} pairs) ===")
        for i, entry in enumerate(meta):
            obj_id = entry["obj_id"]
            pred_path = os.path.join(mode_dir, entry["pred_image"])
            src_path = os.path.join(mode_dir, obj_id, "yaw000_source.png")

            if not os.path.exists(pred_path) or not os.path.exists(src_path):
                continue

            pred_pil = Image.open(pred_path).convert("RGB")
            src_pil = Image.open(src_path).convert("RGB")
            instruction = entry["instruction"]
            angle = entry.get("angle_deg", 0)
            view_name = entry.get("view_name", "")

            clip_t = compute_clip_t(pred_pil, instruction, device)
            dino_src = compute_dino_sim(pred_pil, src_pil, device)

            all_rows.append({
                "mode": mode, "obj_id": obj_id, "angle": angle,
                "view_name": view_name, "clip_t": clip_t, "dino_src": dino_src,
            })

            if (i + 1) % 50 == 0 or i == len(meta) - 1:
                print(f"  [{i+1}/{len(meta)}] CLIP-T={clip_t:.4f} DINO-src={dino_src:.4f}")

    # write CSV
    csv_path = os.path.join(output_dir, f"{prefix}_per_pair.csv")
    fields = ["mode", "obj_id", "angle", "view_name", "clip_t", "dino_src"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nPer-pair CSV: {csv_path} ({len(all_rows)} rows)")

    # aggregate
    summary = {}
    for mode in modes:
        rows = [r for r in all_rows if r["mode"] == mode]
        if not rows:
            continue
        overall = {k: float(np.mean([r[k] for r in rows])) for k in ["clip_t", "dino_src"]}

        per_angle = {}
        angles = sorted(set(r["angle"] for r in rows))
        for a in angles:
            a_rows = [r for r in rows if r["angle"] == a]
            per_angle[str(a)] = {k: float(np.mean([r[k] for r in a_rows]))
                                 for k in ["clip_t", "dino_src"]}

        summary[mode] = {"overall": overall, "per_angle": per_angle, "n_pairs": len(rows)}

    json_path = os.path.join(output_dir, f"{prefix}_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary JSON: {json_path}")

    # print table
    print("\n" + "=" * 50)
    print(f"{prefix}: Benchmark — Overall Metrics (no GT)")
    print("=" * 50)
    print(f"{'Method':<10} {'CLIP-T ↑':>10} {'DINO-src ↑':>12}")
    print("-" * 50)
    for mode in modes:
        if mode not in summary:
            continue
        o = summary[mode]["overall"]
        print(f"{mode:<10} {o['clip_t']:>10.4f} {o['dino_src']:>12.4f}")

    return summary


# ── main ──

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Eval metrics for rotation editing")
    parser.add_argument("--eval", choices=["testset", "benchmark", "benchmark_a", "both"], default="both")
    parser.add_argument("--modes", nargs="+", default=MODES)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_dir", default=OUTPUT_BASE)
    parser.add_argument("--eval_inference_base", default=EVAL_INFERENCE_BASE)
    parser.add_argument("--eval_benchmark_base", default=EVAL_BENCHMARK_BASE)
    parser.add_argument("--eval_benchmark_a_base", default=EVAL_BENCHMARK_A_BASE)
    parser.add_argument("--split_root", default=SPLIT_ROOT)
    args = parser.parse_args()

    print(f"Eval mode: {args.eval}")
    print(f"Models: {args.modes}")
    print(f"Device: {args.device}")
    print(f"Output: {args.output_dir}")

    combined = {}

    if args.eval in ("testset", "both"):
        s1 = eval_testset(args.modes, args.split_root, args.eval_inference_base,
                          args.device, args.output_dir)
        combined["eval1_testset"] = s1

    if args.eval in ("benchmark", "both"):
        s2 = eval_benchmark(args.modes, args.eval_benchmark_base,
                            args.device, args.output_dir)
        combined["eval2_benchmark"] = s2

    if args.eval in ("benchmark_a", "both"):
        s3 = eval_benchmark(args.modes, args.eval_benchmark_a_base,
                            args.device, args.output_dir, prefix="eval3_benchmark_a")
        combined["eval3_benchmark_a"] = s3

    # save combined
    combined_path = os.path.join(args.output_dir, "combined_summary.json")
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2)
    print(f"\nCombined summary: {combined_path}")
    print("Done.")
