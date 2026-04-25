"""
Metrics for SpatialEdit-Bench rotate subset: compare predictions against GT.
Computes PSNR, SSIM, LPIPS, CLIP-I, DINO, FID per mode, with per-angle breakdown.

Aligned with eval_metrics.py (our standard test set eval) for consistent comparison.

Usage:
    python eval_spatialedit_metrics.py --mode ours
    python eval_spatialedit_metrics.py --mode base
    python eval_spatialedit_metrics.py --mode fal
    python eval_spatialedit_metrics.py --all
"""
import os, sys, json, argparse, csv
import numpy as np
from PIL import Image

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

_WORKDIR = "/gemini/platform/public/aigc/aigc_image/zhanghy56_intern/zhangqisong/data-build"
OUTPUT_BASE = f"{_WORKDIR}/DiffSynth-Studio/output/eval_spatialedit"
METRICS_DIR = f"{_WORKDIR}/DiffSynth-Studio/output/eval_spatialedit_metrics"

# lazy-loaded model singletons
_lpips_model = None
_clip_model = None
_clip_preprocess = None
_dino_model = None
_dino_processor = None
_inception_model = None


def load_rgba_as_rgb(path):
    img = Image.open(path)
    if img.mode == "RGBA":
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img).convert("RGB")
    else:
        img = img.convert("RGB")
    return img


def pil_to_tensor(img):
    import torch
    arr = np.array(img).astype(np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return t


# ── model loaders ──

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


def get_inception_model(device):
    global _inception_model
    if _inception_model is None:
        import torch
        from torchvision.models import inception_v3, Inception_V3_Weights
        model = inception_v3(weights=Inception_V3_Weights.DEFAULT)
        model.fc = torch.nn.Identity()
        model = model.to(device)
        model.eval()
        _inception_model = model
    return _inception_model


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
    pred_t = pil_to_tensor(pred_pil).to(device) * 2 - 1
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


def compute_dino_sim(pred_pil, gt_pil, device):
    import torch
    model, processor = get_dino_model(device)
    inputs_a = processor(images=pred_pil, return_tensors="pt").to(device)
    inputs_b = processor(images=gt_pil, return_tensors="pt").to(device)
    with torch.no_grad():
        feat_a = model(**inputs_a).last_hidden_state[:, 0]
        feat_b = model(**inputs_b).last_hidden_state[:, 0]
        feat_a = feat_a / feat_a.norm(dim=-1, keepdim=True)
        feat_b = feat_b / feat_b.norm(dim=-1, keepdim=True)
    return float((feat_a * feat_b).sum().item())


def extract_inception_features(images_pil, device, batch_size=16):
    import torch
    from torchvision import transforms
    model = get_inception_model(device)
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    all_feats = []
    for i in range(0, len(images_pil), batch_size):
        batch = images_pil[i:i + batch_size]
        tensors = torch.stack([transform(img) for img in batch]).to(device)
        with torch.no_grad():
            feats = model(tensors)
        all_feats.append(feats.cpu().numpy())
    return np.concatenate(all_feats, axis=0)


def compute_fid(pred_features, gt_features):
    from scipy import linalg
    mu_pred = np.mean(pred_features, axis=0)
    mu_gt = np.mean(gt_features, axis=0)
    sigma_pred = np.cov(pred_features, rowvar=False)
    sigma_gt = np.cov(gt_features, rowvar=False)
    diff = mu_pred - mu_gt
    covmean, _ = linalg.sqrtm(sigma_pred @ sigma_gt, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = float(diff @ diff + np.trace(sigma_pred + sigma_gt - 2 * covmean))
    return fid


def resize_to_match(pred_pil, gt_pil):
    if pred_pil.size != gt_pil.size:
        pred_pil = pred_pil.resize(gt_pil.size, Image.LANCZOS)
    return pred_pil


# ── main eval ──

def run_metrics(mode, device="cuda"):
    import torch

    pred_dir = os.path.join(OUTPUT_BASE, mode)
    meta_path = os.path.join(pred_dir, "eval_meta.json")
    if not os.path.exists(meta_path):
        print(f"[SKIP] {mode}: no eval_meta.json")
        return None

    with open(meta_path) as f:
        pairs = json.load(f)
    print(f"\n=== {mode} ({len(pairs)} pairs) ===")

    all_rows = []
    pred_images = []
    gt_images = []

    for i, pair in enumerate(pairs):
        pred_pil = load_rgba_as_rgb(pair["pred_path"])
        gt_pil = load_rgba_as_rgb(pair["gt_path"])
        pred_pil = resize_to_match(pred_pil, gt_pil)

        pred_np = np.array(pred_pil)
        gt_np = np.array(gt_pil)

        psnr = compute_psnr(pred_np, gt_np)
        ssim = compute_ssim(pred_np, gt_np)
        lpips_val = compute_lpips_score(pred_pil, gt_pil, device)
        clip_i = compute_clip_i(pred_pil, gt_pil, device)
        dino = compute_dino_sim(pred_pil, gt_pil, device)

        all_rows.append({
            "pair_id": pair["pair_id"],
            "obj_name": pair["obj_name"],
            "angle_idx": pair["angle_idx"],
            "psnr": round(psnr, 4),
            "ssim": round(ssim, 4),
            "lpips": round(lpips_val, 4),
            "clip_i": round(clip_i, 4),
            "dino": round(dino, 4),
        })

        pred_images.append(pred_pil)
        gt_images.append(gt_pil)

        if (i + 1) % 50 == 0 or (i + 1) == len(pairs):
            avg_psnr = np.mean([r["psnr"] for r in all_rows])
            avg_ssim = np.mean([r["ssim"] for r in all_rows])
            print(f"  [{i+1}/{len(pairs)}] PSNR={avg_psnr:.2f} SSIM={avg_ssim:.4f}")

    # FID
    print(f"  Computing FID (InceptionV3)...")
    pred_feats = extract_inception_features(pred_images, device)
    gt_feats = extract_inception_features(gt_images, device)
    fid_val = compute_fid(pred_feats, gt_feats)
    print(f"  {mode} FID = {fid_val:.2f}")

    # Aggregate
    os.makedirs(METRICS_DIR, exist_ok=True)

    avg = {
        "mode": mode,
        "num_pairs": len(all_rows),
        "psnr": round(float(np.mean([r["psnr"] for r in all_rows])), 4),
        "ssim": round(float(np.mean([r["ssim"] for r in all_rows])), 4),
        "lpips": round(float(np.mean([r["lpips"] for r in all_rows])), 4),
        "clip_i": round(float(np.mean([r["clip_i"] for r in all_rows])), 4),
        "dino": round(float(np.mean([r["dino"] for r in all_rows])), 4),
        "fid": round(fid_val, 2),
    }

    # Per-angle summary
    angle_summary = {}
    for a in range(8):
        a_rows = [r for r in all_rows if r["angle_idx"] == a]
        if a_rows:
            angle_summary[f"angle_{a:02d}"] = {
                "psnr": round(float(np.mean([r["psnr"] for r in a_rows])), 4),
                "ssim": round(float(np.mean([r["ssim"] for r in a_rows])), 4),
                "lpips": round(float(np.mean([r["lpips"] for r in a_rows])), 4),
                "clip_i": round(float(np.mean([r["clip_i"] for r in a_rows])), 4),
                "dino": round(float(np.mean([r["dino"] for r in a_rows])), 4),
            }
    avg["per_angle"] = angle_summary

    summary_path = os.path.join(METRICS_DIR, f"{mode}_summary.json")
    with open(summary_path, "w") as f:
        json.dump(avg, f, indent=2)

    per_pair_path = os.path.join(METRICS_DIR, f"{mode}_per_pair.csv")
    with open(per_pair_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "pair_id", "obj_name", "angle_idx", "psnr", "ssim", "lpips", "clip_i", "dino"
        ])
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\n[{mode}] Summary:")
    for k in ["psnr", "ssim", "lpips", "clip_i", "dino", "fid"]:
        print(f"  {k}: {avg[k]}")
    print(f"  Saved: {summary_path}")

    return avg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["base", "ours", "fal"], default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.all:
        modes = ["base", "ours", "fal"]
    elif args.mode:
        modes = [args.mode]
    else:
        parser.error("Specify --mode or --all")

    all_summaries = []
    for mode in modes:
        s = run_metrics(mode, device=args.device)
        if s:
            all_summaries.append(s)

    if len(all_summaries) > 1:
        combined_path = os.path.join(METRICS_DIR, "combined_summary.json")
        with open(combined_path, "w") as f:
            json.dump(all_summaries, f, indent=2)
        print(f"\nCombined summary: {combined_path}")

        print("\n" + "=" * 90)
        print("SpatialEdit-Bench Rotate — Overall Metrics")
        print("=" * 90)
        print(f"{'Method':<8} {'PSNR ↑':>8} {'SSIM ↑':>8} {'LPIPS ↓':>8} "
              f"{'CLIP-I ↑':>9} {'DINO ↑':>8} {'FID ↓':>8}")
        print("-" * 90)
        for s in all_summaries:
            print(f"{s['mode']:<8} {s['psnr']:>8.4f} {s['ssim']:>8.4f} {s['lpips']:>8.4f} "
                  f"{s['clip_i']:>9.4f} {s['dino']:>8.4f} {s['fid']:>8.2f}")
        print("=" * 90)

        # Per-angle PSNR breakdown
        print(f"\n{'Angle':<10}", end="")
        for s in all_summaries:
            print(f" {s['mode']:>10}", end="")
        print()
        print("-" * (10 + 11 * len(all_summaries)))
        for a in range(8):
            key = f"angle_{a:02d}"
            print(f"{key:<10}", end="")
            for s in all_summaries:
                pa = s.get("per_angle", {}).get(key, {})
                val = pa.get("psnr", float("nan"))
                print(f" {val:>10.2f}", end="")
            print()
