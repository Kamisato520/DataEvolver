"""
Stage 3: Image-to-3D — Generate 3D meshes using Hunyuan3D-2.1.
Reads pipeline/data/images/*.png and writes pipeline/data/meshes_raw/{id}.glb.
In the standard stage1-5 pipeline, these raw meshes are then promoted directly
to pipeline/data/meshes/ for downstream scene rendering. Stage 3.5 is no longer
part of the mainline flow.

Two-stage pipeline:
  1. Shape: Hunyuan3DDiTFlowMatchingPipeline → geometry .obj
  2. Paint: Hunyuan3DPaintPipeline → textured .glb (UV + PBR textures)

Use --shape-only to skip paint (debug / fallback to shape-only GLB).

Supports --ids for parallel multi-GPU runs (each worker handles a subset).

Prerequisites:
  cd /aaaidata/zhangqisong/data_build
  git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git
  cd Hunyuan3D-2.1 && pip install -r requirements.txt

  # Compile custom_rasterizer (CUDA extension for paint):
  export PATH=/usr/local/cuda-12.1/bin:$PATH
  export CUDA_HOME=/usr/local/cuda-12.1
  export TORCH_CUDA_ARCH_LIST="8.0"
  cd hy3dpaint/custom_rasterizer && pip install -e .

  # Compile DifferentiableRenderer (C++ pybind11 extension):
  cd hy3dpaint/DifferentiableRenderer && bash compile_mesh_painter.sh
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
import torch

HUNYUAN3D_REPO = "/aaaidata/zhangqisong/data_build/Hunyuan3D-2.1"
MODEL_HUB = "/huggingface/model_hub/Hunyuan3D-2.1"
# Paint model: hunyuan3d-paintpbr-v2-1 lives under the model hub root
# multiview_utils expects parent dir; it appends "hunyuan3d-paintpbr-v2-1" itself
PAINT_MODEL_HUB = "/huggingface/model_hub/Hunyuan3D-2.1"
DINO_MODEL_PATH = "/huggingface/model_hub/dinov2-giant"
REALESRGAN_CKPT = "/aaaidata/zhangqisong/data_build/Hunyuan3D-2.1/hy3dpaint/ckpt/RealESRGAN_x4plus.pth"
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
IMAGES_DIR = os.path.join(DATA_DIR, "images")
IMAGES_RGBA_DIR = os.path.join(DATA_DIR, "images_rgba")
MESHES_DIR = os.path.join(DATA_DIR, "meshes_raw")  # Stage 4 consumes meshes/ after raw->canonical sync.
PROMPTS_PATH = os.path.join(DATA_DIR, "prompts.json")


def resolve_rgb_images_dir(images_dir=None, rgb_images_dir=None):
    """Pick the RGB directory paired with an RGBA input directory."""
    if rgb_images_dir:
        return rgb_images_dir
    if not images_dir or images_dir == IMAGES_DIR:
        return IMAGES_DIR
    candidate = Path(images_dir)
    if candidate.name == "images_rgba":
        sibling = candidate.parent / "images"
        if sibling.exists():
            return str(sibling)
    return IMAGES_DIR


def check_model_ready():
    """Check if Hunyuan3D-2.1 weights are fully downloaded (no .aria2 files)."""
    aria2_files = []
    for root, _, files in os.walk(MODEL_HUB):
        for f in files:
            if f.endswith(".aria2"):
                aria2_files.append(os.path.join(root, f))
    if aria2_files:
        print(f"[Stage 3] ERROR: Model still downloading ({len(aria2_files)} .aria2 files remain):")
        for f in aria2_files[:5]:
            print(f"    {f}")
        print("[Stage 3] Wait for download to complete or resume with huggingface-cli")
        return False
    return True


def setup_paths():
    """Add Hunyuan3D-2.1 sub-packages to Python path."""
    for subdir in ["hy3dshape", "hy3dpaint", "."]:
        path = os.path.join(HUNYUAN3D_REPO, subdir)
        if os.path.exists(path) and path not in sys.path:
            sys.path.insert(0, path)

    # DifferentiableRenderer must be on sys.path for MeshRender import
    diffrender_path = os.path.join(HUNYUAN3D_REPO, "hy3dpaint", "DifferentiableRenderer")
    if os.path.exists(diffrender_path) and diffrender_path not in sys.path:
        sys.path.insert(0, diffrender_path)

    # custom_rasterizer package dir (so `import custom_rasterizer_kernel` works)
    # The .so is in the package dir alongside the __init__.py
    cr_path = os.path.join(HUNYUAN3D_REPO, "hy3dpaint", "custom_rasterizer")
    if os.path.exists(cr_path) and cr_path not in sys.path:
        sys.path.insert(0, cr_path)


def build_paint_pipeline(device):
    """Load Hunyuan3DPaintPipeline with local model weights.

    Key path overrides (all local, no HF download):
      - multiview_pretrained_path → /huggingface/model_hub/Hunyuan3D-2.1
        (multiview_utils.py will resolve → .../hunyuan3d-paintpbr-v2-1/)
      - dino_ckpt_path → /huggingface/model_hub/dinov2-giant
      - realesrgan_ckpt_path → hy3dpaint/ckpt/RealESRGAN_x4plus.pth
    """
    from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig

    paint_cfg = Hunyuan3DPaintConfig(max_num_view=6, resolution=512)

    # Override ALL paths to use local storage (avoid HF download)
    paint_cfg.multiview_pretrained_path = PAINT_MODEL_HUB
    paint_cfg.dino_ckpt_path = DINO_MODEL_PATH
    paint_cfg.realesrgan_ckpt_path = REALESRGAN_CKPT
    paint_cfg.device = device

    # multiview_cfg_path is relative in original code ("hy3dpaint/cfgs/..."),
    # patch to absolute path so it works regardless of cwd
    paint_cfg.multiview_cfg_path = os.path.join(
        HUNYUAN3D_REPO, "hy3dpaint", "cfgs", "hunyuan-paint-pbr.yaml"
    )

    print(f"[Stage 3] Loading paint pipeline from {PAINT_MODEL_HUB} on {device}...")
    paint_pipe = Hunyuan3DPaintPipeline(paint_cfg)
    print("[Stage 3] Paint pipeline loaded.")
    return paint_pipe


def generate_meshes(obj_ids, device="cuda:0", skip_existing=True, shape_only=False,
                    images_dir=None, rgb_images_dir=None, output_dir=None, seed_base=42):
    """Generate textured 3D meshes from images using Hunyuan3D-2.1.

    Pipeline:
        image → rembg(RGBA) → shape pipeline → temp .obj
                                              ↓ (unless --shape-only)
                            paint pipeline (projects image color onto mesh)
                                              ↓
                                    textured .glb (UV + PBR)
    """
    output_dir = output_dir or MESHES_DIR
    os.makedirs(output_dir, exist_ok=True)
    setup_paths()

    # Auto-detect input directory: prefer pre-segmented RGBA if available
    if images_dir is None:
        if os.path.isdir(IMAGES_RGBA_DIR):
            images_dir = IMAGES_RGBA_DIR
        else:
            images_dir = IMAGES_DIR
    use_precut_rgba = (images_dir != IMAGES_DIR)
    rgb_images_dir = resolve_rgb_images_dir(images_dir=images_dir, rgb_images_dir=rgb_images_dir)
    if use_precut_rgba:
        print(f"[Stage 3] Using pre-segmented RGBA images from {images_dir}")
        print(f"[Stage 3] Using paired RGB images from {rgb_images_dir}")

    if not check_model_ready():
        sys.exit(1)

    # Filter already processed
    if skip_existing:
        remaining = [oid for oid in obj_ids
                     if not os.path.exists(os.path.join(output_dir, f"{oid}.glb"))]
        if len(remaining) < len(obj_ids):
            print(f"[Stage 3] Skipping {len(obj_ids) - len(remaining)} existing meshes")
        obj_ids = remaining

    if not obj_ids:
        print("[Stage 3] All meshes already exist, skipping model load")
        return

    print(f"[Stage 3] Loading shape pipeline from {MODEL_HUB} on {device}...")

    try:
        from PIL import Image
        from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
        if not use_precut_rgba:
            from hy3dshape.rembg import BackgroundRemover
    except ImportError as e:
        print(f"[ERROR] Missing dependency: {e}")
        print("Run: cd Hunyuan3D-2.1 && pip install -r requirements.txt")
        sys.exit(1)

    # Load shape pipeline (DiT, ~12GB VRAM)
    try:
        shape_pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            MODEL_HUB,
            torch_dtype=torch.float16,
            device=device,
        )
    except TypeError:
        shape_pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(MODEL_HUB)
        if hasattr(shape_pipe, "to"):
            shape_pipe = shape_pipe.to(device)
        if hasattr(shape_pipe, "half"):
            shape_pipe = shape_pipe.half()

    rembg = BackgroundRemover() if not use_precut_rgba else None
    print(f"[Stage 3] Shape pipeline loaded on {device}. Processing {len(obj_ids)} objects...")

    # Load paint pipeline once (saves ~29GB VRAM re-alloc per object)
    paint_pipe = None
    if not shape_only:
        try:
            paint_pipe = build_paint_pipeline(device)
        except Exception as e:
            print(f"[WARN] Failed to load paint pipeline: {e}")
            print("[WARN] Falling back to shape-only mode (no UV texture).")
            import traceback
            traceback.print_exc()

    for i, obj_id in enumerate(obj_ids):
        img_path_input = os.path.join(images_dir, f"{obj_id}.png")
        img_path_rgb = os.path.join(rgb_images_dir, f"{obj_id}.png")
        out_path = os.path.join(output_dir, f"{obj_id}.glb")

        if not os.path.exists(img_path_input):
            print(f"[WARN] Image not found: {img_path_input}, skipping")
            continue

        print(f"[Stage 3][{device}] ({i+1}/{len(obj_ids)}) Processing: {obj_id} ...")

        try:
            if use_precut_rgba:
                image_rgba = Image.open(img_path_input).convert("RGBA")
                # Validate alpha channel is meaningful
                alpha = image_rgba.split()[-1]
                extrema = alpha.getextrema()
                if extrema == (0, 0) or extrema == (255, 255):
                    print(f"[WARN] {obj_id}: RGBA alpha invalid (extrema={extrema}), fallback to rembg")
                    image = Image.open(img_path_rgb).convert("RGB")
                    from hy3dshape.rembg import BackgroundRemover
                    _rembg = BackgroundRemover()
                    image_rgba = _rembg(image)
                    del _rembg
            else:
                image = Image.open(img_path_input).convert("RGB")
                image_rgba = rembg(image)

            # ── Stage 3a: Shape generation ─────────────────────────────
            with torch.no_grad():
                outputs = shape_pipe(
                    image=image_rgba,
                    num_inference_steps=50,
                    guidance_scale=5.0,
                    seed=seed_base + i,
                )
            mesh = outputs[0]

            if paint_pipe is None:
                # shape-only fallback: export directly to GLB (no UV texture)
                mesh.export(out_path)
                size_kb = os.path.getsize(out_path) / 1024
                print(f"    [shape-only] Saved → {out_path} ({size_kb:.1f} KB)")
                if size_kb < 10:
                    print(f"[WARN] Mesh suspiciously small ({size_kb:.1f} KB)")
                continue

            # ── Stage 3b: Texture painting ────────────────────────────
            with tempfile.TemporaryDirectory() as tmp_dir:
                # Export geometry as .obj (paint pipeline requires .obj input)
                temp_obj = os.path.join(tmp_dir, f"{obj_id}.obj")
                mesh.export(temp_obj)

                temp_textured_obj = os.path.join(tmp_dir, f"{obj_id}_textured.obj")
                try:
                    paint_pipe(
                        mesh_path=temp_obj,
                        image_path=img_path_rgb,      # original RGB for color projection
                        output_mesh_path=temp_textured_obj,
                        use_remesh=True,
                        save_glb=True,
                    )
                    # paint pipeline writes .glb alongside the .obj
                    temp_glb = temp_textured_obj.replace(".obj", ".glb")
                    if not os.path.exists(temp_glb):
                        # some versions name it differently — look for any .glb in tmp
                        glb_candidates = [
                            f for f in os.listdir(tmp_dir) if f.endswith(".glb")
                        ]
                        if glb_candidates:
                            temp_glb = os.path.join(tmp_dir, glb_candidates[0])
                        else:
                            raise FileNotFoundError(
                                f"Paint pipeline did not produce a .glb in {tmp_dir}"
                            )
                    shutil.move(temp_glb, out_path)
                    size_kb = os.path.getsize(out_path) / 1024
                    print(f"    [textured] Saved → {out_path} ({size_kb:.1f} KB)")
                    if size_kb < 50:
                        print(f"[WARN] Textured mesh suspiciously small ({size_kb:.1f} KB)")
                except Exception as paint_err:
                    print(f"[WARN] Paint failed for {obj_id}: {paint_err}")
                    print("[WARN] Falling back to shape-only GLB for this object.")
                    import traceback
                    traceback.print_exc()
                    # fallback: export shape-only .glb so pipeline can continue
                    mesh.export(out_path)
                    size_kb = os.path.getsize(out_path) / 1024
                    print(f"    [shape-only fallback] Saved → {out_path} ({size_kb:.1f} KB)")

        except Exception as e:
            print(f"[WARN] Failed to generate mesh for {obj_id}: {e}")
            import traceback
            traceback.print_exc()

    del shape_pipe
    if rembg is not None:
        del rembg
    if paint_pipe is not None:
        del paint_pipe
    torch.cuda.empty_cache()
    print(f"[Stage 3][{device}] GPU memory released.")


def verify_outputs(obj_ids, output_dir=None):
    """Check all expected meshes exist and have reasonable size."""
    output_dir = output_dir or MESHES_DIR
    missing = []
    for oid in obj_ids:
        path = os.path.join(output_dir, f"{oid}.glb")
        if not os.path.exists(path) or os.path.getsize(path) < 10 * 1024:
            missing.append(oid)

    if missing:
        print(f"[Stage 3] WARN: Missing/small meshes: {missing}")
    else:
        print(f"[Stage 3] ✓ All {len(obj_ids)} meshes generated successfully")
    return len(missing) == 0


def main():
    parser = argparse.ArgumentParser(description="Stage 3: Image-to-3D mesh generation with texture")
    parser.add_argument("--device", default="cuda:0",
                        help="CUDA device (e.g. cuda:0, cuda:1)")
    parser.add_argument("--no-skip", action="store_true",
                        help="Force regeneration even if output .glb already exists")
    parser.add_argument("--ids", default=None,
                        help="Comma-separated list of object IDs to process (e.g. obj_001,obj_002). "
                             "Default: all objects in prompts.json")
    parser.add_argument("--shape-only", action="store_true",
                        help="Skip texture painting, output shape-only GLB (faster, for debugging)")
    parser.add_argument("--images-dir", default=None,
                        help="Override input images directory. "
                             "Default: auto-detect (images_rgba/ if exists, else images/)")
    parser.add_argument("--rgb-images-dir", default=None,
                        help="Optional RGB directory paired with --images-dir when using RGBA inputs")
    parser.add_argument("--output-dir", default=MESHES_DIR,
                        help="Where to write textured/raw GLBs")
    parser.add_argument("--seed-base", type=int, default=42,
                        help="Base seed for deterministic regeneration runs")
    parser.add_argument("--prompts-path", default=PROMPTS_PATH,
                        help="Prompt file used to enumerate valid object IDs")
    args = parser.parse_args()

    with open(args.prompts_path, "r", encoding="utf-8") as f:
        prompts = json.load(f)
    all_ids = [p["id"] for p in prompts]

    if args.ids:
        obj_ids = [x.strip() for x in args.ids.split(",") if x.strip()]
        invalid = [oid for oid in obj_ids if oid not in all_ids]
        if invalid:
            print(f"[WARN] Unknown object IDs: {invalid}")
        obj_ids = [oid for oid in obj_ids if oid in all_ids]
    else:
        obj_ids = all_ids

    mode = "shape-only" if args.shape_only else "shape+paint (textured)"
    print(f"[Stage 3] Mode: {mode}")
    print(f"[Stage 3] Processing {len(obj_ids)} objects on {args.device}: {obj_ids}")

    generate_meshes(
        obj_ids,
        device=args.device,
        skip_existing=not args.no_skip,
        shape_only=args.shape_only,
        images_dir=args.images_dir,
        rgb_images_dir=args.rgb_images_dir,
        output_dir=args.output_dir,
        seed_base=args.seed_base,
    )
    if not verify_outputs(obj_ids, output_dir=args.output_dir):
        sys.exit(1)


if __name__ == "__main__":
    main()
