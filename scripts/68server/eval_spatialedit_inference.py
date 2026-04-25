"""
SpatialEdit-Bench rotation inference: base / ours / fal.

Usage:
    python eval_spatialedit_inference.py --mode ours --device cuda
    python eval_spatialedit_inference.py --mode base --device cuda
    python eval_spatialedit_inference.py --mode fal  --device cuda
"""
import os, sys, json, argparse, time, glob
from pathlib import Path
from PIL import Image
import torch
from safetensors.torch import load_file

_WORKDIR = "/gemini/platform/public/aigc/aigc_image/zhanghy56_intern/zhangqisong/data-build"
sys.path.insert(0, f"{_WORKDIR}/DiffSynth-Studio")
from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig

MODEL_ROOT = "/gemini/platform/public/aigc/aigc_image/zhanghy56_intern/zhangqisong/Qwen-Image-Edit-2511"
OUR_LORA = f"{_WORKDIR}/DiffSynth-Studio/output/rotation8_bright_clockwise_raw_rank32/epoch_0030/lora.safetensors"
FAL_LORA = f"{_WORKDIR}/Qwen-Image-Edit-2511-Multiple-Angles-LoRA/qwen-image-edit-2511-multiple-angles-lora.safetensors"

BENCH_ROOT = f"{_WORKDIR}/SpatialEdit-Bench"
RESULTS_DIR = f"{BENCH_ROOT}/SpatialEdit_Results/spatialedit/fullset/rotate/en"
OUTPUT_BASE = f"{_WORKDIR}/DiffSynth-Studio/output/eval_spatialedit"

NUM_STEPS = 30
SEED = 42


def convert_our_lora_keys(state_dict):
    new_sd = {}
    for k, v in state_dict.items():
        nk = k
        if nk.startswith("pipe.dit."):
            nk = nk[len("pipe.dit."):]
        nk = nk.replace(".lora_A.default.", ".lora_A.")
        nk = nk.replace(".lora_B.default.", ".lora_B.")
        new_sd[nk] = v
    return new_sd


def convert_fal_lora_keys(state_dict):
    new_sd = {}
    for k, v in state_dict.items():
        nk = k
        if nk.startswith("transformer."):
            nk = nk[len("transformer."):]
        new_sd[nk] = v
    return new_sd


def load_pipeline(device="cuda"):
    pipe = QwenImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=[
            ModelConfig(
                model_id="Qwen/Qwen-Image-Edit-2511",
                origin_file_pattern=f"{MODEL_ROOT}/transformer/diffusion_pytorch_model*.safetensors",
            ),
            ModelConfig(
                model_id="Qwen/Qwen-Image",
                origin_file_pattern=f"{MODEL_ROOT}/text_encoder/model*.safetensors",
            ),
            ModelConfig(
                model_id="Qwen/Qwen-Image",
                origin_file_pattern=f"{MODEL_ROOT}/vae/diffusion_pytorch_model.safetensors",
            ),
        ],
        tokenizer_config=ModelConfig(f"{MODEL_ROOT}/tokenizer"),
        processor_config=ModelConfig(f"{MODEL_ROOT}/processor"),
    )
    return pipe


def load_lora_into_pipe(pipe, lora_path, mode="ours"):
    sd = load_file(lora_path)
    if mode == "ours":
        sd = convert_our_lora_keys(sd)
    elif mode == "fal":
        sd = convert_fal_lora_keys(sd)
    pipe.load_lora(pipe.dit, state_dict=sd)


def collect_pairs():
    """Collect all (obj_folder, angle_idx, src_path, prompt, gt_path) tuples."""
    pairs = []
    obj_folders = sorted([
        os.path.join(RESULTS_DIR, d)
        for d in os.listdir(RESULTS_DIR)
        if os.path.isdir(os.path.join(RESULTS_DIR, d))
    ])
    for obj_folder in obj_folders:
        obj_name = os.path.basename(obj_folder)
        for angle_idx in range(8):
            prefix = f"{angle_idx:02d}"
            src_path = os.path.join(obj_folder, f"{prefix}_src.png")
            gt_path = os.path.join(obj_folder, f"{prefix}.png")
            prompt_path = os.path.join(obj_folder, f"{prefix}_prompt.txt")

            if not os.path.exists(src_path) or not os.path.exists(prompt_path):
                continue

            with open(prompt_path, "r", encoding="utf-8") as f:
                prompt = f.read().strip()

            pairs.append({
                "obj_name": obj_name,
                "angle_idx": angle_idx,
                "src_path": src_path,
                "gt_path": gt_path,
                "prompt": prompt,
                "pair_id": f"{obj_name}_angle{angle_idx:02d}",
            })
    return pairs


def run_inference(pipe, pairs, output_dir, seed=SEED, num_steps=NUM_STEPS):
    os.makedirs(output_dir, exist_ok=True)
    print(f"Total pairs: {len(pairs)}")
    results = []
    skipped = 0
    t0 = time.time()

    for i, pair in enumerate(pairs):
        out_path = os.path.join(output_dir, f"{pair['pair_id']}.png")
        if os.path.exists(out_path):
            skipped += 1
            results.append({
                "pair_id": pair["pair_id"],
                "obj_name": pair["obj_name"],
                "angle_idx": pair["angle_idx"],
                "prompt": pair["prompt"],
                "src_path": pair["src_path"],
                "gt_path": pair["gt_path"],
                "pred_path": out_path,
            })
            continue

        if skipped and not results[-1].get("_logged"):
            print(f"Skipped {skipped} existing images, resuming from [{i+1}/{len(pairs)}]")

        src_img = Image.open(pair["src_path"]).convert("RGB")

        output = pipe(
            prompt=pair["prompt"],
            edit_image=[src_img],
            seed=seed,
            num_inference_steps=num_steps,
            height=src_img.size[1],
            width=src_img.size[0],
            edit_image_auto_resize=True,
        )

        if isinstance(output, Image.Image):
            out_img = output
        elif isinstance(output, (list, tuple)):
            out_img = output[0]
        else:
            out_img = output

        out_path = os.path.join(output_dir, f"{pair['pair_id']}.png")
        out_img.save(out_path)

        elapsed = time.time() - t0
        avg = elapsed / (i + 1)
        eta = avg * (len(pairs) - i - 1)
        print(f"[{i+1}/{len(pairs)}] {pair['pair_id']}  ({avg:.1f}s/img, ETA {eta:.0f}s)")

        results.append({
            "pair_id": pair["pair_id"],
            "obj_name": pair["obj_name"],
            "angle_idx": pair["angle_idx"],
            "prompt": pair["prompt"],
            "src_path": pair["src_path"],
            "gt_path": pair["gt_path"],
            "pred_path": out_path,
        })

    meta_path = os.path.join(output_dir, "eval_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Done. {len(results)} images saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["base", "ours", "fal"], required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epoch", type=int, default=30)
    parser.add_argument("--num_steps", type=int, default=NUM_STEPS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--lora_path", default=None, help="Override LoRA path (for custom checkpoints)")
    args = parser.parse_args()

    if args.output_dir:
        out_dir = args.output_dir
    else:
        out_dir = os.path.join(OUTPUT_BASE, args.mode)

    print(f"Mode: {args.mode}")
    print(f"Output: {out_dir}")

    pairs = collect_pairs()
    print(f"Collected {len(pairs)} pairs from {RESULTS_DIR}")

    print("Loading pipeline...")
    pipe = load_pipeline(device=args.device)

    if args.mode == "ours":
        if args.lora_path:
            lora_path = args.lora_path
        else:
            lora_path = OUR_LORA.replace("epoch_0030", f"epoch_{args.epoch:04d}")
        if not os.path.exists(lora_path):
            print(f"[ERROR] LoRA not found: {lora_path}")
            sys.exit(1)
        print(f"Loading our LoRA: {lora_path}")
        load_lora_into_pipe(pipe, lora_path, mode="ours")
    elif args.mode == "fal":
        if not os.path.exists(FAL_LORA):
            print(f"[ERROR] fal LoRA not found: {FAL_LORA}")
            sys.exit(1)
        print(f"Loading fal LoRA: {FAL_LORA}")
        load_lora_into_pipe(pipe, FAL_LORA, mode="fal")
    else:
        print("Using base model (no LoRA)")

    run_inference(pipe, pairs, out_dir, seed=args.seed, num_steps=args.num_steps)
