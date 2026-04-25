"""
Benchmark inference: run rotation editing on ObjectMover-Benchmark images.

For each of the first 50 images, generate 7 rotated views (45° to 315°)
using the same prompt format as training:
  "Rotate this object from front view to {target_view_name}."

Usage:
    python eval_inference_benchmark.py --mode ours --device cuda
    python eval_inference_benchmark.py --mode base --device cuda
    python eval_inference_benchmark.py --mode fal  --device cuda
"""
import os, sys, json, argparse, time, shutil
from pathlib import Path
from PIL import Image
import torch
from safetensors.torch import load_file

sys.path.insert(0, "/aaaidata/zhangqisong/DiffSynth-Studio")
from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig

# ── paths ──
MODEL_ROOT = "/data/wuwenzhuo/Qwen-Image-Edit-2511"
BENCHMARK_DIR_B = "/huggingface/dataset_hub/ObjectMover-Benchmark/objectmoverB/images_resize"
BENCHMARK_DIR_A = "/huggingface/dataset_hub/ObjectMover-Benchmark/objectmoverA/images_resize"
OUR_LORA = (
    "/aaaidata/zhangqisong/DiffSynth-Studio/output"
    "/rotation8_rgb_baseline_rank32/epoch_0030/lora.safetensors"
)
FAL_LORA = (
    "/data/wuwenzhuo/Qwen-Image-Edit-2511-Multiple-Angles-LoRA"
    "/qwen-image-edit-2511-multiple-angles-lora.safetensors"
)
OUTPUT_BASE_B = "/aaaidata/zhangqisong/DiffSynth-Studio/output/eval_benchmark"
OUTPUT_BASE_A = "/aaaidata/zhangqisong/DiffSynth-Studio/output/eval_benchmark_a"

NUM_STEPS = 30
SEED = 42

# ── angle → view name mapping (matches training data) ──
ANGLE_VIEW_MAP = {
    45:  "front-right view",
    90:  "right side view",
    135: "back-right view",
    180: "back view",
    225: "back-left view",
    270: "left side view",
    315: "front-left view",
}


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


def run_benchmark_inference(pipe, output_dir, benchmark_dir, num_objects=30, seed=SEED, num_steps=NUM_STEPS):
    """Run inference on first N benchmark images × 7 target angles."""
    os.makedirs(output_dir, exist_ok=True)
    benchmark = Path(benchmark_dir)

    # collect first 50 images
    all_imgs = sorted([f for f in os.listdir(benchmark) if f.endswith(".png")])[:num_objects]
    total_pairs = len(all_imgs) * len(ANGLE_VIEW_MAP)
    print(f"Benchmark: {len(all_imgs)} objects × {len(ANGLE_VIEW_MAP)} angles = {total_pairs} pairs")

    results = []
    t0 = time.time()
    done = 0

    for img_name in all_imgs:
        obj_id = img_name.replace(".png", "")  # e.g. "test_001"
        obj_dir = os.path.join(output_dir, obj_id)
        os.makedirs(obj_dir, exist_ok=True)

        # copy source as yaw000 reference
        src_path = benchmark / img_name
        src_img = Image.open(src_path).convert("RGB")
        shutil.copy2(src_path, os.path.join(obj_dir, "yaw000_source.png"))

        for angle_deg, view_name in ANGLE_VIEW_MAP.items():
            instruction = f"Rotate this object from front view to {view_name}."

            output = pipe(
                prompt=instruction,
                edit_image=[src_img],
                seed=seed,
                num_inference_steps=num_steps,
                height=src_img.size[1],
                width=src_img.size[0],
                edit_image_auto_resize=True,
                zero_cond_t=True,
            )

            if isinstance(output, (list, tuple)):
                out_img = output[0]
            else:
                out_img = output

            out_name = f"yaw{angle_deg:03d}.png"
            out_img.save(os.path.join(obj_dir, out_name))

            done += 1
            elapsed = time.time() - t0
            avg = elapsed / done
            eta = avg * (total_pairs - done)
            print(f"[{done}/{total_pairs}] {obj_id}/yaw{angle_deg:03d} saved  "
                  f"({avg:.1f}s/img, ETA {eta:.0f}s)")

            results.append({
                "obj_id": obj_id,
                "source_image": img_name,
                "angle_deg": angle_deg,
                "view_name": view_name,
                "instruction": instruction,
                "pred_image": f"{obj_id}/{out_name}",
            })

    meta_path = os.path.join(output_dir, "benchmark_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Done. {len(results)} images saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["base", "ours", "fal"], required=True)
    parser.add_argument("--dataset", choices=["A", "B"], default="B",
                        help="ObjectMover-A (30 objects, has GT) or B (50 objects, no GT)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epoch", type=int, default=30)
    parser.add_argument("--num_steps", type=int, default=NUM_STEPS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--num_objects", type=int, default=None,
                        help="Override number of objects (default: 30 for A, 50 for B)")
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    # resolve dataset-specific defaults
    if args.dataset == "A":
        benchmark_dir = BENCHMARK_DIR_A
        default_num = 30
        default_output_base = OUTPUT_BASE_A
    else:
        benchmark_dir = BENCHMARK_DIR_B
        default_num = 50
        default_output_base = OUTPUT_BASE_B

    num_objects = args.num_objects if args.num_objects is not None else default_num

    if args.output_dir:
        out_dir = args.output_dir
    else:
        out_dir = os.path.join(default_output_base, args.mode)

    print(f"Mode: {args.mode}")
    print(f"Dataset: ObjectMover-{args.dataset}")
    print(f"Benchmark dir: {benchmark_dir}")
    print(f"Num objects: {num_objects}")
    print(f"Output: {out_dir}")
    print(f"Loading pipeline...")

    pipe = load_pipeline(device=args.device)

    if args.mode == "ours":
        lora_path = OUR_LORA.replace("epoch_0030", f"epoch_{args.epoch:04d}")
        print(f"Loading our LoRA: {lora_path}")
        load_lora_into_pipe(pipe, lora_path, mode="ours")
    elif args.mode == "fal":
        print(f"Loading fal LoRA: {FAL_LORA}")
        load_lora_into_pipe(pipe, FAL_LORA, mode="fal")
    else:
        print("Using base model (no LoRA)")

    run_benchmark_inference(pipe, out_dir, benchmark_dir, num_objects=num_objects,
                            seed=args.seed, num_steps=args.num_steps)

