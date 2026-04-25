"""
Eval inference for rotation editing: base model / our LoRA / fal community LoRA.

Usage:
    python eval_inference.py --mode ours
    python eval_inference.py --mode base
    python eval_inference.py --mode fal
"""
import os, sys, json, argparse, time
from pathlib import Path
from PIL import Image
import torch
from safetensors.torch import load_file

# ── 68 服务器路径 ──
_WORKDIR = "/gemini/platform/public/aigc/aigc_image/zhanghy56_intern/zhangqisong/data-build"

sys.path.insert(0, f"{_WORKDIR}/DiffSynth-Studio")
from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig

# ── paths ──
MODEL_ROOT = "/gemini/platform/public/aigc/aigc_image/zhanghy56_intern/zhangqisong/Qwen-Image-Edit-2511"

# 默认用 bbox split；如需 RGB split 改为对应目录名
SPLIT_ROOT = f"{_WORKDIR}/dataset_scene_v7_full50_rotation8_trainready_front2others_splitobj_seed42_bboxmask_bright_final_20260416"

OUR_LORA = f"{_WORKDIR}/DiffSynth-Studio/output/rotation8_bright_clockwise_raw_rank32/epoch_0030/lora.safetensors"

# fal 社区 LoRA（如不做对比可不传，此处保留占位）
FAL_LORA = f"{_WORKDIR}/Qwen-Image-Edit-2511-Multiple-Angles-LoRA/qwen-image-edit-2511-multiple-angles-lora.safetensors"

OUTPUT_BASE = f"{_WORKDIR}/DiffSynth-Studio/output/eval_inference"

NUM_STEPS = 30
SEED = 42


def convert_our_lora_keys(state_dict):
    """Convert our training checkpoint keys to standard DiffSynth format.

    Our keys:  pipe.dit.transformer_blocks.0.attn.to_q.lora_A.default.weight
    Expected:  transformer_blocks.0.attn.to_q.lora_A.weight
    """
    new_sd = {}
    for k, v in state_dict.items():
        nk = k
        if nk.startswith("pipe.dit."):
            nk = nk[len("pipe.dit."):]
        nk = nk.replace(".lora_A.default.", ".lora_A.")
        nk = nk.replace(".lora_B.default.", ".lora_B.")
        new_sd[nk] = v
    return new_sd

def load_pipeline(device="cuda"):
    """Load base Qwen-Image-Edit-2511 pipeline."""
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


def convert_fal_lora_keys(state_dict):
    """Convert fal community LoRA keys (diffusers format) to DiffSynth format.

    Fal keys:  transformer.transformer_blocks.0.attn.to_q.lora_A.weight
    Expected:  transformer_blocks.0.attn.to_q.lora_A.weight
    """
    new_sd = {}
    for k, v in state_dict.items():
        nk = k
        if nk.startswith("transformer."):
            nk = nk[len("transformer."):]
        new_sd[nk] = v
    return new_sd


def load_lora_into_pipe(pipe, lora_path, mode="ours"):
    """Load LoRA weights into the pipeline's dit module."""
    sd = load_file(lora_path)
    if mode == "ours":
        sd = convert_our_lora_keys(sd)
    elif mode == "fal":
        sd = convert_fal_lora_keys(sd)
    pipe.load_lora(pipe.dit, state_dict=sd)


def run_inference(pipe, test_jsonl, output_dir, seed=SEED, num_steps=NUM_STEPS):
    """Run inference on all test pairs and save results."""
    os.makedirs(output_dir, exist_ok=True)
    split_root = Path(SPLIT_ROOT)

    with open(test_jsonl, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    print(f"Total test pairs: {len(rows)}")
    results = []
    t0 = time.time()

    for i, row in enumerate(rows):
        pair_id = row["pair_id"]
        src_path = split_root / row["source_image"]
        instruction = row["instruction"]

        src_img = Image.open(src_path).convert("RGB")

        output = pipe(
            prompt=instruction,
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

        out_path = os.path.join(output_dir, f"{pair_id}.png")
        out_img.save(out_path)

        elapsed = time.time() - t0
        avg = elapsed / (i + 1)
        eta = avg * (len(rows) - i - 1)
        print(f"[{i+1}/{len(rows)}] {pair_id} saved  ({avg:.1f}s/img, ETA {eta:.0f}s)")

        results.append({
            "pair_id": pair_id,
            "source_image": row["source_image"],
            "target_image": row["target_image"],
            "instruction": instruction,
            "pred_image": f"{pair_id}.png",
            "target_rotation_deg": row.get("target_rotation_deg"),
        })

    meta_path = os.path.join(output_dir, "eval_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Done. {len(results)} images saved to {output_dir}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["base", "ours", "fal"], required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epoch", type=int, default=30, help="Our LoRA epoch to use")
    parser.add_argument("--num_steps", type=int, default=NUM_STEPS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    test_jsonl = os.path.join(SPLIT_ROOT, "pairs", "test_pairs.jsonl")

    if args.output_dir:
        out_dir = args.output_dir
    else:
        out_dir = os.path.join(OUTPUT_BASE, args.mode)

    print(f"Mode: {args.mode}")
    print(f"Output: {out_dir}")
    print(f"Loading pipeline...")

    pipe = load_pipeline(device=args.device)

    if args.mode == "ours":
        lora_path = OUR_LORA.replace("epoch_0030", f"epoch_{args.epoch:04d}")
        if not os.path.exists(lora_path):
            print(f"[ERROR] LoRA not found: {lora_path}")
            sys.exit(1)
        print(f"Loading our LoRA: {lora_path}")
        load_lora_into_pipe(pipe, lora_path, mode="ours")
    elif args.mode == "fal":
        if not os.path.exists(FAL_LORA):
            print(f"[ERROR] fal LoRA not found: {FAL_LORA}")
            print("Please transfer it first or skip fal mode.")
            sys.exit(1)
        print(f"Loading fal LoRA: {FAL_LORA}")
        load_lora_into_pipe(pipe, FAL_LORA, mode="fal")
    else:
        print("Using base model (no LoRA)")

    run_inference(pipe, test_jsonl, out_dir, seed=args.seed, num_steps=args.num_steps)

