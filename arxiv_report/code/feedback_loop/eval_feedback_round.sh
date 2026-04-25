#!/bin/bash
set -euo pipefail

CHECKPOINT=""
DATASET_ROOT=""
OUTPUT_DIR=""
WORKDIR="${WORKDIR:-/gemini/platform/public/aigc/aigc_image/zhanghy56_intern/zhangqisong/data-build}"
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
GENERATE_ONLY=0

while [ $# -gt 0 ]; do
  case "$1" in
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --dataset-root) DATASET_ROOT="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --workdir) WORKDIR="$2"; shift 2 ;;
    --repo-root) REPO_ROOT="$2"; shift 2 ;;
    --generate-only) GENERATE_ONLY=1; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [ -z "$CHECKPOINT" ] || [ -z "$DATASET_ROOT" ] || [ -z "$OUTPUT_DIR" ]; then
  echo "Usage: $0 --checkpoint <lora> --dataset-root <root> --output-dir <dir> [--generate-only]" >&2
  exit 2
fi

emit_script() {
  local script
  script=$(cat <<'BASH'
#!/bin/bash
set -euo pipefail

WORKDIR="__WORKDIR__"
REPO_ROOT="__REPO_ROOT__"
CHECKPOINT="__CHECKPOINT__"
DATASET_ROOT="__DATASET_ROOT__"
OUTPUT_DIR="__OUTPUT_DIR__"
DSDIR="${WORKDIR}/DiffSynth-Studio"
RUNTIME_DIR="${OUTPUT_DIR}/runtime"

source "${WORKDIR}/.venv/bin/activate"
export PYTHONPATH="${DSDIR}:${PYTHONPATH:-}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
mkdir -p "$OUTPUT_DIR" "$RUNTIME_DIR"

if [ ! -f "$CHECKPOINT" ]; then
  echo "Missing checkpoint: $CHECKPOINT" >&2
  exit 1
fi
if [ ! -f "${DATASET_ROOT}/pairs/test_pairs.jsonl" ]; then
  echo "Missing test pairs: ${DATASET_ROOT}/pairs/test_pairs.jsonl" >&2
  exit 1
fi

cat > "${RUNTIME_DIR}/eval_testset_inference.py" <<'PY'
import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from safetensors.torch import load_file

workdir = os.environ["WORKDIR"]
sys.path.insert(0, f"{workdir}/DiffSynth-Studio")
from diffsynth.pipelines.qwen_image import ModelConfig, QwenImagePipeline

MODEL_ROOT = os.environ.get(
    "MODEL_ROOT",
    "/gemini/platform/public/aigc/aigc_image/zhanghy56_intern/zhangqisong/Qwen-Image-Edit-2511",
)


def convert_our_lora_keys(state_dict):
    out = {}
    for key, value in state_dict.items():
        new_key = key
        if new_key.startswith("pipe.dit."):
            new_key = new_key[len("pipe.dit."):]
        new_key = new_key.replace(".lora_A.default.", ".lora_A.")
        new_key = new_key.replace(".lora_B.default.", ".lora_B.")
        out[new_key] = value
    return out


def load_pipeline(device):
    return QwenImagePipeline.from_pretrained(
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    rows = [
        json.loads(line)
        for line in (dataset_root / "pairs" / "test_pairs.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    os.makedirs(args.output_dir, exist_ok=True)
    pipe = load_pipeline(args.device)
    pipe.load_lora(pipe.dit, state_dict=convert_our_lora_keys(load_file(args.checkpoint)))

    results = []
    t0 = time.time()
    for index, row in enumerate(rows, start=1):
        src_img = Image.open(dataset_root / row["source_image"]).convert("RGB")
        output = pipe(
            prompt=row["instruction"],
            edit_image=[src_img],
            seed=args.seed,
            num_inference_steps=args.num_steps,
            height=src_img.size[1],
            width=src_img.size[0],
            edit_image_auto_resize=True,
            zero_cond_t=True,
        )
        out_img = output if isinstance(output, Image.Image) else output[0]
        out_name = f"{row['pair_id']}.png"
        out_img.save(Path(args.output_dir) / out_name)
        elapsed = time.time() - t0
        print(f"[{index}/{len(rows)}] {row['pair_id']} ({elapsed / index:.1f}s/img)")
        results.append({
            "pair_id": row["pair_id"],
            "source_image": row["source_image"],
            "target_image": row["target_image"],
            "instruction": row["instruction"],
            "pred_image": out_name,
            "target_rotation_deg": row.get("target_rotation_deg"),
        })

    with (Path(args.output_dir) / "eval_meta.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
PY

echo "=== Test Set inference ==="
TESTSET_INF_BASE="${OUTPUT_DIR}/testset_inference"
CUDA_VISIBLE_DEVICES=${TESTSET_GPU:-0} WORKDIR="$WORKDIR" python "${RUNTIME_DIR}/eval_testset_inference.py" \
  --checkpoint "$CHECKPOINT" \
  --dataset-root "$DATASET_ROOT" \
  --output-dir "${TESTSET_INF_BASE}/ours" \
  --device cuda

echo "=== Test Set metrics ==="
bash "${REPO_ROOT}/scripts/68server/run_eval_metrics.sh" \
  --eval testset \
  --modes ours \
  --eval_inference_base "$TESTSET_INF_BASE" \
  --split_root "$DATASET_ROOT" \
  --output_dir "${OUTPUT_DIR}/testset_metrics"

echo "=== SpatialEdit inference ==="
SPATIAL_INF_DIR="${OUTPUT_DIR}/spatialedit_inference/ours"
CUDA_VISIBLE_DEVICES=${SPATIALEDIT_GPU:-0} python "${REPO_ROOT}/scripts/68server/eval_spatialedit_inference.py" \
  --mode ours \
  --lora_path "$CHECKPOINT" \
  --output_dir "$SPATIAL_INF_DIR" \
  --device cuda

echo "=== SpatialEdit symlink folders ==="
SPATIAL_FOLDERS="${OUTPUT_DIR}/spatialedit_folders/ours"
python - "$SPATIAL_INF_DIR" "$SPATIAL_FOLDERS" <<'PY'
import json
import os
import sys
from pathlib import Path

pred_dir = Path(sys.argv[1])
out_root = Path(sys.argv[2])
meta = json.loads((pred_dir / "eval_meta.json").read_text(encoding="utf-8"))
pred_folder = out_root / "pred"
gt_folder = out_root / "gt"
pred_folder.mkdir(parents=True, exist_ok=True)
gt_folder.mkdir(parents=True, exist_ok=True)
for pair in meta:
    name = pair["pair_id"] + ".png"
    pred_link = pred_folder / name
    gt_link = gt_folder / name
    for link in (pred_link, gt_link):
        if link.exists() or link.is_symlink():
            link.unlink()
    os.symlink(os.path.abspath(pair["pred_path"]), pred_link)
    os.symlink(os.path.abspath(pair["gt_path"]), gt_link)
print(f"Created {len(meta)} symlink pairs under {out_root}")
PY

echo "=== SpatialEdit metrics ==="
mkdir -p "${OUTPUT_DIR}/spatialedit_metrics"
CUDA_VISIBLE_DEVICES=${SPATIALEDIT_METRICS_GPU:-0} python "${DSDIR}/metrics/eval_image_metrics.py" \
  --folder_a "${SPATIAL_FOLDERS}/pred" \
  --folder_b "${SPATIAL_FOLDERS}/gt" \
  --output_csv "${OUTPUT_DIR}/spatialedit_metrics/ours_metrics.csv" \
  --device cuda \
  2>&1 | tee "${OUTPUT_DIR}/spatialedit_metrics/ours_eval.log"
BASH
)
  script=${script//__WORKDIR__/$WORKDIR}
  script=${script//__REPO_ROOT__/$REPO_ROOT}
  script=${script//__CHECKPOINT__/$CHECKPOINT}
  script=${script//__DATASET_ROOT__/$DATASET_ROOT}
  script=${script//__OUTPUT_DIR__/$OUTPUT_DIR}
  printf '%s\n' "$script"
}

if [ "$GENERATE_ONLY" -eq 1 ]; then
  emit_script
else
  emit_script | bash
fi
