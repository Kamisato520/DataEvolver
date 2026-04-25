#!/bin/bash
set -euo pipefail

DATASET_ROOT=""
OUTPUT_DIR=""
WORKDIR="${WORKDIR:-/gemini/platform/public/aigc/aigc_image/zhanghy56_intern/zhangqisong/data-build}"
GENERATE_ONLY=0

while [ $# -gt 0 ]; do
  case "$1" in
    --dataset-root) DATASET_ROOT="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --workdir) WORKDIR="$2"; shift 2 ;;
    --generate-only) GENERATE_ONLY=1; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [ -z "$DATASET_ROOT" ] || [ -z "$OUTPUT_DIR" ]; then
  echo "Usage: $0 --dataset-root <root> --output-dir <dir> [--generate-only]" >&2
  exit 2
fi

emit_script() {
  local script
  script=$(cat <<'BASH'
#!/bin/bash
set -euo pipefail

export CUDA_LAUNCH_BLOCKING=1
export PYTHONFAULTHANDLER=1
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export TORCH_DISTRIBUTED_DEBUG=${TORCH_DISTRIBUTED_DEBUG:-OFF}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE:-0}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}

WORKDIR="__WORKDIR__"
DATASET_ROOT="__DATASET_ROOT__"
OUTPUT_DIR="__OUTPUT_DIR__"
NUM_PROCESSES=${NUM_PROCESSES:-8}

source "${WORKDIR}/.venv/bin/activate"
export PYTHONPATH="${WORKDIR}/DiffSynth-Studio:${PYTHONPATH:-}"
cd "${WORKDIR}/DiffSynth-Studio"

ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-${WORKDIR}/DiffSynth-Studio/accelerate_config_8gpu.yaml}"
TRAIN_METADATA="${DATASET_ROOT}/pairs/train_pairs.jsonl"
MODEL_ROOT="${MODEL_ROOT:-/gemini/platform/public/aigc/aigc_image/zhanghy56_intern/zhangqisong/Qwen-Image-Edit-2511}"
MODEL_ID_WITH_ORIGIN="Qwen/Qwen-Image-Edit-2511:${MODEL_ROOT}/transformer/diffusion_pytorch_model*.safetensors,Qwen/Qwen-Image:${MODEL_ROOT}/text_encoder/model*.safetensors,Qwen/Qwen-Image:${MODEL_ROOT}/vae/diffusion_pytorch_model.safetensors"
TOKENIZER_PATH="${MODEL_ROOT}/tokenizer"
PROCESSOR_PATH="${MODEL_ROOT}/processor"

if [ ! -f "$TRAIN_METADATA" ]; then
  echo "Missing train metadata: $TRAIN_METADATA" >&2
  exit 1
fi

EVAL_IMAGE_PATH=$(python - "$DATASET_ROOT" <<'PY'
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
with (root / "pairs" / "train_pairs.jsonl").open("r", encoding="utf-8") as f:
    row = json.loads(next(line for line in f if line.strip()))
print(root / row.get("source_image", "views/obj_001/yaw000.png"))
PY
)
EVAL_OBJECT_DESCRIPTION=$(python - "$DATASET_ROOT" <<'PY'
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
with (root / "pairs" / "train_pairs.jsonl").open("r", encoding="utf-8") as f:
    row = json.loads(next(line for line in f if line.strip()))
print(row.get("object_description", "object"))
PY
)

mkdir -p "$OUTPUT_DIR" "${OUTPUT_DIR}/tensorboard"

accelerate_cmd=(accelerate launch --num_processes "$NUM_PROCESSES")
if [ -n "$ACCELERATE_CONFIG" ] && [ -f "$ACCELERATE_CONFIG" ]; then
  accelerate_cmd+=(--config_file "$ACCELERATE_CONFIG")
fi

extra_args=()
if [ "${FIND_UNUSED_PARAMETERS:-true}" = true ]; then
  extra_args+=(--find_unused_parameters)
fi

"${accelerate_cmd[@]}" train_clockwise.py \
  --dataset_base_path "$DATASET_ROOT" \
  --dataset_metadata_path "$TRAIN_METADATA" \
  --data_file_keys "image,edit_image" \
  --extra_inputs "edit_image" \
  --max_pixels 1048576 \
  --dataset_repeat "${DATASET_REPEAT:-10}" \
  --model_id_with_origin_paths "$MODEL_ID_WITH_ORIGIN" \
  --tokenizer_path "$TOKENIZER_PATH" \
  --processor_path "$PROCESSOR_PATH" \
  --learning_rate "${LEARNING_RATE:-1e-4}" \
  --num_epochs "${NUM_EPOCHS:-30}" \
  --save_every_epochs "${SAVE_EVERY_EPOCHS:-1}" \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "$OUTPUT_DIR" \
  --lora_base_model "dit" \
  --lora_target_modules "to_q,to_k,to_v,add_q_proj,add_k_proj,add_v_proj,to_out.0,to_add_out,img_mlp.net.2,img_mod.1,txt_mlp.net.2,txt_mod.1" \
  --lora_rank "${LORA_RANK:-32}" \
  --use_gradient_checkpointing \
  --dataset_num_workers "${DATASET_NUM_WORKERS:-4}" \
  --tensorboard_log_dir "${OUTPUT_DIR}/tensorboard" \
  --tensorboard_flush_secs "${TENSORBOARD_FLUSH_SECS:-10}" \
  --eval_image_path "$EVAL_IMAGE_PATH" \
  --eval_every_epochs "${EVAL_EVERY_EPOCHS:-1}" \
  --eval_num_inference_steps "${EVAL_NUM_INFERENCE_STEPS:-30}" \
  --eval_seed "${EVAL_SEED:-42}" \
  --seed "${TRAIN_SEED:-42}" \
  --eval_object_description "$EVAL_OBJECT_DESCRIPTION" \
  "${extra_args[@]}" \
  --zero_cond_t
BASH
)
  script=${script//__WORKDIR__/$WORKDIR}
  script=${script//__DATASET_ROOT__/$DATASET_ROOT}
  script=${script//__OUTPUT_DIR__/$OUTPUT_DIR}
  printf '%s\n' "$script"
}

if [ "$GENERATE_ONLY" -eq 1 ]; then
  emit_script
else
  emit_script | bash
fi
