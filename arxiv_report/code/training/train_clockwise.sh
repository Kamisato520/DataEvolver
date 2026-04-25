export CUDA_LAUNCH_BLOCKING=1
export PYTHONFAULTHANDLER=1
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export TORCH_DISTRIBUTED_DEBUG=${TORCH_DISTRIBUTED_DEBUG:-OFF}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5}
export NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE:-0}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}
NUM_PROCESSES=${NUM_PROCESSES:-6}

# ===== 68 服务器路径配置 =====
WORKDIR="/gemini/platform/public/aigc/aigc_image/zhanghy56_intern/zhangqisong/data-build"

# Python 环境（uv）
source "${WORKDIR}/.venv/bin/activate"
export PYTHONPATH="${WORKDIR}/DiffSynth-Studio:${PYTHONPATH:-}"

# accelerate 配置（6 GPU）
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-${WORKDIR}/DiffSynth-Studio/accelerate_config_6gpu.yaml}"

# ===== 数据集路径 =====
# bright+clockwise bbox 版（当前默认）
SPLIT_ROOT="${WORKDIR}/dataset_scene_v7_full50_rotation8_trainready_front2others_splitobj_seed42_bboxmask_bright_final_20260416"
# 旧暗 bbox 版
# SPLIT_ROOT="${WORKDIR}/dataset_scene_v7_full50_rotation8_trainready_front2others_splitobj_seed42_bboxmask_final_20260414"
# RGB 版
# SPLIT_ROOT="${WORKDIR}/dataset_scene_v7_full50_rotation8_trainready_front2others_splitobj_seed42_final_20260410"

DATASET_BASE_PATH="${SPLIT_ROOT}"

CSV_LIST=(
  "${SPLIT_ROOT}/pairs/train_pairs.jsonl"
)

OUTPUT_DIR_LIST=(
  "${WORKDIR}/DiffSynth-Studio/output/rotation8_bright_clockwise_raw_rank32"
)

# ===== 模型路径（68 服务器） =====
MODEL_ROOT="/gemini/platform/public/aigc/aigc_image/zhanghy56_intern/zhangqisong/Qwen-Image-Edit-2511"
MODEL_ID_WITH_ORIGIN="Qwen/Qwen-Image-Edit-2511:${MODEL_ROOT}/transformer/diffusion_pytorch_model*.safetensors,Qwen/Qwen-Image:${MODEL_ROOT}/text_encoder/model*.safetensors,Qwen/Qwen-Image:${MODEL_ROOT}/vae/diffusion_pytorch_model.safetensors"
TOKENIZER_PATH="${MODEL_ROOT}/tokenizer"
PROCESSOR_PATH="${MODEL_ROOT}/processor"

# ===== 训练参数 =====
LEARNING_RATE="1e-4"
NUM_EPOCHS=30
DATASET_REPEAT=10
LORA_RANK=32
LORA_TARGET_MODULES="to_q,to_k,to_v,add_q_proj,