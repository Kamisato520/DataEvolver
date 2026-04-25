#!/bin/bash
set -euo pipefail

# R2 quality_gate training command (derived from R1 train_command.sh).
# Differences vs R1:
#   - SPLIT_ROOT  -> r2 augmented_dataset (94 obj, 578 pairs)
#   - OUT         -> output/v2_scaling_r2_quality_gate
#   - From base, no R1 ckpt warm-start (per user choice b)
#   - All other hparams identical: 30 epoch, 6 GPU, lr=1e-4, rank=32, prompt v3

export CUDA_LAUNCH_BLOCKING=1
export PYTHONFAULTHANDLER=1
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export TORCH_DISTRIBUTED_DEBUG=${TORCH_DISTRIBUTED_DEBUG:-OFF}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5}
export NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE:-0}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}
NUM_PROCESSES=${NUM_PROCESSES:-6}

WORKDIR=/gemini/platform/public/aigc/aigc_image/zhanghy56_intern/zhangqisong/data-build
source "$WORKDIR/.venv/bin/activate"
export PYTHONPATH="$WORKDIR/DiffSynth-Studio:${PYTHONPATH:-}"

ACCELERATE_CONFIG=${ACCELERATE_CONFIG:-$WORKDIR/DiffSynth-Studio/accelerate_config_6gpu.yaml}
SPLIT_ROOT=$WORKDIR/feedback_loop_runs/v2_scaling_r2_quality_gate/round_1/augmented_dataset
DATASET_BASE_PATH=$SPLIT_ROOT
CSV=$SPLIT_ROOT/pairs/train_pairs.jsonl
OUT=$WORKDIR/DiffSynth-Studio/output/v2_scaling_r2_quality_gate
TB_DIR=$OUT/tensorboard
MODEL_ROOT=/gemini/platform/public/aigc/aigc_image/zhanghy56_intern/zhangqisong/Qwen-Image-Edit-2511
MODEL_ID_WITH_ORIGIN=Qwen/Qwen-Image-Edit-2511:${MODEL_ROOT}/transformer/diffusion_pytorch_model*.safetensors,Qwen/Qwen-Image:${MODEL_ROOT}/text_encoder/model*.safetensors,Qwen/Qwen-Image:${MODEL_ROOT}/vae/diffusion_pytorch_model.safetensors
TOKENIZER_PATH=${MODEL_ROOT}/tokenizer
PROCESSOR_PATH=${MODEL_ROOT}/processor
EVAL_IMAGE_PATH=${SPLIT_ROOT}/views/obj_001/yaw000.png

mkdir -p "$OUT" "$TB_DIR"
cd "$WORKDIR/DiffSynth-Studio"

accelerate launch --num_processes "$NUM_PROCESSES" --config_file "$ACCELERATE_CONFIG" train_clockwise.py \
  --dataset_base_path "$DATASET_BASE_PATH" \
  --dataset_metadata_path "$CSV" \
  --data_file_keys image,edit_image \
  --extra_inputs edit_image \
  --max_pixels 1048576 \
  --dataset_repeat 10 \
  --model_id_with_origin_paths "$MODEL_ID_WITH_ORIGIN" \
  --tokenizer_path "$TOKENIZER_PATH" \
  --processor_path "$PROCESSOR_PATH" \
  --learning_rate 1e-4 \
  --num_epochs 30 \
  --save_every_epochs 1 \
  --remove_prefix_in_ckpt pipe.dit. \
  --output_path "$OUT" \
  --lora_base_model dit \
  --lora_target_modules to_q,to_k,to_v,add_q_proj,add_k_proj,add_v_proj,to_out.0,to_add_out,img_mlp.net.2,img_mod.1,txt_mlp.net.2,txt_mod.1 \
  --lora_rank 32 \
  --use_gradient_checkpointing \
  --dataset_num_workers 4 \
  --tensorboard_log_dir "$TB_DIR" \
  --tensorboard_flush_secs 10 \
  --eval_image_path "$EVAL_IMAGE_PATH" \
  --eval_every_epochs 1 \
  --eval_num_inference_steps 30 \
  --eval_seed 42 \
  --seed 42 \
  --find_unused_parameters \
  --zero_cond_t

echo "Training completed at $(date '+%F %T')"
