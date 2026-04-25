export CUDA_LAUNCH_BLOCKING=1
export PYTHONFAULTHANDLER=1
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export TORCH_DISTRIBUTED_DEBUG=${TORCH_DISTRIBUTED_DEBUG:-OFF}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1,2,3,4,5,7}
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
# bbox 版（当前默认）
SPLIT_ROOT="${WORKDIR}/dataset_scene_v7_full50_rotation8_trainready_front2others_splitobj_seed42_bboxmask_final_20260414"
# RGB 版（注释掉的备选，传上来后解注释）
# SPLIT_ROOT="${WORKDIR}/dataset_scene_v7_full50_rotation8_trainready_front2others_splitobj_seed42_final_20260410"

DATASET_BASE_PATH="${SPLIT_ROOT}"

CSV_LIST=(
  "${SPLIT_ROOT}/pairs/train_pairs.jsonl"
)

OUTPUT_DIR_LIST=(
  "${WORKDIR}/DiffSynth-Studio/output/rotation8_bbox_rank32"
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
LORA_TARGET_MODULES="to_q,to_k,to_v,add_q_proj,add_k_proj,add_v_proj,to_out.0,to_add_out,img_mlp.net.2,img_mod.1,txt_mlp.net.2,txt_mod.1"
REMOVE_PREFIX_IN_CKPT="pipe.dit."
DATASET_NUM_WORKERS=4
FIND_UNUSED_PARAMETERS=true
SAVE_EVERY_EPOCHS=1

# ===== TensorBoard / Eval 配置 =====
EVAL_IMAGE_PATH="${SPLIT_ROOT}/bbox_views/obj_001/yaw000.png"
EVAL_EVERY_EPOCHS=1
EVAL_NUM_INFERENCE_STEPS=30
EVAL_SEED=42
TRAIN_SEED=42
TENSORBOARD_FLUSH_SECS=10

# ===== 训练执行 =====
for i in "${!CSV_LIST[@]}"; do
  csv="${CSV_LIST[$i]}"
  out="${OUTPUT_DIR_LIST[$i]:-${OUTPUT_DIR_LIST[0]}}"
  idx=$((i+1))
  tb_dir="${out}/tensorboard"

  echo ""
  echo "=================================================================="
  echo "开始训练任务 #${idx}"
  echo "METADATA: ${csv}"
  echo "OUT: ${out}"
  echo "TB : ${tb_dir}"
  echo "EVAL IMAGE: ${EVAL_IMAGE_PATH}"
  echo "=================================================================="

  mkdir -p "${out}"
  mkdir -p "${tb_dir}"

  accelerate_cmd=(accelerate launch)
  accelerate_cmd+=(--num_processes "${NUM_PROCESSES}")
  if [ -n "${ACCELERATE_CONFIG}" ] && [ -f "${ACCELERATE_CONFIG}" ]; then
    accelerate_cmd+=(--config_file "${ACCELERATE_CONFIG}")
  fi

  "${accelerate_cmd[@]}" train_clockwise.py \
    --dataset_base_path "${DATASET_BASE_PATH}" \
    --dataset_metadata_path "${csv}" \
    --data_file_keys "image,edit_image" \
    --extra_inputs "edit_image" \
    --max_pixels 1048576 \
    --dataset_repeat "${DATASET_REPEAT}" \
    --model_id_with_origin_paths "${MODEL_ID_WITH_ORIGIN}" \
    --tokenizer_path "${TOKENIZER_PATH}" \
    --processor_path "${PROCESSOR_PATH}" \
    --learning_rate "${LEARNING_RATE}" \
    --num_epochs "${NUM_EPOCHS}" \
    --save_every_epochs "${SAVE_EVERY_EPOCHS}" \
    --remove_prefix_in_ckpt "${REMOVE_PREFIX_IN_CKPT}" \
    --output_path "${out}" \
    --lora_base_model "dit" \
    --lora_target_modules "${LORA_TARGET_MODULES}" \
    --lora_rank "${LORA_RANK}" \
    --use_gradient_checkpointing \
    --dataset_num_workers "${DATASET_NUM_WORKERS}" \
    --tensorboard_log_dir "${tb_dir}" \
    --tensorboard_flush_secs "${TENSORBOARD_FLUSH_SECS}" \
    --eval_image_path "${EVAL_IMAGE_PATH}" \
    --eval_every_epochs "${EVAL_EVERY_EPOCHS}" \
    --eval_num_inference_steps "${EVAL_NUM_INFERENCE_STEPS}" \
    --eval_seed "${EVAL_SEED}" \
    --seed "${TRAIN_SEED}" \
    $( [ "${FIND_UNUSED_PARAMETERS}" = true ] && echo "--find_unused_parameters" || echo "" ) \
    --zero_cond_t

  rc=$?
  if [ $rc -ne 0 ]; then
    echo "训练任务 #${idx} 失败，退出码 ${rc}"
    exit $rc
  fi

  echo "训练任务 #${idx} 完成"
  echo "TensorBoard 日志目录: ${tb_dir}"
done
