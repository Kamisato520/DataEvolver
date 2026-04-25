import json
import os
import math
import argparse
import random
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm.auto import tqdm

import torch
import accelerate
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from safetensors.torch import save_file  # 新增导入

from diffsynth.core import UnifiedDataset
from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
from diffsynth.diffusion import *
from diffsynth.core.data.operators import *

os.environ["TOKENIZERS_PARALLELISM"] = "false"

DEFAULT_VIEW_BUCKETS = [
    "front view",
    "front-right quarter view",
    "right side view",
    "back-right quarter view",
    "back view",
    "back-left quarter view",
    "left side view",
    "front-left quarter view",
]
ALLOWED_VIEW_BUCKETS = set(DEFAULT_VIEW_BUCKETS)
PAIR_TRACE_REQUIRED_KEYS = ("pair_id", "obj_id", "target_rotation_deg")


def normalize_view_bucket(view: Any) -> str:
    text = str(view).strip().lower()
    text = " ".join(text.split())
    if text not in ALLOWED_VIEW_BUCKETS:
        raise ValueError(
            "Unsupported angle_bucket/view prompt: "
            f"{repr(view)}. Allowed values: {sorted(ALLOWED_VIEW_BUCKETS)}"
        )
    return text


# -----------------------------
# Utilities
# -----------------------------
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def default_collate_keep_dict(batch):
    # 这里默认 batch_size=1 最稳
    if len(batch) == 1:
        return batch[0]
    # 如果你后面确认 diffsynth 的 pipeline 能直接吃 list-batch，
    # 可以把这里改成更复杂的 collate。
    return batch[0]


def get_arg_with_default(args, names: List[str], default):
    for name in names:
        if hasattr(args, name):
            return getattr(args, name)
    return default


def _unwrap_trace_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        if value.size == 1:
            return value.item()
        return value.tolist()
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return _unwrap_trace_value(value[0])
    return value


def _resolve_pair_loss_trace_dir(args) -> Optional[str]:
    if getattr(args, "disable_pair_loss_trace", False):
        return None
    return args.pair_loss_trace_dir or os.path.join(args.output_path, "pair_loss_trace")


def _resolve_pair_loss_trace_run_id(args) -> str:
    if args.pair_loss_trace_run_id:
        return str(args.pair_loss_trace_run_id)
    return os.path.basename(os.path.abspath(args.output_path.rstrip("\\/")))


def _open_pair_loss_trace_file(trace_root: str, epoch: int, rank: int):
    epoch_dir = os.path.join(trace_root, f"epoch_{epoch:02d}")
    os.makedirs(epoch_dir, exist_ok=True)
    trace_path = os.path.join(epoch_dir, f"rank_{rank:02d}.jsonl")
    return open(trace_path, "w", encoding="utf-8", buffering=1), trace_path


def _build_pair_loss_trace_record(
    batch: Dict[str, Any],
    *,
    run_id: str,
    epoch: int,
    global_step: int,
    rank: int,
    loss_scalar: float,
) -> dict:
    if not isinstance(batch, dict):
        raise TypeError(
            "pair loss trace requires dict batches. "
            f"Got {type(batch)}; keep batch_size=1 with default_collate_keep_dict."
        )

    missing = [key for key in PAIR_TRACE_REQUIRED_KEYS if key not in batch]
    if missing:
        raise KeyError(
            "pair loss trace requires dataset fields "
            f"{list(PAIR_TRACE_REQUIRED_KEYS)}, but batch is missing {missing}. "
            f"Available keys: {sorted(batch.keys())}"
        )

    pair_id = str(_unwrap_trace_value(batch["pair_id"]))
    obj_id = str(_unwrap_trace_value(batch["obj_id"]))
    target_rotation_deg = int(_unwrap_trace_value(batch["target_rotation_deg"]))
    return {
        "run_id": run_id,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "rank": int(rank),
        "pair_id": pair_id,
        "obj_id": obj_id,
        "target_rotation_deg": target_rotation_deg,
        "loss": float(loss_scalar),
    }


def _device_to_str(device: Any) -> str:
    if isinstance(device, torch.device):
        return str(device)
    return str(device)


def validate_trainable_param_devices(model: torch.nn.Module, expected_device: torch.device, accelerator):
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    local_devices = sorted({str(p.device) for _, p in trainable})
    local_count = len(trainable)
    print(
        f"[DDP-Check][rank={accelerator.process_index}] "
        f"expected_device={expected_device}, "
        f"trainable_param_count={local_count}, "
        f"trainable_param_devices={local_devices}"
    )

    if local_devices != [str(expected_device)]:
        preview = [f"{n}:{p.device}" for n, p in trainable[:20]]
        raise RuntimeError(
            "Trainable params are not all on the local accelerator device. "
            f"expected={expected_device}, actual={local_devices}, samples={preview}"
        )


def pil_to_numpy_hwc(img: Image.Image) -> np.ndarray:
    arr = np.array(img)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    return arr


def make_image_grid(
    images: List[Image.Image],
    titles: Optional[List[str]] = None,
    cols: int = 4,
    pad: int = 12,
    title_h: int = 28,
    bg_color=(255, 255, 255),
) -> Image.Image:
    if len(images) == 0:
        raise ValueError("images is empty")

    w = max(img.width for img in images)
    h = max(img.height for img in images)

    rows = math.ceil(len(images) / cols)
    canvas_w = cols * w + (cols + 1) * pad
    canvas_h = rows * (h + title_h) + (rows + 1) * pad
    canvas = Image.new("RGB", (canvas_w, canvas_h), bg_color)
    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for idx, img in enumerate(images):
        r = idx // cols
        c = idx % cols
        x = pad + c * w
        y = pad + r * (h + title_h)

        if img.size != (w, h):
            img = img.resize((w, h), Image.BILINEAR)

        if titles is not None and idx < len(titles):
            draw.text((x, y), titles[idx], fill=(0, 0, 0), font=font)

        canvas.paste(img, (x, y + title_h))

    return canvas


def extract_first_image_from_pipeline_output(output: Any) -> Image.Image:
    """
    尽量兼容不同 pipeline 输出格式：
    - PIL.Image
    - [PIL.Image, ...]
    - {"images": [...]}
    - 某些对象带 .images
    """
    if isinstance(output, Image.Image):
        return output

    if isinstance(output, (list, tuple)) and len(output) > 0:
        if isinstance(output[0], Image.Image):
            return output[0]

    if isinstance(output, dict):
        if "images" in output and len(output["images"]) > 0:
            return output["images"][0]
        if "image" in output:
            if isinstance(output["image"], Image.Image):
                return output["image"]
            if isinstance(output["image"], (list, tuple)) and len(output["image"]) > 0:
                return output["image"][0]

    if hasattr(output, "images") and len(output.images) > 0:
        return output.images[0]

    raise TypeError(f"Unsupported pipeline output type: {type(output)}")


# -----------------------------
# Model
# -----------------------------
class QwenImageTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None,
        model_id_with_origin_paths=None,
        tokenizer_path=None,
        processor_path=None,
        trainable_models=None,
        lora_base_model=None,
        lora_target_modules="",
        lora_rank=32,
        lora_checkpoint=None,
        preset_lora_path=None,
        preset_lora_model=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        fp8_models=None,
        offload_models=None,
        device="cpu",
        task="sft",
        zero_cond_t=False,
    ):
        super().__init__()

        # Load models
        model_configs = self.parse_model_configs(
            model_paths,
            model_id_with_origin_paths,
            fp8_models=fp8_models,
            offload_models=offload_models,
            device=device,
        )
        tokenizer_config = (
            ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="tokenizer/")
            if tokenizer_path is None
            else ModelConfig(tokenizer_path)
        )
        processor_config = (
            ModelConfig(model_id="Qwen/Qwen-Image-Edit", origin_file_pattern="processor/")
            if processor_path is None
            else ModelConfig(processor_path)
        )

        self.pipe = QwenImagePipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device=device,
            model_configs=model_configs,
            tokenizer_config=tokenizer_config,
            processor_config=processor_config,
        )
        self.pipe = self.split_pipeline_units(task, self.pipe, trainable_models, lora_base_model)

        # Training mode
        self.switch_pipe_to_training_mode(
            self.pipe,
            trainable_models,
            lora_base_model,
            lora_target_modules,
            lora_rank,
            lora_checkpoint,
            preset_lora_path,
            preset_lora_model,
            task=task,
        )

        # Configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None and extra_inputs != "" else []
        self.fp8_models = fp8_models
        self.task = task
        self.zero_cond_t = zero_cond_t

        self.task_to_loss = {
            "sft:data_process": lambda pipe, *args: args,
            "direct_distill:data_process": lambda pipe, *args: args,
            "sft": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
            "sft:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
            "sft:rotate": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
            "direct_distill": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi),
            "direct_distill:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi),
        }

    def get_pipeline_inputs(self, data):
        if isinstance(data, dict) and data.get("instruction"):
            prompt = str(data["instruction"])
        else:
            angle_value = "front view"
            if isinstance(data, dict):
                angle_value = data.get("angle_bucket", "front view")
            prompt = normalize_view_bucket(angle_value)

        inputs_posi = {"prompt": prompt}
        inputs_nega = {"negative_prompt": ""}
        inputs_shared = {
            "cfg_scale": 1,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "edit_image_auto_resize": True,
            "zero_cond_t": self.zero_cond_t,
        }

        if isinstance(data["image"], list):
            inputs_shared.update({
                "input_image": data["image"],
                "height": data["image"][0].size[1],
                "width": data["image"][0].size[0],
            })
        else:
            inputs_shared.update({
                "input_image": data["image"],
                "height": data["image"].size[1],
                "width": data["image"].size[0],
            })

        inputs_shared = self.parse_extra_inputs(data, self.extra_inputs, inputs_shared)
        return inputs_shared, inputs_posi, inputs_nega

    def forward(self, data, inputs=None):
        if inputs is None:
            inputs = self.get_pipeline_inputs(data)

        inputs = self.transfer_data_to_device(inputs, self.pipe.device, self.pipe.torch_dtype)
        for unit in self.pipe.units:
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)

        loss = self.task_to_loss[self.task](self.pipe, *inputs)
        return loss

    @torch.no_grad()
    def infer_single_rotation(
        self,
        image: Image.Image,
        view_prompt: str,
        num_inference_steps: int = 30,
        seed: int = 42,
    ) -> Image.Image:
        self.eval()

        prompt = str(view_prompt)
        seed_everything(seed)

        call_kwargs = dict(
            prompt=prompt,
            negative_prompt="",
            height=image.size[1],
            width=image.size[0],
            cfg_scale=1,
            edit_image_auto_resize=True,
            zero_cond_t=self.zero_cond_t,
            num_inference_steps=num_inference_steps,
        )
        try:
            # 与 inf.py 一致：优先使用 edit_image 接口
            output = self.pipe(edit_image=[image], **call_kwargs)
        except TypeError as e:
            if "edit_image" not in str(e):
                raise
            # 兼容某些版本仍使用 input_image
            output = self.pipe(input_image=image, **call_kwargs)

        out_img = extract_first_image_from_pipeline_output(output)
        self.train()
        return out_img

    @torch.no_grad()
    def infer_rotation_sequence(
        self,
        image: Image.Image,
        view_buckets: Optional[List[str]] = None,
        num_inference_steps: int = 30,
        seed: int = 42,
    ) -> Tuple[List[Image.Image], List[str]]:
        if view_buckets is None:
            view_buckets = DEFAULT_VIEW_BUCKETS

        images = []
        titles = []
        for view_prompt in view_buckets:
            out_img = self.infer_single_rotation(
                image=image,
                view_prompt=view_prompt,
                num_inference_steps=num_inference_steps,
                seed=seed,
            )
            images.append(out_img)
            titles.append(str(view_prompt))
        return images, titles


# -----------------------------
# Parser
# -----------------------------
def qwen_image_parser():
    parser = argparse.ArgumentParser(description="Qwen-Image training with TensorBoard logging.")
    parser = add_general_config(parser)
    parser = add_image_size_config(parser)

    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to tokenizer.")
    parser.add_argument("--processor_path", type=str, default=None, help="Path to processor.")
    parser.add_argument("--zero_cond_t", default=False, action="store_true", help="Enable for Qwen-Image-Edit-2511.")
    parser.add_argument("--save_every_epochs", type=int, default=1, help="Save model every N epochs.")

    # TensorBo