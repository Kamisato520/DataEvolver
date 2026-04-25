#!/usr/bin/env python3
"""Dual-path dataset synthesis loop.

- Blender path: command-template driven, optional placeholder fallback.
- T2I path: provider command templates, optional placeholder fallback.
- Preview-first loop with dual-model evaluation and iterative refinement.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

try:
    from PIL import Image, ImageDraw
except Exception:
    Image = None
    ImageDraw = None

# Reuse gate utilities
import dataset_readiness_gate as gate

THIS_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_DIR = THIS_DIR.parent / "configs"

DEFAULT_T2I_CONFIG = {
    "prompting": {
        "prompt_templates": [
            "{subject}, {scene}, {lighting}, {style}, high quality dataset sample",
            "{subject} in {scene}, {lighting}, {style}, clean composition",
            "research training image: {subject}, {scene}, {lighting}, {style}",
        ],
        "variant_dimensions": {
            "subject": ["target object", "variant object", "auxiliary object"],
            "scene": ["indoor studio", "outdoor environment", "neutral background"],
            "lighting": ["soft daylight", "hard side light", "diffuse light"],
            "style": ["photorealistic", "documentary", "product-shot"],
        },
        "max_unique_prompts": 5000,
    },
    "preview": {"images_per_provider": 8},
    "full": {"images_per_provider": 200},
    "providers": [
        {
            "name": "qwen_image_local",
            "enabled": True,
            "command": "",
            "output_ext": "png",
            "allow_placeholder": True,
        },
        {
            "name": "nano_banana",
            "enabled": False,
            "command": "",
            "output_ext": "png",
            "allow_placeholder": True,
        },
        {
            "name": "gpt_image",
            "enabled": False,
            "command": "",
            "output_ext": "png",
            "allow_placeholder": True,
        },
    ],
    "refinement": {"primary_refine_command": ""},
}

DEFAULT_BLENDER_CONFIG = {
    "object_blend_folder": "",
    "scene_blend_path": "",
    "output_path": "./refine-logs/render-output",
    "allow_placeholder": True,
    "preview_image_count": 8,
    "full_image_count": 200,
    "preview_video_count": 2,
    "full_video_count": 20,
    "command_images": "",
    "command_videos": "",
    "preview_command_images": "",
    "preview_command_videos": "",
    "full_command_images": "",
    "full_command_videos": "",
}


def _default_path(name: str) -> str:
    return str((DEFAULT_CONFIG_DIR / name).resolve())


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict JSON in {path}")
    return data


def deep_update(base: dict, override: dict) -> dict:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def unique_keep_order(items: Sequence[str]) -> List[str]:
    return gate.unique_keep_order(items)


def parse_list_arg(values: Sequence[str]) -> List[str]:
    out: List[str] = []
    for raw in values:
        for item in raw.split(","):
            item = item.strip()
            if item:
                out.append(item)
    return unique_keep_order(out)


def run_shell(cmd: str, env: Optional[dict] = None) -> Tuple[int, str, str]:
    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, env=env, check=False)
    return proc.returncode, proc.stdout, proc.stderr


def create_placeholder_image(path: Path, text: str) -> bool:
    if Image is None:
        return False
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        img = Image.new("RGB", (768, 768), color=(245, 245, 245))
        if ImageDraw is not None:
            draw = ImageDraw.Draw(img)
            draw.multiline_text((20, 20), text[:260], fill=(30, 30, 30), spacing=4)
        img.save(path)
        return True
    except Exception:
        return False


def create_placeholder_mask(path: Path) -> bool:
    if Image is None:
        return False
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        img = Image.new("L", (768, 768), color=255)
        img.save(path)
        return True
    except Exception:
        return False


def create_placeholder_video(path: Path) -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg,
        "-y",
        "-f",
        "lavfi",
        "-i",
        "color=c=gray:s=640x640:d=1",
        "-pix_fmt",
        "yuv420p",
        str(path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    return proc.returncode == 0 and path.exists()


def load_t2i_config(path: Optional[str]) -> dict:
    cfg = json.loads(json.dumps(DEFAULT_T2I_CONFIG))
    p = Path(path) if path else Path(_default_path("t2i_generation.default.json"))
    if p.exists():
        deep_update(cfg, load_json(p))
    return cfg


def load_blender_config(path: Optional[str]) -> dict:
    cfg = json.loads(json.dumps(DEFAULT_BLENDER_CONFIG))
    p = Path(path) if path else Path(_default_path("blender_render.default.json"))
    if p.exists():
        deep_update(cfg, load_json(p))
    return cfg


def read_requirements(requirements_path: Optional[str], missing_data_args: Sequence[str]) -> dict:
    missing = parse_list_arg(missing_data_args)
    details: Dict[str, object] = {}
    if requirements_path and Path(requirements_path).exists():
        payload = load_json(Path(requirements_path))
        details = payload
        missing = unique_keep_order(missing + [str(x) for x in payload.get("missing_data", [])])
    return {"missing_data": missing, "details": details}


def build_requirements_text(requirements: dict) -> str:
    missing = requirements.get("missing_data", [])
    details = requirements.get("details", {})
    parts = []
    if missing:
        parts.append("missing_data=" + ", ".join(str(x) for x in missing))
    if details:
        parts.append("details=" + json.dumps(details, ensure_ascii=False)[:1200])
    return " | ".join(parts)


def generate_prompts(requirements: dict, t2i_cfg: dict, total_needed: int) -> List[str]:
    prompting = t2i_cfg.get("prompting", {})
    templates = prompting.get("prompt_templates", []) or ["{subject}, {scene}, {lighting}, {style}"]
    variants = prompting.get("variant_dimensions", {})
    if not variants:
        variants = {
            "subject": ["target object"],
            "scene": ["neutral background"],
            "lighting": ["soft light"],
            "style": ["photorealistic"],
        }

    keys = list(variants.keys())
    max_unique = int(prompting.get("max_unique_prompts", 5000))
    req_text = build_requirements_text(requirements)

    stride = 1
    strides = {}
    for k in keys:
        strides[k] = stride
        stride *= max(1, len(variants.get(k, [])))

    prompts: List[str] = []
    cap = min(max_unique, max(total_needed * 8, 100))
    for idx in range(cap):
        t = templates[idx % len(templates)]
        vals = {}
        for k in keys:
            arr = variants.get(k) or [k]
            pick = (idx // strides[k]) % len(arr)
            vals[k] = arr[pick]
        p = t
        for k, v in vals.items():
            p = p.replace("{" + k + "}", str(v))
        if req_text:
            p = f"{p}. Requirements: {req_text}"
        prompts.append(p)
        if len(prompts) >= total_needed:
            break

    return unique_keep_order(prompts)[:total_needed]


def _render_template(template: str, kv: dict) -> str:
    out = template
    for k, v in kv.items():
        out = out.replace("{" + k + "}", str(v))
    return out


def _scan_outputs(root: Path) -> dict:
    out = {"root": str(root.resolve()), "images": [], "videos": [], "masks": [], "metadata": []}
    if not root.exists():
        return out
    for cur, _, files in os.walk(root):
        for name in files:
            fp = Path(cur) / name
            ext = fp.suffix.lower()
            low = name.lower()
            if ext in gate.IMAGE_EXTENSIONS:
                out["images"].append(str(fp.resolve()))
            elif ext in gate.VIDEO_EXTENSIONS:
                out["videos"].append(str(fp.resolve()))
            if "mask" in low and ext in gate.IMAGE_EXTENSIONS:
                out["masks"].append(str(fp.resolve()))
            if ext == ".json":
                out["metadata"].append(str(fp.resolve()))
    for k in ("images", "videos", "masks", "metadata"):
        out[k] = sorted(unique_keep_order(out[k]))
    return out


def generate_with_provider(provider: dict, prompts: Sequence[str], output_dir: Path) -> Tuple[List[str], List[str]]:
    generated: List[str] = []
    errors: List[str] = []
    name = str(provider.get("name", "provider"))
    cmd_template = str(provider.get("command", "")).strip()
    ext = str(provider.get("output_ext", "png")).lstrip(".")
    allow_placeholder = bool(provider.get("allow_placeholder", True))

    output_dir.mkdir(parents=True, exist_ok=True)

    for idx, prompt in enumerate(prompts, start=1):
        out_path = output_dir / f"{name}_{idx:05d}.{ext}"
        ok = False

        if cmd_template:
            cmd = (
                cmd_template
                .replace("{prompt}", prompt.replace('"', '\\"'))
                .replace("{output}", str(out_path))
                .replace("{index}", str(idx))
                .replace("{provider}", name)
            )
            rc, _out, err = run_shell(cmd, env=os.environ.copy())
            if rc == 0 and out_path.exists():
                ok = True
            else:
                errors.append(f"{name}#{idx} command failed: {err[-400:]}")

        if not ok and allow_placeholder:
            ok = create_placeholder_image(out_path, prompt)
            if not ok:
                errors.append(f"{name}#{idx} placeholder generation failed")

        if ok and out_path.exists():
            generated.append(str(out_path.resolve()))

    return generated, errors


def run_blender_path(blender_cfg: dict, output_root: Path, mode: str, preview: bool, requirements: dict) -> dict:
    output_root.mkdir(parents=True, exist_ok=True)
    phase = "preview" if preview else "full"
    allow_placeholder = bool(blender_cfg.get("allow_placeholder", True))

    img_dir = output_root / "images" / "blender"
    mask_dir = output_root / "masks" / "blender"
    vid_dir = output_root / "videos" / "blender"
    meta_dir = output_root / "metadata" / "blender"

    obj_folder = str(blender_cfg.get("object_blend_folder", ""))
    scene_path = str(blender_cfg.get("scene_blend_path", ""))

    img_count = int(blender_cfg.get("preview_image_count", 8) if preview else blender_cfg.get("full_image_count", 200))
    vid_count = int(blender_cfg.get("preview_video_count", 2) if preview else blender_cfg.get("full_video_count", 20))

    cmd_images = str(
        blender_cfg.get(f"{phase}_command_images") or blender_cfg.get("command_images") or ""
    ).strip()
    cmd_videos = str(
        blender_cfg.get(f"{phase}_command_videos") or blender_cfg.get("command_videos") or ""
    ).strip()

    steps = []

    if mode in {"images", "both"}:
        if cmd_images:
            cmd = _render_template(
                cmd_images,
                {
                    "output_dir": str(img_dir.resolve()),
                    "object_blend_folder": obj_folder,
                    "scene_blend_path": scene_path,
                    "phase": phase,
                },
            )
            rc, out, err = run_shell(cmd)
            steps.append({"mode": "images", "command": cmd, "return_code": rc, "stdout_tail": out[-1500:], "stderr_tail": err[-1500:]})
        if not _scan_outputs(output_root)["images"] and allow_placeholder:
            req_text = build_requirements_text(requirements)
            for i in range(1, max(1, img_count) + 1):
                img_path = img_dir / f"blender_{i:05d}.png"
                mask_path = mask_dir / f"blender_{i:05d}_mask.png"
                create_placeholder_image(img_path, f"blender placeholder {i}\n{req_text}")
                create_placeholder_mask(mask_path)

    if mode in {"videos", "both"}:
        if cmd_videos:
            cmd = _render_template(
                cmd_videos,
                {
                    "output_dir": str(vid_dir.resolve()),
                    "object_blend_folder": obj_folder,
                    "scene_blend_path": scene_path,
                    "phase": phase,
                },
            )
            rc, out, err = run_shell(cmd)
            steps.append({"mode": "videos", "command": cmd, "return_code": rc, "stdout_tail": out[-1500:], "stderr_tail": err[-1500:]})
        if not _scan_outputs(output_root)["videos"] and allow_placeholder:
            for i in range(1, max(1, vid_count) + 1):
                v_path = vid_dir / f"blender_{i:05d}.mp4"
                ok = create_placeholder_video(v_path)
                meta = {
                    "video_name": v_path.name,
                    "object_name": "placeholder_object",
                    "camera_index": 0,
                    "frame_count": 1,
                    "frames": [{"frame": 1}],
                    "placeholder_video_created": ok,
                }
                meta_path = meta_dir / f"blender_{i:05d}.json"
                meta_path.parent.mkdir(parents=True, exist_ok=True)
                meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    outputs = _scan_outputs(output_root)
    success = (len(outputs["images"]) + len(outputs["videos"])) > 0
    return {
        "success": success,
        "steps": steps,
        "output_root": str(output_root.resolve()),
        "error": None if success else "no blender outputs generated",
        "outputs": outputs,
    }


def merge_outputs(output_maps: Sequence[dict]) -> dict:
    merged = {"root": "", "images": [], "videos": [], "masks": [], "metadata": []}
    for item in output_maps:
        for key in ("images", "videos", "masks", "metadata"):
            merged[key].extend(item.get(key, []))
    for key in ("images", "videos", "masks", "metadata"):
        merged[key] = unique_keep_order(merged[key])
    return merged


def summarize_outputs(outputs: dict) -> dict:
    return {
        "images": len(outputs.get("images", [])),
        "videos": len(outputs.get("videos", [])),
        "masks": len(outputs.get("masks", [])),
        "metadata": len(outputs.get("metadata", [])),
    }


def build_preview_stats(outputs: dict) -> gate.DatasetStats:
    image_count = len(outputs.get("images", []))
    video_count = len(outputs.get("videos", []))
    label_count = len(outputs.get("masks", [])) + len(outputs.get("metadata", []))

    min_w = None
    min_h = None
    for raw in outputs.get("images", [])[:20]:
        size = gate.read_image_size(Path(raw))
        if size is None:
            continue
        w, h = size
        min_w = w if min_w is None else min(min_w, w)
        min_h = h if min_h is None else min(min_h, h)

    return gate.DatasetStats(
        path="SYNTHESIS_PREVIEW",
        image_count=image_count,
        video_count=video_count,
        label_count=label_count,
        primary_sample_count=image_count + video_count,
        sampled_images_for_decode=min(image_count, 20),
        unreadable_images=0,
        min_width=min_w,
        min_height=min_h,
        class_coverage=max(1, gate.estimate_render_class_coverage(outputs)) if (image_count + video_count) > 0 else 0,
        split_counts={"train": image_count + video_count, "val": 0, "test": 0},
        label_completeness=gate.safe_ratio(label_count, image_count + video_count),
        bad_sample_rate=0.0,
    )


def heuristic_refine(missing_data: Sequence[str], blender_overrides: dict, t2i_overrides: dict) -> Tuple[dict, dict]:
    b = json.loads(json.dumps(blender_overrides or {}))
    t = json.loads(json.dumps(t2i_overrides or {}))

    if any("more_samples" in x for x in missing_data):
        t["full_images_per_provider"] = int(min(1000, int(t.get("full_images_per_provider", 200)) + 50))
        b["full_image_count"] = int(min(1000, int(b.get("full_image_count", 200)) + 50))

    if any("more_classes" in x for x in missing_data):
        t["prompt_diversity_boost"] = int(t.get("prompt_diversity_boost", 0)) + 1

    return b, t


def run_primary_refine_command(command_template: str, payload: dict) -> Optional[dict]:
    if not command_template:
        return None
    with tempfile.TemporaryDirectory(prefix="synth_refine_") as tmp:
        in_path = Path(tmp) / "input.json"
        out_path = Path(tmp) / "output.json"
        in_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        cmd = command_template.replace("{input_json}", str(in_path)).replace("{output_json}", str(out_path))
        rc, _out, _err = run_shell(cmd)
        if rc != 0 or not out_path.exists():
            return None
        try:
            return load_json(out_path)
        except Exception:
            return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dual-path dataset synthesis loop")
    parser.add_argument("--requirements-path", default=None)
    parser.add_argument("--missing-data", action="append", default=[])
    parser.add_argument("--idea-path", default="refine-logs/FINAL_PROPOSAL.md")
    parser.add_argument("--llm-eval-config-path", default=_default_path("dataset_llm_eval.default.json"))
    parser.add_argument("--t2i-config-path", default=_default_path("t2i_generation.default.json"))
    parser.add_argument("--blender-config-path", default=_default_path("blender_render.default.json"))
    parser.add_argument("--render-output-mode", choices=["images", "videos", "both"], default="both")

    parser.add_argument("--enable-blender", action="store_true", default=True)
    parser.add_argument("--disable-blender", action="store_false", dest="enable_blender")
    parser.add_argument("--enable-t2i", action="store_true", default=True)
    parser.add_argument("--disable-t2i", action="store_false", dest="enable_t2i")

    parser.add_argument("--preview-max-iterations", type=int, default=3)
    parser.add_argument("--preview-accepted-score", type=float, default=0.75)

    parser.add_argument("--output-root", default="refine-logs/synthesis-output")
    parser.add_argument("--reports-dir", default="refine-logs")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    requirements = read_requirements(args.requirements_path, args.missing_data)
    llm_cfg = gate.load_llm_eval_config(args.llm_eval_config_path)
    t2i_cfg = load_t2i_config(args.t2i_config_path)
    blender_cfg = load_blender_config(args.blender_config_path)

    if not args.enable_blender and not args.enable_t2i:
        print("[synthesis-loop] both paths are disabled")
        return 2

    idea_text = ""
    if args.idea_path and Path(args.idea_path).exists():
        idea_text = Path(args.idea_path).read_text(encoding="utf-8", errors="ignore")

    output_root = Path(args.output_root).resolve()
    preview_root = output_root / "preview"
    full_root = output_root / "full"
    preview_root.mkdir(parents=True, exist_ok=True)
    full_root.mkdir(parents=True, exist_ok=True)

    blender_overrides: dict = {}
    t2i_overrides: dict = {}
    preview_history: List[dict] = []
    accepted_iteration = None

    for iteration in range(1, args.preview_max_iterations + 1):
        iter_dir = preview_root / f"iter_{iteration}"
        iter_dir.mkdir(parents=True, exist_ok=True)

        outputs_for_merge = []
        iter_steps = {"iteration": iteration, "paths": {}, "summary": {}}

        if args.enable_blender:
            b_dir = iter_dir / "blender"
            b_cfg = deep_update(json.loads(json.dumps(blender_cfg)), blender_overrides)
            b_res = run_blender_path(
                blender_cfg=b_cfg,
                output_root=b_dir,
                mode=args.render_output_mode,
                preview=True,
                requirements=requirements,
            )
            iter_steps["paths"]["blender"] = {
                "success": b_res.get("success", False),
                "error": b_res.get("error"),
                "steps": b_res.get("steps", []),
            }
            if b_res.get("success"):
                outputs_for_merge.append(b_res.get("outputs", {}))

        if args.enable_t2i:
            t_dir = iter_dir / "t2i"
            providers = [p for p in t2i_cfg.get("providers", []) if p.get("enabled", False)]
            preview_count = int(t2i_cfg.get("preview", {}).get("images_per_provider", 8))
            prompts = generate_prompts(requirements, t2i_cfg, max(preview_count * max(len(providers), 1), 1))

            t_outputs = {"root": str(t_dir.resolve()), "images": [], "videos": [], "masks": [], "metadata": []}
            provider_logs = []
            for provider in providers:
                n = min(preview_count, len(prompts))
                provider_prompts = prompts[:n]
                p_name = str(provider.get("name", "provider"))
                generated, errors = generate_with_provider(provider, provider_prompts, t_dir / p_name)
                t_outputs["images"].extend(generated)
                provider_logs.append({"provider": p_name, "generated": len(generated), "errors": errors[:10]})

            t_outputs["images"] = unique_keep_order(t_outputs["images"])
            iter_steps["paths"]["t2i"] = {
                "success": len(t_outputs["images"]) > 0,
                "providers": provider_logs,
            }
            if len(t_outputs["images"]) > 0:
                outputs_for_merge.append(t_outputs)

        merged_outputs = merge_outputs(outputs_for_merge)
        preview_stats = build_preview_stats(merged_outputs)
        preview_threshold = gate.EvalResult(
            passed=(preview_stats.primary_sample_count > 0),
            failed_checks=[] if preview_stats.primary_sample_count > 0 else ["preview has no generated samples"],
        )
        preview_missing_seed = requirements.get("missing_data", []) if preview_stats.primary_sample_count <= 0 else []

        payload = gate.build_eval_payload(
            phase=f"preview_iter_{iteration}",
            idea_path=args.idea_path,
            idea_text=idea_text,
            dataset_stats=preview_stats,
            threshold_eval=preview_threshold,
            missing_seed=preview_missing_seed,
            render_counts=summarize_outputs(merged_outputs),
        )
        dual_eval = gate.run_dual_model_eval(
            llm_cfg=llm_cfg,
            phase=f"preview_iter_{iteration}",
            payload=payload,
            threshold_eval=preview_threshold,
            missing_seed=preview_missing_seed,
        )

        iter_steps["summary"] = {
            "preview_counts": summarize_outputs(merged_outputs),
            "dual_eval": dual_eval,
            "blender_overrides": blender_overrides,
            "t2i_overrides": t2i_overrides,
        }
        preview_history.append(iter_steps)

        avg = float(dual_eval.get("consensus", {}).get("avg_match_score", 0.0))
        ok = bool(dual_eval.get("consensus", {}).get("pass", False)) and avg >= args.preview_accepted_score
        if ok:
            accepted_iteration = iteration
            break

        refine_payload = {
            "iteration": iteration,
            "requirements": requirements,
            "dual_eval": dual_eval,
            "blender_overrides": blender_overrides,
            "t2i_overrides": t2i_overrides,
        }
        refine_cmd = str(t2i_cfg.get("refinement", {}).get("primary_refine_command", "")).strip()
        refined = run_primary_refine_command(refine_cmd, refine_payload)
        if refined:
            blender_overrides = deep_update(blender_overrides, refined.get("blender_overrides", {}))
            t2i_overrides = deep_update(t2i_overrides, refined.get("t2i_overrides", {}))
        else:
            blender_overrides, t2i_overrides = heuristic_refine(
                dual_eval.get("missing_data_union", requirements.get("missing_data", [])),
                blender_overrides,
                t2i_overrides,
            )

    accepted = accepted_iteration is not None
    full_outputs_for_merge = []
    full_logs = {"blender": None, "t2i": None}

    if accepted:
        if args.enable_blender:
            b_dir = full_root / "blender"
            b_cfg = deep_update(json.loads(json.dumps(blender_cfg)), blender_overrides)
            b_res = run_blender_path(
                blender_cfg=b_cfg,
                output_root=b_dir,
                mode=args.render_output_mode,
                preview=False,
                requirements=requirements,
            )
            full_logs["blender"] = {
                "success": b_res.get("success", False),
                "error": b_res.get("error"),
                "steps": b_res.get("steps", []),
            }
            if b_res.get("success"):
                full_outputs_for_merge.append(b_res.get("outputs", {}))

        if args.enable_t2i:
            t_dir = full_root / "t2i"
            providers = [p for p in t2i_cfg.get("providers", []) if p.get("enabled", False)]
            full_count_base = int(t2i_cfg.get("full", {}).get("images_per_provider", 200))
            full_count = int(t2i_overrides.get("full_images_per_provider", full_count_base))
            full_count = max(1, full_count)
            prompts = generate_prompts(requirements, t2i_cfg, max(full_count * max(len(providers), 1), 1))
            t_outputs = {"root": str(t_dir.resolve()), "images": [], "videos": [], "masks": [], "metadata": []}
            provider_logs = []
            offset = 0
            for provider in providers:
                p_name = str(provider.get("name", "provider"))
                p_prompts = prompts[offset: offset + full_count]
                offset += full_count
                generated, errors = generate_with_provider(provider, p_prompts, t_dir / p_name)
                t_outputs["images"].extend(generated)
                provider_logs.append({"provider": p_name, "generated": len(generated), "errors": errors[:10]})
            t_outputs["images"] = unique_keep_order(t_outputs["images"])
            full_logs["t2i"] = {"success": len(t_outputs["images"]) > 0, "providers": provider_logs}
            if len(t_outputs["images"]) > 0:
                full_outputs_for_merge.append(t_outputs)

    full_outputs = merge_outputs(full_outputs_for_merge)
    full_outputs["root"] = str(full_root.resolve())

    manifest = {
        "generated_at": gate.now_iso(),
        "success": bool(accepted and (len(full_outputs.get("images", [])) + len(full_outputs.get("videos", [])) > 0)),
        "accepted": bool(accepted),
        "accepted_iteration": accepted_iteration,
        "preview_iterations": len(preview_history),
        "preview_history": preview_history,
        "applied_overrides": {
            "blender": blender_overrides,
            "t2i": t2i_overrides,
        },
        "requirements": requirements,
        "outputs": {
            "root": str(full_root.resolve()),
            "images": full_outputs.get("images", []),
            "videos": full_outputs.get("videos", []),
            "masks": full_outputs.get("masks", []),
            "metadata": full_outputs.get("metadata", []),
            "counts": summarize_outputs(full_outputs),
        },
        "full_logs": full_logs,
    }

    manifest_path = reports_dir / "synthesis_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"[synthesis-loop] manifest: {manifest_path}")
    print(f"[synthesis-loop] accepted_iteration: {accepted_iteration}")
    print(f"[synthesis-loop] success: {manifest['success']}")

    return 0 if manifest["success"] else 2


if __name__ == "__main__":
    sys.exit(main())
