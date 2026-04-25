#!/usr/bin/env python3
"""Dataset readiness gate with dual-model evaluation and synthesis fallback.

Design goals:
- Keep orchestration deterministic and configurable.
- Support Blender/T2I/dual synthesis via dataset_synthesis_loop.py.
- Emit a stable manifest contract for downstream experiment skills.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

try:
    from PIL import Image
except Exception:
    Image = None

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
LABEL_EXTENSIONS = {".json", ".txt", ".csv", ".xml", ".yaml", ".yml"}

ROOT_DIR = Path(__file__).resolve().parents[3]
THIS_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_DIR = THIS_DIR.parent / "configs"

DEFAULT_THRESHOLDS = {
    "min_samples": 1000,
    "min_resolution": [256, 256],
    "min_label_completeness": 0.80,
    "min_class_coverage": 2,
    "max_bad_sample_rate": 0.05,
    "require_splits": ["train", "val", "test"],
}

DEFAULT_LLM_EVAL_CONFIG = {
    "enabled": True,
    "primary_model": {
        "name": "claude",
        "command": "",
    },
    "secondary_model": {
        "name": "gpt-5.4",
        "command": "",
    },
    "consensus": {
        "min_avg_match_score": 0.55,
        "block_on_not_fit": True,
        "drop_if_any_model_rejects": True,
    },
}


def now_iso() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def to_abs(path: str) -> str:
    return str(Path(path).expanduser().resolve())


def safe_ratio(num: float, den: float) -> float:
    if den <= 0:
        return 0.0
    return float(num) / float(den)


def deep_update(base: dict, override: dict) -> dict:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def unique_keep_order(items: Sequence[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in items:
        v = str(item).strip()
        if not v or v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def _default_path(name: str) -> str:
    return str((DEFAULT_CONFIG_DIR / name).resolve())


def load_thresholds(path: Optional[str]) -> dict:
    payload = json.loads(json.dumps(DEFAULT_THRESHOLDS))
    p = Path(path) if path else Path(_default_path("dataset_thresholds.default.json"))
    if p.exists():
        deep_update(payload, load_json(p))
    return payload


def load_llm_eval_config(path: Optional[str]) -> dict:
    payload = json.loads(json.dumps(DEFAULT_LLM_EVAL_CONFIG))
    p = Path(path) if path else Path(_default_path("dataset_llm_eval.default.json"))
    if p.exists():
        deep_update(payload, load_json(p))
    return payload


def discover_dataset_paths(user_paths: Sequence[str], plan_path: Optional[str]) -> List[str]:
    found: List[str] = []
    seen = set()

    def add_dir(path: Path) -> None:
        try:
            rp = str(path.expanduser().resolve())
        except Exception:
            return
        if rp in seen:
            return
        if path.exists() and path.is_dir():
            seen.add(rp)
            found.append(rp)

    for p in user_paths:
        add_dir(Path(p))

    for rel in ("datasets", "dataset", "data"):
        add_dir(ROOT_DIR / rel)

    if plan_path:
        plan = Path(plan_path)
        if plan.exists() and plan.is_file():
            text = plan.read_text(encoding="utf-8", errors="ignore")
            for raw in re.findall(r"(?:\./|\.\./|/)[^\s\"'`]+", text):
                add_dir(Path(raw))

    return found


def read_image_size(path: Path) -> Optional[Tuple[int, int]]:
    if Image is None:
        return None
    try:
        with Image.open(path) as img:
            return img.width, img.height
    except Exception:
        return None


@dataclass
class DatasetStats:
    path: str
    image_count: int
    video_count: int
    label_count: int
    primary_sample_count: int
    sampled_images_for_decode: int
    unreadable_images: int
    min_width: Optional[int]
    min_height: Optional[int]
    class_coverage: int
    split_counts: Dict[str, int]
    label_completeness: float
    bad_sample_rate: float


@dataclass
class EvalResult:
    passed: bool
    failed_checks: List[str]


@dataclass
class ModelReview:
    model_name: str
    source: str
    verdict: str
    match_score: float
    missing_data: List[str]
    unsuitable_patterns: List[str]
    unsuitable_paths: List[str]
    rationale: str
    error: Optional[str] = None


def collect_dataset_stats(path: str, split_names: Sequence[str], decode_limit: int = 200) -> DatasetStats:
    root = Path(path)
    image_count = 0
    video_count = 0
    label_count = 0
    sampled_images = 0
    unreadable_images = 0
    min_width: Optional[int] = None
    min_height: Optional[int] = None
    class_names = set()
    split_counts = {s: 0 for s in split_names}

    for current_root, _, files in os.walk(root):
        current = Path(current_root)
        rel_parts = current.relative_to(root).parts if current != root else tuple()
        for name in files:
            fp = current / name
            ext = fp.suffix.lower()
            if ext in IMAGE_EXTENSIONS:
                image_count += 1
                if rel_parts:
                    head = rel_parts[0]
                    if head in split_counts:
                        split_counts[head] += 1
                        if len(rel_parts) > 1:
                            class_names.add(rel_parts[1])
                    else:
                        class_names.add(head)
                if sampled_images < decode_limit:
                    sampled_images += 1
                    size = read_image_size(fp)
                    if size is None:
                        unreadable_images += 1
                    else:
                        w, h = size
                        min_width = w if min_width is None else min(min_width, w)
                        min_height = h if min_height is None else min(min_height, h)
            elif ext in VIDEO_EXTENSIONS:
                video_count += 1
                if rel_parts and rel_parts[0] in split_counts:
                    split_counts[rel_parts[0]] += 1
            elif ext in LABEL_EXTENSIONS:
                label_count += 1

    primary = image_count + video_count
    return DatasetStats(
        path=to_abs(path),
        image_count=image_count,
        video_count=video_count,
        label_count=label_count,
        primary_sample_count=primary,
        sampled_images_for_decode=sampled_images,
        unreadable_images=unreadable_images,
        min_width=min_width,
        min_height=min_height,
        class_coverage=len(class_names),
        split_counts=split_counts,
        label_completeness=safe_ratio(label_count, primary),
        bad_sample_rate=safe_ratio(unreadable_images, sampled_images),
    )


def aggregate_stats(stats_list: Sequence[DatasetStats], split_names: Sequence[str]) -> DatasetStats:
    split_counts = {s: 0 for s in split_names}
    min_width: Optional[int] = None
    min_height: Optional[int] = None
    total_images = sum(s.image_count for s in stats_list)
    total_videos = sum(s.video_count for s in stats_list)
    total_labels = sum(s.label_count for s in stats_list)
    total_primary = sum(s.primary_sample_count for s in stats_list)
    sampled = sum(s.sampled_images_for_decode for s in stats_list)
    unreadable = sum(s.unreadable_images for s in stats_list)
    class_cov = max((s.class_coverage for s in stats_list), default=0)

    for s in stats_list:
        if s.min_width is not None:
            min_width = s.min_width if min_width is None else min(min_width, s.min_width)
        if s.min_height is not None:
            min_height = s.min_height if min_height is None else min(min_height, s.min_height)
        for split in split_names:
            split_counts[split] += s.split_counts.get(split, 0)

    return DatasetStats(
        path="MULTI_DATASET",
        image_count=total_images,
        video_count=total_videos,
        label_count=total_labels,
        primary_sample_count=total_primary,
        sampled_images_for_decode=sampled,
        unreadable_images=unreadable,
        min_width=min_width,
        min_height=min_height,
        class_coverage=class_cov,
        split_counts=split_counts,
        label_completeness=safe_ratio(total_labels, total_primary),
        bad_sample_rate=safe_ratio(unreadable, sampled),
    )


def evaluate_stats(stats: DatasetStats, thresholds: dict) -> EvalResult:
    failed: List[str] = []
    min_samples = int(thresholds.get("min_samples", 0))
    if stats.primary_sample_count < min_samples:
        failed.append(f"samples {stats.primary_sample_count} < {min_samples}")

    min_res = thresholds.get("min_resolution", [0, 0])
    min_w, min_h = int(min_res[0]), int(min_res[1])
    if min_w > 0 and min_h > 0:
        if stats.min_width is None or stats.min_height is None:
            failed.append("resolution unavailable (image decode unavailable or no images)")
        elif stats.min_width < min_w or stats.min_height < min_h:
            failed.append(f"resolution ({stats.min_width}x{stats.min_height}) < ({min_w}x{min_h})")

    min_label = float(thresholds.get("min_label_completeness", 0.0))
    if stats.label_completeness < min_label:
        failed.append(f"label completeness {stats.label_completeness:.3f} < {min_label:.3f}")

    min_cls = int(thresholds.get("min_class_coverage", 0))
    if stats.class_coverage < min_cls:
        failed.append(f"class coverage {stats.class_coverage} < {min_cls}")

    max_bad = float(thresholds.get("max_bad_sample_rate", 1.0))
    if stats.bad_sample_rate > max_bad:
        failed.append(f"bad sample rate {stats.bad_sample_rate:.3f} > {max_bad:.3f}")

    for split in thresholds.get("require_splits", []):
        if stats.split_counts.get(split, 0) <= 0:
            failed.append(f"missing split: {split}")

    return EvalResult(passed=(len(failed) == 0), failed_checks=failed)


def map_failed_checks_to_missing(failed_checks: Sequence[str]) -> List[str]:
    out: List[str] = []
    for item in failed_checks:
        low = item.lower()
        if "samples" in low:
            out.append("more_samples")
        elif "resolution" in low:
            out.append("higher_resolution")
        elif "label completeness" in low:
            out.append("more_complete_labels")
        elif "class coverage" in low:
            out.append("more_classes")
        elif "missing split" in low:
            split = item.split(":")[-1].strip()
            out.append(f"missing_split_{split}")
        elif "bad sample rate" in low:
            out.append("cleaner_samples")
    return unique_keep_order(out)


def clip_text(text: str, max_chars: int = 16000) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[truncated]..."


def build_eval_payload(
    phase: str,
    idea_path: Optional[str],
    idea_text: str,
    dataset_stats: DatasetStats,
    threshold_eval: EvalResult,
    missing_seed: Sequence[str],
    render_counts: Optional[dict] = None,
) -> dict:
    return {
        "phase": phase,
        "idea_path": to_abs(idea_path) if idea_path else None,
        "idea_text": clip_text(idea_text),
        "dataset_stats": asdict(dataset_stats),
        "threshold_eval": asdict(threshold_eval),
        "seed_missing_data": list(missing_seed),
        "render_counts": render_counts or {},
    }


def heuristic_model_review(model_name: str, phase: str, missing_seed: Sequence[str], threshold_eval: EvalResult) -> ModelReview:
    missing = unique_keep_order(list(missing_seed))
    if phase == "post_render" and threshold_eval.passed:
        verdict = "fit"
    elif len(missing) <= 1:
        verdict = "partial"
    else:
        verdict = "not_fit"

    if threshold_eval.passed and not missing:
        score = 0.85
    elif threshold_eval.passed:
        score = 0.70
    elif len(missing) <= 2:
        score = 0.50
    else:
        score = 0.30

    return ModelReview(
        model_name=model_name,
        source="heuristic",
        verdict=verdict,
        match_score=max(0.0, min(1.0, score)),
        missing_data=missing,
        unsuitable_patterns=[],
        unsuitable_paths=[],
        rationale="heuristic fallback from threshold checks",
        error=None,
    )


def parse_model_review(model_name: str, payload: dict, source: str) -> ModelReview:
    verdict = str(payload.get("verdict", "partial")).strip().lower()
    if verdict not in {"fit", "partial", "not_fit"}:
        verdict = "partial"
    try:
        score = float(payload.get("match_score", 0.5))
    except Exception:
        score = 0.5
    score = max(0.0, min(1.0, score))
    return ModelReview(
        model_name=model_name,
        source=source,
        verdict=verdict,
        match_score=score,
        missing_data=unique_keep_order(payload.get("missing_data", [])),
        unsuitable_patterns=unique_keep_order(payload.get("unsuitable_patterns", [])),
        unsuitable_paths=unique_keep_order(payload.get("unsuitable_paths", [])),
        rationale=str(payload.get("rationale", "") or "no rationale"),
        error=None,
    )


def run_model_eval_command(model_cfg: dict, phase: str, payload: dict) -> Tuple[Optional[ModelReview], Optional[str]]:
    model_name = str(model_cfg.get("name", "model")).strip() or "model"
    cmd_template = str(model_cfg.get("command", "")).strip()
    if not cmd_template:
        return None, "empty command"

    with tempfile.TemporaryDirectory(prefix=f"dual_eval_{model_name}_") as tmp:
        input_path = Path(tmp) / "input.json"
        output_path = Path(tmp) / "output.json"
        input_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        cmd = (
            cmd_template
            .replace("{input_json}", str(input_path))
            .replace("{output_json}", str(output_path))
            .replace("{phase}", phase)
            .replace("{model_name}", model_name)
        )
        proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            msg = proc.stderr[-1000:] if proc.stderr else proc.stdout[-1000:]
            return None, f"command failed (rc={proc.returncode}): {msg}"
        if not output_path.exists():
            return None, "command finished but output_json not found"
        try:
            payload_out = load_json(output_path)
            return parse_model_review(model_name, payload_out, source="command"), None
        except Exception as e:
            return None, f"invalid output_json: {e}"


def run_dual_model_eval(llm_cfg: dict, phase: str, payload: dict, threshold_eval: EvalResult, missing_seed: Sequence[str]) -> dict:
    enabled = bool(llm_cfg.get("enabled", True))
    if not enabled:
        primary = heuristic_model_review("claude", phase, missing_seed, threshold_eval)
        secondary = heuristic_model_review("gpt-5.4", phase, missing_seed, threshold_eval)
    else:
        primary_cfg = llm_cfg.get("primary_model", {})
        secondary_cfg = llm_cfg.get("secondary_model", {})

        primary, p_err = run_model_eval_command(primary_cfg, phase, payload)
        if primary is None:
            primary = heuristic_model_review(str(primary_cfg.get("name", "claude")), phase, missing_seed, threshold_eval)
            primary.error = p_err

        secondary, s_err = run_model_eval_command(secondary_cfg, phase, payload)
        if secondary is None:
            secondary = heuristic_model_review(str(secondary_cfg.get("name", "gpt-5.4")), phase, missing_seed, threshold_eval)
            secondary.error = s_err

    reviews = [primary, secondary]
    consensus_cfg = llm_cfg.get("consensus", {})
    min_avg = float(consensus_cfg.get("min_avg_match_score", 0.55))
    block_on_not_fit = bool(consensus_cfg.get("block_on_not_fit", True))

    avg = safe_ratio(sum(r.match_score for r in reviews), len(reviews))
    any_not_fit = any(r.verdict == "not_fit" for r in reviews)
    passed = avg >= min_avg and not (block_on_not_fit and any_not_fit)

    missing_union = unique_keep_order(list(missing_seed) + [x for r in reviews for x in r.missing_data])
    bad_patterns = unique_keep_order([x for r in reviews for x in r.unsuitable_patterns])
    bad_paths = unique_keep_order([x for r in reviews for x in r.unsuitable_paths])

    return {
        "phase": phase,
        "enabled": enabled,
        "reviews": [asdict(r) for r in reviews],
        "consensus": {
            "avg_match_score": avg,
            "pass": passed,
            "any_not_fit": any_not_fit,
            "min_avg_match_score": min_avg,
            "block_on_not_fit": block_on_not_fit,
        },
        "missing_data_union": missing_union,
        "unsuitable_patterns_union": bad_patterns,
        "unsuitable_paths_union": bad_paths,
    }


def run_command(cmd: Sequence[str], env: Optional[dict] = None) -> Tuple[int, str, str]:
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)
    return proc.returncode, proc.stdout, proc.stderr


def run_synthesis_loop(
    synthesis_loop_script: str,
    reports_dir: str,
    requirements_path: Optional[str],
    missing_data: Sequence[str],
    idea_path: Optional[str],
    llm_eval_config_path: Optional[str],
    t2i_config_path: Optional[str],
    blender_config_path: Optional[str],
    render_output_mode: str,
    synthesis_mode: str,
) -> dict:
    cmd = [
        sys.executable,
        synthesis_loop_script,
        "--reports-dir",
        reports_dir,
        "--render-output-mode",
        render_output_mode,
    ]
    if requirements_path:
        cmd.extend(["--requirements-path", requirements_path])
    for item in missing_data:
        cmd.extend(["--missing-data", item])
    if idea_path:
        cmd.extend(["--idea-path", idea_path])
    if llm_eval_config_path:
        cmd.extend(["--llm-eval-config-path", llm_eval_config_path])
    if t2i_config_path:
        cmd.extend(["--t2i-config-path", t2i_config_path])
    if blender_config_path:
        cmd.extend(["--blender-config-path", blender_config_path])

    if synthesis_mode == "blender":
        cmd.append("--disable-t2i")
    elif synthesis_mode == "t2i":
        cmd.append("--disable-blender")

    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    manifest_path = Path(reports_dir) / "synthesis_manifest.json"
    step = {
        "mode": synthesis_mode,
        "script": to_abs(synthesis_loop_script),
        "command": cmd,
        "return_code": proc.returncode,
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
    }

    if proc.returncode != 0:
        return {
            "triggered": True,
            "success": False,
            "steps": [step],
            "error": "synthesis loop failed",
            "manifest_path": str(manifest_path.resolve()),
        }

    if not manifest_path.exists():
        return {
            "triggered": True,
            "success": False,
            "steps": [step],
            "error": "synthesis manifest not found",
            "manifest_path": str(manifest_path.resolve()),
        }

    try:
        manifest = load_json(manifest_path)
    except Exception as e:
        return {
            "triggered": True,
            "success": False,
            "steps": [step],
            "error": f"invalid synthesis manifest: {e}",
            "manifest_path": str(manifest_path.resolve()),
        }

    outputs = manifest.get("outputs", {})
    return {
        "triggered": True,
        "success": bool(manifest.get("success", False)),
        "synthesis_mode": synthesis_mode,
        "manifest_path": str(manifest_path.resolve()),
        "render_output_root": str(outputs.get("root", "")),
        "steps": [step],
    }


def ffprobe_video(path: Path) -> Optional[dict]:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return None
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,nb_frames",
        "-of",
        "json",
        str(path),
    ]
    rc, out, _ = run_command(cmd)
    if rc != 0:
        return None
    try:
        data = json.loads(out)
        streams = data.get("streams", [])
        return streams[0] if streams else None
    except Exception:
        return None


def is_mask_binary(path: Path) -> Optional[bool]:
    if Image is None:
        return None
    try:
        with Image.open(path) as img:
            vals = set(img.convert("L").getdata())
        return vals.issubset({0, 255})
    except Exception:
        return False


def collect_render_outputs(root: str) -> dict:
    rp = Path(root)
    out = {"root": to_abs(root), "images": [], "videos": [], "masks": [], "metadata": []}
    if not rp.exists():
        return out
    for current_root, _, files in os.walk(rp):
        for name in files:
            fp = Path(current_root) / name
            ext = fp.suffix.lower()
            low = name.lower()
            if ext in IMAGE_EXTENSIONS:
                out["images"].append(str(fp.resolve()))
            elif ext in VIDEO_EXTENSIONS:
                out["videos"].append(str(fp.resolve()))
            if "mask" in low and ext in IMAGE_EXTENSIONS:
                out["masks"].append(str(fp.resolve()))
            if ext == ".json":
                out["metadata"].append(str(fp.resolve()))

    for key in ("images", "videos", "masks", "metadata"):
        out[key] = sorted(unique_keep_order(out[key]))
    return out


def _path_match_any(path: str, patterns: Sequence[str], bad_paths: Sequence[str]) -> bool:
    low = path.lower()
    for p in bad_paths:
        token = p.strip().lower()
        if token and token in low:
            return True
    for p in patterns:
        token = p.strip()
        if not token:
            continue
        try:
            if re.search(token, path, flags=re.IGNORECASE):
                return True
        except re.error:
            if token.lower() in low:
                return True
    return False


def filter_render_outputs(outputs: dict, patterns: Sequence[str], bad_paths: Sequence[str]) -> dict:
    filtered = {"root": outputs.get("root"), "images": [], "videos": [], "masks": [], "metadata": [], "dropped": []}
    for key in ("images", "videos", "masks", "metadata"):
        for item in outputs.get(key, []):
            if _path_match_any(item, patterns, bad_paths):
                filtered["dropped"].append(item)
            else:
                filtered[key].append(item)
    filtered["dropped"] = sorted(unique_keep_order(filtered["dropped"]))
    return filtered


def run_render_qc(outputs: dict, expected_mode: str, require_masks: bool = True, require_metadata: bool = True) -> dict:
    images = [Path(p) for p in outputs.get("images", [])]
    videos = [Path(p) for p in outputs.get("videos", [])]
    masks = [Path(p) for p in outputs.get("masks", [])]
    metadata = [Path(p) for p in outputs.get("metadata", [])]

    checks = {
        "has_images": len(images) > 0,
        "has_videos": len(videos) > 0,
        "has_masks": len(masks) > 0,
        "has_metadata": len(metadata) > 0,
        "image_decode_pass_rate": None,
        "video_probe_pass_rate": None,
        "mask_binary_pass_rate": None,
        "metadata_valid_pass_rate": None,
    }
    failures: List[str] = []

    if expected_mode in {"images", "both"} and not checks["has_images"]:
        failures.append("missing rendered images")
    if expected_mode in {"videos", "both"} and not checks["has_videos"]:
        failures.append("missing rendered videos")
    if require_masks and not checks["has_masks"]:
        failures.append("missing rendered masks")
    if require_metadata and not checks["has_metadata"]:
        failures.append("missing metadata json")

    sample_images = images[:50]
    if sample_images:
        ok = 0
        for p in sample_images:
            if read_image_size(p) is not None:
                ok += 1
        checks["image_decode_pass_rate"] = safe_ratio(ok, len(sample_images))
        if checks["image_decode_pass_rate"] < 0.95:
            failures.append("image decode pass rate < 0.95")

    sample_videos = videos[:30]
    if sample_videos:
        ok = 0
        for p in sample_videos:
            info = ffprobe_video(p)
            if info and int(info.get("width", 0)) > 0 and int(info.get("height", 0)) > 0:
                ok += 1
        checks["video_probe_pass_rate"] = safe_ratio(ok, len(sample_videos))
        if checks["video_probe_pass_rate"] < 0.90:
            failures.append("video probe pass rate < 0.90")

    sample_masks = masks[:50]
    if sample_masks:
        known = 0
        ok = 0
        for p in sample_masks:
            flag = is_mask_binary(p)
            if flag is None:
                continue
            known += 1
            if flag:
                ok += 1
        if known > 0:
            checks["mask_binary_pass_rate"] = safe_ratio(ok, known)
            if checks["mask_binary_pass_rate"] < 0.95:
                failures.append("mask binary pass rate < 0.95")

    sample_meta = metadata[:200]
    if sample_meta:
        ok = 0
        for p in sample_meta:
            try:
                payload = load_json(p)
                if isinstance(payload, dict) and payload:
                    ok += 1
            except Exception:
                pass
        checks["metadata_valid_pass_rate"] = safe_ratio(ok, len(sample_meta))
        if checks["metadata_valid_pass_rate"] < 0.95:
            failures.append("metadata valid pass rate < 0.95")

    return {
        "passed": len(failures) == 0,
        "failures": failures,
        "checks": checks,
        "counts": {
            "images": len(images),
            "videos": len(videos),
            "masks": len(masks),
            "metadata": len(metadata),
        },
    }


def synthetic_sample_count(render_counts: dict, task_type: str, mode: str) -> int:
    image_n = int(render_counts.get("images", 0))
    video_n = int(render_counts.get("videos", 0))
    if task_type == "video":
        return video_n
    if task_type == "image":
        return image_n
    if mode == "both":
        return image_n + video_n
    return max(image_n, video_n)


def estimate_render_class_coverage(outputs: dict) -> int:
    classes = set()
    for key in ("images", "videos"):
        for p in outputs.get(key, []):
            classes.add(Path(p).parent.name)
    return len([x for x in classes if x])


def estimate_outputs_min_resolution(outputs: dict, sample_limit: int = 200) -> Tuple[Optional[int], Optional[int]]:
    min_w = None
    min_h = None
    n = 0
    for p in outputs.get("images", []):
        size = read_image_size(Path(p))
        if size is None:
            continue
        w, h = size
        min_w = w if min_w is None else min(min_w, w)
        min_h = h if min_h is None else min(min_h, h)
        n += 1
        if n >= sample_limit:
            break
    return min_w, min_h


def build_markdown_readiness(
    path: Path,
    args: argparse.Namespace,
    thresholds: dict,
    discovered: Sequence[str],
    real_eval: dict,
    pre_llm_eval: dict,
    render_status: dict,
    render_qc: dict,
    post_llm_eval: dict,
    filter_summary: dict,
    merged_eval: dict,
    final_decision: str,
) -> None:
    lines: List[str] = []
    lines.append("# Dataset Readiness Report")
    lines.append("")
    lines.append(f"- Time: `{now_iso()}`")
    lines.append(f"- Task type: `{args.task_type}`")
    lines.append(f"- Dataset gate: `{args.dataset_gate}`")
    lines.append(f"- Render fallback: `{args.render_fallback}`")
    lines.append(f"- Synthesis mode: `{args.synthesis_mode}`")
    lines.append(f"- Render output mode: `{args.render_output_mode}`")
    lines.append(f"- Data merge mode: `{args.data_merge_mode}`")
    lines.append(f"- Gate policy: `{args.gate_policy}`")
    lines.append("")
    lines.append("## Thresholds")
    lines.append("```json")
    lines.append(json.dumps(thresholds, indent=2, ensure_ascii=False))
    lines.append("```")
    lines.append("")

    lines.append("## Discovered Datasets")
    if discovered:
        for p in discovered:
            lines.append(f"- `{p}`")
    else:
        lines.append("- None")
    lines.append("")

    lines.append("## Real Dataset Evaluation")
    lines.append(f"- Passed: `{real_eval.get('passed')}`")
    failed = real_eval.get("failed_checks", [])
    if failed:
        lines.append("- Failed checks:")
        for item in failed:
            lines.append(f"  - {item}")
    else:
        lines.append("- Failed checks: None")
    lines.append("")

    lines.append("## Dual-LLM Match Evaluation (Pre-Render)")
    lines.append(f"- Consensus pass: `{pre_llm_eval.get('consensus', {}).get('pass')}`")
    lines.append(f"- Avg match score: `{pre_llm_eval.get('consensus', {}).get('avg_match_score')}`")
    for item in pre_llm_eval.get("missing_data_union", []):
        lines.append(f"- Missing: `{item}`")
    for rv in pre_llm_eval.get("reviews", []):
        lines.append(f"- {rv.get('model_name')}: verdict=`{rv.get('verdict')}`, score=`{rv.get('match_score')}`")
    lines.append("")

    lines.append("## Synthesis Status")
    lines.append(f"- Triggered: `{render_status.get('triggered', False)}`")
    lines.append(f"- Success: `{render_status.get('success', False)}`")
    if render_status.get("error"):
        lines.append(f"- Error: `{render_status.get('error')}`")
    lines.append("")

    lines.append("## Render QC")
    lines.append(f"- Passed: `{render_qc.get('passed')}`")
    for f in render_qc.get("failures", []):
        lines.append(f"- Failure: {f}")
    lines.append("")

    lines.append("## Dual-LLM Match Evaluation (Post)")
    lines.append(f"- Consensus pass: `{post_llm_eval.get('consensus', {}).get('pass')}`")
    lines.append(f"- Avg match score: `{post_llm_eval.get('consensus', {}).get('avg_match_score')}`")
    for item in post_llm_eval.get("missing_data_union", []):
        lines.append(f"- Remaining missing: `{item}`")
    for rv in post_llm_eval.get("reviews", []):
        lines.append(f"- {rv.get('model_name')}: verdict=`{rv.get('verdict')}`, score=`{rv.get('match_score')}`")
    lines.append("")

    lines.append("## Filter Summary")
    lines.append(f"- Dropped synthetic files: `{filter_summary.get('dropped_count', 0)}`")
    lines.append(f"- Remaining images: `{filter_summary.get('remaining_images', 0)}`")
    lines.append(f"- Remaining videos: `{filter_summary.get('remaining_videos', 0)}`")
    lines.append("")

    lines.append("## Merged Evaluation")
    lines.append(f"- Passed: `{merged_eval.get('passed')}`")
    for f in merged_eval.get("failed_checks", []):
        lines.append(f"- Failure: {f}")
    lines.append("")

    lines.append("## Final Decision")
    lines.append(f"- `{final_decision}`")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_markdown_render_qc(path: Path, render_qc: dict, outputs: dict) -> None:
    lines: List[str] = []
    lines.append("# Render QC Report")
    lines.append("")
    lines.append(f"- Time: `{now_iso()}`")
    lines.append(f"- Root: `{outputs.get('root')}`")
    lines.append(f"- Passed: `{render_qc.get('passed')}`")
    lines.append("")
    counts = render_qc.get("counts", {})
    lines.append("## Output Counts")
    for k in ("images", "videos", "masks", "metadata"):
        lines.append(f"- {k}: `{counts.get(k, 0)}`")
    lines.append("")
    lines.append("## Checks")
    for k, v in render_qc.get("checks", {}).items():
        lines.append(f"- {k}: `{v}`")
    lines.append("")
    lines.append("## Failures")
    fails = render_qc.get("failures", [])
    if fails:
        for f in fails:
            lines.append(f"- {f}")
    else:
        lines.append("- None")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dataset readiness gate")
    parser.add_argument("--dataset-path", action="append", default=[], help="Candidate dataset path (repeatable)")
    parser.add_argument("--plan-path", default="refine-logs/EXPERIMENT_PLAN.md")
    parser.add_argument("--idea-path", default="refine-logs/FINAL_PROPOSAL.md")
    parser.add_argument("--task-type", choices=["image", "video", "multimodal"], default="multimodal")
    parser.add_argument("--thresholds-path", default=_default_path("dataset_thresholds.default.json"))
    parser.add_argument("--llm-eval-config-path", default=_default_path("dataset_llm_eval.default.json"))

    parser.add_argument("--dataset-gate", action="store_true", default=True)
    parser.add_argument("--no-dataset-gate", action="store_false", dest="dataset_gate")
    parser.add_argument("--render-fallback", choices=["blender", "none"], default="blender")
    parser.add_argument("--synthesis-mode", choices=["blender", "t2i", "dual"], default="dual")
    parser.add_argument("--render-output-mode", choices=["images", "videos", "both"], default="both")
    parser.add_argument("--data-merge-mode", choices=["fill-gap", "replace"], default="fill-gap")
    parser.add_argument(
        "--gate-policy",
        choices=["manual-override-on-fail", "strict-block", "warn-and-continue"],
        default="manual-override-on-fail",
    )
    parser.add_argument("--manual-override", action="store_true")

    parser.add_argument("--blender-config-path", default=_default_path("blender_render.default.json"))
    parser.add_argument("--t2i-config-path", default=_default_path("t2i_generation.default.json"))
    parser.add_argument("--synthesis-loop-script", default=str((THIS_DIR / "dataset_synthesis_loop.py").resolve()))
    parser.add_argument("--requirements-json-path", default=None)
    parser.add_argument("--reports-dir", default="refine-logs")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    thresholds = load_thresholds(args.thresholds_path)
    llm_cfg = load_llm_eval_config(args.llm_eval_config_path)
    split_names = thresholds.get("require_splits", ["train", "val", "test"])

    idea_text = ""
    for hint in [args.idea_path, args.plan_path]:
        p = Path(hint)
        if p.exists():
            idea_text = p.read_text(encoding="utf-8", errors="ignore")
            break

    discovered = discover_dataset_paths(args.dataset_path, args.plan_path)
    stats_list = [collect_dataset_stats(p, split_names) for p in discovered]

    if stats_list:
        real_agg = aggregate_stats(stats_list, split_names)
    else:
        real_agg = DatasetStats(
            path="MULTI_DATASET",
            image_count=0,
            video_count=0,
            label_count=0,
            primary_sample_count=0,
            sampled_images_for_decode=0,
            unreadable_images=0,
            min_width=None,
            min_height=None,
            class_coverage=0,
            split_counts={s: 0 for s in split_names},
            label_completeness=0.0,
            bad_sample_rate=0.0,
        )

    real_eval = evaluate_stats(real_agg, thresholds)
    pre_missing = map_failed_checks_to_missing(real_eval.failed_checks)
    pre_payload = build_eval_payload("pre_render", args.idea_path, idea_text, real_agg, real_eval, pre_missing)
    pre_llm_eval = run_dual_model_eval(llm_cfg, "pre_render", pre_payload, real_eval, pre_missing)
    missing_for_synth = unique_keep_order(pre_llm_eval.get("missing_data_union", []))

    render_status = {"triggered": False, "success": False, "steps": []}
    render_outputs = {"root": "", "images": [], "videos": [], "masks": [], "metadata": []}
    render_qc = {
        "passed": False,
        "failures": ["render not triggered"],
        "checks": {},
        "counts": {"images": 0, "videos": 0, "masks": 0, "metadata": 0},
    }

    should_synth = args.dataset_gate and len(missing_for_synth) > 0 and args.render_fallback != "none"
    if should_synth:
        requirements_path = args.requirements_json_path
        if not requirements_path:
            auto_req = reports_dir / "missing_data_requirements.json"
            auto_req.write_text(
                json.dumps(
                    {"missing_data": missing_for_synth, "reviews": pre_llm_eval.get("reviews", [])},
                    indent=2,
                    ensure_ascii=False,
                ) + "\n",
                encoding="utf-8",
            )
            requirements_path = str(auto_req)

        render_status = run_synthesis_loop(
            synthesis_loop_script=args.synthesis_loop_script,
            reports_dir=str(reports_dir),
            requirements_path=requirements_path,
            missing_data=missing_for_synth,
            idea_path=args.idea_path,
            llm_eval_config_path=args.llm_eval_config_path,
            t2i_config_path=args.t2i_config_path,
            blender_config_path=args.blender_config_path,
            render_output_mode=args.render_output_mode,
            synthesis_mode=args.synthesis_mode,
        )

        if render_status.get("success"):
            render_root = render_status.get("render_output_root", "")
            render_outputs = collect_render_outputs(render_root)
            qc_mode = "images" if args.synthesis_mode == "t2i" else args.render_output_mode
            require_masks = args.synthesis_mode != "t2i"
            require_metadata = args.synthesis_mode != "t2i"
            render_qc = run_render_qc(render_outputs, qc_mode, require_masks, require_metadata)
        else:
            render_root = render_status.get("render_output_root", "")
            render_outputs = collect_render_outputs(render_root)
            render_qc = {
                "passed": False,
                "failures": [render_status.get("error", "synthesis failed")],
                "checks": {},
                "counts": {
                    "images": len(render_outputs.get("images", [])),
                    "videos": len(render_outputs.get("videos", [])),
                    "masks": len(render_outputs.get("masks", [])),
                    "metadata": len(render_outputs.get("metadata", [])),
                },
            }

    provisional_mode = "images" if args.synthesis_mode == "t2i" else args.render_output_mode
    provisional_n = synthetic_sample_count(render_qc.get("counts", {}), args.task_type, provisional_mode)
    provisional_labeled = provisional_n if (render_qc.get("checks", {}).get("has_masks") or args.synthesis_mode == "t2i") else 0
    provisional_cov = estimate_render_class_coverage(render_outputs)
    prov_w, prov_h = estimate_outputs_min_resolution(render_outputs)

    merged_w = prov_w if real_agg.min_width is None else (real_agg.min_width if prov_w is None else min(real_agg.min_width, prov_w))
    merged_h = prov_h if real_agg.min_height is None else (real_agg.min_height if prov_h is None else min(real_agg.min_height, prov_h))

    provisional_merged = DatasetStats(
        path="MERGED_DATASET",
        image_count=real_agg.image_count + int(render_qc.get("counts", {}).get("images", 0)),
        video_count=real_agg.video_count + int(render_qc.get("counts", {}).get("videos", 0)),
        label_count=real_agg.label_count + provisional_labeled,
        primary_sample_count=real_agg.primary_sample_count + provisional_n,
        sampled_images_for_decode=real_agg.sampled_images_for_decode,
        unreadable_images=real_agg.unreadable_images,
        min_width=merged_w,
        min_height=merged_h,
        class_coverage=max(real_agg.class_coverage, provisional_cov),
        split_counts=real_agg.split_counts,
        label_completeness=safe_ratio(real_agg.label_count + provisional_labeled, real_agg.primary_sample_count + provisional_n),
        bad_sample_rate=real_agg.bad_sample_rate,
    )

    provisional_eval = evaluate_stats(provisional_merged, thresholds)
    post_missing = map_failed_checks_to_missing(provisional_eval.failed_checks)
    post_payload = build_eval_payload(
        "post_render" if should_synth else "post_merge",
        args.idea_path,
        idea_text,
        provisional_merged if should_synth else real_agg,
        provisional_eval if should_synth else real_eval,
        post_missing,
        render_counts=render_qc.get("counts", {}),
    )
    post_llm_eval = run_dual_model_eval(
        llm_cfg,
        "post_render" if should_synth else "post_merge",
        post_payload,
        provisional_eval if should_synth else real_eval,
        post_missing,
    )

    if should_synth:
        bad_patterns = post_llm_eval.get("unsuitable_patterns_union", [])
        bad_paths = post_llm_eval.get("unsuitable_paths_union", [])
        filtered_outputs = filter_render_outputs(render_outputs, bad_patterns, bad_paths)
        qc_mode = "images" if args.synthesis_mode == "t2i" else args.render_output_mode
        require_masks = args.synthesis_mode != "t2i"
        require_metadata = args.synthesis_mode != "t2i"
        filtered_qc = run_render_qc(filtered_outputs, qc_mode, require_masks, require_metadata)
    else:
        filtered_outputs = dict(render_outputs)
        filtered_qc = dict(render_qc)

    filter_summary = {
        "dropped_count": len(filtered_outputs.get("dropped", [])),
        "remaining_images": len(filtered_outputs.get("images", [])),
        "remaining_videos": len(filtered_outputs.get("videos", [])),
        "remaining_masks": len(filtered_outputs.get("masks", [])),
    }

    merged_mode = "images" if args.synthesis_mode == "t2i" else args.render_output_mode
    synth_n = synthetic_sample_count(filtered_qc.get("counts", {}), args.task_type, merged_mode)
    synth_labeled = synth_n if (filtered_qc.get("checks", {}).get("has_masks") or args.synthesis_mode == "t2i") else 0
    synth_cov = estimate_render_class_coverage(filtered_outputs)
    final_w, final_h = estimate_outputs_min_resolution(filtered_outputs)
    merged_final_w = final_w if real_agg.min_width is None else (real_agg.min_width if final_w is None else min(real_agg.min_width, final_w))
    merged_final_h = final_h if real_agg.min_height is None else (real_agg.min_height if final_h is None else min(real_agg.min_height, final_h))

    merged_stats = DatasetStats(
        path="MERGED_DATASET",
        image_count=real_agg.image_count + int(filtered_qc.get("counts", {}).get("images", 0)),
        video_count=real_agg.video_count + int(filtered_qc.get("counts", {}).get("videos", 0)),
        label_count=real_agg.label_count + synth_labeled,
        primary_sample_count=real_agg.primary_sample_count + synth_n,
        sampled_images_for_decode=real_agg.sampled_images_for_decode,
        unreadable_images=real_agg.unreadable_images,
        min_width=merged_final_w,
        min_height=merged_final_h,
        class_coverage=max(real_agg.class_coverage, synth_cov),
        split_counts=real_agg.split_counts,
        label_completeness=safe_ratio(real_agg.label_count + synth_labeled, real_agg.primary_sample_count + synth_n),
        bad_sample_rate=real_agg.bad_sample_rate,
    )
    merged_eval = evaluate_stats(merged_stats, thresholds)

    pre_pass = bool(pre_llm_eval.get("consensus", {}).get("pass", False))
    post_pass = bool(post_llm_eval.get("consensus", {}).get("pass", False))
    qc_ok = bool(filtered_qc.get("passed", False)) if should_synth else True
    merged_ok = bool(merged_eval.passed)

    gate_passed = merged_ok and pre_pass and post_pass and qc_ok
    if gate_passed:
        decision = "PASSED_AFTER_RENDER_FILTERED_DUAL_LLM" if should_synth else "PASSED_REAL_DATA_DUAL_LLM"
        next_step = "allow_experiment"
    else:
        if args.gate_policy == "warn-and-continue":
            decision = "WARN_CONTINUE"
            next_step = "allow_experiment"
            gate_passed = True
        elif args.gate_policy == "manual-override-on-fail" and args.manual_override:
            decision = "PASSED_MANUAL_OVERRIDE"
            next_step = "allow_experiment"
            gate_passed = True
        else:
            decision = "BLOCKED"
            next_step = "manual_approval_required" if args.gate_policy == "manual-override-on-fail" else "block_experiment"

    manifest = {
        "generated_at": now_iso(),
        "gate_passed": gate_passed,
        "decision": decision,
        "next_step": next_step,
        "settings": {
            "DATASET_GATE": args.dataset_gate,
            "DATASET_THRESHOLDS_PATH": to_abs(args.thresholds_path),
            "RENDER_FALLBACK": args.render_fallback,
            "SYNTHESIS_MODE": args.synthesis_mode,
            "RENDER_OUTPUT_MODE": args.render_output_mode,
            "DATA_MERGE_MODE": args.data_merge_mode,
            "GATE_POLICY": args.gate_policy,
            "LLM_EVAL_CONFIG_PATH": to_abs(args.llm_eval_config_path),
            "T2I_CONFIG_PATH": to_abs(args.t2i_config_path),
            "BLENDER_CONFIG_PATH": to_abs(args.blender_config_path),
        },
        "discovered_datasets": discovered,
        "real_dataset_eval": {
            "stats": asdict(real_agg),
            "passed": real_eval.passed,
            "failed_checks": real_eval.failed_checks,
        },
        "dual_llm_eval": {
            "pre_render": pre_llm_eval,
            "post_render_or_merge": post_llm_eval,
        },
        "synthetic_data": {
            "triggered": should_synth,
            "status": render_status,
            "outputs": filtered_outputs,
            "qc": filtered_qc,
            "filter_summary": filter_summary,
        },
        "merged_eval": {
            "stats": asdict(merged_stats),
            "passed": merged_eval.passed,
            "failed_checks": merged_eval.failed_checks,
        },
        "training_dataset": {
            "real_paths": discovered,
            "synthetic_root": filtered_outputs.get("root", ""),
            "selected_counts": {
                "real_images": real_agg.image_count,
                "real_videos": real_agg.video_count,
                "synthetic_images": len(filtered_outputs.get("images", [])),
                "synthetic_videos": len(filtered_outputs.get("videos", [])),
            },
            "dropped_files_sample": filtered_outputs.get("dropped", [])[:100],
        },
    }

    manifest_path = reports_dir / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    readiness_path = reports_dir / "DATASET_READINESS.md"
    build_markdown_readiness(
        readiness_path,
        args,
        thresholds,
        discovered,
        manifest["real_dataset_eval"],
        pre_llm_eval,
        render_status,
        filtered_qc,
        post_llm_eval,
        filter_summary,
        manifest["merged_eval"],
        decision,
    )

    qc_path = reports_dir / "RENDER_QC_REPORT.md"
    build_markdown_render_qc(qc_path, filtered_qc, filtered_outputs)

    print(f"[dataset-gate] manifest: {manifest_path}")
    print(f"[dataset-gate] readiness report: {readiness_path}")
    print(f"[dataset-gate] render qc report: {qc_path}")
    print(f"[dataset-gate] decision: {decision}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
