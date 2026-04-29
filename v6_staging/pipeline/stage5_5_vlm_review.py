"""
Stage 5.5: VLM-based render review + CV metrics.

Reviews rendered images for one object using Qwen3-VL-8B-Instruct and
traditional CV metrics. Outputs a JSON review with hybrid quality score.

v6: Added structure_consistency, color_consistency, physics_consistency
    diagnostics with reference image comparison support.

Usage:
  python pipeline/stage5_5_vlm_review.py \
    --renders-dir pipeline/data/renders \
    --obj-id obj_001 \
    --output-dir pipeline/data/vlm_reviews \
    [--prev-rgb path/to/prev.png] \
    [--round-idx 0] \
    [--active-group lighting] \
    [--device cuda:0]
"""

import os
import sys
import json
import base64
import math
import re
import argparse
import copy
from collections import Counter as _Counter
from io import BytesIO
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

VLM_MODEL_PATH = "/huggingface/model_hub/Qwen3-VL-8B-Instruct"
DATA_BUILD_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ACTION_SPACE_PATH = os.path.join(DATA_BUILD_ROOT, "configs", "action_space.json")

# Representative canonical views: (azimuth, elevation)
REP_VIEWS = [(0, 0), (90, 0), (180, 0), (270, 0)]

ISSUE_TAGS = [
    "none", "underexposed", "overexposed", "flat_lighting", "harsh_shadow",
    "weak_subject_separation", "background_too_bright", "background_too_dark",
    "object_too_small", "object_too_large", "off_center_left", "off_center_right",
    "off_center_up", "off_center_down", "object_cutoff_left", "object_cutoff_right",
    "object_cutoff_top", "object_cutoff_bottom", "floating_object",
    "ground_intersection", "geometry_distortion", "partial_occlusion",
    "mask_boundary_error", "mask_hole", "mask_spill", "background_distracting",
    "depth_of_field_too_strong", "depth_of_field_too_weak",
    "mesh_interpenetration", "color_shift", "physical_implausibility",
]

LIGHTING_DIAGNOSIS_VALUES = {
    "flat_no_rim", "flat_low_contrast", "underexposed_global",
    "underexposed_shadow", "harsh_shadow_key", "good"
}

# v6: New diagnostic field valid values
STRUCTURE_VALUES = {"good", "minor_mismatch", "major_mismatch"}
COLOR_VALUES = {"good", "minor_shift", "major_shift"}
PHYSICS_VALUES = {"good", "minor_issue", "major_issue"}

GROUPS = ["lighting", "camera", "object", "scene", "material"]

# Hybrid score weights
VLM_W = {"lighting": 0.25, "object_integrity": 0.30, "composition": 0.20,
          "render_quality_semantic": 0.10, "overall": 0.15}
# Full 4-metric weights (used when mask is available)
CV_W  = {"mask_score": 0.30, "exposure_score": 0.25,
          "sharpness_score": 0.20, "framing_score": 0.25}
# 2-metric weights when no mask (exposure 0.556, sharpness 0.444)
CV_W_NOMASK = {"exposure_score": 0.556, "sharpness_score": 0.444}

# ─────────────────────────────────────────────────────────────────────────────
# CV Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_mask_score(mask_path: str) -> float:
    """Boundary F1 (3px band) + silhouette sanity (holes/fragments)."""
    try:
        import cv2
        mask = np.array(Image.open(mask_path).convert("L"))
        binary = (mask > 127).astype(np.uint8)
        if binary.sum() == 0:
            return 0.0

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        boundary_band = cv2.dilate(binary, kernel) - cv2.erode(binary, kernel)
        edges = cv2.Canny(binary * 255, 50, 150)
        edges_b = (edges > 0).astype(np.uint8)

        tp = float(np.sum(edges_b & (boundary_band > 0)))
        fp = float(np.sum(edges_b & (boundary_band == 0)))
        fn = float(np.sum((edges_b == 0) & (boundary_band > 0)))
        prec = tp / (tp + fp + 1e-8)
        rec  = tp / (tp + fn + 1e-8)
        f1   = 2 * prec * rec / (prec + rec + 1e-8)

        # Fragment penalty
        n_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary)
        total_fg = int(binary.sum())
        if n_labels > 1:
            main_area = int(stats[1:, cv2.CC_STAT_AREA].max())
            frag_penalty = 1.0 - max(0.0, (total_fg - main_area) / (total_fg + 1e-8))
        else:
            frag_penalty = 1.0

        # Hole penalty (large holes inside the object)
        inv = 1 - binary
        n_holes, _, h_stats, _ = cv2.connectedComponentsWithStats(inv)
        large_holes = sum(1 for a in (h_stats[1:, cv2.CC_STAT_AREA] if n_holes > 1 else []) if a > 200)
        hole_penalty = max(0.0, 1.0 - large_holes * 0.1)

        sanity = (frag_penalty + hole_penalty) / 2.0
        return float(np.clip(0.7 * f1 + 0.3 * sanity, 0.0, 1.0))
    except Exception as e:
        print(f"  [mask_score] fallback (0.5): {e}")
        return 0.5


def compute_exposure_score(rgb_path: str, mask_path: Optional[str] = None) -> float:
    """Luminance clipping + median deviation for foreground pixels."""
    try:
        img = np.array(Image.open(rgb_path).convert("RGB")).astype(float) / 255.0
        if mask_path and os.path.exists(mask_path):
            mask = np.array(Image.open(mask_path).convert("L"))
            fg = mask > 127
        else:
            h, w = img.shape[:2]
            fg = np.zeros((h, w), bool)
            fg[h // 4: 3 * h // 4, w // 4: 3 * w // 4] = True

        if fg.sum() < 100:
            return 0.5

        lum = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
        fg_lum = lum[fg]

        bright_clip = float((fg_lum > 0.95).mean())
        dark_clip   = float((fg_lum < 0.05).mean())
        median_lum  = float(np.median(fg_lum))

        clip_pen   = min(1.0, (bright_clip + dark_clip) * 2.0)
        median_pen = min(1.0, abs(median_lum - 0.45) * 1.5)
        return float(np.clip(1.0 - clip_pen * 0.6 - median_pen * 0.4, 0.0, 1.0))
    except Exception as e:
        print(f"  [exposure_score] fallback (0.5): {e}")
        return 0.5


def compute_sharpness_score(rgb_path: str, history_file: Optional[str] = None) -> float:
    """Laplacian variance, log-mapped with bootstrap percentile calibration.
    Falls back to fixed range lo=40, hi=180 (CG render typical) until 10+ samples.
    """
    try:
        import cv2
        img = np.array(Image.open(rgb_path).convert("L"))
        lap_var = float(cv2.Laplacian(img.astype(np.float32), cv2.CV_32F).var())

        history = []
        if history_file and os.path.exists(history_file):
            with open(history_file) as f:
                history = json.load(f)
        history.append(lap_var)
        if len(history) > 500:
            history = history[-500:]
        if history_file:
            os.makedirs(os.path.dirname(history_file), exist_ok=True)
            with open(history_file, "w") as f:
                json.dump(history, f)

        if len(history) >= 10:
            lo = float(np.percentile(history, 10))
            hi = float(np.percentile(history, 90))
        else:
            lo, hi = 40.0, 180.0  # CG render typical range

        score = (math.log1p(lap_var) - math.log1p(lo)) / (math.log1p(hi) - math.log1p(lo) + 1e-6)
        return float(np.clip(score, 0.0, 1.0))
    except Exception as e:
        print(f"  [sharpness_score] fallback (0.5): {e}")
        return 0.5


def compute_framing_score(rgb_path: str, mask_path: Optional[str] = None) -> float:
    """center_score * 0.45 + size_score * 0.35 + (1-cutoff) * 0.20."""
    try:
        import cv2
        img = np.array(Image.open(rgb_path).convert("RGB"))
        h, w = img.shape[:2]

        if mask_path and os.path.exists(mask_path):
            mask = np.array(Image.open(mask_path).convert("L"))
            binary = (mask > 127).astype(np.uint8)
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            _, binary = cv2.threshold(gray, 30, 1, cv2.THRESH_BINARY)

        if binary.sum() == 0:
            return 0.5

        rows = np.any(binary, axis=1)
        cols = np.any(binary, axis=0)
        row_idx = np.where(rows)[0]
        col_idx = np.where(cols)[0]
        rmin, rmax = int(row_idx[0]), int(row_idx[-1])
        cmin, cmax = int(col_idx[0]), int(col_idx[-1])

        obj_cx = (cmin + cmax) / 2.0
        obj_cy = (rmin + rmax) / 2.0
        d = math.hypot(obj_cx - w / 2.0, obj_cy - h / 2.0)
        center_score = 1.0 - d / (math.hypot(w, h) / 2.0 + 1e-8)

        area_frac = float(binary.sum()) / (h * w)
        if area_frac < 0.15:
            size_score = area_frac / 0.15
        elif area_frac > 0.65:
            size_score = 1.0 - (area_frac - 0.65) / 0.35
        else:
            size_score = 1.0

        border = 5
        cutoff = (0.25 * int(rmin < border) + 0.25 * int(rmax > h - border) +
                  0.25 * int(cmin < border) + 0.25 * int(cmax > w - border))

        score = 0.45 * center_score + 0.35 * max(0.0, size_score) + 0.20 * (1.0 - cutoff)
        return float(np.clip(score, 0.0, 1.0))
    except Exception as e:
        print(f"  [framing_score] fallback (0.5): {e}")
        return 0.5


def compute_cv_metrics(rgb_path: str, mask_path: Optional[str], history_file: Optional[str] = None) -> dict:
    """Compute CV quality metrics. When mask is unavailable, only use exposure+sharpness."""
    mask_available = bool(mask_path and os.path.exists(mask_path))

    es = compute_exposure_score(rgb_path, mask_path if mask_available else None)
    ss = compute_sharpness_score(rgb_path, history_file)

    if mask_available:
        ms = compute_mask_score(mask_path)
        fs = compute_framing_score(rgb_path, mask_path)
        cv = (CV_W["mask_score"] * ms + CV_W["exposure_score"] * es
              + CV_W["sharpness_score"] * ss + CV_W["framing_score"] * fs)
        return {
            "mask_score":      round(ms, 4),
            "exposure_score":  round(es, 4),
            "sharpness_score": round(ss, 4),
            "framing_score":   round(fs, 4),
            "cv_score":        round(cv, 4),
            "mask_available":  True,
        }
    else:
        # No mask: exposure(0.556) + sharpness(0.444), framing/mask unreliable
        cv = CV_W_NOMASK["exposure_score"] * es + CV_W_NOMASK["sharpness_score"] * ss
        return {
            "mask_score":      None,
            "exposure_score":  round(es, 4),
            "sharpness_score": round(ss, 4),
            "framing_score":   None,
            "cv_score":        round(cv, 4),
            "mask_available":  False,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Hybrid Score
# ─────────────────────────────────────────────────────────────────────────────

def compute_hybrid_score(vlm_scores: dict, cv_metrics: dict) -> tuple:
    """Returns (hybrid_score, route)."""
    # VLM score: ordinal 1-5 -> 0-1
    s_vlm = sum(VLM_W[k] * (vlm_scores.get(k, 3) - 1) / 4.0 for k in VLM_W)
    s_cv  = cv_metrics["cv_score"]
    hybrid = 0.50 * s_vlm + 0.50 * s_cv

    # Hard gates
    if vlm_scores.get("object_integrity", 3) <= 2:
        hybrid = min(hybrid, 0.69)
    # Mask gate: only when mask is available (Fix 1)
    if cv_metrics.get("mask_available") and cv_metrics.get("mask_score") is not None:
        if cv_metrics["mask_score"] < 0.60:
            hybrid = min(hybrid, 0.74)
    if cv_metrics["exposure_score"] < 0.55:
        hybrid = min(hybrid, 0.74)

    if hybrid >= 0.80:
        route = "pass"
    elif hybrid >= 0.55:
        route = "needs_fix"
    else:
        route = "reject"

    return round(hybrid, 4), route


# ─────────────────────────────────────────────────────────────────────────────
# v6: Reference image resolution
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_reference_image(obj_id, profile_cfg, renders_dir):
    """Resolve reference image path, return absolute path or None."""
    ref_dir = profile_cfg.get("reference_images_dir", None) if profile_cfg else None

    if ref_dir is None:
        # Default: infer from renders_dir sibling directory
        # renders_dir = /aaaidata/.../pipeline/data/renders
        # -> ref_dir = /aaaidata/.../pipeline/data/images
        ref_dir = os.path.join(os.path.dirname(renders_dir), "images")

    if not os.path.isabs(ref_dir):
        # Relative path: resolve relative to DATA_BUILD_ROOT
        ref_dir = os.path.join(DATA_BUILD_ROOT, ref_dir)

    ref_path = os.path.join(ref_dir, f"{obj_id}.png")
    return ref_path if os.path.exists(ref_path) else None


# ─────────────────────────────────────────────────────────────────────────────
# VLM inference
# ─────────────────────────────────────────────────────────────────────────────

_vlm_model = None
_vlm_processor = None


def load_vlm(device: str = "cuda:0"):
    global _vlm_model, _vlm_processor
    if _vlm_model is not None:
        return _vlm_model, _vlm_processor

    print(f"[VLM] Loading Qwen3-VL from {VLM_MODEL_PATH} -> {device}")
    from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor
    import torch

    _vlm_processor = Qwen3VLProcessor.from_pretrained(VLM_MODEL_PATH, trust_remote_code=True)
    _vlm_model = Qwen3VLForConditionalGeneration.from_pretrained(
        VLM_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    _vlm_model.eval()
    print("[VLM] Model loaded.")
    return _vlm_model, _vlm_processor


def _img_to_b64(path: str, max_size: int = 512) -> str:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def _build_prompt(sample_id: str, round_idx: int, active_group: str,
                  has_prev: bool, az: int, el: int,
                  prompt_appendix: str = "",
                  issue_tags_whitelist=None,
                  has_reference: bool = False) -> tuple:
    effective_tags = issue_tags_whitelist if issue_tags_whitelist else ISSUE_TAGS
    allowed_issues = ", ".join(effective_tags)
    allowed_actions_in_group = []
    if os.path.exists(ACTION_SPACE_PATH):
        with open(ACTION_SPACE_PATH) as f:
            aspace = json.load(f)
        allowed_actions_in_group = list(aspace["groups"].get(active_group, {}).get("actions", {}).keys())
        # Fix 6: Include compound actions belonging to this group
        for cname, cdef in aspace.get("compound_actions", {}).items():
            if cdef.get("group") == active_group:
                allowed_actions_in_group.append(cname)
    if not allowed_actions_in_group:
        allowed_actions_in_group = ["NO_OP"]
    allowed_actions_str = ", ".join(["NO_OP"] + allowed_actions_in_group)

    system_msg = (
        "You are a strict dataset render quality inspector. "
        "Your job is to assess rendered 3D object images for use in machine learning datasets. "
        "Focus exclusively on technical render quality. Be conservative and objective. "
        "Return ONLY a valid JSON object, no other text, no markdown fences."
    )

    # v6: Build image description with reference image context
    image_desc_parts = []
    img_idx = 1
    image_desc_parts.append(f"Image {img_idx} is the CURRENT render (view az={az}, el={el}).")
    img_idx += 1
    if has_reference:
        image_desc_parts.append(f"Image {img_idx} is the REFERENCE image (original T2I output, ground truth appearance).")
        img_idx += 1
    if has_prev:
        image_desc_parts.append(f"Image {img_idx} is the PREVIOUS render (round {round_idx-1}).")
    elif not has_reference:
        image_desc_parts.append("Only one image is provided (no previous render or reference for comparison).")
    pairwise_instr = " ".join(image_desc_parts)

    score_guide = (
        "Scoring guide (1=very bad, 2=poor, 3=acceptable, 4=good, 5=excellent):\n"
        "  lighting: overall illumination quality, no harsh shadows, good contrast\n"
        "    If lighting < 4, also report lighting_diagnosis:\n"
        "      flat_no_rim: no rim/edge separation from background\n"
        "      flat_low_contrast: low tonal range, washed out\n"
        "      underexposed_global: entire image too dark\n"
        "      underexposed_shadow: shadow areas too deep/crushed\n"
        "      harsh_shadow_key: hard, distracting key light shadows\n"
        "      good: lighting >= 4\n"
        "  object_integrity: object is complete, no clipping, correct geometry\n"
        "  composition: object centering, appropriate scale, not cut off\n"
        "  render_quality_semantic: no artifacts, blurriness, or texture issues\n"
        "  overall: holistic dataset usability score"
    )

    # v6: Additional diagnostic fields description
    ref_diag_guide = ""
    if has_reference:
        ref_diag_guide = (
            "\n\nAdditional diagnostics (compare current render with the REFERENCE image):\n"
            "  structure_consistency: Does the rendered 3D object structure match the reference? "
            "(good=match, minor_mismatch=small differences, major_mismatch=significant structural deviation)\n"
            "  color_consistency: Is the color palette (hue, saturation, brightness) consistent with reference? "
            "(good=match, minor_shift=slight color difference, major_shift=significant color deviation)\n"
            "  physics_consistency: Is the object physically plausible (no floating, no ground intersection, "
            "correct shadow/contact)? (good=plausible, minor_issue=slight implausibility, major_issue=severe problem)"
        )
    else:
        ref_diag_guide = (
            "\n\nAdditional diagnostics (assess based on the render alone):\n"
            "  structure_consistency: Is the 3D object structurally coherent? "
            "(good=coherent, minor_mismatch=small issues, major_mismatch=significant problems)\n"
            "  color_consistency: Are colors natural and balanced? "
            "(good=balanced, minor_shift=slight issue, major_shift=significant issue)\n"
            "  physics_consistency: Is the object physically plausible? "
            "(good=plausible, minor_issue=slight issue, major_issue=severe problem)"
        )

    # v6: Extended JSON output template with 3 new fields
    json_template = (
        "{"
        "\"schema_version\": \"vlm_review_v1\","
        "\"sample_id\": \"" + sample_id + "\","
        "\"round_idx\": " + str(round_idx) + ","
        "\"vlm_route\": \"<pass|needs_fix|reject>\","
        "\"scores\": {\"lighting\": <1-5>, \"object_integrity\": <1-5>, \"composition\": <1-5>, \"render_quality_semantic\": <1-5>, \"overall\": <1-5>},"
        "\"confidence\": {\"lighting\": \"<low|medium|high>\", \"object_integrity\": \"<low|medium|high>\", \"composition\": \"<low|medium|high>\", \"render_quality_semantic\": \"<low|medium|high>\", \"overall\": \"<low|medium|high>\"},"
        "\"issue_tags\": [\"<tag1>\"],"
        "\"suggested_actions\": [\"<action1>\"],"
        "\"lighting_diagnosis\": \"<flat_no_rim|flat_low_contrast|underexposed_global|underexposed_shadow|harsh_shadow_key|good>\","
        "\"structure_consistency\": \"<good|minor_mismatch|major_mismatch>\","
        "\"color_consistency\": \"<good|minor_shift|major_shift>\","
        "\"physics_consistency\": \"<good|minor_issue|major_issue>\","
        "\"pairwise_vs_prev\": {\"available\": " + ("true" if has_prev else "false") + ", \"winner\": \"<current|previous|tie|none>\", \"lighting\": \"<better|same|worse|na>\", \"object_integrity\": \"<better|same|worse|na>\", \"composition\": \"<better|same|worse|na>\", \"render_quality_semantic\": \"<better|same|worse|na>\"}"
        "}"
    )

    user_msg = (
        f"Analyze this rendered 3D object image.\n"
        f"sample_id={sample_id}, round={round_idx}, view=az{az:03d}_el{el:+03d}, "
        f"active_fix_group={active_group}\n"
        f"{pairwise_instr}\n\n"
        f"{score_guide}"
        f"{ref_diag_guide}\n\n"
        f"Allowed issue_tags (choose up to 3 that clearly apply): {allowed_issues}\n"
        f"Allowed suggested_actions for group '{active_group}' (choose up to 2): {allowed_actions_str}\n\n"
        f"Output ONLY this JSON object (replace <...> with your values):\n"
        f"{json_template}"
    )
    if prompt_appendix:
        user_msg = user_msg + "\n\n" + prompt_appendix
    return system_msg, user_msg


def _extract_json(text: str) -> Optional[dict]:
    """Extract first valid JSON object from model output."""
    # Try direct parse
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    # Find JSON block
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass
    return None


def _validate_review(obj: dict, sample_id: str, round_idx: int) -> dict:
    """Coerce and fill missing fields with safe defaults."""
    obj.setdefault("schema_version", "vlm_review_v1")
    obj.setdefault("sample_id", sample_id)
    obj.setdefault("round_idx", round_idx)
    obj.setdefault("vlm_route", "needs_fix")

    scores_default = {"lighting": 3, "object_integrity": 3, "composition": 3,
                      "render_quality_semantic": 3, "overall": 3}
    scores = obj.get("scores", {})
    for k, v in scores_default.items():
        if k not in scores or not isinstance(scores[k], int):
            scores[k] = v
        else:
            scores[k] = max(1, min(5, int(scores[k])))
    obj["scores"] = scores

    conf_default = {k: "medium" for k in scores_default}
    conf = obj.get("confidence", {})
    for k in conf_default:
        if k not in conf or conf[k] not in ("low", "medium", "high"):
            conf[k] = "medium"
    obj["confidence"] = conf

    tags = obj.get("issue_tags", [])
    tags = [t for t in tags if t in ISSUE_TAGS][:3]
    if not tags:
        tags = ["none"]
    obj["issue_tags"] = tags

    actions = obj.get("suggested_actions", ["NO_OP"])
    actions = [a for a in actions if isinstance(a, str)][:2]
    # Fix 1: Hard-validate actions against action_space (filter stale/illegal names)
    try:
        with open(ACTION_SPACE_PATH) as _f:
            _aspace = json.load(_f)
        valid_actions = set()
        for grp_data in _aspace["groups"].values():
            valid_actions.update(grp_data["actions"].keys())
        valid_actions.update(_aspace.get("compound_actions", {}).keys())
        valid_actions.add("NO_OP")
        actions = [a for a in actions if a in valid_actions]
    except Exception:
        pass  # If action_space unavailable, skip validation
    if not actions:
        actions = ["NO_OP"]
    obj["suggested_actions"] = actions

    # Validate lighting_diagnosis (optional field, default "good")
    diagnosis = obj.get("lighting_diagnosis", "good")
    if diagnosis not in LIGHTING_DIAGNOSIS_VALUES:
        diagnosis = "good"
    obj["lighting_diagnosis"] = diagnosis

    # v6: Validate new diagnostic fields (optional, default "good")
    sc = obj.get("structure_consistency", "good")
    if sc not in STRUCTURE_VALUES:
        sc = "good"
    obj["structure_consistency"] = sc

    cc = obj.get("color_consistency", "good")
    if cc not in COLOR_VALUES:
        cc = "good"
    obj["color_consistency"] = cc

    pc = obj.get("physics_consistency", "good")
    if pc not in PHYSICS_VALUES:
        pc = "good"
    obj["physics_consistency"] = pc

    ppv = obj.get("pairwise_vs_prev", {})
    ppv.setdefault("available", False)
    ppv.setdefault("winner", "none")
    for k in ("lighting", "object_integrity", "composition", "render_quality_semantic"):
        ppv.setdefault(k, "na")
    obj["pairwise_vs_prev"] = ppv

    return obj


def run_vlm_review(rgb_path: str, sample_id: str, round_idx: int,
                   active_group: str, az: int, el: int,
                   prev_rgb_path: Optional[str] = None,
                   device: str = "cuda:0", max_retries: int = 3,
                   prompt_appendix: str = "",
                   issue_tags_whitelist=None,
                   reference_image_path: Optional[str] = None) -> dict:
    """Run VLM inference for one view. Returns validated review dict."""
    import torch

    model, processor = load_vlm(device)
    has_reference = reference_image_path is not None and os.path.exists(reference_image_path)
    system_msg, user_msg = _build_prompt(sample_id, round_idx, active_group,
                                          prev_rgb_path is not None, az, el,
                                          prompt_appendix=prompt_appendix,
                                          issue_tags_whitelist=issue_tags_whitelist,
                                          has_reference=has_reference)

    # Build message content: current render first, then reference, then previous
    content = [{"type": "image", "image": rgb_path}]
    if has_reference:
        content.append({"type": "image", "image": reference_image_path})
    if prev_rgb_path and os.path.exists(prev_rgb_path):
        content.append({"type": "image", "image": prev_rgb_path})
    content.append({"type": "text", "text": user_msg})

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user",   "content": content},
    ]

    for attempt in range(max_retries):
        try:
            text_input = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            # Use qwen_vl_utils for image processing
            from qwen_vl_utils import process_vision_info
            image_inputs, video_inputs = process_vision_info(messages)

            inputs = processor(
                text=[text_input],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                )

            generated = output_ids[0][inputs.input_ids.shape[1]:]
            text_out = processor.decode(generated, skip_special_tokens=True)
            parsed = _extract_json(text_out)

            if parsed is not None:
                return _validate_review(parsed, sample_id, round_idx)

            print(f"  [VLM] attempt {attempt+1}: JSON parse failed, raw='{text_out[:120]}'")
        except Exception as e:
            print(f"  [VLM] attempt {attempt+1} error: {e}")

    # Fallback: return neutral review
    print(f"  [VLM] all {max_retries} attempts failed, returning neutral review")
    return _validate_review({}, sample_id, round_idx)


# ─────────────────────────────────────────────────────────────────────────────
# Per-object review (aggregates multiple views)
# ─────────────────────────────────────────────────────────────────────────────

def review_object(obj_id: str, renders_dir: str, output_dir: str,
                  round_idx: int = 0, active_group: str = "lighting",
                  prev_renders_dir: Optional[str] = None,
                  device: str = "cuda:0",
                  history_file: Optional[str] = None,
                  prompt_appendix: str = "",
                  issue_tags_whitelist=None,
                  reference_image_path: Optional[str] = None,
                  profile_cfg: dict = None) -> dict:
    """
    Review all representative views for obj_id. Returns aggregated result.
    Saves per-view reviews to output_dir/{obj_id}_round{round_idx}_{az}_{el}.json
    and aggregated to output_dir/{obj_id}_round{round_idx}_agg.json.
    """
    obj_renders = os.path.join(renders_dir, obj_id)
    os.makedirs(output_dir, exist_ok=True)

    # v6: Resolve reference image if not explicitly provided
    if reference_image_path is None:
        reference_image_path = _resolve_reference_image(obj_id, profile_cfg, renders_dir)
    if reference_image_path:
        print(f"  [{obj_id}] Reference image: {reference_image_path}")

    per_view_results = []

    for az, el in REP_VIEWS:
        el_str = f"{el:+03d}"
        fname  = f"az{az:03d}_el{el_str}.png"
        rgb_path = os.path.join(obj_renders, fname)
        if not os.path.exists(rgb_path):
            print(f"  [review] View not found, skip: {rgb_path}")
            continue

        prev_rgb = None
        if prev_renders_dir:
            prev_rgb = os.path.join(prev_renders_dir, obj_id, fname)
            if not os.path.exists(prev_rgb):
                prev_rgb = None

        # CV metrics (no mask in pipeline renders)
        cv = compute_cv_metrics(rgb_path, mask_path=None, history_file=history_file)

        # VLM review
        sample_id = f"{obj_id}_az{az:03d}_el{el_str}"
        vlm_review = run_vlm_review(
            rgb_path, sample_id, round_idx, active_group, az, el,
            prev_rgb_path=prev_rgb, device=device,
            prompt_appendix=prompt_appendix,
            issue_tags_whitelist=issue_tags_whitelist,
            reference_image_path=reference_image_path,
        )

        hybrid, route = compute_hybrid_score(vlm_review["scores"], cv)
        vlm_review["cv_metrics"]    = cv
        vlm_review["hybrid_score"]  = hybrid
        vlm_review["hybrid_route"]  = route

        # Save per-view
        view_out = os.path.join(output_dir, f"{obj_id}_r{round_idx:02d}_az{az:03d}_el{el_str}.json")
        with open(view_out, "w") as f:
            json.dump(vlm_review, f, indent=2)

        per_view_results.append(vlm_review)
        print(f"  [{obj_id}] az={az:3d} el={el:+3d} | hybrid={hybrid:.3f} route={route}")

    if not per_view_results:
        print(f"  [review] No views found for {obj_id}")
        return {}

    # Aggregate: mean scores, union issue_tags, majority route
    agg_vlm_scores = {}
    for k in ("lighting", "object_integrity", "composition", "render_quality_semantic", "overall"):
        agg_vlm_scores[k] = round(float(np.mean([r["scores"][k] for r in per_view_results])), 2)

    agg_cv = {}
    for k in ("exposure_score", "sharpness_score", "cv_score"):
        agg_cv[k] = round(float(np.mean([r["cv_metrics"][k] for r in per_view_results])), 4)
    # mask_score / framing_score only when available
    for k in ("mask_score", "framing_score"):
        vals = [r["cv_metrics"][k] for r in per_view_results if r["cv_metrics"].get(k) is not None]
        agg_cv[k] = round(float(np.mean(vals)), 4) if vals else None
    agg_cv["mask_available"] = any(r["cv_metrics"].get("mask_available", False) for r in per_view_results)

    hybrid_scores = [r["hybrid_score"] for r in per_view_results]
    agg_hybrid = round(float(np.mean(hybrid_scores)), 4)
    worst_hybrid = round(float(np.min(hybrid_scores)), 4)

    # Aggregate hybrid with penalty for worst view (dataset quality = min quality view)
    final_hybrid = round(0.7 * agg_hybrid + 0.3 * worst_hybrid, 4)

    routes = [r["hybrid_route"] for r in per_view_results]
    route_counts = {r: routes.count(r) for r in set(routes)}
    final_route = max(route_counts, key=route_counts.get)

    # Collect most frequent issue_tags
    all_tags = []
    for r in per_view_results:
        all_tags.extend(r.get("issue_tags", []))
    all_tags = [t for t in all_tags if t != "none"]
    tag_counts = {}
    for t in all_tags:
        tag_counts[t] = tag_counts.get(t, 0) + 1
    top_tags = sorted(tag_counts, key=tag_counts.get, reverse=True)[:3] or ["none"]

    # Collect suggested actions
    all_actions = []
    for r in per_view_results:
        all_actions.extend(r.get("suggested_actions", []))
    action_counts = {}
    for a in all_actions:
        action_counts[a] = action_counts.get(a, 0) + 1
    top_actions = sorted(action_counts, key=action_counts.get, reverse=True)[:2] or ["NO_OP"]

    # Aggregate lighting_diagnosis (majority vote)
    diag_counts = _Counter(r.get("lighting_diagnosis", "good") for r in per_view_results)
    agg_diagnosis = diag_counts.most_common(1)[0][0]

    # v6: Aggregate structure_consistency (worst-case / any-view)
    struct_values = [r.get("structure_consistency", "good") for r in per_view_results]
    if "major_mismatch" in struct_values:
        agg_structure = "major_mismatch"
    elif "minor_mismatch" in struct_values:
        agg_structure = "minor_mismatch"
    else:
        agg_structure = "good"

    # v6: Aggregate physics_consistency (worst-case / any-view)
    physics_values = [r.get("physics_consistency", "good") for r in per_view_results]
    if "major_issue" in physics_values:
        agg_physics = "major_issue"
    elif "minor_issue" in physics_values:
        agg_physics = "minor_issue"
    else:
        agg_physics = "good"

    # v6: Aggregate color_consistency (majority vote)
    color_counts = _Counter(r.get("color_consistency", "good") for r in per_view_results)
    agg_color = color_counts.most_common(1)[0][0]

    # Per-view CV summary for downstream checks
    per_view_cv = [
        {
            "az": az,
            "el": el,
            "exposure_score": r["cv_metrics"]["exposure_score"],
            "sharpness_score": r["cv_metrics"]["sharpness_score"],
        }
        for r, (az, el) in zip(per_view_results, REP_VIEWS[:len(per_view_results)])
    ]

    aggregated = {
        "schema_version":    "vlm_review_v1",
        "obj_id":            obj_id,
        "round_idx":         round_idx,
        "active_group":      active_group,
        "num_views_reviewed": len(per_view_results),
        "agg_vlm_scores":    agg_vlm_scores,
        "agg_cv_metrics":    agg_cv,
        "hybrid_score":      final_hybrid,
        "worst_view_hybrid": worst_hybrid,
        "hybrid_route":      final_route,
        "issue_tags":        top_tags,
        "suggested_actions": top_actions,
        "lighting_diagnosis": agg_diagnosis,
        "lighting_diagnosis_stats": dict(diag_counts),
        "structure_consistency": agg_structure,
        "color_consistency": agg_color,
        "physics_consistency": agg_physics,
        "per_view_cv":       per_view_cv,
        "per_view":          [
            {"az": r["sample_id"].split("_az")[1].split("_")[0],
             "el": r["sample_id"].split("_el")[1] if "_el" in r["sample_id"] else "0",
             "hybrid_score": r["hybrid_score"],
             "route": r["hybrid_route"]}
            for r in per_view_results
        ],
    }

    agg_out = os.path.join(output_dir, f"{obj_id}_r{round_idx:02d}_agg.json")
    with open(agg_out, "w") as f:
        json.dump(aggregated, f, indent=2)
    print(f"  [{obj_id}] round={round_idx} | final_hybrid={final_hybrid:.3f} | route={final_route} | tags={top_tags} | diagnosis={agg_diagnosis} | struct={agg_structure} | color={agg_color} | physics={agg_physics}")
    return aggregated


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Stage 5.5: VLM render review + CV metrics")
    p.add_argument("--renders-dir",  required=True, help="Root dir with {obj_id}/az*.png")
    p.add_argument("--obj-id",       required=True, help="Object ID to review (e.g. obj_001)")
    p.add_argument("--output-dir",   required=True, help="Directory to save review JSONs")
    p.add_argument("--round-idx",    type=int, default=0)
    p.add_argument("--active-group", default="lighting",
                   choices=["lighting", "camera", "object", "scene", "material"])
    p.add_argument("--prev-renders-dir", default=None,
                   help="Previous-round renders dir for pairwise comparison")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--history-file", default=None,
                   help="JSON file for sharpness rolling percentile history")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = review_object(
        obj_id=args.obj_id,
        renders_dir=args.renders_dir,
        output_dir=args.output_dir,
        round_idx=args.round_idx,
        active_group=args.active_group,
        prev_renders_dir=args.prev_renders_dir,
        device=args.device,
        history_file=args.history_file,
    )
    print(f"\n=== Review summary: hybrid={result.get('hybrid_score','N/A')}, route={result.get('hybrid_route','N/A')} ===")
