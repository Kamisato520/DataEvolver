"""
run_scene_evolution_loop.py - bounded scene-aware render evolution loop.
"""

import argparse
import copy
import json
import os
import subprocess
import sys
from collections import Counter
from typing import Optional

import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "pipeline"))

from stage5_5_vlm_review import review_object  # noqa: E402
from stage5_6_feedback_apply import (  # noqa: E402
    apply_feedback,
    default_control_state,
    load_action_space,
    save_control_state,
)


BLENDER_BIN = os.environ.get("BLENDER_BIN", "/home/wuwenzhuo/blender-4.24/blender")
SCENE_RENDER_SCRIPT = os.path.join(SCRIPT_DIR, "pipeline", "stage4_scene_render.py")
SCENE_TEMPLATE_PATH = os.path.join(SCRIPT_DIR, "configs", "scene_template.json")
SCENE_ACTION_SPACE_PATH = os.path.join(SCRIPT_DIR, "configs", "scene_action_space.json")

PROFILE_DEFAULTS = {
    "dataset_name": "scene_v7",
    "task_type": "scene_insert",
    "accept_threshold": 0.78,
    "reject_threshold": 0.35,
    "preserve_score_threshold": 0.72,
    "explore_threshold": 0.60,
    "stability_threshold": 0.03,
    "unstable_variance_limit": 0.10,
    "unstable_span_limit": 0.10,
    "mid_budget": 1,
    "low_budget": 2,
    "improve_eps": 0.01,
    "max_rounds": 5,
    "patience": 4,
    "review_mode": "scene_insert",
    "review_view_policy": "scene_qc_4",
    "reference_images_dir": "pipeline/data/images",
    "use_pseudo_reference": True,
    "pseudo_reference_dir": "pipeline/data/pseudo_references",
    "use_freeform_vlm_feedback": True,
    "enable_vlm_thinking": True,
    "prompt_appendix": "",
    "action_whitelist": None,
    "issue_tags_whitelist": None,
    "contact_gap_minor": 0.01,
    "contact_gap_major": 0.05,
    "penetration_minor": 0.01,
    "penetration_major": 0.05,
    "target_bbox_height_range": [0.3, 1.5],
    "zoning_score": "hybrid",
    "auto_retry_with_vlm_only": True,
    "initial_control_state_overrides": None,
    "locked_control_state_overrides": None,
}


def load_json(path: str, default=None):
    if not path or not os.path.exists(path):
        return default
    with open(path) as f:
        return json.load(f)


def load_profile(path):
    if path is None:
        return dict(PROFILE_DEFAULTS)
    with open(path) as f:
        data = json.load(f)
    merged = {**PROFILE_DEFAULTS, **data}
    if "unstable_variance_limit" not in merged and "unstable_span_limit" in merged:
        merged["unstable_variance_limit"] = merged["unstable_span_limit"]
    if "unstable_span_limit" not in merged and "unstable_variance_limit" in merged:
        merged["unstable_span_limit"] = merged["unstable_variance_limit"]
    return merged


def load_scene_template(path):
    return load_json(path, default={}) or {}


def discover_obj_ids(meshes_dir):
    return sorted(f[:-4] for f in os.listdir(meshes_dir) if f.endswith(".glb"))


def _resolve_reference_image(obj_id: str, profile_cfg: dict) -> Optional[str]:
    ref_dir = profile_cfg.get("reference_images_dir")
    if ref_dir is None:
        ref_dir = os.path.join(SCRIPT_DIR, "pipeline", "data", "images")
    if not os.path.isabs(ref_dir):
        ref_dir = os.path.join(SCRIPT_DIR, ref_dir)
    path = os.path.join(ref_dir, f"{obj_id}.png")
    return path if os.path.exists(path) else None


def _resolve_pseudo_reference_image(obj_id: str, profile_cfg: dict) -> Optional[str]:
    if not profile_cfg.get("use_pseudo_reference", False):
        return None
    ref_dir = profile_cfg.get("pseudo_reference_dir")
    if ref_dir is None:
        ref_dir = os.path.join(SCRIPT_DIR, "pipeline", "data", "pseudo_references")
    if not os.path.isabs(ref_dir):
        ref_dir = os.path.join(SCRIPT_DIR, ref_dir)
    candidates = [
        os.path.join(ref_dir, obj_id, "pseudo_reference.png"),
        os.path.join(ref_dir, f"{obj_id}.png"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def _resolve_segmented_reference(reference_image_path: Optional[str]) -> Optional[str]:
    if not reference_image_path or not os.path.exists(reference_image_path):
        return None
    norm = reference_image_path.replace("\\", "/")
    if "/images/" in norm:
        rgba_path = norm.replace("/images/", "/images_rgba/")
        if os.path.exists(rgba_path):
            return rgba_path
    return reference_image_path if os.path.exists(reference_image_path) else None


def _estimate_reference_rgb(reference_image_path: Optional[str]) -> Optional[list[float]]:
    source_path = _resolve_segmented_reference(reference_image_path)
    if not source_path or not os.path.exists(source_path):
        return None
    try:
        from PIL import Image
    except Exception:
        return None

    try:
        with Image.open(source_path) as im:
            rgba = im.convert("RGBA")
            arr = np.asarray(rgba).astype(np.float32) / 255.0
    except Exception:
        return None

    rgb = arr[..., :3]
    alpha = arr[..., 3]
    mask = alpha > 0.2
    if int(mask.sum()) < 32:
        mask = ~(rgb.min(axis=-1) > 0.94)
    if int(mask.sum()) < 32:
        mask = np.ones(rgb.shape[:2], dtype=bool)

    pixels = rgb[mask]
    if pixels.size == 0:
        return None

    mean_rgb = pixels.mean(axis=0)
    low_mid_rgb = np.quantile(pixels, 0.35, axis=0)
    ref_rgb = mean_rgb * 0.4 + low_mid_rgb * 0.6
    return [round(float(v), 4) for v in ref_rgb.tolist()]


def classify_score_zone(score, preserve_threshold, explore_threshold):
    if score >= preserve_threshold:
        return "high"
    if score >= explore_threshold:
        return "mid"
    return "low"


def _deep_merge_dict(base: dict, overrides: Optional[dict]) -> dict:
    result = copy.deepcopy(base)
    if not overrides:
        return result
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge_dict(result.get(key, {}), value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _apply_control_overrides(state: dict, overrides: Optional[dict]) -> dict:
    return _deep_merge_dict(state, overrides)


def _aggregate_baseline_diagnostics(diag_list: list) -> dict:
    if not diag_list:
        return {}
    struct_vals = [d.get("structure_consistency", "good") for d in diag_list]
    if "major_mismatch" in struct_vals:
        agg_structure = "major_mismatch"
    elif "minor_mismatch" in struct_vals:
        agg_structure = "minor_mismatch"
    else:
        agg_structure = "good"

    phys_vals = [d.get("physics_consistency", "good") for d in diag_list]
    if "major_issue" in phys_vals:
        agg_physics = "major_issue"
    elif "minor_issue" in phys_vals:
        agg_physics = "minor_issue"
    else:
        agg_physics = "good"

    color_vals = [d.get("color_consistency", "good") for d in diag_list]
    agg_color = Counter(color_vals).most_common(1)[0][0]

    light_vals = [d.get("lighting_diagnosis", "good") for d in diag_list]
    agg_lighting = Counter(light_vals).most_common(1)[0][0]

    latest_prog = {}
    for d in diag_list:
        prog = d.get("programmatic_physics") or {}
        for k, v in prog.items():
            latest_prog[k] = v

    return {
        "structure_consistency": agg_structure,
        "physics_consistency": agg_physics,
        "color_consistency": agg_color,
        "lighting_diagnosis": agg_lighting,
        "programmatic_physics": latest_prog,
    }


def _determine_active_group(review: dict, profile: dict) -> str:
    prog = review.get("programmatic_physics") or {}
    bbox_height = prog.get("bbox_height")
    target_range = profile.get("target_bbox_height_range", [0.3, 1.5])
    scale_issue = False
    if bbox_height is not None:
        scale_issue = float(bbox_height) < float(target_range[0]) or float(bbox_height) > float(target_range[1])

    if prog.get("contact_gap", 0.0) > profile.get("contact_gap_minor", 0.01):
        return "object"
    if prog.get("penetration_depth", 0.0) > profile.get("penetration_minor", 0.01):
        return "object"
    if scale_issue:
        return "object"
    if review.get("physics_consistency", "good") != "good":
        return "object"

    diag = review.get("lighting_diagnosis", "good")
    if diag in ("scene_light_mismatch", "underexposed_global", "flat_low_contrast"):
        return "scene"
    if diag in ("shadow_missing", "harsh_shadow_key"):
        return "lighting"
    if review.get("color_consistency", "good") != "good":
        return "material"
    return "scene"


def _action_group_for_name(action_name: str, aspace: dict) -> Optional[str]:
    if not action_name or action_name == "NO_OP":
        return None
    for group_name, group_data in aspace.get("groups", {}).items():
        if action_name in group_data.get("actions", {}):
            return group_name
    compound = aspace.get("compound_actions", {}).get(action_name)
    if compound:
        return compound.get("group")
    return None


def render_scene_state(obj_id, meshes_dir, output_dir, control_state,
                       scene_template_path, blender_bin=BLENDER_BIN,
                       resolution=512, engine="EEVEE",
                       reference_image_path: Optional[str] = None):
    glb_path = os.path.join(meshes_dir, f"{obj_id}.glb")
    if not os.path.exists(glb_path):
        print(f"  [scene-rerender] GLB not found: {glb_path}")
        return False

    os.makedirs(output_dir, exist_ok=True)
    control_state = copy.deepcopy(control_state)
    ref_rgb = _estimate_reference_rgb(reference_image_path)
    if ref_rgb is not None:
        control_state.setdefault("material", {})
        control_state["material"]["reference_rgb"] = ref_rgb
        control_state["material"]["reference_image_path"] = reference_image_path
    cs_path = os.path.join(output_dir, f"_control_{obj_id}.json")
    with open(cs_path, "w") as f:
        json.dump(control_state, f, indent=2)

    cmd = [
        blender_bin, "-b", "-P", SCENE_RENDER_SCRIPT, "--",
        "--input-dir", meshes_dir,
        "--output-dir", output_dir,
        "--obj-id", obj_id,
        "--resolution", str(resolution),
        "--engine", engine,
        "--control-state", cs_path,
        "--scene-template", scene_template_path,
    ]
    print(f"  [scene-rerender] {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        if result.returncode != 0:
            print(f"  [scene-rerender] Blender failed:\n{result.stderr[-1200:]}")
            return False
        return True
    except subprocess.TimeoutExpired:
        print("  [scene-rerender] Blender timed out")
        return False
    except Exception as e:
        print(f"  [scene-rerender] Error: {e}")
        return False


def evolve_object(obj_id, meshes_dir, evolution_dir, scene_template, profile,
                  device="cuda:0", blender_bin=BLENDER_BIN, max_rounds=None):
    import statistics as _stats

    accept_th = float(profile["accept_threshold"])
    reject_th = float(profile["reject_threshold"])
    preserve_th = float(profile["preserve_score_threshold"])
    explore_th = float(profile["explore_threshold"])
    stability_th = float(profile["stability_threshold"])
    unstable_span = float(profile.get("unstable_span_limit", profile.get("unstable_variance_limit", 0.05)))
    mid_budget = int(profile["mid_budget"])
    low_budget = int(profile["low_budget"])
    improve_eps = float(profile["improve_eps"])
    max_rounds = int(profile.get("max_rounds", max_rounds or 5))
    prompt_appendix = profile.get("prompt_appendix", "")
    issue_tags_wl = profile.get("issue_tags_whitelist")
    review_mode = profile.get("review_mode", "scene_insert")
    zoning_score_key = profile.get("zoning_score", "hybrid")
    initial_overrides = profile.get("initial_control_state_overrides")
    locked_overrides = profile.get("locked_control_state_overrides")

    obj_dir = os.path.join(evolution_dir, obj_id)
    reviews_dir = os.path.join(obj_dir, "reviews")
    states_dir = os.path.join(obj_dir, "states")
    sharp_hist = os.path.join(obj_dir, "sharpness_history.json")
    os.makedirs(reviews_dir, exist_ok=True)
    os.makedirs(states_dir, exist_ok=True)

    aspace = load_action_space(SCENE_ACTION_SPACE_PATH)
    baseline_state = default_control_state(aspace)
    baseline_state = _apply_control_overrides(baseline_state, initial_overrides)
    baseline_state = _apply_control_overrides(baseline_state, locked_overrides)
    best_state = copy.deepcopy(baseline_state)
    best_score = 0.0
    action_blacklist = set()
    state_log = []
    reference_image_path = _resolve_reference_image(obj_id, profile)
    pseudo_reference_path = _resolve_pseudo_reference_image(obj_id, profile)
    rep_views = scene_template.get("qc_views")
    render_resolution = int(scene_template.get("render_resolution", 512))
    render_engine = str(scene_template.get("render_engine", "EEVEE")).upper()

    def _render_review_probe(label: str, round_idx: int, control_state: dict, active_group: str, prev_dir=None, history_file=None):
        control_state = _apply_control_overrides(control_state, locked_overrides)
        render_base = os.path.join(obj_dir, label)
        ok = render_scene_state(
            obj_id=obj_id,
            meshes_dir=meshes_dir,
            output_dir=render_base,
            control_state=control_state,
            scene_template_path=scene_template["__path__"],
            blender_bin=blender_bin,
            resolution=render_resolution,
            engine=render_engine,
            reference_image_path=reference_image_path,
        )
        obj_render_dir = os.path.join(render_base, obj_id)
        has_renders = os.path.isdir(obj_render_dir) and any(name.endswith(".png") and not name.endswith("_mask.png") for name in os.listdir(obj_render_dir))
        if not (ok and has_renders):
            return None, render_base
        rev = review_object(
            obj_id=obj_id,
            renders_dir=render_base,
            output_dir=reviews_dir,
            round_idx=round_idx,
            active_group=active_group,
            prev_renders_dir=prev_dir,
            device=device,
            history_file=history_file,
            prompt_appendix=prompt_appendix,
            issue_tags_whitelist=issue_tags_wl,
            reference_image_path=reference_image_path,
            pseudo_reference_path=pseudo_reference_path,
            profile_cfg=profile,
            review_mode=review_mode,
            action_space_path=SCENE_ACTION_SPACE_PATH,
            rep_views=rep_views,
        )
        return rev, render_base

    probes = []
    probe_dirs = []
    for idx in range(2):
        rev, render_base = _render_review_probe(f"probe{idx}_renders", idx, baseline_state, "scene")
        if rev is None:
            return {
                "obj_id": obj_id,
                "final_hybrid": 0.0,
                "accepted": False,
                "exit_reason": "rejected_render_hard_fail",
                "baseline_zone": "low",
            }
        probes.append(rev)
        probe_dirs.append(render_base)
        print(f"  [{obj_id}] Baseline p{idx}: hybrid={rev['hybrid_score']:.4f} vlm={rev['vlm_only_score']:.4f}")

    hybrid_scores = [r["hybrid_score"] for r in probes]
    vlm_scores = [r["vlm_only_score"] for r in probes]
    score_span = max(hybrid_scores) - min(hybrid_scores)

    if abs(hybrid_scores[0] - hybrid_scores[1]) > stability_th:
        rev, render_base = _render_review_probe("probe2_renders", 2, baseline_state, "scene")
        if rev is not None:
            probes.append(rev)
            probe_dirs.append(render_base)
            hybrid_scores.append(rev["hybrid_score"])
            vlm_scores.append(rev["vlm_only_score"])
            score_span = max(hybrid_scores) - min(hybrid_scores)
            print(f"  [{obj_id}] Baseline p2: hybrid={rev['hybrid_score']:.4f} vlm={rev['vlm_only_score']:.4f}")

    if len(hybrid_scores) == 3:
        confirmed_hybrid = _stats.median(hybrid_scores)
        confirmed_vlm = _stats.median(vlm_scores)
        if score_span > unstable_span:
            return {
                "obj_id": obj_id,
                "final_hybrid": confirmed_hybrid,
                "final_vlm_only": confirmed_vlm,
                "accepted": False,
                "exit_reason": "rejected_unstable_score",
                "baseline_zone": "low",
                "confirmed_score": confirmed_hybrid,
                "confirmed_vlm_only": confirmed_vlm,
            }
    else:
        confirmed_hybrid = sum(hybrid_scores) / len(hybrid_scores)
        confirmed_vlm = sum(vlm_scores) / len(vlm_scores)

    agg_diag = _aggregate_baseline_diagnostics(probes)
    combined_review = dict(probes[-1])
    combined_review.update(agg_diag)
    combined_review["confirmed_score"] = confirmed_hybrid
    combined_review["confirmed_vlm_only"] = confirmed_vlm

    zone_basis = confirmed_vlm if zoning_score_key == "vlm_only" else confirmed_hybrid
    zone = classify_score_zone(zone_basis, preserve_th, explore_th)
    combined_review["zone"] = zone
    combined_review["zoning_score_used"] = zoning_score_key

    best_state = copy.deepcopy(baseline_state)
    best_score = confirmed_hybrid

    baseline_review_path = os.path.join(reviews_dir, f"{obj_id}_baseline_scene_agg.json")
    with open(baseline_review_path, "w") as f:
        json.dump(combined_review, f, indent=2)

    def _write_result(exit_reason: str, final_score: float, final_vlm: float, attempts: int, updates: int):
        result = {
            "obj_id": obj_id,
            "final_hybrid": final_score,
            "final_vlm_only": final_vlm,
            "accepted": final_score >= accept_th,
            "exit_reason": exit_reason,
            "state_log": state_log,
            "confirmed_score": round(confirmed_hybrid, 4),
            "confirmed_vlm_only": round(confirmed_vlm, 4),
            "score_span": round(score_span, 6),
            "score_repeats": len(probes),
            "baseline_zone": zone,
            "zoning_score_used": zoning_score_key,
            "diagnostics": agg_diag,
            "probes_run": len(probes) + attempts,
            "updates_run": updates,
            "best_state_path": os.path.join(states_dir, "control_state_best.json"),
        }
        result["initial_control_state_overrides"] = initial_overrides
        result["locked_control_state_overrides"] = locked_overrides
        save_control_state(best_state, result["best_state_path"])
        result_path = os.path.join(obj_dir, "scene_evolution_result.json")
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[{obj_id}] Exit: {exit_reason} | final={final_score:.4f} | zone={zone}")
        return result

    if agg_diag.get("structure_consistency") == "major_mismatch":
        return _write_result("rejected_mesh", confirmed_hybrid, confirmed_vlm, 0, 0)
    if confirmed_hybrid >= accept_th:
        return _write_result("accepted_baseline", confirmed_hybrid, confirmed_vlm, 0, 0)
    if zone == "high":
        return _write_result("preserved_high_zone", confirmed_hybrid, confirmed_vlm, 0, 0)
    if confirmed_hybrid < reject_th:
        return _write_result("rejected_low_quality", confirmed_hybrid, confirmed_vlm, 0, 0)

    budget = mid_budget if zone == "mid" else low_budget
    active_group = _determine_active_group(combined_review, profile)
    advisor_actions = combined_review.get("advisor_actions") or []
    advisor_group = _action_group_for_name(advisor_actions[0], aspace) if advisor_actions else None
    if advisor_group:
        active_group = advisor_group
    print(f"  [{obj_id}] Zone={zone}, budget={budget}, active_group={active_group}, zoning_score={zoning_score_key}")

    attempts_made = 0
    updates_made = 0
    prev_render_dir = probe_dirs[0]

    for attempt_idx in range(min(budget, max_rounds)):
        attempt_state = copy.deepcopy(baseline_state)
        cs_path = os.path.join(states_dir, f"cs_attempt_{attempt_idx:02d}.json")
        next_cs_path = os.path.join(states_dir, f"cs_attempt_{attempt_idx:02d}_next.json")
        attempt_state = _apply_control_overrides(attempt_state, locked_overrides)
        save_control_state(attempt_state, cs_path)

        feedback = apply_feedback(
            review_json_path=baseline_review_path,
            control_state_path=cs_path,
            output_control_state_path=next_cs_path,
            round_idx=attempt_idx,
            history_json_path=os.path.join(states_dir, "history.json"),
            hybrid_score=confirmed_hybrid,
            action_blacklist=action_blacklist,
            preset_mode=True,
            active_group_override=active_group,
            action_space_path=SCENE_ACTION_SPACE_PATH,
            action_whitelist=advisor_actions or None,
        )

        action = feedback.get("action_taken", "NO_OP")
        attempts_made += 1

        if action == "NO_OP":
            state_log.append({
                "attempt": attempt_idx,
                "zone": zone,
                "action_taken": action,
                "hybrid_score": confirmed_hybrid,
                "score_delta": 0.0,
                "active_group": active_group,
                "action_blacklist": sorted(action_blacklist),
                "exit_reason": "mid_no_improve" if zone == "mid" else "low_exhausted",
            })
            return _write_result("mid_no_improve" if zone == "mid" else "low_exhausted", confirmed_hybrid, confirmed_vlm, attempts_made, updates_made)

        updates_made += 1
        attempt_state = feedback["control_state"]
        attempt_state = _apply_control_overrides(attempt_state, locked_overrides)
        render_label = f"attempt{attempt_idx + 1}_renders"
        attempt_review, attempt_render_dir = _render_review_probe(
            render_label,
            len(probes) + attempt_idx,
            attempt_state,
            active_group,
            prev_dir=prev_render_dir,
            history_file=sharp_hist,
        )

        if attempt_review is None:
            action_blacklist.add(action)
            if zone == "mid":
                return _write_result("rejected_render_hard_fail", confirmed_hybrid, confirmed_vlm, attempts_made, updates_made)
            continue

        attempt_score = attempt_review.get("hybrid_score", 0.0)
        attempt_vlm = attempt_review.get("vlm_only_score", 0.0)
        delta = attempt_score - confirmed_hybrid
        state_log.append({
            "attempt": attempt_idx,
            "zone": zone,
            "action_taken": action,
            "hybrid_score": attempt_score,
            "vlm_only_score": attempt_vlm,
            "score_delta": round(delta, 6),
            "active_group": active_group,
            "action_blacklist": sorted(action_blacklist),
            "exit_reason": "accepted_after_try" if delta > improve_eps else "attempt_failed",
        })

        if attempt_score > confirmed_hybrid + improve_eps:
            best_state = copy.deepcopy(attempt_state)
            best_score = attempt_score
            return _write_result("accepted_after_try", best_score, attempt_vlm, attempts_made, updates_made)

        action_blacklist.add(action)
        if zone == "mid":
            return _write_result("mid_no_improve", confirmed_hybrid, confirmed_vlm, attempts_made, updates_made)

    return _write_result("low_exhausted" if zone == "low" else "mid_no_improve", confirmed_hybrid, confirmed_vlm, attempts_made, updates_made)


def run_evolution(meshes_dir, output_dir, obj_ids=None, device="cuda:0",
                  blender_bin=BLENDER_BIN, profile=None, scene_template_path=SCENE_TEMPLATE_PATH,
                  allow_auto_retry=True):
    os.makedirs(output_dir, exist_ok=True)
    if obj_ids is None:
        obj_ids = discover_obj_ids(meshes_dir)
    prof = profile or dict(PROFILE_DEFAULTS)
    scene_template = load_scene_template(scene_template_path)
    scene_template["__path__"] = scene_template_path

    print(f"[SceneEvolution] Processing {len(obj_ids)} objects: {obj_ids}")
    results = {}
    for obj_id in obj_ids:
        try:
            results[obj_id] = evolve_object(
                obj_id=obj_id,
                meshes_dir=meshes_dir,
                evolution_dir=output_dir,
                scene_template=scene_template,
                profile=prof,
                device=device,
                blender_bin=blender_bin,
            )
        except Exception as e:
            print(f"[SceneEvolution] ERROR on {obj_id}: {e}")
            import traceback
            traceback.print_exc()
            results[obj_id] = {"obj_id": obj_id, "error": str(e), "accepted": False, "baseline_zone": "low"}

    accepted = sum(1 for r in results.values() if r.get("accepted"))
    final_scores = {k: v.get("final_hybrid", 0.0) for k, v in results.items()}
    summary = {
        "total": len(results),
        "accepted": accepted,
        "acceptance_rate": accepted / max(1, len(results)),
        "final_scores": final_scores,
        "results": results,
    }
    with open(os.path.join(output_dir, "scene_validation_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    baseline_zones = [r.get("baseline_zone") for r in results.values() if "error" not in r]
    if (
        allow_auto_retry
        and prof.get("auto_retry_with_vlm_only", True)
        and prof.get("zoning_score", "hybrid") == "hybrid"
        and len(obj_ids) <= 3
        and baseline_zones
        and all(z == "low" for z in baseline_zones)
    ):
        print("[SceneEvolution] All smoke baselines fell into low zone; retrying with zoning_score=vlm_only")
        retry_profile = dict(prof)
        retry_profile["zoning_score"] = "vlm_only"
        retry_profile["auto_retry_with_vlm_only"] = False
        retry_dir = os.path.join(output_dir, "retry_vlm_only")
        summary["vlm_only_retry"] = run_evolution(
            meshes_dir=meshes_dir,
            output_dir=retry_dir,
            obj_ids=obj_ids,
            device=device,
            blender_bin=blender_bin,
            profile=retry_profile,
            scene_template_path=scene_template_path,
            allow_auto_retry=False,
        )
        with open(os.path.join(output_dir, "scene_validation_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

    return summary


def parse_args():
    p = argparse.ArgumentParser(description="Scene-aware VLM render evolution loop")
    p.add_argument("--meshes-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--obj-ids", nargs="+", default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--blender", default=BLENDER_BIN, dest="blender_bin")
    p.add_argument("--profile", default=None)
    p.add_argument("--scene-template", default=SCENE_TEMPLATE_PATH)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    profile = load_profile(args.profile)
    summary = run_evolution(
        meshes_dir=args.meshes_dir,
        output_dir=args.output_dir,
        obj_ids=args.obj_ids,
        device=args.device,
        blender_bin=args.blender_bin,
        profile=profile,
        scene_template_path=args.scene_template,
    )
    print(
        f"\n=== Scene Summary: {summary['accepted']}/{summary['total']} accepted, "
        f"rate={summary['acceptance_rate']:.1%} ==="
    )
