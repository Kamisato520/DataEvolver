"""
run_agent_loop.py - Agent-in-the-loop scene render optimizer (v7.2)

Based on run_scene_evolution_loop.py. Key changes:
  - Simplified VLM prompt: natural language audit on 5 dimensions
  - VLM directly suggests parameter changes (no preset mapping)
  - Simple N-round loop (no zone/budget complexity)
  - Visual progression strip saved for each object: base -> loop1 -> ... -> final
"""

import argparse
import copy
import json
import os
import re
import subprocess
import sys
from typing import Optional

# --- Path setup: file lives in agent_loop_v72/, pipeline/ is in parent data_build/ ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR  = os.path.dirname(SCRIPT_DIR)   # .../data_build/
sys.path.insert(0, os.path.join(PARENT_DIR, "pipeline"))

from stage5_5_vlm_review import load_vlm           # noqa: E402
from stage5_6_feedback_apply import (              # noqa: E402
    default_control_state,
    load_action_space,
)

BLENDER_BIN          = os.environ.get("BLENDER_BIN", "/home/wuwenzhuo/blender-4.24/blender")
SCENE_RENDER_SCRIPT  = os.path.join(PARENT_DIR, "pipeline", "stage4_scene_render.py")
SCENE_TEMPLATE_PATH  = os.path.join(PARENT_DIR, "configs", "scene_template.json")
ACTION_SPACE_PATH    = os.path.join(PARENT_DIR, "configs", "scene_action_space.json")
REFERENCE_IMAGES_DIR = os.path.join(PARENT_DIR, "pipeline", "data", "images_rgba")

DEFAULT_MAX_ROUNDS       = 4
DEFAULT_NO_IMPROVE_PATIENCE = 2

# Adjustable parameter space — keys MUST match actual control_state structure.
# Actual structure from default_control_state():
#   lighting: {key_scale, key_yaw_deg}
#   object:   {offset_z, yaw_deg, scale}
#   scene:    {hdri_yaw_deg, env_strength_scale, contact_shadow_strength}
#   material: {saturation_scale, value_scale, roughness_add}
PARAM_SPACE = {
    "lighting.key_scale": {
        "min": 0.3, "max": 3.0,
        "desc": "主方向灯强度缩放. 低于1.0减弱主灯, 高于1.0增强主灯",
    },
    "lighting.key_yaw_deg": {
        "min": -90.0, "max": 90.0,
        "desc": "主灯水平方向角度(度). 改变主光方向",
    },
    "object.scale": {
        "min": 0.5, "max": 2.0,
        "desc": "物体在场景中的缩放比例",
    },
    "object.yaw_deg": {
        "min": -180.0, "max": 180.0,
        "desc": "物体绕垂直轴旋转(度). 调整物体朝向",
    },
    "object.offset_z": {
        "min": -0.15, "max": 0.15,
        "desc": "物体垂直方向微调(米). 正值上移, 负值下移",
    },
    "scene.env_strength_scale": {
        "min": 0.3, "max": 3.0,
        "desc": "场景HDRI环境光整体强度. 增大使整体变亮",
    },
    "scene.hdri_yaw_deg": {
        "min": -180.0, "max": 180.0,
        "desc": "场景HDRI旋转角度(度). 改变环境光方向, 使物体阴影方向与场景一致",
    },
    "scene.contact_shadow_strength": {
        "min": 0.0, "max": 2.0,
        "desc": "物体与地面接触阴影强度. 增大使物体与地面连接更自然",
    },
    "material.roughness_add": {
        "min": -0.3, "max": 0.3,
        "desc": "材质粗糙度增量. 负值更光滑(反射强), 正值更粗糙(哑光)",
    },
    "material.saturation_scale": {
        "min": 0.5, "max": 1.5,
        "desc": "材质颜色饱和度缩放. 低于1.0偏灰, 高于1.0颜色更鲜艳",
    },
    "material.value_scale": {
        "min": 0.5, "max": 1.5,
        "desc": "材质亮度缩放. 低于1.0偏暗, 高于1.0偏亮",
    },
}


# ---------------------------------------------------------------------------
# Rendering  (unchanged from run_scene_evolution_loop.py)
# ---------------------------------------------------------------------------

def render_scene_state(obj_id, meshes_dir, output_dir, control_state,
                       scene_template_path, blender_bin=BLENDER_BIN,
                       resolution=1024, engine="CYCLES"):
    glb_path = os.path.join(meshes_dir, f"{obj_id}.glb")
    if not os.path.exists(glb_path):
        print(f"  [render] GLB not found: {glb_path}")
        return False

    os.makedirs(output_dir, exist_ok=True)
    cs_path = os.path.join(output_dir, f"_control_{obj_id}.json")
    with open(cs_path, "w") as f:
        json.dump(control_state, f, indent=2)

    cmd = [
        blender_bin, "-b", "-P", SCENE_RENDER_SCRIPT, "--",
        "--input-dir",      meshes_dir,
        "--output-dir",     output_dir,
        "--obj-id",         obj_id,
        "--resolution",     str(resolution),
        "--engine",         engine,
        "--control-state",  cs_path,
        "--scene-template", scene_template_path,
    ]
    print(f"  [render] Blender rendering {obj_id} ...")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        if result.returncode != 0:
            print(f"  [render] Blender failed:\n{result.stderr[-800:]}")
            return False
        return True
    except subprocess.TimeoutExpired:
        print("  [render] Blender timed out")
        return False
    except Exception as e:
        print(f"  [render] Error: {e}")
        return False


def _find_representative_render(render_dir: str, obj_id: str) -> Optional[str]:
    """Return az000_el+00 render path, with fallbacks."""
    for candidate in [
        os.path.join(render_dir, obj_id, "az000_el+00.png"),
        os.path.join(render_dir, obj_id, "az000_el000.png"),
    ]:
        if os.path.exists(candidate):
            return candidate
    obj_subdir = os.path.join(render_dir, obj_id)
    if os.path.isdir(obj_subdir):
        for f in sorted(os.listdir(obj_subdir)):
            if f.endswith(".png") and "_mask" not in f:
                return os.path.join(obj_subdir, f)
    return None


# ---------------------------------------------------------------------------
# Agent Audit  (new: simplified VLM prompt that returns verdict + param_changes)
# ---------------------------------------------------------------------------

def _build_agent_audit_prompt(control_state: dict) -> str:
    """Build the simplified Chinese audit prompt for the agent loop."""
    param_lines = []
    for key, spec in PARAM_SPACE.items():
        group, param = key.split(".", 1)
        cur_val = control_state.get(group, {}).get(param, "N/A")
        param_lines.append(
            f"  {key}: current={cur_val}  range=[{spec['min']}, {spec['max']}]  "
            f"meaning: {spec['desc']}"
        )
    params_str = "\n".join(param_lines)

    return f"""你是一个3D渲染质量审核员。
图片1是T2I参考图（目标效果，已经过SAM前景分割），图片2是当前Blender场景渲染结果。

请从以下5个维度审核渲染质量，对比参考图：
1. 结构一致性：渲染的物体与参考图是否是同一个物体，几何形状是否高度一致
2. 纹理颜色一致性：物体纹理和颜色是否与参考图匹配，无明显色差或色偏
3. 场景合理性：物体在场景中的位置、大小、朝向是否自然合理
4. 物理可信度：物体是否自然地放置在地面上（不悬浮、不穿插地面）
5. 整体美观度：光照、构图、渲染质量整体是否美观自然

当前可调整的渲染参数：
{params_str}

请严格输出以下JSON格式（不要输出任何其他内容，不要使用markdown代码块）：
{{
  "verdict": "accept 或 redo 或 reject_asset",
  "structure_consistency": "good 或 minor 或 major",
  "color_consistency": "good 或 minor 或 major",
  "scene_presence": "good 或 minor 或 major",
  "physics_plausibility": "good 或 minor 或 major",
  "overall_aesthetics": "good 或 minor 或 major",
  "top_issue": "最主要问题标签(如 scene_light_mismatch/color_shift/floating/too_dark/good_enough 等)",
  "summary": "一两句话描述主要问题和改进方向",
  "param_changes": {{
    "参数路径": 新数值
  }}
}}

判断规则：
- 如果5个维度都是good且整体已足够好，verdict为"accept"，param_changes填{{}}
- 如果物体mesh严重错误（与参考图是完全不同的物体），verdict为"reject_asset"，param_changes填{{}}
- 否则verdict为"redo"，param_changes给出具体参数修改建议
- param_changes每次最多修改3个参数，值必须在允许范围内，只列需要改的参数
- 优先修改影响最大的参数（通常scene.hdri_yaw_deg调整光照方向，或lighting.world_ev调整整体亮度）"""


def agent_audit(ref_img_path: str, render_img_path: str,
                control_state: dict, device: str = "cuda:0",
                max_retries: int = 3) -> dict:
    """
    Simplified VLM audit for the agent loop.
    Returns dict: verdict, 5-dim grades, top_issue, summary, param_changes.
    """
    import torch
    from qwen_vl_utils import process_vision_info

    model, processor = load_vlm(device)
    prompt_text = _build_agent_audit_prompt(control_state)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a 3D render quality auditor. "
                "Return ONLY a valid JSON object. No markdown, no extra text."
            ),
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": ref_img_path},
                {"type": "image", "image": render_img_path},
                {"type": "text",  "text":  prompt_text},
            ],
        },
    ]

    for attempt in range(max_retries):
        try:
            text_input = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
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
            text_out   = processor.decode(generated, skip_special_tokens=True)
            parsed     = _extract_audit_json(text_out)

            if parsed is not None:
                parsed = _validate_and_clamp(parsed, control_state)
                print(f"  [audit] verdict={parsed.get('verdict')}  "
                      f"top_issue={parsed.get('top_issue')}")
                print(f"  [audit] summary: {str(parsed.get('summary', ''))[:120]}")
                if parsed.get("param_changes"):
                    print(f"  [audit] param_changes: {parsed['param_changes']}")
                return parsed

            print(f"  [audit] attempt {attempt+1}: JSON parse failed, "
                  f"raw='{text_out[:150]}'")

        except Exception as e:
            print(f"  [audit] attempt {attempt+1} error: {e}")

    # Fallback: no changes
    print(f"  [audit] all {max_retries} attempts failed, returning neutral")
    return {
        "verdict": "redo",
        "structure_consistency": "good",
        "color_consistency": "good",
        "scene_presence": "good",
        "physics_plausibility": "good",
        "overall_aesthetics": "good",
        "top_issue": "parse_failed",
        "summary": "VLM response parse failed",
        "param_changes": {},
    }


def _extract_audit_json(text: str) -> Optional[dict]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r'\{[\s\S]*\}', text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


def _validate_and_clamp(audit: dict, control_state: dict) -> dict:
    if audit.get("verdict") not in {"accept", "redo", "reject_asset"}:
        audit["verdict"] = "redo"
    raw = audit.get("param_changes", {})
    clamped = {}
    for key, value in raw.items():
        if key in PARAM_SPACE:
            spec = PARAM_SPACE[key]
            try:
                clamped[key] = float(max(spec["min"], min(spec["max"], float(value))))
            except (TypeError, ValueError):
                pass
        else:
            print(f"  [audit] Unknown param '{key}', skipping")
    audit["param_changes"] = clamped
    return audit


# ---------------------------------------------------------------------------
# Parameter application
# ---------------------------------------------------------------------------

def apply_param_changes(control_state: dict, param_changes: dict) -> dict:
    new_state = copy.deepcopy(control_state)
    for key, value in param_changes.items():
        parts = key.split(".", 1)
        if len(parts) == 2:
            group, param = parts
            if group in new_state and isinstance(new_state[group], dict):
                if param in new_state[group]:
                    new_state[group][param] = value
                else:
                    print(f"  [apply] Unknown param '{param}' in group '{group}'")
            else:
                print(f"  [apply] Unknown group '{group}'")
    return new_state


# ---------------------------------------------------------------------------
# Progression visualization
# ---------------------------------------------------------------------------

def _load_font(size: int):
    """Try common Linux TTF paths; fallback to Pillow default."""
    from PIL import ImageFont
    for font_path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
    ]:
        if os.path.exists(font_path):
            try:
                return ImageFont.truetype(font_path, size)
            except Exception:
                pass
    return ImageFont.load_default()


def _rgba_to_rgb(img) -> "Image.Image":
    """Composite RGBA image onto white background, return RGB."""
    from PIL import Image
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        return bg
    return img.convert("RGB")


def _format_param_changes(param_changes: dict) -> str:
    """Abbreviate param_changes dict to short string, max 40 chars.

    e.g. {"scene.hdri_yaw_deg": 60.0} -> "hdri_yaw=60.0"
    """
    if not param_changes:
        return ""
    parts = []
    for key, val in param_changes.items():
        short_key = key.split(".", 1)[-1]
        short_key = (short_key.replace("_deg", "")
                               .replace("_scale", "")
                               .replace("_strength", ""))
        if isinstance(val, float):
            parts.append(f"{short_key}={val:.2g}")
        else:
            parts.append(f"{short_key}={val}")
    result = " ".join(parts)
    if len(result) > 40:
        result = result[:37] + "..."
    return result


def generate_progression_image(obj_id: str, obj_output_dir: str,
                                total_rounds: int,
                                ref_img_path: Optional[str] = None) -> Optional[str]:
    """Create annotated progression strip: ref (T2I) → base → round 1 → ... → round N.

    Each frame is 512×512.  Per-frame annotations (top_issue, param_changes) are
    drawn below from persisted audit_XX.json / decision_XX.json files.
    The final render frame gets a verdict badge (ACCEPT / REDO / REJECT).
    """
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        print("  [progression] Pillow not available, skipping")
        return None

    FRAME_W  = 512
    FRAME_H  = 512
    GAP      = 12
    MARGIN   = 16
    LABEL_H  = 70   # pixels below each frame for 3 annotation lines
    HEADER_H = 36
    BORDER   = 2

    font_title = _load_font(16)
    font_label = _load_font(14)
    font_small = _load_font(12)

    # ── Collect frames ────────────────────────────────────────────────
    frames_data = []  # list of dicts

    # Reference (T2I) frame
    if ref_img_path and os.path.exists(ref_img_path):
        try:
            ref_pil = Image.open(ref_img_path)
            ref_pil = _rgba_to_rgb(ref_pil).resize((FRAME_W, FRAME_H), Image.LANCZOS)
            frames_data.append({
                "img": ref_pil, "label": "ref (T2I)",
                "top_issue": "", "param_changes_str": "",
                "verdict": None, "is_last": False,
            })
        except Exception as e:
            print(f"  [progression] Could not load ref image: {e}")

    # Render frames: round_00 (base) through round_N
    for r in range(total_rounds + 1):
        rdir     = os.path.join(obj_output_dir, f"round_{r:02d}")
        img_path = _find_representative_render(rdir, obj_id)
        if not (img_path and os.path.exists(img_path)):
            continue
        try:
            frame_img = Image.open(img_path).convert("RGB").resize(
                (FRAME_W, FRAME_H), Image.LANCZOS
            )
        except Exception:
            continue

        label = "base" if r == 0 else f"round {r}"

        # Load audit annotations (audit produced *after* rendering round r)
        top_issue = ""
        audit_path = os.path.join(obj_output_dir, f"audit_{r:02d}.json")
        if os.path.exists(audit_path):
            try:
                with open(audit_path, encoding="utf-8") as f:
                    top_issue = json.load(f).get("top_issue", "")
            except Exception:
                pass

        # Load param_changes from decision that *follows* this render
        param_changes_str = ""
        decision_path = os.path.join(obj_output_dir, f"decision_{r + 1:02d}.json")
        if r > 0 and os.path.exists(decision_path):
            try:
                with open(decision_path, encoding="utf-8") as f:
                    param_changes_str = _format_param_changes(
                        json.load(f).get("param_changes", {})
                    )
            except Exception:
                pass

        frames_data.append({
            "img": frame_img, "label": label,
            "top_issue": top_issue, "param_changes_str": param_changes_str,
            "verdict": None, "is_last": False,
        })

    if not frames_data:
        return None

    # ── Resolve final verdict ─────────────────────────────────────────
    final_verdict = None
    audit_final_path = os.path.join(obj_output_dir, "audit_final.json")
    if os.path.exists(audit_final_path):
        try:
            with open(audit_final_path, encoding="utf-8") as f:
                final_verdict = json.load(f).get("verdict")
        except Exception:
            pass
    if final_verdict is None and total_rounds > 0:
        last_audit_path = os.path.join(obj_output_dir, f"audit_{total_rounds:02d}.json")
        if os.path.exists(last_audit_path):
            try:
                with open(last_audit_path, encoding="utf-8") as f:
                    final_verdict = json.load(f).get("verdict")
            except Exception:
                pass

    # Mark last render frame (not ref) for verdict badge
    for i in range(len(frames_data) - 1, -1, -1):
        if frames_data[i]["label"] != "ref (T2I)":
            frames_data[i]["is_last"] = True
            frames_data[i]["verdict"] = final_verdict
            break

    # ── Build canvas ──────────────────────────────────────────────────
    n        = len(frames_data)
    canvas_w = MARGIN * 2 + n * FRAME_W + (n - 1) * GAP
    canvas_h = MARGIN + HEADER_H + FRAME_H + LABEL_H + MARGIN
    canvas   = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    draw     = ImageDraw.Draw(canvas)

    # Header
    draw.text((MARGIN, MARGIN),
              f"{obj_id} \u2014 Agent Loop Progression",
              fill=(30, 30, 30), font=font_title)

    for i, fd in enumerate(frames_data):
        x = MARGIN + i * (FRAME_W + GAP)
        y = MARGIN + HEADER_H

        # Colored border: green if accepted, gray otherwise
        if fd["is_last"] and fd["verdict"] == "accept":
            border_color = (34, 139, 34)
        else:
            border_color = (160, 160, 160)

        draw.rectangle(
            [x - BORDER, y - BORDER,
             x + FRAME_W + BORDER - 1, y + FRAME_H + BORDER - 1],
            outline=border_color, width=BORDER,
        )

        canvas.paste(fd["img"], (x, y))

        # Verdict badge on last render frame
        if fd["is_last"] and fd["verdict"]:
            verdict = fd["verdict"]
            if verdict == "accept":
                badge_bg, badge_txt = (34, 139, 34),   "ACCEPT"
            elif verdict == "reject_asset":
                badge_bg, badge_txt = (200, 0, 0),     "REJECT"
            else:
                badge_bg, badge_txt = (210, 105, 30),  "REDO"
            bx = x + FRAME_W - 82
            by = y + FRAME_H - 28
            draw.rectangle([bx - 4, by - 2, bx + 78, by + 20], fill=badge_bg)
            draw.text((bx, by), badge_txt, fill=(255, 255, 255), font=font_label)

        # Annotations below frame
        ann_y = y + FRAME_H + 5
        draw.text((x, ann_y), fd["label"], fill=(30, 30, 30), font=font_label)
        ann_y += 18
        if fd["top_issue"]:
            draw.text((x, ann_y), fd["top_issue"][:35],
                      fill=(140, 60, 60), font=font_small)
        ann_y += 16
        if fd["param_changes_str"]:
            draw.text((x, ann_y), fd["param_changes_str"],
                      fill=(60, 60, 150), font=font_small)

    out_path = os.path.join(obj_output_dir, "progression.png")
    canvas.save(out_path)
    print(f"  [progression] Saved: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Main loop for a single object
# ---------------------------------------------------------------------------

def loop_object(obj_id: str, meshes_dir: str, output_dir: str,
                reference_images_dir: str, scene_template_path: str,
                device: str = "cuda:0", blender_bin: str = BLENDER_BIN,
                max_rounds: int = DEFAULT_MAX_ROUNDS,
                no_improve_patience: int = DEFAULT_NO_IMPROVE_PATIENCE,
                resolution: int = 1024) -> dict:
    """Run agent loop for one object. Returns summary dict."""
    aspace        = load_action_space(ACTION_SPACE_PATH)
    control_state = default_control_state(aspace)

    obj_out = os.path.join(output_dir, obj_id)
    os.makedirs(obj_out, exist_ok=True)

    # Locate reference image
    ref_img = os.path.join(reference_images_dir, f"{obj_id}.png")
    if not os.path.exists(ref_img):
        alt = os.path.join(reference_images_dir.replace("images_rgba", "images"),
                           f"{obj_id}.png")
        if os.path.exists(alt):
            ref_img = alt
        else:
            print(f"  [{obj_id}] Reference image not found: {ref_img}")
            return {"obj_id": obj_id, "error": "reference_not_found"}

    # ── Round 0: baseline render ──────────────────────────────────────
    print(f"\n[{obj_id}] ===== Round 0: baseline render =====")
    round_dir = os.path.join(obj_out, "round_00")
    ok = render_scene_state(obj_id, meshes_dir, round_dir, control_state,
                            scene_template_path, blender_bin, resolution)
    if not ok:
        return {"obj_id": obj_id, "error": "baseline_render_failed"}

    with open(os.path.join(obj_out, "control_state_00.json"), "w") as f:
        json.dump(control_state, f, indent=2)

    final_round    = 0
    final_verdict  = "redo"
    no_improve_cnt = 0

    # ── Rounds 1 .. max_rounds ────────────────────────────────────────
    for round_idx in range(1, max_rounds + 1):
        prev_render_dir = os.path.join(obj_out, f"round_{round_idx-1:02d}")
        current_render  = _find_representative_render(prev_render_dir, obj_id)
        if not current_render:
            print(f"  [{obj_id}] Cannot find render for round {round_idx-1}, stopping")
            break

        print(f"\n[{obj_id}] ===== Round {round_idx}: VLM audit =====")
        audit = agent_audit(ref_img, current_render, control_state, device)

        # Save audit
        with open(os.path.join(obj_out, f"audit_{round_idx-1:02d}.json"),
                  "w", encoding="utf-8") as f:
            json.dump(audit, f, indent=2, ensure_ascii=False)

        final_verdict = audit.get("verdict", "redo")

        if final_verdict == "accept":
            print(f"  [{obj_id}] Accepted at round {round_idx-1}. Done.")
            break

        if final_verdict == "reject_asset":
            print(f"  [{obj_id}] Asset rejected (mesh/asset quality). Done.")
            break

        param_changes = audit.get("param_changes", {})
        if not param_changes:
            no_improve_cnt += 1
            print(f"  [{obj_id}] No param changes suggested "
                  f"(patience {no_improve_cnt}/{no_improve_patience})")
            if no_improve_cnt >= no_improve_patience:
                print(f"  [{obj_id}] Patience exceeded, stopping")
                break
            continue
        else:
            no_improve_cnt = 0

        # Apply changes and save
        control_state = apply_param_changes(control_state, param_changes)
        with open(os.path.join(obj_out, f"control_state_{round_idx:02d}.json"), "w") as f:
            json.dump(control_state, f, indent=2)

        decision = {
            "round":             round_idx,
            "top_issue":         audit.get("top_issue", ""),
            "summary":           audit.get("summary", ""),
            "param_changes":     param_changes,
            "new_control_state": control_state,
        }
        with open(os.path.join(obj_out, f"decision_{round_idx:02d}.json"),
                  "w", encoding="utf-8") as f:
            json.dump(decision, f, indent=2, ensure_ascii=False)

        # Re-render with new params
        print(f"\n[{obj_id}] ===== Round {round_idx}: re-render =====")
        round_dir = os.path.join(obj_out, f"round_{round_idx:02d}")
        ok = render_scene_state(obj_id, meshes_dir, round_dir, control_state,
                                scene_template_path, blender_bin, resolution)
        if not ok:
            print(f"  [{obj_id}] Render failed at round {round_idx}, stopping")
            break

        final_round = round_idx

    # ── Final audit on last render ────────────────────────────────────
    last_render = _find_representative_render(
        os.path.join(obj_out, f"round_{final_round:02d}"), obj_id
    )
    if last_render and final_verdict == "redo":
        print(f"\n[{obj_id}] ===== Final audit =====")
        final_audit = agent_audit(ref_img, last_render, control_state, device)
        with open(os.path.join(obj_out, "audit_final.json"),
                  "w", encoding="utf-8") as f:
            json.dump(final_audit, f, indent=2, ensure_ascii=False)
        final_verdict = final_audit.get("verdict", "redo")

    # ── Progression visualization ─────────────────────────────────────
    prog_path = generate_progression_image(obj_id, obj_out, final_round,
                                           ref_img_path=ref_img)

    result = {
        "obj_id":            obj_id,
        "total_rounds":      final_round,
        "final_verdict":     final_verdict,
        "progression_image": prog_path,
    }
    print(f"\n[{obj_id}] DONE  rounds={final_round}  verdict={final_verdict}")
    return result


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def run_agent_loop(obj_ids, meshes_dir, output_dir, reference_images_dir,
                   scene_template_path, device="cuda:0", blender_bin=BLENDER_BIN,
                   max_rounds=DEFAULT_MAX_ROUNDS, resolution=1024):
    os.makedirs(output_dir, exist_ok=True)
    results = []
    for obj_id in obj_ids:
        try:
            r = loop_object(
                obj_id=obj_id,
                meshes_dir=meshes_dir,
                output_dir=output_dir,
                reference_images_dir=reference_images_dir,
                scene_template_path=scene_template_path,
                device=device,
                blender_bin=blender_bin,
                max_rounds=max_rounds,
                resolution=resolution,
            )
        except Exception as e:
            import traceback
            print(f"[{obj_id}] EXCEPTION: {e}")
            traceback.print_exc()
            r = {"obj_id": obj_id, "error": str(e)}
        results.append(r)

    summary = {
        "total":    len(results),
        "accepted": sum(1 for r in results if r.get("final_verdict") == "accept"),
        "results":  results,
    }
    summary_path = os.path.join(output_dir, "batch_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n=== Batch done: {summary['accepted']}/{summary['total']} accepted ===")
    print(f"    Summary: {summary_path}")
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Agent-in-the-loop scene render optimizer (v7.2)"
    )
    p.add_argument("--meshes-dir",
                   default=os.path.join(PARENT_DIR, "pipeline", "data", "meshes"))
    p.add_argument("--output-dir",
                   default=os.path.join(PARENT_DIR, "pipeline", "data", "agent_loop_v72"))
    p.add_argument("--reference-images-dir", default=REFERENCE_IMAGES_DIR)
    p.add_argument("--obj-ids", nargs="+",
                   default=["obj_009", "obj_005", "obj_001"],
                   help="Objects to process (default: 3-object smoke test)")
    p.add_argument("--device",       default="cuda:0")
    p.add_argument("--blender",      default=BLENDER_BIN, dest="blender_bin")
    p.add_argument("--scene-template", default=SCENE_TEMPLATE_PATH)
    p.add_argument("--max-rounds",   type=int, default=DEFAULT_MAX_ROUNDS)
    p.add_argument("--resolution",   type=int, default=1024)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_agent_loop(
        obj_ids=args.obj_ids,
        meshes_dir=args.meshes_dir,
        output_dir=args.output_dir,
        reference_images_dir=args.reference_images_dir,
        scene_template_path=args.scene_template,
        device=args.device,
        blender_bin=args.blender_bin,
        max_rounds=args.max_rounds,
        resolution=args.resolution,
    )
