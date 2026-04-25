#!/usr/bin/env python3
"""Build the read-only prompt for the feedback-loop Plan Agent."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare import load_eval_sources


DEFAULT_WORKDIR = "/gemini/platform/public/aigc/aigc_image/zhanghy56_intern/zhangqisong/data-build"
DEFAULT_WWZ_CODE_ROOT = "/aaaidata/zhangqisong/data_build_runs/scene_full20_loop_20260404_code"
DEFAULT_BASELINE_DATASET = (
    "dataset_scene_v7_full50_rotation8_trainready_front2others_"
    "splitobj_seed42_bboxmask_bright_objinfo_20260418"
)
DEFAULT_BASELINE_CHECKPOINT = "DiffSynth-Studio/output/rotation8_bright_objinfo_rank32/epoch_0029/lora.safetensors"


def read_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _fmt(value) -> str:
    if value is None:
        return "NA"
    if isinstance(value, (int, float)):
        return f"{value:.4f}"
    return str(value)


def format_metrics_table(source_name: str, record: dict) -> str:
    lines = [
        f"### {source_name}",
        "",
        f"- mode: `{record.get('mode')}`",
        f"- path: `{record.get('path')}`",
        f"- n_pairs: `{record.get('n_pairs')}`",
        "",
        "| Angle | PSNR | SSIM | LPIPS | CLIP-I | DINO |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    per_angle = record.get("per_angle") or {}
    for angle in sorted(per_angle, key=lambda x: int(x)):
        metrics = per_angle[angle]
        lines.append(
            "| {angle} | {psnr} | {ssim} | {lpips} | {clip_i} | {dino} |".format(
                angle=angle,
                psnr=_fmt(metrics.get("psnr")),
                ssim=_fmt(metrics.get("ssim")),
                lpips=_fmt(metrics.get("lpips")),
                clip_i=_fmt(metrics.get("clip_i")),
                dino=_fmt(metrics.get("dino")),
            )
        )
    lines.append("")
    return "\n".join(lines)


def previous_compare_summary(state: Optional[dict], round_number: int) -> str:
    if not state or round_number <= 1:
        return "None; this is the first feedback round."
    previous = None
    for item in state.get("rounds", []):
        if item.get("round") == round_number - 1:
            previous = item
            break
    if not previous:
        return "No previous round record found in FEEDBACK_STATE.json."
    compare_report_path = (previous.get("paths") or {}).get("compare_report")
    compare_payload = read_json(Path(compare_report_path)) if compare_report_path else None
    primary = None
    if compare_payload:
        sources = compare_payload.get("sources") or {}
        primary = sources.get("spatialedit") or sources.get("testset")
        if primary is None and sources:
            primary = next(iter(sources.values()))
    return json.dumps(
        {
            "round": previous.get("round"),
            "verdict": previous.get("verdict"),
            "delta_summary": previous.get("delta_summary"),
            "compare_report": compare_report_path,
            "dataset_feedback_plan": (
                previous.get("v2_dataset_construction") or {}
            ).get("dataset_feedback_plan_path"),
            "weak_angles": (primary or {}).get("weak_angles"),
            "strong_angles": (primary or {}).get("strong_angles"),
            "strong_angle_regressions": (primary or {}).get("strong_angle_regressions"),
            "all_angle_bad_objects": (primary or {}).get("all_angle_bad_objects"),
            "object_angle_outliers": (primary or {}).get("object_angle_outliers"),
        },
        indent=2,
        ensure_ascii=False,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build feedback-loop Plan Agent prompt")
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workdir", default=os.environ.get("WORKDIR", DEFAULT_WORKDIR))
    parser.add_argument("--baseline-eval-dir", default=None)
    parser.add_argument("--baseline-mode", default="ours_objinfo")
    parser.add_argument("--wwz-code-root", default=os.environ.get("WWZ_CODE_ROOT", DEFAULT_WWZ_CODE_ROOT))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    workdir = Path(args.workdir)
    output_path = Path(args.output)
    run_root = output_path.parent
    experiment_root = run_root.parent
    state_path = experiment_root / "FEEDBACK_STATE.json"
    state = read_json(state_path)
    baseline_eval_dir = Path(
        args.baseline_eval_dir
        or os.environ.get("BASELINE_EVAL_DIR", str(experiment_root / "round_0" / "eval_results"))
    )

    metric_sections = []
    warnings = []
    sources = load_eval_sources(baseline_eval_dir, args.baseline_mode) if baseline_eval_dir.exists() else {}
    if sources:
        for name, record in sources.items():
            metric_sections.append(format_metrics_table(name, record))
    else:
        warnings.append(f"Baseline metrics not found or not parseable: {baseline_eval_dir}")

    prompt = f"""# Feedback Loop Plan Agent - Round {args.round}

You are the read-only Plan Agent for the rotation feedback loop.

## Baseline Summary
- Experiment: exp5_objinfo
- Dataset: `{workdir / DEFAULT_BASELINE_DATASET}`
- Checkpoint: `{workdir / DEFAULT_BASELINE_CHECKPOINT}`
- Baseline eval dir: `{baseline_eval_dir}`
- Code policy: local repo is the only code modification source; servers only pull, run, and report logs.
- Server ownership: wwz builds new objects and renders; 68 only merges train-ready data, trains, and evaluates.
- wwz code root: `{args.wwz_code_root}`

## Baseline Metrics
{chr(10).join(metric_sections) if metric_sections else "No baseline metrics table available."}

## Previous Round Compare Summary
```json
{previous_compare_summary(state, args.round)}
```

## Dataset Feedback Loop v2
The intervention is dataset scaling, not training-strategy search. Diagnose weak target rotations and object-specific failures, then expand the dataset by adding new train-only objects and weak-angle pairs.

## Allowed Write Scope For Execution Phase
- `$RUN_ROOT/` only for generated datasets, logs, state, and command files.
- No server-side repo code edits. If code must change, describe it for local implementation instead.

## Hard Constraints
- New dataset roots only; do not overwrite existing datasets.
- `source_image` and `target_image` must point to `views/`, not `bbox_views/`.
- Preserve pinned object-disjoint split for the original 50 objects. New objects are train-only.
- Do not write caches, logs, or training outputs inside dataset roots.
- Use epoch 29 checkpoint for comparison.
- If required source data is missing, stop and report the missing path.
- Data construction dependencies live on wwz. Do not plan Stage 1, Blender, SAM2, Hunyuan3D, Qwen3.5, or Qwen-Image-2512 execution on 68 unless explicitly confirmed.

## Output Format
Write `INTERVENTION_PLAN.md` with:
- Diagnostic summary.
- Dataset feedback plan: weak angles, strong-angle guard, all-angle-bad objects, and object-angle outliers.
- wwz commands for new object concept generation, Stage 1 assets, render-prior warm start, yaw000 VLM loop, and rotation export.
- transfer command from wwz to 68.
- 68 commands for pinned split validation, augmented dataset merge, train, eval, and compare.
- Expected `dataset_feedback_plan.json`, `pinned_split.json`, `golden_config_prior.json`, and `augmented_manifest.json` paths.
- Training, eval, and compare command expectations.
- Expected impact on Test Set and SpatialEdit separately.
"""

    if warnings:
        prompt += "\n## Prompt Build Warnings\n" + "\n".join(f"- {warning}" for warning in warnings) + "\n"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(prompt, encoding="utf-8")
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
