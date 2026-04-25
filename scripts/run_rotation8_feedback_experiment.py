"""
Offline controller for the rotation8 train->dataset feedback experiment.

This controller has two layers:
1. Always analyze pair-level loss traces and emit hard-case / augmentation plans.
2. Optionally materialize feedback/random augmented split roots when candidate
   scene_v7 pair-evolution roots are provided.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Dict, Iterable, List, Optional, Tuple


SCRIPT_PATH = Path(__file__).resolve()
SCRIPTS_DIR = SCRIPT_PATH.parent
REPO_ROOT = SCRIPT_PATH.parent.parent
PIPELINE_ROOT = REPO_ROOT / "pipeline"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from asset_lifecycle import detect_asset_viability, detect_verdict_from_review
from build_rotation8_trainready_dataset import build_dataset as build_trainready_dataset
from export_scene_multiview_from_pair_evolution import (
    collect_round_records,
    discover_pair_dirs,
    select_best_round,
)


EXPORT_ROTATION8_SCRIPT = SCRIPTS_DIR / "export_rotation8_from_best_object_state.py"
TARGET_ROTATIONS = [45, 90, 135, 180, 225, 270, 315]
TRACE_TEXT_KEYS = ("assistant_text", "raw_text", "response_text", "content", "answer", "model_output")
KEEP_SIGNAL_PATTERNS = ("keep", "acceptable", "good enough", "可以了")


def parse_args():
    parser = argparse.ArgumentParser(description="Rotation8 feedback experiment controller")
    parser.add_argument("--trace-root", required=True, help="pair_loss_trace root")
    parser.add_argument("--base-split-root", required=True, help="Base object-disjoint split root")
    parser.add_argument("--output-dir", required=True, help="Experiment output directory")
    parser.add_argument("--warmup-ignore-epochs", type=int, default=5)
    parser.add_argument("--hard-angle-count", type=int, default=2)
    parser.add_argument("--hard-angle-gap-ratio-threshold", type=float, default=1.2)
    parser.add_argument("--hard-pair-top-percent", type=float, default=0.2)
    parser.add_argument("--augmentation-quota-per-angle", type=int, default=10)
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--random-control-seed", type=int, default=4201)
    parser.add_argument("--qc-hybrid-threshold", type=float, default=0.78)
    parser.add_argument("--candidate-link-mode", choices=["symlink", "copy"], default="symlink")
    parser.add_argument("--asset-link-mode", choices=["symlink", "copy"], default="symlink")
    parser.add_argument("--trainready-copy-mode", choices=["copy", "symlink"], default="symlink")
    parser.add_argument("--feedback-source-root", default=None, help="Candidate scene_v7 pair-evolution root for targeted augmentation")
    parser.add_argument("--random-source-root", default=None, help="Candidate scene_v7 pair-evolution root for random-control augmentation")
    parser.add_argument("--materialize-datasets", action="store_true", help="Render/export accepted candidates and build augmented split roots")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--blender-bin", default=None)
    parser.add_argument("--meshes-dir", default=None)
    parser.add_argument("--scene-template", default=None)
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--base-rotation-deg", type=int, default=0)
    return parser.parse_args()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path, default=None):
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def load_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: List[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as f:
            f.write("")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def ensure_clean_dir(path: Path):
    if path.exists() or path.is_symlink():
        if path.is_symlink() or path.is_file():
            path.unlink()
        else:
            shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def link_or_copy_dir(src: Path, dst: Path, mode: str):
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink() or dst.is_file():
            dst.unlink()
        else:
            shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "symlink":
        dst.symlink_to(src, target_is_directory=True)
    else:
        shutil.copytree(src, dst)


def assert_new_output_root(path: Path):
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Output dir already exists and is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def sorted_trace_files(trace_root: Path) -> List[Path]:
    files = [p for p in trace_root.rglob("rank_*.jsonl") if p.is_file()]
    return sorted(files, key=lambda p: str(p))


def normalize_angle(value) -> int:
    return int(value)


def load_pair_loss_trace(trace_root: Path) -> Tuple[List[dict], dict]:
    rows: List[dict] = []
    trace_files = sorted_trace_files(trace_root)
    if not trace_files:
        raise FileNotFoundError(f"No pair trace files found under {trace_root}")

    run_ids = set()
    epochs = set()
    ranks = set()
    pair_ids = set()
    for trace_file in trace_files:
        for row in load_jsonl(trace_file):
            required = {"run_id", "epoch", "global_step", "rank", "pair_id", "obj_id", "target_rotation_deg", "loss"}
            missing = sorted(required - set(row))
            if missing:
                raise KeyError(f"Trace row in {trace_file} missing keys: {missing}")
            row = dict(row)
            row["epoch"] = int(row["epoch"])
            row["global_step"] = int(row["global_step"])
            row["rank"] = int(row["rank"])
            row["pair_id"] = str(row["pair_id"])
            row["obj_id"] = str(row["obj_id"])
            row["target_rotation_deg"] = normalize_angle(row["target_rotation_deg"])
            row["loss"] = float(row["loss"])
            rows.append(row)
            run_ids.add(str(row["run_id"]))
            epochs.add(row["epoch"])
            ranks.add(row["rank"])
            pair_ids.add(row["pair_id"])

    if not rows:
        raise ValueError(f"Trace root {trace_root} contains no rows")
    if len(run_ids) != 1:
        raise ValueError(f"Expected a single run_id in trace root, found {sorted(run_ids)}")

    summary = {
        "trace_root": str(trace_root),
        "trace_files": [str(p) for p in trace_files],
        "run_id": next(iter(run_ids)),
        "total_rows": len(rows),
        "epochs": sorted(epochs),
        "ranks": sorted(ranks),
        "unique_pairs": len(pair_ids),
    }
    return rows, summary


def analyze_hard_cases(rows: List[dict], args) -> dict:
    pair_epoch_losses: Dict[str, Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))
    pair_meta: Dict[str, dict] = {}
    epochs = set()

    for row in rows:
        pair_id = row["pair_id"]
        meta = {
            "pair_id": pair_id,
            "obj_id": row["obj_id"],
            "target_rotation_deg": normalize_angle(row["target_rotation_deg"]),
        }
        existing = pair_meta.get(pair_id)
        if existing is None:
            pair_meta[pair_id] = meta
        elif existing != meta:
            raise ValueError(f"Inconsistent metadata for pair_id={pair_id}: {existing} vs {meta}")
        pair_epoch_losses[pair_id][int(row["epoch"])].append(float(row["loss"]))
        epochs.add(int(row["epoch"]))

    late_epochs = [epoch for epoch in sorted(epochs) if epoch > int(args.warmup_ignore_epochs)]
    if not late_epochs:
        raise ValueError(
            f"No epochs remain after warmup_ignore_epochs={args.warmup_ignore_epochs}. "
            f"Observed epochs: {sorted(epochs)}"
        )

    pair_epoch_mean: Dict[str, Dict[int, float]] = {}
    for pair_id, per_epoch in pair_epoch_losses.items():
        pair_epoch_mean[pair_id] = {
            epoch: float(sum(losses) / len(losses))
            for epoch, losses in sorted(per_epoch.items())
        }

    persistence_counter: Dict[str, int] = defaultdict(int)
    top_fraction = float(args.hard_pair_top_percent)
    for epoch in late_epochs:
        angle_buckets: Dict[int, List[Tuple[str, float]]] = defaultdict(list)
        for pair_id, per_epoch in pair_epoch_mean.items():
            if epoch not in per_epoch:
                continue
            angle = int(pair_meta[pair_id]["target_rotation_deg"])
            angle_buckets[angle].append((pair_id, per_epoch[epoch]))
        for angle, items in angle_buckets.items():
            items = sorted(items, key=lambda item: (-item[1], item[0]))
            top_k = max(1, int(math.ceil(len(items) * top_fraction)))
            for pair_id, _ in items[:top_k]:
                persistence_counter[pair_id] += 1

    pair_rankings: List[dict] = []
    for pair_id, per_epoch in pair_epoch_mean.items():
        late_losses = [per_epoch[epoch] for epoch in late_epochs if epoch in per_epoch]
        if not late_losses:
            continue
        score = float(sum(late_losses) / len(late_losses))
        persistence = int(persistence_counter.get(pair_id, 0))
        ranking = {
            "pair_id": pair_id,
            "obj_id": pair_meta[pair_id]["obj_id"],
            "target_rotation_deg": int(pair_meta[pair_id]["target_rotation_deg"]),
            "pair_score": score,
            "persistence": persistence,
            "late_epoch_count": len(late_losses),
            "late_epoch_mean_losses": {str(epoch): per_epoch[epoch] for epoch in late_epochs if epoch in per_epoch},
            "persistent_hard": persistence > 0,
        }
        pair_rankings.append(ranking)

    pair_rankings.sort(key=lambda item: (-item["pair_score"], -item["persistence"], item["pair_id"]))

    angle_stats: List[dict] = []
    overall_pair_scores = [item["pair_score"] for item in pair_rankings]
    overall_median = float(median(overall_pair_scores))
    angle_groups: Dict[int, List[dict]] = defaultdict(list)
    for ranking in pair_rankings:
        angle_groups[int(ranking["target_rotation_deg"])].append(ranking)

    for angle in sorted(angle_groups):
        group = angle_groups[angle]
        scores = [item["pair_score"] for item in group]
        persistent_count = sum(1 for item in group if item["persistent_hard"])
        angle_stats.append(
            {
                "target_rotation_deg": int(angle),
                "pair_count": len(group),
                "median_pair_score": float(median(scores)),
                "mean_pair_score": float(sum(scores) / len(scores)),
                "persistent_hard_pair_count": persistent_count,
                "persistent_hard_ratio": float(persistent_count / len(group)),
                "top_pairs": [
                    {
                        "pair_id": item["pair_id"],
                        "obj_id": item["obj_id"],
                        "pair_score": item["pair_score"],
                        "persistence": item["persistence"],
                    }
                    for item in group[: min(10, len(group))]
                ],
            }
        )

    angle_stats.sort(
        key=lambda item: (
            -item["median_pair_score"],
            -item["persistent_hard_ratio"],
            item["target_rotation_deg"],
        )
    )

    selected_hard_angles = [
        int(item["target_rotation_deg"])
        for item in angle_stats[: max(0, int(args.hard_angle_count))]
    ]
    topk_median_mean = (
        float(sum(item["median_pair_score"] for item in angle_stats[: len(selected_hard_angles)]) / len(selected_hard_angles))
        if selected_hard_angles
        else 0.0
    )
    gap_ratio = float(topk_median_mean / overall_median) if overall_median > 0 else float("inf")
    decision = "SELECTED" if selected_hard_angles and gap_ratio >= float(args.hard_angle_gap_ratio_threshold) else "NO_UPDATE"
    if decision == "NO_UPDATE":
        selected_hard_angles = []

    return {
        "config": {
            "warmup_ignore_epochs": int(args.warmup_ignore_epochs),
            "hard_pair_top_percent": float(args.hard_pair_top_percent),
            "hard_angle_count": int(args.hard_angle_count),
            "hard_angle_gap_ratio_threshold": float(args.hard_angle_gap_ratio_threshold),
        },
        "trace_summary": {
            "epochs_seen": sorted(epochs),
            "late_epochs": late_epochs,
            "late_epoch_count": len(late_epochs),
            "pair_count": len(pair_rankings),
        },
        "overall_pair_score_median": overall_median,
        "topk_angle_median_mean": topk_median_mean,
        "gap_ratio": gap_ratio,
        "decision": decision,
        "selected_hard_angles": selected_hard_angles,
        "pair_rankings": pair_rankings,
        "angle_stats": angle_stats,
    }


def choose_random_control_angles(hard_angles: List[int], args) -> List[int]:
    if not hard_angles:
        return []
    remaining = [angle for angle in TARGET_ROTATIONS if angle not in set(hard_angles)]
    rng = random.Random(int(args.random_control_seed))
    return sorted(rng.sample(remaining, k=min(len(hard_angles), len(remaining))))


def build_augmentation_manifest(trace_summary: dict, hard_case_report: dict, base_split_root: Path, args) -> dict:
    hard_angles = list(hard_case_report["selected_hard_angles"])
    random_control_angles = choose_random_control_angles(hard_angles, args)
    quota = int(args.augmentation_quota_per_angle)

    return {
        "created_at": now_iso(),
        "run_id": trace_summary["run_id"],
        "base_split_root": str(base_split_root),
        "decision": hard_case_report["decision"],
        "targeted_angles": hard_angles,
        "random_control_angles": random_control_angles,
        "config": {
            "augmentation_quota_per_angle": quota,
            "train_seed": int(args.train_seed),
            "random_control_seed": int(args.random_control_seed),
            "qc_hybrid_threshold": float(args.qc_hybrid_threshold),
            "qc_keep_signal_patterns": list(KEEP_SIGNAL_PATTERNS),
            "reject_asset_viability": "abandon",
            "non_target_policy": "held_out_diagnostic_only",
            "training_protocol": "all groups train from scratch",
        },
        "groups": {
            "baseline": {
                "split_root": str(base_split_root),
                "materialization_status": "ready",
                "training_pair_absorption_rule": "use base split train pairs only",
            },
            "feedback-round1": {
                "source_root": str(args.feedback_source_root) if args.feedback_source_root else None,
                "targeted_angles": hard_angles,
                "quota_per_angle": quota,
                "requested_total_objects": quota * len(hard_angles),
                "training_pair_absorption_rule": "only front -> targeted angle",
                "materialization_status": "pending_candidate_generation" if args.feedback_source_root is None else "planned",
            },
            "random-augment": {
                "source_root": str(args.random_source_root) if args.random_source_root else None,
                "targeted_angles": random_control_angles,
                "quota_per_angle": quota,
                "requested_total_objects": quota * len(random_control_angles),
                "training_pair_absorption_rule": "only front -> sampled control angle",
                "materialization_status": "pending_candidate_generation" if args.random_source_root is None else "planned",
            },
        },
    }


def load_round_trace_text(pair_dir: Path, round_idx: int) -> str:
    reviews_dir = pair_dir / "reviews"
    chunks: List[str] = []
    seen = set()
    for trace_path in sorted(reviews_dir.glob(f"*_r{round_idx:02d}_*_trace.json")):
        payload = load_json(trace_path, default={}) or {}
        attempts = payload.get("attempts") or []
        text = None
        if attempts:
            assistant_text = attempts[-1].get("assistant_text")
            if isinstance(assistant_text, str) and assistant_text.strip():
                text = assistant_text.strip()
        if not text:
            for key in TRACE_TEXT_KEYS:
                value = payload.get(key)
                if isinstance(value, str) and value.strip():
                    text = value.strip()
                    break
        if text and text not in seen:
            chunks.append(text)
            seen.add(text)
    return "\n\n".join(chunks)


def has_keep_signal(text: str, agg: dict) -> bool:
    verdict = detect_verdict_from_review(text, agg)
    if verdict == "keep":
        return True
    lowered = str(text or "").lower()
    return any(pattern in lowered for pattern in KEEP_SIGNAL_PATTERNS)


def discover_candidate_base_pairs(source_root: Path, base_rotation_deg: int) -> List[dict]:
    records = discover_pair_dirs(source_root, requested_pair_names=None)
    filtered = [record for record in records if int(record["rotation_deg"]) == int(base_rotation_deg)]
    filtered.sort(key=lambda item: item["pair_name"])
    return filtered


def evaluate_candidate_pool(source_root: Path, args) -> dict:
    base_rotation_deg = int(args.base_rotation_deg)
    candidate_pairs = discover_candidate_base_pairs(source_root, base_rotation_deg=base_rotation_deg)
    passed: List[dict] = []
    rejected: List[dict] = []

    for pair_record in candidate_pairs:
        pair_dir = Path(pair_record["pair_dir"])
        obj_id = str(pair_record["obj_id"])
        round_records = collect_round_records(pair_dir, obj_id)
        best_round = select_best_round(round_records)
        if best_round is None:
            rejected.append(
                {
                    "obj_id": obj_id,
                    "pair_name": pair_record["pair_name"],
                    "pair_dir": str(pair_dir),
                    "gate_passed": False,
                    "gate_failure_reasons": ["missing_best_round"],
                }
            )
            continue

        round_idx = int(best_round["round_idx"])
        agg = load_json(Path(best_round["agg_path"]), default={}) if best_round.get("agg_path") else {}
        agg = agg or {}
        trace_text = load_round_trace_text(pair_dir, round_idx)
        keep_signal = has_keep_signal(trace_text, agg)
        asset_viability, abandon_reason, abandon_confidence = detect_asset_viability(trace_text, agg)
        hybrid_score_raw = agg.get("hybrid_score", best_round.get("hybrid_score"))
        try:
            hybrid_score = float(hybrid_score_raw)
        except (TypeError, ValueError):
            hybrid_score = None

        failure_reasons: List[str] = []
        if not keep_signal:
            failure_reasons.append("missing_keep_signal")
        if hybrid_score is None:
            failure_reasons.append("missing_hybrid_score")
        elif hybrid_score < float(args.qc_hybrid_threshold):
            failure_reasons.append(
                f"hybrid_score_below_threshold:{hybrid_score:.4f}<{float(args.qc_hybrid_threshold):.4f}"
            )
        if asset_viability == "abandon":
            failure_reasons.append("asset_viability=abandon")

        payload = {
            "obj_id": obj_id,
            "pair_name": pair_record["pair_name"],
            "pair_dir": str(pair_dir),
            "best_round_idx": round_idx,
            "best_hybrid_score": hybrid_score,
            "detected_verdict": detect_verdict_from_review(trace_text, agg),
            "keep_signal": keep_signal,
            "asset_viability": asset_viability,
            "abandon_reason": abandon_reason,
            "abandon_confidence": abandon_confidence,
            "selection_reason": best_round.get("selection_reason"),
            "review_excerpt": " ".join(trace_text.split())[:500],
            "gate_passed": not failure_reasons,
            "gate_failure_reasons": failure_reasons,
        }
        if payload["gate_passed"]:
            passed.append(payload)
        else:
            rejected.append(payload)

    passed.sort(key=lambda item: (-(item["best_hybrid_score"] or -1.0), item["obj_id"]))
    rejected.sort(key=lambda item: (item["obj_id"], item["pair_name"]))
    return {
        "source_root": str(source_root),
        "base_rotation_deg": base_rotation_deg,
        "candidate_count": len(candidate_pairs),
        "accepted": passed,
        "rejected": rejected,
    }


def assign_candidates_to_angles(accepted: List[dict], angles: List[int], quota: int) -> Tuple[Dict[int, List[dict]], List[dict]]:
    assignments: Dict[int, List[dict]] = {int(angle): [] for angle in angles}
    pool = [dict(item) for item in accepted]
    if not angles or quota <= 0:
        return assignments, pool

    cursor = 0
    while pool and any(len(assignments[angle]) < quota for angle in angles):
        angle = int(angles[cursor % len(angles)])
        cursor += 1
        if len(assignments[angle]) >= quota:
            continue
        assignments[angle].append(pool.pop(0))
    return assignments, pool


def update_group_manifest(group_entry: dict, pool: dict, assignments: Dict[int, List[dict]], leftovers: List[dict], quota: int):
    accepted = pool["accepted"]
    rejected = pool["rejected"]
    group_entry["accepted_object_count"] = len(accepted)
    group_entry["rejected_object_count"] = len(rejected)
    group_entry["accepted_counts_by_angle"] = {str(angle): len(assignments.get(int(angle), [])) for angle in assignments}
    group_entry["leftover_accepted_object_count"] = len(leftovers)
    group_entry["selected_objects_by_angle"] = {
        str(angle): [
            {
                "obj_id": item["obj_id"],
                "pair_name": item["pair_name"],
                "best_round_idx": item["best_round_idx"],
                "best_hybrid_score": item["best_hybrid_score"],
                "pair_dir": item["pair_dir"],
            }
            for item in assignments[int(angle)]
        ]
        for angle in assignments
    }
    group_entry["accepted_object_preview"] = [
        {
            "obj_id": item["obj_id"],
            "pair_name": item["pair_name"],
            "best_hybrid_score": item["best_hybrid_score"],
        }
        for item in accepted[: min(10, len(accepted))]
    ]
    group_entry["rejected_object_preview"] = [
        {
            "obj_id": item["obj_id"],
            "pair_name": item["pair_name"],
            "gate_failure_reasons": item["gate_failure_reasons"],
        }
        for item in rejected[: min(10, len(rejected))]
    ]
    group_entry["materialization_status"] = "planned" if accepted else "insufficient_candidates"
    group_entry["requested_total_objects"] = quota * len(assignments)


def load_base_split(base_split_root: Path) -> dict:
    train_rows = load_jsonl(base_split_root / "pairs" / "train_pairs.jsonl")
    val_rows = load_jsonl(base_split_root / "pairs" / "val_pairs.jsonl")
    test_rows = load_jsonl(base_split_root / "pairs" / "test_pairs.jsonl")
    split_payload = load_json(base_split_root / "object_splits.json", default={}) or {}

    if split_payload:
        train_objects = sorted(str(item) for item in split_payload.get("train_objects", []))
        val_objects = sorted(str(item) for item in split_payload.get("val_objects", []))
        test_objects = sorted(str(item) for item in split_payload.get("test_objects", []))
    else:
        train_objects = sorted({row["obj_id"] for row in train_rows})
        val_objects = sorted({row["obj_id"] for row in val_rows})
        test_objects = sorted({row["obj_id"] for row in test_rows})

    return {
        "train_rows": train_rows,
        "val_rows": val_rows,
        "test_rows": test_rows,
        "train_objects": train_objects,
        "val_objects": val_objects,
        "test_objects": test_objects,
    }


def stage_selected_pairs_root(assignments: Dict[int, List[dict]], staged_root: Path, link_mode: str) -> dict:
    ensure_clean_dir(staged_root)
    manifest = {"objects": []}
    seen_names = set()
    for angle, items in assignments.items():
        for item in items:
            pair_name = str(item["pair_name"])
            if pair_name in seen_names:
                raise ValueError(f"Duplicate selected pair_name: {pair_name}")
            seen_names.add(pair_name)
            src = Path(item["pair_dir"]).resolve()
            dst = staged_root / pair_name
            link_or_copy_dir(src, dst, link_mode)
            manifest["objects"].append(
                {
                    "obj_id": item["obj_id"],
                    "pair_name": pair_name,
                    "assigned_angle": int(angle),
                    "pair_dir": str(src),
                }
            )
    manifest["objects"].sort(key=lambda item: (item["assigned_angle"], item["obj_id"]))
    save_json(staged_root / "selection_manifest.json", manifest)
    return manifest


def run_export_rotation8(
    *,
    source_root: Path,
    output_root: Path,
    args,
) -> dict:
    if not args.blender_bin or not args.meshes_dir or not args.scene_template:
        raise ValueError(
            "materialize-datasets requires --blender-bin, --meshes-dir, and --scene-template"
        )
    cmd = [
        args.python_bin,
        str(EXPORT_ROTATION8_SCRIPT),
        "--source-root",
        str(source_root),
        "--output-dir",
        str(output_root),
        "--gpus",
        str(args.gpus),
        "--python",
        str(args.python_bin),
        "--blender",
        str(args.blender_bin),
        "--meshes-dir",
        str(args.meshes_dir),
        "--scene-template",
        str(args.scene_template),
        "--base-rotation-deg",
        str(int(args.base_rotation_deg)),
    ]
    started = time.time()
    completed = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    elapsed_seconds = round(time.time() - started, 3)
    gpu_groups = [chunk for chunk in str(args.gpus).replace(";", ",").split(",") if chunk.strip()]
    gpu_hours = round((elapsed_seconds / 3600.0) * max(1, len(gpu_groups)), 4)
    return {
        "command": cmd,
        "return_code": completed.returncode,
        "elapsed_seconds": elapsed_seconds,
        "gpu_hours_estimate": gpu_hours,
        "stdout_tail": completed.stdout[-8000:],
    }


def candidate_pair_lookup(candidate_trainready_root: Path) -> Dict[Tuple[str, int], dict]:
    lookup: Dict[Tuple[str, int], dict] = {}
    for row in load_jsonl(candidate_trainready_root / "pairs" / "train_pairs.jsonl"):
        key = (str(row["obj_id"]), int(row["target_rotation_deg"]))
        lookup[key] = row
    return lookup


def materialize_group_assets(base_root: Path, candidate_root: Path, output_root: Path, selected_obj_ids: List[str], asset_link_mode: str):
    views_out = output_root / "views"
    objects_out = output_root / "objects"
    views_out.mkdir(parents=True, exist_ok=True)
    objects_out.mkdir(parents=True, exist_ok=True)

    for obj_dir in sorted((base_root / "views").iterdir(), key=lambda p: p.name):
        link_or_copy_dir(obj_dir.resolve(), views_out / obj_dir.name, asset_link_mode)
    for obj_dir in sorted((base_root / "objects").iterdir(), key=lambda p: p.name):
        link_or_copy_dir(obj_dir.resolve(), objects_out / obj_dir.name, asset_link_mode)

    for obj_id in sorted(selected_obj_ids):
        view_src = (candidate_root / "views" / obj_id).resolve()
        obj_src = (candidate_root / "objects" / obj_id).resolve()
        if not view_src.exists():
            raise FileNotFoundError(f"Missing candidate view dir: {view_src}")
        if not obj_src.exists():
            raise FileNotFoundError(f"Missing candidate object dir: {obj_src}")
        if (views_out / obj_id).exists() or (views_out / obj_id).is_symlink():
            raise ValueError(f"Object id collision in views/: {obj_id}")
        if (objects_out / obj_id).exists() or (objects_out / obj_id).is_symlink():
            raise ValueError(f"Object id collision in objects/: {obj_id}")
        link_or_copy_dir(view_src, views_out / obj_id, asset_link_mode)
        link_or_copy_dir(obj_src, objects_out / obj_id, asset_link_mode)


def build_augmented_split_root(
    *,
    base_split_root: Path,
    candidate_trainready_root: Path,
    output_root: Path,
    group_name: str,
    assignments: Dict[int, List[dict]],
    augmentation_manifest: dict,
    asset_link_mode: str,
) -> dict:
    base = load_base_split(base_split_root)
    candidate_lookup = candidate_pair_lookup(candidate_trainready_root)
    selected_objects = []
    added_rows: List[dict] = []
    assignment_payload: Dict[str, dict] = {}

    for angle, items in assignments.items():
        for item in items:
            obj_id = str(item["obj_id"])
            key = (obj_id, int(angle))
            if key not in candidate_lookup:
                raise KeyError(f"Missing candidate pair for {key} in {candidate_trainready_root}")
            row = dict(candidate_lookup[key])
            row["split"] = "train"
            added_rows.append(row)
            selected_objects.append(obj_id)
            assignment_payload[obj_id] = {
                "assigned_angle": int(angle),
                "pair_name": item["pair_name"],
                "source_pair_dir": item["pair_dir"],
                "best_round_idx": item["best_round_idx"],
                "best_hybrid_score": item["best_hybrid_score"],
            }

    if len(selected_objects) != len(set(selected_objects)):
        raise ValueError(f"{group_name} selected duplicate obj_ids: {selected_objects}")

    ensure_clean_dir(output_root)
    materialize_group_assets(
        base_root=base_split_root,
        candidate_root=candidate_trainready_root,
        output_root=output_root,
        selected_obj_ids=selected_objects,
        asset_link_mode=asset_link_mode,
    )

    train_rows = sorted(base["train_rows"] + added_rows, key=lambda row: str(row["pair_id"]))
    val_rows = list(base["val_rows"])
    test_rows = list(base["test_rows"])
    all_rows = train_rows + val_rows + test_rows

    write_jsonl(output_root / "pairs" / "train_pairs.jsonl", train_rows)
    write_jsonl(output_root / "pairs" / "val_pairs.jsonl", val_rows)
    write_jsonl(output_root / "pairs" / "test_pairs.jsonl", test_rows)
    write_jsonl(output_root / "pairs" / "all_pairs.jsonl", all_rows)
    write_csv(output_root / "pairs" / "train_pairs.csv", train_rows)
    write_csv(output_root / "pairs" / "val_pairs.csv", val_rows)
    write_csv(output_root / "pairs" / "test_pairs.csv", test_rows)
    write_csv(output_root / "pairs" / "all_pairs.csv", all_rows)

    object_splits = {
        "seed": int(augmentation_manifest["config"]["train_seed"]),
        "train_objects": sorted(base["train_objects"] + selected_objects),
        "val_objects": list(base["val_objects"]),
        "test_objects": list(base["test_objects"]),
        "added_train_objects": sorted(selected_objects),
        "group_name": group_name,
    }
    save_json(output_root / "object_splits.json", object_splits)

    split_rows = (
        [{"obj_id": obj_id, "split": "train"} for obj_id in object_splits["train_objects"]]
        + [{"obj_id": obj_id, "split": "val"} for obj_id in object_splits["val_objects"]]
        + [{"obj_id": obj_id, "split": "test"} for obj_id in object_splits["test_objects"]]
    )
    write_csv(output_root / "object_splits.csv", split_rows)

    group_assignments = {
        "group_name": group_name,
        "selected_objects": assignment_payload,
        "non_target_policy": "held_out_diagnostic_only",
    }
    save_json(output_root / "augmentation_assignments.json", group_assignments)

    summary = {
        "success": True,
        "created_at": now_iso(),
        "dataset_type": "rotation8_feedback_augmented_split",
        "group_name": group_name,
        "source_base_split_root": str(base_split_root),
        "source_candidate_trainready_root": str(candidate_trainready_root),
        "output_root": str(output_root),
        "asset_link_mode": asset_link_mode,
        "object_counts": {
            "train": len(object_splits["train_objects"]),
            "val": len(object_splits["val_objects"]),
            "test": len(object_splits["test_objects"]),
            "added_train": len(selected_objects),
        },
        "pair_counts": {
            "train": len(train_rows),
            "val": len(val_rows),
            "test": len(test_rows),
            "added_train": len(added_rows),
            "total": len(all_rows),
        },
        "assigned_angles": {
            str(angle): len(items)
            for angle, items in sorted(assignments.items(), key=lambda item: int(item[0]))
        },
    }
    save_json(output_root / "summary.json", summary)
    save_json(
        output_root / "manifest.json",
        {
            "summary": summary,
            "augmentation_assignments": group_assignments,
            "pairs": all_rows,
        },
    )
    return summary


def materialize_group(
    *,
    group_name: str,
    group_entry: dict,
    assignments: Dict[int, List[dict]],
    base_split_root: Path,
    output_root: Path,
    args,
) -> dict:
    selected_count = sum(len(items) for items in assignments.values())
    if selected_count == 0:
        return {
            "group_name": group_name,
            "status": "no_selected_objects",
            "selected_object_count": 0,
        }

    staged_pairs_root = output_root / "staged_pair_roots" / group_name
    stage_selected_pairs_root(assignments, staged_pairs_root, args.candidate_link_mode)

    consistent_root = output_root / "datasets" / f"{group_name}_rotation8_consistent"
    export_result = run_export_rotation8(
        source_root=staged_pairs_root,
        output_root=consistent_root,
        args=args,
    )
    if export_result["return_code"] != 0:
        return {
            "group_name": group_name,
            "status": "export_failed",
            "selected_object_count": selected_count,
            "staged_pairs_root": str(staged_pairs_root),
            "consistent_root": str(consistent_root),
            "export_result": export_result,
        }

    trainready_root = output_root / "datasets" / f"{group_name}_rotation8_trainready"
    build_trainready_dataset(
        source_root=consistent_root,
        output_root=trainready_root,
        source_rotation_deg=int(args.base_rotation_deg),
        target_rotations=list(TARGET_ROTATIONS),
        copy_mode=args.trainready_copy_mode,
    )

    split_root = output_root / "datasets" / f"{group_name}_rotation8_split"
    split_summary = build_augmented_split_root(
        base_split_root=base_split_root,
        candidate_trainready_root=trainready_root,
        output_root=split_root,
        group_name=group_name,
        assignments=assignments,
        augmentation_manifest={"config": {"train_seed": int(args.train_seed)}},
        asset_link_mode=args.asset_link_mode,
    )

    return {
        "group_name": group_name,
        "status": "materialized",
        "selected_object_count": selected_count,
        "staged_pairs_root": str(staged_pairs_root),
        "consistent_root": str(consistent_root),
        "trainready_root": str(trainready_root),
        "split_root": str(split_root),
        "export_result": export_result,
        "split_summary": split_summary,
    }


def build_dataset_lineage(
    *,
    trace_summary: dict,
    hard_case_report: dict,
    augmentation_manifest: dict,
    base_split_root: Path,
    args,
    materialized_groups: dict,
) -> dict:
    return {
        "created_at": now_iso(),
        "controller": "rotation8_feedback_experiment_v1_1",
        "source_training_run_id": trace_summary["run_id"],
        "parent_dataset_root": str(base_split_root),
        "hard_angle_evidence": {
            "decision": hard_case_report["decision"],
            "selected_hard_angles": hard_case_report["selected_hard_angles"],
            "gap_ratio": hard_case_report["gap_ratio"],
            "overall_pair_score_median": hard_case_report["overall_pair_score_median"],
            "topk_angle_median_mean": hard_case_report["topk_angle_median_mean"],
        },
        "control_angle_seed": int(args.random_control_seed),
        "training_eval_generation_seeds": {
            "train_seed": int(args.train_seed),
            "random_control_seed": int(args.random_control_seed),
        },
        "new_root_path": {
            "baseline": str(base_split_root),
            "feedback-round1": materialized_groups.get("feedback-round1", {}).get("split_root"),
            "random-augment": materialized_groups.get("random-augment", {}).get("split_root"),
        },
        "groups": materialized_groups,
        "artifacts": {
            "hard_case_report": "hard_case_report.json",
            "augmentation_manifest": "augmentation_manifest.json",
            "dataset_lineage": "dataset_lineage.json",
        },
        "notes": {
            "official_training_protocol": "all groups train from scratch with the same config",
            "val_test_policy": "reuse original object-disjoint val/test without injecting new objects",
            "non_target_policy": "new objects keep full rotation8 assets, but only assigned angle enters train",
        },
    }


def main():
    args = parse_args()
    base_split_root = Path(args.base_split_root).resolve()
    trace_root = Path(args.trace_root).resolve()
    output_root = Path(args.output_dir).resolve()
    assert_new_output_root(output_root)

    trace_rows, trace_summary = load_pair_loss_trace(trace_root)
    hard_case_report = analyze_hard_cases(trace_rows, args)
    hard_case_report["trace_files"] = trace_summary["trace_files"]
    hard_case_report["run_id"] = trace_summary["run_id"]
    save_json(output_root / "hard_case_report.json", hard_case_report)

    augmentation_manifest = build_augmentation_manifest(trace_summary, hard_case_report, base_split_root, args)
    if hard_case_report["decision"] == "NO_UPDATE":
        augmentation_manifest["groups"]["feedback-round1"]["materialization_status"] = "skipped_no_update"
        augmentation_manifest["groups"]["random-augment"]["materialization_status"] = "skipped_no_update"

    assignments_by_group = {}
    if hard_case_report["decision"] != "NO_UPDATE":
        group_specs = [
            ("feedback-round1", args.feedback_source_root, augmentation_manifest["targeted_angles"]),
            ("random-augment", args.random_source_root, augmentation_manifest["random_control_angles"]),
        ]
        quota = int(args.augmentation_quota_per_angle)
        for group_name, source_root_raw, angles in group_specs:
            if not angles:
                assignments_by_group[group_name] = {}
                continue
            if source_root_raw is None:
                augmentation_manifest["groups"][group_name]["materialization_status"] = "pending_candidate_generation"
                assignments_by_group[group_name] = {int(angle): [] for angle in angles}
                continue
            pool = evaluate_candidate_pool(Path(source_root_raw).resolve(), args)
            assignments, leftovers = assign_candidates_to_angles(pool["accepted"], [int(angle) for angle in angles], quota)
            update_group_manifest(augmentation_manifest["groups"][group_name], pool, assignments, leftovers, quota)
            assignments_by_group[group_name] = assignments

    save_json(output_root / "augmentation_manifest.json", augmentation_manifest)

    materialized_groups: Dict[str, dict] = {
        "baseline": {
            "group_name": "baseline",
            "status": "ready",
            "split_root": str(base_split_root),
            "selected_object_count": 0,
        }
    }

    if args.materialize_datasets:
        if hard_case_report["decision"] == "NO_UPDATE":
            materialized_groups["feedback-round1"] = {"group_name": "feedback-round1", "status": "skipped_no_update"}
            materialized_groups["random-augment"] = {"group_name": "random-augment", "status": "skipped_no_update"}
        else:
            for group_name in ("feedback-round1", "random-augment"):
                source_root = augmentation_manifest["groups"][group_name].get("source_root")
                if source_root is None:
                    materialized_groups[group_name] = {
                        "group_name": group_name,
                        "status": "pending_candidate_generation",
                    }
                    continue
                materialized_groups[group_name] = materialize_group(
                    group_name=group_name,
                    group_entry=augmentation_manifest["groups"][group_name],
                    assignments=assignments_by_group.get(group_name, {}),
                    base_split_root=base_split_root,
                    output_root=output_root,
                    args=args,
                )
                if materialized_groups[group_name]["status"] == "materialized":
                    augmentation_manifest["groups"][group_name]["materialization_status"] = "ready"
                    augmentation_manifest["groups"][group_name]["split_root"] = materialized_groups[group_name]["split_root"]
            save_json(output_root / "augmentation_manifest.json", augmentation_manifest)

    lineage = build_dataset_lineage(
        trace_summary=trace_summary,
        hard_case_report=hard_case_report,
        augmentation_manifest=augmentation_manifest,
        base_split_root=base_split_root,
        args=args,
        materialized_groups=materialized_groups,
    )
    save_json(output_root / "dataset_lineage.json", lineage)

    summary = {
        "run_id": trace_summary["run_id"],
        "decision": hard_case_report["decision"],
        "targeted_angles": augmentation_manifest["targeted_angles"],
        "random_control_angles": augmentation_manifest["random_control_angles"],
        "materialized_groups": {
            name: payload.get("status")
            for name, payload in materialized_groups.items()
        },
        "output_root": str(output_root),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
