#!/usr/bin/env python3
"""Select non-duplicate new object concepts for feedback-driven scaling."""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional


DEFAULT_CANDIDATES = {
    "daily_item": [
        "floor_lamp",
        "laundry_basket",
        "rolling_office_chair",
        "small_bookshelf",
        "tool_cart",
        "garden_hose_reel",
        "portable_air_compressor",
        "folding_table",
        "plastic_storage_bin",
        "standing_fan",
        "camping_stove",
        "pet_carrier",
        "vacuum_cleaner",
        "electric_drill",
        "child_stroller",
        "luggage_cart",
        "cooler_bag",
        "metal_bucket",
        "paint_can",
        "shoe_rack",
    ],
    "vehicle": [
        "cargo_bike",
        "electric_scooter",
        "mini_excavator",
        "forklift",
        "utility_cart",
        "tow_truck",
        "compact_tractor",
        "snowmobile",
        "go_kart",
        "mobility_scooter",
    ],
    "street_object": [
        "bus_stop_sign",
        "construction_barrel",
        "newspaper_box",
        "bike_rack",
        "street_planter",
        "temporary_road_sign",
        "pedestrian_crossing_sign",
        "portable_fence_panel",
        "traffic_delineator_post",
        "outdoor_notice_board",
    ],
    "sports_item": [
        "golf_bag",
        "hockey_stick",
        "snowboard",
        "boxing_bag",
        "kayak",
        "exercise_bike",
        "dumbbell_rack",
        "volleyball",
        "cricket_bat",
        "archery_target",
    ],
}


def read_json(path: Optional[Path], default=None):
    if not path or not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def load_objects(path: Path) -> List[dict]:
    payload = read_json(path, default=[])
    if not isinstance(payload, list):
        raise ValueError(f"Object file must be a list: {path}")
    return [dict(item) for item in payload if isinstance(item, dict)]


def load_existing_names(objects: List[dict], prompts_path: Optional[Path]) -> set[str]:
    names = {normalize_name(str(item.get("name", ""))) for item in objects if item.get("name")}
    prompts = read_json(prompts_path, default=[]) if prompts_path else []
    if isinstance(prompts, list):
        for item in prompts:
            if isinstance(item, dict):
                for key in ("name", "object_name", "description"):
                    if item.get(key):
                        names.add(normalize_name(str(item[key])))
    return {name for name in names if name}


def next_object_number(objects: List[dict], explicit_start: Optional[int]) -> int:
    if explicit_start is not None:
        return explicit_start
    max_seen = 0
    for item in objects:
        match = re.match(r"obj_(\d+)$", str(item.get("id", "")))
        if match:
            max_seen = max(max_seen, int(match.group(1)))
    return max_seen + 1


def largest_remainder_quota(category_counts: Counter, count: int) -> Dict[str, int]:
    total = sum(category_counts.values())
    if total <= 0:
        raise ValueError("Cannot infer category distribution from empty objects")
    raw = {cat: count * n / total for cat, n in category_counts.items()}
    quota = {cat: int(math.floor(value)) for cat, value in raw.items()}
    remaining = count - sum(quota.values())
    ranked = sorted(raw, key=lambda cat: (raw[cat] - quota[cat], category_counts[cat], cat), reverse=True)
    for cat in ranked[:remaining]:
        quota[cat] += 1
    return {cat: quota[cat] for cat in sorted(quota) if quota[cat] > 0}


def load_quota(path: Optional[Path]) -> Optional[Dict[str, int]]:
    payload = read_json(path, default=None) if path else None
    if payload is None:
        return None
    quota = payload.get("category_quota", payload)
    return {str(cat): int(value) for cat, value in quota.items() if int(value) > 0}


def load_candidates(path: Optional[Path]) -> Dict[str, List[str]]:
    payload = read_json(path, default=None) if path else None
    if payload is None:
        return DEFAULT_CANDIDATES
    if isinstance(payload, list):
        by_category: Dict[str, List[str]] = {}
        for item in payload:
            if isinstance(item, dict) and item.get("name") and item.get("category"):
                by_category.setdefault(str(item["category"]), []).append(str(item["name"]))
        return by_category
    return {str(category): [str(name) for name in names] for category, names in payload.items()}


def select_objects(
    *,
    existing_names: set[str],
    count: int,
    start_number: int,
    category_quota: Dict[str, int],
    candidates: Dict[str, List[str]],
) -> tuple[List[dict], List[dict]]:
    selected: List[dict] = []
    used_names = set(existing_names)
    next_number = start_number
    fallback_candidates = []
    for fallback_category in sorted(candidates):
        for fallback_name in candidates.get(fallback_category, []):
            normalized = normalize_name(fallback_name)
            if normalized:
                fallback_candidates.append((fallback_category, normalized))

    fallback_used = []
    for category in sorted(category_quota):
        needed = int(category_quota[category])
        available = [normalize_name(name) for name in candidates.get(category, [])]
        chosen = []
        for name in available:
            if name and name not in used_names:
                chosen.append(name)
                used_names.add(name)
            if len(chosen) == needed:
                break
        while len(chosen) < needed:
            borrowed = None
            for fallback_category, fallback_name in fallback_candidates:
                if fallback_name not in used_names:
                    borrowed = (fallback_category, fallback_name)
                    break
            if borrowed is None:
                synthetic_name = normalize_name(f"feedback_{category}_object_{next_number + len(chosen):03d}")
                borrowed = (category, synthetic_name)
            fallback_category, fallback_name = borrowed
            chosen.append((fallback_name, fallback_category))
            used_names.add(fallback_name)
            fallback_used.append({
                "requested_category": category,
                "selected_category": fallback_category,
                "name": fallback_name,
            })
        chosen = [
            item if isinstance(item, tuple) else (item, category)
            for item in chosen
        ]
        for name, selected_category in chosen:
            selected.append({"id": f"obj_{next_number:03d}", "name": name, "category": selected_category})
            next_number += 1
    if len(selected) != count:
        raise ValueError(f"Selected {len(selected)} objects, expected {count}")
    return selected, fallback_used


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build feedback expansion object list")
    parser.add_argument("--existing-objects-file", required=True)
    parser.add_argument("--existing-prompts-json", default=None)
    parser.add_argument("--candidate-file", default=None)
    parser.add_argument("--category-quota-json", default=None)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--start-id-number", type=int, default=None)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    existing_objects = load_objects(Path(args.existing_objects_file))
    existing_names = load_existing_names(
        existing_objects,
        Path(args.existing_prompts_json) if args.existing_prompts_json else None,
    )
    category_counts = Counter(str(item.get("category")) for item in existing_objects if item.get("category"))
    category_quota = load_quota(Path(args.category_quota_json) if args.category_quota_json else None)
    if category_quota is None:
        category_quota = largest_remainder_quota(category_counts, args.count)
    if sum(category_quota.values()) != args.count:
        raise ValueError(f"Category quota sums to {sum(category_quota.values())}, expected {args.count}")

    selected, fallback_used = select_objects(
        existing_names=existing_names,
        count=args.count,
        start_number=next_object_number(existing_objects, args.start_id_number),
        category_quota=category_quota,
        candidates=load_candidates(Path(args.candidate_file) if args.candidate_file else None),
    )
    write_json(Path(args.output), selected)
    summary = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "output": str(Path(args.output).resolve()),
        "count": len(selected),
        "category_quota": category_quota,
        "fallback_used": fallback_used,
        "object_ids": [item["id"] for item in selected],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
