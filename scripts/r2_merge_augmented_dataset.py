#!/usr/bin/env python3
"""Merge R1 augmented_dataset (70 obj) + new_trainready (24 obj) into r2 augmented_dataset.

Schema unified to 19 cols (drop bbox/raw and t2i_prompt). views/ and objects/ are symlinks
into source dirs to avoid duplication.

Run on 68: python3 r2_merge_augmented_dataset.py
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

W = Path("/gemini/platform/public/aigc/aigc_image/zhanghy56_intern/zhangqisong/data-build")
R1_DIR = W / "feedback_loop_runs/v2_scaling_r1/round_1/augmented_dataset"
NEW_DIR = W / "feedback_loop_runs/v2_scaling_r2_quality_gate/round_1/new_trainready"
OUT_DIR = W / "feedback_loop_runs/v2_scaling_r2_quality_gate/round_1/augmented_dataset"

UNIFIED_FIELDS = [
    "pair_id", "obj_id", "task_type", "split", "prompt_version",
    "source_rotation_deg", "target_rotation_deg",
    "source_view_name", "target_view_name",
    "instruction", "object_description",
    "source_image", "target_image",
    "source_mask", "target_mask",
    "source_render_metadata", "target_render_metadata",
    "source_control_state", "target_control_state",
]


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize_row(row: dict, default_prompt_version: str = "v3") -> dict:
    out = {k: row.get(k, "") for k in UNIFIED_FIELDS}
    if not out["prompt_version"]:
        out["prompt_version"] = default_prompt_version
    return out


def write_pairs(rows: list[dict], pairs_dir: Path, name: str) -> None:
    pairs_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = pairs_dir / f"{name}.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    csv_path = pairs_dir / f"{name}.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=UNIFIED_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def link_subdirs(src_root: Path, dst_root: Path, sub: str) -> int:
    """Create symlinks dst_root/sub/<obj> -> src_root/sub/<obj> for every obj subdir."""
    src = src_root / sub
    dst = dst_root / sub
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for obj_dir in sorted(src.iterdir()):
        if not obj_dir.is_dir():
            continue
        link = dst / obj_dir.name
        if link.exists() or link.is_symlink():
            link.unlink()
        os.symlink(obj_dir.resolve(), link)
        n += 1
    return n


def main() -> None:
    print(f"[INFO] R1_DIR  = {R1_DIR}")
    print(f"[INFO] NEW_DIR = {NEW_DIR}")
    print(f"[INFO] OUT_DIR = {OUT_DIR}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: link views/ and objects/
    n_views_r1 = link_subdirs(R1_DIR, OUT_DIR, "views")
    n_objs_r1 = link_subdirs(R1_DIR, OUT_DIR, "objects")
    n_views_new = link_subdirs(NEW_DIR, OUT_DIR, "views")
    n_objs_new = link_subdirs(NEW_DIR, OUT_DIR, "objects")
    print(f"[LINK] views: R1={n_views_r1}, new={n_views_new}")
    print(f"[LINK] objects: R1={n_objs_r1}, new={n_objs_new}")

    # Step 2: load + normalize pairs
    r1_pairs = R1_DIR / "pairs"
    new_pairs = NEW_DIR / "pairs"

    r1_train = [normalize_row(r) for r in load_jsonl(r1_pairs / "train_pairs.jsonl")]
    r1_val = [normalize_row(r) for r in load_jsonl(r1_pairs / "val_pairs.jsonl")]
    r1_test = [normalize_row(r) for r in load_jsonl(r1_pairs / "test_pairs.jsonl")]
    new_train = [normalize_row(r) for r in load_jsonl(new_pairs / "train_pairs.jsonl")]

    print(f"[LOAD] R1: train={len(r1_train)} val={len(r1_val)} test={len(r1_test)}")
    print(f"[LOAD] new: train={len(new_train)} (val/test = 0, all train per source)")

    # Sanity: new pairs all split=train
    bad = [r for r in new_train if r["split"] != "train"]
    if bad:
        raise ValueError(f"new dataset has non-train rows: {len(bad)}")

    train = r1_train + new_train
    val = r1_val
    test = r1_test
    all_pairs = train + val + test

    # Step 3: write merged pairs
    out_pairs = OUT_DIR / "pairs"
    write_pairs(train, out_pairs, "train_pairs")
    write_pairs(val, out_pairs, "val_pairs")
    write_pairs(test, out_pairs, "test_pairs")
    write_pairs(all_pairs, out_pairs, "all_pairs")

    print(f"[WRITE] train={len(train)} val={len(val)} test={len(test)} all={len(all_pairs)}")

    # Step 4: object splits
    train_objs = sorted({r["obj_id"] for r in train})
    val_objs = sorted({r["obj_id"] for r in val})
    test_objs = sorted({r["obj_id"] for r in test})
    overlap_tv = set(train_objs) & set(val_objs)
    overlap_tt = set(train_objs) & set(test_objs)
    if overlap_tv or overlap_tt:
        print(f"[WARN] obj overlap: train∩val={overlap_tv}, train∩test={overlap_tt}")

    splits_path = OUT_DIR / "object_splits.json"
    with splits_path.open("w", encoding="utf-8") as f:
        json.dump(
            {"train": train_objs, "val": val_objs, "test": test_objs},
            f,
            ensure_ascii=False,
            indent=2,
        )

    csv_path = OUT_DIR / "object_splits.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["obj_id", "split"])
        for s, lst in [("train", train_objs), ("val", val_objs), ("test", test_objs)]:
            for o in lst:
                w.writerow([o, s])

    # Step 5: manifest.json + summary.json
    manifest = {
        "dataset_type": "rotation_edit_trainready_merged",
        "source_dirs": {
            "r1_augmented": str(R1_DIR),
            "new_trainready": str(NEW_DIR),
        },
        "total_objects": len(train_objs) + len(val_objs) + len(test_objs),
        "split_counts": {
            "train_objs": len(train_objs),
            "val_objs": len(val_objs),
            "test_objs": len(test_objs),
            "train_pairs": len(train),
            "val_pairs": len(val),
            "test_pairs": len(test),
            "total_pairs": len(all_pairs),
        },
        "fields": UNIFIED_FIELDS,
        "prompt_version": "v3",
    }
    with (OUT_DIR / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    with (OUT_DIR / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(manifest["split_counts"], f, ensure_ascii=False, indent=2)

    print(f"[DONE] OUT_DIR={OUT_DIR}")
    print(f"[DONE] objects total={manifest['total_objects']}, pairs total={len(all_pairs)}")


if __name__ == "__main__":
    main()
