import csv, os, re
from collections import defaultdict

BASE = "/gemini/platform/public/aigc/aigc_image/zhanghy56_intern/zhangqisong/data-build/SpatialEdit-Bench-Eval/csv_results"

VIEW_KEYWORDS = [
    ("front right", "front-right"),
    ("rear right", "back-right"),
    ("front left", "front-left"),
    ("rear left", "back-left"),
    ("right side", "right side"),
    ("left side", "left side"),
    ("front view", "front"),
    ("rear view", "back"),
]

def extract_view(instruction):
    inst_lower = instruction.lower()
    for kw, label in VIEW_KEYWORDS:
        if kw in inst_lower:
            return label
    return "unknown"

for mode in ["base", "ours", "fal", "ours_objinfo"]:
    csv_path = os.path.join(BASE, mode, "qwen35vl", f"{mode}_rotate_en_vie_score.csv")
    if not os.path.exists(csv_path):
        print(f"[SKIP] {mode}: not found")
        continue

    view_data = defaultdict(list)
    total = 0
    with open(csv_path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sv = float(row["Score_view"])
            sc = float(row["Score_cons"])
            view = extract_view(row["instruction"])
            view_data[view].append({"score_view": sv, "score_cons": sc})
            total += 1

    print(f"\n=== {mode} VIEScore ({total} pairs) ===")
    print(f"{'View':<14} {'N':>3} {'Score_view':>11} {'Score_cons':>11} {'Overall':>8}")
    print("-" * 55)

    ordered_views = ["right side", "front-right", "front", "front-left", "left side", "back-left", "back", "back-right"]
    all_sv, all_sc = [], []
    for view in ordered_views:
        rows = view_data.get(view, [])
        if not rows:
            continue
        sv = sum(r["score_view"] for r in rows) / len(rows)
        sc = sum(r["score_cons"] for r in rows) / len(rows)
        ov = (sv * sc) ** 0.5
        print(f"{view:<14} {len(rows):>3} {sv:>11.4f} {sc:>11.4f} {ov:>8.4f}")
        all_sv.extend([r["score_view"] for r in rows])
        all_sc.extend([r["score_cons"] for r in rows])

    if all_sv:
        avg_sv = sum(all_sv) / len(all_sv)
        avg_sc = sum(all_sc) / len(all_sc)
        avg_ov = (avg_sv * avg_sc) ** 0.5
        print(f"{'AVG':<14} {len(all_sv):>3} {avg_sv:>11.4f} {avg_sc:>11.4f} {avg_ov:>8.4f}")
