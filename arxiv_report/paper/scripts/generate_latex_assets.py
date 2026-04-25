from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / "eval"
ASSETS = ROOT / "assets"
OUT = ROOT / "paper" / "latex_assets"
FIG_DIR = OUT / "figures"
TAB_DIR = OUT / "tables"
DATA_DIR = OUT / "data"

ANGLE_DEG = [45, 90, 135, 180, 225, 270, 315, 360]

COLORS = {
    "baseline": "#4C78A8",
    "r1": "#F58518",
    "positive": "#54A24B",
    "negative": "#E45756",
    "neutral": "#6B7280",
    "threshold": "#B91C1C",
}


def ensure_dirs() -> None:
    for d in [FIG_DIR, TAB_DIR, DATA_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.5,
        }
    )


def savefig(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


def parse_markdown_table(lines: list[str], header_marker: str) -> list[list[str]]:
    start = None
    for i, line in enumerate(lines):
        if header_marker in line:
            start = i
            break
    if start is None:
        raise FileNotFoundError(f"Cannot find markdown table with marker: {header_marker}")

    rows: list[list[str]] = []
    for line in lines[start:]:
        line = line.strip()
        if not line.startswith("|"):
            if rows:
                break
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if all(set(c) <= {"-", ":"} for c in cells):
            continue
        rows.append(cells)
    return rows


def to_float(value: str) -> float:
    value = value.strip().replace("+", "")
    value = re.sub(r"[^\d.\-eE]", "", value)
    if value == "":
        return math.nan
    return float(value)


def load_report_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    md = (EVAL / "derived" / "R1_vs_exp5_spatialedit_bench.md").read_text(encoding="utf-8")
    lines = md.splitlines()

    overall_rows = parse_markdown_table(lines, "| 指标 | exp5 | R1 |")
    overall = pd.DataFrame(overall_rows[1:], columns=["metric", "exp5", "r1", "delta", "direction"])
    for c in ["exp5", "r1", "delta"]:
        overall[c] = overall[c].map(to_float)
    overall["metric_clean"] = overall["metric"].str.replace("↑", "", regex=False).str.replace("↓", "", regex=False).str.strip()

    view_rows = parse_markdown_table(lines, "| angle_idx |")
    view = pd.DataFrame(view_rows[1:], columns=["angle_idx", "description", "exp5", "r1", "delta"])
    for c in ["exp5", "r1", "delta"]:
        view[c] = view[c].map(to_float)
    view["angle_slot"] = view["angle_idx"].map(lambda x: int(str(x)))
    view["angle_deg"] = view["angle_slot"].map(lambda i: ANGLE_DEG[i])

    traditional_rows = parse_markdown_table(lines, "| angle | PSNR_exp5 |")
    trad = pd.DataFrame(
        traditional_rows[1:],
        columns=["angle_idx", "psnr_exp5", "psnr_r1", "delta_psnr", "dino_exp5", "dino_r1", "delta_dino"],
    )
    for c in ["psnr_exp5", "psnr_r1", "delta_psnr", "dino_exp5", "dino_r1", "delta_dino"]:
        trad[c] = trad[c].map(to_float)
    trad["angle_slot"] = trad["angle_idx"].map(lambda x: int(str(x)))
    trad["angle_deg"] = trad["angle_slot"].map(lambda i: ANGLE_DEG[i])

    return overall, view, trad


def load_quality_scores() -> pd.DataFrame:
    objects_path = EVAL / "v2_scaling_r1" / "feedback_expansion_r1_objects.json"
    objects = {item["id"]: item for item in json.loads(objects_path.read_text(encoding="utf-8"))}
    root = EVAL / "v2_scaling_r1" / "vlm_bootstrap"
    rows = []
    for obj_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        review_dir = obj_dir / "reviews"
        reviews = []
        for f in sorted(review_dir.glob("*_agg.json")):
            data = json.loads(f.read_text(encoding="utf-8"))
            if data.get("hybrid_score") is not None:
                reviews.append((int(data.get("round_idx", -1)), data, f.name))
        if not reviews:
            continue
        latest_round, latest, latest_file = max(reviews, key=lambda x: x[0])
        best_round, best, _ = max(reviews, key=lambda x: x[1].get("hybrid_score", -1))
        meta = objects.get(obj_dir.name, {})
        rows.append(
            {
                "obj_id": obj_dir.name,
                "name": meta.get("name", obj_dir.name),
                "category": meta.get("category", "unknown"),
                "latest_round": latest_round,
                "latest_file": latest_file,
                "hybrid_score": float(latest["hybrid_score"]),
                "best_hybrid_score": float(best["hybrid_score"]),
                "route": latest.get("hybrid_route", ""),
                "issue_tags": ";".join(latest.get("issue_tags", [])),
            }
        )
    return pd.DataFrame(rows)


def load_bootstrap_history() -> pd.DataFrame:
    root = EVAL / "v2_scaling_r1" / "vlm_bootstrap"
    rows = []
    for obj_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for f in sorted((obj_dir / "reviews").glob("*_agg.json")):
            data = json.loads(f.read_text(encoding="utf-8"))
            if data.get("hybrid_score") is None:
                continue
            rows.append(
                {
                    "obj_id": obj_dir.name,
                    "round_idx": int(data.get("round_idx", -1)),
                    "hybrid_score": float(data["hybrid_score"]),
                    "hybrid_route": data.get("hybrid_route", ""),
                    "lighting_diagnosis": data.get("lighting_diagnosis", ""),
                }
            )
    return pd.DataFrame(rows)


def load_augmentation_composition() -> tuple[pd.DataFrame, pd.DataFrame]:
    objects_path = EVAL / "v2_scaling_r1" / "feedback_expansion_r1_objects.json"
    objects = pd.DataFrame(json.loads(objects_path.read_text(encoding="utf-8")))
    train_pairs = pd.read_csv(EVAL / "v2_scaling_r1" / "trainready" / "train_pairs.csv")
    angle_counts = train_pairs["target_rotation_deg"].value_counts().sort_index().reset_index()
    angle_counts.columns = ["target_rotation_deg", "pair_count"]
    category_counts = objects["category"].value_counts().reset_index()
    category_counts.columns = ["category", "object_count"]
    return category_counts, angle_counts


def parse_vie_angle(path: str) -> int:
    m = re.search(r"/(\d{2})\.png$", str(path).replace("\\", "/"))
    if not m:
        return -1
    return int(m.group(1))


def load_vie_per_angle() -> pd.DataFrame:
    frames = []
    for label, path in [
        ("exp5", EVAL / "exp5_baseline" / "vie_score.csv"),
        ("R1", EVAL / "v2_scaling_r1" / "vie_score.csv"),
    ]:
        df = pd.read_csv(path)
        df["angle_slot"] = df["edited_image"].map(parse_vie_angle)
        df = df[df["angle_slot"] >= 0].copy()
        df["angle_deg"] = df["angle_slot"].map(lambda i: ANGLE_DEG[i])
        grouped = df.groupby("angle_deg", as_index=False)[["Score_view", "Score_cons"]].mean()
        grouped["configuration"] = label
        frames.append(grouped)
    combined = pd.concat(frames, ignore_index=True)
    return combined


def write_csvs(
    overall: pd.DataFrame,
    view: pd.DataFrame,
    trad: pd.DataFrame,
    quality: pd.DataFrame,
    bootstrap: pd.DataFrame,
    category_counts: pd.DataFrame,
    angle_counts: pd.DataFrame,
    vie_per_angle: pd.DataFrame,
) -> None:
    overall.to_csv(DATA_DIR / "overall_metrics_from_report.csv", index=False)
    view.to_csv(DATA_DIR / "per_angle_score_view_from_report.csv", index=False)
    trad.to_csv(DATA_DIR / "per_angle_psnr_dino_from_report.csv", index=False)
    quality.to_csv(DATA_DIR / "r1_hybrid_scores_from_vlm_bootstrap.csv", index=False)
    bootstrap.to_csv(DATA_DIR / "r1_bootstrap_hybrid_history.csv", index=False)
    category_counts.to_csv(DATA_DIR / "r1_category_counts.csv", index=False)
    angle_counts.to_csv(DATA_DIR / "r1_target_angle_pair_counts.csv", index=False)
    vie_per_angle.to_csv(DATA_DIR / "vie_per_angle_from_csv.csv", index=False)


def plot_overall_relative_change(overall: pd.DataFrame) -> None:
    direction = {
        "PSNR": 1,
        "SSIM": 1,
        "LPIPS": -1,
        "CLIP-I": 1,
        "DINO": 1,
        "FID": -1,
        "Score_view": 1,
        "Score_cons": 1,
        "VIE Overall": 1,
    }
    df = overall.copy()
    df["better_delta"] = df.apply(lambda r: direction[r["metric_clean"]] * (r["r1"] - r["exp5"]), axis=1)
    df["relative_improvement_pct"] = df["better_delta"] / df["exp5"].abs() * 100.0
    df.to_csv(DATA_DIR / "overall_signed_relative_change.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    colors = [COLORS["positive"] if v >= 0 else COLORS["negative"] for v in df["relative_improvement_pct"]]
    ax.bar(df["metric_clean"], df["relative_improvement_pct"], color=colors, width=0.68)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Signed relative change (%)\nhigher is better")
    ax.set_title("R1 vs. exp5: direction-normalized metric change")
    ax.tick_params(axis="x", rotation=35)
    ymin = min(df["relative_improvement_pct"].min() - 1.0, -1.0)
    ymax = max(df["relative_improvement_pct"].max() + 1.0, 1.0)
    ax.set_ylim(ymin, ymax)
    for i, v in enumerate(df["relative_improvement_pct"]):
        va = "bottom" if v >= 0 else "top"
        y = v + (0.35 if v >= 0 else -0.35)
        ax.text(i, y, f"{v:+.1f}", ha="center", va=va, fontsize=7)
    savefig(fig, "overall_signed_relative_change")


def plot_overall_metric_groups(overall: pd.DataFrame) -> None:
    groups = [
        ("Pixel / perceptual", ["PSNR", "SSIM", "LPIPS"]),
        ("Semantic / identity", ["CLIP-I", "DINO"]),
        ("VLM view / consistency", ["Score_view", "Score_cons", "VIE Overall"]),
        ("Distribution", ["FID"]),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(9.2, 2.6), gridspec_kw={"width_ratios": [3, 2, 3, 1.2]})
    for ax, (title, metrics) in zip(axes, groups):
        sub = overall.set_index("metric_clean").loc[metrics].reset_index()
        x = np.arange(len(sub))
        width = 0.36
        ax.bar(x - width / 2, sub["exp5"], width, label="exp5", color=COLORS["baseline"])
        ax.bar(x + width / 2, sub["r1"], width, label="R1", color=COLORS["r1"])
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(sub["metric_clean"], rotation=35, ha="right")
        if title == "Pixel / perceptual":
            ax.legend(frameon=False, loc="best")
    fig.suptitle("Overall metric values on SpatialEdit-Bench", y=1.03, fontsize=11)
    savefig(fig, "overall_metric_values_grouped")


def plot_per_angle(view: pd.DataFrame, trad: pd.DataFrame) -> None:
    merged = trad.merge(view[["angle_deg", "exp5", "r1", "delta"]], on="angle_deg", suffixes=("", "_score_view"))
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 6.2), sharex=True)
    specs = [
        ("delta_score_view", "Score_view delta", COLORS["positive"]),
        ("delta_psnr", "PSNR delta", COLORS["positive"]),
        ("delta_dino", "DINO delta", COLORS["negative"]),
    ]
    merged = merged.rename(columns={"delta": "delta_score_view"})
    for ax, (col, ylabel, base_color) in zip(axes, specs):
        vals = merged[col].values
        colors = [COLORS["positive"] if v >= 0 else COLORS["negative"] for v in vals]
        ax.bar(merged["angle_deg"].astype(str), vals, color=colors, width=0.72)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylabel(ylabel)
        pad = max(abs(vals.min()), abs(vals.max())) * 0.2 + 1e-6
        ax.set_ylim(vals.min() - pad, vals.max() + pad)
    axes[-1].set_xlabel("Angle slot mapped by compare.py (degrees)")
    fig.suptitle("Per-angle diagnostic deltas: R1 minus exp5", y=1.01, fontsize=11)
    savefig(fig, "per_angle_diagnostic_deltas")


def plot_dataset_scaling_quality(quality: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.3), gridspec_kw={"width_ratios": [1.0, 2.2]})

    ax = axes[0]
    configs = ["exp5", "R1"]
    train = [245, 305]
    val = [49, 49]
    test = [56, 56]
    x = np.arange(len(configs))
    ax.bar(x, train, label="train", color=COLORS["baseline"])
    ax.bar(x, val, bottom=train, label="val", color="#72B7B2")
    ax.bar(x, test, bottom=np.array(train) + np.array(val), label="test", color="#B279A2")
    ax.set_xticks(x)
    ax.set_xticklabels(configs)
    ax.set_ylabel("Pairs")
    ax.set_title("Dataset split size")
    ax.legend(frameon=False, fontsize=7)
    ax.annotate("+60 train pairs", xy=(1, 305), xytext=(0.35, 355), arrowprops={"arrowstyle": "->", "lw": 0.8}, fontsize=8)

    ax = axes[1]
    q = quality.sort_values("hybrid_score").reset_index(drop=True)
    bars = ax.bar(np.arange(len(q)), q["hybrid_score"], color=COLORS["r1"], width=0.78)
    ax.axhline(0.6, color=COLORS["threshold"], linestyle="--", linewidth=1.1, label="quality gate = 0.6")
    median = float(q["hybrid_score"].median())
    ax.axhline(median, color=COLORS["neutral"], linestyle=":", linewidth=1.0, label=f"median = {median:.2f}")
    ax.set_ylim(0, max(0.68, q["hybrid_score"].max() + 0.08))
    ax.set_title("R1 new-object render quality")
    ax.set_ylabel("latest hybrid score")
    ax.set_xticks(np.arange(len(q)))
    ax.set_xticklabels(q["obj_id"].str.replace("obj_", ""), rotation=90, fontsize=6)
    ax.legend(frameon=False, loc="upper left", fontsize=7)
    savefig(fig, "dataset_scaling_and_quality")


def plot_issue_tags(quality: pd.DataFrame) -> None:
    counter: Counter[str] = Counter()
    for tags in quality["issue_tags"].fillna(""):
        for tag in str(tags).split(";"):
            tag = tag.strip()
            if tag:
                counter[tag] += 1
    if not counter:
        return
    items = counter.most_common(10)
    labels = [k.replace("_", " ") for k, _ in items][::-1]
    vals = [v for _, v in items][::-1]
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    ax.barh(labels, vals, color=COLORS["r1"])
    ax.set_xlabel("Count across latest R1 object reviews")
    ax.set_title("Most frequent VLM/CV issue tags")
    for i, v in enumerate(vals):
        ax.text(v + 0.1, i, str(v), va="center", fontsize=8)
    savefig(fig, "r1_issue_tag_counts")


def plot_bootstrap_convergence(bootstrap: pd.DataFrame) -> None:
    if bootstrap.empty:
        return
    fig, ax = plt.subplots(figsize=(6.8, 3.3))
    for obj_id, sub in bootstrap.groupby("obj_id"):
        sub = sub.sort_values("round_idx")
        ax.plot(sub["round_idx"], sub["hybrid_score"], color="#9CA3AF", alpha=0.42, linewidth=0.9)
    mean_curve = bootstrap.groupby("round_idx", as_index=False)["hybrid_score"].mean()
    ax.plot(mean_curve["round_idx"], mean_curve["hybrid_score"], color=COLORS["r1"], linewidth=2.2, marker="o", label="mean")
    ax.axhline(0.6, color=COLORS["threshold"], linestyle="--", linewidth=1.1, label="quality gate = 0.6")
    ax.set_xlabel("VLM refinement round")
    ax.set_ylabel("Hybrid score")
    ax.set_title("R1 VLM bootstrap convergence remains below quality gate")
    ax.set_ylim(0.25, 0.68)
    ax.legend(frameon=False, loc="upper left")
    savefig(fig, "r1_bootstrap_hybrid_convergence")


def plot_augmentation_composition(category_counts: pd.DataFrame, angle_counts: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0), gridspec_kw={"width_ratios": [1.4, 1.0]})
    ax = axes[0]
    cat = category_counts.sort_values("object_count", ascending=True)
    ax.barh(cat["category"].str.replace("_", " "), cat["object_count"], color=COLORS["baseline"])
    ax.set_xlabel("New objects")
    ax.set_title("R1 object categories")
    for i, v in enumerate(cat["object_count"]):
        ax.text(v + 0.15, i, str(v), va="center", fontsize=8)
    ax.set_xlim(0, max(cat["object_count"]) + 2)

    ax = axes[1]
    angle = angle_counts.copy()
    ax.bar(angle["target_rotation_deg"].astype(str), angle["pair_count"], color=COLORS["r1"], width=0.65)
    ax.set_xlabel("Target yaw (deg)")
    ax.set_ylabel("Train pairs")
    ax.set_title("Weak-angle additions")
    ax.set_ylim(0, max(angle["pair_count"]) + 5)
    for i, v in enumerate(angle["pair_count"]):
        ax.text(i, v + 0.6, str(v), ha="center", fontsize=8)
    savefig(fig, "r1_augmentation_composition")


def plot_vie_per_angle(vie_per_angle: pd.DataFrame) -> None:
    if vie_per_angle.empty:
        return
    pivot_view = vie_per_angle.pivot(index="angle_deg", columns="configuration", values="Score_view").sort_index()
    pivot_cons = vie_per_angle.pivot(index="angle_deg", columns="configuration", values="Score_cons").sort_index()
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.0), sharex=True)
    x = np.arange(len(pivot_view.index))
    width = 0.36
    for ax, pivot, title, ylabel in [
        (axes[0], pivot_view, "VLM view score", "Score_view"),
        (axes[1], pivot_cons, "VLM consistency score", "Score_cons"),
    ]:
        ax.bar(x - width / 2, pivot["exp5"], width, label="exp5", color=COLORS["baseline"])
        ax.bar(x + width / 2, pivot["R1"], width, label="R1", color=COLORS["r1"])
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels([str(int(a)) for a in pivot.index], rotation=0)
        ax.set_ylim(0, 1.05)
    axes[0].legend(frameon=False, loc="lower right")
    fig.supxlabel("Angle slot mapped by compare.py (degrees)", y=-0.02)
    savefig(fig, "vie_per_angle_view_consistency")


def create_qualitative_grid() -> None:
    img_dir = ASSETS / "qualitative" / "ground_truth"
    objects = ["obj_051", "obj_052", "obj_053"]
    yaws = ["000", "090", "180", "270"]
    paths = [[img_dir / f"{obj}_yaw{yaw}_gt.png" for yaw in yaws] for obj in objects]
    if not all(p.exists() for row in paths for p in row):
        return

    fig, axes = plt.subplots(len(objects), len(yaws), figsize=(7.2, 5.1))
    for i, obj in enumerate(objects):
        for j, yaw in enumerate(yaws):
            ax = axes[i, j]
            img = Image.open(paths[i][j]).convert("RGB")
            img = ImageOps.contain(img, (512, 512))
            ax.imshow(img)
            ax.set_axis_off()
            if i == 0:
                ax.set_title(f"yaw{yaw}")
            if j == 0:
                ax.text(-0.04, 0.5, obj, transform=ax.transAxes, rotation=90, ha="right", va="center", fontsize=9)
    fig.suptitle("Representative scene-aware rotation renders", y=0.98, fontsize=11)
    savefig(fig, "qualitative_rotation_grid")


def latex_escape(s: str) -> str:
    return str(s).replace("_", "\\_")


def write_latex_tables(overall: pd.DataFrame, view: pd.DataFrame, trad: pd.DataFrame, quality: pd.DataFrame) -> None:
    direction_label = {
        "PSNR": "better",
        "SSIM": "better",
        "LPIPS": "better",
        "CLIP-I": "better",
        "DINO": "worse",
        "FID": "worse",
        "Score_view": "better",
        "Score_cons": "slightly worse",
        "VIE Overall": "better",
    }
    metric_latex = {
        "PSNR": "PSNR ($\\uparrow$)",
        "SSIM": "SSIM ($\\uparrow$)",
        "LPIPS": "LPIPS ($\\downarrow$)",
        "CLIP-I": "CLIP-I ($\\uparrow$)",
        "DINO": "DINO ($\\uparrow$)",
        "FID": "FID ($\\downarrow$)",
        "Score_view": "Score\\_view ($\\uparrow$)",
        "Score_cons": "Score\\_cons ($\\uparrow$)",
        "VIE Overall": "VIE Overall ($\\uparrow$)",
    }
    overall_lines = [
        "% Auto-generated by arxiv_report/paper/scripts/generate_latex_assets.py",
        "\\begin{tabular}{lrrrl}",
        "\\toprule",
        "Metric & exp5 & R1 & $\\Delta$ & Interpretation \\\\",
        "\\midrule",
    ]
    for _, r in overall.iterrows():
        metric = metric_latex[r["metric_clean"]]
        interp = direction_label[r["metric_clean"]]
        overall_lines.append(f"{metric} & {r['exp5']:.4g} & {r['r1']:.4g} & {r['delta']:+.4g} & {interp} \\\\")
    overall_lines += ["\\bottomrule", "\\end{tabular}", ""]
    (TAB_DIR / "overall_metrics_booktabs.tex").write_text("\n".join(overall_lines), encoding="utf-8")

    dataset_lines = [
        "% Auto-generated by arxiv_report/paper/scripts/generate_latex_assets.py",
        "\\begin{tabular}{lrrrrl}",
        "\\toprule",
        "Configuration & Train objects & Train pairs & Val pairs & Test pairs & Intervention \\\\",
        "\\midrule",
        "exp5 baseline & 35 & 245 & 49 & 56 & object-info prompt baseline \\\\",
        "v2 Scaling R1 & 35+20 & 305 & 49 & 56 & +60 weak-angle train pairs \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        "",
    ]
    (TAB_DIR / "dataset_stats_booktabs.tex").write_text("\n".join(dataset_lines), encoding="utf-8")

    per_angle = trad.merge(view[["angle_deg", "exp5", "r1", "delta"]], on="angle_deg")
    per_angle_lines = [
        "% Auto-generated by arxiv_report/paper/scripts/generate_latex_assets.py",
        "\\begin{tabular}{rrrrrr}",
        "\\toprule",
        "Angle & Score\\_view exp5 & Score\\_view R1 & $\\Delta$ & PSNR $\\Delta$ & DINO $\\Delta$ \\\\",
        "\\midrule",
    ]
    for _, r in per_angle.iterrows():
        per_angle_lines.append(
            f"{int(r['angle_deg'])} & {r['exp5']:.4f} & {r['r1']:.4f} & {r['delta']:+.4f} & {r['delta_psnr']:+.4f} & {r['delta_dino']:+.4f} \\\\"
        )
    per_angle_lines += ["\\bottomrule", "\\end{tabular}", ""]
    (TAB_DIR / "per_angle_diagnostics_booktabs.tex").write_text("\n".join(per_angle_lines), encoding="utf-8")

    q = quality.copy()
    q_summary = {
        "new_objects": len(q),
        "below_gate": int((q["hybrid_score"] < 0.6).sum()),
        "median": float(q["hybrid_score"].median()),
        "max": float(q["hybrid_score"].max()),
        "min": float(q["hybrid_score"].min()),
    }
    quality_lines = [
        "% Auto-generated by arxiv_report/paper/scripts/generate_latex_assets.py",
        "\\begin{tabular}{ll}",
        "\\toprule",
        "Diagnostic item & Value \\\\",
        "\\midrule",
        f"New R1 objects & {q_summary['new_objects']} \\\\",
        f"Objects below hybrid-score gate 0.6 & {q_summary['below_gate']} \\\\",
        f"Median latest hybrid score & {q_summary['median']:.3f} \\\\",
        f"Min / max latest hybrid score & {q_summary['min']:.3f} / {q_summary['max']:.3f} \\\\",
        "Verdict & inspect \\\\",
        "Next action & render-quality gating before merge \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        "",
    ]
    (TAB_DIR / "r1_quality_diagnosis_booktabs.tex").write_text("\n".join(quality_lines), encoding="utf-8")


def write_composition_table(category_counts: pd.DataFrame, angle_counts: pd.DataFrame) -> None:
    lines = [
        "% Auto-generated by arxiv_report/paper/scripts/generate_latex_assets.py",
        "\\begin{tabular}{lrl}",
        "\\toprule",
        "Group & Count & Unit \\\\",
        "\\midrule",
    ]
    for _, r in category_counts.sort_values("category").iterrows():
        lines.append(f"{latex_escape(r['category'])} & {int(r['object_count'])} & objects \\\\")
    for _, r in angle_counts.iterrows():
        lines.append(f"target yaw {int(r['target_rotation_deg'])}$^\\circ$ & {int(r['pair_count'])} & pairs \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    (TAB_DIR / "r1_augmentation_composition_booktabs.tex").write_text("\n".join(lines), encoding="utf-8")


def write_asset_readme() -> None:
    text = """# LaTeX Asset Preparation

Generated by `arxiv_report/paper/scripts/generate_latex_assets.py`.

## Figures

- `figures/overall_signed_relative_change.pdf`: direction-normalized relative changes, where positive means better.
- `figures/overall_metric_values_grouped.pdf`: grouped absolute metric values for exp5 and v2 Scaling R1.
- `figures/per_angle_diagnostic_deltas.pdf`: per-angle deltas for Score_view, PSNR, and DINO.
- `figures/vie_per_angle_view_consistency.pdf`: per-angle grouped bars for Score_view and Score_cons from VIEScore CSVs.
- `figures/dataset_scaling_and_quality.pdf`: train/val/test pair counts plus R1 new-object hybrid scores.
- `figures/r1_bootstrap_hybrid_convergence.pdf`: VLM bootstrap hybrid-score trajectories and quality-gate threshold.
- `figures/r1_augmentation_composition.pdf`: R1 new-object category mix and weak-angle pair counts.
- `figures/r1_issue_tag_counts.pdf`: most frequent VLM/CV issue tags in latest R1 object reviews.
- `figures/qualitative_rotation_grid.pdf`: representative rendered GT rotation views for obj_051--obj_053.

Each PDF has a PNG preview with the same basename.

## Tables

- `tables/overall_metrics_booktabs.tex`
- `tables/dataset_stats_booktabs.tex`
- `tables/per_angle_diagnostics_booktabs.tex`
- `tables/r1_quality_diagnosis_booktabs.tex`
- `tables/r1_augmentation_composition_booktabs.tex`

## Data

Intermediate CSV files are written under `data/` so that plotted values can be audited.
"""
    (OUT / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    configure_matplotlib()
    overall, view, trad = load_report_tables()
    quality = load_quality_scores()
    bootstrap = load_bootstrap_history()
    category_counts, angle_counts = load_augmentation_composition()
    vie_per_angle = load_vie_per_angle()
    write_csvs(overall, view, trad, quality, bootstrap, category_counts, angle_counts, vie_per_angle)
    plot_overall_relative_change(overall)
    plot_overall_metric_groups(overall)
    plot_per_angle(view, trad)
    plot_vie_per_angle(vie_per_angle)
    plot_dataset_scaling_quality(quality)
    plot_bootstrap_convergence(bootstrap)
    plot_augmentation_composition(category_counts, angle_counts)
    plot_issue_tags(quality)
    create_qualitative_grid()
    write_latex_tables(overall, view, trad, quality)
    write_composition_table(category_counts, angle_counts)
    write_asset_readme()
    print(f"Wrote LaTeX assets to {OUT}")


if __name__ == "__main__":
    main()
