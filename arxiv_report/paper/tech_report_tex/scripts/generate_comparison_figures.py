from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures" / "generated"


SPATIALEDIT = {
    "Base": {"PSNR": 15.66, "SSIM": 0.6623, "LPIPS": 0.3304, "CLIP-I": 0.8807, "DINO": 0.8517, "FID": 65.47},
    "Public LoRA": {"PSNR": 15.76, "SSIM": 0.6545, "LPIPS": 0.3443, "CLIP-I": 0.8747, "DINO": 0.8405, "FID": 68.35},
    "Bright Dir.": {"PSNR": 16.66, "SSIM": 0.7310, "LPIPS": 0.2555, "CLIP-I": 0.9059, "DINO": 0.8858, "FID": 51.88},
    "Semantic": {"PSNR": 16.63, "SSIM": 0.7296, "LPIPS": 0.2564, "CLIP-I": 0.9050, "DINO": 0.8895, "FID": 50.83},
    "Feedback R1": {"PSNR": 16.68, "SSIM": 0.7310, "LPIPS": 0.2546, "CLIP-I": 0.9499, "DINO": 0.8837, "FID": 55.93},
}


VLM = {
    "Base": {"Score_view": 0.7746, "Score_cons": 0.9020},
    "Public LoRA": {"Score_view": 0.7234, "Score_cons": 0.8658},
    "Bright Dir.": {"Score_view": 0.7746, "Score_cons": 0.9682},
    "Semantic": {"Score_view": 0.7705, "Score_cons": 0.9709},
    "Feedback R1": {"Score_view": 0.7828, "Score_cons": 0.9676},
}


LOWER_IS_BETTER = {"LPIPS", "FID"}
COLORS = {
    "Base": "#6B7280",
    "Public LoRA": "#9CA3AF",
    "Bright Dir.": "#4C78A8",
    "Semantic": "#54A24B",
    "Feedback R1": "#F58518",
}


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 8,
            "legend.fontsize": 7,
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
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


def plot_spatialedit_relative() -> None:
    configs = list(SPATIALEDIT.keys())
    metrics = ["PSNR", "SSIM", "LPIPS", "CLIP-I", "DINO", "FID"]
    x = np.arange(len(metrics))
    width = 0.15

    fig, ax = plt.subplots(figsize=(7.2, 2.7))
    for i, cfg in enumerate(configs):
        vals = []
        for metric in metrics:
            base = SPATIALEDIT["Base"][metric]
            value = SPATIALEDIT[cfg][metric]
            if metric in LOWER_IS_BETTER:
                delta = (base - value) / base * 100.0
            else:
                delta = (value - base) / base * 100.0
            vals.append(delta)
        ax.bar(x + (i - 2) * width, vals, width=width, label=cfg, color=COLORS[cfg])

    ax.axhline(0, color="#111827", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Direction-normalized change vs. base (%)")
    ax.set_title("SpatialEdit-Bench external comparison")
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.28), frameon=False)
    savefig(fig, "spatialedit_external_relative_change")


def plot_vlm_scores() -> None:
    configs = list(VLM.keys())
    metrics = ["Score_view", "Score_cons"]
    x = np.arange(len(metrics))
    width = 0.15

    fig, ax = plt.subplots(figsize=(5.6, 2.6))
    for i, cfg in enumerate(configs):
        vals = [VLM[cfg][metric] for metric in metrics]
        ax.bar(x + (i - 2) * width, vals, width=width, label=cfg, color=COLORS[cfg])

    ax.set_xticks(x)
    ax.set_xticklabels(["Score_view", "Score_cons"])
    ax.set_ylim(0.68, 1.0)
    ax.set_ylabel("VLM score")
    ax.set_title("View and consistency judge scores")
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.28), frameon=False)
    savefig(fig, "spatialedit_vlm_score_bars")


def main() -> None:
    configure_matplotlib()
    plot_spatialedit_relative()
    plot_vlm_scores()


if __name__ == "__main__":
    main()
