from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUT_DIR = Path("paper_figures")


COLORS = {
    "input": "#EAF2FF",
    "cluster": "#D9EAD3",
    "base": "#D9EAF7",
    "gate": "#FFF2CC",
    "expert": "#FCE4D6",
    "fusion": "#EADCF8",
    "transfer": "#E6E0F8",
    "loss": "#F4F4F4",
    "edge": "#3A3A3A",
    "muted": "#777777",
}


def box(ax, xy, wh, text, fc, ec="#333333", lw=1.2, fs=9.0, weight="regular"):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fs,
        fontweight=weight,
        color="#222222",
        linespacing=1.18,
    )
    return patch


def arrow(
    ax,
    start,
    end,
    color="#333333",
    lw=1.2,
    style="-|>",
    rad=0.0,
    ms=11,
    linestyle="-",
):
    arr = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=ms,
        linewidth=lw,
        color=color,
        linestyle=linestyle,
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=3,
        shrinkB=3,
    )
    ax.add_patch(arr)
    return arr


def draw_main():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(12.0, 6.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.02,
        0.975,
        "GIMoE: Cluster-Aware Forecasting with Penalty-Guided Residual Experts",
        fontsize=14.5,
        fontweight="bold",
        ha="left",
        va="top",
    )
    ax.text(
        0.02,
        0.94,
        "Current paper scope: cluster-aware base prediction, penalty-keyed residual experts, and source-to-target transfer through cluster matching.",
        fontsize=9.2,
        color="#555555",
        ha="left",
        va="top",
    )

    # Left: data and clustering.
    box(
        ax,
        (0.035, 0.57),
        (0.16, 0.15),
        "Multivariate history\nX in R^(C x L)\ntrain/val/test split",
        COLORS["input"],
        fs=9.3,
        weight="bold",
    )
    box(
        ax,
        (0.035, 0.35),
        (0.16, 0.16),
        "Train-only channel\nclustering\ncorr / shape portrait\ncluster id c_i",
        COLORS["cluster"],
        fs=9.0,
        weight="bold",
    )
    arrow(ax, (0.115, 0.57), (0.115, 0.51))

    # Main model blocks.
    box(
        ax,
        (0.25, 0.57),
        (0.19, 0.16),
        "Cluster-aware\nbase predictor\none head per cluster\n$\\hat{Y}^{base}$",
        COLORS["base"],
        fs=9.0,
        weight="bold",
    )
    box(
        ax,
        (0.25, 0.35),
        (0.19, 0.16),
        "Cluster-level gate\nfeatures from input\n+ cluster portrait\nselect top-k penalties",
        COLORS["gate"],
        fs=9.0,
        weight="bold",
    )
    box(
        ax,
        (0.52, 0.35),
        (0.22, 0.19),
        "Penalty-guided\nresidual expert bank\nper-cluster experts keyed by\nlevel / trend / range / delta / ...\nDelta Y_p",
        COLORS["expert"],
        fs=8.7,
        weight="bold",
    )
    box(
        ax,
        (0.53, 0.59),
        (0.20, 0.15),
        "Residual fusion\nY_hat = Y_hat_base\n+ sum_p g_p alpha_p Delta Y_p",
        COLORS["fusion"],
        fs=9.0,
        weight="bold",
    )
    box(
        ax,
        (0.79, 0.58),
        (0.16, 0.16),
        "Forecast output\nY_hat in R^(C x H)\nMSE / MAE",
        "#E2F0D9",
        fs=9.1,
        weight="bold",
    )
    box(
        ax,
        (0.52, 0.16),
        (0.23, 0.13),
        "Training objective\nforecast loss + routed penalty loss\npenalties supervise residual experts",
        COLORS["loss"],
        fs=8.8,
    )

    # Main arrows.
    arrow(ax, (0.195, 0.645), (0.25, 0.645))
    arrow(ax, (0.195, 0.43), (0.25, 0.43))
    arrow(ax, (0.44, 0.645), (0.53, 0.665))
    arrow(ax, (0.44, 0.43), (0.52, 0.43))
    arrow(ax, (0.63, 0.54), (0.63, 0.59))
    arrow(ax, (0.73, 0.665), (0.79, 0.665))
    arrow(ax, (0.63, 0.35), (0.63, 0.29), linestyle="--", color="#666666", ms=9)
    arrow(ax, (0.87, 0.58), (0.745, 0.25), color="#777777", rad=-0.10, ms=9, linestyle="--")
    arrow(ax, (0.115, 0.35), (0.25, 0.60), color="#4A7C59", rad=-0.25)
    arrow(ax, (0.115, 0.35), (0.25, 0.40), color="#4A7C59", rad=0.0)

    # Small labels.
    ax.text(0.217, 0.67, "X, c_i", fontsize=8, color="#444444")
    ax.text(0.217, 0.455, "cluster features", fontsize=8, color="#444444")
    ax.text(0.465, 0.675, "base forecast", fontsize=8, color="#444444")
    ax.text(0.462, 0.455, "routing mask g_p", fontsize=8, color="#444444")
    ax.text(0.645, 0.56, "residuals", fontsize=8, color="#444444")

    # Transfer panel.
    transfer_panel = FancyBboxPatch(
        (0.035, 0.05),
        0.42,
        0.22,
        boxstyle="round,pad=0.02,rounding_size=0.025",
        linewidth=1.0,
        edgecolor="#6C5BA7",
        facecolor="#F6F3FF",
    )
    ax.add_patch(transfer_panel)
    ax.text(
        0.055,
        0.25,
        "Transfer path (secondary evidence)",
        fontsize=10.2,
        fontweight="bold",
        color="#4B3F82",
        ha="left",
        va="center",
    )
    box(
        ax,
        (0.06, 0.135),
        (0.11, 0.08),
        "Source model\ncluster memory\nprototypes",
        COLORS["transfer"],
        fs=7.8,
    )
    box(
        ax,
        (0.205, 0.135),
        (0.11, 0.08),
        "Target train\nchannels\n(no test labels)",
        COLORS["input"],
        fs=7.8,
    )
    box(
        ax,
        (0.345, 0.135),
        (0.085, 0.08),
        "Match by\ncorr / cycle\nroute",
        COLORS["cluster"],
        fs=7.8,
    )
    box(
        ax,
        (0.205, 0.065),
        (0.225, 0.05),
        "Reuse matched cluster heads and GIMoE residual experts\nfor zero-shot transfer or target fine-tuning",
        "#FFFFFF",
        fs=7.4,
    )
    arrow(ax, (0.17, 0.175), (0.205, 0.175), color="#6C5BA7", ms=9)
    arrow(ax, (0.315, 0.175), (0.345, 0.175), color="#6C5BA7", ms=9)
    arrow(ax, (0.387, 0.135), (0.318, 0.115), color="#6C5BA7", ms=9)
    arrow(ax, (0.14, 0.35), (0.12, 0.27), color="#6C5BA7", rad=0.15, ms=9)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ["svg", "pdf", "png"]:
        path = OUT_DIR / f"gimoe_main_architecture.{suffix}"
        if suffix == "png":
            fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        else:
            fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def draw_clean():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(11.2, 5.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.04, 0.94, "Main forecasting framework", fontsize=12, fontweight="bold", ha="left")
    ax.text(
        0.04,
        0.905,
        "GIMoE organizes base forecasting and structured residual correction around train-only channel clusters.",
        fontsize=8.6,
        color="#555555",
        ha="left",
    )

    # Main path.
    box(ax, (0.04, 0.61), (0.13, 0.13), "Input window\n$X_{t-L:t}$", COLORS["input"], fs=9, weight="bold")
    box(ax, (0.21, 0.61), (0.14, 0.13), "Channel\nclustering\n$c_i$", COLORS["cluster"], fs=9, weight="bold")
    box(ax, (0.40, 0.61), (0.17, 0.13), "Cluster-aware\nbase predictor\n$\\hat{Y}^{base}$", COLORS["base"], fs=9, weight="bold")
    box(ax, (0.76, 0.61), (0.16, 0.13), "Forecast\n$\\hat{Y}$", "#E2F0D9", fs=9.2, weight="bold")

    box(ax, (0.40, 0.36), (0.17, 0.13), "Cluster gate\n$q(g\\mid X,c)$", COLORS["gate"], fs=9, weight="bold")
    box(
        ax,
        (0.61, 0.36),
        (0.18, 0.13),
        "Penalty-keyed\nresidual experts\n$E_{c,p}(X,\\hat{Y}^{base})$",
        COLORS["expert"],
        fs=8.7,
        weight="bold",
    )
    box(
        ax,
        (0.61, 0.61),
        (0.12, 0.13),
        "Residual\nfusion\n$\\hat{Y}^{base}+\\sum_p g_p\\alpha_p\\Delta_p$",
        COLORS["fusion"],
        fs=8.0,
        weight="bold",
    )
    box(
        ax,
        (0.61, 0.16),
        (0.18, 0.10),
        "Training losses\nMSE/MAE + routed penalties",
        COLORS["loss"],
        fs=8.2,
    )

    arrow(ax, (0.17, 0.675), (0.21, 0.675))
    arrow(ax, (0.35, 0.675), (0.40, 0.675))
    arrow(ax, (0.57, 0.675), (0.61, 0.675))
    arrow(ax, (0.73, 0.675), (0.76, 0.675))
    arrow(ax, (0.48, 0.61), (0.48, 0.49), color="#666666")
    arrow(ax, (0.57, 0.425), (0.61, 0.425))
    arrow(ax, (0.70, 0.49), (0.68, 0.61))
    arrow(ax, (0.70, 0.36), (0.70, 0.26), color="#777777", linestyle="--", ms=9)
    arrow(ax, (0.68, 0.61), (0.69, 0.26), color="#777777", linestyle="--", rad=-0.08, ms=9)
    arrow(ax, (0.28, 0.61), (0.48, 0.49), color="#4A7C59", rad=-0.15)
    arrow(ax, (0.28, 0.61), (0.62, 0.41), color="#4A7C59", rad=-0.18)

    # Transfer inset.
    panel = FancyBboxPatch(
        (0.04, 0.07),
        0.48,
        0.21,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        linewidth=1.0,
        edgecolor="#6C5BA7",
        facecolor="#F6F3FF",
    )
    ax.add_patch(panel)
    ax.text(0.06, 0.245, "Transfer usage", fontsize=10, fontweight="bold", color="#4B3F82", ha="left")
    box(ax, (0.065, 0.12), (0.11, 0.075), "Source\nmemory", COLORS["transfer"], fs=7.9)
    box(ax, (0.215, 0.12), (0.11, 0.075), "Target train\nchannels", COLORS["input"], fs=7.9)
    box(ax, (0.365, 0.12), (0.11, 0.075), "Cluster\nmatching", COLORS["cluster"], fs=7.9)
    arrow(ax, (0.175, 0.158), (0.215, 0.158), color="#6C5BA7", ms=9)
    arrow(ax, (0.325, 0.158), (0.365, 0.158), color="#6C5BA7", ms=9)
    arrow(ax, (0.28, 0.61), (0.28, 0.28), color="#6C5BA7", linestyle="--", rad=0.0, ms=9)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ["svg", "pdf", "png"]:
        path = OUT_DIR / f"gimoe_main_architecture_clean.{suffix}"
        if suffix == "png":
            fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        else:
            fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    draw_main()
    draw_clean()
