from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "ett_val_calibration_diagnostics"
CLEAN = OUT / "clean_figures"


def _setup_style():
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "#222222",
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "grid.color": "#D9D9D9",
            "grid.linewidth": 0.8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def plot_transfer_pct():
    import matplotlib.pyplot as plt

    df = pd.read_csv(ROOT / "outputs" / "ettm1_to_ett_val_calibrated_transfer" / "best_by_val.csv")
    df["horizon"] = df["horizon"].astype(int)
    df = df.sort_values(["target", "horizon"])
    targets = ["ETTh1", "ETTh2", "ETTm2"]
    before_color = "#8A8A8A"
    after_color = "#2F6FA8"
    threshold_color = "#B04A3A"

    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.2), dpi=220, sharey=True)
    for ax, target in zip(axes, targets):
        sub = df[df["target"] == target].copy()
        x = np.arange(len(sub))
        old = sub["old_pct_vs_target_self"].to_numpy(float)
        new = sub["pct_vs_target_self"].to_numpy(float)
        for xi, o, n in zip(x, old, new):
            ax.plot([xi, xi], [o, n], color="#B8B8B8", lw=1.6, zorder=1)
        ax.scatter(x, old, s=46, color="#FFFFFF", edgecolor=before_color, lw=1.3, label="Before", zorder=3)
        ax.scatter(x, new, s=58, color=after_color, edgecolor="white", lw=0.8, label="Val-calibrated", zorder=4)
        ax.axhline(0, color="#222222", lw=1.1)
        ax.axhline(10, color=threshold_color, lw=1.0, ls=(0, (4, 3)))
        ax.set_xticks(x)
        ax.set_xticklabels([f"H{int(v)}" for v in sub["horizon"]], fontsize=9)
        ax.set_xlim(-0.35, len(sub) - 0.65)
        ax.set_title(target, fontsize=12, fontweight="bold", color="#222222")
        ax.grid(axis="y", alpha=0.75)
        for xi, n in zip(x, new):
            ax.text(
                xi,
                n - 2.4 if n < 0 else n + 1.8,
                f"{n:+.1f}%",
                ha="center",
                va="top" if n < 0 else "bottom",
                fontsize=8,
                color=after_color,
            )
        ax.set_ylim(-55, 16)
    axes[0].set_ylabel("Test MSE vs target self-train (%)", fontsize=10)
    axes[0].legend(frameon=False, loc="lower left", bbox_to_anchor=(0, -0.28), ncol=2)
    fig.suptitle("ETTm1 → ETT transfer after validation calibration", fontsize=14, fontweight="bold", y=1.03)
    fig.text(0.88, 0.93, "10% threshold", color="#B04A3A", fontsize=8)
    fig.suptitle("ETTm1 -> ETT transfer after validation calibration", fontsize=14, fontweight="bold", y=1.03)
    fig.tight_layout()
    fig.savefig(CLEAN / "transfer_pct_dumbbell.png", bbox_inches="tight")
    plt.close(fig)


def plot_decomposition_and_bias():
    import matplotlib.pyplot as plt

    df = pd.read_csv(OUT / "diagnostics.csv")
    df["case"] = df["target"] + " H" + df["horizon"].astype(str)
    df = df.sort_values(["target", "horizon"]).reset_index(drop=True)
    y = np.arange(len(df))

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.1), dpi=220)
    ax = axes[0]
    ax.scatter(df["target_self_mse"], y, s=64, marker="s", color="#4C78A8", label="Target self", zorder=3)
    ax.scatter(df["route_no_cal_test_mse"], y, s=54, color="#9B9B9B", label="Transfer before calibration", zorder=3)
    ax.scatter(df["cal_test_mse"], y, s=70, color="#3F8D63", label="After val calibration", zorder=4)
    for yi, before, after in zip(y, df["route_no_cal_test_mse"], df["cal_test_mse"]):
        ax.plot([before, after], [yi, yi], color="#AFC7B8", lw=2.0, zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels(df["case"], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Test MSE")
    ax.set_title("MSE decomposition", fontsize=12, fontweight="bold")
    ax.grid(axis="x", alpha=0.75)
    ax.legend(frameon=False, loc="lower right", fontsize=8)

    ax = axes[1]
    ax.barh(y, df["bias_abs_reduction_pct"], color="#3F8D63", height=0.56)
    ax.set_yticks(y)
    ax.set_yticklabels([])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Mean test-bias reduction (%)")
    ax.set_title("What calibration fixes", fontsize=12, fontweight="bold")
    ax.grid(axis="x", alpha=0.75)
    for yi, v in zip(y, df["bias_abs_reduction_pct"]):
        ax.text(v + 1.4, yi, f"{v:.1f}%", va="center", fontsize=8, color="#222222")
    fig.suptitle("Validation calibration mainly removes stable bias", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(CLEAN / "calibration_decomposition_clean.png", bbox_inches="tight")
    plt.close(fig)


def plot_residual_consistency():
    import matplotlib.pyplot as plt

    with (OUT / "diagnostics_full.json").open("r", encoding="utf-8") as f:
        rows = json.load(f)
    rows = sorted(rows, key=lambda r: (r["target"], int(r["horizon"])))

    fig, axes = plt.subplots(2, 3, figsize=(11.8, 6.6), dpi=220, sharex=False, sharey=False)
    for ax, row in zip(axes.ravel(), rows):
        x = np.asarray(row["residual_mean_val_flat"], dtype=float)
        y = np.asarray(row["residual_mean_test_flat"], dtype=float)
        ax.scatter(x, y, s=12, color="#3C6E71", alpha=0.55, edgecolor="none")
        lo = float(min(x.min(), y.min()))
        hi = float(max(x.max(), y.max()))
        pad = (hi - lo) * 0.08 + 1e-6
        lo -= pad
        hi += pad
        ax.plot([lo, hi], [lo, hi], color="#333333", lw=1.0, ls=(0, (4, 3)))
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_title(f"{row['target']} H{row['horizon']}  r={row['residual_mean_val_test_corr']:.2f}", fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.5)
    for ax in axes[:, 0]:
        ax.set_ylabel("Test residual bias")
    for ax in axes[-1, :]:
        ax.set_xlabel("Validation residual bias")
    fig.suptitle("Validation bias persists into test", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(CLEAN / "val_test_bias_consistency_small_multiples.png", bbox_inches="tight")
    plt.close(fig)


def plot_horizon_bias_reduction():
    import matplotlib.pyplot as plt

    with (OUT / "diagnostics_full.json").open("r", encoding="utf-8") as f:
        rows = json.load(f)
    rows = sorted(rows, key=lambda r: (r["target"], int(r["horizon"])))
    fig, axes = plt.subplots(2, 3, figsize=(12.4, 6.8), dpi=220)
    for ax, row in zip(axes.ravel(), rows):
        before = np.asarray(row["test_bias_by_horizon_before"], dtype=float)
        after = np.asarray(row["test_bias_by_horizon_after"], dtype=float)
        h = len(before)
        # Downsample visually for long horizons while preserving trend.
        if h > 240:
            bins = np.array_split(np.arange(h), 120)
            xs = np.array([b.mean() for b in bins])
            before_plot = np.array([before[b].mean() for b in bins])
            after_plot = np.array([after[b].mean() for b in bins])
        else:
            xs = np.arange(h)
            before_plot = before
            after_plot = after
        ax.fill_between(xs, before_plot, color="#D65F45", alpha=0.18)
        ax.plot(xs, before_plot, color="#D65F45", lw=1.2, label="Before")
        ax.fill_between(xs, after_plot, color="#3F8D63", alpha=0.18)
        ax.plot(xs, after_plot, color="#3F8D63", lw=1.2, label="After")
        ax.set_title(f"{row['target']} H{row['horizon']}", fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.5)
    axes[0, 0].legend(frameon=False, fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("Mean abs test bias")
    for ax in axes[-1, :]:
        ax.set_xlabel("Horizon step")
    fig.suptitle("Horizon-wise test bias before and after calibration", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(CLEAN / "horizon_bias_reduction_small_multiples.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    CLEAN.mkdir(parents=True, exist_ok=True)
    _setup_style()
    plot_transfer_pct()
    plot_decomposition_and_bias()
    plot_residual_consistency()
    plot_horizon_bias_reduction()
    print(CLEAN)


if __name__ == "__main__":
    main()
