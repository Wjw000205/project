import argparse
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.reader import read_csv_time_series
from src.data.windows import global_zscore
from src.utils.clustering import cluster_channels_by_corr
from src.utils.pearson import pearson_corr_matrix
from src.utils.yaml_io import load_yaml


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Plot ETTm1 raw channel curves, global Pearson correlation matrix, and global clustering summary."
    )
    ap.add_argument("--config", type=str, default="configs/ETTm1.yaml", help="YAML config path.")
    ap.add_argument("--start", type=int, default=0, help="Start index of the plotted slice.")
    ap.add_argument("--points", type=int, default=1000, help="Number of plotted points per channel.")
    ap.add_argument(
        "--output",
        type=str,
        default="outputs/ETTm1/ettm1_channels_corr_cluster.png",
        help="Output figure path.",
    )
    ap.add_argument("--dpi", type=int, default=160, help="Figure DPI.")
    return ap.parse_args()


def set_matplotlib_style() -> None:
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def slice_dates(csv_path: Path, start: int, points: int, date_col: int) -> pd.Series:
    df = pd.read_csv(csv_path, header=0)
    date_name = df.columns[int(date_col)]
    stop = min(len(df), int(start) + int(points))
    if start < 0 or start >= len(df):
        raise ValueError(f"start={start} is out of range for dataset length {len(df)}")
    return pd.to_datetime(df.iloc[start:stop][date_name], errors="coerce")


def normalize_like_train(data_tc: torch.Tensor, cfg: dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    norm_cfg = cfg.get("normalize", {})
    if not bool(norm_cfg.get("global_zscore", True)):
        mean = data_tc.mean(dim=0)
        std = data_tc.std(dim=0).clamp_min(1e-6)
        return data_tc, mean, std

    train_ratio = float(cfg["data"]["train_ratio"])
    t_train = int(data_tc.shape[0] * train_ratio)
    if bool(norm_cfg.get("train_only", False)):
        train_seg = data_tc[:t_train]
        mean_c = train_seg.mean(dim=0, keepdim=True)
        std_c = train_seg.std(dim=0, keepdim=True).clamp_min(1e-6)
        normed = (data_tc - mean_c) / std_c
        return normed, mean_c.squeeze(0), std_c.squeeze(0)

    return global_zscore(data_tc)


def cluster_summary_lines(clusters: dict, channel_names: Sequence[str]) -> List[str]:
    lines: List[str] = []
    for cid in sorted(clusters.keys()):
        names = [channel_names[idx] for idx in clusters[cid]]
        lines.append(f"Cluster {cid}: {', '.join(names)}")
    return lines


def main() -> None:
    args = parse_args()
    set_matplotlib_style()

    cfg = load_yaml(args.config)
    data_cfg = cfg["data"]
    cluster_cfg = cfg["cluster"]
    csv_path = ROOT / data_cfg["csv_path"]
    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data_tc, channel_names = read_csv_time_series(
        csv_path=str(csv_path),
        date_col=int(data_cfg.get("date_col", 0)),
        dtype=torch.float32,
    )
    if data_tc.shape[1] != 7:
        raise ValueError(f"Expected 7 channels for ETTm1, got {data_tc.shape[1]}")

    data_norm_tc, _, _ = normalize_like_train(data_tc, cfg)

    start = int(args.start)
    stop = min(int(data_norm_tc.shape[0]), start + int(args.points))
    if start < 0 or start >= int(data_norm_tc.shape[0]):
        raise ValueError(f"start={start} is out of range for dataset length {data_norm_tc.shape[0]}")
    if stop - start < 2:
        raise ValueError("Selected slice must contain at least 2 points")

    plot_slice = data_tc[start:stop]
    corr_cc = pearson_corr_matrix(data_norm_tc)
    cluster_id_c, clusters = cluster_channels_by_corr(
        corr_cc=corr_cc,
        data_tc=data_norm_tc,
        n_clusters=cluster_cfg.get("n_clusters", None),
        distance_threshold=cluster_cfg.get("distance_threshold", None),
        linkage=cluster_cfg.get("linkage", "average"),
        method=cluster_cfg.get("method", "agglomerative"),
        kmeans_n_init=int(cluster_cfg.get("kmeans_n_init", 10)),
        kmeans_max_iter=int(cluster_cfg.get("kmeans_max_iter", 300)),
        spectral_affinity=cluster_cfg.get("spectral_affinity", "corr"),
        rbf_gamma=float(cluster_cfg.get("rbf_gamma", 1.0)),
        dbscan_eps=cluster_cfg.get("dbscan_eps", None),
        dbscan_min_samples=int(cluster_cfg.get("dbscan_min_samples", 5)),
        random_state=cluster_cfg.get("random_state", 0),
        min_cluster_size=int(cluster_cfg.get("min_cluster_size", 2)),
        merge_small_clusters=bool(cluster_cfg.get("merge_small_clusters", True)),
        no_merge_if_channels_lt=int(cluster_cfg.get("no_merge_if_channels_lt", 10)),
    )

    dates = slice_dates(csv_path, start, stop - start, int(data_cfg.get("date_col", 0)))
    x = np.arange(start, stop)
    y_plot = plot_slice.detach().cpu().numpy()
    corr_np = corr_cc.detach().cpu().numpy()
    cluster_lines = cluster_summary_lines(clusters, channel_names)
    assign_text = " | ".join(
        f"{name}->C{int(cid)}" for name, cid in zip(channel_names, cluster_id_c.detach().cpu().tolist())
    )

    fig = plt.figure(figsize=(18, 9), constrained_layout=True)
    gs = fig.add_gridspec(
        nrows=2,
        ncols=2,
        height_ratios=[4.2, 1.35],
        width_ratios=[1.9, 1.0],
        hspace=0.14,
        wspace=0.18,
    )

    ax_left = fig.add_subplot(gs[0, 0])
    ax_right = fig.add_subplot(gs[0, 1])
    ax_bottom = fig.add_subplot(gs[1, :])

    cmap = plt.get_cmap("tab10")
    for idx, name in enumerate(channel_names):
        ax_left.plot(x, y_plot[:, idx], linewidth=1.2, alpha=0.95, color=cmap(idx), label=name)

    tick_count = min(8, len(x))
    tick_positions = np.linspace(0, len(x) - 1, num=tick_count, dtype=int)
    tick_labels = []
    for pos in tick_positions:
        ts = dates.iloc[pos]
        tick_labels.append(ts.strftime("%m-%d %H:%M") if pd.notna(ts) else str(x[pos]))

    ax_left.set_xticks(x[tick_positions])
    ax_left.set_xticklabels(tick_labels, rotation=20, ha="right")
    ax_left.set_xlabel("Time")
    ax_left.set_ylabel("Raw value")
    ax_left.set_title(
        f"ETTm1 raw channels ({stop - start} plotted points, global clustering)"
    )
    ax_left.grid(alpha=0.25, linestyle="--", linewidth=0.6)
    ax_left.legend(ncol=4, fontsize=9, frameon=True, loc="upper right")

    im = ax_right.imshow(corr_np, cmap="coolwarm", vmin=-1.0, vmax=1.0, aspect="equal")
    ax_right.set_xticks(np.arange(len(channel_names)))
    ax_right.set_yticks(np.arange(len(channel_names)))
    ax_right.set_xticklabels(channel_names, rotation=35, ha="right")
    ax_right.set_yticklabels(channel_names)
    ax_right.set_title("Global Pearson Correlation Matrix")
    for i in range(corr_np.shape[0]):
        for j in range(corr_np.shape[1]):
            color = "white" if abs(corr_np[i, j]) > 0.55 else "black"
            ax_right.text(j, i, f"{corr_np[i, j]:.2f}", ha="center", va="center", fontsize=8, color=color)
    fig.colorbar(im, ax=ax_right, fraction=0.046, pad=0.04)

    ax_bottom.axis("off")
    method = cluster_cfg.get("method", "agglomerative")
    threshold = cluster_cfg.get("distance_threshold", None)
    title = f"Global Clustering Result (method={method}"
    if threshold is not None:
        title += f", distance_threshold={threshold}"
    title += ")"
    body = "\n".join(cluster_lines + ["", assign_text])
    ax_bottom.text(
        0.01,
        0.92,
        title,
        fontsize=13,
        fontweight="bold",
        va="top",
        ha="left",
        transform=ax_bottom.transAxes,
    )
    ax_bottom.text(
        0.01,
        0.72,
        body,
        fontsize=11,
        va="top",
        ha="left",
        transform=ax_bottom.transAxes,
        family="monospace",
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "#f6f6f6", "edgecolor": "#cccccc"},
    )

    fig.savefig(output_path, dpi=int(args.dpi), bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figure to: {output_path}")
    print(f"Plot slice: start={start}, points={stop - start}, values=raw")
    print("Global cluster summary:")
    for line in cluster_lines:
        print(f"  {line}")
    print(f"Assignments: {assign_text}")


if __name__ == "__main__":
    main()

