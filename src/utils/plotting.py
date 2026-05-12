import os
import re
from typing import Dict, List, Tuple

import matplotlib
import torch


matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt


def _safe_name(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-\u4e00-\u9fa5]+", "_", s)


def save_cluster_metric_curves(
    out_dir: str,
    train_metric_hist: List[torch.Tensor],
    val_metric_hist: List[torch.Tensor],
    metric_name: str = "loss",
    dpi: int = 140,
):
    if len(val_metric_hist) == 0:
        return
    os.makedirs(out_dir, exist_ok=True)

    val = torch.stack([v.detach().cpu() for v in val_metric_hist], dim=0).numpy()
    train = None
    if len(train_metric_hist) == len(val_metric_hist):
        train = torch.stack([t.detach().cpu() for t in train_metric_hist], dim=0).numpy()

    epochs = list(range(1, val.shape[0] + 1))
    k_count = val.shape[1]
    for k in range(k_count):
        plt.figure()
        if train is not None:
            plt.plot(epochs, train[:, k], label="train")
        plt.plot(epochs, val[:, k], label="val")
        plt.xlabel("epoch")
        plt.ylabel(metric_name)
        plt.title(f"cluster {k}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"cluster_{k}.png"), dpi=dpi)
        plt.close()


@torch.no_grad()
def save_channel_plots(
    out_dir: str,
    channel_names: List[str],
    plot_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    best_sample: Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]],
    worst_sample: Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]],
    input_len: int,
    pred_len: int,
    dpi: int = 140,
):
    os.makedirs(out_dir, exist_ok=True)
    sorted_idx = sorted(plot_cache.keys())

    for channel_idx, name in enumerate(channel_names):
        ch_dir = os.path.join(out_dir, _safe_name(name))
        os.makedirs(ch_dir, exist_ok=True)

        for plot_num, window_idx in enumerate(sorted_idx):
            x_cL, y_cH, yhat_cH = plot_cache[window_idx]
            _plot_one(
                os.path.join(ch_dir, f"rand_{plot_num}_idx_{window_idx}.png"),
                x_cL[channel_idx],
                y_cH[channel_idx],
                yhat_cH[channel_idx],
                input_len,
                pred_len,
                title=f"{name} | random idx={window_idx}",
                dpi=dpi,
            )

        if channel_idx in best_sample:
            x, y, yhat, mse = best_sample[channel_idx]
            _plot_one(
                os.path.join(ch_dir, "best.png"),
                x,
                y,
                yhat,
                input_len,
                pred_len,
                title=f"{name} | best mse={mse:.6f}",
                dpi=dpi,
            )
        if channel_idx in worst_sample:
            x, y, yhat, mse = worst_sample[channel_idx]
            _plot_one(
                os.path.join(ch_dir, "worst.png"),
                x,
                y,
                yhat,
                input_len,
                pred_len,
                title=f"{name} | worst mse={mse:.6f}",
                dpi=dpi,
            )


def _plot_one(
    path: str,
    x_L: torch.Tensor,
    y_H: torch.Tensor,
    yhat_H: torch.Tensor,
    input_len: int,
    pred_len: int,
    title: str,
    dpi: int,
):
    x = x_L.detach().cpu().numpy()
    y = y_H.detach().cpu().numpy()
    yhat = yhat_H.detach().cpu().numpy()

    plt.figure()
    plt.title(title)
    t0 = list(range(input_len))
    t1 = list(range(input_len, input_len + pred_len))
    plt.plot(t0, x, label="history")
    plt.plot(t1, y, label="true")
    plt.plot(t1, yhat, label="pred")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    plt.close()
