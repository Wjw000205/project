import os
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd


def _compute_prototypes(data_tc: torch.Tensor, cluster_id_c: torch.Tensor, k: int) -> torch.Tensor:
    """
    data_tc: [T, C]
    cluster_id_c: [C]
    returns: [K, T]
    """
    T, C = data_tc.shape
    device = data_tc.device
    prot = torch.zeros(k, T, device=device, dtype=data_tc.dtype)
    cnt = torch.zeros(k, device=device, dtype=data_tc.dtype)
    idx = cluster_id_c.view(-1, 1).expand(C, T)  # [C, T]
    prot.scatter_add_(0, idx, data_tc.t())      # [K, T]
    cnt.scatter_add_(0, cluster_id_c, torch.ones(C, device=device, dtype=data_tc.dtype))
    prot = prot / cnt.clamp_min(1.0).view(-1, 1)
    return prot


def _high_freq_ratio(x: np.ndarray) -> float:
    n = x.shape[0]
    if n < 4:
        return 0.0
    x0 = x - x.mean()
    spec = np.fft.rfft(x0)
    power = (spec.real ** 2 + spec.imag ** 2)
    if power.size <= 1:
        return 0.0
    power = power[1:]
    total = power.sum()
    if total <= 0:
        return 0.0
    start = int(np.floor(power.size * 2 / 3))
    hf = power[start:].sum()
    return float(hf / total)


def _diff_energy(x: np.ndarray) -> float:
    if x.shape[0] < 2:
        return 0.0
    d1 = np.diff(x)
    return float(np.mean(d1 ** 2))


def _jump_rate(x: np.ndarray, thr: float) -> float:
    if x.shape[0] < 2:
        return 0.0
    d1 = np.diff(x)
    return float(np.mean(np.abs(d1) > thr))


def _trend_strength(x: np.ndarray) -> float:
    n = x.shape[0]
    if n < 2:
        return 0.0
    t = np.arange(n, dtype=np.float64)
    x0 = x.astype(np.float64)
    sx = x0.std()
    if sx <= 1e-12:
        return 0.0
    corr = np.corrcoef(t, x0)[0, 1]
    if not np.isfinite(corr):
        return 0.0
    return float(abs(corr))


def _variance(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.var(x))


def _downsample(x: np.ndarray, max_points: int) -> Tuple[np.ndarray, np.ndarray]:
    n = x.shape[0]
    if max_points <= 0 or n <= max_points:
        idx = np.arange(n)
        return x, idx
    idx = np.linspace(0, n - 1, max_points, dtype=np.int64)
    return x[idx], idx


def _scale_metric_values(raw: np.ndarray, mode: str) -> np.ndarray:
    mode_norm = str(mode or "per_metric_minmax").lower()
    if mode_norm in {"per_metric_minmax", "scaled", "minmax"}:
        minv = raw.min(axis=0)
        maxv = raw.max(axis=0)
        span = np.maximum(maxv - minv, 1e-12)
        return (raw - minv) / span
    if mode_norm in {"raw_0_1", "raw_prob", "probability"}:
        return np.clip(raw, 0.0, 1.0)
    if mode_norm in {"cluster_max", "per_cluster_max"}:
        denom = np.maximum(raw.max(axis=1, keepdims=True), 1e-12)
        return raw / denom
    raise ValueError(f"Unknown metric_scale_mode: {mode}")


def _save_combined_radar(
    out_dir: str,
    metric_names_use: List[str],
    scaled: np.ndarray,
    raw: np.ndarray,
    portrait_title_use: str,
    dpi: int,
) -> str:
    K = scaled.shape[0]
    fig = plt.figure(figsize=(6.2, 5.2))
    ax = fig.add_subplot(111, polar=True)

    base_angles = np.linspace(0, 2 * np.pi, len(metric_names_use), endpoint=False)
    angles = np.concatenate([base_angles, [base_angles[0]]])
    cmap = plt.get_cmap("tab10")

    for k in range(K):
        vals = np.concatenate([scaled[k], [scaled[k, 0]]])
        color = cmap(k % 10)
        ax.plot(angles, vals, linewidth=1.8, color=color, label=f"cluster {k}")
        ax.fill(angles, vals, color=color, alpha=0.08)

    ax.set_ylim(0, 1.0)
    ax.set_xticks(base_angles)
    ax.set_xticklabels(metric_names_use, fontsize=9)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], fontsize=8)
    ax.set_title(f"{portrait_title_use} (all clusters)")
    ax.legend(loc="upper right", bbox_to_anchor=(1.32, 1.12), frameon=True, fontsize=8)

    txt_lines = []
    for k in range(K):
        top_idx = int(np.argmax(raw[k]))
        txt_lines.append(f"C{k}: top={metric_names_use[top_idx]} ({raw[k, top_idx]:.3f})")
    fig.text(0.02, 0.02, " | ".join(txt_lines), fontsize=8)

    fig.tight_layout()
    combined_path = os.path.join(out_dir, "all_clusters_radar.png")
    fig.savefig(combined_path, dpi=dpi)
    plt.close(fig)
    return combined_path


def save_cluster_portraits(
    out_dir: str,
    data_tc: torch.Tensor,
    cluster_id_c: torch.Tensor,
    jump_thr: float,
    dpi: int = 140,
    max_points: int = 2000,
    metric_names: Optional[List[str]] = None,
    metric_values_km: Optional[torch.Tensor] = None,
    portrait_title: Optional[str] = None,
    metric_scale_mode: str = "per_metric_minmax",
) -> Dict[str, str]:
    """
    Saves per-cluster portrait images and a metrics table.
    Returns paths of saved artifacts.
    """
    os.makedirs(out_dir, exist_ok=True)
    K = int(cluster_id_c.max().item() + 1)
    sizes = torch.bincount(cluster_id_c, minlength=K).detach().cpu().tolist()

    prototypes_kt = _compute_prototypes(data_tc, cluster_id_c, K).detach().cpu()
    proto_np = prototypes_kt.numpy()
    proto_path = os.path.join(out_dir, "cluster_prototypes.npy")
    np.save(proto_path, proto_np)

    use_custom_metrics = (metric_names is not None) and (metric_values_km is not None)
    if use_custom_metrics:
        metric_names_use = list(metric_names)
        if isinstance(metric_values_km, torch.Tensor):
            metric_values = metric_values_km.detach().cpu().numpy()
        else:
            metric_values = np.asarray(metric_values_km)
        if metric_values.ndim != 2 or metric_values.shape[0] != K or metric_values.shape[1] != len(metric_names_use):
            raise ValueError("metric_values_km must have shape [K, M] matching metric_names.")
        portrait_title_use = portrait_title or "penalty portrait"
    else:
        metric_names_use = ["hf_ratio", "diff_energy", "jump_rate", "trend_strength", "variance"]
        metric_values = None
        portrait_title_use = portrait_title or "shape portrait (scaled)"

    rows: List[Dict[str, float]] = []
    for k in range(K):
        row = {
            "cluster": k,
            "size": int(sizes[k]),
        }
        if use_custom_metrics:
            for i, name in enumerate(metric_names_use):
                row[name] = float(metric_values[k, i])
        else:
            x = proto_np[k]
            row.update({
                "hf_ratio": _high_freq_ratio(x),
                "diff_energy": _diff_energy(x),
                "jump_rate": _jump_rate(x, jump_thr),
                "trend_strength": _trend_strength(x),
                "variance": _variance(x),
            })
        rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, "cluster_portrait_metrics.csv")
    df.to_csv(csv_path, index=False)

    raw = df[metric_names_use].to_numpy(dtype=np.float64)
    scaled = _scale_metric_values(raw, metric_scale_mode if use_custom_metrics else "per_metric_minmax")
    combined_path = _save_combined_radar(out_dir, metric_names_use, scaled, raw, portrait_title_use, dpi)

    for k in range(K):
        fig = plt.figure(figsize=(9, 3.6))
        gs = fig.add_gridspec(1, 2, width_ratios=[2.2, 1.6])
        ax0 = fig.add_subplot(gs[0, 0])
        ax1 = fig.add_subplot(gs[0, 1], polar=True)

        x = proto_np[k]
        xs, idx = _downsample(x, max_points)
        ax0.plot(idx, xs, color="#2d6cdf", linewidth=1.2)
        ax0.set_title(f"cluster {k} prototype (size={sizes[k]})")
        ax0.set_xlabel("t")
        ax0.set_ylabel("value")
        ax0.grid(True, alpha=0.25)

        vals = scaled[k]
        angles = np.linspace(0, 2 * np.pi, len(metric_names_use), endpoint=False)
        vals = np.concatenate([vals, [vals[0]]])
        angles = np.concatenate([angles, [angles[0]]])
        ax1.plot(angles, vals, color="#f08c00", linewidth=1.4)
        ax1.fill(angles, vals, color="#f08c00", alpha=0.18)
        ax1.set_ylim(0, 1.0)
        ax1.set_xticks(angles[:-1])
        ax1.set_xticklabels(metric_names_use, fontsize=8)
        ax1.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax1.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], fontsize=7)
        ax1.set_title(portrait_title_use)

        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"cluster_{k}_portrait.png"), dpi=dpi)
        plt.close(fig)

    return {
        "dir": out_dir,
        "metrics_csv": csv_path,
        "prototypes_npy": proto_path,
        "all_clusters_radar": combined_path,
    }
