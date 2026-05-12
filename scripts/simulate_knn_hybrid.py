import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors


REPO_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.compare_moe_on_off import evaluate_run, load_yaml, prepare_data_context


@dataclass
class RetrievalBank:
    key: int
    label: str
    features_nd: np.ndarray
    future_template_nh: np.ndarray
    nn: NearestNeighbors


def _parse_int_list(text: str) -> List[int]:
    values = []
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        values.append(int(item))
    if len(values) == 0:
        raise ValueError("Expected at least one integer value.")
    return values


def _parse_float_list(text: str) -> List[float]:
    values = []
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        values.append(float(item))
    if len(values) == 0:
        raise ValueError("Expected at least one float value.")
    return values


def _adaptive_pool_2d(x_nl: torch.Tensor, out_len: int) -> torch.Tensor:
    if out_len <= 0:
        return x_nl.new_zeros((x_nl.shape[0], 0))
    return F.adaptive_avg_pool1d(x_nl.unsqueeze(1), output_size=out_len).squeeze(1)


def build_shape_features(
    hist_nl: torch.Tensor,
    shape_bins: int,
    diff_bins: int,
    eps: float = 1.0e-6,
) -> torch.Tensor:
    mean_n1 = hist_nl.mean(dim=-1, keepdim=True)
    std_n1 = hist_nl.std(dim=-1, keepdim=True).clamp_min(eps)
    z_nl = (hist_nl - mean_n1) / std_n1

    feat_parts = [_adaptive_pool_2d(z_nl, shape_bins)]
    if diff_bins > 0 and hist_nl.shape[1] >= 2:
        dz_nl = z_nl[:, 1:] - z_nl[:, :-1]
        feat_parts.append(_adaptive_pool_2d(dz_nl, diff_bins))

    t_l = torch.linspace(-1.0, 1.0, steps=hist_nl.shape[1], device=hist_nl.device, dtype=hist_nl.dtype).view(1, -1)
    slope_n1 = ((z_nl * t_l).mean(dim=-1, keepdim=True) / t_l.pow(2).mean(dim=-1, keepdim=True).clamp_min(eps))
    last_n1 = z_nl[:, -1:].contiguous()
    range_n1 = (z_nl.max(dim=-1, keepdim=True).values - z_nl.min(dim=-1, keepdim=True).values)
    feat_parts.extend([slope_n1, last_n1, range_n1])
    return torch.cat(feat_parts, dim=-1)


def build_future_template(
    hist_nl: torch.Tensor,
    fut_nh: torch.Tensor,
    anchor_mode: str,
    eps: float = 1.0e-6,
) -> torch.Tensor:
    anchor_mode = str(anchor_mode).lower()
    hist_std_n1 = hist_nl.std(dim=-1, keepdim=True).clamp_min(eps)
    if anchor_mode == "mean":
        anchor_n1 = hist_nl.mean(dim=-1, keepdim=True)
    elif anchor_mode == "last":
        anchor_n1 = hist_nl[:, -1:].contiguous()
    else:
        raise ValueError(f"Unsupported anchor_mode={anchor_mode}")
    return (fut_nh - anchor_n1) / hist_std_n1


def reconstruct_from_template(
    hist_nl: torch.Tensor,
    template_nh: np.ndarray,
    anchor_mode: str,
    eps: float = 1.0e-6,
) -> np.ndarray:
    hist_std_n1 = hist_nl.std(dim=-1, keepdim=True).clamp_min(eps).cpu().numpy()
    if str(anchor_mode).lower() == "mean":
        anchor_n1 = hist_nl.mean(dim=-1, keepdim=True).cpu().numpy()
    else:
        anchor_n1 = hist_nl[:, -1:].contiguous().cpu().numpy()
    return anchor_n1 + template_nh * hist_std_n1


def knn_confidence(
    dist_bk: np.ndarray,
    tpl_bkh: np.ndarray,
    weight_bk: np.ndarray,
    feature_dim: int,
    distance_sharpness: float,
    confidence_floor: float,
) -> np.ndarray:
    mean_dist_b = (dist_bk * weight_bk).sum(axis=1).astype(np.float32)
    dist_scale = np.sqrt(max(int(feature_dim), 1))
    distance_conf_b = np.exp(-float(distance_sharpness) * mean_dist_b / max(dist_scale, 1.0e-6))
    tpl_mean_b1h = (tpl_bkh * weight_bk[..., None]).sum(axis=1, keepdims=True)
    disp_b = np.sqrt(((tpl_bkh - tpl_mean_b1h) ** 2 * weight_bk[..., None]).sum(axis=(1, 2)) / max(tpl_bkh.shape[2], 1))
    agreement_conf_b = 1.0 / (1.0 + disp_b.astype(np.float32))
    conf_b = np.clip(distance_conf_b * agreement_conf_b, 0.0, 1.0)
    floor = float(max(0.0, min(confidence_floor, 1.0)))
    return (floor + (1.0 - floor) * conf_b).astype(np.float32)


def resolve_run_dir(config_path: Path, run_dir_arg: str | None) -> Path:
    if run_dir_arg is not None:
        run_dir = Path(run_dir_arg)
        return run_dir if run_dir.is_absolute() else (REPO_ROOT / run_dir).resolve()

    cfg = load_yaml(config_path)
    base_out = Path(cfg["exp"]["out_dir"])
    if not base_out.is_absolute():
        base_out = (REPO_ROOT / base_out).resolve()

    compare_run_dir = base_out / "moe_compare" / "runs" / "moe_on"
    if compare_run_dir.exists():
        return compare_run_dir
    return base_out


def make_scope_label(scope: str, key: int) -> str:
    if scope == "same_channel":
        return f"channel_{key}"
    if scope == "same_cluster":
        return f"cluster_{key}"
    raise ValueError(f"Unsupported scope={scope}")


def collect_bank_series(
    xtr_ncl: torch.Tensor,
    ytr_nch: torch.Tensor,
    cluster_id_c: torch.Tensor,
    scope: str,
    key: int,
    train_stride: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    stride = max(1, int(train_stride))
    x_sub = xtr_ncl[::stride]
    y_sub = ytr_nch[::stride]

    if scope == "same_channel":
        return x_sub[:, key, :].contiguous(), y_sub[:, key, :].contiguous()
    if scope == "same_cluster":
        members = (cluster_id_c == key).nonzero(as_tuple=False).view(-1)
        x_bank = x_sub.index_select(1, members).permute(1, 0, 2).reshape(-1, x_sub.shape[-1]).contiguous()
        y_bank = y_sub.index_select(1, members).permute(1, 0, 2).reshape(-1, y_sub.shape[-1]).contiguous()
        return x_bank, y_bank
    raise ValueError(f"Unsupported scope={scope}")


def build_retrieval_banks(
    xtr_ncl: torch.Tensor,
    ytr_nch: torch.Tensor,
    cluster_id_c: torch.Tensor,
    scope: str,
    shape_bins: int,
    diff_bins: int,
    train_stride: int,
    weight_metric: str,
    anchor_mode: str,
) -> Dict[int, RetrievalBank]:
    if scope == "same_channel":
        keys: Iterable[int] = range(xtr_ncl.shape[1])
    elif scope == "same_cluster":
        keys = range(int(cluster_id_c.max().item()) + 1)
    else:
        raise ValueError(f"Unsupported scope={scope}")

    banks: Dict[int, RetrievalBank] = {}
    for key in keys:
        hist_nl, fut_nh = collect_bank_series(
            xtr_ncl=xtr_ncl,
            ytr_nch=ytr_nch,
            cluster_id_c=cluster_id_c,
            scope=scope,
            key=int(key),
            train_stride=train_stride,
        )
        features_nd = build_shape_features(hist_nl, shape_bins=shape_bins, diff_bins=diff_bins).cpu().numpy().astype(np.float32)
        future_template_nh = build_future_template(hist_nl, fut_nh, anchor_mode=anchor_mode).cpu().numpy().astype(np.float32)
        nn = NearestNeighbors(
            n_neighbors=1,
            metric=weight_metric,
            algorithm="auto",
            n_jobs=-1,
        )
        nn.fit(features_nd)
        banks[int(key)] = RetrievalBank(
            key=int(key),
            label=make_scope_label(scope, int(key)),
            features_nd=features_nd,
            future_template_nh=future_template_nh,
            nn=nn,
        )
    return banks


def resolve_bank_key(scope: str, channel_idx: int, cluster_id_c: torch.Tensor) -> int:
    if scope == "same_channel":
        return int(channel_idx)
    if scope == "same_cluster":
        return int(cluster_id_c[channel_idx].item())
    raise ValueError(f"Unsupported scope={scope}")


def compute_metrics(
    pred_nch: torch.Tensor,
    true_nch: torch.Tensor,
    mean_c: torch.Tensor,
    std_c: torch.Tensor,
) -> Tuple[float, float, float, np.ndarray, np.ndarray, np.ndarray]:
    se_nch = (pred_nch - true_nch).pow(2)
    mse_c = se_nch.mean(dim=(0, 2)).cpu().numpy()
    mae_norm_c = (pred_nch - true_nch).abs().mean(dim=(0, 2)).cpu().numpy()

    pred_raw = pred_nch * std_c.view(1, -1, 1) + mean_c.view(1, -1, 1)
    true_raw = true_nch * std_c.view(1, -1, 1) + mean_c.view(1, -1, 1)
    mae_raw_c = (pred_raw - true_raw).abs().mean(dim=(0, 2)).cpu().numpy()
    return float(mse_c.mean()), float(mae_norm_c.mean()), float(mae_raw_c.mean()), mse_c, mae_norm_c, mae_raw_c


def save_alpha_plot(results_df: pd.DataFrame, out_path: Path) -> None:
    plot_df = results_df[results_df["method"].isin(["hybrid", "hybrid_dynamic"])].copy()
    if plot_df.empty:
        return

    plt.figure(figsize=(7.5, 4.8))
    for method in ["hybrid", "hybrid_dynamic"]:
        method_df = plot_df[plot_df["method"] == method]
        for k in sorted(method_df["k"].unique()):
            sub = method_df[method_df["k"] == k].sort_values("alpha")
            label = f"{method} k={int(k)}"
            plt.plot(sub["alpha"], sub["avg_mse"], marker="o", linewidth=1.5, label=label)
    plt.xlabel("alpha")
    plt.ylabel("avg MSE (normalized)")
    plt.title("KNN Hybrid Sweep")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--run-dir", type=str, default=None)
    ap.add_argument("--out-dir", type=str, default=None)
    ap.add_argument("--scope", type=str, default="same_channel", choices=["same_channel", "same_cluster"])
    ap.add_argument("--shape-bins", type=int, default=24)
    ap.add_argument("--diff-bins", type=int, default=12)
    ap.add_argument("--train-stride", type=int, default=2)
    ap.add_argument("--query-batch-size", type=int, default=512)
    ap.add_argument("--k-grid", type=str, default="2,4,8,16")
    ap.add_argument("--alpha-grid", type=str, default="0.1,0.2,0.3,0.4,0.5")
    ap.add_argument("--distance-weight", type=str, default="inverse", choices=["uniform", "inverse"])
    ap.add_argument("--anchor-mode", type=str, default="last", choices=["last", "mean"])
    ap.add_argument("--eval-batch-size", type=int, default=256)
    ap.add_argument("--no-dynamic-confidence", action="store_false", dest="dynamic_confidence")
    ap.add_argument("--distance-sharpness", type=float, default=1.0)
    ap.add_argument("--confidence-floor", type=float, default=0.0)
    ap.set_defaults(dynamic_confidence=True)
    args = ap.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (REPO_ROOT / config_path).resolve()
    run_dir = resolve_run_dir(config_path, args.run_dir)

    out_dir = Path(args.out_dir) if args.out_dir is not None else (run_dir / "knn_hybrid_sim")
    if not out_dir.is_absolute():
        out_dir = (REPO_ROOT / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_yaml(config_path)
    device_name = cfg["exp"].get("device", "cpu")
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")

    print(f"Config: {config_path}")
    print(f"Run dir: {run_dir}")
    print(f"Out dir: {out_dir}")
    print(f"Device: {device}")

    context = prepare_data_context(cfg)
    xtr_norm, ytr_norm = context.xtr_norm.contiguous(), context.ytr_norm.contiguous()
    xte_norm, yte_norm = context.xte_norm.contiguous(), context.yte_norm.contiguous()
    cluster_id_c = context.cluster_id_c.contiguous()

    print("Evaluate base checkpoint...")
    base_eval = evaluate_run(
        context=context,
        run_cfg=cfg,
        run_dir=run_dir,
        device=device,
        eval_batch_size=int(args.eval_batch_size),
    )
    mean_c = context.mean_c.float()
    std_c = context.std_c.float()
    base_pred_norm = ((base_eval.yhat_raw.float() - mean_c.view(1, -1, 1)) / std_c.view(1, -1, 1)).contiguous()

    base_avg_mse, base_avg_mae_norm, base_avg_mae_raw, base_mse_c, base_mae_norm_c, base_mae_raw_c = compute_metrics(
        pred_nch=base_pred_norm,
        true_nch=yte_norm.float(),
        mean_c=mean_c,
        std_c=std_c,
    )
    print(
        f"Base avg_mse={base_avg_mse:.6f}, "
        f"avg_mae_norm={base_avg_mae_norm:.6f}, "
        f"avg_mae_raw={base_avg_mae_raw:.6f}"
    )

    print("Build retrieval banks...")
    banks = build_retrieval_banks(
        xtr_ncl=xtr_norm.float(),
        ytr_nch=ytr_norm.float(),
        cluster_id_c=cluster_id_c,
        scope=args.scope,
        shape_bins=int(args.shape_bins),
        diff_bins=int(args.diff_bins),
        train_stride=int(args.train_stride),
        weight_metric="euclidean",
        anchor_mode=args.anchor_mode,
    )

    k_grid = sorted(set(_parse_int_list(args.k_grid)))
    alpha_grid = sorted(set(_parse_float_list(args.alpha_grid)))
    max_k = max(k_grid)
    query_batch_size = max(1, int(args.query_batch_size))

    se_sum_map: Dict[Tuple[str, int, float], torch.Tensor] = {}
    ae_norm_sum_map: Dict[Tuple[str, int, float], torch.Tensor] = {}
    ae_raw_sum_map: Dict[Tuple[str, int, float], torch.Tensor] = {}
    confidence_sum_map: Dict[Tuple[str, int, float], float] = {}
    effective_alpha_sum_map: Dict[Tuple[str, int, float], float] = {}
    confidence_count_map: Dict[Tuple[str, int, float], int] = {}
    for k in k_grid:
        for alpha in alpha_grid:
            key = ("hybrid", int(k), float(alpha))
            se_sum_map[key] = torch.zeros(xte_norm.shape[1], dtype=torch.float64)
            ae_norm_sum_map[key] = torch.zeros(xte_norm.shape[1], dtype=torch.float64)
            ae_raw_sum_map[key] = torch.zeros(xte_norm.shape[1], dtype=torch.float64)
            confidence_sum_map[key] = 0.0
            effective_alpha_sum_map[key] = 0.0
            confidence_count_map[key] = 0
            if bool(args.dynamic_confidence):
                dyn_key = ("hybrid_dynamic", int(k), float(alpha))
                se_sum_map[dyn_key] = torch.zeros(xte_norm.shape[1], dtype=torch.float64)
                ae_norm_sum_map[dyn_key] = torch.zeros(xte_norm.shape[1], dtype=torch.float64)
                ae_raw_sum_map[dyn_key] = torch.zeros(xte_norm.shape[1], dtype=torch.float64)
                confidence_sum_map[dyn_key] = 0.0
                effective_alpha_sum_map[dyn_key] = 0.0
                confidence_count_map[dyn_key] = 0
        knn_key = ("knn_only", int(k), 1.0)
        se_sum_map[knn_key] = torch.zeros(xte_norm.shape[1], dtype=torch.float64)
        ae_norm_sum_map[knn_key] = torch.zeros(xte_norm.shape[1], dtype=torch.float64)
        ae_raw_sum_map[knn_key] = torch.zeros(xte_norm.shape[1], dtype=torch.float64)

    print("Run KNN hybrid sweep...")
    for c in range(xte_norm.shape[1]):
        bank_key = resolve_bank_key(args.scope, c, cluster_id_c)
        bank = banks[bank_key]
        query_hist = xte_norm[:, c, :].float().contiguous()
        query_feat = build_shape_features(query_hist, shape_bins=int(args.shape_bins), diff_bins=int(args.diff_bins)).cpu().numpy().astype(np.float32)
        base_pred_ch = base_pred_norm[:, c, :].float().cpu()
        true_ch = yte_norm[:, c, :].float().cpu()
        std_raw = float(std_c[c].item())

        bank.nn.set_params(n_neighbors=max_k)
        for start in range(0, query_feat.shape[0], query_batch_size):
            end = min(start + query_batch_size, query_feat.shape[0])
            dist_bd, idx_bd = bank.nn.kneighbors(query_feat[start:end], n_neighbors=max_k, return_distance=True)
            tpl_bkh = bank.future_template_nh[idx_bd]

            if args.distance_weight == "inverse":
                w_bk = 1.0 / np.maximum(dist_bd, 1.0e-6)
            else:
                w_bk = np.ones_like(dist_bd, dtype=np.float32)

            cum_w_bk = np.cumsum(w_bk, axis=1)
            cum_tpl_bkh = np.cumsum(tpl_bkh * w_bk[..., None], axis=1)

            hist_batch = query_hist[start:end]
            true_batch = true_ch[start:end]
            base_batch = base_pred_ch[start:end]

            for k in k_grid:
                avg_tpl_bh = cum_tpl_bkh[:, k - 1, :] / np.maximum(cum_w_bk[:, k - 1:k], 1.0e-6)
                knn_pred_bh = reconstruct_from_template(
                    hist_nl=hist_batch,
                    template_nh=avg_tpl_bh.astype(np.float32),
                    anchor_mode=args.anchor_mode,
                )
                knn_pred_bh = torch.from_numpy(knn_pred_bh).to(dtype=torch.float32)

                se_knn = (knn_pred_bh - true_batch).pow(2).sum()
                ae_knn_norm = (knn_pred_bh - true_batch).abs().sum()
                se_sum_map[("knn_only", int(k), 1.0)][c] += float(se_knn.item())
                ae_norm_sum_map[("knn_only", int(k), 1.0)][c] += float(ae_knn_norm.item())
                ae_raw_sum_map[("knn_only", int(k), 1.0)][c] += float(ae_knn_norm.item()) * std_raw

                for alpha in alpha_grid:
                    hybrid_bh = (1.0 - float(alpha)) * base_batch + float(alpha) * knn_pred_bh
                    se = (hybrid_bh - true_batch).pow(2).sum()
                    ae_norm = (hybrid_bh - true_batch).abs().sum()
                    key = ("hybrid", int(k), float(alpha))
                    se_sum_map[key][c] += float(se.item())
                    ae_norm_sum_map[key][c] += float(ae_norm.item())
                    ae_raw_sum_map[key][c] += float(ae_norm.item()) * std_raw
                    confidence_sum_map[key] += float(end - start)
                    effective_alpha_sum_map[key] += float(alpha) * float(end - start)
                    confidence_count_map[key] += int(end - start)
                    if bool(args.dynamic_confidence):
                        dist_k = dist_bd[:, :k]
                        tpl_k = tpl_bkh[:, :k, :]
                        weight_k = w_bk[:, :k] / np.maximum(cum_w_bk[:, k - 1:k], 1.0e-6)
                        conf_b = knn_confidence(
                            dist_bk=dist_k,
                            tpl_bkh=tpl_k,
                            weight_bk=weight_k,
                            feature_dim=query_feat.shape[1],
                            distance_sharpness=float(args.distance_sharpness),
                            confidence_floor=float(args.confidence_floor),
                        )
                        alpha_b1 = torch.from_numpy((float(alpha) * conf_b).reshape(-1, 1)).to(dtype=torch.float32)
                        hybrid_dyn_bh = (1.0 - alpha_b1) * base_batch + alpha_b1 * knn_pred_bh
                        se_dyn = (hybrid_dyn_bh - true_batch).pow(2).sum()
                        ae_dyn_norm = (hybrid_dyn_bh - true_batch).abs().sum()
                        dyn_key = ("hybrid_dynamic", int(k), float(alpha))
                        se_sum_map[dyn_key][c] += float(se_dyn.item())
                        ae_norm_sum_map[dyn_key][c] += float(ae_dyn_norm.item())
                        ae_raw_sum_map[dyn_key][c] += float(ae_dyn_norm.item()) * std_raw
                        confidence_sum_map[dyn_key] += float(conf_b.sum())
                        effective_alpha_sum_map[dyn_key] += float((float(alpha) * conf_b).sum())
                        confidence_count_map[dyn_key] += int(conf_b.shape[0])

        print(f"  channel={context.channel_names[c]} bank={bank.label}")

    rows = [{
        "method": "model_only",
        "scope": args.scope,
        "k": 0,
        "alpha": 0.0,
        "avg_mse": base_avg_mse,
        "avg_mae_norm": base_avg_mae_norm,
        "avg_mae_raw": base_avg_mae_raw,
        "delta_mse": 0.0,
        "delta_mse_pct": 0.0,
        "shape_bins": int(args.shape_bins),
        "diff_bins": int(args.diff_bins),
        "train_stride": int(args.train_stride),
        "distance_weight": args.distance_weight,
        "anchor_mode": args.anchor_mode,
        "mean_confidence": "",
        "mean_effective_alpha": "",
    }]

    denom_per_channel = float(xte_norm.shape[0] * xte_norm.shape[2])
    for key, se_sum_c in se_sum_map.items():
        method, k, alpha = key
        ae_norm_sum_c = ae_norm_sum_map[key]
        ae_raw_sum_c = ae_raw_sum_map[key]
        mse_c = (se_sum_c / denom_per_channel).numpy()
        mae_norm_c = (ae_norm_sum_c / denom_per_channel).numpy()
        mae_raw_c = (ae_raw_sum_c / denom_per_channel).numpy()
        avg_mse = float(mse_c.mean())
        avg_mae_norm = float(mae_norm_c.mean())
        avg_mae_raw = float(mae_raw_c.mean())
        rows.append({
            "method": method,
            "scope": args.scope,
            "k": int(k),
            "alpha": float(alpha),
            "avg_mse": avg_mse,
            "avg_mae_norm": avg_mae_norm,
            "avg_mae_raw": avg_mae_raw,
            "delta_mse": avg_mse - base_avg_mse,
            "delta_mse_pct": (avg_mse - base_avg_mse) / max(base_avg_mse, 1.0e-12) * 100.0,
            "shape_bins": int(args.shape_bins),
            "diff_bins": int(args.diff_bins),
            "train_stride": int(args.train_stride),
            "distance_weight": args.distance_weight,
            "anchor_mode": args.anchor_mode,
            "mean_confidence": (
                confidence_sum_map[key] / confidence_count_map[key]
                if key in confidence_count_map and confidence_count_map[key] > 0
                else ""
            ),
            "mean_effective_alpha": (
                effective_alpha_sum_map[key] / confidence_count_map[key]
                if key in confidence_count_map and confidence_count_map[key] > 0
                else ""
            ),
        })

    results_df = pd.DataFrame(rows).sort_values(["avg_mse", "method", "k", "alpha"]).reset_index(drop=True)
    results_path = out_dir / "results.csv"
    results_df.to_csv(results_path, index=False)

    best_row = results_df.iloc[0].to_dict()
    best_key = (str(best_row["method"]), int(best_row["k"]), float(best_row["alpha"]))

    channel_rows = []
    base_denom = float(xte_norm.shape[0] * xte_norm.shape[2])
    if best_row["method"] == "model_only":
        best_mse_c = base_mse_c
        best_mae_norm_c = base_mae_norm_c
        best_mae_raw_c = base_mae_raw_c
    else:
        best_mse_c = (se_sum_map[best_key] / base_denom).numpy()
        best_mae_norm_c = (ae_norm_sum_map[best_key] / base_denom).numpy()
        best_mae_raw_c = (ae_raw_sum_map[best_key] / base_denom).numpy()

    for c, channel in enumerate(context.channel_names):
        channel_rows.append({
            "channel": channel,
            "cluster_id": int(cluster_id_c[c].item()),
            "base_mse": float(base_mse_c[c]),
            "best_mse": float(best_mse_c[c]),
            "mse_gain": float(base_mse_c[c] - best_mse_c[c]),
            "base_mae_norm": float(base_mae_norm_c[c]),
            "best_mae_norm": float(best_mae_norm_c[c]),
            "mae_norm_gain": float(base_mae_norm_c[c] - best_mae_norm_c[c]),
            "base_mae_raw": float(base_mae_raw_c[c]),
            "best_mae_raw": float(best_mae_raw_c[c]),
            "mae_raw_gain": float(base_mae_raw_c[c] - best_mae_raw_c[c]),
        })
    channel_df = pd.DataFrame(channel_rows).sort_values("mse_gain", ascending=False)
    channel_path = out_dir / "best_channel_metrics.csv"
    channel_df.to_csv(channel_path, index=False)

    save_alpha_plot(results_df, out_dir / "mse_vs_alpha.png")

    summary = {
        "config_path": str(config_path),
        "run_dir": str(run_dir),
        "out_dir": str(out_dir),
        "scope": args.scope,
        "shape_bins": int(args.shape_bins),
        "diff_bins": int(args.diff_bins),
        "train_stride": int(args.train_stride),
        "distance_weight": args.distance_weight,
        "anchor_mode": args.anchor_mode,
        "dynamic_confidence": bool(args.dynamic_confidence),
        "distance_sharpness": float(args.distance_sharpness),
        "confidence_floor": float(args.confidence_floor),
        "base_avg_mse": base_avg_mse,
        "base_avg_mae_norm": base_avg_mae_norm,
        "base_avg_mae_raw": base_avg_mae_raw,
        "best": best_row,
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Saved results to: {results_path}")
    print(f"Saved best-channel metrics to: {channel_path}")
    print("Top results:")
    print(results_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
