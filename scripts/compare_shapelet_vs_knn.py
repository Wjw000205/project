import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import Ridge
from sklearn.neighbors import NearestNeighbors


REPO_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.compare_moe_on_off import evaluate_run, load_yaml, prepare_data_context


@dataclass
class KNNBank:
    key: int
    label: str
    features_nd: np.ndarray
    future_template_nh: np.ndarray
    nn: NearestNeighbors


@dataclass
class ShapeletModel:
    key: int
    label: str
    lengths: List[int]
    centers_by_length: List[np.ndarray]
    feat_mean_f: np.ndarray
    feat_std_f: np.ndarray
    target_mean_h: np.ndarray
    target_std_h: np.ndarray
    target_clip_lo: float
    target_clip_hi: float
    ridge: Ridge
    train_samples: int
    num_shapelets: int


def _parse_int_list(text: str) -> List[int]:
    values = []
    for item in str(text).split(","):
        item = item.strip()
        if item:
            values.append(int(item))
    if len(values) == 0:
        raise ValueError("Expected non-empty integer list.")
    return values


def _parse_float_list(text: str) -> List[float]:
    values = []
    for item in str(text).split(","):
        item = item.strip()
        if item:
            values.append(float(item))
    if len(values) == 0:
        raise ValueError("Expected non-empty float list.")
    return values


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


def _adaptive_pool_2d(x_nl: torch.Tensor, out_len: int) -> torch.Tensor:
    if out_len <= 0:
        return x_nl.new_zeros((x_nl.shape[0], 0))
    return F.adaptive_avg_pool1d(x_nl.unsqueeze(1), output_size=out_len).squeeze(1)


def zscore_2d(x_nl: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    mean_n1 = x_nl.mean(dim=-1, keepdim=True)
    std_n1 = x_nl.std(dim=-1, keepdim=True).clamp_min(eps)
    return (x_nl - mean_n1) / std_n1


def build_global_shape_features(
    hist_nl: torch.Tensor,
    shape_bins: int,
    diff_bins: int,
    eps: float = 1.0e-6,
) -> torch.Tensor:
    z_nl = zscore_2d(hist_nl, eps=eps)
    feat_parts = [_adaptive_pool_2d(z_nl, shape_bins)]
    if diff_bins > 0 and hist_nl.shape[1] >= 2:
        dz_nl = z_nl[:, 1:] - z_nl[:, :-1]
        feat_parts.append(_adaptive_pool_2d(dz_nl, diff_bins))
    t_l = torch.linspace(-1.0, 1.0, steps=hist_nl.shape[1], device=hist_nl.device, dtype=hist_nl.dtype).view(1, -1)
    slope_n1 = (z_nl * t_l).mean(dim=-1, keepdim=True) / t_l.pow(2).mean(dim=-1, keepdim=True).clamp_min(eps)
    last_n1 = z_nl[:, -1:].contiguous()
    range_n1 = z_nl.max(dim=-1, keepdim=True).values - z_nl.min(dim=-1, keepdim=True).values
    feat_parts.extend([slope_n1, last_n1, range_n1])
    return torch.cat(feat_parts, dim=-1)


def build_future_template(
    hist_nl: torch.Tensor,
    fut_nh: torch.Tensor,
    anchor_mode: str,
    eps: float = 1.0e-6,
) -> torch.Tensor:
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
    if anchor_mode == "mean":
        anchor_n1 = hist_nl.mean(dim=-1, keepdim=True).cpu().numpy()
    elif anchor_mode == "last":
        anchor_n1 = hist_nl[:, -1:].contiguous().cpu().numpy()
    else:
        raise ValueError(f"Unsupported anchor_mode={anchor_mode}")
    return anchor_n1 + template_nh * hist_std_n1


def make_scope_label(scope: str, key: int) -> str:
    if scope == "same_channel":
        return f"channel_{key}"
    if scope == "same_cluster":
        return f"cluster_{key}"
    raise ValueError(f"Unsupported scope={scope}")


def resolve_bank_key(scope: str, channel_idx: int, cluster_id_c: torch.Tensor) -> int:
    if scope == "same_channel":
        return int(channel_idx)
    if scope == "same_cluster":
        return int(cluster_id_c[channel_idx].item())
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


def compute_metrics(
    pred_nch: torch.Tensor,
    true_nch: torch.Tensor,
    mean_c: torch.Tensor,
    std_c: torch.Tensor,
) -> Tuple[float, float, float]:
    mse = float((pred_nch - true_nch).pow(2).mean().item())
    mae_norm = float((pred_nch - true_nch).abs().mean().item())
    pred_raw = pred_nch * std_c.view(1, -1, 1) + mean_c.view(1, -1, 1)
    true_raw = true_nch * std_c.view(1, -1, 1) + mean_c.view(1, -1, 1)
    mae_raw = float((pred_raw - true_raw).abs().mean().item())
    return mse, mae_norm, mae_raw


def build_knn_banks(
    xtr_ncl: torch.Tensor,
    ytr_nch: torch.Tensor,
    cluster_id_c: torch.Tensor,
    scope: str,
    shape_bins: int,
    diff_bins: int,
    train_stride: int,
    anchor_mode: str,
) -> Dict[int, KNNBank]:
    if scope == "same_channel":
        keys: Iterable[int] = range(xtr_ncl.shape[1])
    else:
        keys = range(int(cluster_id_c.max().item()) + 1)

    banks: Dict[int, KNNBank] = {}
    for key in keys:
        hist_nl, fut_nh = collect_bank_series(xtr_ncl, ytr_nch, cluster_id_c, scope, int(key), train_stride)
        feat_nd = build_global_shape_features(hist_nl, shape_bins, diff_bins).cpu().numpy().astype(np.float32)
        tpl_nh = build_future_template(hist_nl, fut_nh, anchor_mode).cpu().numpy().astype(np.float32)
        nn = NearestNeighbors(metric="euclidean", algorithm="auto", n_jobs=-1)
        nn.fit(feat_nd)
        banks[int(key)] = KNNBank(
            key=int(key),
            label=make_scope_label(scope, int(key)),
            features_nd=feat_nd,
            future_template_nh=tpl_nh,
            nn=nn,
        )
    return banks


def run_knn_hybrid(
    xte_ncl: torch.Tensor,
    base_pred_nch: torch.Tensor,
    cluster_id_c: torch.Tensor,
    banks: Dict[int, KNNBank],
    scope: str,
    shape_bins: int,
    diff_bins: int,
    anchor_mode: str,
    k: int,
    alpha: float,
    query_batch_size: int,
) -> torch.Tensor:
    out = base_pred_nch.clone()
    for c in range(xte_ncl.shape[1]):
        bank_key = resolve_bank_key(scope, c, cluster_id_c)
        bank = banks[bank_key]
        query_hist = xte_ncl[:, c, :].float().contiguous()
        query_feat = build_global_shape_features(query_hist, shape_bins, diff_bins).cpu().numpy().astype(np.float32)
        pred_ch = base_pred_nch[:, c, :].float().cpu()
        bank.nn.set_params(n_neighbors=min(int(k), bank.features_nd.shape[0]))
        blended = torch.empty_like(pred_ch)
        for start in range(0, query_feat.shape[0], query_batch_size):
            end = min(start + query_batch_size, query_feat.shape[0])
            k_eff = min(int(k), bank.features_nd.shape[0])
            dist_bd, idx_bd = bank.nn.kneighbors(query_feat[start:end], n_neighbors=k_eff, return_distance=True)
            tpl_bkh = bank.future_template_nh[idx_bd]
            w_bk = 1.0 / np.maximum(dist_bd, 1.0e-6)
            tpl_bh = (tpl_bkh * w_bk[..., None]).sum(axis=1) / np.maximum(w_bk.sum(axis=1, keepdims=True), 1.0e-6)
            knn_pred = reconstruct_from_template(query_hist[start:end], tpl_bh.astype(np.float32), anchor_mode)
            knn_pred = torch.from_numpy(knn_pred).to(dtype=pred_ch.dtype)
            blended[start:end] = (1.0 - float(alpha)) * pred_ch[start:end] + float(alpha) * knn_pred
        out[:, c, :] = blended
    return out


def subsample_bank(
    hist_nl: torch.Tensor,
    fut_nh: torch.Tensor,
    max_samples: int,
    rng: np.random.Generator,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if hist_nl.shape[0] <= max_samples:
        return hist_nl, fut_nh
    idx = rng.choice(hist_nl.shape[0], size=int(max_samples), replace=False)
    idx_t = torch.tensor(idx, dtype=torch.long)
    return hist_nl.index_select(0, idx_t), fut_nh.index_select(0, idx_t)


def build_shapelet_centers(
    hist_nl: torch.Tensor,
    lengths: List[int],
    shapelets_per_length: List[int],
    candidate_limit: int,
    random_state: int,
) -> List[np.ndarray]:
    centers_by_length: List[np.ndarray] = []
    hist_len = int(hist_nl.shape[1])
    for length, num_shapelets in zip(lengths, shapelets_per_length):
        if length <= 1 or length > hist_len:
            raise ValueError(f"Invalid shapelet length={length} for history length={hist_len}")
        step = max(1, length // 2)
        candidates = []
        for start in range(0, hist_len - length + 1, step):
            candidates.append(hist_nl[:, start:start + length])
        cand_nl = torch.cat(candidates, dim=0)
        if cand_nl.shape[0] > candidate_limit:
            rng = np.random.default_rng(random_state + length)
            idx = rng.choice(cand_nl.shape[0], size=int(candidate_limit), replace=False)
            cand_nl = cand_nl.index_select(0, torch.tensor(idx, dtype=torch.long))
        cand_nl = zscore_2d(cand_nl).cpu().numpy().astype(np.float32)
        if cand_nl.shape[0] <= num_shapelets:
            centers = cand_nl[:num_shapelets]
            if centers.shape[0] < num_shapelets:
                pad_idx = np.random.default_rng(random_state + 7 * length).choice(centers.shape[0], size=num_shapelets - centers.shape[0], replace=True)
                centers = np.concatenate([centers, centers[pad_idx]], axis=0)
        else:
            kmeans = MiniBatchKMeans(
                n_clusters=int(num_shapelets),
                random_state=int(random_state + length),
                batch_size=min(4096, cand_nl.shape[0]),
                n_init=3,
                max_iter=100,
            )
            kmeans.fit(cand_nl)
            centers = kmeans.cluster_centers_.astype(np.float32)
        centers_by_length.append(centers)
    return centers_by_length


def encode_shapelet_features(
    hist_nl: torch.Tensor,
    lengths: List[int],
    centers_by_length: List[np.ndarray],
    batch_size: int,
) -> np.ndarray:
    outputs = []
    for start in range(0, hist_nl.shape[0], batch_size):
        batch = hist_nl[start:start + batch_size].float().contiguous()
        feat_parts = []
        for length, centers in zip(lengths, centers_by_length):
            patches = batch.unfold(1, int(length), 1)
            bsz, num_patches, _ = patches.shape
            patches = patches.reshape(-1, int(length))
            patches = zscore_2d(patches).reshape(bsz, num_patches, int(length))
            centers_t = torch.from_numpy(centers).to(device=batch.device, dtype=batch.dtype)
            p2 = patches.pow(2).sum(dim=-1, keepdim=True)
            c2 = centers_t.pow(2).sum(dim=-1).view(1, 1, -1)
            cross = torch.matmul(patches, centers_t.t())
            min_dist = (p2 + c2 - 2.0 * cross).clamp_min(0.0).min(dim=1).values
            sim = 1.0 / (1.0 + (min_dist / float(length)))
            feat_parts.append(sim)
        outputs.append(torch.cat(feat_parts, dim=-1).cpu())
    return torch.cat(outputs, dim=0).numpy().astype(np.float32)


def build_shapelet_models(
    xtr_ncl: torch.Tensor,
    ytr_nch: torch.Tensor,
    cluster_id_c: torch.Tensor,
    scope: str,
    train_stride: int,
    anchor_mode: str,
    lengths: List[int],
    shapelets_per_length: List[int],
    train_sample_size: int,
    candidate_limit: int,
    ridge_alpha: float,
    encode_batch_size: int,
    random_state: int,
) -> Dict[int, ShapeletModel]:
    if scope == "same_channel":
        keys: Iterable[int] = range(xtr_ncl.shape[1])
    else:
        keys = range(int(cluster_id_c.max().item()) + 1)

    models: Dict[int, ShapeletModel] = {}
    rng = np.random.default_rng(random_state)
    for key in keys:
        hist_nl, fut_nh = collect_bank_series(xtr_ncl, ytr_nch, cluster_id_c, scope, int(key), train_stride)
        hist_fit, fut_fit = subsample_bank(hist_nl, fut_nh, train_sample_size, rng)
        future_template = build_future_template(hist_fit, fut_fit, anchor_mode).cpu().numpy().astype(np.float32)
        centers_by_length = build_shapelet_centers(
            hist_fit,
            lengths=lengths,
            shapelets_per_length=shapelets_per_length,
            candidate_limit=candidate_limit,
            random_state=random_state + int(key) * 1000,
        )
        feat_nf = encode_shapelet_features(hist_fit, lengths, centers_by_length, batch_size=encode_batch_size)
        feat_mean = feat_nf.mean(axis=0, keepdims=True).astype(np.float32)
        feat_std = feat_nf.std(axis=0, keepdims=True).astype(np.float32)
        feat_std = np.clip(feat_std, 1.0e-6, None)
        feat_nf = (feat_nf - feat_mean) / feat_std
        target_mean = future_template.mean(axis=0, keepdims=True).astype(np.float32)
        target_std = future_template.std(axis=0, keepdims=True).astype(np.float32)
        target_std = np.clip(target_std, 1.0e-6, None)
        target_nf = (future_template - target_mean) / target_std
        clip_lo = float(np.quantile(future_template, 0.01))
        clip_hi = float(np.quantile(future_template, 0.99))
        ridge = Ridge(alpha=float(ridge_alpha), fit_intercept=True, random_state=random_state)
        ridge.fit(feat_nf, target_nf)
        models[int(key)] = ShapeletModel(
            key=int(key),
            label=make_scope_label(scope, int(key)),
            lengths=list(lengths),
            centers_by_length=centers_by_length,
            feat_mean_f=feat_mean.squeeze(0),
            feat_std_f=feat_std.squeeze(0),
            target_mean_h=target_mean.squeeze(0),
            target_std_h=target_std.squeeze(0),
            target_clip_lo=clip_lo,
            target_clip_hi=clip_hi,
            ridge=ridge,
            train_samples=int(hist_fit.shape[0]),
            num_shapelets=int(sum(shapelets_per_length)),
        )
    return models


def run_shapelet_hybrid(
    xte_ncl: torch.Tensor,
    base_pred_nch: torch.Tensor,
    cluster_id_c: torch.Tensor,
    models: Dict[int, ShapeletModel],
    scope: str,
    anchor_mode: str,
    alpha: float,
    encode_batch_size: int,
) -> torch.Tensor:
    out = base_pred_nch.clone()
    for c in range(xte_ncl.shape[1]):
        model_key = resolve_bank_key(scope, c, cluster_id_c)
        model = models[model_key]
        query_hist = xte_ncl[:, c, :].float().contiguous()
        feat_nf = encode_shapelet_features(query_hist, model.lengths, model.centers_by_length, batch_size=encode_batch_size)
        feat_nf = (feat_nf - model.feat_mean_f[None, :]) / model.feat_std_f[None, :]
        pred_tpl_std = model.ridge.predict(feat_nf).astype(np.float32)
        pred_tpl_std = np.clip(pred_tpl_std, -5.0, 5.0)
        pred_tpl = pred_tpl_std * model.target_std_h[None, :] + model.target_mean_h[None, :]
        pred_tpl = np.clip(pred_tpl, model.target_clip_lo, model.target_clip_hi)
        shapelet_pred = reconstruct_from_template(query_hist, pred_tpl, anchor_mode)
        shapelet_pred = torch.from_numpy(shapelet_pred).to(dtype=out.dtype)
        out[:, c, :] = (1.0 - float(alpha)) * out[:, c, :] + float(alpha) * shapelet_pred
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--run-dir", type=str, default=None)
    ap.add_argument("--out-dir", type=str, default=None)
    ap.add_argument("--scope", type=str, default="same_cluster", choices=["same_channel", "same_cluster"])
    ap.add_argument("--train-stride", type=int, default=4)
    ap.add_argument("--k-grid", type=str, default="8,16")
    ap.add_argument("--alpha-grid", type=str, default="0.05,0.08,0.1,0.12,0.15,0.2")
    ap.add_argument("--shape-bins", type=int, default=24)
    ap.add_argument("--diff-bins", type=int, default=12)
    ap.add_argument("--anchor-mode", type=str, default="last", choices=["last", "mean"])
    ap.add_argument("--query-batch-size", type=int, default=512)
    ap.add_argument("--shapelet-lengths", type=str, default="24,48")
    ap.add_argument("--shapelets-per-length", type=str, default="8,8")
    ap.add_argument("--shapelet-train-samples", type=int, default=4096)
    ap.add_argument("--shapelet-candidate-limit", type=int, default=12000)
    ap.add_argument("--shapelet-ridge-alpha", type=float, default=4.0)
    ap.add_argument("--eval-batch-size", type=int, default=256)
    args = ap.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (REPO_ROOT / config_path).resolve()
    run_dir = resolve_run_dir(config_path, args.run_dir)
    out_dir = Path(args.out_dir) if args.out_dir is not None else (run_dir.parent.parent / "shapelet_vs_knn")
    if not out_dir.is_absolute():
        out_dir = (REPO_ROOT / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_yaml(config_path)
    device_name = cfg["exp"].get("device", "cpu")
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    context = prepare_data_context(cfg)

    print(f"Config: {config_path}")
    print(f"Run dir: {run_dir}")
    print(f"Out dir: {out_dir}")
    print(f"Device: {device}")

    t0 = perf_counter()
    base_eval = evaluate_run(
        context=context,
        run_cfg=cfg,
        run_dir=run_dir,
        device=device,
        eval_batch_size=int(args.eval_batch_size),
    )
    mean_c = context.mean_c.float()
    std_c = context.std_c.float()
    xtr_ncl = context.xtr_norm.float().contiguous()
    ytr_nch = context.ytr_norm.float().contiguous()
    xte_ncl = context.xte_norm.float().contiguous()
    yte_nch = context.yte_norm.float().contiguous()
    cluster_id_c = context.cluster_id_c.contiguous()
    base_pred_nch = ((base_eval.yhat_raw.float() - mean_c.view(1, -1, 1)) / std_c.view(1, -1, 1)).contiguous()
    base_mse, base_mae_norm, base_mae_raw = compute_metrics(base_pred_nch, yte_nch, mean_c, std_c)
    print(f"Base model: avg_mse={base_mse:.6f}, avg_mae_norm={base_mae_norm:.6f}, avg_mae_raw={base_mae_raw:.6f}")

    k_grid = sorted(set(_parse_int_list(args.k_grid)))
    alpha_grid = sorted(set(_parse_float_list(args.alpha_grid)))
    shapelet_lengths = _parse_int_list(args.shapelet_lengths)
    shapelets_per_length = _parse_int_list(args.shapelets_per_length)
    if len(shapelet_lengths) != len(shapelets_per_length):
        raise ValueError("shapelet-lengths and shapelets-per-length must have the same length.")

    t_knn0 = perf_counter()
    knn_banks = build_knn_banks(
        xtr_ncl=xtr_ncl,
        ytr_nch=ytr_nch,
        cluster_id_c=cluster_id_c,
        scope=args.scope,
        shape_bins=int(args.shape_bins),
        diff_bins=int(args.diff_bins),
        train_stride=int(args.train_stride),
        anchor_mode=args.anchor_mode,
    )
    knn_bank_sec = perf_counter() - t_knn0
    print(f"KNN banks ready in {knn_bank_sec:.2f}s")

    t_shapelet0 = perf_counter()
    shapelet_models = build_shapelet_models(
        xtr_ncl=xtr_ncl,
        ytr_nch=ytr_nch,
        cluster_id_c=cluster_id_c,
        scope=args.scope,
        train_stride=int(args.train_stride),
        anchor_mode=args.anchor_mode,
        lengths=shapelet_lengths,
        shapelets_per_length=shapelets_per_length,
        train_sample_size=int(args.shapelet_train_samples),
        candidate_limit=int(args.shapelet_candidate_limit),
        ridge_alpha=float(args.shapelet_ridge_alpha),
        encode_batch_size=int(args.query_batch_size),
        random_state=int(cfg["exp"].get("seed", 0)),
    )
    shapelet_fit_sec = perf_counter() - t_shapelet0
    print(f"Shapelet models ready in {shapelet_fit_sec:.2f}s")

    rows = [{
        "method": "model_only",
        "scope": args.scope,
        "k": 0,
        "alpha": 0.0,
        "avg_mse": base_mse,
        "avg_mae_norm": base_mae_norm,
        "avg_mae_raw": base_mae_raw,
        "delta_mse": 0.0,
        "delta_mse_pct": 0.0,
        "runtime_sec": 0.0,
    }]

    best_preds: Dict[Tuple[str, int, float], torch.Tensor] = {}

    for k in k_grid:
        for alpha in alpha_grid:
            t1 = perf_counter()
            pred = run_knn_hybrid(
                xte_ncl=xte_ncl,
                base_pred_nch=base_pred_nch,
                cluster_id_c=cluster_id_c,
                banks=knn_banks,
                scope=args.scope,
                shape_bins=int(args.shape_bins),
                diff_bins=int(args.diff_bins),
                anchor_mode=args.anchor_mode,
                k=int(k),
                alpha=float(alpha),
                query_batch_size=int(args.query_batch_size),
            )
            runtime = perf_counter() - t1
            mse, mae_norm, mae_raw = compute_metrics(pred, yte_nch, mean_c, std_c)
            key = ("knn_hybrid", int(k), float(alpha))
            best_preds[key] = pred
            rows.append({
                "method": "knn_hybrid",
                "scope": args.scope,
                "k": int(k),
                "alpha": float(alpha),
                "avg_mse": mse,
                "avg_mae_norm": mae_norm,
                "avg_mae_raw": mae_raw,
                "delta_mse": mse - base_mse,
                "delta_mse_pct": (mse - base_mse) / max(base_mse, 1.0e-12) * 100.0,
                "runtime_sec": runtime,
            })
            print(f"KNN  k={int(k):2d} alpha={float(alpha):.3f} -> mse={mse:.6f}")

    for alpha in alpha_grid:
        t1 = perf_counter()
        pred = run_shapelet_hybrid(
            xte_ncl=xte_ncl,
            base_pred_nch=base_pred_nch,
            cluster_id_c=cluster_id_c,
            models=shapelet_models,
            scope=args.scope,
            anchor_mode=args.anchor_mode,
            alpha=float(alpha),
            encode_batch_size=int(args.query_batch_size),
        )
        runtime = perf_counter() - t1
        mse, mae_norm, mae_raw = compute_metrics(pred, yte_nch, mean_c, std_c)
        key = ("shapelet_hybrid", 0, float(alpha))
        best_preds[key] = pred
        rows.append({
            "method": "shapelet_hybrid",
            "scope": args.scope,
            "k": 0,
            "alpha": float(alpha),
            "avg_mse": mse,
            "avg_mae_norm": mae_norm,
            "avg_mae_raw": mae_raw,
            "delta_mse": mse - base_mse,
            "delta_mse_pct": (mse - base_mse) / max(base_mse, 1.0e-12) * 100.0,
            "runtime_sec": runtime,
        })
        print(f"Shapelet alpha={float(alpha):.3f} -> mse={mse:.6f}")

    results_df = pd.DataFrame(rows).sort_values(["avg_mse", "method", "k", "alpha"]).reset_index(drop=True)
    results_path = out_dir / "results.csv"
    results_df.to_csv(results_path, index=False)

    best_knn = results_df[results_df["method"] == "knn_hybrid"].sort_values("avg_mse").head(1)
    best_shapelet = results_df[results_df["method"] == "shapelet_hybrid"].sort_values("avg_mse").head(1)
    if best_knn.empty or best_shapelet.empty:
        raise RuntimeError("Missing KNN or shapelet results.")
    best_knn_row = best_knn.iloc[0].to_dict()
    best_shapelet_row = best_shapelet.iloc[0].to_dict()

    pred_knn = best_preds[("knn_hybrid", int(best_knn_row["k"]), float(best_knn_row["alpha"]))]
    pred_shapelet = best_preds[("shapelet_hybrid", 0, float(best_shapelet_row["alpha"]))]
    channel_rows = []
    for c, channel in enumerate(context.channel_names):
        base_mse_c = float((base_pred_nch[:, c, :] - yte_nch[:, c, :]).pow(2).mean().item())
        knn_mse_c = float((pred_knn[:, c, :] - yte_nch[:, c, :]).pow(2).mean().item())
        shp_mse_c = float((pred_shapelet[:, c, :] - yte_nch[:, c, :]).pow(2).mean().item())
        channel_rows.append({
            "channel": channel,
            "cluster_id": int(cluster_id_c[c].item()),
            "base_mse": base_mse_c,
            "knn_best_mse": knn_mse_c,
            "shapelet_best_mse": shp_mse_c,
            "knn_gain_vs_base": base_mse_c - knn_mse_c,
            "shapelet_gain_vs_base": base_mse_c - shp_mse_c,
            "shapelet_minus_knn": shp_mse_c - knn_mse_c,
        })
    channel_df = pd.DataFrame(channel_rows).sort_values("shapelet_minus_knn")
    channel_path = out_dir / "channel_comparison.csv"
    channel_df.to_csv(channel_path, index=False)

    summary = {
        "config_path": str(config_path),
        "run_dir": str(run_dir),
        "out_dir": str(out_dir),
        "scope": args.scope,
        "base": {
            "avg_mse": base_mse,
            "avg_mae_norm": base_mae_norm,
            "avg_mae_raw": base_mae_raw,
        },
        "best_knn": best_knn_row,
        "best_shapelet": best_shapelet_row,
        "shapelet_vs_knn_delta_mse": float(best_shapelet_row["avg_mse"] - best_knn_row["avg_mse"]),
        "knn_bank_sec": float(knn_bank_sec),
        "shapelet_fit_sec": float(shapelet_fit_sec),
        "shapelet_lengths": shapelet_lengths,
        "shapelets_per_length": shapelets_per_length,
        "shapelet_train_samples": int(args.shapelet_train_samples),
        "shapelet_candidate_limit": int(args.shapelet_candidate_limit),
        "elapsed_sec": float(perf_counter() - t0),
    }
    summary_path = out_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Saved results to: {results_path}")
    print(f"Saved channel comparison to: {channel_path}")
    print(f"Saved summary to: {summary_path}")
    print("Top results:")
    print(results_df.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
