import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.compare_moe_on_off import evaluate_run, load_yaml, prepare_data_context
from src.data.windows import make_strict_windows
from src.utils.knn_shape import build_future_template, build_shape_features, reconstruct_from_template


@dataclass
class RollingBank:
    key: int
    label: str
    starts_n: np.ndarray
    features_nd: np.ndarray
    future_template_nh: np.ndarray

    @property
    def size(self) -> int:
        return int(self.features_nd.shape[0])


def _parse_int_list(text: str) -> List[int]:
    values = []
    for item in str(text).split(","):
        item = item.strip()
        if item:
            values.append(int(item))
    if len(values) == 0:
        raise ValueError("Expected at least one integer.")
    return values


def _parse_float_list(text: str) -> List[float]:
    values = []
    for item in str(text).split(","):
        item = item.strip()
        if item:
            values.append(float(item))
    if len(values) == 0:
        raise ValueError("Expected at least one float.")
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


def collect_full_bank_series(
    xall_ncl: torch.Tensor,
    yall_nch: torch.Tensor,
    cluster_id_c: torch.Tensor,
    scope: str,
    key: int,
    bank_stride: int,
) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    stride = max(1, int(bank_stride))
    starts = np.arange(xall_ncl.shape[0], dtype=np.int64)[::stride]
    x_sub = xall_ncl[::stride]
    y_sub = yall_nch[::stride]

    if scope == "same_channel":
        return x_sub[:, key, :].contiguous(), y_sub[:, key, :].contiguous(), starts
    if scope == "same_cluster":
        members = (cluster_id_c == key).nonzero(as_tuple=False).view(-1)
        x_bank = x_sub.index_select(1, members).permute(1, 0, 2).reshape(-1, x_sub.shape[-1]).contiguous()
        y_bank = y_sub.index_select(1, members).permute(1, 0, 2).reshape(-1, y_sub.shape[-1]).contiguous()
        start_bank = np.tile(starts, reps=int(members.numel()))
        return x_bank, y_bank, start_bank
    raise ValueError(f"Unsupported scope={scope}")


def build_rolling_banks(
    xall_ncl: torch.Tensor,
    yall_nch: torch.Tensor,
    cluster_id_c: torch.Tensor,
    scope: str,
    shape_bins: int,
    diff_bins: int,
    bank_stride: int,
    anchor_mode: str,
) -> Dict[int, RollingBank]:
    if scope == "same_channel":
        keys: Iterable[int] = range(xall_ncl.shape[1])
    elif scope == "same_cluster":
        keys = range(int(cluster_id_c.max().item()) + 1)
    else:
        raise ValueError(f"Unsupported scope={scope}")

    banks: Dict[int, RollingBank] = {}
    for key in keys:
        hist_nl, fut_nh, starts_n = collect_full_bank_series(
            xall_ncl=xall_ncl,
            yall_nch=yall_nch,
            cluster_id_c=cluster_id_c,
            scope=scope,
            key=int(key),
            bank_stride=bank_stride,
        )
        feat_nd = build_shape_features(hist_nl, shape_bins=shape_bins, diff_bins=diff_bins).cpu().numpy().astype(np.float32)
        tpl_nh = build_future_template(hist_nl, fut_nh, anchor_mode=anchor_mode).cpu().numpy().astype(np.float32)
        order = np.argsort(starts_n, kind="stable")
        banks[int(key)] = RollingBank(
            key=int(key),
            label=make_scope_label(scope, int(key)),
            starts_n=starts_n[order],
            features_nd=feat_nd[order],
            future_template_nh=tpl_nh[order],
        )
    return banks


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


def fixed_bank_predict(
    query_hist_qcl: torch.Tensor,
    base_pred_qch: torch.Tensor,
    cluster_id_c: torch.Tensor,
    banks: Dict[int, RollingBank],
    scope: str,
    shape_bins: int,
    diff_bins: int,
    anchor_mode: str,
    k: int,
    alpha: float,
    max_start_allowed: int,
    query_batch_size: int,
) -> torch.Tensor:
    out = base_pred_qch.clone()
    for c in range(query_hist_qcl.shape[1]):
        bank = banks[resolve_bank_key(scope, c, cluster_id_c)]
        allowed_count = int(np.searchsorted(bank.starts_n, int(max_start_allowed), side="right"))
        if allowed_count <= 0:
            continue
        query_hist = query_hist_qcl[:, c, :].float().contiguous()
        query_feat = build_shape_features(query_hist, shape_bins=shape_bins, diff_bins=diff_bins).cpu().numpy().astype(np.float32)
        bank_feat = torch.from_numpy(bank.features_nd[:allowed_count])
        bank_tpl = bank.future_template_nh[:allowed_count]
        pred_out = torch.empty_like(out[:, c, :])
        k_eff = min(int(k), allowed_count)
        bank_sq = bank_feat.pow(2).sum(dim=1)
        for start in range(0, query_feat.shape[0], query_batch_size):
            end = min(start + query_batch_size, query_feat.shape[0])
            q_feat = torch.from_numpy(query_feat[start:end])
            q_sq = q_feat.pow(2).sum(dim=1, keepdim=True)
            dist = (q_sq + bank_sq.view(1, -1) - 2.0 * torch.matmul(q_feat, bank_feat.t())).clamp_min(0.0)
            topv, topi = torch.topk(dist, k=k_eff, dim=1, largest=False)
            w = 1.0 / topv.clamp_min(1.0e-6)
            tpl = bank_tpl[topi.numpy()]
            tpl = (tpl * w.unsqueeze(-1).numpy()).sum(axis=1) / np.maximum(w.sum(dim=1, keepdim=True).numpy(), 1.0e-6)
            knn_pred = reconstruct_from_template(query_hist[start:end], tpl.astype(np.float32), anchor_mode)
            knn_pred = torch.from_numpy(knn_pred).to(dtype=pred_out.dtype)
            pred_out[start:end] = (1.0 - float(alpha)) * base_pred_qch[start:end, c, :] + float(alpha) * knn_pred
        out[:, c, :] = pred_out
    return out


def rolling_bank_predict(
    query_hist_qcl: torch.Tensor,
    base_pred_qch: torch.Tensor,
    query_start_abs_q: np.ndarray,
    cluster_id_c: torch.Tensor,
    banks: Dict[int, RollingBank],
    scope: str,
    shape_bins: int,
    diff_bins: int,
    anchor_mode: str,
    k: int,
    alpha: float,
    pred_len: int,
    query_batch_size: int,
    bank_chunk_size: int,
) -> torch.Tensor:
    out = base_pred_qch.clone()
    valid_limit_q = query_start_abs_q - int(pred_len)
    for c in range(query_hist_qcl.shape[1]):
        bank = banks[resolve_bank_key(scope, c, cluster_id_c)]
        query_hist = query_hist_qcl[:, c, :].float().contiguous()
        query_feat = build_shape_features(query_hist, shape_bins=shape_bins, diff_bins=diff_bins).cpu().numpy().astype(np.float32)
        bank_feat = torch.from_numpy(bank.features_nd)
        bank_sq = bank_feat.pow(2).sum(dim=1)
        pred_out = torch.empty_like(out[:, c, :])
        allowed_count_q = np.searchsorted(bank.starts_n, valid_limit_q, side="right").astype(np.int64)
        for q0 in range(0, query_feat.shape[0], query_batch_size):
            q1 = min(q0 + query_batch_size, query_feat.shape[0])
            q_feat = torch.from_numpy(query_feat[q0:q1])
            q_sq = q_feat.pow(2).sum(dim=1, keepdim=True)
            allowed_count = allowed_count_q[q0:q1]
            best_dist = torch.full((q1 - q0, int(k)), float("inf"), dtype=q_feat.dtype)
            best_idx = torch.full((q1 - q0, int(k)), -1, dtype=torch.long)
            for b0 in range(0, bank.size, bank_chunk_size):
                b1 = min(b0 + bank_chunk_size, bank.size)
                dist = (q_sq + bank_sq[b0:b1].view(1, -1) - 2.0 * torch.matmul(q_feat, bank_feat[b0:b1].t())).clamp_min(0.0)
                valid = torch.from_numpy((allowed_count[:, None] > np.arange(b0, b1, dtype=np.int64)[None, :]))
                dist = torch.where(valid, dist, torch.full_like(dist, float("inf")))
                cand_dist = torch.cat([best_dist, dist], dim=1)
                cand_idx_new = torch.arange(b0, b1, dtype=torch.long).view(1, -1).expand(q1 - q0, -1)
                cand_idx = torch.cat([best_idx, cand_idx_new], dim=1)
                topv, topi = torch.topk(cand_dist, k=int(k), dim=1, largest=False)
                best_dist = topv
                best_idx = cand_idx.gather(1, topi)

            row_pred = []
            for row in range(q1 - q0):
                valid_idx = best_idx[row][torch.isfinite(best_dist[row])].cpu().numpy()
                if valid_idx.size == 0:
                    row_pred.append(base_pred_qch[q0 + row, c, :].numpy())
                    continue
                valid_dist = best_dist[row][torch.isfinite(best_dist[row])].cpu().numpy()
                tpl = bank.future_template_nh[valid_idx]
                w = 1.0 / np.maximum(valid_dist, 1.0e-6)
                tpl = (tpl * w[:, None]).sum(axis=0, keepdims=True) / np.maximum(w.sum(), 1.0e-6)
                knn_pred = reconstruct_from_template(query_hist[q0 + row:q0 + row + 1], tpl.astype(np.float32), anchor_mode)[0]
                base_pred = base_pred_qch[q0 + row, c, :].numpy()
                row_pred.append((1.0 - float(alpha)) * base_pred + float(alpha) * knn_pred)
            pred_out[q0:q1] = torch.from_numpy(np.stack(row_pred, axis=0)).to(dtype=pred_out.dtype)
        out[:, c, :] = pred_out
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--run-dir", type=str, default=None)
    ap.add_argument("--out-dir", type=str, default=None)
    ap.add_argument("--scope", type=str, default="same_cluster", choices=["same_channel", "same_cluster"])
    ap.add_argument("--bank-stride", type=int, default=4)
    ap.add_argument("--shape-bins", type=int, default=24)
    ap.add_argument("--diff-bins", type=int, default=12)
    ap.add_argument("--anchor-mode", type=str, default="last", choices=["last", "mean"])
    ap.add_argument("--k-grid", type=str, default="16,32")
    ap.add_argument("--alpha-grid", type=str, default="0.1,0.12,0.15,0.2")
    ap.add_argument("--query-batch-size", type=int, default=256)
    ap.add_argument("--bank-chunk-size", type=int, default=8192)
    ap.add_argument("--eval-batch-size", type=int, default=256)
    args = ap.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (REPO_ROOT / config_path).resolve()
    run_dir = resolve_run_dir(config_path, args.run_dir)
    out_dir = Path(args.out_dir) if args.out_dir is not None else (run_dir / "rolling_knn_sim")
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
    base_pred_nch = ((base_eval.yhat_raw.float() - mean_c.view(1, -1, 1)) / std_c.view(1, -1, 1)).contiguous()
    xte_ncl = context.xte_norm.float().contiguous()
    yte_nch = context.yte_norm.float().contiguous()
    cluster_id_c = context.cluster_id_c.contiguous()
    base_mse, base_mae_norm, base_mae_raw = compute_metrics(base_pred_nch, yte_nch, mean_c, std_c)
    print(f"Base: avg_mse={base_mse:.6f}, avg_mae_norm={base_mae_norm:.6f}, avg_mae_raw={base_mae_raw:.6f}")

    xall_ncl, yall_nch = make_strict_windows(context.norm_data_tc, context.L, context.H, 0, context.norm_data_tc.shape[0])
    banks = build_rolling_banks(
        xall_ncl=xall_ncl.float(),
        yall_nch=yall_nch.float(),
        cluster_id_c=cluster_id_c,
        scope=args.scope,
        shape_bins=int(args.shape_bins),
        diff_bins=int(args.diff_bins),
        bank_stride=int(args.bank_stride),
        anchor_mode=args.anchor_mode,
    )
    query_start_abs_q = context.t_val + np.arange(xte_ncl.shape[0], dtype=np.int64)
    fixed_max_start_allowed = int(context.t_val - context.H)

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

    k_grid = sorted(set(_parse_int_list(args.k_grid)))
    alpha_grid = sorted(set(_parse_float_list(args.alpha_grid)))
    best_preds: Dict[Tuple[str, int, float], torch.Tensor] = {}

    for k in k_grid:
        for alpha in alpha_grid:
            t1 = perf_counter()
            pred_fixed = fixed_bank_predict(
                query_hist_qcl=xte_ncl,
                base_pred_qch=base_pred_nch,
                cluster_id_c=cluster_id_c,
                banks=banks,
                scope=args.scope,
                shape_bins=int(args.shape_bins),
                diff_bins=int(args.diff_bins),
                anchor_mode=args.anchor_mode,
                k=int(k),
                alpha=float(alpha),
                max_start_allowed=fixed_max_start_allowed,
                query_batch_size=int(args.query_batch_size),
            )
            sec = perf_counter() - t1
            mse, mae_norm, mae_raw = compute_metrics(pred_fixed, yte_nch, mean_c, std_c)
            key = ("fixed_bank", int(k), float(alpha))
            best_preds[key] = pred_fixed
            rows.append({
                "method": "fixed_bank",
                "scope": args.scope,
                "k": int(k),
                "alpha": float(alpha),
                "avg_mse": mse,
                "avg_mae_norm": mae_norm,
                "avg_mae_raw": mae_raw,
                "delta_mse": mse - base_mse,
                "delta_mse_pct": (mse - base_mse) / max(base_mse, 1.0e-12) * 100.0,
                "runtime_sec": sec,
            })
            print(f"fixed_bank k={int(k):2d} alpha={float(alpha):.3f} -> mse={mse:.6f}")

    for k in k_grid:
        for alpha in alpha_grid:
            t1 = perf_counter()
            pred_roll = rolling_bank_predict(
                query_hist_qcl=xte_ncl,
                base_pred_qch=base_pred_nch,
                query_start_abs_q=query_start_abs_q,
                cluster_id_c=cluster_id_c,
                banks=banks,
                scope=args.scope,
                shape_bins=int(args.shape_bins),
                diff_bins=int(args.diff_bins),
                anchor_mode=args.anchor_mode,
                k=int(k),
                alpha=float(alpha),
                pred_len=int(context.H),
                query_batch_size=int(args.query_batch_size),
                bank_chunk_size=int(args.bank_chunk_size),
            )
            sec = perf_counter() - t1
            mse, mae_norm, mae_raw = compute_metrics(pred_roll, yte_nch, mean_c, std_c)
            key = ("rolling_bank", int(k), float(alpha))
            best_preds[key] = pred_roll
            rows.append({
                "method": "rolling_bank",
                "scope": args.scope,
                "k": int(k),
                "alpha": float(alpha),
                "avg_mse": mse,
                "avg_mae_norm": mae_norm,
                "avg_mae_raw": mae_raw,
                "delta_mse": mse - base_mse,
                "delta_mse_pct": (mse - base_mse) / max(base_mse, 1.0e-12) * 100.0,
                "runtime_sec": sec,
            })
            print(f"rolling_bank k={int(k):2d} alpha={float(alpha):.3f} -> mse={mse:.6f}")

    results_df = pd.DataFrame(rows).sort_values(["avg_mse", "method", "k", "alpha"]).reset_index(drop=True)
    results_path = out_dir / "results.csv"
    results_df.to_csv(results_path, index=False)

    best_fixed = results_df[results_df["method"] == "fixed_bank"].sort_values("avg_mse").head(1).iloc[0].to_dict()
    best_roll = results_df[results_df["method"] == "rolling_bank"].sort_values("avg_mse").head(1).iloc[0].to_dict()
    pred_fixed_best = best_preds[("fixed_bank", int(best_fixed["k"]), float(best_fixed["alpha"]))]
    pred_roll_best = best_preds[("rolling_bank", int(best_roll["k"]), float(best_roll["alpha"]))]

    channel_rows = []
    for c, channel in enumerate(context.channel_names):
        base_mse_c = float((base_pred_nch[:, c, :] - yte_nch[:, c, :]).pow(2).mean().item())
        fixed_mse_c = float((pred_fixed_best[:, c, :] - yte_nch[:, c, :]).pow(2).mean().item())
        roll_mse_c = float((pred_roll_best[:, c, :] - yte_nch[:, c, :]).pow(2).mean().item())
        channel_rows.append({
            "channel": channel,
            "cluster_id": int(cluster_id_c[c].item()),
            "base_mse": base_mse_c,
            "fixed_best_mse": fixed_mse_c,
            "rolling_best_mse": roll_mse_c,
            "fixed_gain_vs_base": base_mse_c - fixed_mse_c,
            "rolling_gain_vs_base": base_mse_c - roll_mse_c,
            "rolling_minus_fixed": roll_mse_c - fixed_mse_c,
        })
    channel_df = pd.DataFrame(channel_rows).sort_values("rolling_minus_fixed")
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
        "best_fixed_bank": best_fixed,
        "best_rolling_bank": best_roll,
        "rolling_vs_fixed_delta_mse": float(best_roll["avg_mse"] - best_fixed["avg_mse"]),
        "elapsed_sec": float(perf_counter() - t0),
        "bank_sizes": {str(k): int(v.size) for k, v in banks.items()},
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
