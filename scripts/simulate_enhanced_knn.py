import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.compare_moe_on_off import load_eval_modules, load_yaml, prepare_data_context
from src.data.windows import WindowTensorDataset, make_strict_windows
from src.utils.knn_shape import build_future_template, build_shape_features, reconstruct_from_template


@dataclass
class EnhancedBank:
    starts_n: np.ndarray
    hist_feat_nd: np.ndarray
    joint_feat_nd: np.ndarray
    future_template_nh: np.ndarray
    residual_template_nh: np.ndarray

    @property
    def size(self) -> int:
        return int(self.starts_n.shape[0])


def _parse_int_list(text: str) -> List[int]:
    values = []
    for item in str(text).split(","):
        item = item.strip()
        if item:
            values.append(int(item))
    if len(values) == 0:
        raise ValueError("Expected at least one integer.")
    return sorted(set(values))


def _parse_float_list(text: str) -> List[float]:
    values = []
    for item in str(text).split(","):
        item = item.strip()
        if item:
            values.append(float(item))
    if len(values) == 0:
        raise ValueError("Expected at least one float.")
    return sorted(set(values))


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def predict_model(
    model: torch.nn.Module,
    x_ncl: torch.Tensor,
    cluster_id_c: torch.Tensor,
    pred_len: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    loader = DataLoader(
        WindowTensorDataset(x_ncl, torch.zeros((x_ncl.shape[0], x_ncl.shape[1], pred_len), dtype=x_ncl.dtype)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    out = torch.empty((x_ncl.shape[0], x_ncl.shape[1], pred_len), dtype=torch.float32)
    cluster_id_c = cluster_id_c.to(device)
    model.eval()
    with torch.no_grad():
        for x, _, idx in loader:
            x = x.to(device, non_blocking=True)
            yhat = model(x, cluster_id_c)
            out[idx.long()] = yhat.detach().cpu()
    return out


def compute_metrics(pred_nch: torch.Tensor, true_nch: torch.Tensor) -> Tuple[float, float]:
    mse = float((pred_nch - true_nch).pow(2).mean().item())
    mae = float((pred_nch - true_nch).abs().mean().item())
    return mse, mae


def build_bank_for_channel(
    hist_nl: torch.Tensor,
    fut_nh: torch.Tensor,
    base_nh: torch.Tensor,
    starts_n: np.ndarray,
    hist_shape_bins: int,
    hist_diff_bins: int,
    pred_shape_bins: int,
    pred_diff_bins: int,
    anchor_mode: str,
    eps: float = 1.0e-6,
) -> EnhancedBank:
    hist_feat_nd = build_shape_features(hist_nl, shape_bins=hist_shape_bins, diff_bins=hist_diff_bins).cpu().numpy().astype(np.float32)
    pred_feat_nd = build_shape_features(base_nh, shape_bins=pred_shape_bins, diff_bins=pred_diff_bins).cpu().numpy().astype(np.float32)
    joint_feat_nd = np.concatenate([hist_feat_nd, pred_feat_nd], axis=1).astype(np.float32)
    future_template_nh = build_future_template(hist_nl, fut_nh, anchor_mode=anchor_mode).cpu().numpy().astype(np.float32)
    hist_std_n1 = hist_nl.std(dim=-1, keepdim=True).clamp_min(eps).cpu().numpy().astype(np.float32)
    residual_template_nh = ((fut_nh - base_nh).cpu().numpy().astype(np.float32)) / hist_std_n1
    order = np.argsort(starts_n, kind="stable")
    return EnhancedBank(
        starts_n=starts_n[order],
        hist_feat_nd=hist_feat_nd[order],
        joint_feat_nd=joint_feat_nd[order],
        future_template_nh=future_template_nh[order],
        residual_template_nh=residual_template_nh[order],
    )


def build_enhanced_banks(
    x_bank_ncl: torch.Tensor,
    y_bank_nch: torch.Tensor,
    base_bank_pred_nch: torch.Tensor,
    bank_stride: int,
    hist_shape_bins: int,
    hist_diff_bins: int,
    pred_shape_bins: int,
    pred_diff_bins: int,
    anchor_mode: str,
) -> Dict[int, EnhancedBank]:
    stride = max(1, int(bank_stride))
    starts = np.arange(x_bank_ncl.shape[0], dtype=np.int64)[::stride]
    x_sub = x_bank_ncl[::stride].float().contiguous()
    y_sub = y_bank_nch[::stride].float().contiguous()
    b_sub = base_bank_pred_nch[::stride].float().contiguous()
    banks: Dict[int, EnhancedBank] = {}
    for c in range(x_sub.shape[1]):
        banks[c] = build_bank_for_channel(
            hist_nl=x_sub[:, c, :],
            fut_nh=y_sub[:, c, :],
            base_nh=b_sub[:, c, :],
            starts_n=starts.copy(),
            hist_shape_bins=hist_shape_bins,
            hist_diff_bins=hist_diff_bins,
            pred_shape_bins=pred_shape_bins,
            pred_diff_bins=pred_diff_bins,
            anchor_mode=anchor_mode,
        )
    return banks


def compute_topk_causal(
    query_feat_qd: np.ndarray,
    bank_feat_nd: np.ndarray,
    starts_n: np.ndarray,
    query_start_abs_q: np.ndarray,
    pred_len: int,
    max_k: int,
    query_batch_size: int,
    bank_chunk_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    q_feat = torch.from_numpy(query_feat_qd)
    q_sq = q_feat.pow(2).sum(dim=1, keepdim=True)
    b_feat = torch.from_numpy(bank_feat_nd)
    b_sq = b_feat.pow(2).sum(dim=1)
    allowed_count_q = np.searchsorted(starts_n, query_start_abs_q - int(pred_len), side="right").astype(np.int64)
    best_dist = torch.full((q_feat.shape[0], int(max_k)), float("inf"), dtype=q_feat.dtype)
    best_idx = torch.full((q_feat.shape[0], int(max_k)), -1, dtype=torch.long)

    for q0 in range(0, q_feat.shape[0], int(query_batch_size)):
        q1 = min(q0 + int(query_batch_size), q_feat.shape[0])
        allowed_count = allowed_count_q[q0:q1]
        local_best_dist = best_dist[q0:q1].clone()
        local_best_idx = best_idx[q0:q1].clone()
        for b0 in range(0, b_feat.shape[0], int(bank_chunk_size)):
            b1 = min(b0 + int(bank_chunk_size), b_feat.shape[0])
            dist = (q_sq[q0:q1] + b_sq[b0:b1].view(1, -1) - 2.0 * torch.matmul(q_feat[q0:q1], b_feat[b0:b1].t())).clamp_min(0.0)
            valid = torch.from_numpy((allowed_count[:, None] > np.arange(b0, b1, dtype=np.int64)[None, :]))
            dist = torch.where(valid, dist, torch.full_like(dist, float("inf")))
            cand_dist = torch.cat([local_best_dist, dist], dim=1)
            cand_idx_new = torch.arange(b0, b1, dtype=torch.long).view(1, -1).expand(q1 - q0, -1)
            cand_idx = torch.cat([local_best_idx, cand_idx_new], dim=1)
            topv, topi = torch.topk(cand_dist, k=int(max_k), dim=1, largest=False)
            local_best_dist = topv
            local_best_idx = cand_idx.gather(1, topi)
        best_dist[q0:q1] = local_best_dist
        best_idx[q0:q1] = local_best_idx
    return best_dist.numpy().astype(np.float32), best_idx.numpy().astype(np.int64)


def build_neighbor_cache(
    x_query_qcl: torch.Tensor,
    base_query_qch: torch.Tensor,
    query_start_abs_q: np.ndarray,
    banks: Dict[int, EnhancedBank],
    hist_shape_bins: int,
    hist_diff_bins: int,
    pred_shape_bins: int,
    pred_diff_bins: int,
    max_k: int,
    query_batch_size: int,
    bank_chunk_size: int,
    pred_len: int,
) -> Dict[str, Dict[int, Tuple[np.ndarray, np.ndarray]]]:
    cache = {"hist": {}, "joint": {}}
    for c in range(x_query_qcl.shape[1]):
        hist_feat = build_shape_features(x_query_qcl[:, c, :].float().contiguous(), shape_bins=hist_shape_bins, diff_bins=hist_diff_bins).cpu().numpy().astype(np.float32)
        pred_feat = build_shape_features(base_query_qch[:, c, :].float().contiguous(), shape_bins=pred_shape_bins, diff_bins=pred_diff_bins).cpu().numpy().astype(np.float32)
        joint_feat = np.concatenate([hist_feat, pred_feat], axis=1).astype(np.float32)
        bank = banks[c]
        cache["hist"][c] = compute_topk_causal(
            query_feat_qd=hist_feat,
            bank_feat_nd=bank.hist_feat_nd,
            starts_n=bank.starts_n,
            query_start_abs_q=query_start_abs_q,
            pred_len=pred_len,
            max_k=max_k,
            query_batch_size=query_batch_size,
            bank_chunk_size=bank_chunk_size,
        )
        cache["joint"][c] = compute_topk_causal(
            query_feat_qd=joint_feat,
            bank_feat_nd=bank.joint_feat_nd,
            starts_n=bank.starts_n,
            query_start_abs_q=query_start_abs_q,
            pred_len=pred_len,
            max_k=max_k,
            query_batch_size=query_batch_size,
            bank_chunk_size=bank_chunk_size,
        )
    return cache


def _alpha_from_dispersion(alpha: float, tpl_bkh: np.ndarray, weight_bk: np.ndarray, adaptive: str) -> np.ndarray:
    if adaptive != "agreement":
        return np.full((tpl_bkh.shape[0], 1), float(alpha), dtype=np.float32)
    tpl_mean = (tpl_bkh * weight_bk[..., None]).sum(axis=1, keepdims=True)
    disp = np.sqrt(((tpl_bkh - tpl_mean) ** 2 * weight_bk[..., None]).sum(axis=(1, 2)) / max(tpl_bkh.shape[2], 1))
    alpha_eff = float(alpha) / (1.0 + disp)
    return alpha_eff.reshape(-1, 1).astype(np.float32)


def predict_from_cache(
    x_query_qcl: torch.Tensor,
    base_query_qch: torch.Tensor,
    banks: Dict[int, EnhancedBank],
    cache: Dict[str, Dict[int, Tuple[np.ndarray, np.ndarray]]],
    feature_key: str,
    template_key: str,
    k: int,
    alpha: float,
    anchor_mode: str,
    adaptive: str,
    batch_size: int,
) -> torch.Tensor:
    out = base_query_qch.clone()
    for c in range(x_query_qcl.shape[1]):
        dist_qk, idx_qk = cache[feature_key][c]
        bank = banks[c]
        templates_nh = bank.future_template_nh if template_key == "future" else bank.residual_template_nh
        pred_list = []
        for q0 in range(0, x_query_qcl.shape[0], int(batch_size)):
            q1 = min(q0 + int(batch_size), x_query_qcl.shape[0])
            dist = dist_qk[q0:q1, :int(k)]
            idx = idx_qk[q0:q1, :int(k)]
            valid = np.isfinite(dist) & (idx >= 0)
            w = np.where(valid, 1.0 / np.maximum(dist, 1.0e-6), 0.0).astype(np.float32)
            w_sum = np.maximum(w.sum(axis=1, keepdims=True), 1.0e-6)
            weight = w / w_sum
            tpl_bkh = templates_nh[idx]
            tpl_bkh = np.where(valid[..., None], tpl_bkh, 0.0)
            tpl_bh = (tpl_bkh * weight[..., None]).sum(axis=1)
            alpha_eff = _alpha_from_dispersion(alpha=float(alpha), tpl_bkh=tpl_bkh, weight_bk=weight, adaptive=adaptive)
            hist = x_query_qcl[q0:q1, c, :]
            base = base_query_qch[q0:q1, c, :].cpu().numpy().astype(np.float32)
            if template_key == "future":
                knn_pred = reconstruct_from_template(hist, tpl_bh.astype(np.float32), anchor_mode=anchor_mode)
                pred = (1.0 - alpha_eff) * base + alpha_eff * knn_pred
            else:
                hist_std = hist.std(dim=-1, keepdim=True).clamp_min(1.0e-6).cpu().numpy().astype(np.float32)
                resid = tpl_bh.astype(np.float32) * hist_std
                pred = base + alpha_eff * resid
            pred_list.append(torch.from_numpy(pred))
        out[:, c, :] = torch.cat(pred_list, dim=0).to(dtype=out.dtype)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="outputs/ETTm1/best_config_search_configs/mse_0p9.yaml")
    ap.add_argument("--run-dir", type=str, default="outputs/ETTm1/best_config_search_runs/mse_0p9")
    ap.add_argument("--out-dir", type=str, default="outputs/ETTm1/enhanced_knn_eval")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--eval-batch-size", type=int, default=256)
    ap.add_argument("--query-batch-size", type=int, default=256)
    ap.add_argument("--bank-chunk-size", type=int, default=8192)
    ap.add_argument("--bank-stride", type=int, default=2)
    ap.add_argument("--hist-shape-bins", type=int, default=24)
    ap.add_argument("--hist-diff-bins", type=int, default=12)
    ap.add_argument("--pred-shape-bins", type=int, default=16)
    ap.add_argument("--pred-diff-bins", type=int, default=8)
    ap.add_argument("--anchor-mode", type=str, default="last", choices=["last", "mean"])
    ap.add_argument("--max-k", type=int, default=64)
    ap.add_argument("--direct-k-grid", type=str, default="32,40,48")
    ap.add_argument("--direct-alpha-grid", type=str, default="0.22,0.25,0.28,0.30,0.35")
    ap.add_argument("--residual-k-grid", type=str, default="24,32,40,48,64")
    ap.add_argument("--residual-alpha-grid", type=str, default="0.5,0.7,1.0,1.2,1.5")
    ap.add_argument("--combo-gamma-grid", type=str, default="0.15,0.25,0.35,0.50,0.70")
    args = ap.parse_args()

    config_path = resolve_path(args.config)
    run_dir = resolve_path(args.run_dir)
    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_yaml(config_path)
    context = prepare_data_context(cfg)
    device_name = args.device or cfg["exp"].get("device", "cpu")
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")

    xva, yva = make_strict_windows(context.norm_data_tc, context.L, context.H, context.t_train, context.t_val)
    xte, yte = context.xte_norm.float().contiguous(), context.yte_norm.float().contiguous()
    xall, yall = make_strict_windows(context.norm_data_tc, context.L, context.H, 0, context.norm_data_tc.shape[0])
    start_bank = np.arange(xall.shape[0], dtype=np.int64)[:: int(args.bank_stride)]
    xbank = xall[:: int(args.bank_stride)].float().contiguous()
    ybank = yall[:: int(args.bank_stride)].float().contiguous()
    del xall
    del yall

    t0 = perf_counter()
    bundle = load_eval_modules(cfg, run_dir / "best_checkpoint.pt", context.K, device)
    model = bundle["model"]
    cluster_id_c = context.cluster_id_c
    base_val = predict_model(model, xva.float().contiguous(), cluster_id_c, pred_len=context.H, batch_size=int(args.eval_batch_size), device=device).contiguous()
    base_test = predict_model(model, xte, cluster_id_c, pred_len=context.H, batch_size=int(args.eval_batch_size), device=device).contiguous()
    base_bank = predict_model(model, xbank, cluster_id_c, pred_len=context.H, batch_size=int(args.eval_batch_size), device=device).contiguous()

    banks = build_enhanced_banks(
        x_bank_ncl=xbank,
        y_bank_nch=ybank,
        base_bank_pred_nch=base_bank,
        bank_stride=1,
        hist_shape_bins=int(args.hist_shape_bins),
        hist_diff_bins=int(args.hist_diff_bins),
        pred_shape_bins=int(args.pred_shape_bins),
        pred_diff_bins=int(args.pred_diff_bins),
        anchor_mode=str(args.anchor_mode),
    )

    query_val = context.t_train + np.arange(xva.shape[0], dtype=np.int64)
    query_test = context.t_val + np.arange(xte.shape[0], dtype=np.int64)

    # Align bank starts after subsampling to stride windows.
    for c in banks:
        banks[c].starts_n = start_bank.copy()

    cache_val = build_neighbor_cache(
        x_query_qcl=xva.float().contiguous(),
        base_query_qch=base_val,
        query_start_abs_q=query_val,
        banks=banks,
        hist_shape_bins=int(args.hist_shape_bins),
        hist_diff_bins=int(args.hist_diff_bins),
        pred_shape_bins=int(args.pred_shape_bins),
        pred_diff_bins=int(args.pred_diff_bins),
        max_k=int(args.max_k),
        query_batch_size=int(args.query_batch_size),
        bank_chunk_size=int(args.bank_chunk_size),
        pred_len=int(context.H),
    )
    cache_test = build_neighbor_cache(
        x_query_qcl=xte,
        base_query_qch=base_test,
        query_start_abs_q=query_test,
        banks=banks,
        hist_shape_bins=int(args.hist_shape_bins),
        hist_diff_bins=int(args.hist_diff_bins),
        pred_shape_bins=int(args.pred_shape_bins),
        pred_diff_bins=int(args.pred_diff_bins),
        max_k=int(args.max_k),
        query_batch_size=int(args.query_batch_size),
        bank_chunk_size=int(args.bank_chunk_size),
        pred_len=int(context.H),
    )

    rows = []
    base_val_mse, base_val_mae = compute_metrics(base_val, yva.float())
    base_test_mse, base_test_mae = compute_metrics(base_test, yte)
    rows.append({
        "method": "base",
        "feature": "base",
        "template": "base",
        "adaptive": "none",
        "k": 0,
        "alpha": 0.0,
        "gamma": 0.0,
        "val_mse": base_val_mse,
        "val_mae_norm": base_val_mae,
        "test_mse": base_test_mse,
        "test_mae_norm": base_test_mae,
    })
    print(f"Base: val_mse={base_val_mse:.6f}, test_mse={base_test_mse:.6f}")

    best_family: Dict[str, dict] = {}

    def evaluate_family(
        family: str,
        feature: str,
        template: str,
        adaptive_modes: List[str],
        k_grid: List[int],
        alpha_grid: List[float],
    ) -> None:
        for adaptive in adaptive_modes:
            for k in k_grid:
                for alpha in alpha_grid:
                    name = f"{family}_{feature}_{template}_{adaptive}_k{k}_a{alpha:.2f}"
                    pred_val = predict_from_cache(
                        x_query_qcl=xva.float().contiguous(),
                        base_query_qch=base_val,
                        banks=banks,
                        cache=cache_val,
                        feature_key=feature,
                        template_key=template,
                        k=int(k),
                        alpha=float(alpha),
                        anchor_mode=str(args.anchor_mode),
                        adaptive=adaptive,
                        batch_size=int(args.query_batch_size),
                    )
                    pred_test = predict_from_cache(
                        x_query_qcl=xte,
                        base_query_qch=base_test,
                        banks=banks,
                        cache=cache_test,
                        feature_key=feature,
                        template_key=template,
                        k=int(k),
                        alpha=float(alpha),
                        anchor_mode=str(args.anchor_mode),
                        adaptive=adaptive,
                        batch_size=int(args.query_batch_size),
                    )
                    val_mse, val_mae = compute_metrics(pred_val, yva.float())
                    test_mse, test_mae = compute_metrics(pred_test, yte)
                    rows.append({
                        "method": family,
                        "feature": feature,
                        "template": template,
                        "adaptive": adaptive,
                        "k": int(k),
                        "alpha": float(alpha),
                        "gamma": 0.0,
                        "val_mse": val_mse,
                        "val_mae_norm": val_mae,
                        "test_mse": test_mse,
                        "test_mae_norm": test_mae,
                    })
                    best = best_family.get(family, None)
                    if best is None or val_mse < float(best["row"]["val_mse"]):
                        best_family[family] = {
                            "name": name,
                            "row": rows[-1].copy(),
                            "pred_val": pred_val.clone(),
                            "pred_test": pred_test.clone(),
                        }
                    print(f"{name}: val_mse={val_mse:.6f}, test_mse={test_mse:.6f}")

    evaluate_family(
        family="direct",
        feature="hist",
        template="future",
        adaptive_modes=["none", "agreement"],
        k_grid=_parse_int_list(args.direct_k_grid),
        alpha_grid=_parse_float_list(args.direct_alpha_grid),
    )
    evaluate_family(
        family="direct",
        feature="joint",
        template="future",
        adaptive_modes=["none", "agreement"],
        k_grid=_parse_int_list(args.direct_k_grid),
        alpha_grid=_parse_float_list(args.direct_alpha_grid),
    )
    evaluate_family(
        family="residual",
        feature="hist",
        template="residual",
        adaptive_modes=["none", "agreement"],
        k_grid=_parse_int_list(args.residual_k_grid),
        alpha_grid=_parse_float_list(args.residual_alpha_grid),
    )
    evaluate_family(
        family="residual",
        feature="joint",
        template="residual",
        adaptive_modes=["none", "agreement"],
        k_grid=_parse_int_list(args.residual_k_grid),
        alpha_grid=_parse_float_list(args.residual_alpha_grid),
    )

    results_df = pd.DataFrame(rows).sort_values(["val_mse", "test_mse", "method", "feature", "template", "adaptive", "k", "alpha"]).reset_index(drop=True)
    direct_best = best_family["direct"]
    residual_best = best_family["residual"]
    direct_best_row = dict(direct_best["row"])
    residual_best_row = dict(residual_best["row"])
    direct_name = str(direct_best["name"])
    residual_name = str(residual_best["name"])

    combo_rows = []
    for gamma in _parse_float_list(args.combo_gamma_grid):
        name = f"combo_gamma_{gamma:.2f}"
        pred_val = (1.0 - float(gamma)) * direct_best["pred_val"] + float(gamma) * residual_best["pred_val"]
        pred_test = (1.0 - float(gamma)) * direct_best["pred_test"] + float(gamma) * residual_best["pred_test"]
        val_mse, val_mae = compute_metrics(pred_val, yva.float())
        test_mse, test_mae = compute_metrics(pred_test, yte)
        combo_rows.append({
            "method": "combo",
            "feature": f"{direct_best_row['feature']}+{residual_best_row['feature']}",
            "template": "future+residual",
            "adaptive": f"{direct_best_row['adaptive']}+{residual_best_row['adaptive']}",
            "k": int(direct_best_row["k"]),
            "alpha": float(direct_best_row["alpha"]),
            "gamma": float(gamma),
            "val_mse": val_mse,
            "val_mae_norm": val_mae,
            "test_mse": test_mse,
            "test_mae_norm": test_mae,
            "direct_name": direct_name,
            "residual_name": residual_name,
        })
        print(f"{name}: val_mse={val_mse:.6f}, test_mse={test_mse:.6f}")
    combo_df = pd.DataFrame(combo_rows).sort_values(["val_mse", "test_mse"]).reset_index(drop=True)
    best_combo = None if combo_df.shape[0] == 0 else combo_df.iloc[0].to_dict()

    results_path = out_dir / "results.csv"
    combo_path = out_dir / "combo_results.csv"
    summary_path = out_dir / "summary.json"
    results_df.to_csv(results_path, index=False)
    combo_df.to_csv(combo_path, index=False)
    summary = {
        "config_path": str(config_path),
        "run_dir": str(run_dir),
        "base": results_df.iloc[results_df.index[results_df["method"] == "base"][0]].to_dict(),
        "best_direct": direct_best_row,
        "best_residual": residual_best_row,
        "best_combo": best_combo,
        "elapsed_sec": float(perf_counter() - t0),
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Saved results to: {results_path}")
    print(f"Saved combo results to: {combo_path}")
    print(f"Saved summary to: {summary_path}")
    print("Top results:")
    print(results_df.head(16).to_string(index=False))
    if combo_df.shape[0] > 0:
        print("Top combo results:")
        print(combo_df.head(16).to_string(index=False))


if __name__ == "__main__":
    main()
