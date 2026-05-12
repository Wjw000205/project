import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Dict, Iterable, List

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
from src.utils.knn_shape import KNNShapeConfig, ShapeKNNHybrid, build_future_template, build_shape_features, reconstruct_from_template


@dataclass
class ClusterRecallBank:
    key: int
    label: str
    starts_n: np.ndarray
    features_nd: np.ndarray
    future_template_nh: np.ndarray

    @property
    def size(self) -> int:
        return int(self.features_nd.shape[0])


def _parse_int_list(text: str) -> List[int]:
    out = []
    for item in str(text).split(","):
        item = item.strip()
        if item:
            out.append(int(item))
    if len(out) == 0:
        raise ValueError("Expected at least one integer.")
    return sorted(set(out))


def _parse_float_list(text: str) -> List[float]:
    out = []
    for item in str(text).split(","):
        item = item.strip()
        if item:
            out.append(float(item))
    if len(out) == 0:
        raise ValueError("Expected at least one float.")
    return sorted(set(out))


def resolve_path(path_text: str | None) -> Path | None:
    if path_text is None:
        return None
    path = Path(path_text)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def predict_model(
    model: torch.nn.Module,
    x_ncl: torch.Tensor,
    cluster_id_c: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    loader = DataLoader(WindowTensorDataset(x_ncl, x_ncl.new_zeros((x_ncl.shape[0], x_ncl.shape[1], 1))), batch_size=batch_size, shuffle=False, num_workers=0)
    out = torch.empty((x_ncl.shape[0], x_ncl.shape[1], model.H), dtype=torch.float32)
    cluster_id_c = cluster_id_c.to(device)
    model.eval()
    with torch.no_grad():
        for x, _, idx in loader:
            x = x.to(device, non_blocking=True)
            yhat = model(x, cluster_id_c)
            out[idx.long()] = yhat.detach().cpu()
    return out


def compute_mse(pred_nch: torch.Tensor, true_nch: torch.Tensor) -> float:
    return float((pred_nch - true_nch).pow(2).mean().item())


def compute_mae(pred_nch: torch.Tensor, true_nch: torch.Tensor) -> float:
    return float((pred_nch - true_nch).abs().mean().item())


def build_cluster_recall_banks(
    xall_ncl: torch.Tensor,
    yall_nch: torch.Tensor,
    cluster_id_c: torch.Tensor,
    shape_bins: int,
    diff_bins: int,
    bank_stride: int,
    anchor_mode: str,
) -> Dict[int, ClusterRecallBank]:
    stride = max(1, int(bank_stride))
    starts = np.arange(xall_ncl.shape[0], dtype=np.int64)[::stride]
    banks: Dict[int, ClusterRecallBank] = {}
    cluster_keys: Iterable[int] = [int(v) for v in torch.unique(cluster_id_c.detach().cpu(), sorted=True).tolist()]
    for key in cluster_keys:
        members = (cluster_id_c == key).nonzero(as_tuple=False).view(-1)
        if members.numel() == 0:
            continue
        x_cluster_nl = xall_ncl[::stride].index_select(1, members).mean(dim=1).contiguous()
        y_cluster_nh = yall_nch[::stride].index_select(1, members).mean(dim=1).contiguous()
        feat_nd = build_shape_features(x_cluster_nl, shape_bins=shape_bins, diff_bins=diff_bins).cpu().numpy().astype(np.float32)
        tpl_nh = build_future_template(x_cluster_nl, y_cluster_nh, anchor_mode=anchor_mode).cpu().numpy().astype(np.float32)
        banks[key] = ClusterRecallBank(
            key=key,
            label=f"cluster_{key}",
            starts_n=starts.copy(),
            features_nd=feat_nd,
            future_template_nh=tpl_nh,
        )
    if len(banks) == 0:
        raise ValueError("Cluster recall bank is empty.")
    return banks


def predict_cluster_templates(
    query_cluster_hist_qkl: Dict[int, torch.Tensor],
    query_start_abs_q: np.ndarray,
    banks: Dict[int, ClusterRecallBank],
    shape_bins: int,
    diff_bins: int,
    anchor_mode: str,
    k: int,
    query_batch_size: int,
    bank_chunk_size: int,
    pred_len: int,
) -> Dict[int, np.ndarray]:
    out: Dict[int, np.ndarray] = {}
    valid_limit_q = query_start_abs_q - int(pred_len)
    for key, query_hist_ql in query_cluster_hist_qkl.items():
        bank = banks[key]
        q_feat = build_shape_features(query_hist_ql.float().contiguous(), shape_bins=shape_bins, diff_bins=diff_bins).cpu().numpy().astype(np.float32)
        q_feat_t = torch.from_numpy(q_feat)
        q_sq = q_feat_t.pow(2).sum(dim=1, keepdim=True)
        bank_feat = torch.from_numpy(bank.features_nd)
        bank_sq = bank_feat.pow(2).sum(dim=1)
        allowed_count_q = np.searchsorted(bank.starts_n, valid_limit_q, side="right").astype(np.int64)
        tpl_out = np.zeros((query_hist_ql.shape[0], pred_len), dtype=np.float32)

        for q0 in range(0, q_feat.shape[0], query_batch_size):
            q1 = min(q0 + query_batch_size, q_feat.shape[0])
            allowed_count = allowed_count_q[q0:q1]
            best_dist = torch.full((q1 - q0, int(k)), float("inf"), dtype=q_feat_t.dtype)
            best_idx = torch.full((q1 - q0, int(k)), -1, dtype=torch.long)
            for b0 in range(0, bank.size, bank_chunk_size):
                b1 = min(b0 + bank_chunk_size, bank.size)
                dist = (q_sq[q0:q1] + bank_sq[b0:b1].view(1, -1) - 2.0 * torch.matmul(q_feat_t[q0:q1], bank_feat[b0:b1].t())).clamp_min(0.0)
                valid = torch.from_numpy((allowed_count[:, None] > np.arange(b0, b1, dtype=np.int64)[None, :]))
                dist = torch.where(valid, dist, torch.full_like(dist, float("inf")))
                cand_dist = torch.cat([best_dist, dist], dim=1)
                cand_idx_new = torch.arange(b0, b1, dtype=torch.long).view(1, -1).expand(q1 - q0, -1)
                cand_idx = torch.cat([best_idx, cand_idx_new], dim=1)
                topv, topi = torch.topk(cand_dist, k=int(k), dim=1, largest=False)
                best_dist = topv
                best_idx = cand_idx.gather(1, topi)

            for row in range(q1 - q0):
                finite_mask = torch.isfinite(best_dist[row])
                idx = best_idx[row][finite_mask].cpu().numpy()
                if idx.size == 0:
                    continue
                dist = best_dist[row][finite_mask].cpu().numpy()
                w = 1.0 / np.maximum(dist, 1.0e-6)
                tpl = bank.future_template_nh[idx]
                tpl_out[q0 + row] = (tpl * w[:, None]).sum(axis=0) / np.maximum(w.sum(), 1.0e-6)
        out[key] = tpl_out
    return out


def apply_cluster_templates(
    query_hist_qcl: torch.Tensor,
    base_pred_qch: torch.Tensor,
    cluster_id_c: torch.Tensor,
    cluster_tpl_qkh: Dict[int, np.ndarray],
    alpha: float,
    anchor_mode: str,
) -> torch.Tensor:
    out = base_pred_qch.clone()
    for key, tpl_qh in cluster_tpl_qkh.items():
        members = (cluster_id_c == key).nonzero(as_tuple=False).view(-1)
        if members.numel() == 0:
            continue
        for c in members.tolist():
            knn_pred = reconstruct_from_template(query_hist_qcl[:, c, :], tpl_qh.astype(np.float32), anchor_mode)
            knn_pred_t = torch.from_numpy(knn_pred).to(dtype=out.dtype)
            out[:, c, :] = (1.0 - float(alpha)) * base_pred_qch[:, c, :] + float(alpha) * knn_pred_t
    return out


def evaluate_method(
    pred_val_nch: torch.Tensor,
    pred_test_nch: torch.Tensor,
    yva_nch: torch.Tensor,
    yte_nch: torch.Tensor,
) -> dict:
    return {
        "val_mse": compute_mse(pred_val_nch, yva_nch),
        "test_mse": compute_mse(pred_test_nch, yte_nch),
        "val_mae_norm": compute_mae(pred_val_nch, yva_nch),
        "test_mae_norm": compute_mae(pred_test_nch, yte_nch),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="outputs/ETTm1/best_config_search_configs/mse_0p9.yaml")
    ap.add_argument("--run-dir", type=str, default="outputs/ETTm1/best_config_search_runs/mse_0p9")
    ap.add_argument("--out-dir", type=str, default="outputs/ETTm1/cluster_recall_eval")
    ap.add_argument("--eval-batch-size", type=int, default=256)
    ap.add_argument("--query-batch-size", type=int, default=256)
    ap.add_argument("--bank-chunk-size", type=int, default=8192)
    ap.add_argument("--shape-bins", type=int, default=24)
    ap.add_argument("--diff-bins", type=int, default=12)
    ap.add_argument("--anchor-mode", type=str, default="last", choices=["last", "mean"])
    ap.add_argument("--same-channel-k-grid", type=str, default="32,40")
    ap.add_argument("--same-channel-alpha-grid", type=str, default="0.25,0.28,0.30")
    ap.add_argument("--same-cluster-k-grid", type=str, default="48,56,64")
    ap.add_argument("--same-cluster-alpha-grid", type=str, default="0.22,0.25,0.28")
    ap.add_argument("--cluster-k-grid", type=str, default="8,16,24,32,40")
    ap.add_argument("--cluster-alpha-grid", type=str, default="0.05,0.08,0.10,0.12,0.15,0.20,0.25")
    ap.add_argument("--cluster-bank-stride-grid", type=str, default="2,4")
    ap.add_argument("--combo-beta-grid", type=str, default="0.02,0.05,0.08,0.10,0.12,0.15,0.20")
    args = ap.parse_args()

    config_path = resolve_path(args.config)
    run_dir = resolve_path(args.run_dir)
    out_dir = resolve_path(args.out_dir)
    assert config_path is not None and run_dir is not None and out_dir is not None
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_yaml(config_path)
    context = prepare_data_context(cfg)
    xva, yva = make_strict_windows(context.norm_data_tc, context.L, context.H, context.t_train, context.t_val)
    xte, yte = context.xte_norm.float().contiguous(), context.yte_norm.float().contiguous()
    xall, yall = make_strict_windows(context.norm_data_tc, context.L, context.H, 0, context.norm_data_tc.shape[0])
    device = torch.device("cpu")

    t0 = perf_counter()
    bundle = load_eval_modules(cfg, run_dir / "best_checkpoint.pt", context.K, device)
    model = bundle["model"]
    cluster_id_c = context.cluster_id_c

    base_val = predict_model(model, xva.float().contiguous(), cluster_id_c, batch_size=int(args.eval_batch_size), device=device).contiguous()
    base_test = predict_model(model, xte, cluster_id_c, batch_size=int(args.eval_batch_size), device=device).contiguous()
    query_val = context.t_train + np.arange(xva.shape[0], dtype=np.int64)
    query_test = context.t_val + np.arange(xte.shape[0], dtype=np.int64)

    rows = []
    method_preds_val: Dict[str, torch.Tensor] = {"base": base_val}
    method_preds_test: Dict[str, torch.Tensor] = {"base": base_test}
    base_metrics = evaluate_method(base_val, base_test, yva.float(), yte)
    rows.append({
        "method": "base",
        "scope": "base",
        "bank_stride": 0,
        "k": 0,
        "alpha": 0.0,
        "beta": 0.0,
        **base_metrics,
    })
    print(f"Base: val_mse={base_metrics['val_mse']:.6f}, test_mse={base_metrics['test_mse']:.6f}")

    # same_channel rolling KNN
    for k in _parse_int_list(args.same_channel_k_grid):
        for alpha in _parse_float_list(args.same_channel_alpha_grid):
            name = f"same_channel_k{k}_a{alpha:.2f}"
            cfg_knn = KNNShapeConfig(
                enable=True,
                mode="rolling",
                scope="same_channel",
                bank_split="history",
                use_for_model_selection=False,
                k=int(k),
                alpha=float(alpha),
                shape_bins=int(args.shape_bins),
                diff_bins=int(args.diff_bins),
                bank_stride=2,
                distance_weight="inverse",
                anchor_mode=str(args.anchor_mode),
                bank_chunk_size=int(args.bank_chunk_size),
            )
            hybrid = ShapeKNNHybrid.fit(xall.float(), yall.float(), cluster_id_c, cfg_knn)
            pred_val = hybrid.hybridize_batch(xva.float(), base_val, cluster_id_c, query_start_abs_b=query_val)
            pred_test = hybrid.hybridize_batch(xte, base_test, cluster_id_c, query_start_abs_b=query_test)
            method_preds_val[name] = pred_val
            method_preds_test[name] = pred_test
            metrics = evaluate_method(pred_val, pred_test, yva.float(), yte)
            rows.append({
                "method": "same_channel",
                "scope": "same_channel",
                "bank_stride": 2,
                "k": int(k),
                "alpha": float(alpha),
                "beta": 0.0,
                **metrics,
            })
            print(f"{name}: val_mse={metrics['val_mse']:.6f}, test_mse={metrics['test_mse']:.6f}")

    # same_cluster rolling KNN
    for k in _parse_int_list(args.same_cluster_k_grid):
        for alpha in _parse_float_list(args.same_cluster_alpha_grid):
            name = f"same_cluster_k{k}_a{alpha:.2f}"
            cfg_knn = KNNShapeConfig(
                enable=True,
                mode="rolling",
                scope="same_cluster",
                bank_split="history",
                use_for_model_selection=False,
                k=int(k),
                alpha=float(alpha),
                shape_bins=int(args.shape_bins),
                diff_bins=int(args.diff_bins),
                bank_stride=2,
                distance_weight="inverse",
                anchor_mode=str(args.anchor_mode),
                bank_chunk_size=int(args.bank_chunk_size),
            )
            hybrid = ShapeKNNHybrid.fit(xall.float(), yall.float(), cluster_id_c, cfg_knn)
            pred_val = hybrid.hybridize_batch(xva.float(), base_val, cluster_id_c, query_start_abs_b=query_val)
            pred_test = hybrid.hybridize_batch(xte, base_test, cluster_id_c, query_start_abs_b=query_test)
            method_preds_val[name] = pred_val
            method_preds_test[name] = pred_test
            metrics = evaluate_method(pred_val, pred_test, yva.float(), yte)
            rows.append({
                "method": "same_cluster",
                "scope": "same_cluster",
                "bank_stride": 2,
                "k": int(k),
                "alpha": float(alpha),
                "beta": 0.0,
                **metrics,
            })
            print(f"{name}: val_mse={metrics['val_mse']:.6f}, test_mse={metrics['test_mse']:.6f}")

    # cluster recall
    cluster_hist_val = {
        int(key): xva.index_select(1, (cluster_id_c == key).nonzero(as_tuple=False).view(-1)).mean(dim=1).contiguous()
        for key in torch.unique(cluster_id_c, sorted=True).tolist()
    }
    cluster_hist_test = {
        int(key): xte.index_select(1, (cluster_id_c == key).nonzero(as_tuple=False).view(-1)).mean(dim=1).contiguous()
        for key in torch.unique(cluster_id_c, sorted=True).tolist()
    }
    for bank_stride in _parse_int_list(args.cluster_bank_stride_grid):
        banks = build_cluster_recall_banks(
            xall_ncl=xall.float(),
            yall_nch=yall.float(),
            cluster_id_c=cluster_id_c,
            shape_bins=int(args.shape_bins),
            diff_bins=int(args.diff_bins),
            bank_stride=int(bank_stride),
            anchor_mode=str(args.anchor_mode),
        )
        for k in _parse_int_list(args.cluster_k_grid):
            for alpha in _parse_float_list(args.cluster_alpha_grid):
                name = f"cluster_recall_s{bank_stride}_k{k}_a{alpha:.2f}"
                tpl_val = predict_cluster_templates(
                    query_cluster_hist_qkl=cluster_hist_val,
                    query_start_abs_q=query_val,
                    banks=banks,
                    shape_bins=int(args.shape_bins),
                    diff_bins=int(args.diff_bins),
                    anchor_mode=str(args.anchor_mode),
                    k=int(k),
                    query_batch_size=int(args.query_batch_size),
                    bank_chunk_size=int(args.bank_chunk_size),
                    pred_len=int(context.H),
                )
                tpl_test = predict_cluster_templates(
                    query_cluster_hist_qkl=cluster_hist_test,
                    query_start_abs_q=query_test,
                    banks=banks,
                    shape_bins=int(args.shape_bins),
                    diff_bins=int(args.diff_bins),
                    anchor_mode=str(args.anchor_mode),
                    k=int(k),
                    query_batch_size=int(args.query_batch_size),
                    bank_chunk_size=int(args.bank_chunk_size),
                    pred_len=int(context.H),
                )
                pred_val = apply_cluster_templates(xva.float(), base_val, cluster_id_c, tpl_val, alpha=float(alpha), anchor_mode=str(args.anchor_mode))
                pred_test = apply_cluster_templates(xte, base_test, cluster_id_c, tpl_test, alpha=float(alpha), anchor_mode=str(args.anchor_mode))
                method_preds_val[name] = pred_val
                method_preds_test[name] = pred_test
                metrics = evaluate_method(pred_val, pred_test, yva.float(), yte)
                rows.append({
                    "method": "cluster_recall",
                    "scope": "cluster_recall",
                    "bank_stride": int(bank_stride),
                    "k": int(k),
                    "alpha": float(alpha),
                    "beta": 0.0,
                    **metrics,
                })
                print(f"{name}: val_mse={metrics['val_mse']:.6f}, test_mse={metrics['test_mse']:.6f}")

    results_df = pd.DataFrame(rows).sort_values(["val_mse", "test_mse", "method"]).reset_index(drop=True)
    same_channel_best = results_df[results_df["method"] == "same_channel"].iloc[0].to_dict()
    same_cluster_best = results_df[results_df["method"] == "same_cluster"].iloc[0].to_dict()
    cluster_recall_best = results_df[results_df["method"] == "cluster_recall"].iloc[0].to_dict()

    # combine best same_channel with top cluster_recall candidates
    combo_rows = []
    channel_name = f"same_channel_k{int(same_channel_best['k'])}_a{float(same_channel_best['alpha']):.2f}"
    top_cluster_names = results_df[results_df["method"] == "cluster_recall"].head(5)
    for _, row in top_cluster_names.iterrows():
        cluster_name = f"cluster_recall_s{int(row['bank_stride'])}_k{int(row['k'])}_a{float(row['alpha']):.2f}"
        for beta in _parse_float_list(args.combo_beta_grid):
            name = f"combo_{channel_name}_{cluster_name}_b{beta:.2f}"
            pred_val = (1.0 - float(beta)) * method_preds_val[channel_name] + float(beta) * method_preds_val[cluster_name]
            pred_test = (1.0 - float(beta)) * method_preds_test[channel_name] + float(beta) * method_preds_test[cluster_name]
            metrics = evaluate_method(pred_val, pred_test, yva.float(), yte)
            combo_rows.append({
                "method": "channel_plus_cluster",
                "scope": "channel_plus_cluster",
                "bank_stride": int(row["bank_stride"]),
                "k": int(row["k"]),
                "alpha": float(row["alpha"]),
                "beta": float(beta),
                "channel_name": channel_name,
                "cluster_name": cluster_name,
                **metrics,
            })
            print(f"{name}: val_mse={metrics['val_mse']:.6f}, test_mse={metrics['test_mse']:.6f}")

    combo_df = pd.DataFrame(combo_rows).sort_values(["val_mse", "test_mse"]).reset_index(drop=True)
    combo_best = None if combo_df.shape[0] == 0 else combo_df.iloc[0].to_dict()

    results_path = out_dir / "results.csv"
    combo_path = out_dir / "combo_results.csv"
    summary_path = out_dir / "summary.json"
    results_df.to_csv(results_path, index=False)
    combo_df.to_csv(combo_path, index=False)

    summary = {
        "config_path": str(config_path),
        "run_dir": str(run_dir),
        "base": base_metrics,
        "best_same_channel": same_channel_best,
        "best_same_cluster": same_cluster_best,
        "best_cluster_recall": cluster_recall_best,
        "best_channel_plus_cluster": combo_best,
        "elapsed_sec": float(perf_counter() - t0),
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Saved results to: {results_path}")
    print(f"Saved combo results to: {combo_path}")
    print(f"Saved summary to: {summary_path}")
    print("Top results:")
    print(results_df.head(12).to_string(index=False))
    if combo_df.shape[0] > 0:
        print("Top combo results:")
        print(combo_df.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
