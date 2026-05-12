"""
Per-cluster penalty effectivity probe.

Q: for each cluster, which penalty's structural failure mode is base_only
   actually committing? This tells us if "penalty doesn't work" is global, or
   if it's "we picked the wrong penalty per cluster".

Pipeline:
  1. Fit leader clustering on train-only data -> cluster_id_c [C].
  2. Train base_only (SimpleBasePredictor) for 40 epochs on full data.
  3. On test set, compute penalty(y_base, y) and penalty(y_last, y) PER CLUSTER,
     for ALL supported penalties (not just the v2 default 4).
  4. Report per-cluster rel_gap and recommend top 1-2 penalties per cluster.

Output: outputs/<exp>/penalty_per_cluster.json + console table.
"""
from __future__ import annotations
import argparse
import json
import os
import time
from typing import Dict, List

import torch
import torch.nn.functional as F

from src.utils.yaml_io import load_yaml
from src.utils.pearson import pearson_corr_matrix
from src.utils.clustering import cluster_channels_by_corr
from src.models.penalties import supported_penalty_names, build_penalty_bank
from src.models.gi_moe import SimpleBasePredictor
from src.data.reader import read_csv_time_series
from src.data.windows import global_zscore
from scripts.run_gi_moe import build_loaders, _seed, _split_train_only_zscore


def _train_block_for_clustering(cfg, device) -> torch.Tensor:
    data, _ = read_csv_time_series(cfg["data"]["csv_path"], date_col=cfg["data"]["date_col"])
    max_rows = int(cfg["data"].get("max_rows", data.shape[0]))
    data = data[:max_rows]
    if cfg["normalize"].get("train_only", True):
        normed, t_train, _ = _split_train_only_zscore(
            data, float(cfg["data"]["train_ratio"]), float(cfg["data"]["val_ratio"]))
    else:
        normed, _, _ = global_zscore(data)
        t_train = int(data.shape[0] * cfg["data"]["train_ratio"])
    return normed[:t_train].to(device)


def _train_base(cfg, device) -> SimpleBasePredictor:
    L = int(cfg["window"]["input_len"]); H = int(cfg["window"]["pred_len"])
    base = SimpleBasePredictor(
        input_len=L, pred_len=H,
        hidden_dim=int(cfg["model"].get("hidden_dim", 256)),
        dropout=float(cfg["model"].get("dropout", 0.2)),
    ).to(device)
    dl_tr, dl_va, dl_te, _ = build_loaders(cfg, device)
    opt = torch.optim.Adam(
        base.parameters(),
        lr=float(cfg["train"].get("lr", 1.0e-3)),
        weight_decay=float(cfg["train"].get("weight_decay", 0.0)),
    )
    mae_weight = float(cfg.get("moe_loss", {}).get("mae_weight", 0.3))
    epochs = int(cfg["train"].get("epochs", 40))
    print(f"[probe] training base for {epochs} epochs on {device} ...")
    t0 = time.time()
    for ep in range(1, epochs + 1):
        base.train()
        for x, y, _ in dl_tr:
            x = x.to(device); y = y.to(device)
            yhat = base(x)
            loss = F.mse_loss(yhat, y) + mae_weight * F.l1_loss(yhat, y)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(base.parameters(), 1.0); opt.step()
    print(f"[probe] base trained in {time.time()-t0:.1f}s")
    return base, dl_te


def _fit_clusters(cfg, device, num_channels: int):
    ccfg = cfg.get("cluster", {}) or {"method": "leader", "n_clusters": 3,
                                      "distance_threshold": 0.7,
                                      "linkage": "average",
                                      "min_cluster_size": 2,
                                      "merge_small_clusters": True,
                                      "no_merge_if_channels_lt": 7,
                                      "random_state": 2026}
    train_block = _train_block_for_clustering(cfg, device)
    corr_cc = pearson_corr_matrix(train_block)
    cluster_ids_c, _ = cluster_channels_by_corr(
        corr_cc=corr_cc.cpu(),
        data_tc=train_block.cpu(),
        n_clusters=int(ccfg.get("n_clusters", 3)),
        distance_threshold=float(ccfg.get("distance_threshold", 0.7)),
        linkage=str(ccfg.get("linkage", "average")),
        method=str(ccfg.get("method", "leader")),
        kmeans_n_init=int(ccfg.get("kmeans_n_init", 10)),
        kmeans_max_iter=int(ccfg.get("kmeans_max_iter", 300)),
        random_state=int(ccfg.get("random_state", 2026)),
        min_cluster_size=int(ccfg.get("min_cluster_size", 2)),
        merge_small_clusters=bool(ccfg.get("merge_small_clusters", True)),
        no_merge_if_channels_lt=int(ccfg.get("no_merge_if_channels_lt", 7)),
    )
    cluster_id_c = cluster_ids_c.to(device).long()
    K = int(cluster_id_c.max().item()) + 1
    sizes = [int((cluster_id_c == k).sum().item()) for k in range(K)]
    print(f"[probe] K={K} sizes={sizes}")
    return cluster_id_c, K, sizes


@torch.no_grad()
def _per_cluster_per_penalty(base, loader, penalty_fns, cluster_id_c, K, device):
    base.eval()
    # accumulate per (cluster, penalty): sum and count of mean-over-H values per (b, c).
    sums_b: Dict[str, List[float]] = {p: [0.0] * K for p in penalty_fns}
    sums_l: Dict[str, List[float]] = {p: [0.0] * K for p in penalty_fns}
    sums_t: Dict[str, List[float]] = {p: [0.0] * K for p in penalty_fns}
    counts: List[int] = [0] * K
    se_b = [0.0] * K
    n_pts = [0] * K
    for x, y, _ in loader:
        x = x.to(device); y = y.to(device)
        y_base = base(x)
        y_last = x[..., -1:].expand_as(y)
        # per-cluster channel-grouped means
        for k in range(K):
            idx = (cluster_id_c == k).nonzero(as_tuple=False).view(-1)
            if idx.numel() == 0:
                continue
            yb_k = y_base.index_select(1, idx)
            yl_k = y_last.index_select(1, idx)
            yt_k = y.index_select(1, idx)
            for p, fn in penalty_fns.items():
                sums_b[p][k] += float(fn(yb_k, yt_k).mean().item())
                sums_l[p][k] += float(fn(yl_k, yt_k).mean().item())
                sums_t[p][k] += float(fn(yt_k, yt_k).mean().item())
            counts[k] += 1
            se_b[k] += float(((yb_k - yt_k) ** 2).sum().item())
            n_pts[k] += int(yt_k.numel())
    report: Dict[str, dict] = {"_meta": {"cluster_sizes": [int((cluster_id_c == k).sum().item()) for k in range(K)]}}
    for k in range(K):
        c = max(counts[k], 1)
        cluster_report = {"base_test_mse": se_b[k] / max(n_pts[k], 1)}
        per_pen = {}
        for p in penalty_fns:
            b = sums_b[p][k] / c; l = sums_l[p][k] / c; t = sums_t[p][k] / c
            # rel_gap: positive = base far from "perfect score" (truth-as-truth = t) compared to trivial baseline gap.
            # If baseline penalizes more than base does (normal case where last is bad), gap = (b - t) / (l - t).
            # Guard: avoid div-by-zero when |l - t| is tiny.
            denom = (l - t)
            if abs(denom) < 1.0e-6:
                rel = float("nan")
            else:
                rel = (b - t) / denom
            per_pen[p] = {"base": b, "last_value": l, "truth": t, "rel_gap": rel}
        cluster_report["penalties"] = per_pen
        report[f"cluster_{k}"] = cluster_report
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--top-k", type=int, default=2, help="how many penalties to recommend per cluster")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    if args.seed is not None:
        cfg["exp"]["seed"] = int(args.seed)
    device = torch.device(cfg["exp"].get("device", "cuda:0") if torch.cuda.is_available() else "cpu")
    _seed(int(cfg["exp"].get("seed", 2026)))

    # Build penalty bank using ALL supported penalties (so we can compare).
    all_pens = list(supported_penalty_names())
    # Drop penalties whose truth-as-truth value is nonzero (one-sided / y_hat-only),
    # they are misaligned with our "rel_gap toward 0" framework. Keep all here;
    # the rel_gap output will tell the user which are well-defined.
    penalty_fns = build_penalty_bank(all_pens, jump_thr=float(cfg.get("moe_loss", {}).get("jump_threshold", 2.0)))

    _, _, _, num_channels = build_loaders(cfg, device)
    cluster_id_c, K, sizes = _fit_clusters(cfg, device, num_channels)
    base, dl_te = _train_base(cfg, device)
    report = _per_cluster_per_penalty(base, dl_te, penalty_fns, cluster_id_c, K, device)
    report["_meta"]["all_penalties"] = all_pens

    # Print compact table per cluster and per-penalty rel_gap.
    print(f"\n[probe] === per-cluster penalty effectivity ===")
    print(f"  config: {args.config}, K={K}, sizes={sizes}")
    print()
    header = f"  {'penalty':12s} " + " ".join(f"{'k='+str(k):>10s}" for k in range(K))
    print(header)
    print(f"  {'(rel_gap)':12s} " + " ".join(f"{'sz='+str(sizes[k]):>10s}" for k in range(K)))
    for p in all_pens:
        row = f"  {p:12s} "
        for k in range(K):
            v = report[f"cluster_{k}"]["penalties"][p]["rel_gap"]
            row += f"{v:>10.3f} " if not (v != v) else "      nan  "  # nan check
        print(row)
    print()
    print(f"  base test_mse per cluster: " + " ".join(
        f"k{k}={report[f'cluster_{k}']['base_test_mse']:.3f}" for k in range(K)))
    print()

    # Recommend top-k per cluster (highest positive rel_gap, ignoring NaN/negative).
    print(f"[probe] === recommended top-{args.top_k} penalties per cluster ===")
    recommendations: Dict[str, List[str]] = {}
    for k in range(K):
        pens = report[f"cluster_{k}"]["penalties"]
        scored = []
        for name, vals in pens.items():
            gap = vals["rel_gap"]
            if gap != gap:  # NaN
                continue
            # we want positive gap (base worse than truth-as-truth direction).
            # ignore strongly negative (penalty fights truth).
            if gap > 0:
                scored.append((name, gap))
        scored.sort(key=lambda t: -t[1])
        top = [name for name, _ in scored[:args.top_k]]
        recommendations[f"cluster_{k}"] = top
        scores = ", ".join(f"{n}({s:.2f})" for n, s in scored[:args.top_k])
        print(f"  cluster {k} (size={sizes[k]}): {scores}")
    report["_meta"]["recommendations_top_k"] = recommendations

    out_path = os.path.join(cfg["exp"].get("out_dir", "outputs"), "penalty_per_cluster.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n[probe] saved -> {out_path}")


if __name__ == "__main__":
    main()
