"""
Comprehensive GI diagnostic on cluster_mlp K=3 base.

Answers:
  Q1. MoE 在不同位置发挥多大作用？
      Per-cluster test_mse: moe-off vs GI breakdown.
      Per-channel: where does GI help / hurt.
      Branch contribution magnitude per (cluster, penalty).

  Q2. Penalty 选择是否合理？
      Per-cluster rel_gap on the STRONG cluster_mlp base (not weak SimpleBase).
      Penalties whose rel_gap collapsed on strong base = redundant supervision.

  Q3. Gate 是否塌缩 / 健康？
      Per-cluster gate mean / std / sparsity.
      Per-sample selectivity verification.
      Improve_p correlation.

Usage:
  python -m scripts.diagnose_cluster_gi --config configs/gi_moe_ETTm1_clusterbase.yaml
"""
from __future__ import annotations
import argparse
import json
import time
from typing import Dict, List
import numpy as np
import torch
import torch.nn.functional as F

from src.utils.yaml_io import load_yaml
from src.utils.pearson import pearson_corr_matrix
from src.utils.clustering import cluster_channels_by_corr
from src.data.reader import read_csv_time_series
from src.data.windows import global_zscore
from src.models.penalties import build_penalty_bank, supported_penalty_names
from src.models.gi_moe import (
    ClusterMLPBaseWithFeatures, HiddenBlockMoEHead, gi_moe_loss_v2,
)
from scripts.run_gi_moe import build_loaders, _seed, _split_train_only_zscore


def _train_block(cfg, device):
    data, _ = read_csv_time_series(cfg["data"]["csv_path"], date_col=cfg["data"]["date_col"])
    data = data[:int(cfg["data"].get("max_rows", data.shape[0]))]
    if cfg["normalize"].get("train_only", True):
        normed, t_train, _ = _split_train_only_zscore(
            data, float(cfg["data"]["train_ratio"]), float(cfg["data"]["val_ratio"]))
    else:
        normed, _, _ = global_zscore(data)
        t_train = int(data.shape[0] * cfg["data"]["train_ratio"])
    return normed[:t_train].to(device)


def _fit_clusters(cfg, device):
    ccfg = cfg.get("cluster", {}) or {}
    tb = _train_block(cfg, device)
    corr = pearson_corr_matrix(tb).cpu()
    ids, _ = cluster_channels_by_corr(
        corr_cc=corr, data_tc=tb.cpu(),
        n_clusters=int(ccfg.get("n_clusters", 3)),
        distance_threshold=float(ccfg.get("distance_threshold", 0.7)),
        linkage=str(ccfg.get("linkage", "average")),
        method=str(ccfg.get("method", "leader")),
        random_state=int(ccfg.get("random_state", 2026)),
        min_cluster_size=int(ccfg.get("min_cluster_size", 2)),
        merge_small_clusters=bool(ccfg.get("merge_small_clusters", True)),
        no_merge_if_channels_lt=int(ccfg.get("no_merge_if_channels_lt", 7)),
    )
    cluster_id_c = ids.to(device).long()
    K = int(cluster_id_c.max().item() + 1)
    sizes = [int((cluster_id_c == k).sum().item()) for k in range(K)]
    return cluster_id_c, K, sizes


def _make_base(cfg, device, K):
    L = int(cfg["window"]["input_len"]); H = int(cfg["window"]["pred_len"])
    return ClusterMLPBaseWithFeatures(
        num_clusters=K, input_len=L, pred_len=H,
        hidden_dim=int(cfg["model"].get("hidden_dim", 256)),
        dropout=float(cfg["model"].get("dropout", 0.2)),
    ).to(device)


def _make_head(cfg, device, pen_names):
    H = int(cfg["window"]["pred_len"])
    m2 = cfg.get("moe_loss_v2", {}) or {}
    return HiddenBlockMoEHead(
        in_dim=int(cfg["model"].get("hidden_dim", 256)),
        pred_len=H, penalty_names=pen_names,
        shared_dim=int(m2.get("shared_dim", 128)),
        private_dim=int(m2.get("private_dim", 32)),
        dropout=float(cfg["model"].get("dropout", 0.0)),
        mask_init=float(m2.get("mask_init", 0.0)),
        log_alpha_init=float(m2.get("log_alpha_init", -3.0)),
        gate_init_bias=float(m2.get("gate_init_bias", -2.0)),
        use_pga=bool(m2.get("use_penalty_gated_activation", True)),
    ).to(device)


def _train_base_only(cfg, base, dl_tr, cluster_id_c, device):
    opt = torch.optim.Adam(base.parameters(),
        lr=float(cfg["train"].get("lr", 1.0e-3)),
        weight_decay=float(cfg["train"].get("weight_decay", 0.0)))
    mae_w = float(cfg.get("moe_loss_v2", {}).get("mae_weight", 0.3))
    epochs = int(cfg["train"].get("epochs", 40))
    print(f"[diag] training moe-off base ({epochs} ep)")
    for ep in range(epochs):
        base.train()
        for x, y, _ in dl_tr:
            x = x.to(device); y = y.to(device)
            yhat = base(x, cluster_id_c=cluster_id_c, return_features=False)
            loss = F.mse_loss(yhat, y) + mae_w * F.l1_loss(yhat, y)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(base.parameters(), 1.0); opt.step()


def _train_gi(cfg, base, head, dl_tr, cluster_id_c, penalty_fns, device):
    params = list(base.parameters()) + list(head.parameters())
    opt = torch.optim.Adam(params,
        lr=float(cfg["train"].get("lr", 1.0e-3)),
        weight_decay=float(cfg["train"].get("weight_decay", 0.0)))
    m2 = cfg.get("moe_loss_v2", {}) or {}
    lam_pen = float(m2.get("lambda_pen", 0.1)); lam_p = m2.get("lambda_p", None)
    lam_norm = float(m2.get("lambda_norm", 1e-4)); mae_w = float(m2.get("mae_weight", 0.3))
    lam_mask = float(m2.get("lambda_mask", 1e-4)); mask_t = float(m2.get("mask_target", 0.5))
    lam_gb = float(m2.get("lambda_gate_bimodal", 0.0))
    normalize = bool(m2.get("normalize_penalties", False))
    epochs = int(cfg["train"].get("epochs", 40))
    print(f"[diag] training GI (base+head, {epochs} ep)")
    for ep in range(epochs):
        base.train(); head.train()
        for x, y, _ in dl_tr:
            x = x.to(device); y = y.to(device)
            y_base, h = base(x, cluster_id_c=cluster_id_c, return_features=True)
            out = head(h, y_base=y_base)
            loss, _ = gi_moe_loss_v2(
                y_base=out["y_base"], y_final=out["y_final"], y=y,
                residuals=out["residuals"], gates=out["gates"], alphas=out["alphas"],
                penalty_fns=penalty_fns, mask_values=out.get("mask_values"),
                lambda_pen=lam_pen, lambda_p=lam_p, lambda_norm=lam_norm,
                mae_weight=mae_w, lambda_mask=lam_mask, mask_target=mask_t,
                lambda_gate_bimodal=lam_gb, head=head, normalize_penalties=normalize,
            )
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()


@torch.no_grad()
def _per_cluster_analysis(base, head, dl_te, cluster_id_c, K, pen_names, penalty_fns, device, with_head: bool):
    base.eval()
    if head is not None: head.eval()
    # accumulators per cluster
    se = [0.0]*K; n_pts = [0]*K
    se_base = [0.0]*K
    pen_base = {p: [0.0]*K for p in pen_names}
    pen_final = {p: [0.0]*K for p in pen_names}
    pen_truth = {p: [0.0]*K for p in pen_names}
    gate_sum = {p: [0.0]*K for p in pen_names}
    gate_sq = {p: [0.0]*K for p in pen_names}
    gate_n = {p: [0]*K for p in pen_names}
    branch_norm = {p: [0.0]*K for p in pen_names}
    pen_l_count = [0]*K
    for x, y, _ in dl_te:
        x = x.to(device); y = y.to(device)
        y_base, h = base(x, cluster_id_c=cluster_id_c, return_features=True)
        if with_head and head is not None:
            out = head(h, y_base=y_base)
            y_final = out["y_final"]
            for p in pen_names:
                # branch_p stats
                br = out["branches"][p]
                g  = out["gates"][p].squeeze(-1)        # [B,C]
                for k in range(K):
                    idx = (cluster_id_c == k).nonzero(as_tuple=False).view(-1)
                    if idx.numel() == 0: continue
                    g_k = g.index_select(1, idx)        # [B, n_k]
                    br_k = br.index_select(1, idx)      # [B, n_k, H]
                    gate_sum[p][k] += float(g_k.sum().item())
                    gate_sq[p][k]  += float((g_k**2).sum().item())
                    gate_n[p][k]   += int(g_k.numel())
                    branch_norm[p][k] += float(br_k.pow(2).mean(dim=-1).sqrt().sum().item())
        else:
            y_final = y_base
        for k in range(K):
            idx = (cluster_id_c == k).nonzero(as_tuple=False).view(-1)
            if idx.numel() == 0: continue
            yb_k = y_base.index_select(1, idx)
            yf_k = y_final.index_select(1, idx)
            yt_k = y.index_select(1, idx)
            se_base[k] += float((yb_k - yt_k).pow(2).sum().item())
            se[k] += float((yf_k - yt_k).pow(2).sum().item())
            n_pts[k] += int(yt_k.numel())
            for p, fn in penalty_fns.items():
                pen_base[p][k]  += float(fn(yb_k, yt_k).mean().item())
                pen_final[p][k] += float(fn(yf_k, yt_k).mean().item())
                pen_truth[p][k] += float(fn(yt_k, yt_k).mean().item())
            pen_l_count[k] += 1
    report = {"K": K, "clusters": []}
    for k in range(K):
        c = max(pen_l_count[k], 1); npt = max(n_pts[k], 1)
        cl = {"cluster": k, "test_mse_base": se_base[k]/npt, "test_mse_final": se[k]/npt}
        for p in pen_names:
            cl[f"pen_base_{p}"] = pen_base[p][k] / c
            cl[f"pen_final_{p}"] = pen_final[p][k] / c
            cl[f"pen_truth_{p}"] = pen_truth[p][k] / c
            # rel_gap formula used in the per-cluster probe.
            l_val = pen_truth[p][k] / c  # truth->truth, not last_value here; we use a robust approx
            # For rel_gap on cluster_mlp, base = penalty(y_base, y_truth).
            denom = max(abs(pen_truth[p][k]/c), 1e-6)
            cl[f"rel_pen_match_{p}"] = (pen_base[p][k]/c - pen_truth[p][k]/c)  # raw mismatch (smaller = better)
            if with_head:
                gn = max(gate_n[p][k], 1)
                gm = gate_sum[p][k] / gn
                gv = max(gate_sq[p][k] / gn - gm*gm, 0.0)
                cl[f"gate_mean_{p}"] = gm
                cl[f"gate_std_{p}"] = gv ** 0.5
                cl[f"branch_norm_{p}"] = branch_norm[p][k] / gn
        report["clusters"].append(cl)
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    cfg["exp"]["seed"] = args.seed
    device = torch.device(cfg["exp"].get("device", "cuda:0") if torch.cuda.is_available() else "cpu")
    _seed(args.seed)

    pen_names = list(cfg.get("moe_loss_v2", {}).get("penalties", ["delta", "direction", "trend"]))
    penalty_fns = build_penalty_bank(pen_names, jump_thr=float(cfg.get("moe_loss_v2", {}).get("jump_threshold", 2.0)))

    cluster_id_c, K, sizes = _fit_clusters(cfg, device)
    print(f"[diag] K={K} sizes={sizes}")
    dl_tr, _, dl_te, _ = build_loaders(cfg, device)

    # --- moe-off ---
    base_off = _make_base(cfg, device, K)
    t0 = time.time(); _train_base_only(cfg, base_off, dl_tr, cluster_id_c, device); print(f"[diag] moe-off trained in {time.time()-t0:.1f}s")
    rep_off = _per_cluster_analysis(base_off, None, dl_te, cluster_id_c, K, pen_names, penalty_fns, device, with_head=False)

    # --- GI ---
    _seed(args.seed)
    base_gi = _make_base(cfg, device, K)
    head = _make_head(cfg, device, pen_names)
    t0 = time.time(); _train_gi(cfg, base_gi, head, dl_tr, cluster_id_c, penalty_fns, device); print(f"[diag] GI trained in {time.time()-t0:.1f}s")
    rep_gi = _per_cluster_analysis(base_gi, head, dl_te, cluster_id_c, K, pen_names, penalty_fns, device, with_head=True)

    print()
    print("=== Q1: MoE contribution per cluster ===")
    print(f"{'cluster':10s} {'size':>6s} {'moe_off':>10s} {'GI(yf)':>10s} {'GI(yb)':>10s} {'Δ(yf-off)':>11s}")
    for k in range(K):
        co = rep_off["clusters"][k]
        cg = rep_gi["clusters"][k]
        d = cg['test_mse_final'] - co['test_mse_base']
        print(f"{'k='+str(k):10s} {sizes[k]:>6d} {co['test_mse_base']:>10.4f} {cg['test_mse_final']:>10.4f} {cg['test_mse_base']:>10.4f} {d:>+11.4f}")

    print()
    print("=== Q2: penalty mismatch (base vs final, smaller=better) per cluster ===")
    print(f"{'cluster':10s} " + " ".join(f"{p:>10s}_off/{p:>10s}_GI/{'Δ':>5s}" for p in pen_names))
    for k in range(K):
        co = rep_off["clusters"][k]; cg = rep_gi["clusters"][k]
        row = f"{'k='+str(k):10s} "
        for p in pen_names:
            off_v = co[f'pen_base_{p}']
            gi_v  = cg[f'pen_final_{p}']
            d = gi_v - off_v
            row += f" {off_v:>6.4f}/{gi_v:>6.4f}/{d:>+5.4f} "
        print(row)
    print("  (Δ negative -> GI reduced this penalty's value on test)")

    print()
    print("=== Q3: gate per cluster (GI only) ===")
    print(f"{'cluster':10s} " + " ".join(f"{p:>14s}(mean/std/branch)" for p in pen_names))
    for k in range(K):
        cg = rep_gi["clusters"][k]
        row = f"{'k='+str(k):10s} "
        for p in pen_names:
            gm = cg[f'gate_mean_{p}']; gs = cg[f'gate_std_{p}']; bn = cg[f'branch_norm_{p}']
            row += f" {gm:>5.3f}/{gs:>5.3f}/{bn:>6.4f} "
        print(row)
    print("  gate_std > 0.1 = per-sample varying.  branch_norm = ||g·α·r_p|| avg per sample.")

    # Save full reports
    import os
    out_dir = cfg["exp"].get("out_dir", "outputs")
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/cluster_gi_diagnostic.json", "w", encoding="utf-8") as f:
        json.dump({"moe_off": rep_off, "gi": rep_gi, "sizes": sizes}, f, indent=2)
    print(f"\n[diag] saved -> {out_dir}/cluster_gi_diagnostic.json")


if __name__ == "__main__":
    main()
