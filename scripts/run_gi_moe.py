"""
Standalone runner for GI-MoE Loss (v1 Adapter + v2 Hidden-Block).

Modes (`--mode`):
  A. base                              -- base predictor only
  B. hidden_block_mse_only             -- v2 head, MSE/MAE only
  C. hidden_block_ordinary_penalty     -- v2 head, MSE + penalty(y_final, y)
  D. hidden_block_gi_moe_loss          -- v2 head, masked-visibility GI loss
  E. adapter_gi_moe_loss               -- v1 adapter bank, masked-visibility GI loss
  + adapter_mse_only / adapter_ordinary_penalty (v1 baselines)

No cluster, no KNN, no dynamic prototype, no neuron-mask gradient hook.
"""
from __future__ import annotations
import argparse
import json
import os
import time
from typing import Dict, List

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.utils.yaml_io import load_yaml
from src.utils.seed import set_seed
from src.data.reader import read_csv_time_series
from src.data.windows import global_zscore, make_strict_windows, WindowTensorDataset
from src.models.penalties import build_penalty_bank
from src.models.gi_moe import (
    SimpleBasePredictor,
    ClusterMLPBaseWithFeatures,
    PenaltyAdapterBank,
    HiddenBlockMoEHead,
    gi_moe_loss,
    gi_moe_loss_v2,
    loss_mse_only,
    loss_ordinary_penalty,
    verify_gi_moe_grad_isolation,
    verify_gi_moe_v2_grad_isolation,
)
from src.utils.pearson import pearson_corr_matrix
from src.utils.clustering import cluster_channels_by_corr


# -------------------------------------------------------------------------
def _seed(seed: int) -> None:
    try:
        set_seed(int(seed), deterministic=False)
    except Exception:
        torch.manual_seed(int(seed))


def _split_train_only_zscore(data_tc: torch.Tensor, train_ratio: float, val_ratio: float):
    T = data_tc.shape[0]
    t_train = int(T * train_ratio)
    t_val = int(T * (train_ratio + val_ratio))
    train_block = data_tc[:t_train]
    mean = train_block.mean(dim=0, keepdim=True)
    std = train_block.std(dim=0, keepdim=True).clamp_min(1.0e-6)
    return (data_tc - mean) / std, t_train, t_val


def build_loaders(cfg, device):
    data, _ = read_csv_time_series(cfg["data"]["csv_path"], date_col=cfg["data"]["date_col"])
    max_rows = int(cfg["data"].get("max_rows", data.shape[0]))
    data = data[:max_rows]
    if cfg["normalize"].get("train_only", True):
        normed, t_train, t_val = _split_train_only_zscore(
            data, float(cfg["data"]["train_ratio"]), float(cfg["data"]["val_ratio"]))
    else:
        normed, _, _ = global_zscore(data)
        T = data.shape[0]
        t_train = int(T * cfg["data"]["train_ratio"])
        t_val = int(T * (cfg["data"]["train_ratio"] + cfg["data"]["val_ratio"]))
    L = int(cfg["window"]["input_len"])
    H = int(cfg["window"]["pred_len"])
    T = normed.shape[0]
    x_tr, y_tr = make_strict_windows(normed, L, H, 0, t_train)
    x_va, y_va = make_strict_windows(normed, L, H, t_train, t_val)
    x_te, y_te = make_strict_windows(normed, L, H, t_val, T)
    bs = int(cfg["train"]["batch_size"])
    nw = int(cfg["train"].get("num_workers", 0))
    return (
        DataLoader(WindowTensorDataset(x_tr, y_tr), batch_size=bs, shuffle=True, num_workers=nw),
        DataLoader(WindowTensorDataset(x_va, y_va), batch_size=bs, shuffle=False, num_workers=nw),
        DataLoader(WindowTensorDataset(x_te, y_te), batch_size=bs, shuffle=False, num_workers=nw),
        normed.shape[1],
    )


# -------------------------------------------------------------------------
def _call_base(base, x, *, cluster_id_c=None, return_features=False):
    """Uniform call to base. ClusterMLPBaseWithFeatures takes cluster_id_c;
    SimpleBasePredictor ignores it."""
    if isinstance(base, ClusterMLPBaseWithFeatures):
        return base(x, cluster_id_c=cluster_id_c, return_features=return_features)
    return base(x, return_features=return_features)


def _forward(mode: str, base, bank_v1, head_v2, x, cluster_id_c=None):
    if mode == "base":
        y_base = _call_base(base, x, cluster_id_c=cluster_id_c, return_features=False)
        return {"y_base": y_base, "y_final": y_base}
    if mode.startswith("adapter_"):
        y_base, h = _call_base(base, x, cluster_id_c=cluster_id_c, return_features=True)
        residuals, gates = bank_v1(h, y_base)
        y_final = bank_v1.mix(y_base, residuals, gates)
        return {"y_base": y_base, "y_final": y_final, "residuals": residuals, "gates": gates}
    if mode.startswith("hidden_block_"):
        # B1 fix: y_base from base.decode (strong path).
        y_base, h = _call_base(base, x, cluster_id_c=cluster_id_c, return_features=True)
        out = head_v2(h, y_base=y_base)
        return out
    raise ValueError(f"unknown mode: {mode}")


def _eval(mode, base, bank_v1, head_v2, loader, device, cluster_id_c=None):
    base.eval()
    if bank_v1 is not None:
        bank_v1.eval()
    if head_v2 is not None:
        head_v2.eval()
    se_f = ae_f = se_b = ae_b = n = 0.0
    with torch.no_grad():
        for x, y, _ in loader:
            x = x.to(device); y = y.to(device)
            fwd = _forward(mode, base, bank_v1, head_v2, x, cluster_id_c=cluster_id_c)
            y_final = fwd["y_final"]; y_base = fwd["y_base"]
            n_b = float(y.numel())
            se_f += float(((y_final - y) ** 2).sum().item()); ae_f += float((y_final - y).abs().sum().item())
            se_b += float(((y_base - y) ** 2).sum().item());  ae_b += float((y_base  - y).abs().sum().item())
            n += n_b
    return {"mse": se_f / max(n, 1.0), "mae": ae_f / max(n, 1.0),
            "y_base_mse": se_b / max(n, 1.0), "y_base_mae": ae_b / max(n, 1.0)}


# -------------------------------------------------------------------------
def run(cfg, mode_override: str = "", verify_grad: bool = False):
    device = torch.device(cfg["exp"].get("device", "cuda:0") if torch.cuda.is_available() else "cpu")
    _seed(int(cfg["exp"].get("seed", 2026)))

    dl_tr, dl_va, dl_te, num_channels = build_loaders(cfg, device)
    L = int(cfg["window"]["input_len"])
    H = int(cfg["window"]["pred_len"])
    hidden_dim = int(cfg["model"].get("hidden_dim", 256))
    predictor_kind = str(cfg["model"].get("predictor", "simple")).lower()
    dropout = float(cfg["model"].get("dropout", 0.2))

    # Build cluster_id_c if needed (for cluster_mlp predictor).
    cluster_id_c = None
    if predictor_kind in {"cluster_mlp", "cluster", "k3"}:
        # Fit leader clustering on train-only block.
        ccfg = cfg.get("cluster", {}) or {}
        # Reload train block for clustering.
        data, _ = read_csv_time_series(cfg["data"]["csv_path"], date_col=cfg["data"]["date_col"])
        max_rows = int(cfg["data"].get("max_rows", data.shape[0])); data = data[:max_rows]
        if cfg["normalize"].get("train_only", True):
            normed, t_train, _ = _split_train_only_zscore(
                data, float(cfg["data"]["train_ratio"]), float(cfg["data"]["val_ratio"]))
        else:
            normed, _, _ = global_zscore(data)
            t_train = int(data.shape[0] * cfg["data"]["train_ratio"])
        train_block = normed[:t_train].to(device)
        corr_cc = pearson_corr_matrix(train_block)
        cluster_ids, _ = cluster_channels_by_corr(
            corr_cc=corr_cc.cpu(), data_tc=train_block.cpu(),
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
        cluster_id_c = cluster_ids.to(device).long()
        K_eff = int(cluster_id_c.max().item() + 1)
        sizes = [int((cluster_id_c == k).sum().item()) for k in range(K_eff)]
        print(f"[cluster] K={K_eff} sizes={sizes}")
        base = ClusterMLPBaseWithFeatures(
            num_clusters=K_eff, input_len=L, pred_len=H,
            hidden_dim=hidden_dim, dropout=dropout,
        ).to(device)
    else:
        base = SimpleBasePredictor(
            input_len=L, pred_len=H, hidden_dim=hidden_dim, dropout=dropout,
        ).to(device)

    mode = (mode_override or cfg.get("moe_loss", {}).get("mode", "adapter_gi_moe_loss")).strip()

    # --- v1 adapter bank (built lazily if mode needs it) ---
    mcfg = cfg.get("moe_loss", {}) or {}
    pen_names: List[str] = list(mcfg.get("penalties", ["amp_under", "delta", "jitter", "smooth"]))
    penalty_fns = build_penalty_bank(pen_names, jump_thr=float(mcfg.get("jump_threshold", 2.0)))
    bank_v1 = PenaltyAdapterBank(
        penalty_names=pen_names, hidden_dim=hidden_dim, output_dim=H,
        adapter_dim=int(mcfg.get("adapter_dim", 32)),
        gate_init_bias=float(mcfg.get("gate_init_bias", -2.0)),
    ).to(device)

    # --- v2 hidden-block head ---
    mcfg2 = cfg.get("moe_loss_v2", {}) or {}
    head_v2 = HiddenBlockMoEHead(
        in_dim=hidden_dim, pred_len=H,
        penalty_names=list(mcfg2.get("penalties", pen_names)),
        shared_dim=int(mcfg2.get("shared_dim", 128)),
        private_dim=int(mcfg2.get("private_dim", 32)),
        dropout=float(cfg["model"].get("dropout", 0.0)),
        mask_init=float(mcfg2.get("mask_init", 0.0)),
        log_alpha_init=float(mcfg2.get("log_alpha_init", -3.0)),
        gate_init_bias=float(mcfg2.get("gate_init_bias", -2.0)),
        use_pga=bool(mcfg2.get("use_penalty_gated_activation", True)),
    ).to(device)

    # Verify-grad path: choose v1 or v2 based on mode (default: both)
    if verify_grad:
        x, y, _ = next(iter(dl_tr))
        x = x.to(device); y = y.to(device)
        print("=== verifying v1 adapter gradient isolation ===")
        verify_gi_moe_grad_isolation(
            base_model=base, bank=bank_v1, x=x, y=y,
            penalty_fns=penalty_fns, target_penalty=pen_names[0], verbose=True,
        )
        print("=== verifying v2 hidden-block gradient isolation ===")
        verify_gi_moe_v2_grad_isolation(
            base_model=base, head=head_v2, x=x, y=y,
            penalty_fns=penalty_fns, target_penalty=pen_names[0], verbose=True,
        )
        print("[verify] PASS")
        return {"verify": True}

    # --- Optimizer: include only modules used by this mode (cleaner) ---
    params: List[nn.Parameter] = list(base.parameters())
    if mode.startswith("adapter_"):
        params += list(bank_v1.parameters())
    if mode.startswith("hidden_block_"):
        params += list(head_v2.parameters())
    opt = torch.optim.Adam(
        params,
        lr=float(cfg["train"].get("lr", 1.0e-3)),
        weight_decay=float(cfg["train"].get("weight_decay", 0.0)),
    )
    epochs = int(cfg["train"].get("epochs", 40))

    out_dir = cfg["exp"].get("out_dir", "outputs/gi_moe")
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, f"log_{mode}.jsonl")
    print(f"[run] mode={mode} penalties={pen_names} epochs={epochs} device={device} -> {log_path}")

    lambda_pen = float(mcfg.get("lambda_pen", 0.1))
    lambda_p_v1 = mcfg.get("lambda_p", None)
    lambda_norm_v1 = float(mcfg.get("lambda_norm", 1.0e-4))
    mae_weight_v1 = float(mcfg.get("mae_weight", 0.3))

    lambda_pen_v2 = float(mcfg2.get("lambda_pen", 0.1))
    lambda_p_v2 = mcfg2.get("lambda_p", None)
    lambda_norm_v2 = float(mcfg2.get("lambda_norm", 1.0e-4))
    lambda_mask_v2 = float(mcfg2.get("lambda_mask", 1.0e-4))
    mask_target_v2 = float(mcfg2.get("mask_target", 0.5))
    mae_weight_v2 = float(mcfg2.get("mae_weight", 0.3))
    lambda_gate_v2 = float(mcfg2.get("lambda_gate", 0.5))
    bce_tau_v2 = float(mcfg2.get("bce_tau", 0.01))
    detach_gate_v2 = bool(mcfg2.get("detach_gate_from_main", False))
    lambda_gate_bimodal_v2 = float(mcfg2.get("lambda_gate_bimodal", 0.0))
    normalize_penalties_v2 = bool(mcfg2.get("normalize_penalties", False))

    best_val = float("inf")
    history: List[dict] = []
    t0 = time.time()
    for ep in range(1, epochs + 1):
        base.train(); bank_v1.train(); head_v2.train()
        train_se = train_ae = n_seen = 0.0
        pen_acc = {p: 0.0 for p in pen_names}
        gate_mean_acc = {p: 0.0 for p in pen_names}
        gate_std_acc = {p: 0.0 for p in pen_names}
        res_norm_acc = {p: 0.0 for p in pen_names}
        branch_norm_acc = {p: 0.0 for p in pen_names}
        mask_mean_acc = {p: 0.0 for p in pen_names}
        alpha_acc = {p: 0.0 for p in pen_names}
        ratio_acc = 0.0; nb = 0; pen_total_acc = 0.0
        y_base_mse_acc = y_final_mse_acc = 0.0

        for x, y, _ in dl_tr:
            x = x.to(device); y = y.to(device)
            fwd = _forward(mode, base, bank_v1, head_v2, x, cluster_id_c=cluster_id_c)
            y_base = fwd["y_base"]; y_final = fwd["y_final"]

            if mode == "base":
                loss, info = loss_mse_only(y_final, y, mae_weight=mae_weight_v1)
            elif mode == "adapter_mse_only":
                loss, info = loss_mse_only(y_final, y, mae_weight=mae_weight_v1)
            elif mode == "adapter_ordinary_penalty":
                loss, info = loss_ordinary_penalty(
                    y_final, y, penalty_fns,
                    lambda_pen=lambda_pen, lambda_p=lambda_p_v1, mae_weight=mae_weight_v1)
            elif mode == "adapter_gi_moe_loss":
                loss, info = gi_moe_loss(
                    y_base=y_base, y_final=y_final, y=y,
                    residuals=fwd["residuals"], gates=fwd["gates"], penalty_fns=penalty_fns,
                    lambda_pen=lambda_pen, lambda_p=lambda_p_v1,
                    lambda_norm=lambda_norm_v1, mae_weight=mae_weight_v1)
            elif mode == "hidden_block_mse_only":
                loss, info = loss_mse_only(y_final, y, mae_weight=mae_weight_v2)
            elif mode == "hidden_block_ordinary_penalty":
                loss, info = loss_ordinary_penalty(
                    y_final, y, penalty_fns,
                    lambda_pen=lambda_pen_v2, lambda_p=lambda_p_v2, mae_weight=mae_weight_v2)
            elif mode == "hidden_block_gi_moe_loss":
                loss, info = gi_moe_loss_v2(
                    y_base=y_base, y_final=y_final, y=y,
                    residuals=fwd["residuals"], gates=fwd["gates"], alphas=fwd["alphas"],
                    penalty_fns=penalty_fns, mask_values=fwd.get("mask_values"),
                    lambda_pen=lambda_pen_v2, lambda_p=lambda_p_v2,
                    lambda_norm=lambda_norm_v2, mae_weight=mae_weight_v2,
                    lambda_mask=lambda_mask_v2, mask_target=mask_target_v2,
                    lambda_gate_bimodal=lambda_gate_bimodal_v2,
                    head=head_v2, normalize_penalties=normalize_penalties_v2)
            elif mode == "hidden_block_gi_bce_gate":
                # Per-sample gate supervision via improve_p (decouples gate from MSE-only signal).
                loss, info = gi_moe_loss_v2(
                    y_base=y_base, y_final=y_final, y=y,
                    residuals=fwd["residuals"], gates=fwd["gates"], alphas=fwd["alphas"],
                    penalty_fns=penalty_fns, mask_values=fwd.get("mask_values"),
                    lambda_pen=lambda_pen_v2, lambda_p=lambda_p_v2,
                    lambda_norm=lambda_norm_v2, mae_weight=mae_weight_v2,
                    lambda_mask=lambda_mask_v2, mask_target=mask_target_v2,
                    bce_gate_supervision=True,
                    lambda_gate=lambda_gate_v2, bce_tau=bce_tau_v2,
                    detach_gate_from_main=detach_gate_v2,
                    lambda_gate_bimodal=lambda_gate_bimodal_v2,
                    head=head_v2, normalize_penalties=normalize_penalties_v2)
            else:
                raise ValueError(f"unknown mode: {mode}")

            opt.zero_grad(set_to_none=True)
            loss.backward()
            # Split grad clip: base, bank, head each clipped INDEPENDENTLY so
            # one module's large gradient norm doesn't shrink the others' lr.
            # (Diagnostic on cluster_mlp showed unified clip dragged base lr
            # down when head had large penalty-gradient norm even when its
            # branches were dead → base degraded.)
            torch.nn.utils.clip_grad_norm_(base.parameters(), max_norm=1.0)
            if mode.startswith("adapter_"):
                torch.nn.utils.clip_grad_norm_(bank_v1.parameters(), max_norm=1.0)
            if mode.startswith("hidden_block_"):
                torch.nn.utils.clip_grad_norm_(head_v2.parameters(), max_norm=1.0)
            opt.step()

            with torch.no_grad():
                train_se += float(((y_final - y) ** 2).sum().item())
                train_ae += float((y_final - y).abs().sum().item())
                n_seen += float(y.numel())
                num = (y_final - y_base).pow(2).mean().sqrt()
                den = y_base.pow(2).mean().sqrt().clamp_min(1e-8)
                ratio_acc += float((num / den).item())
                pen_total_acc += float(info.get("L_pen_total", torch.zeros(())).item()) if "L_pen_total" in info else 0.0
                y_base_mse_acc += float(info["y_base_mse"].item()) if "y_base_mse" in info else 0.0
                y_final_mse_acc += float(info["y_final_mse"].item()) if "y_final_mse" in info else 0.0
                if mode.startswith("adapter_") and "residuals" in fwd:
                    for p in pen_names:
                        r = fwd["residuals"][p]; g = fwd["gates"][p]
                        gate_mean_acc[p] += float(g.mean().item())
                        gate_std_acc[p] += float(g.std().item())
                        res_norm_acc[p] += float(r.pow(2).mean().sqrt().item())
                        branch_norm_acc[p] += float((g * r).pow(2).mean().sqrt().item())
                        if f"L_pen_{p}" in info:
                            pen_acc[p] += float(info[f"L_pen_{p}"].item())
                elif mode.startswith("hidden_block_") and "residuals" in fwd:
                    for p in pen_names:
                        r = fwd["residuals"][p]; g = fwd["gates"][p]; a = fwd["alphas"][p]
                        gate_mean_acc[p] += float(g.mean().item())
                        gate_std_acc[p] += float(g.std().item())
                        res_norm_acc[p] += float(r.pow(2).mean().sqrt().item())
                        branch_norm_acc[p] += float((g * a * r).pow(2).mean().sqrt().item())
                        alpha_acc[p] += float(a.item())
                        if fwd.get("mask_values") is not None:
                            mask_mean_acc[p] += float(fwd["mask_values"][p].mean().item())
                        if f"L_pen_{p}" in info:
                            pen_acc[p] += float(info[f"L_pen_{p}"].item())
                nb += 1

        train_mse = train_se / max(n_seen, 1.0)
        train_mae = train_ae / max(n_seen, 1.0)
        val_metrics = _eval(mode, base, bank_v1, head_v2, dl_va, device, cluster_id_c=cluster_id_c)
        test_metrics = _eval(mode, base, bank_v1, head_v2, dl_te, device, cluster_id_c=cluster_id_c)

        rec = {
            "epoch": ep, "mode": mode,
            "train_mse": train_mse, "train_mae": train_mae,
            "val_mse": val_metrics["mse"], "val_mae": val_metrics["mae"],
            "test_mse": test_metrics["mse"], "test_mae": test_metrics["mae"],
            "val_y_base_mse": val_metrics["y_base_mse"], "val_y_base_mae": val_metrics["y_base_mae"],
            "loss_pen_total": pen_total_acc / max(nb, 1),
            "y_base_mse_train": y_base_mse_acc / max(nb, 1),
            "y_final_mse_train": y_final_mse_acc / max(nb, 1),
            "total_residual_ratio": ratio_acc / max(nb, 1),
        }
        for p in pen_names:
            rec[f"loss_pen_{p}"] = pen_acc[p] / max(nb, 1)
            rec[f"gate_mean_{p}"] = gate_mean_acc[p] / max(nb, 1)
            rec[f"gate_std_{p}"] = gate_std_acc[p] / max(nb, 1)
            rec[f"residual_norm_{p}"] = res_norm_acc[p] / max(nb, 1)
            rec[f"branch_norm_{p}"] = branch_norm_acc[p] / max(nb, 1)
            if mode.startswith("hidden_block_"):
                rec[f"alpha_{p}"] = alpha_acc[p] / max(nb, 1)
                rec[f"mask_mean_{p}"] = mask_mean_acc[p] / max(nb, 1)
        history.append(rec)

        if val_metrics["mse"] < best_val:
            best_val = val_metrics["mse"]

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

        extras = ""
        if mode.startswith("hidden_block_"):
            a_summary = " ".join(f"a:{p}:{rec[f'alpha_{p}']:.3f}" for p in pen_names)
            extras = " " + a_summary
        gate_summary = " ".join(f"{p}:{rec[f'gate_mean_{p}']:.2f}" for p in pen_names) if not mode == "base" else ""
        print(f"[ep {ep:03d}/{epochs}] train_mse={train_mse:.4f} val_mse={val_metrics['mse']:.4f} "
              f"test_mse={test_metrics['mse']:.4f} y_base_val={val_metrics['y_base_mse']:.4f} "
              f"ratio={rec['total_residual_ratio']:.3f}"
              + (f" gates[{gate_summary}]" if gate_summary else "") + extras)

    elapsed = time.time() - t0
    summary = {"mode": mode, "epochs": epochs, "best_val_mse": best_val,
               "final": history[-1] if history else None, "elapsed_sec": elapsed}
    with open(os.path.join(out_dir, f"summary_{mode}.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[done] mode={mode} best_val_mse={best_val:.4f} time={elapsed:.1f}s")
    return summary


# -------------------------------------------------------------------------
ABLATION_MODES = [
    "base",
    "adapter_mse_only",
    "adapter_ordinary_penalty",
    "adapter_gi_moe_loss",
    "hidden_block_mse_only",
    "hidden_block_ordinary_penalty",
    "hidden_block_gi_moe_loss",
    "hidden_block_gi_bce_gate",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--mode", default="", help="one of: " + ", ".join(ABLATION_MODES))
    ap.add_argument("--verify-grad", action="store_true")
    ap.add_argument("--ablation-all", action="store_true",
                    help="run all 7 modes sequentially")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    if args.seed is not None:
        cfg["exp"]["seed"] = int(args.seed)
        out_dir = cfg["exp"].get("out_dir", "outputs/gi_moe")
        cfg["exp"]["out_dir"] = f"{out_dir}_seed{args.seed}"
        os.makedirs(cfg["exp"]["out_dir"], exist_ok=True)

    if args.ablation_all:
        results = {}
        for m in ABLATION_MODES:
            print("\n" + "=" * 80 + f"\n[ablation] mode={m}\n" + "=" * 80)
            results[m] = run(cfg, mode_override=m, verify_grad=False)
        out_dir = cfg["exp"]["out_dir"]
        with open(os.path.join(out_dir, "ablation_all.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print("\n[ablation_all] best_val_mse / final_test_mse:")
        for m, r in results.items():
            print(f"  {m:34s} val={r['best_val_mse']:.4f}  test={r['final']['test_mse']:.4f}")
        return

    run(cfg, mode_override=args.mode, verify_grad=args.verify_grad)


if __name__ == "__main__":
    main()
