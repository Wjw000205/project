"""
Prediction-level diagnostic for v2 GI-MoE (hidden_block_gi_moe_loss).

Stops asking 'does test_mse drop?' and instead inspects WHAT the system is doing:

Q1. Does branch r_p actually flow through to y_final?
    - Compute |gate_p · alpha_p · r_p| / |y_base| per sample, distribution.

Q2. Does y_final improve on the penalty metric vs y_base?
    - penalty_p(y_base, y),  penalty_p(y_final, y),  penalty_p(y, y)=0.
    - If y_final's score < y_base's score, penalty is doing its job at training-objective level.

Q3. Is gate firing correlated with actual improve_p?
    - For each penalty p:
        improve_p_real(b, c) = penalty_p(y_base, y) - penalty_p(y_base + branch_p, y)
        gate_p_mean(b, c)    = gates[p].mean(-1)
        corr(gate_p_mean, improve_p_real) -> if low, gate is firing randomly.

Q4. Has base already learned the penalty-targeted feature?
    - Already covered by rel_gap probe. Here we also look at per-sample base
      penalty distribution (histogram), to see if base is uniformly bad or
      only bad on a subset.

Q5. Mismatch type analysis: where exactly does y_base fail and does y_final fix it?
    - Pick worst-K samples by y_base test_mse; show how y_final differs and which
      penalty's branch dominated the fix.

Usage:
  python -m scripts.diagnose_predictions \\
      --config configs/gi_moe_ETTm1_directional.yaml \\
      --mode hidden_block_gi_moe_loss
"""
from __future__ import annotations
import argparse
import json
import os
import time
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from src.utils.yaml_io import load_yaml
from src.models.penalties import build_penalty_bank
from src.models.gi_moe import (
    SimpleBasePredictor,
    HiddenBlockMoEHead,
    PenaltyAdapterBank,
    gi_moe_loss_v2,
    gi_moe_loss,
)
from scripts.run_gi_moe import build_loaders, _seed


# -------------------------------------------------------------------------
def _build_models(cfg, device, num_channels):
    L = int(cfg["window"]["input_len"]); H = int(cfg["window"]["pred_len"])
    hidden_dim = int(cfg["model"].get("hidden_dim", 256))
    base = SimpleBasePredictor(
        input_len=L, pred_len=H, hidden_dim=hidden_dim,
        dropout=float(cfg["model"].get("dropout", 0.2)),
    ).to(device)
    mcfg2 = cfg.get("moe_loss_v2", {}) or {}
    pen_names = list(mcfg2.get("penalties", ["delta", "direction", "trend"]))
    head = HiddenBlockMoEHead(
        in_dim=hidden_dim, pred_len=H, penalty_names=pen_names,
        shared_dim=int(mcfg2.get("shared_dim", 128)),
        private_dim=int(mcfg2.get("private_dim", 32)),
        dropout=float(cfg["model"].get("dropout", 0.0)),
        mask_init=float(mcfg2.get("mask_init", 0.0)),
        log_alpha_init=float(mcfg2.get("log_alpha_init", -3.0)),
        gate_init_bias=float(mcfg2.get("gate_init_bias", -2.0)),
        use_pga=bool(mcfg2.get("use_penalty_gated_activation", True)),
    ).to(device)
    return base, head, pen_names


def _train(base, head, mode, cfg, dl_tr, device, penalty_fns, pen_names):
    mcfg2 = cfg.get("moe_loss_v2", {}) or {}
    params = list(base.parameters()) + list(head.parameters())
    opt = torch.optim.Adam(
        params, lr=float(cfg["train"].get("lr", 1.0e-3)),
        weight_decay=float(cfg["train"].get("weight_decay", 0.0)),
    )
    mae_w = float(mcfg2.get("mae_weight", 0.3))
    lam_pen = float(mcfg2.get("lambda_pen", 0.1))
    lam_p = mcfg2.get("lambda_p", None)
    lam_norm = float(mcfg2.get("lambda_norm", 1.0e-4))
    lam_mask = float(mcfg2.get("lambda_mask", 1.0e-4))
    mask_t = float(mcfg2.get("mask_target", 0.5))
    epochs = int(cfg["train"].get("epochs", 40))
    print(f"[diag] training base+head for {epochs} epochs, mode={mode}")
    t0 = time.time()
    for ep in range(1, epochs + 1):
        base.train(); head.train()
        for x, y, _ in dl_tr:
            x = x.to(device); y = y.to(device)
            y_base_v, h = base(x, return_features=True)
            out = head(h, y_base=y_base_v)
            y_base = out["y_base"]; y_final = out["y_final"]
            if mode == "base_only":
                # train base via simple decode (head bypassed)
                y_pred_base = base(x)
                loss = F.mse_loss(y_pred_base, y) + mae_w * F.l1_loss(y_pred_base, y)
            else:
                loss, _ = gi_moe_loss_v2(
                    y_base=y_base, y_final=y_final, y=y,
                    residuals=out["residuals"], gates=out["gates"], alphas=out["alphas"],
                    penalty_fns=penalty_fns,
                    mask_values=out.get("mask_values"),
                    lambda_pen=lam_pen, lambda_p=lam_p,
                    lambda_norm=lam_norm, mae_weight=mae_w,
                    lambda_mask=lam_mask, mask_target=mask_t,
                )
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
    print(f"[diag] trained in {time.time()-t0:.1f}s")


@torch.no_grad()
def _collect_diagnostics(base, head, loader, device, penalty_fns, pen_names):
    base.eval(); head.eval()
    # Accumulate per-batch tensors and statistics.
    all_branch_rel: Dict[str, List[float]] = {p: [] for p in pen_names}
    all_gate_means: Dict[str, List[float]] = {p: [] for p in pen_names}
    all_improve: Dict[str, List[float]] = {p: [] for p in pen_names}
    all_alpha: Dict[str, List[float]] = {p: [] for p in pen_names}
    pen_base_sum = {p: 0.0 for p in pen_names}
    pen_final_sum = {p: 0.0 for p in pen_names}
    pen_truth_sum = {p: 0.0 for p in pen_names}
    counts = 0
    se_base = se_final = n_pts = 0.0
    sample_records = []

    for x, y, _ in loader:
        x = x.to(device); y = y.to(device)
        y_base_v, h = base(x, return_features=True)
        out = head(h, y_base=y_base_v)
        y_base = out["y_base"]; y_final = out["y_final"]

        # global mse
        se_base += float(((y_base - y) ** 2).sum().item())
        se_final += float(((y_final - y) ** 2).sum().item())
        n_pts += float(y.numel())

        base_norm = y_base.pow(2).mean(dim=-1).sqrt().clamp_min(1e-8)  # [B,C]

        for p in pen_names:
            r = out["residuals"][p]
            g = out["gates"][p]
            a = out["alphas"][p]
            branch = g * a * r                                          # [B,C,H]
            branch_norm = branch.pow(2).mean(dim=-1).sqrt()             # [B,C]
            rel = (branch_norm / base_norm).cpu().numpy()
            all_branch_rel[p].extend(rel.flatten().tolist())
            # gate is now [B,C,1] (per-sample scalar); squeeze for stats.
            all_gate_means[p].extend(g.squeeze(-1).cpu().numpy().flatten().tolist())
            all_alpha[p].append(float(a.item()))

            # Per-sample improve_p = penalty_p(y_base, y) - penalty_p(y_base + branch, y)
            pen_base = penalty_fns[p](y_base, y)                       # [B,C]
            pen_base_branch = penalty_fns[p](y_base + branch, y)       # [B,C]
            improve = (pen_base - pen_base_branch).cpu().numpy()
            all_improve[p].extend(improve.flatten().tolist())

            # Aggregate penalty stats (global level — using y_base / y_final / truth-y)
            pen_base_sum[p] += float(pen_base.mean().item())
            pen_final_sum[p] += float(penalty_fns[p](y_final, y).mean().item())
            pen_truth_sum[p] += float(penalty_fns[p](y, y).mean().item())
        counts += 1

    # Worst-K samples by y_base MSE (pick e.g. 5 worst).
    # Redo a pass to collect them with full tensors.
    sample_records = []
    with torch.no_grad():
        worst_buf = []  # list of (mse, x, y, y_base, y_final, branches_per_p, gates_per_p, alphas_per_p)
        for x, y, _ in loader:
            x = x.to(device); y = y.to(device)
            y_base_v, h = base(x, return_features=True)
            out = head(h, y_base=y_base_v)
            y_base = out["y_base"]; y_final = out["y_final"]
            sample_mse = (y_base - y).pow(2).mean(dim=-1)             # [B, C]
            B, C = sample_mse.shape
            for b in range(B):
                for c in range(C):
                    item = (float(sample_mse[b, c].item()),
                            b, c,
                            y_base[b, c].cpu().numpy(),
                            y_final[b, c].cpu().numpy(),
                            y[b, c].cpu().numpy(),
                            {p: float((out["gates"][p][b, c] * out["alphas"][p] * out["residuals"][p][b, c]).pow(2).mean().sqrt().item()) for p in pen_names},
                            {p: float(out["gates"][p][b, c].mean().item()) for p in pen_names},
                            )
                    worst_buf.append(item)
            if len(worst_buf) > 3000:
                worst_buf.sort(key=lambda r: -r[0])
                worst_buf = worst_buf[:5]
        worst_buf.sort(key=lambda r: -r[0])
        sample_records = worst_buf[:5]

    report = {
        "global_mse": {"base": se_base / max(n_pts, 1.0), "final": se_final / max(n_pts, 1.0)},
        "penalty_stats_global": {
            p: {
                "y_base": pen_base_sum[p] / max(counts, 1),
                "y_final": pen_final_sum[p] / max(counts, 1),
                "y_truth_as_truth": pen_truth_sum[p] / max(counts, 1),
            } for p in pen_names
        },
        "alpha_avg": {p: float(np.mean(all_alpha[p])) for p in pen_names},
    }

    # Distribution stats
    for p in pen_names:
        rels = np.array(all_branch_rel[p])
        gms = np.array(all_gate_means[p])
        imps = np.array(all_improve[p])
        report.setdefault("distributions", {})[p] = {
            "branch_rel_y_base": {"mean": float(rels.mean()), "median": float(np.median(rels)),
                                  "p10": float(np.percentile(rels, 10)),
                                  "p90": float(np.percentile(rels, 90)),
                                  "max": float(rels.max())},
            "gate_mean_per_sample": {"mean": float(gms.mean()), "std": float(gms.std()),
                                     "p10": float(np.percentile(gms, 10)),
                                     "p90": float(np.percentile(gms, 90))},
            "improve_p_real_per_sample": {"mean": float(imps.mean()), "std": float(imps.std()),
                                          "frac_positive": float((imps > 0).mean()),
                                          "p10": float(np.percentile(imps, 10)),
                                          "p90": float(np.percentile(imps, 90))},
            "corr_gate_vs_improve": float(np.corrcoef(gms, imps)[0, 1]),
        }

    return report, sample_records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--mode", default="hidden_block_gi_moe_loss")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    if args.seed is not None:
        cfg["exp"]["seed"] = int(args.seed)
    device = torch.device(cfg["exp"].get("device", "cuda:0") if torch.cuda.is_available() else "cpu")
    _seed(int(cfg["exp"].get("seed", 2026)))

    mcfg2 = cfg.get("moe_loss_v2", {}) or {}
    pen_names = list(mcfg2.get("penalties", ["delta", "direction", "trend"]))
    penalty_fns = build_penalty_bank(pen_names, jump_thr=float(mcfg2.get("jump_threshold", 2.0)))

    dl_tr, dl_va, dl_te, num_channels = build_loaders(cfg, device)
    base, head, pen_names = _build_models(cfg, device, num_channels)
    _train(base, head, args.mode, cfg, dl_tr, device, penalty_fns, pen_names)
    report, samples = _collect_diagnostics(base, head, dl_te, device, penalty_fns, pen_names)

    # Print compact summary.
    print(f"\n[diag] === GLOBAL MSE ===")
    print(f"  y_base test_mse:   {report['global_mse']['base']:.4f}")
    print(f"  y_final test_mse:  {report['global_mse']['final']:.4f}")
    print(f"  delta:             {report['global_mse']['final'] - report['global_mse']['base']:+.4f}  "
          f"(negative = y_final beats y_base)")

    print(f"\n[diag] === Q2 PENALTY-METRIC IMPROVEMENT (global mean over test) ===")
    print(f"  {'penalty':12s} {'y_base':>10s} {'y_final':>10s} {'truth':>10s} {'delta':>10s}")
    for p in pen_names:
        s = report["penalty_stats_global"][p]
        d = s["y_final"] - s["y_base"]
        print(f"  {p:12s} {s['y_base']:>10.4f} {s['y_final']:>10.4f} {s['y_truth_as_truth']:>10.4f} {d:>+10.4f}")
    print(f"  Interpretation: 'delta' negative = y_final reduced this penalty's value -> penalty doing its job.")

    print(f"\n[diag] === Q1 BRANCH CONTRIBUTION (||g·α·r_p|| / ||y_base|| per sample) ===")
    print(f"  {'penalty':12s} {'mean':>8s} {'median':>8s} {'p10':>8s} {'p90':>8s} {'max':>8s} {'alpha_avg':>10s}")
    for p in pen_names:
        d = report["distributions"][p]["branch_rel_y_base"]
        print(f"  {p:12s} {d['mean']:>8.4f} {d['median']:>8.4f} {d['p10']:>8.4f} {d['p90']:>8.4f} {d['max']:>8.4f} {report['alpha_avg'][p]:>10.4f}")
    print(f"  Interpretation: if median < 0.01, branch is effectively inert.")

    print(f"\n[diag] === Q3 PER-SAMPLE GATE vs ACTUAL IMPROVE_P ===")
    print(f"  {'penalty':12s} {'gate_mean':>10s} {'gate_std':>10s} {'imp_mean':>10s} {'imp_std':>10s} {'frac_imp>0':>11s} {'corr':>8s}")
    for p in pen_names:
        d = report["distributions"][p]
        print(f"  {p:12s} {d['gate_mean_per_sample']['mean']:>10.4f} {d['gate_mean_per_sample']['std']:>10.4f} "
              f"{d['improve_p_real_per_sample']['mean']:>10.4f} {d['improve_p_real_per_sample']['std']:>10.4f} "
              f"{d['improve_p_real_per_sample']['frac_positive']:>11.4f} {d['corr_gate_vs_improve']:>8.3f}")
    print(f"  Interpretation:")
    print(f"    - frac_imp>0 < 0.5: branch HURTS more samples than it helps")
    print(f"    - corr > 0.3: gate is correctly routing to needy samples")
    print(f"    - corr ~ 0: gate routing is uncorrelated with where penalty actually helps")

    print(f"\n[diag] === Q5 WORST-5 SAMPLES BY y_base MSE ===")
    for rank, (mse, b, c, yb, yf, yt, branch_norms, gate_means) in enumerate(samples, 1):
        improve = float(((yb - yt) ** 2).mean() - ((yf - yt) ** 2).mean())
        print(f"  rank {rank}: batch_b={b} c={c} | y_base_mse={mse:.3f} | y_final-y_base mse delta={-improve:+.3f}")
        print(f"           branch_norms = " + ", ".join(f"{p}:{branch_norms[p]:.3f}" for p in pen_names))
        print(f"           gate_means   = " + ", ".join(f"{p}:{gate_means[p]:.3f}" for p in pen_names))

    out_path = args.out or os.path.join(cfg["exp"].get("out_dir", "outputs"), "diagnostic_report.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # Drop numpy arrays from sample records for JSON-friendliness.
    serializable_samples = [
        {"mse_y_base": s[0], "batch": s[1], "channel": s[2],
         "branch_norms": s[6], "gate_means": s[7]}
        for s in samples
    ]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"report": report, "worst_samples": serializable_samples}, f, indent=2)
    print(f"\n[diag] saved -> {out_path}")


if __name__ == "__main__":
    main()
