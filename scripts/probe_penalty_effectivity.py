"""
Penalty Effectivity Probe.

Question: does base_only (a converged plain MLP) actually fail at the structural
modes our penalty pool is designed for?

If base_only's predictions:
  - have systematic amplitude collapse (predicted_std < truth_std) -> amp_under
    SHOULD be large -> the penalty has real room to help.
  - are too smooth on first-difference -> delta SHOULD be large.
  - are too jittery -> jitter SHOULD be large.
  - have first-derivative mismatch -> smooth SHOULD be large.

We also compute, as reference, a trivial constant predictor (mean of past) and
the truth-against-truth (which should be exactly 0). This gives us scales.

For each penalty we report:
    base_only_mean    -- what base achieves on test
    last_value_mean   -- trivial repeat-last-value predictor baseline
    truth_mean        -- penalty(y, y) — exactly 0 for sanity
    relative_gap      -- (base_only_mean - 0) normalized by last_value_mean

If relative_gap is small (<0.05), base_only is already close to the "perfect"
score on that penalty -> adding that penalty as a supervisor cannot help much.
If relative_gap is large (>0.3), there is genuine room for that penalty.

Usage:
    python -m scripts.probe_penalty_effectivity --config configs/gi_moe_ETTm1.yaml
"""
from __future__ import annotations
import argparse
import json
import os
import time
from typing import Dict

import torch
from torch import nn

from src.utils.yaml_io import load_yaml
from src.models.penalties import build_penalty_bank
from src.models.gi_moe import SimpleBasePredictor
from scripts.run_gi_moe import build_loaders, _seed


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
            loss = torch.nn.functional.mse_loss(yhat, y) + mae_weight * torch.nn.functional.l1_loss(yhat, y)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(base.parameters(), 1.0); opt.step()
    print(f"[probe] base trained in {time.time()-t0:.1f}s")
    return base, dl_te


@torch.no_grad()
def _per_penalty_stats(base, loader, penalty_fns, device) -> Dict[str, Dict[str, float]]:
    base.eval()
    sums = {p: 0.0 for p in penalty_fns}
    sums_last = {p: 0.0 for p in penalty_fns}
    sums_truth = {p: 0.0 for p in penalty_fns}
    n_batches = 0
    se_base = se_last = n_y = 0.0
    for x, y, _ in loader:
        x = x.to(device); y = y.to(device)
        y_base = base(x)
        y_last = x[..., -1:].expand_as(y)
        for p, fn in penalty_fns.items():
            sums[p] += float(fn(y_base, y).mean().item())
            sums_last[p] += float(fn(y_last, y).mean().item())
            sums_truth[p] += float(fn(y, y).mean().item())
        se_base += float(((y_base - y) ** 2).sum().item())
        se_last += float(((y_last - y) ** 2).sum().item())
        n_y += float(y.numel())
        n_batches += 1
    n = max(n_batches, 1)
    base_mse = se_base / max(n_y, 1.0)
    last_mse = se_last / max(n_y, 1.0)
    report: Dict[str, Dict[str, float]] = {
        "_mse": {"base_only": base_mse, "last_value": last_mse, "truth": 0.0},
    }
    for p in penalty_fns:
        b = sums[p] / n; l = sums_last[p] / n; t = sums_truth[p] / n
        # relative gap: how much room remains compared to trivial baseline.
        denom = max(abs(l), 1.0e-8)
        rel_gap = (b - t) / denom
        report[p] = {
            "base_only": b,
            "last_value": l,
            "truth": t,
            "relative_gap_vs_last": rel_gap,
        }
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    if args.seed is not None:
        cfg["exp"]["seed"] = int(args.seed)
    device = torch.device(cfg["exp"].get("device", "cuda:0") if torch.cuda.is_available() else "cpu")
    _seed(int(cfg["exp"].get("seed", 2026)))

    mcfg = cfg.get("moe_loss", {}) or {}
    pen_names = list(mcfg.get("penalties", ["amp_under", "delta", "jitter", "smooth"]))
    penalty_fns = build_penalty_bank(pen_names, jump_thr=float(mcfg.get("jump_threshold", 2.0)))

    base, dl_te = _train_base(cfg, device)
    report = _per_penalty_stats(base, dl_te, penalty_fns, device)

    print(f"\n[probe] === penalty effectivity on test set ===")
    print(f"  config:          {args.config}")
    print(f"  base test_mse:   {report['_mse']['base_only']:.4f}")
    print(f"  last test_mse:   {report['_mse']['last_value']:.4f}")
    print()
    print(f"  {'penalty':12s} {'base':>10s} {'last_value':>12s} {'truth':>10s} {'rel_gap':>10s}")
    for p in pen_names:
        r = report[p]
        print(f"  {p:12s} {r['base_only']:>10.4f} {r['last_value']:>12.4f} {r['truth']:>10.4f} {r['relative_gap_vs_last']:>10.3f}")
    print()
    print(f"  Interpretation:")
    print(f"    rel_gap > 0.3   = penalty has real room to help (base is well below trivial baseline).")
    print(f"    rel_gap < 0.05  = base is already near-zero on this penalty; supervisor cannot push much.")

    # save
    out_path = os.path.join(cfg["exp"].get("out_dir", "outputs"), "penalty_effectivity.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n[probe] saved -> {out_path}")


if __name__ == "__main__":
    main()
