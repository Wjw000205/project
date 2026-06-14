from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.compare_knn_shape_variants import make_loader  # noqa: E402
from scripts.compare_moe_on_off import load_eval_modules, load_yaml, prepare_data_context  # noqa: E402
from src.data.windows import make_strict_windows  # noqa: E402
from src.utils.knn_shape import KNNShapeConfig, ShapeKNNHybrid, predict_bank_outputs  # noqa: E402


FIELDS = [
    "method",
    "shrink",
    "val_mse",
    "val_mae",
    "test_mse",
    "test_mae",
    "bank_split",
    "k",
    "alpha",
    "time_feature_mode",
    "time_periods",
    "time_feature_weight",
]


def resolve_path(path_text: str | Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def metrics(pred: torch.Tensor, true: torch.Tensor) -> tuple[float, float]:
    return float((pred - true).pow(2).mean().item()), float((pred - true).abs().mean().item())


def fit_apply(
    pred_val: torch.Tensor,
    true_val: torch.Tensor,
    pred_test: torch.Tensor,
    *,
    method: str,
    shrink: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    eps = 1.0e-6
    if method == "none":
        cal_val = pred_val
        cal_test = pred_test
    elif method == "bias_channel":
        bias = (true_val - pred_val).mean(dim=(0, 2), keepdim=True)
        cal_val = pred_val + bias
        cal_test = pred_test + bias
    elif method == "bias_channel_horizon":
        bias = (true_val - pred_val).mean(dim=0, keepdim=True)
        cal_val = pred_val + bias
        cal_test = pred_test + bias
    elif method == "affine_channel":
        x = pred_val
        y = true_val
        mx = x.mean(dim=(0, 2), keepdim=True)
        my = y.mean(dim=(0, 2), keepdim=True)
        vx = (x - mx).pow(2).mean(dim=(0, 2), keepdim=True).clamp_min(eps)
        cov = ((x - mx) * (y - my)).mean(dim=(0, 2), keepdim=True)
        a = (cov / vx).clamp(0.5, 1.5)
        b = my - a * mx
        cal_val = a * pred_val + b
        cal_test = a * pred_test + b
    elif method == "affine_channel_horizon":
        x = pred_val
        y = true_val
        mx = x.mean(dim=0, keepdim=True)
        my = y.mean(dim=0, keepdim=True)
        vx = (x - mx).pow(2).mean(dim=0, keepdim=True).clamp_min(eps)
        cov = ((x - mx) * (y - my)).mean(dim=0, keepdim=True)
        a = (cov / vx).clamp(0.5, 1.5)
        b = my - a * mx
        cal_val = a * pred_val + b
        cal_test = a * pred_test + b
    else:
        raise ValueError(f"Unknown calibration method: {method}")

    shrink = float(shrink)
    return pred_val + shrink * (cal_val - pred_val), pred_test + shrink * (cal_test - pred_test)


def make_knn_config(args: argparse.Namespace) -> KNNShapeConfig:
    return KNNShapeConfig.from_dict(
        {
            "enable": True,
            "mode": args.mode,
            "scope": args.scope,
            "bank_split": args.bank_split,
            "bank_stride": args.bank_stride,
            "feature_mode": args.feature_mode,
            "template_mode": args.template_mode,
            "distance_weight": args.distance_weight,
            "anchor_mode": args.anchor_mode,
            "shape_bins": args.shape_bins,
            "diff_bins": args.diff_bins,
            "pred_shape_bins": args.pred_shape_bins,
            "pred_diff_bins": args.pred_diff_bins,
            "time_feature_mode": args.time_feature_mode,
            "time_periods": args.time_periods,
            "time_feature_weight": args.time_feature_weight,
            "k": args.k,
            "alpha": args.alpha,
            "adaptive_alpha": args.adaptive_alpha,
            "distance_sharpness": args.distance_sharpness,
            "confidence_floor": args.confidence_floor,
        }
    )


def make_bank(context, knn_cfg: KNNShapeConfig):
    if knn_cfg.bank_split == "train":
        starts = torch.arange(0, int(context.xtr_norm.shape[0]), dtype=torch.long)
        return context.xtr_norm, context.ytr_norm, starts
    if knn_cfg.bank_split == "pre_test":
        x_bank, y_bank = make_strict_windows(context.norm_data_tc, context.L, context.H, 0, context.t_val)
        starts = torch.arange(0, int(x_bank.shape[0]), dtype=torch.long)
        return x_bank, y_bank, starts
    raise ValueError("This calibration probe supports bank_split=train or pre_test.")


@torch.no_grad()
def collect_predictions(
    *,
    model,
    hybrid: ShapeKNNHybrid,
    loader,
    eval_start: int,
    cluster_id_c: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    preds = []
    trues = []
    model.eval()
    for x, y, idx in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        idx = idx.to(device=device, dtype=torch.long, non_blocking=True)
        base = model(x, cluster_id_c)
        pred = hybrid.hybridize_batch(x, base, cluster_id_c, query_start_abs_b=eval_start + idx)
        preds.append(pred.detach().cpu())
        trues.append(y.detach().cpu())
    return torch.cat(preds, dim=0), torch.cat(trues, dim=0)


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDS})


def main() -> None:
    ap = argparse.ArgumentParser(description="Probe validation calibration on phase-aware KNN predictions.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--eval-batch-size", type=int, default=None)
    ap.add_argument("--mode", choices=["fixed", "rolling"], default="fixed")
    ap.add_argument("--scope", choices=["same_channel", "same_cluster"], default="same_channel")
    ap.add_argument("--bank-split", choices=["train", "pre_test"], default="train")
    ap.add_argument("--bank-stride", type=int, default=2)
    ap.add_argument("--feature-mode", choices=["hist", "joint"], default="joint")
    ap.add_argument("--template-mode", choices=["future", "residual"], default="residual")
    ap.add_argument("--distance-weight", choices=["inverse", "uniform"], default="inverse")
    ap.add_argument("--anchor-mode", choices=["last", "mean"], default="last")
    ap.add_argument("--shape-bins", type=int, default=24)
    ap.add_argument("--diff-bins", type=int, default=12)
    ap.add_argument("--pred-shape-bins", type=int, default=16)
    ap.add_argument("--pred-diff-bins", type=int, default=8)
    ap.add_argument("--time-feature-mode", default="forecast_phase")
    ap.add_argument("--time-periods", default="96,672")
    ap.add_argument("--time-feature-weight", type=float, default=1.0)
    ap.add_argument("--k", type=int, default=42)
    ap.add_argument("--alpha", type=float, default=1.35)
    ap.add_argument("--adaptive-alpha", default="confidence")
    ap.add_argument("--distance-sharpness", type=float, default=2.0)
    ap.add_argument("--confidence-floor", type=float, default=0.0)
    args = ap.parse_args()

    config_path = resolve_path(args.config)
    run_dir = resolve_path(args.run_dir)
    checkpoint_path = run_dir / "best_checkpoint.pt"
    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_yaml(config_path)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    context = prepare_data_context(cfg)
    batch_size = int(args.eval_batch_size or cfg["train"]["batch_size"])

    bundle = load_eval_modules(cfg, checkpoint_path, context.K, device)
    model = bundle["model"]
    cluster_id_c = context.cluster_id_c.to(device)
    knn_cfg = make_knn_config(args)
    x_bank, y_bank, starts = make_bank(context, knn_cfg)
    base_bank_pred = None
    if knn_cfg.needs_base_bank_prediction():
        base_bank_pred = predict_bank_outputs(
            model=model,
            x_bank_ncl=x_bank,
            cluster_id_c=cluster_id_c,
            batch_size=max(batch_size, 64),
            device=device,
        )
    hybrid = ShapeKNNHybrid.fit(
        x_bank_ncl=x_bank,
        y_bank_nch=y_bank,
        cluster_id_c=cluster_id_c,
        cfg=knn_cfg,
        start_offsets_n=starts,
        base_bank_pred_nch=base_bank_pred,
    )

    xva, yva = make_strict_windows(context.norm_data_tc, context.L, context.H, context.t_train, context.t_val)
    val_loader = make_loader(xva, yva, batch_size)
    test_loader = make_loader(context.xte_norm, context.yte_norm, batch_size)
    pred_val, true_val = collect_predictions(
        model=model,
        hybrid=hybrid,
        loader=val_loader,
        eval_start=context.t_train,
        cluster_id_c=cluster_id_c,
        device=device,
    )
    pred_test, true_test = collect_predictions(
        model=model,
        hybrid=hybrid,
        loader=test_loader,
        eval_start=context.t_val,
        cluster_id_c=cluster_id_c,
        device=device,
    )

    rows: list[dict[str, Any]] = []
    methods = ["none", "bias_channel", "bias_channel_horizon", "affine_channel", "affine_channel_horizon"]
    shrinks = [0.125, 0.25, 0.5, 0.75, 1.0]
    for method in methods:
        method_shrinks = [1.0] if method == "none" else shrinks
        for shrink in method_shrinks:
            cal_val, cal_test = fit_apply(pred_val, true_val, pred_test, method=method, shrink=shrink)
            val_mse, val_mae = metrics(cal_val, true_val)
            test_mse, test_mae = metrics(cal_test, true_test)
            rows.append(
                {
                    "method": method,
                    "shrink": float(shrink),
                    "val_mse": val_mse,
                    "val_mae": val_mae,
                    "test_mse": test_mse,
                    "test_mae": test_mae,
                    "bank_split": knn_cfg.bank_split,
                    "k": int(knn_cfg.k),
                    "alpha": float(knn_cfg.alpha),
                    "time_feature_mode": knn_cfg.time_feature_mode,
                    "time_periods": ",".join(str(int(v)) for v in knn_cfg.time_periods),
                    "time_feature_weight": float(knn_cfg.time_feature_weight),
                }
            )
    rows = sorted(rows, key=lambda row: (float(row["val_mse"]), float(row["val_mae"])))
    write_rows(out_dir / "calibration_results.csv", rows)
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump({"config_path": str(config_path), "run_dir": str(run_dir), "rows": rows}, f, ensure_ascii=False, indent=2)
    print("Top calibration rows:")
    for row in rows[:10]:
        print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
