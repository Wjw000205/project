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
    "rank",
    "prediction_source",
    "periods",
    "alpha",
    "val_mse",
    "val_mae",
    "test_mse",
    "test_mae",
    "prototype_fit_split",
    "min_count",
    "mean_count",
]


def resolve_path(path_text: str | Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def parse_int_groups(text: str) -> list[tuple[int, ...]]:
    groups = []
    for raw_group in str(text).split(";"):
        values = tuple(int(part.strip()) for part in raw_group.split(",") if part.strip())
        if values:
            groups.append(values)
    if not groups:
        raise ValueError("Expected at least one period group.")
    return groups


def parse_float_list(text: str) -> list[float]:
    values = [float(part.strip()) for part in str(text).split(",") if part.strip()]
    if not values:
        raise ValueError("Expected at least one float value.")
    return values


def metrics(pred: torch.Tensor, true: torch.Tensor) -> tuple[float, float]:
    return float((pred - true).pow(2).mean().item()), float((pred - true).abs().mean().item())


def make_knn_config(args: argparse.Namespace) -> KNNShapeConfig:
    return KNNShapeConfig.from_dict(
        {
            "enable": True,
            "mode": "fixed",
            "scope": "same_channel",
            "bank_split": "train",
            "bank_stride": int(args.bank_stride),
            "feature_mode": "joint",
            "template_mode": "residual",
            "distance_weight": "inverse",
            "anchor_mode": "last",
            "shape_bins": 24,
            "diff_bins": 12,
            "pred_shape_bins": 16,
            "pred_diff_bins": 8,
            "time_feature_mode": "forecast_phase",
            "time_periods": args.time_periods,
            "time_feature_weight": float(args.time_feature_weight),
            "k": int(args.knn_k),
            "alpha": float(args.knn_alpha),
            "adaptive_alpha": "confidence",
            "distance_sharpness": 2.0,
            "confidence_floor": 0.0,
        }
    )


@torch.no_grad()
def collect_base_predictions(
    *,
    model,
    loader,
    starts_base: int,
    cluster_id_c: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    preds = []
    trues = []
    starts = []
    model.eval()
    for x, y, idx in loader:
        x = x.to(device, non_blocking=True)
        idx = idx.to(device=device, dtype=torch.long, non_blocking=True)
        pred = model(x, cluster_id_c)
        preds.append(pred.detach().cpu())
        trues.append(y.detach().cpu())
        starts.append((int(starts_base) + idx).detach().cpu())
    return torch.cat(preds, dim=0), torch.cat(trues, dim=0), torch.cat(starts, dim=0)


@torch.no_grad()
def collect_knn_predictions(
    *,
    model,
    hybrid: ShapeKNNHybrid,
    loader,
    starts_base: int,
    cluster_id_c: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    preds = []
    trues = []
    starts = []
    model.eval()
    for x, y, idx in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        idx = idx.to(device=device, dtype=torch.long, non_blocking=True)
        base = model(x, cluster_id_c)
        pred = hybrid.hybridize_batch(x, base, cluster_id_c, query_start_abs_b=int(starts_base) + idx)
        preds.append(pred.detach().cpu())
        trues.append(y.detach().cpu())
        starts.append((int(starts_base) + idx).detach().cpu())
    return torch.cat(preds, dim=0), torch.cat(trues, dim=0), torch.cat(starts, dim=0)


def fit_phase_prototype(
    residual_nch: torch.Tensor,
    starts_n: torch.Tensor,
    input_len: int,
    period: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    period = max(int(period), 1)
    c = int(residual_nch.shape[1])
    h = int(residual_nch.shape[2])
    sums = torch.zeros((period, c, h), dtype=residual_nch.dtype)
    counts = torch.zeros(period, dtype=residual_nch.dtype)
    phases = torch.remainder(starts_n.to(torch.long) + int(input_len), period)
    for phase in range(period):
        mask = phases == phase
        if bool(mask.any().item()):
            sums[phase] = residual_nch[mask].mean(dim=0)
            counts[phase] = float(mask.sum().item())
    return sums, counts


def lookup_phase_prototype(
    prototypes: list[torch.Tensor],
    starts_n: torch.Tensor,
    input_len: int,
) -> torch.Tensor:
    parts = []
    for proto_pch in prototypes:
        period = int(proto_pch.shape[0])
        phases = torch.remainder(starts_n.to(torch.long) + int(input_len), period)
        parts.append(proto_pch.index_select(0, phases))
    return torch.stack(parts, dim=0).mean(dim=0)


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDS})


def main() -> None:
    ap = argparse.ArgumentParser(description="Probe train-only phase residual prototypes on input-96 predictions.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--eval-batch-size", type=int, default=None)
    ap.add_argument("--period-groups", default="96;672;96,672")
    ap.add_argument("--alphas", default="0.1,0.2,0.3,0.5,0.75,1.0")
    ap.add_argument("--prediction-sources", default="base,knn")
    ap.add_argument("--prototype-fit-split", choices=["train", "pre_test"], default="train")
    ap.add_argument("--knn-k", type=int, default=42)
    ap.add_argument("--knn-alpha", type=float, default=1.35)
    ap.add_argument("--bank-stride", type=int, default=2)
    ap.add_argument("--time-periods", default="96,672")
    ap.add_argument("--time-feature-weight", type=float, default=1.0)
    args = ap.parse_args()

    config_path = resolve_path(args.config)
    run_dir = resolve_path(args.run_dir)
    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_yaml(config_path)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    context = prepare_data_context(cfg)
    batch_size = int(args.eval_batch_size or cfg["train"]["batch_size"])
    checkpoint_path = run_dir / "best_checkpoint.pt"
    bundle = load_eval_modules(cfg, checkpoint_path, context.K, device)
    model = bundle["model"]
    cluster_id_c = context.cluster_id_c.to(device)

    train_loader = make_loader(context.xtr_norm, context.ytr_norm, batch_size)
    xva, yva = make_strict_windows(context.norm_data_tc, context.L, context.H, context.t_train, context.t_val)
    val_loader = make_loader(xva, yva, batch_size)
    test_loader = make_loader(context.xte_norm, context.yte_norm, batch_size)

    train_base, train_true, train_starts = collect_base_predictions(
        model=model,
        loader=train_loader,
        starts_base=0,
        cluster_id_c=cluster_id_c,
        device=device,
    )
    val_base, val_true, val_starts = collect_base_predictions(
        model=model,
        loader=val_loader,
        starts_base=context.t_train,
        cluster_id_c=cluster_id_c,
        device=device,
    )
    test_base, test_true, test_starts = collect_base_predictions(
        model=model,
        loader=test_loader,
        starts_base=context.t_val,
        cluster_id_c=cluster_id_c,
        device=device,
    )

    val_knn = test_knn = None
    if "knn" in {part.strip().lower() for part in str(args.prediction_sources).split(",")}:
        knn_cfg = make_knn_config(args)
        starts_bank = torch.arange(0, int(context.xtr_norm.shape[0]), dtype=torch.long)
        base_bank_pred = predict_bank_outputs(
            model=model,
            x_bank_ncl=context.xtr_norm,
            cluster_id_c=cluster_id_c,
            batch_size=max(batch_size, 64),
            device=device,
        )
        hybrid = ShapeKNNHybrid.fit(
            x_bank_ncl=context.xtr_norm,
            y_bank_nch=context.ytr_norm,
            cluster_id_c=cluster_id_c,
            cfg=knn_cfg,
            start_offsets_n=starts_bank,
            base_bank_pred_nch=base_bank_pred,
        )
        val_knn, _, _ = collect_knn_predictions(
            model=model,
            hybrid=hybrid,
            loader=val_loader,
            starts_base=context.t_train,
            cluster_id_c=cluster_id_c,
            device=device,
        )
        test_knn, _, _ = collect_knn_predictions(
            model=model,
            hybrid=hybrid,
            loader=test_loader,
            starts_base=context.t_val,
            cluster_id_c=cluster_id_c,
            device=device,
        )

    if args.prototype_fit_split == "pre_test":
        fit_x, fit_y = make_strict_windows(context.norm_data_tc, context.L, context.H, 0, context.t_val)
        fit_loader = make_loader(fit_x, fit_y, batch_size)
        fit_base, fit_true, fit_starts = collect_base_predictions(
            model=model,
            loader=fit_loader,
            starts_base=0,
            cluster_id_c=cluster_id_c,
            device=device,
        )
        fit_residual = fit_true - fit_base
    else:
        fit_starts = train_starts
        fit_residual = train_true - train_base
    period_groups = parse_int_groups(args.period_groups)
    alphas = parse_float_list(args.alphas)
    sources = [part.strip().lower() for part in str(args.prediction_sources).split(",") if part.strip()]

    rows: list[dict[str, Any]] = []
    prototype_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for period_group in period_groups:
        prototypes = []
        counts = []
        for period in period_group:
            if int(period) not in prototype_cache:
                prototype_cache[int(period)] = fit_phase_prototype(fit_residual, fit_starts, context.L, int(period))
            proto, count = prototype_cache[int(period)]
            prototypes.append(proto)
            counts.append(count)
        val_corr = lookup_phase_prototype(prototypes, val_starts, context.L)
        test_corr = lookup_phase_prototype(prototypes, test_starts, context.L)
        count_cat = torch.cat([c.reshape(-1) for c in counts])
        for source in sources:
            if source == "base":
                val_pred = val_base
                test_pred = test_base
            elif source == "knn":
                if val_knn is None or test_knn is None:
                    continue
                val_pred = val_knn
                test_pred = test_knn
            else:
                raise ValueError(f"Unsupported prediction source: {source}")
            for alpha in alphas:
                cal_val = val_pred + float(alpha) * val_corr
                cal_test = test_pred + float(alpha) * test_corr
                val_mse, val_mae = metrics(cal_val, val_true)
                test_mse, test_mae = metrics(cal_test, test_true)
                rows.append(
                    {
                        "prediction_source": source,
                        "periods": ",".join(str(v) for v in period_group),
                        "alpha": float(alpha),
                        "val_mse": val_mse,
                        "val_mae": val_mae,
                        "test_mse": test_mse,
                        "test_mae": test_mae,
                        "prototype_fit_split": str(args.prototype_fit_split),
                        "min_count": float(count_cat[count_cat > 0].min().item()) if bool((count_cat > 0).any().item()) else 0.0,
                        "mean_count": float(count_cat[count_cat > 0].mean().item()) if bool((count_cat > 0).any().item()) else 0.0,
                    }
                )

    rows = sorted(rows, key=lambda row: (float(row["val_mse"]), float(row["val_mae"])))
    for idx, row in enumerate(rows, start=1):
        row["rank"] = idx
    write_rows(out_dir / "phase_residual_prototype_results.csv", rows)
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump({"config_path": str(config_path), "run_dir": str(run_dir), "rows": rows}, f, ensure_ascii=False, indent=2)
    print("Top phase residual prototype rows:")
    for row in rows[:12]:
        print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
