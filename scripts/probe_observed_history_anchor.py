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
    "blend_target",
    "lags",
    "alpha",
    "val_mse",
    "val_mae",
    "test_mse",
    "test_mae",
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
        raise ValueError("Expected at least one lag group.")
    return groups


def parse_float_list(text: str) -> list[float]:
    values = [float(part.strip()) for part in str(text).split(",") if part.strip()]
    if not values:
        raise ValueError("Expected at least one alpha.")
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
def collect_predictions(
    *,
    model,
    hybrid: ShapeKNNHybrid | None,
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
        pred = model(x, cluster_id_c)
        if hybrid is not None:
            pred = hybrid.hybridize_batch(x, pred, cluster_id_c, query_start_abs_b=int(starts_base) + idx)
        preds.append(pred.detach().cpu())
        trues.append(y.detach().cpu())
        starts.append((int(starts_base) + idx).detach().cpu())
    return torch.cat(preds, dim=0), torch.cat(trues, dim=0), torch.cat(starts, dim=0)


def observed_history_anchor(
    data_tc: torch.Tensor,
    starts_n: torch.Tensor,
    input_len: int,
    pred_len: int,
    lags: tuple[int, ...],
) -> torch.Tensor:
    n = int(starts_n.numel())
    c = int(data_tc.shape[1])
    h = int(pred_len)
    out = torch.zeros((n, c, h), dtype=data_tc.dtype)
    cnt = torch.zeros((n, 1, h), dtype=data_tc.dtype)
    starts = starts_n.to(torch.long)
    for lag in lags:
        lag = int(lag)
        if lag <= 0:
            continue
        for step in range(h):
            forecast_idx = starts + int(input_len) + step
            hist_idx = forecast_idx - lag
            valid = (hist_idx >= 0) & (hist_idx < starts + int(input_len))
            if not bool(valid.any().item()):
                continue
            out[valid, :, step] += data_tc.index_select(0, hist_idx[valid])
            cnt[valid, :, step] += 1.0
    return out / cnt.clamp_min(1.0)


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDS})


def main() -> None:
    ap = argparse.ArgumentParser(description="Probe observed-history seasonal anchors for input-96 forecasts.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--eval-batch-size", type=int, default=None)
    ap.add_argument("--lag-groups", default="96;192;288;336;96,192;96,192,288;96,192,288,336;672;96,672")
    ap.add_argument("--alphas", default="0.1,0.2,0.3,0.5,0.75,1.0")
    ap.add_argument("--prediction-sources", default="base,knn")
    ap.add_argument("--blend-targets", default="prediction,base")
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

    xva, yva = make_strict_windows(context.norm_data_tc, context.L, context.H, context.t_train, context.t_val)
    val_loader = make_loader(xva, yva, batch_size)
    test_loader = make_loader(context.xte_norm, context.yte_norm, batch_size)
    val_base, val_true, val_starts = collect_predictions(
        model=model,
        hybrid=None,
        loader=val_loader,
        starts_base=context.t_train,
        cluster_id_c=cluster_id_c,
        device=device,
    )
    test_base, test_true, test_starts = collect_predictions(
        model=model,
        hybrid=None,
        loader=test_loader,
        starts_base=context.t_val,
        cluster_id_c=cluster_id_c,
        device=device,
    )

    val_knn = test_knn = None
    sources = [part.strip().lower() for part in str(args.prediction_sources).split(",") if part.strip()]
    if "knn" in sources:
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
        val_knn, _, _ = collect_predictions(
            model=model,
            hybrid=hybrid,
            loader=val_loader,
            starts_base=context.t_train,
            cluster_id_c=cluster_id_c,
            device=device,
        )
        test_knn, _, _ = collect_predictions(
            model=model,
            hybrid=hybrid,
            loader=test_loader,
            starts_base=context.t_val,
            cluster_id_c=cluster_id_c,
            device=device,
        )

    lag_groups = parse_int_groups(args.lag_groups)
    alphas = parse_float_list(args.alphas)
    blend_targets = [part.strip().lower() for part in str(args.blend_targets).split(",") if part.strip()]
    rows: list[dict[str, Any]] = []
    for lags in lag_groups:
        val_anchor = observed_history_anchor(context.norm_data_tc, val_starts, context.L, context.H, lags)
        test_anchor = observed_history_anchor(context.norm_data_tc, test_starts, context.L, context.H, lags)
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
            for target in blend_targets:
                if target == "prediction":
                    val_delta = val_anchor - val_pred
                    test_delta = test_anchor - test_pred
                elif target == "base":
                    val_delta = val_anchor - val_base
                    test_delta = test_anchor - test_base
                else:
                    raise ValueError(f"Unsupported blend target: {target}")
                for alpha in alphas:
                    cal_val = val_pred + float(alpha) * val_delta
                    cal_test = test_pred + float(alpha) * test_delta
                    val_mse, val_mae = metrics(cal_val, val_true)
                    test_mse, test_mae = metrics(cal_test, test_true)
                    rows.append(
                        {
                            "prediction_source": source,
                            "blend_target": target,
                            "lags": ",".join(str(v) for v in lags),
                            "alpha": float(alpha),
                            "val_mse": val_mse,
                            "val_mae": val_mae,
                            "test_mse": test_mse,
                            "test_mae": test_mae,
                        }
                    )
    rows = sorted(rows, key=lambda row: (float(row["val_mse"]), float(row["val_mae"])))
    for idx, row in enumerate(rows, start=1):
        row["rank"] = idx
    write_rows(out_dir / "observed_history_anchor_results.csv", rows)
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump({"config_path": str(config_path), "run_dir": str(run_dir), "rows": rows}, f, ensure_ascii=False, indent=2)
    print("Top observed-history anchor rows:")
    for row in rows[:14]:
        print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
