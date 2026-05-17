import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.compare_moe_on_off import (  # noqa: E402
    compute_penalty_scale,
    load_eval_modules,
    load_yaml,
    prepare_data_context,
)
from src.data.windows import WindowTensorDataset, make_strict_windows  # noqa: E402
from src.models.penalties import build_penalty_bank  # noqa: E402
from src.train import eval_loop, reduce_cluster_metric  # noqa: E402
from src.utils.knn_shape import (  # noqa: E402
    KNNShapeConfig,
    ShapeKNNHybrid,
    predict_bank_outputs,
)


RESULT_FIELDS = [
    "variant",
    "adaptive_alpha",
    "k",
    "alpha",
    "val_avg_mae",
    "val_avg_mse",
    "val_delta_mse_pct",
    "val_confidence",
    "val_effective_alpha",
    "test_avg_mae",
    "test_avg_mse",
    "test_delta_mse_pct",
    "test_confidence",
    "test_effective_alpha",
]


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def parse_modes(text: str) -> List[str]:
    modes = []
    for item in str(text).split(","):
        item = item.strip().lower()
        if item:
            modes.append(item)
    if len(modes) == 0:
        raise ValueError("Expected at least one adaptive alpha mode.")
    return modes


def make_loader(x: torch.Tensor, y: torch.Tensor, batch_size: int) -> DataLoader:
    return DataLoader(
        WindowTensorDataset(x, y),
        batch_size=max(1, int(batch_size)),
        shuffle=False,
        num_workers=0,
    )


def make_knn_config(
    base_cfg: dict,
    args: argparse.Namespace,
    adaptive_alpha: str,
    pred_len: int,
) -> KNNShapeConfig:
    knn_cfg = dict(base_cfg.get("knn_hybrid", {}))
    knn_cfg["enable"] = True
    knn_cfg["adaptive_alpha"] = adaptive_alpha
    if args.mode is not None:
        knn_cfg["mode"] = args.mode
    if args.scope is not None:
        knn_cfg["scope"] = args.scope
    if args.bank_split is not None:
        knn_cfg["bank_split"] = args.bank_split
    if args.k is not None:
        knn_cfg["k"] = int(args.k)
    if args.alpha is not None:
        knn_cfg["alpha"] = float(args.alpha)
    if args.confidence_floor is not None:
        knn_cfg["confidence_floor"] = float(args.confidence_floor)
    if args.distance_sharpness is not None:
        knn_cfg["distance_sharpness"] = float(args.distance_sharpness)
    if args.bank_stride is not None:
        knn_cfg["bank_stride"] = int(args.bank_stride)
    return KNNShapeConfig.from_dict(knn_cfg).resolved_for_horizon(pred_len)


def run_eval(
    *,
    split_name: str,
    loader: DataLoader,
    eval_start: int,
    hybrid: ShapeKNNHybrid | None,
    model,
    gate,
    lambda_kp: torch.Tensor,
    penalty_names: List[str],
    penalty_fns: Dict[str, Any],
    cluster_id_c: torch.Tensor,
    K: int,
    moe_cfg: dict,
    device: torch.device,
    select_ranks: List[int],
    channel_count: int,
    mse_weight: float,
    gate_entropy_weight: float,
    gate_balance_weight: float,
    gate_soft_weight: float,
    gate_entropy_target_frac: float,
    penalty_scale: torch.Tensor,
    dynamic_lambda,
    lambda_min_kp: torch.Tensor,
    cluster_weight_k: torch.Tensor,
    pred_residual=None,
    pred_residual_gate=None,
    pred_residual_scale_c=None,
    residual_correction_ch=None,
) -> Dict[str, Any]:
    if hybrid is not None:
        hybrid.reset_confidence_stats()
    loss_k, mse_k, _, _, mae_c, *_ = eval_loop(
        model,
        gate,
        lambda_kp,
        penalty_names,
        penalty_fns,
        loader,
        cluster_id_c,
        K,
        moe_cfg,
        device,
        select_ranks=select_ranks,
        collect_plot=False,
        channel_count=channel_count,
        mse_weight=mse_weight,
        gate_entropy_weight=gate_entropy_weight,
        gate_balance_weight=gate_balance_weight,
        gate_soft_weight=gate_soft_weight,
        gate_entropy_target_frac=gate_entropy_target_frac,
        penalty_scale=penalty_scale,
        dynamic_lambda=dynamic_lambda,
        lambda_min_kp=lambda_min_kp,
        pred_residual=pred_residual,
        pred_residual_gate=pred_residual_gate,
        pred_residual_scale_c=pred_residual_scale_c,
        residual_correction_ch=residual_correction_ch,
        knn_hybrid=hybrid,
        eval_start=eval_start,
    )
    confidence_stats = {} if hybrid is None else (hybrid.get_confidence_stats() or {})
    return {
        f"{split_name}_avg_loss": float(reduce_cluster_metric(loss_k, cluster_weight_k).item()),
        f"{split_name}_avg_mse": float(reduce_cluster_metric(mse_k, cluster_weight_k).item()),
        f"{split_name}_avg_mae": float(mae_c.mean().item()),
        f"{split_name}_confidence": confidence_stats.get("mean_confidence", ""),
        f"{split_name}_effective_alpha": confidence_stats.get("mean_effective_alpha", ""),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compare ShapeKNNHybrid adaptive-alpha variants with the same trained checkpoint."
    )
    ap.add_argument("--config", type=str, required=True, help="Run config used for the checkpoint.")
    ap.add_argument("--run-dir", type=str, default=None, help="Directory containing best_checkpoint.pt.")
    ap.add_argument("--out-csv", type=str, default=None)
    ap.add_argument("--modes", type=str, default="none,agreement,distance,confidence")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--eval-batch-size", type=int, default=None)
    ap.add_argument("--mode", type=str, default=None, choices=["fixed", "rolling"])
    ap.add_argument("--scope", type=str, default=None, choices=["same_channel", "same_cluster"])
    ap.add_argument("--bank-split", type=str, default=None, choices=["train", "pre_test", "history"])
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--confidence-floor", type=float, default=None)
    ap.add_argument("--distance-sharpness", type=float, default=None)
    ap.add_argument("--bank-stride", type=int, default=None)
    args = ap.parse_args()

    config_path = resolve_path(args.config)
    cfg = load_yaml(config_path)
    run_dir = resolve_path(args.run_dir) if args.run_dir else resolve_path(str(cfg["exp"]["out_dir"]))
    checkpoint_path = run_dir / "best_checkpoint.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    out_csv = resolve_path(args.out_csv) if args.out_csv else (run_dir / "knn_shape_variant_compare.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    device_name = args.device or cfg["exp"].get("device", "cpu")
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    batch_size = int(args.eval_batch_size or cfg["train"]["batch_size"])

    context = prepare_data_context(cfg)
    bundle = load_eval_modules(cfg, checkpoint_path, context.K, device)
    model = bundle["model"]
    gate = bundle["gate"]
    dynamic_lambda = bundle["dynamic_lambda"]
    pred_residual = bundle.get("pred_residual")
    lambda_kp = bundle["base_lambda_kp"]
    lambda_min_kp = bundle["lambda_min_kp"]
    penalty_names = bundle["penalty_names"]

    cluster_id_c = context.cluster_id_c.to(device)
    cluster_sizes = torch.bincount(cluster_id_c, minlength=context.K).float().to(device)
    cluster_weight_k = cluster_sizes / cluster_sizes.sum().clamp_min(1.0)

    penalty_fns = build_penalty_bank(penalty_names, jump_thr=float(cfg["penalties"]["jump_threshold"]))
    penalty_scale = compute_penalty_scale(
        xtr_norm=context.xtr_norm,
        ytr_norm=context.ytr_norm,
        penalty_names=penalty_names,
        penalty_fns=penalty_fns,
        pred_len=context.H,
        batch_size=batch_size,
        device=device,
        floor=float(cfg["train"].get("penalty_scale_floor", 1.0e-3)),
    )

    xva, yva = make_strict_windows(context.norm_data_tc, context.L, context.H, context.t_train, context.t_val)
    val_loader = make_loader(xva, yva, batch_size)
    test_loader = make_loader(context.xte_norm, context.yte_norm, batch_size)

    raw_select_ranks = cfg.get("moe", {}).get("select_ranks", [1])
    select_ranks = [int(v) for v in raw_select_ranks]
    moe_cfg = cfg["moe"]

    common_eval_kwargs = {
        "model": model,
        "gate": gate,
        "lambda_kp": lambda_kp,
        "penalty_names": penalty_names,
        "penalty_fns": penalty_fns,
        "cluster_id_c": cluster_id_c,
        "K": context.K,
        "moe_cfg": moe_cfg,
        "device": device,
        "select_ranks": select_ranks,
        "channel_count": len(context.channel_names),
        "mse_weight": float(cfg["train"].get("mse_weight", 1.0)),
        "gate_entropy_weight": float(moe_cfg.get("gate_entropy_weight", 0.0)),
        "gate_balance_weight": float(moe_cfg.get("gate_balance_weight", 0.0)),
        "gate_soft_weight": float(moe_cfg.get("gate_soft_weight", 0.0)),
        "gate_entropy_target_frac": float(moe_cfg.get("gate_entropy_target_frac", 0.0)),
        "penalty_scale": penalty_scale,
        "dynamic_lambda": dynamic_lambda,
        "lambda_min_kp": lambda_min_kp,
        "cluster_weight_k": cluster_weight_k,
        "pred_residual": pred_residual,
    }

    rows: List[Dict[str, Any]] = []
    base_row: Dict[str, Any] = {
        "variant": "base",
        "adaptive_alpha": "",
        "k": 0,
        "alpha": 0.0,
    }
    base_row.update(
        run_eval(
            split_name="val",
            loader=val_loader,
            eval_start=context.t_train,
            hybrid=None,
            **common_eval_kwargs,
        )
    )
    base_row.update(
        run_eval(
            split_name="test",
            loader=test_loader,
            eval_start=context.t_val,
            hybrid=None,
            **common_eval_kwargs,
        )
    )
    rows.append(base_row)

    modes = parse_modes(args.modes)
    cached_bank_preds: Dict[str, torch.Tensor | None] = {}

    def get_bank_for_split(knn_cfg: KNNShapeConfig, split: str):
        if split == "val":
            x_bank = context.xtr_norm
            y_bank = context.ytr_norm
            if knn_cfg.mode == "rolling" and knn_cfg.bank_split in {"pre_test", "history"}:
                x_bank, y_bank = make_strict_windows(context.norm_data_tc, context.L, context.H, 0, context.t_val)
            cache_key = f"val:{x_bank.shape[0]}:{knn_cfg.needs_base_bank_prediction()}"
        else:
            x_bank = context.xtr_norm
            y_bank = context.ytr_norm
            if knn_cfg.bank_split in {"pre_test", "history"}:
                x_bank, y_bank = make_strict_windows(context.norm_data_tc, context.L, context.H, 0, context.t_val)
                if knn_cfg.mode == "rolling" and knn_cfg.bank_split == "history":
                    x_bank, y_bank = make_strict_windows(
                        context.norm_data_tc,
                        context.L,
                        context.H,
                        0,
                        int(context.norm_data_tc.shape[0]),
                    )
            cache_key = f"test:{x_bank.shape[0]}:{knn_cfg.needs_base_bank_prediction()}"

        base_bank_pred = None
        if knn_cfg.needs_base_bank_prediction():
            if cache_key not in cached_bank_preds:
                cached_bank_preds[cache_key] = predict_bank_outputs(
                    model=model,
                    x_bank_ncl=x_bank,
                    cluster_id_c=cluster_id_c,
                    batch_size=max(batch_size, 64),
                    device=device,
                )
            base_bank_pred = cached_bank_preds[cache_key]
        return x_bank, y_bank, base_bank_pred

    for adaptive_alpha in modes:
        knn_cfg = make_knn_config(cfg, args, adaptive_alpha, pred_len=context.H)
        print(
            "Evaluate "
            f"adaptive_alpha={adaptive_alpha}, k={knn_cfg.k}, alpha={knn_cfg.alpha}, "
            f"mode={knn_cfg.mode}, bank_split={knn_cfg.bank_split}, scope={knn_cfg.scope}"
        )

        x_val_bank, y_val_bank, base_val_bank_pred = get_bank_for_split(knn_cfg, "val")
        h_val = ShapeKNNHybrid.fit(
            x_bank_ncl=x_val_bank,
            y_bank_nch=y_val_bank,
            cluster_id_c=cluster_id_c,
            cfg=knn_cfg,
            base_bank_pred_nch=base_val_bank_pred,
        )
        x_test_bank, y_test_bank, base_test_bank_pred = get_bank_for_split(knn_cfg, "test")
        h_test = ShapeKNNHybrid.fit(
            x_bank_ncl=x_test_bank,
            y_bank_nch=y_test_bank,
            cluster_id_c=cluster_id_c,
            cfg=knn_cfg,
            base_bank_pred_nch=base_test_bank_pred,
        )

        row: Dict[str, Any] = {
            "variant": f"knn_{adaptive_alpha}",
            "adaptive_alpha": adaptive_alpha,
            "k": h_test.cfg.k,
            "alpha": h_test.cfg.alpha,
        }
        row.update(
            run_eval(
                split_name="val",
                loader=val_loader,
                eval_start=context.t_train,
                hybrid=h_val,
                **common_eval_kwargs,
            )
        )
        row.update(
            run_eval(
                split_name="test",
                loader=test_loader,
                eval_start=context.t_val,
                hybrid=h_test,
                **common_eval_kwargs,
            )
        )
        rows.append(row)

    base_val_mse = float(rows[0]["val_avg_mse"])
    base_test_mse = float(rows[0]["test_avg_mse"])
    for row in rows:
        row["val_delta_mse"] = float(row["val_avg_mse"]) - base_val_mse
        row["val_delta_mse_pct"] = 100.0 * float(row["val_delta_mse"]) / max(base_val_mse, 1.0e-12)
        row["test_delta_mse"] = float(row["test_avg_mse"]) - base_test_mse
        row["test_delta_mse_pct"] = 100.0 * float(row["test_delta_mse"]) / max(base_test_mse, 1.0e-12)

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in RESULT_FIELDS})

    summary = {
        "config_path": str(config_path),
        "run_dir": str(run_dir),
        "out_csv": str(out_csv),
        "rows": rows,
    }
    summary_path = out_csv.with_suffix(".json")
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Saved KNN variant comparison to: {out_csv}")
    print(f"Saved JSON summary to: {summary_path}")
    print("Rows sorted by val_avg_mse:")
    for row in sorted(rows, key=lambda r: float(r["val_avg_mse"])):
        print(json.dumps({field: row.get(field, "") for field in RESULT_FIELDS}, ensure_ascii=False))


if __name__ == "__main__":
    main()
