from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

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


def resolve_path(path_text: str | Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def parse_lags(text: str) -> list[int]:
    lags = []
    for item in str(text).split(","):
        item = item.strip()
        if item:
            value = int(item)
            if value > 0:
                lags.append(value)
    if not lags:
        raise ValueError("Expected at least one positive history-anchor lag.")
    return lags


def make_loader(x: torch.Tensor, y: torch.Tensor, batch_size: int) -> DataLoader:
    return DataLoader(
        WindowTensorDataset(x, y),
        batch_size=max(1, int(batch_size)),
        shuffle=False,
        num_workers=0,
    )


def evaluate_split(
    *,
    split_name: str,
    loader: DataLoader,
    eval_start: int,
    history_anchor_cfg: dict[str, Any],
    common_kwargs: dict[str, Any],
    cluster_weight_k: torch.Tensor,
) -> dict[str, float]:
    loss_k, mse_k, _, _, mae_c, *_ = eval_loop(
        loader=loader,
        eval_start=int(eval_start),
        knn_hybrid=None,
        history_anchor_cfg=history_anchor_cfg,
        **common_kwargs,
    )
    return {
        f"{split_name}_avg_loss": float(reduce_cluster_metric(loss_k, cluster_weight_k).item()),
        f"{split_name}_avg_mse": float(reduce_cluster_metric(mse_k, cluster_weight_k).item()),
        f"{split_name}_avg_mae": float(mae_c.mean().item()),
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    rows = payload["rows"]
    lines = [
        "# Input-96 Main History Anchor Evaluation",
        "",
        "KNN is disabled for all rows (`knn_hybrid.enable=false`, `knn_hybrid=None`).",
        "",
        "| Variant | Val MSE | Val MAE | Test MSE | Test MAE |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {variant} | {val_avg_mse:.6f} | {val_avg_mae:.6f} | {test_avg_mse:.6f} | {test_avg_mae:.6f} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "History anchor config:",
            "",
            "```json",
            json.dumps(payload["history_anchor"], indent=2),
            "```",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate input-96 main-model history-anchor adapter with KNN off.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--out-dir", default="outputs/input96_history_anchor_eval")
    parser.add_argument("--device", default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--lags", default="96,192,288")
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--blend-target", choices=["prediction", "base"], default="prediction")
    args = parser.parse_args()

    config_path = resolve_path(args.config)
    cfg = load_yaml(config_path)
    cfg.setdefault("knn_hybrid", {})["enable"] = False
    cfg.setdefault("model", {})["history_anchor"] = {
        "enable": True,
        "lags": parse_lags(args.lags),
        "alpha": float(args.alpha),
        "blend_target": str(args.blend_target),
    }
    run_dir = resolve_path(args.run_dir) if args.run_dir else resolve_path(cfg["exp"]["out_dir"])
    checkpoint_path = run_dir / "best_checkpoint.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device_name = args.device or cfg["exp"].get("device", "cpu")
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    context = prepare_data_context(cfg)
    bundle = load_eval_modules(cfg, checkpoint_path, context.K, device)

    cluster_id_c = context.cluster_id_c.to(device)
    cluster_sizes = torch.bincount(cluster_id_c, minlength=context.K).float().to(device)
    cluster_weight_k = cluster_sizes / cluster_sizes.sum().clamp_min(1.0)
    batch_size = int(args.eval_batch_size or cfg["train"]["batch_size"])
    penalty_names = bundle["penalty_names"]
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
    common_kwargs = {
        "model": bundle["model"],
        "gate": bundle["gate"],
        "lambda_kp": bundle["base_lambda_kp"],
        "penalty_names": penalty_names,
        "penalty_fns": penalty_fns,
        "cluster_id_c": cluster_id_c,
        "K": context.K,
        "moe_cfg": cfg["moe"],
        "device": device,
        "select_ranks": [int(v) for v in raw_select_ranks],
        "collect_plot": False,
        "channel_count": len(context.channel_names),
        "mse_weight": float(cfg["train"].get("mse_weight", 1.0)),
        "gate_entropy_weight": float(cfg["moe"].get("gate_entropy_weight", 0.0)),
        "gate_balance_weight": float(cfg["moe"].get("gate_balance_weight", 0.0)),
        "gate_soft_weight": float(cfg["moe"].get("gate_soft_weight", 0.0)),
        "gate_entropy_target_frac": float(cfg["moe"].get("gate_entropy_target_frac", 0.0)),
        "penalty_scale": penalty_scale,
        "dynamic_lambda": bundle["dynamic_lambda"],
        "lambda_min_kp": bundle["lambda_min_kp"],
        "pred_residual": bundle.get("pred_residual"),
        "observed_history_tc": context.norm_data_tc,
        "input_len": context.L,
    }

    variants = [
        ("base_no_anchor", {"enable": False}),
        ("main_history_anchor", cfg["model"]["history_anchor"]),
    ]
    rows = []
    for variant, history_anchor_cfg in variants:
        val = evaluate_split(
            split_name="val",
            loader=val_loader,
            eval_start=context.t_train,
            history_anchor_cfg=history_anchor_cfg,
            common_kwargs=common_kwargs,
            cluster_weight_k=cluster_weight_k,
        )
        test = evaluate_split(
            split_name="test",
            loader=test_loader,
            eval_start=context.t_val,
            history_anchor_cfg=history_anchor_cfg,
            common_kwargs=common_kwargs,
            cluster_weight_k=cluster_weight_k,
        )
        rows.append(
            {
                "variant": variant,
                "val_avg_loss": val["val_avg_loss"],
                "val_avg_mse": val["val_avg_mse"],
                "val_avg_mae": val["val_avg_mae"],
                "test_avg_loss": test["test_avg_loss"],
                "test_avg_mse": test["test_avg_mse"],
                "test_avg_mae": test["test_avg_mae"],
            }
        )

    payload = {
        "config_path": str(config_path),
        "run_dir": str(run_dir),
        "checkpoint_path": str(checkpoint_path),
        "device": str(device),
        "input_len": int(context.L),
        "pred_len": int(context.H),
        "knn_hybrid_enable": False,
        "history_anchor": cfg["model"]["history_anchor"],
        "rows": rows,
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown(out_dir / "summary.md", payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
