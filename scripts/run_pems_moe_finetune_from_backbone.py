from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]

FIELDS = [
    "phase",
    "dataset",
    "horizon",
    "variant",
    "status",
    "test_mse",
    "test_mae",
    "val_mse",
    "val_mae",
    "backbone_mse",
    "backbone_mae",
    "mse_gain_pct",
    "mae_gain_pct",
    "config_path",
    "out_dir",
    "returncode",
    "error",
]


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_yaml(path: Path, cfg: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=False)


def read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDS})


def run_train(py: str, cfg_path: Path, out_dir: Path) -> tuple[int, str]:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    proc = subprocess.run(
        [py, "-u", "-m", "src.train", "--config", str(cfg_path)],
        cwd=ROOT,
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "stdout.log").write_text(proc.stdout, encoding="utf-8", errors="replace")
    return int(proc.returncode), proc.stdout


def summary_metrics(path: Path) -> dict[str, float]:
    with path.open("r", encoding="utf-8") as f:
        summary = json.load(f)
    test = summary.get("test") or {}
    val = summary.get("val") or {}
    return {
        "test_mse": float(test["avg_mse"]),
        "test_mae": float(test["avg_mae"]),
        "val_mse": float(val["avg_mse"]),
        "val_mae": float(val["avg_mae"]),
    }


def summarize_run(
    *,
    dataset: str,
    horizon: int,
    variant: str,
    cfg_path: Path,
    out_dir: Path,
    returncode: int,
    output: str,
    backbone: dict[str, float],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "phase": "moe",
        "dataset": dataset,
        "horizon": int(horizon),
        "variant": variant,
        "status": "ok" if returncode == 0 else "error",
        "config_path": str(cfg_path),
        "out_dir": str(out_dir),
        "returncode": returncode,
        "error": "" if returncode == 0 else output[-3000:],
        "backbone_mse": backbone["test_mse"],
        "backbone_mae": backbone["test_mae"],
    }
    summary_path = out_dir / "run_summary.json"
    if not summary_path.exists():
        row["status"] = "error"
        row["error"] = row.get("error") or "run_summary.json missing"
        return row
    metrics = summary_metrics(summary_path)
    row.update(
        {
            "test_mse": metrics["test_mse"],
            "test_mae": metrics["test_mae"],
            "val_mse": metrics["val_mse"],
            "val_mae": metrics["val_mae"],
            "mse_gain_pct": (backbone["test_mse"] - metrics["test_mse"]) / backbone["test_mse"] * 100.0,
            "mae_gain_pct": (backbone["test_mae"] - metrics["test_mae"]) / backbone["test_mae"] * 100.0,
        }
    )
    return row


def localize(cfg: dict[str, Any], *, out_dir: Path, name: str, device: str) -> dict[str, Any]:
    cfg = copy.deepcopy(cfg)
    cfg.setdefault("exp", {})
    cfg["exp"]["name"] = name
    cfg["exp"]["out_dir"] = str(out_dir)
    cfg["exp"]["device"] = device
    cfg.setdefault("corr", {})["save_path"] = str(out_dir / "corr.npy")
    cfg.setdefault("portrait", {})["out_dir"] = str(out_dir / "cluster_portraits")
    cfg.setdefault("knn_hybrid", {})["path"] = str(out_dir / "knn_shape_bank.pt")
    cfg.setdefault("memory", {})["path"] = str(out_dir / "cluster_memory.pt")
    cfg.setdefault("memory", {})["checkpoint_path"] = str(out_dir / "best_checkpoint.pt")
    cfg.setdefault("eval", {})["skip_test"] = False
    cfg.setdefault("plot", {})["enable"] = False
    return cfg


def make_cfg(base: dict[str, Any], args: argparse.Namespace, out_dir: Path, checkpoint: Path, variant: str) -> dict[str, Any]:
    cfg = localize(
        base,
        out_dir=out_dir,
        name=f"{args.dataset}_H{args.horizon}_{variant}",
        device=args.device,
    )
    cfg.setdefault("data", {})["csv_path"] = f"data/{args.dataset}.csv"
    cfg.setdefault("window", {})["input_len"] = 96
    cfg["window"]["pred_len"] = int(args.horizon)
    cfg["window"]["lazy"] = True
    cfg.setdefault("normalize", {})["global_zscore"] = True
    cfg["normalize"]["train_only"] = True
    cfg.setdefault("cluster", {})["train_only"] = True
    cfg.setdefault("knn_hybrid", {})["enable"] = False

    penalties = list(cfg.get("penalties", {}).get("enabled", ["amp_under", "delta", "diff_amp", "direction"]))
    cfg.setdefault("penalties", {})["enabled"] = penalties
    moe = cfg.setdefault("moe", {})
    moe["enable"] = True
    moe["freeze_backbone"] = True
    moe["topk"] = int(moe.get("topk", 1))
    moe["select_ranks"] = [1]
    moe["lambda_init"] = {name: 0.0 for name in penalties}
    moe["lambda_min"] = {name: 0.0 for name in penalties}
    moe["lambda_schedule"] = {name: "none" for name in penalties}
    moe.setdefault("dynamic_lambda", {})["enable"] = False
    moe.setdefault("learnable_lambda", {})["enable"] = False
    moe["gate_temperature"] = float(args.gate_temperature)
    moe["gate_noise_std"] = float(args.gate_noise_std)
    moe["gate_soft_weight"] = float(args.gate_soft_weight)
    moe["allow_skip"] = bool(args.allow_skip)
    moe["skip_cost"] = float(args.skip_cost)
    moe["skip_init_bias"] = float(args.skip_init_bias)
    moe["router_mode"] = str(args.router_mode)
    moe["router_penalty_context_weight"] = float(args.router_penalty_context_weight)
    moe["router_detach_penalty_context"] = bool(args.router_detach_penalty_context)
    moe["train_stat_anchor_expert"] = {
        "enable": True,
        "period": int(args.period),
        "alpha": 0.0,
        "mode": "phase_mean",
        "reference": "last",
        "blend_target": "prediction",
        "scale_selection": {
            "enable": True,
            "metric": "mse",
            "max_scale": float(args.stat_max),
            "steps": int(args.stat_steps),
        },
    }
    moe["train_residual_anchor_expert"] = {
        "enable": True,
        "period": int(args.period),
        "alpha": 0.0,
        "blend_target": "prediction",
        "scale_selection": {
            "enable": True,
            "metric": "mse",
            "max_scale": float(args.resid_max),
            "steps": int(args.resid_steps),
            "horizon_segments": int(args.segments),
        },
    }
    moe["pred_side_residual"] = {
        "enable": True,
        "corrector_hidden": int(args.pred_hidden),
        "init_alpha": float(args.pred_init_alpha),
        "alpha_scale": float(args.pred_alpha_scale),
        "use_y_base_input": True,
        "feature_mode": str(args.pred_feature_mode),
        "residual_clip": float(args.pred_residual_clip),
        "intervention_enable": bool(args.pred_intervention),
        "intervention_init": float(args.pred_intervention_init),
        "specialization_weight": float(args.pred_specialization_weight),
        "norm_weight": float(args.pred_norm_weight),
        "intervention_weight": float(args.pred_intervention_weight),
        "detach_routed_penalty_pred": bool(args.pred_detach_routed_penalty_pred),
        "selection_policy": str(args.pred_selection_policy),
        "selection_scale_min": float(args.pred_selection_scale_min),
        "selection_scale_max": float(args.pred_selection_scale_max),
        "selection_scale_steps": int(args.pred_selection_scale_steps),
        "selection_min_abs_improvement": float(args.pred_selection_min_abs_improvement),
        "selection_min_rel_improvement": float(args.pred_selection_min_rel_improvement),
    }
    cfg.setdefault("train", {})
    cfg["train"]["epochs"] = int(args.moe_epochs)
    cfg["train"]["batch_size"] = int(args.batch_size)
    cfg["train"]["lr"] = float(args.moe_lr)
    cfg["train"]["mse_weight"] = float(args.moe_mse_weight)
    cfg["train"].setdefault("mae_objective", {}).update(
        {"enable": True, "kind": "l1", "weight": float(args.moe_mae_weight), "warmup_epochs": 1}
    )
    cfg["train"]["selection_metric"] = "val_mse"
    cfg["train"].setdefault("lr_scheduler", {}).update(
        {"name": "plateau", "factor": 0.5, "patience": 2, "min_lr": 1.0e-6}
    )
    cfg.setdefault("early_stop", {}).update({"patience": int(args.moe_patience), "min_delta": 1.0e-6})
    cfg["calibration"] = (
        {"enable": False}
        if args.calibration == ""
        else {"enable": True, "method": args.calibration, "shrink": float(args.calibration_shrink), "max_abs": 0.0}
    )
    cfg["memory"] = {
        "enable": False,
        "save_checkpoint": False,
        "path": str(out_dir / "cluster_memory.pt"),
        "checkpoint_path": str(out_dir / "best_checkpoint.pt"),
    }
    cfg["finetune"] = {
        "enable": True,
        "checkpoint_path": str(checkpoint),
        "strict_window": True,
        "strict_model": True,
        "cluster_map": "index",
        "load_model": True,
        "load_gate": False,
        "load_dynamic_lambda": False,
        "load_learnable_lambda": False,
        "load_pred_residual": False,
    }
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser(description="Fine-tune frozen MoE modules from an existing PEMS backbone checkpoint.")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--horizon", type=int, required=True)
    ap.add_argument("--backbone-config", type=Path, required=True)
    ap.add_argument("--backbone-checkpoint", type=Path, required=True)
    ap.add_argument("--backbone-summary", type=Path, default=None)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--period", type=int, default=288)
    ap.add_argument("--stat-max", type=float, default=0.2)
    ap.add_argument("--stat-steps", type=int, default=9)
    ap.add_argument("--resid-max", type=float, default=0.8)
    ap.add_argument("--resid-steps", type=int, default=33)
    ap.add_argument("--segments", type=int, default=4)
    ap.add_argument("--calibration", choices=["", "median", "mean"], default="")
    ap.add_argument("--calibration-shrink", type=float, default=1.0)
    ap.add_argument("--moe-epochs", type=int, default=8)
    ap.add_argument("--moe-lr", type=float, default=3.0e-4)
    ap.add_argument("--moe-mse-weight", type=float, default=1.0)
    ap.add_argument("--moe-mae-weight", type=float, default=0.2)
    ap.add_argument("--moe-patience", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--pred-hidden", type=int, default=16)
    ap.add_argument("--pred-init-alpha", type=float, default=-5.0)
    ap.add_argument("--pred-alpha-scale", type=float, default=0.1)
    ap.add_argument("--pred-feature-mode", choices=["legacy", "safe_augmented"], default="safe_augmented")
    ap.add_argument("--pred-residual-clip", type=float, default=0.0)
    ap.add_argument("--pred-intervention", action="store_true")
    ap.add_argument("--pred-intervention-init", type=float, default=-2.0)
    ap.add_argument("--pred-specialization-weight", type=float, default=0.0)
    ap.add_argument("--pred-norm-weight", type=float, default=1.0e-4)
    ap.add_argument("--pred-intervention-weight", type=float, default=0.0)
    ap.add_argument("--pred-detach-routed-penalty-pred", action="store_true")
    ap.add_argument("--pred-selection-policy", default="val_mse_scale")
    ap.add_argument("--pred-selection-scale-min", type=float, default=0.0)
    ap.add_argument("--pred-selection-scale-max", type=float, default=0.5)
    ap.add_argument("--pred-selection-scale-steps", type=int, default=11)
    ap.add_argument("--pred-selection-min-abs-improvement", type=float, default=0.0)
    ap.add_argument("--pred-selection-min-rel-improvement", type=float, default=0.0)
    ap.add_argument("--gate-temperature", type=float, default=1.2)
    ap.add_argument("--gate-noise-std", type=float, default=0.0)
    ap.add_argument("--gate-soft-weight", type=float, default=0.0)
    ap.add_argument("--allow-skip", action="store_true")
    ap.add_argument("--skip-cost", type=float, default=0.15)
    ap.add_argument("--skip-init-bias", type=float, default=-2.0)
    ap.add_argument("--router-mode", choices=["learned", "penalty_context", "penalty_only"], default="learned")
    ap.add_argument("--router-penalty-context-weight", type=float, default=0.0)
    ap.add_argument("--router-detach-penalty-context", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--rerun", action="store_true")
    args = ap.parse_args()

    cfg_path = args.backbone_config if args.backbone_config.is_absolute() else ROOT / args.backbone_config
    checkpoint = args.backbone_checkpoint if args.backbone_checkpoint.is_absolute() else ROOT / args.backbone_checkpoint
    summary_path = args.backbone_summary
    if summary_path is None:
        summary_path = checkpoint.parent / "run_summary.json"
    elif not summary_path.is_absolute():
        summary_path = ROOT / summary_path
    backbone = summary_metrics(summary_path)

    rows = read_rows(args.out_root / "results.csv")
    completed = {(row.get("phase"), row.get("variant")) for row in rows if row.get("status") == "ok"}
    base = read_yaml(cfg_path)
    lr_tag = f"{float(args.moe_lr):.0e}".replace("+", "").replace("-", "m").replace(".", "p")
    variant = (
        f"p288_stat{int(args.stat_max * 1000):03d}_resid{int(args.resid_max * 1000):03d}_seg{args.segments}"
        f"_prh{int(args.pred_hidden)}_a{int(float(args.pred_alpha_scale) * 1000):03d}"
        f"_ep{int(args.moe_epochs)}_lr{lr_tag}"
        f"_{args.pred_selection_policy}"
    )
    if args.calibration:
        variant += f"_cal{args.calibration}_s{int(args.calibration_shrink * 1000):03d}"
    out_dir = args.out_root / "runs" / variant
    cfg_out = args.out_root / "configs" / f"{variant}.yaml"
    write_yaml(cfg_out, make_cfg(base, args, out_dir, checkpoint, variant))
    if (out_dir / "run_summary.json").exists() and not args.rerun:
        print(f"[reuse] {variant}", flush=True)
        rc, output = 0, ""
    else:
        print(f"[run] {variant}", flush=True)
        rc, output = run_train(args.python, cfg_out, out_dir)
    if ("moe", variant) not in completed or args.rerun:
        rows.append(
            summarize_run(
                dataset=args.dataset,
                horizon=args.horizon,
                variant=variant,
                cfg_path=cfg_out,
                out_dir=out_dir,
                returncode=rc,
                output=output,
                backbone=backbone,
            )
        )
    write_rows(args.out_root / "results.csv", rows)


if __name__ == "__main__":
    main()
