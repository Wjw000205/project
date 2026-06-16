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


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDS})


def read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


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
    return cfg


def base_common(
    cfg: dict[str, Any],
    *,
    dataset: str,
    horizon: int,
    hidden_dim: int,
    batch_size: int,
    device: str,
    out_dir: Path,
    name: str,
) -> dict[str, Any]:
    cfg = localize(cfg, out_dir=out_dir, name=name, device=device)
    cfg.setdefault("data", {})["csv_path"] = f"data/{dataset}.csv"
    cfg["data"]["date_col"] = 0
    cfg["data"]["train_ratio"] = 0.7
    cfg["data"]["val_ratio"] = 0.1
    cfg["data"]["test_ratio"] = 0.2
    cfg["data"]["max_rows"] = 0
    cfg.setdefault("window", {})["input_len"] = 96
    cfg["window"]["pred_len"] = int(horizon)
    cfg["window"]["lazy"] = True
    cfg.setdefault("normalize", {})["global_zscore"] = True
    cfg["normalize"]["train_only"] = True
    cfg.setdefault("cluster", {})["train_only"] = True
    cfg.setdefault("model", {})["predictor"] = "context_channel_head_mlp"
    cfg["model"]["hidden_dim"] = int(hidden_dim)
    cfg["model"]["dropout"] = 0.0
    cfg.setdefault("train", {})["batch_size"] = int(batch_size)
    cfg["train"]["lr"] = 1.0e-3
    cfg["train"]["mse_weight"] = 0.5
    cfg["train"]["selection_metric"] = "val_mae"
    cfg["train"].setdefault("mae_objective", {}).update(
        {"enable": True, "kind": "l1", "weight": 1.5, "warmup_epochs": 3}
    )
    cfg["train"].setdefault("lr_scheduler", {}).update(
        {"name": "plateau", "factor": 0.5, "patience": 2, "min_lr": 1.0e-6}
    )
    cfg.setdefault("early_stop", {}).update({"patience": 4, "min_delta": 1.0e-6})
    cfg["plot"] = {"enable": False}
    cfg["portrait"] = {"enable": False, "out_dir": str(out_dir / "cluster_portraits")}
    cfg.setdefault("knn_hybrid", {})["enable"] = False
    return cfg


def make_backbone_cfg(base: dict[str, Any], args: argparse.Namespace, out_dir: Path) -> dict[str, Any]:
    cfg = base_common(
        copy.deepcopy(base),
        dataset=args.dataset,
        horizon=args.horizon,
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        device=args.device,
        out_dir=out_dir,
        name=f"{args.dataset}_H{args.horizon}_cch_h{args.hidden_dim}_backbone",
    )
    cfg.setdefault("moe", {})["enable"] = False
    cfg["train"]["epochs"] = int(args.backbone_epochs)
    cfg["memory"] = {
        "enable": False,
        "save_checkpoint": True,
        "path": str(out_dir / "cluster_memory.pt"),
        "checkpoint_path": str(out_dir / "best_checkpoint.pt"),
    }
    cfg["calibration"] = {"enable": False}
    return cfg


def make_moe_cfg(base: dict[str, Any], args: argparse.Namespace, out_dir: Path, checkpoint: Path, variant: str) -> dict[str, Any]:
    cfg = base_common(
        copy.deepcopy(base),
        dataset=args.dataset,
        horizon=args.horizon,
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        device=args.device,
        out_dir=out_dir,
        name=f"{args.dataset}_H{args.horizon}_cch_h{args.hidden_dim}_{variant}",
    )
    penalties = list(cfg.get("penalties", {}).get("enabled", ["amp_under", "delta", "diff_amp", "direction"]))
    cfg.setdefault("penalties", {})["enabled"] = penalties
    cfg.setdefault("moe", {})["enable"] = True
    cfg["moe"]["freeze_backbone"] = True
    cfg["moe"]["lambda_init"] = {name: 0.0 for name in penalties}
    cfg["moe"]["lambda_min"] = {name: 0.0 for name in penalties}
    cfg["moe"]["lambda_schedule"] = {name: "none" for name in penalties}
    cfg["moe"].setdefault("dynamic_lambda", {})["enable"] = False
    cfg["moe"].setdefault("learnable_lambda", {})["enable"] = False
    pred_side = cfg["moe"].setdefault("pred_side_residual", {})
    if bool(args.pred_residual):
        pred_side.update(
            {
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
                "selection_holdout_fraction": float(args.pred_selection_holdout_fraction),
                "selection_holdout_min_windows": int(args.pred_selection_holdout_min_windows),
            }
        )
    else:
        pred_side["enable"] = False
    cfg["moe"]["train_stat_anchor_expert"] = {
        "enable": True,
        "period": int(args.period),
        "alpha": 0.0,
        "mode": "phase_mean",
        "reference": "last",
        "blend_target": "prediction",
        "scale_selection": {"enable": True, "metric": "mse", "max_scale": float(args.stat_max), "steps": int(args.stat_steps)},
    }
    cfg["moe"]["train_residual_anchor_expert"] = {
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
    cfg["train"]["epochs"] = int(args.moe_epochs)
    cfg["train"]["lr"] = float(args.moe_lr)
    cfg["train"]["mse_weight"] = float(args.moe_mse_weight)
    cfg["train"].setdefault("mae_objective", {})["weight"] = float(args.moe_mae_weight)
    cfg["train"]["selection_metric"] = "val_mse"
    cfg.setdefault("early_stop", {}).update(
        {"patience": int(args.moe_patience), "min_delta": float(args.moe_min_delta)}
    )
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
    }
    return cfg


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


def summarize(
    *,
    phase: str,
    dataset: str,
    horizon: int,
    variant: str,
    cfg_path: Path,
    out_dir: Path,
    returncode: int,
    output: str,
    backbone: dict[str, float] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "phase": phase,
        "dataset": dataset,
        "horizon": int(horizon),
        "variant": variant,
        "status": "ok" if returncode == 0 else "error",
        "config_path": str(cfg_path),
        "out_dir": str(out_dir),
        "returncode": returncode,
        "error": "" if returncode == 0 else output[-3000:],
    }
    summary_path = out_dir / "run_summary.json"
    if not summary_path.exists():
        row["status"] = "error"
        row["error"] = row.get("error") or "run_summary.json missing"
        return row
    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)
    val = summary.get("val") or {}
    test = summary.get("test") or {}
    row.update({"val_mse": val.get("avg_mse", ""), "val_mae": val.get("avg_mae", ""), "test_mse": test.get("avg_mse", ""), "test_mae": test.get("avg_mae", "")})
    if backbone is not None and row["test_mse"] != "":
        bmse = float(backbone["test_mse"])
        bmae = float(backbone["test_mae"])
        mse = float(row["test_mse"])
        mae = float(row["test_mae"])
        row.update({"backbone_mse": bmse, "backbone_mae": bmae, "mse_gain_pct": (bmse - mse) / bmse * 100.0, "mae_gain_pct": (bmae - mae) / bmae * 100.0})
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description="Strict CCH backbone then frozen p288 MoE for a PEMS cell.")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--horizon", type=int, required=True)
    ap.add_argument("--hidden-dim", type=int, default=160)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--backbone-epochs", type=int, default=36)
    ap.add_argument("--period", type=int, default=288)
    ap.add_argument("--stat-max", type=float, default=0.2)
    ap.add_argument("--stat-steps", type=int, default=9)
    ap.add_argument("--resid-max", type=float, default=0.8)
    ap.add_argument("--resid-steps", type=int, default=33)
    ap.add_argument("--segments", type=int, default=4)
    ap.add_argument("--calibration", choices=["", "median", "mean"], default="")
    ap.add_argument("--calibration-shrink", type=float, default=1.0)
    ap.add_argument("--moe-epochs", type=int, default=1)
    ap.add_argument("--moe-lr", type=float, default=0.0)
    ap.add_argument("--moe-mse-weight", type=float, default=0.5)
    ap.add_argument("--moe-mae-weight", type=float, default=1.5)
    ap.add_argument("--moe-patience", type=int, default=4)
    ap.add_argument("--moe-min-delta", type=float, default=1.0e-6)
    ap.add_argument("--pred-residual", action="store_true")
    ap.add_argument("--pred-hidden", type=int, default=16)
    ap.add_argument("--pred-init-alpha", type=float, default=-4.0)
    ap.add_argument("--pred-alpha-scale", type=float, default=0.2)
    ap.add_argument("--pred-feature-mode", choices=["legacy", "safe_augmented"], default="safe_augmented")
    ap.add_argument("--pred-residual-clip", type=float, default=0.0)
    ap.add_argument("--pred-intervention", action="store_true")
    ap.add_argument("--pred-intervention-init", type=float, default=-2.0)
    ap.add_argument("--pred-specialization-weight", type=float, default=0.0)
    ap.add_argument("--pred-norm-weight", type=float, default=1.0e-4)
    ap.add_argument("--pred-intervention-weight", type=float, default=0.0)
    ap.add_argument("--pred-detach-routed-penalty-pred", action="store_true")
    ap.add_argument("--pred-selection-policy", default="none")
    ap.add_argument("--pred-selection-scale-min", type=float, default=0.0)
    ap.add_argument("--pred-selection-scale-max", type=float, default=1.0)
    ap.add_argument("--pred-selection-scale-steps", type=int, default=21)
    ap.add_argument("--pred-selection-min-abs-improvement", type=float, default=0.0)
    ap.add_argument("--pred-selection-min-rel-improvement", type=float, default=0.0)
    ap.add_argument("--pred-selection-holdout-fraction", type=float, default=0.4)
    ap.add_argument("--pred-selection-holdout-min-windows", type=int, default=256)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--out-root", type=Path, default=None)
    ap.add_argument("--rerun", action="store_true")
    args = ap.parse_args()

    if args.out_root is None:
        args.out_root = ROOT / "outputs" / f"codex_table_target_20260615_{args.dataset.lower()}_h{args.horizon}_h{args.hidden_dim}_strict_backbone_moe"
    base = read_yaml(ROOT / "configs" / f"{args.dataset}_H{args.horizon}.yaml")
    results_path = args.out_root / "results.csv"
    rows: list[dict[str, Any]] = read_rows(results_path)
    completed = {(row.get("phase"), row.get("variant")) for row in rows if row.get("status") == "ok"}

    backbone_variant = f"cch_h{args.hidden_dim}_backbone"
    backbone_dir = args.out_root / "runs" / backbone_variant
    backbone_cfg = args.out_root / "configs" / f"{backbone_variant}.yaml"
    write_yaml(backbone_cfg, make_backbone_cfg(base, args, backbone_dir))
    if (backbone_dir / "run_summary.json").exists() and not args.rerun:
        print(f"[reuse] {backbone_variant}", flush=True)
        rc, out = 0, ""
    else:
        print(f"[run] {backbone_variant}", flush=True)
        rc, out = run_train(args.python, backbone_cfg, backbone_dir)
    backbone_row = summarize(phase="backbone", dataset=args.dataset, horizon=args.horizon, variant=backbone_variant, cfg_path=backbone_cfg, out_dir=backbone_dir, returncode=rc, output=out)
    if ("backbone", backbone_variant) not in completed or args.rerun:
        rows.append(backbone_row)
    write_rows(results_path, rows)
    if backbone_row["status"] != "ok":
        raise SystemExit("backbone failed")

    checkpoint = backbone_dir / "best_checkpoint.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    backbone_metrics = {"test_mse": float(backbone_row["test_mse"]), "test_mae": float(backbone_row["test_mae"])}

    moe_variant = f"p288_h{args.hidden_dim}_stat{int(args.stat_max*1000):03d}_resid{int(args.resid_max*1000):03d}_seg{args.segments}"
    if args.calibration:
        moe_variant += f"_cal{args.calibration}_s{int(args.calibration_shrink*1000):03d}"
    if args.pred_residual:
        lr_tag = f"{float(args.moe_lr):.0e}".replace("+", "").replace("-", "m").replace(".", "p")
        moe_variant += (
            f"_prh{int(args.pred_hidden)}_a{int(float(args.pred_alpha_scale) * 1000):03d}"
            f"_ep{int(args.moe_epochs)}_lr{lr_tag}_{str(args.pred_selection_policy).lower()}"
        )
    moe_dir = args.out_root / "runs" / moe_variant
    moe_cfg = args.out_root / "configs" / f"{moe_variant}.yaml"
    write_yaml(moe_cfg, make_moe_cfg(base, args, moe_dir, checkpoint, moe_variant))
    if (moe_dir / "run_summary.json").exists() and not args.rerun:
        print(f"[reuse] {moe_variant}", flush=True)
        rc, out = 0, ""
    else:
        print(f"[run] {moe_variant}", flush=True)
        rc, out = run_train(args.python, moe_cfg, moe_dir)
    if ("moe", moe_variant) not in completed or args.rerun:
        rows.append(summarize(phase="moe", dataset=args.dataset, horizon=args.horizon, variant=moe_variant, cfg_path=moe_cfg, out_dir=moe_dir, returncode=rc, output=out, backbone=backbone_metrics))
    write_rows(results_path, rows)


if __name__ == "__main__":
    main()
