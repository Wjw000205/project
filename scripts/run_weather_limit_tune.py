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

BASE_CONFIGS = {
    "h96_resid080_stat040": (
        ROOT
        / "outputs"
        / "fresh_input_len96_20260614_weather_h96_mae_p144_moe_probe"
        / "configs"
        / "p144_stat030_resid060_cal000.yaml"
    ),
    "h96_resid100_stat050": (
        ROOT
        / "outputs"
        / "fresh_input_len96_20260614_weather_h96_mae_p144_moe_probe"
        / "configs"
        / "p144_stat030_resid060_cal000.yaml"
    ),
    "h192_mae25_wd1e4": (
        ROOT
        / "outputs"
        / "fresh_input_len96_20260614_weather_h192_mae_arch_probe"
        / "configs"
        / "cch_h320_chadapt_r4_s005_mse03_mae20_valmae.yaml"
    ),
    "h192_mae30": (
        ROOT
        / "outputs"
        / "fresh_input_len96_20260614_weather_h192_mae_arch_probe"
        / "configs"
        / "cch_h320_chadapt_r4_s005_mse03_mae20_valmae.yaml"
    ),
    "h336_mae20": (
        ROOT
        / "outputs"
        / "fresh_input_len96_20260614_weather_h336_cch_backbone_probe"
        / "configs"
        / "cch_h128_do005_wd5e4_mae07_basis_r8.yaml"
    ),
    "h336_h160_mae20": (
        ROOT
        / "outputs"
        / "fresh_input_len96_20260614_weather_h336_cch_backbone_probe"
        / "configs"
        / "cch_h128_do005_wd5e4_mae07_basis_r8.yaml"
    ),
    "h720_mae15": ROOT / "outputs" / "main_table_weather_strict_retry" / "configs" / "weather_pred_720.yaml",
    "h720_h160_mae15": ROOT / "outputs" / "main_table_weather_strict_retry" / "configs" / "weather_pred_720.yaml",
    "h96_bias_freeze": (
        ROOT
        / "outputs"
        / "codex_table_target_20260614"
        / "weather_limits_utf8"
        / "configs"
        / "H96_h96_resid100_stat050.yaml"
    ),
    "h192_bias_freeze": (
        ROOT
        / "outputs"
        / "codex_table_target_20260614"
        / "weather_limits_utf8"
        / "configs"
        / "H192_h192_mae25_wd1e4.yaml"
    ),
    "h192_bias_unfreeze": (
        ROOT
        / "outputs"
        / "codex_table_target_20260614"
        / "weather_limits_utf8"
        / "configs"
        / "H192_h192_mae25_wd1e4.yaml"
    ),
    "h336_bias_freeze": (
        ROOT
        / "outputs"
        / "codex_table_target_20260614"
        / "weather_limits_utf8"
        / "configs"
        / "H336_h336_mae20.yaml"
    ),
    "h336_bias_unfreeze": (
        ROOT
        / "outputs"
        / "codex_table_target_20260614"
        / "weather_limits_utf8"
        / "configs"
        / "H336_h336_mae20.yaml"
    ),
}

BIAS_CHECKPOINTS = {
    "h96_bias_freeze": (
        ROOT
        / "outputs"
        / "fresh_input_len96_20260614_weather_h96_mae_arch_refine2"
        / "runs"
        / "r4_s005_mse03_mae20_valmae"
        / "best_checkpoint.pt"
    ),
    "h192_bias_freeze": (
        ROOT
        / "outputs"
        / "codex_table_target_20260614"
        / "weather_limits_utf8"
        / "runs"
        / "H192"
        / "h192_mae25_wd1e4"
        / "best_checkpoint.pt"
    ),
    "h192_bias_unfreeze": (
        ROOT
        / "outputs"
        / "codex_table_target_20260614"
        / "weather_limits_utf8"
        / "runs"
        / "H192"
        / "h192_mae25_wd1e4"
        / "best_checkpoint.pt"
    ),
    "h336_bias_freeze": (
        ROOT
        / "outputs"
        / "codex_table_target_20260614"
        / "weather_limits_utf8"
        / "runs"
        / "H336"
        / "h336_mae20"
        / "best_checkpoint.pt"
    ),
    "h336_bias_unfreeze": (
        ROOT
        / "outputs"
        / "codex_table_target_20260614"
        / "weather_limits_utf8"
        / "runs"
        / "H336"
        / "h336_mae20"
        / "best_checkpoint.pt"
    ),
}


FIELDS = [
    "variant",
    "status",
    "horizon",
    "val_mse",
    "val_mae",
    "test_mse",
    "test_mae",
    "best_epoch",
    "total_sec",
    "avg_epoch_sec",
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


def infer_horizon(cfg: dict[str, Any]) -> int:
    return int((cfg.get("window") or {}).get("pred_len"))


def localize_paths(cfg: dict[str, Any], *, out_dir: Path, variant: str, device: str) -> dict[str, Any]:
    cfg = copy.deepcopy(cfg)
    cfg.setdefault("exp", {})
    cfg["exp"]["name"] = f"weather_limit_{variant}"
    cfg["exp"]["out_dir"] = str(out_dir)
    cfg["exp"]["device"] = device
    cfg.setdefault("corr", {})["save_path"] = str(out_dir / "corr.npy")
    cfg.setdefault("portrait", {})["out_dir"] = str(out_dir / "cluster_portraits")
    cfg.setdefault("memory", {})["path"] = str(out_dir / "cluster_memory.pt")
    cfg.setdefault("memory", {})["checkpoint_path"] = str(out_dir / "best_checkpoint.pt")
    cfg.setdefault("eval", {})["skip_test"] = False
    cfg.setdefault("plot", {})["enable"] = False
    cfg.setdefault("portrait", {})["enable"] = False
    cfg.setdefault("memory", {})["enable"] = False
    return cfg


def apply_variant(cfg: dict[str, Any], variant: str) -> dict[str, Any]:
    cfg = copy.deepcopy(cfg)
    train = cfg.setdefault("train", {})
    early = cfg.setdefault("early_stop", {})
    model = cfg.setdefault("model", {})
    mae = train.setdefault("mae_objective", {})

    if variant == "h96_resid080_stat040":
        train["epochs"] = 1
        cfg["moe"]["train_stat_anchor_expert"]["scale_selection"]["max_scale"] = 0.4
        cfg["moe"]["train_residual_anchor_expert"]["scale_selection"]["max_scale"] = 0.8
    elif variant == "h96_resid100_stat050":
        train["epochs"] = 1
        cfg["moe"]["train_stat_anchor_expert"]["scale_selection"]["max_scale"] = 0.5
        cfg["moe"]["train_residual_anchor_expert"]["scale_selection"]["max_scale"] = 1.0
    elif variant == "h192_mae25_wd1e4":
        train.update({"epochs": 60, "mse_weight": 0.2, "selection_metric": "val_mae", "weight_decay": 1.0e-4})
        mae.update({"enable": True, "kind": "l1", "weight": 2.5, "warmup_epochs": 0})
        early["patience"] = 7
    elif variant == "h192_mae30":
        train.update({"epochs": 60, "mse_weight": 0.1, "selection_metric": "val_mae"})
        mae.update({"enable": True, "kind": "l1", "weight": 3.0, "warmup_epochs": 0})
        early["patience"] = 7
    elif variant == "h336_mae20":
        train.update({"epochs": 45, "mse_weight": 0.3, "selection_metric": "val_mae"})
        mae.update({"enable": True, "kind": "l1", "weight": 2.0, "warmup_epochs": 0})
        early["patience"] = 7
    elif variant == "h336_h160_mae20":
        model["hidden_dim"] = 160
        train.update({"epochs": 45, "mse_weight": 0.3, "selection_metric": "val_mae"})
        mae.update({"enable": True, "kind": "l1", "weight": 2.0, "warmup_epochs": 0})
        early["patience"] = 7
    elif variant == "h720_mae15":
        train.update({"epochs": 30, "mse_weight": 0.5, "selection_metric": "val_mae"})
        mae.update({"enable": True, "kind": "l1", "weight": 1.5, "warmup_epochs": 0})
        early["patience"] = 7
    elif variant == "h720_h160_mae15":
        model["hidden_dim"] = 160
        model["dropout"] = 0.1
        train.update({"epochs": 30, "mse_weight": 0.5, "selection_metric": "val_mae"})
        mae.update({"enable": True, "kind": "l1", "weight": 1.5, "warmup_epochs": 0})
        early["patience"] = 7
    elif variant in BIAS_CHECKPOINTS:
        freeze_base = variant.endswith("_freeze")
        cfg.setdefault("moe", {})["freeze_backbone"] = False
        train["freeze_backbone"] = False
        model["horizon_bias_adapter"] = {
            "enable": True,
            "init_bias": 0.0,
            "scale": 1.0,
            "freeze_base": freeze_base,
        }
        train.update(
            {
                "epochs": 24 if freeze_base else 30,
                "lr": 8.0e-3 if freeze_base else 4.0e-4,
                "mse_weight": 0.1 if freeze_base else 0.2,
                "selection_metric": "val_mae",
                "weight_decay": 0.0 if freeze_base else float(train.get("weight_decay", 1.0e-4)),
            }
        )
        mae.update({"enable": True, "kind": "l1", "weight": 3.0 if freeze_base else 2.5, "warmup_epochs": 0})
        early["patience"] = 5 if freeze_base else 7
        cfg["finetune"] = {
            "enable": True,
            "checkpoint_path": str(BIAS_CHECKPOINTS[variant]),
            "strict_window": True,
            "strict_model": False,
            "cluster_map": "index",
            "load_model": True,
            "load_gate": False,
            "load_dynamic_lambda": False,
            "load_learnable_lambda": False,
        }
        cfg.setdefault("memory", {})["save_checkpoint"] = True
    else:
        raise ValueError(f"Unknown variant: {variant}")
    return cfg


def run_train(py: str, cfg_path: Path, out_dir: Path) -> tuple[int, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
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
    (out_dir / "stdout.log").write_text(proc.stdout, encoding="utf-8", errors="replace")
    return int(proc.returncode), proc.stdout


def summarize(variant: str, cfg_path: Path, out_dir: Path, returncode: int, output: str) -> dict[str, Any]:
    row: dict[str, Any] = {
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
    row["horizon"] = (summary.get("windowing") or {}).get("pred_len", "")
    val = summary.get("val") or {}
    test = summary.get("test") or {}
    timing = summary.get("timing") or {}
    row.update(
        {
            "val_mse": val.get("avg_mse", ""),
            "val_mae": val.get("avg_mae", ""),
            "test_mse": test.get("avg_mse", ""),
            "test_mae": test.get("avg_mae", ""),
            "best_epoch": json.dumps(summary.get("best_epoch", "")),
            "total_sec": timing.get("total_sec", ""),
            "avg_epoch_sec": timing.get("avg_epoch_sec", ""),
        }
    )
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description="Weather MAE-focused local limit tuning.")
    ap.add_argument("--out-root", type=Path, default=ROOT / "outputs" / "weather_limit_tune")
    ap.add_argument("--variants", nargs="+", default=list(BASE_CONFIGS))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    rows: list[dict[str, Any]] = []
    for variant in args.variants:
        base_path = BASE_CONFIGS[variant]
        cfg = apply_variant(read_yaml(base_path), variant)
        horizon = infer_horizon(cfg)
        out_dir = args.out_root / "runs" / f"H{horizon}" / variant
        cfg = localize_paths(cfg, out_dir=out_dir, variant=variant, device=args.device)
        cfg_path = args.out_root / "configs" / f"H{horizon}_{variant}.yaml"
        write_yaml(cfg_path, cfg)
        print(f"[run] {variant} H{horizon}", flush=True)
        if args.dry_run:
            row = {"variant": variant, "status": "planned", "horizon": horizon, "config_path": str(cfg_path), "out_dir": str(out_dir)}
        else:
            returncode, output = run_train(args.python, cfg_path, out_dir)
            row = summarize(variant, cfg_path, out_dir, returncode, output)
        rows.append(row)
        write_rows(args.out_root / "results.csv", rows)
    write_rows(args.out_root / "results.csv", rows)


if __name__ == "__main__":
    main()
