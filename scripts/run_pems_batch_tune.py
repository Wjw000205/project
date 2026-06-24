from __future__ import annotations

import argparse
import copy
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]

DATASETS = ["PEMS03", "PEMS04", "PEMS07", "PEMS08"]
HORIZONS = [12, 24, 48, 96]

FIELDS = [
    "status",
    "phase",
    "dataset",
    "horizon",
    "candidate",
    "predictor",
    "batch_size",
    "hidden_dim",
    "dropout",
    "lambda_scale",
    "mse_weight",
    "mae_weight",
    "selection_metric",
    "lr",
    "epochs",
    "val_mae",
    "val_mse",
    "test_mae",
    "test_mse",
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


def deep_update(dst: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            deep_update(dst[key], value)
        else:
            dst[key] = copy.deepcopy(value)
    return dst


def candidates() -> list[dict[str, Any]]:
    return [
        {
            "name": "bs64_h192_do020_l003",
            "batch_size": 64,
            "hidden_dim": 192,
            "dropout": 0.20,
            "lambda_scale": 0.03,
            "lr": 1.0e-3,
        },
        {
            "name": "bs128_h192_do020_l003",
            "batch_size": 128,
            "hidden_dim": 192,
            "dropout": 0.20,
            "lambda_scale": 0.03,
            "lr": 1.0e-3,
        },
        {
            "name": "bs64_h256_do010_l003",
            "batch_size": 64,
            "hidden_dim": 256,
            "dropout": 0.10,
            "lambda_scale": 0.03,
            "lr": 1.0e-3,
        },
        {
            "name": "bs64_h192_do010_l001",
            "batch_size": 64,
            "hidden_dim": 192,
            "dropout": 0.10,
            "lambda_scale": 0.01,
            "lr": 1.0e-3,
        },
        {
            "name": "bs32_h192_do020_l003_mse100_mae030",
            "batch_size": 32,
            "hidden_dim": 192,
            "dropout": 0.20,
            "lambda_scale": 0.03,
            "mse_weight": 1.0,
            "mae_weight": 0.30,
            "lr": 1.0e-3,
        },
        {
            "name": "bs32_h256_do020_l003_mse100_mae030",
            "batch_size": 32,
            "hidden_dim": 256,
            "dropout": 0.20,
            "lambda_scale": 0.03,
            "mse_weight": 1.0,
            "mae_weight": 0.30,
            "lr": 1.0e-3,
        },
        {
            "name": "bs16_h192_do020_l003_mse100_mae030",
            "batch_size": 16,
            "hidden_dim": 192,
            "dropout": 0.20,
            "lambda_scale": 0.03,
            "mse_weight": 1.0,
            "mae_weight": 0.30,
            "lr": 1.0e-3,
        },
        {
            "name": "bs32_h192_do010_l001_mse100_mae030",
            "batch_size": 32,
            "hidden_dim": 192,
            "dropout": 0.10,
            "lambda_scale": 0.01,
            "mse_weight": 1.0,
            "mae_weight": 0.30,
            "lr": 1.0e-3,
        },
        {
            "name": "bs64_mlp_h192_do010_l001_season288_mse100_mae030",
            "predictor": "mlp",
            "batch_size": 64,
            "hidden_dim": 192,
            "dropout": 0.10,
            "lambda_scale": 0.01,
            "mse_weight": 1.0,
            "mae_weight": 0.30,
            "lr": 1.0e-3,
            "model_overrides": {"seasonal_residual": True, "seasonal_period": 288, "seasonal_num_periods": 1},
        },
        {
            "name": "bs32_mlp_h192_do010_l001_season288_mse100_mae030",
            "predictor": "mlp",
            "batch_size": 32,
            "hidden_dim": 192,
            "dropout": 0.10,
            "lambda_scale": 0.01,
            "mse_weight": 1.0,
            "mae_weight": 0.30,
            "lr": 1.0e-3,
            "model_overrides": {"seasonal_residual": True, "seasonal_period": 288, "seasonal_num_periods": 1},
        },
        {
            "name": "bs64_chh_h128_do010_l001_mse100_mae030",
            "predictor": "channel_head_mlp",
            "batch_size": 64,
            "hidden_dim": 128,
            "dropout": 0.10,
            "lambda_scale": 0.01,
            "mse_weight": 1.0,
            "mae_weight": 0.30,
            "lr": 1.0e-3,
        },
        {
            "name": "bs64_cch_h128_do010_l001_mse100_mae030",
            "predictor": "context_channel_head_mlp",
            "batch_size": 64,
            "hidden_dim": 128,
            "dropout": 0.10,
            "lambda_scale": 0.01,
            "mse_weight": 1.0,
            "mae_weight": 0.30,
            "lr": 1.0e-3,
        },
        {
            "name": "bs64_chdlinear_k25_mse100_mae030",
            "predictor": "channel_dlinear",
            "batch_size": 64,
            "hidden_dim": 128,
            "dropout": 0.0,
            "lambda_scale": 0.01,
            "mse_weight": 1.0,
            "mae_weight": 0.30,
            "lr": 1.0e-3,
            "model_overrides": {"dlinear_kernel_size": 25},
        },
        {
            "name": "bs64_cch_h128_do010_l001_mse080_mae080_valmae",
            "predictor": "context_channel_head_mlp",
            "batch_size": 64,
            "hidden_dim": 128,
            "dropout": 0.10,
            "lambda_scale": 0.01,
            "mse_weight": 0.80,
            "mae_weight": 0.80,
            "selection_metric": "val_mae",
            "lr": 1.0e-3,
        },
        {
            "name": "bs64_cch_h128_do010_l001_mse050_mae100_valmae",
            "predictor": "context_channel_head_mlp",
            "batch_size": 64,
            "hidden_dim": 128,
            "dropout": 0.10,
            "lambda_scale": 0.01,
            "mse_weight": 0.50,
            "mae_weight": 1.00,
            "selection_metric": "val_mae",
            "lr": 1.0e-3,
        },
        {
            "name": "bs64_cch_h192_do010_l001_mse080_mae080_valmae",
            "predictor": "context_channel_head_mlp",
            "batch_size": 64,
            "hidden_dim": 192,
            "dropout": 0.10,
            "lambda_scale": 0.01,
            "mse_weight": 0.80,
            "mae_weight": 0.80,
            "selection_metric": "val_mae",
            "lr": 1.0e-3,
        },
        {
            "name": "bs64_cch_h128_do005_l001_mse080_mae080_valmae",
            "predictor": "context_channel_head_mlp",
            "batch_size": 64,
            "hidden_dim": 128,
            "dropout": 0.05,
            "lambda_scale": 0.01,
            "mse_weight": 0.80,
            "mae_weight": 0.80,
            "selection_metric": "val_mae",
            "lr": 1.0e-3,
        },
        {
            "name": "bs64_cch_h96_do000_l001_mse080_mae120_valmae",
            "predictor": "context_channel_head_mlp",
            "batch_size": 64,
            "hidden_dim": 96,
            "dropout": 0.0,
            "lambda_scale": 0.01,
            "mse_weight": 0.80,
            "mae_weight": 1.20,
            "selection_metric": "val_mae",
            "lr": 1.0e-3,
        },
        {
            "name": "bs64_cch_h64_do000_l001_mse080_mae120_valmae",
            "predictor": "context_channel_head_mlp",
            "batch_size": 64,
            "hidden_dim": 64,
            "dropout": 0.0,
            "lambda_scale": 0.01,
            "mse_weight": 0.80,
            "mae_weight": 1.20,
            "selection_metric": "val_mae",
            "lr": 1.0e-3,
        },
        {
            "name": "bs32_cch_h128_do000_l001_mse080_mae120_valmae",
            "predictor": "context_channel_head_mlp",
            "batch_size": 32,
            "hidden_dim": 128,
            "dropout": 0.0,
            "lambda_scale": 0.01,
            "mse_weight": 0.80,
            "mae_weight": 1.20,
            "selection_metric": "val_mae",
            "lr": 1.0e-3,
        },
        {
            "name": "bs32_cch_h128_do000_l001_mse050_mae150_valmae",
            "predictor": "context_channel_head_mlp",
            "batch_size": 32,
            "hidden_dim": 128,
            "dropout": 0.0,
            "lambda_scale": 0.01,
            "mse_weight": 0.50,
            "mae_weight": 1.50,
            "selection_metric": "val_mae",
            "lr": 1.0e-3,
        },
        {
            "name": "bs32_cch_h160_do000_l001_mse050_mae150_valmae",
            "predictor": "context_channel_head_mlp",
            "batch_size": 32,
            "hidden_dim": 160,
            "dropout": 0.0,
            "lambda_scale": 0.01,
            "mse_weight": 0.50,
            "mae_weight": 1.50,
            "selection_metric": "val_mae",
            "lr": 1.0e-3,
        },
        {
            "name": "bs32_cch_h192_do000_l001_mse050_mae150_valmae",
            "predictor": "context_channel_head_mlp",
            "batch_size": 32,
            "hidden_dim": 192,
            "dropout": 0.0,
            "lambda_scale": 0.01,
            "mse_weight": 0.50,
            "mae_weight": 1.50,
            "selection_metric": "val_mae",
            "lr": 1.0e-3,
        },
        {
            "name": "bs32_cch_h192_do000_l001_mse050_mae150_valmae_s2031",
            "predictor": "context_channel_head_mlp",
            "batch_size": 32,
            "hidden_dim": 192,
            "dropout": 0.0,
            "lambda_scale": 0.01,
            "mse_weight": 0.50,
            "mae_weight": 1.50,
            "selection_metric": "val_mae",
            "lr": 1.0e-3,
            "seed": 2031,
        },
        {
            "name": "bs32_cch_h224_do000_l001_mse050_mae150_valmae",
            "predictor": "context_channel_head_mlp",
            "batch_size": 32,
            "hidden_dim": 224,
            "dropout": 0.0,
            "lambda_scale": 0.01,
            "mse_weight": 0.50,
            "mae_weight": 1.50,
            "selection_metric": "val_mae",
            "lr": 1.0e-3,
        },
        {
            "name": "bs32_cch_h128_do000_l001_mse030_mae200_valmae",
            "predictor": "context_channel_head_mlp",
            "batch_size": 32,
            "hidden_dim": 128,
            "dropout": 0.0,
            "lambda_scale": 0.01,
            "mse_weight": 0.30,
            "mae_weight": 2.00,
            "selection_metric": "val_mae",
            "lr": 1.0e-3,
        },
        {
            "name": "bs32_cch_h128_do000_l001_mse020_mae250_valmae",
            "predictor": "context_channel_head_mlp",
            "batch_size": 32,
            "hidden_dim": 128,
            "dropout": 0.0,
            "lambda_scale": 0.01,
            "mse_weight": 0.20,
            "mae_weight": 2.50,
            "selection_metric": "val_mae",
            "lr": 1.0e-3,
        },
        {
            "name": "bs32_cch_h128_do000_l001_mse010_mae300_valmae",
            "predictor": "context_channel_head_mlp",
            "batch_size": 32,
            "hidden_dim": 128,
            "dropout": 0.0,
            "lambda_scale": 0.01,
            "mse_weight": 0.10,
            "mae_weight": 3.00,
            "selection_metric": "val_mae",
            "lr": 1.0e-3,
        },
        {
            "name": "bs16_cch_h128_do000_l001_mse030_mae200_valmae",
            "predictor": "context_channel_head_mlp",
            "batch_size": 16,
            "hidden_dim": 128,
            "dropout": 0.0,
            "lambda_scale": 0.01,
            "mse_weight": 0.30,
            "mae_weight": 2.00,
            "selection_metric": "val_mae",
            "lr": 1.0e-3,
        },
        {
            "name": "bs32_cch_h160_do000_l001_mse030_mae200_valmae",
            "predictor": "context_channel_head_mlp",
            "batch_size": 32,
            "hidden_dim": 160,
            "dropout": 0.0,
            "lambda_scale": 0.01,
            "mse_weight": 0.30,
            "mae_weight": 2.00,
            "selection_metric": "val_mae",
            "lr": 1.0e-3,
        },
        {
            "name": "bs32_cch_h128_do000_l001_mse030_mae200_valmae_s2031",
            "predictor": "context_channel_head_mlp",
            "batch_size": 32,
            "hidden_dim": 128,
            "dropout": 0.0,
            "lambda_scale": 0.01,
            "mse_weight": 0.30,
            "mae_weight": 2.00,
            "selection_metric": "val_mae",
            "lr": 1.0e-3,
            "seed": 2031,
        },
        {
            "name": "bs64_cch_h128_do000_l001_mse050_mae150_valmae",
            "predictor": "context_channel_head_mlp",
            "batch_size": 64,
            "hidden_dim": 128,
            "dropout": 0.0,
            "lambda_scale": 0.01,
            "mse_weight": 0.50,
            "mae_weight": 1.50,
            "selection_metric": "val_mae",
            "lr": 1.0e-3,
        },
        {
            "name": "bs64_cch_h128_do000_l001_mse050_mae150_valmae_anchor_p288",
            "predictor": "context_channel_head_mlp",
            "batch_size": 64,
            "hidden_dim": 128,
            "dropout": 0.0,
            "lambda_scale": 0.01,
            "mse_weight": 0.50,
            "mae_weight": 1.50,
            "selection_metric": "val_mae",
            "lr": 1.0e-3,
            "moe_overrides": {
                "train_stat_anchor_expert": {
                    "enable": True,
                    "period": 288,
                    "alpha": 0.0,
                    "mode": "phase_mean",
                    "reference": "last",
                    "blend_target": "prediction",
                    "scale_selection": {
                        "enable": True,
                        "metric": "mse",
                        "max_scale": 0.3,
                        "steps": 13,
                    },
                },
                "train_residual_anchor_expert": {
                    "enable": True,
                    "period": 288,
                    "alpha": 0.0,
                    "blend_target": "prediction",
                    "scale_selection": {
                        "enable": True,
                        "metric": "mse",
                        "max_scale": 1.2,
                        "steps": 49,
                        "horizon_segments": 4,
                    },
                },
            },
        },
        {
            "name": "bs64_cch_h128_do000_l001_mse050_mae150_valmae_anchor_p288_residmae",
            "predictor": "context_channel_head_mlp",
            "batch_size": 64,
            "hidden_dim": 128,
            "dropout": 0.0,
            "lambda_scale": 0.01,
            "mse_weight": 0.50,
            "mae_weight": 1.50,
            "selection_metric": "val_mae",
            "lr": 1.0e-3,
            "moe_overrides": {
                "train_stat_anchor_expert": {
                    "enable": True,
                    "period": 288,
                    "alpha": 0.0,
                    "mode": "phase_mean",
                    "reference": "last",
                    "blend_target": "prediction",
                    "scale_selection": {
                        "enable": True,
                        "metric": "mse",
                        "max_scale": 0.3,
                        "steps": 13,
                    },
                },
                "train_residual_anchor_expert": {
                    "enable": True,
                    "period": 288,
                    "alpha": 0.0,
                    "blend_target": "prediction",
                    "scale_selection": {
                        "enable": True,
                        "metric": "mae",
                        "max_scale": 1.2,
                        "steps": 49,
                        "horizon_segments": 4,
                    },
                },
            },
        },
        {
            "name": "bs64_cch_h128_do000_l001_mse050_mae150_valmae_s2031",
            "predictor": "context_channel_head_mlp",
            "batch_size": 64,
            "hidden_dim": 128,
            "dropout": 0.0,
            "lambda_scale": 0.01,
            "mse_weight": 0.50,
            "mae_weight": 1.50,
            "selection_metric": "val_mae",
            "lr": 1.0e-3,
            "seed": 2031,
        },
        {
            "name": "bs64_cch_h128_do000_l001_mse050_mae150_valmae_s2037",
            "predictor": "context_channel_head_mlp",
            "batch_size": 64,
            "hidden_dim": 128,
            "dropout": 0.0,
            "lambda_scale": 0.01,
            "mse_weight": 0.50,
            "mae_weight": 1.50,
            "selection_metric": "val_mae",
            "lr": 1.0e-3,
            "seed": 2037,
        },
        {
            "name": "bs64_cch_h128_do000_l001_mse050_mae150_valmae_hbias",
            "predictor": "context_channel_head_mlp",
            "batch_size": 64,
            "hidden_dim": 128,
            "dropout": 0.0,
            "lambda_scale": 0.01,
            "mse_weight": 0.50,
            "mae_weight": 1.50,
            "selection_metric": "val_mae",
            "lr": 1.0e-3,
            "model_overrides": {
                "horizon_bias_adapter": {
                    "enable": True,
                    "init_bias": 0.0,
                    "scale": 1.0,
                    "freeze_base": False,
                }
            },
        },
        {
            "name": "bs32_cch_h128_do000_l001_mse030_mae200_valmae_hbias",
            "predictor": "context_channel_head_mlp",
            "batch_size": 32,
            "hidden_dim": 128,
            "dropout": 0.0,
            "lambda_scale": 0.01,
            "mse_weight": 0.30,
            "mae_weight": 2.00,
            "selection_metric": "val_mae",
            "lr": 1.0e-3,
            "model_overrides": {
                "horizon_bias_adapter": {
                    "enable": True,
                    "init_bias": 0.0,
                    "scale": 1.0,
                    "freeze_base": False,
                }
            },
        },
    ]


def configure(
    base_cfg: dict[str, Any],
    *,
    dataset: str,
    horizon: int,
    input_len: int = 96,
    cand: dict[str, Any],
    phase: str,
    out_dir: Path,
    epochs: int,
    skip_test: bool,
    device: str,
    save_checkpoint: bool = False,
    backbone_only: bool = False,
    freeze_backbone: bool = False,
    finetune_checkpoint: Path | str | None = None,
    lr_override: float | None = None,
) -> dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    run_name = f"{dataset}_H{horizon}_{cand['name']}_{phase}"

    cfg.setdefault("exp", {})
    cfg["exp"]["name"] = run_name
    cfg["exp"]["out_dir"] = str(out_dir)
    cfg["exp"]["device"] = device
    if "seed" in cand:
        cfg["exp"]["seed"] = int(cand["seed"])

    cfg.setdefault("data", {})
    cfg["data"]["csv_path"] = f"data/{dataset}.csv"
    cfg["data"]["date_col"] = 0
    cfg["data"]["train_ratio"] = 0.7
    cfg["data"]["val_ratio"] = 0.1
    cfg["data"]["test_ratio"] = 0.2
    cfg["data"]["max_rows"] = 0

    cfg.setdefault("window", {})
    cfg["window"]["input_len"] = int(input_len)
    cfg["window"]["pred_len"] = int(horizon)
    cfg["window"]["lazy"] = True

    cfg.setdefault("normalize", {})
    cfg["normalize"]["global_zscore"] = True
    cfg["normalize"]["train_only"] = True

    cfg.setdefault("corr", {})
    cfg["corr"]["compute"] = True
    cfg["corr"]["save_path"] = str(out_dir / "corr.npy")

    cfg.setdefault("cluster", {})
    cfg["cluster"]["train_only"] = True
    if "seed" in cand:
        cfg["cluster"]["random_state"] = int(cand["seed"])

    cfg.setdefault("model", {})
    cfg["model"]["predictor"] = str(cand.get("predictor", "mlp"))
    cfg["model"]["hidden_dim"] = int(cand["hidden_dim"])
    cfg["model"]["dropout"] = float(cand["dropout"])
    cfg["model"].update(dict(cand.get("model_overrides", {}) or {}))

    penalties = list(cfg.get("penalties", {}).get("enabled", ["amp_under", "delta", "diff_amp", "direction"]))
    cfg.setdefault("penalties", {})
    cfg["penalties"]["enabled"] = penalties
    cfg["penalties"]["jump_threshold"] = float(cfg["penalties"].get("jump_threshold", 0.6))

    lambda_scale = float(cand["lambda_scale"])
    cfg.setdefault("moe", {})
    if backbone_only:
        cfg["moe"]["enable"] = False
    cfg["moe"]["lambda_init"] = {name: lambda_scale for name in penalties}
    cfg["moe"]["lambda_min"] = {name: 0.0 for name in penalties}
    cfg["moe"]["lambda_schedule"] = {name: "none" for name in penalties}
    cfg["moe"]["gate_temperature"] = float(cfg["moe"].get("gate_temperature", 1.2))
    cfg["moe"]["gate_noise_std"] = float(cfg["moe"].get("gate_noise_std", 0.2))
    cfg["moe"].setdefault("dynamic_lambda", {})["enable"] = False
    cfg["moe"].setdefault("learnable_lambda", {})["enable"] = False
    cfg["moe"].setdefault("pred_side_residual", {})["enable"] = False
    deep_update(cfg["moe"], dict(cand.get("moe_overrides", {}) or {}))
    if freeze_backbone and not backbone_only:
        cfg["moe"]["freeze_backbone"] = True

    cfg.setdefault("train", {})
    cfg["train"]["epochs"] = int(epochs)
    cfg["train"]["batch_size"] = int(cand["batch_size"])
    cfg["train"]["lr"] = float(cand["lr"] if lr_override is None else lr_override)
    if "mse_weight" in cand:
        cfg["train"]["mse_weight"] = float(cand["mse_weight"])
    cfg["train"]["selection_metric"] = str(cand.get("selection_metric", "val_mse"))
    cfg["train"]["penalty_warmup_epochs"] = min(int(cfg["train"].get("penalty_warmup_epochs", 5)), 3)
    cfg["train"].setdefault("mae_objective", {})
    cfg["train"]["mae_objective"]["enable"] = True
    cfg["train"]["mae_objective"]["kind"] = "l1"
    cfg["train"]["mae_objective"]["weight"] = float(
        cand.get("mae_weight", cfg["train"]["mae_objective"].get("weight", 0.6))
    )
    cfg["train"]["mae_objective"]["warmup_epochs"] = min(
        int(cfg["train"]["mae_objective"].get("warmup_epochs", 5)),
        3,
    )
    cfg["train"].setdefault("lr_scheduler", {})
    cfg["train"]["lr_scheduler"]["name"] = "plateau"
    cfg["train"]["lr_scheduler"]["factor"] = 0.5
    cfg["train"]["lr_scheduler"]["patience"] = 2
    cfg["train"]["lr_scheduler"]["min_lr"] = 1.0e-6

    cfg.setdefault("early_stop", {})
    cfg["early_stop"]["patience"] = 4
    cfg["early_stop"]["min_delta"] = 1.0e-6

    cfg["eval"] = {"skip_test": bool(skip_test)}
    cfg["plot"] = {"enable": False}
    cfg["portrait"] = {"enable": False, "out_dir": str(out_dir / "cluster_portraits")}
    cfg["memory"] = {
        "enable": False,
        "save_checkpoint": bool(save_checkpoint),
        "path": str(out_dir / "cluster_memory.pt"),
        "checkpoint_path": str(out_dir / "best_checkpoint.pt"),
    }
    if finetune_checkpoint is not None:
        cfg["finetune"] = {
            "enable": True,
            "checkpoint_path": str(finetune_checkpoint),
            "strict_window": True,
            "strict_model": True,
            "cluster_map": "index",
            "load_model": True,
            "load_gate": False,
            "load_dynamic_lambda": False,
            "load_learnable_lambda": False,
        }
    return cfg


def run_train(python_exe: str, cfg_path: Path, out_dir: Path) -> tuple[int, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [python_exe, "-u", "-m", "src.train", "--config", str(cfg_path)],
        cwd=str(ROOT),
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    (out_dir / "stdout.log").write_text(proc.stdout, encoding="utf-8", errors="replace")
    return int(proc.returncode), proc.stdout


def summary_row(
    *,
    phase: str,
    dataset: str,
    horizon: int,
    cand: dict[str, Any],
    cfg_path: Path,
    out_dir: Path,
    epochs: int,
    returncode: int,
    output: str,
) -> dict[str, Any]:
    row = {
        "status": "ok",
        "phase": phase,
        "dataset": dataset,
        "horizon": int(horizon),
        "candidate": cand["name"],
        "predictor": cand.get("predictor", "mlp"),
        "batch_size": cand["batch_size"],
        "hidden_dim": cand["hidden_dim"],
        "dropout": cand["dropout"],
        "lambda_scale": cand["lambda_scale"],
        "mse_weight": cand.get("mse_weight", ""),
        "mae_weight": cand.get("mae_weight", ""),
        "selection_metric": cand.get("selection_metric", "val_mse"),
        "lr": cand["lr"],
        "epochs": int(epochs),
        "config_path": str(cfg_path),
        "out_dir": str(out_dir),
        "returncode": int(returncode),
    }
    if returncode != 0:
        row["status"] = "oom" if "out of memory" in output.lower() else "error"
        row["error"] = output[-4000:]
        return row
    summary_path = out_dir / "run_summary.json"
    if not summary_path.exists():
        row["status"] = "error"
        row["error"] = "run_summary.json not found"
        return row
    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)
    val = summary.get("val") or {}
    test = summary.get("test") or {}
    timing = summary.get("timing") or {}
    row["val_mae"] = val.get("avg_mae", "")
    row["val_mse"] = val.get("avg_mse", "")
    row["test_mae"] = test.get("avg_mae", "")
    row["test_mse"] = test.get("avg_mse", "")
    row["total_sec"] = timing.get("total_sec", "")
    row["avg_epoch_sec"] = timing.get("avg_epoch_sec", "")
    return row


def safe_float(value: Any) -> float:
    try:
        if value in ("", None):
            return float("inf")
        return float(value)
    except (TypeError, ValueError):
        return float("inf")


def parse_csv_list(raw: str, cast=str) -> list[Any]:
    return [cast(item.strip()) for item in raw.split(",") if item.strip()]


def filter_candidates(cand_list: list[dict[str, Any]], names: str = "") -> list[dict[str, Any]]:
    requested = [item.strip() for item in str(names or "").split(",") if item.strip()]
    if not requested:
        return cand_list
    by_name = {str(cand["name"]): cand for cand in cand_list}
    missing = [name for name in requested if name not in by_name]
    if missing:
        raise ValueError(f"Unknown PEMS candidates: {missing}. Supported: {sorted(by_name)}")
    return [by_name[name] for name in requested]


def main() -> None:
    ap = argparse.ArgumentParser(description="Targeted batch-size and parameter tuning for PEMS configs.")
    ap.add_argument("--out-root", type=Path, default=ROOT / "outputs" / "pems_batch_tune")
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--horizons", default=",".join(str(h) for h in HORIZONS))
    ap.add_argument("--phase", choices=["search", "final"], default="search")
    ap.add_argument("--candidates", default="", help="Comma-separated candidate names for targeted search.")
    ap.add_argument("--candidate", default="", help="Candidate name for final phase; empty uses best search row.")
    ap.add_argument("--candidate-limit", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--input-len", type=int, default=96)
    ap.add_argument("--save-checkpoint", action="store_true")
    ap.add_argument("--backbone-only", action="store_true", help="Disable MoE and train/evaluate the backbone path first.")
    ap.add_argument("--freeze-backbone", action="store_true", help="Freeze the warm-started backbone during the MoE stage.")
    ap.add_argument(
        "--finetune-root",
        type=Path,
        default=None,
        help="Warm-start from <root>/final/<dataset>/H<horizon>/<candidate>/best_checkpoint.pt.",
    )
    ap.add_argument(
        "--finetune-candidate",
        default="",
        help="Candidate directory name under --finetune-root; empty uses the active candidate name.",
    )
    ap.add_argument("--lr-override", type=float, default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--rerun", action="store_true")
    args = ap.parse_args()

    datasets = parse_csv_list(args.datasets, str)
    horizons = parse_csv_list(args.horizons, int)
    cand_list = filter_candidates(candidates(), args.candidates)
    if args.candidate_limit > 0:
        cand_list = cand_list[: int(args.candidate_limit)]

    result_path = args.out_root / f"{args.phase}_results.csv"
    rows = read_rows(result_path)
    completed = {
        (row.get("phase"), row.get("dataset"), int(row.get("horizon", -1)), row.get("candidate"))
        for row in rows
        if row.get("status") == "ok"
    }

    for dataset in datasets:
        for horizon in horizons:
            base_cfg = read_yaml(ROOT / "configs" / f"{dataset}_H{horizon}.yaml")
            if args.phase == "search":
                active_candidates = cand_list
                skip_test = True
            else:
                if args.candidate:
                    chosen = next(c for c in cand_list if c["name"] == args.candidate)
                else:
                    search_rows = [
                        row
                        for row in read_rows(args.out_root / "search_results.csv")
                        if row.get("dataset") == dataset
                        and int(row.get("horizon", -1)) == int(horizon)
                        and row.get("status") == "ok"
                    ]
                    if not search_rows:
                        raise RuntimeError(f"No search rows for {dataset} H{horizon}")
                    best = min(search_rows, key=lambda row: safe_float(row.get("val_mse")))
                    chosen = next(c for c in cand_list if c["name"] == best["candidate"])
                active_candidates = [chosen]
                skip_test = False

            for cand in active_candidates:
                key = (args.phase, dataset, int(horizon), cand["name"])
                if key in completed and not args.rerun:
                    print(f"[skip] {dataset} H{horizon} {cand['name']} {args.phase}", flush=True)
                    continue
                out_dir = args.out_root / args.phase / dataset / f"H{horizon}" / cand["name"]
                cfg_path = args.out_root / "configs" / args.phase / dataset / f"H{horizon}_{cand['name']}.yaml"
                finetune_checkpoint = None
                if args.finetune_root is not None:
                    finetune_candidate = args.finetune_candidate or cand["name"]
                    finetune_checkpoint = (
                        args.finetune_root
                        / "final"
                        / dataset
                        / f"H{horizon}"
                        / finetune_candidate
                        / "best_checkpoint.pt"
                    )
                    if not finetune_checkpoint.exists():
                        raise FileNotFoundError(f"Fine-tune checkpoint not found: {finetune_checkpoint}")
                cfg = configure(
                    base_cfg,
                    dataset=dataset,
                    horizon=int(horizon),
                    input_len=int(args.input_len),
                    cand=cand,
                    phase=args.phase,
                    out_dir=out_dir,
                    epochs=int(args.epochs),
                    skip_test=skip_test,
                    device=args.device,
                    save_checkpoint=bool(args.save_checkpoint),
                    backbone_only=bool(args.backbone_only),
                    freeze_backbone=bool(args.freeze_backbone),
                    finetune_checkpoint=finetune_checkpoint,
                    lr_override=args.lr_override,
                )
                write_yaml(cfg_path, cfg)
                print(f"[run] {dataset} H{horizon} {cand['name']} {args.phase}", flush=True)
                returncode, output = run_train(args.python, cfg_path, out_dir)
                rows.append(
                    summary_row(
                        phase=args.phase,
                        dataset=dataset,
                        horizon=int(horizon),
                        cand=cand,
                        cfg_path=cfg_path,
                        out_dir=out_dir,
                        epochs=int(args.epochs),
                        returncode=returncode,
                        output=output,
                    )
                )
                write_rows(result_path, rows)
    write_rows(result_path, rows)
    print(f"Saved {result_path}", flush=True)


if __name__ == "__main__":
    main()
