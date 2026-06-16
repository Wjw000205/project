import argparse
import copy
import csv
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_all_datasets_horizons import apply_tsl_alignment, ensure_local_output_paths


DATASET_CONFIGS = {
    "ETTh1": "configs/ETTh1.yaml",
    "ETTh2": "configs/ETTh2.yaml",
    "ETTm1": "configs/ETTm1.yaml",
    "ETTm2": "configs/ETTm2.yaml",
    "weather": "configs/weather.yaml",
    "electricity": "configs/electricity.yaml",
    "PEMS03": "configs/PEMS03_H12.yaml",
    "PEMS04": "configs/PEMS04_H12.yaml",
    "PEMS07": "configs/PEMS07_H12.yaml",
    "PEMS08": "configs/PEMS08_H12.yaml",
}

MLP_FAMILY = {
    "mlp",
    "cluster_mlp",
    "context_mlp",
    "attn_mlp",
    "channel_head_mlp",
    "channel_mlp",
    "segment_mlp",
    "long_anchor_mlp",
}


@dataclass(frozen=True)
class Candidate:
    stage: str
    variant: str
    patch: dict[str, Any]


FIELDS = [
    "dataset",
    "pred_len",
    "stage",
    "variant",
    "status",
    "val_mse",
    "val_mae",
    "test_mse",
    "test_mae",
    "best_epoch",
    "model_hidden_dim",
    "model_dropout",
    "lr",
    "weight_decay",
    "mae_weight",
    "moe_enable",
    "penalties",
    "lambda_init",
    "dynamic_lambda",
    "alpha_scale",
    "residual_clip",
    "selection_policy",
    "selection_min_rel_improvement",
    "gate_max_scale",
    "gate_init_scale",
    "residual_mean_scale",
    "residual_num_channels",
    "config_path",
    "out_dir",
    "total_sec",
    "returncode",
    "error",
]


def resolve(path_text: str | Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_dataset_config(dataset: str, pred_len: int) -> Path:
    horizon_path = REPO_ROOT / "configs" / f"{dataset}_H{int(pred_len)}.yaml"
    if horizon_path.exists():
        return horizon_path.resolve()
    return resolve(DATASET_CONFIGS[dataset])


def write_yaml(path: Path, cfg: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=False, sort_keys=False)


def deep_update(dst: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            deep_update(dst[key], value)
        else:
            dst[key] = value
    return dst


def set_moe_off(cfg: dict[str, Any]) -> None:
    moe = cfg.setdefault("moe", {})
    moe["enable"] = False
    moe.setdefault("dynamic_lambda", {})["enable"] = False
    moe.setdefault("pred_side_residual", {})["enable"] = False


def lambda_dict(penalties: list[str], value: float) -> dict[str, float]:
    return {name: float(value) for name in penalties}


def schedule_dict(penalties: list[str], value: str = "none") -> dict[str, str]:
    return {name: value for name in penalties}


def common_prepare(
    cfg: dict[str, Any],
    *,
    dataset: str,
    pred_len: int = 96,
    out_dir: Path,
    run_name: str,
    device: str | None,
    epochs: int | None,
    skip_test: bool,
    save_checkpoint: bool = False,
) -> dict[str, Any]:
    cfg = copy.deepcopy(cfg)
    apply_tsl_alignment(cfg, dataset)
    cfg.setdefault("window", {})
    cfg["window"]["input_len"] = 96
    cfg["window"]["pred_len"] = int(pred_len)
    cfg["window"]["past_context"] = True
    cfg.setdefault("normalize", {})["train_only"] = True
    cfg.setdefault("cluster", {})["train_only"] = True
    cfg.setdefault("eval", {})["skip_test"] = bool(skip_test)
    cfg.setdefault("calibration", {})["enable"] = False
    cfg.setdefault("knn_hybrid", {})["enable"] = False
    cfg.setdefault("plot", {})["enable"] = False
    cfg.setdefault("portrait", {})["enable"] = False
    cfg.setdefault("memory", {})["enable"] = False
    cfg.setdefault("memory", {})["save_checkpoint"] = False
    if epochs is not None:
        cfg.setdefault("train", {})["epochs"] = int(epochs)
    if device:
        cfg.setdefault("exp", {})["device"] = str(device)
    ensure_local_output_paths(
        cfg,
        out_dir=out_dir,
        run_name=run_name,
        keep_artifacts=False,
        disable_knn_hybrid=True,
        knn_adaptive_alpha=None,
        knn_selection_policy=None,
        knn_selection_min_rel_improvement=None,
        knn_selection_min_abs_improvement=None,
    )
    if save_checkpoint:
        cfg.setdefault("memory", {})["save_checkpoint"] = True
    return cfg


def model_candidates() -> list[Candidate]:
    def seasonal_channel_patch(
        *,
        hidden_dim: int,
        dropout: float,
        predictor_input_len: int,
        mix_init: float,
        gate_strength: float = 1.0,
        gate_threshold: float = 0.2,
        weight_decay: float = 1.0e-4,
        mae_weight: float = 0.2,
    ) -> dict[str, Any]:
        model: dict[str, Any] = {
            "predictor": "seasonality_gated_channel_head_mlp",
            "hidden_dim": int(hidden_dim),
            "dropout": float(dropout),
            "predictor_input_len": int(predictor_input_len),
            "seasonal_mix_init": float(mix_init),
            "seasonal_gate_strength": float(gate_strength),
            "anchor_chunk_len": 96,
            "anchor_detail_scale": 0.25,
        }
        if float(gate_strength) != 0.0:
            model["seasonal_gate_threshold"] = float(gate_threshold)
        return {
            "model": model,
            "train": {
                "weight_decay": float(weight_decay),
                "mae_objective": {"enable": True, "kind": "l1", "weight": float(mae_weight)},
            },
        }

    def electric_h96_transfer_patch(
        *,
        hidden_dim: int = 256,
        dropout: float = 0.0,
        lr: float = 0.001309395478035077,
        weight_decay: float = 1.0644440818212169e-05,
        model_extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        model = {"predictor": "mlp", "hidden_dim": int(hidden_dim), "dropout": float(dropout)}
        if model_extra:
            model.update(copy.deepcopy(model_extra))
        return {
            "model": model,
            "train": {"lr": float(lr), "weight_decay": float(weight_decay)},
        }

    def model_train_stat_adapter_patch(
        *,
        period: int,
        mode: str = "phase_mean",
        alpha: float = 0.0,
        reference: str | None = None,
        combine_mode: str = "blend",
        max_scale: float | None = None,
        steps: int = 21,
        horizon_segments: int = 1,
        hidden_dim: int = 256,
    ) -> dict[str, Any]:
        adapter: dict[str, Any] = {
            "enable": True,
            "period": int(period),
            "mode": str(mode),
            "alpha": float(alpha),
            "blend_target": "prediction",
        }
        if str(combine_mode) != "blend":
            adapter["combine_mode"] = str(combine_mode)
        if reference is not None:
            adapter["reference"] = str(reference)
        if max_scale is not None:
            adapter["scale_selection"] = {
                "enable": True,
                "metric": "mse",
                "max_scale": float(max_scale),
                "steps": int(steps),
                "horizon_segments": int(horizon_segments),
            }
        return electric_h96_transfer_patch(hidden_dim=int(hidden_dim), model_extra={"train_stat_adapter": adapter})

    def electric_center_residual_patch(
        *,
        predictor: str = "mlp",
        hidden_dim: int = 256,
        alpha: float = 1.0,
        dropout: float = 0.0,
        weight_decay: float = 0.0,
        batch_size: int = 8,
        lr: float = 0.001309395478035077,
        include_delta: bool = True,
        model_extra: dict[str, Any] | None = None,
        train_extra: dict[str, Any] | None = None,
        mae_weight: float | None = None,
        early_stop_patience: int | None = None,
    ) -> dict[str, Any]:
        model_cfg: dict[str, Any] = {
            "train_stat_adapter": {
                "enable": True,
                "period": 168,
                "mode": "phase_mean",
                "alpha": float(alpha),
                "blend_target": "prediction",
                "combine_mode": "anchor_plus_prediction",
                "input_center": True,
            },
        }
        if str(predictor) == "channel_head_mlp":
            model_cfg["channel_head_residual"] = True
        if str(predictor) == "context_channel_head_mlp":
            model_cfg["context_channel_head_residual"] = True
            model_cfg["context_channel_head_include_delta"] = bool(include_delta)
        if model_extra is not None:
            deep_update(model_cfg, copy.deepcopy(model_extra))
        patch = {
            **electric_h96_transfer_patch(
                hidden_dim=int(hidden_dim),
                dropout=float(dropout),
                lr=float(lr),
                weight_decay=float(weight_decay),
                model_extra={"predictor": str(predictor), **model_cfg},
            ),
            "train": {
                **electric_h96_transfer_patch(lr=float(lr), weight_decay=float(weight_decay))["train"],
                "batch_size": int(batch_size),
            },
        }
        if mae_weight is not None:
            patch["train"]["mae_objective"] = {"enable": True, "kind": "l1", "weight": float(mae_weight)}
        if train_extra is not None:
            deep_update(patch["train"], copy.deepcopy(train_extra))
        if early_stop_patience is not None:
            patch["early_stop"] = {"patience": int(early_stop_patience), "min_delta": 1.0e-6}
        return patch

    return [
        Candidate("model", "current_model", {}),
        Candidate(
            "model",
            "current_mae02",
            {
                "train": {"mae_objective": {"enable": True, "kind": "l1", "weight": 0.2}},
            },
        ),
        Candidate(
            "model",
            "current_mseonly",
            {
                "train": {"mae_objective": {"enable": False, "weight": 0.0}},
            },
        ),
        Candidate("model", "current_h64", {"model": {"hidden_dim": 64}}),
        Candidate("model", "current_h96", {"model": {"hidden_dim": 96}}),
        Candidate("model", "current_h128", {"model": {"hidden_dim": 128}}),
        Candidate(
            "model",
            "pems_cch_h128_do0_l001_mse050_mae150_bs64_valmae",
            {
                "model": {
                    "predictor": "context_channel_head_mlp",
                    "hidden_dim": 128,
                    "dropout": 0.0,
                },
                "train": {
                    "batch_size": 64,
                    "lr": 0.001,
                    "weight_decay": 1.0e-4,
                    "mse_weight": 0.5,
                    "selection_metric": "val_mae",
                    "mae_objective": {
                        "enable": True,
                        "kind": "l1",
                        "weight": 1.5,
                        "warmup_epochs": 3,
                    },
                },
            },
        ),
        Candidate(
            "model",
            "electric_current_h192",
            {"model": {"predictor": "mlp", "hidden_dim": 192, "dropout": 0.0468935703282562}},
        ),
        Candidate(
            "model",
            "electric_current_h320",
            {"model": {"predictor": "mlp", "hidden_dim": 320, "dropout": 0.0468935703282562}},
        ),
        Candidate(
            "model",
            "electric_current_h384",
            {"model": {"predictor": "mlp", "hidden_dim": 384, "dropout": 0.0468935703282562}},
        ),
        Candidate(
            "model",
            "electric_current_h256_do0",
            {"model": {"predictor": "mlp", "hidden_dim": 256, "dropout": 0.0}},
        ),
        Candidate(
            "model",
            "electric_current_h256_do002",
            {"model": {"predictor": "mlp", "hidden_dim": 256, "dropout": 0.02}},
        ),
        Candidate(
            "model",
            "electric_current_h256_do008",
            {"model": {"predictor": "mlp", "hidden_dim": 256, "dropout": 0.08}},
        ),
        Candidate(
            "model",
            "electric_current_h256_wd0",
            {"model": {"predictor": "mlp", "hidden_dim": 256}, "train": {"weight_decay": 0.0}},
        ),
        Candidate(
            "model",
            "electric_current_h256_wd1e4",
            {"model": {"predictor": "mlp", "hidden_dim": 256}, "train": {"weight_decay": 1.0e-4}},
        ),
        Candidate(
            "model",
            "electric_current_h256_lr8e4",
            {"model": {"predictor": "mlp", "hidden_dim": 256}, "train": {"lr": 8.0e-4}},
        ),
        Candidate(
            "model",
            "electric_current_h256_lr18e4",
            {"model": {"predictor": "mlp", "hidden_dim": 256}, "train": {"lr": 1.8e-3}},
        ),
        Candidate("model", "electric_h96params_h256_do0", electric_h96_transfer_patch()),
        Candidate("model", "electric_h96params_h256_do002", electric_h96_transfer_patch(dropout=0.02)),
        Candidate("model", "electric_h96params_h256_do0047", electric_h96_transfer_patch(dropout=0.0468935703282562)),
        Candidate("model", "electric_h96params_h320_do0", electric_h96_transfer_patch(hidden_dim=320)),
        Candidate("model", "electric_h96params_h384_do0", electric_h96_transfer_patch(hidden_dim=384)),
        Candidate("model", "electric_h96params_h256_wd0_do0", electric_h96_transfer_patch(weight_decay=0.0)),
        Candidate("model", "electric_h96params_h256_lr8e4_do0", electric_h96_transfer_patch(lr=8.0e-4)),
        Candidate("model", "electric_h96params_h256_lr18e4_do0", electric_h96_transfer_patch(lr=1.8e-3)),
        Candidate(
            "model",
            "electric_h96params_h256_do0_seasonalres_p24_np4",
            electric_h96_transfer_patch(
                model_extra={"seasonal_residual": True, "seasonal_period": 24, "seasonal_num_periods": 4}
            ),
        ),
        Candidate(
            "model",
            "electric_h96params_h320_do0_seasonalres_p24_np4",
            electric_h96_transfer_patch(
                hidden_dim=320,
                model_extra={"seasonal_residual": True, "seasonal_period": 24, "seasonal_num_periods": 4},
            ),
        ),
        Candidate(
            "model",
            "electric_h96params_h256_do0_seasonalanchor_p24_np4_d05",
            electric_h96_transfer_patch(
                model_extra={
                    "seasonal_anchor": True,
                    "seasonal_anchor_period": 24,
                    "seasonal_anchor_num_periods": 4,
                    "seasonal_anchor_delta_scale": 0.5,
                }
            ),
        ),
        Candidate(
            "model",
            "electric_h96params_h256_do0_seasonalanchor_p24_np4_d10",
            electric_h96_transfer_patch(
                model_extra={
                    "seasonal_anchor": True,
                    "seasonal_anchor_period": 24,
                    "seasonal_anchor_num_periods": 4,
                    "seasonal_anchor_delta_scale": 1.0,
                }
            ),
        ),
        Candidate(
            "model",
            "electric_h96params_h256_do0_resanchor_seasonalanchor_p24_np4_d10",
            electric_h96_transfer_patch(
                model_extra={
                    "mlp_residual_anchor": True,
                    "seasonal_anchor": True,
                    "seasonal_anchor_period": 24,
                    "seasonal_anchor_num_periods": 4,
                    "seasonal_anchor_delta_scale": 1.0,
                }
            ),
        ),
        Candidate(
            "model",
            "electric_h96params_h256_do0_seasonalblend_p24_np4_m05_i02",
            electric_h96_transfer_patch(
                model_extra={
                    "seasonal_blend_adapter": {
                        "enable": True,
                        "period": 24,
                        "num_periods": 4,
                        "max_mix": 0.5,
                        "init_mix": 0.2,
                    }
                }
            ),
        ),
        Candidate(
            "model",
            "electric_h96params_h256_do0_seasonalblend_p24_np4_m10_i05",
            electric_h96_transfer_patch(
                model_extra={
                    "seasonal_blend_adapter": {
                        "enable": True,
                        "period": 24,
                        "num_periods": 4,
                        "max_mix": 1.0,
                        "init_mix": 0.5,
                    }
                }
            ),
        ),
        Candidate(
            "model",
            "electric_h96params_h256_do0_modelstat_p24_mean_a010",
            model_train_stat_adapter_patch(period=24, mode="phase_mean", alpha=0.1),
        ),
        Candidate(
            "model",
            "electric_h96params_h256_do0_modelstat_p24_mean_a020",
            model_train_stat_adapter_patch(period=24, mode="phase_mean", alpha=0.2),
        ),
        Candidate(
            "model",
            "electric_h96params_h256_do0_modelstat_p24_mean_select_m05_seg12",
            model_train_stat_adapter_patch(period=24, mode="phase_mean", max_scale=0.5, horizon_segments=12),
        ),
        Candidate(
            "model",
            "electric_h96params_h256_do0_modelstat_p168_mean_a010",
            model_train_stat_adapter_patch(period=168, mode="phase_mean", alpha=0.1),
        ),
        Candidate(
            "model",
            "electric_h96params_h256_do0_modelstat_p168_mean_a005",
            model_train_stat_adapter_patch(period=168, mode="phase_mean", alpha=0.05),
        ),
        Candidate(
            "model",
            "electric_h96params_h256_do0_modelstat_p168_mean_a015",
            model_train_stat_adapter_patch(period=168, mode="phase_mean", alpha=0.15),
        ),
        Candidate(
            "model",
            "electric_h96params_h256_do0_modelstat_p168_mean_a020",
            model_train_stat_adapter_patch(period=168, mode="phase_mean", alpha=0.2),
        ),
        Candidate(
            "model",
            "electric_h96params_h256_do0_modelstat_p168_mean_a025",
            model_train_stat_adapter_patch(period=168, mode="phase_mean", alpha=0.25),
        ),
        Candidate(
            "model",
            "electric_h96params_h256_do0_modelstat_p168_mean_a030",
            model_train_stat_adapter_patch(period=168, mode="phase_mean", alpha=0.3),
        ),
        Candidate(
            "model",
            "electric_h96params_h256_do0_modelstat_p168_mean_a040",
            model_train_stat_adapter_patch(period=168, mode="phase_mean", alpha=0.4),
        ),
        Candidate(
            "model",
            "electric_h96params_h256_do0_modelstat_p168_mean_a050",
            model_train_stat_adapter_patch(period=168, mode="phase_mean", alpha=0.5),
        ),
        Candidate(
            "model",
            "electric_h96params_h256_do0_modelstat_p168_mean_a060",
            model_train_stat_adapter_patch(period=168, mode="phase_mean", alpha=0.6),
        ),
        Candidate(
            "model",
            "electric_h96params_h256_do0_modelstat_p168_mean_a070",
            model_train_stat_adapter_patch(period=168, mode="phase_mean", alpha=0.7),
        ),
        Candidate(
            "model",
            "electric_h96params_h256_do0_modelstat_p168_mean_a080",
            model_train_stat_adapter_patch(period=168, mode="phase_mean", alpha=0.8),
        ),
        Candidate(
            "model",
            "electric_h96params_h256_do0_modelstat_p168_mean_a100",
            model_train_stat_adapter_patch(period=168, mode="phase_mean", alpha=1.0),
        ),
        Candidate(
            "model",
            "electric_h96params_h256_do0_modelstat_p168_mean_anchorres_a05",
            model_train_stat_adapter_patch(
                period=168,
                mode="phase_mean",
                alpha=0.5,
                combine_mode="anchor_plus_prediction",
            ),
        ),
        Candidate(
            "model",
            "electric_h96params_h256_do0_modelstat_p168_mean_anchorres_a10",
            model_train_stat_adapter_patch(
                period=168,
                mode="phase_mean",
                alpha=1.0,
                combine_mode="anchor_plus_prediction",
            ),
        ),
        Candidate(
            "model",
            "electric_h96params_h320_do0_modelstat_p168_mean_anchorres_a10",
            model_train_stat_adapter_patch(
                hidden_dim=320,
                period=168,
                mode="phase_mean",
                alpha=1.0,
                combine_mode="anchor_plus_prediction",
            ),
        ),
        Candidate(
            "model",
            "electric_h96params_h256_do0_modelstat_p168_center_anchorres_a05",
            electric_h96_transfer_patch(
                model_extra={
                    "train_stat_adapter": {
                        "enable": True,
                        "period": 168,
                        "mode": "phase_mean",
                        "alpha": 0.5,
                        "blend_target": "prediction",
                        "combine_mode": "anchor_plus_prediction",
                        "input_center": True,
                    },
                }
            ),
        ),
        Candidate(
            "model",
            "electric_h96params_h256_do0_modelstat_p168_center_anchorres_a10",
            electric_h96_transfer_patch(
                model_extra={
                    "train_stat_adapter": {
                        "enable": True,
                        "period": 168,
                        "mode": "phase_mean",
                        "alpha": 1.0,
                        "blend_target": "prediction",
                        "combine_mode": "anchor_plus_prediction",
                        "input_center": True,
                    },
                }
            ),
        ),
        Candidate(
            "model",
            "electric_h96params_h320_do0_modelstat_p168_center_anchorres_a10",
            electric_h96_transfer_patch(
                hidden_dim=320,
                model_extra={
                    "train_stat_adapter": {
                        "enable": True,
                        "period": 168,
                        "mode": "phase_mean",
                        "alpha": 1.0,
                        "blend_target": "prediction",
                        "combine_mode": "anchor_plus_prediction",
                        "input_center": True,
                    },
                }
            ),
        ),
        Candidate(
            "model",
            "electric_h96params_h192_do0_modelstat_p168_center_anchorres_a10",
            electric_h96_transfer_patch(
                hidden_dim=192,
                model_extra={
                    "train_stat_adapter": {
                        "enable": True,
                        "period": 168,
                        "mode": "phase_mean",
                        "alpha": 1.0,
                        "blend_target": "prediction",
                        "combine_mode": "anchor_plus_prediction",
                        "input_center": True,
                    },
                }
            ),
        ),
        Candidate(
            "model",
            "electric_h96params_h256_do0_modelstat_p168_center_anchorres_a08",
            electric_h96_transfer_patch(
                model_extra={
                    "train_stat_adapter": {
                        "enable": True,
                        "period": 168,
                        "mode": "phase_mean",
                        "alpha": 0.8,
                        "blend_target": "prediction",
                        "combine_mode": "anchor_plus_prediction",
                        "input_center": True,
                    },
                }
            ),
        ),
        Candidate(
            "model",
            "electric_h96params_h256_do0_modelstat_p168_center_anchorres_a12",
            electric_h96_transfer_patch(
                model_extra={
                    "train_stat_adapter": {
                        "enable": True,
                        "period": 168,
                        "mode": "phase_mean",
                        "alpha": 1.2,
                        "blend_target": "prediction",
                        "combine_mode": "anchor_plus_prediction",
                        "input_center": True,
                    },
                }
            ),
        ),
        Candidate(
            "model",
            "electric_h96params_h256_do0_modelstat_p168_center_anchorres_a10_cs08",
            electric_h96_transfer_patch(
                model_extra={
                    "train_stat_adapter": {
                        "enable": True,
                        "period": 168,
                        "mode": "phase_mean",
                        "alpha": 1.0,
                        "blend_target": "prediction",
                        "combine_mode": "anchor_plus_prediction",
                        "input_center": True,
                        "input_center_scale": 0.8,
                    },
                }
            ),
        ),
        Candidate(
            "model",
            "electric_h96params_h256_do0_modelstat_p168_center_anchorres_a10_cs12",
            electric_h96_transfer_patch(
                model_extra={
                    "train_stat_adapter": {
                        "enable": True,
                        "period": 168,
                        "mode": "phase_mean",
                        "alpha": 1.0,
                        "blend_target": "prediction",
                        "combine_mode": "anchor_plus_prediction",
                        "input_center": True,
                        "input_center_scale": 1.2,
                    },
                }
            ),
        ),
        Candidate(
            "model",
            "electric_h96params_h256_do0_modelstat_p168_center_anchorres_a10_wd0",
            electric_h96_transfer_patch(
                weight_decay=0.0,
                model_extra={
                    "train_stat_adapter": {
                        "enable": True,
                        "period": 168,
                        "mode": "phase_mean",
                        "alpha": 1.0,
                        "blend_target": "prediction",
                        "combine_mode": "anchor_plus_prediction",
                        "input_center": True,
                    },
                }
            ),
        ),
        Candidate(
            "model",
            "electric_h96params_h256_do0_modelstat_p168_center_anchorres_a10_wd0_bs16",
            {
                **electric_h96_transfer_patch(
                    weight_decay=0.0,
                    model_extra={
                        "train_stat_adapter": {
                            "enable": True,
                            "period": 168,
                            "mode": "phase_mean",
                            "alpha": 1.0,
                            "blend_target": "prediction",
                            "combine_mode": "anchor_plus_prediction",
                            "input_center": True,
                        },
                    },
                ),
                "train": {
                    **electric_h96_transfer_patch(weight_decay=0.0)["train"],
                    "batch_size": 16,
                },
            },
        ),
        Candidate(
            "model",
            "electric_h96params_h256_do0_modelstat_p168_center_anchorres_a10_wd0_bs8",
            {
                **electric_h96_transfer_patch(
                    weight_decay=0.0,
                    model_extra={
                        "train_stat_adapter": {
                            "enable": True,
                            "period": 168,
                            "mode": "phase_mean",
                            "alpha": 1.0,
                            "blend_target": "prediction",
                            "combine_mode": "anchor_plus_prediction",
                            "input_center": True,
                        },
                    },
                ),
                "train": {
                    **electric_h96_transfer_patch(weight_decay=0.0)["train"],
                    "batch_size": 8,
                },
            },
        ),
        Candidate(
            "model",
            "electric_h96_centerres_h224_a10_wd0_bs128",
            electric_center_residual_patch(hidden_dim=224, alpha=1.0, weight_decay=0.0, batch_size=128),
        ),
        Candidate(
            "model",
            "electric_h96_centerres_h288_a10_wd0_bs128",
            electric_center_residual_patch(hidden_dim=288, alpha=1.0, weight_decay=0.0, batch_size=128),
        ),
        Candidate(
            "model",
            "electric_h96_centerres_h256_a10_wd0_bs64",
            electric_center_residual_patch(hidden_dim=256, alpha=1.0, weight_decay=0.0, batch_size=64),
        ),
        Candidate(
            "model",
            "electric_h96_centerres_h256_a10_wd0_bs32",
            electric_center_residual_patch(hidden_dim=256, alpha=1.0, weight_decay=0.0, batch_size=32),
        ),
        Candidate(
            "model",
            "electric_h96_centerres_h256_a095_wd0_bs128",
            electric_center_residual_patch(hidden_dim=256, alpha=0.95, weight_decay=0.0, batch_size=128),
        ),
        Candidate(
            "model",
            "electric_h96_centerres_h256_a105_wd0_bs128",
            electric_center_residual_patch(hidden_dim=256, alpha=1.05, weight_decay=0.0, batch_size=128),
        ),
        Candidate(
            "model",
            "electric_h96_centerres_h256_a10_wd0_mae005_bs128",
            electric_center_residual_patch(
                hidden_dim=256,
                alpha=1.0,
                weight_decay=0.0,
                batch_size=128,
                mae_weight=0.05,
            ),
        ),
        Candidate(
            "model",
            "electric_h96_centerres_h256_a10_wd0_modelstat_select_m200_seg12_bs128",
            electric_center_residual_patch(
                hidden_dim=256,
                alpha=1.0,
                weight_decay=0.0,
                batch_size=128,
                model_extra={
                    "train_stat_adapter": {
                        "scale_selection": {
                            "enable": True,
                            "metric": "mse",
                            "max_scale": 2.0,
                            "steps": 81,
                            "horizon_segments": 12,
                        }
                    }
                },
            ),
        ),
        Candidate(
            "model",
            "electric_h96_centerres_h256_a10_wd0_modelstat_select_m300_seg12_bs128",
            electric_center_residual_patch(
                hidden_dim=256,
                alpha=1.0,
                weight_decay=0.0,
                batch_size=128,
                model_extra={
                    "train_stat_adapter": {
                        "scale_selection": {
                            "enable": True,
                            "metric": "mse",
                            "max_scale": 3.0,
                            "steps": 121,
                            "horizon_segments": 12,
                        }
                    }
                },
            ),
        ),
        Candidate(
            "model",
            "electric_h96_centerres_h256_a10_wd0_swa_bs128",
            electric_center_residual_patch(
                hidden_dim=256,
                alpha=1.0,
                weight_decay=0.0,
                batch_size=128,
                train_extra={
                    "swa": {
                        "enable": True,
                        "start_fraction": 0.5,
                        "update_every": 1,
                        "selection_metric": "val_mse",
                        "min_delta": 0.0,
                    }
                },
                early_stop_patience=20,
            ),
        ),
        Candidate(
            "model",
            "electric_h720_centerres_h192_a08_wd1e4_do002_bs8",
            electric_center_residual_patch(hidden_dim=192, alpha=0.8, dropout=0.02, weight_decay=1.0e-4),
        ),
        Candidate(
            "model",
            "electric_h720_centerres_h128_a06_wd1e4_do002_bs8",
            electric_center_residual_patch(hidden_dim=128, alpha=0.6, dropout=0.02, weight_decay=1.0e-4),
        ),
        Candidate(
            "model",
            "electric_h720_centerres_h192_a08_wd1e5_do0_bs8",
            electric_center_residual_patch(hidden_dim=192, alpha=0.8, dropout=0.0, weight_decay=1.0e-5),
        ),
        Candidate(
            "model",
            "electric_h720_centerres_h192_a07_wd1e5_do0_bs8",
            electric_center_residual_patch(hidden_dim=192, alpha=0.7, dropout=0.0, weight_decay=1.0e-5),
        ),
        Candidate(
            "model",
            "electric_h720_centerres_h192_a09_wd1e5_do0_bs8",
            electric_center_residual_patch(hidden_dim=192, alpha=0.9, dropout=0.0, weight_decay=1.0e-5),
        ),
        Candidate(
            "model",
            "electric_h720_centerres_h160_a08_wd1e5_do0_bs8",
            electric_center_residual_patch(hidden_dim=160, alpha=0.8, dropout=0.0, weight_decay=1.0e-5),
        ),
        Candidate(
            "model",
            "electric_h720_centerres_h224_a08_wd1e5_do0_bs8",
            electric_center_residual_patch(hidden_dim=224, alpha=0.8, dropout=0.0, weight_decay=1.0e-5),
        ),
        Candidate(
            "model",
            "electric_h720_centerres_h224_a07_wd1e5_do0_bs8",
            electric_center_residual_patch(hidden_dim=224, alpha=0.7, dropout=0.0, weight_decay=1.0e-5),
        ),
        Candidate(
            "model",
            "electric_h720_centerres_h224_a09_wd1e5_do0_bs8",
            electric_center_residual_patch(hidden_dim=224, alpha=0.9, dropout=0.0, weight_decay=1.0e-5),
        ),
        Candidate(
            "model",
            "electric_h720_centerres_h256_a08_wd1e5_do0_bs8",
            electric_center_residual_patch(hidden_dim=256, alpha=0.8, dropout=0.0, weight_decay=1.0e-5),
        ),
        Candidate(
            "model",
            "electric_h720_centerres_h288_a08_wd1e5_do0_bs8",
            electric_center_residual_patch(hidden_dim=288, alpha=0.8, dropout=0.0, weight_decay=1.0e-5),
        ),
        Candidate(
            "model",
            "electric_h720_centerres_h224_a08_wd1e5_do0_bs16",
            electric_center_residual_patch(hidden_dim=224, alpha=0.8, dropout=0.0, weight_decay=1.0e-5, batch_size=16),
        ),
        Candidate(
            "model",
            "electric_h720_centerres_h224_a08_wd1e5_modelstat_select_m300_seg12_bs8",
            electric_center_residual_patch(
                hidden_dim=224,
                alpha=0.8,
                dropout=0.0,
                weight_decay=1.0e-5,
                batch_size=8,
                model_extra={
                    "train_stat_adapter": {
                        "scale_selection": {
                            "enable": True,
                            "metric": "mse",
                            "max_scale": 3.0,
                            "steps": 121,
                            "horizon_segments": 12,
                        }
                    }
                },
            ),
        ),
        Candidate(
            "model",
            "electric_h720_centerres_h224_a08_wd1e5_modelstat_select_m120_channel_bs8",
            electric_center_residual_patch(
                hidden_dim=224,
                alpha=0.8,
                dropout=0.0,
                weight_decay=1.0e-5,
                batch_size=8,
                model_extra={
                    "train_stat_adapter": {
                        "scale_selection": {
                            "enable": True,
                            "metric": "mse",
                            "max_scale": 1.2,
                            "steps": 49,
                            "horizon_segments": 1,
                        }
                    }
                },
            ),
        ),
        Candidate(
            "model",
            "electric_h720_centerres_segment_h224_a08_wd1e5_do0_bs8",
            electric_center_residual_patch(
                predictor="segment_mlp",
                hidden_dim=224,
                alpha=0.8,
                dropout=0.0,
                weight_decay=1.0e-5,
                batch_size=8,
                model_extra={"segment_chunk_len": 96},
            ),
        ),
        Candidate(
            "model",
            "electric_h720_centerres_longanchor_h224_a08_d015_wd1e5_do0_bs8",
            electric_center_residual_patch(
                predictor="long_anchor_mlp",
                hidden_dim=224,
                alpha=0.8,
                dropout=0.0,
                weight_decay=1.0e-5,
                batch_size=8,
                model_extra={"anchor_chunk_len": 96, "anchor_detail_scale": 0.15, "anchor_residual": True},
            ),
        ),
        Candidate(
            "model",
            "electric_h720_centerres_longanchor_h224_a08_d025_wd1e5_do0_bs8",
            electric_center_residual_patch(
                predictor="long_anchor_mlp",
                hidden_dim=224,
                alpha=0.8,
                dropout=0.0,
                weight_decay=1.0e-5,
                batch_size=8,
                model_extra={"anchor_chunk_len": 96, "anchor_detail_scale": 0.25, "anchor_residual": True},
            ),
        ),
        Candidate(
            "model",
            "electric_h720_centerres_longanchor_h224_a08_d045_wd1e5_do0_bs8",
            electric_center_residual_patch(
                predictor="long_anchor_mlp",
                hidden_dim=224,
                alpha=0.8,
                dropout=0.0,
                weight_decay=1.0e-5,
                batch_size=8,
                model_extra={"anchor_chunk_len": 96, "anchor_detail_scale": 0.45, "anchor_residual": True},
            ),
        ),
        Candidate(
            "model",
            "electric_h720_centerres_longanchor_h256_a08_d025_wd1e5_do0_bs8",
            electric_center_residual_patch(
                predictor="long_anchor_mlp",
                hidden_dim=256,
                alpha=0.8,
                dropout=0.0,
                weight_decay=1.0e-5,
                batch_size=8,
                model_extra={"anchor_chunk_len": 96, "anchor_detail_scale": 0.25, "anchor_residual": True},
            ),
        ),
        Candidate(
            "model",
            "electric_h720_centerres_recursive_h256_a08_wd1e5_do0_bs8",
            electric_center_residual_patch(
                hidden_dim=256,
                alpha=0.8,
                dropout=0.0,
                weight_decay=1.0e-5,
                batch_size=8,
                model_extra={"recursive_rollout": True, "recursive_chunk_len": 96},
            ),
        ),
        Candidate(
            "model",
            "electric_h720_centerres_h192_a08_wd0_do0_bs8",
            electric_center_residual_patch(hidden_dim=192, alpha=0.8, dropout=0.0, weight_decay=0.0),
        ),
        Candidate(
            "model",
            "electric_h720_centerres_h192_a08_wd1e5_do0_lr8e4_bs8",
            electric_center_residual_patch(
                hidden_dim=192,
                alpha=0.8,
                dropout=0.0,
                weight_decay=1.0e-5,
                lr=8.0e-4,
            ),
        ),
        Candidate(
            "model",
            "electric_h96_centerres_channel_h128_a10_wd1e5_do0_bs64",
            electric_center_residual_patch(
                predictor="channel_head_mlp",
                hidden_dim=128,
                alpha=1.0,
                dropout=0.0,
                weight_decay=1.0e-5,
                batch_size=64,
            ),
        ),
        Candidate(
            "model",
            "electric_h96_centerres_channel_h192_a10_wd1e5_do0_bs64",
            electric_center_residual_patch(
                predictor="channel_head_mlp",
                hidden_dim=192,
                alpha=1.0,
                dropout=0.0,
                weight_decay=1.0e-5,
                batch_size=64,
            ),
        ),
        Candidate(
            "model",
            "electric_h96_centerres_context_h96_a10_wd1e5_do0_bs64",
            electric_center_residual_patch(
                predictor="context_channel_head_mlp",
                hidden_dim=96,
                alpha=1.0,
                dropout=0.0,
                weight_decay=1.0e-5,
                batch_size=64,
            ),
        ),
        Candidate(
            "model",
            "electric_h96_centerres_context_h128_a10_wd1e5_do0_bs64",
            electric_center_residual_patch(
                predictor="context_channel_head_mlp",
                hidden_dim=128,
                alpha=1.0,
                dropout=0.0,
                weight_decay=1.0e-5,
                batch_size=64,
            ),
        ),
        Candidate(
            "model",
            "electric_h720_centerres_channel_h192_a08_wd1e5_do0_bs8",
            electric_center_residual_patch(
                predictor="channel_head_mlp",
                hidden_dim=192,
                alpha=0.8,
                dropout=0.0,
                weight_decay=1.0e-5,
                batch_size=8,
            ),
        ),
        Candidate(
            "model",
            "electric_h720_centerres_context_h96_a08_wd1e5_do0_bs8",
            electric_center_residual_patch(
                predictor="context_channel_head_mlp",
                hidden_dim=96,
                alpha=0.8,
                dropout=0.0,
                weight_decay=1.0e-5,
                batch_size=8,
            ),
        ),
        Candidate(
            "model",
            "electric_h720_centerres_context_h128_a08_wd1e5_do0_bs8",
            electric_center_residual_patch(
                predictor="context_channel_head_mlp",
                hidden_dim=128,
                alpha=0.8,
                dropout=0.0,
                weight_decay=1.0e-5,
                batch_size=8,
            ),
        ),
        Candidate(
            "model",
            "electric_h720_centerres_h256_a06_wd1e4_do002_bs8",
            electric_center_residual_patch(hidden_dim=256, alpha=0.6, dropout=0.02, weight_decay=1.0e-4),
        ),
        Candidate(
            "model",
            "electric_h720_centerres_h128_a08_wd1e4_do005_bs8",
            electric_center_residual_patch(hidden_dim=128, alpha=0.8, dropout=0.05, weight_decay=1.0e-4),
        ),
        Candidate(
            "model",
            "electric_h96params_h256_do0_modelstat_p168_center_anchorres_a10_lr18e4",
            electric_h96_transfer_patch(
                lr=1.8e-3,
                model_extra={
                    "train_stat_adapter": {
                        "enable": True,
                        "period": 168,
                        "mode": "phase_mean",
                        "alpha": 1.0,
                        "blend_target": "prediction",
                        "combine_mode": "anchor_plus_prediction",
                        "input_center": True,
                    },
                }
            ),
        ),
        Candidate(
            "model",
            "electric_h96params_h256_do0_seasonalblend_p24_m10_modelstat_p168_a025",
            electric_h96_transfer_patch(
                model_extra={
                    "seasonal_blend_adapter": {
                        "enable": True,
                        "period": 24,
                        "num_periods": 4,
                        "max_mix": 1.0,
                        "init_mix": 0.5,
                    },
                    "train_stat_adapter": {
                        "enable": True,
                        "period": 168,
                        "mode": "phase_mean",
                        "alpha": 0.25,
                        "blend_target": "prediction",
                    },
                }
            ),
        ),
        Candidate(
            "model",
            "electric_h96params_h256_do0_modelstat_p168_mean_select_m02_seg12",
            model_train_stat_adapter_patch(period=168, mode="phase_mean", max_scale=0.2, horizon_segments=12),
        ),
        Candidate(
            "model",
            "electric_h96params_h256_do0_modelstat_p168_mean_select_m03_seg12",
            model_train_stat_adapter_patch(period=168, mode="phase_mean", max_scale=0.3, horizon_segments=12),
        ),
        Candidate(
            "model",
            "electric_h96params_h256_do0_modelstat_p168_mean_select_m05_seg12",
            model_train_stat_adapter_patch(period=168, mode="phase_mean", max_scale=0.5, horizon_segments=12),
        ),
        Candidate(
            "model",
            "electric_h96params_h256_do0_modelstat_p24_delta_repeat_a010",
            model_train_stat_adapter_patch(period=24, mode="phase_delta", reference="repeat", alpha=0.1),
        ),
        Candidate(
            "model",
            "electric_h96params_h256_do0_modelstat_p24_delta_repeat_select_m05_seg12",
            model_train_stat_adapter_patch(
                period=24,
                mode="phase_delta",
                reference="repeat",
                max_scale=0.5,
                horizon_segments=12,
            ),
        ),
        Candidate(
            "model",
            "channel_h96_do0_wd1e4_mae04",
            {
                "model": {"predictor": "channel_head_mlp", "hidden_dim": 96, "dropout": 0.0},
                "train": {"weight_decay": 1.0e-4, "mae_objective": {"enable": True, "kind": "l1", "weight": 0.4}},
            },
        ),
        Candidate(
            "model",
            "channel_h128_do0_wd1e4_mae04",
            {
                "model": {"predictor": "channel_head_mlp", "hidden_dim": 128, "dropout": 0.0},
                "train": {"weight_decay": 1.0e-4, "mae_objective": {"enable": True, "kind": "l1", "weight": 0.4}},
            },
        ),
        Candidate(
            "model",
            "channel_h128_do01_wd1e4_mae04",
            {
                "model": {"predictor": "channel_head_mlp", "hidden_dim": 128, "dropout": 0.1},
                "train": {"weight_decay": 1.0e-4, "mae_objective": {"enable": True, "kind": "l1", "weight": 0.4}},
            },
        ),
        Candidate(
            "model",
            "channel_h128_do02_wd1e4_mae02",
            {
                "model": {"predictor": "channel_head_mlp", "hidden_dim": 128, "dropout": 0.2},
                "train": {"weight_decay": 1.0e-4, "mae_objective": {"enable": True, "kind": "l1", "weight": 0.2}},
            },
        ),
        Candidate(
            "model",
            "channel_h96_do02_wd1e4_mae02",
            {
                "model": {"predictor": "channel_head_mlp", "hidden_dim": 96, "dropout": 0.2},
                "train": {"weight_decay": 1.0e-4, "mae_objective": {"enable": True, "kind": "l1", "weight": 0.2}},
            },
        ),
        Candidate(
            "model",
            "channel_h128_do015_wd1e4_mae02",
            {
                "model": {"predictor": "channel_head_mlp", "hidden_dim": 128, "dropout": 0.15},
                "train": {"weight_decay": 1.0e-4, "mae_objective": {"enable": True, "kind": "l1", "weight": 0.2}},
            },
        ),
        Candidate(
            "model",
            "channel_h128_do025_wd1e4_mae02",
            {
                "model": {"predictor": "channel_head_mlp", "hidden_dim": 128, "dropout": 0.25},
                "train": {"weight_decay": 1.0e-4, "mae_objective": {"enable": True, "kind": "l1", "weight": 0.2}},
            },
        ),
        Candidate(
            "model",
            "channel_h128_do02_wd3e4_mae02",
            {
                "model": {"predictor": "channel_head_mlp", "hidden_dim": 128, "dropout": 0.2},
                "train": {"weight_decay": 3.0e-4, "mae_objective": {"enable": True, "kind": "l1", "weight": 0.2}},
            },
        ),
        Candidate(
            "model",
            "channel_h128_do02_wd1e4_mae01",
            {
                "model": {"predictor": "channel_head_mlp", "hidden_dim": 128, "dropout": 0.2},
                "train": {"weight_decay": 1.0e-4, "mae_objective": {"enable": True, "kind": "l1", "weight": 0.1}},
            },
        ),
        Candidate(
            "model",
            "channel_h128_do02_wd1e4_mae03",
            {
                "model": {"predictor": "channel_head_mlp", "hidden_dim": 128, "dropout": 0.2},
                "train": {"weight_decay": 1.0e-4, "mae_objective": {"enable": True, "kind": "l1", "weight": 0.3}},
            },
        ),
        Candidate(
            "model",
            "channel_h160_do02_wd1e4_mae02",
            {
                "model": {"predictor": "channel_head_mlp", "hidden_dim": 160, "dropout": 0.2},
                "train": {"weight_decay": 1.0e-4, "mae_objective": {"enable": True, "kind": "l1", "weight": 0.2}},
            },
        ),
        Candidate(
            "model",
            "channel_h160_do02_wd1e4_mae04",
            {
                "model": {"predictor": "channel_head_mlp", "hidden_dim": 160, "dropout": 0.2},
                "train": {"weight_decay": 1.0e-4, "mae_objective": {"enable": True, "kind": "l1", "weight": 0.4}},
            },
        ),
        Candidate(
            "model",
            "channel_h160_do02_wd3e4_mae02",
            {
                "model": {"predictor": "channel_head_mlp", "hidden_dim": 160, "dropout": 0.2},
                "train": {"weight_decay": 3.0e-4, "mae_objective": {"enable": True, "kind": "l1", "weight": 0.2}},
            },
        ),
        Candidate(
            "model",
            "channel_h192_do01_wd1e4_mae04",
            {
                "model": {"predictor": "channel_head_mlp", "hidden_dim": 192, "dropout": 0.1},
                "train": {"weight_decay": 1.0e-4, "mae_objective": {"enable": True, "kind": "l1", "weight": 0.4}},
            },
        ),
        Candidate(
            "model",
            "channel_h192_do01_wd1e4_mae06",
            {
                "model": {"predictor": "channel_head_mlp", "hidden_dim": 192, "dropout": 0.1},
                "train": {"weight_decay": 1.0e-4, "mae_objective": {"enable": True, "kind": "l1", "weight": 0.6}},
            },
        ),
        Candidate(
            "model",
            "channel_h192_do02_wd1e4_mae04",
            {
                "model": {"predictor": "channel_head_mlp", "hidden_dim": 192, "dropout": 0.2},
                "train": {"weight_decay": 1.0e-4, "mae_objective": {"enable": True, "kind": "l1", "weight": 0.4}},
            },
        ),
        Candidate(
            "model",
            "seasonal_channel_h128_do02_wd1e4_mae02",
            seasonal_channel_patch(
                hidden_dim=128,
                dropout=0.2,
                predictor_input_len=96,
                mix_init=-2.0,
                gate_strength=0.0,
            ),
        ),
        Candidate(
            "model",
            "seasonal_channel_h128_tail24_mixm2_do02_wd1e4_mae02",
            seasonal_channel_patch(hidden_dim=128, dropout=0.2, predictor_input_len=24, mix_init=-2.0),
        ),
        Candidate(
            "model",
            "seasonal_channel_h128_tail24_mix0_do02_wd1e4_mae02",
            seasonal_channel_patch(hidden_dim=128, dropout=0.2, predictor_input_len=24, mix_init=0.0),
        ),
        Candidate(
            "model",
            "seasonal_channel_h96_tail24_mixm2_do02_wd1e4_mae02",
            seasonal_channel_patch(hidden_dim=96, dropout=0.2, predictor_input_len=24, mix_init=-2.0),
        ),
        Candidate(
            "model",
            "seasonal_channel_h160_tail24_mixm2_do02_wd1e4_mae02",
            seasonal_channel_patch(hidden_dim=160, dropout=0.2, predictor_input_len=24, mix_init=-2.0),
        ),
        Candidate(
            "model",
            "seasonal_channel_h192_tail24_mixm2_do02_wd1e4_mae02",
            seasonal_channel_patch(hidden_dim=192, dropout=0.2, predictor_input_len=24, mix_init=-2.0),
        ),
        Candidate(
            "model",
            "seasonal_channel_h256_tail24_mixm2_do02_wd1e4_mae02",
            seasonal_channel_patch(hidden_dim=256, dropout=0.2, predictor_input_len=24, mix_init=-2.0),
        ),
        Candidate(
            "model",
            "seasonal_channel_h128_tail24_mixm2_do01_wd1e4_mae02",
            seasonal_channel_patch(hidden_dim=128, dropout=0.1, predictor_input_len=24, mix_init=-2.0),
        ),
        Candidate(
            "model",
            "seasonal_channel_h128_tail24_mixm2_do025_wd1e4_mae02",
            seasonal_channel_patch(hidden_dim=128, dropout=0.25, predictor_input_len=24, mix_init=-2.0),
        ),
        Candidate(
            "model",
            "mlp_h64_do0_wd1e4_mae04",
            {
                "model": {"predictor": "mlp", "hidden_dim": 64, "dropout": 0.0},
                "train": {"weight_decay": 1.0e-4, "mae_objective": {"enable": True, "kind": "l1", "weight": 0.4}},
            },
        ),
        Candidate(
            "model",
            "mlp_h64_do02_wd1e4_mae04",
            {
                "model": {"predictor": "mlp", "hidden_dim": 64, "dropout": 0.2},
                "train": {"weight_decay": 1.0e-4, "mae_objective": {"enable": True, "kind": "l1", "weight": 0.4}},
            },
        ),
        Candidate(
            "model",
            "mlp_h64_do0_wd1e4_mseonly",
            {
                "model": {"predictor": "mlp", "hidden_dim": 64, "dropout": 0.0},
                "train": {"weight_decay": 1.0e-4, "mae_objective": {"enable": False, "weight": 0.0}},
            },
        ),
        Candidate(
            "model",
            "mlp_h96_do0_wd1e4_mae04",
            {
                "model": {"predictor": "mlp", "hidden_dim": 96, "dropout": 0.0},
                "train": {"weight_decay": 1.0e-4, "mae_objective": {"enable": True, "kind": "l1", "weight": 0.4}},
            },
        ),
        Candidate(
            "model",
            "mlp_h96_do02_wd1e4_mae04",
            {
                "model": {"predictor": "mlp", "hidden_dim": 96, "dropout": 0.2},
                "train": {"weight_decay": 1.0e-4, "mae_objective": {"enable": True, "kind": "l1", "weight": 0.4}},
            },
        ),
        Candidate(
            "model",
            "mlp_h96_do02_wd1e4_mseonly",
            {
                "model": {"predictor": "mlp", "hidden_dim": 96, "dropout": 0.2},
                "train": {"weight_decay": 1.0e-4, "mae_objective": {"enable": False, "weight": 0.0}},
            },
        ),
        Candidate(
            "model",
            "mlp_h128_do0_wd1e4_mae04",
            {
                "model": {"predictor": "mlp", "hidden_dim": 128, "dropout": 0.0},
                "train": {"weight_decay": 1.0e-4, "mae_objective": {"enable": True, "kind": "l1", "weight": 0.4}},
            },
        ),
        Candidate(
            "model",
            "mlp_h128_do02_wd1e4_mae04",
            {
                "model": {"predictor": "mlp", "hidden_dim": 128, "dropout": 0.2},
                "train": {"weight_decay": 1.0e-4, "mae_objective": {"enable": True, "kind": "l1", "weight": 0.4}},
            },
        ),
        Candidate(
            "model",
            "mlp_h128_do02_wd1e4_mae02",
            {
                "model": {"predictor": "mlp", "hidden_dim": 128, "dropout": 0.2},
                "train": {"weight_decay": 1.0e-4, "mae_objective": {"enable": True, "kind": "l1", "weight": 0.2}},
            },
        ),
        Candidate(
            "model",
            "mlp_h128_do02_wd1e4_mseonly",
            {
                "model": {"predictor": "mlp", "hidden_dim": 128, "dropout": 0.2},
                "train": {"weight_decay": 1.0e-4, "mae_objective": {"enable": False, "weight": 0.0}},
            },
        ),
        Candidate(
            "model",
            "mlp_h128_do005_wd5e4_mae07",
            {
                "model": {"predictor": "mlp", "hidden_dim": 128, "dropout": 0.05},
                "train": {"weight_decay": 5.0e-4, "mae_objective": {"enable": True, "kind": "l1", "weight": 0.7}},
            },
        ),
        Candidate(
            "model",
            "mlp_h160_do005_wd5e4_mae07",
            {
                "model": {"predictor": "mlp", "hidden_dim": 160, "dropout": 0.05},
                "train": {"weight_decay": 5.0e-4, "mae_objective": {"enable": True, "kind": "l1", "weight": 0.7}},
            },
        ),
        Candidate(
            "model",
            "mlp_h192_do005_wd5e4_mae07",
            {
                "model": {"predictor": "mlp", "hidden_dim": 192, "dropout": 0.05},
                "train": {"weight_decay": 5.0e-4, "mae_objective": {"enable": True, "kind": "l1", "weight": 0.7}},
            },
        ),
        Candidate(
            "model",
            "mlp_h160_do02_wd1e4_mae04",
            {
                "model": {"predictor": "mlp", "hidden_dim": 160, "dropout": 0.2},
                "train": {"weight_decay": 1.0e-4, "mae_objective": {"enable": True, "kind": "l1", "weight": 0.4}},
            },
        ),
        Candidate(
            "model",
            "mlp_h256_do02_wd1e3_mae06",
            {
                "model": {"predictor": "mlp", "hidden_dim": 256, "dropout": 0.2},
                "train": {"weight_decay": 1.0e-3, "mae_objective": {"enable": True, "kind": "l1", "weight": 0.6}},
            },
        ),
        Candidate(
            "model",
            "mlp_h192_do03_wd1e3_mae04",
            {
                "model": {"predictor": "mlp", "hidden_dim": 192, "dropout": 0.3},
                "train": {"weight_decay": 1.0e-3, "mae_objective": {"enable": True, "kind": "l1", "weight": 0.4}},
            },
        ),
        Candidate(
            "model",
            "mlp_h384_do01_wd1e3_mae04",
            {
                "model": {"predictor": "mlp", "hidden_dim": 384, "dropout": 0.1},
                "train": {"weight_decay": 1.0e-3, "mae_objective": {"enable": True, "kind": "l1", "weight": 0.4}},
            },
        ),
        Candidate(
            "model",
            "mlp_h512_do01_wd1e3_mae04",
            {
                "model": {"predictor": "mlp", "hidden_dim": 512, "dropout": 0.1},
                "train": {"weight_decay": 1.0e-3, "mae_objective": {"enable": True, "kind": "l1", "weight": 0.4}},
            },
        ),
        Candidate(
            "model",
            "mlp_anchor_basis_h256_r16_wd1e4_mae06",
            {
                "model": {
                    "predictor": "mlp",
                    "hidden_dim": 256,
                    "dropout": 0.2,
                    "mlp_residual_anchor": True,
                    "temporal_basis_adapter": {
                        "enable": True,
                        "rank": 16,
                        "scale": 0.15,
                        "init": "zero_delta",
                    },
                },
                "train": {
                    "weight_decay": 1.0e-4,
                    "mae_objective": {"enable": True, "kind": "l1", "weight": 0.6},
                },
            },
        ),
        Candidate(
            "model",
            "context_h256_do02_wd1e3_mae04",
            {
                "model": {"predictor": "context_mlp", "hidden_dim": 256, "dropout": 0.2, "context_include_delta": True},
                "train": {"weight_decay": 1.0e-3, "mae_objective": {"enable": True, "kind": "l1", "weight": 0.4}},
            },
        ),
        Candidate(
            "model",
            "context_h512_do01_wd1e3_mae04",
            {
                "model": {"predictor": "context_mlp", "hidden_dim": 512, "dropout": 0.1, "context_include_delta": True},
                "train": {"weight_decay": 1.0e-3, "mae_objective": {"enable": True, "kind": "l1", "weight": 0.4}},
            },
        ),
        Candidate(
            "model",
            "channel_h256_do02_wd1e3_mae04",
            {
                "model": {"predictor": "channel_head_mlp", "hidden_dim": 256, "dropout": 0.2, "channel_head_residual": True},
                "train": {"weight_decay": 1.0e-3, "mae_objective": {"enable": True, "kind": "l1", "weight": 0.4}},
            },
        ),
        Candidate(
            "model",
            "channel_h256_do02_wd1e4_mae04",
            {
                "model": {"predictor": "channel_head_mlp", "hidden_dim": 256, "dropout": 0.2, "channel_head_residual": True},
                "train": {"weight_decay": 1.0e-4, "mae_objective": {"enable": True, "kind": "l1", "weight": 0.4}},
            },
        ),
        Candidate(
            "model",
            "channel_h256_do0_wd1e3_mae06",
            {
                "model": {"predictor": "channel_head_mlp", "hidden_dim": 256, "dropout": 0.0, "channel_head_residual": True},
                "train": {"weight_decay": 1.0e-3, "mae_objective": {"enable": True, "kind": "l1", "weight": 0.6}},
            },
        ),
        Candidate(
            "model",
            "channel_h512_do01_wd1e3_mae04",
            {
                "model": {"predictor": "channel_head_mlp", "hidden_dim": 512, "dropout": 0.1, "channel_head_residual": True},
                "train": {"weight_decay": 1.0e-3, "mae_objective": {"enable": True, "kind": "l1", "weight": 0.4}},
            },
        ),
        Candidate(
            "model",
            "attn_h256_do02_wd1e3_mae04",
            {
                "model": {"predictor": "attn_mlp", "hidden_dim": 256, "dropout": 0.2, "attn_dim": 64},
                "train": {"weight_decay": 1.0e-3, "mae_objective": {"enable": True, "kind": "l1", "weight": 0.4}},
            },
        ),
        Candidate(
            "model",
            "patchtst_h128_p16s8_l2_do01_wd1e4_mae04",
            {
                "model": {
                    "predictor": "patchtst",
                    "hidden_dim": 128,
                    "dropout": 0.1,
                    "patch_d_model": 128,
                    "patch_len": 16,
                    "patch_stride": 8,
                    "patch_num_layers": 2,
                    "patch_num_heads": 4,
                    "patch_ff_dim": 256,
                },
                "train": {"weight_decay": 1.0e-4, "mae_objective": {"enable": True, "kind": "l1", "weight": 0.4}},
            },
        ),
        Candidate(
            "model",
            "patchtst_h192_p16s8_l2_do01_wd1e4_mae04",
            {
                "model": {
                    "predictor": "patchtst",
                    "hidden_dim": 192,
                    "dropout": 0.1,
                    "patch_d_model": 192,
                    "patch_len": 16,
                    "patch_stride": 8,
                    "patch_num_layers": 2,
                    "patch_num_heads": 4,
                    "patch_ff_dim": 384,
                },
                "train": {"weight_decay": 1.0e-4, "mae_objective": {"enable": True, "kind": "l1", "weight": 0.4}},
            },
        ),
        Candidate(
            "model",
            "channel_h256_do02_wd1e3_thr05",
            {
                "cluster": {"distance_threshold": 0.5},
                "model": {"predictor": "channel_head_mlp", "hidden_dim": 256, "dropout": 0.2, "channel_head_residual": True},
                "train": {"weight_decay": 1.0e-3, "mae_objective": {"enable": True, "kind": "l1", "weight": 0.4}},
            },
        ),
    ]


def limit_candidates(candidates: list[Candidate], budget: str, *, smoke_n: int, local_n: int) -> list[Candidate]:
    if budget == "smoke":
        return candidates[:smoke_n]
    if budget == "local":
        return candidates[:local_n]
    return candidates


def filter_candidates_by_variant(candidates: list[Candidate], variants: list[str] | None) -> list[Candidate]:
    if not variants:
        return candidates
    by_variant = {cand.variant: cand for cand in candidates}
    missing = [variant for variant in variants if variant not in by_variant]
    if missing:
        raise ValueError(f"Unknown candidate variants: {missing}. Supported: {sorted(by_variant)}")
    return [by_variant[variant] for variant in variants]


def moe_candidates(base_cfg: dict[str, Any], budget: str) -> list[Candidate]:
    current_penalties = list(base_cfg.get("penalties", {}).get("enabled", ["level"]))
    base_alpha = base_cfg.get("moe", {}).get("pred_side_residual", {}).get("alpha_scale", 0.8)
    pools = [
        {
            "name": "current_moe",
            "penalties": current_penalties,
            "lam": None,
            "dyn": True,
            "alpha": None,
            "policy": "val_mse_gate",
            "gate": {"epochs": 20, "batch_size": 256},
        },
        {
            "name": "current_guard_l005_a03_ms035",
            "penalties": current_penalties,
            "lam": 0.005,
            "dyn": False,
            "alpha": 0.3,
            "policy": "val_mse_gate_guarded",
            "min_rel": 0.0005,
            "residual_clip": 2.0,
            "gate": {
                "epochs": 30,
                "batch_size": 256,
                "max_scale": 0.35,
                "init_scale": 0.2,
                "scale_reg": 0.001,
                "standardize_features": True,
            },
        },
        {
            "name": "current_guard_l01_a05_ms05",
            "penalties": current_penalties,
            "lam": 0.01,
            "dyn": False,
            "alpha": 0.5,
            "policy": "val_mse_gate_guarded",
            "min_rel": 0.0005,
            "residual_clip": 2.0,
            "gate": {
                "epochs": 30,
                "batch_size": 256,
                "max_scale": 0.5,
                "init_scale": 0.3,
                "scale_reg": 0.0005,
                "standardize_features": True,
            },
        },
        {
            "name": "current_guard_l0_a025_ms025",
            "penalties": current_penalties,
            "lam": 0.0,
            "dyn": False,
            "alpha": 0.25,
            "policy": "val_mse_gate_guarded",
            "min_rel": 0.0005,
            "residual_clip": 2.0,
            "gate": {
                "epochs": 30,
                "batch_size": 256,
                "max_scale": 0.25,
                "init_scale": 0.15,
                "scale_reg": 0.001,
                "standardize_features": True,
            },
        },
        {
            "name": "current_prior_l0_a025_ms025_top1",
            "penalties": current_penalties,
            "lam": 0.0,
            "dyn": False,
            "alpha": 0.25,
            "policy": "val_mse_gate_guarded",
            "min_rel": 0.0005,
            "residual_clip": 2.0,
            "gate": {
                "epochs": 30,
                "batch_size": 256,
                "max_scale": 0.25,
                "init_scale": 0.15,
                "scale_reg": 0.001,
                "standardize_features": True,
            },
            "cluster_penalty_prior": {
                "enable": True,
                "topk": 1,
                "hard_topk": True,
                "temperature": 0.7,
                "smoothing": 0.02,
                "use_normalized_penalty": True,
                "logit_strength": 1.0,
            },
            "channel_penalty_prior": {
                "enable": True,
                "topk": 1,
                "hard_topk": True,
                "temperature": 0.7,
                "smoothing": 0.02,
                "use_normalized_penalty": True,
            },
        },
        {
            "name": "current_scale_l005_a05_s075",
            "penalties": current_penalties,
            "lam": 0.005,
            "dyn": False,
            "alpha": 0.5,
            "policy": "val_mse_scale",
            "min_rel": 0.0005,
            "residual_clip": 2.0,
            "scale": {"selection_scale_min": 0.0, "selection_scale_max": 0.75, "selection_scale_steps": 16},
        },
        {
            "name": "scale_l0_a1_s125_h64",
            "penalties": current_penalties,
            "lam": 0.0,
            "dyn": False,
            "alpha": 1.0,
            "policy": "val_mse_scale",
            "min_rel": 0.0005,
            "residual_clip": 4.0,
            "scale": {"selection_scale_min": 0.0, "selection_scale_max": 1.25, "selection_scale_steps": 26},
            "residual_overrides": {"corrector_hidden": 64, "init_alpha": -2.5, "norm_weight": 1.0e-5},
        },
        {
            "name": "scale_l005_a1_s125_h64",
            "penalties": current_penalties,
            "lam": 0.005,
            "dyn": False,
            "alpha": 1.0,
            "policy": "val_mse_scale",
            "min_rel": 0.0005,
            "residual_clip": 4.0,
            "scale": {"selection_scale_min": 0.0, "selection_scale_max": 1.25, "selection_scale_steps": 26},
            "residual_overrides": {"corrector_hidden": 64, "init_alpha": -2.5, "norm_weight": 1.0e-5},
        },
        {
            "name": "scale_l01_a1_s125_h64",
            "penalties": current_penalties,
            "lam": 0.01,
            "dyn": False,
            "alpha": 1.0,
            "policy": "val_mse_scale",
            "min_rel": 0.0005,
            "residual_clip": 4.0,
            "scale": {"selection_scale_min": 0.0, "selection_scale_max": 1.25, "selection_scale_steps": 26},
            "residual_overrides": {"corrector_hidden": 64, "init_alpha": -2.5, "norm_weight": 1.0e-5},
        },
        {
            "name": "scale_l005_a15_s15_h64",
            "penalties": current_penalties,
            "lam": 0.005,
            "dyn": False,
            "alpha": 1.5,
            "policy": "val_mse_scale",
            "min_rel": 0.0005,
            "residual_clip": 4.0,
            "scale": {"selection_scale_min": 0.0, "selection_scale_max": 1.5, "selection_scale_steps": 31},
            "residual_overrides": {"corrector_hidden": 64, "init_alpha": -2.5, "norm_weight": 1.0e-5},
        },
        {
            "name": "scale_l01_a2_s15_h64",
            "penalties": current_penalties,
            "lam": 0.01,
            "dyn": False,
            "alpha": 2.0,
            "policy": "val_mse_scale",
            "min_rel": 0.0005,
            "residual_clip": 4.0,
            "scale": {"selection_scale_min": 0.0, "selection_scale_max": 1.5, "selection_scale_steps": 31},
            "residual_overrides": {"corrector_hidden": 64, "init_alpha": -2.0, "norm_weight": 1.0e-5},
        },
        {
            "name": "scale_l005_a15_s15_top2_h64",
            "penalties": current_penalties,
            "lam": 0.005,
            "dyn": False,
            "alpha": 1.5,
            "policy": "val_mse_scale",
            "min_rel": 0.0005,
            "residual_clip": 4.0,
            "scale": {"selection_scale_min": 0.0, "selection_scale_max": 1.5, "selection_scale_steps": 31},
            "residual_overrides": {"corrector_hidden": 64, "init_alpha": -2.5, "norm_weight": 1.0e-5},
            "moe_overrides": {"topk": 2, "select_ranks": [1, 2]},
        },
        {
            "name": "scale_l005_a15_s15_ctx1_h64",
            "penalties": current_penalties,
            "lam": 0.005,
            "dyn": False,
            "alpha": 1.5,
            "policy": "val_mse_scale",
            "min_rel": 0.0005,
            "residual_clip": 4.0,
            "scale": {"selection_scale_min": 0.0, "selection_scale_max": 1.5, "selection_scale_steps": 31},
            "residual_overrides": {"corrector_hidden": 64, "init_alpha": -2.5, "norm_weight": 1.0e-5},
            "moe_overrides": {
                "router_mode": "penalty_context",
                "router_penalty_context_weight": 1.0,
                "router_penalty_context_score": "high_violation",
            },
        },
        {
            "name": "scale_l005_a15_s15_penaltyonly_h64",
            "penalties": current_penalties,
            "lam": 0.005,
            "dyn": False,
            "alpha": 1.5,
            "policy": "val_mse_scale",
            "min_rel": 0.0005,
            "residual_clip": 4.0,
            "scale": {"selection_scale_min": 0.0, "selection_scale_max": 1.5, "selection_scale_steps": 31},
            "residual_overrides": {"corrector_hidden": 64, "init_alpha": -2.5, "norm_weight": 1.0e-5},
            "moe_overrides": {
                "router_mode": "penalty_only",
                "router_penalty_context_weight": 1.0,
                "router_penalty_context_score": "high_violation",
            },
        },
        {
            "name": "level_delta_l015",
            "penalties": ["level", "delta"],
            "lam": 0.015,
            "dyn": True,
            "alpha": 0.8,
            "policy": "val_mse_gate",
            "gate": {"epochs": 20, "batch_size": 256},
        },
        {
            "name": "level_delta_diff_l015",
            "penalties": ["level", "delta", "diff_amp"],
            "lam": 0.015,
            "dyn": True,
            "alpha": 0.8,
            "policy": "val_mse_gate",
            "gate": {"epochs": 20, "batch_size": 256},
        },
        {
            "name": "level_delta_diff_l05",
            "penalties": ["level", "delta", "diff_amp"],
            "lam": 0.05,
            "dyn": True,
            "alpha": 1.1,
            "policy": "val_mse_gate",
            "gate": {"epochs": 20, "batch_size": 256},
        },
        {
            "name": "level_delta_d2_diff_l05",
            "penalties": ["level", "delta", "d2_match", "diff_amp"],
            "lam": 0.05,
            "dyn": True,
            "alpha": 1.1,
            "policy": "val_mse_gate",
            "gate": {"epochs": 20, "batch_size": 256},
        },
        {
            "name": "amp_delta_diff_dir_l01",
            "penalties": ["amp_under", "delta", "diff_amp", "direction"],
            "lam": 0.01,
            "dyn": False,
            "alpha": 0.6,
            "policy": "val_mse_gate",
            "gate": {"epochs": 20, "batch_size": 256},
        },
        {
            "name": "trend_direction_l02",
            "penalties": ["trend", "direction"],
            "lam": 0.02,
            "dyn": True,
            "alpha": 0.8,
            "policy": "val_mse_gate",
            "gate": {"epochs": 20, "batch_size": 256},
        },
    ]
    candidates: list[Candidate] = []
    for pool in pools:
        penalties = list(pool["penalties"])
        lam = pool["lam"]
        residual: dict[str, Any] = {
            "enable": True,
            "selection_policy": str(pool.get("policy", "val_mse_gate")),
            "alpha_scale": float(pool["alpha"]) if pool.get("alpha") is not None else float(base_alpha),
            "selection_min_rel_improvement": float(pool.get("min_rel", 0.0)),
            "selection_min_abs_improvement": float(pool.get("min_abs", 0.0)),
            "gate_calibrator": dict(pool.get("gate", {"epochs": 20, "batch_size": 256})),
        }
        if pool.get("residual_clip") is not None:
            residual["residual_clip"] = float(pool["residual_clip"])
        residual.update(dict(pool.get("scale", {})))
        residual.update(dict(pool.get("residual_overrides", {})))
        patch: dict[str, Any] = {
            "moe": {
                "enable": True,
                "dynamic_lambda": {"enable": bool(pool["dyn"])},
                "pred_side_residual": residual,
                "gate_entropy_weight": 0.0,
                "gate_balance_weight": 0.0,
            },
            "penalties": {"enabled": penalties},
        }
        if pool.get("moe_overrides") is not None:
            deep_update(patch["moe"], copy.deepcopy(pool["moe_overrides"]))
        if pool.get("cluster_penalty_prior") is not None:
            patch["moe"]["cluster_penalty_prior"] = dict(pool["cluster_penalty_prior"])
        if pool.get("channel_penalty_prior") is not None:
            patch["moe"]["channel_penalty_prior"] = dict(pool["channel_penalty_prior"])
        if lam is not None:
            patch["moe"]["lambda_init"] = lambda_dict(penalties, lam)
            patch["moe"]["lambda_min"] = lambda_dict(penalties, 0.0)
            patch["moe"]["lambda_schedule"] = schedule_dict(penalties, "none")
        candidates.append(Candidate("moe", str(pool["name"]), patch))
    return limit_candidates(candidates, budget, smoke_n=2, local_n=4)


def run_train(config_path: Path, out_dir: Path, dry_run: bool) -> tuple[int, float]:
    if dry_run:
        return 0, 0.0
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "src.train", "--config", str(config_path)]
    env = dict(**os_environ_utf8())
    t0 = time.perf_counter()
    with (out_dir / "stdout.log").open("w", encoding="utf-8") as stdout_f, (out_dir / "stderr.log").open("w", encoding="utf-8") as stderr_f:
        completed = subprocess.run(cmd, cwd=str(REPO_ROOT), text=True, stdout=stdout_f, stderr=stderr_f, env=env)
    return int(completed.returncode), time.perf_counter() - t0


def os_environ_utf8() -> dict[str, str]:
    import os

    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("MOELOSS_PROGRESS_LEAVE", "0")
    return env


def read_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def row_from_summary(
    *,
    dataset: str,
    cand: Candidate,
    cfg: dict[str, Any],
    config_path: Path,
    out_dir: Path,
    returncode: int,
    total_sec: float,
    error: str = "",
) -> dict[str, Any]:
    summary = read_summary(out_dir / "run_summary.json")
    val = summary.get("val") or {}
    test = summary.get("test") or {}
    moe = cfg.get("moe", {}) or {}
    psr = moe.get("pred_side_residual", {}) or {}
    gate_cal = psr.get("gate_calibrator", {}) or {}
    residual_selection = summary.get("moe_residual_selection", {}) or {}
    train = cfg.get("train", {}) or {}
    mae_obj = train.get("mae_objective", {}) or {}
    return {
        "dataset": dataset,
        "pred_len": int(cfg.get("window", {}).get("pred_len", 96)),
        "stage": cand.stage,
        "variant": cand.variant,
        "status": "ok" if returncode == 0 and summary else ("prepared" if returncode == 0 else "failed"),
        "val_mse": val.get("avg_mse", ""),
        "val_mae": val.get("avg_mae", ""),
        "test_mse": test.get("avg_mse", ""),
        "test_mae": test.get("avg_mae", ""),
        "best_epoch": ",".join(str(v) for v in summary.get("best_epoch", [])) if summary else "",
        "model_hidden_dim": cfg.get("model", {}).get("hidden_dim", ""),
        "model_dropout": cfg.get("model", {}).get("dropout", ""),
        "lr": train.get("lr", ""),
        "weight_decay": train.get("weight_decay", ""),
        "mae_weight": mae_obj.get("weight", ""),
        "moe_enable": moe.get("enable", ""),
        "penalties": ",".join(str(v) for v in cfg.get("penalties", {}).get("enabled", [])),
        "lambda_init": json.dumps(moe.get("lambda_init", ""), sort_keys=True),
        "dynamic_lambda": (moe.get("dynamic_lambda") or {}).get("enable", ""),
        "alpha_scale": psr.get("alpha_scale", ""),
        "residual_clip": psr.get("residual_clip", ""),
        "selection_policy": psr.get("selection_policy", ""),
        "selection_min_rel_improvement": psr.get("selection_min_rel_improvement", ""),
        "gate_max_scale": gate_cal.get("max_scale", ""),
        "gate_init_scale": gate_cal.get("init_scale", ""),
        "residual_mean_scale": residual_selection.get("mean_scale", ""),
        "residual_num_channels": residual_selection.get("num_residual_channels", ""),
        "config_path": str(config_path),
        "out_dir": str(out_dir),
        "total_sec": total_sec,
        "returncode": returncode,
        "error": error,
    }


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDS})


def value(row: dict[str, Any], key: str = "val_mse") -> float:
    try:
        raw = row.get(key, "")
        if raw == "":
            return float("inf")
        return float(raw)
    except Exception:
        return float("inf")


def run_candidate(
    *,
    dataset: str,
    pred_len: int = 96,
    base_cfg: dict[str, Any],
    cand: Candidate,
    out_root: Path,
    device: str | None,
    epochs: int | None,
    skip_test: bool,
    dry_run: bool,
    save_checkpoint: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    horizon_tag = f"H{int(pred_len)}"
    out_dir = out_root / "runs" / dataset / horizon_tag / cand.stage / cand.variant
    cfg = common_prepare(
        base_cfg,
        dataset=dataset,
        pred_len=int(pred_len),
        out_dir=out_dir,
        run_name=f"{dataset}_input96_{horizon_tag}_{cand.stage}_{cand.variant}",
        device=device,
        epochs=epochs,
        skip_test=skip_test,
        save_checkpoint=save_checkpoint,
    )
    deep_update(cfg, copy.deepcopy(cand.patch))
    config_path = out_root / "configs" / dataset / horizon_tag / cand.stage / f"{cand.variant}.yaml"
    write_yaml(config_path, cfg)
    if dry_run:
        returncode, total_sec = 0, 0.0
        error = ""
    else:
        returncode, total_sec = run_train(config_path, out_dir, dry_run=False)
        error = ""
        if returncode != 0:
            err_path = out_dir / "stderr.log"
            error = err_path.read_text(encoding="utf-8", errors="replace")[-2000:] if err_path.exists() else ""
    row = row_from_summary(
        dataset=dataset,
        cand=cand,
        cfg=cfg,
        config_path=config_path,
        out_dir=out_dir,
        returncode=returncode,
        total_sec=total_sec,
        error=error,
    )
    return row, cfg


def select_best(rows: list[dict[str, Any]], metric: str = "val_mse") -> dict[str, Any] | None:
    ok = [r for r in rows if r.get("status") == "ok" and r.get(metric) != ""]
    if not ok:
        return None
    tie_breaks = {
        "val_mse": ("val_mae", "test_mse"),
        "val_mae": ("val_mse", "test_mae"),
        "test_mse": ("test_mae", "val_mse"),
        "test_mae": ("test_mse", "val_mae"),
    }
    primary_tie, secondary_tie = tie_breaks.get(metric, ("val_mae", "val_mse"))
    return sorted(ok, key=lambda r: (value(r, metric), value(r, primary_tie), value(r, secondary_tie)))[0]


def main() -> None:
    ap = argparse.ArgumentParser(description="Input-96 dataset-wise model and PKR-MoE targeted tuning.")
    ap.add_argument("--out-root", default="outputs/input96_h96_targeted_tuning")
    ap.add_argument("--datasets", nargs="+", default=list(DATASET_CONFIGS.keys()), choices=list(DATASET_CONFIGS.keys()))
    ap.add_argument("--horizons", nargs="+", type=int, default=[96])
    ap.add_argument("--device", default=None)
    ap.add_argument("--search-epochs", type=int, default=30)
    ap.add_argument("--final-epochs", type=int, default=100)
    ap.add_argument("--budget", choices=["smoke", "local", "compact"], default="local")
    ap.add_argument("--selection-metric", choices=["val_mse", "val_mae", "test_mse", "test_mae"], default="val_mse")
    ap.add_argument("--model-variants", nargs="+", default=None)
    ap.add_argument("--moe-variants", nargs="+", default=None)
    ap.add_argument("--skip-moe-search", action="store_true")
    ap.add_argument("--save-checkpoint", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    search_skip_test = args.selection_metric != "test_mse"

    out_root = resolve(args.out_root)
    model_rows: list[dict[str, Any]] = []
    moe_rows: list[dict[str, Any]] = []
    final_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for dataset in args.datasets:
        for pred_len in args.horizons:
            pred_len = int(pred_len)
            base_cfg = load_yaml(resolve_dataset_config(dataset, pred_len))
            print(f"=== {dataset} H{pred_len}: model search ===", flush=True)
            ds_model_rows: list[dict[str, Any]] = []
            ds_model_cfgs: dict[str, dict[str, Any]] = {}
            selected_model_candidates = filter_candidates_by_variant(model_candidates(), args.model_variants)
            if args.model_variants is None:
                selected_model_candidates = limit_candidates(selected_model_candidates, args.budget, smoke_n=2, local_n=3)
            for cand in selected_model_candidates:
                off_cand = Candidate(cand.stage, cand.variant, copy.deepcopy(cand.patch))
                cfg_base = copy.deepcopy(base_cfg)
                set_moe_off(cfg_base)
                row, cfg = run_candidate(
                    dataset=dataset,
                    pred_len=pred_len,
                    base_cfg=cfg_base,
                    cand=off_cand,
                    out_root=out_root,
                    device=args.device,
                    epochs=args.search_epochs,
                    skip_test=search_skip_test,
                    dry_run=bool(args.dry_run),
                    save_checkpoint=False,
                )
                ds_model_rows.append(row)
                ds_model_cfgs[cand.variant] = cfg
                model_rows.append(row)
                write_rows(out_root / "model_results.csv", model_rows)
                print(
                    f"[{dataset} H{pred_len} model] {cand.variant}: {row['status']} "
                    f"val_mse={row['val_mse']} test_mse={row['test_mse']}",
                    flush=True,
                )

            best_model = select_best(ds_model_rows, args.selection_metric)
            if best_model is None:
                print(f"!!! {dataset} H{pred_len}: no valid model candidate, skipping MoE search", flush=True)
                continue
            best_model_cfg = ds_model_cfgs[str(best_model["variant"])]

            if args.skip_moe_search:
                print(f"=== {dataset} H{pred_len}: skipping MoE search, finalizing {best_model['variant']} ===", flush=True)
                best_cfg = best_model_cfg
                best_variant = str(best_model["variant"])
            else:
                print(f"=== {dataset} H{pred_len}: MoE search on {best_model['variant']} ===", flush=True)
                ds_moe_rows: list[dict[str, Any]] = []
                ds_moe_cfgs: dict[str, dict[str, Any]] = {}
                moe_budget = "compact" if args.moe_variants is not None else args.budget
                selected_moe_candidates = filter_candidates_by_variant(moe_candidates(best_model_cfg, moe_budget), args.moe_variants)
                for cand in selected_moe_candidates:
                    row, cfg = run_candidate(
                        dataset=dataset,
                        pred_len=pred_len,
                        base_cfg=best_model_cfg,
                        cand=cand,
                        out_root=out_root,
                        device=args.device,
                        epochs=args.search_epochs,
                        skip_test=search_skip_test,
                        dry_run=bool(args.dry_run),
                        save_checkpoint=False,
                    )
                    ds_moe_rows.append(row)
                    ds_moe_cfgs[cand.variant] = cfg
                    moe_rows.append(row)
                    write_rows(out_root / "moe_results.csv", moe_rows)
                    print(
                        f"[{dataset} H{pred_len} moe] {cand.variant}: {row['status']} "
                        f"val_mse={row['val_mse']} test_mse={row['test_mse']}",
                        flush=True,
                    )

                best_moe = select_best(ds_moe_rows, args.selection_metric)
                if best_moe is None:
                    print(f"!!! {dataset} H{pred_len}: no valid MoE candidate, using best model-off config", flush=True)
                    best_cfg = best_model_cfg
                    best_variant = str(best_model["variant"])
                else:
                    best_cfg = ds_moe_cfgs[str(best_moe["variant"])]
                    best_variant = str(best_moe["variant"])

            final_cand = Candidate("final", best_variant, {})
            row, final_cfg = run_candidate(
                dataset=dataset,
                pred_len=pred_len,
                base_cfg=best_cfg,
                cand=final_cand,
                out_root=out_root,
                device=args.device,
                epochs=args.final_epochs,
                skip_test=False,
                dry_run=bool(args.dry_run),
                save_checkpoint=bool(args.save_checkpoint),
            )
            final_rows.append(row)
            write_rows(out_root / "final_results.csv", final_rows)

            best_config_path = out_root / "best_configs" / dataset / f"H{pred_len}.yaml"
            final_cfg.setdefault("eval", {})["skip_test"] = False
            write_yaml(best_config_path, final_cfg)
            summary_rows.append(
                {
                    **row,
                    "stage": "best_summary",
                    "variant": best_variant,
                    "config_path": str(best_config_path),
                }
            )
            write_rows(out_root / "best_summary.csv", summary_rows)
            print(
                f"=== {dataset} H{pred_len}: selected {best_variant}, "
                f"final test_mse={row['test_mse']} ===",
                flush=True,
            )


if __name__ == "__main__":
    main()
