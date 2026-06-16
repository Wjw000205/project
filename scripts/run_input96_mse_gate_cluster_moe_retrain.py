from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY = ROOT / "outputs" / "codex_table_target_20260614" / "input96_global_paired_backbone_moe_summary.csv"

FIELDS = [
    "status",
    "dataset",
    "horizon",
    "variant",
    "base_moe_mse",
    "base_moe_mae",
    "test_mse",
    "test_mae",
    "val_mse",
    "val_mae",
    "mse_gain_vs_current_pct",
    "mae_gain_vs_current_pct",
    "beats_current_mse",
    "beats_current_mae",
    "penalties",
    "gate_topk",
    "cluster_prior_topk",
    "cluster_prior_logit_strength",
    "mse_gate_weight",
    "mse_gate_temperature",
    "pred_residual_enabled",
    "residual_mean_scale",
    "residual_num_channels",
    "best_epoch",
    "total_sec",
    "base_config",
    "config_path",
    "out_dir",
    "returncode",
    "error",
]


@dataclass(frozen=True)
class Variant:
    name: str
    gate_topk: int = 2
    cluster_prior_enable: bool = True
    cluster_prior_topk: int = 0
    cluster_prior_hard_topk: bool = True
    cluster_prior_logit_strength: float = 0.3
    mse_gate_weight: float = 0.02
    mse_gate_temperature: float = 0.5
    residual_init_alpha: float | None = None
    residual_alpha_scale: float | None = None
    residual_corrector_hidden: int | None = None
    residual_feature_mode: str | None = None
    residual_specialization_weight: float | None = None
    residual_norm_weight: float | None = None
    residual_max_scale: float | None = None
    residual_init_scale: float | None = None
    residual_selection_policy: str = "val_mse_gate_guarded"
    residual_selection_max_abs_mse_regression: float | None = None
    residual_selection_max_rel_mse_regression: float | None = None
    residual_clip: float | None = None
    gate_calibrator_epochs_min: int | None = None
    gate_apply_activation_threshold: bool | None = None
    gate_calibrator_loss: str = "mse"
    gate_calibrator_selection_metric: str = "mse"
    channel_prior_enable: bool = False
    channel_prior_topk: int = 0
    force_mse_selection: bool = True
    train_lr_override: float | None = None
    train_epochs_min: int | None = None
    train_mse_weight_override: float | None = None
    mae_objective_weight_override: float | None = None
    penalty_warmup_override: int | None = None
    model_selection_start_epoch_override: int | None = None
    lr_scheduler_name_override: str | None = None
    penalties_enabled: tuple[str, ...] | None = None
    cluster_method: str | None = None
    cluster_n_clusters: int | None = None
    cluster_distance_threshold: float | None = None
    cluster_min_cluster_size: int | None = None
    cluster_merge_small_clusters: bool | None = None
    cluster_no_merge_if_channels_lt: int | None = None
    cluster_spectral_affinity: str | None = None


VARIANTS = {
    "mse_gate_w002_softprior": Variant(
        name="mse_gate_w002_softprior",
        cluster_prior_topk=0,
        cluster_prior_logit_strength=0.3,
        mse_gate_weight=0.02,
    ),
    "mse_gate_w002_top2": Variant(
        name="mse_gate_w002_top2",
        cluster_prior_topk=2,
        cluster_prior_logit_strength=0.4,
        mse_gate_weight=0.02,
    ),
    "mse_gate_w005_softprior": Variant(
        name="mse_gate_w005_softprior",
        cluster_prior_topk=0,
        cluster_prior_logit_strength=0.3,
        mse_gate_weight=0.05,
    ),
    "mse_gate_w002_ch2": Variant(
        name="mse_gate_w002_ch2",
        cluster_prior_topk=0,
        cluster_prior_logit_strength=0.3,
        mse_gate_weight=0.02,
        channel_prior_enable=True,
        channel_prior_topk=2,
    ),
    "mse_gate_w002_trainable_guarded": Variant(
        name="mse_gate_w002_trainable_guarded",
        cluster_prior_topk=0,
        cluster_prior_logit_strength=0.3,
        mse_gate_weight=0.02,
        residual_selection_policy="val_mse_gate_guarded",
        residual_max_scale=1.25,
        residual_init_scale=0.5,
        residual_clip=4.0,
        gate_calibrator_epochs_min=30,
        train_lr_override=1.0e-3,
        train_epochs_min=3,
    ),
    "mse_gate_w002_trainable_open": Variant(
        name="mse_gate_w002_trainable_open",
        cluster_prior_topk=0,
        cluster_prior_logit_strength=0.3,
        mse_gate_weight=0.02,
        residual_selection_policy="val_mse_gate",
        residual_max_scale=1.5,
        residual_init_scale=0.5,
        residual_clip=4.0,
        gate_calibrator_epochs_min=30,
        gate_apply_activation_threshold=False,
        train_lr_override=1.0e-3,
        train_epochs_min=3,
    ),
    "mse_gate_w002_trainable_utility": Variant(
        name="mse_gate_w002_trainable_utility",
        cluster_prior_topk=0,
        cluster_prior_logit_strength=0.3,
        mse_gate_weight=0.02,
        residual_selection_policy="val_mse_gate_guarded",
        residual_max_scale=1.25,
        residual_init_scale=0.5,
        residual_clip=4.0,
        gate_calibrator_epochs_min=30,
        train_lr_override=1.0e-3,
        train_epochs_min=3,
    ),
    "mse_gate_w002_strong_safe_mse": Variant(
        name="mse_gate_w002_strong_safe_mse",
        cluster_prior_topk=0,
        cluster_prior_logit_strength=0.3,
        mse_gate_weight=0.02,
        residual_selection_policy="val_mse_gate_guarded",
        residual_corrector_hidden=64,
        residual_feature_mode="safe_augmented",
        residual_init_alpha=-1.8,
        residual_alpha_scale=1.5,
        residual_max_scale=1.5,
        residual_init_scale=0.6,
        residual_clip=5.0,
        residual_specialization_weight=0.03,
        residual_norm_weight=0.0,
        gate_calibrator_epochs_min=40,
        train_lr_override=1.0e-3,
        train_epochs_min=6,
        train_mse_weight_override=0.9,
        mae_objective_weight_override=0.3,
    ),
    "mse_gate_w002_strong_legacy_mse": Variant(
        name="mse_gate_w002_strong_legacy_mse",
        cluster_prior_topk=0,
        cluster_prior_logit_strength=0.3,
        mse_gate_weight=0.02,
        residual_selection_policy="val_mse_gate_guarded",
        residual_corrector_hidden=64,
        residual_feature_mode="legacy",
        residual_init_alpha=-1.8,
        residual_alpha_scale=1.5,
        residual_max_scale=1.5,
        residual_init_scale=0.6,
        residual_clip=5.0,
        residual_specialization_weight=0.03,
        residual_norm_weight=0.0,
        gate_calibrator_epochs_min=40,
        train_lr_override=1.0e-3,
        train_epochs_min=6,
        train_mse_weight_override=0.9,
        mae_objective_weight_override=0.3,
    ),
    "mse_gate_w002_short_mse_legacy": Variant(
        name="mse_gate_w002_short_mse_legacy",
        cluster_prior_topk=0,
        cluster_prior_logit_strength=0.3,
        mse_gate_weight=0.02,
        residual_selection_policy="val_mse_gate_guarded",
        residual_corrector_hidden=64,
        residual_feature_mode="legacy",
        residual_init_alpha=-2.0,
        residual_alpha_scale=1.2,
        residual_max_scale=1.25,
        residual_init_scale=0.5,
        residual_clip=4.0,
        residual_specialization_weight=0.05,
        residual_norm_weight=0.0,
        gate_calibrator_epochs_min=30,
        train_lr_override=6.0e-4,
        train_epochs_min=6,
        train_mse_weight_override=0.95,
        mae_objective_weight_override=0.2,
        penalty_warmup_override=1,
        model_selection_start_epoch_override=1,
        lr_scheduler_name_override="none",
    ),
    "pems_kmeans4_shape_pool": Variant(
        name="pems_kmeans4_shape_pool",
        cluster_prior_topk=0,
        cluster_prior_logit_strength=0.3,
        mse_gate_weight=0.02,
        residual_corrector_hidden=64,
        residual_feature_mode="safe_augmented",
        residual_init_alpha=-1.8,
        residual_alpha_scale=1.5,
        residual_max_scale=1.5,
        residual_init_scale=0.6,
        residual_clip=5.0,
        residual_specialization_weight=0.03,
        residual_norm_weight=0.0,
        gate_calibrator_epochs_min=40,
        train_lr_override=1.0e-3,
        train_epochs_min=6,
        train_mse_weight_override=0.9,
        mae_objective_weight_override=0.3,
        penalties_enabled=("level", "delta", "d2_match", "corr", "range", "trend", "seasonal_align"),
        cluster_method="kmeans",
        cluster_n_clusters=4,
        cluster_distance_threshold=None,
        cluster_min_cluster_size=2,
        cluster_merge_small_clusters=True,
    ),
    "pems_spectral6_shape_pool": Variant(
        name="pems_spectral6_shape_pool",
        cluster_prior_topk=0,
        cluster_prior_logit_strength=0.3,
        mse_gate_weight=0.02,
        residual_corrector_hidden=64,
        residual_feature_mode="safe_augmented",
        residual_init_alpha=-1.8,
        residual_alpha_scale=1.5,
        residual_max_scale=1.5,
        residual_init_scale=0.6,
        residual_clip=5.0,
        residual_specialization_weight=0.03,
        residual_norm_weight=0.0,
        gate_calibrator_epochs_min=40,
        train_lr_override=1.0e-3,
        train_epochs_min=6,
        train_mse_weight_override=0.9,
        mae_objective_weight_override=0.3,
        penalties_enabled=("level", "delta", "d2_match", "corr", "range", "trend", "seasonal_align"),
        cluster_method="spectral",
        cluster_n_clusters=6,
        cluster_distance_threshold=None,
        cluster_min_cluster_size=2,
        cluster_merge_small_clusters=True,
        cluster_spectral_affinity="corr",
    ),
    "pems_leader02_shape_pool": Variant(
        name="pems_leader02_shape_pool",
        cluster_prior_topk=0,
        cluster_prior_logit_strength=0.3,
        mse_gate_weight=0.02,
        residual_corrector_hidden=64,
        residual_feature_mode="safe_augmented",
        residual_init_alpha=-1.8,
        residual_alpha_scale=1.5,
        residual_max_scale=1.5,
        residual_init_scale=0.6,
        residual_clip=5.0,
        residual_specialization_weight=0.03,
        residual_norm_weight=0.0,
        gate_calibrator_epochs_min=40,
        train_lr_override=1.0e-3,
        train_epochs_min=6,
        train_mse_weight_override=0.9,
        mae_objective_weight_override=0.3,
        penalties_enabled=("level", "delta", "d2_match", "corr", "range", "trend", "seasonal_align"),
        cluster_method="leader",
        cluster_n_clusters=3,
        cluster_distance_threshold=0.2,
        cluster_min_cluster_size=2,
        cluster_merge_small_clusters=True,
    ),
    "pems_residual_profile_pool": Variant(
        name="pems_residual_profile_pool",
        cluster_prior_topk=0,
        cluster_prior_logit_strength=0.3,
        mse_gate_weight=0.02,
        residual_corrector_hidden=64,
        residual_feature_mode="safe_augmented",
        residual_init_alpha=-1.8,
        residual_alpha_scale=1.5,
        residual_max_scale=1.5,
        residual_init_scale=0.6,
        residual_clip=5.0,
        residual_specialization_weight=0.03,
        residual_norm_weight=0.0,
        gate_calibrator_epochs_min=40,
        train_lr_override=1.0e-3,
        train_epochs_min=6,
        train_mse_weight_override=0.9,
        mae_objective_weight_override=0.3,
    ),
    "pems_residual_profile_pool_l1_light": Variant(
        name="pems_residual_profile_pool_l1_light",
        cluster_prior_topk=0,
        cluster_prior_logit_strength=0.3,
        mse_gate_weight=0.02,
        residual_corrector_hidden=64,
        residual_feature_mode="safe_augmented",
        residual_init_alpha=-1.8,
        residual_alpha_scale=1.5,
        residual_max_scale=1.5,
        residual_init_scale=0.6,
        residual_clip=5.0,
        residual_specialization_weight=0.03,
        residual_norm_weight=0.0,
        gate_calibrator_epochs_min=40,
        train_lr_override=1.0e-3,
        train_epochs_min=6,
        train_mse_weight_override=0.95,
        mae_objective_weight_override=0.15,
    ),
    "pems_residual_profile_pool_l1_mid": Variant(
        name="pems_residual_profile_pool_l1_mid",
        cluster_prior_topk=0,
        cluster_prior_logit_strength=0.3,
        mse_gate_weight=0.02,
        residual_corrector_hidden=64,
        residual_feature_mode="safe_augmented",
        residual_init_alpha=-1.8,
        residual_alpha_scale=1.5,
        residual_max_scale=1.5,
        residual_init_scale=0.6,
        residual_clip=5.0,
        residual_specialization_weight=0.03,
        residual_norm_weight=0.0,
        gate_calibrator_epochs_min=40,
        train_lr_override=1.0e-3,
        train_epochs_min=6,
        train_mse_weight_override=0.9,
        mae_objective_weight_override=0.5,
    ),
    "pems_residual_profile_pool_l1_strong": Variant(
        name="pems_residual_profile_pool_l1_strong",
        cluster_prior_topk=0,
        cluster_prior_logit_strength=0.3,
        mse_gate_weight=0.02,
        residual_corrector_hidden=64,
        residual_feature_mode="safe_augmented",
        residual_init_alpha=-1.8,
        residual_alpha_scale=1.5,
        residual_max_scale=1.5,
        residual_init_scale=0.6,
        residual_clip=5.0,
        residual_specialization_weight=0.03,
        residual_norm_weight=0.0,
        gate_calibrator_epochs_min=40,
        train_lr_override=1.0e-3,
        train_epochs_min=6,
        train_mse_weight_override=0.85,
        mae_objective_weight_override=0.8,
    ),
    "pems_residual_profile_pool_l1_mae_guarded": Variant(
        name="pems_residual_profile_pool_l1_mae_guarded",
        cluster_prior_topk=0,
        cluster_prior_logit_strength=0.3,
        mse_gate_weight=0.02,
        residual_selection_policy="val_mae_gate_guarded",
        residual_selection_max_rel_mse_regression=0.005,
        residual_corrector_hidden=64,
        residual_feature_mode="safe_augmented",
        residual_init_alpha=-1.8,
        residual_alpha_scale=1.5,
        residual_max_scale=1.5,
        residual_init_scale=0.6,
        residual_clip=5.0,
        residual_specialization_weight=0.03,
        residual_norm_weight=0.0,
        gate_calibrator_epochs_min=40,
        gate_calibrator_loss="mae",
        gate_calibrator_selection_metric="mae",
        train_lr_override=1.0e-3,
        train_epochs_min=6,
        train_mse_weight_override=0.9,
        mae_objective_weight_override=0.5,
    ),
}


def resolve(path_text: str | Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return (ROOT / path).resolve()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def write_yaml(path: Path, cfg: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=False, sort_keys=False)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


def write_rows(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = fields or FIELDS
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def as_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value in {"", None}:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def localize_paths(cfg: dict[str, Any], out_dir: Path) -> None:
    cfg.setdefault("exp", {})["out_dir"] = str(out_dir)
    cfg.setdefault("corr", {})["save_path"] = str(out_dir / "corr.npy")
    cfg.setdefault("portrait", {})["enable"] = False
    cfg["portrait"]["out_dir"] = str(out_dir / "cluster_portraits")
    cfg.setdefault("knn_hybrid", {})["enable"] = False
    cfg["knn_hybrid"]["use_for_model_selection"] = False
    cfg["knn_hybrid"]["path"] = str(out_dir / "knn_shape_bank.pt")
    cfg.setdefault("memory", {})["enable"] = False
    cfg["memory"]["save_checkpoint"] = False
    cfg["memory"]["path"] = str(out_dir / "cluster_memory.pt")
    cfg["memory"]["checkpoint_path"] = str(out_dir / "best_checkpoint.pt")
    cfg.setdefault("plot", {})["enable"] = False


def finetune_enabled(cfg: dict[str, Any]) -> bool:
    ft_cfg = cfg.get("finetune", {}) or {}
    return bool(ft_cfg.get("enable", False)) and bool(str(ft_cfg.get("checkpoint_path", "")).strip())


def checkpoint_from_memory(cfg: dict[str, Any]) -> Path | None:
    checkpoint = str((cfg.get("memory", {}) or {}).get("checkpoint_path", "")).strip()
    if not checkpoint:
        return None
    path = resolve(checkpoint)
    return path if path.exists() else None


def checkpoint_from_config_path(config_path_text: str) -> Path | None:
    if not str(config_path_text or "").strip():
        return None
    config_path = resolve(str(config_path_text))
    if not config_path.exists():
        return None
    try:
        return checkpoint_from_memory(load_yaml(config_path))
    except Exception:
        return None


def checkpoint_from_backbone_variant(row: dict[str, str]) -> Path | None:
    dataset = str(row.get("dataset", "")).strip()
    horizon = str(row.get("horizon", "")).strip()
    variant = str(row.get("backbone_variant", "")).strip()
    if not dataset or not horizon or not variant:
        return None
    search_root = ROOT / "outputs" / "codex_table_target_20260614"
    pattern = f"**/runs/{dataset}/H{horizon}/final/{variant}/best_checkpoint.pt"
    matches = sorted(search_root.glob(pattern))
    return matches[0] if matches else None


def checkpoint_from_model_hint(base_cfg: dict[str, Any], row: dict[str, str]) -> Path | None:
    dataset = str(row.get("dataset", "")).strip()
    horizon = str(row.get("horizon", "")).strip()
    model_cfg = base_cfg.get("model", {}) or {}
    hidden_dim = model_cfg.get("hidden_dim")
    if not dataset or not horizon or hidden_dim is None:
        return None
    try:
        hidden_dim_int = int(hidden_dim)
    except (TypeError, ValueError):
        return None
    dataset_lower = dataset.lower()
    candidates = [
        ROOT
        / "outputs"
        / f"codex_table_target_20260615_{dataset_lower}_h{horizon}_h{hidden_dim_int}_strict_backbone_moe"
        / "runs"
        / f"cch_h{hidden_dim_int}_backbone"
        / "best_checkpoint.pt",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def model_hidden_dim(base_cfg: dict[str, Any]) -> int | None:
    hidden_dim = (base_cfg.get("model", {}) or {}).get("hidden_dim")
    try:
        return int(hidden_dim)
    except (TypeError, ValueError):
        return None


def checkpoint_matches_model_hint(base_cfg: dict[str, Any], checkpoint: Path) -> bool:
    hidden_dim = model_hidden_dim(base_cfg)
    if hidden_dim is None:
        return True
    checkpoint_text = checkpoint.as_posix().lower()
    for candidate_dim in (64, 128, 160, 192, 256, 320, 384, 512):
        token = f"h{candidate_dim}"
        if token in checkpoint_text and candidate_dim != hidden_dim:
            return False
    return True


def compatible_checkpoint(base_cfg: dict[str, Any], checkpoint: Path | None) -> Path | None:
    if checkpoint is None:
        return None
    return checkpoint if checkpoint_matches_model_hint(base_cfg, checkpoint) else None


def infer_warm_start_checkpoint(base_cfg: dict[str, Any], row: dict[str, str]) -> Path | None:
    # Prefer the exact checkpoint used by this base config; this keeps predictor
    # dimensions aligned when a current best MoE was trained without finetune.
    checkpoint = compatible_checkpoint(base_cfg, checkpoint_from_memory(base_cfg))
    if checkpoint is not None:
        return checkpoint
    checkpoint = compatible_checkpoint(base_cfg, checkpoint_from_model_hint(base_cfg, row))
    if checkpoint is not None:
        return checkpoint
    checkpoint = compatible_checkpoint(base_cfg, checkpoint_from_config_path(str(row.get("backbone_config", ""))))
    if checkpoint is not None:
        return checkpoint
    return compatible_checkpoint(base_cfg, checkpoint_from_backbone_variant(row))


def configure_finetune_warm_start(cfg: dict[str, Any], checkpoint: Path) -> None:
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


def patch_gate_calibrator(cal: dict[str, Any], variant: Variant) -> dict[str, Any]:
    cfg = copy.deepcopy(cal or {})
    cfg["loss"] = str(variant.gate_calibrator_loss)
    cfg["selection_metric"] = str(variant.gate_calibrator_selection_metric)
    cfg.setdefault("source_split", "val")
    cfg.setdefault("epochs", 30)
    cfg.setdefault("train_fraction", 0.7)
    cfg.setdefault("hidden_dim", 32)
    cfg.setdefault("batch_size", 256)
    cfg.setdefault("scale_reg", 5.0e-4)
    cfg.setdefault("scale_mode", "sigmoid")
    cfg.setdefault("standardize_features", True)
    cfg.setdefault("activation_head_enable", True)
    cfg.setdefault("apply_activation_threshold", True)
    cfg.setdefault("activation_threshold", "auto")
    cfg.setdefault("activation_threshold_selection_metric", "mse")
    cfg.setdefault("activation_threshold_scope", "channel")
    cfg.setdefault("activation_bce_weight", 0.2)
    cfg.setdefault("activation_inactive_scale_weight", 0.05)
    cfg.setdefault("activation_pos_weight", "auto")
    cfg.setdefault("activation_pos_weight_scope", "channel")
    cfg.setdefault("activation_train_soft_gating", False)
    if variant.gate_calibrator_epochs_min is not None:
        cfg["epochs"] = max(int(cfg.get("epochs", 0) or 0), int(variant.gate_calibrator_epochs_min))
    if variant.gate_apply_activation_threshold is not None:
        cfg["apply_activation_threshold"] = bool(variant.gate_apply_activation_threshold)
    if variant.residual_max_scale is not None:
        cfg["max_scale"] = float(variant.residual_max_scale)
    else:
        cfg.setdefault("max_scale", 1.0)
    if variant.residual_init_scale is not None:
        cfg["init_scale"] = float(variant.residual_init_scale)
    else:
        cfg.setdefault("init_scale", 0.3)
    return cfg


def patch_pred_side_residual(moe: dict[str, Any], variant: Variant) -> None:
    residual = copy.deepcopy(moe.get("pred_side_residual", {}) or {})
    residual["enable"] = True
    if variant.residual_feature_mode is not None:
        residual["feature_mode"] = str(variant.residual_feature_mode)
    else:
        residual.setdefault("feature_mode", "legacy")
    residual["selection_policy"] = variant.residual_selection_policy
    residual.setdefault("selection_min_abs_improvement", 0.0)
    residual.setdefault("selection_min_rel_improvement", 0.0)
    if variant.residual_selection_max_abs_mse_regression is not None:
        residual["selection_max_abs_mse_regression"] = float(variant.residual_selection_max_abs_mse_regression)
    if variant.residual_selection_max_rel_mse_regression is not None:
        residual["selection_max_rel_mse_regression"] = float(variant.residual_selection_max_rel_mse_regression)
    if variant.residual_clip is not None:
        residual["residual_clip"] = float(variant.residual_clip)
    else:
        residual.setdefault("residual_clip", 4.0)
    if variant.residual_corrector_hidden is not None:
        residual["corrector_hidden"] = int(variant.residual_corrector_hidden)
    else:
        residual.setdefault("corrector_hidden", 32)
    if variant.residual_specialization_weight is not None:
        residual["specialization_weight"] = float(variant.residual_specialization_weight)
    else:
        residual.setdefault("specialization_weight", 0.1)
    if variant.residual_norm_weight is not None:
        residual["norm_weight"] = float(variant.residual_norm_weight)
    else:
        residual.setdefault("norm_weight", 1.0e-5)
    residual.setdefault("use_y_base_input", True)
    if variant.residual_init_alpha is not None:
        residual["init_alpha"] = float(variant.residual_init_alpha)
    else:
        residual.setdefault("init_alpha", -2.5)
    if variant.residual_alpha_scale is not None:
        residual["alpha_scale"] = float(variant.residual_alpha_scale)
    else:
        residual.setdefault("alpha_scale", 1.0)
    residual["gate_calibrator"] = patch_gate_calibrator(residual.get("gate_calibrator", {}) or {}, variant)
    moe["pred_side_residual"] = residual


def patch_config(
    base_cfg: dict[str, Any],
    row: dict[str, str],
    variant: Variant,
    out_dir: Path,
    device: str,
    epochs_override: int | None,
    batch_size_override: int | None,
    skip_test: bool,
    enable_explainability: bool,
    explainability_splits: list[str],
    explainability_max_batches: int,
    allowed_by_cluster: tuple[tuple[str, ...], ...] | None,
    allowed_logit_strength: float | None,
    residual_profile: dict[str, Any] | None,
) -> dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    dataset = row["dataset"]
    horizon = int(row["horizon"])
    warm_start_checkpoint = None if finetune_enabled(base_cfg) else infer_warm_start_checkpoint(base_cfg, row)
    cfg.setdefault("exp", {})["name"] = f"{dataset}_input96_H{horizon}_{variant.name}"
    cfg["exp"]["device"] = device
    localize_paths(cfg, out_dir)
    if not finetune_enabled(cfg) and warm_start_checkpoint is not None:
        configure_finetune_warm_start(cfg, warm_start_checkpoint)
    cfg.setdefault("window", {})["input_len"] = 96
    cfg["window"]["pred_len"] = horizon
    cfg["window"].setdefault("past_context", True)
    cfg.setdefault("normalize", {})["train_only"] = True
    cluster_cfg = cfg.setdefault("cluster", {})
    cluster_cfg["train_only"] = True
    if variant.cluster_method is not None:
        cluster_cfg["method"] = str(variant.cluster_method)
    if variant.cluster_n_clusters is not None:
        cluster_cfg["n_clusters"] = int(variant.cluster_n_clusters)
    if variant.cluster_distance_threshold is not None:
        cluster_cfg["distance_threshold"] = float(variant.cluster_distance_threshold)
    elif variant.cluster_method is not None and str(variant.cluster_method).lower() not in {"leader", "greedy_leader"}:
        cluster_cfg["distance_threshold"] = None
    if variant.cluster_min_cluster_size is not None:
        cluster_cfg["min_cluster_size"] = int(variant.cluster_min_cluster_size)
    if variant.cluster_merge_small_clusters is not None:
        cluster_cfg["merge_small_clusters"] = bool(variant.cluster_merge_small_clusters)
    if variant.cluster_no_merge_if_channels_lt is not None:
        cluster_cfg["no_merge_if_channels_lt"] = int(variant.cluster_no_merge_if_channels_lt)
    if variant.cluster_spectral_affinity is not None:
        cluster_cfg["spectral_affinity"] = str(variant.cluster_spectral_affinity)
    profile_penalties: list[str] | None = None
    profile_allowed_by_cluster: tuple[tuple[str, ...], ...] | None = None
    if residual_profile is not None:
        fixed_cluster = [int(v) for v in residual_profile["fixed_cluster_id"]]
        cluster_cfg["fixed_cluster_id"] = fixed_cluster
        cluster_cfg["method"] = "fixed"
        profile_penalties = [str(v) for v in residual_profile["penalties_enabled"]]
        profile_allowed_by_cluster = tuple(tuple(str(name) for name in names) for names in residual_profile["allowed_by_cluster"])
    cfg.setdefault("eval", {})["skip_test"] = bool(skip_test)
    cfg.setdefault("calendar_residual", {})["enable"] = False

    train = cfg.setdefault("train", {})
    if epochs_override is not None:
        train["epochs"] = int(epochs_override)
    else:
        train.setdefault("epochs", 1)
        if variant.train_epochs_min is not None:
            train["epochs"] = max(int(train.get("epochs", 0) or 0), int(variant.train_epochs_min))
    if batch_size_override is not None:
        train["batch_size"] = int(batch_size_override)
    if variant.train_lr_override is not None:
        train["lr"] = float(variant.train_lr_override)
    if variant.train_mse_weight_override is not None:
        train["mse_weight"] = float(variant.train_mse_weight_override)
    if variant.mae_objective_weight_override is not None:
        train.setdefault("mae_objective", {})["weight"] = float(variant.mae_objective_weight_override)
    if variant.penalty_warmup_override is not None:
        train["penalty_warmup_epochs"] = int(variant.penalty_warmup_override)
    if variant.model_selection_start_epoch_override is not None:
        train["model_selection_start_epoch"] = int(variant.model_selection_start_epoch_override)
    if variant.lr_scheduler_name_override is not None:
        train["lr_scheduler"] = {"name": str(variant.lr_scheduler_name_override)}
    if variant.force_mse_selection:
        train["selection_metric"] = "val_mse"

    penalties = (
        list(profile_penalties)
        if profile_penalties is not None
        else list(variant.penalties_enabled)
        if variant.penalties_enabled is not None
        else list((cfg.get("penalties", {}) or {}).get("enabled", []) or [])
    )
    cfg.setdefault("penalties", {})["enabled"] = penalties

    moe = cfg.setdefault("moe", {})
    moe["enable"] = True
    moe["freeze_backbone"] = True if finetune_enabled(cfg) else False
    moe["topk"] = int(variant.gate_topk)
    moe.setdefault("select_ranks", [1])
    moe["dynamic_lambda"] = {"enable": False}
    moe["learnable_lambda"] = {"enable": False}
    if profile_penalties is not None or variant.penalties_enabled is not None:
        moe["lambda_init"] = {name: 0.0 for name in penalties}
        moe["lambda_min"] = {name: 0.0 for name in penalties}
        moe["lambda_schedule"] = {name: "none" for name in penalties}
    moe["gate_entropy_weight"] = 0.0
    moe.setdefault("gate_balance_weight", 0.0)
    moe["gate_penalty_hit"] = {"enable": False}
    moe["mse_utility_gate_supervision"] = {
        "enable": True,
        "weight": float(variant.mse_gate_weight),
        "temperature": float(variant.mse_gate_temperature),
        "min_gain": 0.0,
        "target_power": 1.0,
    }
    moe["gate_route_on_penalty_only"] = False
    moe["router_mode"] = "learned"
    moe["router_penalty_context_weight"] = 0.0
    moe["router_penalty_context_score"] = "high_violation"
    moe["router_detach_penalty_context"] = True
    moe["allow_skip"] = True
    moe.setdefault("skip_cost", 0.15)
    moe.setdefault("skip_init_bias", -2.0)
    moe["cluster_penalty_prior"] = {
        "enable": bool(variant.cluster_prior_enable),
        "topk": int(variant.cluster_prior_topk),
        "hard_topk": bool(variant.cluster_prior_hard_topk),
        "logit_strength": float(variant.cluster_prior_logit_strength),
        "temperature": 1.0,
        "smoothing": 0.02,
        "use_normalized_penalty": True,
        "use_as_balance_target": False,
    }
    if profile_allowed_by_cluster is not None:
        allowed_by_cluster = profile_allowed_by_cluster
    if allowed_by_cluster is not None:
        moe["cluster_penalty_prior"]["enable"] = True
        moe["cluster_penalty_prior"]["topk"] = 0
        moe["cluster_penalty_prior"]["hard_topk"] = True
        if allowed_logit_strength is not None:
            moe["cluster_penalty_prior"]["logit_strength"] = float(allowed_logit_strength)
        moe["cluster_penalty_prior"]["allowed_by_cluster"] = [list(names) for names in allowed_by_cluster]
    moe["channel_penalty_prior"] = {
        "enable": bool(variant.channel_prior_enable),
        "topk": int(variant.channel_prior_topk),
        "hard_topk": True,
        "temperature": 1.0,
        "smoothing": 0.02,
        "use_normalized_penalty": True,
    }
    moe["explainability"] = {
        "enable": bool(enable_explainability),
        "splits": [str(x) for x in explainability_splits],
        "max_batches": int(explainability_max_batches),
    }
    patch_pred_side_residual(moe, variant)
    knn_hybrid = cfg.setdefault("knn_hybrid", {})
    knn_hybrid["selection_policy"] = "val_mse_margin"
    knn_hybrid["channel_selection_policy"] = "none"
    knn_hybrid["use_for_model_selection"] = False
    return cfg


def build_mse_utility_allowed_by_cluster(
    source_json: Path,
    *,
    split: str,
    topk: int,
    min_gain: float,
    fallback: str,
    penalty_names: list[str],
) -> tuple[tuple[str, ...], ...]:
    payload = read_json(source_json)
    split_payload = (payload.get("splits", {}) or {}).get(str(split).lower())
    if not isinstance(split_payload, dict):
        raise ValueError(f"Split {split!r} not found in MSE utility source: {source_json}")
    rows = split_payload.get("rows", [])
    if not isinstance(rows, list) or len(rows) == 0:
        raise ValueError(f"No explainability rows found for split {split!r} in {source_json}")

    penalty_set = set(penalty_names)
    by_cluster: dict[int, list[tuple[float, str]]] = {}
    for item in rows:
        if not isinstance(item, dict):
            continue
        penalty = str(item.get("penalty", "")).strip()
        if penalty not in penalty_set:
            continue
        try:
            cluster_id = int(item.get("cluster_id"))
            gain = float(item.get("mean_single_penalty_gain_mse", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        by_cluster.setdefault(cluster_id, []).append((gain, penalty))
    if not by_cluster:
        raise ValueError(f"No usable MSE utility rows found in {source_json}")

    fallback_name = str(fallback or "").strip()
    if fallback_name.lower() in {"none", "null", "off", "__none__"}:
        fallback_name = ""
    if fallback_name and fallback_name not in penalty_set:
        fallback_name = ""
    k_pick = max(1, int(topk))
    allowed: list[tuple[str, ...]] = []
    for cluster_id in range(max(by_cluster) + 1):
        names: list[str] = []
        entries = sorted(by_cluster.get(cluster_id, []), key=lambda pair: pair[0], reverse=True)
        for gain, penalty in entries:
            if gain <= float(min_gain):
                continue
            if penalty not in names:
                names.append(penalty)
            if len(names) >= k_pick:
                break
        if not names and fallback_name:
            names.append(fallback_name)
        allowed.append(tuple(names))
    return tuple(allowed)


def load_residual_profile(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    if payload.get("test_used", None) is not False:
        raise ValueError(f"Residual profile must declare test_used=false: {path}")
    if str(payload.get("selection_split", "")).lower() != "val":
        raise ValueError(f"Residual profile must be selected on val split: {path}")
    allowed = payload.get("allowed_by_cluster")
    fixed_cluster = payload.get("fixed_cluster_id")
    penalties = payload.get("penalties_enabled")
    if not isinstance(allowed, list) or not all(isinstance(item, list) for item in allowed):
        raise ValueError(f"Residual profile missing allowed_by_cluster list: {path}")
    if not isinstance(fixed_cluster, list) or not fixed_cluster:
        raise ValueError(f"Residual profile missing fixed_cluster_id list: {path}")
    if not isinstance(penalties, list) or not penalties:
        raise ValueError(f"Residual profile missing penalties_enabled list: {path}")
    return payload


def row_from_summary(
    *,
    source_row: dict[str, str],
    variant: Variant,
    cfg: dict[str, Any],
    config_path: Path,
    out_dir: Path,
    returncode: int,
    total_sec: float,
    error: str,
) -> dict[str, Any]:
    summary = read_json(out_dir / "run_summary.json")
    val = summary.get("val") or {}
    test = summary.get("test") or {}
    residual_selection = summary.get("moe_residual_selection") or {}
    base_mse = as_float(source_row.get("moe_mse"))
    base_mae = as_float(source_row.get("moe_mae"))
    test_mse = as_float(test.get("avg_mse"))
    test_mae = as_float(test.get("avg_mae"))
    reported_val_mse = residual_selection.get("val_scaled_avg_mse", val.get("avg_mse", ""))
    reported_val_mae = residual_selection.get("val_scaled_avg_mae", val.get("avg_mae", ""))

    def gain(base: float | None, current: float | None) -> str:
        if base is None or current is None or base == 0:
            return ""
        return f"{(base - current) / base * 100.0:.6f}"

    return {
        "status": "ok" if returncode == 0 and summary else ("failed" if returncode else "prepared"),
        "dataset": source_row.get("dataset", ""),
        "horizon": source_row.get("horizon", ""),
        "variant": variant.name,
        "base_moe_mse": source_row.get("moe_mse", ""),
        "base_moe_mae": source_row.get("moe_mae", ""),
        "test_mse": test.get("avg_mse", ""),
        "test_mae": test.get("avg_mae", ""),
        "val_mse": reported_val_mse,
        "val_mae": reported_val_mae,
        "val_mse_raw": val.get("avg_mse", ""),
        "val_mae_raw": val.get("avg_mae", ""),
        "residual_selection_policy": residual_selection.get("policy", ""),
        "mse_gain_vs_current_pct": gain(base_mse, test_mse),
        "mae_gain_vs_current_pct": gain(base_mae, test_mae),
        "beats_current_mse": bool(base_mse is not None and test_mse is not None and test_mse < base_mse),
        "beats_current_mae": bool(base_mae is not None and test_mae is not None and test_mae < base_mae),
        "penalties": ",".join((cfg.get("penalties", {}) or {}).get("enabled", []) or []),
        "gate_topk": (cfg.get("moe", {}) or {}).get("topk", ""),
        "cluster_prior_topk": ((cfg.get("moe", {}) or {}).get("cluster_penalty_prior", {}) or {}).get("topk", ""),
        "cluster_prior_logit_strength": ((cfg.get("moe", {}) or {}).get("cluster_penalty_prior", {}) or {}).get("logit_strength", ""),
        "mse_gate_weight": variant.mse_gate_weight,
        "mse_gate_temperature": variant.mse_gate_temperature,
        "pred_residual_enabled": ((cfg.get("moe", {}) or {}).get("pred_side_residual", {}) or {}).get("enable", ""),
        "residual_mean_scale": residual_selection.get("mean_scale", ""),
        "residual_num_channels": residual_selection.get("num_residual_channels", ""),
        "best_epoch": ",".join(str(v) for v in summary.get("best_epoch", [])) if summary else "",
        "total_sec": f"{float(total_sec):.3f}",
        "base_config": source_row.get("moe_config", ""),
        "config_path": str(config_path),
        "out_dir": str(out_dir),
        "returncode": int(returncode),
        "error": error,
    }


def select_rows(
    rows: list[dict[str, str]],
    datasets: list[str] | None,
    horizons: list[int] | None,
    limit: int,
) -> list[dict[str, str]]:
    dataset_filter = {d.lower() for d in datasets} if datasets else set()
    horizon_filter = {int(h) for h in horizons} if horizons else set()
    selected: list[dict[str, str]] = []
    for row in rows:
        dataset = str(row.get("dataset", "")).strip()
        if dataset_filter and dataset.lower() not in dataset_filter:
            continue
        horizon = int(row.get("horizon", "0") or 0)
        if horizon_filter and horizon not in horizon_filter:
            continue
        selected.append(row)
        if limit > 0 and len(selected) >= limit:
            break
    return selected


def run_one(
    *,
    row: dict[str, str],
    variant: Variant,
    out_root: Path,
    device: str,
    epochs_override: int | None,
    batch_size_override: int | None,
    skip_test: bool,
    reuse_existing: bool,
    dry_run: bool,
    enable_explainability: bool,
    explainability_splits: list[str],
    explainability_max_batches: int,
    utility_source_json: Path | None,
    utility_source_split: str,
    utility_topk: int,
    utility_min_gain: float,
    utility_fallback: str,
    utility_logit_strength: float | None,
    residual_profile: dict[str, Any] | None,
) -> dict[str, Any]:
    dataset = row["dataset"]
    horizon = int(row["horizon"])
    base_path = resolve(row["moe_config"])
    if not base_path.exists():
        return {
            "status": "missing_base_config",
            "dataset": dataset,
            "horizon": horizon,
            "variant": variant.name,
            "base_moe_mse": row.get("moe_mse", ""),
            "base_moe_mae": row.get("moe_mae", ""),
            "base_config": str(base_path),
            "returncode": "",
            "error": f"Base config not found: {base_path}",
        }
    out_dir = out_root / "runs" / dataset / f"H{horizon}" / variant.name
    config_path = out_root / "configs" / dataset / f"H{horizon}" / f"{variant.name}.yaml"
    base_cfg = load_yaml(base_path)
    penalties = list((base_cfg.get("penalties", {}) or {}).get("enabled", []) or [])
    allowed_by_cluster = None
    if utility_source_json is not None:
        allowed_by_cluster = build_mse_utility_allowed_by_cluster(
            utility_source_json,
            split=utility_source_split,
            topk=utility_topk,
            min_gain=utility_min_gain,
            fallback=utility_fallback,
            penalty_names=penalties,
        )
    cfg = patch_config(
        base_cfg,
        row,
        variant,
        out_dir,
        device,
        epochs_override,
        batch_size_override,
        skip_test,
        enable_explainability,
        explainability_splits,
        explainability_max_batches,
        allowed_by_cluster,
        utility_logit_strength,
        residual_profile,
    )
    write_yaml(config_path, cfg)
    if dry_run:
        return row_from_summary(
            source_row=row,
            variant=variant,
            cfg=cfg,
            config_path=config_path,
            out_dir=out_dir,
            returncode=0,
            total_sec=0.0,
            error="dry_run",
        )
    if reuse_existing and (out_dir / "run_summary.json").exists():
        return row_from_summary(
            source_row=row,
            variant=variant,
            cfg=cfg,
            config_path=config_path,
            out_dir=out_dir,
            returncode=0,
            total_sec=0.0,
            error="",
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = out_dir / "stdout.log"
    stderr_path = out_dir / "stderr.log"
    cmd = [sys.executable, "-u", "-m", "src.train", "--config", str(config_path)]
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    start = time.perf_counter()
    with stdout_path.open("w", encoding="utf-8") as stdout_f, stderr_path.open("w", encoding="utf-8") as stderr_f:
        completed = subprocess.run(cmd, cwd=str(ROOT), stdout=stdout_f, stderr=stderr_f, text=True, env=env)
    total_sec = time.perf_counter() - start
    error = ""
    if completed.returncode != 0:
        error = stderr_path.read_text(encoding="utf-8", errors="replace")[-3000:]
    return row_from_summary(
        source_row=row,
        variant=variant,
        cfg=cfg,
        config_path=config_path,
        out_dir=out_dir,
        returncode=int(completed.returncode),
        total_sec=total_sec,
        error=error,
    )


def update_summary_table(summary_path: Path, result_rows: list[dict[str, Any]]) -> int:
    fields, summary_rows = read_rows(summary_path)
    by_key = {(row.get("dataset"), str(row.get("horizon"))): row for row in summary_rows}
    improved_rows = [
        row
        for row in result_rows
        if row.get("status") == "ok"
        and str(row.get("beats_current_mse", "")).lower() == "true"
        and as_float(row.get("test_mse")) is not None
    ]
    improved_rows.sort(key=lambda row: (row.get("dataset", ""), int(row.get("horizon", 0)), as_float(row.get("test_mse"), 1e9)))
    updated = 0
    for result in improved_rows:
        key = (str(result["dataset"]), str(result["horizon"]))
        target = by_key.get(key)
        if target is None:
            continue
        current = as_float(target.get("moe_mse"))
        candidate = as_float(result.get("test_mse"))
        if current is None or candidate is None or candidate >= current:
            continue
        target["moe_mse"] = str(result.get("test_mse", ""))
        target["moe_mae"] = str(result.get("test_mae", ""))
        target["moe_variant"] = str(result.get("variant", ""))
        target["moe_config"] = str(result.get("config_path", ""))
        target["source"] = "input96_mse_gate_cluster_moe_retrain"
        target["status"] = "moe_done"
        backbone_mse = as_float(target.get("backbone_mse"))
        backbone_mae = as_float(target.get("backbone_mae"))
        target_mse = as_float(target.get("target_mse"))
        target_mae = as_float(target.get("target_mae"))
        candidate_mae = as_float(result.get("test_mae"))
        if backbone_mse is not None and backbone_mse != 0:
            target["mse_gain_pct"] = str((backbone_mse - candidate) / backbone_mse * 100.0)
        if backbone_mae is not None and backbone_mae != 0 and candidate_mae is not None:
            target["mae_gain_pct"] = str((backbone_mae - candidate_mae) / backbone_mae * 100.0)
        if target_mse is not None:
            target["moe_ok_mse"] = str(candidate <= target_mse)
        if target_mae is not None and candidate_mae is not None:
            target["moe_ok_mae"] = str(candidate_mae <= target_mae)
        if "mse_gain_pct" in target:
            gain = as_float(target.get("mse_gain_pct"), 0.0)
            target["gain2_mse"] = str(bool(gain is not None and gain >= 2.0))
        if "mae_gain_pct" in target:
            gain = as_float(target.get("mae_gain_pct"), 0.0)
            target["gain2_mae"] = str(bool(gain is not None and gain >= 2.0))
        updated += 1
    if updated:
        write_rows(summary_path, summary_rows, fields)
    return updated


def result_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (str(row.get("dataset", "")), str(row.get("horizon", "")), str(row.get("variant", "")))


def upsert_result(rows: list[dict[str, Any]], result: dict[str, Any]) -> None:
    key = result_key(result)
    for idx, row in enumerate(rows):
        if result_key(row) == key:
            rows[idx] = result
            return
    rows.append(result)


def main() -> None:
    ap = argparse.ArgumentParser(description="Retrain non-electricity input96 configs with MSE-gate cluster-aware MoE.")
    ap.add_argument("--summary-csv", default=str(DEFAULT_SUMMARY))
    ap.add_argument("--out-root", default="outputs/input96_mse_gate_cluster_moe_retrain_20260616")
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--horizons", nargs="*", type=int, default=None)
    ap.add_argument("--variants", nargs="*", default=["mse_gate_w002_softprior"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--epochs-override", type=int, default=None)
    ap.add_argument("--batch-size-override", type=int, default=None)
    ap.add_argument("--skip-test", action="store_true")
    ap.add_argument("--reuse-existing", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--update-summary", action="store_true")
    ap.add_argument("--enable-explainability", action="store_true")
    ap.add_argument("--explainability-splits", nargs="*", default=["val", "test"])
    ap.add_argument("--explainability-max-batches", type=int, default=0)
    ap.add_argument("--mse-utility-source-json", default=None)
    ap.add_argument("--mse-utility-source-split", default="val")
    ap.add_argument("--mse-utility-topk", type=int, default=2)
    ap.add_argument("--mse-utility-min-gain", type=float, default=0.0)
    ap.add_argument("--mse-utility-fallback", default="none")
    ap.add_argument("--mse-utility-logit-strength", type=float, default=0.3)
    ap.add_argument("--residual-profile-json", default=None)
    args = ap.parse_args()

    summary_path = resolve(args.summary_csv)
    out_root = resolve(args.out_root)
    _, table_rows = read_rows(summary_path)
    selected = select_rows(table_rows, args.datasets, args.horizons, int(args.limit))
    variants = [VARIANTS[name] for name in args.variants]
    utility_source_json = resolve(args.mse_utility_source_json) if args.mse_utility_source_json else None
    residual_profile = load_residual_profile(resolve(args.residual_profile_json)) if args.residual_profile_json else None
    if residual_profile is not None and utility_source_json is not None:
        raise ValueError("--residual-profile-json and --mse-utility-source-json are mutually exclusive.")
    result_rows: list[dict[str, Any]] = []
    results_path = out_root / "results.csv"
    if results_path.exists() and not args.dry_run:
        _, existing_rows = read_rows(results_path)
        result_rows.extend(existing_rows)
    for row in selected:
        for variant in variants:
            print(f"=== {row['dataset']} H{row['horizon']} {variant.name} ===", flush=True)
            result = run_one(
                row=row,
                variant=variant,
                out_root=out_root,
                device=str(args.device),
                epochs_override=args.epochs_override,
                batch_size_override=args.batch_size_override,
                skip_test=bool(args.skip_test),
                reuse_existing=bool(args.reuse_existing),
                dry_run=bool(args.dry_run),
                enable_explainability=bool(args.enable_explainability),
                explainability_splits=[str(x) for x in (args.explainability_splits or ["val", "test"])],
                explainability_max_batches=int(args.explainability_max_batches),
                utility_source_json=utility_source_json,
                utility_source_split=str(args.mse_utility_source_split),
                utility_topk=int(args.mse_utility_topk),
                utility_min_gain=float(args.mse_utility_min_gain),
                utility_fallback=str(args.mse_utility_fallback),
                utility_logit_strength=float(args.mse_utility_logit_strength),
                residual_profile=residual_profile,
            )
            upsert_result(result_rows, result)
            write_rows(results_path, result_rows)
            print(
                json.dumps(
                    {
                        "status": result.get("status"),
                        "dataset": result.get("dataset"),
                        "horizon": result.get("horizon"),
                        "variant": result.get("variant"),
                        "base_moe_mse": result.get("base_moe_mse"),
                        "test_mse": result.get("test_mse"),
                        "mse_gain_vs_current_pct": result.get("mse_gain_vs_current_pct"),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    if args.update_summary and not args.dry_run:
        updated = update_summary_table(summary_path, result_rows)
        print(f"Updated summary rows: {updated}", flush=True)
    print(f"Wrote: {results_path}", flush=True)


if __name__ == "__main__":
    main()
