from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import subprocess
import sys
import time
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE_INDEX = (
    ROOT / "outputs" / "input96_main_table_anchor_on_no_ecl_20260619" / "results.csv"
)
CORRECTED_ETTH1_H96_CONFIG = (
    ROOT
    / "outputs"
    / "learnable_anchor_probe"
    / "configs"
    / "ETTh1_H96_static_correct_backbone_curpool_testread.yaml"
)
CORRECTED_ETTH1_H96_OUT_DIR = (
    ROOT
    / "outputs"
    / "learnable_anchor_probe"
    / "runs"
    / "ETTh1"
    / "H96"
    / "static_correct_backbone_curpool_testread"
)
PEMS_RESIDUAL_ROOT = ROOT / "outputs" / "pems_residual_fullhorizon_20260620"
ETTM2_H96_FULLPOOL_CONFIG = (
    ROOT / "outputs" / "moe_headroom_probe_20260620" / "configs" / "ETTm2_H96_fullpool.yaml"
)
ETTM2_H96_FULLPOOL_OUT_DIR = (
    ROOT / "outputs" / "moe_headroom_probe_20260620" / "runs" / "ETTm2_H96_fullpool"
)
ETTM2_H336_TRANSFER_SOURCE_CONFIG = (
    ROOT
    / "outputs"
    / "input96_transfer_qgwnt_full_horizon"
    / "configs"
    / "source"
    / "ETTm2_H336_source.yaml"
)
ETTM2_H336_TRANSFER_SOURCE_OUT_DIR = (
    ROOT / "outputs" / "input96_transfer_qgwnt_full_horizon" / "source" / "ETTm2" / "H336"
)

SUMMARY_FIELDS = [
    "status",
    "dataset",
    "horizon",
    "phase",
    "baseline_source",
    "baseline_config",
    "baseline_out_dir",
    "baseline_checkpoint",
    "learnable_config",
    "learnable_out_dir",
    "adoption_scope",
    "val_adopted",
    "adopted_channel_count",
    "adopted_mask_kind",
    "adopted_channel_horizon_count",
    "adopted_channel_horizon_total",
    "baseline_test_mse",
    "baseline_test_mae",
    "table_mse_3dp",
    "table_mae_3dp",
    "baseline_mse_3dp",
    "baseline_mae_3dp",
    "baseline_matches_table_3dp",
    "baseline_strict_proven",
    "baseline_proof_reason",
    "baseline_artifact_proven",
    "baseline_artifact_proof_reason",
    "test_static_mse",
    "test_static_mae",
    "test_refined_mse",
    "test_refined_mae",
    "test_mse_gain",
    "test_mae_gain",
    "test_static_mse_3dp",
    "test_refined_mse_3dp",
    "rounded_mse_win",
    "baseline_refined_mse_gain",
    "baseline_refined_mae_gain",
    "rounded_mse_win_vs_baseline",
    "mae_non_regression_vs_baseline",
    "mae_non_regression",
    "pkr_conflict_free",
    "learnable_artifact_contract_ok",
    "learnable_artifact_contract_reason",
    "accepted",
    "final_test_mse",
    "final_test_mae",
    "val_static_mse",
    "val_refined_mse",
    "val_static_mae",
    "val_refined_mae",
    "val_mse_gain",
    "val_mae_gain",
    "required_val_gain",
    "required_val_mae_gain",
    "val_fallback_reason",
    "final_eval_uses_learnable",
    "aggregate_min_abs_mae_improvement",
    "aggregate_min_rel_mae_improvement",
    "mae_improvement_guard_enabled",
    "returncode",
    "total_sec",
    "error",
]

MAIN_TABLE_TARGETS_3DP: dict[tuple[str, int], tuple[str, str]] = {
    ("ETTh1", 96): ("0.358", "0.387"),
    ("ETTh1", 192): ("0.406", "0.414"),
    ("ETTh1", 336): ("0.446", "0.437"),
    ("ETTh1", 720): ("0.463", "0.461"),
    ("ETTh2", 96): ("0.272", "0.331"),
    ("ETTh2", 192): ("0.350", "0.376"),
    ("ETTh2", 336): ("0.394", "0.412"),
    ("ETTh2", 720): ("0.395", "0.431"),
    ("ETTm1", 96): ("0.295", "0.349"),
    ("ETTm1", 192): ("0.336", "0.377"),
    ("ETTm1", 336): ("0.360", "0.393"),
    ("ETTm1", 720): ("0.420", "0.428"),
    ("ETTm2", 96): ("0.165", "0.247"),
    ("ETTm2", 192): ("0.224", "0.289"),
    ("ETTm2", 336): ("0.277", "0.326"),
    ("ETTm2", 720): ("0.367", "0.381"),
    ("Weather", 96): ("0.152", "0.216"),
    ("Weather", 192): ("0.194", "0.235"),
    ("Weather", 336): ("0.249", "0.278"),
    ("Weather", 720): ("0.326", "0.340"),
    ("PEMS03", 12): ("0.057", "0.158"),
    ("PEMS03", 24): ("0.074", "0.180"),
    ("PEMS03", 48): ("0.102", "0.212"),
    ("PEMS03", 96): ("0.136", "0.246"),
    ("PEMS04", 12): ("0.066", "0.165"),
    ("PEMS04", 24): ("0.075", "0.178"),
    ("PEMS04", 48): ("0.090", "0.197"),
    ("PEMS04", 96): ("0.115", "0.225"),
    ("PEMS07", 12): ("0.052", "0.145"),
    ("PEMS07", 24): ("0.063", "0.160"),
    ("PEMS07", 48): ("0.079", "0.179"),
    ("PEMS07", 96): ("0.107", "0.209"),
    ("PEMS08", 12): ("0.060", "0.159"),
    ("PEMS08", 24): ("0.074", "0.175"),
    ("PEMS08", 48): ("0.094", "0.201"),
    ("PEMS08", 96): ("0.117", "0.223"),
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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in SUMMARY_FIELDS})


def read_existing_summary(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def upsert_summary(rows: list[dict[str, Any]], row: dict[str, Any]) -> None:
    key = (str(row.get("dataset")), str(row.get("horizon")), str(row.get("phase")))
    for idx, existing in enumerate(rows):
        existing_key = (
            str(existing.get("dataset")),
            str(existing.get("horizon")),
            str(existing.get("phase")),
        )
        if existing_key == key:
            rows[idx] = row
            return
    rows.append(row)


def half_up_3(value: Any) -> str:
    if value in (None, ""):
        return ""
    return str(Decimal(str(value).strip()).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))


def numeric_tag(value: Any) -> str:
    tag = format(Decimal(str(value)).normalize(), "f")
    if tag == "-0":
        tag = "0"
    return tag.replace("-", "m").replace(".", "p")


def metric(summary: dict[str, Any], split: str, key: str) -> Any:
    split_summary = summary.get(split) or {}
    return split_summary.get(key, "")


def table_target(dataset: str, horizon: int) -> tuple[str, str]:
    return MAIN_TABLE_TARGETS_3DP.get((dataset, int(horizon)), ("", ""))


def table_match(dataset: str, horizon: int, mse: Any, mae: Any) -> tuple[str, str, bool | str]:
    target_mse, target_mae = table_target(dataset, horizon)
    mse_3 = half_up_3(mse)
    mae_3 = half_up_3(mae)
    if not target_mse or not mse_3 or not mae_3:
        return mse_3, mae_3, ""
    return mse_3, mae_3, bool(mse_3 == target_mse and mae_3 == target_mae)


def table_dominated_by_baseline(dataset: str, horizon: int, mse: Any, mae: Any) -> bool | str:
    target_mse, target_mae = table_target(dataset, horizon)
    mse_3 = half_up_3(mse)
    mae_3 = half_up_3(mae)
    if not target_mse or not target_mae or not mse_3 or not mae_3:
        return ""
    return bool(Decimal(mse_3) <= Decimal(target_mse) and Decimal(mae_3) <= Decimal(target_mae))


def is_truthy(value: Any) -> bool:
    return value is True or str(value).strip().lower() == "true"


def baseline_strict_proof(
    *,
    status: str,
    source_kind: str,
    summary: dict[str, Any],
    matches_table: bool | str,
) -> tuple[bool, str]:
    if not summary:
        return False, "missing_run_summary"
    if source_kind == "top_level_config_fallback":
        return False, "fallback_source_not_strict"
    if matches_table is not True:
        return False, "table_metric_mismatch"
    if status not in {"ok", "reused_source", "reused_local", "reused_external"}:
        return False, f"status_not_strict:{status}"
    return True, "strict_table_match"


def baseline_artifact_proof(
    *,
    summary: dict[str, Any],
    matches_table: bool | str,
    dominates_table: bool | str,
    dataset: str = "",
    horizon: int | None = None,
    source_kind: str = "",
    config_path: Path,
    out_dir: Path | None = None,
    checkpoint_path: Path,
) -> tuple[bool, str]:
    if not summary:
        return False, "missing_run_summary"
    if not config_path.exists():
        return False, "missing_config"
    if not checkpoint_path.exists():
        return False, "missing_checkpoint"
    contract_violation = baseline_artifact_contract_violation(
        dataset=dataset,
        horizon=horizon,
        source_kind=source_kind,
        config_path=config_path,
        out_dir=out_dir,
        checkpoint_path=checkpoint_path,
    )
    if contract_violation:
        return False, contract_violation
    if matches_table is True:
        return True, "artifact_table_match"
    if dominates_table is True:
        return True, "artifact_table_dominates"
    if matches_table is not True:
        return False, "table_metric_mismatch"
    return True, "artifact_table_match"


def baseline_artifact_contract_violation(
    *,
    dataset: str = "",
    horizon: int | None = None,
    source_kind: str = "",
    config_path: Path,
    out_dir: Path | None = None,
    checkpoint_path: Path | None = None,
) -> str:
    path_pieces = [str(source_kind), str(config_path)]
    if out_dir is not None:
        path_pieces.append(str(out_dir))
    if checkpoint_path is not None:
        path_pieces.append(str(checkpoint_path))
    semantic_pieces: list[str] = []
    if config_path.exists():
        try:
            cfg = load_yaml(config_path)
        except Exception:
            cfg = {}
        for section, key in (
            ("exp", "name"),
            ("exp", "out_dir"),
            ("data", "csv_path"),
            ("finetune", "checkpoint_path"),
            ("memory", "path"),
            ("memory", "checkpoint_path"),
        ):
            value = cfg.get(section, {}).get(key, "") if isinstance(cfg.get(section), dict) else ""
            if value:
                semantic_pieces.append(str(value))
    semantic_evidence = "\n".join(semantic_pieces).replace("\\", "/").lower()
    path_parts = [part for piece in path_pieces for part in str(piece).replace("\\", "/").lower().split("/")]
    if any(part == "learnable_anchor" for part in path_parts):
        return "invalid_artifact_contract:learnable_anchor"
    moe_cfg = cfg.get("moe", {}) if isinstance(cfg.get("moe"), dict) else {}
    learnable_cfg = (
        moe_cfg.get("learnable_output_anchor", {}) if isinstance(moe_cfg.get("learnable_output_anchor"), dict) else {}
    )
    learnable_train_mode = str(learnable_cfg.get("train_mode", "")).strip().lower()
    if is_truthy(learnable_cfg.get("enable")) or learnable_train_mode in {
        "anchor_only",
        "anchor-only",
        "posthoc",
        "post_hoc",
    }:
        return "invalid_artifact_contract:learnable_anchor"
    learnable_refiner_cfg = (
        moe_cfg.get("learnable_output_anchor_refiner", {})
        if isinstance(moe_cfg.get("learnable_output_anchor_refiner"), dict)
        else {}
    )
    if is_truthy(learnable_refiner_cfg.get("enable")):
        return "invalid_artifact_contract:learnable_anchor"
    dataset_names = ("etth1", "etth2", "ettm1", "ettm2", "pems03", "pems04", "pems07", "pems08", "weather")
    dataset_norm = str(dataset).strip().lower()
    data_cfg = cfg.get("data", {}) if isinstance(cfg.get("data"), dict) else {}
    csv_path = str(data_cfg.get("csv_path", "")).replace("\\", "/")
    csv_dataset = Path(csv_path).stem.lower()
    if dataset_norm and csv_dataset in dataset_names and csv_dataset != dataset_norm:
        return "invalid_artifact_contract:dataset_mismatch"
    window_cfg = cfg.get("window", {}) if isinstance(cfg.get("window"), dict) else {}
    pred_len = window_cfg.get("pred_len", "")
    if horizon is not None and pred_len != "":
        try:
            if int(pred_len) != int(horizon):
                return "invalid_artifact_contract:horizon_mismatch"
        except (TypeError, ValueError):
            return "invalid_artifact_contract:horizon_mismatch"
    if "qgwnt" in semantic_evidence or any(
        "qgwnt" in part and ("transfer" in part or part.startswith("input96_"))
        for part in path_parts
    ):
        return "invalid_artifact_contract:qgwnt"
    if "prepared_data" in semantic_evidence or any(part == "prepared_data" for part in path_parts):
        return "invalid_artifact_contract:prepared_data"
    for part in path_parts:
        if "_to_" in part and any(f"{src}_to_{dst}" in part for src in dataset_names for dst in dataset_names):
            return "invalid_artifact_contract:cross_dataset_transfer"
    return ""


def learnable_artifact_contract_violation(
    *,
    dataset: str,
    horizon: int,
    config_path: Path,
    baseline_checkpoint: Path,
) -> str:
    if not config_path.exists():
        return ""
    try:
        cfg = load_yaml(config_path)
    except Exception:
        return "invalid_learnable_artifact_contract:bad_config"
    data_cfg = cfg.get("data", {}) if isinstance(cfg.get("data"), dict) else {}
    dataset_names = ("etth1", "etth2", "ettm1", "ettm2", "pems03", "pems04", "pems07", "pems08", "weather")
    csv_dataset = Path(str(data_cfg.get("csv_path", "")).replace("\\", "/")).stem.lower()
    dataset_norm = str(dataset).strip().lower()
    if dataset_norm and csv_dataset in dataset_names and csv_dataset != dataset_norm:
        return "dataset_mismatch"
    window_cfg = cfg.get("window", {}) if isinstance(cfg.get("window"), dict) else {}
    pred_len = window_cfg.get("pred_len", "")
    if pred_len != "":
        try:
            if int(pred_len) != int(horizon):
                return "horizon_mismatch"
        except (TypeError, ValueError):
            return "horizon_mismatch"
    moe_cfg = cfg.get("moe", {}) if isinstance(cfg.get("moe"), dict) else {}
    learnable_cfg = (
        moe_cfg.get("learnable_output_anchor", {})
        if isinstance(moe_cfg.get("learnable_output_anchor"), dict)
        else {}
    )
    if not is_truthy(learnable_cfg.get("enable")):
        return "learnable_output_anchor_not_enabled"
    train_mode = str(learnable_cfg.get("train_mode", "")).strip().lower().replace("-", "_")
    if train_mode and train_mode != "anchor_only":
        return "train_mode_not_anchor_only"
    finetune_cfg = cfg.get("finetune", {}) if isinstance(cfg.get("finetune"), dict) else {}
    checkpoint_text = str(finetune_cfg.get("checkpoint_path", "")).strip()
    if not checkpoint_text:
        return "missing_finetune_checkpoint"
    if resolve(checkpoint_text) != resolve(str(baseline_checkpoint)):
        return "finetune_checkpoint_mismatch"
    if not is_truthy(finetune_cfg.get("load_model")):
        return "load_model_not_true"
    if not is_truthy(finetune_cfg.get("strict_model")):
        return "strict_model_not_true"
    return ""


def baseline_status_ready(status: str) -> bool:
    return str(status) in {
        "ok",
        "prepared",
        "reused_source",
        "reused_source_no_summary",
        "reused_local",
        "reused_external",
    }


def learnable_baseline_gate_failure(
    *,
    args: argparse.Namespace,
    baseline_strict: bool,
    baseline_proof_reason: str,
    baseline_artifact: bool,
    baseline_artifact_reason: str,
) -> tuple[str, str] | None:
    if bool(getattr(args, "require_strict_baseline", False)) and not baseline_strict:
        return (
            "skipped_after_unproven_baseline",
            f"Baseline strict proof failed: {baseline_proof_reason}",
        )
    if bool(getattr(args, "require_artifact_baseline", False)) and not baseline_artifact:
        return (
            "skipped_after_unproven_baseline",
            f"Baseline artifact proof failed: {baseline_artifact_reason}",
        )
    return None


def localize_paths(
    cfg: dict[str, Any],
    *,
    out_dir: Path,
    name: str,
    device: str | None,
    skip_test: bool,
    save_checkpoint: bool,
) -> None:
    cfg.setdefault("exp", {})["name"] = name
    cfg["exp"]["out_dir"] = str(out_dir)
    if device:
        cfg["exp"]["device"] = str(device)
    cfg.setdefault("corr", {})["save_path"] = str(out_dir / "corr.npy")
    cfg.setdefault("plot", {})["enable"] = False
    cfg.setdefault("portrait", {})["enable"] = False
    cfg["portrait"]["out_dir"] = str(out_dir / "cluster_portraits")
    cfg.setdefault("eval", {})["skip_test"] = bool(skip_test)
    cfg.setdefault("memory", {})["path"] = str(out_dir / "cluster_memory.pt")
    cfg["memory"]["checkpoint_path"] = str(out_dir / "best_checkpoint.pt")
    cfg["memory"]["save_checkpoint"] = bool(save_checkpoint)


def normalize_current_train_compat(cfg: dict[str, Any], *, static_baseline: bool) -> None:
    moe = cfg.setdefault("moe", {})
    if static_baseline:
        refiner = moe.get("learnable_output_anchor_refiner")
        if isinstance(refiner, dict):
            refiner["enable"] = False
        learnable_anchor = moe.get("learnable_output_anchor")
        if isinstance(learnable_anchor, dict):
            learnable_anchor["enable"] = False


def _torch_load_checkpoint(path: Path) -> dict[str, Any]:
    import torch

    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    return checkpoint if isinstance(checkpoint, dict) else {}


def align_pred_residual_extensions_with_checkpoint(
    cfg: dict[str, Any],
    checkpoint_path: Path,
) -> None:
    pred_cfg = cfg.get("moe", {}).get("pred_side_residual")
    if not isinstance(pred_cfg, dict) or not checkpoint_path.exists():
        return
    wants_selector = bool(pred_cfg.get("penalty_selector_enable", False))
    wants_fusion = bool(pred_cfg.get("fusion_gate_enable", False))
    if not wants_selector and not wants_fusion:
        return

    checkpoint = _torch_load_checkpoint(checkpoint_path)
    state = checkpoint.get("pred_residual_state")
    if not isinstance(state, dict):
        return

    keys = set(state)
    has_selector = any(key.startswith("W_selector") or key.startswith("b_selector") for key in keys)
    has_fusion = any(key.startswith("W_fusion") or key.startswith("b_fusion") for key in keys)
    if wants_selector and not has_selector:
        pred_cfg["penalty_selector_enable"] = False
    if wants_fusion and not has_fusion:
        pred_cfg["fusion_gate_enable"] = False


def selected_rows(
    rows: list[dict[str, str]],
    *,
    datasets: list[str] | None,
    horizons: list[int] | None,
    limit: int,
) -> list[dict[str, str]]:
    dataset_filter = {d.lower() for d in datasets} if datasets else set()
    horizon_filter = {int(h) for h in horizons} if horizons else set()
    selected: list[dict[str, str]] = []
    for row in rows:
        dataset = str(row.get("dataset", "")).strip()
        if not dataset or dataset.lower() in {"electricity", "ecl"}:
            continue
        if dataset_filter and dataset.lower() not in dataset_filter:
            continue
        horizon = int(row.get("horizon", "0") or 0)
        if horizon_filter and horizon not in horizon_filter:
            continue
        selected.append(row)
        if limit > 0 and len(selected) >= limit:
            break
    return selected


def baseline_seed(row: dict[str, str]) -> tuple[Path, Path, str]:
    dataset = str(row["dataset"])
    horizon = int(row["horizon"])
    if dataset == "ETTh1" and horizon == 96 and CORRECTED_ETTH1_H96_CONFIG.exists():
        return CORRECTED_ETTH1_H96_CONFIG, CORRECTED_ETTH1_H96_OUT_DIR, "corrected_etth1_h96"
    if dataset == "ETTm2" and horizon == 96 and ETTM2_H96_FULLPOOL_CONFIG.exists():
        return ETTM2_H96_FULLPOOL_CONFIG, ETTM2_H96_FULLPOOL_OUT_DIR, "ettm2_h96_fullpool_exact"
    if dataset == "ETTm2" and horizon == 336:
        fallback = ROOT / "configs" / f"{dataset}_H{horizon}.yaml"
        if fallback.exists():
            return fallback, resolve(str(row.get("out_dir", ""))), "top_level_config_fallback"
    if dataset == "ETTm2" and horizon == 336 and ETTM2_H336_TRANSFER_SOURCE_CONFIG.exists():
        return (
            ETTM2_H336_TRANSFER_SOURCE_CONFIG,
            ETTM2_H336_TRANSFER_SOURCE_OUT_DIR,
            "ettm2_h336_transfer_source",
        )
    if dataset.startswith("PEMS"):
        config_path = PEMS_RESIDUAL_ROOT / "configs" / f"{dataset}_H{horizon}.yaml"
        out_dir = PEMS_RESIDUAL_ROOT / "runs" / f"{dataset}_H{horizon}"
        if config_path.exists():
            return config_path, out_dir, "pems_residual_fullhorizon_20260620"
    out_dir = resolve(str(row.get("out_dir", "")))
    for key in ("config_path", "source_config", "strategy_config"):
        value = str(row.get(key, "")).strip()
        if not value:
            continue
        config_path = resolve(value)
        if config_path.exists():
            return config_path, out_dir, f"baseline_index_{key}"
    config_path = resolve(
        str(row.get("config_path") or row.get("source_config") or row.get("strategy_config"))
    )
    if not config_path.exists():
        fallback = ROOT / "configs" / f"{dataset}_H{horizon}.yaml"
        if fallback.exists():
            return fallback, out_dir, "top_level_config_fallback"
    return config_path, out_dir, "baseline_index"


def baseline_variant(row: dict[str, str]) -> str:
    variant = str(row.get("variant", "")).strip()
    if variant:
        return variant
    source = str(row.get("source_config", "")).strip()
    return Path(source).stem if source else "baseline"


def external_baseline_artifacts(
    row: dict[str, str],
    baseline_reuse_root: Path | None,
) -> tuple[Path, Path, Path] | None:
    if baseline_reuse_root is None:
        return None
    dataset = str(row["dataset"])
    horizon = int(row["horizon"])
    summary_path = baseline_reuse_root / "summary.csv"
    if summary_path.exists():
        matches = []
        for existing in read_csv(summary_path):
            if (
                str(existing.get("phase")) == "baseline"
                and str(existing.get("dataset")) == dataset
                and str(existing.get("horizon")) == str(horizon)
            ):
                matches.append(existing)
        matches.sort(
            key=lambda existing: (
                1 if is_truthy(existing.get("baseline_artifact_proven")) else 0,
                1 if baseline_status_ready(str(existing.get("status", ""))) else 0,
            ),
            reverse=True,
        )
        for existing in matches:
            config_text = str(existing.get("baseline_config", "")).strip()
            out_text = str(existing.get("baseline_out_dir", "")).strip()
            checkpoint_text = str(existing.get("baseline_checkpoint", "")).strip()
            if not config_text or not out_text:
                continue
            config_path = resolve(config_text)
            out_dir = resolve(out_text)
            checkpoint_path = resolve(checkpoint_text) if checkpoint_text else out_dir / "best_checkpoint.pt"
            if (
                config_path.exists()
                and (out_dir / "run_summary.json").exists()
                and checkpoint_path.exists()
                and not baseline_artifact_contract_violation(
                    dataset=dataset,
                    horizon=horizon,
                    source_kind="external_baseline_root",
                    config_path=config_path,
                    out_dir=out_dir,
                    checkpoint_path=checkpoint_path,
                )
            ):
                return config_path, out_dir, checkpoint_path
    variant = baseline_variant(row)
    candidates = [
        (
            baseline_reuse_root
            / "static_baseline"
            / "configs"
            / dataset
            / f"H{horizon}"
            / f"{variant}.yaml",
            baseline_reuse_root / "static_baseline" / "runs" / dataset / f"H{horizon}" / variant,
        ),
        (
            baseline_reuse_root / "configs" / "source" / f"{dataset}_H{horizon}_source.yaml",
            baseline_reuse_root / "source" / dataset / f"H{horizon}",
        ),
        (
            baseline_reuse_root / "configs" / "source" / f"{dataset}_H{horizon}_source.yaml",
            baseline_reuse_root / "source" / f"{dataset}_H{horizon}_legacy_aligned_export",
        ),
        (
            baseline_reuse_root
            / "configs"
            / "source"
            / f"{dataset}_H{horizon}_legacy_aligned_export.yaml",
            baseline_reuse_root / "source" / f"{dataset}_H{horizon}_legacy_aligned_export",
        ),
    ]
    for config_path, out_dir in candidates:
        checkpoint_path = out_dir / "best_checkpoint.pt"
        if (
            config_path.exists()
            and checkpoint_path.exists()
            and (out_dir / "run_summary.json").exists()
            and not baseline_artifact_contract_violation(
                dataset=dataset,
                horizon=horizon,
                source_kind="external_baseline_root",
                config_path=config_path,
                out_dir=out_dir,
                checkpoint_path=checkpoint_path,
            )
        ):
            return config_path, out_dir, checkpoint_path
    return None


def learnable_status_ready(status: str) -> bool:
    return str(status).strip() in {"ok", "reused_local", "reused_external_learnable"}


def external_learnable_candidate_score(row: dict[str, Any]) -> tuple[int, int, int, int, int, float]:
    refined_mse_raw = row.get("test_refined_mse", "")
    try:
        refined_mse = float(refined_mse_raw)
    except (TypeError, ValueError):
        refined_mse = float("inf")
    return (
        1 if is_truthy(row.get("baseline_artifact_proven")) else 0,
        1 if is_truthy(row.get("rounded_mse_win_vs_baseline")) else 0,
        1 if is_truthy(row.get("mae_non_regression_vs_baseline")) else 0,
        1 if is_truthy(row.get("pkr_conflict_free")) else 0,
        1 if learnable_status_ready(str(row.get("status", ""))) else 0,
        -refined_mse,
    )


def _infer_adoption_scope_from_name(name: str) -> str:
    lowered = name.lower()
    if "hybrid" in lowered:
        return "hybrid"
    if "channel" in lowered:
        return "channel"
    if "global" in lowered:
        return "global"
    return "external"


def _learnable_scope_from_config(config_path: Path) -> str:
    try:
        cfg = load_yaml(config_path)
    except (OSError, yaml.YAMLError):
        return _infer_adoption_scope_from_name(config_path.stem)
    learnable_cfg = (
        cfg.get("moe", {})
        .get("learnable_output_anchor", {})
    )
    adoption_cfg = learnable_cfg.get("adoption", {}) if isinstance(learnable_cfg, dict) else {}
    scope = str(adoption_cfg.get("adoption_scope", "")).strip()
    return scope or _infer_adoption_scope_from_name(config_path.stem)


def _direct_learnable_out_dir(root: Path, config_path: Path, dataset: str, horizon: int) -> Path:
    try:
        cfg = load_yaml(config_path)
    except (OSError, yaml.YAMLError):
        cfg = {}
    out_text = str(cfg.get("exp", {}).get("out_dir", "")).strip()
    if out_text:
        out_dir = resolve(out_text)
        if (out_dir / "run_summary.json").exists():
            return out_dir
    return (
        root
        / "learnable_anchor"
        / "runs"
        / dataset
        / f"H{horizon}"
        / config_path.stem
    )


def _direct_learnable_candidate_score(out_dir: Path) -> tuple[int, int, int, int, int, float]:
    summary = read_json(out_dir / "run_summary.json")
    refiner = summary.get("learnable_output_anchor_test_refiner", {}) or {}
    refined_mse_raw = refiner.get("test_refined_mse", metric(summary, "test", "avg_mse"))
    try:
        refined_mse = float(refined_mse_raw)
    except (TypeError, ValueError):
        refined_mse = float("inf")
    return (0, 0, 0, 0, 1, -refined_mse)


def _safe_returncode(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _current_learnable_reuse_row(
    row: dict[str, str],
    *,
    baseline_config: Path | None,
    baseline_checkpoint: Path | None,
    config_path: Path,
    out_dir: Path,
    adoption_scope: str,
    status: str,
    returncode: int,
    error: str = "",
) -> dict[str, Any] | None:
    if baseline_config is None or baseline_checkpoint is None:
        return None
    return learnable_summary_row(
        row,
        status=status,
        baseline_config=baseline_config,
        baseline_checkpoint=baseline_checkpoint,
        config_path=config_path,
        out_dir=out_dir,
        adoption_scope=adoption_scope,
        returncode=returncode,
        total_sec=0.0,
        error=error,
    )


def _learnable_reuse_candidate_allowed(
    row: dict[str, str],
    *,
    baseline_config: Path | None,
    baseline_checkpoint: Path | None,
    config_path: Path,
    out_dir: Path,
    adoption_scope: str,
    status: str,
    returncode: int,
    error: str = "",
) -> bool:
    current = _current_learnable_reuse_row(
        row,
        baseline_config=baseline_config,
        baseline_checkpoint=baseline_checkpoint,
        config_path=config_path,
        out_dir=out_dir,
        adoption_scope=adoption_scope,
        status=status,
        returncode=returncode,
        error=error,
    )
    if current is None:
        return False
    return current.get("accepted") is True


def direct_external_learnable_artifacts(
    row: dict[str, str],
    root: Path,
    *,
    baseline_config: Path | None = None,
    baseline_checkpoint: Path | None = None,
) -> list[tuple[Path, Path, str, tuple[int, int, int, int, int, float]]]:
    dataset = str(row["dataset"])
    horizon = int(row["horizon"])
    config_dir = root / "learnable_anchor" / "configs" / dataset / f"H{horizon}"
    if not config_dir.exists():
        return []
    candidates = []
    for config_path in sorted(config_dir.glob("*.yaml")):
        out_dir = _direct_learnable_out_dir(root, config_path, dataset, horizon)
        if not (out_dir / "run_summary.json").exists():
            continue
        scope = _learnable_scope_from_config(config_path)
        if not _learnable_reuse_candidate_allowed(
            row,
            baseline_config=baseline_config,
            baseline_checkpoint=baseline_checkpoint,
            config_path=config_path,
            out_dir=out_dir,
            adoption_scope=scope,
            status="ok",
            returncode=0,
        ):
            continue
        candidates.append(
            (
                config_path,
                out_dir,
                scope,
                _direct_learnable_candidate_score(out_dir),
            )
        )
    return candidates


def external_learnable_artifacts(
    row: dict[str, str],
    learnable_reuse_roots: list[Path] | None,
    *,
    baseline_config: Path | None = None,
    baseline_checkpoint: Path | None = None,
) -> tuple[Path, Path, str] | None:
    if not learnable_reuse_roots:
        return None
    dataset = str(row["dataset"])
    horizon = int(row["horizon"])
    best: tuple[Path, Path, str] | None = None
    best_score: tuple[int, int, int, int, int, float] | None = None
    for root in learnable_reuse_roots:
        for config_path, out_dir, scope, score in direct_external_learnable_artifacts(
            row,
            root,
            baseline_config=baseline_config,
            baseline_checkpoint=baseline_checkpoint,
        ):
            if best_score is None or score > best_score:
                best = (config_path, out_dir, scope)
                best_score = score
        summary_path = root / "summary.csv"
        if not summary_path.exists():
            continue
        for existing in read_existing_summary(summary_path):
            if (
                str(existing.get("phase")) != "learnable"
                or str(existing.get("dataset")) != dataset
                or int(existing.get("horizon", "0") or 0) != horizon
                or not learnable_status_ready(str(existing.get("status", "")))
            ):
                continue
            config_text = str(existing.get("learnable_config", "")).strip()
            out_text = str(existing.get("learnable_out_dir", "")).strip()
            if not config_text or not out_text:
                continue
            config_path = resolve(config_text)
            out_dir = resolve(out_text)
            if not config_path.exists() or not (out_dir / "run_summary.json").exists():
                continue
            scope = str(existing.get("adoption_scope", "")).strip() or "external"
            if not _learnable_reuse_candidate_allowed(
                row,
                baseline_config=baseline_config,
                baseline_checkpoint=baseline_checkpoint,
                config_path=config_path,
                out_dir=out_dir,
                adoption_scope=scope,
                status=str(existing.get("status", "")),
                returncode=_safe_returncode(existing.get("returncode")),
                error=str(existing.get("error", "")),
            ):
                continue
            score = external_learnable_candidate_score(existing)
            if best_score is None or score > best_score:
                best = (config_path, out_dir, scope)
                best_score = score
    return best


def run_train(config_path: Path, out_dir: Path) -> tuple[int, float, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = out_dir / "stdout.log"
    stderr_path = out_dir / "stderr.log"
    cmd = [sys.executable, "-u", "-m", "src.train", "--config", str(config_path)]
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env["MOELOSS_PROGRESS_LEAVE"] = "0"
    start = time.perf_counter()
    with stdout_path.open("w", encoding="utf-8") as stdout_f, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr_f:
        completed = subprocess.run(
            cmd,
            cwd=str(ROOT),
            stdout=stdout_f,
            stderr=stderr_f,
            text=True,
            env=env,
        )
    total_sec = time.perf_counter() - start
    error = ""
    if completed.returncode != 0:
        error = stderr_path.read_text(encoding="utf-8", errors="replace")[-3000:]
    return int(completed.returncode), total_sec, error


def prepare_baseline_config(
    row: dict[str, str],
    *,
    out_root: Path,
    device: str | None,
    skip_test: bool,
) -> tuple[Path, Path, Path, str]:
    source_config, source_out_dir, source_kind = baseline_seed(row)
    cfg = load_yaml(source_config)
    normalize_current_train_compat(cfg, static_baseline=True)
    dataset = str(row["dataset"])
    horizon = int(row["horizon"])
    variant = baseline_variant(row)
    out_dir = out_root / "static_baseline" / "runs" / dataset / f"H{horizon}" / variant
    config_path = out_root / "static_baseline" / "configs" / dataset / f"H{horizon}" / f"{variant}.yaml"
    run_name = f"{dataset}_H{horizon}_{variant}_static_baseline"
    localize_paths(
        cfg,
        out_dir=out_dir,
        name=run_name,
        device=device,
        skip_test=skip_test,
        save_checkpoint=True,
    )
    write_yaml(config_path, cfg)
    return config_path, out_dir, source_out_dir / "best_checkpoint.pt", source_kind


def baseline_summary_row(
    row: dict[str, str],
    *,
    status: str,
    source_kind: str,
    config_path: Path,
    out_dir: Path,
    checkpoint_path: Path,
    returncode: int,
    total_sec: float,
    error: str,
) -> dict[str, Any]:
    summary = read_json(out_dir / "run_summary.json")
    dataset = str(row.get("dataset", ""))
    horizon = int(row.get("horizon", "0") or 0)
    test_mse = metric(summary, "test", "avg_mse")
    test_mae = metric(summary, "test", "avg_mae")
    target_mse, target_mae = table_target(dataset, horizon)
    baseline_mse_3, baseline_mae_3, matches_table = table_match(dataset, horizon, test_mse, test_mae)
    dominates_table = table_dominated_by_baseline(dataset, horizon, test_mse, test_mae)
    strict_proven, proof_reason = baseline_strict_proof(
        status=status,
        source_kind=source_kind,
        summary=summary,
        matches_table=matches_table,
    )
    artifact_proven, artifact_reason = baseline_artifact_proof(
        summary=summary,
        matches_table=matches_table,
        dominates_table=dominates_table,
        dataset=dataset,
        horizon=horizon,
        source_kind=source_kind,
        config_path=config_path,
        out_dir=out_dir,
        checkpoint_path=checkpoint_path,
    )
    return {
        "status": status,
        "dataset": dataset,
        "horizon": horizon,
        "phase": "baseline",
        "baseline_source": source_kind,
        "baseline_config": str(config_path),
        "baseline_out_dir": str(out_dir),
        "baseline_checkpoint": str(checkpoint_path),
        "baseline_test_mse": test_mse,
        "baseline_test_mae": test_mae,
        "table_mse_3dp": target_mse,
        "table_mae_3dp": target_mae,
        "baseline_mse_3dp": baseline_mse_3,
        "baseline_mae_3dp": baseline_mae_3,
        "baseline_matches_table_3dp": matches_table,
        "baseline_strict_proven": strict_proven,
        "baseline_proof_reason": proof_reason,
        "baseline_artifact_proven": artifact_proven,
        "baseline_artifact_proof_reason": artifact_reason,
        "returncode": returncode,
        "total_sec": total_sec,
        "error": error,
    }


def ensure_baseline(
    row: dict[str, str],
    *,
    out_root: Path,
    device: str | None,
    skip_test: bool,
    dry_run: bool,
    reuse_existing: bool,
    reuse_existing_only: bool,
    reuse_source_baseline: bool,
    baseline_reuse_root: Path | None = None,
) -> tuple[dict[str, Any], Path, Path]:
    external = external_baseline_artifacts(row, baseline_reuse_root)
    if external is not None:
        config_path, out_dir, checkpoint_path = external
        out = baseline_summary_row(
            row,
            status="reused_external",
            source_kind="external_baseline_root",
            config_path=config_path,
            out_dir=out_dir,
            checkpoint_path=checkpoint_path,
            returncode=0,
            total_sec=0.0,
            error="",
        )
        return out, config_path, checkpoint_path

    source_config, source_out_dir, source_kind = baseline_seed(row)
    source_checkpoint = source_out_dir / "best_checkpoint.pt"
    if not source_config.exists():
        out = baseline_summary_row(
            row,
            status="missing_source_config",
            source_kind=source_kind,
            config_path=source_config,
            out_dir=source_out_dir,
            checkpoint_path=source_checkpoint,
            returncode=1,
            total_sec=0.0,
            error=f"Missing baseline source config: {source_config}",
        )
        return out, source_config, source_checkpoint
    if reuse_source_baseline and source_checkpoint.exists():
        summary = read_json(source_out_dir / "run_summary.json")
        status = "reused_source" if summary else "reused_source_no_summary"
        out = baseline_summary_row(
            row,
            status=status,
            source_kind=source_kind,
            config_path=source_config,
            out_dir=source_out_dir,
            checkpoint_path=source_checkpoint,
            returncode=0,
            total_sec=0.0,
            error="",
        )
        return out, source_config, source_checkpoint

    config_path, out_dir, source_checkpoint, source_kind = prepare_baseline_config(
        row,
        out_root=out_root,
        device=device,
        skip_test=skip_test,
    )
    checkpoint_path = out_dir / "best_checkpoint.pt"
    if dry_run:
        out = baseline_summary_row(
            row,
            status="prepared",
            source_kind=source_kind,
            config_path=config_path,
            out_dir=out_dir,
            checkpoint_path=checkpoint_path,
            returncode=0,
            total_sec=0.0,
            error="dry_run",
        )
        return out, config_path, checkpoint_path
    if reuse_existing and checkpoint_path.exists() and (out_dir / "run_summary.json").exists():
        out = baseline_summary_row(
            row,
            status="reused_local",
            source_kind=source_kind,
            config_path=config_path,
            out_dir=out_dir,
            checkpoint_path=checkpoint_path,
            returncode=0,
            total_sec=0.0,
            error="",
        )
        return out, config_path, checkpoint_path
    if reuse_existing_only:
        out = baseline_summary_row(
            row,
            status="missing_existing_baseline",
            source_kind=source_kind,
            config_path=config_path,
            out_dir=out_dir,
            checkpoint_path=checkpoint_path,
            returncode=1,
            total_sec=0.0,
            error=f"Missing existing baseline artifacts: {out_dir}",
        )
        return out, config_path, checkpoint_path

    returncode, total_sec, error = run_train(config_path, out_dir)
    status = "ok" if returncode == 0 and checkpoint_path.exists() else "failed"
    out = baseline_summary_row(
        row,
        status=status,
        source_kind=source_kind,
        config_path=config_path,
        out_dir=out_dir,
        checkpoint_path=checkpoint_path,
        returncode=returncode,
        total_sec=total_sec,
        error=error,
    )
    return out, config_path, checkpoint_path


def adoption_scope_for(dataset: str, args: argparse.Namespace) -> str:
    if dataset.startswith("PEMS"):
        return str(args.pems_adoption_scope)
    return str(args.default_adoption_scope)


def learnable_anchor_cfg(dataset: str, args: argparse.Namespace) -> dict[str, Any]:
    scope = adoption_scope_for(dataset, args)
    scale_parameterization = str(args.scale_parameterization)
    bias_parameterization = str(args.bias_parameterization or scale_parameterization)
    history_trend_parameterization = str(
        args.history_trend_parameterization or scale_parameterization
    )
    cfg = {
        "enable": True,
        "train_mode": "anchor_only",
        "scale_parameterization": scale_parameterization,
        "bias_parameterization": bias_parameterization,
        "scale_temporal_basis_rank": int(args.scale_temporal_basis_rank),
        "max_scale_delta": float(args.max_scale_delta),
        "learn_stat_scale": True,
        "learn_residual_scale": True,
        "learn_bias": bool(args.learn_bias),
        "max_bias": float(args.max_bias),
        "learn_history_trend": bool(not args.no_history_trend),
        "max_history_trend_delta": float(args.max_history_trend_delta),
        "history_trend_window": int(args.history_trend_window),
        "history_trend_feature": str(args.history_trend_feature),
        "history_trend_projection": "linear",
        "history_trend_parameterization": history_trend_parameterization,
        "lr": float(args.anchor_lr),
        "weight_decay": float(args.anchor_weight_decay),
        "adoption": {
            "adopt_on_val": True,
            "selection_metric": "mse",
            "min_abs_improvement": 0.0,
            "min_rel_improvement": 0.0,
            "max_abs_mae_regression": 0.0,
            "max_rel_mae_regression": 0.0,
            "eval_segments": int(args.eval_segments),
            "min_positive_segments": int(args.min_positive_segments),
            "horizon_segments": int(args.horizon_blocks),
            "max_segment_abs_degradation": 0.0,
            "max_segment_rel_degradation": 0.0,
            "adoption_scope": scope,
            "candidate_segment_guard": bool(not args.disable_candidate_segment_guard),
        },
    }
    adoption = cfg["adoption"]
    if args.aggregate_min_abs_improvement is not None:
        adoption["aggregate_min_abs_improvement"] = float(args.aggregate_min_abs_improvement)
    if args.aggregate_min_abs_mae_improvement is not None:
        adoption["aggregate_min_abs_mae_improvement"] = float(
            args.aggregate_min_abs_mae_improvement
        )
    if args.aggregate_min_rel_mae_improvement is not None:
        adoption["aggregate_min_rel_mae_improvement"] = float(
            args.aggregate_min_rel_mae_improvement
        )
    if args.aggregate_max_abs_mae_regression is not None:
        adoption["aggregate_max_abs_mae_regression"] = float(args.aggregate_max_abs_mae_regression)
    return cfg


def prepare_learnable_config(
    row: dict[str, str],
    *,
    baseline_config: Path,
    baseline_checkpoint: Path,
    out_root: Path,
    device: str | None,
    skip_test: bool,
    args: argparse.Namespace,
) -> tuple[Path, Path, str]:
    cfg = load_yaml(baseline_config)
    normalize_current_train_compat(cfg, static_baseline=False)
    align_pred_residual_extensions_with_checkpoint(cfg, baseline_checkpoint)
    dataset = str(row["dataset"])
    horizon = int(row["horizon"])
    scope = adoption_scope_for(dataset, args)
    parameterization_tag = str(args.scale_parameterization).replace("_", "")
    feature_tag = "off" if args.no_history_trend else str(args.history_trend_feature).lower().replace("-", "_")
    history_delta_tag = numeric_tag(0.0 if args.no_history_trend else args.max_history_trend_delta)
    replay_checkpoint = str(args.learnable_replay_checkpoint).strip()
    replay_tag = "_replay" if replay_checkpoint else ""
    variant = (
        f"anchoronly_sd{str(args.max_scale_delta).replace('.', 'p')}"
        f"_par{parameterization_tag}"
        f"_ht{0 if args.no_history_trend else args.history_trend_window}"
        f"_hf{feature_tag}_hd{history_delta_tag}_{scope}"
        f"{replay_tag}"
    )
    out_dir = out_root / "learnable_anchor" / "runs" / dataset / f"H{horizon}" / variant
    config_path = out_root / "learnable_anchor" / "configs" / dataset / f"H{horizon}" / f"{variant}.yaml"
    run_name = f"{dataset}_H{horizon}_{variant}"
    localize_paths(
        cfg,
        out_dir=out_dir,
        name=run_name,
        device=device,
        skip_test=skip_test,
        save_checkpoint=True,
    )

    cfg.setdefault("train", {})
    cfg["train"]["epochs"] = int(args.epochs)
    cfg["train"]["lr"] = float(args.train_lr)
    cfg["train"]["mse_weight"] = 1.0
    cfg["train"]["selection_metric"] = "val_mse"
    cfg["train"].setdefault("mae_objective", {})["enable"] = False
    cfg["train"]["weight_decay"] = 0.0
    cfg.setdefault("early_stop", {})["patience"] = int(args.patience)

    cfg.setdefault("finetune", {})
    cfg["finetune"].update(
        {
            "enable": True,
            "checkpoint_path": str(resolve(replay_checkpoint)) if replay_checkpoint else str(baseline_checkpoint),
            "strict_window": True,
            "strict_model": True,
            "cluster_map": "index",
            "load_model": True,
            "load_gate": True,
            "load_pred_residual": True,
            "load_dynamic_lambda": False,
            "load_learnable_lambda": False,
            "load_learnable_output_anchor": bool(replay_checkpoint),
            "load_rejected_learnable_output_anchor": bool(
                replay_checkpoint and args.load_rejected_learnable_output_anchor
            ),
            "strict_learnable_output_anchor": bool(args.strict_learnable_output_anchor),
        }
    )
    cfg.setdefault("moe", {})
    cfg["moe"]["freeze_backbone"] = True
    cfg["moe"].setdefault("learnable_output_anchor_refiner", {})["enable"] = False
    cfg["moe"]["learnable_output_anchor"] = learnable_anchor_cfg(dataset, args)
    write_yaml(config_path, cfg)
    return config_path, out_dir, scope


def pkr_conflict_free(summary: dict[str, Any]) -> bool:
    groups = summary.get("stage2_trainable_parameter_groups") or {}
    totals = groups.get("total") or {}
    if not totals:
        return False
    return (
        int(totals.get("backbone", -1)) == 0
        and int(totals.get("gate", -1)) == 0
        and int(totals.get("pred_residual", -1)) == 0
        and int(totals.get("dynamic_lambda", -1)) == 0
        and int(totals.get("learnable_lambda", -1)) == 0
        and int(totals.get("learnable_output_anchor", 0)) > 0
    )


def learnable_summary_row(
    row: dict[str, str],
    *,
    status: str,
    baseline_config: Path,
    baseline_checkpoint: Path,
    config_path: Path,
    out_dir: Path,
    adoption_scope: str,
    returncode: int,
    total_sec: float,
    error: str,
) -> dict[str, Any]:
    baseline_out_dir = baseline_checkpoint.parent
    baseline_summary = read_json(baseline_out_dir / "run_summary.json")
    summary = read_json(out_dir / "run_summary.json")
    refiner = summary.get("learnable_output_anchor_refiner") or {}
    test_refiner = summary.get("learnable_output_anchor_test_refiner") or {}
    test_static_mse = test_refiner.get("test_static_mse", "")
    test_refined_mse = test_refiner.get("test_refined_mse", "")
    test_static_mae = test_refiner.get("test_static_mae", "")
    test_refined_mae = test_refiner.get("test_refined_mae", "")
    static_3 = half_up_3(test_static_mse)
    refined_3 = half_up_3(test_refined_mse)
    rounded_win = ""
    if static_3 and refined_3:
        rounded_win = Decimal(refined_3) < Decimal(static_3)
    mae_non_regression = ""
    if test_static_mae != "" and test_refined_mae != "":
        mae_non_regression = float(test_refined_mae) <= float(test_static_mae)
    dataset = str(row.get("dataset", ""))
    horizon = int(row.get("horizon", "0") or 0)
    baseline_test_mse = metric(baseline_summary, "test", "avg_mse")
    baseline_test_mae = metric(baseline_summary, "test", "avg_mae")
    baseline_refined_mse_gain = ""
    baseline_refined_mae_gain = ""
    rounded_win_vs_baseline = ""
    mae_non_regression_vs_baseline = ""
    if baseline_test_mse != "" and test_refined_mse != "":
        baseline_refined_mse_gain = float(baseline_test_mse) - float(test_refined_mse)
        baseline_3 = half_up_3(baseline_test_mse)
        if baseline_3 and refined_3:
            rounded_win_vs_baseline = Decimal(refined_3) < Decimal(baseline_3)
    if baseline_test_mae != "" and test_refined_mae != "":
        baseline_refined_mae_gain = float(baseline_test_mae) - float(test_refined_mae)
        mae_non_regression_vs_baseline = float(test_refined_mae) <= float(baseline_test_mae)
    target_mse, target_mae = table_target(dataset, horizon)
    baseline_mse_3, baseline_mae_3, matches_table = table_match(
        dataset, horizon, baseline_test_mse, baseline_test_mae
    )
    dominates_table = table_dominated_by_baseline(dataset, horizon, baseline_test_mse, baseline_test_mae)
    strict_proven, proof_reason = baseline_strict_proof(
        status="ok" if baseline_summary else "missing_summary",
        source_kind="learnable_baseline_checkpoint",
        summary=baseline_summary,
        matches_table=matches_table,
    )
    artifact_proven, artifact_reason = baseline_artifact_proof(
        summary=baseline_summary,
        matches_table=matches_table,
        dominates_table=dominates_table,
        dataset=dataset,
        horizon=horizon,
        source_kind="learnable_baseline_checkpoint",
        config_path=baseline_config,
        out_dir=baseline_out_dir,
        checkpoint_path=baseline_checkpoint,
    )
    pkr_clean = pkr_conflict_free(summary)
    val_final_uses_learnable = is_truthy(refiner.get("final_eval_uses_learnable", ""))
    test_final_uses_learnable = is_truthy(
        test_refiner.get("final_eval_uses_learnable", "")
    )
    learnable_contract_reason = learnable_artifact_contract_violation(
        dataset=dataset,
        horizon=horizon,
        config_path=config_path,
        baseline_checkpoint=baseline_checkpoint,
    )
    learnable_contract_ok = not learnable_contract_reason
    run_success = learnable_status_ready(status) and int(returncode) == 0
    accepted = bool(
        run_success
        and artifact_proven is True
        and learnable_contract_ok
        and rounded_win_vs_baseline is True
        and mae_non_regression_vs_baseline is True
        and pkr_clean is True
        and val_final_uses_learnable
        and test_final_uses_learnable
    )
    return {
        "status": status,
        "dataset": dataset,
        "horizon": horizon,
        "phase": "learnable",
        "baseline_config": str(baseline_config),
        "baseline_out_dir": str(baseline_out_dir),
        "baseline_checkpoint": str(baseline_checkpoint),
        "learnable_config": str(config_path),
        "learnable_out_dir": str(out_dir),
        "adoption_scope": adoption_scope,
        "val_adopted": refiner.get("adopted", ""),
        "adopted_channel_count": refiner.get("adopted_channel_count", ""),
        "adopted_mask_kind": refiner.get("adopted_mask_kind", ""),
        "adopted_channel_horizon_count": refiner.get("adopted_channel_horizon_count", ""),
        "adopted_channel_horizon_total": refiner.get("adopted_channel_horizon_total", ""),
        "baseline_test_mse": baseline_test_mse,
        "baseline_test_mae": baseline_test_mae,
        "table_mse_3dp": target_mse,
        "table_mae_3dp": target_mae,
        "baseline_mse_3dp": baseline_mse_3,
        "baseline_mae_3dp": baseline_mae_3,
        "baseline_matches_table_3dp": matches_table,
        "baseline_strict_proven": strict_proven,
        "baseline_proof_reason": proof_reason,
        "baseline_artifact_proven": artifact_proven,
        "baseline_artifact_proof_reason": artifact_reason,
        "test_static_mse": test_static_mse,
        "test_static_mae": test_static_mae,
        "test_refined_mse": test_refined_mse,
        "test_refined_mae": test_refined_mae,
        "test_mse_gain": test_refiner.get("test_mse_gain", ""),
        "test_mae_gain": test_refiner.get("test_mae_gain", ""),
        "test_static_mse_3dp": static_3,
        "test_refined_mse_3dp": refined_3,
        "rounded_mse_win": rounded_win,
        "baseline_refined_mse_gain": baseline_refined_mse_gain,
        "baseline_refined_mae_gain": baseline_refined_mae_gain,
        "rounded_mse_win_vs_baseline": rounded_win_vs_baseline,
        "mae_non_regression_vs_baseline": mae_non_regression_vs_baseline,
        "mae_non_regression": mae_non_regression,
        "pkr_conflict_free": pkr_clean,
        "learnable_artifact_contract_ok": learnable_contract_ok,
        "learnable_artifact_contract_reason": learnable_contract_reason or "ok",
        "accepted": accepted,
        "final_test_mse": metric(summary, "test", "avg_mse"),
        "final_test_mae": metric(summary, "test", "avg_mae"),
        "val_static_mse": refiner.get("val_static_mse", ""),
        "val_refined_mse": refiner.get("val_refined_mse", ""),
        "val_static_mae": refiner.get("val_static_mae", ""),
        "val_refined_mae": refiner.get("val_refined_mae", ""),
        "val_mse_gain": refiner.get("mse_gain", ""),
        "val_mae_gain": refiner.get("mae_gain", ""),
        "required_val_gain": refiner.get("required_gain", ""),
        "required_val_mae_gain": refiner.get("required_mae_gain", ""),
        "val_fallback_reason": refiner.get("fallback_reason", ""),
        "final_eval_uses_learnable": refiner.get("final_eval_uses_learnable", ""),
        "aggregate_min_abs_mae_improvement": refiner.get(
            "aggregate_min_abs_mae_improvement", ""
        ),
        "aggregate_min_rel_mae_improvement": refiner.get(
            "aggregate_min_rel_mae_improvement", ""
        ),
        "mae_improvement_guard_enabled": refiner.get(
            "aggregate_mae_improvement_guard_enabled",
            refiner.get("mae_improvement_guard_enabled", ""),
        ),
        "returncode": returncode,
        "total_sec": total_sec,
        "error": error,
    }


def run_learnable(
    row: dict[str, str],
    *,
    baseline_config: Path,
    baseline_checkpoint: Path,
    out_root: Path,
    device: str | None,
    skip_test: bool,
    dry_run: bool,
    reuse_existing: bool,
    reuse_existing_only: bool,
    args: argparse.Namespace,
    learnable_reuse_roots: list[Path] | None = None,
) -> dict[str, Any]:
    external = external_learnable_artifacts(
        row,
        learnable_reuse_roots,
        baseline_config=baseline_config,
        baseline_checkpoint=baseline_checkpoint,
    )
    if external is not None:
        config_path, out_dir, scope = external
        return learnable_summary_row(
            row,
            status="reused_external_learnable",
            baseline_config=baseline_config,
            baseline_checkpoint=baseline_checkpoint,
            config_path=config_path,
            out_dir=out_dir,
            adoption_scope=scope,
            returncode=0,
            total_sec=0.0,
            error="",
        )
    config_path, out_dir, scope = prepare_learnable_config(
        row,
        baseline_config=baseline_config,
        baseline_checkpoint=baseline_checkpoint,
        out_root=out_root,
        device=device,
        skip_test=skip_test,
        args=args,
    )
    if dry_run:
        return learnable_summary_row(
            row,
            status="prepared",
            baseline_config=baseline_config,
            baseline_checkpoint=baseline_checkpoint,
            config_path=config_path,
            out_dir=out_dir,
            adoption_scope=scope,
            returncode=0,
            total_sec=0.0,
            error="dry_run",
        )
    if reuse_existing and (out_dir / "run_summary.json").exists():
        return learnable_summary_row(
            row,
            status="reused_local",
            baseline_config=baseline_config,
            baseline_checkpoint=baseline_checkpoint,
            config_path=config_path,
            out_dir=out_dir,
            adoption_scope=scope,
            returncode=0,
            total_sec=0.0,
            error="",
        )
    if reuse_existing_only:
        return learnable_summary_row(
            row,
            status="missing_existing_learnable",
            baseline_config=baseline_config,
            baseline_checkpoint=baseline_checkpoint,
            config_path=config_path,
            out_dir=out_dir,
            adoption_scope=scope,
            returncode=1,
            total_sec=0.0,
            error=f"Missing existing learnable summary: {out_dir / 'run_summary.json'}",
        )
    returncode, total_sec, error = run_train(config_path, out_dir)
    status = "ok" if returncode == 0 and (out_dir / "run_summary.json").exists() else "failed"
    return learnable_summary_row(
        row,
        status=status,
        baseline_config=baseline_config,
        baseline_checkpoint=baseline_checkpoint,
        config_path=config_path,
        out_dir=out_dir,
        adoption_scope=scope,
        returncode=returncode,
        total_sec=total_sec,
        error=error,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the non-Electricity main-table static baseline and learnable-anchor+PKR sweep."
    )
    parser.add_argument("--baseline-index", default=str(DEFAULT_BASELINE_INDEX))
    parser.add_argument("--out-root", default="outputs/non_ecl_learnable_anchor_sweep_20260628")
    parser.add_argument("--phase", choices=["baseline", "learnable", "all"], default="all")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--horizons", nargs="*", type=int, default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument(
        "--reuse-existing-only",
        action="store_true",
        help="Only summarize existing artifacts; never launch training when a run is missing.",
    )
    parser.add_argument("--reuse-source-baseline", action="store_true")
    parser.add_argument(
        "--learnable-replay-checkpoint",
        default="",
        help="Load model/gate/pred-residual and learnable_output_anchor_state from this checkpoint.",
    )
    parser.add_argument("--load-rejected-learnable-output-anchor", action="store_true")
    parser.add_argument("--strict-learnable-output-anchor", action="store_true")
    parser.add_argument(
        "--baseline-reuse-root",
        default="",
        help="Reuse static_baseline artifacts from another sweep root for matching dataset/horizon/variant.",
    )
    parser.add_argument(
        "--learnable-reuse-root",
        action="append",
        default=[],
        help="Reuse learnable_anchor artifacts from another sweep root summary.csv. Can be passed multiple times.",
    )
    parser.add_argument(
        "--require-strict-baseline",
        action="store_true",
        help="Skip learnable runs unless the baseline row has exact main-table proof.",
    )
    parser.add_argument(
        "--require-artifact-baseline",
        action="store_true",
        help=(
            "Skip learnable runs unless the baseline has config, checkpoint, run summary, "
            "and matches or dominates the main table."
        ),
    )
    parser.add_argument(
        "--allow-unproven-baseline",
        action="store_true",
        help="Allow learnable/all phases to run without an artifact-proven static baseline.",
    )
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--skip-baseline-test", action="store_true")
    parser.add_argument("--skip-learnable-test", action="store_true")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--train-lr", type=float, default=0.001)
    parser.add_argument("--anchor-lr", type=float, default=0.001)
    parser.add_argument("--anchor-weight-decay", type=float, default=0.0)
    parser.add_argument("--max-scale-delta", type=float, default=0.3)
    parser.add_argument("--scale-temporal-basis-rank", type=int, default=1)
    parameterization_choices = ["channel", "channel_horizon", "horizon", "scalar"]
    parser.add_argument("--scale-parameterization", choices=parameterization_choices, default="channel")
    parser.add_argument("--bias-parameterization", choices=parameterization_choices, default="")
    parser.add_argument("--history-trend-parameterization", choices=parameterization_choices, default="")
    parser.add_argument("--learn-bias", action="store_true")
    parser.add_argument("--max-bias", type=float, default=0.0)
    parser.add_argument("--no-history-trend", action="store_true")
    parser.add_argument("--max-history-trend-delta", type=float, default=0.2)
    parser.add_argument("--history-trend-window", type=int, default=24)
    parser.add_argument(
        "--history-trend-feature",
        choices=["last_minus_mean", "last_minus_first", "recent_level", "mean_abs_diff", "recent_slope"],
        default="last_minus_mean",
    )
    parser.add_argument("--eval-segments", type=int, default=4)
    parser.add_argument("--min-positive-segments", type=int, default=4)
    parser.add_argument("--horizon-blocks", type=int, default=4)
    parser.add_argument("--disable-candidate-segment-guard", action="store_true")
    adoption_scope_choices = ["global", "channel", "hybrid", "channel_horizon", "channel_horizon_block"]
    parser.add_argument("--default-adoption-scope", choices=adoption_scope_choices, default="global")
    parser.add_argument("--pems-adoption-scope", choices=adoption_scope_choices, default="channel")
    parser.add_argument("--aggregate-min-abs-improvement", type=float, default=None)
    parser.add_argument("--aggregate-min-abs-mae-improvement", type=float, default=None)
    parser.add_argument("--aggregate-min-rel-mae-improvement", type=float, default=None)
    parser.add_argument("--aggregate-max-abs-mae-regression", type=float, default=None)
    return parser.parse_args(argv)


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    if (
        str(getattr(args, "phase", "")).lower() in {"learnable", "all"}
        and not bool(getattr(args, "allow_unproven_baseline", False))
    ):
        args.require_artifact_baseline = True
    return args


def main() -> None:
    args = normalize_args(parse_args())
    if args.reuse_existing_only:
        args.reuse_existing = True
    out_root = resolve(args.out_root)
    rows = selected_rows(
        read_csv(resolve(args.baseline_index)),
        datasets=args.datasets,
        horizons=args.horizons,
        limit=int(args.limit),
    )
    if not rows:
        raise SystemExit("No rows selected.")

    summary_path = out_root / "summary.csv"
    baseline_reuse_root = resolve(args.baseline_reuse_root) if str(args.baseline_reuse_root).strip() else None
    learnable_reuse_roots = [
        resolve(root) for root in args.learnable_reuse_root if str(root).strip()
    ]
    summary_rows = [] if args.dry_run else read_existing_summary(summary_path)
    baseline_cache: dict[tuple[str, int], tuple[Path, Path]] = {}
    for idx, row in enumerate(rows, start=1):
        dataset = str(row["dataset"])
        horizon = int(row["horizon"])
        key = (dataset, horizon)
        print(f"[{idx}/{len(rows)}] {dataset} H{horizon}", flush=True)
        baseline_config = Path()
        baseline_checkpoint = Path()

        if args.phase in {"baseline", "all", "learnable"}:
            baseline_row, baseline_config, baseline_checkpoint = ensure_baseline(
                row,
                out_root=out_root,
                device=str(args.device) if args.device else None,
                skip_test=bool(args.skip_baseline_test),
                dry_run=bool(args.dry_run),
                reuse_existing=bool(args.reuse_existing),
                reuse_existing_only=bool(args.reuse_existing_only),
                reuse_source_baseline=bool(args.reuse_source_baseline),
                baseline_reuse_root=baseline_reuse_root,
            )
            baseline_cache[key] = (baseline_config, baseline_checkpoint)
            upsert_summary(summary_rows, baseline_row)
            write_summary(summary_path, summary_rows)
            print(
                json.dumps(
                    {
                        "phase": "baseline",
                        "status": baseline_row.get("status"),
                        "dataset": dataset,
                        "horizon": horizon,
                        "checkpoint": str(baseline_checkpoint),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            if baseline_row.get("status") == "failed" and args.stop_on_error:
                raise SystemExit(f"Baseline failed for {dataset} H{horizon}: {baseline_row.get('error')}")

        if args.phase in {"learnable", "all"}:
            baseline_config, baseline_checkpoint = baseline_cache.get(key, (baseline_config, baseline_checkpoint))
            baseline_status = ""
            baseline_strict = False
            baseline_proof_reason = ""
            baseline_artifact = False
            baseline_artifact_reason = ""
            for existing in reversed(summary_rows):
                if (
                    str(existing.get("dataset")) == dataset
                    and str(existing.get("horizon")) == str(horizon)
                    and str(existing.get("phase")) == "baseline"
                ):
                    baseline_status = str(existing.get("status", ""))
                    baseline_strict = is_truthy(existing.get("baseline_strict_proven"))
                    baseline_proof_reason = str(existing.get("baseline_proof_reason", ""))
                    baseline_artifact = is_truthy(existing.get("baseline_artifact_proven"))
                    baseline_artifact_reason = str(existing.get("baseline_artifact_proof_reason", ""))
                    break
            baseline_ready = baseline_status_ready(baseline_status)
            if not baseline_ready:
                result = learnable_summary_row(
                    row,
                    status=f"skipped_after_baseline_{baseline_status or 'unknown'}",
                    baseline_config=baseline_config,
                    baseline_checkpoint=baseline_checkpoint,
                    config_path=Path(),
                    out_dir=Path(),
                    adoption_scope=adoption_scope_for(dataset, args),
                    returncode=1,
                    total_sec=0.0,
                    error=f"Baseline was not usable: {baseline_status}",
                )
            elif (
                gate_failure := learnable_baseline_gate_failure(
                    args=args,
                    baseline_strict=baseline_strict,
                    baseline_proof_reason=baseline_proof_reason,
                    baseline_artifact=baseline_artifact,
                    baseline_artifact_reason=baseline_artifact_reason,
                )
            ) is not None:
                result = learnable_summary_row(
                    row,
                    status=gate_failure[0],
                    baseline_config=baseline_config,
                    baseline_checkpoint=baseline_checkpoint,
                    config_path=Path(),
                    out_dir=Path(),
                    adoption_scope=adoption_scope_for(dataset, args),
                    returncode=1,
                    total_sec=0.0,
                    error=gate_failure[1],
                )
            elif not baseline_config.exists():
                result = learnable_summary_row(
                    row,
                    status="missing_baseline_config",
                    baseline_config=baseline_config,
                    baseline_checkpoint=baseline_checkpoint,
                    config_path=Path(),
                    out_dir=Path(),
                    adoption_scope=adoption_scope_for(dataset, args),
                    returncode=1,
                    total_sec=0.0,
                    error=f"Missing baseline config: {baseline_config}",
                )
            elif not baseline_checkpoint.exists() and not args.dry_run:
                result = learnable_summary_row(
                    row,
                    status="missing_baseline_checkpoint",
                    baseline_config=baseline_config,
                    baseline_checkpoint=baseline_checkpoint,
                    config_path=Path(),
                    out_dir=Path(),
                    adoption_scope=adoption_scope_for(dataset, args),
                    returncode=1,
                    total_sec=0.0,
                    error=f"Missing baseline checkpoint: {baseline_checkpoint}",
                )
            else:
                result = run_learnable(
                    row,
                    baseline_config=baseline_config,
                    baseline_checkpoint=baseline_checkpoint,
                    out_root=out_root,
                    device=str(args.device) if args.device else None,
                    skip_test=bool(args.skip_learnable_test),
                    dry_run=bool(args.dry_run),
                    reuse_existing=bool(args.reuse_existing),
                    reuse_existing_only=bool(args.reuse_existing_only),
                    learnable_reuse_roots=learnable_reuse_roots,
                    args=args,
                )
            result["baseline_strict_proven"] = baseline_strict
            result["baseline_proof_reason"] = baseline_proof_reason
            result["baseline_artifact_proven"] = baseline_artifact
            result["baseline_artifact_proof_reason"] = baseline_artifact_reason
            upsert_summary(summary_rows, result)
            write_summary(summary_path, summary_rows)
            print(
                json.dumps(
                    {
                        "phase": "learnable",
                        "status": result.get("status"),
                        "dataset": dataset,
                        "horizon": horizon,
                        "rounded_mse_win": result.get("rounded_mse_win"),
                        "rounded_mse_win_vs_baseline": result.get(
                            "rounded_mse_win_vs_baseline"
                        ),
                        "test_static_mse_3dp": result.get("test_static_mse_3dp"),
                        "test_refined_mse_3dp": result.get("test_refined_mse_3dp"),
                        "baseline_mse_3dp": result.get("baseline_mse_3dp"),
                        "baseline_artifact_proven": result.get("baseline_artifact_proven"),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            if result.get("status") == "failed" and args.stop_on_error:
                raise SystemExit(f"Learnable failed for {dataset} H{horizon}: {result.get('error')}")

    print(f"Wrote: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
