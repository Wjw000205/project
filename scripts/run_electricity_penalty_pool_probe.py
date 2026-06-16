from __future__ import annotations

import argparse
import copy
import csv
import json
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_BASE_CONFIGS = {
    96: ROOT
    / "outputs"
    / "e_h96_alpha095_frozen_resid_moe_extreme"
    / "configs"
    / "electricity"
    / "H96"
    / "moe_activation"
    / "trainresid_seg12_p168_mse2000_s801.yaml",
    192: ROOT
    / "outputs"
    / "electricity_strict_20260615_moe_h192"
    / "configs"
    / "electricity"
    / "H192"
    / "moe_activation"
    / "trainresid_seg12_p168_mse2000_s801.yaml",
    336: ROOT
    / "outputs"
    / "electricity_strict_20260615_moe_h336"
    / "configs"
    / "electricity"
    / "H336"
    / "moe_activation"
    / "trainresid_seg12_p168_mse2000_s801.yaml",
    720: ROOT
    / "outputs"
    / "e_h720_best224_frozen_resid_moe_wide"
    / "configs"
    / "electricity"
    / "H720"
    / "moe_activation"
    / "trainresid_seg12_p168_mse300_s121.yaml",
}

FIELDS = [
    "status",
    "horizon",
    "variant",
    "penalties",
    "router_mode",
    "router_penalty_context_weight",
    "gate_topk",
    "select_ranks",
    "gate_balance_weight",
    "mse_utility_gate_enable",
    "mse_utility_gate_weight",
    "mse_utility_gate_temperature",
    "mse_utility_gate_min_gain",
    "mse_utility_gate_target_power",
    "cluster_penalty_prior_enable",
    "cluster_penalty_prior_topk",
    "cluster_penalty_prior_hard_topk",
    "cluster_penalty_prior_logit_strength",
    "cluster_penalty_prior_always_include",
    "channel_penalty_prior_enable",
    "channel_penalty_prior_topk",
    "seasonal_anchor_period",
    "seasonal_anchor_num_periods",
    "seasonal_anchor_scale",
    "train_residual_anchor_enable",
    "train_residual_anchor_steps",
    "val_mse",
    "val_mae",
    "test_mse",
    "test_mae",
    "mse_gain_vs_strict_best_pct",
    "mae_gain_vs_strict_best_pct",
    "residual_mean_scale",
    "residual_num_channels",
    "best_epoch",
    "total_sec",
    "config_path",
    "out_dir",
    "returncode",
    "error",
]

STRICT_BEST = {
    96: {"mse": 0.13797663152217865, "mae": 0.2359694540500641},
    192: {"mse": 0.15302953124046326, "mae": 0.2505868971347809},
    336: {"mse": 0.1679292619228363, "mae": 0.2669973373413086},
    720: {"mse": 0.20272107422351837, "mae": 0.3013533353805542},
}


@dataclass(frozen=True)
class PoolCandidate:
    variant: str
    penalties: tuple[str, ...]
    seasonal_anchor_period: int = 0
    seasonal_anchor_num_periods: int = 1
    seasonal_anchor_scale: float = 0.0
    gate_topk: int = 1
    select_ranks: tuple[int, ...] = (1,)
    gate_balance_weight: float = 0.0
    mse_utility_gate_enable: bool = False
    mse_utility_gate_weight: float = 0.0
    mse_utility_gate_temperature: float = 1.0
    mse_utility_gate_min_gain: float = 0.0
    mse_utility_gate_target_power: float = 1.0
    router_mode: str = "penalty_context"
    router_penalty_context_weight: float = 1.0
    gate_route_on_penalty_only: bool = True
    cluster_penalty_prior_enable: bool = False
    cluster_penalty_prior_topk: int = 0
    cluster_penalty_prior_hard_topk: bool = True
    cluster_penalty_prior_logit_strength: float = 0.0
    cluster_penalty_prior_temperature: float = 1.0
    cluster_penalty_prior_smoothing: float = 0.02
    cluster_penalty_prior_use_as_balance_target: bool = False
    cluster_penalty_prior_always_include: tuple[str, ...] = ()
    cluster_allowed_by_cluster: tuple[tuple[str, ...], ...] | None = None
    channel_penalty_prior_enable: bool = False
    channel_penalty_prior_topk: int = 0
    channel_penalty_prior_hard_topk: bool = True
    channel_penalty_prior_temperature: float = 1.0
    channel_penalty_prior_smoothing: float = 0.02
    lambda_scale: float = 0.0
    residual_init_alpha: float = -2.5
    residual_alpha_scale: float = 1.0
    residual_corrector_hidden: int = 64
    residual_specialization_weight: float = 0.1
    residual_norm_weight: float = 1.0e-5
    selection_policy: str = "val_mse_gate_guarded"
    gate_calibrator_epochs: int = 40
    gate_calibrator_max_scale: float = 1.0
    gate_calibrator_init_scale: float = 0.3
    gate_calibrator_apply_activation_threshold: bool = True
    gate_calibrator_activation_bce_weight: float = 0.2
    train_residual_anchor_max_scale: float | None = None
    explainability_enable: bool = False


POOL_CANDIDATES = [
    PoolCandidate(
        "current_plus_seasonal_p24_ctx",
        ("amp_under", "delta", "diff_amp", "direction", "seasonal_align"),
        seasonal_anchor_period=24,
        seasonal_anchor_num_periods=4,
        seasonal_anchor_scale=0.6,
    ),
    PoolCandidate(
        "lddf_plus_seasonal_p24_ctx",
        ("level", "delta", "d2_match", "diff_amp", "seasonal_align"),
        seasonal_anchor_period=24,
        seasonal_anchor_num_periods=4,
        seasonal_anchor_scale=0.6,
    ),
    PoolCandidate(
        "load_shape_seasonal_p24_ctx",
        ("level", "range", "trend", "delta", "seasonal_align"),
        seasonal_anchor_period=24,
        seasonal_anchor_num_periods=4,
        seasonal_anchor_scale=0.6,
    ),
    PoolCandidate(
        "load_shape_level_range_ctx",
        ("level", "range"),
    ),
    PoolCandidate(
        "load_shape_cluster_top1_prior",
        ("level", "range", "trend", "delta", "seasonal_align"),
        seasonal_anchor_period=24,
        seasonal_anchor_num_periods=4,
        seasonal_anchor_scale=0.6,
        router_mode="learned",
        router_penalty_context_weight=0.0,
        cluster_penalty_prior_enable=True,
        cluster_penalty_prior_topk=1,
        cluster_penalty_prior_hard_topk=True,
        cluster_penalty_prior_logit_strength=0.8,
        explainability_enable=True,
    ),
    PoolCandidate(
        "load_shape_cluster_top2_prior",
        ("level", "range", "trend", "delta", "seasonal_align"),
        seasonal_anchor_period=24,
        seasonal_anchor_num_periods=4,
        seasonal_anchor_scale=0.6,
        router_mode="learned",
        router_penalty_context_weight=0.0,
        cluster_penalty_prior_enable=True,
        cluster_penalty_prior_topk=2,
        cluster_penalty_prior_hard_topk=True,
        cluster_penalty_prior_logit_strength=0.8,
        explainability_enable=True,
    ),
    PoolCandidate(
        "load_shape_cluster_top2_ctx",
        ("level", "range", "trend", "delta", "seasonal_align"),
        seasonal_anchor_period=24,
        seasonal_anchor_num_periods=4,
        seasonal_anchor_scale=0.6,
        router_mode="penalty_context",
        router_penalty_context_weight=0.5,
        cluster_penalty_prior_enable=True,
        cluster_penalty_prior_topk=2,
        cluster_penalty_prior_hard_topk=True,
        cluster_penalty_prior_logit_strength=0.5,
        explainability_enable=True,
    ),
    PoolCandidate(
        "load_shape_cluster_top1_plus_seasonal",
        ("level", "range", "trend", "delta", "seasonal_align"),
        seasonal_anchor_period=24,
        seasonal_anchor_num_periods=4,
        seasonal_anchor_scale=0.6,
        router_mode="learned",
        router_penalty_context_weight=0.0,
        cluster_penalty_prior_enable=True,
        cluster_penalty_prior_topk=1,
        cluster_penalty_prior_hard_topk=True,
        cluster_penalty_prior_logit_strength=0.5,
        cluster_penalty_prior_always_include=("seasonal_align",),
        explainability_enable=True,
    ),
    PoolCandidate(
        "load_shape_cluster_top1_plus_seasonal_ch2",
        ("level", "range", "trend", "delta", "seasonal_align"),
        seasonal_anchor_period=24,
        seasonal_anchor_num_periods=4,
        seasonal_anchor_scale=0.6,
        router_mode="learned",
        router_penalty_context_weight=0.0,
        cluster_penalty_prior_enable=True,
        cluster_penalty_prior_topk=1,
        cluster_penalty_prior_hard_topk=True,
        cluster_penalty_prior_logit_strength=0.5,
        cluster_penalty_prior_always_include=("seasonal_align",),
        channel_penalty_prior_enable=True,
        channel_penalty_prior_topk=2,
        explainability_enable=True,
    ),
    PoolCandidate(
        "load_shape_cluster_top2_plus_seasonal_ch2_gate2",
        ("level", "range", "trend", "delta", "seasonal_align"),
        seasonal_anchor_period=24,
        seasonal_anchor_num_periods=4,
        seasonal_anchor_scale=0.6,
        gate_topk=2,
        router_mode="learned",
        router_penalty_context_weight=0.0,
        cluster_penalty_prior_enable=True,
        cluster_penalty_prior_topk=2,
        cluster_penalty_prior_hard_topk=True,
        cluster_penalty_prior_logit_strength=0.5,
        cluster_penalty_prior_always_include=("seasonal_align",),
        channel_penalty_prior_enable=True,
        channel_penalty_prior_topk=2,
        explainability_enable=True,
    ),
    PoolCandidate(
        "load_shape_cluster_top2_plus_seasonal_ch2_gate2_bal",
        ("level", "range", "trend", "delta", "seasonal_align"),
        seasonal_anchor_period=24,
        seasonal_anchor_num_periods=4,
        seasonal_anchor_scale=0.6,
        gate_topk=2,
        gate_balance_weight=0.01,
        router_mode="learned",
        router_penalty_context_weight=0.0,
        cluster_penalty_prior_enable=True,
        cluster_penalty_prior_topk=2,
        cluster_penalty_prior_hard_topk=True,
        cluster_penalty_prior_logit_strength=0.4,
        cluster_penalty_prior_use_as_balance_target=True,
        cluster_penalty_prior_always_include=("seasonal_align",),
        channel_penalty_prior_enable=True,
        channel_penalty_prior_topk=2,
        explainability_enable=True,
    ),
    PoolCandidate(
        "load_shape_h192_valgain_cluster_map_gate2",
        ("level", "range", "trend", "delta", "seasonal_align"),
        seasonal_anchor_period=24,
        seasonal_anchor_num_periods=4,
        seasonal_anchor_scale=0.6,
        gate_topk=2,
        router_mode="learned",
        router_penalty_context_weight=0.0,
        cluster_penalty_prior_enable=True,
        cluster_penalty_prior_topk=0,
        cluster_penalty_prior_hard_topk=True,
        cluster_penalty_prior_logit_strength=0.3,
        cluster_allowed_by_cluster=(
            ("seasonal_align", "trend"),
            ("seasonal_align",),
            ("seasonal_align", "range"),
            ("trend",),
            ("seasonal_align",),
            ("seasonal_align", "trend"),
            ("seasonal_align",),
            ("seasonal_align", "level"),
            ("seasonal_align", "delta"),
            ("seasonal_align", "trend"),
            ("seasonal_align",),
        ),
        explainability_enable=True,
    ),
    PoolCandidate(
        "load_shape_mse_utility_cluster_map_gate2",
        ("level", "range", "trend", "delta", "seasonal_align"),
        seasonal_anchor_period=24,
        seasonal_anchor_num_periods=4,
        seasonal_anchor_scale=0.6,
        gate_topk=2,
        router_mode="learned",
        router_penalty_context_weight=0.0,
        cluster_penalty_prior_enable=True,
        cluster_penalty_prior_topk=0,
        cluster_penalty_prior_hard_topk=True,
        cluster_penalty_prior_logit_strength=0.3,
        explainability_enable=True,
    ),
    PoolCandidate(
        "load_shape_mse_utility_gate_w002",
        ("level", "range", "trend", "delta", "seasonal_align"),
        seasonal_anchor_period=24,
        seasonal_anchor_num_periods=4,
        seasonal_anchor_scale=0.6,
        gate_topk=2,
        mse_utility_gate_enable=True,
        mse_utility_gate_weight=0.02,
        mse_utility_gate_temperature=0.5,
        router_mode="learned",
        router_penalty_context_weight=0.0,
        cluster_penalty_prior_enable=True,
        cluster_penalty_prior_topk=0,
        cluster_penalty_prior_hard_topk=True,
        cluster_penalty_prior_logit_strength=0.3,
        explainability_enable=True,
    ),
    PoolCandidate(
        "load_shape_mse_utility_gate_w002_nomask",
        ("level", "range", "trend", "delta", "seasonal_align"),
        seasonal_anchor_period=24,
        seasonal_anchor_num_periods=4,
        seasonal_anchor_scale=0.6,
        gate_topk=2,
        mse_utility_gate_enable=True,
        mse_utility_gate_weight=0.02,
        mse_utility_gate_temperature=0.5,
        router_mode="learned",
        router_penalty_context_weight=0.0,
        cluster_penalty_prior_enable=False,
        explainability_enable=True,
    ),
    PoolCandidate(
        "load_shape_mse_utility_gate_w002_ch2",
        ("level", "range", "trend", "delta", "seasonal_align"),
        seasonal_anchor_period=24,
        seasonal_anchor_num_periods=4,
        seasonal_anchor_scale=0.6,
        gate_topk=2,
        mse_utility_gate_enable=True,
        mse_utility_gate_weight=0.02,
        mse_utility_gate_temperature=0.5,
        router_mode="learned",
        router_penalty_context_weight=0.0,
        cluster_penalty_prior_enable=False,
        channel_penalty_prior_enable=True,
        channel_penalty_prior_topk=2,
        explainability_enable=True,
    ),
    PoolCandidate(
        "load_shape_mse_utility_gate_w005",
        ("level", "range", "trend", "delta", "seasonal_align"),
        seasonal_anchor_period=24,
        seasonal_anchor_num_periods=4,
        seasonal_anchor_scale=0.6,
        gate_topk=2,
        mse_utility_gate_enable=True,
        mse_utility_gate_weight=0.05,
        mse_utility_gate_temperature=0.5,
        router_mode="learned",
        router_penalty_context_weight=0.0,
        cluster_penalty_prior_enable=True,
        cluster_penalty_prior_topk=0,
        cluster_penalty_prior_hard_topk=True,
        cluster_penalty_prior_logit_strength=0.3,
        explainability_enable=True,
    ),
    PoolCandidate(
        "load_shape_mse_utility_gate_w010",
        ("level", "range", "trend", "delta", "seasonal_align"),
        seasonal_anchor_period=24,
        seasonal_anchor_num_periods=4,
        seasonal_anchor_scale=0.6,
        gate_topk=2,
        mse_utility_gate_enable=True,
        mse_utility_gate_weight=0.10,
        mse_utility_gate_temperature=0.5,
        router_mode="learned",
        router_penalty_context_weight=0.0,
        cluster_penalty_prior_enable=True,
        cluster_penalty_prior_topk=0,
        cluster_penalty_prior_hard_topk=True,
        cluster_penalty_prior_logit_strength=0.3,
        explainability_enable=True,
    ),
    PoolCandidate(
        "load_shape_mse_utility_gate_w002_rank12",
        ("level", "range", "trend", "delta", "seasonal_align"),
        seasonal_anchor_period=24,
        seasonal_anchor_num_periods=4,
        seasonal_anchor_scale=0.6,
        gate_topk=2,
        select_ranks=(1, 2),
        mse_utility_gate_enable=True,
        mse_utility_gate_weight=0.02,
        mse_utility_gate_temperature=0.5,
        router_mode="learned",
        router_penalty_context_weight=0.0,
        cluster_penalty_prior_enable=True,
        cluster_penalty_prior_topk=0,
        cluster_penalty_prior_hard_topk=True,
        cluster_penalty_prior_logit_strength=0.3,
        explainability_enable=True,
    ),
    PoolCandidate(
        "wide_shape_cluster_top2_prior",
        ("level", "range", "trend", "delta", "seasonal_align", "amp_under", "diff_amp", "direction", "d2_match", "corr"),
        seasonal_anchor_period=24,
        seasonal_anchor_num_periods=4,
        seasonal_anchor_scale=0.6,
        router_mode="learned",
        router_penalty_context_weight=0.0,
        cluster_penalty_prior_enable=True,
        cluster_penalty_prior_topk=2,
        cluster_penalty_prior_hard_topk=True,
        cluster_penalty_prior_logit_strength=0.8,
        residual_specialization_weight=0.05,
        explainability_enable=True,
    ),
    PoolCandidate(
        "wide_shape_cluster_top2_ctx",
        ("level", "range", "trend", "delta", "seasonal_align", "amp_under", "diff_amp", "direction", "d2_match", "corr"),
        seasonal_anchor_period=24,
        seasonal_anchor_num_periods=4,
        seasonal_anchor_scale=0.6,
        router_mode="penalty_context",
        router_penalty_context_weight=0.5,
        cluster_penalty_prior_enable=True,
        cluster_penalty_prior_topk=2,
        cluster_penalty_prior_hard_topk=True,
        cluster_penalty_prior_logit_strength=0.5,
        residual_specialization_weight=0.05,
        explainability_enable=True,
    ),
    PoolCandidate(
        "load_shape_seasonal_p24_strongresid_ctx",
        ("level", "range", "trend", "delta", "seasonal_align"),
        seasonal_anchor_period=24,
        seasonal_anchor_num_periods=4,
        seasonal_anchor_scale=0.6,
        residual_init_alpha=-2.0,
        residual_alpha_scale=1.5,
        gate_calibrator_max_scale=1.5,
        gate_calibrator_init_scale=0.5,
        train_residual_anchor_max_scale=30.0,
    ),
    PoolCandidate(
        "load_shape_seasonal_p24_softgate_ctx",
        ("level", "range", "trend", "delta", "seasonal_align"),
        seasonal_anchor_period=24,
        seasonal_anchor_num_periods=4,
        seasonal_anchor_scale=0.6,
        residual_init_alpha=-2.0,
        residual_alpha_scale=1.25,
        gate_calibrator_max_scale=1.25,
        gate_calibrator_init_scale=0.5,
        gate_calibrator_apply_activation_threshold=False,
        gate_calibrator_activation_bce_weight=0.05,
    ),
    PoolCandidate(
        "load_shape_seasonal_p24_w05_ctx",
        ("level", "range", "trend", "delta", "seasonal_align"),
        seasonal_anchor_period=24,
        seasonal_anchor_num_periods=4,
        seasonal_anchor_scale=0.6,
        router_penalty_context_weight=0.5,
    ),
    PoolCandidate(
        "load_shape_seasonal_p24_w20_ctx",
        ("level", "range", "trend", "delta", "seasonal_align"),
        seasonal_anchor_period=24,
        seasonal_anchor_num_periods=4,
        seasonal_anchor_scale=0.6,
        router_penalty_context_weight=2.0,
    ),
    PoolCandidate(
        "load_shape_seasonal_p24_s04_ctx",
        ("level", "range", "trend", "delta", "seasonal_align"),
        seasonal_anchor_period=24,
        seasonal_anchor_num_periods=4,
        seasonal_anchor_scale=0.4,
    ),
    PoolCandidate(
        "load_shape_seasonal_p24_s08_ctx",
        ("level", "range", "trend", "delta", "seasonal_align"),
        seasonal_anchor_period=24,
        seasonal_anchor_num_periods=4,
        seasonal_anchor_scale=0.8,
    ),
    PoolCandidate(
        "load_shape_seasonal_p24_w15_ctx",
        ("level", "range", "trend", "delta", "seasonal_align"),
        seasonal_anchor_period=24,
        seasonal_anchor_num_periods=4,
        seasonal_anchor_scale=0.6,
        router_penalty_context_weight=1.5,
    ),
    PoolCandidate(
        "load_shape_diff_seasonal_p24_ctx",
        ("level", "range", "trend", "delta", "diff_amp", "seasonal_align"),
        seasonal_anchor_period=24,
        seasonal_anchor_num_periods=4,
        seasonal_anchor_scale=0.6,
    ),
    PoolCandidate(
        "lddf_no_seasonal_ctx",
        ("level", "delta", "d2_match", "diff_amp"),
        seasonal_anchor_period=0,
        seasonal_anchor_scale=0.0,
    ),
    PoolCandidate(
        "lddf_plus_seasonal_p96_ctx",
        ("level", "delta", "d2_match", "diff_amp", "seasonal_align"),
        seasonal_anchor_period=96,
        seasonal_anchor_num_periods=1,
        seasonal_anchor_scale=0.6,
    ),
]


def resolve(path_text: str | Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return (ROOT / path).resolve()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_yaml(path: Path, cfg: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=False, sort_keys=False)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_mse_utility_allowed_by_cluster(
    source_json: Path,
    *,
    split: str,
    topk: int,
    min_gain: float,
    fallback: str,
) -> tuple[tuple[str, ...], ...]:
    payload = read_json(source_json)
    split_payload = (payload.get("splits", {}) or {}).get(str(split).lower())
    if not isinstance(split_payload, dict):
        raise ValueError(f"Split {split!r} not found in MSE utility source: {source_json}")
    rows = split_payload.get("rows", [])
    if not isinstance(rows, list) or len(rows) == 0:
        raise ValueError(f"No explainability rows found for split {split!r} in {source_json}")

    by_cluster: dict[int, list[tuple[float, str]]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        penalty = str(row.get("penalty", "")).strip()
        if not penalty:
            continue
        try:
            cluster_id = int(row.get("cluster_id"))
        except (TypeError, ValueError):
            continue
        try:
            gain = float(row.get("mean_single_penalty_gain_mse", 0.0) or 0.0)
        except (TypeError, ValueError):
            gain = 0.0
        by_cluster.setdefault(cluster_id, []).append((gain, penalty))
    if not by_cluster:
        raise ValueError(f"No usable MSE utility rows found in {source_json}")

    k_max = max(by_cluster)
    allowed: list[tuple[str, ...]] = []
    k_pick = max(1, int(topk))
    fallback_name = str(fallback or "").strip()
    if fallback_name.lower() in {"none", "null", "off", "__none__"}:
        fallback_name = ""
    for k in range(k_max + 1):
        entries = sorted(by_cluster.get(k, []), key=lambda item: item[0], reverse=True)
        names: list[str] = []
        for gain, penalty in entries:
            if gain <= float(min_gain):
                continue
            if penalty not in names:
                names.append(penalty)
            if len(names) >= k_pick:
                break
        if fallback_name and fallback_name not in names:
            names.append(fallback_name)
        allowed.append(tuple(names))
    return tuple(allowed)


def apply_mse_utility_to_candidate(
    cand: PoolCandidate,
    allowed_by_cluster: tuple[tuple[str, ...], ...],
    *,
    fallback: str,
    logit_strength: float,
) -> PoolCandidate:
    penalty_set = set(cand.penalties)
    fallback_name = str(fallback or "").strip()
    if fallback_name.lower() in {"none", "null", "off", "__none__"}:
        fallback_name = ""
    filtered_rows: list[tuple[str, ...]] = []
    for row in allowed_by_cluster:
        names = []
        for name in row:
            if name in penalty_set and name not in names:
                names.append(name)
        if not names and fallback_name in penalty_set:
            names.append(fallback_name)
        filtered_rows.append(tuple(names))
    return replace(
        cand,
        cluster_penalty_prior_enable=True,
        cluster_penalty_prior_topk=0,
        cluster_penalty_prior_hard_topk=True,
        cluster_penalty_prior_logit_strength=float(logit_strength),
        cluster_penalty_prior_always_include=(),
        cluster_allowed_by_cluster=tuple(filtered_rows),
    )


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDS})


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


def lambda_dict(names: tuple[str, ...], value: float) -> dict[str, float]:
    return {name: float(value) for name in names}


def selector_cfg() -> dict[str, Any]:
    return {
        "enable": True,
        "source_split": "val",
        "epochs": 30,
        "train_fraction": 0.75,
        "hidden_dim": 64,
        "batch_size": 256,
        "lr": 1.0e-3,
        "weight_decay": 1.0e-4,
        "standardize_features": True,
        "class_weight": "none",
        "class_weight_max": 8.0,
        "init_skip_bias": 0.0,
        "use_penalty_identity": True,
        "positive_sample_weight": 1.0,
        "negative_sample_weight": 1.0,
        "label_min_abs_improvement": 0.0,
        "label_min_rel_improvement": 0.0,
    }


def gate_calibrator_cfg(cand: PoolCandidate | None = None) -> dict[str, Any]:
    cand = cand or PoolCandidate("_default", tuple())
    return {
        "source_split": "val",
        "loss": "mse",
        "selection_metric": "mse",
        "epochs": int(cand.gate_calibrator_epochs),
        "train_fraction": 0.75,
        "hidden_dim": 64,
        "batch_size": 256,
        "max_scale": float(cand.gate_calibrator_max_scale),
        "init_scale": float(cand.gate_calibrator_init_scale),
        "scale_reg": 5.0e-4,
        "standardize_features": True,
        "activation_head_enable": True,
        "apply_activation_threshold": bool(cand.gate_calibrator_apply_activation_threshold),
        "activation_threshold": "auto",
        "activation_threshold_selection_metric": "mse",
        "activation_threshold_scope": "channel",
        "activation_bce_weight": float(cand.gate_calibrator_activation_bce_weight),
        "activation_inactive_scale_weight": 0.05,
        "activation_pos_weight": "auto",
        "activation_pos_weight_scope": "channel",
        "activation_train_soft_gating": False,
    }


def build_config(
    base_cfg: dict[str, Any],
    cand: PoolCandidate,
    *,
    horizon: int,
    out_dir: Path,
    device: str,
    keep_train_residual_anchor: bool,
    train_residual_anchor_steps: int,
    enable_candidate_selector: bool,
    batch_size_override: int = 0,
    lazy_windows: bool = False,
    skip_test: bool = False,
    disable_gate_hit: bool = False,
    disable_explainability: bool = False,
    explainability_splits: tuple[str, ...] = ("val", "test"),
) -> dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    cfg.setdefault("exp", {})["name"] = f"electricity_input96_H{horizon}_pool_{cand.variant}"
    cfg["exp"]["device"] = str(device)
    localize_paths(cfg, out_dir)
    cfg.setdefault("window", {})["input_len"] = 96
    cfg["window"]["pred_len"] = int(horizon)
    cfg["window"]["past_context"] = True
    if bool(lazy_windows):
        cfg["window"]["lazy"] = True
    cfg.setdefault("normalize", {})["train_only"] = True
    cfg.setdefault("cluster", {})["train_only"] = True
    cfg.setdefault("eval", {})["skip_test"] = bool(skip_test)
    cfg.setdefault("calendar_residual", {})["enable"] = False
    cfg.setdefault("train", {})["epochs"] = 1
    if int(batch_size_override) > 0:
        cfg["train"]["batch_size"] = int(batch_size_override)
    cfg["train"]["selection_metric"] = "val_mse"
    if not bool((base_cfg.get("moe", {}) or {}).get("enable", False)):
        base_memory = base_cfg.get("memory", {}) or {}
        base_checkpoint = base_memory.get("checkpoint_path")
        if not base_checkpoint:
            base_checkpoint = str(Path(str(base_cfg.get("exp", {}).get("out_dir", ""))) / "best_checkpoint.pt")
        cfg["finetune"] = {
            "enable": True,
            "checkpoint_path": str(base_checkpoint),
            "strict_window": True,
            "strict_model": True,
            "cluster_map": "index",
            "load_model": True,
            "load_gate": False,
            "load_dynamic_lambda": False,
            "load_learnable_lambda": False,
        }

    penalties = tuple(cand.penalties)
    cfg.setdefault("penalties", {})["enabled"] = list(penalties)
    cfg["penalties"]["jump_threshold"] = float(cfg["penalties"].get("jump_threshold", 0.6))

    moe = cfg.setdefault("moe", {})
    moe["enable"] = True
    moe["freeze_backbone"] = True
    moe["topk"] = int(cand.gate_topk)
    moe["select_ranks"] = [int(v) for v in cand.select_ranks]
    moe["dynamic_lambda"] = {"enable": False}
    moe["learnable_lambda"] = {"enable": False}
    moe["lambda_init"] = lambda_dict(penalties, cand.lambda_scale)
    moe["lambda_min"] = lambda_dict(penalties, 0.0)
    moe["lambda_schedule"] = {name: "none" for name in penalties}
    moe["gate_entropy_weight"] = 0.0
    moe["gate_balance_weight"] = float(cand.gate_balance_weight)
    moe["gate_penalty_hit"] = {"enable": not bool(disable_gate_hit)}
    moe["mse_utility_gate_supervision"] = {
        "enable": bool(cand.mse_utility_gate_enable),
        "weight": float(cand.mse_utility_gate_weight),
        "temperature": float(cand.mse_utility_gate_temperature),
        "min_gain": float(cand.mse_utility_gate_min_gain),
        "target_power": float(cand.mse_utility_gate_target_power),
    }
    moe["gate_route_on_penalty_only"] = bool(cand.gate_route_on_penalty_only)
    moe["router_mode"] = str(cand.router_mode)
    moe["router_penalty_context_weight"] = float(cand.router_penalty_context_weight)
    moe["router_penalty_context_score"] = "high_violation"
    moe["router_detach_penalty_context"] = True
    moe["allow_skip"] = True
    moe["skip_cost"] = float(moe.get("skip_cost", 0.15))
    moe["skip_init_bias"] = float(moe.get("skip_init_bias", -2.0))
    moe["cluster_penalty_prior"] = {
        "enable": bool(cand.cluster_penalty_prior_enable),
        "topk": int(cand.cluster_penalty_prior_topk),
        "hard_topk": bool(cand.cluster_penalty_prior_hard_topk),
        "logit_strength": float(cand.cluster_penalty_prior_logit_strength),
        "temperature": float(cand.cluster_penalty_prior_temperature),
        "smoothing": float(cand.cluster_penalty_prior_smoothing),
        "use_normalized_penalty": True,
        "use_as_balance_target": bool(cand.cluster_penalty_prior_use_as_balance_target),
    }
    if cand.cluster_penalty_prior_always_include:
        moe["cluster_penalty_prior"]["always_include"] = list(cand.cluster_penalty_prior_always_include)
    if cand.cluster_allowed_by_cluster is not None:
        moe["cluster_penalty_prior"]["allowed_by_cluster"] = [
            list(row) for row in cand.cluster_allowed_by_cluster
        ]
    moe["channel_penalty_prior"] = {
        "enable": bool(cand.channel_penalty_prior_enable),
        "topk": int(cand.channel_penalty_prior_topk),
        "hard_topk": bool(cand.channel_penalty_prior_hard_topk),
        "temperature": float(cand.channel_penalty_prior_temperature),
        "smoothing": float(cand.channel_penalty_prior_smoothing),
        "use_normalized_penalty": True,
    }
    moe["explainability"] = {
        "enable": bool(cand.explainability_enable) and not bool(disable_explainability),
        "splits": [str(x) for x in explainability_splits],
        "max_batches": 0,
    }

    train_resid = moe.setdefault("train_residual_anchor_expert", {})
    train_resid["enable"] = bool(keep_train_residual_anchor)
    if keep_train_residual_anchor and train_residual_anchor_steps > 0:
        train_resid.setdefault("scale_selection", {})["steps"] = int(train_residual_anchor_steps)
    if keep_train_residual_anchor and cand.train_residual_anchor_max_scale is not None:
        train_resid.setdefault("scale_selection", {})["max_scale"] = float(cand.train_residual_anchor_max_scale)

    residual = {
        "enable": True,
        "feature_mode": "legacy",
        "selection_policy": str(cand.selection_policy),
        "selection_min_abs_improvement": 0.0,
        "selection_min_rel_improvement": 0.0,
        "residual_clip": 4.0,
        "corrector_hidden": int(cand.residual_corrector_hidden),
        "init_alpha": float(cand.residual_init_alpha),
        "alpha_scale": float(cand.residual_alpha_scale),
        "specialization_weight": float(cand.residual_specialization_weight),
        "norm_weight": float(cand.residual_norm_weight),
        "use_y_base_input": True,
        "gate_calibrator": gate_calibrator_cfg(cand),
    }
    if enable_candidate_selector:
        residual["candidate_selector"] = selector_cfg()
    if "seasonal_align" in penalties and cand.seasonal_anchor_period > 0 and cand.seasonal_anchor_scale != 0.0:
        residual.update(
            {
                "seasonal_anchor_names": ["seasonal_align"],
                "seasonal_anchor_period": int(cand.seasonal_anchor_period),
                "seasonal_anchor_num_periods": int(cand.seasonal_anchor_num_periods),
                "seasonal_anchor_scale": float(cand.seasonal_anchor_scale),
            }
        )
    moe["pred_side_residual"] = residual
    return cfg


def row_from_summary(
    *,
    horizon: int,
    cand: PoolCandidate,
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
    best = STRICT_BEST.get(int(horizon), {})

    def gain(metric: str, current: Any) -> str:
        try:
            base = float(best[metric])
            cur = float(current)
            return f"{(base - cur) / base * 100.0:.6f}"
        except Exception:
            return ""

    return {
        "status": "ok" if returncode == 0 and summary else ("failed" if returncode else "prepared"),
        "horizon": int(horizon),
        "variant": cand.variant,
        "penalties": ",".join(cand.penalties),
        "router_mode": cand.router_mode,
        "router_penalty_context_weight": cand.router_penalty_context_weight,
        "gate_topk": cand.gate_topk,
        "select_ranks": ",".join(str(v) for v in cand.select_ranks),
        "gate_balance_weight": cand.gate_balance_weight,
        "mse_utility_gate_enable": cand.mse_utility_gate_enable,
        "mse_utility_gate_weight": cand.mse_utility_gate_weight,
        "mse_utility_gate_temperature": cand.mse_utility_gate_temperature,
        "mse_utility_gate_min_gain": cand.mse_utility_gate_min_gain,
        "mse_utility_gate_target_power": cand.mse_utility_gate_target_power,
        "cluster_penalty_prior_enable": cand.cluster_penalty_prior_enable,
        "cluster_penalty_prior_topk": cand.cluster_penalty_prior_topk,
        "cluster_penalty_prior_hard_topk": cand.cluster_penalty_prior_hard_topk,
        "cluster_penalty_prior_logit_strength": cand.cluster_penalty_prior_logit_strength,
        "cluster_penalty_prior_always_include": ",".join(cand.cluster_penalty_prior_always_include),
        "channel_penalty_prior_enable": cand.channel_penalty_prior_enable,
        "channel_penalty_prior_topk": cand.channel_penalty_prior_topk,
        "seasonal_anchor_period": cand.seasonal_anchor_period,
        "seasonal_anchor_num_periods": cand.seasonal_anchor_num_periods,
        "seasonal_anchor_scale": cand.seasonal_anchor_scale,
        "train_residual_anchor_enable": (cfg.get("moe", {}).get("train_residual_anchor_expert", {}) or {}).get("enable", ""),
        "train_residual_anchor_steps": (
            (cfg.get("moe", {}).get("train_residual_anchor_expert", {}) or {})
            .get("scale_selection", {})
            .get("steps", "")
        ),
        "val_mse": val.get("avg_mse", ""),
        "val_mae": val.get("avg_mae", ""),
        "test_mse": test.get("avg_mse", ""),
        "test_mae": test.get("avg_mae", ""),
        "mse_gain_vs_strict_best_pct": gain("mse", test.get("avg_mse", "")),
        "mae_gain_vs_strict_best_pct": gain("mae", test.get("avg_mae", "")),
        "residual_mean_scale": residual_selection.get("mean_scale", ""),
        "residual_num_channels": residual_selection.get("num_residual_channels", ""),
        "best_epoch": ",".join(str(v) for v in summary.get("best_epoch", [])) if summary else "",
        "total_sec": f"{float(total_sec):.3f}",
        "config_path": str(config_path),
        "out_dir": str(out_dir),
        "returncode": int(returncode),
        "error": error,
    }


def run_one(
    *,
    horizon: int,
    base_cfg: dict[str, Any],
    cand: PoolCandidate,
    out_root: Path,
    device: str,
    keep_train_residual_anchor: bool,
    train_residual_anchor_steps: int,
    enable_candidate_selector: bool,
    reuse_existing: bool,
    batch_size_override: int,
    lazy_windows: bool,
    skip_test: bool,
    disable_gate_hit: bool,
    disable_explainability: bool,
    explainability_splits: tuple[str, ...],
) -> dict[str, Any]:
    out_dir = out_root / "runs" / f"H{horizon}" / cand.variant
    config_path = out_root / "configs" / f"H{horizon}" / f"{cand.variant}.yaml"
    cfg = build_config(
        base_cfg,
        cand,
        horizon=horizon,
        out_dir=out_dir,
        device=device,
        keep_train_residual_anchor=keep_train_residual_anchor,
        train_residual_anchor_steps=train_residual_anchor_steps,
        enable_candidate_selector=enable_candidate_selector,
        batch_size_override=batch_size_override,
        lazy_windows=lazy_windows,
        skip_test=skip_test,
        disable_gate_hit=disable_gate_hit,
        disable_explainability=disable_explainability,
        explainability_splits=explainability_splits,
    )
    write_yaml(config_path, cfg)
    if reuse_existing and (out_dir / "run_summary.json").exists():
        return row_from_summary(
            horizon=horizon,
            cand=cand,
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
    start = time.perf_counter()
    with stdout_path.open("w", encoding="utf-8") as stdout_f, stderr_path.open("w", encoding="utf-8") as stderr_f:
        completed = subprocess.run(cmd, cwd=str(ROOT), stdout=stdout_f, stderr=stderr_f, text=True)
    total_sec = time.perf_counter() - start
    error = ""
    if completed.returncode != 0:
        error = stderr_path.read_text(encoding="utf-8", errors="replace")[-3000:]
    return row_from_summary(
        horizon=horizon,
        cand=cand,
        cfg=cfg,
        config_path=config_path,
        out_dir=out_dir,
        returncode=int(completed.returncode),
        total_sec=total_sec,
        error=error,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Probe electricity penalty pools without changing PKR-MoE internals.")
    ap.add_argument("--horizon", type=int, default=336, choices=sorted(DEFAULT_BASE_CONFIGS))
    ap.add_argument("--base-config", default=None)
    ap.add_argument("--out-root", default="outputs/electricity_penalty_pool_probe_20260615")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--variants", nargs="*", default=None)
    ap.add_argument("--keep-train-residual-anchor", action="store_true")
    ap.add_argument("--train-residual-anchor-steps", type=int, default=101)
    ap.add_argument("--enable-candidate-selector", action="store_true")
    ap.add_argument("--reuse-existing", action="store_true")
    ap.add_argument("--batch-size-override", type=int, default=0)
    ap.add_argument("--lazy-windows", action="store_true")
    ap.add_argument("--skip-test", action="store_true")
    ap.add_argument("--disable-gate-hit", action="store_true")
    ap.add_argument("--disable-explainability", action="store_true")
    ap.add_argument("--explainability-splits", nargs="*", default=("val", "test"))
    ap.add_argument("--mse-utility-source-json", default=None)
    ap.add_argument("--mse-utility-source-split", default="val")
    ap.add_argument("--mse-utility-topk", type=int, default=2)
    ap.add_argument("--mse-utility-min-gain", type=float, default=0.0)
    ap.add_argument("--mse-utility-fallback", default="seasonal_align")
    ap.add_argument("--mse-utility-logit-strength", type=float, default=0.3)
    args = ap.parse_args()

    horizon = int(args.horizon)
    base_path = resolve(args.base_config) if args.base_config else DEFAULT_BASE_CONFIGS[horizon]
    if not base_path.exists():
        raise FileNotFoundError(f"Base config not found: {base_path}")
    base_cfg = load_yaml(base_path)
    out_root = resolve(args.out_root)
    variants = set(args.variants or [])
    candidates = [cand for cand in POOL_CANDIDATES if not variants or cand.variant in variants]
    if not candidates:
        raise ValueError(f"No variants selected from {[cand.variant for cand in POOL_CANDIDATES]}")
    if args.mse_utility_source_json:
        utility_allowed = build_mse_utility_allowed_by_cluster(
            resolve(args.mse_utility_source_json),
            split=str(args.mse_utility_source_split),
            topk=int(args.mse_utility_topk),
            min_gain=float(args.mse_utility_min_gain),
            fallback=str(args.mse_utility_fallback),
        )
        candidates = [
            apply_mse_utility_to_candidate(
                cand,
                utility_allowed,
                fallback=str(args.mse_utility_fallback),
                logit_strength=float(args.mse_utility_logit_strength),
            )
            for cand in candidates
        ]
        print(
            "MSE utility allowed_by_cluster enabled: "
            f"source={resolve(args.mse_utility_source_json)}, "
            f"split={args.mse_utility_source_split}, topk={args.mse_utility_topk}, "
            f"min_gain={args.mse_utility_min_gain}, fallback={args.mse_utility_fallback}, "
            f"allowed={utility_allowed}",
            flush=True,
        )

    rows: list[dict[str, Any]] = []
    results_path = out_root / "pool_results.csv"
    for cand in candidates:
        print(f"=== H{horizon} {cand.variant}: {','.join(cand.penalties)} ===", flush=True)
        row = run_one(
            horizon=horizon,
            base_cfg=base_cfg,
            cand=cand,
            out_root=out_root,
            device=str(args.device),
            keep_train_residual_anchor=bool(args.keep_train_residual_anchor),
            train_residual_anchor_steps=int(args.train_residual_anchor_steps),
            enable_candidate_selector=bool(args.enable_candidate_selector),
            reuse_existing=bool(args.reuse_existing),
            batch_size_override=int(args.batch_size_override),
            lazy_windows=bool(args.lazy_windows),
            skip_test=bool(args.skip_test),
            disable_gate_hit=bool(args.disable_gate_hit),
            disable_explainability=bool(args.disable_explainability),
            explainability_splits=tuple(str(x) for x in (args.explainability_splits or ("val", "test"))),
        )
        rows.append(row)
        write_rows(results_path, rows)
        print(json.dumps({k: row.get(k) for k in ("status", "variant", "val_mse", "test_mse", "test_mae")}, ensure_ascii=False), flush=True)

    ok_rows = [row for row in rows if row.get("status") == "ok" and row.get("test_mse") not in {"", None}]
    if ok_rows:
        best = sorted(ok_rows, key=lambda row: float(row["test_mse"]))[0]
        print(f"Best: {best['variant']} test_mse={best['test_mse']} test_mae={best['test_mae']}", flush=True)
    print(f"Wrote: {results_path}", flush=True)


if __name__ == "__main__":
    main()
