from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

try:
    import numpy as np
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, WhiteKernel
except Exception:  # pragma: no cover - runtime fallback for minimal envs
    np = None
    GaussianProcessRegressor = None
    Matern = None
    WhiteKernel = None

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_input96_h96_targeted_tuning import (  # noqa: E402
    Candidate as RunCandidate,
    DATASET_CONFIGS,
    load_yaml,
    read_summary,
    resolve,
    run_candidate,
    value,
    write_yaml,
)
from scripts.run_input96_moe_positive_search import apply_moe_training_controls  # noqa: E402


PENALTY_POOLS: dict[str, list[str]] = {
    "current": ["jump", "amp_under", "level", "delta"],
    "jump_level_delta": ["jump", "level", "delta"],
    "amp_level_delta": ["amp_under", "level", "delta"],
    "jump_amp_level": ["jump", "amp_under", "level"],
    "level_delta": ["level", "delta"],
    "jump_amp": ["jump", "amp_under"],
}

LAMBDA_PROFILES = ["flat", "amp_heavy", "jump_heavy", "level_heavy", "delta_heavy"]
ROUTER_MODES = ["learned", "penalty_context", "penalty_only"]
FEATURE_MODES = ["legacy", "safe_augmented"]

FIELDS = [
    "dataset",
    "pred_len",
    "phase",
    "trial",
    "variant",
    "status",
    "objective",
    "objective_source",
    "val_mse",
    "val_mae",
    "val_scaled_mse",
    "val_scaled_mae",
    "val_scaled_full_mse",
    "val_scaled_full_mae",
    "val_pred_base_mse",
    "val_residual_mse",
    "test_mse",
    "test_mae",
    "best_epoch",
    "penalty_pool",
    "penalties",
    "lambda_scale",
    "lambda_profile",
    "lambda_values",
    "alpha_scale",
    "residual_clip",
    "selection_scale_max",
    "selection_scale_steps",
    "selection_min_rel_improvement",
    "corrector_hidden",
    "init_alpha",
    "norm_weight",
    "lr",
    "weight_decay",
    "topk",
    "router_mode",
    "router_context_weight",
    "feature_mode",
    "gate_temperature",
    "gate_noise_std",
    "skip_cost",
    "selection_policy",
    "selection_holdout_fraction",
    "selection_holdout_min_windows",
    "selection_max_residual_channels",
    "selection_eval_segments",
    "selection_min_positive_segments",
    "selection_max_segment_rel_degradation",
    "selection_max_segment_abs_degradation",
    "residual_mean_scale",
    "residual_num_channels",
    "config_path",
    "out_dir",
    "total_sec",
    "returncode",
    "candidate_json",
    "error",
]


@dataclass(frozen=True)
class FrozenMoeCandidate:
    penalty_pool: str
    lambda_scale: float
    lambda_profile: str
    alpha_scale: float
    residual_clip: float
    selection_scale_max: float
    selection_scale_steps: int
    selection_min_rel_improvement: float
    corrector_hidden: int
    init_alpha: float
    norm_weight: float
    lr: float
    weight_decay: float
    topk: int
    router_mode: str
    router_context_weight: float
    feature_mode: str
    gate_temperature: float
    gate_noise_std: float
    skip_cost: float
    selection_policy: str = "val_mse_scale"
    selection_holdout_fraction: float = 0.0
    selection_holdout_min_windows: int = 256
    selection_max_residual_channels: int = 0
    selection_eval_segments: int = 1
    selection_min_positive_segments: int = 0
    selection_max_segment_rel_degradation: float = 0.0
    selection_max_segment_abs_degradation: float = 0.0

    def key(self) -> tuple[Any, ...]:
        return (
            self.penalty_pool,
            round(float(self.lambda_scale), 6),
            self.lambda_profile,
            round(float(self.alpha_scale), 4),
            round(float(self.residual_clip), 4),
            round(float(self.selection_scale_max), 4),
            int(self.selection_scale_steps),
            round(float(self.selection_min_rel_improvement), 6),
            int(self.corrector_hidden),
            round(float(self.init_alpha), 4),
            round(float(self.norm_weight), 8),
            round(float(self.lr), 8),
            round(float(self.weight_decay), 8),
            int(self.topk),
            self.router_mode,
            round(float(self.router_context_weight), 4),
            self.feature_mode,
            round(float(self.gate_temperature), 4),
            round(float(self.gate_noise_std), 4),
            round(float(self.skip_cost), 4),
            self.selection_policy,
            round(float(self.selection_holdout_fraction), 4),
            int(self.selection_holdout_min_windows),
            int(self.selection_max_residual_channels),
            int(self.selection_eval_segments),
            int(self.selection_min_positive_segments),
            round(float(self.selection_max_segment_rel_degradation), 6),
            round(float(self.selection_max_segment_abs_degradation), 8),
        )

    def short_id(self) -> str:
        parts = [
            _abbr(self.penalty_pool),
            f"l{_slug_float(self.lambda_scale)}",
            _abbr(self.lambda_profile),
            f"a{_slug_float(self.alpha_scale)}",
            f"c{_slug_float(self.residual_clip)}",
            f"s{_slug_float(self.selection_scale_max)}",
            f"n{int(self.selection_scale_steps)}",
            f"h{int(self.corrector_hidden)}",
            f"ia{_slug_float(self.init_alpha)}",
            f"lr{_slug_float(self.lr)}",
            f"wd{_slug_float(self.weight_decay)}",
            f"k{int(self.topk)}",
            _abbr(self.router_mode),
            _abbr(self.feature_mode),
        ]
        if self.selection_policy != "val_mse_scale":
            parts.append(_abbr(self.selection_policy))
        if int(self.selection_max_residual_channels) > 0:
            parts.append(f"mc{int(self.selection_max_residual_channels)}")
        if int(self.selection_eval_segments) > 1:
            parts.append(f"sg{int(self.selection_eval_segments)}p{int(self.selection_min_positive_segments)}")
        return "_".join(parts)


def _abbr(text: str) -> str:
    return "".join(part[:3] for part in text.split("_"))


def _slug_float(raw: float) -> str:
    value_text = f"{float(raw):.4g}"
    return value_text.replace(".", "p").replace("-", "m").replace("+", "")


def short_variant_name(trial: int, cand: FrozenMoeCandidate) -> str:
    digest = hashlib.sha1(candidate_to_json(cand).encode("utf-8")).hexdigest()[:10]
    slug = cand.short_id()
    if len(slug) > 48:
        slug = slug[:48].rstrip("_")
    return f"trial_{int(trial):03d}_{slug}_{digest}"


def safe_float(raw: Any, default: float = math.inf) -> float:
    try:
        if raw is None or raw == "":
            return default
        out = float(raw)
        if math.isnan(out):
            return default
        return out
    except (TypeError, ValueError):
        return default


def candidate_to_json(cand: FrozenMoeCandidate) -> str:
    return json.dumps(asdict(cand), sort_keys=True, separators=(",", ":"))


def candidate_from_json(raw: str) -> FrozenMoeCandidate:
    data = json.loads(raw)
    data.setdefault("selection_policy", "val_mse_scale")
    data.setdefault("selection_holdout_fraction", 0.0)
    data.setdefault("selection_holdout_min_windows", 256)
    data.setdefault("selection_max_residual_channels", 0)
    data.setdefault("selection_eval_segments", 1)
    data.setdefault("selection_min_positive_segments", 0)
    data.setdefault("selection_max_segment_rel_degradation", 0.0)
    data.setdefault("selection_max_segment_abs_degradation", 0.0)
    return FrozenMoeCandidate(**data)


def lambda_values(cand: FrozenMoeCandidate) -> dict[str, float]:
    penalties = PENALTY_POOLS[cand.penalty_pool]
    weights = {name: 1.0 for name in penalties}
    if cand.lambda_profile == "amp_heavy":
        for name in ("amp_under",):
            if name in weights:
                weights[name] = 2.0
        for name in ("level", "delta"):
            if name in weights:
                weights[name] = 0.7
    elif cand.lambda_profile == "jump_heavy":
        if "jump" in weights:
            weights["jump"] = 2.0
        if "level" in weights:
            weights["level"] = 0.7
    elif cand.lambda_profile == "level_heavy":
        if "level" in weights:
            weights["level"] = 2.0
        if "jump" in weights:
            weights["jump"] = 0.7
    elif cand.lambda_profile == "delta_heavy":
        if "delta" in weights:
            weights["delta"] = 2.0
        if "amp_under" in weights:
            weights["amp_under"] = 0.7
    return {name: float(cand.lambda_scale) * float(weights[name]) for name in penalties}


def candidate_to_patch(cand: FrozenMoeCandidate) -> dict[str, Any]:
    penalties = PENALTY_POOLS[cand.penalty_pool]
    topk = max(1, min(int(cand.topk), len(penalties)))
    router_context_weight = 0.0 if cand.router_mode == "learned" else float(cand.router_context_weight)
    residual = {
        "enable": True,
        "selection_policy": str(cand.selection_policy),
        "alpha_scale": float(cand.alpha_scale),
        "residual_clip": float(cand.residual_clip),
        "selection_min_abs_improvement": 0.0,
        "selection_min_rel_improvement": float(cand.selection_min_rel_improvement),
        "selection_scale_min": 0.0,
        "selection_scale_max": float(cand.selection_scale_max),
        "selection_scale_steps": int(cand.selection_scale_steps),
        "corrector_hidden": int(cand.corrector_hidden),
        "init_alpha": float(cand.init_alpha),
        "norm_weight": float(cand.norm_weight),
        "feature_mode": str(cand.feature_mode),
        "use_y_base_input": True,
        "diagnostics": {"enable": True},
    }
    if cand.selection_policy == "val_mse_scale_holdout":
        residual["selection_holdout_fraction"] = float(cand.selection_holdout_fraction)
        residual["selection_holdout_min_windows"] = int(cand.selection_holdout_min_windows)
    if int(cand.selection_max_residual_channels) > 0:
        residual["selection_max_residual_channels"] = int(cand.selection_max_residual_channels)
    if int(cand.selection_eval_segments) > 1:
        residual["selection_eval_segments"] = int(cand.selection_eval_segments)
        residual["selection_min_positive_segments"] = int(cand.selection_min_positive_segments)
        residual["selection_max_segment_rel_degradation"] = float(cand.selection_max_segment_rel_degradation)
        residual["selection_max_segment_abs_degradation"] = float(cand.selection_max_segment_abs_degradation)
    values = lambda_values(cand)
    return {
        "moe": {
            "enable": True,
            "topk": topk,
            "select_ranks": list(range(1, topk + 1)),
            "dynamic_lambda": {"enable": False},
            "learnable_lambda": {"enable": False},
            "pred_side_residual": residual,
            "lambda_init": values,
            "lambda_min": {name: 0.0 for name in penalties},
            "lambda_schedule": {name: "none" for name in penalties},
            "gate_entropy_weight": 0.0,
            "gate_balance_weight": 0.0,
            "gate_route_on_penalty_only": True,
            "router_mode": str(cand.router_mode),
            "router_penalty_context_weight": router_context_weight,
            "router_penalty_context_score": "high_violation",
            "router_detach_penalty_context": True,
            "allow_skip": True,
            "skip_cost": float(cand.skip_cost),
            "gate_temperature": float(cand.gate_temperature),
            "gate_noise_std": float(cand.gate_noise_std),
        },
        "penalties": {"enabled": list(penalties)},
        "train": {"lr": float(cand.lr), "weight_decay": float(cand.weight_decay)},
    }


def seed_candidates() -> list[FrozenMoeCandidate]:
    seeds = [
        FrozenMoeCandidate(
            "current",
            0.005,
            "flat",
            0.5,
            2.0,
            0.75,
            16,
            0.0005,
            32,
            -3.0,
            0.0,
            5.0e-4,
            1.0e-5,
            1,
            "learned",
            0.0,
            "legacy",
            1.0,
            0.2,
            0.15,
        ),
        FrozenMoeCandidate(
            "current",
            0.01,
            "flat",
            2.0,
            4.0,
            1.5,
            31,
            0.0005,
            64,
            -2.0,
            1.0e-5,
            5.0e-4,
            1.0e-5,
            1,
            "learned",
            0.0,
            "legacy",
            1.0,
            0.2,
            0.15,
        ),
        FrozenMoeCandidate(
            "current",
            0.0,
            "flat",
            1.0,
            4.0,
            1.25,
            26,
            0.0005,
            64,
            -2.5,
            1.0e-5,
            5.0e-4,
            1.0e-5,
            1,
            "learned",
            0.0,
            "legacy",
            1.0,
            0.2,
            0.15,
        ),
        FrozenMoeCandidate(
            "current",
            0.005,
            "flat",
            1.5,
            4.0,
            1.5,
            31,
            0.0005,
            64,
            -2.5,
            1.0e-5,
            5.0e-4,
            1.0e-5,
            2,
            "learned",
            0.0,
            "legacy",
            1.0,
            0.2,
            0.15,
        ),
        FrozenMoeCandidate(
            "current",
            0.005,
            "flat",
            1.5,
            4.0,
            1.5,
            31,
            0.0005,
            64,
            -2.5,
            1.0e-5,
            5.0e-4,
            1.0e-5,
            1,
            "penalty_context",
            1.0,
            "legacy",
            1.0,
            0.2,
            0.15,
        ),
        FrozenMoeCandidate(
            "jump_amp_level",
            0.005,
            "amp_heavy",
            1.5,
            4.0,
            1.5,
            31,
            0.0002,
            64,
            -2.3,
            1.0e-5,
            4.0e-4,
            1.0e-5,
            1,
            "penalty_only",
            1.0,
            "legacy",
            1.0,
            0.1,
            0.15,
        ),
    ]
    unique: dict[tuple[Any, ...], FrozenMoeCandidate] = {}
    for cand in seeds:
        unique[cand.key()] = cand
    return list(unique.values())


def with_selection_controls(
    cand: FrozenMoeCandidate,
    *,
    selection_policy: str,
    selection_holdout_fraction: float,
    selection_holdout_min_windows: int,
    selection_max_residual_channels: int = 0,
    selection_eval_segments: int = 1,
    selection_min_positive_segments: int = 0,
    selection_max_segment_rel_degradation: float = 0.0,
    selection_max_segment_abs_degradation: float = 0.0,
) -> FrozenMoeCandidate:
    return replace(
        cand,
        selection_policy=str(selection_policy),
        selection_holdout_fraction=float(selection_holdout_fraction),
        selection_holdout_min_windows=int(selection_holdout_min_windows),
        selection_max_residual_channels=int(selection_max_residual_channels),
        selection_eval_segments=int(selection_eval_segments),
        selection_min_positive_segments=int(selection_min_positive_segments),
        selection_max_segment_rel_degradation=float(selection_max_segment_rel_degradation),
        selection_max_segment_abs_degradation=float(selection_max_segment_abs_degradation),
    )


def sample_candidate(
    rng: random.Random,
    *,
    selection_policy: str = "val_mse_scale",
    selection_holdout_fraction: float = 0.0,
    selection_holdout_min_windows: int = 256,
    selection_max_residual_channels: int = 0,
    selection_eval_segments: int = 1,
    selection_min_positive_segments: int = 0,
    selection_max_segment_rel_degradation: float = 0.0,
    selection_max_segment_abs_degradation: float = 0.0,
) -> FrozenMoeCandidate:
    lambda_scale = 0.0 if rng.random() < 0.15 else 10 ** rng.uniform(math.log10(0.001), math.log10(0.03))
    router_mode = rng.choice(ROUTER_MODES)
    return FrozenMoeCandidate(
        penalty_pool=rng.choice(list(PENALTY_POOLS)),
        lambda_scale=float(lambda_scale),
        lambda_profile=rng.choice(LAMBDA_PROFILES),
        alpha_scale=rng.uniform(0.25, 3.0),
        residual_clip=rng.choice([2.0, 4.0, 6.0, 8.0]),
        selection_scale_max=rng.uniform(0.45, 2.5),
        selection_scale_steps=rng.choice([16, 21, 26, 31, 41]),
        selection_min_rel_improvement=rng.choice([0.0, 0.0002, 0.0005, 0.001]),
        corrector_hidden=rng.choice([32, 48, 64, 96, 128]),
        init_alpha=rng.uniform(-3.5, -1.0),
        norm_weight=rng.choice([0.0, 1.0e-6, 1.0e-5, 1.0e-4]),
        lr=10 ** rng.uniform(math.log10(1.0e-4), math.log10(1.0e-3)),
        weight_decay=10 ** rng.uniform(math.log10(1.0e-6), math.log10(1.0e-4)),
        topk=rng.choice([1, 1, 2]),
        router_mode=router_mode,
        router_context_weight=0.0 if router_mode == "learned" else rng.uniform(0.5, 2.0),
        feature_mode=rng.choice(["legacy", "legacy", "legacy", "safe_augmented"]),
        gate_temperature=rng.uniform(0.7, 1.5),
        gate_noise_std=rng.choice([0.0, 0.05, 0.1, 0.2]),
        skip_cost=rng.uniform(0.05, 0.3),
        selection_policy=str(selection_policy),
        selection_holdout_fraction=float(selection_holdout_fraction),
        selection_holdout_min_windows=int(selection_holdout_min_windows),
        selection_max_residual_channels=int(selection_max_residual_channels),
        selection_eval_segments=int(selection_eval_segments),
        selection_min_positive_segments=int(selection_min_positive_segments),
        selection_max_segment_rel_degradation=float(selection_max_segment_rel_degradation),
        selection_max_segment_abs_degradation=float(selection_max_segment_abs_degradation),
    )


def candidate_to_features(candidates: list[FrozenMoeCandidate]) -> Any:
    if np is None:
        raise RuntimeError("numpy is required for Bayesian candidate encoding.")
    cat_values = {
        "penalty_pool": list(PENALTY_POOLS),
        "lambda_profile": LAMBDA_PROFILES,
        "selection_scale_steps": [16, 21, 26, 31, 41],
        "corrector_hidden": [32, 48, 64, 96, 128],
        "topk": [1, 2],
        "router_mode": ROUTER_MODES,
        "feature_mode": FEATURE_MODES,
        "selection_policy": ["val_mse_scale", "val_mse_scale_holdout"],
    }
    rows: list[list[float]] = []
    for cand in candidates:
        row = [
            1.0 if cand.lambda_scale == 0.0 else 0.0,
            math.log10(max(float(cand.lambda_scale), 1.0e-5)),
            float(cand.alpha_scale) / 3.0,
            float(cand.residual_clip) / 8.0,
            float(cand.selection_scale_max) / 2.5,
            float(cand.selection_min_rel_improvement) * 1000.0,
            (float(cand.init_alpha) + 3.5) / 2.5,
            math.log10(max(float(cand.norm_weight), 1.0e-8)),
            math.log10(float(cand.lr)),
            math.log10(float(cand.weight_decay)),
            float(cand.router_context_weight) / 2.0,
            float(cand.gate_temperature) / 1.5,
            float(cand.gate_noise_std),
            float(cand.skip_cost),
            float(cand.selection_holdout_fraction),
            float(cand.selection_max_residual_channels) / 8.0,
            float(cand.selection_eval_segments) / 8.0,
            float(cand.selection_min_positive_segments) / 8.0,
            float(cand.selection_max_segment_rel_degradation),
            float(cand.selection_max_segment_abs_degradation),
        ]
        for attr, values in cat_values.items():
            current = getattr(cand, attr)
            row.extend(1.0 if current == value else 0.0 for value in values)
        rows.append(row)
    return np.asarray(rows, dtype=float)


def normal_pdf(x: Any) -> Any:
    return np.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def normal_cdf(x: Any) -> Any:
    return 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))


def choose_next_candidate(
    *,
    rng: random.Random,
    observed: list[tuple[FrozenMoeCandidate, float]],
    tried: set[tuple[Any, ...]],
    pool_size: int,
    selection_policy: str = "val_mse_scale",
    selection_holdout_fraction: float = 0.0,
    selection_holdout_min_windows: int = 256,
    selection_max_residual_channels: int = 0,
    selection_eval_segments: int = 1,
    selection_min_positive_segments: int = 0,
    selection_max_segment_rel_degradation: float = 0.0,
    selection_max_segment_abs_degradation: float = 0.0,
) -> FrozenMoeCandidate:
    if GaussianProcessRegressor is None or np is None or len(observed) < 4:
        for _ in range(max(1000, pool_size * 20)):
            cand = sample_candidate(
                rng,
                selection_policy=selection_policy,
                selection_holdout_fraction=selection_holdout_fraction,
                selection_holdout_min_windows=selection_holdout_min_windows,
                selection_max_residual_channels=selection_max_residual_channels,
                selection_eval_segments=selection_eval_segments,
                selection_min_positive_segments=selection_min_positive_segments,
                selection_max_segment_rel_degradation=selection_max_segment_rel_degradation,
                selection_max_segment_abs_degradation=selection_max_segment_abs_degradation,
            )
            if cand.key() not in tried:
                return cand
        return sample_candidate(
            rng,
            selection_policy=selection_policy,
            selection_holdout_fraction=selection_holdout_fraction,
            selection_holdout_min_windows=selection_holdout_min_windows,
            selection_max_residual_channels=selection_max_residual_channels,
            selection_eval_segments=selection_eval_segments,
            selection_min_positive_segments=selection_min_positive_segments,
            selection_max_segment_rel_degradation=selection_max_segment_rel_degradation,
            selection_max_segment_abs_degradation=selection_max_segment_abs_degradation,
        )

    unique_pool: list[FrozenMoeCandidate] = []
    seen = set(tried)
    attempts = 0
    while len(unique_pool) < int(pool_size) and attempts < int(pool_size) * 30:
        attempts += 1
        cand = sample_candidate(
            rng,
            selection_policy=selection_policy,
            selection_holdout_fraction=selection_holdout_fraction,
            selection_holdout_min_windows=selection_holdout_min_windows,
            selection_max_residual_channels=selection_max_residual_channels,
            selection_eval_segments=selection_eval_segments,
            selection_min_positive_segments=selection_min_positive_segments,
            selection_max_segment_rel_degradation=selection_max_segment_rel_degradation,
            selection_max_segment_abs_degradation=selection_max_segment_abs_degradation,
        )
        if cand.key() in seen:
            continue
        seen.add(cand.key())
        unique_pool.append(cand)
    if not unique_pool:
        return sample_candidate(
            rng,
            selection_policy=selection_policy,
            selection_holdout_fraction=selection_holdout_fraction,
            selection_holdout_min_windows=selection_holdout_min_windows,
            selection_max_residual_channels=selection_max_residual_channels,
            selection_eval_segments=selection_eval_segments,
            selection_min_positive_segments=selection_min_positive_segments,
            selection_max_segment_rel_degradation=selection_max_segment_rel_degradation,
            selection_max_segment_abs_degradation=selection_max_segment_abs_degradation,
        )

    y = np.asarray([score for _, score in observed], dtype=float)
    x_train = candidate_to_features([cand for cand, _ in observed])
    x_pool = candidate_to_features(unique_pool)
    try:
        kernel = Matern(nu=2.5) + WhiteKernel(noise_level=1.0e-6)
        gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=rng.randint(0, 2**31 - 1))
        gp.fit(x_train, y)
        mu, std = gp.predict(x_pool, return_std=True)
        best = float(np.min(y))
        std = np.maximum(std, 1.0e-9)
        improvement = best - mu
        z = improvement / std
        ei = improvement * normal_cdf(z) + std * normal_pdf(z)
        return unique_pool[int(np.argmax(ei))]
    except Exception:
        return rng.choice(unique_pool)


def objective_from_summary(
    summary: dict[str, Any],
    *,
    max_residual_channels: int | None = None,
    max_mean_scale: float | None = None,
    channel_penalty: float = 0.0,
    mean_scale_penalty: float = 0.0,
    residual_degradation_penalty: float = 0.0,
) -> tuple[float, str]:
    selection = summary.get("moe_residual_selection") or {}
    scaled_full = safe_float(selection.get("val_scaled_full_avg_mse"))
    scaled = safe_float(selection.get("val_scaled_avg_mse"))
    penalty = 0.0
    guarded = False
    if max_residual_channels is not None and float(channel_penalty) > 0.0:
        channels = safe_float(selection.get("num_residual_channels"), default=0.0)
        excess_channels = max(0.0, channels - float(max_residual_channels))
        if excess_channels > 0.0:
            guarded = True
            penalty += float(channel_penalty) * excess_channels
    if max_mean_scale is not None and float(mean_scale_penalty) > 0.0:
        mean_scale = safe_float(selection.get("mean_scale"), default=0.0)
        excess_scale = max(0.0, mean_scale - float(max_mean_scale))
        if excess_scale > 0.0:
            guarded = True
            penalty += float(mean_scale_penalty) * excess_scale
    degradation_guarded = False
    if float(residual_degradation_penalty) > 0.0:
        base_mse = safe_float(selection.get("val_pred_base_avg_mse"))
        residual_mse = safe_float(selection.get("val_residual_avg_mse"))
        if math.isfinite(base_mse) and math.isfinite(residual_mse):
            degradation = max(0.0, residual_mse - base_mse)
            if degradation > 0.0:
                degradation_guarded = True
                penalty += float(residual_degradation_penalty) * degradation
    if math.isfinite(scaled_full):
        source = "moe_residual_selection.val_scaled_full_avg_mse"
        if guarded:
            source += "+complexity_guard"
        if degradation_guarded:
            source += "+residual_degradation_guard"
        return scaled_full + penalty, source
    if math.isfinite(scaled):
        source = "moe_residual_selection.val_scaled_avg_mse"
        if guarded:
            source += "+complexity_guard"
        if degradation_guarded:
            source += "+residual_degradation_guard"
        return scaled + penalty, source
    val = summary.get("val") or {}
    raw = safe_float(val.get("avg_mse"))
    if math.isfinite(raw):
        return raw, "val.avg_mse"
    return math.inf, "missing"


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


def observed_from_rows(rows: list[dict[str, Any]]) -> list[tuple[FrozenMoeCandidate, float]]:
    observed: list[tuple[FrozenMoeCandidate, float]] = []
    for row in rows:
        if row.get("status") != "ok":
            continue
        objective = safe_float(row.get("objective"))
        if not math.isfinite(objective):
            continue
        try:
            observed.append((candidate_from_json(str(row["candidate_json"])), objective))
        except Exception:
            continue
    return observed


def tried_from_rows(rows: list[dict[str, Any]]) -> set[tuple[Any, ...]]:
    tried: set[tuple[Any, ...]] = set()
    for row in rows:
        try:
            tried.add(candidate_from_json(str(row["candidate_json"])).key())
        except Exception:
            continue
    return tried


def best_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    ok = [row for row in rows if row.get("status") == "ok" and math.isfinite(safe_float(row.get("objective")))]
    if not ok:
        return None
    return sorted(ok, key=lambda row: (safe_float(row.get("objective")), value(row, "val_mae")))[0]


def row_updates(
    *,
    row: dict[str, Any],
    cand: FrozenMoeCandidate,
    trial: int,
    phase: str,
    max_residual_channels: int | None = None,
    max_mean_scale: float | None = None,
    channel_penalty: float = 0.0,
    mean_scale_penalty: float = 0.0,
    residual_degradation_penalty: float = 0.0,
) -> dict[str, Any]:
    out_dir = Path(str(row.get("out_dir", "")))
    summary = read_summary(out_dir / "run_summary.json")
    objective, source = objective_from_summary(
        summary,
        max_residual_channels=max_residual_channels,
        max_mean_scale=max_mean_scale,
        channel_penalty=channel_penalty,
        mean_scale_penalty=mean_scale_penalty,
        residual_degradation_penalty=residual_degradation_penalty,
    )
    selection = summary.get("moe_residual_selection") or {}
    values = lambda_values(cand)
    out = dict(row)
    out.update(
        {
            "phase": phase,
            "trial": int(trial),
            "objective": "" if not math.isfinite(objective) else objective,
            "objective_source": source,
            "val_scaled_mse": selection.get("val_scaled_avg_mse", ""),
            "val_scaled_mae": selection.get("val_scaled_avg_mae", ""),
            "val_scaled_full_mse": selection.get("val_scaled_full_avg_mse", ""),
            "val_scaled_full_mae": selection.get("val_scaled_full_avg_mae", ""),
            "val_pred_base_mse": selection.get("val_pred_base_avg_mse", ""),
            "val_residual_mse": selection.get("val_residual_avg_mse", ""),
            "penalty_pool": cand.penalty_pool,
            "penalties": ",".join(PENALTY_POOLS[cand.penalty_pool]),
            "lambda_scale": cand.lambda_scale,
            "lambda_profile": cand.lambda_profile,
            "lambda_values": json.dumps(values, sort_keys=True),
            "alpha_scale": cand.alpha_scale,
            "residual_clip": cand.residual_clip,
            "selection_scale_max": cand.selection_scale_max,
            "selection_scale_steps": cand.selection_scale_steps,
            "selection_min_rel_improvement": cand.selection_min_rel_improvement,
            "corrector_hidden": cand.corrector_hidden,
            "init_alpha": cand.init_alpha,
            "norm_weight": cand.norm_weight,
            "lr": cand.lr,
            "weight_decay": cand.weight_decay,
            "topk": cand.topk,
            "router_mode": cand.router_mode,
            "router_context_weight": cand.router_context_weight,
            "feature_mode": cand.feature_mode,
            "gate_temperature": cand.gate_temperature,
            "gate_noise_std": cand.gate_noise_std,
            "skip_cost": cand.skip_cost,
            "selection_policy": cand.selection_policy,
            "selection_holdout_fraction": cand.selection_holdout_fraction,
            "selection_holdout_min_windows": cand.selection_holdout_min_windows,
            "selection_max_residual_channels": cand.selection_max_residual_channels,
            "selection_eval_segments": cand.selection_eval_segments,
            "selection_min_positive_segments": cand.selection_min_positive_segments,
            "selection_max_segment_rel_degradation": cand.selection_max_segment_rel_degradation,
            "selection_max_segment_abs_degradation": cand.selection_max_segment_abs_degradation,
            "candidate_json": candidate_to_json(cand),
        }
    )
    return out


def write_best_outputs(out_root: Path, rows: list[dict[str, Any]], dataset: str, horizon: int) -> None:
    best = best_row(rows)
    if best is None:
        return
    write_rows(out_root / "best_by_val.csv", [best])
    config_path = Path(str(best["config_path"]))
    if config_path.exists():
        best_cfg = load_yaml(config_path)
        best_cfg.setdefault("eval", {})["skip_test"] = False
        write_yaml(out_root / "best_configs" / dataset / f"H{int(horizon)}.yaml", best_cfg)


def select_seed_candidate(
    tried: set[tuple[Any, ...]],
    *,
    selection_policy: str = "val_mse_scale",
    selection_holdout_fraction: float = 0.0,
    selection_holdout_min_windows: int = 256,
    selection_max_residual_channels: int = 0,
    selection_eval_segments: int = 1,
    selection_min_positive_segments: int = 0,
    selection_max_segment_rel_degradation: float = 0.0,
    selection_max_segment_abs_degradation: float = 0.0,
) -> FrozenMoeCandidate | None:
    for cand in seed_candidates():
        cand = with_selection_controls(
            cand,
            selection_policy=selection_policy,
            selection_holdout_fraction=selection_holdout_fraction,
            selection_holdout_min_windows=selection_holdout_min_windows,
            selection_max_residual_channels=selection_max_residual_channels,
            selection_eval_segments=selection_eval_segments,
            selection_min_positive_segments=selection_min_positive_segments,
            selection_max_segment_rel_degradation=selection_max_segment_rel_degradation,
            selection_max_segment_abs_degradation=selection_max_segment_abs_degradation,
        )
        if cand.key() not in tried:
            return cand
    return None


def run_one_candidate(
    *,
    dataset: str,
    horizon: int,
    base_cfg: dict[str, Any],
    out_root: Path,
    device: str | None,
    epochs: int,
    trial: int,
    cand: FrozenMoeCandidate,
    dry_run: bool,
    skip_test: bool,
    phase: str,
    variant_override: str | None = None,
    max_residual_channels: int | None = None,
    max_mean_scale: float | None = None,
    channel_penalty: float = 0.0,
    mean_scale_penalty: float = 0.0,
    residual_degradation_penalty: float = 0.0,
) -> dict[str, Any]:
    variant = str(variant_override) if variant_override else short_variant_name(trial, cand)
    run_cand = RunCandidate(phase, variant, candidate_to_patch(cand))
    row, _cfg = run_candidate(
        dataset=dataset,
        pred_len=int(horizon),
        base_cfg=base_cfg,
        cand=run_cand,
        out_root=out_root,
        device=device,
        epochs=int(epochs),
        skip_test=bool(skip_test),
        dry_run=bool(dry_run),
    )
    return row_updates(
        row=row,
        cand=cand,
        trial=trial,
        phase=phase,
        max_residual_channels=max_residual_channels,
        max_mean_scale=max_mean_scale,
        channel_penalty=channel_penalty,
        mean_scale_penalty=mean_scale_penalty,
        residual_degradation_penalty=residual_degradation_penalty,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Bayesian search for frozen-backbone input-96 MoE residual parameters.")
    ap.add_argument("--base-config", required=True)
    ap.add_argument("--dataset", default="ETTh2", choices=list(DATASET_CONFIGS.keys()))
    ap.add_argument("--horizon", type=int, default=96)
    ap.add_argument("--out-root", default="outputs/input96_frozen_moe_bayes_search")
    ap.add_argument("--device", default=None)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--trials", type=int, default=12)
    ap.add_argument("--init-trials", type=int, default=6)
    ap.add_argument("--candidate-pool-size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--selection-policy", choices=["val_mse_scale", "val_mse_scale_holdout"], default="val_mse_scale")
    ap.add_argument("--selection-holdout-fraction", type=float, default=0.4)
    ap.add_argument("--selection-holdout-min-windows", type=int, default=256)
    ap.add_argument("--selection-max-residual-channels", type=int, default=0)
    ap.add_argument("--selection-eval-segments", type=int, default=1)
    ap.add_argument("--selection-min-positive-segments", type=int, default=0)
    ap.add_argument("--selection-max-segment-rel-degradation", type=float, default=0.0)
    ap.add_argument("--selection-max-segment-abs-degradation", type=float, default=0.0)
    ap.add_argument("--objective-max-residual-channels", type=int, default=None)
    ap.add_argument("--objective-max-mean-scale", type=float, default=None)
    ap.add_argument("--objective-channel-penalty", type=float, default=0.0)
    ap.add_argument("--objective-mean-scale-penalty", type=float, default=0.0)
    ap.add_argument("--objective-residual-degradation-penalty", type=float, default=0.0)
    ap.add_argument("--warm-start-checkpoint", default=None)
    ap.add_argument("--freeze-backbone", dest="freeze_backbone", action="store_true", default=True)
    ap.add_argument("--no-freeze-backbone", dest="freeze_backbone", action="store_false")
    ap.add_argument("--run-final-test", action="store_true")
    ap.add_argument("--final-epochs", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    rng = random.Random(int(args.seed))
    out_root = resolve(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    result_path = out_root / "bayes_results.csv"

    base_cfg = load_yaml(resolve(args.base_config))
    apply_moe_training_controls(
        base_cfg,
        warm_start_checkpoint=args.warm_start_checkpoint,
        freeze_backbone=bool(args.freeze_backbone),
        lr=None,
        weight_decay=None,
    )

    rows = read_rows(result_path)
    observed = observed_from_rows(rows)
    tried = tried_from_rows(rows)
    print(
        f"=== Frozen MoE Bayes {args.dataset} H{int(args.horizon)}: "
        f"{len(rows)}/{int(args.trials)} rows, {len(observed)} observed ===",
        flush=True,
    )

    while len(rows) < int(args.trials):
        trial = len(rows)
        selection_policy = str(args.selection_policy)
        selection_holdout_fraction = float(args.selection_holdout_fraction if selection_policy == "val_mse_scale_holdout" else 0.0)
        selection_holdout_min_windows = int(args.selection_holdout_min_windows)
        selection_max_residual_channels = int(args.selection_max_residual_channels)
        selection_eval_segments = int(args.selection_eval_segments)
        selection_min_positive_segments = int(args.selection_min_positive_segments)
        selection_max_segment_rel_degradation = float(args.selection_max_segment_rel_degradation)
        selection_max_segment_abs_degradation = float(args.selection_max_segment_abs_degradation)
        cand = select_seed_candidate(
            tried,
            selection_policy=selection_policy,
            selection_holdout_fraction=selection_holdout_fraction,
            selection_holdout_min_windows=selection_holdout_min_windows,
            selection_max_residual_channels=selection_max_residual_channels,
            selection_eval_segments=selection_eval_segments,
            selection_min_positive_segments=selection_min_positive_segments,
            selection_max_segment_rel_degradation=selection_max_segment_rel_degradation,
            selection_max_segment_abs_degradation=selection_max_segment_abs_degradation,
        )
        if cand is None:
            if len(observed) < int(args.init_trials):
                for _ in range(1000):
                    cand = sample_candidate(
                        rng,
                        selection_policy=selection_policy,
                        selection_holdout_fraction=selection_holdout_fraction,
                        selection_holdout_min_windows=selection_holdout_min_windows,
                        selection_max_residual_channels=selection_max_residual_channels,
                        selection_eval_segments=selection_eval_segments,
                        selection_min_positive_segments=selection_min_positive_segments,
                        selection_max_segment_rel_degradation=selection_max_segment_rel_degradation,
                        selection_max_segment_abs_degradation=selection_max_segment_abs_degradation,
                    )
                    if cand.key() not in tried:
                        break
            else:
                cand = choose_next_candidate(
                    rng=rng,
                    observed=observed,
                    tried=tried,
                    pool_size=int(args.candidate_pool_size),
                    selection_policy=selection_policy,
                    selection_holdout_fraction=selection_holdout_fraction,
                    selection_holdout_min_windows=selection_holdout_min_windows,
                    selection_max_residual_channels=selection_max_residual_channels,
                    selection_eval_segments=selection_eval_segments,
                    selection_min_positive_segments=selection_min_positive_segments,
                    selection_max_segment_rel_degradation=selection_max_segment_rel_degradation,
                    selection_max_segment_abs_degradation=selection_max_segment_abs_degradation,
                )
        tried.add(cand.key())
        print(f"[trial {trial:03d}] {cand.short_id()}", flush=True)
        row = run_one_candidate(
            dataset=str(args.dataset),
            horizon=int(args.horizon),
            base_cfg=base_cfg,
            out_root=out_root,
            device=args.device,
            epochs=int(args.epochs),
            trial=trial,
            cand=cand,
            dry_run=bool(args.dry_run),
            skip_test=True,
            phase="bayes_moe",
            max_residual_channels=args.objective_max_residual_channels,
            max_mean_scale=args.objective_max_mean_scale,
            channel_penalty=float(args.objective_channel_penalty),
            mean_scale_penalty=float(args.objective_mean_scale_penalty),
            residual_degradation_penalty=float(args.objective_residual_degradation_penalty),
        )
        rows.append(row)
        objective = safe_float(row.get("objective"))
        if row.get("status") == "ok" and math.isfinite(objective):
            observed.append((cand, objective))
        write_rows(result_path, rows)
        write_best_outputs(out_root, rows, str(args.dataset), int(args.horizon))
        best = best_row(rows)
        best_text = "" if best is None else f" best={best.get('objective')} ({best.get('variant')})"
        print(
            f"  -> {row.get('status')} obj={row.get('objective')} "
            f"raw_val={row.get('val_mse')} scaled_val={row.get('val_scaled_mse')}{best_text}",
            flush=True,
        )

    if args.run_final_test:
        best = best_row(rows)
        if best is None:
            raise SystemExit("No successful search row for final test.")
        cand = candidate_from_json(str(best["candidate_json"]))
        best_cfg = load_yaml(Path(str(best["config_path"])))
        final_epochs = int(args.final_epochs if args.final_epochs is not None else args.epochs)
        final_row = run_one_candidate(
            dataset=str(args.dataset),
            horizon=int(args.horizon),
            base_cfg=best_cfg,
            out_root=out_root,
            device=args.device,
            epochs=final_epochs,
            trial=0,
            cand=cand,
            dry_run=bool(args.dry_run),
            skip_test=False,
            phase="bayes_final",
            max_residual_channels=args.objective_max_residual_channels,
            max_mean_scale=args.objective_max_mean_scale,
            channel_penalty=float(args.objective_channel_penalty),
            mean_scale_penalty=float(args.objective_mean_scale_penalty),
            residual_degradation_penalty=float(args.objective_residual_degradation_penalty),
        )
        write_rows(out_root / "final_results.csv", [final_row])
        print(
            f"=== Final {final_row.get('status')} test_mse={final_row.get('test_mse')} "
            f"test_mae={final_row.get('test_mae')} ===",
            flush=True,
        )


if __name__ == "__main__":
    main()
