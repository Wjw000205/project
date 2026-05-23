from __future__ import annotations

import argparse
import csv
import copy
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=False), encoding="utf-8")


def set_paths(cfg: dict[str, Any], out_dir: Path, name: str) -> None:
    cfg.setdefault("exp", {})
    cfg["exp"]["name"] = name
    cfg["exp"]["out_dir"] = str(out_dir)
    cfg.setdefault("corr", {})
    cfg["corr"]["save_path"] = str(out_dir / "corr.npy")
    cfg.setdefault("plot", {})
    cfg["plot"]["enable"] = False
    cfg.setdefault("portrait", {})
    cfg["portrait"]["enable"] = False
    cfg["portrait"]["out_dir"] = str(out_dir / "cluster_portraits")
    cfg.setdefault("memory", {})
    cfg["memory"]["enable"] = False
    cfg["memory"]["save_checkpoint"] = False
    cfg["memory"]["path"] = str(out_dir / "cluster_memory.pt")
    cfg["memory"]["checkpoint_path"] = str(out_dir / "best_checkpoint.pt")
    cfg.setdefault("knn_hybrid", {})
    cfg["knn_hybrid"]["enable"] = False
    cfg["knn_hybrid"]["path"] = str(out_dir / "knn_shape_bank.pt")


def deep_update(dst: dict[str, Any], src: dict[str, Any]) -> None:
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            deep_update(dst[key], value)
        else:
            dst[key] = copy.deepcopy(value)


def adaptive_residual_overrides(fusion_init: float = -1.5) -> dict[str, Any]:
    return {
        "penalty_selector_enable": True,
        "selector_temperature": 1.0,
        "selector_use_cluster_context": True,
        "fusion_gate_enable": True,
        "fusion_init": fusion_init,
        "fusion_use_cluster_context": True,
    }


def channel_expert_residual_overrides(
    fusion_init: float = -1.5,
    mode_type: str = "override",
) -> dict[str, Any]:
    cfg = adaptive_residual_overrides(fusion_init=fusion_init)
    cfg["channel_expert_adapters"] = {
        "enable": True,
        "mode": "merged_singletons",
        "mode_type": mode_type,
    }
    return cfg


def cluster_prior_overrides(topk: int, select_ranks: list[int], logit_strength: float = 0.0) -> dict[str, Any]:
    return {
        "select_ranks": select_ranks,
        "explainability": {
            "enable": True,
            "splits": ["train", "val", "test"],
            "max_batches": 0,
        },
        "cluster_penalty_prior": {
            "enable": True,
            "topk": topk,
            "hard_topk": True,
            "temperature": 0.7,
            "smoothing": 0.0,
            "use_normalized_penalty": True,
            "logit_strength": logit_strength,
            "use_as_balance_target": False,
        },
    }


def channel_prior_overrides(topk: int, select_ranks: list[int]) -> dict[str, Any]:
    cfg = cluster_prior_overrides(topk=topk, select_ranks=select_ranks)
    cfg["channel_penalty_prior"] = {
        "enable": True,
        "topk": topk,
        "hard_topk": True,
        "temperature": 0.7,
        "smoothing": 0.0,
        "use_normalized_penalty": True,
    }
    return cfg


def rnn_safe_moe(overrides: dict[str, Any]) -> dict[str, Any]:
    cfg = copy.deepcopy(overrides)
    cfg["dynamic_lambda"] = {"enable": False}
    cfg["learnable_lambda"] = {"enable": False, "bilevel": {"enable": False}}
    return cfg


def residual_overrides_with(**kwargs: Any) -> dict[str, Any]:
    cfg = adaptive_residual_overrides()
    deep_update(cfg, kwargs)
    return cfg


def candidates() -> list[dict[str, Any]]:
    return [
        {
            "name": "mlp_current_cluster",
            "predictor": "mlp",
            "cluster": {},
            "model": {},
            "pred_residual": {},
        },
        {
            "name": "mlp_current_cluster_bs32",
            "predictor": "mlp",
            "cluster": {},
            "model": {},
            "train": {"batch_size": 32},
            "pred_residual": {},
        },
        {
            "name": "mlp_current_cluster_bs64",
            "predictor": "mlp",
            "cluster": {},
            "model": {},
            "train": {"batch_size": 64},
            "pred_residual": {},
        },
        {
            "name": "mlp_weak_residual_gate",
            "predictor": "mlp",
            "cluster": {},
            "model": {},
            "pred_residual": {
                "alpha_scale": 0.8,
                "residual_clip": 2.0,
                "selection_policy": "val_mse_gate_guarded",
                "selection_min_rel_improvement": 0.0005,
            },
        },
        {
            "name": "mlp_weak_residual_gate_bs64",
            "predictor": "mlp",
            "cluster": {},
            "model": {},
            "train": {"batch_size": 64},
            "pred_residual": {
                "alpha_scale": 0.8,
                "residual_clip": 2.0,
                "selection_policy": "val_mse_gate_guarded",
                "selection_min_rel_improvement": 0.0005,
            },
        },
        {
            "name": "mlp_weak_residual_gate_bs32",
            "predictor": "mlp",
            "cluster": {},
            "model": {},
            "train": {"batch_size": 32},
            "pred_residual": {
                "alpha_scale": 0.8,
                "residual_clip": 2.0,
                "selection_policy": "val_mse_gate_guarded",
                "selection_min_rel_improvement": 0.0005,
            },
        },
        {
            "name": "channel_head_current_cluster",
            "predictor": "channel_head_mlp",
            "cluster": {},
            "model": {},
            "pred_residual": {},
        },
        {
            "name": "context_channel_head_current_cluster",
            "predictor": "context_channel_head_mlp",
            "cluster": {},
            "model": {},
            "pred_residual": {},
        },
        {
            "name": "context_channel_head_selector_fusion",
            "predictor": "context_channel_head_mlp",
            "cluster": {},
            "model": {},
            "moe": {"explainability": {"enable": True, "splits": ["train", "val", "test"], "max_batches": 0}},
            "pred_residual": adaptive_residual_overrides(),
        },
        {
            "name": "context_channel_head_prior_top1_fusion",
            "predictor": "context_channel_head_mlp",
            "cluster": {},
            "model": {},
            "moe": cluster_prior_overrides(topk=1, select_ranks=[1]),
            "pred_residual": adaptive_residual_overrides(),
        },
        {
            "name": "context_channel_head_prior_top2_select1_fusion",
            "predictor": "context_channel_head_mlp",
            "cluster": {},
            "model": {},
            "moe": cluster_prior_overrides(topk=2, select_ranks=[1]),
            "pred_residual": adaptive_residual_overrides(),
        },
        {
            "name": "context_channel_head_prior_top2_select12_fusion",
            "predictor": "context_channel_head_mlp",
            "cluster": {},
            "model": {},
            "moe": cluster_prior_overrides(topk=2, select_ranks=[1, 2]),
            "pred_residual": adaptive_residual_overrides(),
        },
        {
            "name": "channel_head_selector_fusion",
            "predictor": "channel_head_mlp",
            "cluster": {},
            "model": {},
            "moe": {"explainability": {"enable": True, "splits": ["train", "val", "test"], "max_batches": 0}},
            "pred_residual": adaptive_residual_overrides(),
        },
        {
            "name": "channel_head_prior_top1_fusion",
            "predictor": "channel_head_mlp",
            "cluster": {},
            "model": {},
            "moe": cluster_prior_overrides(topk=1, select_ranks=[1]),
            "pred_residual": adaptive_residual_overrides(),
        },
        {
            "name": "channel_head_prior_top2_select12_fusion",
            "predictor": "channel_head_mlp",
            "cluster": {},
            "model": {},
            "moe": cluster_prior_overrides(topk=2, select_ranks=[1, 2]),
            "pred_residual": adaptive_residual_overrides(),
        },
        {
            "name": "channel_head_feature_w05_prior_top2_select12_fusion",
            "predictor": "channel_head_mlp",
            "cluster": {
                "feature_aware": {
                    "enable": True,
                    "weight": 0.5,
                    "acf_lags": [1, 24, 96],
                }
            },
            "model": {},
            "moe": cluster_prior_overrides(topk=2, select_ranks=[1, 2]),
            "pred_residual": adaptive_residual_overrides(),
        },
        {
            "name": "channel_head_feature_w10_keep_singletons_prior_top2_select12_fusion",
            "predictor": "channel_head_mlp",
            "cluster": {
                "feature_aware": {
                    "enable": True,
                    "weight": 1.0,
                    "acf_lags": [1, 24, 96],
                },
                "singleton_merge_strategy": "keep",
            },
            "model": {},
            "moe": cluster_prior_overrides(topk=2, select_ranks=[1, 2]),
            "pred_residual": adaptive_residual_overrides(),
        },
        {
            "name": "channel_head_feature_w10_guarded_t130_prior_top2_select12_fusion",
            "predictor": "channel_head_mlp",
            "cluster": {
                "feature_aware": {
                    "enable": True,
                    "weight": 1.0,
                    "acf_lags": [1, 24, 96],
                },
                "singleton_merge_strategy": "guarded_pool",
                "singleton_merge_distance_threshold": 1.30,
            },
            "model": {},
            "moe": cluster_prior_overrides(topk=2, select_ranks=[1, 2]),
            "pred_residual": adaptive_residual_overrides(),
        },
        {
            "name": "channel_head_feature_w10_guarded_t150_prior_top2_select12_fusion",
            "predictor": "channel_head_mlp",
            "cluster": {
                "feature_aware": {
                    "enable": True,
                    "weight": 1.0,
                    "acf_lags": [1, 24, 96],
                },
                "singleton_merge_strategy": "guarded_pool",
                "singleton_merge_distance_threshold": 1.50,
            },
            "model": {},
            "moe": cluster_prior_overrides(topk=2, select_ranks=[1, 2]),
            "pred_residual": adaptive_residual_overrides(),
        },
        {
            "name": "channel_head_feature_w10_guarded_t155_prior_top2_select12_fusion",
            "predictor": "channel_head_mlp",
            "cluster": {
                "feature_aware": {
                    "enable": True,
                    "weight": 1.0,
                    "acf_lags": [1, 24, 96],
                },
                "singleton_merge_strategy": "guarded_pool",
                "singleton_merge_distance_threshold": 1.55,
            },
            "model": {},
            "moe": cluster_prior_overrides(topk=2, select_ranks=[1, 2]),
            "pred_residual": adaptive_residual_overrides(),
        },
        {
            "name": "channel_head_feature_w10_guarded_t130_hd128_prior_top2_select12_fusion",
            "predictor": "channel_head_mlp",
            "cluster": {
                "feature_aware": {
                    "enable": True,
                    "weight": 1.0,
                    "acf_lags": [1, 24, 96],
                },
                "singleton_merge_strategy": "guarded_pool",
                "singleton_merge_distance_threshold": 1.30,
            },
            "model": {"hidden_dim": 128},
            "moe": cluster_prior_overrides(topk=2, select_ranks=[1, 2]),
            "pred_residual": adaptive_residual_overrides(),
        },
        {
            "name": "channel_head_feature_w10_guarded_t155_hd128_prior_top2_select12_fusion",
            "predictor": "channel_head_mlp",
            "cluster": {
                "feature_aware": {
                    "enable": True,
                    "weight": 1.0,
                    "acf_lags": [1, 24, 96],
                },
                "singleton_merge_strategy": "guarded_pool",
                "singleton_merge_distance_threshold": 1.55,
            },
            "model": {"hidden_dim": 128},
            "moe": cluster_prior_overrides(topk=2, select_ranks=[1, 2]),
            "pred_residual": adaptive_residual_overrides(),
        },
        {
            "name": "channel_head_feature_w10_prior_top2_select12_fusion",
            "predictor": "channel_head_mlp",
            "cluster": {
                "feature_aware": {
                    "enable": True,
                    "weight": 1.0,
                    "acf_lags": [1, 24, 96],
                },
                "singleton_merge_strategy": "pool",
            },
            "model": {},
            "moe": cluster_prior_overrides(topk=2, select_ranks=[1, 2]),
            "pred_residual": adaptive_residual_overrides(),
        },
        {
            "name": "channel_head_feature_w10_channel_prior_top2_select12_fusion",
            "predictor": "channel_head_mlp",
            "cluster": {
                "feature_aware": {
                    "enable": True,
                    "weight": 1.0,
                    "acf_lags": [1, 24, 96],
                },
                "singleton_merge_strategy": "pool",
            },
            "model": {},
            "moe": channel_prior_overrides(topk=2, select_ranks=[1, 2]),
            "pred_residual": adaptive_residual_overrides(),
        },
        {
            "name": "channel_head_feature_w10_channel_prior_top1_select1_fusion",
            "predictor": "channel_head_mlp",
            "cluster": {
                "feature_aware": {
                    "enable": True,
                    "weight": 1.0,
                    "acf_lags": [1, 24, 96],
                },
                "singleton_merge_strategy": "pool",
            },
            "model": {},
            "moe": channel_prior_overrides(topk=1, select_ranks=[1]),
            "pred_residual": adaptive_residual_overrides(),
        },
        {
            "name": "channel_head_feature_w10_channel_prior_top1_select1_fusion_bs32",
            "predictor": "channel_head_mlp",
            "cluster": {
                "feature_aware": {
                    "enable": True,
                    "weight": 1.0,
                    "acf_lags": [1, 24, 96],
                },
                "singleton_merge_strategy": "pool",
            },
            "model": {},
            "train": {"batch_size": 32},
            "moe": channel_prior_overrides(topk=1, select_ranks=[1]),
            "pred_residual": adaptive_residual_overrides(),
        },
        {
            "name": "input96_channel_head_feature_w10_channel_prior_top1_select1_fusion_bs32",
            "predictor": "channel_head_mlp",
            "window": {"input_len": 96},
            "cluster": {
                "feature_aware": {
                    "enable": True,
                    "weight": 1.0,
                    "acf_lags": [1, 24, 96],
                },
                "singleton_merge_strategy": "pool",
            },
            "model": {},
            "train": {"batch_size": 32},
            "moe": channel_prior_overrides(topk=1, select_ranks=[1]),
            "pred_residual": adaptive_residual_overrides(),
        },
        {
            "name": "longctx336_base96_channel_head_feature_w10_channel_prior_top1_select1_fusion_bs32",
            "predictor": "channel_head_mlp",
            "window": {"input_len": 336},
            "cluster": {
                "feature_aware": {
                    "enable": True,
                    "weight": 1.0,
                    "acf_lags": [1, 24, 96],
                },
                "singleton_merge_strategy": "pool",
            },
            "model": {"predictor_input_len": 96},
            "train": {"batch_size": 32},
            "moe": channel_prior_overrides(topk=1, select_ranks=[1]),
            "pred_residual": adaptive_residual_overrides(),
        },
        {
            "name": "longctx336_summary_channel_head_feature_w10_channel_prior_top1_select1_fusion_bs32",
            "predictor": "long_context_channel_head_mlp",
            "window": {"input_len": 336},
            "cluster": {
                "feature_aware": {
                    "enable": True,
                    "weight": 1.0,
                    "acf_lags": [1, 24, 96],
                },
                "singleton_merge_strategy": "pool",
            },
            "model": {"predictor_input_len": 96},
            "train": {"batch_size": 32},
            "moe": channel_prior_overrides(topk=1, select_ranks=[1]),
            "pred_residual": adaptive_residual_overrides(),
        },
        {
            "name": "longctx336_summary_channel_head_h128_do010_prior_top1_select1_fusion_bs32",
            "predictor": "long_context_channel_head_mlp",
            "window": {"input_len": 336},
            "cluster": {
                "feature_aware": {
                    "enable": True,
                    "weight": 1.0,
                    "acf_lags": [1, 24, 96],
                },
                "singleton_merge_strategy": "pool",
            },
            "model": {"predictor_input_len": 96, "hidden_dim": 128, "dropout": 0.10},
            "train": {"batch_size": 32},
            "moe": channel_prior_overrides(topk=1, select_ranks=[1]),
            "pred_residual": adaptive_residual_overrides(),
        },
        {
            "name": "longctx336_summary_channel_head_recursive96_prior_top1_select1_fusion_bs32",
            "predictor": "long_context_channel_head_mlp",
            "window": {"input_len": 336},
            "cluster": {
                "feature_aware": {
                    "enable": True,
                    "weight": 1.0,
                    "acf_lags": [1, 24, 96],
                },
                "singleton_merge_strategy": "pool",
            },
            "model": {"predictor_input_len": 96, "recursive_rollout": True, "recursive_chunk_len": 96},
            "train": {"batch_size": 32},
            "moe": channel_prior_overrides(topk=1, select_ranks=[1]),
            "pred_residual": adaptive_residual_overrides(),
        },
        {
            "name": "longctx336_seasonal_profile_channel_head_prior_top1_select1_fusion_bs32",
            "predictor": "long_context_channel_head_mlp",
            "window": {"input_len": 336},
            "cluster": {
                "feature_aware": {
                    "enable": True,
                    "weight": 1.0,
                    "acf_lags": [1, 24, 96],
                },
                "singleton_merge_strategy": "pool",
            },
            "model": {
                "predictor_input_len": 96,
                "long_context_include_seasonal_profile": True,
            },
            "train": {"batch_size": 32},
            "moe": channel_prior_overrides(topk=1, select_ranks=[1]),
            "pred_residual": adaptive_residual_overrides(),
        },
        {
            "name": "longctx336_profile_anchor_channel_head_detail025_prior_top1_select1_fusion_bs32",
            "predictor": "long_context_anchor_channel_head_mlp",
            "window": {"input_len": 336},
            "cluster": {
                "feature_aware": {
                    "enable": True,
                    "weight": 1.0,
                    "acf_lags": [1, 24, 96],
                },
                "singleton_merge_strategy": "pool",
            },
            "model": {
                "predictor_input_len": 96,
                "long_context_include_seasonal_profile": True,
                "anchor_chunk_len": 96,
                "anchor_detail_scale": 0.25,
            },
            "train": {"batch_size": 32},
            "moe": channel_prior_overrides(topk=1, select_ranks=[1]),
            "pred_residual": adaptive_residual_overrides(),
        },
        {
            "name": "channel_head_seasonal_align_adapter_prior_top2_select12_fusion_bs32",
            "predictor": "channel_head_mlp",
            "window": {"input_len": 336},
            "cluster": {
                "feature_aware": {
                    "enable": True,
                    "weight": 1.0,
                    "acf_lags": [1, 24, 96],
                },
                "singleton_merge_strategy": "pool",
            },
            "penalties": {
                "enabled": [
                    "level",
                    "range",
                    "delta",
                    "diff_amp",
                    "trend",
                    "direction",
                    "seasonal_align",
                ],
            },
            "train": {"batch_size": 32},
            "moe": channel_prior_overrides(topk=2, select_ranks=[1, 2]),
            "pred_residual": residual_overrides_with(
                seasonal_anchor_names=["seasonal_align"],
                seasonal_anchor_period=96,
                seasonal_anchor_num_periods=2,
                seasonal_anchor_scale=0.5,
            ),
        },
        {
            "name": "channel_head_trend_seasonal_align_s010_prior_top1_fusion_bs32",
            "predictor": "channel_head_mlp",
            "window": {"input_len": 336},
            "cluster": {
                "feature_aware": {
                    "enable": True,
                    "weight": 1.0,
                    "acf_lags": [1, 24, 96],
                },
                "singleton_merge_strategy": "pool",
            },
            "penalties": {"enabled": ["trend", "seasonal_align"]},
            "train": {"batch_size": 32},
            "moe": {
                **channel_prior_overrides(topk=1, select_ranks=[1]),
                "lambda_init": {"trend": 0.05, "seasonal_align": 0.03},
                "lambda_min": {"trend": 0.0, "seasonal_align": 0.0},
            },
            "pred_residual": residual_overrides_with(
                alpha_scale=0.5,
                seasonal_anchor_names=["seasonal_align"],
                seasonal_anchor_period=96,
                seasonal_anchor_num_periods=2,
                seasonal_anchor_scale=0.10,
                selection_policy="val_mse_gate_guarded",
                selection_min_rel_improvement=0.0005,
            ),
        },
        {
            "name": "channel_head_trend_seasonal_align_s005_prior_top1_fusion_bs32",
            "predictor": "channel_head_mlp",
            "window": {"input_len": 336},
            "cluster": {
                "feature_aware": {
                    "enable": True,
                    "weight": 1.0,
                    "acf_lags": [1, 24, 96],
                },
                "singleton_merge_strategy": "pool",
            },
            "penalties": {"enabled": ["trend", "seasonal_align"]},
            "train": {"batch_size": 32},
            "moe": {
                **channel_prior_overrides(topk=1, select_ranks=[1]),
                "lambda_init": {"trend": 0.05, "seasonal_align": 0.02},
                "lambda_min": {"trend": 0.0, "seasonal_align": 0.0},
            },
            "pred_residual": residual_overrides_with(
                alpha_scale=0.5,
                seasonal_anchor_names=["seasonal_align"],
                seasonal_anchor_period=96,
                seasonal_anchor_num_periods=2,
                seasonal_anchor_scale=0.05,
                selection_policy="val_mse_gate_guarded",
                selection_min_rel_improvement=0.0005,
            ),
        },
        {
            "name": "channel_head_trend_seasonal_align_s025_prior_top1_fusion_bs32",
            "predictor": "channel_head_mlp",
            "window": {"input_len": 336},
            "cluster": {
                "feature_aware": {
                    "enable": True,
                    "weight": 1.0,
                    "acf_lags": [1, 24, 96],
                },
                "singleton_merge_strategy": "pool",
            },
            "penalties": {"enabled": ["trend", "seasonal_align"]},
            "train": {"batch_size": 32},
            "moe": {
                **channel_prior_overrides(topk=1, select_ranks=[1]),
                "lambda_init": {"trend": 0.05, "seasonal_align": 0.03},
                "lambda_min": {"trend": 0.0, "seasonal_align": 0.0},
            },
            "pred_residual": residual_overrides_with(
                alpha_scale=0.5,
                seasonal_anchor_names=["seasonal_align"],
                seasonal_anchor_period=96,
                seasonal_anchor_num_periods=2,
                seasonal_anchor_scale=0.25,
                selection_policy="val_mse_gate_guarded",
                selection_min_rel_improvement=0.0005,
            ),
        },
        {
            "name": "channel_head_trend_seasonal_align_s050_prior_top1_fusion_bs32",
            "predictor": "channel_head_mlp",
            "window": {"input_len": 336},
            "cluster": {
                "feature_aware": {
                    "enable": True,
                    "weight": 1.0,
                    "acf_lags": [1, 24, 96],
                },
                "singleton_merge_strategy": "pool",
            },
            "penalties": {"enabled": ["trend", "seasonal_align"]},
            "train": {"batch_size": 32},
            "moe": {
                **channel_prior_overrides(topk=1, select_ranks=[1]),
                "lambda_init": {"trend": 0.05, "seasonal_align": 0.05},
                "lambda_min": {"trend": 0.0, "seasonal_align": 0.0},
            },
            "pred_residual": residual_overrides_with(
                alpha_scale=0.5,
                seasonal_anchor_names=["seasonal_align"],
                seasonal_anchor_period=96,
                seasonal_anchor_num_periods=2,
                seasonal_anchor_scale=0.50,
                selection_policy="val_mse_gate_guarded",
                selection_min_rel_improvement=0.0005,
            ),
        },
        {
            "name": "seasonal_hybrid336_profile_anchor_mix_init_m2_gate4_detail025_prior_top1_select1_fusion_bs32",
            "predictor": "seasonality_gated_channel_head_mlp",
            "window": {"input_len": 336},
            "cluster": {
                "feature_aware": {
                    "enable": True,
                    "weight": 1.0,
                    "acf_lags": [1, 24, 96],
                },
                "singleton_merge_strategy": "pool",
            },
            "model": {
                "predictor_input_len": 96,
                "long_context_include_seasonal_profile": True,
                "anchor_chunk_len": 96,
                "anchor_detail_scale": 0.25,
                "seasonal_mix_init": -2.0,
                "seasonal_gate_strength": 4.0,
                "seasonal_gate_threshold": 0.75,
            },
            "train": {"batch_size": 32},
            "moe": channel_prior_overrides(topk=1, select_ranks=[1]),
            "pred_residual": adaptive_residual_overrides(),
        },
        {
            "name": "seasonal_hybrid336_profile_anchor_mix_init_m3_gate6_detail025_prior_top1_select1_fusion_bs32",
            "predictor": "seasonality_gated_channel_head_mlp",
            "window": {"input_len": 336},
            "cluster": {
                "feature_aware": {
                    "enable": True,
                    "weight": 1.0,
                    "acf_lags": [1, 24, 96],
                },
                "singleton_merge_strategy": "pool",
            },
            "model": {
                "predictor_input_len": 96,
                "long_context_include_seasonal_profile": True,
                "anchor_chunk_len": 96,
                "anchor_detail_scale": 0.25,
                "seasonal_mix_init": -3.0,
                "seasonal_gate_strength": 6.0,
                "seasonal_gate_threshold": 0.75,
            },
            "train": {"batch_size": 32},
            "moe": channel_prior_overrides(topk=1, select_ranks=[1]),
            "pred_residual": adaptive_residual_overrides(),
        },
        {
            "name": "longctx336_profile_anchor_channel_head_detail010_prior_top1_select1_fusion_bs32",
            "predictor": "long_context_anchor_channel_head_mlp",
            "window": {"input_len": 336},
            "cluster": {
                "feature_aware": {
                    "enable": True,
                    "weight": 1.0,
                    "acf_lags": [1, 24, 96],
                },
                "singleton_merge_strategy": "pool",
            },
            "model": {
                "predictor_input_len": 96,
                "long_context_include_seasonal_profile": True,
                "anchor_chunk_len": 96,
                "anchor_detail_scale": 0.10,
            },
            "train": {"batch_size": 32},
            "moe": channel_prior_overrides(topk=1, select_ranks=[1]),
            "pred_residual": adaptive_residual_overrides(),
        },
        {
            "name": "longctx336_profile_anchor_tdstrong_mseheavy_detail025_prior_top1_select1_fusion_bs32",
            "predictor": "long_context_anchor_channel_head_mlp",
            "window": {"input_len": 336},
            "cluster": {
                "feature_aware": {
                    "enable": True,
                    "weight": 1.0,
                    "acf_lags": [1, 24, 96],
                },
                "singleton_merge_strategy": "pool",
            },
            "model": {
                "hidden_dim": 256,
                "dropout": 0.20,
                "predictor_input_len": 96,
                "long_context_include_seasonal_profile": True,
                "anchor_chunk_len": 96,
                "anchor_detail_scale": 0.25,
            },
            "penalties": {"enabled": ["trend", "direction"]},
            "train": {
                "batch_size": 32,
                "mse_weight": 0.9,
                "mae_objective": {"enable": True, "kind": "l1", "weight": 0.6, "warmup_epochs": 5},
            },
            "moe": {
                **channel_prior_overrides(topk=1, select_ranks=[1]),
                "lambda_init": {"trend": 0.07875, "direction": 0.1575},
                "lambda_min": {"trend": 0.0, "direction": 0.0},
                "lambda_schedule": {"trend": "none", "direction": "none"},
                "gate_init_bias": {"enable": True, "values": {"default": 0.0}},
            },
            "pred_residual": residual_overrides_with(
                residual_clip=0.0,
                alpha_scale=1.1,
                gate_calibrator={
                    "loss": "mse",
                    "selection_metric": "mse",
                    "epochs": 30,
                    "train_fraction": 0.7,
                    "hidden_dim": 32,
                    "batch_size": 256,
                    "max_scale": 1.0,
                    "init_scale": 0.4,
                    "scale_reg": 5.0e-4,
                    "scale_mode": "sigmoid",
                    "standardize_features": True,
                },
            ),
        },
        {
            "name": "input96_channel_head_feature_w10_channel_prior_top2_select12_scale_bs32",
            "predictor": "channel_head_mlp",
            "window": {"input_len": 96},
            "cluster": {
                "feature_aware": {
                    "enable": True,
                    "weight": 1.0,
                    "acf_lags": [1, 24, 96],
                },
                "singleton_merge_strategy": "pool",
            },
            "model": {},
            "train": {"batch_size": 32},
            "moe": channel_prior_overrides(topk=2, select_ranks=[1, 2]),
            "pred_residual": residual_overrides_with(
                selection_policy="val_mse_scale",
                selection_scale_min=0.0,
                selection_scale_max=1.5,
                selection_scale_steps=16,
            ),
        },
        {
            "name": "input96_channel_head_feature_w10_channel_prior_top2_select12_gate_train_bs32",
            "predictor": "channel_head_mlp",
            "window": {"input_len": 96},
            "cluster": {
                "feature_aware": {
                    "enable": True,
                    "weight": 1.0,
                    "acf_lags": [1, 24, 96],
                },
                "singleton_merge_strategy": "pool",
            },
            "model": {},
            "train": {"batch_size": 32},
            "moe": channel_prior_overrides(topk=2, select_ranks=[1, 2]),
            "pred_residual": residual_overrides_with(
                gate_calibrator={
                    "source_split": "train",
                    "scale_mode": "signed_tanh",
                    "max_scale": 0.75,
                    "init_scale": 0.25,
                    "scale_reg": 0.001,
                },
                selection_policy="val_mse_gate_guarded",
                selection_min_rel_improvement=0.001,
            ),
        },
        {
            "name": "input96_channel_lstm_mixer_feature_w10_channel_prior_top1_select1_bs32",
            "predictor": "channel_lstm_mixer",
            "window": {"input_len": 96},
            "cluster": {
                "feature_aware": {
                    "enable": True,
                    "weight": 1.0,
                    "acf_lags": [1, 24, 96],
                },
                "singleton_merge_strategy": "pool",
            },
            "model": {
                "hidden_dim": 256,
                "dropout": 0.0,
                "lstm_num_layers": 1,
                "backbone_mix_init": -2.0,
            },
            "train": {"batch_size": 32},
            "moe": rnn_safe_moe(channel_prior_overrides(topk=1, select_ranks=[1])),
            "pred_residual": adaptive_residual_overrides(),
        },
        {
            "name": "longctx336_base96_channel_lstm_mixer_feature_w10_channel_prior_top1_select1_bs32",
            "predictor": "channel_lstm_mixer",
            "window": {"input_len": 336},
            "cluster": {
                "feature_aware": {
                    "enable": True,
                    "weight": 1.0,
                    "acf_lags": [1, 24, 96],
                },
                "singleton_merge_strategy": "pool",
            },
            "model": {
                "predictor_input_len": 96,
                "hidden_dim": 256,
                "dropout": 0.0,
                "lstm_num_layers": 1,
                "backbone_mix_init": -2.0,
            },
            "train": {"batch_size": 32},
            "moe": rnn_safe_moe(channel_prior_overrides(topk=1, select_ranks=[1])),
            "pred_residual": adaptive_residual_overrides(),
        },
        {
            "name": "input96_channel_lstm_mixer_feature_w10_channel_prior_top1_select1_mixm3_bs32",
            "predictor": "channel_lstm_mixer",
            "window": {"input_len": 96},
            "cluster": {
                "feature_aware": {
                    "enable": True,
                    "weight": 1.0,
                    "acf_lags": [1, 24, 96],
                },
                "singleton_merge_strategy": "pool",
            },
            "model": {
                "hidden_dim": 256,
                "dropout": 0.0,
                "lstm_num_layers": 1,
                "backbone_mix_init": -3.0,
            },
            "train": {"batch_size": 32},
            "moe": rnn_safe_moe(channel_prior_overrides(topk=1, select_ranks=[1])),
            "pred_residual": adaptive_residual_overrides(),
        },
        {
            "name": "input96_channel_lstm_mixer_feature_w10_channel_prior_top1_select1_mixm1_bs32",
            "predictor": "channel_lstm_mixer",
            "window": {"input_len": 96},
            "cluster": {
                "feature_aware": {
                    "enable": True,
                    "weight": 1.0,
                    "acf_lags": [1, 24, 96],
                },
                "singleton_merge_strategy": "pool",
            },
            "model": {
                "hidden_dim": 256,
                "dropout": 0.0,
                "lstm_num_layers": 1,
                "backbone_mix_init": -1.0,
            },
            "train": {"batch_size": 32},
            "moe": rnn_safe_moe(channel_prior_overrides(topk=1, select_ranks=[1])),
            "pred_residual": adaptive_residual_overrides(),
        },
        {
            "name": "input96_channel_lstm_mixer_seasonal_anchor96_prior_top1_bs32",
            "predictor": "channel_lstm_mixer",
            "window": {"input_len": 96},
            "cluster": {
                "feature_aware": {
                    "enable": True,
                    "weight": 1.0,
                    "acf_lags": [1, 24, 96],
                },
                "singleton_merge_strategy": "pool",
            },
            "model": {
                "hidden_dim": 256,
                "dropout": 0.0,
                "lstm_num_layers": 1,
                "backbone_mix_init": -2.0,
                "seasonal_anchor": True,
                "seasonal_anchor_period": 96,
                "seasonal_anchor_delta_scale": 1.0,
            },
            "train": {"batch_size": 32},
            "moe": rnn_safe_moe(channel_prior_overrides(topk=1, select_ranks=[1])),
            "pred_residual": adaptive_residual_overrides(),
        },
        {
            "name": "input96_channel_head_seasonal_anchor96_prior_top1_bs32",
            "predictor": "channel_head_mlp",
            "window": {"input_len": 96},
            "cluster": {
                "feature_aware": {
                    "enable": True,
                    "weight": 1.0,
                    "acf_lags": [1, 24, 96],
                },
                "singleton_merge_strategy": "pool",
            },
            "model": {
                "seasonal_anchor": True,
                "seasonal_anchor_period": 96,
                "seasonal_anchor_delta_scale": 1.0,
            },
            "train": {"batch_size": 32},
            "moe": rnn_safe_moe(channel_prior_overrides(topk=1, select_ranks=[1])),
            "pred_residual": adaptive_residual_overrides(),
        },
        {
            "name": "input96_channel_lstm_hardroute_hull_mufl_lull_prior_top1_bs32",
            "predictor": "channel_lstm_mixer",
            "window": {"input_len": 96},
            "cluster": {
                "feature_aware": {
                    "enable": True,
                    "weight": 1.0,
                    "acf_lags": [1, 24, 96],
                },
                "singleton_merge_strategy": "pool",
            },
            "model": {
                "hidden_dim": 256,
                "dropout": 0.0,
                "lstm_num_layers": 1,
                "backbone_hard_route": True,
                "backbone_lstm_channel_indices": [1, 2, 5],
            },
            "train": {"batch_size": 32},
            "moe": rnn_safe_moe(channel_prior_overrides(topk=1, select_ranks=[1])),
            "pred_residual": adaptive_residual_overrides(),
        },
        {
            "name": "input96_channel_lstm_mixer_feature_w10_channel_prior_top2_select12_gate_train_bs32",
            "predictor": "channel_lstm_mixer",
            "window": {"input_len": 96},
            "cluster": {
                "feature_aware": {
                    "enable": True,
                    "weight": 1.0,
                    "acf_lags": [1, 24, 96],
                },
                "singleton_merge_strategy": "pool",
            },
            "model": {
                "hidden_dim": 256,
                "dropout": 0.0,
                "lstm_num_layers": 1,
                "backbone_mix_init": -2.0,
            },
            "train": {"batch_size": 32},
            "moe": rnn_safe_moe(channel_prior_overrides(topk=2, select_ranks=[1, 2])),
            "pred_residual": residual_overrides_with(
                gate_calibrator={
                    "source_split": "train",
                    "scale_mode": "signed_tanh",
                    "max_scale": 0.75,
                    "init_scale": 0.25,
                    "scale_reg": 0.001,
                },
                selection_policy="val_mse_gate_guarded",
                selection_min_rel_improvement=0.001,
            ),
        },
        {
            "name": "input96_lstm_revin_feature_w10_channel_prior_top1_select1_fusion_bs64",
            "predictor": "lstm_revin",
            "window": {"input_len": 96},
            "cluster": {
                "feature_aware": {
                    "enable": True,
                    "weight": 1.0,
                    "acf_lags": [1, 24, 96],
                },
                "singleton_merge_strategy": "pool",
            },
            "model": {"hidden_dim": 256, "dropout": 0.0, "lstm_num_layers": 1},
            "train": {"batch_size": 64},
            "moe": rnn_safe_moe(channel_prior_overrides(topk=1, select_ranks=[1])),
            "pred_residual": adaptive_residual_overrides(),
        },
        {
            "name": "input96_lstm_revin_feature_w10_channel_prior_top2_select12_scale_bs64",
            "predictor": "lstm_revin",
            "window": {"input_len": 96},
            "cluster": {
                "feature_aware": {
                    "enable": True,
                    "weight": 1.0,
                    "acf_lags": [1, 24, 96],
                },
                "singleton_merge_strategy": "pool",
            },
            "model": {"hidden_dim": 256, "dropout": 0.0, "lstm_num_layers": 1},
            "train": {"batch_size": 64},
            "moe": rnn_safe_moe(channel_prior_overrides(topk=2, select_ranks=[1, 2])),
            "pred_residual": residual_overrides_with(
                selection_policy="val_mse_scale",
                selection_scale_min=0.0,
                selection_scale_max=1.5,
                selection_scale_steps=16,
            ),
        },
        {
            "name": "input96_lstm_revin_feature_w10_channel_prior_top2_select12_gate_train_bs64",
            "predictor": "lstm_revin",
            "window": {"input_len": 96},
            "cluster": {
                "feature_aware": {
                    "enable": True,
                    "weight": 1.0,
                    "acf_lags": [1, 24, 96],
                },
                "singleton_merge_strategy": "pool",
            },
            "model": {"hidden_dim": 256, "dropout": 0.0, "lstm_num_layers": 1},
            "train": {"batch_size": 64},
            "moe": rnn_safe_moe(channel_prior_overrides(topk=2, select_ranks=[1, 2])),
            "pred_residual": residual_overrides_with(
                gate_calibrator={
                    "source_split": "train",
                    "scale_mode": "signed_tanh",
                    "max_scale": 0.75,
                    "init_scale": 0.25,
                    "scale_reg": 0.001,
                },
                selection_policy="val_mse_gate_guarded",
                selection_min_rel_improvement=0.001,
            ),
        },
        {
            "name": "input96_lstm_revin_feature_w10_channel_prior_top2_select12_gate_train_safe64_a08_bs64",
            "predictor": "lstm_revin",
            "window": {"input_len": 96},
            "cluster": {
                "feature_aware": {
                    "enable": True,
                    "weight": 1.0,
                    "acf_lags": [1, 24, 96],
                },
                "singleton_merge_strategy": "pool",
            },
            "model": {"hidden_dim": 256, "dropout": 0.0, "lstm_num_layers": 1},
            "train": {"batch_size": 64},
            "moe": rnn_safe_moe(channel_prior_overrides(topk=2, select_ranks=[1, 2])),
            "pred_residual": residual_overrides_with(
                feature_mode="safe_augmented",
                corrector_hidden=64,
                alpha_scale=0.8,
                residual_clip=4.0,
                gate_calibrator={
                    "source_split": "train",
                    "scale_mode": "signed_tanh",
                    "max_scale": 0.75,
                    "init_scale": 0.25,
                    "scale_reg": 0.001,
                },
                selection_policy="val_mse_gate_guarded",
                selection_min_rel_improvement=0.001,
            ),
        },
        {
            "name": "input96_lstm_revin_feature_w10_channel_prior_top2_select12_gate_train_safe64_a12_bs64",
            "predictor": "lstm_revin",
            "window": {"input_len": 96},
            "cluster": {
                "feature_aware": {
                    "enable": True,
                    "weight": 1.0,
                    "acf_lags": [1, 24, 96],
                },
                "singleton_merge_strategy": "pool",
            },
            "model": {"hidden_dim": 256, "dropout": 0.0, "lstm_num_layers": 1},
            "train": {"batch_size": 64},
            "moe": rnn_safe_moe(channel_prior_overrides(topk=2, select_ranks=[1, 2])),
            "pred_residual": residual_overrides_with(
                feature_mode="safe_augmented",
                corrector_hidden=64,
                alpha_scale=1.2,
                residual_clip=4.0,
                gate_calibrator={
                    "source_split": "train",
                    "scale_mode": "signed_tanh",
                    "max_scale": 0.75,
                    "init_scale": 0.25,
                    "scale_reg": 0.001,
                },
                selection_policy="val_mse_gate_guarded",
                selection_min_rel_improvement=0.001,
            ),
        },
        {
            "name": "input96_lstm_revin_feature_w10_channel_prior_top2_select12_gate_train_h128_drop01_bs64",
            "predictor": "lstm_revin",
            "window": {"input_len": 96},
            "cluster": {
                "feature_aware": {
                    "enable": True,
                    "weight": 1.0,
                    "acf_lags": [1, 24, 96],
                },
                "singleton_merge_strategy": "pool",
            },
            "model": {"hidden_dim": 128, "dropout": 0.1, "lstm_num_layers": 1},
            "train": {"batch_size": 64},
            "moe": rnn_safe_moe(channel_prior_overrides(topk=2, select_ranks=[1, 2])),
            "pred_residual": residual_overrides_with(
                gate_calibrator={
                    "source_split": "train",
                    "scale_mode": "signed_tanh",
                    "max_scale": 0.75,
                    "init_scale": 0.25,
                    "scale_reg": 0.001,
                },
                selection_policy="val_mse_gate_guarded",
                selection_min_rel_improvement=0.001,
            ),
        },
        {
            "name": "input96_lstm_revin_feature_w10_channel_prior_top2_select12_strong_bs64",
            "predictor": "lstm_revin",
            "window": {"input_len": 96},
            "cluster": {
                "feature_aware": {
                    "enable": True,
                    "weight": 1.0,
                    "acf_lags": [1, 24, 96],
                },
                "singleton_merge_strategy": "pool",
            },
            "model": {"hidden_dim": 256, "dropout": 0.0, "lstm_num_layers": 1},
            "train": {"batch_size": 64},
            "moe": rnn_safe_moe(channel_prior_overrides(topk=2, select_ranks=[1, 2])),
            "pred_residual": residual_overrides_with(alpha_scale=2.4, residual_clip=4.0),
        },
        {
            "name": "channel_head_feature_w10_channel_prior_top1_select1_fusion_bs32_channel_adapters",
            "predictor": "channel_head_mlp",
            "cluster": {
                "feature_aware": {
                    "enable": True,
                    "weight": 1.0,
                    "acf_lags": [1, 24, 96],
                },
                "singleton_merge_strategy": "pool",
            },
            "model": {},
            "train": {"batch_size": 32},
            "moe": channel_prior_overrides(topk=1, select_ranks=[1]),
            "pred_residual": channel_expert_residual_overrides(),
        },
        {
            "name": "channel_head_feature_w10_channel_prior_top1_select1_fusion_bs32_channel_delta_adapters",
            "predictor": "channel_head_mlp",
            "cluster": {
                "feature_aware": {
                    "enable": True,
                    "weight": 1.0,
                    "acf_lags": [1, 24, 96],
                },
                "singleton_merge_strategy": "pool",
            },
            "model": {},
            "train": {"batch_size": 32},
            "moe": channel_prior_overrides(topk=1, select_ranks=[1]),
            "pred_residual": channel_expert_residual_overrides(mode_type="delta"),
        },
        {
            "name": "channel_head_feature_w10_channel_prior_top3_select123_fusion",
            "predictor": "channel_head_mlp",
            "cluster": {
                "feature_aware": {
                    "enable": True,
                    "weight": 1.0,
                    "acf_lags": [1, 24, 96],
                },
                "singleton_merge_strategy": "pool",
            },
            "model": {},
            "moe": channel_prior_overrides(topk=3, select_ranks=[1, 2, 3]),
            "pred_residual": adaptive_residual_overrides(),
        },
        {
            "name": "mlp_selector_fusion",
            "predictor": "mlp",
            "cluster": {},
            "model": {},
            "moe": {"explainability": {"enable": True, "splits": ["train", "val", "test"], "max_batches": 0}},
            "pred_residual": adaptive_residual_overrides(),
        },
        {
            "name": "mlp_prior_top1_fusion",
            "predictor": "mlp",
            "cluster": {},
            "model": {},
            "moe": cluster_prior_overrides(topk=1, select_ranks=[1]),
            "pred_residual": adaptive_residual_overrides(),
        },
        {
            "name": "mlp_prior_top2_select12_fusion",
            "predictor": "mlp",
            "cluster": {},
            "model": {},
            "moe": cluster_prior_overrides(topk=2, select_ranks=[1, 2]),
            "pred_residual": adaptive_residual_overrides(),
        },
        {
            "name": "mlp_feature_w10_keep_singletons_prior_top2_select12_fusion",
            "predictor": "mlp",
            "cluster": {
                "feature_aware": {
                    "enable": True,
                    "weight": 1.0,
                    "acf_lags": [1, 24, 96],
                },
                "singleton_merge_strategy": "keep",
            },
            "model": {},
            "moe": cluster_prior_overrides(topk=2, select_ranks=[1, 2]),
            "pred_residual": adaptive_residual_overrides(),
        },
        {
            "name": "mlp_feature_w10_prior_top2_select12_fusion",
            "predictor": "mlp",
            "cluster": {
                "feature_aware": {
                    "enable": True,
                    "weight": 1.0,
                    "acf_lags": [1, 24, 96],
                },
                "singleton_merge_strategy": "pool",
            },
            "model": {},
            "moe": cluster_prior_overrides(topk=2, select_ranks=[1, 2]),
            "pred_residual": adaptive_residual_overrides(),
        },
        {
            "name": "mlp_no_merge_th085",
            "predictor": "mlp",
            "cluster": {
                "distance_threshold": 0.85,
                "merge_small_clusters": False,
                "min_cluster_size": 1,
            },
            "model": {},
            "pred_residual": {},
        },
        {
            "name": "channel_head_no_merge_th085",
            "predictor": "channel_head_mlp",
            "cluster": {
                "distance_threshold": 0.85,
                "merge_small_clusters": False,
                "min_cluster_size": 1,
            },
            "model": {},
            "pred_residual": {},
        },
        {
            "name": "channel_head_no_merge_th075",
            "predictor": "channel_head_mlp",
            "cluster": {
                "distance_threshold": 0.75,
                "merge_small_clusters": False,
                "min_cluster_size": 1,
            },
            "model": {},
            "pred_residual": {},
        },
        {
            "name": "channel_head_no_merge_th080",
            "predictor": "channel_head_mlp",
            "cluster": {
                "distance_threshold": 0.80,
                "merge_small_clusters": False,
                "min_cluster_size": 1,
            },
            "model": {},
            "pred_residual": {},
        },
        {
            "name": "channel_head_no_merge_th090",
            "predictor": "channel_head_mlp",
            "cluster": {
                "distance_threshold": 0.90,
                "merge_small_clusters": False,
                "min_cluster_size": 1,
            },
            "model": {},
            "pred_residual": {},
        },
        {
            "name": "mlp_no_merge_th075",
            "predictor": "mlp",
            "cluster": {
                "distance_threshold": 0.75,
                "merge_small_clusters": False,
                "min_cluster_size": 1,
            },
            "model": {},
            "pred_residual": {},
        },
        {
            "name": "mlp_no_merge_th080",
            "predictor": "mlp",
            "cluster": {
                "distance_threshold": 0.80,
                "merge_small_clusters": False,
                "min_cluster_size": 1,
            },
            "model": {},
            "pred_residual": {},
        },
        {
            "name": "mlp_no_merge_th090",
            "predictor": "mlp",
            "cluster": {
                "distance_threshold": 0.90,
                "merge_small_clusters": False,
                "min_cluster_size": 1,
            },
            "model": {},
            "pred_residual": {},
        },
        {
            "name": "channel_head_hd384",
            "predictor": "channel_head_mlp",
            "cluster": {},
            "model": {"hidden_dim": 384},
            "pred_residual": {},
        },
        {
            "name": "channel_head_hd512",
            "predictor": "channel_head_mlp",
            "cluster": {},
            "model": {"hidden_dim": 512},
            "pred_residual": {},
        },
        {
            "name": "channel_head_do010",
            "predictor": "channel_head_mlp",
            "cluster": {},
            "model": {"dropout": 0.10},
            "pred_residual": {},
        },
        {
            "name": "channel_head_hd384_do010",
            "predictor": "channel_head_mlp",
            "cluster": {},
            "model": {"hidden_dim": 384, "dropout": 0.10},
            "pred_residual": {},
        },
        {
            "name": "mlp_hd384",
            "predictor": "mlp",
            "cluster": {},
            "model": {"hidden_dim": 384},
            "pred_residual": {},
        },
        {
            "name": "mlp_do010",
            "predictor": "mlp",
            "cluster": {},
            "model": {"dropout": 0.10},
            "pred_residual": {},
        },
        {
            "name": "segment_mlp_chunk96",
            "predictor": "segment_mlp",
            "cluster": {},
            "model": {"segment_chunk_len": 96},
            "pred_residual": {},
        },
        {
            "name": "long_anchor_mlp",
            "predictor": "long_anchor_mlp",
            "cluster": {},
            "model": {"anchor_chunk_len": 96, "anchor_detail_scale": 0.5, "anchor_residual": True},
            "pred_residual": {},
        },
        {
            "name": "long_anchor_mlp_detail025",
            "predictor": "long_anchor_mlp",
            "cluster": {},
            "model": {"anchor_chunk_len": 96, "anchor_detail_scale": 0.25, "anchor_residual": True},
            "pred_residual": {},
        },
        {
            "name": "mlp_recursive96",
            "predictor": "mlp",
            "cluster": {},
            "model": {"recursive_rollout": True, "recursive_chunk_len": 96},
            "pred_residual": {},
        },
        {
            "name": "channel_head_recursive96",
            "predictor": "channel_head_mlp",
            "cluster": {},
            "model": {"recursive_rollout": True, "recursive_chunk_len": 96},
            "pred_residual": {},
        },
        {
            "name": "dlinear_k25",
            "predictor": "dlinear",
            "cluster": {},
            "model": {"dlinear_kernel_size": 25},
            "pred_residual": {},
        },
        {
            "name": "dlinear_k13",
            "predictor": "dlinear",
            "cluster": {},
            "model": {"dlinear_kernel_size": 13},
            "pred_residual": {},
        },
        {
            "name": "channel_dlinear_k25",
            "predictor": "channel_dlinear",
            "cluster": {},
            "model": {"dlinear_kernel_size": 25},
            "pred_residual": {},
        },
        {
            "name": "channel_dlinear_k13",
            "predictor": "channel_dlinear",
            "cluster": {},
            "model": {"dlinear_kernel_size": 13},
            "pred_residual": {},
        },
        {
            "name": "patchtst_light",
            "predictor": "patchtst",
            "cluster": {},
            "model": {
                "patch_d_model": 128,
                "patch_len": 16,
                "patch_stride": 8,
                "patch_num_layers": 2,
                "patch_num_heads": 4,
                "patch_ff_dim": 256,
            },
            "pred_residual": {},
        },
    ]


def run_one(config_path: Path, out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "stdout.log").open("w", encoding="utf-8") as stdout_f, (
        out_dir / "stderr.log"
    ).open("w", encoding="utf-8") as stderr_f:
        proc = subprocess.run(
            [sys.executable, "-m", "src.train", "--config", str(config_path)],
            cwd=str(REPO_ROOT),
            text=True,
            stdout=stdout_f,
            stderr=stderr_f,
        )
    return int(proc.returncode)


def load_metrics(out_dir: Path) -> dict[str, Any]:
    path = out_dir / "run_summary.json"
    if not path.exists():
        return {}
    s = json.loads(path.read_text(encoding="utf-8"))
    return {
        "test_mse": s.get("test", {}).get("avg_mse", ""),
        "test_mae": s.get("test", {}).get("avg_mae", ""),
        "val_mse": s.get("val", {}).get("avg_mse", ""),
        "val_mae": s.get("val", {}).get("avg_mae", ""),
        "best_epoch": s.get("best_epoch", ""),
        "test_per_cluster_mse": s.get("test", {}).get("per_cluster_mse", ""),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ETTm2")
    ap.add_argument("--horizon", type=int, default=720)
    ap.add_argument("--out-root", default="outputs/ettm2_growth_fix_probe")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-candidates", type=int, default=3)
    ap.add_argument(
        "--candidates",
        nargs="*",
        default=None,
        help="Optional candidate names to run. Defaults to the built-in order.",
    )
    ap.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse an existing run_summary.json instead of retraining that candidate.",
    )
    ap.add_argument("--batch-size-override", type=int, default=None)
    ap.add_argument("--epochs-override", type=int, default=None)
    args = ap.parse_args()

    base_path = REPO_ROOT / "outputs" / "main_table_tsl_aligned" / "configs" / f"{args.dataset}_pred_{args.horizon}.yaml"
    base = read_yaml(base_path)
    out_root = REPO_ROOT / args.out_root
    rows: list[dict[str, Any]] = []
    fields = [
        "name",
        "status",
        "test_mse",
        "test_mae",
        "val_mse",
        "val_mae",
        "best_epoch",
        "test_per_cluster_mse",
        "predictor",
        "distance_threshold",
        "merge_small_clusters",
        "singleton_merge_strategy",
        "singleton_merge_distance_threshold",
        "min_cluster_size",
        "feature_aware_enable",
        "feature_aware_weight",
        "hidden_dim",
        "dropout",
        "batch_size",
        "penalty_selector_enable",
        "fusion_gate_enable",
        "channel_expert_adapters_enable",
        "channel_expert_adapters_mode_type",
        "channel_expert_adapters_delta_init",
        "cluster_penalty_prior_enable",
        "cluster_penalty_prior_topk",
        "channel_penalty_prior_enable",
        "channel_penalty_prior_topk",
        "select_ranks",
        "config_path",
        "out_dir",
        "seconds",
        "returncode",
    ]
    results_path = out_root / f"{args.dataset}_H{args.horizon}_results.csv"
    existing_by_name: dict[str, dict[str, Any]] = {}
    if results_path.exists():
        with results_path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if row.get("name"):
                    existing_by_name[str(row["name"])] = dict(row)
    selected_candidates = candidates()
    if args.candidates:
        requested = set(args.candidates)
        selected_candidates = [cand for cand in selected_candidates if cand["name"] in requested]
        missing = sorted(requested.difference({cand["name"] for cand in selected_candidates}))
        if missing:
            raise ValueError(f"Unknown candidate names: {missing}")
    else:
        selected_candidates = selected_candidates[: max(1, int(args.max_candidates))]

    for cand in selected_candidates:
        cfg = copy.deepcopy(base)
        name = f"{args.dataset}_H{args.horizon}_{cand['name']}"
        cfg.setdefault("exp", {})
        cfg["exp"]["device"] = args.device
        cfg.setdefault("model", {})
        cfg["model"]["predictor"] = cand["predictor"]
        for key, value in cand.get("model", {}).items():
            cfg["model"][key] = value
        if cand.get("train"):
            cfg.setdefault("train", {})
            for key, value in cand.get("train", {}).items():
                cfg["train"][key] = value
        if args.batch_size_override is not None:
            cfg.setdefault("train", {})
            cfg["train"]["batch_size"] = int(args.batch_size_override)
        if args.epochs_override is not None:
            cfg.setdefault("train", {})
            cfg["train"]["epochs"] = int(args.epochs_override)
        if cand.get("window"):
            cfg.setdefault("window", {})
            for key, value in cand.get("window", {}).items():
                cfg["window"][key] = value
        if cand.get("penalties"):
            cfg.setdefault("penalties", {})
            deep_update(cfg["penalties"], cand["penalties"])
        cfg.setdefault("cluster", {})
        for key, value in cand["cluster"].items():
            cfg["cluster"][key] = value
        moe_overrides = cand.get("moe", {})
        if moe_overrides:
            cfg.setdefault("moe", {})
            deep_update(cfg["moe"], moe_overrides)
        pred_residual_overrides = cand.get("pred_residual", {})
        if pred_residual_overrides:
            cfg.setdefault("moe", {})
            cfg["moe"].setdefault("pred_side_residual", {})
            for key, value in pred_residual_overrides.items():
                cfg["moe"]["pred_side_residual"][key] = value
        out_dir = out_root / "runs" / name
        config_path = out_root / "configs" / f"{name}.yaml"
        set_paths(cfg, out_dir, name)
        write_yaml(config_path, cfg)
        t0 = time.perf_counter()
        if args.skip_existing and (out_dir / "run_summary.json").exists():
            rc = 0
        else:
            rc = run_one(config_path, out_dir)
        sec = time.perf_counter() - t0
        metrics = load_metrics(out_dir)
        row = {
            "name": name,
            "status": "ok" if rc == 0 else "failed",
            **metrics,
            "predictor": cfg["model"]["predictor"],
            "distance_threshold": cfg.get("cluster", {}).get("distance_threshold", ""),
            "merge_small_clusters": cfg.get("cluster", {}).get("merge_small_clusters", ""),
            "singleton_merge_strategy": cfg.get("cluster", {}).get("singleton_merge_strategy", ""),
            "singleton_merge_distance_threshold": cfg.get("cluster", {}).get("singleton_merge_distance_threshold", ""),
            "min_cluster_size": cfg.get("cluster", {}).get("min_cluster_size", ""),
            "feature_aware_enable": cfg.get("cluster", {}).get("feature_aware", {}).get("enable", ""),
            "feature_aware_weight": cfg.get("cluster", {}).get("feature_aware", {}).get("weight", ""),
            "hidden_dim": cfg.get("model", {}).get("hidden_dim", ""),
            "dropout": cfg.get("model", {}).get("dropout", ""),
            "batch_size": cfg.get("train", {}).get("batch_size", ""),
            "penalty_selector_enable": cfg.get("moe", {}).get("pred_side_residual", {}).get("penalty_selector_enable", ""),
            "fusion_gate_enable": cfg.get("moe", {}).get("pred_side_residual", {}).get("fusion_gate_enable", ""),
            "channel_expert_adapters_enable": cfg.get("moe", {}).get("pred_side_residual", {}).get("channel_expert_adapters", {}).get("enable", ""),
            "channel_expert_adapters_mode_type": cfg.get("moe", {}).get("pred_side_residual", {}).get("channel_expert_adapters", {}).get("mode_type", ""),
            "channel_expert_adapters_delta_init": cfg.get("moe", {}).get("pred_side_residual", {}).get("channel_expert_adapters", {}).get("delta_init", ""),
            "cluster_penalty_prior_enable": cfg.get("moe", {}).get("cluster_penalty_prior", {}).get("enable", ""),
            "cluster_penalty_prior_topk": cfg.get("moe", {}).get("cluster_penalty_prior", {}).get("topk", ""),
            "channel_penalty_prior_enable": cfg.get("moe", {}).get("channel_penalty_prior", {}).get("enable", ""),
            "channel_penalty_prior_topk": cfg.get("moe", {}).get("channel_penalty_prior", {}).get("topk", ""),
            "select_ranks": cfg.get("moe", {}).get("select_ranks", ""),
            "config_path": str(config_path),
            "out_dir": str(out_dir),
            "seconds": sec,
            "returncode": rc,
        }
        existing_by_name[str(row["name"])] = row
        rows = list(existing_by_name.values())
        out_root.mkdir(parents=True, exist_ok=True)
        with results_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        print(f"{name}: {row['status']} test_mse={row.get('test_mse')} test_mae={row.get('test_mae')}", flush=True)
        if rc != 0:
            break


if __name__ == "__main__":
    main()
