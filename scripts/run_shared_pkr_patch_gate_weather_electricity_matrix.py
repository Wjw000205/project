from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "outputs" / "shared_pkr_patch_gate_weather_electricity_matrix_20260711"
DATASETS = ("Weather", "Electricity")
HORIZONS = (96, 192, 336, 720)
PATCH_LEN = 24

ROOT_CONFIG_PREFIX = {
    "Weather": "weather",
    "Electricity": "electricity",
}
WEATHER_PENALTIES = ("amp_under", "delta", "diff_amp", "direction")
ELECTRICITY_H96_PENALTIES = (
    "amp_under",
    "range",
    "delta",
    "diff_amp",
    "direction",
)
REGIME_CONTEXT = {
    "Weather": (96, 144, 1008),
    "Electricity": (24, 96, 168),
}

BANK_TEMPLATE = (
    ROOT
    / "outputs"
    / "shared_moe_cluster_ablation_20260709"
    / "configs"
    / "ETTm1"
    / "H96"
    / "shared_moe_gate96_r64_valonly.yaml"
)
GATE_TEMPLATE = (
    ROOT
    / "outputs"
    / "ettm1_shared_pkr_patch_gate_recall_20260710"
    / "configs"
    / "ETTm1"
    / "H96"
    / "shared_pkr_patch24_regimectx192_384_672_utilitypolicy_ep12_valonly.yaml"
)

BACKBONE_CHECKPOINTS = {
    ("Electricity", 96): ROOT
    / "outputs"
    / "e_h96_bestwd0_ckpt"
    / "runs"
    / "electricity"
    / "H96"
    / "final"
    / "electric_h96params_h256_do0_modelstat_p168_center_anchorres_a10_wd0"
    / "best_checkpoint.pt",
    ("Electricity", 192): ROOT
    / "outputs"
    / "electricity_strict_20260615_backbones"
    / "runs"
    / "electricity"
    / "H192"
    / "final"
    / "electric_h96params_h256_do0_modelstat_p168_center_anchorres_a10_wd0"
    / "best_checkpoint.pt",
    ("Electricity", 336): ROOT
    / "outputs"
    / "electricity_strict_20260615_backbones"
    / "runs"
    / "electricity"
    / "H336"
    / "final"
    / "electric_h96params_h256_do0_modelstat_p168_center_anchorres_a10_wd0"
    / "best_checkpoint.pt",
    ("Electricity", 720): ROOT
    / "outputs"
    / "e_h720_best224_ckpt"
    / "runs"
    / "electricity"
    / "H720"
    / "final"
    / "electric_h720_centerres_h224_a08_wd1e5_do0_bs8"
    / "best_checkpoint.pt",
}


def electricity_model(hidden_dim: int, alpha: float, include_legacy_noops: bool = False) -> dict[str, Any]:
    model: dict[str, Any] = {
        "predictor": "mlp",
        "hidden_dim": hidden_dim,
        "dropout": 0.0,
        "train_stat_adapter": {
            "enable": True,
            "period": 168,
            "mode": "phase_mean",
            "alpha": alpha,
            "blend_target": "prediction",
            "combine_mode": "anchor_plus_prediction",
            "input_center": True,
        },
    }
    if include_legacy_noops:
        model.update(
            {
                "context_channel_head_include_delta": True,
                "channel_head_residual": True,
                "context_channel_head_residual": True,
            }
        )
    return model


ELECTRICITY_MODELS = {
    96: electricity_model(256, 1.0, include_legacy_noops=True),
    192: electricity_model(256, 1.0),
    336: electricity_model(256, 1.0),
    720: electricity_model(224, 0.8),
}

# Frozen from validation before the single authorized test read.
SELECTED_TEST_SOURCE = {
    ("Weather", 96): "baseline",
    ("Weather", 192): "shared_gate",
    ("Weather", 336): "shared_gate",
    ("Weather", 720): "shared_gate",
    ("Electricity", 96): "shared_gate",
    ("Electricity", 192): "shared_gate",
    ("Electricity", 336): "baseline",
    ("Electricity", 720): "shared_gate",
}


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=False)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def repo_path(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def resolve_config_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def penalties_for(dataset: str, horizon: int) -> tuple[str, ...]:
    if dataset == "Electricity" and horizon == 96:
        return ELECTRICITY_H96_PENALTIES
    return WEATHER_PENALTIES


def train_batch_size(dataset: str, horizon: int) -> int:
    if dataset == "Weather":
        return 64
    if horizon <= 192:
        return 64
    if horizon == 336:
        return 32
    return 16


def root_config_path(dataset: str, horizon: int) -> Path:
    return ROOT / "configs" / f"{ROOT_CONFIG_PREFIX[dataset]}_H{horizon}.yaml"


def backbone_checkpoint(dataset: str, horizon: int, cfg: dict[str, Any]) -> Path:
    if dataset == "Electricity":
        return BACKBONE_CHECKPOINTS[(dataset, horizon)]
    return resolve_config_path(cfg["finetune"]["checkpoint_path"])


def output_paths(dataset: str, horizon: int, stage: str) -> tuple[Path, Path]:
    names = {
        "backbone": "frozen_backbone_replay_valonly",
        "bank": "shared_native_pkr_bank_anchorpath_ep6_valonly",
        "bank_replay": "shared_native_pkr_bank_anchorpath_replay_valonly",
        "gate": "shared_pkr_patch24_anchorpath_causalregime_incremental_ep12_valonly",
        "audit": "shared_pkr_patch24_anchorpath_causalregime_incremental_blockaudit6_valonly",
        "score": "shared_pkr_patch24_anchorpath_causalregime_scorecurve_valonly",
        "calibrate": "shared_pkr_patch24_anchorpath_train_tail_calibrated_valonly",
    }
    if stage == "test":
        source = SELECTED_TEST_SOURCE[(dataset, horizon)]
        name = f"selected_{source}_single_test_read"
    else:
        name = names[stage]
    config = OUT_ROOT / "configs" / dataset / f"H{horizon}" / f"{name}.yaml"
    run = OUT_ROOT / "runs" / dataset / f"H{horizon}" / name
    return config, run


def set_output_paths(
    cfg: dict[str, Any], dataset: str, horizon: int, stage: str
) -> tuple[Path, Path]:
    config_path, run_dir = output_paths(dataset, horizon, stage)
    run_rel = repo_path(run_dir)
    cfg["exp"] = copy.deepcopy(cfg.get("exp") or {})
    cfg["exp"].update(
        {
            "name": f"{dataset}_H{horizon}_{run_dir.name}",
            "out_dir": run_rel,
            "seed": 2026,
            "deterministic": True,
            "device": "cuda:0",
        }
    )
    cfg["corr"] = copy.deepcopy(cfg.get("corr") or {})
    cfg["corr"].update({"compute": True, "save_path": f"{run_rel}/corr.npy"})
    cfg["portrait"] = {"enable": False, "out_dir": f"{run_rel}/cluster_portraits"}
    cfg["plot"] = {"enable": False}
    cfg["memory"] = {
        "path": f"{run_rel}/cluster_memory.pt",
        "checkpoint_path": f"{run_rel}/best_checkpoint.pt",
        "enable": False,
        "save_checkpoint": True,
    }
    cfg["eval"] = {"skip_test": True}
    return config_path, run_dir


def configure_penalties(cfg: dict[str, Any], names: tuple[str, ...]) -> None:
    cfg["penalties"]["enabled"] = list(names)
    moe = cfg["moe"]
    moe["lambda_init"] = {name: 0.0 for name in names}
    moe["lambda_min"] = {name: 0.0 for name in names}
    moe["lambda_schedule"] = {name: "none" for name in names}
    moe["gate_init_bias"] = {"enable": True, "values": {"default": 0.0}}
    moe["cluster_penalty_prior"] = {"enable": False}
    moe["channel_penalty_prior"] = {"enable": False}


def enable_best_output_anchor_path(cfg: dict[str, Any]) -> None:
    moe = cfg["moe"]
    moe["pred_side_residual"]["train_with_eval_anchors"] = True
    moe.pop("train_stat_anchor_expert", None)
    moe.pop("train_residual_anchor_expert", None)
    cfg["calendar_residual"] = {"enable": False}


def make_backbone_replay_config(
    dataset: str,
    horizon: int,
) -> tuple[Path, Path, dict[str, Any]]:
    cfg = read_yaml(root_config_path(dataset, horizon))
    checkpoint = backbone_checkpoint(dataset, horizon, cfg)
    if dataset == "Electricity":
        cfg["model"] = copy.deepcopy(ELECTRICITY_MODELS[horizon])
    cfg["window"]["lazy"] = True
    cfg["window"]["past_context"] = True
    cfg["moe"]["enable"] = False
    cfg["moe"].setdefault("dynamic_lambda", {})["enable"] = False
    cfg["moe"].setdefault("learnable_lambda", {})["enable"] = False
    cfg["moe"].setdefault("pred_side_residual", {})["enable"] = False
    cfg["train"].update(
        {
            "epochs": 1,
            "batch_size": train_batch_size(dataset, horizon),
            "lr": 0.0,
            "selection_metric": "val_mse",
            "lr_scheduler": {"name": "none"},
        }
    )
    cfg["early_stop"] = {"patience": 1, "min_delta": 1.0e-6}
    config_path, run_dir = set_output_paths(cfg, dataset, horizon, "backbone")
    cfg["finetune"] = {
        "enable": True,
        "checkpoint_path": repo_path(checkpoint),
        "strict_window": True,
        "strict_model": True,
        "cluster_map": "index",
        "load_model": True,
        "load_gate": False,
        "load_dynamic_lambda": False,
        "load_learnable_lambda": False,
    }
    return config_path, run_dir, cfg


def make_bank_config(
    dataset: str,
    horizon: int,
    bank_template: dict[str, Any],
) -> tuple[Path, Path, dict[str, Any]]:
    cfg = read_yaml(root_config_path(dataset, horizon))
    checkpoint = backbone_checkpoint(dataset, horizon, cfg)
    if dataset == "Electricity":
        cfg["model"] = copy.deepcopy(ELECTRICITY_MODELS[horizon])

    for key in ("moe", "penalties", "train", "early_stop"):
        cfg[key] = copy.deepcopy(bank_template[key])
    if "diagnostics" in bank_template:
        cfg["diagnostics"] = copy.deepcopy(bank_template["diagnostics"])
    else:
        cfg.pop("diagnostics", None)
    names = penalties_for(dataset, horizon)
    configure_penalties(cfg, names)
    cfg["moe"]["shared_across_clusters"] = True
    cfg["moe"]["freeze_backbone"] = True
    enable_best_output_anchor_path(cfg)
    cfg["window"]["lazy"] = True
    cfg["window"]["past_context"] = True
    cfg["train"]["batch_size"] = train_batch_size(dataset, horizon)

    config_path, run_dir = set_output_paths(cfg, dataset, horizon, "bank")
    cfg["finetune"] = {
        "enable": True,
        "checkpoint_path": repo_path(checkpoint),
        "strict_window": True,
        "strict_model": True,
        "strict_pred_residual": False,
        "cluster_map": "index",
        "load_model": True,
        "load_gate": False,
        "load_pred_residual": False,
        "load_dynamic_lambda": False,
        "load_learnable_lambda": False,
    }
    return config_path, run_dir, cfg


def make_gate_config(
    dataset: str,
    horizon: int,
    bank_cfg: dict[str, Any],
    bank_checkpoint: Path,
    gate_template: dict[str, Any],
) -> tuple[Path, Path, dict[str, Any]]:
    cfg = copy.deepcopy(bank_cfg)
    for key in ("moe", "penalties", "train", "early_stop", "diagnostics"):
        if key in gate_template:
            cfg[key] = copy.deepcopy(gate_template[key])
        else:
            cfg.pop(key, None)
    names = penalties_for(dataset, horizon)
    configure_penalties(cfg, names)
    cfg["moe"]["shared_across_clusters"] = True
    cfg["moe"]["freeze_backbone"] = True
    enable_best_output_anchor_path(cfg)

    router = cfg["moe"]["pred_side_residual"]["patch_router"]
    router["patch_len"] = PATCH_LEN
    if horizon > int(cfg["window"]["input_len"]):
        router["short_history_mode"] = "cycle"
    else:
        router.pop("short_history_mode", None)
    router["regime_context"] = {
        "enable": True,
        "lengths": list(REGIME_CONTEXT[dataset]),
    }
    router["expected_mse_weight"] = 1.0
    recall = router["hierarchical_recall"]
    recall.update(
        {
            "proposal_gain_listwise_weight": 1.0,
            "proposal_rescue_ce_weight": 1.0,
            "ranking_ce_weight": 0.0,
            "risk_sign_bce_weight": 1.0,
            "selected_utility_policy_weight": 0.0,
            "selected_adoption_bce_weight": 0.5,
            "selected_adoption_recall_weight": 0.0,
            "selected_false_adopt_weight": 0.0,
            "false_adopt_max_probability": 0.2,
        }
    )
    conditional_risk = recall["expert_conditional_risk"]
    conditional_risk["proposal_topk"] = 2
    conditional_risk["proposal_rescue"] = True
    conditional_risk["adoption_source"] = "benefit_probability"
    conditional_risk["pairwise_rank"].update({"enable": True, "loss_weight": 1.0})
    cfg["train"]["batch_size"] = train_batch_size(dataset, horizon)

    config_path, run_dir = set_output_paths(cfg, dataset, horizon, "gate")
    cfg["finetune"] = {
        "enable": True,
        "checkpoint_path": repo_path(bank_checkpoint),
        "strict_window": True,
        "strict_model": True,
        "strict_pred_residual": False,
        "cluster_map": "index",
        "load_model": True,
        "load_gate": False,
        "load_pred_residual": True,
        "load_dynamic_lambda": False,
        "load_learnable_lambda": False,
    }
    return config_path, run_dir, cfg


def make_bank_replay_config(
    dataset: str,
    horizon: int,
    bank_cfg: dict[str, Any],
    bank_checkpoint: Path,
) -> tuple[Path, Path, dict[str, Any]]:
    cfg = copy.deepcopy(bank_cfg)
    cfg["train"]["epochs"] = 1
    cfg["train"]["batch_size"] = train_batch_size(dataset, horizon)
    cfg["train"]["lr"] = 0.0
    cfg["train"]["lr_scheduler"] = {"name": "none"}
    config_path, run_dir = set_output_paths(cfg, dataset, horizon, "bank_replay")
    cfg["finetune"] = {
        "enable": True,
        "checkpoint_path": repo_path(bank_checkpoint),
        "strict_window": True,
        "strict_model": True,
        "strict_pred_residual": False,
        "cluster_map": "index",
        "load_model": True,
        "load_gate": False,
        "load_pred_residual": True,
        "load_dynamic_lambda": False,
        "load_learnable_lambda": False,
    }
    return config_path, run_dir, cfg


def make_audit_config(
    dataset: str,
    horizon: int,
    gate_cfg: dict[str, Any],
    gate_checkpoint: Path,
) -> tuple[Path, Path, dict[str, Any]]:
    cfg = copy.deepcopy(gate_cfg)
    diagnostics = cfg["moe"]["pred_side_residual"].setdefault("diagnostics", {})
    diagnostics["validation_temporal_blocks"] = 6
    cfg["train"]["epochs"] = 1
    cfg["train"]["batch_size"] = train_batch_size(dataset, horizon)
    cfg["train"]["lr"] = 0.0
    cfg["train"]["lr_scheduler"] = {"name": "none"}
    config_path, run_dir = set_output_paths(cfg, dataset, horizon, "audit")
    cfg["finetune"] = {
        "enable": True,
        "checkpoint_path": repo_path(gate_checkpoint),
        "strict_window": True,
        "strict_model": True,
        "strict_pred_residual": False,
        "cluster_map": "index",
        "load_model": True,
        "load_gate": False,
        "load_pred_residual": True,
        "load_dynamic_lambda": False,
        "load_learnable_lambda": False,
    }
    return config_path, run_dir, cfg


def make_score_curve_config(
    dataset: str,
    horizon: int,
    gate_cfg: dict[str, Any],
    gate_checkpoint: Path,
) -> tuple[Path, Path, dict[str, Any]]:
    cfg = copy.deepcopy(gate_cfg)
    diagnostics = cfg["moe"]["pred_side_residual"].setdefault("diagnostics", {})
    diagnostics["score_threshold_curve"] = True
    diagnostics["score_threshold_curve_max_windows"] = 512
    diagnostics["score_threshold_curve_heads"] = ["executed_risk_score"]
    diagnostics["validation_temporal_blocks"] = 6
    cfg["train"]["epochs"] = 1
    cfg["train"]["batch_size"] = train_batch_size(dataset, horizon)
    cfg["train"]["lr"] = 0.0
    cfg["train"]["lr_scheduler"] = {"name": "none"}
    config_path, run_dir = set_output_paths(cfg, dataset, horizon, "score")
    cfg["finetune"] = {
        "enable": True,
        "checkpoint_path": repo_path(gate_checkpoint),
        "strict_window": True,
        "strict_model": True,
        "strict_pred_residual": False,
        "cluster_map": "index",
        "load_model": True,
        "load_gate": False,
        "load_pred_residual": True,
        "load_dynamic_lambda": False,
        "load_learnable_lambda": False,
    }
    return config_path, run_dir, cfg


def make_calibration_config(
    dataset: str,
    horizon: int,
    gate_cfg: dict[str, Any],
    gate_checkpoint: Path,
) -> tuple[Path, Path, dict[str, Any]]:
    cfg = copy.deepcopy(gate_cfg)
    router = cfg["moe"]["pred_side_residual"]["patch_router"]
    risk = router["hierarchical_recall"]["expert_conditional_risk"]
    risk["temporal_calibration"] = {
        "enable": True,
        "tail_fraction": 0.2,
        "temporal_blocks": 4,
        "purge_windows": int(cfg["window"]["input_len"]) + horizon - 1,
        "min_gain_cost_ratio": 1.0,
        "min_block_net_gain": 0.0,
        "per_penalty": False,
    }
    diagnostics = cfg["moe"]["pred_side_residual"].setdefault("diagnostics", {})
    diagnostics["validation_temporal_blocks"] = 6
    cfg["train"]["epochs"] = 1
    cfg["train"]["batch_size"] = train_batch_size(dataset, horizon)
    cfg["train"]["lr"] = 0.0
    cfg["train"]["lr_scheduler"] = {"name": "none"}
    config_path, run_dir = set_output_paths(cfg, dataset, horizon, "calibrate")
    cfg["finetune"] = {
        "enable": True,
        "checkpoint_path": repo_path(gate_checkpoint),
        "strict_window": True,
        "strict_model": True,
        "strict_pred_residual": False,
        "cluster_map": "index",
        "load_model": True,
        "load_gate": False,
        "load_pred_residual": True,
        "load_dynamic_lambda": False,
        "load_learnable_lambda": False,
    }
    return config_path, run_dir, cfg


def make_test_config(
    dataset: str,
    horizon: int,
    backbone_cfg: dict[str, Any],
    gate_cfg: dict[str, Any],
    gate_checkpoint: Path,
) -> tuple[Path, Path, dict[str, Any]]:
    source = SELECTED_TEST_SOURCE[(dataset, horizon)]
    cfg = copy.deepcopy(backbone_cfg if source == "baseline" else gate_cfg)
    cfg["train"].update(
        {
            "epochs": 1,
            "batch_size": train_batch_size(dataset, horizon),
            "lr": 0.0,
            "selection_metric": "val_mse",
            "lr_scheduler": {"name": "none"},
        }
    )
    cfg["early_stop"] = {"patience": 1, "min_delta": 1.0e-6}

    diagnostics = cfg.setdefault("diagnostics", {})
    diagnostics["save_prediction_intermediates"] = False
    diagnostics.setdefault("stage2_loss_audit", {})["enable"] = False
    diagnostics.setdefault("stage2_route_audit", {})["enable"] = False

    if source == "shared_gate":
        pred_diagnostics = cfg["moe"]["pred_side_residual"].setdefault(
            "diagnostics", {}
        )
        pred_diagnostics.update(
            {
                "enable": False,
                "train_oracle": False,
                "score_threshold_curve": False,
                "validation_temporal_blocks": 0,
            }
        )
        cfg["moe"]["gate_penalty_hit"] = {"enable": False}
        cfg["moe"]["explainability"] = {"enable": False}
        cfg["finetune"] = {
            "enable": True,
            "checkpoint_path": repo_path(gate_checkpoint),
            "strict_window": True,
            "strict_model": True,
            "strict_pred_residual": False,
            "cluster_map": "index",
            "load_model": True,
            "load_gate": False,
            "load_pred_residual": True,
            "load_dynamic_lambda": False,
            "load_learnable_lambda": False,
        }

    config_path, run_dir = set_output_paths(cfg, dataset, horizon, "test")
    cfg["eval"] = {"skip_test": False}
    return config_path, run_dir, cfg


def validate_config(
    cfg: dict[str, Any], dataset: str, horizon: int, stage: str
) -> None:
    assert Path(cfg["data"]["csv_path"]).stem.lower() == dataset.lower()
    assert int(cfg["window"]["input_len"]) == 96
    assert int(cfg["window"]["pred_len"]) == horizon
    assert cfg["window"]["lazy"] is True
    assert cfg["eval"]["skip_test"] is (stage != "test")
    if stage == "test" and SELECTED_TEST_SOURCE[(dataset, horizon)] == "baseline":
        assert cfg["moe"]["enable"] is False
        assert int(cfg["train"]["epochs"]) == 1
        assert float(cfg["train"]["lr"]) == 0.0
        assert cfg["finetune"]["load_model"] is True
        assert cfg["finetune"]["load_gate"] is False
        return
    if stage == "backbone":
        assert cfg["moe"]["enable"] is False
        assert int(cfg["train"]["epochs"]) == 1
        assert float(cfg["train"]["lr"]) == 0.0
        checkpoint = resolve_config_path(cfg["finetune"]["checkpoint_path"])
        assert checkpoint.exists(), checkpoint
        if dataset == "Electricity":
            assert cfg["model"] == ELECTRICITY_MODELS[horizon]
        return
    assert cfg["moe"]["shared_across_clusters"] is True
    assert cfg["moe"]["freeze_backbone"] is True
    names = penalties_for(dataset, horizon)
    assert tuple(cfg["penalties"]["enabled"]) == names
    assert cfg["moe"]["lambda_init"] == {name: 0.0 for name in names}
    assert cfg["moe"]["lambda_min"] == {name: 0.0 for name in names}
    assert cfg["moe"]["lambda_schedule"] == {name: "none" for name in names}
    assert "train_stat_anchor_expert" not in cfg["moe"]
    assert "train_residual_anchor_expert" not in cfg["moe"]
    assert cfg["moe"]["cluster_penalty_prior"]["enable"] is False
    assert cfg["moe"]["pred_side_residual"]["train_with_eval_anchors"] is True
    checkpoint = resolve_config_path(cfg["finetune"]["checkpoint_path"])
    if stage == "bank":
        assert checkpoint.exists(), checkpoint
        assert cfg["finetune"]["load_pred_residual"] is False
        assert int(cfg["train"]["epochs"]) == 6
        if dataset == "Electricity":
            assert cfg["model"] == ELECTRICITY_MODELS[horizon]
        return
    if stage == "bank_replay":
        assert cfg["finetune"]["load_pred_residual"] is True
        assert int(cfg["train"]["epochs"]) == 1
        assert float(cfg["train"]["lr"]) == 0.0
        return

    router = cfg["moe"]["pred_side_residual"]["patch_router"]
    recall = router["hierarchical_recall"]
    risk = recall["expert_conditional_risk"]
    assert router["enable"] is True
    assert int(router["patch_len"]) == PATCH_LEN
    assert horizon % PATCH_LEN == 0
    assert router.get("short_history_mode") == ("cycle" if horizon > 96 else None)
    assert router["regime_context"] == {
        "enable": True,
        "lengths": list(REGIME_CONTEXT[dataset]),
    }
    assert float(router["expected_mse_weight"]) == 1.0
    assert float(recall["proposal_gain_listwise_weight"]) == 1.0
    assert float(recall["proposal_rescue_ce_weight"]) == 1.0
    assert float(recall["ranking_ce_weight"]) == 0.0
    assert float(recall["risk_sign_bce_weight"]) == 1.0
    assert float(recall["selected_adoption_bce_weight"]) == 0.5
    assert float(recall["selected_false_adopt_weight"]) == 0.0
    assert risk["proposal_topk"] == 2
    assert risk["proposal_rescue"] is True
    assert risk["pairwise_rank"]["enable"] is True
    assert float(risk["pairwise_rank"]["loss_weight"]) == 1.0
    assert cfg["finetune"]["load_pred_residual"] is True
    assert int(cfg["train"]["epochs"]) == (12 if stage == "gate" else 1)
    if stage == "test":
        assert float(cfg["train"]["lr"]) == 0.0
        assert cfg["moe"]["pred_side_residual"]["diagnostics"]["enable"] is False
        assert cfg["moe"]["gate_penalty_hit"]["enable"] is False
        assert cfg["moe"]["explainability"]["enable"] is False
    if stage in {"audit", "score"}:
        assert cfg["moe"]["pred_side_residual"]["diagnostics"][
            "validation_temporal_blocks"
        ] == 6
        assert float(cfg["train"]["lr"]) == 0.0
    if stage == "score":
        assert cfg["moe"]["pred_side_residual"]["diagnostics"][
            "score_threshold_curve"
        ] is True
        assert cfg["moe"]["pred_side_residual"]["diagnostics"][
            "score_threshold_curve_max_windows"
        ] == 512
        assert cfg["moe"]["pred_side_residual"]["diagnostics"][
            "score_threshold_curve_heads"
        ] == ["executed_risk_score"]
    if stage == "calibrate":
        calibration = risk["temporal_calibration"]
        assert calibration["enable"] is True
        assert float(calibration["tail_fraction"]) == 0.2
        assert int(calibration["temporal_blocks"]) == 4
        assert calibration["per_penalty"] is False


def prepare_cell(
    dataset: str,
    horizon: int,
    bank_template: dict[str, Any],
    gate_template: dict[str, Any],
) -> dict[str, Path]:
    backbone_config, backbone_run, backbone_cfg = make_backbone_replay_config(
        dataset, horizon
    )
    validate_config(backbone_cfg, dataset, horizon, "backbone")
    write_yaml(backbone_config, backbone_cfg)

    bank_config, bank_run, bank_cfg = make_bank_config(dataset, horizon, bank_template)
    validate_config(bank_cfg, dataset, horizon, "bank")
    write_yaml(bank_config, bank_cfg)
    bank_checkpoint = bank_run / "best_checkpoint.pt"

    bank_replay_config, bank_replay_run, bank_replay_cfg = make_bank_replay_config(
        dataset, horizon, bank_cfg, bank_checkpoint
    )
    validate_config(bank_replay_cfg, dataset, horizon, "bank_replay")
    write_yaml(bank_replay_config, bank_replay_cfg)

    gate_config, gate_run, gate_cfg = make_gate_config(
        dataset, horizon, bank_cfg, bank_checkpoint, gate_template
    )
    validate_config(gate_cfg, dataset, horizon, "gate")
    write_yaml(gate_config, gate_cfg)
    gate_checkpoint = gate_run / "best_checkpoint.pt"

    audit_config, audit_run, audit_cfg = make_audit_config(
        dataset, horizon, gate_cfg, gate_checkpoint
    )
    validate_config(audit_cfg, dataset, horizon, "audit")
    write_yaml(audit_config, audit_cfg)

    score_config, score_run, score_cfg = make_score_curve_config(
        dataset, horizon, gate_cfg, gate_checkpoint
    )
    validate_config(score_cfg, dataset, horizon, "score")
    write_yaml(score_config, score_cfg)

    calibration_config, calibration_run, calibration_cfg = make_calibration_config(
        dataset, horizon, gate_cfg, gate_checkpoint
    )
    validate_config(calibration_cfg, dataset, horizon, "calibrate")
    write_yaml(calibration_config, calibration_cfg)

    test_config, test_run, test_cfg = make_test_config(
        dataset,
        horizon,
        backbone_cfg,
        gate_cfg,
        gate_checkpoint,
    )
    validate_config(test_cfg, dataset, horizon, "test")
    write_yaml(test_config, test_cfg)
    return {
        "backbone_config": backbone_config,
        "backbone_run": backbone_run,
        "bank_config": bank_config,
        "bank_run": bank_run,
        "bank_checkpoint": bank_checkpoint,
        "bank_replay_config": bank_replay_config,
        "bank_replay_run": bank_replay_run,
        "gate_config": gate_config,
        "gate_run": gate_run,
        "gate_checkpoint": gate_checkpoint,
        "audit_config": audit_config,
        "audit_run": audit_run,
        "score_config": score_config,
        "score_run": score_run,
        "calibration_config": calibration_config,
        "calibration_run": calibration_run,
        "test_config": test_config,
        "test_run": test_run,
    }


def trainable_backbone_count(summary: dict[str, Any]) -> int | None:
    groups = summary.get("stage2_trainable_parameter_groups") or {}
    total = groups.get("total") or {}
    value = total.get("backbone")
    return None if value is None else int(value)


def summarize_gate(dataset: str, horizon: int, summary_path: Path) -> dict[str, Any]:
    summary = read_json(summary_path)
    backbone_summary_path = output_paths(dataset, horizon, "backbone")[1] / "run_summary.json"
    backbone_summary = read_json(backbone_summary_path) if backbone_summary_path.exists() else {}
    shared = summary.get("shared_moe") or {}
    patch = ((summary.get("moe_residual") or {}).get("patch_router") or {})
    oracle = patch.get("oracle_diagnostic") or {}
    final_base_mse = (backbone_summary.get("val") or {}).get("avg_mse")
    final_selected_mse = (summary.get("val") or {}).get("avg_mse")
    final_gain_pct = None
    if final_base_mse is not None and final_selected_mse is not None and final_base_mse > 0:
        final_gain_pct = 100.0 * (final_base_mse - final_selected_mse) / final_base_mse
    row = {
        "dataset": dataset,
        "horizon": horizon,
        "summary_path": repo_path(summary_path),
        "best_epoch": shared.get("best_epoch", summary.get("best_epoch")),
        "backbone_trainable": trainable_backbone_count(summary),
        "shared_moe": bool(shared.get("shared_across_clusters")),
        "penalty_names": summary.get("penalty_names"),
        "val_avg_mse": (summary.get("val") or {}).get("avg_mse"),
        "val_avg_mae": (summary.get("val") or {}).get("avg_mae"),
        "final_metric_base_mse": final_base_mse,
        "final_metric_gain_pct": final_gain_pct,
        "base_patch_mse": oracle.get("base_patch_mse"),
        "selected_patch_mse": oracle.get("selected_patch_mse"),
        "oracle_patch_mse": oracle.get("oracle_patch_mse"),
        "selected_gain_pct": oracle.get("selected_gain_pct"),
        "oracle_gain_pct": oracle.get("oracle_gain_pct"),
        "proposal_oracle_best_recall_at_k": oracle.get(
            "proposal_oracle_best_recall_at_k"
        ),
        "selected_utility_recall": oracle.get("selected_utility_recall"),
        "selected_utility_precision": oracle.get("selected_utility_precision"),
        "selected_gain_to_cost_ratio": oracle.get("selected_gain_to_cost_ratio"),
        "shortlist_pairwise_accuracy": oracle.get("shortlist_pairwise_accuracy"),
        "skip_rate": (oracle.get("selected_class_rate") or {}).get("skip"),
    }
    gain = row["selected_gain_pct"]
    oracle_gain = row["oracle_gain_pct"]
    proposal = row["proposal_oracle_best_recall_at_k"]
    if final_gain_pct is not None and final_gain_pct <= 0.0:
        row["diagnosis"] = "negative_final_metric_despite_patch_diagnostic"
    elif oracle_gain is not None and oracle_gain <= 0.1:
        row["diagnosis"] = "candidate_quality_or_no_oracle_space"
    elif gain is not None and gain > 0.0:
        row["diagnosis"] = "positive_selected_utility"
    elif proposal is not None and proposal < 0.5:
        row["diagnosis"] = "proposal_recall_or_routing_target"
    else:
        row["diagnosis"] = "risk_selection_or_train_val_shift"
    return row


def add_temporal_audit(row: dict[str, Any], summary_path: Path) -> dict[str, Any]:
    summary = read_json(summary_path)
    patch = ((summary.get("moe_residual") or {}).get("patch_router") or {})
    temporal = patch.get("validation_temporal_block_metrics") or {}
    blocks = temporal.get("blocks") or []
    gains = [float(block["selected_gain_pct"]) for block in blocks]
    row.update(
        {
            "audit_summary_path": repo_path(summary_path),
            "temporal_block_gains_pct": gains,
            "positive_temporal_blocks": sum(gain > 0.0 for gain in gains),
            "worst_temporal_block_gain_pct": min(gains) if gains else None,
            "best_temporal_block_gain_pct": max(gains) if gains else None,
        }
    )
    return row


def update_matrix_result(row: dict[str, Any]) -> None:
    path = OUT_ROOT / "matrix_results.json"
    payload = {"protocol": {}, "results": []}
    if path.exists():
        payload = read_json(path)
    payload["protocol"] = {
        "test_read": False,
        "backbone_frozen": True,
        "shared_across_clusters": True,
        "base_path": "root_train_stat_plus_train_residual_anchors",
        "patch_len": PATCH_LEN,
        "regime_context_lengths": {
            dataset: list(lengths) for dataset, lengths in REGIME_CONTEXT.items()
        },
        "bank_epochs": 6,
        "gate_epochs": 12,
        "gate_objective": {
            "expected_mse_weight": 1.0,
            "proposal_gain_listwise_weight": 1.0,
            "proposal_rescue_ce_weight": 1.0,
            "risk_sign_bce_weight": 1.0,
            "pairwise_rank_weight": 1.0,
            "selected_adoption_bce_weight": 0.5,
        },
    }
    results = [
        item
        for item in payload.get("results", [])
        if (item.get("dataset"), item.get("horizon"))
        != (row["dataset"], row["horizon"])
    ]
    results.append(row)
    payload["results"] = sorted(
        results, key=lambda item: (item["dataset"], item["horizon"])
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2)
        handle.write("\n")


def summarize_test(
    dataset: str,
    horizon: int,
    config_path: Path,
    summary_path: Path,
) -> dict[str, Any]:
    cfg = read_yaml(config_path)
    summary = read_json(summary_path)
    test = summary.get("test") or {}
    shared = summary.get("shared_moe") or {}
    pre_moe_base_mse = test.get(
        "pre_moe_base_avg_mse", test.get("base_avg_mse")
    )
    selected_mse = test.get("avg_mse")
    internal_gain_pct = test.get(
        "gain_pct_vs_pre_moe_base", test.get("gain_pct_vs_base")
    )
    source = SELECTED_TEST_SOURCE[(dataset, horizon)]
    if source == "baseline":
        reference_base_mse = selected_mse
        reference_base_source = "selected_noop_system_from_this_single_test_read"
    elif dataset == "Weather":
        reference_summary_path = ROOT / "outputs" / f"weather_H{horizon}" / "run_summary.json"
        reference_summary = read_json(reference_summary_path)
        reference_base_mse = (reference_summary.get("test") or {}).get("avg_mse")
        reference_val_mse = (reference_summary.get("val") or {}).get("avg_mse")
        replay_summary = read_json(
            output_paths(dataset, horizon, "backbone")[1] / "run_summary.json"
        )
        replay_val_mse = (replay_summary.get("val") or {}).get("avg_mse")
        if abs(float(reference_val_mse) - float(replay_val_mse)) > 1.0e-6:
            raise ValueError(
                f"Weather-H{horizon} reference baseline val mismatch: "
                f"{reference_val_mse} vs {replay_val_mse}"
            )
        reference_base_source = repo_path(reference_summary_path)
    else:
        reference_base_mse = pre_moe_base_mse
        reference_base_source = "same_forward_pre_moe_path_no_output_anchor"
    reference_gain_pct = float(
        100.0
        * (float(reference_base_mse) - float(selected_mse))
        / max(abs(float(reference_base_mse)), 1.0e-12)
    )
    return {
        "dataset": dataset,
        "horizon": horizon,
        "selected_source": source,
        "config_path": repo_path(config_path),
        "summary_path": repo_path(summary_path),
        "checkpoint_path": cfg["finetune"]["checkpoint_path"],
        "shared_across_clusters": bool(shared.get("shared_across_clusters")),
        "backbone_trainable": trainable_backbone_count(summary),
        "val_avg_mse": (summary.get("val") or {}).get("avg_mse"),
        "test_pre_moe_base_avg_mse": pre_moe_base_mse,
        "test_internal_gain_pct_vs_pre_moe": internal_gain_pct,
        "test_reference_base_avg_mse": reference_base_mse,
        "test_reference_base_source": reference_base_source,
        "test_selected_avg_mse": selected_mse,
        "test_selected_avg_mae": test.get("avg_mae"),
        "test_gain_pct_vs_reference_base": reference_gain_pct,
    }


def update_single_test_result(row: dict[str, Any]) -> None:
    path = OUT_ROOT / "single_test_results.json"
    payload = {"protocol": {}, "results": []}
    if path.exists():
        payload = read_json(path)
    payload["protocol"] = {
        "selection_frozen_before_test": True,
        "selection_basis": "validation matrix and temporal-block audit",
        "test_reads_per_cell": 1,
        "paired_pre_moe_metric": "accumulated from the pre-MoE prediction in the same test forward pass",
        "weather_reference_baseline": "existing root run_summary whose validation MSE exactly matches the frozen matrix baseline; no new test read",
        "electricity_reference_baseline": "same-forward pre-MoE path because Electricity has no default output-anchor expert",
        "baseline_cell_comparison": "the selected baseline itself",
        "test_used_for_tuning": False,
    }
    results = [
        item
        for item in payload.get("results", [])
        if (item.get("dataset"), item.get("horizon"))
        != (row["dataset"], row["horizon"])
    ]
    results.append(row)
    payload["results"] = sorted(
        results, key=lambda item: (item["dataset"], item["horizon"])
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2)
        handle.write("\n")

    lines = [
        "# Frozen-selection single test read",
        "",
        "The system choice was frozen from validation before test. Each cell traversed the test loader once; the paired base metric was accumulated in that same forward pass.",
        "",
        "| Dataset | H | Selected | Base test MSE | Selected test MSE | Gain vs base | Test MAE |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for item in payload["results"]:
        lines.append(
            "| {dataset} | {horizon} | {selected_source} | {base:.6f} | "
            "{selected:.6f} | {gain:+.4f}% | {mae:.6f} |".format(
                dataset=item["dataset"],
                horizon=item["horizon"],
                selected_source=item["selected_source"],
                base=float(
                    item.get(
                        "test_reference_base_avg_mse",
                        item.get("test_comparison_base_avg_mse"),
                    )
                ),
                selected=float(item["test_selected_avg_mse"]),
                gain=float(
                    item.get(
                        "test_gain_pct_vs_reference_base",
                        item.get("test_gain_pct_vs_comparison_base"),
                    )
                ),
                mae=float(item["test_selected_avg_mae"]),
            )
        )
    summary_path = OUT_ROOT / "single_test_summary.md"
    with summary_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines) + "\n")


def run_training(python: str, config_path: Path, run_dir: Path, force: bool) -> None:
    summary_path = run_dir / "run_summary.json"
    if summary_path.exists() and not force:
        print(f"[reuse] {repo_path(summary_path)}", flush=True)
        return
    command = [python, "-u", "-m", "src.train", "--config", str(config_path)]
    print(f"[run] {' '.join(command)}", flush=True)
    started = time.time()
    env = dict(os.environ)
    env.update({"PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"})
    completed = subprocess.run(command, cwd=ROOT, env=env)
    elapsed = time.time() - started
    if completed.returncode != 0:
        raise RuntimeError(
            f"training failed ({completed.returncode}) after {elapsed:.1f}s: {config_path}"
        )
    if not summary_path.exists():
        raise RuntimeError(f"run_summary.json missing after successful command: {summary_path}")
    print(f"[done] {dataset_horizon(run_dir)} in {elapsed:.1f}s", flush=True)


def dataset_horizon(run_dir: Path) -> str:
    relative = run_dir.relative_to(OUT_ROOT / "runs")
    return f"{relative.parts[0]}-{relative.parts[1]}-{relative.parts[2]}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the frozen shared native-PKR patch-gate Weather/Electricity matrix."
    )
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument(
        "--horizons", nargs="+", type=int, choices=HORIZONS, default=list(HORIZONS)
    )
    parser.add_argument(
        "--stage",
        choices=(
            "prepare",
            "backbone",
            "bank",
            "bank-replay",
            "gate",
            "audit",
            "score",
            "calibrate",
            "test",
            "all",
        ),
        default="prepare",
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bank_template = read_yaml(BANK_TEMPLATE)
    gate_template = read_yaml(GATE_TEMPLATE)
    cells = [(dataset, horizon) for dataset in args.datasets for horizon in args.horizons]
    prepared = {
        cell: prepare_cell(cell[0], cell[1], bank_template, gate_template) for cell in cells
    }
    print(f"[prepared] {len(prepared)} cells under {repo_path(OUT_ROOT)}", flush=True)
    if args.stage == "prepare":
        return 0

    for dataset, horizon in cells:
        paths = prepared[(dataset, horizon)]
        if args.stage == "test":
            source = SELECTED_TEST_SOURCE[(dataset, horizon)]
            selected_checkpoint = (
                paths["gate_checkpoint"]
                if source == "shared_gate"
                else resolve_config_path(
                    read_yaml(paths["test_config"])["finetune"]["checkpoint_path"]
                )
            )
            if not selected_checkpoint.exists():
                raise FileNotFoundError(
                    f"selected {source} checkpoint missing for {dataset}-H{horizon}: "
                    f"{selected_checkpoint}"
                )
            run_training(
                args.python,
                paths["test_config"],
                paths["test_run"],
                args.force,
            )
            row = summarize_test(
                dataset,
                horizon,
                paths["test_config"],
                paths["test_run"] / "run_summary.json",
            )
            update_single_test_result(row)
            print(
                "[test] " + json.dumps(row, ensure_ascii=True, sort_keys=True),
                flush=True,
            )
            continue
        if args.stage == "backbone":
            run_training(
                args.python, paths["backbone_config"], paths["backbone_run"], args.force
            )
            continue
        if args.stage in {"bank", "all"}:
            run_training(args.python, paths["bank_config"], paths["bank_run"], args.force)
        if args.stage == "bank-replay":
            if not paths["bank_checkpoint"].exists():
                raise FileNotFoundError(
                    f"shared bank checkpoint missing for {dataset}-H{horizon}: "
                    f"{paths['bank_checkpoint']}"
                )
            run_training(
                args.python,
                paths["bank_replay_config"],
                paths["bank_replay_run"],
                args.force,
            )
        if args.stage in {"gate", "all"}:
            if not paths["bank_checkpoint"].exists():
                raise FileNotFoundError(
                    f"shared bank checkpoint missing for {dataset}-H{horizon}: "
                    f"{paths['bank_checkpoint']}"
                )
            run_training(args.python, paths["gate_config"], paths["gate_run"], args.force)
            row = summarize_gate(dataset, horizon, paths["gate_run"] / "run_summary.json")
            update_matrix_result(row)
            print("[result] " + json.dumps(row, ensure_ascii=True, sort_keys=True), flush=True)
        if args.stage in {"audit", "all"}:
            if not paths["gate_checkpoint"].exists():
                raise FileNotFoundError(
                    f"patch-gate checkpoint missing for {dataset}-H{horizon}: "
                    f"{paths['gate_checkpoint']}"
                )
            run_training(args.python, paths["audit_config"], paths["audit_run"], args.force)
            audit_summary_path = paths["audit_run"] / "run_summary.json"
            row = summarize_gate(dataset, horizon, audit_summary_path)
            trained_gate_summary_path = paths["gate_run"] / "run_summary.json"
            trained_gate_summary = read_json(trained_gate_summary_path)
            trained_shared = trained_gate_summary.get("shared_moe") or {}
            row["best_epoch"] = trained_shared.get(
                "best_epoch",
                trained_gate_summary.get("best_epoch"),
            )
            row["trained_gate_summary_path"] = repo_path(trained_gate_summary_path)
            row = add_temporal_audit(row, audit_summary_path)
            update_matrix_result(row)
            print("[audit] " + json.dumps(row, ensure_ascii=True, sort_keys=True), flush=True)
        if args.stage == "score":
            if not paths["gate_checkpoint"].exists():
                raise FileNotFoundError(
                    f"patch-gate checkpoint missing for {dataset}-H{horizon}: "
                    f"{paths['gate_checkpoint']}"
                )
            run_training(args.python, paths["score_config"], paths["score_run"], args.force)
            score_summary = read_json(paths["score_run"] / "run_summary.json")
            curve = (
                ((score_summary.get("moe_residual") or {}).get("patch_router") or {}).get(
                    "score_threshold_curve"
                )
                or {}
            )
            compact = {
                split: {
                    key: value
                    for key, value in (curve.get(split) or {}).items()
                    if key not in {"per_channel", "per_penalty", "temporal_blocks"}
                }
                for split in ("train", "validation")
            }
            print("[score] " + json.dumps(compact, ensure_ascii=True, sort_keys=True), flush=True)
        if args.stage == "calibrate":
            if not paths["gate_checkpoint"].exists():
                raise FileNotFoundError(
                    f"patch-gate checkpoint missing for {dataset}-H{horizon}: "
                    f"{paths['gate_checkpoint']}"
                )
            run_training(
                args.python,
                paths["calibration_config"],
                paths["calibration_run"],
                args.force,
            )
            calibration_summary_path = paths["calibration_run"] / "run_summary.json"
            row = summarize_gate(dataset, horizon, calibration_summary_path)
            row = add_temporal_audit(row, calibration_summary_path)
            calibration_summary = read_json(calibration_summary_path)
            row["temporal_calibration"] = (
                (((calibration_summary.get("moe_residual") or {}).get("patch_router") or {}).get(
                    "temporal_calibration"
                ))
            )
            print(
                "[calibrate] " + json.dumps(row, ensure_ascii=True, sort_keys=True),
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
