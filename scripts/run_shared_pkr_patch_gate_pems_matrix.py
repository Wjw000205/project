from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "outputs" / "shared_pkr_patch_gate_pems_matrix_20260711"
DATASETS = ("PEMS03", "PEMS04", "PEMS07", "PEMS08")
HORIZONS = (12, 24, 48, 96)
PENALTIES = ("amp_under", "delta", "diff_amp", "direction")
PATCH_LEN = 12
REGIME_CONTEXT = (96, 288, 2016)

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

EXISTING_BANKS: dict[tuple[str, int], Path] = {}


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


def backbone_checkpoint_path(dataset: str, horizon: int) -> Path:
    if dataset == "PEMS08" and horizon == 96:
        return (
            ROOT
            / "outputs"
            / "pems08_h96_backbone_capacity"
            / "runs"
            / "hid192_b2"
            / "best_checkpoint.pt"
        )
    return (
        ROOT
        / "outputs"
        / "pems_depth_rollout"
        / "runs"
        / f"{dataset}_H{horizon}_hid192_b2"
        / "best_checkpoint.pt"
    )


def output_paths(
    dataset: str,
    horizon: int,
    stage: str,
    variant_tag: str = "",
) -> tuple[Path, Path]:
    names = {
        "bank": "shared_four_pkr_bank_ep6_valonly",
        "gate": "shared_pkr_patch12_regimectx96_288_2016_incremental_ep12_valonly",
        "audit": "shared_pkr_patch12_regimectx96_288_2016_incremental_blockaudit6_valonly",
    }
    name = names[stage]
    if variant_tag and stage in {"gate", "audit"}:
        name = name.removesuffix("_valonly") + f"_{variant_tag}_valonly"
    config = OUT_ROOT / "configs" / dataset / f"H{horizon}" / f"{name}.yaml"
    run = OUT_ROOT / "runs" / dataset / f"H{horizon}" / name
    return config, run


def set_output_paths(
    cfg: dict[str, Any],
    dataset: str,
    horizon: int,
    stage: str,
    variant_tag: str = "",
) -> tuple[Path, Path]:
    config_path, run_dir = output_paths(dataset, horizon, stage, variant_tag)
    run_rel = repo_path(run_dir)
    suffix = {
        "bank": "bank_ep6",
        "gate": "gate_ep12",
        "audit": "gate_blockaudit6",
    }[stage]
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


def disable_output_anchors(cfg: dict[str, Any]) -> None:
    moe = cfg["moe"]
    residual = moe["pred_side_residual"]
    residual["train_with_eval_anchors"] = False
    moe["train_stat_anchor_expert"] = {"enable": False}
    moe["train_residual_anchor_expert"] = {"enable": False}
    cfg["calendar_residual"] = {"enable": False}


def configure_pems_penalties(cfg: dict[str, Any]) -> None:
    cfg["penalties"]["enabled"] = list(PENALTIES)
    moe = cfg["moe"]
    moe["lambda_init"] = {name: 0.0 for name in PENALTIES}
    moe["lambda_min"] = {name: 0.0 for name in PENALTIES}
    moe["lambda_schedule"] = {name: "none" for name in PENALTIES}
    moe["gate_init_bias"] = {"enable": True, "values": {"default": 0.0}}
    moe["cluster_penalty_prior"] = {"enable": False}


def make_bank_config(
    dataset: str,
    horizon: int,
    bank_template: dict[str, Any],
) -> tuple[Path, Path, dict[str, Any]]:
    base_path = ROOT / "configs" / f"{dataset}_H{horizon}.yaml"
    cfg = read_yaml(base_path)
    backbone_finetune = copy.deepcopy(cfg.get("finetune") or {})
    backbone_finetune["checkpoint_path"] = repo_path(
        backbone_checkpoint_path(dataset, horizon)
    )

    for key in ("moe", "penalties", "train", "early_stop"):
        cfg[key] = copy.deepcopy(bank_template[key])
    if "diagnostics" in bank_template:
        cfg["diagnostics"] = copy.deepcopy(bank_template["diagnostics"])
    else:
        cfg.pop("diagnostics", None)
    configure_pems_penalties(cfg)
    cfg["moe"]["shared_across_clusters"] = True
    cfg["moe"]["freeze_backbone"] = True
    disable_output_anchors(cfg)

    config_path, run_dir = set_output_paths(cfg, dataset, horizon, "bank")
    backbone_finetune.update(
        {
            "enable": True,
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
    )
    cfg["finetune"] = backbone_finetune
    return config_path, run_dir, cfg


def make_gate_config(
    dataset: str,
    horizon: int,
    bank_cfg: dict[str, Any],
    bank_checkpoint: Path,
    gate_template: dict[str, Any],
    selected_false_adopt_weight: float,
    false_adopt_max_probability: float,
    proposal_topk: int,
    ranking_ce_weight: float,
    variant_tag: str,
) -> tuple[Path, Path, dict[str, Any]]:
    cfg = copy.deepcopy(bank_cfg)
    for key in ("moe", "penalties", "train", "early_stop", "diagnostics"):
        if key in gate_template:
            cfg[key] = copy.deepcopy(gate_template[key])
        else:
            cfg.pop(key, None)
    configure_pems_penalties(cfg)
    cfg["moe"]["shared_across_clusters"] = True
    cfg["moe"]["freeze_backbone"] = True
    disable_output_anchors(cfg)

    router = cfg["moe"]["pred_side_residual"]["patch_router"]
    router["patch_len"] = PATCH_LEN
    router.pop("short_history_mode", None)
    router["regime_context"] = {
        "enable": True,
        "lengths": list(REGIME_CONTEXT),
    }
    router["expected_mse_weight"] = 1.0
    recall = router["hierarchical_recall"]
    recall.update(
        {
            "proposal_gain_listwise_weight": 1.0,
            "proposal_rescue_ce_weight": 1.0,
            "ranking_ce_weight": ranking_ce_weight,
            "risk_sign_bce_weight": 1.0,
            "selected_utility_policy_weight": 0.0,
            "selected_adoption_bce_weight": 0.5,
            "selected_adoption_recall_weight": 0.0,
            "selected_false_adopt_weight": selected_false_adopt_weight,
            "false_adopt_max_probability": false_adopt_max_probability,
        }
    )
    conditional_risk = recall["expert_conditional_risk"]
    conditional_risk["proposal_topk"] = proposal_topk
    conditional_risk["proposal_rescue"] = proposal_topk == 2
    recall["proposal_rescue_ce_weight"] = 1.0 if proposal_topk == 2 else 0.0
    conditional_risk["adoption_source"] = "benefit_probability"
    conditional_risk["pairwise_rank"].update({"enable": True, "loss_weight": 1.0})

    config_path, run_dir = set_output_paths(cfg, dataset, horizon, "gate", variant_tag)
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
    variant_tag: str,
) -> tuple[Path, Path, dict[str, Any]]:
    cfg = copy.deepcopy(gate_cfg)
    diagnostics = cfg["moe"]["pred_side_residual"].setdefault("diagnostics", {})
    diagnostics["validation_temporal_blocks"] = 6
    cfg["train"]["epochs"] = 1
    cfg["train"]["batch_size"] = 64 if dataset == "PEMS07" else 256
    cfg["train"]["lr"] = 0.0
    cfg["train"]["lr_scheduler"] = {"name": "none"}
    config_path, run_dir = set_output_paths(cfg, dataset, horizon, "audit", variant_tag)
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


def validate_config(
    cfg: dict[str, Any],
    dataset: str,
    horizon: int,
    stage: str,
    selected_false_adopt_weight: float = 0.0,
    false_adopt_max_probability: float = 0.2,
    proposal_topk: int = 2,
    ranking_ce_weight: float = 0.0,
) -> None:
    assert Path(cfg["data"]["csv_path"]).stem == dataset
    assert int(cfg["window"]["input_len"]) == 96
    assert int(cfg["window"]["pred_len"]) == horizon
    assert cfg["eval"]["skip_test"] is True
    assert cfg["moe"]["shared_across_clusters"] is True
    assert cfg["moe"]["freeze_backbone"] is True
    assert tuple(cfg["penalties"]["enabled"]) == PENALTIES
    assert cfg["moe"]["pred_side_residual"]["train_with_eval_anchors"] is False
    assert cfg["moe"]["train_stat_anchor_expert"]["enable"] is False
    assert cfg["moe"]["train_residual_anchor_expert"]["enable"] is False
    assert cfg["moe"]["cluster_penalty_prior"]["enable"] is False
    assert cfg["model"]["predictor"] == "context_channel_head_mlp"
    assert int(cfg["model"]["hidden_dim"]) == 192
    assert int(cfg["model"]["context_channel_head_blocks"]) == 2
    assert cfg["moe"]["lambda_init"] == {name: 0.0 for name in PENALTIES}
    assert cfg["moe"]["lambda_min"] == {name: 0.0 for name in PENALTIES}
    assert cfg["moe"]["lambda_schedule"] == {name: "none" for name in PENALTIES}
    if stage == "bank":
        backbone_checkpoint = resolve_config_path(cfg["finetune"]["checkpoint_path"])
        assert backbone_checkpoint.exists(), backbone_checkpoint
        assert cfg["finetune"]["load_pred_residual"] is False
        assert int(cfg["train"]["epochs"]) == 6
    if stage in {"gate", "audit"}:
        router = cfg["moe"]["pred_side_residual"]["patch_router"]
        recall = router["hierarchical_recall"]
        risk = recall["expert_conditional_risk"]
        assert router["enable"] is True
        assert int(router["patch_len"]) == PATCH_LEN
        assert horizon % PATCH_LEN == 0
        assert router["regime_context"] == {
            "enable": True,
            "lengths": list(REGIME_CONTEXT),
        }
        assert float(router["expected_mse_weight"]) == 1.0
        assert float(recall["proposal_gain_listwise_weight"]) == 1.0
        assert float(recall["ranking_ce_weight"]) == ranking_ce_weight
        assert float(recall["proposal_rescue_ce_weight"]) == (
            1.0 if proposal_topk == 2 else 0.0
        )
        assert float(recall["risk_sign_bce_weight"]) == 1.0
        assert float(recall["selected_utility_policy_weight"]) == 0.0
        assert float(recall["selected_adoption_bce_weight"]) == 0.5
        assert float(recall["selected_adoption_recall_weight"]) == 0.0
        assert float(recall["selected_false_adopt_weight"]) == selected_false_adopt_weight
        assert float(recall["false_adopt_max_probability"]) == false_adopt_max_probability
        assert risk["proposal_topk"] == proposal_topk
        assert bool(risk["proposal_rescue"]) == (proposal_topk == 2)
        assert risk["pairwise_rank"]["enable"] is True
        assert float(risk["pairwise_rank"]["loss_weight"]) == 1.0
        assert risk["adoption_source"] == "benefit_probability"
        assert cfg["finetune"]["load_pred_residual"] is True
        assert int(cfg["train"]["epochs"]) == (12 if stage == "gate" else 1)
        if stage == "audit":
            assert cfg["moe"]["pred_side_residual"]["diagnostics"][
                "validation_temporal_blocks"
            ] == 6
            assert float(cfg["train"]["lr"]) == 0.0
            assert int(cfg["train"]["batch_size"]) == (
                64 if dataset == "PEMS07" else 256
            )


def prepare_cell(
    dataset: str,
    horizon: int,
    bank_template: dict[str, Any],
    gate_template: dict[str, Any],
    selected_false_adopt_weight: float,
    false_adopt_max_probability: float,
    proposal_topk: int,
    ranking_ce_weight: float,
    variant_tag: str,
) -> dict[str, Path]:
    bank_config_path, bank_run_dir, bank_cfg = make_bank_config(dataset, horizon, bank_template)
    validate_config(bank_cfg, dataset, horizon, "bank")
    write_yaml(bank_config_path, bank_cfg)

    bank_checkpoint = EXISTING_BANKS.get((dataset, horizon), bank_run_dir / "best_checkpoint.pt")
    gate_config_path, gate_run_dir, gate_cfg = make_gate_config(
        dataset,
        horizon,
        bank_cfg,
        bank_checkpoint,
        gate_template,
        selected_false_adopt_weight,
        false_adopt_max_probability,
        proposal_topk,
        ranking_ce_weight,
        variant_tag,
    )
    validate_config(
        gate_cfg,
        dataset,
        horizon,
        "gate",
        selected_false_adopt_weight,
        false_adopt_max_probability,
        proposal_topk,
        ranking_ce_weight,
    )
    write_yaml(gate_config_path, gate_cfg)
    gate_checkpoint = gate_run_dir / "best_checkpoint.pt"
    audit_config_path, audit_run_dir, audit_cfg = make_audit_config(
        dataset,
        horizon,
        gate_cfg,
        gate_checkpoint,
        variant_tag,
    )
    validate_config(
        audit_cfg,
        dataset,
        horizon,
        "audit",
        selected_false_adopt_weight,
        false_adopt_max_probability,
        proposal_topk,
        ranking_ce_weight,
    )
    write_yaml(audit_config_path, audit_cfg)
    return {
        "bank_config": bank_config_path,
        "bank_run": bank_run_dir,
        "bank_checkpoint": bank_checkpoint,
        "gate_config": gate_config_path,
        "gate_run": gate_run_dir,
        "gate_checkpoint": gate_checkpoint,
        "audit_config": audit_config_path,
        "audit_run": audit_run_dir,
    }


def materialize_root_config(dataset: str, horizon: int, source_path: Path) -> Path:
    cfg = read_yaml(source_path)
    recall_cfg = cfg["moe"]["pred_side_residual"]["patch_router"]["hierarchical_recall"]
    assert float(recall_cfg.pop("risk_calibration_weight", 0.0)) == 0.0
    run_rel = f"outputs/{dataset}_H{horizon}"
    cfg["exp"]["name"] = f"{dataset}_H{horizon}"
    cfg["exp"]["out_dir"] = run_rel
    cfg["corr"]["save_path"] = f"{run_rel}/corr.npy"
    cfg["portrait"]["out_dir"] = f"{run_rel}/cluster_portraits"
    cfg["memory"]["path"] = f"{run_rel}/cluster_memory.pt"
    cfg["memory"]["checkpoint_path"] = f"{run_rel}/best_checkpoint.pt"
    cfg["eval"]["skip_test"] = False

    assert Path(cfg["data"]["csv_path"]).stem == dataset
    assert int(cfg["window"]["pred_len"]) == horizon
    assert cfg["moe"]["shared_across_clusters"] is True
    assert cfg["moe"]["freeze_backbone"] is True
    assert tuple(cfg["penalties"]["enabled"]) == PENALTIES
    bank_checkpoint = resolve_config_path(cfg["finetune"]["checkpoint_path"])
    assert bank_checkpoint.exists(), bank_checkpoint

    target_path = ROOT / "configs" / f"{dataset}_H{horizon}.yaml"
    write_yaml(target_path, cfg)
    return target_path


def trainable_backbone_count(summary: dict[str, Any]) -> int | None:
    groups = summary.get("stage2_trainable_parameter_groups") or {}
    total = groups.get("total") or {}
    value = total.get("backbone")
    return None if value is None else int(value)


def summarize_gate(
    dataset: str,
    horizon: int,
    summary_path: Path,
    variant_tag: str,
) -> dict[str, Any]:
    summary = read_json(summary_path)
    shared = summary.get("shared_moe") or {}
    patch = ((summary.get("moe_residual") or {}).get("patch_router") or {})
    oracle = patch.get("oracle_diagnostic") or {}
    row = {
        "dataset": dataset,
        "horizon": horizon,
        "variant": variant_tag or "base",
        "summary_path": repo_path(summary_path),
        "best_epoch": shared.get("best_epoch", summary.get("best_epoch")),
        "backbone_trainable": trainable_backbone_count(summary),
        "shared_moe": bool(shared.get("shared_across_clusters")),
        "penalty_names": summary.get("penalty_names"),
        "val_avg_mse": (summary.get("val") or {}).get("avg_mse"),
        "val_avg_mae": (summary.get("val") or {}).get("avg_mae"),
        "base_patch_mse": oracle.get("base_patch_mse"),
        "selected_patch_mse": oracle.get("selected_patch_mse"),
        "oracle_patch_mse": oracle.get("oracle_patch_mse"),
        "selected_gain_pct": oracle.get("selected_gain_pct"),
        "oracle_gain_pct": oracle.get("oracle_gain_pct"),
        "selected_utility_recall": oracle.get("selected_utility_recall"),
        "selected_utility_precision": oracle.get("selected_utility_precision"),
        "selected_gain_to_cost_ratio": oracle.get("selected_gain_to_cost_ratio"),
        "proposal_oracle_best_recall_at_k": oracle.get("proposal_oracle_best_recall_at_k"),
        "shortlist_pairwise_accuracy": oracle.get("shortlist_pairwise_accuracy"),
        "skip_rate": (oracle.get("selected_class_rate") or {}).get("skip"),
        "selected_false_adopt_weight": (patch.get("hierarchical_recall") or {}).get(
            "selected_false_adopt_weight"
        ),
        "false_adopt_max_probability": (patch.get("hierarchical_recall") or {}).get(
            "false_adopt_max_probability"
        ),
        "proposal_topk": patch.get("expert_risk_proposal_topk"),
        "ranking_ce_weight": (patch.get("hierarchical_recall") or {}).get(
            "ranking_ce_weight"
        ),
    }
    gain = row["selected_gain_pct"]
    oracle_gain = row["oracle_gain_pct"]
    proposal = row["proposal_oracle_best_recall_at_k"]
    if oracle_gain is not None and oracle_gain <= 0.1:
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
    row = dict(row)
    row["variant"] = row.get("variant") or "base"
    payload["protocol"] = {
        "test_read": False,
        "backbone_frozen": True,
        "shared_across_clusters": True,
        "penalties": list(PENALTIES),
        "patch_len": PATCH_LEN,
        "regime_context_lengths": list(REGIME_CONTEXT),
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
    results = []
    for item in payload.get("results", []):
        item = dict(item)
        item["variant"] = item.get("variant") or "base"
        if (item["dataset"], item["horizon"], item["variant"]) != (
            row["dataset"],
            row["horizon"],
            row["variant"],
        ):
            results.append(item)
    results.append(row)
    payload["results"] = sorted(
        results,
        key=lambda item: (item["dataset"], item["horizon"], item["variant"]),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2)
        handle.write("\n")


def run_training(python: str, config_path: Path, run_dir: Path, force: bool) -> None:
    summary_path = run_dir / "run_summary.json"
    if summary_path.exists() and not force:
        print(f"[reuse] {repo_path(summary_path)}", flush=True)
        return
    command = [python, "-u", "-m", "src.train", "--config", str(config_path)]
    print(f"[run] {' '.join(command)}", flush=True)
    started = time.time()
    completed = subprocess.run(command, cwd=ROOT)
    elapsed = time.time() - started
    if completed.returncode != 0:
        raise RuntimeError(f"training failed ({completed.returncode}) after {elapsed:.1f}s: {config_path}")
    if not summary_path.exists():
        raise RuntimeError(f"run_summary.json missing after successful command: {summary_path}")
    print(f"[done] {dataset_horizon(run_dir)} in {elapsed:.1f}s", flush=True)


def dataset_horizon(run_dir: Path) -> str:
    relative = run_dir.relative_to(OUT_ROOT / "runs")
    return f"{relative.parts[0]}-{relative.parts[1]}-{relative.parts[2]}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the frozen shared four-PKR patch-gate PEMS matrix.")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--horizons", nargs="+", type=int, choices=HORIZONS, default=list(HORIZONS))
    parser.add_argument(
        "--stage",
        choices=("prepare", "bank", "gate", "audit", "all", "materialize"),
        default="prepare",
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--selected-false-adopt-weight", type=float, default=0.0)
    parser.add_argument("--false-adopt-max-probability", type=float, default=0.2)
    parser.add_argument("--proposal-topk", type=int, default=2)
    parser.add_argument("--ranking-ce-weight", type=float, default=0.0)
    parser.add_argument("--variant-tag", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.selected_false_adopt_weight < 0.0:
        raise ValueError("--selected-false-adopt-weight must be non-negative.")
    if not 0.0 < args.false_adopt_max_probability < 1.0:
        raise ValueError("--false-adopt-max-probability must be in (0,1).")
    if not 1 <= args.proposal_topk <= len(PENALTIES):
        raise ValueError(f"--proposal-topk must be in [1,{len(PENALTIES)}].")
    if args.ranking_ce_weight < 0.0:
        raise ValueError("--ranking-ce-weight must be non-negative.")
    variant_tag = args.variant_tag.strip()
    if any(not (char.isalnum() or char in {"-", "_"}) for char in variant_tag):
        raise ValueError("--variant-tag may contain only letters, digits, '-' and '_'.")
    if (
        args.selected_false_adopt_weight != 0.0
        or args.false_adopt_max_probability != 0.2
        or args.proposal_topk != 2
        or args.ranking_ce_weight != 0.0
    ) and not variant_tag:
        raise ValueError("Non-default gate parameters require --variant-tag.")
    if args.stage == "materialize" and variant_tag:
        raise ValueError("Root configs can only materialize the selected base variant.")
    bank_template = read_yaml(BANK_TEMPLATE)
    gate_template = read_yaml(GATE_TEMPLATE)
    cells = [(dataset, horizon) for dataset in args.datasets for horizon in args.horizons]
    prepared = {
        cell: prepare_cell(
            cell[0],
            cell[1],
            bank_template,
            gate_template,
            args.selected_false_adopt_weight,
            args.false_adopt_max_probability,
            args.proposal_topk,
            args.ranking_ce_weight,
            variant_tag,
        )
        for cell in cells
    }
    print(f"[prepared] {len(prepared)} cells under {repo_path(OUT_ROOT)}", flush=True)
    if args.stage == "materialize":
        for dataset, horizon in cells:
            target_path = materialize_root_config(
                dataset,
                horizon,
                prepared[(dataset, horizon)]["gate_config"],
            )
            print(f"[materialized] {repo_path(target_path)}", flush=True)
        return 0
    if args.stage == "prepare":
        return 0

    for dataset, horizon in cells:
        paths = prepared[(dataset, horizon)]
        if args.stage in {"bank", "all"} and (dataset, horizon) not in EXISTING_BANKS:
            run_training(args.python, paths["bank_config"], paths["bank_run"], args.force)
        if args.stage in {"gate", "all"}:
            if not paths["bank_checkpoint"].exists():
                raise FileNotFoundError(
                    f"shared bank checkpoint missing for {dataset}-H{horizon}: {paths['bank_checkpoint']}"
                )
            run_training(args.python, paths["gate_config"], paths["gate_run"], args.force)
            row = summarize_gate(
                dataset,
                horizon,
                paths["gate_run"] / "run_summary.json",
                variant_tag,
            )
            update_matrix_result(row)
            print("[result] " + json.dumps(row, ensure_ascii=True, sort_keys=True), flush=True)
        if args.stage in {"audit", "all"}:
            if not paths["gate_checkpoint"].exists():
                raise FileNotFoundError(
                    f"patch-gate checkpoint missing for {dataset}-H{horizon}: "
                    f"{paths['gate_checkpoint']}"
                )
            run_training(args.python, paths["audit_config"], paths["audit_run"], args.force)
            row = summarize_gate(
                dataset,
                horizon,
                paths["gate_run"] / "run_summary.json",
                variant_tag,
            )
            row = add_temporal_audit(row, paths["audit_run"] / "run_summary.json")
            update_matrix_result(row)
            print("[audit] " + json.dumps(row, ensure_ascii=True, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
