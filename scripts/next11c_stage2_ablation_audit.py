import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml


HISTORY_GATE_FEATURE_DIM = 10
HISTORY_BASE_GATE_FEATURE_DIM = 19


def _get(d: Dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    return bool(value)


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _variant_stage(name: str) -> str:
    if name == "anchors":
        return "b_anchors"
    if name == "full":
        return "c_full"
    if name == "moe_only_no_anchors":
        return "d_moe_only_no_anchors"
    return name


def _gate_feature_dim(cfg: Dict[str, Any]) -> int:
    mode = str(_get(cfg, "moe.gate_feature_mode", "history") or "history").lower()
    if mode in {"history_base", "history+base", "input_base", "safe_augmented"}:
        return HISTORY_BASE_GATE_FEATURE_DIM
    return HISTORY_GATE_FEATURE_DIM


def _gate_param_count(*, k: int, p: int, feat_dim: int, hidden_dim: int, allow_skip: bool) -> int:
    if k <= 0 or p <= 0:
        return 0
    per_cluster = feat_dim * hidden_dim + hidden_dim + hidden_dim * p + p
    if allow_skip:
        per_cluster += hidden_dim + 1
    return k * per_cluster


def _pred_residual_input_dim(cfg: Dict[str, Any]) -> int:
    input_len = _as_int(_get(cfg, "window.input_len"), 0)
    pred_len = _as_int(_get(cfg, "window.pred_len"), 0)
    psr = _get(cfg, "moe.pred_side_residual", {}) or {}
    use_y_base = _as_bool(psr.get("use_y_base_input"), True)
    mode = str(psr.get("feature_mode", "legacy") or "legacy").lower()
    if mode == "safe_augmented":
        return input_len + pred_len + 10 + (2 * pred_len if use_y_base else 0)
    return input_len + (pred_len if use_y_base else 0)


def _pred_residual_param_count(*, cfg: Dict[str, Any], k: int, c: int, p: int) -> int:
    psr = _get(cfg, "moe.pred_side_residual", {}) or {}
    if not _as_bool(psr.get("enable"), False) or k <= 0 or p <= 0:
        return 0
    hidden_dim = _as_int(psr.get("corrector_hidden"), 32)
    pred_len = _as_int(_get(cfg, "window.pred_len"), 0)
    input_dim = _pred_residual_input_dim(cfg)
    per_expert = input_dim * hidden_dim + hidden_dim + hidden_dim * pred_len + pred_len + 1 + hidden_dim + 1
    total_experts = k * p
    channel_cfg = psr.get("channel_expert_adapters", {}) or {}
    if _as_bool(channel_cfg.get("enable"), False):
        mode = str(channel_cfg.get("mode", "merged_singletons") or "merged_singletons").lower()
        if mode in {"all", "all_channels"}:
            total_experts += c * p
    if _as_bool(psr.get("penalty_selector_enable"), False):
        selector_input_dim = input_dim * (3 if _as_bool(psr.get("selector_use_cluster_context"), True) else 1)
        total_experts_params = total_experts * per_expert
        total_experts_params += k * (selector_input_dim * p + p)
        return total_experts_params
    if _as_bool(psr.get("fusion_gate_enable"), False):
        fusion_input_dim = input_dim * (3 if _as_bool(psr.get("fusion_use_cluster_context"), True) else 1)
        total_experts_params = total_experts * per_expert
        total_experts_params += k * (fusion_input_dim + 1)
        return total_experts_params
    return total_experts * per_expert


def _config_paths(config_root: Path) -> Iterable[Path]:
    return sorted(config_root.glob("*/*.yaml"))


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else None


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def _artifact_has_loss_components(summary: Optional[Dict[str, Any]]) -> bool:
    if not summary:
        return False
    diag = summary.get("stage2_loss_diagnostics")
    return isinstance(diag, dict) and bool(diag.get("epochs"))


def _artifact_has_gradient_components(summary: Optional[Dict[str, Any]]) -> bool:
    if not summary:
        return False
    diag = summary.get("stage2_loss_diagnostics")
    if not isinstance(diag, dict):
        return False
    groups = diag.get("trainable_parameter_groups")
    return isinstance(groups, dict) and bool(groups)


def _trainable_groups(cfg: Dict[str, Any], pred_residual_count: int, gate_count: int) -> List[str]:
    groups: List[str] = []
    if not _as_bool(_get(cfg, "moe.freeze_backbone"), False):
        groups.append("backbone")
    if gate_count > 0:
        groups.append("gate")
    if pred_residual_count > 0:
        groups.append("pred_residual")
    if _as_bool(_get(cfg, "moe.dynamic_lambda.enable"), False):
        groups.append("dynamic_lambda")
    if _as_bool(_get(cfg, "moe.learnable_lambda.enable"), False):
        groups.append("learnable_lambda")
    return groups


def _infer_cell_dims(run_root: Path) -> Dict[str, Dict[str, int]]:
    dims: Dict[str, Dict[str, int]] = {}
    for summary_path in sorted(run_root.glob("*/*/run_summary.json")):
        summary = _load_json(summary_path)
        if not summary:
            continue
        cell = summary_path.parent.parent.name
        best_epoch = summary.get("best_epoch")
        k = len(best_epoch) if isinstance(best_epoch, list) and best_epoch else 0
        c = len(_get(summary, "val.per_channel_mse", []) or [])
        if k > 0 or c > 0:
            cur = dims.setdefault(cell, {})
            if k > 0:
                cur["num_clusters"] = k
            if c > 0:
                cur["num_channels"] = c
    return dims


def audit_one(config_path: Path, run_root: Path, cell_dims: Optional[Dict[str, Dict[str, int]]] = None) -> Dict[str, Any]:
    cfg = _load_yaml(config_path)
    cell = config_path.parent.name
    variant = config_path.stem
    run_dir = run_root / cell / variant
    summary = _load_json(run_dir / "run_summary.json")
    penalty_names = list(_get(cfg, "penalties.enabled", []) or [])
    p = len(penalty_names)
    best_epoch = (summary or {}).get("best_epoch")
    k = len(best_epoch) if isinstance(best_epoch, list) and best_epoch else 0
    c = len(_get(summary or {}, "val.per_channel_mse", []) or [])
    dims = (cell_dims or {}).get(cell, {})
    if k <= 0:
        fixed_cluster_id = _get(cfg, "cluster.fixed_cluster_id")
        if isinstance(fixed_cluster_id, list) and fixed_cluster_id:
            k = max(int(v) for v in fixed_cluster_id) + 1
        else:
            k = _as_int(dims.get("num_clusters"), _as_int(_get(cfg, "cluster.n_clusters"), 0))
    if c <= 0:
        fixed_cluster_id = _get(cfg, "cluster.fixed_cluster_id")
        if isinstance(fixed_cluster_id, list) and fixed_cluster_id:
            c = len(fixed_cluster_id)
        else:
            c = _as_int(dims.get("num_channels"), _as_int(_get(cfg, "data.num_channels"), 0))

    epochs = _as_int(_get(cfg, "train.epochs"), 0)
    penalty_warmup = _as_int(_get(cfg, "train.penalty_warmup_epochs"), 0)
    raw_selection_start = _get(cfg, "train.model_selection_start_epoch")
    effective_selection_start = _as_int(raw_selection_start, max(1, penalty_warmup + 1))
    if epochs > 0:
        effective_selection_start = max(1, min(effective_selection_start, epochs))
    early_stop_patience = _get(cfg, "early_stop.patience")
    train_patience = _get(cfg, "train.patience", early_stop_patience)
    psr = _get(cfg, "moe.pred_side_residual", {}) or {}
    mse_gate = _get(cfg, "moe.mse_utility_gate_supervision", {}) or {}
    gate_count = _gate_param_count(
        k=k,
        p=p,
        feat_dim=_gate_feature_dim(cfg),
        hidden_dim=_as_int(_get(cfg, "moe.gate_hidden_dim"), 64),
        allow_skip=_as_bool(_get(cfg, "moe.allow_skip"), False),
    )
    adapter_count = _pred_residual_param_count(cfg=cfg, k=k, c=c, p=p)
    trainable_groups = _trainable_groups(cfg, adapter_count, gate_count)
    residual_summary = _get(summary or {}, "moe_residual", {}) or {}
    route_dist = residual_summary.get("effective_route_by_penalty")
    residual_rms = residual_summary.get("residual_base_rms_ratio")
    alpha_mean = residual_summary.get("alpha_mean")
    stage = _variant_stage(variant)
    is_moe_stage2 = _as_bool(_get(cfg, "moe.enable"), False)
    pred_side_enabled = _as_bool(psr.get("enable"), False)
    undertrained = bool(is_moe_stage2 and pred_side_enabled and epochs <= 1)

    return {
        "cell": cell,
        "variant": variant,
        "stage": stage,
        "config_path": str(config_path),
        "run_dir": str(run_dir),
        "run_summary_exists": bool(summary is not None),
        "train_epochs": epochs,
        "train_patience": train_patience,
        "early_stop_patience": early_stop_patience,
        "train_model_selection_start_epoch_raw": raw_selection_start,
        "train_model_selection_start_epoch_effective": effective_selection_start,
        "penalty_warmup_epochs": penalty_warmup,
        "candidate_supervision_weight": _as_float(psr.get("candidate_supervision_weight"), 0.0),
        "candidate_supervision_warmup_epochs": _get(psr, "candidate_supervision.warmup_epochs", None),
        "gate_utility_supervision_enable": _as_bool(mse_gate.get("enable"), False),
        "gate_utility_supervision_weight": _as_float(mse_gate.get("weight"), 0.0),
        "gate_utility_supervision_warmup_epochs": mse_gate.get("warmup_epochs"),
        "skip_supervision_weight": _as_float(_get(cfg, "moe.skip_supervision_weight"), 0.0),
        "moe_freeze_backbone": _as_bool(_get(cfg, "moe.freeze_backbone"), False),
        "moe_enable": is_moe_stage2,
        "pred_side_residual_enable": pred_side_enabled,
        "train_stat_anchor_enable": _as_bool(_get(cfg, "moe.train_stat_anchor_expert.enable"), False),
        "train_residual_anchor_enable": _as_bool(_get(cfg, "moe.train_residual_anchor_expert.enable"), False),
        "penalty_names": penalty_names,
        "num_clusters": k,
        "num_channels": c,
        "trainable_parameter_groups_inferred": trainable_groups,
        "adapter_parameter_count_inferred": adapter_count,
        "gate_parameter_count_inferred": gate_count,
        "residual_experts_receive_gradients_inferred": bool(pred_side_enabled and adapter_count > 0 and epochs > 0),
        "gate_receives_gradients_inferred": bool(gate_count > 0 and epochs > 0),
        "gradient_evidence_logged": _artifact_has_gradient_components(summary),
        "loss_component_evidence_logged": _artifact_has_loss_components(summary),
        "best_checkpoint_configured": str(_get(cfg, "memory.checkpoint_path", run_dir / "best_checkpoint.pt")),
        "memory_save_checkpoint": _as_bool(_get(cfg, "memory.save_checkpoint"), False),
        "best_checkpoint_saved": bool((run_dir / "best_checkpoint.pt").exists()),
        "eval_skip_test": _as_bool(_get(cfg, "eval.skip_test"), True),
        "best_epoch": best_epoch,
        "val_avg_mse": _get(summary or {}, "val.avg_mse"),
        "val_avg_mae": _get(summary or {}, "val.avg_mae"),
        "selected_val_scaled_mse": _get(summary or {}, "moe_residual_selection.val_scaled_avg_mse"),
        "selected_val_scaled_mae": _get(summary or {}, "moe_residual_selection.val_scaled_avg_mae"),
        "residual_base_rms_ratio": residual_rms,
        "alpha_mean": alpha_mean,
        "actual_route_distribution": route_dist,
        "undertrained_stage2_ablation": undertrained,
        "audit_notes": (
            "old artifact lacks gradient/loss-component evidence; infer trainability from config and nonzero residual summary"
            if not _artifact_has_gradient_components(summary)
            else ""
        ),
    }


def write_markdown(rows: List[Dict[str, Any]], path: Path) -> None:
    headers = [
        "cell",
        "stage",
        "epochs",
        "patience",
        "selection_start",
        "warmup",
        "freeze",
        "groups",
        "adapter_params",
        "gate_params",
        "grad_logged",
        "ckpt_saved",
        "skip_test",
        "undertrained",
        "val_mse",
        "val_mae",
    ]
    lines = ["# NEXT-11c Step 0 Stage-2 Config Audit", ""]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for r in rows:
        values = [
            r["cell"],
            r["stage"],
            r["train_epochs"],
            r["train_patience"],
            r["train_model_selection_start_epoch_effective"],
            r["penalty_warmup_epochs"],
            r["moe_freeze_backbone"],
            ",".join(r["trainable_parameter_groups_inferred"]),
            r["adapter_parameter_count_inferred"],
            r["gate_parameter_count_inferred"],
            r["gradient_evidence_logged"],
            r["best_checkpoint_saved"],
            r["eval_skip_test"],
            r["undertrained_stage2_ablation"],
            r["val_avg_mse"],
            r["val_avg_mae"],
        ]
        lines.append("| " + " | ".join(str(v) for v in values) + " |")
    lines.extend(
        [
            "",
            "## Findings",
            "",
            "- `undertrained_stage2_ablation=true` means `moe.enable=true`, `pred_side_residual.enable=true`, and `train.epochs <= 1`.",
            "- `gradient_evidence_logged=false` means the old artifact did not serialize per-module gradient diagnostics; trainability is inferred from config and residual summaries.",
            "- `best_checkpoint_saved=false` means the old Stage-2 run cannot be reloaded for fair post-hoc diagnostics.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit NEXT-8 ETT H96 Stage-2 ablation configs.")
    parser.add_argument("--config-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    cell_dims = _infer_cell_dims(args.run_root)
    rows = [audit_one(path, args.run_root, cell_dims=cell_dims) for path in _config_paths(args.config_root)]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "stage2_config_audit.json"
    md_path = args.out_dir / "stage2_config_audit.md"
    json_path.write_text(json.dumps({"rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(rows, md_path)
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
