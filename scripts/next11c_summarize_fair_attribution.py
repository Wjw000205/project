import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import yaml


VARIANTS = ["a_backbone_eval", "b_anchors", "d_moe_only_no_anchors", "c_full"]


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return data


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def _get(d: Mapping[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = d
    for part in path.split("."):
        if not isinstance(cur, Mapping) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _pct_delta(new: Optional[float], base: Optional[float]) -> Optional[float]:
    if new is None or base is None or base == 0:
        return None
    return (new / base - 1.0) * 100.0


def _gain_pct(base: Optional[float], new: Optional[float]) -> Optional[float]:
    if new is None or base is None or base == 0:
        return None
    return (base - new) / base * 100.0


def _fmt(value: Optional[float], digits: int = 6) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def _fmt_pct(value: Optional[float], digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.{digits}f}%"


def _last_epoch_route(summary: Mapping[str, Any]) -> Dict[str, Any]:
    epochs = _get(summary, "stage2_loss_diagnostics.epochs", []) or []
    if not epochs:
        return {}
    route = epochs[-1].get("route", {}) if isinstance(epochs[-1], Mapping) else {}
    return route if isinstance(route, dict) else {}


def _last_epoch_gradient(summary: Mapping[str, Any]) -> Dict[str, float]:
    epochs = _get(summary, "stage2_loss_diagnostics.epochs", []) or []
    if not epochs:
        return {}
    grad = epochs[-1].get("gradient_l2_mean", {}) if isinstance(epochs[-1], Mapping) else {}
    return grad if isinstance(grad, dict) else {}


def _last5_val_mse_range_pct(summary: Mapping[str, Any]) -> Optional[float]:
    epochs = _get(summary, "stage2_loss_diagnostics.epochs", []) or []
    vals = [
        _as_float(epoch.get("val_mse"))
        for epoch in epochs[-5:]
        if isinstance(epoch, Mapping) and epoch.get("val_mse") is not None
    ]
    vals = [v for v in vals if v is not None]
    if len(vals) < 2:
        return None
    best = min(vals)
    if best == 0:
        return None
    return (max(vals) - min(vals)) / best * 100.0


def _config_for_summary(summary: Mapping[str, Any]) -> Dict[str, Any]:
    raw = summary.get("config_path")
    if not raw:
        return {}
    path = Path(str(raw))
    if not path.exists():
        return {}
    return _load_yaml(path)


def summarize_variant(cell: str, variant: str, root: Path) -> Dict[str, Any]:
    summary_path = root / "runs" / cell / variant / "run_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    summary = _load_json(summary_path)
    cfg = _config_for_summary(summary)
    selection = summary.get("moe_residual_selection", {}) or {}
    residual = summary.get("moe_residual", {}) or {}
    val = summary.get("val", {}) or {}
    test = summary.get("test")
    selected = summary.get("selected", {}) or {}
    route = _last_epoch_route(summary)
    gradient = _last_epoch_gradient(summary)

    is_moe_variant = variant in {"d_moe_only_no_anchors", "c_full"}
    scaled_mse = _as_float(selection.get("val_scaled_avg_mse"))
    scaled_mae = _as_float(selection.get("val_scaled_avg_mae"))
    if is_moe_variant and scaled_mse is not None and scaled_mae is not None:
        final_mse = scaled_mse
        final_mae = scaled_mae
        final_source = "moe_residual_selection.val_scaled"
    else:
        final_mse = _as_float(val.get("avg_mse"))
        final_mae = _as_float(val.get("avg_mae"))
        final_source = "val.avg"
    test_mse = _as_float(selected.get("avg_mse")) if is_moe_variant else None
    test_mae = _as_float(selected.get("avg_mae")) if is_moe_variant else None
    test_source = "selected.avg" if test_mse is not None and test_mae is not None else "test.avg"
    if isinstance(test, Mapping):
        if test_mse is None:
            test_mse = _as_float(test.get("avg_mse"))
        if test_mae is None:
            test_mae = _as_float(test.get("avg_mae"))

    raw_mse = _as_float(selection.get("val_residual_avg_mse"))
    raw_mae = _as_float(selection.get("val_residual_avg_mae"))
    if raw_mse is None:
        raw_mse = _as_float(val.get("avg_mse"))
    if raw_mae is None:
        raw_mae = _as_float(val.get("avg_mae"))
    run_base_mse = _as_float(selection.get("val_pred_base_avg_mse"))
    run_base_mae = _as_float(selection.get("val_pred_base_avg_mae"))

    epochs = _get(summary, "stage2_loss_diagnostics.epochs", []) or []
    configured_epochs = _as_int(_get(cfg, "train.epochs"), 0)
    diag_epochs = len(epochs)
    early_stopped = bool(configured_epochs > 0 and diag_epochs > 0 and diag_epochs < configured_epochs)
    last5_range_pct = _last5_val_mse_range_pct(summary)
    plateau = bool(last5_range_pct is not None and last5_range_pct <= 0.10)
    if configured_epochs == 0:
        sufficiency = "eval-only"
    elif early_stopped:
        sufficiency = f"sufficient: early_stop after {diag_epochs}/{configured_epochs}"
    elif plateau:
        sufficiency = f"sufficient: last5 val_mse range {last5_range_pct:.3f}%"
    else:
        sufficiency = "not_sufficient: no early_stop/no plateau"

    return {
        "cell": cell,
        "variant": variant,
        "root": str(root),
        "summary_path": str(summary_path),
        "config_path": str(summary.get("config_path", "")),
        "skip_test": bool(_get(summary, "eval.skip_test", False)),
        "test_is_null": test is None,
        "configured_epochs": configured_epochs,
        "patience": _get(cfg, "early_stop.patience"),
        "penalty_warmup_epochs": _get(cfg, "train.penalty_warmup_epochs", 0),
        "lr_warmup_epochs": _get(cfg, "train.lr_warmup_epochs", 0),
        "model_selection_start_epoch": _get(cfg, "train.model_selection_start_epoch"),
        "best_epoch": summary.get("best_epoch"),
        "diag_epochs": diag_epochs,
        "best_checkpoint_saved": (summary_path.parent / "best_checkpoint.pt").exists(),
        "final_metric_source": final_source,
        "final_val_mse": final_mse,
        "final_val_mae": final_mae,
        "final_test_mse": test_mse,
        "final_test_mae": test_mae,
        "test_metric_source": test_source,
        "raw_val_mse": raw_mse,
        "raw_val_mae": raw_mae,
        "run_base_mse": run_base_mse,
        "run_base_mae": run_base_mae,
        "raw_route_gain_mse_pct": _gain_pct(run_base_mse, raw_mse),
        "raw_route_gain_mae_pct": _gain_pct(run_base_mae, raw_mae),
        "selected_scaled_gain_mse_pct": _gain_pct(run_base_mse, final_mse),
        "selected_scaled_gain_mae_pct": _gain_pct(run_base_mae, final_mae),
        "route_distribution": residual.get("effective_route_by_penalty"),
        "skip_noop_rate": route.get("skip_noop_rate"),
        "skip_prob": route.get("skip_prob"),
        "residual_delta_rms": residual.get("residual_delta_rms"),
        "residual_base_rms_ratio": residual.get("residual_base_rms_ratio"),
        "trainable_parameter_groups": _get(summary, "stage2_loss_diagnostics.trainable_parameter_groups.total"),
        "last_epoch_gradient_l2_mean": gradient,
        "early_stopped": early_stopped,
        "last5_val_mse_range_pct": last5_range_pct,
        "sufficiency": sufficiency,
    }


def contribution(cell: str, name: str, baseline: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "cell": cell,
        "contribution": name,
        "baseline_variant": baseline["variant"],
        "new_variant": new["variant"],
        "baseline_mse": baseline["final_val_mse"],
        "baseline_mae": baseline["final_val_mae"],
        "new_mse": new["final_val_mse"],
        "new_mae": new["final_val_mae"],
        "mse_delta_abs": (
            new["final_val_mse"] - baseline["final_val_mse"]
            if new["final_val_mse"] is not None and baseline["final_val_mse"] is not None
            else None
        ),
        "mae_delta_abs": (
            new["final_val_mae"] - baseline["final_val_mae"]
            if new["final_val_mae"] is not None and baseline["final_val_mae"] is not None
            else None
        ),
        "mse_delta_pct": _pct_delta(new["final_val_mse"], baseline["final_val_mse"]),
        "mae_delta_pct": _pct_delta(new["final_val_mae"], baseline["final_val_mae"]),
        "raw_route_gain_mse_pct": new.get("raw_route_gain_mse_pct"),
        "selected_scaled_gain_mse_pct": new.get("selected_scaled_gain_mse_pct"),
        "sufficiency": new.get("sufficiency"),
    }


def test_contribution(cell: str, name: str, baseline: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "cell": cell,
        "contribution": name,
        "baseline_variant": baseline["variant"],
        "new_variant": new["variant"],
        "baseline_mse": baseline["final_test_mse"],
        "baseline_mae": baseline["final_test_mae"],
        "new_mse": new["final_test_mse"],
        "new_mae": new["final_test_mae"],
        "mse_delta_pct": _pct_delta(new["final_test_mse"], baseline["final_test_mse"]),
        "mae_delta_pct": _pct_delta(new["final_test_mae"], baseline["final_test_mae"]),
        "val_mse_delta_pct": _pct_delta(new["final_val_mse"], baseline["final_val_mse"]),
        "val_mae_delta_pct": _pct_delta(new["final_val_mae"], baseline["final_val_mae"]),
    }


def load_undertrained_rows(path: Optional[Path]) -> List[Dict[str, Any]]:
    if path is None or not path.exists():
        return []
    audit = _load_json(path)
    rows = audit.get("rows", []) if isinstance(audit.get("rows"), list) else []
    return [
        {
            "cell": row.get("cell"),
            "variant": row.get("stage") or row.get("variant"),
            "train_epochs": row.get("train_epochs"),
            "penalty_warmup_epochs": row.get("penalty_warmup_epochs"),
            "eval_skip_test": row.get("eval_skip_test"),
            "best_checkpoint_saved": row.get("best_checkpoint_saved"),
            "val_avg_mse": row.get("val_avg_mse"),
            "val_avg_mae": row.get("val_avg_mae"),
        }
        for row in rows
        if row.get("undertrained_stage2_ablation")
    ]


def build_report(
    cell_roots: Mapping[str, Path],
    old_audit_path: Optional[Path],
    variant_roots: Optional[Mapping[str, Mapping[str, Path]]] = None,
) -> Dict[str, Any]:
    cells: Dict[str, Dict[str, Any]] = {}
    contribs: List[Dict[str, Any]] = []
    test_contribs: List[Dict[str, Any]] = []
    for cell, root in cell_roots.items():
        overrides = variant_roots.get(cell, {}) if variant_roots is not None else {}
        variants = {
            variant: summarize_variant(cell, variant, overrides.get(variant, root))
            for variant in VARIANTS
        }
        cells[cell] = {
            "root": str(root),
            "variant_roots": {variant: str(overrides.get(variant, root)) for variant in VARIANTS},
            "variants": variants,
        }
        a = variants["a_backbone_eval"]
        b = variants["b_anchors"]
        d = variants["d_moe_only_no_anchors"]
        c = variants["c_full"]
        contribs.extend(
            [
                contribution(cell, "MoE-only contribution (d - a)", a, d),
                contribution(cell, "Anchor contribution (b - a)", a, b),
                contribution(cell, "MoE-on-anchor contribution (c - b)", b, c),
                contribution(cell, "Full pipeline contribution (c - a)", a, c),
            ]
        )
        if all(row["final_test_mse"] is not None for row in (a, b, d, c)):
            test_contribs.extend(
                [
                    test_contribution(cell, "MoE-only contribution (d - a)", a, d),
                    test_contribution(cell, "Anchor contribution (b - a)", a, b),
                    test_contribution(cell, "MoE-on-anchor contribution (c - b)", b, c),
                    test_contribution(cell, "Full pipeline contribution (c - a)", a, c),
                ]
            )
    test_read = bool(test_contribs)
    return {
        "protocol": {
            "selection": "validation only",
            "test_read": test_read,
            "stage2_warmup": "disabled: penalty_warmup_epochs=0 and lr_warmup_epochs=0",
            "final_metric_rule": "a/b use val.avg; d/c use moe_residual_selection.val_scaled when present",
            "test_metric_rule": "use selected.avg when the eval path selects a MoE residual variant; otherwise use test.avg",
            "loss_rule": "stage1 loss is not compared with stage2 total loss",
        },
        "cells": cells,
        "contributions": contribs,
        "test_contributions": test_contribs,
        "undertrained_old_rows_invalid": load_undertrained_rows(old_audit_path),
    }


def markdown(report: Mapping[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# NEXT-11c Fair Stage-2 MoE Ablation Attribution")
    lines.append("")
    test_read = bool(_get(report, "protocol.test_read", False))
    if test_read:
        lines.append("Protocol: schedule was frozen on validation before this single test read. Stage-2 warmup is disabled because the backbone is frozen and Stage-2 trains only adapter/gate. Stage-1 training loss is not compared with Stage-2 total loss.")
    else:
        lines.append("Protocol: validation-only selection; no test read in this report. Stage-2 warmup is disabled because the backbone is frozen and Stage-2 trains only adapter/gate. Stage-1 training loss is not compared with Stage-2 total loss.")
    lines.append("")
    invalid = report.get("undertrained_old_rows_invalid", []) or []
    if invalid:
        lines.append("## Invalid old NEXT-8 Stage-2 rows")
        lines.append("")
        lines.append("| cell | variant | epochs | penalty_warmup | skip_test | checkpoint | old val mse/mae |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for row in invalid:
            lines.append(
                "| {cell} | {variant} | {epochs} | {warmup} | {skip} | {ckpt} | {mse}/{mae} |".format(
                    cell=row.get("cell"),
                    variant=row.get("variant"),
                    epochs=row.get("train_epochs"),
                    warmup=row.get("penalty_warmup_epochs"),
                    skip=row.get("eval_skip_test"),
                    ckpt=row.get("best_checkpoint_saved"),
                    mse=_fmt(_as_float(row.get("val_avg_mse"))),
                    mae=_fmt(_as_float(row.get("val_avg_mae"))),
                )
            )
        lines.append("")
    lines.append("## Contribution table")
    lines.append("")
    lines.append("| cell | contribution | baseline -> new | baseline val | new val | MSE delta | MAE delta | raw route gain | selected/scaled gain | sufficiency |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---|")
    for row in report.get("contributions", []):
        lines.append(
            "| {cell} | {name} | {base}->{new} | {bm}/{ba} | {nm}/{na} | {dm} | {da} | {rg} | {sg} | {suff} |".format(
                cell=row["cell"],
                name=row["contribution"],
                base=row["baseline_variant"],
                new=row["new_variant"],
                bm=_fmt(row["baseline_mse"]),
                ba=_fmt(row["baseline_mae"]),
                nm=_fmt(row["new_mse"]),
                na=_fmt(row["new_mae"]),
                dm=_fmt_pct(row["mse_delta_pct"]),
                da=_fmt_pct(row["mae_delta_pct"]),
                rg=_fmt_pct(row.get("raw_route_gain_mse_pct")),
                sg=_fmt_pct(row.get("selected_scaled_gain_mse_pct")),
                suff=row.get("sufficiency"),
            )
        )
    lines.append("")
    test_rows = report.get("test_contributions", []) or []
    if test_rows:
        lines.append("## Test contribution table")
        lines.append("")
        lines.append("| cell | contribution | baseline -> new | baseline test | new test | test MSE delta | test MAE delta | val MSE delta | val MAE delta |")
        lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|")
        for row in test_rows:
            lines.append(
                "| {cell} | {name} | {base}->{new} | {bm}/{ba} | {nm}/{na} | {dm} | {da} | {vdm} | {vda} |".format(
                    cell=row["cell"],
                    name=row["contribution"],
                    base=row["baseline_variant"],
                    new=row["new_variant"],
                    bm=_fmt(row["baseline_mse"]),
                    ba=_fmt(row["baseline_mae"]),
                    nm=_fmt(row["new_mse"]),
                    na=_fmt(row["new_mae"]),
                    dm=_fmt_pct(row["mse_delta_pct"]),
                    da=_fmt_pct(row["mae_delta_pct"]),
                    vdm=_fmt_pct(row["val_mse_delta_pct"]),
                    vda=_fmt_pct(row["val_mae_delta_pct"]),
                )
            )
        lines.append("")
    lines.append("## Variant diagnostics")
    lines.append("")
    lines.append("| cell | variant | final source | final val | test source | final test | raw val | best_epoch | diag_epochs | route distribution | skip/no-op | residual_delta_rms | gradients |")
    lines.append("|---|---|---|---:|---|---:|---:|---|---:|---|---:|---:|---|")
    cells = report.get("cells", {}) or {}
    for cell, data in cells.items():
        for variant in VARIANTS:
            row = data["variants"][variant]
            route = row.get("route_distribution")
            if isinstance(route, Mapping):
                route_s = ", ".join(f"{k}:{float(v):.3f}" for k, v in route.items())
            else:
                route_s = "n/a"
            grad = row.get("last_epoch_gradient_l2_mean")
            if isinstance(grad, Mapping) and grad:
                grad_s = ", ".join(f"{k}:{float(v):.4g}" for k, v in grad.items())
            else:
                grad_s = "n/a"
            lines.append(
                "| {cell} | {variant} | {source} | {fm}/{fa} | {test_source} | {tm}/{ta} | {rm}/{ra} | {best} | {diag} | {route} | {skip} | {rms} | {grad} |".format(
                    cell=cell,
                    variant=variant,
                    source=row["final_metric_source"],
                    fm=_fmt(row["final_val_mse"]),
                    fa=_fmt(row["final_val_mae"]),
                    test_source=row["test_metric_source"],
                    tm=_fmt(row["final_test_mse"]),
                    ta=_fmt(row["final_test_mae"]),
                    rm=_fmt(row["raw_val_mse"]),
                    ra=_fmt(row["raw_val_mae"]),
                    best=row.get("best_epoch"),
                    diag=row.get("diag_epochs"),
                    route=route_s,
                    skip=_fmt(row.get("skip_noop_rate")),
                    rms=_fmt(row.get("residual_delta_rms")),
                    grad=grad_s,
                )
            )
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    if test_read:
        lines.append("- The old 1-epoch Stage-2 rows are invalid for final attribution.")
        lines.append("- The single legal test read contradicts the MoE-only validation lift in both cells: d regresses slightly against a on ETTm2-H96 and ETTh1-H96 test.")
        lines.append("- Anchors generalize cleanly and remain the dominant contribution on both cells.")
        lines.append("- MoE-on-anchor is small and cell-dependent on test: ETTm2-H96 improves over anchors, while ETTh1-H96 is MSE-neutral/slightly worse and MAE-neutral/slightly better.")
        lines.append("- Stop here for this test read; do not tune on test. The failure layer for MoE-only is train-val/test utility shift with secondary selection/adoption policy.")
    else:
        lines.append("- The old 1-epoch Stage-2 rows are invalid for final attribution.")
        lines.append("- Fair no-warmup Stage-2 training makes MoE-only nonzero but still much smaller than anchors on ETTm2-H96 and ETTh1-H96.")
        lines.append("- Anchors make residual routing easier: c improves over b on selected/scaled validation metrics in both cells, but raw routing can still be weaker than the selected/scaled path.")
        lines.append("- Skip/no-op remains unused in these ablation configs, so no-regret behavior here comes from the selection/scaling eval path rather than an active skip route.")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize NEXT-11c fair Stage-2 attribution.")
    parser.add_argument("--ettm2-root", type=Path, default=Path("outputs/next11c_fair_stage2_audit/fair_valonly_nowarmup"))
    parser.add_argument("--ettm2-stage2-root", type=Path, default=None)
    parser.add_argument("--etth1-root", type=Path, default=Path("outputs/next11c_fair_stage2_audit/fair_valonly_nowarmup"))
    parser.add_argument("--etth1-stage2-root", type=Path, default=Path("outputs/next11c_fair_stage2_audit/fair_valonly_nowarmup_e40"))
    parser.add_argument("--old-audit", type=Path, default=Path("outputs/next11c_fair_stage2_audit/step0_config_audit/stage2_config_audit.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/next11c_fair_stage2_audit/fair_attribution_report"))
    args = parser.parse_args()

    variant_roots: Dict[str, Dict[str, Path]] = {}
    if args.ettm2_stage2_root is not None:
        variant_roots["ETTm2_H96"] = {
            "d_moe_only_no_anchors": args.ettm2_stage2_root,
            "c_full": args.ettm2_stage2_root,
        }
    if args.etth1_stage2_root is not None:
        variant_roots["ETTh1_H96"] = {
            "d_moe_only_no_anchors": args.etth1_stage2_root,
            "c_full": args.etth1_stage2_root,
        }
    report = build_report(
        {"ETTm2_H96": args.ettm2_root, "ETTh1_H96": args.etth1_root},
        args.old_audit,
        variant_roots,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "fair_stage2_attribution.json"
    md_path = args.out_dir / "fair_stage2_attribution.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(markdown(report), encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
