from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MATRIX_ROOT = ROOT / "outputs" / "shared_pkr_patch_gate_matrix_20260710"
PENALTIES = ["level", "delta", "d2_match", "diff_amp"]
DATASETS = ("ETTm1", "ETTm2", "ETTh1", "ETTh2")
HORIZONS = (96, 192, 336, 720)


def output(path: str) -> Path:
    return ROOT / "outputs" / Path(path)


BASELINE_SUMMARIES = {
    ("ETTm1", 96): output(
        "non_ecl_learnable_anchor_sweep_20260628_probe/static_baseline/runs/"
        "ETTm1/H96/mse_gate_w002_strong_safe_mse/run_summary.json"
    ),
    ("ETTm1", 192): output(
        "non_ecl_learnable_anchor_sweep_20260628_probe/static_baseline/runs/"
        "ETTm1/H192/mse_gate_w002_ch2/run_summary.json"
    ),
    ("ETTm1", 336): output(
        "non_ecl_learnable_anchor_sweep_20260628_probe/static_baseline/runs/"
        "ETTm1/H336/mse_gate_w005_softprior/run_summary.json"
    ),
    ("ETTm1", 720): output(
        "non_ecl_learnable_anchor_sweep_20260628_probe/static_baseline/runs/"
        "ETTm1/H720/hist020_trainstatresid_mean_p96_stat020_resid120_seg7/"
        "run_summary.json"
    ),
    ("ETTm2", 96): output(
        "non_ecl_learnable_anchor_sweep_20260628_probe/static_baseline/runs/"
        "ETTm2/H96/mse_gate_w002_top2_h96_cfull/run_summary.json"
    ),
    ("ETTm2", 192): output(
        "non_ecl_baseline_repro_ettm2_h192_valid_valonly_20260629/static_baseline/"
        "runs/ETTm2/H192/mse_gate_w002_top2_h96_cfull/run_summary.json"
    ),
    ("ETTm2", 336): output(
        "non_ecl_baseline_repro_ettm2_h336_valid_valonly_20260629/static_baseline/"
        "runs/ETTm2/H336/mse_gate_w002_top2_h96_cfull/run_summary.json"
    ),
    ("ETTm2", 720): output(
        "non_ecl_baseline_repro_ettm2_h720_valid_valonly_20260629/static_baseline/"
        "runs/ETTm2/H720/trainstatresid_mean_p96_stat020seg7_resid120_seg12_h96_cfull/"
        "run_summary.json"
    ),
    ("ETTh1", 96): output(
        "non_ecl_learnable_anchor_sweep_20260628_probe/static_baseline/runs/"
        "ETTh1/H96/mse_gate_w002_softprior/run_summary.json"
    ),
    ("ETTh1", 192): output(
        "non_ecl_learnable_anchor_sweep_20260628_probe/static_baseline/runs/"
        "ETTh1/H192/p24_stat020seg12_resid120_allmae_seg12/run_summary.json"
    ),
    ("ETTh1", 336): output(
        "non_ecl_learnable_anchor_sweep_20260628_probe/static_baseline/runs/"
        "ETTh1/H336/trainstatresid_mean_p96_stat020_resid080_seg4/run_summary.json"
    ),
    ("ETTh1", 720): output(
        "non_ecl_learnable_anchor_sweep_20260628_probe/static_baseline/runs/"
        "ETTh1/H720/statsel0p15_resid0p6_mse/run_summary.json"
    ),
    ("ETTh2", 96): output(
        "non_ecl_learnable_anchor_sweep_20260628_probe/static_baseline/runs/"
        "ETTh2/H96/gate_mae_alpha1p2_clip3_h96_anchorpath/run_summary.json"
    ),
    ("ETTh2", 192): output(
        "non_ecl_learnable_anchor_sweep_20260628_probe/static_baseline/runs/"
        "ETTh2/H192/mse_gate_w002_top2_h96_anchorpath/run_summary.json"
    ),
    ("ETTh2", 336): output(
        "non_ecl_learnable_anchor_sweep_20260628_probe/static_baseline/runs/"
        "ETTh2/H336/mse_gate_w005_softprior_h96_anchorpath/run_summary.json"
    ),
    ("ETTh2", 720): output(
        "non_ecl_learnable_anchor_sweep_20260628_probe/static_baseline/runs/"
        "ETTh2/H720/mse_gate_w002_top2_h96_anchorpath/run_summary.json"
    ),
}

EXISTING_BANKS = {
    ("ETTm1", 96): output(
        "shared_moe_cluster_ablation_20260709/runs/ETTm1/H96/"
        "shared_moe_gate96_r64_valonly/run_summary.json"
    ),
    ("ETTh1", 96): output(
        "shared_moe_cluster_ablation_20260709/runs/ETTh1/H96/"
        "shared_moe_ettm1_recipe_valonly/run_summary.json"
    ),
}

ETTM1_H96_REFERENCE = output(
    "ettm1_shared_pkr_patch_gate_recall_20260710/runs/ETTm1/H96/"
    "shared_pkr_patch24_regimectx192_384_672_utilitypolicy_ep12_valonly/"
    "run_summary.json"
)
ETTM1_H96_AUDIT = output(
    "ettm1_shared_pkr_patch_gate_recall_20260710/runs/ETTm1/H96/"
    "shared_pkr_patch24_regimectx192_384_672_utilitypolicy_blockaudit6_valonly/"
    "run_summary.json"
)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def selected_val(summary: dict[str, Any]) -> tuple[float, float]:
    selection = summary.get("moe_residual_selection") or {}
    mse = selection.get("val_scaled_avg_mse")
    mae = selection.get("val_scaled_avg_mae")
    val = summary.get("val") or {}
    return (
        float(val.get("avg_mse") if mse is None else mse),
        float(val.get("avg_mae") if mae is None else mae),
    )


def bank_summary_path(dataset: str, horizon: int) -> Path:
    existing = EXISTING_BANKS.get((dataset, horizon))
    if existing is not None:
        return existing
    return (
        MATRIX_ROOT
        / "runs"
        / dataset
        / f"H{horizon}"
        / "shared_four_pkr_bank_ep6_valonly"
        / "run_summary.json"
    )


def gate_summary_path(dataset: str, horizon: int) -> Path:
    if (dataset, horizon) == ("ETTm1", 96):
        return ETTM1_H96_REFERENCE
    return (
        MATRIX_ROOT
        / "runs"
        / dataset
        / f"H{horizon}"
        / "shared_pkr_patch24_regimectx192_384_672_utilitypolicy_ep12_valonly"
        / "run_summary.json"
    )


def audit_summary_path(dataset: str, horizon: int) -> Path:
    if (dataset, horizon) == ("ETTm1", 96):
        return ETTM1_H96_AUDIT
    return (
        MATRIX_ROOT
        / "runs"
        / dataset
        / f"H{horizon}"
        / "shared_pkr_patch24_regimectx192_384_672_utilitypolicy_blockaudit6_valonly"
        / "run_summary.json"
    )


def build_row(dataset: str, horizon: int) -> dict[str, Any]:
    gate_path = gate_summary_path(dataset, horizon)
    audit_path = audit_summary_path(dataset, horizon)
    bank_path = bank_summary_path(dataset, horizon)
    baseline_path = BASELINE_SUMMARIES[(dataset, horizon)]
    for path in (gate_path, audit_path, bank_path, baseline_path):
        if not path.exists():
            raise FileNotFoundError(path)

    gate = read_json(gate_path)
    audit = read_json(audit_path)
    bank = read_json(bank_path)
    baseline = read_json(baseline_path)
    assert gate.get("test") is None
    assert (gate.get("eval") or {}).get("skip_test") is True
    assert gate.get("penalty_names") == PENALTIES
    assert (gate.get("shared_moe") or {}).get("shared_across_clusters") is True
    assert gate["stage2_trainable_parameter_groups"]["total"]["backbone"] == 0

    oracle = gate["moe_residual"]["patch_router"]["oracle_diagnostic"]
    train_oracle = gate["moe_residual"]["patch_router"].get(
        "train_oracle_diagnostic"
    ) or {}
    block_metrics = audit["moe_residual"]["patch_router"][
        "validation_temporal_block_metrics"
    ]
    assert block_metrics["num_blocks"] == 6
    assert block_metrics["test_read"] is False
    block_gains = [float(block["selected_gain_pct"]) for block in block_metrics["blocks"]]
    bank_selection = bank.get("moe_residual_selection") or {}
    bank_base = float(
        bank_selection.get(
            "val_pred_base_avg_mse",
            oracle["base_patch_mse"],
        )
    )
    bank_raw = float(
        bank_selection.get(
            "val_residual_avg_mse",
            (bank.get("val") or {})["avg_mse"],
        )
    )
    bank_static = float(
        bank_selection.get("val_scaled_avg_mse", bank_raw)
    )
    baseline_mse, baseline_mae = selected_val(baseline)
    gate_val_mse = float((gate.get("val") or {})["avg_mse"])
    gate_val_mae = float((gate.get("val") or {})["avg_mae"])
    gate_gain = float(oracle["selected_gain_pct"])
    train_gain = train_oracle.get("selected_gain_pct")
    shift_failure = bool(
        train_gain is not None and float(train_gain) > 0.0 and gate_gain <= 0.0
    )
    shared = gate.get("shared_moe") or {}
    return {
        "dataset": dataset,
        "horizon": horizon,
        "reference_existing": (dataset, horizon) == ("ETTm1", 96),
        "best_epoch": int(shared.get("best_epoch", 0)),
        "history_patch_projection": gate["moe_residual"]["patch_router"].get(
            "history_patch_projection",
            "tail" if horizon == 96 else None,
        ),
        "bank_base_mse": bank_base,
        "bank_raw_mse": bank_raw,
        "bank_static_mse": bank_static,
        "gate_patch_base_mse": float(oracle["base_patch_mse"]),
        "gate_patch_selected_mse": float(oracle["selected_patch_mse"]),
        "gate_val_mse": gate_val_mse,
        "gate_val_mae": gate_val_mae,
        "gate_gain_pct": gate_gain,
        "oracle_gain_pct": float(oracle["oracle_gain_pct"]),
        "train_gate_gain_pct": None if train_gain is None else float(train_gain),
        "selected_recall": float(oracle["selected_utility_recall"]),
        "selected_precision": float(oracle["selected_utility_precision"]),
        "gain_cost_ratio": float(oracle["selected_gain_to_cost_ratio"]),
        "proposal_oracle_recall_at_2": float(
            oracle["proposal_oracle_best_recall_at_k"]
        ),
        "pairwise_accuracy": float(oracle["shortlist_pairwise_accuracy"]),
        "per_cluster_val_mse": baseline_mse,
        "per_cluster_val_mae": baseline_mae,
        "gap_vs_per_cluster_pct": 100.0 * (gate_val_mse / baseline_mse - 1.0),
        "gain_vs_bank_static_pct": 100.0 * (bank_static - gate_val_mse) / bank_static,
        "positive_vs_raw_base": gate_gain > 0.0,
        "beats_bank_static": gate_val_mse < bank_static,
        "beats_per_cluster": gate_val_mse < baseline_mse,
        "train_val_shift_failure": shift_failure,
        "positive_temporal_blocks": sum(gain > 0.0 for gain in block_gains),
        "temporal_block_gains_pct": block_gains,
        "min_temporal_block_gain_pct": min(block_gains),
        "stable_6_of_6": all(gain > 0.0 for gain in block_gains),
        "gate_summary_path": rel(gate_path),
        "audit_summary_path": rel(audit_path),
        "bank_summary_path": rel(bank_path),
        "per_cluster_summary_path": rel(baseline_path),
    }


def format_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Shared Four-PKR Patch-Gate ETT Matrix",
        "",
        "Protocol: frozen existing backbone, one shared MoE across clusters, "
        "unchanged `level/delta/d2_match/diff_amp`, patch length 24, causal "
        "regime context 192/384/672, validation only, no test read.",
        "",
        "| Cell | Per-cluster val | Bank static val | Patch base -> gate | Gate gain | Oracle | Recall / precision | G/C | Blocks | Verdict |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in payload["results"]:
        verdict = "positive"
        if row["train_val_shift_failure"]:
            verdict = "train->val shift"
        elif not row["positive_vs_raw_base"]:
            verdict = "negative"
        lines.append(
            "| {dataset}-H{horizon} | {per_cluster_val_mse:.6f} | "
            "{bank_static_mse:.6f} | {gate_patch_base_mse:.6f} -> "
            "{gate_patch_selected_mse:.6f} | {gate_gain_pct:+.3f}% | "
            "{oracle_gain_pct:.2f}% | {selected_recall:.1%} / "
            "{selected_precision:.1%} | {gain_cost_ratio:.3f} | "
            "{positive_temporal_blocks}/6 | {verdict} |".format(
                **row,
                verdict=verdict,
            )
        )
    aggregate = payload["aggregate"]
    lines.extend(
        [
            "",
            f"New cells positive vs raw base: {aggregate['new_positive_count']}/15.",
            f"All cells beating bank static selector: {aggregate['beats_bank_static_count']}/16.",
            f"All cells beating canonical per-cluster val: {aggregate['beats_per_cluster_count']}/16.",
            f"All cells positive in 6/6 temporal blocks: {aggregate['stable_6_of_6_count']}/16.",
            f"Explicit train->val shift failures: {', '.join(aggregate['shift_failure_cells'])}.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    rows = [
        build_row(dataset, horizon)
        for dataset in DATASETS
        for horizon in HORIZONS
    ]
    new_rows = [row for row in rows if not row["reference_existing"]]
    aggregate = {
        "cell_count": len(rows),
        "new_cell_count": len(new_rows),
        "new_positive_count": sum(row["positive_vs_raw_base"] for row in new_rows),
        "beats_bank_static_count": sum(row["beats_bank_static"] for row in rows),
        "beats_per_cluster_count": sum(row["beats_per_cluster"] for row in rows),
        "shift_failure_cells": [
            f"{row['dataset']}-H{row['horizon']}"
            for row in rows
            if row["train_val_shift_failure"]
        ],
        "stable_6_of_6_count": sum(row["stable_6_of_6"] for row in rows),
        "stable_6_of_6_cells": [
            f"{row['dataset']}-H{row['horizon']}"
            for row in rows
            if row["stable_6_of_6"]
        ],
        "best_gate_gain_cell": max(
            rows,
            key=lambda row: row["gate_gain_pct"],
        )["dataset"]
        + "-H"
        + str(max(rows, key=lambda row: row["gate_gain_pct"])["horizon"]),
        "closest_per_cluster_cell": min(
            rows,
            key=lambda row: abs(row["gap_vs_per_cluster_pct"]),
        )["dataset"]
        + "-H"
        + str(
            min(rows, key=lambda row: abs(row["gap_vs_per_cluster_pct"]))[
                "horizon"
            ]
        ),
        "test_read": False,
    }
    payload = {
        "protocol": {
            "backbone_frozen": True,
            "shared_across_clusters": True,
            "penalties": PENALTIES,
            "patch_len": 24,
            "regime_context_lengths": [192, 384, 672],
            "short_history_mode": "cycle when pred_len > input_len",
            "test_read": False,
        },
        "aggregate": aggregate,
        "results": rows,
    }
    MATRIX_ROOT.mkdir(parents=True, exist_ok=True)
    json_path = MATRIX_ROOT / "matrix_comparison.json"
    csv_path = MATRIX_ROOT / "matrix_comparison.csv"
    markdown_path = MATRIX_ROOT / "matrix_comparison.md"
    with json_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2)
        handle.write("\n")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    markdown_path.write_text(format_markdown(payload), encoding="utf-8", newline="\n")
    print(json.dumps(aggregate, ensure_ascii=True, indent=2))
    print(rel(markdown_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
