from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_input96_frozen_moe_bayes_search import (  # noqa: E402
    candidate_from_json,
    candidate_to_patch,
    read_rows,
)
from scripts.run_input96_h96_targeted_tuning import Candidate, load_yaml, resolve, run_candidate  # noqa: E402
from scripts.run_input96_moe_positive_search import apply_moe_training_controls  # noqa: E402


BEST_GATE = {
    "max_scale": 2.0,
    "init_scale": 0.5,
    "scale_reg": 0.0001,
    "standardize_features": True,
    "epochs": 30,
    "batch_size": 256,
}

CHANNEL_DELTA = {"channel_expert_adapters": {"enable": True, "mode": "all", "mode_type": "delta"}}
NO_EXTRA_LOSS = {"specialization_weight": 0.0, "norm_weight": 0.0, "intervention_weight": 0.0}


def expert_variants() -> list[tuple[str, dict[str, Any]]]:
    return [
        ("base_gate_ms2", {}),
        ("no_intervention_reg", {"intervention_weight": 0.0}),
        ("no_norm_reg", {"norm_weight": 0.0}),
        ("no_extra_residual_loss", {"specialization_weight": 0.0, "norm_weight": 0.0, "intervention_weight": 0.0}),
        ("spec0p05", {"specialization_weight": 0.05}),
        ("spec0p2", {"specialization_weight": 0.2}),
        ("hidden96", {"corrector_hidden": 96}),
        ("hidden128", {"corrector_hidden": 128}),
        ("safe_augmented", {"feature_mode": "safe_augmented"}),
        (
            "channel_all_delta",
            dict(CHANNEL_DELTA),
        ),
        (
            "channel_all_override",
            {"channel_expert_adapters": {"enable": True, "mode": "all", "mode_type": "override"}},
        ),
        (
            "channel_delta_no_extra",
            {**CHANNEL_DELTA, **NO_EXTRA_LOSS},
        ),
        (
            "channel_delta_no_extra_safe_aug",
            {**CHANNEL_DELTA, **NO_EXTRA_LOSS, "feature_mode": "safe_augmented"},
        ),
        (
            "channel_delta_no_extra_gate_ms15",
            {**CHANNEL_DELTA, **NO_EXTRA_LOSS, "gate_calibrator": {"max_scale": 1.5, "init_scale": 0.3, "scale_reg": 0.0005}},
        ),
        (
            "channel_delta_no_extra_gate_ms25",
            {**CHANNEL_DELTA, **NO_EXTRA_LOSS, "gate_calibrator": {"max_scale": 2.5, "init_scale": 0.5, "scale_reg": 0.0001}},
        ),
    ]


def source_trial_candidate(source_root: Path, trial: int):
    rows = read_rows(source_root / "bayes_results.csv")
    for row in rows:
        if int(row.get("trial", -1)) == int(trial):
            return candidate_from_json(str(row["candidate_json"]))
    raise SystemExit(f"Trial {trial} not found in {source_root / 'bayes_results.csv'}")


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted(set().union(*(row.keys() for row in rows))) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Probe residual expert capacity and regularization for input-96 frozen MoE.")
    ap.add_argument(
        "--base-config",
        default="outputs/fresh_input_len96_20260609_etth2_backbone_ckpt/configs/ETTh2/H96/common_backbone_h96/current_model.yaml",
    )
    ap.add_argument(
        "--warm-start-checkpoint",
        default="outputs/fresh_input_len96_20260609_etth2_backbone_ckpt/runs/ETTh2/H96/common_backbone_h96/current_model/best_checkpoint.pt",
    )
    ap.add_argument("--source-root", default="outputs/fresh_input_len96_20260609_etth2_moe_bayes_h96")
    ap.add_argument("--source-trial", type=int, default=3)
    ap.add_argument("--out-root", default="outputs/fresh_input_len96_20260610_etth2_moe_residual_expert_probe")
    ap.add_argument("--dataset", default="ETTh2")
    ap.add_argument("--horizon", type=int, default=96)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    out_root = resolve(args.out_root)
    base_cfg = load_yaml(resolve(args.base_config))
    apply_moe_training_controls(
        base_cfg,
        warm_start_checkpoint=str(args.warm_start_checkpoint),
        freeze_backbone=True,
        lr=None,
        weight_decay=None,
    )

    source_cand = source_trial_candidate(resolve(args.source_root), int(args.source_trial))
    base_patch = candidate_to_patch(source_cand)
    variants = expert_variants()
    if int(args.limit) > 0:
        variants = variants[: int(args.limit)]

    result_path = out_root / "expert_probe_results.csv"
    rows = read_rows(result_path)
    done = {str(row.get("variant")) for row in rows if row.get("status") == "ok"}

    for name, residual_patch in variants:
        if name in done:
            print(f"[skip] {name}", flush=True)
            continue
        patch = json.loads(json.dumps(base_patch))
        residual = patch["moe"]["pred_side_residual"]
        residual["selection_policy"] = "val_mse_gate_guarded"
        residual["selection_min_rel_improvement"] = 0.0005
        for key in [
            "selection_holdout_fraction",
            "selection_holdout_min_windows",
            "selection_max_residual_channels",
            "selection_eval_segments",
            "selection_min_positive_segments",
            "selection_max_segment_rel_degradation",
            "selection_max_segment_abs_degradation",
        ]:
            residual.pop(key, None)
        residual["gate_calibrator"] = dict(BEST_GATE)
        residual_patch = dict(residual_patch)
        gate_patch = residual_patch.pop("gate_calibrator", None)
        if gate_patch is not None:
            residual["gate_calibrator"].update(dict(gate_patch))
        residual.update(residual_patch)
        run_cand = Candidate("expert_probe", name, patch)
        row, _ = run_candidate(
            dataset=str(args.dataset),
            pred_len=int(args.horizon),
            base_cfg=base_cfg,
            cand=run_cand,
            out_root=out_root,
            device=str(args.device),
            epochs=int(args.epochs),
            skip_test=False,
            dry_run=bool(args.dry_run),
        )
        rows.append(row)
        write_rows(result_path, rows)
        print(
            f"{name} status={row.get('status')} val={row.get('val_mse')} "
            f"test={row.get('test_mse')} mae={row.get('test_mae')}",
            flush=True,
        )


if __name__ == "__main__":
    main()
