from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_input96_frozen_moe_bayes_search import (  # noqa: E402
    candidate_from_json,
    read_rows,
    run_one_candidate,
    with_selection_controls,
    write_rows,
)
from scripts.run_input96_h96_targeted_tuning import load_yaml, resolve  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Recheck selected frozen MoE candidates with holdout scale selection.")
    ap.add_argument("--source-root", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--trials", nargs="+", type=int, required=True)
    ap.add_argument("--dataset", default="ETTh2")
    ap.add_argument("--horizon", type=int, default=96)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--holdout-fraction", type=float, default=0.4)
    ap.add_argument("--holdout-min-windows", type=int, default=256)
    ap.add_argument("--selection-max-residual-channels", type=int, default=0)
    ap.add_argument("--selection-eval-segments", type=int, default=1)
    ap.add_argument("--selection-min-positive-segments", type=int, default=0)
    ap.add_argument("--selection-max-segment-rel-degradation", type=float, default=0.0)
    ap.add_argument("--selection-max-segment-abs-degradation", type=float, default=0.0)
    args = ap.parse_args()

    source_root = resolve(args.source_root)
    out_root = resolve(args.out_root)
    source_rows = read_rows(source_root / "bayes_results.csv")
    by_trial = {int(row["trial"]): row for row in source_rows}
    rows: list[dict[str, object]] = []
    for trial in args.trials:
        if int(trial) not in by_trial:
            raise SystemExit(f"Trial {trial} not found in {source_root / 'bayes_results.csv'}")
        source = by_trial[int(trial)]
        cand = with_selection_controls(
            candidate_from_json(str(source["candidate_json"])),
            selection_policy="val_mse_scale_holdout",
            selection_holdout_fraction=float(args.holdout_fraction),
            selection_holdout_min_windows=int(args.holdout_min_windows),
            selection_max_residual_channels=int(args.selection_max_residual_channels),
            selection_eval_segments=int(args.selection_eval_segments),
            selection_min_positive_segments=int(args.selection_min_positive_segments),
            selection_max_segment_rel_degradation=float(args.selection_max_segment_rel_degradation),
            selection_max_segment_abs_degradation=float(args.selection_max_segment_abs_degradation),
        )
        cfg = load_yaml(Path(str(source["config_path"])))
        row = run_one_candidate(
            dataset=str(args.dataset),
            horizon=int(args.horizon),
            base_cfg=cfg,
            out_root=out_root,
            device=str(args.device),
            epochs=int(args.epochs),
            trial=int(trial),
            cand=cand,
            dry_run=False,
            skip_test=False,
            phase="holdout_recheck",
            variant_override=(
                f"trial_{int(trial):03d}_holdout"
                if int(args.selection_max_residual_channels) <= 0 and int(args.selection_eval_segments) <= 1
                else (
                    f"trial_{int(trial):03d}_holdout"
                    f"_mc{int(args.selection_max_residual_channels)}"
                    f"_sg{int(args.selection_eval_segments)}p{int(args.selection_min_positive_segments)}"
                )
            ),
        )
        row["source_trial"] = int(trial)
        rows.append(row)
        write_rows(out_root / "holdout_recheck_results.csv", rows)
        print(
            f"source_trial={trial} status={row.get('status')} "
            f"val_scaled={row.get('val_scaled_mse')} test_mse={row.get('test_mse')} "
            f"test_mae={row.get('test_mae')}",
            flush=True,
        )

    # Keep a compact source-trial table because write_rows only writes known fields.
    compact_path = out_root / "holdout_recheck_compact.csv"
    compact_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "source_trial",
        "status",
        "objective",
        "val_mse",
        "val_scaled_mse",
        "test_mse",
        "test_mae",
        "alpha_scale",
        "lambda_scale",
        "corrector_hidden",
        "topk",
        "router_mode",
        "config_path",
        "out_dir",
    ]
    with compact_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


if __name__ == "__main__":
    main()
