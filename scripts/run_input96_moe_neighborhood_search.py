from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_input96_frozen_moe_bayes_search import (  # noqa: E402
    FrozenMoeCandidate,
    candidate_from_json,
    read_rows,
    run_one_candidate,
    safe_float,
    with_selection_controls,
    write_rows,
)
from scripts.run_input96_h96_targeted_tuning import load_yaml, resolve  # noqa: E402


def trial_candidate(source_root: Path, trial: int) -> tuple[FrozenMoeCandidate, dict[str, Any]]:
    rows = read_rows(source_root / "bayes_results.csv")
    for row in rows:
        if int(row.get("trial", -1)) == int(trial):
            return candidate_from_json(str(row["candidate_json"])), row
    raise SystemExit(f"Trial {trial} not found in {source_root / 'bayes_results.csv'}")


def neighbor_candidates(base: FrozenMoeCandidate) -> list[tuple[str, FrozenMoeCandidate]]:
    base = with_selection_controls(
        base,
        selection_policy="val_mse_scale_holdout",
        selection_holdout_fraction=0.4,
        selection_holdout_min_windows=256,
        selection_max_residual_channels=3,
    )
    return [
        ("mc2", replace(base, selection_max_residual_channels=2)),
        ("alpha1p25", replace(base, alpha_scale=1.25)),
        ("alpha1p75", replace(base, alpha_scale=1.75)),
        ("lambda0p003", replace(base, lambda_scale=0.003)),
        ("lambda0p006", replace(base, lambda_scale=0.006)),
        ("lambda0p0075", replace(base, lambda_scale=0.0075)),
        ("lambda0p006_alpha1p6", replace(base, lambda_scale=0.006, alpha_scale=1.6)),
        ("lambda0p0075_alpha1p6", replace(base, lambda_scale=0.0075, alpha_scale=1.6)),
        ("lambda0p0075_lr0p0004", replace(base, lambda_scale=0.0075, lr=4.0e-4)),
        ("lambda0p0075_scale1p0", replace(base, lambda_scale=0.0075, selection_scale_max=1.0, selection_scale_steps=41)),
        ("lr0p0003", replace(base, lr=3.0e-4)),
        ("scale1p0_n41", replace(base, selection_scale_max=1.0, selection_scale_steps=41)),
        ("topk1", replace(base, topk=1)),
        ("hidden48", replace(base, corrector_hidden=48)),
        ("clip3", replace(base, residual_clip=3.0)),
    ]


def write_compact(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "variant",
        "status",
        "objective",
        "val_mse",
        "val_scaled_mse",
        "test_mse",
        "test_mae",
        "residual_num_channels",
        "residual_mean_scale",
        "alpha_scale",
        "lambda_scale",
        "lr",
        "topk",
        "corrector_hidden",
        "residual_clip",
        "config_path",
        "out_dir",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in sorted(rows, key=lambda r: safe_float(r.get("test_mse"))):
            writer.writerow({field: row.get(field, "") for field in fields})


def main() -> None:
    ap = argparse.ArgumentParser(description="Small neighborhood sweep around a stable frozen MoE candidate.")
    ap.add_argument("--source-root", default="outputs/fresh_input_len96_20260609_etth2_moe_bayes_h96")
    ap.add_argument("--source-trial", type=int, default=3)
    ap.add_argument("--out-root", default="outputs/fresh_input_len96_20260610_etth2_moe_trial003_neighborhood")
    ap.add_argument("--dataset", default="ETTh2")
    ap.add_argument("--horizon", type=int, default=96)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    source_root = resolve(args.source_root)
    out_root = resolve(args.out_root)
    base, source_row = trial_candidate(source_root, int(args.source_trial))
    source_cfg = load_yaml(Path(str(source_row["config_path"])))
    candidates = neighbor_candidates(base)
    if int(args.limit) > 0:
        candidates = candidates[: int(args.limit)]

    result_path = out_root / "neighborhood_results.csv"
    rows = read_rows(result_path)
    done_variants = {str(row.get("variant")) for row in rows if row.get("status") == "ok"}

    for idx, (name, cand) in enumerate(candidates):
        variant = f"trial_{int(args.source_trial):03d}_nb_{name}"
        if variant in done_variants:
            print(f"[skip] {variant}", flush=True)
            continue
        row = run_one_candidate(
            dataset=str(args.dataset),
            horizon=int(args.horizon),
            base_cfg=source_cfg,
            out_root=out_root,
            device=str(args.device),
            epochs=int(args.epochs),
            trial=idx,
            cand=cand,
            dry_run=bool(args.dry_run),
            skip_test=False,
            phase="neighborhood",
            variant_override=variant,
        )
        rows.append(row)
        write_rows(result_path, rows)
        write_compact(out_root / "neighborhood_compact.csv", rows)
        print(
            f"{variant} status={row.get('status')} val_scaled={row.get('val_scaled_mse')} "
            f"test_mse={row.get('test_mse')} test_mae={row.get('test_mae')}",
            flush=True,
        )

    write_compact(out_root / "neighborhood_compact.csv", rows)


if __name__ == "__main__":
    main()
