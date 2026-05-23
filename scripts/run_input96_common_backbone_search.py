from __future__ import annotations

import argparse
import copy
import csv
from pathlib import Path
from typing import Any

from run_input96_h96_targeted_tuning import (
    Candidate,
    DATASET_CONFIGS,
    REPO_ROOT,
    load_yaml,
    model_candidates,
    resolve,
    run_candidate,
    set_moe_off,
    value,
)


FAST_VARIANTS = {
    "current_model",
    "mlp_h128_do0_wd1e4_mae04",
    "mlp_h256_do02_wd1e3_mae06",
    "context_h256_do02_wd1e3_mae04",
    "channel_h256_do02_wd1e3_mae04",
    "attn_h256_do02_wd1e3_mae04",
}

RUN_FIELDS = [
    "backbone_variant",
    "dataset",
    "status",
    "test_mse",
    "test_mae",
    "val_mse",
    "val_mae",
    "ref_mse",
    "mse_ratio_vs_ref",
    "mse_delta_vs_ref",
    "config_path",
    "out_dir",
    "returncode",
]

SUMMARY_FIELDS = [
    "backbone_variant",
    "ok_count",
    "mean_test_mse",
    "mean_mse_ratio_vs_ref",
    "max_mse_ratio_vs_ref",
    "mean_mse_delta_vs_ref",
    "max_mse_delta_vs_ref",
    "datasets",
]


def load_h96_refs(path: Path) -> dict[str, float]:
    refs: dict[str, float] = {}
    if not path.exists():
        return refs
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if str(row.get("pred_len", "")) != "96":
                continue
            dataset = str(row.get("dataset", ""))
            raw = row.get("base_test_mse") or row.get("test_mse") or row.get("mse")
            try:
                refs[dataset] = float(raw)
            except Exception:
                pass
    return refs


def limit_backbones(candidates: list[Candidate], budget: str, variants: list[str] | None) -> list[Candidate]:
    if variants:
        wanted = set(variants)
        return [c for c in candidates if c.variant in wanted]
    if budget == "smoke":
        return candidates[:1]
    if budget == "fast":
        return [c for c in candidates if c.variant in FAST_VARIANTS]
    return candidates


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_variant: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        by_variant.setdefault(str(row["backbone_variant"]), []).append(row)

    summary: list[dict[str, Any]] = []
    for variant, group in by_variant.items():
        mses = [value(row, "test_mse") for row in group]
        ratios = [value(row, "mse_ratio_vs_ref") for row in group if row.get("mse_ratio_vs_ref") != ""]
        deltas = [value(row, "mse_delta_vs_ref") for row in group if row.get("mse_delta_vs_ref") != ""]
        summary.append(
            {
                "backbone_variant": variant,
                "ok_count": len(group),
                "mean_test_mse": sum(mses) / len(mses) if mses else "",
                "mean_mse_ratio_vs_ref": sum(ratios) / len(ratios) if ratios else "",
                "max_mse_ratio_vs_ref": max(ratios) if ratios else "",
                "mean_mse_delta_vs_ref": sum(deltas) / len(deltas) if deltas else "",
                "max_mse_delta_vs_ref": max(deltas) if deltas else "",
                "datasets": ",".join(str(row["dataset"]) for row in group),
            }
        )
    return sorted(
        summary,
        key=lambda row: (
            value(row, "mean_mse_ratio_vs_ref"),
            value(row, "max_mse_ratio_vs_ref"),
            value(row, "mean_test_mse"),
        ),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Search one shared input-96 H96 backbone across all selected datasets.")
    ap.add_argument("--out-root", default="outputs/input96_common_backbone_search")
    ap.add_argument("--datasets", nargs="+", default=list(DATASET_CONFIGS.keys()), choices=list(DATASET_CONFIGS.keys()))
    ap.add_argument("--budget", choices=["smoke", "fast", "compact"], default="fast")
    ap.add_argument("--variants", nargs="+", default=None)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--device", default=None)
    ap.add_argument("--reference-csv", default="outputs/ett_horizon_sweep/results.csv")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    out_root = resolve(args.out_root)
    refs = load_h96_refs(resolve(args.reference_csv))
    candidates = limit_backbones(model_candidates(), args.budget, args.variants)
    if not candidates:
        raise SystemExit("No backbone candidates selected.")

    rows: list[dict[str, Any]] = []
    for cand in candidates:
        print(f"=== backbone {cand.variant} ===", flush=True)
        for dataset in args.datasets:
            base_cfg = load_yaml(resolve(DATASET_CONFIGS[dataset]))
            set_moe_off(base_cfg)
            row, _ = run_candidate(
                dataset=dataset,
                base_cfg=base_cfg,
                cand=Candidate("common_backbone_h96", cand.variant, copy.deepcopy(cand.patch)),
                out_root=out_root,
                device=args.device,
                epochs=args.epochs,
                skip_test=False,
                dry_run=bool(args.dry_run),
            )
            ref = refs.get(dataset)
            test_mse = value(row, "test_mse")
            row["backbone_variant"] = cand.variant
            row["ref_mse"] = ref if ref is not None else ""
            row["mse_ratio_vs_ref"] = (test_mse / ref) if ref else ""
            row["mse_delta_vs_ref"] = (test_mse - ref) if ref else ""
            rows.append(row)
            write_csv(out_root / "backbone_runs.csv", rows, RUN_FIELDS)
            summary = summarize(rows)
            write_csv(out_root / "backbone_summary.csv", summary, SUMMARY_FIELDS)
            print(
                f"[{cand.variant} {dataset}] {row['status']} "
                f"mse={row['test_mse']} ref={row['ref_mse']} ratio={row['mse_ratio_vs_ref']}",
                flush=True,
            )

    summary = summarize(rows)
    write_csv(out_root / "backbone_summary.csv", summary, SUMMARY_FIELDS)
    if summary:
        print(f"BEST_BACKBONE {summary[0]['backbone_variant']}", flush=True)
        print(f"SUMMARY {out_root / 'backbone_summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
