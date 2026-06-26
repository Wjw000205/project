from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY = ROOT / "outputs" / "codex_table_target_20260614" / "input96_global_paired_backbone_moe_summary.csv"
DEFAULT_RESULTS = ROOT / "outputs" / "input96_main_table_anchor_on_no_ecl_20260619" / "results.csv"
DEFAULT_EXCLUDE_DATASETS = {"electricity", "ecl", "weather"}

FIELDS = [
    "dataset",
    "horizon",
    "status",
    "old_variant",
    "new_variant",
    "strategy_name",
    "old_mse",
    "new_mse",
    "mse_improve_pct",
    "old_mae",
    "new_mae",
    "mae_improve_pct",
    "mse_improved",
    "mae_improved",
    "both_improved",
    "old_config",
    "new_config",
    "strategy_config",
    "out_dir",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def as_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def improve_pct(old: float | None, new: float | None) -> float | None:
    if old is None or new is None or old == 0:
        return None
    return (old - new) / old * 100.0


def fmt(value: Any, digits: int = 6) -> str:
    number = as_float(value)
    if number is None:
        return ""
    return f"{number:.{digits}f}"


def fmt_pct(value: Any) -> str:
    number = as_float(value)
    if number is None:
        return ""
    return f"{number:+.3f}%"


def compare(summary_csv: Path, results_csv: Path, exclude_datasets: set[str], ok_only: bool) -> list[dict[str, Any]]:
    old_rows = {
        (str(row.get("dataset", "")), str(row.get("horizon", ""))): row
        for row in read_csv(summary_csv)
    }
    rows: list[dict[str, Any]] = []
    for result in read_csv(results_csv):
        key = (str(result.get("dataset", "")), str(result.get("horizon", "")))
        if key[0].lower() in exclude_datasets:
            continue
        if ok_only and result.get("status", "") != "ok":
            continue
        old = old_rows.get(key)
        if old is None:
            continue
        old_mse = as_float(old.get("moe_mse"))
        old_mae = as_float(old.get("moe_mae"))
        new_mse = as_float(result.get("test_mse"))
        new_mae = as_float(result.get("test_mae"))
        mse_gain = improve_pct(old_mse, new_mse)
        mae_gain = improve_pct(old_mae, new_mae)
        mse_improved = mse_gain is not None and mse_gain > 0
        mae_improved = mae_gain is not None and mae_gain > 0
        rows.append(
            {
                "dataset": key[0],
                "horizon": key[1],
                "status": result.get("status", ""),
                "old_variant": old.get("moe_variant", ""),
                "new_variant": result.get("variant", ""),
                "strategy_name": result.get("strategy_name", ""),
                "old_mse": old_mse,
                "new_mse": new_mse,
                "mse_improve_pct": mse_gain,
                "old_mae": old_mae,
                "new_mae": new_mae,
                "mae_improve_pct": mae_gain,
                "mse_improved": mse_improved,
                "mae_improved": mae_improved,
                "both_improved": mse_improved and mae_improved,
                "old_config": old.get("moe_config", ""),
                "new_config": result.get("config_path", ""),
                "strategy_config": result.get("strategy_config", ""),
                "out_dir": result.get("out_dir", ""),
            }
        )
    return sorted(rows, key=lambda r: (r["dataset"], int(r["horizon"])))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDS})


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    complete = [row for row in rows if row.get("status") == "ok"]
    improved = [row for row in complete if row.get("both_improved")]
    with path.open("w", encoding="utf-8") as f:
        f.write("# Input96 Anchor-On Rerun Comparison\n\n")
        f.write(f"- Completed rows: {len(complete)}/{len(rows)}\n")
        f.write(f"- Both-metric improved rows: {len(improved)}\n\n")
        f.write("| dataset | H | status | variant | strategy | old MSE | new MSE | dMSE | old MAE | new MAE | dMAE | both |\n")
        f.write("|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|---|\n")
        for row in rows:
            f.write(
                "| "
                + " | ".join(
                    [
                        str(row["dataset"]),
                        str(row["horizon"]),
                        str(row["status"]),
                        str(row["new_variant"]),
                        str(row["strategy_name"]),
                        fmt(row["old_mse"]),
                        fmt(row["new_mse"]),
                        fmt_pct(row["mse_improve_pct"]),
                        fmt(row["old_mae"]),
                        fmt(row["new_mae"]),
                        fmt_pct(row["mae_improve_pct"]),
                        "yes" if row["both_improved"] else "",
                    ]
                )
                + " |\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare input-96 anchor-on rerun results against the current main summary CSV.")
    parser.add_argument("--summary-csv", default=str(DEFAULT_SUMMARY))
    parser.add_argument("--results-csv", default=str(DEFAULT_RESULTS))
    parser.add_argument("--out-csv", default="")
    parser.add_argument("--out-md", default="")
    parser.add_argument("--exclude-datasets", nargs="*", default=sorted(DEFAULT_EXCLUDE_DATASETS))
    parser.add_argument("--include-incomplete", action="store_true")
    args = parser.parse_args()

    summary_csv = Path(args.summary_csv)
    results_csv = Path(args.results_csv)
    if not summary_csv.is_absolute():
        summary_csv = ROOT / summary_csv
    if not results_csv.is_absolute():
        results_csv = ROOT / results_csv
    out_csv = Path(args.out_csv) if args.out_csv else results_csv.with_name("comparison_vs_current_main.csv")
    out_md = Path(args.out_md) if args.out_md else results_csv.with_name("comparison_vs_current_main.md")
    if not out_csv.is_absolute():
        out_csv = ROOT / out_csv
    if not out_md.is_absolute():
        out_md = ROOT / out_md

    rows = compare(
        summary_csv,
        results_csv,
        {name.lower() for name in args.exclude_datasets},
        ok_only=not bool(args.include_incomplete),
    )
    write_csv(out_csv, rows)
    write_markdown(out_md, rows)
    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_md}")


if __name__ == "__main__":
    main()
