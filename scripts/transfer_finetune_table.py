import argparse
import csv
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRANSFER_CSVS = [
    ROOT / "outputs" / "ettm1_current_full_horizon_transfer_finetune" / "transfer_finetune.csv",
    ROOT / "outputs" / "ettm2_current_full_horizon_transfer_finetune" / "transfer_finetune.csv",
    ROOT / "outputs" / "ettm2_to_etth2_h336_pure_transfer" / "transfer_finetune.csv",
]
DEFAULT_MARKDOWN = ROOT / "outputs" / "experiment_excel_summary" / "paper_style_experiment_summary.md"
EXPECTED_LRS = ("0.0001", "5e-05", "2e-05")

SOURCE_ORDER = ["ETTm1", "ETTm2"]
TARGET_ORDER = {
    "ETTm1": ["ETTh1", "ETTh2", "ETTm2"],
    "ETTm2": ["ETTh1", "ETTh2", "ETTm1"],
}
HORIZON_ORDER = [96, 192, 336, 720]

DEFAULT_MAIN_SELF_METRICS = {
    ("ETTh1", 96): (0.3610, 0.3904),
    ("ETTh1", 192): (0.4069, 0.4186),
    ("ETTh1", 336): (0.4328, 0.4339),
    ("ETTh1", 720): (0.4499, 0.4616),
    ("ETTh2", 96): (0.2873, 0.3464),
    ("ETTh2", 192): (0.3564, 0.3925),
    ("ETTh2", 336): (0.3808, 0.4149),
    ("ETTh2", 720): (0.4066, 0.4379),
    ("ETTm1", 96): (0.2834, 0.3381),
    ("ETTm1", 192): (0.3241, 0.3643),
    ("ETTm1", 336): (0.3599, 0.3859),
    ("ETTm1", 720): (0.4225, 0.4224),
    ("ETTm2", 96): (0.1649, 0.2534),
    ("ETTm2", 192): (0.2229, 0.2907),
    ("ETTm2", 336): (0.2773, 0.3281),
    ("ETTm2", 720): (0.3663, 0.3842),
}

TRANSFER_FIELDS = [
    "Source",
    "Target",
    "H",
    "Source self MSE/MAE",
    "Target self MSE/MAE",
    "Zero-shot MSE/MAE",
    "Fine-tune MSE/MAE",
    "FT gain vs zero-shot",
    "FT gain vs target self",
    "Zero-shot note",
    "Leakage controls",
]

ZERO_SHOT_DEGRADATION_NOTES = {
    ("ETTm1", "ETTm2", 96): "Target self is strong; zero-shot still needs target adaptation.",
    ("ETTm1", "ETTm2", 192): "Target self is strong; zero-shot still needs target adaptation.",
    ("ETTm2", "ETTm1", 96): "Source-target scale mismatch; fine-tune restores target level.",
    ("ETTm2", "ETTm1", 192): "Source-target scale mismatch; fine-tune restores target level.",
    ("ETTm2", "ETTh1", 336): "Long-horizon route mismatch; fixed zero-shot over-corrects.",
    ("ETTm2", "ETTm1", 336): "Long-horizon route mismatch; fixed zero-shot over-corrects.",
    ("ETTm2", "ETTh1", 720): "Long-horizon route mismatch; fixed zero-shot over-corrects.",
    ("ETTm2", "ETTm1", 720): "Long-horizon route mismatch; fixed zero-shot over-corrects.",
}


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv_rows(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _float_or_none(value: Any) -> float | None:
    try:
        if value == "" or value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    try:
        if value == "" or value is None:
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _lr_key(value: Any) -> str:
    parsed = _float_or_none(value)
    if parsed is None:
        return str(value)
    return f"{parsed:g}"


def select_best_validation_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    best: dict[tuple[str, str, int], tuple[float, dict[str, Any]]] = {}
    for row in rows:
        if str(row.get("status", "")).lower() != "ok":
            continue
        horizon = _int_or_none(row.get("pred_len"))
        val_mse = _float_or_none(row.get("finetune_val_mse"))
        source = str(row.get("source", ""))
        target = str(row.get("target", ""))
        if not source or not target or horizon is None or val_mse is None:
            continue
        key = (source, target, horizon)
        current = best.get(key)
        if current is None or val_mse < current[0]:
            best[key] = (val_mse, row)

    return [
        best[key][1]
        for key in sorted(best, key=lambda item: _sort_key(*item))
    ]


def _sort_key(source: str, target: str, horizon: int) -> tuple[int, int, int]:
    source_idx = SOURCE_ORDER.index(source) if source in SOURCE_ORDER else len(SOURCE_ORDER)
    targets = TARGET_ORDER.get(source, [])
    target_idx = targets.index(target) if target in targets else len(targets)
    horizon_idx = HORIZON_ORDER.index(horizon) if horizon in HORIZON_ORDER else len(HORIZON_ORDER)
    return source_idx, target_idx, horizon_idx


def zero_shot_degradation_note(source: str, target: str, horizon: int) -> str:
    return ZERO_SHOT_DEGRADATION_NOTES.get((source, target, int(horizon)), "")


def lr_coverage(rows: Iterable[dict[str, Any]]) -> dict[tuple[str, str, int], set[str]]:
    coverage: dict[tuple[str, str, int], set[str]] = {}
    for row in rows:
        if str(row.get("status", "")).lower() != "ok":
            continue
        horizon = _int_or_none(row.get("pred_len"))
        if horizon is None:
            continue
        key = (str(row.get("source", "")), str(row.get("target", "")), horizon)
        coverage.setdefault(key, set()).add(_lr_key(row.get("finetune_lr")))
    return coverage


def missing_lr_cells(rows: Iterable[dict[str, Any]], expected_lrs: Iterable[str] = EXPECTED_LRS) -> list[str]:
    expected = {_lr_key(lr) for lr in expected_lrs}
    all_keys = [
        (source, target, horizon)
        for source in SOURCE_ORDER
        for target in TARGET_ORDER[source]
        for horizon in HORIZON_ORDER
    ]
    missing = []
    coverage = lr_coverage(rows)
    for key in sorted(all_keys, key=lambda item: _sort_key(*item)):
        seen = coverage.get(key, set())
        lack = sorted(expected - seen)
        if lack:
            missing.append(f"{key[0]}->{key[1]} H{key[2]} missing {','.join(lack)}")
    return missing


def _fmt_metric(mse: Any, mae: Any) -> str:
    mse_f = _float_or_none(mse)
    mae_f = _float_or_none(mae)
    if mse_f is None or mae_f is None:
        return ""
    return f"{mse_f:.4f}/{mae_f:.4f}"


def _fmt_lr(value: Any) -> str:
    parsed = _float_or_none(value)
    if parsed is None:
        return str(value)
    return f"{parsed:.0e}"


def _fmt_gain(value: Any) -> str:
    parsed = _float_or_none(value)
    if parsed is None:
        return ""
    return f"{parsed:.2f}%"


def parse_main_table_self_metrics(markdown: str) -> dict[tuple[str, int], tuple[float, float]]:
    metrics: dict[tuple[str, int], tuple[float, float]] = {}
    for line in markdown.splitlines():
        if line.startswith("## 2."):
            break
        if not (line.startswith("| ETTh") or line.startswith("| ETTm")):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) != 5:
            continue
        dataset = cells[0]
        for horizon, cell in zip(HORIZON_ORDER, cells[1:]):
            if "/" not in cell:
                continue
            mse_s, mae_s = [part.strip() for part in cell.split("/", maxsplit=1)]
            mse = _float_or_none(mse_s)
            mae = _float_or_none(mae_s)
            if mse is not None and mae is not None:
                metrics[(dataset, horizon)] = (mse, mae)
    return metrics or dict(DEFAULT_MAIN_SELF_METRICS)


def align_table1_self_metrics(
    row: dict[str, Any],
    self_metrics: dict[tuple[str, int], tuple[float, float]] | None = None,
) -> dict[str, Any]:
    aligned = dict(row)
    metrics = self_metrics or DEFAULT_MAIN_SELF_METRICS
    horizon = _int_or_none(row.get("pred_len"))
    if horizon is None:
        return aligned
    source = str(row.get("source", ""))
    target = str(row.get("target", ""))
    source_metric = metrics.get((source, horizon))
    target_metric = metrics.get((target, horizon))
    if source_metric is not None:
        aligned["source_test_mse"], aligned["source_test_mae"] = source_metric
    if target_metric is not None:
        aligned["target_self_test_mse"], aligned["target_self_test_mae"] = target_metric
        target_mse = target_metric[0]
        ft = _float_or_none(row.get("finetune_test_mse"))
        if target_mse and ft is not None:
            aligned["finetune_gain_pct_vs_target_self"] = (target_mse - ft) / target_mse * 100.0
    return aligned


def build_markdown_rows(
    selected_rows: Iterable[dict[str, Any]],
    self_metrics: dict[tuple[str, int], tuple[float, float]] | None = None,
) -> list[str]:
    lines = [
        "| " + " | ".join(TRANSFER_FIELDS) + " |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for raw_row in selected_rows:
        row = align_table1_self_metrics(raw_row, self_metrics)
        source = str(row.get("source", ""))
        target = str(row.get("target", ""))
        horizon = _int_or_none(row.get("pred_len")) or 0
        ft_gain_vs_zero = row.get("finetune_gain_pct_vs_zero_shot")
        if _float_or_none(ft_gain_vs_zero) is None:
            zero = _float_or_none(row.get("zero_shot_mse"))
            ft = _float_or_none(row.get("finetune_test_mse"))
            ft_gain_vs_zero = (zero - ft) / zero * 100.0 if zero and ft is not None else ""
        ft_gain_vs_target = row.get("finetune_gain_pct_vs_target_self")
        if _float_or_none(ft_gain_vs_target) is None:
            target_self = _float_or_none(row.get("target_self_test_mse"))
            ft = _float_or_none(row.get("finetune_test_mse"))
            ft_gain_vs_target = (target_self - ft) / target_self * 100.0 if target_self and ft is not None else ""

        cells = [
            source,
            target,
            str(horizon),
            _fmt_metric(row.get("source_test_mse"), row.get("source_test_mae")),
            _fmt_metric(row.get("target_self_test_mse"), row.get("target_self_test_mae")),
            _fmt_metric(row.get("zero_shot_mse"), row.get("zero_shot_mae")),
            _fmt_metric(row.get("finetune_test_mse"), row.get("finetune_test_mae")),
            _fmt_gain(ft_gain_vs_zero),
            _fmt_gain(ft_gain_vs_target),
            zero_shot_degradation_note(source, target, horizon),
            "train-only route/norm/cluster; calib=False; KNN=False",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def replace_transfer_table(markdown: str, selected_rows: Iterable[dict[str, Any]]) -> str:
    marker = "| Source | Target | H | Source self MSE/MAE | Target self MSE/MAE | Zero-shot MSE/MAE | Fine-tune MSE/MAE |"
    start = markdown.find(marker)
    if start < 0:
        raise ValueError("Could not find transfer table header in markdown.")
    end = markdown.find("\n\n", start)
    if end < 0:
        raise ValueError("Could not find transfer table end in markdown.")
    table = "\n".join(build_markdown_rows(selected_rows, parse_main_table_self_metrics(markdown)))
    return markdown[:start] + table + markdown[end:]


def update_markdown_table(markdown_path: Path, selected_rows: Iterable[dict[str, Any]]) -> None:
    text = markdown_path.read_text(encoding="utf-8")
    markdown_path.write_text(replace_transfer_table(text, selected_rows), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select transfer fine-tune rows by validation MSE and update Table 9.")
    parser.add_argument("--csv", action="append", type=Path, default=None, help="Transfer fine-tune CSV. Can be repeated.")
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--selected-csv", type=Path, default=ROOT / "outputs" / "experiment_excel_summary" / "transfer_table9_finetune_val_selected.csv")
    parser.add_argument("--allow-missing-lrs", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    csv_paths = args.csv or DEFAULT_TRANSFER_CSVS
    rows: list[dict[str, Any]] = []
    for path in csv_paths:
        rows.extend(read_csv_rows(path))
    missing = missing_lr_cells(rows)
    if missing and not args.allow_missing_lrs:
        raise SystemExit("Missing LR runs:\n" + "\n".join(missing))
    selected = select_best_validation_rows(rows)
    self_metrics = parse_main_table_self_metrics(args.markdown.read_text(encoding="utf-8"))
    selected = [align_table1_self_metrics(row, self_metrics) for row in selected]
    write_csv_rows(args.selected_csv, selected, list(selected[0].keys()) if selected else [])
    update_markdown_table(args.markdown, selected)
    print(f"Selected rows: {len(selected)}")
    print(f"Wrote: {args.selected_csv}")
    print(f"Updated: {args.markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
