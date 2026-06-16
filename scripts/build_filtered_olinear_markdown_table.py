from __future__ import annotations

import csv
import html as htmlmod
import re
from decimal import Decimal, ROUND_HALF_UP
from html.parser import HTMLParser
from pathlib import Path
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_CSV = ROOT / "outputs/codex_table_target_20260614/input96_global_paired_backbone_moe_summary.csv"
OUT_MD = ROOT / "outputs/codex_table_target_20260614/input96_olinear_filtered_comparison.md"
ARXIV_HTML = "https://arxiv.org/html/2505.08550v2"
TABLE_ID = "A7.T15"

REMOVE_DATASETS = {"Exchange", "Traffic", "Solar-Energy"}
REMOVE_MODELS = {"TimeMixer++ 2025a"}

SOURCE_TO_SUMMARY_DATASET = {
    "ECL": "Electricity",
}


class TopLevelTableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.depth = 0
        self.in_tr = False
        self.in_cell = False
        self.rows: list[list[tuple[str, int, int]]] = []
        self.row: list[tuple[str, int, int]] = []
        self.cell_parts: list[str] = []
        self.span = (1, 1)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr = {k: v for k, v in attrs}
        if tag == "table":
            self.depth += 1
        if self.depth == 1 and tag == "tr":
            self.in_tr = True
            self.row = []
        if self.depth == 1 and tag in {"td", "th"}:
            self.in_cell = True
            self.cell_parts = []
            self.span = (int(attr.get("rowspan") or 1), int(attr.get("colspan") or 1))

    def handle_data(self, data: str) -> None:
        if self.in_cell:
            self.cell_parts.append(data)

    def handle_entityref(self, name: str) -> None:
        if self.in_cell:
            self.cell_parts.append(htmlmod.unescape(f"&{name};"))

    def handle_charref(self, name: str) -> None:
        if self.in_cell:
            self.cell_parts.append(htmlmod.unescape(f"&#{name};"))

    def handle_endtag(self, tag: str) -> None:
        if self.depth == 1 and tag in {"td", "th"} and self.in_cell:
            text = " ".join("".join(self.cell_parts).split())
            self.row.append((text, self.span[0], self.span[1]))
            self.in_cell = False
            self.cell_parts = []
        if self.depth == 1 and tag == "tr" and self.in_tr:
            self.rows.append(self.row)
            self.in_tr = False
        if tag == "table":
            self.depth -= 1


def download_olinear_table() -> list[list[str]]:
    html = urlopen(ARXIV_HTML, timeout=30).read().decode("utf-8", "ignore")
    match = re.search(
        rf'<figure class="ltx_table" id="{re.escape(TABLE_ID)}".*?</figure>',
        html,
        re.S,
    )
    if not match:
        raise RuntimeError(f"could not find table {TABLE_ID} in {ARXIV_HTML}")
    table_match = re.search(r"<table\b.*</table>", match.group(0), re.S)
    if not table_match:
        raise RuntimeError(f"could not find table body for {TABLE_ID}")
    parser = TopLevelTableParser()
    parser.feed(table_match.group(0))
    return expand_spans(parser.rows)


def expand_spans(rows: list[list[tuple[str, int, int]]]) -> list[list[str]]:
    expanded: list[list[str]] = []
    pending: dict[int, tuple[str, int]] = {}
    for raw_row in rows:
        row: list[str] = []
        col = 0

        def fill_pending() -> None:
            nonlocal col
            while col in pending:
                text, remaining = pending[col]
                row.append(text)
                if remaining <= 1:
                    del pending[col]
                else:
                    pending[col] = (text, remaining - 1)
                col += 1

        for text, rowspan, colspan in raw_row:
            fill_pending()
            for offset in range(colspan):
                row.append(text)
                if rowspan > 1:
                    pending[col + offset] = (text, rowspan - 1)
            col += colspan
        fill_pending()
        expanded.append(row)
    return expanded


def clean_number(text: str) -> Decimal:
    cleaned = text.replace("\\ul", "").replace("−", "-")
    match = re.search(r"-?\d+(?:\.\d+)?", cleaned)
    if not match:
        raise ValueError(f"not a numeric cell: {text!r}")
    return Decimal(match.group(0))


def q3(value: Decimal) -> Decimal:
    return value.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)


def fmt(value: Decimal) -> str:
    return f"{q3(value):.3f}"


def color_cell(value: Decimal, rank: str | None) -> str:
    text = fmt(value)
    if rank == "best":
        return f'<span style="color:red">{text}</span>'
    if rank == "second":
        return f'<span style="color:blue">{text}</span>'
    return text


def read_ours() -> dict[tuple[str, str], tuple[Decimal, Decimal]]:
    rows: dict[tuple[str, str], tuple[Decimal, Decimal]] = {}
    with SUMMARY_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            rows[(row["dataset"], row["horizon"])] = (
                Decimal(row["moe_mse"]),
                Decimal(row["moe_mae"]),
            )
    return rows


def avg_ours(source_dataset: str, metric_idx: int, ours: dict[tuple[str, str], tuple[Decimal, Decimal]]) -> Decimal:
    summary_dataset = SOURCE_TO_SUMMARY_DATASET.get(source_dataset, source_dataset)
    horizons = ["12", "24", "48", "96"] if source_dataset.startswith("PEMS") else ["96", "192", "336", "720"]
    vals = [ours[(summary_dataset, h)][metric_idx] for h in horizons]
    return sum(vals, Decimal("0")) / Decimal(len(vals))


def model_name(header_text: str) -> str:
    if header_text == "OLinear (Ours)":
        return "OLinear"
    return re.sub(r"\s+(\d{4}[a-z]?)$", r" (\1)", header_text)


def build_rows(grid: list[list[str]]) -> tuple[list[str], list[dict[str, object]]]:
    source_model_columns: list[tuple[str, int, int]] = []
    for col in range(2, len(grid[1]), 2):
        source_name = grid[1][col]
        if source_name in REMOVE_MODELS:
            continue
        source_model_columns.append((model_name(source_name), col, col + 1))

    model_names = ["PKR-MoE (Ours)"] + [name for name, _, _ in source_model_columns]
    ours = read_ours()
    data_rows: list[dict[str, object]] = []

    for row in grid[3:-1]:
        dataset, horizon = row[0], row[1]
        if dataset in REMOVE_DATASETS:
            continue
        values: dict[str, tuple[Decimal, Decimal]] = {}
        if horizon == "Avg":
            values["PKR-MoE (Ours)"] = (avg_ours(dataset, 0, ours), avg_ours(dataset, 1, ours))
        else:
            summary_dataset = SOURCE_TO_SUMMARY_DATASET.get(dataset, dataset)
            values["PKR-MoE (Ours)"] = ours[(summary_dataset, horizon)]
        for name, mse_col, mae_col in source_model_columns:
            values[name] = (clean_number(row[mse_col]), clean_number(row[mae_col]))
        data_rows.append({"dataset": dataset, "horizon": horizon, "values": values})
    return model_names, data_rows


def ranks_for_metric(model_names: list[str], values: dict[str, tuple[Decimal, Decimal]], metric_idx: int) -> dict[str, str | None]:
    rounded = {name: q3(values[name][metric_idx]) for name in model_names}
    sorted_unique = sorted(set(rounded.values()))
    best = sorted_unique[0]
    second = sorted_unique[1] if len(sorted_unique) > 1 else None
    ranks: dict[str, str | None] = {}
    for name, value in rounded.items():
        if value == best:
            ranks[name] = "best"
        elif second is not None and value == second:
            ranks[name] = "second"
        else:
            ranks[name] = None
    return ranks


def render_markdown(model_names: list[str], data_rows: list[dict[str, object]]) -> str:
    headers = ["Dataset", "Horizon"]
    for name in model_names:
        headers.extend([f"{name} MSE", f"{name} MAE"])

    lines = [
        "# Input-96 Forecasting Comparison",
        "",
        "Source table: OLinear Table 15. Edits: removed the requested model column and dataset rows; replaced the first model column with the current PKR-MoE summary values.",
        "",
        '<span style="color:red">Red</span> = best, <span style="color:blue">Blue</span> = second best within each row and metric.',
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    best_counts = {name: [0, 0] for name in model_names}
    for item in data_rows:
        dataset = str(item["dataset"])
        horizon = str(item["horizon"])
        values = item["values"]  # type: ignore[assignment]
        assert isinstance(values, dict)
        mse_ranks = ranks_for_metric(model_names, values, 0)  # type: ignore[arg-type]
        mae_ranks = ranks_for_metric(model_names, values, 1)  # type: ignore[arg-type]
        row = [dataset, horizon]
        for name in model_names:
            mse_rank = mse_ranks[name]
            mae_rank = mae_ranks[name]
            if mse_rank == "best":
                best_counts[name][0] += 1
            if mae_rank == "best":
                best_counts[name][1] += 1
            pair = values[name]  # type: ignore[index]
            row.append(color_cell(pair[0], mse_rank))
            row.append(color_cell(pair[1], mae_rank))
        lines.append("| " + " | ".join(row) + " |")

    count_row = ["1st Count", ""]
    for name in model_names:
        count_row.extend([str(best_counts[name][0]), str(best_counts[name][1])])
    lines.append("| " + " | ".join(count_row) + " |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    grid = download_olinear_table()
    model_names, data_rows = build_rows(grid)
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text(render_markdown(model_names, data_rows), encoding="utf-8")
    print(OUT_MD)
    print(f"models={len(model_names)} rows={len(data_rows)}")


if __name__ == "__main__":
    main()
