import argparse
import csv
import re
import shlex
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MARKDOWN = ROOT / "outputs" / "experiment_excel_summary" / "paper_style_experiment_summary.md"
DEFAULT_OVERHEAD_CSV = ROOT / "outputs" / "experiment_excel_summary" / "h96_overhead_estimate.csv"
DEFAULT_OUTPUT_CSV = ROOT / "outputs" / "experiment_excel_summary" / "table8c_parameter_comparison.csv"
DEFAULT_TSL_ROOT = Path(r"F:\Python program\Time-Series-Library")

TSL_SCRIPT_GLOBS = ("PatchTST_*.sh", "iTransformer_*.sh", "TimeMixer_*.sh")

RUN_DEFAULTS: dict[str, Any] = {
    "task_name": "long_term_forecast",
    "seq_len": 96,
    "label_len": 48,
    "pred_len": 96,
    "enc_in": 7,
    "dec_in": 7,
    "c_out": 7,
    "d_model": 512,
    "n_heads": 8,
    "e_layers": 2,
    "d_layers": 1,
    "d_ff": 2048,
    "moving_avg": 25,
    "factor": 1,
    "dropout": 0.1,
    "embed": "timeF",
    "freq": "h",
    "activation": "gelu",
    "channel_independence": 1,
    "decomp_method": "moving_avg",
    "use_norm": 1,
    "down_sampling_layers": 0,
    "down_sampling_window": 1,
    "down_sampling_method": None,
    "top_k": 5,
    "num_class": 0,
}

INT_ARGS = {
    "seq_len",
    "label_len",
    "pred_len",
    "enc_in",
    "dec_in",
    "c_out",
    "d_model",
    "n_heads",
    "e_layers",
    "d_layers",
    "d_ff",
    "moving_avg",
    "factor",
    "channel_independence",
    "use_norm",
    "down_sampling_layers",
    "down_sampling_window",
    "top_k",
    "num_class",
    "itr",
    "batch_size",
}
FLOAT_ARGS = {"dropout", "learning_rate"}


def _to_int(value: Any) -> int:
    return int(float(str(value)))


def _to_float(value: Any) -> float:
    return float(str(value))


def _fmt_int(value: int) -> str:
    return f"{int(value):,}"


def _fmt_pct(value: Any) -> str:
    try:
        return f"{float(value):.1f}%"
    except (TypeError, ValueError):
        return ""


def _fmt_ratio(params: int, reference_params: int) -> str:
    if reference_params <= 0:
        return ""
    return f"{params / reference_params:.1f}x"


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def read_adapter_overhead_row(path: Path = DEFAULT_OVERHEAD_CSV) -> dict[str, Any]:
    rows = read_csv_rows(path)
    if not rows:
        raise ValueError(f"No rows in overhead CSV: {path}")
    row = rows[0]
    adapter_params = _to_int(row["pkr_moe_params"]) - _to_int(row["moe_off_params"])
    horizon_match = re.search(r"H=?(\d+)", row.get("setting", ""))
    horizon = int(horizon_match.group(1)) if horizon_match else 96
    dataset = row.get("setting", "").split()[0] if row.get("setting") else "ETTm1"
    return {
        "display_name": "PKR adapter/add-on",
        "model": "PKR-MoE",
        "dataset": dataset,
        "horizon": horizon,
        "params": adapter_params,
        "time_overhead": row.get("epoch_time_overhead_pct", ""),
        "gpu_overhead": row.get("gpu_alloc_overhead_pct", ""),
        "config_source": path.name,
    }


def read_pkr_overhead_row(path: Path = DEFAULT_OVERHEAD_CSV) -> dict[str, Any]:
    rows = read_csv_rows(path)
    if not rows:
        raise ValueError(f"No rows in overhead CSV: {path}")
    row = rows[0]
    base_params = _to_int(row["moe_off_params"])
    full_params = _to_int(row["pkr_moe_params"])
    horizon_match = re.search(r"H=?(\d+)", row.get("setting", ""))
    horizon = int(horizon_match.group(1)) if horizon_match else 96
    dataset = row.get("setting", "").split()[0] if row.get("setting") else "ETTm1"
    return {
        "display_name": "PKR-MoE full model",
        "model": "PKR-MoE",
        "dataset": dataset,
        "horizon": horizon,
        "params": full_params,
        "base_params": base_params,
        "adapter_params": full_params - base_params,
        "time_overhead": row.get("epoch_time_overhead_pct", ""),
        "gpu_overhead": row.get("gpu_alloc_overhead_pct", ""),
        "config_source": path.name,
    }


def _strip_assignment_value(value: str) -> str:
    value = value.strip()
    if (value.startswith("'") and value.endswith("'")) or (value.startswith('"') and value.endswith('"')):
        return value[1:-1]
    return value


def _expand_shell_vars(text: str, variables: dict[str, str]) -> str:
    def repl(match: re.Match[str]) -> str:
        return variables.get(match.group(1), match.group(0))

    return re.sub(r"\$([A-Za-z_][A-Za-z0-9_]*)", repl, text)


def _parse_command_args(command: str, variables: dict[str, str]) -> dict[str, str]:
    expanded = _expand_shell_vars(command, variables)
    tokens = shlex.split(expanded, posix=True)
    args: dict[str, str] = {}
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token.startswith("--"):
            key = token[2:].replace("-", "_")
            if i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
                args[key] = tokens[i + 1]
                i += 2
            else:
                args[key] = "1"
                i += 1
        else:
            i += 1
    return args


def parse_tsl_long_term_configs(script_paths: Iterable[Path]) -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []
    for path in script_paths:
        variables: dict[str, str] = {}
        current_command: list[str] = []
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or line.startswith("export "):
                continue

            assignment = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)=(.+)$", line)
            if assignment and not current_command and not line.startswith("python "):
                variables[assignment.group(1)] = _strip_assignment_value(assignment.group(2))
                continue

            if line.startswith("python ") or current_command:
                continuation = line.endswith("\\")
                current_command.append(line[:-1].strip() if continuation else line)
                if continuation:
                    continue
                args = _parse_command_args(" ".join(current_command), variables)
                if args.get("task_name") == "long_term_forecast":
                    configs.append(
                        {
                            "config_path": str(path),
                            "model": args.get("model", variables.get("model_name", "")),
                            "data": args.get("data", ""),
                            "horizon": _to_int(args.get("pred_len", RUN_DEFAULTS["pred_len"])),
                            "args": args,
                        }
                    )
                current_command = []
    return configs


def find_tsl_script_paths(tsl_root: Path) -> list[Path]:
    script_dir = tsl_root / "scripts" / "long_term_forecast" / "ETT_script"
    paths: list[Path] = []
    for pattern in TSL_SCRIPT_GLOBS:
        paths.extend(sorted(script_dir.glob(pattern)))
    return sorted(paths, key=lambda path: path.name)


def _coerce_arg(key: str, value: Any) -> Any:
    if key in INT_ARGS:
        return _to_int(value)
    if key in FLOAT_ARGS:
        return _to_float(value)
    if value == "None":
        return None
    return value


def _config_value(config: dict[str, Any], key: str) -> Any:
    values = dict(RUN_DEFAULTS)
    for arg_key, value in config.get("args", {}).items():
        values[arg_key] = _coerce_arg(arg_key, value)
    return values[key]


def _attention_layer_params(d_model: int, n_heads: int) -> int:
    inner_dim = (d_model // n_heads) * n_heads
    return 3 * (d_model * inner_dim + inner_dim) + (inner_dim * d_model + d_model)


def _encoder_layer_params(d_model: int, d_ff: int, n_heads: int) -> int:
    attention = _attention_layer_params(d_model, n_heads)
    feed_forward = (d_model * d_ff + d_ff) + (d_ff * d_model + d_model)
    layer_norms = 4 * d_model
    return attention + feed_forward + layer_norms


def _patchtst_params(config: dict[str, Any]) -> int:
    seq_len = _config_value(config, "seq_len")
    pred_len = _config_value(config, "pred_len")
    d_model = _config_value(config, "d_model")
    d_ff = _config_value(config, "d_ff")
    e_layers = _config_value(config, "e_layers")
    n_heads = _config_value(config, "n_heads")
    patch_len = 16
    stride = 8

    patch_embedding = patch_len * d_model
    encoder = e_layers * _encoder_layer_params(d_model, d_ff, n_heads)
    encoder_batch_norm = 2 * d_model
    patch_num = int((seq_len - patch_len) / stride + 2)
    head_nf = d_model * patch_num
    head = head_nf * pred_len + pred_len
    return patch_embedding + encoder + encoder_batch_norm + head


def _itransformer_params(config: dict[str, Any]) -> int:
    seq_len = _config_value(config, "seq_len")
    pred_len = _config_value(config, "pred_len")
    d_model = _config_value(config, "d_model")
    d_ff = _config_value(config, "d_ff")
    e_layers = _config_value(config, "e_layers")
    n_heads = _config_value(config, "n_heads")

    inverted_embedding = seq_len * d_model + d_model
    encoder = e_layers * _encoder_layer_params(d_model, d_ff, n_heads)
    encoder_layer_norm = 2 * d_model
    projection = d_model * pred_len + pred_len
    return inverted_embedding + encoder + encoder_layer_norm + projection


def _temporal_embedding_params(embed: str, freq: str, d_model: int) -> int:
    if embed == "timeF":
        freq_dims = {"h": 4, "t": 5, "s": 6, "m": 1, "a": 1, "w": 2, "d": 3, "b": 3}
        return freq_dims.get(freq, 4) * d_model
    if embed == "fixed":
        return 0
    # Learnable calendar embeddings: hour, weekday, day, month, plus minute for minutely data.
    calendar_slots = 24 + 7 + 32 + 13 + (4 if freq == "t" else 0)
    return calendar_slots * d_model


def _linear_params(in_features: int, out_features: int, bias: bool = True) -> int:
    return in_features * out_features + (out_features if bias else 0)


def _timemixer_pdm_block_params(config: dict[str, Any]) -> int:
    seq_len = _config_value(config, "seq_len")
    d_model = _config_value(config, "d_model")
    d_ff = _config_value(config, "d_ff")
    channel_independence = _config_value(config, "channel_independence")
    down_layers = _config_value(config, "down_sampling_layers")
    down_window = _config_value(config, "down_sampling_window")

    params = 2 * d_model  # layer_norm
    if not channel_independence:
        params += _linear_params(d_model, d_ff) + _linear_params(d_ff, d_model)

    for i in range(down_layers):
        high = seq_len // (down_window ** i)
        low = seq_len // (down_window ** (i + 1))
        params += _linear_params(high, low) + _linear_params(low, low)

    for i in reversed(range(down_layers)):
        low = seq_len // (down_window ** (i + 1))
        high = seq_len // (down_window ** i)
        params += _linear_params(low, high) + _linear_params(high, high)

    params += _linear_params(d_model, d_ff) + _linear_params(d_ff, d_model)
    return params


def _timemixer_params(config: dict[str, Any]) -> int:
    seq_len = _config_value(config, "seq_len")
    pred_len = _config_value(config, "pred_len")
    enc_in = _config_value(config, "enc_in")
    c_out = _config_value(config, "c_out")
    d_model = _config_value(config, "d_model")
    e_layers = _config_value(config, "e_layers")
    embed = _config_value(config, "embed")
    freq = _config_value(config, "freq")
    channel_independence = _config_value(config, "channel_independence")
    down_layers = _config_value(config, "down_sampling_layers")
    down_window = _config_value(config, "down_sampling_window")

    embedding_channels = 1 if channel_independence else enc_in
    embedding = embedding_channels * d_model * 3
    embedding += _temporal_embedding_params(embed, freq, d_model)

    normalize_layers = (down_layers + 1) * (2 * enc_in)
    pdm_blocks = e_layers * _timemixer_pdm_block_params(config)

    predict_layers = 0
    for i in range(down_layers + 1):
        length = seq_len // (down_window ** i)
        predict_layers += _linear_params(length, pred_len)

    if channel_independence:
        projection = _linear_params(d_model, 1)
        extra_output = 0
    else:
        projection = _linear_params(d_model, c_out)
        extra_output = 0
        for i in range(down_layers + 1):
            length = seq_len // (down_window ** i)
            extra_output += _linear_params(length, length)
            extra_output += _linear_params(length, pred_len)

    return embedding + normalize_layers + pdm_blocks + predict_layers + projection + extra_output


def estimate_model_params(config: dict[str, Any]) -> int:
    model = config["model"]
    if model == "PatchTST":
        return _patchtst_params(config)
    if model == "iTransformer":
        return _itransformer_params(config)
    if model == "TimeMixer":
        return _timemixer_params(config)
    raise ValueError(f"Unsupported model for static parameter estimate: {model}")


def baseline_display_dataset(model: str, dataset: str) -> str:
    if model == "iTransformer" and dataset.startswith("ETT"):
        return "ETT proxy"
    return dataset


def estimate_baseline_rows(tsl_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for config in parse_tsl_long_term_configs(find_tsl_script_paths(tsl_root)):
        params = estimate_model_params(config)
        args = config["args"]
        rows.append(
            {
                "display_name": f"{config['model']} full model",
                "model": config["model"],
                "dataset": baseline_display_dataset(config["model"], config["data"]),
                "horizon": config["horizon"],
                "params": params,
                "seq_len": _coerce_arg("seq_len", args.get("seq_len", RUN_DEFAULTS["seq_len"])),
                "d_model": _coerce_arg("d_model", args.get("d_model", RUN_DEFAULTS["d_model"])),
                "d_ff": _coerce_arg("d_ff", args.get("d_ff", RUN_DEFAULTS["d_ff"])),
                "e_layers": _coerce_arg("e_layers", args.get("e_layers", RUN_DEFAULTS["e_layers"])),
                "n_heads": _coerce_arg("n_heads", args.get("n_heads", RUN_DEFAULTS["n_heads"])),
                "config_dataset": config["data"],
                "config_source": Path(config["config_path"]).name,
            }
        )
    return sorted(rows, key=lambda row: (row["horizon"], row["model"], row["dataset"]))


def select_markdown_baseline_rows(pkr_row: dict[str, Any], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    horizon = pkr_row["horizon"]
    return [
        row
        for row in rows
        if row["horizon"] == horizon and row["model"] in {"PatchTST", "iTransformer", "TimeMixer"}
    ]


def build_parameter_comparison_table(
    pkr_row: dict[str, Any],
    baseline_rows: Iterable[dict[str, Any]],
) -> list[str]:
    pkr_params = int(pkr_row["params"])
    lines = [
        "| Model/config | Parameter scope | Dataset | H | Full params | PKR add-on params | vs PKR full | Config/source note |",
        "|---|---|---|---:|---:|---:|---:|---|",
        (
            f"| {pkr_row['display_name']} | base + adapter | {pkr_row['dataset']} | {pkr_row['horizon']} | "
            f"{_fmt_int(pkr_params)} | {_fmt_int(int(pkr_row.get('adapter_params', 0)))} | 1.0x | "
            f"{pkr_row.get('config_source', '')}; base={_fmt_int(int(pkr_row.get('base_params', 0)))}; "
            "add-on=cluster x penalty residual adapters |"
        ),
    ]
    for row in baseline_rows:
        params = int(row["params"])
        model = row.get("model", "")
        scope = "full model estimate"
        if model == "TimeMixer":
            scope = "full model estimate; TSL compact CI config"
        elif model == "iTransformer":
            scope = "full model estimate; ETT proxy"
        lines.append(
            f"| {row['display_name']} | {scope} | {row['dataset']} | {row['horizon']} | "
            f"{_fmt_int(params)} | - | {_fmt_ratio(params, pkr_params)} | {row.get('config_source', '')} |"
        )
    return lines


def _range_text(rows: list[dict[str, Any]], model: str) -> str:
    params = [int(row["params"]) for row in rows if row["model"] == model]
    if not params:
        return f"{model}: no ETT script found"
    return f"{model}: {_fmt_int(min(params))}-{_fmt_int(max(params))} params"


def build_parameter_overhead_section(pkr_row: dict[str, Any], baseline_rows: list[dict[str, Any]]) -> str:
    markdown_rows = select_markdown_baseline_rows(pkr_row, baseline_rows)
    table = "\n".join(build_parameter_comparison_table(pkr_row, markdown_rows))
    ranges = "; ".join(
        _range_text(baseline_rows, model)
        for model in ["PatchTST", "iTransformer", "TimeMixer"]
    )
    return "\n".join(
        [
            "Table 8c reports parameter counts for the selected ETTm1 H=96 setting. "
            "For fair model-size comparison, the table uses full PKR-MoE parameters against external full-model estimates; "
            "the PKR adapter/add-on count is kept as an auxiliary column. "
            "External baseline counts are full trainable parameters estimated with static formulas from the PatchTST, "
            "iTransformer, and TimeMixer source code and shell configurations in `F:\\Python program\\Time-Series-Library`. "
            "The local TSL ETT scripts include only `iTransformer_ETTh2.sh`; because ETT variants use the same 7 input "
            "channels, its count is shown as an ETT proxy rather than a dataset-specific iTransformer run. "
            "TimeMixer is marked as the compact channel-independent ETT script configuration (`d_model=16/32`, `d_ff=32`).",
            "",
            table,
            "",
            (
                f"For the PKR row, full params are {_fmt_int(int(pkr_row.get('params', 0)))}; the add-on count "
                f"{_fmt_int(int(pkr_row.get('adapter_params', 0)))} is full PKR-MoE minus MoE-off base "
                f"{_fmt_int(int(pkr_row.get('base_params', 0)))} from the existing H=96 overhead log. "
                f"Training-time overhead is {_fmt_pct(pkr_row.get('time_overhead'))} and GPU allocation overhead is "
                f"{_fmt_pct(pkr_row.get('gpu_overhead'))}. Across all counted ETT-script "
                f"horizons, full-model estimates are {ranges}. The complete per-horizon estimate is written to "
                "`outputs/experiment_excel_summary/table8c_parameter_comparison.csv`."
            ),
        ]
    )


def replace_parameter_overhead_section(markdown: str, replacement: str) -> str:
    heading = "### Parameter and Runtime Overhead"
    next_heading = "### PKR-MoE Evidence Summary"
    start = markdown.find(heading)
    if start < 0:
        raise ValueError("Could not find parameter overhead heading.")
    body_start = markdown.find("\n\n", start)
    if body_start < 0:
        raise ValueError("Could not find parameter overhead body.")
    end = markdown.find(next_heading, body_start)
    if end < 0:
        raise ValueError("Could not find next section heading.")
    return markdown[: body_start + 2] + replacement.rstrip() + "\n\n" + markdown[end:]


def write_parameter_outputs(
    markdown_path: Path,
    output_csv: Path,
    pkr_row: dict[str, Any],
    baseline_rows: list[dict[str, Any]],
) -> None:
    pkr_params = int(pkr_row["params"])
    csv_rows = [
        {
            **pkr_row,
            "parameter_scope": "base + adapter",
            "seq_len": "",
            "d_model": "",
            "d_ff": "",
            "e_layers": "",
            "n_heads": "",
            "vs_pkr_full": "1.0",
        }
    ]
    for row in baseline_rows:
        scope = "full model estimate"
        if row["model"] == "TimeMixer":
            scope = "full model estimate; TSL compact CI config"
        elif row["model"] == "iTransformer":
            scope = "full model estimate; ETT proxy"
        csv_rows.append({**row, "adapter_params": "", "base_params": "", "parameter_scope": scope, "vs_pkr_full": f"{int(row['params']) / pkr_params:.6f}"})
    write_csv_rows(output_csv, csv_rows)

    text = markdown_path.read_text(encoding="utf-8")
    section = build_parameter_overhead_section(pkr_row, baseline_rows)
    markdown_path.write_text(replace_parameter_overhead_section(text, section), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update Table 8c with PKR full-model and TSL baseline parameter counts.")
    parser.add_argument("--tsl-root", type=Path, default=DEFAULT_TSL_ROOT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--overhead-csv", type=Path, default=DEFAULT_OVERHEAD_CSV)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pkr_row = read_pkr_overhead_row(args.overhead_csv)
    baseline_rows = estimate_baseline_rows(args.tsl_root)
    write_parameter_outputs(args.markdown, args.output_csv, pkr_row, baseline_rows)
    print(f"PKR full params: {pkr_row['params']}")
    print(f"PKR adapter/add-on params: {pkr_row['adapter_params']}")
    print(f"Counted baseline configs: {len(baseline_rows)}")
    print(f"Wrote: {args.output_csv}")
    print(f"Updated: {args.markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
