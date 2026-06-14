import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.parameter_overhead_table import (
    baseline_display_dataset,
    build_parameter_comparison_table,
    estimate_model_params,
    parse_tsl_long_term_configs,
    read_adapter_overhead_row,
    read_pkr_overhead_row,
)


def test_read_adapter_overhead_row_uses_delta_instead_of_full_moe_params(tmp_path: Path) -> None:
    csv_path = tmp_path / "h96_overhead_estimate.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "setting",
                "moe_off_params",
                "pkr_moe_params",
                "epoch_time_overhead_pct",
                "gpu_alloc_overhead_pct",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "setting": "ETTm1 H96",
                "moe_off_params": "332832",
                "pkr_moe_params": "562770",
                "epoch_time_overhead_pct": "116.378",
                "gpu_alloc_overhead_pct": "36.009",
            }
        )

    row = read_adapter_overhead_row(csv_path)

    assert row["params"] == 229938
    assert row["display_name"] == "PKR adapter/add-on"


def test_read_pkr_overhead_row_keeps_full_base_and_adapter_counts(tmp_path: Path) -> None:
    csv_path = tmp_path / "h96_overhead_estimate.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "setting",
                "moe_off_params",
                "pkr_moe_params",
                "epoch_time_overhead_pct",
                "gpu_alloc_overhead_pct",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "setting": "ETTm1 H96",
                "moe_off_params": "332832",
                "pkr_moe_params": "562770",
                "epoch_time_overhead_pct": "116.378",
                "gpu_alloc_overhead_pct": "36.009",
            }
        )

    row = read_pkr_overhead_row(csv_path)

    assert row["display_name"] == "PKR-MoE full model"
    assert row["params"] == 562770
    assert row["base_params"] == 332832
    assert row["adapter_params"] == 229938


def test_parse_tsl_long_term_configs_expands_shell_variables(tmp_path: Path) -> None:
    script = tmp_path / "PatchTST_ETTm1.sh"
    script.write_text(
        "\n".join(
            [
                "model_name=PatchTST",
                "seq_len=96",
                "python -u run.py \\",
                "  --task_name long_term_forecast \\",
                "  --model $model_name \\",
                "  --data ETTm1 \\",
                "  --model_id ETTm1_$seq_len'_'192 \\",
                "  --seq_len $seq_len \\",
                "  --pred_len 192 \\",
                "  --enc_in 7 \\",
                "  --c_out 7",
            ]
        ),
        encoding="utf-8",
    )

    configs = parse_tsl_long_term_configs([script])

    assert configs == [
        {
            "config_path": str(script),
            "model": "PatchTST",
            "data": "ETTm1",
            "horizon": 192,
            "args": {
                "task_name": "long_term_forecast",
                "model": "PatchTST",
                "data": "ETTm1",
                "model_id": "ETTm1_96_192",
                "seq_len": "96",
                "pred_len": "192",
                "enc_in": "7",
                "c_out": "7",
            },
        }
    ]


def test_parameter_comparison_table_uses_pkr_full_for_full_model_ratio() -> None:
    table = "\n".join(
        build_parameter_comparison_table(
            {
                "display_name": "PKR-MoE full model",
                "dataset": "ETTm1",
                "horizon": 96,
                "params": 562770,
                "base_params": 332832,
                "adapter_params": 229938,
            },
            [
                {
                    "display_name": "PatchTST full model",
                    "dataset": "ETTm1",
                    "horizon": 96,
                    "params": 3740000,
                    "config_source": "PatchTST_ETTm1.sh",
                }
            ],
        )
    )

    assert "MoE-off params" not in table
    assert "PKR-MoE params" not in table
    assert "vs PKR adapter" not in table
    assert "vs PKR full" in table
    assert "6.6x" in table
    assert "PKR-MoE full model" in table
    assert "562,770" in table
    assert "229,938" in table
    assert "base + adapter" in table
    assert "cluster x penalty residual" in table
    assert "full model estimate" in table
    assert "PatchTST full model" in table


def test_estimate_patchtst_params_uses_static_formula() -> None:
    params = estimate_model_params(
        {
            "model": "PatchTST",
            "args": {
                "task_name": "long_term_forecast",
                "seq_len": "96",
                "pred_len": "8",
                "d_model": "8",
                "d_ff": "16",
                "e_layers": "1",
                "n_heads": "2",
            },
        }
    )

    assert params == 1520


def test_itransformer_ett_script_is_labeled_as_proxy_dataset() -> None:
    assert baseline_display_dataset("iTransformer", "ETTh2") == "ETT proxy"
    assert baseline_display_dataset("PatchTST", "ETTm1") == "ETTm1"
