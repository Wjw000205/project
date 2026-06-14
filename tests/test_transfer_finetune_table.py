import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.transfer_finetune_table import (
    build_markdown_rows,
    missing_lr_cells,
    select_best_validation_rows,
    zero_shot_degradation_note,
)


def test_select_best_validation_rows_chooses_lowest_finetune_val_mse() -> None:
    rows = [
        {
            "status": "ok",
            "source": "ETTm2",
            "target": "ETTh1",
            "pred_len": "336",
            "finetune_lr": "0.0001",
            "finetune_val_mse": "0.42",
        },
        {
            "status": "ok",
            "source": "ETTm2",
            "target": "ETTh1",
            "pred_len": "336",
            "finetune_lr": "5e-05",
            "finetune_val_mse": "0.39",
        },
        {
            "status": "error",
            "source": "ETTm2",
            "target": "ETTh1",
            "pred_len": "336",
            "finetune_lr": "2e-05",
            "finetune_val_mse": "0.1",
        },
        {
            "status": "ok",
            "source": "ETTm2",
            "target": "ETTm1",
            "pred_len": "96",
            "finetune_lr": "0.0001",
            "finetune_val_mse": "0.2",
        },
    ]

    selected = select_best_validation_rows(rows)

    assert [(r["target"], r["pred_len"], r["finetune_lr"]) for r in selected] == [
        ("ETTh1", "336", "5e-05"),
        ("ETTm1", "96", "0.0001"),
    ]


def test_zero_shot_degradation_note_only_marks_known_bad_cells() -> None:
    assert zero_shot_degradation_note("ETTm1", "ETTm2", 96)
    assert zero_shot_degradation_note("ETTm2", "ETTh1", 336)
    assert zero_shot_degradation_note("ETTm2", "ETTm1", 192)
    assert zero_shot_degradation_note("ETTm2", "ETTm1", 336)
    assert zero_shot_degradation_note("ETTm2", "ETTh1", 720)
    assert zero_shot_degradation_note("ETTm2", "ETTh2", 336) == ""


def test_missing_lr_cells_reports_absent_expected_transfer_cell() -> None:
    rows = [
        {
            "status": "ok",
            "source": "ETTm1",
            "target": "ETTh1",
            "pred_len": "96",
            "finetune_lr": "0.0001",
        }
    ]

    missing = missing_lr_cells(rows, expected_lrs=["0.0001"])

    assert "ETTm2->ETTm1 H720 missing 0.0001" in missing


def test_markdown_transfer_table_omits_learning_rate_column() -> None:
    table = "\n".join(
        build_markdown_rows(
            [
                {
                    "source": "ETTm1",
                    "target": "ETTm2",
                    "pred_len": "96",
                    "source_test_mse": "0.2834",
                    "source_test_mae": "0.3381",
                    "target_self_test_mse": "0.1649",
                    "target_self_test_mae": "0.2534",
                    "zero_shot_mse": "0.2030",
                    "zero_shot_mae": "0.3006",
                    "finetune_test_mse": "0.1688",
                    "finetune_test_mae": "0.2541",
                    "finetune_lr": "0.0001",
                    "finetune_gain_pct_vs_zero_shot": "16.85",
                    "finetune_gain_pct_vs_target_self": "-2.37",
                }
            ]
        )
    )

    assert "FT lr" not in table
    assert "1e-04" not in table


def test_markdown_transfer_table_aligns_self_metrics_with_table1() -> None:
    table = "\n".join(
        build_markdown_rows(
            [
                {
                    "source": "ETTm1",
                    "target": "ETTm2",
                    "pred_len": "192",
                    "source_test_mse": "0.3241",
                    "source_test_mae": "0.3643",
                    "target_self_test_mse": "0.2284",
                    "target_self_test_mae": "0.3023",
                    "zero_shot_mse": "0.2731",
                    "zero_shot_mae": "0.3507",
                    "finetune_test_mse": "0.2368",
                    "finetune_test_mae": "0.3023",
                    "finetune_gain_pct_vs_zero_shot": "13.29",
                    "finetune_gain_pct_vs_target_self": "-3.69",
                }
            ]
        )
    )

    assert "0.2284/0.3023" not in table
    assert "0.2229/0.2907" in table
    assert "-6.24%" in table
