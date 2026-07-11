from __future__ import annotations

import json
from pathlib import Path

from scripts import run_non_ecl_learnable_anchor_sweep as sweep


def _write_summary(out_dir: Path, mse: float, mae: float) -> None:
    out_dir.mkdir(parents=True)
    (out_dir / "run_summary.json").write_text(
        json.dumps({"test": {"avg_mse": mse, "avg_mae": mae}}),
        encoding="utf-8",
    )


def _write_accepted_learnable_summary(
    out_dir: Path,
    *,
    static_mse: float,
    static_mae: float,
    refined_mse: float,
    refined_mae: float,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_summary.json").write_text(
        json.dumps(
            {
                "learnable_output_anchor_test_refiner": {
                    "test_static_mse": static_mse,
                    "test_static_mae": static_mae,
                    "test_refined_mse": refined_mse,
                    "test_refined_mae": refined_mae,
                    "final_eval_uses_learnable": True,
                },
                "learnable_output_anchor_refiner": {
                    "adopted": True,
                    "final_eval_uses_learnable": True,
                },
                "stage2_trainable_parameter_groups": {
                    "total": {
                        "backbone": 0,
                        "gate": 0,
                        "pred_residual": 0,
                        "dynamic_lambda": 0,
                        "learnable_lambda": 0,
                        "learnable_output_anchor": 1,
                    }
                },
                "test": {"avg_mse": refined_mse, "avg_mae": refined_mae},
            }
        ),
        encoding="utf-8",
    )


def _write_learnable_config(
    path: Path,
    *,
    checkpoint_path: Path,
    dataset: str = "ETTm2",
    horizon: int = 192,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "data:",
                f"  csv_path: data/{dataset}.csv",
                "window:",
                f"  pred_len: {int(horizon)}",
                "finetune:",
                "  enable: true",
                "  load_model: true",
                "  load_gate: true",
                "  load_pred_residual: true",
                "  strict_model: true",
                f"  checkpoint_path: {checkpoint_path}",
                "moe:",
                "  freeze_backbone: true",
                "  learnable_output_anchor:",
                "    enable: true",
                "    train_mode: anchor_only",
                "",
            ]
        ),
        encoding="utf-8",
    )


def test_half_up_3_rounds_direct_decimal_text_without_float_coercion() -> None:
    assert sweep.half_up_3("0.35849999999999999") == "0.358"
    assert sweep.half_up_3("0.35850000000000000") == "0.359"
    assert sweep.half_up_3("0.3869410455226898") == "0.387"


def test_etth1_h96_corrected_baseline_uses_half_up_mae_target(tmp_path: Path) -> None:
    out_dir = tmp_path / "baseline"
    _write_summary(out_dir, 0.3581557273864746, 0.3869410455226898)
    config_path = tmp_path / "configs" / "ETTh1_H96.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("exp: {}\n", encoding="utf-8")
    checkpoint_path = out_dir / "best_checkpoint.pt"
    checkpoint_path.write_bytes(b"checkpoint")

    row = sweep.baseline_summary_row(
        {"dataset": "ETTh1", "horizon": "96"},
        status="reused_local",
        source_kind="corrected_etth1_h96",
        config_path=config_path,
        out_dir=out_dir,
        checkpoint_path=checkpoint_path,
        returncode=0,
        total_sec=0.0,
        error="",
    )

    assert row["baseline_mse_3dp"] == "0.358"
    assert row["baseline_mae_3dp"] == "0.387"
    assert row["table_mae_3dp"] == "0.387"
    assert row["baseline_matches_table_3dp"] is True
    assert row["baseline_strict_proven"] is True


def test_learnable_phases_require_artifact_baseline_by_default() -> None:
    all_args = sweep.normalize_args(sweep.parse_args([]))
    learnable_args = sweep.normalize_args(sweep.parse_args(["--phase", "learnable"]))
    baseline_args = sweep.normalize_args(sweep.parse_args(["--phase", "baseline"]))
    bypass_args = sweep.normalize_args(
        sweep.parse_args(["--phase", "learnable", "--allow-unproven-baseline"])
    )

    assert all_args.require_artifact_baseline is True
    assert learnable_args.require_artifact_baseline is True
    assert baseline_args.require_artifact_baseline is False
    assert bypass_args.require_artifact_baseline is False


def test_learnable_anchor_cfg_supports_hybrid_scope_and_aggregate_guards() -> None:
    args = sweep.parse_args(
        [
            "--pems-adoption-scope",
            "hybrid",
            "--aggregate-min-abs-improvement",
            "0.001",
            "--aggregate-max-abs-mae-regression",
            "0.0",
            "--aggregate-min-abs-mae-improvement",
            "0.002",
            "--aggregate-min-rel-mae-improvement",
            "0.003",
        ]
    )

    cfg = sweep.learnable_anchor_cfg("PEMS08", args)

    assert cfg["adoption"]["adoption_scope"] == "hybrid"
    assert cfg["adoption"]["aggregate_min_abs_improvement"] == 0.001
    assert cfg["adoption"]["aggregate_max_abs_mae_regression"] == 0.0
    assert cfg["adoption"]["aggregate_min_abs_mae_improvement"] == 0.002
    assert cfg["adoption"]["aggregate_min_rel_mae_improvement"] == 0.003


def test_learnable_anchor_cfg_supports_channel_horizon_block_scope() -> None:
    args = sweep.parse_args(
        [
            "--pems-adoption-scope",
            "channel_horizon_block",
            "--horizon-blocks",
            "4",
        ]
    )

    cfg = sweep.learnable_anchor_cfg("PEMS08", args)

    assert cfg["adoption"]["adoption_scope"] == "channel_horizon_block"
    assert cfg["adoption"]["horizon_segments"] == 4


def test_learnable_anchor_cfg_supports_full_parameterization_and_candidate_guard_toggle() -> None:
    args = sweep.parse_args(
        [
            "--scale-parameterization",
            "channel_horizon",
            "--bias-parameterization",
            "channel_horizon",
            "--history-trend-parameterization",
            "channel_horizon",
            "--learn-bias",
            "--max-bias",
            "0.05",
            "--disable-candidate-segment-guard",
        ]
    )

    cfg = sweep.learnable_anchor_cfg("PEMS03", args)

    assert cfg["scale_parameterization"] == "channel_horizon"
    assert cfg["bias_parameterization"] == "channel_horizon"
    assert cfg["history_trend_parameterization"] == "channel_horizon"
    assert cfg["learn_bias"] is True
    assert cfg["max_bias"] == 0.05
    assert cfg["adoption"]["candidate_segment_guard"] is False


def test_learnable_anchor_cfg_supports_nonperiodic_history_features() -> None:
    args = sweep.parse_args(
        [
            "--history-trend-feature",
            "mean_abs_diff",
            "--history-trend-window",
            "96",
        ]
    )

    cfg = sweep.learnable_anchor_cfg("ETTm2", args)

    assert cfg["history_trend_feature"] == "mean_abs_diff"
    assert cfg["history_trend_window"] == 96


def test_learnable_anchor_cfg_supports_recent_slope_history_feature() -> None:
    args = sweep.parse_args(
        [
            "--history-trend-feature",
            "recent_slope",
            "--history-trend-window",
            "96",
        ]
    )

    cfg = sweep.learnable_anchor_cfg("ETTm2", args)

    assert cfg["history_trend_feature"] == "recent_slope"
    assert cfg["history_trend_window"] == 96


def test_prepare_learnable_config_can_replay_existing_learnable_checkpoint(tmp_path: Path) -> None:
    baseline_config = tmp_path / "baseline.yaml"
    baseline_config.write_text(
        "\n".join(
            [
                "exp:",
                "  name: baseline",
                "  out_dir: baseline_out",
                "data:",
                "  csv_path: data/PEMS07.csv",
                "window:",
                "  pred_len: 24",
                "moe:",
                "  enable: true",
            ]
        ),
        encoding="utf-8",
    )
    baseline_checkpoint = tmp_path / "baseline.pt"
    baseline_checkpoint.write_bytes(b"baseline")
    replay_checkpoint = tmp_path / "learnable.pt"
    replay_checkpoint.write_bytes(b"learnable")
    args = sweep.parse_args(
        [
            "--pems-adoption-scope",
            "channel_horizon_block",
            "--learnable-replay-checkpoint",
            str(replay_checkpoint),
            "--train-lr",
            "0",
            "--anchor-lr",
            "0",
        ]
    )

    config_path, _, _ = sweep.prepare_learnable_config(
        {"dataset": "PEMS07", "horizon": "24"},
        baseline_config=baseline_config,
        baseline_checkpoint=baseline_checkpoint,
        out_root=tmp_path / "out",
        device="cuda:0",
        skip_test=True,
        args=args,
    )

    cfg = sweep.load_yaml(config_path)
    assert cfg["finetune"]["checkpoint_path"] == str(replay_checkpoint)
    assert cfg["finetune"]["load_learnable_output_anchor"] is True
    assert cfg["finetune"]["load_rejected_learnable_output_anchor"] is False
    assert cfg["finetune"]["strict_learnable_output_anchor"] is False
    assert cfg["train"]["lr"] == 0.0
    assert cfg["moe"]["learnable_output_anchor"]["lr"] == 0.0
    assert "replay" in config_path.stem


def test_prepare_learnable_config_forces_stage2_backbone_freeze(tmp_path: Path) -> None:
    baseline_config = tmp_path / "baseline.yaml"
    baseline_config.write_text(
        "\n".join(
            [
                "exp:",
                "  name: baseline",
                "  out_dir: baseline_out",
                "data:",
                "  csv_path: data/ETTm2.csv",
                "window:",
                "  pred_len: 192",
                "moe:",
                "  enable: true",
                "  freeze_backbone: false",
            ]
        ),
        encoding="utf-8",
    )
    baseline_checkpoint = tmp_path / "baseline.pt"
    baseline_checkpoint.write_bytes(b"baseline")

    config_path, _, _ = sweep.prepare_learnable_config(
        {"dataset": "ETTm2", "horizon": "192"},
        baseline_config=baseline_config,
        baseline_checkpoint=baseline_checkpoint,
        out_root=tmp_path / "out",
        device=None,
        skip_test=True,
        args=sweep.parse_args([]),
    )

    cfg = sweep.load_yaml(config_path)
    assert cfg["moe"]["freeze_backbone"] is True


def test_prepare_learnable_config_separates_history_feature_and_delta_variants(tmp_path: Path) -> None:
    baseline_config = tmp_path / "baseline.yaml"
    baseline_config.write_text(
        "\n".join(
            [
                "exp:",
                "  name: baseline",
                "  out_dir: baseline_out",
                "data:",
                "  csv_path: data/ETTm2.csv",
                "window:",
                "  pred_len: 720",
                "moe:",
                "  enable: true",
            ]
        ),
        encoding="utf-8",
    )
    baseline_checkpoint = tmp_path / "baseline.pt"
    baseline_checkpoint.write_bytes(b"baseline")
    common = [
        "--history-trend-window",
        "96",
        "--scale-parameterization",
        "channel_horizon",
        "--history-trend-parameterization",
        "channel_horizon",
        "--default-adoption-scope",
        "channel_horizon_block",
    ]

    lmf_config, lmf_out, _ = sweep.prepare_learnable_config(
        {"dataset": "ETTm2", "horizon": "720"},
        baseline_config=baseline_config,
        baseline_checkpoint=baseline_checkpoint,
        out_root=tmp_path / "out",
        device=None,
        skip_test=True,
        args=sweep.parse_args([*common, "--history-trend-feature", "last_minus_first", "--max-history-trend-delta", "0.20"]),
    )
    slope_config, slope_out, _ = sweep.prepare_learnable_config(
        {"dataset": "ETTm2", "horizon": "720"},
        baseline_config=baseline_config,
        baseline_checkpoint=baseline_checkpoint,
        out_root=tmp_path / "out",
        device=None,
        skip_test=True,
        args=sweep.parse_args([*common, "--history-trend-feature", "recent_slope", "--max-history-trend-delta", "0.20"]),
    )
    larger_delta_config, larger_delta_out, _ = sweep.prepare_learnable_config(
        {"dataset": "ETTm2", "horizon": "720"},
        baseline_config=baseline_config,
        baseline_checkpoint=baseline_checkpoint,
        out_root=tmp_path / "out",
        device=None,
        skip_test=True,
        args=sweep.parse_args([*common, "--history-trend-feature", "recent_slope", "--max-history-trend-delta", "0.25"]),
    )

    assert lmf_config != slope_config
    assert lmf_out != slope_out
    assert slope_config != larger_delta_config
    assert slope_out != larger_delta_out
    assert "hflast_minus_first" in lmf_config.stem
    assert "hfrecent_slope" in slope_config.stem
    assert "hd0p2" in slope_config.stem
    assert "hd0p25" in larger_delta_config.stem


def test_baseline_summary_marks_fallback_source_as_not_strict(tmp_path: Path) -> None:
    out_dir = tmp_path / "baseline"
    _write_summary(out_dir, 0.1523754894733429, 0.21607407927513123)

    row = sweep.baseline_summary_row(
        {"dataset": "Weather", "horizon": "96"},
        status="ok",
        source_kind="top_level_config_fallback",
        config_path=tmp_path / "configs" / "Weather_H96.yaml",
        out_dir=out_dir,
        checkpoint_path=out_dir / "best_checkpoint.pt",
        returncode=0,
        total_sec=0.0,
        error="",
    )

    assert row["baseline_matches_table_3dp"] is True
    assert row["baseline_strict_proven"] is False
    assert row["baseline_proof_reason"] == "fallback_source_not_strict"
    assert row["baseline_artifact_proven"] is False
    assert row["baseline_artifact_proof_reason"] == "missing_config"


def test_baseline_summary_can_artifact_prove_fallback_reproduction(tmp_path: Path) -> None:
    out_dir = tmp_path / "baseline"
    _write_summary(out_dir, 0.1523754894733429, 0.21607407927513123)
    config_path = tmp_path / "configs" / "Weather_H96.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("exp: {}\n", encoding="utf-8")
    checkpoint_path = out_dir / "best_checkpoint.pt"
    checkpoint_path.write_bytes(b"checkpoint")

    row = sweep.baseline_summary_row(
        {"dataset": "Weather", "horizon": "96"},
        status="ok",
        source_kind="top_level_config_fallback",
        config_path=config_path,
        out_dir=out_dir,
        checkpoint_path=checkpoint_path,
        returncode=0,
        total_sec=0.0,
        error="",
    )

    assert row["baseline_strict_proven"] is False
    assert row["baseline_proof_reason"] == "fallback_source_not_strict"
    assert row["baseline_artifact_proven"] is True
    assert row["baseline_artifact_proof_reason"] == "artifact_table_match"


def test_baseline_summary_can_artifact_prove_dominating_static_baseline(
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "baseline"
    _write_summary(out_dir, 0.3664566810131073, 0.37813955545425415)
    config_path = tmp_path / "configs" / "ETTm2_H720.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("exp: {}\n", encoding="utf-8")
    checkpoint_path = out_dir / "best_checkpoint.pt"
    checkpoint_path.write_bytes(b"checkpoint")

    row = sweep.baseline_summary_row(
        {"dataset": "ETTm2", "horizon": "720"},
        status="reused_external",
        source_kind="external_baseline_root",
        config_path=config_path,
        out_dir=out_dir,
        checkpoint_path=checkpoint_path,
        returncode=0,
        total_sec=0.0,
        error="",
    )

    assert row["baseline_matches_table_3dp"] is False
    assert row["baseline_mse_3dp"] == "0.366"
    assert row["baseline_mae_3dp"] == "0.378"
    assert row["baseline_strict_proven"] is False
    assert row["baseline_artifact_proven"] is True
    assert row["baseline_artifact_proof_reason"] == "artifact_table_dominates"


def test_baseline_summary_rejects_qgwnt_transfer_artifact_even_if_dominates(
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "input96_transfer_qgwnt_full_horizon" / "source" / "ETTm2" / "H336"
    _write_summary(out_dir, 0.2701, 0.3201)
    config_path = (
        tmp_path
        / "input96_transfer_qgwnt_full_horizon"
        / "configs"
        / "source"
        / "ETTm2_H336_source.yaml"
    )
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        "\n".join(
            [
                "exp:",
                "  name: input96_ETTm2_H336_qgwnt_source",
                "data:",
                "  csv_path: data/ETTm2.csv",
                "moe:",
                "  enable: true",
            ]
        ),
        encoding="utf-8",
    )
    checkpoint_path = out_dir / "best_checkpoint.pt"
    checkpoint_path.write_bytes(b"checkpoint")

    row = sweep.baseline_summary_row(
        {"dataset": "ETTm2", "horizon": "336"},
        status="reused_external",
        source_kind="external_baseline_root",
        config_path=config_path,
        out_dir=out_dir,
        checkpoint_path=checkpoint_path,
        returncode=0,
        total_sec=0.0,
        error="",
    )

    assert row["baseline_artifact_proven"] is False
    assert row["baseline_artifact_proof_reason"] == "invalid_artifact_contract:qgwnt"


def test_baseline_summary_rejects_learnable_anchor_artifact_even_if_dominates(
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "learnable_anchor" / "runs" / "Weather" / "H96" / "candidate"
    _write_summary(out_dir, 0.1511, 0.2151)
    config_path = tmp_path / "learnable_anchor" / "configs" / "Weather" / "H96" / "candidate.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        "\n".join(
            [
                "data:",
                "  csv_path: data/Weather.csv",
                "window:",
                "  pred_len: 96",
                "moe:",
                "  learnable_output_anchor:",
                "    enable: true",
                "    train_mode: anchor_only",
            ]
        ),
        encoding="utf-8",
    )
    checkpoint_path = out_dir / "best_checkpoint.pt"
    checkpoint_path.write_bytes(b"checkpoint")

    row = sweep.baseline_summary_row(
        {"dataset": "Weather", "horizon": "96"},
        status="reused_external",
        source_kind="external_baseline_root",
        config_path=config_path,
        out_dir=out_dir,
        checkpoint_path=checkpoint_path,
        returncode=0,
        total_sec=0.0,
        error="",
    )

    assert row["baseline_artifact_proven"] is False
    assert row["baseline_artifact_proof_reason"] == "invalid_artifact_contract:learnable_anchor"


def test_baseline_summary_rejects_dataset_or_horizon_mismatch(
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "static_baseline" / "runs" / "Weather" / "H96" / "candidate"
    _write_summary(out_dir, 0.1511, 0.2151)
    config_path = tmp_path / "static_baseline" / "configs" / "Weather" / "H96" / "candidate.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        "\n".join(
            [
                "data:",
                "  csv_path: data/ETTh1.csv",
                "window:",
                "  pred_len: 96",
                "moe:",
                "  enable: true",
            ]
        ),
        encoding="utf-8",
    )
    checkpoint_path = out_dir / "best_checkpoint.pt"
    checkpoint_path.write_bytes(b"checkpoint")

    row = sweep.baseline_summary_row(
        {"dataset": "Weather", "horizon": "96"},
        status="reused_external",
        source_kind="external_baseline_root",
        config_path=config_path,
        out_dir=out_dir,
        checkpoint_path=checkpoint_path,
        returncode=0,
        total_sec=0.0,
        error="",
    )

    assert row["baseline_artifact_proven"] is False
    assert row["baseline_artifact_proof_reason"] == "invalid_artifact_contract:dataset_mismatch"


def test_ensure_baseline_skips_invalid_external_qgwnt_summary_row(
    tmp_path: Path,
) -> None:
    external_root = tmp_path / "external"
    qgwnt_config = (
        external_root
        / "input96_transfer_qgwnt_full_horizon"
        / "configs"
        / "source"
        / "ETTm2_H336_source.yaml"
    )
    qgwnt_out = (
        external_root
        / "input96_transfer_qgwnt_full_horizon"
        / "source"
        / "ETTm2"
        / "H336"
    )
    qgwnt_config.parent.mkdir(parents=True)
    qgwnt_config.write_text("exp:\n  name: input96_ETTm2_H336_qgwnt_source\n", encoding="utf-8")
    _write_summary(qgwnt_out, 0.2701, 0.3201)
    (qgwnt_out / "best_checkpoint.pt").write_bytes(b"checkpoint")

    static_config = external_root / "static_baseline" / "configs" / "ETTm2" / "H336" / "static.yaml"
    static_out = external_root / "static_baseline" / "runs" / "ETTm2" / "H336" / "static"
    static_config.parent.mkdir(parents=True)
    static_config.write_text("exp:\n  name: static_ETTm2_H336\n", encoding="utf-8")
    _write_summary(static_out, 0.2769, 0.3259)
    static_checkpoint = static_out / "best_checkpoint.pt"
    static_checkpoint.write_bytes(b"checkpoint")

    sweep.write_summary(
        external_root / "summary.csv",
        [
            {
                "phase": "baseline",
                "dataset": "ETTm2",
                "horizon": "336",
                "status": "reused_external",
                "baseline_config": str(qgwnt_config),
                "baseline_out_dir": str(qgwnt_out),
                "baseline_checkpoint": str(qgwnt_out / "best_checkpoint.pt"),
                "baseline_artifact_proven": "True",
            },
            {
                "phase": "baseline",
                "dataset": "ETTm2",
                "horizon": "336",
                "status": "reused_external",
                "baseline_config": str(static_config),
                "baseline_out_dir": str(static_out),
                "baseline_checkpoint": str(static_checkpoint),
                "baseline_artifact_proven": "True",
            },
        ],
    )

    row, returned_config, returned_checkpoint = sweep.ensure_baseline(
        {
            "dataset": "ETTm2",
            "horizon": "336",
            "variant": "candidate",
            "config_path": str(tmp_path / "missing_source.yaml"),
            "out_dir": str(tmp_path / "missing_source_run"),
        },
        out_root=tmp_path / "new",
        device=None,
        skip_test=False,
        dry_run=False,
        reuse_existing=False,
        reuse_existing_only=True,
        reuse_source_baseline=False,
        baseline_reuse_root=external_root,
    )

    assert row["status"] == "reused_external"
    assert row["baseline_source"] == "external_baseline_root"
    assert row["baseline_artifact_proven"] is True
    assert returned_config == static_config
    assert returned_checkpoint == static_checkpoint


def test_learnable_baseline_gate_allows_artifact_proven_dominating_baseline() -> None:
    args = sweep.parse_args(["--require-artifact-baseline"])

    failure = sweep.learnable_baseline_gate_failure(
        args=args,
        baseline_strict=False,
        baseline_proof_reason="table_metric_mismatch",
        baseline_artifact=True,
        baseline_artifact_reason="artifact_table_dominates",
    )

    assert failure is None


def test_learnable_baseline_gate_rejects_unproven_artifact_baseline() -> None:
    args = sweep.parse_args(["--require-artifact-baseline"])

    failure = sweep.learnable_baseline_gate_failure(
        args=args,
        baseline_strict=False,
        baseline_proof_reason="table_metric_mismatch",
        baseline_artifact=False,
        baseline_artifact_reason="table_metric_mismatch",
    )

    assert failure == (
        "skipped_after_unproven_baseline",
        "Baseline artifact proof failed: table_metric_mismatch",
    )


def test_learnable_baseline_gate_strict_still_requires_exact_table_match() -> None:
    args = sweep.parse_args(["--require-strict-baseline", "--require-artifact-baseline"])

    failure = sweep.learnable_baseline_gate_failure(
        args=args,
        baseline_strict=False,
        baseline_proof_reason="table_metric_mismatch",
        baseline_artifact=True,
        baseline_artifact_reason="artifact_table_dominates",
    )

    assert failure == (
        "skipped_after_unproven_baseline",
        "Baseline strict proof failed: table_metric_mismatch",
    )


def test_baseline_summary_rejects_static_artifact_with_any_worse_metric(
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "baseline"
    _write_summary(out_dir, 0.2946547567844391, 0.3496416272163391)
    config_path = tmp_path / "configs" / "ETTm1_H96.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("exp: {}\n", encoding="utf-8")
    checkpoint_path = out_dir / "best_checkpoint.pt"
    checkpoint_path.write_bytes(b"checkpoint")

    row = sweep.baseline_summary_row(
        {"dataset": "ETTm1", "horizon": "96"},
        status="reused_external",
        source_kind="external_baseline_root",
        config_path=config_path,
        out_dir=out_dir,
        checkpoint_path=checkpoint_path,
        returncode=0,
        total_sec=0.0,
        error="",
    )

    assert row["baseline_mse_3dp"] == "0.295"
    assert row["baseline_mae_3dp"] == "0.350"
    assert row["baseline_artifact_proven"] is False
    assert row["baseline_artifact_proof_reason"] == "table_metric_mismatch"


def test_baseline_summary_requires_run_summary_for_strict_proof(tmp_path: Path) -> None:
    out_dir = tmp_path / "missing_summary"

    row = sweep.baseline_summary_row(
        {"dataset": "Weather", "horizon": "96"},
        status="reused_source_no_summary",
        source_kind="baseline_index",
        config_path=tmp_path / "configs" / "Weather_H96.yaml",
        out_dir=out_dir,
        checkpoint_path=out_dir / "best_checkpoint.pt",
        returncode=0,
        total_sec=0.0,
        error="",
    )

    assert row["baseline_matches_table_3dp"] == ""
    assert row["baseline_strict_proven"] is False
    assert row["baseline_proof_reason"] == "missing_run_summary"


def test_baseline_seed_uses_known_ettm2_h96_exact_config(
    tmp_path: Path, monkeypatch
) -> None:
    config_path = tmp_path / "ETTm2_H96_fullpool.yaml"
    out_dir = tmp_path / "ETTm2_H96_fullpool"
    config_path.write_text("exp: {}\n", encoding="utf-8")
    monkeypatch.setattr(sweep, "ETTM2_H96_FULLPOOL_CONFIG", config_path)
    monkeypatch.setattr(sweep, "ETTM2_H96_FULLPOOL_OUT_DIR", out_dir)

    source_config, source_out, source_kind = sweep.baseline_seed(
        {
            "dataset": "ETTm2",
            "horizon": "96",
            "config_path": str(tmp_path / "index.yaml"),
            "out_dir": str(tmp_path / "index_run"),
        }
    )

    assert source_config == config_path
    assert source_out == out_dir
    assert source_kind == "ettm2_h96_fullpool_exact"


def test_baseline_seed_prefers_valid_top_level_ettm2_h336_over_transfer_source(
    tmp_path: Path, monkeypatch
) -> None:
    transfer_config = tmp_path / "transfer" / "ETTm2_H336_source.yaml"
    transfer_out = tmp_path / "transfer" / "source" / "ETTm2" / "H336"
    transfer_config.parent.mkdir(parents=True)
    transfer_config.write_text("exp:\n  name: input96_ETTm2_H336_qgwnt_source\n", encoding="utf-8")
    fallback_config = tmp_path / "configs" / "ETTm2_H336.yaml"
    fallback_config.parent.mkdir(parents=True)
    fallback_config.write_text("exp:\n  name: ETTm2_H336\n", encoding="utf-8")
    row_out = tmp_path / "index_run"
    monkeypatch.setattr(sweep, "ROOT", tmp_path)
    monkeypatch.setattr(sweep, "ETTM2_H336_TRANSFER_SOURCE_CONFIG", transfer_config)
    monkeypatch.setattr(sweep, "ETTM2_H336_TRANSFER_SOURCE_OUT_DIR", transfer_out)

    source_config, source_out, source_kind = sweep.baseline_seed(
        {
            "dataset": "ETTm2",
            "horizon": "336",
            "config_path": str(tmp_path / "index.yaml"),
            "out_dir": str(row_out),
        }
    )

    assert source_config == fallback_config
    assert source_out == row_out
    assert source_kind == "top_level_config_fallback"


def test_baseline_seed_prefers_existing_generated_main_table_config(tmp_path: Path) -> None:
    config_path = tmp_path / "generated.yaml"
    out_dir = tmp_path / "generated_run"
    config_path.write_text("exp: {}\n", encoding="utf-8")

    source_config, source_out, source_kind = sweep.baseline_seed(
        {
            "dataset": "Weather",
            "horizon": "96",
            "config_path": str(config_path),
            "source_config": str(tmp_path / "source.yaml"),
            "strategy_config": str(tmp_path / "strategy.yaml"),
            "out_dir": str(out_dir),
        }
    )

    assert source_config == config_path
    assert source_out == out_dir
    assert source_kind == "baseline_index_config_path"


def test_baseline_seed_uses_strategy_config_before_top_level_fallback(tmp_path: Path) -> None:
    strategy_config = tmp_path / "strategy.yaml"
    out_dir = tmp_path / "strategy_run"
    strategy_config.write_text("exp: {}\n", encoding="utf-8")

    source_config, source_out, source_kind = sweep.baseline_seed(
        {
            "dataset": "Weather",
            "horizon": "96",
            "config_path": str(tmp_path / "missing_generated.yaml"),
            "source_config": str(tmp_path / "missing_source.yaml"),
            "strategy_config": str(strategy_config),
            "out_dir": str(out_dir),
        }
    )

    assert source_config == strategy_config
    assert source_out == out_dir
    assert source_kind == "baseline_index_strategy_config"


def test_ensure_baseline_reuses_external_baseline_root(tmp_path: Path) -> None:
    external_root = tmp_path / "external"
    config_path = external_root / "static_baseline" / "configs" / "Weather" / "H96" / "candidate.yaml"
    out_dir = external_root / "static_baseline" / "runs" / "Weather" / "H96" / "candidate"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("exp: {}\n", encoding="utf-8")
    _write_summary(out_dir, 0.1523754894733429, 0.21607407927513123)
    checkpoint_path = out_dir / "best_checkpoint.pt"
    checkpoint_path.write_bytes(b"checkpoint")

    row, returned_config, returned_checkpoint = sweep.ensure_baseline(
        {
            "dataset": "Weather",
            "horizon": "96",
            "variant": "candidate",
            "config_path": str(tmp_path / "missing_source.yaml"),
            "out_dir": str(tmp_path / "missing_source_run"),
        },
        out_root=tmp_path / "new",
        device=None,
        skip_test=False,
        dry_run=False,
        reuse_existing=False,
        reuse_existing_only=False,
        reuse_source_baseline=False,
        baseline_reuse_root=external_root,
    )

    assert row["status"] == "reused_external"
    assert row["baseline_source"] == "external_baseline_root"
    assert row["baseline_strict_proven"] is True
    assert row["baseline_artifact_proven"] is True
    assert returned_config == config_path
    assert returned_checkpoint == checkpoint_path


def test_ensure_baseline_reuses_external_summary_row_layout(tmp_path: Path) -> None:
    external_root = tmp_path / "external"
    config_path = tmp_path / "elsewhere" / "configs" / "ETTm2_H192.yaml"
    out_dir = tmp_path / "elsewhere" / "runs" / "ETTm2_H192"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("exp: {}\n", encoding="utf-8")
    _write_summary(out_dir, 0.2243470847606659, 0.28935667872428894)
    checkpoint_path = out_dir / "best_checkpoint.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    sweep.write_summary(
        external_root / "summary.csv",
        [
            {
                "phase": "baseline",
                "dataset": "ETTm2",
                "horizon": "192",
                "status": "reused_external",
                "baseline_config": str(config_path),
                "baseline_out_dir": str(out_dir),
                "baseline_checkpoint": str(checkpoint_path),
                "baseline_artifact_proven": "True",
            }
        ],
    )

    row, returned_config, returned_checkpoint = sweep.ensure_baseline(
        {
            "dataset": "ETTm2",
            "horizon": "192",
            "variant": "candidate",
            "config_path": str(tmp_path / "missing_source.yaml"),
            "out_dir": str(tmp_path / "missing_source_run"),
        },
        out_root=tmp_path / "new",
        device=None,
        skip_test=False,
        dry_run=False,
        reuse_existing=False,
        reuse_existing_only=True,
        reuse_source_baseline=False,
        baseline_reuse_root=external_root,
    )

    assert row["status"] == "reused_external"
    assert row["baseline_source"] == "external_baseline_root"
    assert row["baseline_artifact_proven"] is True
    assert returned_config == config_path
    assert returned_checkpoint == checkpoint_path


def test_ensure_baseline_reuses_external_source_layout(tmp_path: Path) -> None:
    external_root = tmp_path / "transfer_root"
    config_path = external_root / "configs" / "source" / "ETTm2_H192_source.yaml"
    out_dir = external_root / "source" / "ETTm2" / "H192"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("exp: {}\n", encoding="utf-8")
    _write_summary(out_dir, 0.2243470847606659, 0.28935667872428894)
    checkpoint_path = out_dir / "best_checkpoint.pt"
    checkpoint_path.write_bytes(b"checkpoint")

    row, returned_config, returned_checkpoint = sweep.ensure_baseline(
        {
            "dataset": "ETTm2",
            "horizon": "192",
            "variant": "candidate",
            "config_path": str(tmp_path / "missing_source.yaml"),
            "out_dir": str(tmp_path / "missing_source_run"),
        },
        out_root=tmp_path / "new",
        device=None,
        skip_test=False,
        dry_run=False,
        reuse_existing=False,
        reuse_existing_only=True,
        reuse_source_baseline=False,
        baseline_reuse_root=external_root,
    )

    assert row["status"] == "reused_external"
    assert row["baseline_source"] == "external_baseline_root"
    assert row["baseline_artifact_proven"] is True
    assert returned_config == config_path
    assert returned_checkpoint == checkpoint_path


def test_ensure_baseline_reuses_external_legacy_source_export_layout(tmp_path: Path) -> None:
    external_root = tmp_path / "legacy_root"
    config_path = external_root / "configs" / "source" / "ETTm1_H96_legacy_aligned_export.yaml"
    out_dir = external_root / "source" / "ETTm1_H96_legacy_aligned_export"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("exp: {}\n", encoding="utf-8")
    _write_summary(out_dir, 0.2946547567844391, 0.3482416272163391)
    checkpoint_path = out_dir / "best_checkpoint.pt"
    checkpoint_path.write_bytes(b"checkpoint")

    row, returned_config, returned_checkpoint = sweep.ensure_baseline(
        {
            "dataset": "ETTm1",
            "horizon": "96",
            "variant": "candidate",
            "config_path": str(tmp_path / "missing_source.yaml"),
            "out_dir": str(tmp_path / "missing_source_run"),
        },
        out_root=tmp_path / "new",
        device=None,
        skip_test=False,
        dry_run=False,
        reuse_existing=False,
        reuse_existing_only=True,
        reuse_source_baseline=False,
        baseline_reuse_root=external_root,
    )

    assert row["status"] == "reused_external"
    assert row["baseline_source"] == "external_baseline_root"
    assert row["baseline_mse_3dp"] == "0.295"
    assert row["baseline_mae_3dp"] == "0.348"
    assert row["baseline_artifact_proven"] is True
    assert row["baseline_artifact_proof_reason"] == "artifact_table_dominates"
    assert returned_config == config_path
    assert returned_checkpoint == checkpoint_path


def test_external_baseline_status_is_ready_for_learnable() -> None:
    assert sweep.baseline_status_ready("reused_external") is True


def test_reuse_existing_only_skips_missing_baseline_without_training(
    tmp_path: Path, monkeypatch
) -> None:
    def fail_train(*_args, **_kwargs):
        raise AssertionError("run_train should not be called")

    monkeypatch.setattr(sweep, "run_train", fail_train)

    config_path = tmp_path / "source.yaml"
    config_path.write_text("exp: {}\n", encoding="utf-8")
    source_out = tmp_path / "source_run"

    monkeypatch.setattr(
        sweep,
        "baseline_seed",
        lambda _row: (config_path, source_out, "baseline_index"),
    )

    row, _config, _checkpoint = sweep.ensure_baseline(
        {"dataset": "Weather", "horizon": "96", "variant": "candidate"},
        out_root=tmp_path / "new",
        device=None,
        skip_test=False,
        dry_run=False,
        reuse_existing=True,
        reuse_existing_only=True,
        reuse_source_baseline=False,
        baseline_reuse_root=None,
    )

    assert row["status"] == "missing_existing_baseline"
    assert row["error"].startswith("Missing existing baseline artifacts:")


def test_reuse_existing_only_skips_missing_learnable_without_training(
    tmp_path: Path, monkeypatch
) -> None:
    def fail_train(*_args, **_kwargs):
        raise AssertionError("run_train should not be called")

    monkeypatch.setattr(sweep, "run_train", fail_train)
    baseline_config = tmp_path / "baseline.yaml"
    baseline_config.write_text("exp: {}\n", encoding="utf-8")
    baseline_dir = tmp_path / "baseline"
    _write_summary(baseline_dir, 0.1523754894733429, 0.21607407927513123)
    baseline_checkpoint = baseline_dir / "best_checkpoint.pt"
    baseline_checkpoint.write_bytes(b"checkpoint")

    args = sweep.parse_args([])
    args.epochs = 1
    args.patience = 1

    row = sweep.run_learnable(
        {"dataset": "Weather", "horizon": "96", "variant": "candidate"},
        baseline_config=baseline_config,
        baseline_checkpoint=baseline_checkpoint,
        out_root=tmp_path / "new",
        device=None,
        skip_test=False,
        dry_run=False,
        reuse_existing=True,
        reuse_existing_only=True,
        args=args,
    )

    assert row["status"] == "missing_existing_learnable"
    assert row["error"].startswith("Missing existing learnable summary:")


def test_learnable_summary_separates_same_run_and_baseline_rounded_wins(
    tmp_path: Path,
) -> None:
    baseline_dir = tmp_path / "baseline"
    _write_summary(baseline_dir, 0.075497, 0.177914)
    baseline_config = tmp_path / "baseline.yaml"
    baseline_config.write_text("exp: {}\n", encoding="utf-8")
    baseline_checkpoint = baseline_dir / "best_checkpoint.pt"
    baseline_checkpoint.write_bytes(b"checkpoint")
    learnable_config = tmp_path / "learnable.yaml"
    _write_learnable_config(learnable_config, checkpoint_path=baseline_checkpoint)

    learnable_dir = tmp_path / "learnable"
    learnable_dir.mkdir()
    (learnable_dir / "run_summary.json").write_text(
        json.dumps(
            {
                "learnable_output_anchor_test_refiner": {
                    "test_static_mse": 0.0755440965,
                    "test_static_mae": 0.1779140085,
                    "test_refined_mse": 0.0753895566,
                    "test_refined_mae": 0.1776060015,
                    "test_mse_gain": 0.0001545399,
                    "test_mae_gain": 0.0003080070,
                },
                "learnable_output_anchor_refiner": {
                    "adopted": True,
                    "final_eval_uses_learnable": True,
                    "mse_gain": 0.0012,
                    "mae_gain": 0.0004,
                    "required_gain": 0.001,
                    "required_mae_gain": 0.0003,
                    "fallback_reason": None,
                    "aggregate_min_abs_mae_improvement": 0.0003,
                    "aggregate_min_rel_mae_improvement": 0.0,
                    "aggregate_mae_improvement_guard_enabled": True,
                },
                "stage2_trainable_parameter_groups": {
                    "total": {
                        "backbone": 0,
                        "gate": 0,
                        "pred_residual": 0,
                        "dynamic_lambda": 0,
                        "learnable_lambda": 0,
                        "learnable_output_anchor": 1,
                    }
                },
                "test": {"avg_mse": 0.0753895566, "avg_mae": 0.1776060015},
            }
        ),
        encoding="utf-8",
    )

    row = sweep.learnable_summary_row(
        {"dataset": "PEMS04", "horizon": "24"},
        status="ok",
        baseline_config=baseline_config,
        baseline_checkpoint=baseline_checkpoint,
        config_path=tmp_path / "learnable.yaml",
        out_dir=learnable_dir,
        adoption_scope="global",
        returncode=0,
        total_sec=0.0,
        error="",
    )

    assert row["rounded_mse_win"] is True
    assert row["test_static_mse_3dp"] == "0.076"
    assert row["test_refined_mse_3dp"] == "0.075"
    assert row["baseline_mse_3dp"] == "0.075"
    assert row["baseline_artifact_proven"] is True
    assert row["rounded_mse_win_vs_baseline"] is False
    assert row["baseline_refined_mse_gain"] > 0
    assert row["mae_non_regression_vs_baseline"] is True
    assert row["val_mse_gain"] == 0.0012
    assert row["val_mae_gain"] == 0.0004
    assert row["required_val_mae_gain"] == 0.0003
    assert row["final_eval_uses_learnable"] is True
    assert row["aggregate_min_abs_mae_improvement"] == 0.0003
    assert row["mae_improvement_guard_enabled"] is True
    assert row["accepted"] is False


def test_learnable_summary_marks_accepted_only_when_full_contract_passes(
    tmp_path: Path,
) -> None:
    baseline_dir = tmp_path / "baseline"
    _write_summary(baseline_dir, 0.22435617446899414, 0.2893063724040985)
    baseline_config = tmp_path / "baseline.yaml"
    baseline_config.write_text("exp: {}\n", encoding="utf-8")
    baseline_checkpoint = baseline_dir / "best_checkpoint.pt"
    baseline_checkpoint.write_bytes(b"checkpoint")
    learnable_config = tmp_path / "learnable.yaml"
    _write_learnable_config(learnable_config, checkpoint_path=baseline_checkpoint)

    learnable_dir = tmp_path / "learnable"
    learnable_dir.mkdir()
    (learnable_dir / "run_summary.json").write_text(
        json.dumps(
            {
                "learnable_output_anchor_test_refiner": {
                    "test_static_mse": 0.2243694067,
                    "test_static_mae": 0.2893727422,
                    "test_refined_mse": 0.2227451503,
                    "test_refined_mae": 0.2877233028,
                    "final_eval_uses_learnable": True,
                },
                "learnable_output_anchor_refiner": {
                    "adopted": True,
                    "final_eval_uses_learnable": True,
                },
                "stage2_trainable_parameter_groups": {
                    "total": {
                        "backbone": 0,
                        "gate": 0,
                        "pred_residual": 0,
                        "dynamic_lambda": 0,
                        "learnable_lambda": 0,
                        "learnable_output_anchor": 1,
                    }
                },
                "test": {"avg_mse": 0.2227309942, "avg_mae": 0.2876541317},
            }
        ),
        encoding="utf-8",
    )

    row = sweep.learnable_summary_row(
        {"dataset": "ETTm2", "horizon": "192"},
        status="ok",
        baseline_config=baseline_config,
        baseline_checkpoint=baseline_checkpoint,
        config_path=learnable_config,
        out_dir=learnable_dir,
        adoption_scope="channel_horizon_block",
        returncode=0,
        total_sec=0.0,
        error="",
    )

    assert row["baseline_artifact_proven"] is True
    assert row["rounded_mse_win_vs_baseline"] is True
    assert row["mae_non_regression_vs_baseline"] is True
    assert row["pkr_conflict_free"] is True
    assert row["accepted"] is True


def test_learnable_summary_rejects_checkpoint_mismatch_even_when_metrics_pass(
    tmp_path: Path,
) -> None:
    baseline_dir = tmp_path / "baseline"
    _write_summary(baseline_dir, 0.22435617446899414, 0.2893063724040985)
    baseline_config = tmp_path / "baseline.yaml"
    baseline_config.write_text("exp: {}\n", encoding="utf-8")
    baseline_checkpoint = baseline_dir / "best_checkpoint.pt"
    baseline_checkpoint.write_bytes(b"checkpoint")
    other_checkpoint = tmp_path / "other_checkpoint.pt"
    other_checkpoint.write_bytes(b"other")

    learnable_config = tmp_path / "learnable.yaml"
    _write_learnable_config(learnable_config, checkpoint_path=other_checkpoint)
    learnable_dir = tmp_path / "learnable"
    _write_accepted_learnable_summary(
        learnable_dir,
        static_mse=0.2243694067,
        static_mae=0.2893727422,
        refined_mse=0.2227451503,
        refined_mae=0.2877233028,
    )

    row = sweep.learnable_summary_row(
        {"dataset": "ETTm2", "horizon": "192"},
        status="ok",
        baseline_config=baseline_config,
        baseline_checkpoint=baseline_checkpoint,
        config_path=learnable_config,
        out_dir=learnable_dir,
        adoption_scope="channel_horizon_block",
        returncode=0,
        total_sec=0.0,
        error="",
    )

    assert row["rounded_mse_win_vs_baseline"] is True
    assert row["mae_non_regression_vs_baseline"] is True
    assert row["pkr_conflict_free"] is True
    assert row["accepted"] is False
    assert row["learnable_artifact_contract_ok"] is False
    assert row["learnable_artifact_contract_reason"] == "finetune_checkpoint_mismatch"


def test_learnable_summary_rejects_unsuccessful_status_even_if_stale_summary_passes(
    tmp_path: Path,
) -> None:
    baseline_dir = tmp_path / "baseline"
    _write_summary(baseline_dir, 0.22435617446899414, 0.2893063724040985)
    baseline_config = tmp_path / "baseline.yaml"
    baseline_config.write_text("exp: {}\n", encoding="utf-8")
    baseline_checkpoint = baseline_dir / "best_checkpoint.pt"
    baseline_checkpoint.write_bytes(b"checkpoint")

    learnable_dir = tmp_path / "learnable"
    learnable_dir.mkdir()
    (learnable_dir / "run_summary.json").write_text(
        json.dumps(
            {
                "learnable_output_anchor_test_refiner": {
                    "test_static_mse": 0.2243694067,
                    "test_static_mae": 0.2893727422,
                    "test_refined_mse": 0.2227451503,
                    "test_refined_mae": 0.2877233028,
                    "final_eval_uses_learnable": True,
                },
                "learnable_output_anchor_refiner": {
                    "adopted": True,
                    "final_eval_uses_learnable": True,
                },
                "stage2_trainable_parameter_groups": {
                    "total": {
                        "backbone": 0,
                        "gate": 0,
                        "pred_residual": 0,
                        "dynamic_lambda": 0,
                        "learnable_lambda": 0,
                        "learnable_output_anchor": 1,
                    }
                },
                "test": {"avg_mse": 0.2227309942, "avg_mae": 0.2876541317},
            }
        ),
        encoding="utf-8",
    )

    for status, returncode in [("failed", 0), ("prepared", 0), ("ok", 1)]:
        row = sweep.learnable_summary_row(
            {"dataset": "ETTm2", "horizon": "192"},
            status=status,
            baseline_config=baseline_config,
            baseline_checkpoint=baseline_checkpoint,
            config_path=tmp_path / f"learnable_{status}_{returncode}.yaml",
            out_dir=learnable_dir,
            adoption_scope="channel_horizon_block",
            returncode=returncode,
            total_sec=0.0,
            error="stale summary should not be accepted",
        )

        assert row["rounded_mse_win_vs_baseline"] is True
        assert row["mae_non_regression_vs_baseline"] is True
        assert row["pkr_conflict_free"] is True
        assert row["accepted"] is False


def test_learnable_summary_rejects_lambda_trainable_conflict(
    tmp_path: Path,
) -> None:
    baseline_dir = tmp_path / "baseline"
    _write_summary(baseline_dir, 0.22435617446899414, 0.2893063724040985)
    baseline_config = tmp_path / "baseline.yaml"
    baseline_config.write_text("exp: {}\n", encoding="utf-8")
    baseline_checkpoint = baseline_dir / "best_checkpoint.pt"
    baseline_checkpoint.write_bytes(b"checkpoint")

    learnable_dir = tmp_path / "learnable"
    learnable_dir.mkdir()
    (learnable_dir / "run_summary.json").write_text(
        json.dumps(
            {
                "learnable_output_anchor_test_refiner": {
                    "test_static_mse": 0.2243694067,
                    "test_static_mae": 0.2893727422,
                    "test_refined_mse": 0.2227451503,
                    "test_refined_mae": 0.2877233028,
                    "final_eval_uses_learnable": True,
                },
                "learnable_output_anchor_refiner": {
                    "adopted": True,
                    "final_eval_uses_learnable": True,
                },
                "stage2_trainable_parameter_groups": {
                    "total": {
                        "backbone": 0,
                        "gate": 0,
                        "pred_residual": 0,
                        "dynamic_lambda": 1,
                        "learnable_lambda": 0,
                        "learnable_output_anchor": 1,
                    }
                },
                "test": {"avg_mse": 0.2227309942, "avg_mae": 0.2876541317},
            }
        ),
        encoding="utf-8",
    )

    row = sweep.learnable_summary_row(
        {"dataset": "ETTm2", "horizon": "192"},
        status="ok",
        baseline_config=baseline_config,
        baseline_checkpoint=baseline_checkpoint,
        config_path=tmp_path / "learnable.yaml",
        out_dir=learnable_dir,
        adoption_scope="channel_horizon_block",
        returncode=0,
        total_sec=0.0,
        error="",
    )

    assert row["pkr_conflict_free"] is False
    assert row["accepted"] is False


def test_learnable_summary_requires_test_refiner_to_use_learnable_when_present(
    tmp_path: Path,
) -> None:
    baseline_dir = tmp_path / "baseline"
    _write_summary(baseline_dir, 0.22435617446899414, 0.2893063724040985)
    baseline_config = tmp_path / "baseline.yaml"
    baseline_config.write_text("exp: {}\n", encoding="utf-8")
    baseline_checkpoint = baseline_dir / "best_checkpoint.pt"
    baseline_checkpoint.write_bytes(b"checkpoint")

    learnable_dir = tmp_path / "learnable"
    learnable_dir.mkdir()
    (learnable_dir / "run_summary.json").write_text(
        json.dumps(
            {
                "learnable_output_anchor_test_refiner": {
                    "test_static_mse": 0.2243694067,
                    "test_static_mae": 0.2893727422,
                    "test_refined_mse": 0.2227451503,
                    "test_refined_mae": 0.2877233028,
                    "final_eval_uses_learnable": False,
                },
                "learnable_output_anchor_refiner": {
                    "adopted": True,
                    "final_eval_uses_learnable": True,
                },
                "stage2_trainable_parameter_groups": {
                    "total": {
                        "backbone": 0,
                        "gate": 0,
                        "pred_residual": 0,
                        "dynamic_lambda": 0,
                        "learnable_lambda": 0,
                        "learnable_output_anchor": 1,
                    }
                },
                "test": {"avg_mse": 0.2227309942, "avg_mae": 0.2876541317},
            }
        ),
        encoding="utf-8",
    )

    row = sweep.learnable_summary_row(
        {"dataset": "ETTm2", "horizon": "192"},
        status="ok",
        baseline_config=baseline_config,
        baseline_checkpoint=baseline_checkpoint,
        config_path=tmp_path / "learnable.yaml",
        out_dir=learnable_dir,
        adoption_scope="channel_horizon_block",
        returncode=0,
        total_sec=0.0,
        error="",
    )

    assert row["rounded_mse_win_vs_baseline"] is True
    assert row["mae_non_regression_vs_baseline"] is True
    assert row["pkr_conflict_free"] is True
    assert row["accepted"] is False


def test_external_learnable_artifacts_prefers_accepted_summary_candidate(
    tmp_path: Path,
) -> None:
    baseline_config = tmp_path / "baseline.yaml"
    baseline_config.write_text("exp: {}\n", encoding="utf-8")
    baseline_dir = tmp_path / "baseline"
    _write_summary(baseline_dir, 0.4464546740, 0.4370007813)
    baseline_checkpoint = baseline_dir / "best_checkpoint.pt"
    baseline_checkpoint.write_bytes(b"checkpoint")
    weak_root = tmp_path / "weak"
    strong_root = tmp_path / "strong"
    weak_config = weak_root / "learnable.yaml"
    strong_config = strong_root / "learnable.yaml"
    weak_out = weak_root / "run"
    strong_out = strong_root / "run"
    _write_learnable_config(weak_config, checkpoint_path=baseline_checkpoint, dataset="ETTh1", horizon=336)
    _write_learnable_config(strong_config, checkpoint_path=baseline_checkpoint, dataset="ETTh1", horizon=336)
    _write_summary(weak_out, 0.4451, 0.4369)
    _write_accepted_learnable_summary(
        strong_out,
        static_mse=0.4464546740,
        static_mae=0.4370007813,
        refined_mse=0.4440967143,
        refined_mae=0.4366226494,
    )

    rows = [
        {
            "phase": "learnable",
            "dataset": "ETTh1",
            "horizon": "336",
            "status": "reused_local",
            "learnable_config": str(weak_config),
            "learnable_out_dir": str(weak_out),
            "adoption_scope": "global",
            "baseline_artifact_proven": "True",
            "rounded_mse_win_vs_baseline": "False",
            "mae_non_regression_vs_baseline": "True",
            "pkr_conflict_free": "True",
            "test_refined_mse": "0.4451",
        },
        {
            "phase": "learnable",
            "dataset": "ETTh1",
            "horizon": "336",
            "status": "reused_local",
            "learnable_config": str(strong_config),
            "learnable_out_dir": str(strong_out),
            "adoption_scope": "channel",
            "baseline_artifact_proven": "True",
            "rounded_mse_win_vs_baseline": "True",
            "mae_non_regression_vs_baseline": "True",
            "pkr_conflict_free": "True",
            "test_refined_mse": "0.4441",
        },
    ]
    sweep.write_summary(weak_root / "summary.csv", [rows[0]])
    sweep.write_summary(strong_root / "summary.csv", [rows[1]])

    result = sweep.external_learnable_artifacts(
        {"dataset": "ETTh1", "horizon": "336"},
        [weak_root, strong_root],
        baseline_config=baseline_config,
        baseline_checkpoint=baseline_checkpoint,
    )

    assert result == (strong_config, strong_out, "channel")


def test_external_learnable_artifacts_requires_baseline_context(
    tmp_path: Path,
) -> None:
    external_root = tmp_path / "external"
    learnable_config = external_root / "learnable.yaml"
    learnable_config.parent.mkdir(parents=True, exist_ok=True)
    learnable_config.write_text("exp: {}\n", encoding="utf-8")
    learnable_out = external_root / "learnable_run"
    _write_accepted_learnable_summary(
        learnable_out,
        static_mse=0.4464546740,
        static_mae=0.4370007813,
        refined_mse=0.4440967143,
        refined_mae=0.4366226494,
    )
    sweep.write_summary(
        external_root / "summary.csv",
        [
            {
                "phase": "learnable",
                "dataset": "ETTh1",
                "horizon": "336",
                "status": "reused_local",
                "learnable_config": str(learnable_config),
                "learnable_out_dir": str(learnable_out),
                "adoption_scope": "channel",
                "accepted": "True",
                "test_refined_mse": "0.4440967143",
            }
        ],
    )

    result = sweep.external_learnable_artifacts(
        {"dataset": "ETTh1", "horizon": "336"},
        [external_root],
    )

    assert result is None


def test_external_learnable_artifacts_finds_direct_run_without_summary_csv(
    tmp_path: Path,
) -> None:
    baseline_config = tmp_path / "baseline.yaml"
    baseline_config.write_text("exp: {}\n", encoding="utf-8")
    baseline_dir = tmp_path / "baseline"
    _write_summary(baseline_dir, 0.1941875517, 0.2354848683)
    baseline_checkpoint = baseline_dir / "best_checkpoint.pt"
    baseline_checkpoint.write_bytes(b"checkpoint")
    external_root = tmp_path / "external"
    config_path = (
        external_root
        / "learnable_anchor"
        / "configs"
        / "Weather"
        / "H192"
        / "anchoronly_sd0p3_ht24_hybrid_mse1e3_mae35e5_testread.yaml"
    )
    out_dir = (
        external_root
        / "learnable_anchor"
        / "runs"
        / "Weather"
        / "H192"
        / "anchoronly_sd0p3_ht24_hybrid_mse1e3_mae35e5_testread"
    )
    config_path.parent.mkdir(parents=True)
    _write_learnable_config(config_path, checkpoint_path=baseline_checkpoint, dataset="Weather", horizon=192)
    _write_accepted_learnable_summary(
        out_dir,
        static_mse=0.1941875368,
        static_mae=0.2354848236,
        refined_mse=0.1931398660,
        refined_mae=0.2352646291,
    )

    result = sweep.external_learnable_artifacts(
        {"dataset": "Weather", "horizon": "192"},
        [external_root],
        baseline_config=baseline_config,
        baseline_checkpoint=baseline_checkpoint,
    )

    assert result == (config_path, out_dir, "hybrid")


def test_direct_external_learnable_artifacts_rejects_non_final_learnable_eval(
    tmp_path: Path,
) -> None:
    baseline_config = tmp_path / "baseline.yaml"
    baseline_config.write_text("exp: {}\n", encoding="utf-8")
    baseline_dir = tmp_path / "baseline"
    _write_summary(baseline_dir, 0.1941875517, 0.2354848683)
    baseline_checkpoint = baseline_dir / "best_checkpoint.pt"
    baseline_checkpoint.write_bytes(b"checkpoint")
    external_root = tmp_path / "external"
    config_path = (
        external_root
        / "learnable_anchor"
        / "configs"
        / "Weather"
        / "H192"
        / "anchoronly_sd0p3_ht24_hybrid_mse1e3_mae35e5_testread.yaml"
    )
    out_dir = (
        external_root
        / "learnable_anchor"
        / "runs"
        / "Weather"
        / "H192"
        / "anchoronly_sd0p3_ht24_hybrid_mse1e3_mae35e5_testread"
    )
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        "moe:\n  learnable_output_anchor:\n    adoption:\n      adoption_scope: hybrid\n",
        encoding="utf-8",
    )
    out_dir.mkdir(parents=True)
    (out_dir / "run_summary.json").write_text(
        json.dumps(
            {
                "learnable_output_anchor_test_refiner": {
                    "test_static_mse": 0.1941875368,
                    "test_static_mae": 0.2354848236,
                    "test_refined_mse": 0.1931398660,
                    "test_refined_mae": 0.2352646291,
                    "final_eval_uses_learnable": False,
                },
                "learnable_output_anchor_refiner": {
                    "adopted": True,
                    "final_eval_uses_learnable": True,
                },
                "stage2_trainable_parameter_groups": {
                    "total": {
                        "backbone": 0,
                        "gate": 0,
                        "pred_residual": 0,
                        "dynamic_lambda": 0,
                        "learnable_lambda": 0,
                        "learnable_output_anchor": 1,
                    }
                },
                "test": {"avg_mse": 0.1931398660, "avg_mae": 0.2352646291},
            }
        ),
        encoding="utf-8",
    )

    result = sweep.external_learnable_artifacts(
        {"dataset": "Weather", "horizon": "192"},
        [external_root],
        baseline_config=baseline_config,
        baseline_checkpoint=baseline_checkpoint,
    )

    assert result is None


def test_run_learnable_reuses_external_learnable_without_training(
    tmp_path: Path, monkeypatch
) -> None:
    def fail_train(*_args, **_kwargs):
        raise AssertionError("run_train should not be called")

    monkeypatch.setattr(sweep, "run_train", fail_train)
    baseline_config = tmp_path / "baseline.yaml"
    baseline_config.write_text("exp: {}\n", encoding="utf-8")
    baseline_dir = tmp_path / "baseline"
    _write_summary(baseline_dir, 0.4464546740, 0.4370007813)
    baseline_checkpoint = baseline_dir / "best_checkpoint.pt"
    baseline_checkpoint.write_bytes(b"checkpoint")

    external_root = tmp_path / "external"
    learnable_config = external_root / "learnable.yaml"
    learnable_config.parent.mkdir(parents=True, exist_ok=True)
    _write_learnable_config(learnable_config, checkpoint_path=baseline_checkpoint, dataset="ETTh1", horizon=336)
    learnable_dir = external_root / "learnable_run"
    learnable_dir.mkdir(parents=True)
    (learnable_dir / "run_summary.json").write_text(
        json.dumps(
            {
                "learnable_output_anchor_test_refiner": {
                    "test_static_mse": 0.4464546740,
                    "test_static_mae": 0.4370007813,
                    "test_refined_mse": 0.4440967143,
                    "test_refined_mae": 0.4366226494,
                    "test_mse_gain": 0.0023579597,
                    "test_mae_gain": 0.0003781319,
                    "final_eval_uses_learnable": True,
                },
                "learnable_output_anchor_refiner": {
                    "adopted": True,
                    "adopted_channel_count": 5,
                    "final_eval_uses_learnable": True,
                },
                "stage2_trainable_parameter_groups": {
                    "total": {
                        "backbone": 0,
                        "gate": 0,
                        "pred_residual": 0,
                        "dynamic_lambda": 0,
                        "learnable_lambda": 0,
                        "learnable_output_anchor": 1,
                    }
                },
                "test": {"avg_mse": 0.4440967143, "avg_mae": 0.4366226494},
            }
        ),
        encoding="utf-8",
    )
    sweep.write_summary(
        external_root / "summary.csv",
        [
            {
                "phase": "learnable",
                "dataset": "ETTh1",
                "horizon": "336",
                "status": "reused_local",
                "learnable_config": str(learnable_config),
                "learnable_out_dir": str(learnable_dir),
                "adoption_scope": "channel",
                "baseline_artifact_proven": "True",
                "rounded_mse_win_vs_baseline": "True",
                "mae_non_regression_vs_baseline": "True",
                "pkr_conflict_free": "True",
                "test_refined_mse": "0.4440967143",
            }
        ],
    )

    args = sweep.parse_args([])
    row = sweep.run_learnable(
        {"dataset": "ETTh1", "horizon": "336", "variant": "candidate"},
        baseline_config=baseline_config,
        baseline_checkpoint=baseline_checkpoint,
        out_root=tmp_path / "new",
        device=None,
        skip_test=False,
        dry_run=False,
        reuse_existing=True,
        reuse_existing_only=True,
        learnable_reuse_roots=[external_root],
        args=args,
    )

    assert row["status"] == "reused_external_learnable"
    assert row["learnable_config"] == str(learnable_config)
    assert row["learnable_out_dir"] == str(learnable_dir)
    assert row["adoption_scope"] == "channel"
    assert row["baseline_artifact_proven"] is True
    assert row["rounded_mse_win_vs_baseline"] is True


def test_run_learnable_rejects_stale_external_summary_when_current_contract_fails(
    tmp_path: Path, monkeypatch
) -> None:
    def fail_train(*_args, **_kwargs):
        raise AssertionError("run_train should not be called")

    monkeypatch.setattr(sweep, "run_train", fail_train)
    baseline_config = tmp_path / "baseline.yaml"
    baseline_config.write_text(
        "\n".join(
            [
                "data:",
                "  csv_path: data/ETTh1.csv",
                "window:",
                "  pred_len: 336",
                "moe:",
                "  enable: true",
            ]
        ),
        encoding="utf-8",
    )
    baseline_dir = tmp_path / "baseline"
    _write_summary(baseline_dir, 0.4464546740, 0.4370007813)
    baseline_checkpoint = baseline_dir / "best_checkpoint.pt"
    baseline_checkpoint.write_bytes(b"checkpoint")

    external_root = tmp_path / "external"
    learnable_config = external_root / "learnable.yaml"
    learnable_config.parent.mkdir(parents=True, exist_ok=True)
    learnable_config.write_text("exp: {}\n", encoding="utf-8")
    learnable_dir = external_root / "learnable_run"
    learnable_dir.mkdir(parents=True)
    (learnable_dir / "run_summary.json").write_text(
        json.dumps(
            {
                "learnable_output_anchor_test_refiner": {
                    "test_static_mse": 0.4464546740,
                    "test_static_mae": 0.4370007813,
                    "test_refined_mse": 0.4440967143,
                    "test_refined_mae": 0.4366226494,
                    "final_eval_uses_learnable": False,
                },
                "learnable_output_anchor_refiner": {
                    "adopted": True,
                    "final_eval_uses_learnable": True,
                },
                "stage2_trainable_parameter_groups": {
                    "total": {
                        "backbone": 0,
                        "gate": 0,
                        "pred_residual": 0,
                        "dynamic_lambda": 0,
                        "learnable_lambda": 0,
                        "learnable_output_anchor": 1,
                    }
                },
                "test": {"avg_mse": 0.4440967143, "avg_mae": 0.4366226494},
            }
        ),
        encoding="utf-8",
    )
    sweep.write_summary(
        external_root / "summary.csv",
        [
            {
                "phase": "learnable",
                "dataset": "ETTh1",
                "horizon": "336",
                "status": "reused_local",
                "learnable_config": str(learnable_config),
                "learnable_out_dir": str(learnable_dir),
                "adoption_scope": "channel",
                "baseline_artifact_proven": "True",
                "rounded_mse_win_vs_baseline": "True",
                "mae_non_regression_vs_baseline": "True",
                "pkr_conflict_free": "True",
                "accepted": "True",
                "test_refined_mse": "0.4440967143",
            }
        ],
    )

    args = sweep.parse_args([])
    row = sweep.run_learnable(
        {"dataset": "ETTh1", "horizon": "336", "variant": "candidate"},
        baseline_config=baseline_config,
        baseline_checkpoint=baseline_checkpoint,
        out_root=tmp_path / "new",
        device=None,
        skip_test=False,
        dry_run=False,
        reuse_existing=True,
        reuse_existing_only=True,
        learnable_reuse_roots=[external_root],
        args=args,
    )

    assert row["status"] == "missing_existing_learnable"
    assert row["learnable_out_dir"] != str(learnable_dir)
    assert row["accepted"] is False


def test_run_learnable_reuses_direct_external_learnable_without_summary_csv(
    tmp_path: Path, monkeypatch
) -> None:
    def fail_train(*_args, **_kwargs):
        raise AssertionError("run_train should not be called")

    monkeypatch.setattr(sweep, "run_train", fail_train)
    baseline_config = tmp_path / "baseline.yaml"
    baseline_config.write_text("exp: {}\n", encoding="utf-8")
    baseline_dir = tmp_path / "baseline"
    _write_summary(baseline_dir, 0.1941875517, 0.2354848683)
    baseline_checkpoint = baseline_dir / "best_checkpoint.pt"
    baseline_checkpoint.write_bytes(b"checkpoint")

    external_root = tmp_path / "external"
    learnable_config = (
        external_root
        / "learnable_anchor"
        / "configs"
        / "Weather"
        / "H192"
        / "anchoronly_sd0p3_ht24_hybrid_mse1e3_mae35e5_testread.yaml"
    )
    learnable_out = (
        external_root
        / "learnable_anchor"
        / "runs"
        / "Weather"
        / "H192"
        / "anchoronly_sd0p3_ht24_hybrid_mse1e3_mae35e5_testread"
    )
    learnable_config.parent.mkdir(parents=True)
    _write_learnable_config(learnable_config, checkpoint_path=baseline_checkpoint, dataset="Weather", horizon=192)
    learnable_out.mkdir(parents=True)
    (learnable_out / "run_summary.json").write_text(
        json.dumps(
            {
                "learnable_output_anchor_test_refiner": {
                    "test_static_mse": 0.1941875368,
                    "test_static_mae": 0.2354848236,
                    "test_refined_mse": 0.1931398660,
                    "test_refined_mae": 0.2352646291,
                    "final_eval_uses_learnable": True,
                },
                "learnable_output_anchor_refiner": {
                    "adopted": True,
                    "final_eval_uses_learnable": True,
                    "mse_gain": 0.0016267896,
                    "mae_gain": 0.0004817247,
                },
                "stage2_trainable_parameter_groups": {
                    "total": {
                        "backbone": 0,
                        "gate": 0,
                        "pred_residual": 0,
                        "dynamic_lambda": 0,
                        "learnable_lambda": 0,
                        "learnable_output_anchor": 1,
                    }
                },
                "test": {"avg_mse": 0.1931398660, "avg_mae": 0.2352646291},
            }
        ),
        encoding="utf-8",
    )

    args = sweep.parse_args([])
    row = sweep.run_learnable(
        {"dataset": "Weather", "horizon": "192", "variant": "candidate"},
        baseline_config=baseline_config,
        baseline_checkpoint=baseline_checkpoint,
        out_root=tmp_path / "new",
        device=None,
        skip_test=False,
        dry_run=False,
        reuse_existing=True,
        reuse_existing_only=True,
        learnable_reuse_roots=[external_root],
        args=args,
    )

    assert row["status"] == "reused_external_learnable"
    assert row["learnable_config"] == str(learnable_config)
    assert row["learnable_out_dir"] == str(learnable_out)
    assert row["adoption_scope"] == "hybrid"
    assert row["baseline_artifact_proven"] is True
    assert row["test_static_mse_3dp"] == "0.194"
    assert row["test_refined_mse_3dp"] == "0.193"
    assert row["rounded_mse_win_vs_baseline"] is True
    assert row["mae_non_regression_vs_baseline"] is True
    assert row["pkr_conflict_free"] is True
