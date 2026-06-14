from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import yaml

from src.web_visualizer_core import (
    build_training_command,
    clean_training_log_text,
    discover_configs,
    discover_runs,
    format_training_log_tail,
    load_prediction_sample,
    load_run_payload,
    materialize_training_config,
    parse_training_progress,
)


def _write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def test_discover_configs_reads_dataset_and_horizon(tmp_path: Path) -> None:
    _write_yaml(
        tmp_path / "configs" / "ETTm1_H96.yaml",
        {
            "exp": {"name": "ETTm1_pred_96", "out_dir": "outputs/ETTm1_H96", "device": "cuda:0"},
            "data": {"csv_path": "data/ETTm1.csv"},
            "window": {"input_len": 336, "pred_len": 96},
            "train": {"epochs": 100},
        },
    )

    configs = discover_configs(tmp_path)

    assert configs == [
        {
            "id": "configs/ETTm1_H96.yaml",
            "name": "ETTm1_H96",
            "dataset": "ETTm1",
            "input_len": 336,
            "pred_len": 96,
            "epochs": 100,
            "device": "cuda:0",
            "out_dir": "outputs/ETTm1_H96",
        }
    ]


def test_discover_runs_marks_prediction_replay_availability(tmp_path: Path) -> None:
    run_dir = tmp_path / "outputs" / "ETTm1_H96"
    run_dir.mkdir(parents=True)
    (run_dir / "run_summary.json").write_text(
        json.dumps(
            {
                "config_path": "configs/ETTm1_H96.yaml",
                "penalty_names": ["level", "delta"],
                "selected": {"avg_mse": 0.31, "avg_mae": 0.41},
                "test": {"avg_mse": 0.32, "avg_mae": 0.42},
                "windowing": {"num_train_windows": 10},
            }
        ),
        encoding="utf-8",
    )
    np.savez_compressed(run_dir / "prediction_intermediates.npz", idx=np.array([7]))

    runs = discover_runs(tmp_path)

    assert len(runs) == 1
    assert runs[0]["id"] == "outputs/ETTm1_H96"
    assert runs[0]["label"] == "ETTm1_H96"
    assert runs[0]["config_path"] == "configs/ETTm1_H96.yaml"
    assert runs[0]["avg_mse"] == 0.31
    assert runs[0]["avg_mae"] == 0.41
    assert runs[0]["has_prediction_intermediates"] is True


def test_discover_runs_prioritizes_prediction_replay_runs(tmp_path: Path) -> None:
    old_replay = tmp_path / "outputs" / "old_replay"
    new_plain = tmp_path / "outputs" / "new_plain"
    old_replay.mkdir(parents=True)
    new_plain.mkdir(parents=True)
    (old_replay / "run_summary.json").write_text(json.dumps({"selected": {"avg_mse": 0.4}}), encoding="utf-8")
    (new_plain / "run_summary.json").write_text(json.dumps({"selected": {"avg_mse": 0.1}}), encoding="utf-8")
    np.savez_compressed(old_replay / "prediction_intermediates.npz", idx=np.array([1]))
    os.utime(old_replay / "run_summary.json", (10, 10))
    os.utime(new_plain / "run_summary.json", (20, 20))

    runs = discover_runs(tmp_path, max_runs=1)

    assert [run["id"] for run in runs] == ["outputs/old_replay"]


def test_load_run_payload_reads_summary_csv_and_prediction_meta(tmp_path: Path) -> None:
    run_dir = tmp_path / "outputs" / "ETTm1_H96"
    run_dir.mkdir(parents=True)
    (run_dir / "run_summary.json").write_text(
        json.dumps({"penalty_names": ["level"], "selected": {"avg_mse": 0.2}}),
        encoding="utf-8",
    )
    (run_dir / "cluster_penalty_probs.csv").write_text(
        "cluster,penalty,avg_prob,active_rate\n0,level,0.8,1.0\n",
        encoding="utf-8",
    )
    (run_dir / "prediction_intermediates_meta.json").write_text(
        json.dumps({"sample_count": 2, "channel_names": ["OT"], "penalty_names": ["level"]}),
        encoding="utf-8",
    )
    np.savez_compressed(run_dir / "prediction_intermediates.npz", idx=np.array([1, 2]))

    payload = load_run_payload(tmp_path, "outputs/ETTm1_H96")

    assert payload["id"] == "outputs/ETTm1_H96"
    assert payload["summary"]["selected"]["avg_mse"] == 0.2
    assert payload["cluster_penalty_rows"] == [
        {"cluster": 0, "penalty": "level", "avg_prob": 0.8, "active_rate": 1.0}
    ]
    assert payload["prediction_meta"]["sample_count"] == 2
    assert payload["has_prediction_intermediates"] is True


def test_load_prediction_sample_returns_curves_and_moe_routes(tmp_path: Path) -> None:
    run_dir = tmp_path / "outputs" / "ETTm1_H96"
    run_dir.mkdir(parents=True)
    (run_dir / "prediction_intermediates_meta.json").write_text(
        json.dumps({"channel_names": ["A", "B"], "penalty_names": ["level", "delta"]}),
        encoding="utf-8",
    )
    np.savez_compressed(
        run_dir / "prediction_intermediates.npz",
        idx=np.array([42]),
        x=np.array([[[1.0, 2.0], [3.0, 4.0]]]),
        y_true=np.array([[[2.0, 3.0], [4.0, 5.0]]]),
        y_base=np.array([[[1.5, 2.5], [3.5, 4.5]]]),
        y_residual_raw=np.array([[[1.7, 2.7], [3.7, 4.7]]]),
        y_final=np.array([[[1.8, 2.8], [3.8, 4.8]]]),
        gate_probs=np.array([[[0.2, 0.8], [0.9, 0.1]]]),
        gate_mask=np.array([[[0.0, 1.0], [1.0, 0.0]]]),
        skip_prob=np.array([[0.1, 0.2]]),
        residual_gate_scale=np.array([[[0.5], [0.7]]]),
        cluster_id=np.array([0, 1]),
    )

    sample = load_prediction_sample(tmp_path, "outputs/ETTm1_H96", sample_index=0)

    assert sample["idx"] == 42
    assert sample["channel_names"] == ["A", "B"]
    assert sample["series"]["y_final"][1] == [3.8, 4.8]
    assert sample["channel_cluster_ids"] == [0, 1]
    assert sample["clusters"] == [
        {
            "cluster": 0,
            "probabilities": [{"penalty": "level", "prob": 0.2}, {"penalty": "delta", "prob": 0.8}],
            "penalty_participation": [
                {"penalty": "level", "selected": False, "gate_prob": 0.2, "effective_strength": 0.0},
                {"penalty": "delta", "selected": True, "gate_prob": 0.8, "effective_strength": 0.72},
            ],
            "selected_penalties": ["delta"],
            "top_penalty": "delta",
            "skip_prob": 0.1,
        },
        {
            "cluster": 1,
            "probabilities": [{"penalty": "level", "prob": 0.9}, {"penalty": "delta", "prob": 0.1}],
            "penalty_participation": [
                {"penalty": "level", "selected": True, "gate_prob": 0.9, "effective_strength": 0.72},
                {"penalty": "delta", "selected": False, "gate_prob": 0.1, "effective_strength": 0.0},
            ],
            "selected_penalties": ["level"],
            "top_penalty": "level",
            "skip_prob": 0.2,
        },
    ]
    assert sample["residual_gate_scale"] == [0.5, 0.7]


def test_materialize_training_config_overrides_horizon_and_diagnostics(tmp_path: Path) -> None:
    source = tmp_path / "configs" / "ETTm1.yaml"
    _write_yaml(
        source,
        {
            "exp": {"name": "ETTm1", "out_dir": "outputs/old", "device": "cuda:0"},
            "data": {"csv_path": "data/ETTm1.csv"},
            "window": {"input_len": 336, "pred_len": 96},
            "eval": {"skip_test": True},
            "diagnostics": {"save_prediction_intermediates": False},
        },
    )

    result = materialize_training_config(
        tmp_path,
        "configs/ETTm1.yaml",
        run_id="demo-run",
        pred_len=192,
        device="cpu",
        sample_count=5,
    )

    generated = yaml.safe_load(Path(result["config_path"]).read_text(encoding="utf-8"))
    assert result["run_dir"].endswith("outputs/web_visualizer/runs/demo-run")
    assert generated["exp"]["out_dir"].endswith("outputs/web_visualizer/runs/demo-run")
    assert generated["exp"]["device"] == "cpu"
    assert generated["window"]["pred_len"] == 192
    assert generated["eval"]["skip_test"] is False
    assert generated["diagnostics"]["save_prediction_intermediates"] is True
    assert generated["diagnostics"]["prediction_sample_count"] == 5
    assert generated["diagnostics"]["prediction_sample_strategy"] == "stratified_random"
    assert isinstance(generated["diagnostics"]["prediction_sample_seed"], int)


def test_materialize_training_config_redirects_output_artifact_paths(tmp_path: Path) -> None:
    source = tmp_path / "configs" / "electricity_H96.yaml"
    old_run = (
        "outputs/bayes_search_electricity_weather_8configs/final_runs/electricity/H96/"
        "trial_0021_amp_range_dir_mlp_h256_do0p0469_l0p0269_level_heavy_lr0p0013_legacy"
    )
    _write_yaml(
        source,
        {
            "exp": {"name": "electricity_H96", "out_dir": old_run, "device": "cuda:0"},
            "data": {"csv_path": "data/electricity.csv"},
            "window": {"input_len": 336, "pred_len": 96},
            "corr": {"compute": True, "save_path": f"{old_run}/corr.npy"},
            "portrait": {"enable": False, "out_dir": f"{old_run}/cluster_portraits"},
            "knn_hybrid": {"enable": False, "path": f"{old_run}/knn_shape_bank.pt"},
            "memory": {
                "enable": False,
                "save_checkpoint": False,
                "path": f"{old_run}/cluster_memory.pt",
                "checkpoint_path": f"{old_run}/best_checkpoint.pt",
            },
        },
    )

    result = materialize_training_config(
        tmp_path,
        "configs/electricity_H96.yaml",
        run_id="web-run",
        pred_len=96,
        device="cuda:0",
        sample_count=32,
    )

    run_dir = Path(result["run_dir"]).as_posix()
    generated = yaml.safe_load(Path(result["config_path"]).read_text(encoding="utf-8"))
    assert generated["corr"]["save_path"] == f"{run_dir}/corr.npy"
    assert generated["portrait"]["out_dir"] == f"{run_dir}/cluster_portraits"
    assert generated["knn_hybrid"]["path"] == f"{run_dir}/knn_shape_bank.pt"
    assert generated["memory"]["path"] == f"{run_dir}/cluster_memory.pt"
    assert generated["memory"]["checkpoint_path"] == f"{run_dir}/best_checkpoint.pt"
    assert old_run not in Path(result["config_path"]).read_text(encoding="utf-8")


def test_materialize_training_config_preserves_cuda_device_by_default(tmp_path: Path) -> None:
    source = tmp_path / "configs" / "ETTm1.yaml"
    _write_yaml(
        source,
        {
            "exp": {"name": "ETTm1", "out_dir": "outputs/old", "device": "cuda:0"},
            "data": {"csv_path": "data/ETTm1.csv"},
            "window": {"input_len": 336, "pred_len": 96},
        },
    )

    result = materialize_training_config(
        tmp_path,
        "configs/ETTm1.yaml",
        run_id="cuda-run",
        pred_len=96,
        sample_count=4,
    )

    generated = yaml.safe_load(Path(result["config_path"]).read_text(encoding="utf-8"))
    assert generated["exp"]["device"] == "cuda:0"


def test_build_training_command_uses_my_fram_conda_environment() -> None:
    cmd = build_training_command(
        "F:/repo/outputs/web_visualizer/runs/demo/config.yaml",
        conda_bat="F:/Anaconda3/condabin/conda.bat",
        env_name="my_fram",
    )

    assert cmd == [
        "F:/Anaconda3/condabin/conda.bat",
        "run",
        "--no-capture-output",
        "-n",
        "my_fram",
        "python",
        "-m",
        "src.train",
        "--config",
        "F:/repo/outputs/web_visualizer/runs/demo/config.yaml",
    ]


def test_parse_training_progress_reads_epoch_batch_progress() -> None:
    progress = parse_training_progress(
        "\rTrain ETTm1 H=96 [#####-----] 14/100  14.0% elapsed=00:12 eta=01:14 "
        "| epoch=2/10 batch=4/10 loss=0.123456"
    )

    assert progress["epoch_current"] == 2
    assert progress["epoch_total"] == 10
    assert progress["batch_current"] == 4
    assert progress["batch_total"] == 10
    assert progress["epoch_percent"] == 40.0
    assert progress["global_percent"] == 14.0
    assert progress["loss"] == 0.123456
    assert progress["phase"] == "training"


def test_parse_training_progress_reads_epoch_summary() -> None:
    progress = parse_training_progress("[Epoch 003] loss=0.321000 | val_loss=0.654000")

    assert progress["epoch_current"] == 3
    assert progress["epoch_percent"] == 100.0
    assert progress["loss"] == 0.321
    assert progress["val_loss"] == 0.654
    assert progress["phase"] == "epoch_summary"


def test_format_training_log_tail_filters_repeated_live_progress_lines() -> None:
    raw = (
        "Loaded data: T=100, C=7\n"
        "\rTrain ETTm1 H=96 [----------------------------] 257/53400   0.5% elapsed=00:13 eta=44:49 "
        "| epoch=1/100 batch=257/534 loss=0.900000"
        "\rTrain ETTm1 H=96 [----------------------------] 258/53400   0.5% elapsed=00:13 eta=44:48 "
        "| epoch=1/100 batch=258/534 loss=0.800000"
        "\nSaved run summary to: outputs/demo/run_summary.json\n"
    )

    assert parse_training_progress(raw)["batch_current"] == 258
    text = format_training_log_tail(raw)

    assert "Loaded data: T=100, C=7" in text
    assert "Saved run summary" in text
    assert "batch=257/534" not in text
    assert "batch=258/534" not in text
    assert clean_training_log_text(raw).count("batch=") == 2


def test_training_progress_handles_terminal_truncated_suffix() -> None:
    raw = (
        "Train ETTm1 H=96 [#######---------------------] 12816/53400  24.0% "
        "elapsed=08:31 eta=26:58 | epoch=24/100 batch=534/..."
    )

    progress = parse_training_progress(raw)
    text = format_training_log_tail(raw)

    assert progress["epoch_current"] == 24
    assert progress["epoch_total"] == 100
    assert progress["batch_current"] == 534
    assert progress["batch_total"] == 534
    assert progress["epoch_percent"] == 100.0
    assert progress["global_percent"] == 24.0
    assert text == ""
