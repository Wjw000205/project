from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_input96_transfer_rerun as transfer_rerun


def test_make_finetune_config_keeps_input96_and_matches_legacy_finetune_protocol(tmp_path: Path) -> None:
    source_cfg = {
        "exp": {"seed": 2026},
        "data": {"csv_path": "data/ETTm1.csv", "date_col": 0, "max_rows": 57600},
        "window": {"input_len": 96, "pred_len": 96, "past_context": True},
        "normalize": {"global_zscore": True, "train_only": True},
        "cluster": {"method": "leader", "train_only": True},
        "corr": {"compute": True},
        "model": {"predictor": "mlp", "hidden_dim": 32},
        "moe": {
            "enable": True,
            "pred_side_residual": {
                "enable": True,
                "selection_policy": "val_mse_candidate_channel_guarded",
            },
        },
        "penalties": {"enabled": ["level"]},
        "train": {"epochs": 6, "batch_size": 64, "lr": 0.001},
        "early_stop": {"patience": 10},
    }
    target_cfg = {
        "data": {"csv_path": "data/ETTh1.csv", "date_col": 0, "max_rows": 17420},
        "window": {"input_len": 96, "pred_len": 96, "past_context": True},
    }

    cfg = transfer_rerun.make_finetune_config(
        source="ETTm1",
        target="ETTh1",
        source_cfg=source_cfg,
        target_cfg=target_cfg,
        source_checkpoint=Path("source/best_checkpoint.pt"),
        source_memory=Path("source/cluster_memory.pt"),
        fixed_cluster_id=[2, 1, 0],
        out_dir=tmp_path / "run",
        lr=1.0e-4,
        epochs=12,
        batch_size=32,
        device="cpu",
    )

    assert cfg["window"]["input_len"] == 96
    assert cfg["window"]["pred_len"] == 96
    assert cfg["cluster"]["fixed_cluster_id"] == [2, 1, 0]
    assert cfg["train"]["batch_size"] == 32
    assert cfg["train"]["weight_decay"] == 0.0001
    assert cfg["moe"]["pred_side_residual"]["selection_policy"] == "val_mse_candidate_channel"
    assert cfg["finetune"]["strict_window"] is True
    assert cfg["finetune"]["cluster_map"] == "index"
    assert cfg["finetune"]["load_gate"] is True
    assert cfg["finetune"]["load_dynamic_lambda"] is True
    assert cfg["finetune"]["load_learnable_lambda"] is True
    assert "load_pred_residual" not in cfg["finetune"]
    assert "partial_model_state" not in cfg["finetune"]
    assert "strict_pred_residual" not in cfg["finetune"]
    assert "corr_align" not in cfg["finetune"]
    assert Path(cfg["finetune"]["checkpoint_path"]) == Path("source/best_checkpoint.pt")
    assert cfg["memory"]["save_checkpoint"] is True
    assert cfg["eval"]["skip_test"] is False


def test_prepare_finetune_target_cfg_uses_root_target_config_not_transfer_template(tmp_path: Path) -> None:
    cfg = transfer_rerun.prepare_finetune_target_cfg(
        source="ETTm1",
        target="ETTm2",
        out_root=tmp_path,
    )

    assert cfg["data"]["train_ratio"] == 0.6
    assert cfg["data"]["val_ratio"] == 0.2
    assert cfg["window"]["input_len"] == 96
    assert cfg["window"]["pred_len"] == 96


def test_prepare_source_uses_declared_source_config(monkeypatch, tmp_path: Path) -> None:
    configured_source = tmp_path / "configured_source.yaml"
    exported_cfg = tmp_path / "exported.yaml"
    checkpoint = tmp_path / "best_checkpoint.pt"
    memory = tmp_path / "cluster_memory.pt"
    summary = tmp_path / "run_summary.json"
    captured: dict[str, Path | None] = {}

    def fake_export_source(**kwargs):
        captured["source_config_path"] = kwargs.get("source_config_path")
        memory.touch()
        return exported_cfg, checkpoint, memory, summary

    monkeypatch.setattr(transfer_rerun, "export_source", fake_export_source)
    monkeypatch.setattr(transfer_rerun, "read_yaml", lambda path: {"data": {"csv_path": "data/ETTm2.csv"}})
    monkeypatch.setattr(
        transfer_rerun,
        "load_cluster_checkpoint",
        lambda path, device: {"meta": {"input_len": 96, "pred_len": 96}},
    )
    monkeypatch.setattr(
        transfer_rerun,
        "load_json",
        lambda path: {"test": {"avg_mse": 0.1, "avg_mae": 0.2}},
    )

    info = {"config": configured_source}
    prepared = transfer_rerun.prepare_source(
        "ETTm2",
        info,
        tmp_path,
        device="cpu",
        py="python",
        rerun_source=False,
        source_epochs=1,
    )

    assert captured["source_config_path"] == configured_source
    assert prepared["config"] == exported_cfg


def test_build_transfer_config_keeps_input96_and_uses_legacy_zero_shot_protocol(tmp_path: Path) -> None:
    source_root = transfer_rerun.ROOT / "outputs" / "_test_input96_transfer_source"
    cfg_path = transfer_rerun.build_transfer_config(
        source="ETTm1",
        target="ETTh1",
        source_info={
            "checkpoint": source_root / "best_checkpoint.pt",
            "memory": source_root / "cluster_memory.pt",
            "summary": source_root / "run_summary.json",
            "source_cfg": {"data": {"csv_path": "data/ETTm1.csv", "date_col": 0}},
        },
        out_root=tmp_path,
        device="cpu",
        batch_size=32,
        resample_method="last",
    )

    cfg = transfer_rerun.read_yaml(cfg_path)

    assert cfg["data"]["train_ratio"] == 0.6
    assert cfg["data"]["val_ratio"] == 0.2
    assert cfg["window"]["input_len"] == 96
    assert cfg["window"]["pred_len"] == 96
    assert cfg["transfer"]["corr_mode"] == "cycle_template"
    assert cfg["transfer"]["route_fit_scope"] == "train"
    assert cfg["transfer"]["use_pred_residual"] is True
    assert cfg["transfer"]["cluster_balance_repair"] == {
        "enable": True,
        "target_counts": "source",
        "min_unique_clusters": 2,
    }
    assert cfg["transfer"]["resample"]["method"] == "last"
    assert cfg["eval"]["batch_size"] == 32
    assert cfg["eval"]["split"] == "test"


def test_select_best_finetune_rows_uses_validation_mse() -> None:
    rows = [
        {"status": "ok", "source": "ETTm1", "target": "ETTh1", "finetune_lr": "0.0001", "finetune_val_mse": "0.30"},
        {"status": "ok", "source": "ETTm1", "target": "ETTh1", "finetune_lr": "5e-05", "finetune_val_mse": "0.20"},
        {"status": "error", "source": "ETTm1", "target": "ETTh1", "finetune_lr": "2e-05", "finetune_val_mse": "0.10"},
        {"status": "ok", "source": "ETTm2", "target": "ETTh2", "finetune_lr": "0.0001", "finetune_val_mse": "0.40"},
    ]

    selected = transfer_rerun.select_best_finetune_rows(rows)

    assert selected[("ETTm1", "ETTh1")]["finetune_lr"] == "5e-05"
    assert selected[("ETTm2", "ETTh2")]["finetune_lr"] == "0.0001"
