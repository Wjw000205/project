from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_input96_qgwnt_full_horizon_transfer as qgwnt


def _source_cfg() -> dict:
    return {
        "exp": {"seed": 2026},
        "data": {"csv_path": "data/ETTm1.csv", "date_col": 0, "max_rows": 57600},
        "window": {"input_len": 96, "pred_len": 192, "past_context": True},
        "normalize": {"global_zscore": True, "train_only": True},
        "cluster": {"method": "leader", "train_only": True},
        "corr": {"compute": True},
        "model": {"predictor": "channel_head_mlp", "hidden_dim": 32},
        "moe": {
            "enable": True,
            "freeze_backbone": True,
            "pred_side_residual": {
                "enable": True,
                "selection_policy": "val_mse_candidate_channel_guarded",
            },
        },
        "penalties": {"enabled": ["level"]},
        "train": {"epochs": 6, "batch_size": 64, "lr": 0.001},
        "early_stop": {"patience": 10},
    }


def _target_cfg() -> dict:
    return {
        "data": {"csv_path": "data/ETTh1.csv", "date_col": 0, "max_rows": 17420},
        "window": {"input_len": 96, "pred_len": 192, "past_context": True},
    }


def test_make_qgwnt_config_keeps_input96_horizon_and_unfreezes_backbone(tmp_path: Path) -> None:
    cfg = qgwnt.make_qgwnt_finetune_config(
        source="ETTm1",
        target="ETTh1",
        horizon=192,
        source_cfg=_source_cfg(),
        target_cfg=_target_cfg(),
        source_checkpoint=Path("source/best_checkpoint.pt"),
        source_memory=Path("source/cluster_memory.pt"),
        fixed_cluster_id=[0, 1, 0],
        out_dir=tmp_path / "run",
        device="cpu",
        batch_size=32,
        skip_test=True,
    )

    assert cfg["window"]["input_len"] == 96
    assert cfg["window"]["pred_len"] == 192
    assert cfg["cluster"]["fixed_cluster_id"] == [0, 1, 0]
    assert cfg["moe"]["freeze_backbone"] is False
    assert cfg["moe"]["pred_side_residual"]["selection_policy"] == "val_mse_candidate_channel"
    assert cfg["train"]["lr"] == qgwnt.QGWNT_LR
    assert cfg["train"]["epochs"] == qgwnt.QGWNT_EPOCHS
    assert cfg["eval"]["skip_test"] is True
    assert cfg["memory"]["save_checkpoint"] is True
    assert cfg["finetune"]["load_gate"] is True
    assert cfg["finetune"]["partial_model_state"] is True
    assert cfg["finetune"]["load_pred_residual"] is True
    assert cfg["finetune"]["strict_pred_residual"] is False


def test_make_qgwnt_config_keeps_ettm2_strict_path_without_partial(tmp_path: Path) -> None:
    cfg = qgwnt.make_qgwnt_finetune_config(
        source="ETTm2",
        target="ETTm1",
        horizon=336,
        source_cfg=_source_cfg(),
        target_cfg=_target_cfg(),
        source_checkpoint=Path("source/best_checkpoint.pt"),
        source_memory=Path("source/cluster_memory.pt"),
        fixed_cluster_id=[0, 0, 1],
        out_dir=tmp_path / "run",
        device="cpu",
        batch_size=32,
        skip_test=False,
    )

    assert cfg["window"]["input_len"] == 96
    assert cfg["window"]["pred_len"] == 336
    assert cfg["eval"]["skip_test"] is False
    assert cfg["finetune"]["strict_model"] is True
    assert "partial_model_state" not in cfg["finetune"]
    assert "load_pred_residual" not in cfg["finetune"]


def test_build_transfer_config_uses_train_route_and_validation_split(tmp_path: Path) -> None:
    cfg_path = qgwnt.build_transfer_config(
        source="ETTm1",
        target="ETTh1",
        horizon=720,
        source_info={
            "checkpoint": tmp_path / "best_checkpoint.pt",
            "memory": tmp_path / "cluster_memory.pt",
            "summary": tmp_path / "run_summary.json",
            "source_cfg": {"data": {"csv_path": "data/ETTm1.csv", "date_col": 0}},
        },
        out_root=tmp_path,
        device="cpu",
        batch_size=16,
        resample_method="last",
        eval_split="val",
    )
    cfg = qgwnt.read_yaml(cfg_path)

    assert cfg["window"]["input_len"] == 96
    assert cfg["window"]["pred_len"] == 720
    assert cfg["transfer"]["route_fit_scope"] == "train"
    assert cfg["transfer"]["cluster_balance_repair"]["enable"] is True
    assert cfg["eval"]["split"] == "val"


def test_make_source_config_preserves_epochs_when_override_is_zero(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(qgwnt, "config_path", lambda dataset, horizon: Path("unused.yaml"))
    monkeypatch.setattr(qgwnt, "read_yaml", lambda path: _source_cfg())

    _, cfg = qgwnt.make_source_config(
        source="ETTm1",
        horizon=192,
        out_root=tmp_path,
        device="cpu",
        source_epochs=0,
    )

    assert cfg["train"]["epochs"] == 6
    assert cfg["window"]["input_len"] == 96
    assert cfg["window"]["pred_len"] == 192
    assert cfg["memory"]["save_checkpoint"] is True
