from __future__ import annotations

from scripts.run_pems_batch_tune import candidates, configure, filter_candidates


def test_pems_batch_tune_config_enables_lazy_windows(tmp_path) -> None:
    cfg = configure(
        {},
        dataset="PEMS07",
        horizon=12,
        cand=candidates()[0],
        phase="search",
        out_dir=tmp_path / "run",
        epochs=8,
        skip_test=True,
        device="cuda:0",
    )

    assert cfg["window"]["input_len"] == 96
    assert cfg["window"]["pred_len"] == 12
    assert cfg["window"]["lazy"] is True
    assert cfg["knn_hybrid"]["enable"] is False


def test_pems_batch_tune_config_accepts_input_len_override(tmp_path) -> None:
    cfg = configure(
        {},
        dataset="PEMS07",
        horizon=12,
        cand=candidates()[0],
        phase="search",
        out_dir=tmp_path / "run",
        epochs=8,
        skip_test=True,
        device="cuda:0",
        input_len=96,
    )

    assert cfg["window"]["input_len"] == 96
    assert cfg["window"]["pred_len"] == 12
    assert cfg["window"]["lazy"] is True


def test_pems_batch_tune_config_applies_optional_loss_weights(tmp_path) -> None:
    cand = dict(candidates()[0])
    cand["mse_weight"] = 1.0
    cand["mae_weight"] = 0.25

    cfg = configure(
        {"train": {"mse_weight": 0.9, "mae_objective": {"weight": 0.6}}},
        dataset="PEMS04",
        horizon=12,
        cand=cand,
        phase="search",
        out_dir=tmp_path / "run",
        epochs=8,
        skip_test=True,
        device="cuda:0",
    )

    assert cfg["train"]["mse_weight"] == 1.0
    assert cfg["train"]["mae_objective"]["weight"] == 0.25


def test_pems_batch_tune_config_applies_predictor_overrides(tmp_path) -> None:
    cand = dict(candidates()[0])
    cand["predictor"] = "channel_dlinear"
    cand["model_overrides"] = {
        "dlinear_kernel_size": 25,
        "seasonal_residual": True,
        "seasonal_period": 288,
    }

    cfg = configure(
        {},
        dataset="PEMS08",
        horizon=12,
        cand=cand,
        phase="search",
        out_dir=tmp_path / "run",
        epochs=8,
        skip_test=True,
        device="cuda:0",
    )

    assert cfg["model"]["predictor"] == "channel_dlinear"
    assert cfg["model"]["dlinear_kernel_size"] == 25
    assert cfg["model"]["seasonal_residual"] is True
    assert cfg["model"]["seasonal_period"] == 288


def test_pems_batch_tune_config_applies_selection_metric(tmp_path) -> None:
    cand = dict(candidates()[0])
    cand["selection_metric"] = "val_mae"

    cfg = configure(
        {},
        dataset="PEMS03",
        horizon=12,
        cand=cand,
        phase="search",
        out_dir=tmp_path / "run",
        epochs=8,
        skip_test=True,
        device="cuda:0",
    )

    assert cfg["train"]["selection_metric"] == "val_mae"


def test_pems_batch_tune_config_applies_candidate_seed(tmp_path) -> None:
    cand = dict(candidates()[0])
    cand["seed"] = 2031

    cfg = configure(
        {"exp": {"seed": 2026}, "cluster": {"random_state": 2026}},
        dataset="PEMS08",
        horizon=96,
        cand=cand,
        phase="final",
        out_dir=tmp_path / "run",
        epochs=8,
        skip_test=False,
        device="cuda:0",
    )

    assert cfg["exp"]["seed"] == 2031
    assert cfg["cluster"]["random_state"] == 2031


def test_pems_batch_tune_config_can_save_checkpoint(tmp_path) -> None:
    cfg = configure(
        {},
        dataset="PEMS03",
        horizon=48,
        cand=candidates()[0],
        phase="final",
        out_dir=tmp_path / "run",
        epochs=8,
        skip_test=False,
        device="cuda:0",
        save_checkpoint=True,
    )

    assert cfg["memory"]["save_checkpoint"] is True


def test_pems_batch_tune_config_can_disable_moe_for_backbone_only(tmp_path) -> None:
    cfg = configure(
        {"moe": {"enable": True}},
        dataset="PEMS04",
        horizon=12,
        cand=candidates()[0],
        phase="final",
        out_dir=tmp_path / "run",
        epochs=8,
        skip_test=False,
        device="cuda:0",
        backbone_only=True,
    )

    assert cfg["moe"]["enable"] is False


def test_pems_batch_tune_config_can_finetune_from_backbone_checkpoint(tmp_path) -> None:
    checkpoint = tmp_path / "backbone" / "best_checkpoint.pt"
    cfg = configure(
        {"moe": {"enable": True}},
        dataset="PEMS04",
        horizon=24,
        cand=candidates()[0],
        phase="final",
        out_dir=tmp_path / "run",
        epochs=8,
        skip_test=False,
        device="cuda:0",
        finetune_checkpoint=checkpoint,
    )

    assert cfg["moe"]["enable"] is True
    assert cfg["finetune"]["enable"] is True
    assert cfg["finetune"]["checkpoint_path"] == str(checkpoint)
    assert cfg["finetune"]["load_model"] is True
    assert cfg["finetune"]["load_gate"] is False


def test_pems_batch_tune_config_can_enable_calibration_and_override_lr(tmp_path) -> None:
    cfg = configure(
        {},
        dataset="PEMS07",
        horizon=12,
        cand=candidates()[0],
        phase="final",
        out_dir=tmp_path / "run",
        epochs=1,
        skip_test=False,
        device="cuda:0",
        calibration_cfg={"enable": True, "method": "median", "shrink": 1.0, "max_abs": 0.0},
        lr_override=0.0,
    )

    assert cfg["calibration"] == {"enable": True, "method": "median", "shrink": 1.0, "max_abs": 0.0}
    assert cfg["train"]["lr"] == 0.0


def test_pems_batch_tune_config_can_freeze_backbone_for_moe_stage(tmp_path) -> None:
    cfg = configure(
        {"moe": {"enable": True}},
        dataset="PEMS04",
        horizon=12,
        cand=candidates()[0],
        phase="final",
        out_dir=tmp_path / "run",
        epochs=3,
        skip_test=False,
        device="cuda:0",
        freeze_backbone=True,
    )

    assert cfg["moe"]["enable"] is True
    assert cfg["moe"]["freeze_backbone"] is True


def test_pems_batch_tune_config_applies_moe_overrides(tmp_path) -> None:
    cand = dict(candidates()[0])
    cand["moe_overrides"] = {
        "train_stat_anchor_expert": {
            "enable": True,
            "period": 288,
            "scale_selection": {"enable": True, "metric": "mse", "max_scale": 0.3, "steps": 13},
        },
        "train_residual_anchor_expert": {
            "enable": True,
            "period": 288,
            "scale_selection": {"enable": True, "metric": "mae", "max_scale": 1.2, "steps": 49},
        },
    }

    cfg = configure(
        {"moe": {"enable": True}},
        dataset="PEMS07",
        horizon=24,
        cand=cand,
        phase="final",
        out_dir=tmp_path / "run",
        epochs=3,
        skip_test=False,
        device="cuda:0",
    )

    assert cfg["moe"]["train_stat_anchor_expert"]["period"] == 288
    assert cfg["moe"]["train_stat_anchor_expert"]["scale_selection"]["max_scale"] == 0.3
    assert cfg["moe"]["train_residual_anchor_expert"]["period"] == 288
    assert cfg["moe"]["train_residual_anchor_expert"]["scale_selection"]["metric"] == "mae"


def test_pems_batch_tune_includes_anchor_moe_candidates() -> None:
    by_name = {cand["name"]: cand for cand in candidates()}

    mse_anchor = by_name["bs64_cch_h128_do000_l001_mse050_mae150_valmae_anchor_p288"]
    mae_anchor = by_name["bs64_cch_h128_do000_l001_mse050_mae150_valmae_anchor_p288_residmae"]

    assert mse_anchor["moe_overrides"]["train_stat_anchor_expert"]["period"] == 288
    assert mse_anchor["moe_overrides"]["train_residual_anchor_expert"]["scale_selection"]["metric"] == "mse"
    assert mae_anchor["moe_overrides"]["train_residual_anchor_expert"]["scale_selection"]["metric"] == "mae"


def test_pems_batch_tune_includes_low_memory_h128_context_candidate() -> None:
    by_name = {cand["name"]: cand for cand in candidates()}

    cand = by_name["bs32_cch_h128_do000_l001_mse080_mae120_valmae"]

    assert cand["predictor"] == "context_channel_head_mlp"
    assert cand["batch_size"] == 32
    assert cand["hidden_dim"] == 128
    assert cand["dropout"] == 0.0
    assert cand["selection_metric"] == "val_mae"


def test_pems_batch_tune_includes_h96_backbone_refinement_candidates() -> None:
    by_name = {cand["name"]: cand for cand in candidates()}

    wider = by_name["bs32_cch_h160_do000_l001_mse050_mae150_valmae"]
    widest = by_name["bs32_cch_h192_do000_l001_mse050_mae150_valmae"]
    h224 = by_name["bs32_cch_h224_do000_l001_mse050_mae150_valmae"]
    h192_s2031 = by_name["bs32_cch_h192_do000_l001_mse050_mae150_valmae_s2031"]
    stronger_mae = by_name["bs32_cch_h128_do000_l001_mse030_mae200_valmae"]

    assert wider["predictor"] == "context_channel_head_mlp"
    assert wider["batch_size"] == 32
    assert wider["hidden_dim"] == 160
    assert wider["mse_weight"] == 0.50
    assert wider["mae_weight"] == 1.50
    assert widest["batch_size"] == 32
    assert widest["hidden_dim"] == 192
    assert widest["mse_weight"] == 0.50
    assert widest["mae_weight"] == 1.50
    assert h224["batch_size"] == 32
    assert h224["hidden_dim"] == 224
    assert h224["mse_weight"] == 0.50
    assert h224["mae_weight"] == 1.50
    assert h192_s2031["hidden_dim"] == 192
    assert h192_s2031["seed"] == 2031
    assert stronger_mae["hidden_dim"] == 128
    assert stronger_mae["mse_weight"] == 0.30
    assert stronger_mae["mae_weight"] == 2.00


def test_pems_batch_tune_filter_candidates_by_name_preserves_order() -> None:
    selected = filter_candidates(
        candidates(),
        "bs64_cch_h96_do000_l001_mse080_mae120_valmae,bs32_cch_h128_do000_l001_mse050_mae150_valmae",
    )

    assert [cand["name"] for cand in selected] == [
        "bs64_cch_h96_do000_l001_mse080_mae120_valmae",
        "bs32_cch_h128_do000_l001_mse050_mae150_valmae",
    ]
