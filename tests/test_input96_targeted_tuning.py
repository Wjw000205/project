from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

from scripts.run_input96_h96_targeted_tuning import (
    Candidate,
    DATASET_CONFIGS,
    common_prepare,
    filter_candidates_by_variant,
    model_candidates,
    moe_candidates,
    resolve_dataset_config,
    run_candidate,
)
from scripts.run_input96_moe_positive_search import apply_history_anchor_controls, apply_moe_training_controls
from scripts.run_input96_moe_activation_search import activation_candidates


def minimal_cfg() -> dict:
    return {
        "exp": {"name": "base", "out_dir": "outputs/base", "device": "cpu"},
        "data": {"csv_path": "data/ETTh1.csv", "date_col": 0},
        "window": {"input_len": 336, "pred_len": 96},
        "normalize": {},
        "corr": {},
        "cluster": {},
        "model": {"predictor": "mlp", "hidden_dim": 64, "dropout": 0.1},
        "moe": {"enable": False},
        "penalties": {"enabled": ["trend"]},
        "train": {"epochs": 1},
        "eval": {},
    }


def test_common_prepare_accepts_non_h96_horizon(tmp_path) -> None:
    cfg = common_prepare(
        minimal_cfg(),
        dataset="ETTh1",
        pred_len=336,
        out_dir=tmp_path / "run",
        run_name="ETTh1_input96_H336",
        device="cpu",
        epochs=2,
        skip_test=True,
    )

    assert cfg["window"]["input_len"] == 96
    assert cfg["window"]["pred_len"] == 336
    assert cfg["train"]["epochs"] == 2
    assert cfg["eval"]["skip_test"] is True


def test_common_prepare_can_enable_checkpoint_save(tmp_path) -> None:
    cfg = common_prepare(
        minimal_cfg(),
        dataset="ETTh1",
        pred_len=336,
        out_dir=tmp_path / "run",
        run_name="ETTh1_input96_H336",
        device="cpu",
        epochs=2,
        skip_test=False,
        save_checkpoint=True,
    )

    assert cfg["memory"]["save_checkpoint"] is True
    assert str(cfg["memory"]["checkpoint_path"]).endswith("best_checkpoint.pt")


def test_resolve_dataset_config_prefers_horizon_specific_yaml() -> None:
    assert resolve_dataset_config("ETTh1", 336).name == "ETTh1_H336.yaml"
    assert resolve_dataset_config("PEMS08", 96).name == "PEMS08_H96.yaml"


def test_run_candidate_keeps_horizon_in_dry_run_paths(tmp_path) -> None:
    row, cfg = run_candidate(
        dataset="ETTh1",
        pred_len=336,
        base_cfg=minimal_cfg(),
        cand=Candidate("model", "tiny", {"model": {"hidden_dim": 32}}),
        out_root=tmp_path,
        device="cpu",
        epochs=1,
        skip_test=True,
        dry_run=True,
    )

    assert cfg["window"]["input_len"] == 96
    assert cfg["window"]["pred_len"] == 336
    assert row["pred_len"] == 336
    assert "H336" in row["config_path"]
    assert "H336" in row["out_dir"]


def test_ettm1_backbone_search_cli_accepts_non_h96_horizon(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    out_root = tmp_path / "ettm1_h192"

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_input96_ettm1_no_knn_backbone_search.py",
            "--out-root",
            str(out_root),
            "--variants",
            "patchtst_d96_p8s4_l2_do005_wd5e5_mae04",
            "--horizon",
            "192",
            "--epochs",
            "1",
            "--skip-test",
            "--dry-run",
            "--device",
            "cpu",
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "ETTm1 H192" in proc.stdout
    with (out_root / "backbone_rows.csv").open(newline="", encoding="utf-8") as f:
        row = next(csv.DictReader(f))
    assert row["pred_len"] == "192"
    assert "H192" in row["config_path"]
    assert "H192" in row["out_dir"]


def test_model_candidates_include_small_capacity_etth2_options() -> None:
    variants = {cand.variant for cand in model_candidates()}

    assert "mlp_h64_do0_wd1e4_mae04" in variants
    assert "mlp_h96_do0_wd1e4_mae04" in variants
    assert "mlp_h96_do02_wd1e4_mseonly" in variants
    assert "mlp_h160_do02_wd1e4_mae04" in variants
    assert "channel_h128_do01_wd1e4_mae04" in variants
    assert "channel_h128_do015_wd1e4_mae02" in variants
    assert "channel_h128_do025_wd1e4_mae02" in variants
    assert "channel_h128_do02_wd3e4_mae02" in variants
    assert "channel_h160_do02_wd1e4_mae02" in variants
    assert "channel_h160_do02_wd1e4_mae04" in variants
    assert "channel_h192_do01_wd1e4_mae04" in variants
    assert "channel_h192_do01_wd1e4_mae06" in variants
    assert "channel_h192_do02_wd1e4_mae04" in variants
    assert "channel_h256_do02_wd1e4_mae04" in variants
    assert "seasonal_channel_h192_tail24_mixm2_do02_wd1e4_mae02" in variants
    assert "channel_h256_do0_wd1e3_mae06" in variants
    assert "electric_current_h320" in variants
    assert "electric_current_h256_do0" in variants


def test_model_candidates_include_electricity_backbone_refinement_options() -> None:
    candidates = {cand.variant: cand for cand in model_candidates()}

    h320 = candidates["electric_current_h320"].patch
    do0 = candidates["electric_current_h256_do0"].patch
    wd0 = candidates["electric_current_h256_wd0"].patch
    lr_low = candidates["electric_current_h256_lr8e4"].patch

    assert h320["model"]["predictor"] == "mlp"
    assert h320["model"]["hidden_dim"] == 320
    assert h320["model"]["dropout"] == 0.0468935703282562
    assert do0["model"]["hidden_dim"] == 256
    assert do0["model"]["dropout"] == 0.0
    assert wd0["train"]["weight_decay"] == 0.0
    assert lr_low["train"]["lr"] == 8.0e-4


def test_model_candidates_include_electricity_h96_param_transfer_options() -> None:
    candidates = {cand.variant: cand for cand in model_candidates()}

    do0 = candidates["electric_h96params_h256_do0"].patch
    h320 = candidates["electric_h96params_h320_do0"].patch
    lr_high = candidates["electric_h96params_h256_lr18e4_do0"].patch

    assert do0["model"] == {"predictor": "mlp", "hidden_dim": 256, "dropout": 0.0}
    assert do0["train"]["lr"] == 0.001309395478035077
    assert do0["train"]["weight_decay"] == 1.0644440818212169e-05
    assert h320["model"]["hidden_dim"] == 320
    assert h320["model"]["dropout"] == 0.0
    assert h320["train"]["weight_decay"] == 1.0644440818212169e-05
    assert lr_high["train"]["lr"] == 1.8e-3


def test_model_candidates_include_electricity_seasonal_mlp_options() -> None:
    candidates = {cand.variant: cand for cand in model_candidates()}

    residual = candidates["electric_h96params_h256_do0_seasonalres_p24_np4"].patch
    anchor = candidates["electric_h96params_h256_do0_seasonalanchor_p24_np4_d05"].patch
    blend = candidates["electric_h96params_h256_do0_seasonalblend_p24_np4_m05_i02"].patch

    assert residual["model"]["predictor"] == "mlp"
    assert residual["model"]["hidden_dim"] == 256
    assert residual["model"]["dropout"] == 0.0
    assert residual["model"]["seasonal_residual"] is True
    assert residual["model"]["seasonal_period"] == 24
    assert residual["model"]["seasonal_num_periods"] == 4
    assert residual["train"]["lr"] == 0.001309395478035077
    assert residual["train"]["weight_decay"] == 1.0644440818212169e-05

    assert anchor["model"]["seasonal_anchor"] is True
    assert anchor["model"]["seasonal_anchor_period"] == 24
    assert anchor["model"]["seasonal_anchor_num_periods"] == 4
    assert anchor["model"]["seasonal_anchor_delta_scale"] == 0.5

    adapter = blend["model"]["seasonal_blend_adapter"]
    assert adapter["enable"] is True
    assert adapter["period"] == 24
    assert adapter["num_periods"] == 4
    assert adapter["max_mix"] == 0.5
    assert adapter["init_mix"] == 0.2


def test_model_candidates_include_electricity_model_train_stat_adapter_options() -> None:
    candidates = {cand.variant: cand for cand in model_candidates()}

    p24 = candidates["electric_h96params_h256_do0_modelstat_p24_mean_a020"].patch
    p168_small = candidates["electric_h96params_h256_do0_modelstat_p168_mean_a005"].patch
    p168_mid = candidates["electric_h96params_h256_do0_modelstat_p168_mean_a030"].patch
    p168_high = candidates["electric_h96params_h256_do0_modelstat_p168_mean_a080"].patch
    p168_resid = candidates["electric_h96params_h256_do0_modelstat_p168_mean_anchorres_a10"].patch
    p168_center = candidates["electric_h96params_h256_do0_modelstat_p168_center_anchorres_a10"].patch
    p168_center_scaled = candidates["electric_h96params_h256_do0_modelstat_p168_center_anchorres_a10_cs08"].patch
    p168_center_bs8 = candidates[
        "electric_h96params_h256_do0_modelstat_p168_center_anchorres_a10_wd0_bs8"
    ].patch
    p168 = candidates["electric_h96params_h256_do0_modelstat_p168_mean_select_m05_seg12"].patch
    p168_low = candidates["electric_h96params_h256_do0_modelstat_p168_mean_select_m02_seg12"].patch
    delta = candidates["electric_h96params_h256_do0_modelstat_p24_delta_repeat_select_m05_seg12"].patch

    assert p24["model"]["predictor"] == "mlp"
    assert p24["model"]["train_stat_adapter"] == {
        "enable": True,
        "period": 24,
        "mode": "phase_mean",
        "alpha": 0.2,
        "blend_target": "prediction",
    }
    assert "moe" not in p24
    assert p168_small["model"]["train_stat_adapter"]["alpha"] == 0.05
    assert p168_mid["model"]["train_stat_adapter"]["alpha"] == 0.3
    assert p168_high["model"]["train_stat_adapter"]["alpha"] == 0.8
    assert p168_resid["model"]["train_stat_adapter"]["combine_mode"] == "anchor_plus_prediction"
    assert p168_resid["model"]["train_stat_adapter"]["alpha"] == 1.0
    assert p168_center["model"]["train_stat_adapter"]["input_center"] is True
    assert p168_center_scaled["model"]["train_stat_adapter"]["input_center_scale"] == 0.8
    assert p168_center_bs8["model"]["train_stat_adapter"]["input_center"] is True
    assert p168_center_bs8["model"]["train_stat_adapter"]["combine_mode"] == "anchor_plus_prediction"
    assert p168_center_bs8["train"]["batch_size"] == 8
    assert p168_center_bs8["train"]["weight_decay"] == 0.0

    adapter = p168["model"]["train_stat_adapter"]
    assert adapter["period"] == 168
    assert adapter["mode"] == "phase_mean"
    assert adapter["scale_selection"] == {
        "enable": True,
        "metric": "mse",
        "max_scale": 0.5,
        "steps": 21,
        "horizon_segments": 12,
    }
    assert p168_low["model"]["train_stat_adapter"]["scale_selection"]["max_scale"] == 0.2

    delta_adapter = delta["model"]["train_stat_adapter"]
    assert delta_adapter["period"] == 24
    assert delta_adapter["mode"] == "phase_delta"
    assert delta_adapter["reference"] == "repeat"


def test_model_candidates_include_electricity_h720_centered_residual_options() -> None:
    candidates = {cand.variant: cand for cand in model_candidates()}

    a08 = candidates["electric_h720_centerres_h192_a08_wd1e4_do002_bs8"].patch
    a06 = candidates["electric_h720_centerres_h128_a06_wd1e4_do002_bs8"].patch
    wd1e5 = candidates["electric_h720_centerres_h192_a08_wd1e5_do0_bs8"].patch
    a07 = candidates["electric_h720_centerres_h192_a07_wd1e5_do0_bs8"].patch
    lr_low = candidates["electric_h720_centerres_h192_a08_wd1e5_do0_lr8e4_bs8"].patch
    h224_a09 = candidates["electric_h720_centerres_h224_a09_wd1e5_do0_bs8"].patch
    h256_a08 = candidates["electric_h720_centerres_h256_a08_wd1e5_do0_bs8"].patch

    assert a08["model"]["predictor"] == "mlp"
    assert a08["model"]["hidden_dim"] == 192
    assert a08["model"]["dropout"] == 0.02
    assert a08["model"]["train_stat_adapter"] == {
        "enable": True,
        "period": 168,
        "mode": "phase_mean",
        "alpha": 0.8,
        "blend_target": "prediction",
        "combine_mode": "anchor_plus_prediction",
        "input_center": True,
    }
    assert a08["train"]["batch_size"] == 8
    assert a08["train"]["weight_decay"] == 1.0e-4
    assert a06["model"]["hidden_dim"] == 128
    assert a06["model"]["train_stat_adapter"]["alpha"] == 0.6
    assert wd1e5["train"]["weight_decay"] == 1.0e-5
    assert wd1e5["model"]["dropout"] == 0.0
    assert a07["model"]["train_stat_adapter"]["alpha"] == 0.7
    assert lr_low["train"]["lr"] == 8.0e-4
    assert h224_a09["model"]["hidden_dim"] == 224
    assert h224_a09["model"]["train_stat_adapter"]["alpha"] == 0.9
    assert h256_a08["model"]["hidden_dim"] == 256
    assert h256_a08["model"]["train_stat_adapter"]["alpha"] == 0.8


def test_model_candidates_include_electricity_h96_centered_residual_refine_options() -> None:
    candidates = {cand.variant: cand for cand in model_candidates()}

    bs64 = candidates["electric_h96_centerres_h256_a10_wd0_bs64"].patch
    alpha_low = candidates["electric_h96_centerres_h256_a095_wd0_bs128"].patch
    mae = candidates["electric_h96_centerres_h256_a10_wd0_mae005_bs128"].patch
    select = candidates["electric_h96_centerres_h256_a10_wd0_modelstat_select_m300_seg12_bs128"].patch
    swa = candidates["electric_h96_centerres_h256_a10_wd0_swa_bs128"].patch

    assert bs64["model"]["predictor"] == "mlp"
    assert bs64["model"]["hidden_dim"] == 256
    assert bs64["model"]["train_stat_adapter"]["input_center"] is True
    assert bs64["train"]["batch_size"] == 64
    assert bs64["train"]["weight_decay"] == 0.0
    assert alpha_low["model"]["train_stat_adapter"]["alpha"] == 0.95
    assert mae["train"]["mae_objective"] == {"enable": True, "kind": "l1", "weight": 0.05}
    assert select["model"]["train_stat_adapter"]["scale_selection"] == {
        "enable": True,
        "metric": "mse",
        "max_scale": 3.0,
        "steps": 121,
        "horizon_segments": 12,
    }
    assert swa["train"]["swa"]["enable"] is True
    assert swa["early_stop"]["patience"] == 20


def test_model_candidates_include_electricity_h720_mlp_family_arch_options() -> None:
    candidates = {cand.variant: cand for cand in model_candidates()}

    segment = candidates["electric_h720_centerres_segment_h224_a08_wd1e5_do0_bs8"].patch
    anchor = candidates["electric_h720_centerres_longanchor_h224_a08_d025_wd1e5_do0_bs8"].patch
    recursive = candidates["electric_h720_centerres_recursive_h256_a08_wd1e5_do0_bs8"].patch
    phase_select = candidates["electric_h720_centerres_h224_a08_wd1e5_modelstat_select_m300_seg12_bs8"].patch
    phase_select_channel = candidates["electric_h720_centerres_h224_a08_wd1e5_modelstat_select_m120_channel_bs8"].patch

    assert segment["model"]["predictor"] == "segment_mlp"
    assert segment["model"]["segment_chunk_len"] == 96
    assert segment["model"]["train_stat_adapter"]["alpha"] == 0.8
    assert anchor["model"]["predictor"] == "long_anchor_mlp"
    assert anchor["model"]["anchor_chunk_len"] == 96
    assert anchor["model"]["anchor_detail_scale"] == 0.25
    assert anchor["model"]["anchor_residual"] is True
    assert recursive["model"]["predictor"] == "mlp"
    assert recursive["model"]["recursive_rollout"] is True
    assert recursive["model"]["recursive_chunk_len"] == 96
    assert recursive["train"]["batch_size"] == 8
    assert phase_select["model"]["train_stat_adapter"]["scale_selection"]["max_scale"] == 3.0
    assert phase_select["model"]["train_stat_adapter"]["scale_selection"]["horizon_segments"] == 12
    assert phase_select_channel["model"]["train_stat_adapter"]["scale_selection"] == {
        "enable": True,
        "metric": "mse",
        "max_scale": 1.2,
        "steps": 49,
        "horizon_segments": 1,
    }


def test_model_candidates_include_electricity_centered_channel_context_options() -> None:
    candidates = {cand.variant: cand for cand in model_candidates()}

    h96_channel = candidates["electric_h96_centerres_channel_h192_a10_wd1e5_do0_bs64"].patch
    h96_context = candidates["electric_h96_centerres_context_h128_a10_wd1e5_do0_bs64"].patch
    h720_context = candidates["electric_h720_centerres_context_h128_a08_wd1e5_do0_bs8"].patch

    assert h96_channel["model"]["predictor"] == "channel_head_mlp"
    assert h96_channel["model"]["hidden_dim"] == 192
    assert h96_channel["model"]["train_stat_adapter"]["input_center"] is True
    assert h96_channel["model"]["train_stat_adapter"]["combine_mode"] == "anchor_plus_prediction"
    assert h96_channel["train"]["batch_size"] == 64
    assert h96_context["model"]["predictor"] == "context_channel_head_mlp"
    assert h96_context["model"]["context_channel_head_include_delta"] is True
    assert h96_context["model"]["train_stat_adapter"]["alpha"] == 1.0
    assert h720_context["model"]["predictor"] == "context_channel_head_mlp"
    assert h720_context["model"]["train_stat_adapter"]["alpha"] == 0.8
    assert h720_context["train"]["batch_size"] == 8


def test_targeted_tuning_cli_accepts_val_mae_selection(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    out_root = tmp_path / "val_mae_selection"

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_input96_h96_targeted_tuning.py",
            "--out-root",
            str(out_root),
            "--datasets",
            "ETTh1",
            "--horizons",
            "192",
            "--model-variants",
            "current_model",
            "--skip-moe-search",
            "--search-epochs",
            "1",
            "--final-epochs",
            "1",
            "--selection-metric",
            "val_mae",
            "--dry-run",
            "--device",
            "cpu",
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
    )

    assert proc.returncode == 0, proc.stderr


def test_model_candidates_include_current_low_hd_options() -> None:
    candidates = {cand.variant: cand for cand in model_candidates()}

    h64 = candidates["current_h64"]
    h96 = candidates["current_h96"]
    h128 = candidates["current_h128"]

    assert h64.patch["model"] == {"hidden_dim": 64}
    assert h96.patch["model"] == {"hidden_dim": 96}
    assert h128.patch["model"] == {"hidden_dim": 128}


def test_model_candidates_include_general_backbone_options() -> None:
    candidates = {cand.variant: cand for cand in model_candidates()}

    seasonal = candidates["seasonal_channel_h128_do02_wd1e4_mae02"].patch
    assert seasonal["model"] == {
        "predictor": "seasonality_gated_channel_head_mlp",
        "hidden_dim": 128,
        "dropout": 0.2,
        "predictor_input_len": 96,
        "seasonal_mix_init": -2.0,
        "seasonal_gate_strength": 0.0,
        "anchor_chunk_len": 96,
        "anchor_detail_scale": 0.25,
    }
    assert seasonal["train"]["weight_decay"] == 1.0e-4
    assert seasonal["train"]["mae_objective"]["weight"] == 0.2

    tail24 = candidates["seasonal_channel_h128_tail24_mix0_do02_wd1e4_mae02"].patch
    assert tail24["model"]["predictor"] == "seasonality_gated_channel_head_mlp"
    assert tail24["model"]["predictor_input_len"] == 24
    assert tail24["model"]["seasonal_mix_init"] == 0.0
    assert tail24["model"]["seasonal_gate_strength"] == 1.0
    assert tail24["model"]["seasonal_gate_threshold"] == 0.2

    h96_tail24 = candidates["seasonal_channel_h96_tail24_mixm2_do02_wd1e4_mae02"].patch
    h160_tail24 = candidates["seasonal_channel_h160_tail24_mixm2_do02_wd1e4_mae02"].patch
    h256_tail24 = candidates["seasonal_channel_h256_tail24_mixm2_do02_wd1e4_mae02"].patch
    h128_do01 = candidates["seasonal_channel_h128_tail24_mixm2_do01_wd1e4_mae02"].patch
    h128_do025 = candidates["seasonal_channel_h128_tail24_mixm2_do025_wd1e4_mae02"].patch
    assert h96_tail24["model"]["hidden_dim"] == 96
    assert h160_tail24["model"]["hidden_dim"] == 160
    assert h256_tail24["model"]["hidden_dim"] == 256
    assert h128_do01["model"]["dropout"] == 0.1
    assert h128_do025["model"]["dropout"] == 0.25
    assert h96_tail24["model"]["predictor_input_len"] == 24
    assert h160_tail24["model"]["seasonal_mix_init"] == -2.0

    patchtst = candidates["patchtst_h128_p16s8_l2_do01_wd1e4_mae04"].patch
    assert patchtst["model"] == {
        "predictor": "patchtst",
        "hidden_dim": 128,
        "dropout": 0.1,
        "patch_d_model": 128,
        "patch_len": 16,
        "patch_stride": 8,
        "patch_num_layers": 2,
        "patch_num_heads": 4,
        "patch_ff_dim": 256,
    }
    assert patchtst["train"]["weight_decay"] == 1.0e-4
    assert patchtst["train"]["mae_objective"]["weight"] == 0.4

    anchor_basis = candidates["mlp_anchor_basis_h256_r16_wd1e4_mae06"].patch
    assert anchor_basis["model"]["predictor"] == "mlp"
    assert anchor_basis["model"]["mlp_residual_anchor"] is True
    assert anchor_basis["model"]["temporal_basis_adapter"] == {
        "enable": True,
        "rank": 16,
        "scale": 0.15,
        "init": "zero_delta",
    }


def test_moe_candidates_include_weak_guarded_positive_options() -> None:
    cfg = minimal_cfg()
    cfg["penalties"]["enabled"] = ["jump", "amp_under", "level", "delta"]
    cfg["moe"] = {"pred_side_residual": {"alpha_scale": 1.6}}

    variants = {cand.variant: cand for cand in moe_candidates(cfg, "compact")}
    cand = variants["current_guard_l005_a03_ms035"]
    patch = cand.patch
    residual = patch["moe"]["pred_side_residual"]
    gate = residual["gate_calibrator"]

    assert patch["moe"]["dynamic_lambda"]["enable"] is False
    assert patch["penalties"]["enabled"] == ["jump", "amp_under", "level", "delta"]
    assert patch["moe"]["lambda_init"] == {
        "jump": 0.005,
        "amp_under": 0.005,
        "level": 0.005,
        "delta": 0.005,
    }
    assert residual["selection_policy"] == "val_mse_gate_guarded"
    assert residual["selection_min_rel_improvement"] == 0.0005
    assert residual["alpha_scale"] == 0.3
    assert residual["residual_clip"] == 2.0
    assert gate["max_scale"] == 0.35
    assert gate["init_scale"] == 0.2


def test_moe_candidates_include_penalty_prior_activation_options() -> None:
    cfg = minimal_cfg()
    cfg["penalties"]["enabled"] = ["jump", "amp_under", "level", "delta"]

    variants = {cand.variant: cand for cand in moe_candidates(cfg, "compact")}
    patch = variants["current_prior_l0_a025_ms025_top1"].patch

    assert patch["moe"]["lambda_init"] == {
        "jump": 0.0,
        "amp_under": 0.0,
        "level": 0.0,
        "delta": 0.0,
    }
    assert patch["moe"]["cluster_penalty_prior"] == {
        "enable": True,
        "topk": 1,
        "hard_topk": True,
        "temperature": 0.7,
        "smoothing": 0.02,
        "use_normalized_penalty": True,
        "logit_strength": 1.0,
    }
    assert patch["moe"]["channel_penalty_prior"] == {
        "enable": True,
        "topk": 1,
        "hard_topk": True,
        "temperature": 0.7,
        "smoothing": 0.02,
        "use_normalized_penalty": True,
    }


def test_filter_candidates_by_variant_keeps_requested_order() -> None:
    candidates = [
        Candidate("moe", "a", {}),
        Candidate("moe", "b", {}),
        Candidate("moe", "c", {}),
    ]

    selected = filter_candidates_by_variant(candidates, ["c", "a"])

    assert [cand.variant for cand in selected] == ["c", "a"]


def test_apply_moe_training_controls_sets_warm_start_and_freeze() -> None:
    cfg = minimal_cfg()

    apply_moe_training_controls(
        cfg,
        warm_start_checkpoint="outputs/baseline/best_checkpoint.pt",
        freeze_backbone=True,
        lr=2.0e-4,
        weight_decay=1.0e-5,
    )

    assert cfg["moe"]["freeze_backbone"] is True
    assert cfg["train"]["lr"] == 2.0e-4
    assert cfg["train"]["weight_decay"] == 1.0e-5
    assert cfg["finetune"] == {
        "enable": True,
        "checkpoint_path": "outputs/baseline/best_checkpoint.pt",
        "strict_window": True,
        "strict_model": True,
        "cluster_map": "index",
        "load_model": True,
        "load_gate": False,
        "load_dynamic_lambda": False,
        "load_learnable_lambda": False,
    }


def test_apply_history_anchor_controls_disables_knn_and_sets_model_anchor() -> None:
    cfg = {"model": {}, "knn_hybrid": {"enable": True}}

    apply_history_anchor_controls(
        cfg,
        lags="96,192,288",
        alpha=0.2,
        blend_target="prediction",
    )

    assert cfg["knn_hybrid"]["enable"] is False
    assert cfg["model"]["history_anchor"] == {
        "enable": True,
        "lags": [96, 192, 288],
        "alpha": 0.2,
        "blend_target": "prediction",
        "history_scope": "input_window",
    }


def test_activation_candidates_enable_activation_head_gate() -> None:
    candidates = {cand.variant: cand for cand in activation_candidates({"penalties": {"enabled": ["level", "delta"]}})}

    patch = candidates["act_train_a1_ms1_bce02"].patch
    residual = patch["moe"]["pred_side_residual"]
    gate = residual["gate_calibrator"]

    assert patch["moe"]["dynamic_lambda"]["enable"] is False
    assert residual["selection_policy"] == "val_mse_gate_guarded"
    assert gate["source_split"] == "train"
    assert gate["activation_head_enable"] is True
    assert gate["apply_activation_threshold"] is True
    assert gate["activation_threshold"] == "auto"


def test_activation_candidates_include_candidate_selector_variants() -> None:
    candidates = {cand.variant: cand for cand in activation_candidates({"penalties": {"enabled": ["level", "delta"]}})}

    patch = candidates["selector_val_a1_pos2"].patch
    residual = patch["moe"]["pred_side_residual"]
    selector = residual["candidate_selector"]

    assert residual["selection_policy"] == "val_mse_gate_guarded"
    assert selector["enable"] is True
    assert selector["source_split"] == "val"
    assert selector["positive_sample_weight"] == 2.0
    assert patch["moe"]["dynamic_lambda"]["enable"] is False


def test_activation_candidates_include_long_lag_history_anchor_expert() -> None:
    candidates = {cand.variant: cand for cand in activation_candidates({"penalties": {"enabled": ["level", "delta"]}})}

    patch = candidates["anchor_expert_longlags_a0200"].patch
    expert = patch["moe"]["history_anchor_expert"]

    assert expert["enable"] is True
    assert expert["lags"] == [96, 192, 288, 384, 480, 576, 672, 768]
    assert expert["alpha"] == 0.2
    assert expert["history_scope"] == "input_window"
    assert patch["moe"]["pred_side_residual"]["enable"] is False


def test_activation_candidates_include_history_train_stat_residual_combo() -> None:
    candidates = {cand.variant: cand for cand in activation_candidates({"penalties": {"enabled": ["level", "delta"]}})}

    patch = candidates["hist020_trainstatresid_mean_p96_stat020_resid260_seg7"].patch
    moe = patch["moe"]

    assert moe["history_anchor_expert"] == {
        "enable": True,
        "lags": [96, 192, 288],
        "alpha": 0.2,
        "blend_target": "prediction",
        "history_scope": "input_window",
    }
    assert moe["train_stat_anchor_expert"]["period"] == 96
    assert moe["train_stat_anchor_expert"]["scale_selection"]["max_scale"] == 0.2
    assert moe["train_residual_anchor_expert"]["scale_selection"] == {
        "enable": True,
        "metric": "mse",
        "max_scale": 2.6,
        "steps": 105,
        "horizon_segments": 7,
    }

    h720_patch = candidates["hist035_trainstatresid_mean_p96_stat020_resid120_seg7"].patch
    assert h720_patch["moe"]["history_anchor_expert"]["alpha"] == 0.35
    assert h720_patch["moe"]["train_residual_anchor_expert"]["scale_selection"]["max_scale"] == 1.2
    assert h720_patch["moe"]["train_residual_anchor_expert"]["scale_selection"]["steps"] == 49

    h96_patch = candidates["hist020_trainstatresid_mean_p96_stat020_resid160_seg7"].patch
    assert h96_patch["moe"]["history_anchor_expert"]["alpha"] == 0.2
    assert h96_patch["moe"]["train_residual_anchor_expert"]["scale_selection"]["max_scale"] == 1.6
    assert h96_patch["moe"]["train_residual_anchor_expert"]["scale_selection"]["steps"] == 65

    h192_patch = candidates["hist020_trainstatresid_mean_p96_stat020_resid200_seg7"].patch
    assert h192_patch["moe"]["history_anchor_expert"]["alpha"] == 0.2
    assert h192_patch["moe"]["train_residual_anchor_expert"]["scale_selection"]["max_scale"] == 2.0
    assert h192_patch["moe"]["train_residual_anchor_expert"]["scale_selection"]["steps"] == 81


def test_activation_candidates_include_train_stat_anchor_expert() -> None:
    candidates = {cand.variant: cand for cand in activation_candidates({"penalties": {"enabled": ["level", "delta"]}})}

    patch = candidates["trainstat_period96_a0200"].patch
    expert = patch["moe"]["train_stat_anchor_expert"]

    assert expert == {
        "enable": True,
        "period": 96,
        "alpha": 0.2,
        "mode": "phase_mean",
        "reference": "last",
        "blend_target": "prediction",
    }
    assert patch["moe"]["pred_side_residual"]["enable"] is False


def test_activation_candidates_include_train_stat_fine_alpha_sweep() -> None:
    candidates = {cand.variant: cand for cand in activation_candidates({"penalties": {"enabled": ["level", "delta"]}})}

    assert candidates["trainstat_period96_a0090"].patch["moe"]["train_stat_anchor_expert"]["alpha"] == 0.09
    assert candidates["trainstat_period96_a0110"].patch["moe"]["train_stat_anchor_expert"]["alpha"] == 0.11


def test_activation_candidates_include_train_residual_anchor_expert() -> None:
    candidates = {cand.variant: cand for cand in activation_candidates({"penalties": {"enabled": ["level", "delta"]}})}

    patch = candidates["trainresid_select_p96_mse050_s21"].patch
    expert = patch["moe"]["train_residual_anchor_expert"]

    assert expert["enable"] is True
    assert expert["period"] == 96
    assert expert["alpha"] == 0.0
    assert expert["blend_target"] == "prediction"
    assert expert["scale_selection"] == {
        "enable": True,
        "metric": "mse",
        "max_scale": 0.5,
        "steps": 21,
    }
    assert patch["moe"]["history_anchor_expert"]["enable"] is False
    assert patch["moe"]["train_stat_anchor_expert"]["enable"] is False
    assert patch["moe"]["pred_side_residual"]["enable"] is False


def test_activation_candidates_include_wider_train_residual_scale_sweep() -> None:
    candidates = {cand.variant: cand for cand in activation_candidates({"penalties": {"enabled": ["level", "delta"]}})}

    patch = candidates["trainresid_seg4_p96_mse080_s33"].patch
    expert = patch["moe"]["train_residual_anchor_expert"]

    assert expert["scale_selection"] == {
        "enable": True,
        "metric": "mse",
        "max_scale": 0.8,
        "steps": 33,
        "horizon_segments": 4,
    }
    assert patch["moe"]["history_anchor_expert"]["enable"] is False
    assert patch["moe"]["train_stat_anchor_expert"]["enable"] is False


def test_activation_candidates_include_weekly_train_residual_anchor_expert() -> None:
    candidates = {cand.variant: cand for cand in activation_candidates({"penalties": {"enabled": ["level", "delta"]}})}

    patch = candidates["trainresid_seg12_p168_mse080_s33"].patch
    expert = patch["moe"]["train_residual_anchor_expert"]
    wide_patch = candidates["trainresid_seg12_p168_mse200_s81"].patch
    wide_expert = wide_patch["moe"]["train_residual_anchor_expert"]
    very_wide_patch = candidates["trainresid_seg12_p168_mse800_s321"].patch
    very_wide_expert = very_wide_patch["moe"]["train_residual_anchor_expert"]
    extreme_patch = candidates["trainresid_seg12_p168_mse2000_s801"].patch
    extreme_expert = extreme_patch["moe"]["train_residual_anchor_expert"]
    daily_patch = candidates["trainresid_seg12_p24_mse120_s49"].patch
    p96_patch = candidates["trainresid_seg12_p96_mse120_s49"].patch

    assert expert["enable"] is True
    assert expert["period"] == 168
    assert expert["alpha"] == 0.0
    assert expert["scale_selection"] == {
        "enable": True,
        "metric": "mse",
        "max_scale": 0.8,
        "steps": 33,
        "horizon_segments": 12,
    }
    assert patch["moe"]["history_anchor_expert"]["enable"] is False
    assert patch["moe"]["train_stat_anchor_expert"]["enable"] is False
    assert patch["moe"]["pred_side_residual"]["enable"] is False
    assert wide_expert["period"] == 168
    assert wide_expert["scale_selection"]["max_scale"] == 2.0
    assert wide_expert["scale_selection"]["steps"] == 81
    assert very_wide_expert["scale_selection"]["max_scale"] == 8.0
    assert very_wide_expert["scale_selection"]["steps"] == 321
    assert extreme_expert["scale_selection"]["max_scale"] == 20.0
    assert extreme_expert["scale_selection"]["steps"] == 801
    assert daily_patch["moe"]["train_residual_anchor_expert"]["period"] == 24
    assert daily_patch["moe"]["train_residual_anchor_expert"]["scale_selection"]["max_scale"] == 1.2
    assert p96_patch["moe"]["train_residual_anchor_expert"]["period"] == 96


def test_activation_candidates_include_train_stat_residual_combo() -> None:
    candidates = {cand.variant: cand for cand in activation_candidates({"penalties": {"enabled": ["level", "delta"]}})}

    patch = candidates["trainstatresid_mean_p96_stat030_resid050"].patch
    stat = patch["moe"]["train_stat_anchor_expert"]
    residual = patch["moe"]["train_residual_anchor_expert"]

    assert stat["enable"] is True
    assert stat["scale_selection"] == {
        "enable": True,
        "metric": "mse",
        "max_scale": 0.3,
        "steps": 13,
    }
    assert residual["enable"] is True
    assert residual["scale_selection"] == {
        "enable": True,
        "metric": "mse",
        "max_scale": 0.5,
        "steps": 21,
    }
    assert patch["moe"]["history_anchor_expert"]["enable"] is False


def test_activation_candidates_include_wider_train_stat_residual_combo() -> None:
    candidates = {cand.variant: cand for cand in activation_candidates({"penalties": {"enabled": ["level", "delta"]}})}

    patch = candidates["trainstatresid_mean_p96_stat030_resid080_seg7"].patch
    stat = patch["moe"]["train_stat_anchor_expert"]
    residual = patch["moe"]["train_residual_anchor_expert"]

    assert stat["scale_selection"] == {
        "enable": True,
        "metric": "mse",
        "max_scale": 0.3,
        "steps": 13,
    }
    assert residual["scale_selection"] == {
        "enable": True,
        "metric": "mse",
        "max_scale": 0.8,
        "steps": 33,
        "horizon_segments": 7,
    }
    assert patch["moe"]["history_anchor_expert"]["enable"] is False


def test_activation_candidates_include_etth1_h192_fine_train_stat_residual_combo() -> None:
    candidates = {cand.variant: cand for cand in activation_candidates({"penalties": {"enabled": ["level", "delta"]}})}

    stat018 = candidates["trainstatresid_mean_p96_stat018_resid080_seg4"].patch
    stat0225 = candidates["trainstatresid_mean_p96_stat0225_resid080_seg4"].patch
    resid090 = candidates["trainstatresid_mean_p96_stat020_resid090_seg4"].patch
    seg7 = candidates["trainstatresid_mean_p96_stat020_resid080_seg7"].patch
    resid085_seg7 = candidates["trainstatresid_mean_p96_stat020_resid085_seg7"].patch
    resid090_seg7 = candidates["trainstatresid_mean_p96_stat020_resid090_seg7"].patch
    seg12 = candidates["trainstatresid_mean_p96_stat020_resid080_seg12"].patch
    statseg4 = candidates["trainstatresid_mean_p96_stat020seg4_resid080_seg12"].patch
    statseg7 = candidates["trainstatresid_mean_p96_stat020seg7_resid080_seg12"].patch
    statseg7_resid090 = candidates["trainstatresid_mean_p96_stat020seg7_resid090_seg12"].patch
    statseg12 = candidates["trainstatresid_mean_p96_stat020seg12_resid080_seg12"].patch

    assert stat018["moe"]["train_stat_anchor_expert"]["scale_selection"] == {
        "enable": True,
        "metric": "mse",
        "max_scale": 0.18,
        "steps": 10,
    }
    assert stat0225["moe"]["train_stat_anchor_expert"]["scale_selection"]["max_scale"] == 0.225
    assert stat0225["moe"]["train_stat_anchor_expert"]["scale_selection"]["steps"] == 10
    assert resid090["moe"]["train_residual_anchor_expert"]["scale_selection"] == {
        "enable": True,
        "metric": "mse",
        "max_scale": 0.9,
        "steps": 37,
        "horizon_segments": 4,
    }
    assert seg7["moe"]["train_residual_anchor_expert"]["scale_selection"]["horizon_segments"] == 7
    assert resid085_seg7["moe"]["train_residual_anchor_expert"]["scale_selection"]["max_scale"] == 0.85
    assert resid085_seg7["moe"]["train_residual_anchor_expert"]["scale_selection"]["steps"] == 35
    assert resid090_seg7["moe"]["train_residual_anchor_expert"]["scale_selection"]["horizon_segments"] == 7
    assert seg12["moe"]["train_residual_anchor_expert"]["scale_selection"]["horizon_segments"] == 12
    assert statseg4["moe"]["train_stat_anchor_expert"]["scale_selection"]["horizon_segments"] == 4
    assert statseg4["moe"]["train_residual_anchor_expert"]["scale_selection"]["horizon_segments"] == 12
    assert statseg7["moe"]["train_stat_anchor_expert"]["scale_selection"]["horizon_segments"] == 7
    assert statseg7_resid090["moe"]["train_stat_anchor_expert"]["scale_selection"]["horizon_segments"] == 7
    assert statseg7_resid090["moe"]["train_residual_anchor_expert"]["scale_selection"]["max_scale"] == 0.9
    assert statseg12["moe"]["train_stat_anchor_expert"]["scale_selection"]["horizon_segments"] == 12


def test_activation_candidates_include_etth1_h336_fine_train_stat_residual_combo() -> None:
    candidates = {cand.variant: cand for cand in activation_candidates({"penalties": {"enabled": ["level", "delta"]}})}

    stat025 = candidates["trainstatresid_mean_p96_stat025_resid080_seg7"].patch
    stat0225 = candidates["trainstatresid_mean_p96_stat0225_resid080_seg7"].patch
    residual090 = candidates["trainstatresid_mean_p96_stat030_resid090_seg7"].patch
    stat025_residual090 = candidates["trainstatresid_mean_p96_stat025_resid090_seg7"].patch
    stat025_residual110 = candidates["trainstatresid_mean_p96_stat025_resid110_seg7"].patch
    stat018_residual120 = candidates["trainstatresid_mean_p96_stat018_resid120_seg7"].patch
    stat0225_residual120 = candidates["trainstatresid_mean_p96_stat0225_resid120_seg7"].patch
    stat020_residual140 = candidates["trainstatresid_mean_p96_stat020_resid140_seg7"].patch
    stat020_residual160 = candidates["trainstatresid_mean_p96_stat020_resid160_seg7"].patch
    stat020_residual180 = candidates["trainstatresid_mean_p96_stat020_resid180_seg7"].patch
    stat020_residual200 = candidates["trainstatresid_mean_p96_stat020_resid200_seg7"].patch
    stat020_residual220 = candidates["trainstatresid_mean_p96_stat020_resid220_seg7"].patch
    stat020_residual240 = candidates["trainstatresid_mean_p96_stat020_resid240_seg7"].patch
    stat020_residual260 = candidates["trainstatresid_mean_p96_stat020_resid260_seg7"].patch
    stat020_residual280 = candidates["trainstatresid_mean_p96_stat020_resid280_seg7"].patch
    stat020_residual200_residmae = candidates["trainstatresid_mean_p96_stat020_resid200_residmae_seg7"].patch
    stat020_residual260_residmae = candidates["trainstatresid_mean_p96_stat020_resid260_residmae_seg7"].patch
    stat020_residual200_mae = candidates["trainstatresid_mean_p96_stat020_resid200_mae_seg7"].patch
    stat020_residual260_mae = candidates["trainstatresid_mean_p96_stat020_resid260_mae_seg7"].patch
    stat020_residual120_seg12 = candidates["trainstatresid_mean_p96_stat020_resid120_seg12"].patch
    stat020_residual120_seg16 = candidates["trainstatresid_mean_p96_stat020_resid120_seg16"].patch
    stat020seg7_residual120_seg12 = candidates["trainstatresid_mean_p96_stat020seg7_resid120_seg12"].patch
    stat020seg7_residual120_seg16 = candidates["trainstatresid_mean_p96_stat020seg7_resid120_seg16"].patch
    stat018seg7_residual120_seg16 = candidates["trainstatresid_mean_p96_stat018seg7_resid120_seg16"].patch
    stat0225seg7_residual120_seg16 = candidates["trainstatresid_mean_p96_stat0225seg7_resid120_seg16"].patch
    stat025seg7_residual120_seg16 = candidates["trainstatresid_mean_p96_stat025seg7_resid120_seg16"].patch
    stat0275seg7_residual120_seg16 = candidates["trainstatresid_mean_p96_stat0275seg7_resid120_seg16"].patch
    stat030seg7_residual120_seg16 = candidates["trainstatresid_mean_p96_stat030seg7_resid120_seg16"].patch
    stat035seg7_residual120_seg16 = candidates["trainstatresid_mean_p96_stat035seg7_resid120_seg16"].patch
    stat040seg7_residual120_seg16 = candidates["trainstatresid_mean_p96_stat040seg7_resid120_seg16"].patch
    stat045seg7_residual120_seg16 = candidates["trainstatresid_mean_p96_stat045seg7_resid120_seg16"].patch
    stat030seg7_residual140_seg16 = candidates["trainstatresid_mean_p96_stat030seg7_resid140_seg16"].patch
    stat020seg7_residual140_seg16 = candidates["trainstatresid_mean_p96_stat020seg7_resid140_seg16"].patch
    stat020seg12_residual120_seg12 = candidates["trainstatresid_mean_p96_stat020seg12_resid120_seg12"].patch
    p24_stat020_residual120 = candidates["trainstatresid_mean_p24_stat020_resid120_seg12"].patch
    p24_stat025_residual100 = candidates["trainstatresid_mean_p24_stat025_resid100_seg12"].patch
    p24_stat025_residual140 = candidates["trainstatresid_mean_p24_stat025_resid140_seg12"].patch
    p24_stat025_residual160 = candidates["trainstatresid_mean_p24_stat025_resid160_seg12"].patch
    p24_stat025_residual180 = candidates["trainstatresid_mean_p24_stat025_resid180_seg12"].patch
    p24_stat025_residual200 = candidates["trainstatresid_mean_p24_stat025_resid200_seg12"].patch
    p24_stat025_residual220 = candidates["trainstatresid_mean_p24_stat025_resid220_seg12"].patch
    p24_stat018_residual180 = candidates["trainstatresid_mean_p24_stat018_resid180_seg12"].patch
    p24_stat019_residual180 = candidates["trainstatresid_mean_p24_stat019_resid180_seg12"].patch
    p24_stat020_residual160 = candidates["trainstatresid_mean_p24_stat020_resid160_seg12"].patch
    p24_stat020_residual180 = candidates["trainstatresid_mean_p24_stat020_resid180_seg12"].patch
    p24_stat020_residual200 = candidates["trainstatresid_mean_p24_stat020_resid200_seg12"].patch
    p24_stat02125_residual180 = candidates["trainstatresid_mean_p24_stat02125_resid180_seg12"].patch
    p24_stat0225_residual180 = candidates["trainstatresid_mean_p24_stat0225_resid180_seg12"].patch
    p24_stat02375_residual180 = candidates["trainstatresid_mean_p24_stat02375_resid180_seg12"].patch
    p24_stat0225_residual160 = candidates["trainstatresid_mean_p24_stat0225_resid160_seg12"].patch
    p24_stat0225_residual200 = candidates["trainstatresid_mean_p24_stat0225_resid200_seg12"].patch
    p24_stat0275_residual180 = candidates["trainstatresid_mean_p24_stat0275_resid180_seg12"].patch
    p24_stat025seg7_residual180 = candidates["trainstatresid_mean_p24_stat025seg7_resid180_seg12"].patch
    p24_stat025_residual180_seg16 = candidates["trainstatresid_mean_p24_stat025_resid180_seg16"].patch
    p24_stat030_residual100 = candidates["trainstatresid_mean_p24_stat030_resid100_seg12"].patch
    p24_stat030_residual120 = candidates["trainstatresid_mean_p24_stat030_resid120_seg12"].patch
    p24_stat030_residual140 = candidates["trainstatresid_mean_p24_stat030_resid140_seg12"].patch
    p24_stat035_residual100 = candidates["trainstatresid_mean_p24_stat035_resid100_seg12"].patch
    p24_stat035_residual120 = candidates["trainstatresid_mean_p24_stat035_resid120_seg12"].patch
    p24_stat030_residual120_seg16 = candidates["trainstatresid_mean_p24_stat030_resid120_seg16"].patch
    p24_stat030_residual120_mae = candidates["trainstatresid_mean_p24_stat030_resid120_mae_seg12"].patch
    p24_stat030_residual120_residmae = candidates[
        "trainstatresid_mean_p24_stat030_resid120_residmae_seg12"
    ].patch
    p24_stat025_residual120_seg7 = candidates["trainstatresid_mean_p24_stat025_resid120_seg7"].patch
    p168_stat020_residual120 = candidates["trainstatresid_mean_p168_stat020_resid120_seg12"].patch
    seg12 = candidates["trainstatresid_mean_p96_stat030_resid080_seg12"].patch

    assert stat025["moe"]["train_stat_anchor_expert"]["scale_selection"] == {
        "enable": True,
        "metric": "mse",
        "max_scale": 0.25,
        "steps": 11,
    }
    assert stat0225["moe"]["train_stat_anchor_expert"]["scale_selection"]["max_scale"] == 0.225
    assert stat0225["moe"]["train_stat_anchor_expert"]["scale_selection"]["steps"] == 10
    assert residual090["moe"]["train_residual_anchor_expert"]["scale_selection"] == {
        "enable": True,
        "metric": "mse",
        "max_scale": 0.9,
        "steps": 37,
        "horizon_segments": 7,
    }
    assert stat025_residual090["moe"]["train_stat_anchor_expert"]["scale_selection"]["max_scale"] == 0.25
    assert stat025_residual090["moe"]["train_residual_anchor_expert"]["scale_selection"]["max_scale"] == 0.9
    assert stat025_residual110["moe"]["train_residual_anchor_expert"]["scale_selection"] == {
        "enable": True,
        "metric": "mse",
        "max_scale": 1.1,
        "steps": 45,
        "horizon_segments": 7,
    }
    assert stat018_residual120["moe"]["train_stat_anchor_expert"]["scale_selection"]["max_scale"] == 0.18
    assert stat018_residual120["moe"]["train_residual_anchor_expert"]["scale_selection"]["max_scale"] == 1.2
    assert stat0225_residual120["moe"]["train_stat_anchor_expert"]["scale_selection"]["max_scale"] == 0.225
    assert stat020_residual140["moe"]["train_residual_anchor_expert"]["scale_selection"]["max_scale"] == 1.4
    assert stat020_residual140["moe"]["train_residual_anchor_expert"]["scale_selection"]["steps"] == 57
    assert stat020_residual160["moe"]["train_residual_anchor_expert"]["scale_selection"]["max_scale"] == 1.6
    assert stat020_residual160["moe"]["train_residual_anchor_expert"]["scale_selection"]["steps"] == 65
    assert stat020_residual180["moe"]["train_residual_anchor_expert"]["scale_selection"]["max_scale"] == 1.8
    assert stat020_residual180["moe"]["train_residual_anchor_expert"]["scale_selection"]["steps"] == 73
    assert stat020_residual200["moe"]["train_residual_anchor_expert"]["scale_selection"]["max_scale"] == 2.0
    assert stat020_residual200["moe"]["train_residual_anchor_expert"]["scale_selection"]["steps"] == 81
    assert stat020_residual220["moe"]["train_residual_anchor_expert"]["scale_selection"]["max_scale"] == 2.2
    assert stat020_residual220["moe"]["train_residual_anchor_expert"]["scale_selection"]["steps"] == 89
    assert stat020_residual240["moe"]["train_residual_anchor_expert"]["scale_selection"]["max_scale"] == 2.4
    assert stat020_residual240["moe"]["train_residual_anchor_expert"]["scale_selection"]["steps"] == 97
    assert stat020_residual260["moe"]["train_residual_anchor_expert"]["scale_selection"]["max_scale"] == 2.6
    assert stat020_residual260["moe"]["train_residual_anchor_expert"]["scale_selection"]["steps"] == 105
    assert stat020_residual280["moe"]["train_residual_anchor_expert"]["scale_selection"]["max_scale"] == 2.8
    assert stat020_residual280["moe"]["train_residual_anchor_expert"]["scale_selection"]["steps"] == 113
    assert stat020_residual200_residmae["moe"]["train_stat_anchor_expert"]["scale_selection"]["metric"] == "mse"
    assert stat020_residual200_residmae["moe"]["train_residual_anchor_expert"]["scale_selection"]["metric"] == "mae"
    assert stat020_residual260_residmae["moe"]["train_residual_anchor_expert"]["scale_selection"]["max_scale"] == 2.6
    assert stat020_residual260_residmae["moe"]["train_residual_anchor_expert"]["scale_selection"]["metric"] == "mae"
    assert stat020_residual200_mae["moe"]["train_stat_anchor_expert"]["scale_selection"]["metric"] == "mae"
    assert stat020_residual200_mae["moe"]["train_residual_anchor_expert"]["scale_selection"]["metric"] == "mae"
    assert stat020_residual260_mae["moe"]["train_stat_anchor_expert"]["scale_selection"]["metric"] == "mae"
    assert stat020_residual260_mae["moe"]["train_residual_anchor_expert"]["scale_selection"]["max_scale"] == 2.6
    assert stat020_residual120_seg12["moe"]["train_residual_anchor_expert"]["scale_selection"]["horizon_segments"] == 12
    assert stat020_residual120_seg16["moe"]["train_residual_anchor_expert"]["scale_selection"]["horizon_segments"] == 16
    assert stat020seg7_residual120_seg12["moe"]["train_stat_anchor_expert"]["scale_selection"]["horizon_segments"] == 7
    assert stat020seg7_residual120_seg12["moe"]["train_residual_anchor_expert"]["scale_selection"]["horizon_segments"] == 12
    assert stat020seg7_residual120_seg16["moe"]["train_stat_anchor_expert"]["scale_selection"]["horizon_segments"] == 7
    assert stat020seg7_residual120_seg16["moe"]["train_residual_anchor_expert"]["scale_selection"]["horizon_segments"] == 16
    assert stat018seg7_residual120_seg16["moe"]["train_stat_anchor_expert"]["scale_selection"]["max_scale"] == 0.18
    assert stat0225seg7_residual120_seg16["moe"]["train_stat_anchor_expert"]["scale_selection"]["max_scale"] == 0.225
    assert stat025seg7_residual120_seg16["moe"]["train_stat_anchor_expert"]["scale_selection"]["max_scale"] == 0.25
    assert stat0275seg7_residual120_seg16["moe"]["train_stat_anchor_expert"]["scale_selection"]["max_scale"] == 0.275
    assert stat030seg7_residual120_seg16["moe"]["train_stat_anchor_expert"]["scale_selection"]["max_scale"] == 0.3
    assert stat035seg7_residual120_seg16["moe"]["train_stat_anchor_expert"]["scale_selection"]["max_scale"] == 0.35
    assert stat040seg7_residual120_seg16["moe"]["train_stat_anchor_expert"]["scale_selection"]["max_scale"] == 0.4
    assert stat045seg7_residual120_seg16["moe"]["train_stat_anchor_expert"]["scale_selection"]["max_scale"] == 0.45
    assert stat030seg7_residual140_seg16["moe"]["train_residual_anchor_expert"]["scale_selection"]["max_scale"] == 1.4
    assert stat020seg7_residual140_seg16["moe"]["train_residual_anchor_expert"]["scale_selection"]["max_scale"] == 1.4
    assert stat020seg12_residual120_seg12["moe"]["train_stat_anchor_expert"]["scale_selection"]["horizon_segments"] == 12
    assert p24_stat020_residual120["moe"]["train_stat_anchor_expert"]["period"] == 24
    assert p24_stat020_residual120["moe"]["train_residual_anchor_expert"]["period"] == 24
    assert p24_stat020_residual120["moe"]["train_residual_anchor_expert"]["scale_selection"]["horizon_segments"] == 12
    assert p24_stat025_residual100["moe"]["train_residual_anchor_expert"]["scale_selection"]["max_scale"] == 1.0
    assert p24_stat025_residual140["moe"]["train_residual_anchor_expert"]["scale_selection"]["max_scale"] == 1.4
    assert p24_stat025_residual160["moe"]["train_residual_anchor_expert"]["scale_selection"]["max_scale"] == 1.6
    assert p24_stat025_residual180["moe"]["train_residual_anchor_expert"]["scale_selection"]["max_scale"] == 1.8
    assert p24_stat025_residual200["moe"]["train_residual_anchor_expert"]["scale_selection"]["max_scale"] == 2.0
    assert p24_stat025_residual220["moe"]["train_residual_anchor_expert"]["scale_selection"]["max_scale"] == 2.2
    assert p24_stat018_residual180["moe"]["train_stat_anchor_expert"]["scale_selection"]["max_scale"] == 0.18
    assert p24_stat019_residual180["moe"]["train_stat_anchor_expert"]["scale_selection"]["max_scale"] == 0.19
    assert p24_stat020_residual160["moe"]["train_residual_anchor_expert"]["scale_selection"]["max_scale"] == 1.6
    assert p24_stat020_residual180["moe"]["train_stat_anchor_expert"]["scale_selection"]["max_scale"] == 0.2
    assert p24_stat020_residual200["moe"]["train_residual_anchor_expert"]["scale_selection"]["max_scale"] == 2.0
    assert p24_stat02125_residual180["moe"]["train_stat_anchor_expert"]["scale_selection"]["max_scale"] == 0.2125
    assert p24_stat0225_residual180["moe"]["train_stat_anchor_expert"]["scale_selection"]["max_scale"] == 0.225
    assert p24_stat02375_residual180["moe"]["train_stat_anchor_expert"]["scale_selection"]["max_scale"] == 0.2375
    assert p24_stat0225_residual160["moe"]["train_residual_anchor_expert"]["scale_selection"]["max_scale"] == 1.6
    assert p24_stat0225_residual200["moe"]["train_residual_anchor_expert"]["scale_selection"]["max_scale"] == 2.0
    assert p24_stat0275_residual180["moe"]["train_stat_anchor_expert"]["scale_selection"]["max_scale"] == 0.275
    assert p24_stat025seg7_residual180["moe"]["train_stat_anchor_expert"]["scale_selection"]["horizon_segments"] == 7
    assert p24_stat025_residual180_seg16["moe"]["train_residual_anchor_expert"]["scale_selection"]["horizon_segments"] == 16
    assert candidates["trainstatresid_mean_p24_stat030_resid080_seg12"].patch["moe"]["train_residual_anchor_expert"][
        "scale_selection"
    ]["max_scale"] == 0.8
    assert candidates["trainstatresid_mean_p24_stat030_resid090_seg12"].patch["moe"]["train_residual_anchor_expert"][
        "scale_selection"
    ]["max_scale"] == 0.9
    assert candidates["trainstatresid_mean_p24_stat02625_resid100_seg12"].patch["moe"]["train_stat_anchor_expert"][
        "scale_selection"
    ]["max_scale"] == 0.2625
    assert candidates["trainstatresid_mean_p24_stat0275_resid080_seg12"].patch["moe"]["train_residual_anchor_expert"][
        "scale_selection"
    ]["max_scale"] == 0.8
    assert candidates["trainstatresid_mean_p24_stat0275_resid090_seg12"].patch["moe"]["train_residual_anchor_expert"][
        "scale_selection"
    ]["max_scale"] == 0.9
    assert candidates["trainstatresid_mean_p24_stat0275_resid100_seg12"].patch["moe"]["train_stat_anchor_expert"][
        "scale_selection"
    ]["max_scale"] == 0.275
    assert candidates["trainstatresid_mean_p24_stat0275_resid110_seg12"].patch["moe"]["train_residual_anchor_expert"][
        "scale_selection"
    ]["max_scale"] == 1.1
    assert candidates["trainstatresid_mean_p24_stat02875_resid100_seg12"].patch["moe"]["train_stat_anchor_expert"][
        "scale_selection"
    ]["max_scale"] == 0.2875
    assert candidates["trainstatresid_mean_p24_stat02875_resid090_seg12"].patch["moe"][
        "train_residual_anchor_expert"
    ]["scale_selection"]["max_scale"] == 0.9
    assert candidates["trainstatresid_mean_p24_stat02875_resid110_seg12"].patch["moe"][
        "train_residual_anchor_expert"
    ]["scale_selection"]["max_scale"] == 1.1
    assert candidates["trainstatresid_mean_p24_stat029_resid100_seg12"].patch["moe"]["train_stat_anchor_expert"][
        "scale_selection"
    ]["max_scale"] == 0.29
    assert candidates["trainstatresid_mean_p24_stat0295_resid100_seg12"].patch["moe"]["train_stat_anchor_expert"][
        "scale_selection"
    ]["max_scale"] == 0.295
    assert p24_stat030_residual100["moe"]["train_residual_anchor_expert"]["scale_selection"]["max_scale"] == 1.0
    assert candidates["trainstatresid_mean_p24_stat0325_resid100_seg12"].patch["moe"]["train_stat_anchor_expert"][
        "scale_selection"
    ]["max_scale"] == 0.325
    assert p24_stat030_residual120["moe"]["train_stat_anchor_expert"]["scale_selection"]["max_scale"] == 0.3
    assert p24_stat030_residual140["moe"]["train_residual_anchor_expert"]["scale_selection"]["max_scale"] == 1.4
    assert p24_stat035_residual100["moe"]["train_stat_anchor_expert"]["scale_selection"]["max_scale"] == 0.35
    assert p24_stat035_residual120["moe"]["train_stat_anchor_expert"]["scale_selection"]["max_scale"] == 0.35
    assert p24_stat030_residual120_seg16["moe"]["train_residual_anchor_expert"]["scale_selection"]["horizon_segments"] == 16
    assert p24_stat030_residual120_mae["moe"]["train_stat_anchor_expert"]["scale_selection"]["metric"] == "mae"
    assert p24_stat030_residual120_mae["moe"]["train_residual_anchor_expert"]["scale_selection"]["metric"] == "mae"
    assert p24_stat030_residual120_residmae["moe"]["train_stat_anchor_expert"]["scale_selection"]["metric"] == "mse"
    assert p24_stat030_residual120_residmae["moe"]["train_residual_anchor_expert"]["scale_selection"]["metric"] == "mae"
    assert p24_stat025_residual120_seg7["moe"]["train_residual_anchor_expert"]["scale_selection"]["horizon_segments"] == 7
    assert p168_stat020_residual120["moe"]["train_stat_anchor_expert"]["period"] == 168
    assert p168_stat020_residual120["moe"]["train_residual_anchor_expert"]["period"] == 168
    assert seg12["moe"]["train_residual_anchor_expert"]["scale_selection"]["horizon_segments"] == 12


def test_activation_candidates_include_boundary_train_stat_residual_combo() -> None:
    candidates = {cand.variant: cand for cand in activation_candidates({"penalties": {"enabled": ["level", "delta"]}})}

    patch = candidates["trainstatresid_mean_p96_stat020_resid120_seg4"].patch
    stat = patch["moe"]["train_stat_anchor_expert"]
    residual = patch["moe"]["train_residual_anchor_expert"]

    assert stat["scale_selection"]["max_scale"] == 0.2
    assert stat["scale_selection"]["steps"] == 9
    assert residual["scale_selection"] == {
        "enable": True,
        "metric": "mse",
        "max_scale": 1.2,
        "steps": 49,
        "horizon_segments": 4,
    }
    assert patch["moe"]["history_anchor_expert"]["enable"] is False


def test_dataset_configs_include_pems_input96_targets() -> None:
    assert DATASET_CONFIGS["PEMS03"] == "configs/PEMS03_H12.yaml"
    assert DATASET_CONFIGS["PEMS08"] == "configs/PEMS08_H12.yaml"


def test_activation_candidates_include_daily_train_stat_residual_combo() -> None:
    candidates = {cand.variant: cand for cand in activation_candidates({"penalties": {"enabled": ["level", "delta"]}})}

    patch = candidates["trainstatresid_mean_p288_stat020_resid120_seg4"].patch
    stat = patch["moe"]["train_stat_anchor_expert"]
    residual = patch["moe"]["train_residual_anchor_expert"]

    assert stat["period"] == 288
    assert residual["period"] == 288
    assert residual["scale_selection"]["horizon_segments"] == 4

    stat050 = candidates["trainstatresid_mean_p288_stat050_resid200_seg4"].patch
    assert stat050["moe"]["train_stat_anchor_expert"]["scale_selection"]["max_scale"] == 0.5
    assert stat050["moe"]["train_stat_anchor_expert"]["scale_selection"]["steps"] == 21
    assert stat050["moe"]["train_residual_anchor_expert"]["scale_selection"]["max_scale"] == 2.0
    assert stat050["moe"]["train_residual_anchor_expert"]["scale_selection"]["horizon_segments"] == 4

    resid240 = candidates["trainstatresid_mean_p288_stat040_resid240_seg4"].patch
    assert resid240["moe"]["train_stat_anchor_expert"]["scale_selection"]["max_scale"] == 0.4
    assert resid240["moe"]["train_residual_anchor_expert"]["scale_selection"]["max_scale"] == 2.4
    assert resid240["moe"]["train_residual_anchor_expert"]["scale_selection"]["steps"] == 97

    mae_metric = candidates["trainstatresid_mean_p288_stat050_resid200_mae_seg4"].patch
    assert mae_metric["moe"]["train_stat_anchor_expert"]["scale_selection"]["metric"] == "mae"
    assert mae_metric["moe"]["train_residual_anchor_expert"]["scale_selection"]["metric"] == "mae"

    residual_mae = candidates["trainstatresid_mean_p288_stat050_resid200_residmae_seg4"].patch
    assert residual_mae["moe"]["train_stat_anchor_expert"]["scale_selection"]["metric"] == "mse"
    assert residual_mae["moe"]["train_residual_anchor_expert"]["scale_selection"]["metric"] == "mae"
