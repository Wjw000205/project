from __future__ import annotations

import argparse
import copy
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def deep_update(dst: dict[str, Any], src: dict[str, Any]) -> dict[str, Any]:
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            deep_update(dst[key], value)
        else:
            dst[key] = copy.deepcopy(value)
    return dst


def posix_path(path: Path) -> str:
    return str(path).replace("\\", "/")


def base_common(base: dict[str, Any], out_root: Path, label: str, exp_name: str, device: str) -> dict[str, Any]:
    run_dir = out_root / "runs" / label
    cfg = copy.deepcopy(base)
    deep_update(
        cfg,
        {
            "exp": {
                "name": exp_name,
                "out_dir": posix_path(run_dir.relative_to(ROOT)),
                "seed": 2026,
                "deterministic": True,
                "device": device,
            },
            "data": {
                "csv_path": "data/ETTh2.csv",
                "date_col": 0,
                "max_rows": 14400,
                "train_ratio": 0.6,
                "val_ratio": 0.2,
                "test_ratio": 0.2,
            },
            "window": {"input_len": 336, "pred_len": 720, "past_context": True},
            "normalize": {"global_zscore": True, "train_only": True},
            "corr": {"compute": True, "save_path": posix_path((run_dir / "corr.npy").relative_to(ROOT))},
            "cluster": {
                "method": "leader",
                "n_clusters": 3,
                "linkage": "average",
                "kmeans_n_init": 10,
                "kmeans_max_iter": 300,
                "spectral_affinity": "corr",
                "rbf_gamma": 1.0,
                "dbscan_eps": 0.7,
                "dbscan_min_samples": 2,
                "random_state": 2026,
                "min_cluster_size": 2,
                "no_merge_if_channels_lt": 7,
                "train_only": True,
                "feature_aware": {"enable": False, "weight": 0.0, "acf_lags": [1, 24, 96]},
                "singleton_merge_strategy": "pool",
            },
            "model": {
                "context_channel_head_include_delta": True,
                "channel_head_residual": True,
                "context_channel_head_residual": True,
            },
            "moe": {
                "topk": 1,
                "enable": True,
                "freeze_lambda": False,
                "gate_hidden_dim": 32,
                "select_ranks": [1],
                "detach_penalty_grad": False,
                "pred_side_residual": {
                    "enable": True,
                    "init_alpha": -3.0,
                    "specialization_weight": 0.05,
                    "norm_weight": 0.0001,
                    "use_y_base_input": True,
                    "intervention_enable": False,
                    "intervention_init": -2.0,
                    "intervention_weight": 0.001,
                    "detach_routed_penalty_pred": False,
                    "selection_min_abs_improvement": 0.0,
                    "selection_min_rel_improvement": 0.0,
                    "penalty_guard": {
                        "enable": False,
                        "metric": "mse",
                        "allow_multi": True,
                        "min_abs_improvement": 0.0,
                        "min_rel_improvement": 0.0,
                    },
                    "channel_guard": {"enable": False},
                    "validation_guard": {
                        "enable": False,
                        "select_fraction": 0.5,
                        "min_abs_improvement": 0.0,
                        "min_rel_improvement": 0.002,
                    },
                    "diagnostics": {"enable": True},
                    "residual_clip": 4.0,
                    "selection_policy": "val_mse_candidate_channel",
                    "penalty_selector_enable": True,
                    "selector_temperature": 1.0,
                    "selector_use_cluster_context": True,
                    "fusion_gate_enable": True,
                    "fusion_init": -0.5,
                    "fusion_use_cluster_context": True,
                    "channel_expert_adapters": {
                        "enable": True,
                        "mode": "merged_singletons",
                        "mode_type": "override",
                    },
                },
                "dynamic_lambda": {
                    "enable": False,
                    "mode": "multiscale",
                    "hidden_dim": 32,
                    "segment_bins": [4, 8],
                    "max_factor": 1.5,
                    "mix": 0.6,
                    "dropout": 0.0,
                    "reg_weight": 0.0001,
                },
                "learnable_lambda": {"enable": False},
                "gate_entropy_weight": 0.0,
                "gate_balance_weight": 0.0,
                "gate_route_on_penalty_only": True,
                "router_mode": "learned",
                "router_penalty_context_weight": 0.0,
                "router_detach_penalty_context": True,
                "allow_skip": True,
                "skip_init_bias": -2.0,
                "gate_soft_weight": 0.0,
                "gate_prob_floor": 0.0,
                "gate_entropy_target_frac": 0.7,
                "gate_logit_clip": 5.0,
                "gate_init_bias": {"enable": True, "values": {"default": 0.0}},
                "explainability": {"enable": True, "splits": ["train", "val", "test"], "max_batches": 0},
                "cluster_penalty_prior": {
                    "enable": False,
                    "topk": 0,
                    "hard_topk": True,
                    "temperature": 1.0,
                    "smoothing": 0.02,
                    "use_normalized_penalty": True,
                    "use_as_balance_target": False,
                },
                "channel_penalty_prior": {
                    "enable": False,
                    "topk": 1,
                    "hard_topk": True,
                    "temperature": 1.0,
                    "smoothing": 0.02,
                    "use_normalized_penalty": True,
                },
            },
            "train": {
                "epochs": 50,
                "batch_size": 96,
                "mse_weight": 0.9,
                "mae_objective": {"enable": True, "kind": "l1", "weight": 0.4, "warmup_epochs": 5},
                "selection_metric": "val_mse",
                "grad_clip": 1.0,
                "lr_scheduler": {"name": "plateau", "factor": 0.5, "patience": 3, "min_lr": 1.0e-6},
                "penalty_warmup_epochs": 0,
            },
            "early_stop": {"patience": 10, "min_delta": 1.0e-6},
            "plot": {"enable": False},
            "portrait": {"enable": False, "out_dir": posix_path((run_dir / "cluster_portraits").relative_to(ROOT))},
            "eval": {"skip_test": False},
            "memory": {
                "enable": False,
                "save_checkpoint": False,
                "path": posix_path((run_dir / "cluster_memory.pt").relative_to(ROOT)),
                "checkpoint_path": posix_path((run_dir / "best_checkpoint.pt").relative_to(ROOT)),
            },
        },
    )
    return cfg


def trial_configs(base: dict[str, Any], out_root: Path, device: str) -> list[tuple[str, dict[str, Any]]]:
    trials: list[tuple[str, dict[str, Any]]] = []

    cfg = base_common(
        base,
        out_root,
        "trial_0010_trend_dir",
        "ETTh2_H720_trend_dir_channelheadmlp_h64_do0p276_l0p0213_amp_heavy_lr0p00073_wd7em05_wu0_bs96_dt0p75_fa0p35_gt1p4_gn0p15_sk0p28_pk0_ps2p2_gated_ra0p87_legacy",
        device,
    )
    deep_update(
        cfg,
        {
            "cluster": {
                "distance_threshold": 0.7461676871187886,
                "merge_small_clusters": True,
                "feature_aware": {"enable": True, "weight": 0.35, "acf_lags": [1, 24, 96]},
            },
            "model": {"predictor": "channel_head_mlp", "hidden_dim": 64, "dropout": 0.27607278652090145},
            "moe": {
                "lambda_init": {
                    "delta": 0.02134689470223942,
                    "trend": 0.012808136821343652,
                    "direction": 0.02134689470223942,
                },
                "lambda_min": {"delta": 0.0, "trend": 0.0, "direction": 0.0},
                "lambda_schedule": {"delta": "none", "trend": "none", "direction": "none"},
                "skip_cost": 0.28198219425146365,
                "gate_temperature": 1.434267094160957,
                "gate_noise_std": 0.15477095616662198,
                "pred_side_residual": {
                    "corrector_hidden": 16,
                    "alpha_scale": 0.869281322911342,
                    "feature_mode": "legacy",
                },
                "cluster_penalty_prior": {"logit_strength": 2.2354268676582865},
            },
            "penalties": {"enabled": ["delta", "trend", "direction"], "jump_threshold": 0.6},
            "train": {"lr": 0.0007321919161979784, "weight_decay": 7.290413667211864e-05},
        },
    )
    trials.append(("trial_0010", cfg))

    cfg = base_common(
        base,
        out_root,
        "trial_0015_lddf",
        "ETTh2_H720_lddf_channelheadmlp_h64_do0p184_l0p0535_amp_heavy_lr0p00038_wd0p0005_wu0_bs96_dt0p78_fa0_gt1p2_gn0p19_sk0p18_pk0_ps0p74_gated_ra0p28_safe_augmented",
        device,
    )
    deep_update(
        cfg,
        {
            "cluster": {
                "distance_threshold": 0.7820955422998104,
                "merge_small_clusters": True,
                "feature_aware": {"enable": False, "weight": 0.0, "acf_lags": [1, 24, 96]},
            },
            "model": {"predictor": "channel_head_mlp", "hidden_dim": 64, "dropout": 0.1837712123193864},
            "moe": {
                "lambda_init": {
                    "level": 0.03208768716055112,
                    "delta": 0.05347947860091853,
                    "d2_match": 0.05347947860091853,
                    "diff_amp": 0.05347947860091853,
                },
                "lambda_min": {"level": 0.0, "delta": 0.0, "d2_match": 0.0, "diff_amp": 0.0},
                "lambda_schedule": {"level": "none", "delta": "none", "d2_match": "none", "diff_amp": "none"},
                "skip_cost": 0.18055852084231583,
                "gate_temperature": 1.2377918880105412,
                "gate_noise_std": 0.18644363730054986,
                "pred_side_residual": {
                    "corrector_hidden": 16,
                    "alpha_scale": 0.2782093334491576,
                    "feature_mode": "safe_augmented",
                },
                "cluster_penalty_prior": {"logit_strength": 0.7404729009145958},
            },
            "penalties": {"enabled": ["level", "delta", "d2_match", "diff_amp"], "jump_threshold": 0.6},
            "train": {"lr": 0.0003823869778094861, "weight_decay": 0.0004568488333079209},
        },
    )
    trials.append(("trial_0015", cfg))

    cfg = base_common(
        base,
        out_root,
        "trial_0025_amp_dir_context",
        "ETTh2_H720_amp_dir_contextchannelheadmlp_h192_do0p0588_l0p0257_flat_lr0p0004_wd0p0009_wu0_bs96_dt0p83_fa0p2_gt1p8_gn0p25_sk0p33_pk0_ps0p74_gated_ra0p55_legacy",
        device,
    )
    deep_update(
        cfg,
        {
            "cluster": {
                "distance_threshold": 0.8296040297067719,
                "merge_small_clusters": False,
                "feature_aware": {"enable": True, "weight": 0.2, "acf_lags": [1, 24, 96]},
            },
            "model": {
                "predictor": "context_channel_head_mlp",
                "hidden_dim": 192,
                "dropout": 0.058812510640864755,
            },
            "moe": {
                "lambda_init": {
                    "amp_under": 0.025736218801562183,
                    "delta": 0.025736218801562183,
                    "diff_amp": 0.025736218801562183,
                    "direction": 0.025736218801562183,
                },
                "lambda_min": {"amp_under": 0.0, "delta": 0.0, "diff_amp": 0.0, "direction": 0.0},
                "lambda_schedule": {
                    "amp_under": "none",
                    "delta": "none",
                    "diff_amp": "none",
                    "direction": "none",
                },
                "skip_cost": 0.3251141947626946,
                "gate_temperature": 1.7811823601314352,
                "gate_noise_std": 0.253531446203685,
                "pred_side_residual": {
                    "corrector_hidden": 48,
                    "alpha_scale": 0.5524919660734575,
                    "feature_mode": "legacy",
                },
                "cluster_penalty_prior": {"logit_strength": 0.74188355184835},
            },
            "penalties": {"enabled": ["amp_under", "delta", "diff_amp", "direction"], "jump_threshold": 0.6},
            "train": {"lr": 0.0003970030151414102, "weight_decay": 0.0008951784151525129},
        },
    )
    trials.append(("trial_0025", cfg))

    return trials


def metric(summary: dict[str, Any], split: str, name: str) -> Any:
    value = summary.get(split, {})
    if isinstance(value, dict):
        return value.get(f"avg_{name}", value.get(name))
    return None


def summarize(label: str, cfg_path: Path, cfg: dict[str, Any], returncode: int) -> dict[str, Any]:
    run_dir = ROOT / cfg["exp"]["out_dir"]
    summary_path = run_dir / "run_summary.json"
    row: dict[str, Any] = {
        "label": label,
        "returncode": returncode,
        "config_path": posix_path(cfg_path.relative_to(ROOT)),
        "out_dir": cfg["exp"]["out_dir"],
        "predictor": cfg.get("model", {}).get("predictor"),
        "hidden_dim": cfg.get("model", {}).get("hidden_dim"),
        "dropout": cfg.get("model", {}).get("dropout"),
        "penalties": ",".join(cfg.get("penalties", {}).get("enabled", [])),
        "lr": cfg.get("train", {}).get("lr"),
        "weight_decay": cfg.get("train", {}).get("weight_decay"),
        "batch_size": cfg.get("train", {}).get("batch_size"),
        "summary_path": posix_path(summary_path.relative_to(ROOT)),
    }
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)
        row.update(
            {
                "val_mse": metric(summary, "val", "mse"),
                "val_mae": metric(summary, "val", "mae"),
                "test_mse": metric(summary, "test", "mse"),
                "test_mae": metric(summary, "test", "mae"),
                "selected_variant": summary.get("selected_variant"),
                "best_epoch": summary.get("best_epoch"),
                "total_sec": summary.get("total_sec"),
            }
        )
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", default="configs/ETTh2.yaml")
    parser.add_argument("--out-root", default="outputs/manual_etth2_h720_reruns")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--reuse-existing", action="store_true")
    args = parser.parse_args()

    base = read_yaml(ROOT / args.base_config)
    out_root = ROOT / args.out_root
    cfg_dir = out_root / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    for label, cfg in trial_configs(base, out_root, args.device):
        cfg_path = cfg_dir / f"{label}.yaml"
        write_yaml(cfg_path, cfg)
        run_dir = ROOT / cfg["exp"]["out_dir"]
        summary_path = run_dir / "run_summary.json"
        returncode = 0
        if args.prepare_only:
            print(f"[prepared] {cfg_path}")
        elif args.reuse_existing and summary_path.exists():
            print(f"[reuse] {summary_path}")
        else:
            print(f"[run] {label}: {cfg_path}", flush=True)
            completed = subprocess.run([args.python, "-u", "-m", "src.train", "--config", str(cfg_path)], cwd=ROOT)
            returncode = completed.returncode
        rows.append(summarize(label, cfg_path, cfg, returncode))

    result_path = out_root / "manual_trial_results.csv"
    fieldnames = [
        "label",
        "returncode",
        "val_mse",
        "val_mae",
        "test_mse",
        "test_mae",
        "selected_variant",
        "best_epoch",
        "total_sec",
        "predictor",
        "hidden_dim",
        "dropout",
        "penalties",
        "lr",
        "weight_decay",
        "batch_size",
        "config_path",
        "summary_path",
        "out_dir",
    ]
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with result_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[saved] {result_path}")


if __name__ == "__main__":
    main()
