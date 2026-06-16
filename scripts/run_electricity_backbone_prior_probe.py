from __future__ import annotations

import argparse
import copy
import csv
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]

BASE_CONFIGS = {
    96: ROOT
    / "outputs"
    / "e_h96_alpha095_final_probe"
    / "configs"
    / "electricity"
    / "H96"
    / "final"
    / "electric_h96_centerres_h256_a095_wd0_bs128.yaml",
    192: ROOT
    / "outputs"
    / "electricity_strict_20260615_backbones"
    / "configs"
    / "electricity"
    / "H192"
    / "final"
    / "electric_h96params_h256_do0_modelstat_p168_center_anchorres_a10_wd0.yaml",
    336: ROOT
    / "outputs"
    / "electricity_strict_20260615_backbones"
    / "configs"
    / "electricity"
    / "H336"
    / "final"
    / "electric_h96params_h256_do0_modelstat_p168_center_anchorres_a10_wd0.yaml",
    720: ROOT
    / "outputs"
    / "e_h720_best224_ckpt"
    / "configs"
    / "electricity"
    / "H720"
    / "final"
    / "electric_h720_centerres_h224_a08_wd1e5_do0_bs8.yaml",
}

FIELDS = [
    "status",
    "horizon",
    "variant",
    "val_mse",
    "val_mae",
    "test_mse",
    "test_mae",
    "base_test_mse",
    "base_test_mae",
    "mse_gain_vs_base_pct",
    "mae_gain_vs_base_pct",
    "best_epoch",
    "total_sec",
    "config_path",
    "out_dir",
    "returncode",
    "error",
]


@dataclass(frozen=True)
class BackboneCandidate:
    variant: str
    model_patch: dict[str, Any]
    train_patch: dict[str, Any]
    early_stop_patch: dict[str, Any]
    finetune: bool = True


def candidate_map() -> dict[str, BackboneCandidate]:
    return {
        "channel_r4_s025_lr3e4": BackboneCandidate(
            variant="channel_r4_s025_lr3e4",
            model_patch={
                "channel_adapter": {
                    "enable": True,
                    "rank": 4,
                    "scale": 0.25,
                    "init": "zero_delta",
                }
            },
            train_patch={"lr": 3.0e-4, "weight_decay": 1.0e-5, "epochs": 35},
            early_stop_patch={"patience": 5},
        ),
        "channel_r4_s025_frozen_lr1e3": BackboneCandidate(
            variant="channel_r4_s025_frozen_lr1e3",
            model_patch={
                "channel_adapter": {
                    "enable": True,
                    "rank": 4,
                    "scale": 0.25,
                    "init": "zero_delta",
                    "freeze_base": True,
                }
            },
            train_patch={"lr": 1.0e-3, "weight_decay": 1.0e-5, "epochs": 40},
            early_stop_patch={"patience": 6},
        ),
        "channel_r8_s010_lr3e4": BackboneCandidate(
            variant="channel_r8_s010_lr3e4",
            model_patch={
                "channel_adapter": {
                    "enable": True,
                    "rank": 8,
                    "scale": 0.10,
                    "init": "zero_delta",
                }
            },
            train_patch={"lr": 3.0e-4, "weight_decay": 1.0e-5, "epochs": 35},
            early_stop_patch={"patience": 5},
        ),
        "basis_r16_s010_lr3e4": BackboneCandidate(
            variant="basis_r16_s010_lr3e4",
            model_patch={
                "temporal_basis_adapter": {
                    "enable": True,
                    "rank": 16,
                    "scale": 0.10,
                    "init": "zero_delta",
                }
            },
            train_patch={"lr": 3.0e-4, "weight_decay": 1.0e-5, "epochs": 35},
            early_stop_patch={"patience": 5},
        ),
        "basis_r16_s010_frozen_lr1e3": BackboneCandidate(
            variant="basis_r16_s010_frozen_lr1e3",
            model_patch={
                "temporal_basis_adapter": {
                    "enable": True,
                    "rank": 16,
                    "scale": 0.10,
                    "init": "zero_delta",
                    "freeze_base": True,
                }
            },
            train_patch={"lr": 1.0e-3, "weight_decay": 1.0e-5, "epochs": 40},
            early_stop_patch={"patience": 6},
        ),
        "basis_r16_s005_frozen_lr1e3": BackboneCandidate(
            variant="basis_r16_s005_frozen_lr1e3",
            model_patch={
                "temporal_basis_adapter": {
                    "enable": True,
                    "rank": 16,
                    "scale": 0.05,
                    "init": "zero_delta",
                    "freeze_base": True,
                }
            },
            train_patch={"lr": 1.0e-3, "weight_decay": 1.0e-5, "epochs": 45},
            early_stop_patch={"patience": 6},
        ),
        "basis_r16_s020_frozen_lr1e3": BackboneCandidate(
            variant="basis_r16_s020_frozen_lr1e3",
            model_patch={
                "temporal_basis_adapter": {
                    "enable": True,
                    "rank": 16,
                    "scale": 0.20,
                    "init": "zero_delta",
                    "freeze_base": True,
                }
            },
            train_patch={"lr": 1.0e-3, "weight_decay": 1.0e-5, "epochs": 45},
            early_stop_patch={"patience": 6},
        ),
        "basis_r32_s010_frozen_lr1e3": BackboneCandidate(
            variant="basis_r32_s010_frozen_lr1e3",
            model_patch={
                "temporal_basis_adapter": {
                    "enable": True,
                    "rank": 32,
                    "scale": 0.10,
                    "init": "zero_delta",
                    "freeze_base": True,
                }
            },
            train_patch={"lr": 1.0e-3, "weight_decay": 1.0e-5, "epochs": 45},
            early_stop_patch={"patience": 6},
        ),
        "basis_r32_s020_frozen_lr1e3": BackboneCandidate(
            variant="basis_r32_s020_frozen_lr1e3",
            model_patch={
                "temporal_basis_adapter": {
                    "enable": True,
                    "rank": 32,
                    "scale": 0.20,
                    "init": "zero_delta",
                    "freeze_base": True,
                }
            },
            train_patch={"lr": 1.0e-3, "weight_decay": 1.0e-5, "epochs": 45},
            early_stop_patch={"patience": 6},
        ),
        "basis_r16_s010_channel_r4_s020_frozen_lr1e3": BackboneCandidate(
            variant="basis_r16_s010_channel_r4_s020_frozen_lr1e3",
            model_patch={
                "temporal_basis_adapter": {
                    "enable": True,
                    "rank": 16,
                    "scale": 0.10,
                    "init": "zero_delta",
                    "freeze_base": True,
                },
                "channel_adapter": {
                    "enable": True,
                    "rank": 4,
                    "scale": 0.20,
                    "init": "zero_delta",
                    "freeze_base": False,
                },
            },
            train_patch={"lr": 1.0e-3, "weight_decay": 1.0e-5, "epochs": 45},
            early_stop_patch={"patience": 6},
        ),
        "bias_s005_lr3e4": BackboneCandidate(
            variant="bias_s005_lr3e4",
            model_patch={
                "horizon_bias_adapter": {
                    "enable": True,
                    "init_bias": 0.0,
                    "scale": 0.05,
                }
            },
            train_patch={"lr": 3.0e-4, "weight_decay": 1.0e-5, "epochs": 35},
            early_stop_patch={"patience": 5},
        ),
        "bias_s010_frozen_lr1e3": BackboneCandidate(
            variant="bias_s010_frozen_lr1e3",
            model_patch={
                "horizon_bias_adapter": {
                    "enable": True,
                    "init_bias": 0.0,
                    "scale": 0.10,
                    "freeze_base": True,
                }
            },
            train_patch={"lr": 1.0e-3, "weight_decay": 1.0e-5, "epochs": 40},
            early_stop_patch={"patience": 6},
        ),
        "seasonal_anchor_p24_np4_d025_lr3e4": BackboneCandidate(
            variant="seasonal_anchor_p24_np4_d025_lr3e4",
            model_patch={
                "seasonal_anchor": True,
                "seasonal_anchor_period": 24,
                "seasonal_anchor_num_periods": 4,
                "seasonal_anchor_delta_scale": 0.25,
            },
            train_patch={"lr": 3.0e-4, "weight_decay": 1.0e-5, "epochs": 35},
            early_stop_patch={"patience": 5},
        ),
        "seasonal_blend_p24_np4_m020_i002_lr3e4": BackboneCandidate(
            variant="seasonal_blend_p24_np4_m020_i002_lr3e4",
            model_patch={
                "seasonal_blend_adapter": {
                    "enable": True,
                    "period": 24,
                    "num_periods": 4,
                    "max_mix": 0.20,
                    "init_mix": 0.02,
                }
            },
            train_patch={"lr": 3.0e-4, "weight_decay": 1.0e-5, "epochs": 35},
            early_stop_patch={"patience": 5},
        ),
        "channel_r4_basis_r16_lr2e4": BackboneCandidate(
            variant="channel_r4_basis_r16_lr2e4",
            model_patch={
                "channel_adapter": {
                    "enable": True,
                    "rank": 4,
                    "scale": 0.20,
                    "init": "zero_delta",
                },
                "temporal_basis_adapter": {
                    "enable": True,
                    "rank": 16,
                    "scale": 0.05,
                    "init": "zero_delta",
                },
            },
            train_patch={"lr": 2.0e-4, "weight_decay": 1.0e-5, "epochs": 35},
            early_stop_patch={"patience": 5},
        ),
        "longctx_anchor_tail24_profile_h256_lr1e3": BackboneCandidate(
            variant="longctx_anchor_tail24_profile_h256_lr1e3",
            model_patch={
                "predictor": "long_context_anchor_channel_head_mlp",
                "hidden_dim": 256,
                "dropout": 0.0,
                "predictor_input_len": 24,
                "long_context_include_seasonal_profile": True,
                "anchor_chunk_len": 24,
                "anchor_detail_scale": 0.25,
            },
            train_patch={"lr": 1.0e-3, "weight_decay": 1.0e-5, "epochs": 100, "batch_size": 64},
            early_stop_patch={"patience": 8},
            finetune=False,
        ),
        "mlp_h224_p168_a10_wd1e5_lr1309": BackboneCandidate(
            variant="mlp_h224_p168_a10_wd1e5_lr1309",
            model_patch={
                "predictor": "mlp",
                "hidden_dim": 224,
                "dropout": 0.0,
                "train_stat_adapter": {
                    "enable": True,
                    "period": 168,
                    "mode": "phase_mean",
                    "alpha": 1.0,
                    "blend_target": "prediction",
                    "combine_mode": "anchor_plus_prediction",
                    "input_center": True,
                },
            },
            train_patch={"lr": 0.001309395478035077, "weight_decay": 1.0e-5, "epochs": 100, "batch_size": 64},
            early_stop_patch={"patience": 6},
            finetune=False,
        ),
        "mlp_h288_p168_a10_wd1e5_lr1309": BackboneCandidate(
            variant="mlp_h288_p168_a10_wd1e5_lr1309",
            model_patch={
                "predictor": "mlp",
                "hidden_dim": 288,
                "dropout": 0.0,
                "train_stat_adapter": {
                    "enable": True,
                    "period": 168,
                    "mode": "phase_mean",
                    "alpha": 1.0,
                    "blend_target": "prediction",
                    "combine_mode": "anchor_plus_prediction",
                    "input_center": True,
                },
            },
            train_patch={"lr": 0.001309395478035077, "weight_decay": 1.0e-5, "epochs": 100, "batch_size": 64},
            early_stop_patch={"patience": 6},
            finetune=False,
        ),
        "mlp_h320_p168_a10_wd1e5_lr1309": BackboneCandidate(
            variant="mlp_h320_p168_a10_wd1e5_lr1309",
            model_patch={
                "predictor": "mlp",
                "hidden_dim": 320,
                "dropout": 0.0,
                "train_stat_adapter": {
                    "enable": True,
                    "period": 168,
                    "mode": "phase_mean",
                    "alpha": 1.0,
                    "blend_target": "prediction",
                    "combine_mode": "anchor_plus_prediction",
                    "input_center": True,
                },
            },
            train_patch={"lr": 0.001309395478035077, "weight_decay": 1.0e-5, "epochs": 100, "batch_size": 64},
            early_stop_patch={"patience": 6},
            finetune=False,
        ),
        "longctx_anchor_tail48_profile_h256_lr1e3": BackboneCandidate(
            variant="longctx_anchor_tail48_profile_h256_lr1e3",
            model_patch={
                "predictor": "long_context_anchor_channel_head_mlp",
                "hidden_dim": 256,
                "dropout": 0.0,
                "predictor_input_len": 48,
                "long_context_include_seasonal_profile": True,
                "anchor_chunk_len": 24,
                "anchor_detail_scale": 0.25,
            },
            train_patch={"lr": 1.0e-3, "weight_decay": 1.0e-5, "epochs": 100, "batch_size": 64},
            early_stop_patch={"patience": 8},
            finetune=False,
        ),
        "seasonal_gated_tail24_profile_h256_mixm2_gate4_lr1e3": BackboneCandidate(
            variant="seasonal_gated_tail24_profile_h256_mixm2_gate4_lr1e3",
            model_patch={
                "predictor": "seasonality_gated_channel_head_mlp",
                "hidden_dim": 256,
                "dropout": 0.0,
                "predictor_input_len": 24,
                "long_context_include_seasonal_profile": True,
                "anchor_chunk_len": 24,
                "anchor_detail_scale": 0.25,
                "seasonal_mix_init": -2.0,
                "seasonal_gate_strength": 4.0,
                "seasonal_gate_threshold": 0.75,
            },
            train_patch={"lr": 1.0e-3, "weight_decay": 1.0e-5, "epochs": 100, "batch_size": 64},
            early_stop_patch={"patience": 8},
            finetune=False,
        ),
        "seasonal_gated_tail48_profile_h256_mixm2_gate4_lr1e3": BackboneCandidate(
            variant="seasonal_gated_tail48_profile_h256_mixm2_gate4_lr1e3",
            model_patch={
                "predictor": "seasonality_gated_channel_head_mlp",
                "hidden_dim": 256,
                "dropout": 0.0,
                "predictor_input_len": 48,
                "long_context_include_seasonal_profile": True,
                "anchor_chunk_len": 24,
                "anchor_detail_scale": 0.25,
                "seasonal_mix_init": -2.0,
                "seasonal_gate_strength": 4.0,
                "seasonal_gate_threshold": 0.75,
            },
            train_patch={"lr": 1.0e-3, "weight_decay": 1.0e-5, "epochs": 100, "batch_size": 64},
            early_stop_patch={"patience": 8},
            finetune=False,
        ),
    }


def deep_update(dst: dict[str, Any], src: dict[str, Any]) -> dict[str, Any]:
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            deep_update(dst[key], value)
        else:
            dst[key] = copy.deepcopy(value)
    return dst


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_yaml(path: Path, cfg: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=False, sort_keys=False)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def localize_paths(cfg: dict[str, Any], out_dir: Path) -> None:
    cfg.setdefault("exp", {})["out_dir"] = str(out_dir)
    cfg.setdefault("corr", {})["save_path"] = str(out_dir / "corr.npy")
    cfg.setdefault("portrait", {})["enable"] = False
    cfg["portrait"]["out_dir"] = str(out_dir / "cluster_portraits")
    cfg.setdefault("knn_hybrid", {})["enable"] = False
    cfg["knn_hybrid"]["use_for_model_selection"] = False
    cfg["knn_hybrid"]["path"] = str(out_dir / "knn_shape_bank.pt")
    cfg.setdefault("memory", {})["enable"] = False
    cfg["memory"]["save_checkpoint"] = True
    cfg["memory"]["path"] = str(out_dir / "cluster_memory.pt")
    cfg["memory"]["checkpoint_path"] = str(out_dir / "best_checkpoint.pt")
    cfg.setdefault("plot", {})["enable"] = False


def base_checkpoint_path(base_cfg: dict[str, Any]) -> str:
    memory = base_cfg.get("memory", {}) or {}
    checkpoint = memory.get("checkpoint_path")
    if checkpoint:
        return str(checkpoint)
    out_dir = Path(str(base_cfg.get("exp", {}).get("out_dir", "")))
    return str(out_dir / "best_checkpoint.pt")


def build_config(base_cfg: dict[str, Any], cand: BackboneCandidate, *, horizon: int, out_dir: Path, device: str) -> dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    cfg.setdefault("exp", {})["name"] = f"electricity_input96_H{horizon}_backbone_prior_{cand.variant}"
    cfg["exp"]["device"] = str(device)
    localize_paths(cfg, out_dir)
    cfg.setdefault("window", {})["input_len"] = 96
    cfg["window"]["pred_len"] = int(horizon)
    cfg["window"]["past_context"] = True
    cfg.setdefault("normalize", {})["train_only"] = True
    cfg.setdefault("cluster", {})["train_only"] = True
    cfg.setdefault("eval", {})["skip_test"] = False
    cfg.setdefault("calendar_residual", {})["enable"] = False

    model = cfg.setdefault("model", {})
    deep_update(model, cand.model_patch)

    train = cfg.setdefault("train", {})
    deep_update(train, cand.train_patch)
    train["selection_metric"] = "val_mse"

    early_stop = cfg.setdefault("early_stop", {})
    deep_update(early_stop, cand.early_stop_patch)

    moe = cfg.setdefault("moe", {})
    moe["enable"] = False
    moe["freeze_backbone"] = False

    if cand.finetune:
        cfg["finetune"] = {
            "enable": True,
            "checkpoint_path": base_checkpoint_path(base_cfg),
            "strict_window": True,
            "strict_model": False,
            "cluster_map": "index",
            "load_model": True,
            "load_gate": False,
            "load_dynamic_lambda": False,
            "load_learnable_lambda": False,
        }
    else:
        cfg.pop("finetune", None)
    return cfg


def summary_row(
    *,
    horizon: int,
    cand: BackboneCandidate,
    cfg_path: Path,
    out_dir: Path,
    returncode: int,
    total_sec: float,
    error: str,
    base_mse: float,
    base_mae: float,
) -> dict[str, Any]:
    summary = read_json(out_dir / "run_summary.json")
    val = summary.get("val") or {}
    test = summary.get("test") or {}
    test_mse = test.get("avg_mse", "")
    test_mae = test.get("avg_mae", "")

    def gain(base: float, current: Any) -> str:
        try:
            cur = float(current)
            return f"{(base - cur) / base * 100.0:.6f}"
        except Exception:
            return ""

    return {
        "status": "ok" if returncode == 0 and summary else ("failed" if returncode else "prepared"),
        "horizon": int(horizon),
        "variant": cand.variant,
        "val_mse": val.get("avg_mse", ""),
        "val_mae": val.get("avg_mae", ""),
        "test_mse": test_mse,
        "test_mae": test_mae,
        "base_test_mse": base_mse,
        "base_test_mae": base_mae,
        "mse_gain_vs_base_pct": gain(base_mse, test_mse),
        "mae_gain_vs_base_pct": gain(base_mae, test_mae),
        "best_epoch": ",".join(str(v) for v in summary.get("best_epoch", [])) if summary else "",
        "total_sec": f"{float(total_sec):.3f}",
        "config_path": str(cfg_path),
        "out_dir": str(out_dir),
        "returncode": int(returncode),
        "error": error,
    }


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDS})


def run_one(
    *,
    horizon: int,
    base_cfg: dict[str, Any],
    cand: BackboneCandidate,
    out_root: Path,
    device: str,
    reuse_existing: bool,
) -> dict[str, Any]:
    base_summary = read_json(Path(str(base_cfg["exp"]["out_dir"])) / "run_summary.json")
    base_test = base_summary.get("test") or {}
    base_mse = float(base_test.get("avg_mse"))
    base_mae = float(base_test.get("avg_mae"))
    out_dir = out_root / "runs" / f"H{horizon}" / cand.variant
    cfg_path = out_root / "configs" / f"H{horizon}" / f"{cand.variant}.yaml"
    cfg = build_config(base_cfg, cand, horizon=horizon, out_dir=out_dir, device=device)
    write_yaml(cfg_path, cfg)
    if reuse_existing and (out_dir / "run_summary.json").exists():
        return summary_row(
            horizon=horizon,
            cand=cand,
            cfg_path=cfg_path,
            out_dir=out_dir,
            returncode=0,
            total_sec=0.0,
            error="",
            base_mse=base_mse,
            base_mae=base_mae,
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = out_dir / "stdout.log"
    stderr_path = out_dir / "stderr.log"
    cmd = [sys.executable, "-u", "-m", "src.train", "--config", str(cfg_path)]
    start = time.perf_counter()
    with stdout_path.open("w", encoding="utf-8") as stdout_f, stderr_path.open("w", encoding="utf-8") as stderr_f:
        completed = subprocess.run(cmd, cwd=str(ROOT), stdout=stdout_f, stderr=stderr_f, text=True)
    total_sec = time.perf_counter() - start
    error = ""
    if completed.returncode != 0:
        error = stderr_path.read_text(encoding="utf-8", errors="replace")[-3000:]
    return summary_row(
        horizon=horizon,
        cand=cand,
        cfg_path=cfg_path,
        out_dir=out_dir,
        returncode=completed.returncode,
        total_sec=total_sec,
        error=error,
        base_mse=base_mse,
        base_mae=base_mae,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe electricity backbone priors from strict input_len=96 checkpoints.")
    parser.add_argument("--horizon", type=int, action="append", choices=sorted(BASE_CONFIGS), required=True)
    parser.add_argument("--out-root", type=str, default="outputs/electricity_backbone_prior_probe_20260615")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--variants", type=str, default="channel_r4_s025_lr3e4,basis_r16_s010_lr3e4,bias_s005_lr3e4")
    parser.add_argument("--reuse-existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    variants = [v.strip() for v in str(args.variants).split(",") if v.strip()]
    candidates = candidate_map()
    unknown = [v for v in variants if v not in candidates]
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}. Available: {sorted(candidates)}")

    out_root = (ROOT / args.out_root).resolve()
    rows: list[dict[str, Any]] = []
    for horizon in args.horizon:
        base_cfg = load_yaml(BASE_CONFIGS[int(horizon)])
        for variant in variants:
            cand = candidates[variant]
            print(f"=== H{horizon} {variant} ===", flush=True)
            row = run_one(
                horizon=int(horizon),
                base_cfg=base_cfg,
                cand=cand,
                out_root=out_root,
                device=str(args.device),
                reuse_existing=bool(args.reuse_existing),
            )
            rows.append(row)
            write_rows(out_root / "backbone_prior_results.csv", rows)
            print(
                json.dumps(
                    {
                        "status": row["status"],
                        "horizon": row["horizon"],
                        "variant": row["variant"],
                        "val_mse": row["val_mse"],
                        "test_mse": row["test_mse"],
                        "test_mae": row["test_mae"],
                        "mse_gain_vs_base_pct": row["mse_gain_vs_base_pct"],
                    },
                    ensure_ascii=True,
                ),
                flush=True,
            )
    ok_rows = [r for r in rows if r.get("status") == "ok" and r.get("test_mse") != ""]
    if ok_rows:
        best = min(ok_rows, key=lambda r: float(r["test_mse"]))
        print(f"Best: H{best['horizon']} {best['variant']} test_mse={best['test_mse']} test_mae={best['test_mae']}")
    print(f"Wrote: {out_root / 'backbone_prior_results.csv'}")


if __name__ == "__main__":
    main()
