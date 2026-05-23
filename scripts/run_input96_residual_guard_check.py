from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def write_yaml(path: Path, cfg: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")


def deep_copy(obj: Any) -> Any:
    return json.loads(json.dumps(obj))


def set_run_paths(cfg: dict[str, Any], out_dir: Path, name: str, device: str | None) -> None:
    cfg.setdefault("exp", {})["name"] = name
    cfg.setdefault("exp", {})["out_dir"] = str(out_dir)
    if device:
        cfg["exp"]["device"] = str(device)
    cfg.setdefault("corr", {})["save_path"] = str(out_dir / "corr.npy")
    cfg.setdefault("portrait", {})["out_dir"] = str(out_dir / "cluster_portraits")
    cfg.setdefault("knn_hybrid", {})["path"] = str(out_dir / "knn_shape_bank.pt")
    cfg.setdefault("memory", {})["path"] = str(out_dir / "cluster_memory.pt")
    cfg.setdefault("memory", {})["checkpoint_path"] = str(out_dir / "best_checkpoint.pt")


def base_input96_cfg(base_cfg: dict[str, Any], out_dir: Path, name: str, device: str | None) -> dict[str, Any]:
    cfg = deep_copy(base_cfg)
    cfg.setdefault("window", {})["input_len"] = 96
    cfg.setdefault("window", {})["pred_len"] = 96
    cfg.setdefault("eval", {})["skip_test"] = False
    cfg.setdefault("plot", {})["enable"] = False
    cfg.setdefault("portrait", {})["enable"] = False
    cfg.setdefault("knn_hybrid", {})["enable"] = False
    cfg.setdefault("calibration", {})["enable"] = False
    cfg.setdefault("memory", {})["enable"] = False
    cfg.setdefault("memory", {})["save_checkpoint"] = False
    cfg.pop("diagnostics", None)
    set_run_paths(cfg, out_dir, name, device)
    return cfg


def apply_variant(cfg: dict[str, Any], variant: str) -> None:
    moe = cfg.setdefault("moe", {})
    residual = moe.setdefault("pred_side_residual", {})
    gate = residual.setdefault("gate_calibrator", {})

    def configure_channel_head_base() -> None:
        cfg.setdefault("model", {}).update(
            {
                "predictor": "channel_head_mlp",
                "hidden_dim": 256,
                "dropout": 0.0,
                "channel_head_residual": True,
            }
        )
        cfg.setdefault("train", {})["weight_decay"] = 1.0e-3

    def configure_scale_guard() -> None:
        residual["selection_policy"] = "val_mse_scale"
        residual["selection_scale_min"] = 0.0
        residual["selection_scale_max"] = 1.0
        residual["selection_scale_steps"] = 21

    def set_penalties(names: list[str], lam: float, dynamic: bool) -> None:
        cfg.setdefault("penalties", {})["enabled"] = names
        moe["lambda_init"] = {name: float(lam) for name in names}
        moe["lambda_min"] = {name: 0.0 for name in names}
        moe["lambda_schedule"] = {name: "none" for name in names}
        moe.setdefault("dynamic_lambda", {})["enable"] = bool(dynamic)
    if variant == "current_gate":
        residual["selection_policy"] = "val_mse_gate"
    elif variant == "gate_guarded":
        residual["selection_policy"] = "val_mse_gate_guarded"
    elif variant == "scale_grid":
        residual["selection_policy"] = "val_mse_scale"
        residual["selection_scale_min"] = 0.0
        residual["selection_scale_max"] = 1.0
        residual["selection_scale_steps"] = 21
    elif variant == "channel_binary":
        residual["selection_policy"] = "val_mse_channel"
    elif variant == "activation_head_guarded":
        residual["selection_policy"] = "val_mse_gate_guarded"
        gate["activation_head_enable"] = True
        gate["apply_activation_threshold"] = True
        gate["activation_threshold"] = "auto"
        gate["activation_threshold_selection_metric"] = "mse"
        gate["activation_threshold_scope"] = "channel"
        gate["activation_bce_weight"] = 0.2
        gate["activation_inactive_scale_weight"] = 0.05
        gate["activation_pos_weight"] = "auto"
        gate["activation_pos_weight_scope"] = "channel"
    elif variant == "residual_disabled":
        residual["enable"] = False
    elif variant == "mlp_h128_scale":
        cfg.setdefault("model", {}).update({"predictor": "mlp", "hidden_dim": 128, "dropout": 0.0})
        cfg.setdefault("train", {})["weight_decay"] = 1.0e-4
        residual["selection_policy"] = "val_mse_scale"
        residual["selection_scale_min"] = 0.0
        residual["selection_scale_max"] = 1.0
        residual["selection_scale_steps"] = 21
    elif variant == "channel_h256_scale":
        cfg.setdefault("model", {}).update(
            {"predictor": "channel_head_mlp", "hidden_dim": 256, "dropout": 0.2, "channel_head_residual": True}
        )
        cfg.setdefault("train", {})["weight_decay"] = 1.0e-3
        residual["selection_policy"] = "val_mse_scale"
        residual["selection_scale_min"] = 0.0
        residual["selection_scale_max"] = 1.0
        residual["selection_scale_steps"] = 21
    elif variant.startswith("channel_h") and variant.endswith("_scale") and len(variant.split("_")) == 4:
        stem = variant.removeprefix("channel_h").removesuffix("_scale")
        parts = stem.split("_do", 1)
        try:
            hidden_dim = int(parts[0])
            dropout = float(parts[1].replace("p", ".")) if len(parts) > 1 else 0.2
        except ValueError as exc:
            raise ValueError(f"Invalid channel-head variant: {variant}") from exc
        cfg.setdefault("model", {}).update(
            {
                "predictor": "channel_head_mlp",
                "hidden_dim": hidden_dim,
                "dropout": dropout,
                "channel_head_residual": True,
            }
        )
        cfg.setdefault("train", {})["weight_decay"] = 1.0e-3
        residual["selection_policy"] = "val_mse_scale"
        residual["selection_scale_min"] = 0.0
        residual["selection_scale_max"] = 1.0
        residual["selection_scale_steps"] = 21
    elif variant == "channel_h256_do0p0_moe_off":
        configure_channel_head_base()
        moe["enable"] = False
        moe.setdefault("dynamic_lambda", {})["enable"] = False
        residual["enable"] = False
    elif variant == "channel_h256_do0p0_no_residual":
        configure_channel_head_base()
        residual["enable"] = False
    elif variant == "channel_h256_do0p0_zero_lambda_scale":
        configure_channel_head_base()
        set_penalties(["jump", "amp_under", "level", "delta"], 0.0, False)
        residual["alpha_scale"] = 0.8
        configure_scale_guard()
    elif variant == "channel_h256_do0p0_low_lambda_scale":
        configure_channel_head_base()
        set_penalties(["jump", "amp_under", "level", "delta"], 0.01, False)
        residual["alpha_scale"] = 0.8
        configure_scale_guard()
    elif variant == "channel_h256_do0p0_level_amp_l005":
        configure_channel_head_base()
        set_penalties(["level", "amp_under"], 0.005, False)
        residual["alpha_scale"] = 0.5
        configure_scale_guard()
    elif variant == "channel_h256_do0p0_jump_level_l005":
        configure_channel_head_base()
        set_penalties(["jump", "level"], 0.005, False)
        residual["alpha_scale"] = 0.5
        configure_scale_guard()
    elif variant == "channel_h256_do0p0_shape_l005":
        configure_channel_head_base()
        set_penalties(["level", "delta", "diff_amp"], 0.005, False)
        residual["alpha_scale"] = 0.5
        configure_scale_guard()
    elif variant == "context_h256_scale":
        cfg.setdefault("model", {}).update(
            {"predictor": "context_mlp", "hidden_dim": 256, "dropout": 0.2, "context_include_delta": True}
        )
        cfg.setdefault("train", {})["weight_decay"] = 1.0e-3
        residual["selection_policy"] = "val_mse_scale"
        residual["selection_scale_min"] = 0.0
        residual["selection_scale_max"] = 1.0
        residual["selection_scale_steps"] = 21
    elif variant == "attn_h256_scale":
        cfg.setdefault("model", {}).update({"predictor": "attn_mlp", "hidden_dim": 256, "dropout": 0.2, "attn_dim": 64})
        cfg.setdefault("train", {})["weight_decay"] = 1.0e-3
        residual["selection_policy"] = "val_mse_scale"
        residual["selection_scale_min"] = 0.0
        residual["selection_scale_max"] = 1.0
        residual["selection_scale_steps"] = 21
    elif variant.startswith("mlp_thr"):
        try:
            threshold = float(variant.removeprefix("mlp_thr").replace("p", "."))
        except ValueError as exc:
            raise ValueError(f"Invalid threshold variant: {variant}") from exc
        cfg.setdefault("cluster", {})["distance_threshold"] = threshold
        cfg.setdefault("cluster", {})["merge_small_clusters"] = True
        cfg.setdefault("model", {}).update({"predictor": "mlp", "hidden_dim": 256, "dropout": 0.2})
        residual["selection_policy"] = "val_mse_scale"
        residual["selection_scale_min"] = 0.0
        residual["selection_scale_max"] = 1.0
        residual["selection_scale_steps"] = 21
    elif variant.startswith("mlp_nm_thr"):
        try:
            threshold = float(variant.removeprefix("mlp_nm_thr").replace("p", "."))
        except ValueError as exc:
            raise ValueError(f"Invalid threshold variant: {variant}") from exc
        cfg.setdefault("cluster", {})["distance_threshold"] = threshold
        cfg.setdefault("cluster", {})["merge_small_clusters"] = False
        cfg.setdefault("model", {}).update({"predictor": "mlp", "hidden_dim": 256, "dropout": 0.2})
        residual["selection_policy"] = "val_mse_scale"
        residual["selection_scale_min"] = 0.0
        residual["selection_scale_max"] = 1.0
        residual["selection_scale_steps"] = 21
    elif variant.startswith("channel_thr"):
        try:
            threshold = float(variant.removeprefix("channel_thr").replace("p", "."))
        except ValueError as exc:
            raise ValueError(f"Invalid threshold variant: {variant}") from exc
        cfg.setdefault("cluster", {})["distance_threshold"] = threshold
        cfg.setdefault("cluster", {})["merge_small_clusters"] = True
        cfg.setdefault("model", {}).update(
            {"predictor": "channel_head_mlp", "hidden_dim": 256, "dropout": 0.2, "channel_head_residual": True}
        )
        cfg.setdefault("train", {})["weight_decay"] = 1.0e-3
        residual["selection_policy"] = "val_mse_scale"
        residual["selection_scale_min"] = 0.0
        residual["selection_scale_max"] = 1.0
        residual["selection_scale_steps"] = 21
    elif variant.startswith("channel_nm_thr"):
        try:
            threshold = float(variant.removeprefix("channel_nm_thr").replace("p", "."))
        except ValueError as exc:
            raise ValueError(f"Invalid threshold variant: {variant}") from exc
        cfg.setdefault("cluster", {})["distance_threshold"] = threshold
        cfg.setdefault("cluster", {})["merge_small_clusters"] = False
        cfg.setdefault("model", {}).update(
            {"predictor": "channel_head_mlp", "hidden_dim": 256, "dropout": 0.2, "channel_head_residual": True}
        )
        cfg.setdefault("train", {})["weight_decay"] = 1.0e-3
        residual["selection_policy"] = "val_mse_scale"
        residual["selection_scale_min"] = 0.0
        residual["selection_scale_max"] = 1.0
        residual["selection_scale_steps"] = 21
    else:
        raise ValueError(f"Unknown variant: {variant}")


def run_train(config_path: Path) -> int:
    cfg = load_yaml(config_path)
    out_dir = Path(cfg["exp"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("MOELOSS_PROGRESS_LEAVE", "0")
    with (out_dir / "stdout.log").open("w", encoding="utf-8") as stdout_f, (
        out_dir / "stderr.log"
    ).open("w", encoding="utf-8") as stderr_f:
        proc = subprocess.run(
            [sys.executable, "-m", "src.train", "--config", str(config_path)],
            cwd=str(REPO_ROOT),
            text=True,
            stdout=stdout_f,
            stderr=stderr_f,
            env=env,
        )
    return int(proc.returncode)


def summarize(out_root: Path, variants: list[str]) -> None:
    rows: list[dict[str, Any]] = []
    for variant in variants:
        run_dir = out_root / "runs" / variant
        summary_path = run_dir / "run_summary.json"
        if not summary_path.exists():
            rows.append({"variant": variant, "status": "missing"})
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        selection = summary.get("moe_residual_selection") or {}
        gate_hit = (summary.get("moe_gate_penalty_hit") or {}).get("test") or {}
        test = summary.get("test") or {}
        rows.append(
            {
                "variant": variant,
                "status": "ok",
                "test_mse": test.get("avg_mse"),
                "test_mae": test.get("avg_mae"),
                "val_mse": (summary.get("val") or {}).get("avg_mse"),
                "base_val_mse": selection.get("val_pred_base_avg_mse"),
                "raw_val_mse": selection.get("val_residual_avg_mse"),
                "scaled_val_mse": selection.get("val_scaled_avg_mse"),
                "residual_channels": ",".join(selection.get("residual_channels", []) or []),
                "base_channels": ",".join(selection.get("base_channels", []) or []),
                "mean_scale": selection.get("mean_scale"),
                "gate_selected_gain_pct": gate_hit.get("selected_top1_gain_pct_vs_base"),
                "best_epoch": json.dumps(summary.get("best_epoch", []), ensure_ascii=False),
            }
        )
    fields = [
        "variant",
        "status",
        "test_mse",
        "test_mae",
        "val_mse",
        "base_val_mse",
        "raw_val_mse",
        "scaled_val_mse",
        "residual_channels",
        "base_channels",
        "mean_scale",
        "gate_selected_gain_pct",
        "best_epoch",
    ]
    with (out_root / "results.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Check input_len=96 residual guard variants on ETTh2 H96.")
    ap.add_argument("--base-config", default="outputs/ett_horizon_sweep/configs/ETTh2_pred_96.yaml")
    ap.add_argument("--out-root", default="outputs/input96_residual_guard_check/ETTh2_H96")
    ap.add_argument("--device", default=None)
    ap.add_argument("--variants", nargs="+", default=["current_gate", "gate_guarded", "scale_grid", "channel_binary"])
    ap.add_argument("--run", action="store_true")
    args = ap.parse_args()

    out_root = Path(args.out_root).resolve()
    base_cfg = load_yaml(Path(args.base_config).resolve())
    config_paths: list[Path] = []
    for variant in args.variants:
        cfg = base_input96_cfg(base_cfg, out_root / "runs" / variant, f"ETTh2_h96_input96_{variant}", args.device)
        apply_variant(cfg, variant)
        config_path = out_root / "configs" / f"{variant}.yaml"
        write_yaml(config_path, cfg)
        config_paths.append(config_path)

    if args.run:
        for config_path in config_paths:
            print(f"Running {config_path}", flush=True)
            rc = run_train(config_path)
            print(f"Return code {rc}: {config_path}", flush=True)
            if rc != 0:
                raise SystemExit(rc)
    summarize(out_root, list(args.variants))
    print(f"Wrote {out_root / 'results.csv'}", flush=True)


if __name__ == "__main__":
    main()
