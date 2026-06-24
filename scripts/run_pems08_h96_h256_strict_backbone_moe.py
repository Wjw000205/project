from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = (
    ROOT
    / "outputs"
    / "fresh_input_len96_20260614_pems08_h96_h256_probe"
    / "configs"
    / "p288_h256_stat050_resid200_residmae_seg4.yaml"
)

FIELDS = [
    "phase",
    "variant",
    "status",
    "test_mse",
    "test_mae",
    "val_mse",
    "val_mae",
    "backbone_mse",
    "backbone_mae",
    "mse_gain_pct",
    "mae_gain_pct",
    "config_path",
    "out_dir",
    "returncode",
    "error",
]


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_yaml(path: Path, cfg: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=False)


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDS})


def localize(cfg: dict[str, Any], *, out_dir: Path, name: str, device: str) -> dict[str, Any]:
    cfg = copy.deepcopy(cfg)
    cfg.setdefault("exp", {})
    cfg["exp"]["name"] = name
    cfg["exp"]["out_dir"] = str(out_dir)
    cfg["exp"]["device"] = device
    cfg.setdefault("corr", {})["save_path"] = str(out_dir / "corr.npy")
    cfg.setdefault("portrait", {})["out_dir"] = str(out_dir / "cluster_portraits")
    cfg.setdefault("memory", {})["path"] = str(out_dir / "cluster_memory.pt")
    cfg.setdefault("memory", {})["checkpoint_path"] = str(out_dir / "best_checkpoint.pt")
    cfg.setdefault("eval", {})["skip_test"] = False
    return cfg


def backbone_config(base: dict[str, Any], *, out_dir: Path, device: str, epochs: int) -> dict[str, Any]:
    cfg = localize(base, out_dir=out_dir, name="PEMS08_H96_h256_backbone_strict", device=device)
    cfg.setdefault("window", {})["input_len"] = 96
    cfg["window"]["pred_len"] = 96
    cfg.setdefault("model", {})["predictor"] = "context_channel_head_mlp"
    cfg["model"]["hidden_dim"] = 256
    cfg["model"]["dropout"] = 0.0
    cfg.setdefault("moe", {})["enable"] = False
    cfg.setdefault("train", {})["epochs"] = int(epochs)
    cfg["train"]["batch_size"] = 32
    cfg["train"]["lr"] = 1.0e-3
    cfg["train"]["mse_weight"] = 0.5
    cfg["train"]["selection_metric"] = "val_mae"
    cfg["train"].setdefault("mae_objective", {}).update(
        {"enable": True, "kind": "l1", "weight": 1.5, "warmup_epochs": 3}
    )
    cfg.setdefault("memory", {})["save_checkpoint"] = True
    return cfg


def moe_config(
    base: dict[str, Any],
    *,
    out_dir: Path,
    device: str,
    checkpoint: Path,
    variant: str,
    residual_metric: str,
) -> dict[str, Any]:
    cfg = localize(base, out_dir=out_dir, name=f"PEMS08_H96_h256_strict_{variant}", device=device)
    cfg.setdefault("window", {})["input_len"] = 96
    cfg["window"]["pred_len"] = 96
    cfg.setdefault("model", {})["predictor"] = "context_channel_head_mlp"
    cfg["model"]["hidden_dim"] = 256
    cfg["model"]["dropout"] = 0.0
    cfg.setdefault("moe", {})["enable"] = True
    cfg["moe"]["freeze_backbone"] = True
    cfg["moe"].setdefault("train_stat_anchor_expert", {})["enable"] = True
    cfg["moe"]["train_stat_anchor_expert"].update({"period": 288, "alpha": 0.0, "mode": "phase_mean"})
    cfg["moe"]["train_stat_anchor_expert"].setdefault("scale_selection", {}).update(
        {"enable": True, "metric": "mse", "max_scale": 0.5, "steps": 21}
    )
    cfg["moe"].setdefault("train_residual_anchor_expert", {})["enable"] = True
    cfg["moe"]["train_residual_anchor_expert"].update({"period": 288, "alpha": 0.0})
    cfg["moe"]["train_residual_anchor_expert"].setdefault("scale_selection", {}).update(
        {"enable": True, "metric": residual_metric, "max_scale": 2.0, "steps": 81, "horizon_segments": 4}
    )
    cfg.setdefault("train", {})["epochs"] = 1
    cfg["train"]["lr"] = 0.0
    cfg["train"]["batch_size"] = 32
    cfg["train"]["selection_metric"] = "val_mse"
    cfg.setdefault("finetune", {})
    cfg["finetune"] = {
        "enable": True,
        "checkpoint_path": str(checkpoint),
        "strict_window": True,
        "strict_model": True,
        "cluster_map": "index",
        "load_model": True,
        "load_gate": False,
        "load_dynamic_lambda": False,
        "load_learnable_lambda": False,
    }
    return cfg


def run_train(py: str, cfg_path: Path, out_dir: Path) -> tuple[int, str]:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    proc = subprocess.run(
        [py, "-u", "-m", "src.train", "--config", str(cfg_path)],
        cwd=ROOT,
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "stdout.log").write_text(proc.stdout, encoding="utf-8", errors="replace")
    return int(proc.returncode), proc.stdout


def summarize(
    *,
    phase: str,
    variant: str,
    cfg_path: Path,
    out_dir: Path,
    returncode: int,
    output: str,
    backbone: dict[str, float] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "phase": phase,
        "variant": variant,
        "status": "ok" if returncode == 0 else "error",
        "config_path": str(cfg_path),
        "out_dir": str(out_dir),
        "returncode": returncode,
        "error": "" if returncode == 0 else output[-3000:],
    }
    summary_path = out_dir / "run_summary.json"
    if not summary_path.exists():
        row["status"] = "error"
        row["error"] = row.get("error") or "run_summary.json missing"
        return row
    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)
    val = summary.get("val") or {}
    test = summary.get("test") or {}
    row.update(
        {
            "val_mse": val.get("avg_mse", ""),
            "val_mae": val.get("avg_mae", ""),
            "test_mse": test.get("avg_mse", ""),
            "test_mae": test.get("avg_mae", ""),
        }
    )
    if backbone is not None and row["test_mse"] != "":
        bmse = float(backbone["test_mse"])
        bmae = float(backbone["test_mae"])
        mse = float(row["test_mse"])
        mae = float(row["test_mae"])
        row.update(
            {
                "backbone_mse": bmse,
                "backbone_mae": bmae,
                "mse_gain_pct": (bmse - mse) / bmse * 100.0,
                "mae_gain_pct": (bmae - mae) / bmae * 100.0,
            }
        )
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description="PEMS08 H96 strict h256 backbone then frozen MoE.")
    ap.add_argument(
        "--out-root",
        type=Path,
        default=ROOT / "outputs" / "codex_table_target_20260615_pems08_h96_h256_strict_backbone_moe",
    )
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--backbone-epochs", type=int, default=36)
    ap.add_argument("--rerun", action="store_true")
    args = ap.parse_args()

    base = read_yaml(BASE_CONFIG)
    rows: list[dict[str, Any]] = []

    backbone_dir = args.out_root / "runs" / "backbone_h256"
    backbone_cfg_path = args.out_root / "configs" / "backbone_h256.yaml"
    write_yaml(backbone_cfg_path, backbone_config(base, out_dir=backbone_dir, device=args.device, epochs=args.backbone_epochs))
    if (backbone_dir / "run_summary.json").exists() and not args.rerun:
        print("[reuse] backbone_h256", flush=True)
        rc, out = 0, ""
    else:
        print("[run] backbone_h256", flush=True)
        rc, out = run_train(args.python, backbone_cfg_path, backbone_dir)
    backbone_row = summarize(
        phase="backbone",
        variant="backbone_h256",
        cfg_path=backbone_cfg_path,
        out_dir=backbone_dir,
        returncode=rc,
        output=out,
    )
    rows.append(backbone_row)
    write_rows(args.out_root / "results.csv", rows)
    if backbone_row["status"] != "ok":
        raise SystemExit("backbone run failed")

    backbone_metrics = {"test_mse": float(backbone_row["test_mse"]), "test_mae": float(backbone_row["test_mae"])}
    checkpoint = backbone_dir / "best_checkpoint.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    variants = {
        "p288_h256_stat050_resid200_mse_seg4": "mse",
        "p288_h256_stat050_resid200_residmae_seg4": "mae",
    }
    for variant, metric in variants.items():
        out_dir = args.out_root / "runs" / variant
        cfg_path = args.out_root / "configs" / f"{variant}.yaml"
        write_yaml(
            cfg_path,
            moe_config(base, out_dir=out_dir, device=args.device, checkpoint=checkpoint, variant=variant, residual_metric=metric),
        )
        if (out_dir / "run_summary.json").exists() and not args.rerun:
            print(f"[reuse] {variant}", flush=True)
            rc, out = 0, ""
        else:
            print(f"[run] {variant}", flush=True)
            rc, out = run_train(args.python, cfg_path, out_dir)
        row = summarize(
            phase="moe",
            variant=variant,
            cfg_path=cfg_path,
            out_dir=out_dir,
            returncode=rc,
            output=out,
            backbone=backbone_metrics,
        )
        rows.append(row)
        write_rows(args.out_root / "results.csv", rows)


if __name__ == "__main__":
    main()
