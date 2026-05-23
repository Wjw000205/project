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


def write_yaml(path: Path, cfg: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=False)


def read_summary(run_dir: Path) -> dict[str, Any]:
    with (run_dir / "run_summary.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def patch_paths(cfg: dict[str, Any], out_dir: Path, device: str | None) -> dict[str, Any]:
    cfg = copy.deepcopy(cfg)
    cfg.setdefault("exp", {})["out_dir"] = str(out_dir)
    if device:
        cfg["exp"]["device"] = str(device)
    cfg.setdefault("corr", {})["save_path"] = str(out_dir / "corr.npy")
    cfg.setdefault("plot", {})["enable"] = False
    cfg.setdefault("portrait", {})["enable"] = False
    cfg["portrait"]["out_dir"] = str(out_dir / "cluster_portraits")
    cfg.setdefault("knn_hybrid", {})["enable"] = False
    cfg["knn_hybrid"]["use_for_model_selection"] = False
    cfg["knn_hybrid"]["path"] = str(out_dir / "knn_shape_bank.pt")
    cfg.setdefault("calibration", {})["enable"] = False
    cfg["memory"] = {
        "enable": False,
        "save_checkpoint": False,
        "path": str(out_dir / "cluster_memory.pt"),
        "checkpoint_path": str(out_dir / "best_checkpoint.pt"),
    }
    cfg.setdefault("eval", {})["skip_test"] = False
    return cfg


def run_train(config_path: Path, *, reuse_existing: bool) -> int:
    cfg = read_yaml(config_path)
    out_dir = Path(cfg["exp"]["out_dir"])
    if reuse_existing and (out_dir / "run_summary.json").exists():
        print(f"[reuse] {out_dir}", flush=True)
        return 0
    cmd = [sys.executable, "-m", "src.train", "--config", str(config_path)]
    print(f"[run] {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, cwd=str(ROOT)).returncode


def metric_row(label: str, config_path: Path, run_dir: Path, returncode: int, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    row: dict[str, Any] = {
        "label": label,
        "status": "ok" if returncode == 0 and (run_dir / "run_summary.json").exists() else "failed",
        "config_path": str(config_path),
        "out_dir": str(run_dir),
        "returncode": returncode,
    }
    if extra:
        row.update(extra)
    if (run_dir / "run_summary.json").exists():
        summary = read_summary(run_dir)
        row.update(
            {
                "val_mse": summary.get("val", {}).get("avg_mse", ""),
                "val_mae": summary.get("val", {}).get("avg_mae", ""),
                "test_mse": summary.get("test", {}).get("avg_mse", ""),
                "test_mae": summary.get("test", {}).get("avg_mae", ""),
                "best_epoch": ",".join(str(v) for v in summary.get("best_epoch", [])),
            }
        )
    return row


def prepare_and_run(
    source_config: Path,
    out_root: Path,
    group: str,
    label: str,
    *,
    device: str | None,
    reuse_existing: bool,
    seed: int | None = None,
    patch: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = read_yaml(source_config)
    if seed is not None:
        cfg.setdefault("exp", {})["seed"] = int(seed)
        cfg.setdefault("cluster", {})["random_state"] = int(seed)
    if patch:
        deep_update(cfg, patch)
    run_dir = out_root / "runs" / group / label
    cfg = patch_paths(cfg, run_dir, device)
    cfg.setdefault("exp", {})["name"] = f"paper_rerun_ettm1_{group}_{label}"
    config_path = out_root / "configs" / group / f"{label}.yaml"
    write_yaml(config_path, cfg)
    code = run_train(config_path, reuse_existing=reuse_existing)
    return metric_row(label, config_path, run_dir, code)


def deep_update(target: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            deep_update(target[key], value)
        else:
            target[key] = value
    return target


def current_h96_config() -> Path:
    generated = ROOT / "outputs" / "main_table_tsl_aligned" / "configs" / "ETTm1_pred_96.yaml"
    if generated.exists():
        return generated
    return ROOT / "configs" / "ETTm1.yaml"


def disable_moe_patch() -> dict[str, Any]:
    return {
        "moe": {
            "enable": False,
            "dynamic_lambda": {"enable": False},
            "pred_side_residual": {"enable": False},
        }
    }


def lambda_zero_patch(source_config: Path) -> dict[str, Any]:
    cfg = read_yaml(source_config)
    penalties = list(cfg.get("penalties", {}).get("enabled", []))
    return {
        "moe": {
            "dynamic_lambda": {"enable": False},
            "lambda_init": {name: 0.0 for name in penalties},
            "lambda_min": {name: 0.0 for name in penalties},
            "pred_side_residual": {"enable": True},
        }
    }


def fixed_lambda_patch() -> dict[str, Any]:
    return {
        "moe": {
            "dynamic_lambda": {"enable": False},
            "pred_side_residual": {"enable": True},
        }
    }


def penalty_loss_only_patch() -> dict[str, Any]:
    return {
        "moe": {
            "enable": True,
            "dynamic_lambda": {"enable": True},
            "pred_side_residual": {"enable": False},
        }
    }


def rerun_module(out_root: Path, device: str | None, reuse_existing: bool, source_config: Path) -> list[dict[str, Any]]:
    items = [
        ("moe_off", disable_moe_patch()),
        ("zero_lambda_residual", lambda_zero_patch(source_config)),
        ("fixed_lambda_residual", fixed_lambda_patch()),
        ("penalty_loss_only", penalty_loss_only_patch()),
        ("full_pkr_moe", {}),
    ]
    rows = []
    for label, patch in items:
        row = prepare_and_run(
            source_config,
            out_root,
            "module",
            label,
            device=device,
            reuse_existing=reuse_existing,
            patch=patch,
        )
        cfg = read_yaml(Path(row["config_path"]))
        row["moe_enable"] = cfg.get("moe", {}).get("enable", "")
        row["dynamic_lambda_enable"] = cfg.get("moe", {}).get("dynamic_lambda", {}).get("enable", "")
        row["pred_side_residual_enable"] = cfg.get("moe", {}).get("pred_side_residual", {}).get("enable", "")
        rows.append(row)
    ref = next((r for r in rows if r["label"] == "moe_off" and r["status"] == "ok"), None)
    if ref:
        ref_mse = float(ref["test_mse"])
        for row in rows:
            if row["status"] == "ok":
                row["gain_vs_moe_off_pct"] = (ref_mse - float(row["test_mse"])) / ref_mse * 100.0
    write_csv(
        out_root / "module_results.csv",
        rows,
        [
            "status",
            "label",
            "moe_enable",
            "dynamic_lambda_enable",
            "pred_side_residual_enable",
            "test_mse",
            "test_mae",
            "val_mse",
            "val_mae",
            "gain_vs_moe_off_pct",
            "best_epoch",
            "config_path",
            "out_dir",
            "returncode",
        ],
    )
    return rows


def rerun_detach(out_root: Path, device: str | None, reuse_existing: bool, source_config: Path) -> list[dict[str, Any]]:
    items = [
        ("no_detach", {}),
        ("detach_penalty_grad", {"moe": {"detach_penalty_grad": True}}),
        ("detach_routed_penalty_pred", {"moe": {"pred_side_residual": {"detach_routed_penalty_pred": True}}}),
        (
            "detach_both",
            {"moe": {"detach_penalty_grad": True, "pred_side_residual": {"detach_routed_penalty_pred": True}}},
        ),
    ]
    rows = []
    for label, patch in items:
        row = prepare_and_run(
            source_config,
            out_root,
            "detach",
            label,
            device=device,
            reuse_existing=reuse_existing,
            patch=patch,
        )
        cfg = read_yaml(Path(row["config_path"]))
        row["detach_penalty_grad"] = cfg.get("moe", {}).get("detach_penalty_grad", "")
        row["detach_routed_penalty_pred"] = cfg.get("moe", {}).get("pred_side_residual", {}).get(
            "detach_routed_penalty_pred", ""
        )
        rows.append(row)
    ref = next((r for r in rows if r["label"] == "no_detach" and r["status"] == "ok"), None)
    if ref:
        ref_mse = float(ref["test_mse"])
        for row in rows:
            if row["status"] == "ok":
                row["change_vs_no_detach_pct"] = (ref_mse - float(row["test_mse"])) / ref_mse * 100.0
    write_csv(
        out_root / "detach_results.csv",
        rows,
        [
            "status",
            "label",
            "detach_penalty_grad",
            "detach_routed_penalty_pred",
            "test_mse",
            "test_mae",
            "val_mse",
            "val_mae",
            "change_vs_no_detach_pct",
            "best_epoch",
            "config_path",
            "out_dir",
            "returncode",
        ],
    )
    return rows


def rerun_backbone(out_root: Path, device: str | None, reuse_existing: bool, source_config: Path) -> list[dict[str, Any]]:
    pairs = [
        (
            "mlp",
            {"model": {"predictor": "mlp"}},
            {"model": {"predictor": "mlp"}},
        ),
        (
            "dlinear_k25",
            {"model": {"predictor": "dlinear", "dlinear_kernel_size": 25}},
            {"model": {"predictor": "dlinear", "dlinear_kernel_size": 25}},
        ),
        (
            "dlinear_k13",
            {"model": {"predictor": "dlinear", "dlinear_kernel_size": 13}},
            {"model": {"predictor": "dlinear", "dlinear_kernel_size": 13}},
        ),
        (
            "nlinear",
            {"model": {"predictor": "nlinear"}},
            {"model": {"predictor": "nlinear"}},
        ),
    ]
    rows = []
    for name, off_patch, on_patch in pairs:
        off_patch = deep_update(copy.deepcopy(off_patch), disable_moe_patch())
        off = prepare_and_run(
            source_config,
            out_root,
            "backbone",
            f"{name}_off",
            device=device,
            reuse_existing=reuse_existing,
            patch=off_patch,
        )
        on = prepare_and_run(
            source_config,
            out_root,
            "backbone",
            f"{name}_on",
            device=device,
            reuse_existing=reuse_existing,
            patch=on_patch,
        )
        row = {
            "backbone": name,
            "status": "ok" if off["status"] == "ok" and on["status"] == "ok" else "failed",
            "moe_off_test_mse": off.get("test_mse", ""),
            "moe_on_test_mse": on.get("test_mse", ""),
            "moe_off_test_mae": off.get("test_mae", ""),
            "moe_on_test_mae": on.get("test_mae", ""),
            "moe_off_val_mse": off.get("val_mse", ""),
            "moe_on_val_mse": on.get("val_mse", ""),
            "moe_off_best_epoch": off.get("best_epoch", ""),
            "moe_on_best_epoch": on.get("best_epoch", ""),
            "moe_off_config": off.get("config_path", ""),
            "moe_on_config": on.get("config_path", ""),
            "moe_off_run": off.get("out_dir", ""),
            "moe_on_run": on.get("out_dir", ""),
        }
        if row["status"] == "ok":
            off_mse = float(row["moe_off_test_mse"])
            on_mse = float(row["moe_on_test_mse"])
            row["relative_gain_pct"] = (off_mse - on_mse) / off_mse * 100.0
        rows.append(row)
    write_csv(
        out_root / "backbone_results.csv",
        rows,
        [
            "status",
            "backbone",
            "moe_off_test_mse",
            "moe_on_test_mse",
            "relative_gain_pct",
            "moe_off_test_mae",
            "moe_on_test_mae",
            "moe_off_val_mse",
            "moe_on_val_mse",
            "moe_off_best_epoch",
            "moe_on_best_epoch",
            "moe_off_config",
            "moe_on_config",
            "moe_off_run",
            "moe_on_run",
        ],
    )
    return rows


def rerun_seeds(
    out_root: Path,
    device: str | None,
    reuse_existing: bool,
    seeds: list[int],
    source_config: Path,
) -> list[dict[str, Any]]:
    rows = []
    for seed in seeds:
        row = prepare_and_run(
            source_config,
            out_root,
            "multiseed",
            f"seed_{seed}",
            device=device,
            reuse_existing=reuse_existing,
            seed=seed,
        )
        row["seed"] = seed
        rows.append(row)
    ok_rows = [r for r in rows if r["status"] == "ok"]
    if ok_rows:
        mses = [float(r["test_mse"]) for r in ok_rows]
        maes = [float(r["test_mae"]) for r in ok_rows]
        summary = {
            "dataset": "ETTm1",
            "seeds": ",".join(str(s) for s in seeds),
            "count": len(ok_rows),
            "test_mse_mean": sum(mses) / len(mses),
            "test_mse_min": min(mses),
            "test_mse_max": max(mses),
            "test_mae_mean": sum(maes) / len(maes),
            "test_mae_min": min(maes),
            "test_mae_max": max(maes),
        }
        if len(mses) > 1:
            import statistics

            summary["test_mse_std"] = statistics.stdev(mses)
            summary["test_mae_std"] = statistics.stdev(maes)
        write_csv(
            out_root / "seed_summary.csv",
            [summary],
            [
                "dataset",
                "seeds",
                "count",
                "test_mse_mean",
                "test_mse_std",
                "test_mse_min",
                "test_mse_max",
                "test_mae_mean",
                "test_mae_std",
                "test_mae_min",
                "test_mae_max",
            ],
        )
    write_csv(
        out_root / "seed_results.csv",
        rows,
        [
            "status",
            "seed",
            "test_mse",
            "test_mae",
            "val_mse",
            "val_mae",
            "best_epoch",
            "config_path",
            "out_dir",
            "returncode",
        ],
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=ROOT / "outputs" / "paper_rerun_ettm1_md_experiments")
    parser.add_argument(
        "--groups",
        nargs="+",
        choices=["module", "detach", "backbone", "seeds"],
        default=["module", "detach", "backbone", "seeds"],
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2024, 2025, 2026, 2027, 2028])
    parser.add_argument("--source-config", type=Path, default=None)
    parser.add_argument("--reuse-existing", action="store_true")
    args = parser.parse_args()

    source_config = args.source_config or current_h96_config()
    if not source_config.is_absolute():
        source_config = (ROOT / source_config).resolve()
    print(f"[source] {source_config}", flush=True)
    args.out_root.mkdir(parents=True, exist_ok=True)
    if "module" in args.groups:
        rerun_module(args.out_root, args.device, args.reuse_existing, source_config)
    if "detach" in args.groups:
        rerun_detach(args.out_root, args.device, args.reuse_existing, source_config)
    if "backbone" in args.groups:
        rerun_backbone(args.out_root, args.device, args.reuse_existing, source_config)
    if "seeds" in args.groups:
        rerun_seeds(args.out_root, args.device, args.reuse_existing, args.seeds, source_config)


if __name__ == "__main__":
    main()
