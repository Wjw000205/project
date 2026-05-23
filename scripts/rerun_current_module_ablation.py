from __future__ import annotations

import argparse
import copy
import csv
import json
import statistics
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


def deep_update(target: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            deep_update(target[key], value)
        else:
            target[key] = value
    return target


def current_cluster_moe_patch(args: argparse.Namespace) -> dict[str, Any]:
    """Default ablation subject: train-only cluster top-k penalty prior + residual MoE.

    This is the current module variant, not the older fixed penalty-pool MoE:
    train split penalty portraits define per-cluster allowed penalties; the gate
    still decides per-input routing/skip and residual participation strength.
    """
    prior_topk = max(1, int(args.prior_topk))
    return {
        "moe": {
            "enable": True,
            "topk": prior_topk,
            "select_ranks": [1],
            "allow_skip": True,
            "skip_cost": float(args.skip_cost),
            "skip_init_bias": float(args.skip_init_bias),
            "gate_route_on_penalty_only": True,
            "router_mode": "learned",
            "router_penalty_context_weight": 0.0,
            "router_detach_penalty_context": True,
            "cluster_penalty_prior": {
                "enable": True,
                "topk": prior_topk,
                "hard_topk": True,
                "temperature": float(args.prior_temperature),
                "smoothing": float(args.prior_smoothing),
                "use_normalized_penalty": True,
                "logit_strength": float(args.prior_logit_strength),
                "use_as_balance_target": False,
            },
            "explainability": {
                "enable": True,
                "splits": ["train", "val", "test"],
                "max_batches": 0,
            },
            "pred_side_residual": {
                "enable": True,
                "penalty_selector_enable": True,
                "selector_temperature": 1.0,
                "selector_use_cluster_context": True,
                "fusion_gate_enable": True,
                "fusion_init": float(args.fusion_init),
                "fusion_use_cluster_context": True,
                "intervention_enable": True,
                "intervention_init": float(args.intervention_init),
            },
        }
    }


def annotate_moe_fields(row: dict[str, Any]) -> None:
    cfg = read_yaml(Path(row["config_path"]))
    moe = cfg.get("moe", {}) or {}
    residual = moe.get("pred_side_residual", {}) or {}
    prior = moe.get("cluster_penalty_prior", {}) or {}
    row["moe_enable"] = moe.get("enable", "")
    row["dynamic_lambda_enable"] = (moe.get("dynamic_lambda", {}) or {}).get("enable", "")
    row["pred_side_residual_enable"] = residual.get("enable", "")
    row["cluster_penalty_prior_enable"] = prior.get("enable", "")
    row["cluster_penalty_prior_topk"] = prior.get("topk", "")
    row["penalty_selector_enable"] = residual.get("penalty_selector_enable", "")
    row["fusion_gate_enable"] = residual.get("fusion_gate_enable", "")
    row["allow_skip"] = moe.get("allow_skip", "")


def normalize_cfg(
    cfg: dict[str, Any],
    *,
    dataset: str,
    run_dir: Path,
    device: str | None,
    pred_len: int,
    input_len: int,
    epochs: int | None,
    batch_size: int | None,
) -> dict[str, Any]:
    cfg = copy.deepcopy(cfg)
    cfg.setdefault("exp", {})["out_dir"] = str(run_dir)
    cfg["exp"]["name"] = f"current_module_{dataset}_{run_dir.parent.name}_{run_dir.name}"
    if device:
        cfg["exp"]["device"] = str(device)

    cfg.setdefault("window", {})["input_len"] = int(input_len)
    cfg["window"]["pred_len"] = int(pred_len)
    cfg.setdefault("normalize", {})["train_only"] = True
    cfg.setdefault("cluster", {})["train_only"] = True
    cfg.setdefault("corr", {})["save_path"] = str(run_dir / "corr.npy")

    cfg.setdefault("plot", {})["enable"] = False
    cfg.setdefault("portrait", {})["enable"] = False
    cfg["portrait"]["out_dir"] = str(run_dir / "cluster_portraits")
    cfg.setdefault("knn_hybrid", {})["enable"] = False
    cfg["knn_hybrid"]["use_for_model_selection"] = False
    cfg["knn_hybrid"]["path"] = str(run_dir / "knn_shape_bank.pt")
    cfg.setdefault("calibration", {})["enable"] = False
    cfg.setdefault("eval", {})["skip_test"] = False
    cfg["eval"]["save_predictions"] = False

    cfg["memory"] = {
        "enable": False,
        "save_checkpoint": False,
        "path": str(run_dir / "cluster_memory.pt"),
        "checkpoint_path": str(run_dir / "best_checkpoint.pt"),
    }
    if epochs is not None:
        cfg.setdefault("train", {})["epochs"] = int(epochs)
        gate_cal = cfg.setdefault("moe", {}).setdefault("pred_side_residual", {}).setdefault("gate_calibrator", {})
        if "epochs" in gate_cal:
            gate_cal["epochs"] = min(int(gate_cal["epochs"]), int(epochs))
    if batch_size is not None:
        cfg.setdefault("train", {})["batch_size"] = int(batch_size)
    return cfg


def run_train(config_path: Path, *, reuse_existing: bool, python: str) -> int:
    cfg = read_yaml(config_path)
    out_dir = Path(cfg["exp"]["out_dir"])
    if reuse_existing and (out_dir / "run_summary.json").exists():
        print(f"[reuse] {out_dir}", flush=True)
        return 0
    cmd = [python, "-u", "-m", "src.train", "--config", str(config_path)]
    print(f"[run] {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, cwd=str(ROOT))
    return int(proc.returncode)


def metric_row(
    *,
    dataset: str,
    group: str,
    label: str,
    config_path: Path,
    run_dir: Path,
    returncode: int,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "dataset": dataset,
        "group": group,
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
        cfg_for_row = read_yaml(config_path)
        window_cfg = cfg_for_row.get("window", {}) or {}
        window_summary = summary.get("windowing", {}) or {}
        row.update(
            {
                "input_len": window_summary.get("input_len", window_cfg.get("input_len", "")),
                "pred_len": window_summary.get("pred_len", window_cfg.get("pred_len", "")),
                "val_mse": summary.get("val", {}).get("avg_mse", ""),
                "val_mae": summary.get("val", {}).get("avg_mae", ""),
                "test_mse": summary.get("test", {}).get("avg_mse", ""),
                "test_mae": summary.get("test", {}).get("avg_mae", ""),
                "best_epoch": ",".join(str(v) for v in summary.get("best_epoch", [])),
                "penalties": ",".join(str(v) for v in summary.get("penalty_names", [])),
            }
        )
    return row


def make_run(
    base_cfg: dict[str, Any],
    *,
    dataset: str,
    out_root: Path,
    group: str,
    label: str,
    patch: dict[str, Any],
    device: str | None,
    pred_len: int,
    input_len: int,
    epochs: int | None,
    batch_size: int | None,
    reuse_existing: bool,
    python: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    deep_update(cfg, patch)
    run_dir = out_root / "runs" / dataset / group / label
    cfg = normalize_cfg(
        cfg,
        dataset=dataset,
        run_dir=run_dir,
        device=device,
        pred_len=pred_len,
        input_len=input_len,
        epochs=epochs,
        batch_size=batch_size,
    )
    config_path = out_root / "configs" / dataset / group / f"{label}.yaml"
    write_yaml(config_path, cfg)
    code = run_train(config_path, reuse_existing=reuse_existing, python=python)
    return metric_row(
        dataset=dataset,
        group=group,
        label=label,
        config_path=config_path,
        run_dir=run_dir,
        returncode=code,
        extra=extra,
    )


def disable_moe_patch() -> dict[str, Any]:
    return {
        "moe": {
            "enable": False,
            "allow_skip": False,
            "dynamic_lambda": {"enable": False},
            "pred_side_residual": {"enable": False},
        }
    }


def zero_lambda_patch(cfg: dict[str, Any]) -> dict[str, Any]:
    penalties = list(cfg.get("penalties", {}).get("enabled", []))
    return {
        "moe": {
            "enable": True,
            "dynamic_lambda": {"enable": False},
            "lambda_init": {name: 0.0 for name in penalties},
            "lambda_min": {name: 0.0 for name in penalties},
            "pred_side_residual": {"enable": True},
        }
    }


def fixed_lambda_patch() -> dict[str, Any]:
    return {"moe": {"enable": True, "dynamic_lambda": {"enable": False}, "pred_side_residual": {"enable": True}}}


def penalty_loss_only_patch() -> dict[str, Any]:
    return {"moe": {"enable": True, "pred_side_residual": {"enable": False}}}


def module_runs(
    base_cfg: dict[str, Any],
    *,
    dataset: str,
    out_root: Path,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    specs = [
        ("moe_off", disable_moe_patch()),
        ("zero_lambda_residual", zero_lambda_patch(base_cfg)),
        ("fixed_lambda_residual", fixed_lambda_patch()),
        ("penalty_loss_only", penalty_loss_only_patch()),
        ("full_current", {}),
    ]
    rows = []
    for label, patch in specs:
        row = make_run(base_cfg, dataset=dataset, out_root=out_root, group="module", label=label, patch=patch, **run_kwargs(args))
        annotate_moe_fields(row)
        rows.append(row)
    add_gain(rows, ref_label="moe_off")
    return rows


def detach_runs(
    base_cfg: dict[str, Any],
    *,
    dataset: str,
    out_root: Path,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    specs = [
        ("no_detach", {}),
        ("detach_penalty_grad", {"moe": {"detach_penalty_grad": True}}),
        ("detach_routed_penalty_pred", {"moe": {"pred_side_residual": {"detach_routed_penalty_pred": True}}}),
        (
            "detach_both",
            {"moe": {"detach_penalty_grad": True, "pred_side_residual": {"detach_routed_penalty_pred": True}}},
        ),
    ]
    rows = []
    for label, patch in specs:
        row = make_run(base_cfg, dataset=dataset, out_root=out_root, group="detach", label=label, patch=patch, **run_kwargs(args))
        annotate_moe_fields(row)
        cfg = read_yaml(Path(row["config_path"]))
        row["detach_penalty_grad"] = cfg.get("moe", {}).get("detach_penalty_grad", "")
        row["detach_routed_penalty_pred"] = (
            cfg.get("moe", {}).get("pred_side_residual", {}).get("detach_routed_penalty_pred", "")
        )
        rows.append(row)
    add_gain(rows, ref_label="no_detach", gain_col="gain_vs_no_detach_pct")
    return rows


def backbone_runs(
    base_cfg: dict[str, Any],
    *,
    dataset: str,
    out_root: Path,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    specs = [
        ("mlp", {"model": {"predictor": "mlp"}}),
        ("nlinear", {"model": {"predictor": "nlinear"}}),
        ("dlinear_k25", {"model": {"predictor": "dlinear", "moving_avg": 25}}),
        ("dlinear_k13", {"model": {"predictor": "dlinear", "moving_avg": 13}}),
    ]
    rows = []
    for name, model_patch in specs:
        off_patch = copy.deepcopy(model_patch)
        deep_update(off_patch, disable_moe_patch())
        on_patch = copy.deepcopy(model_patch)
        off = make_run(
            base_cfg,
            dataset=dataset,
            out_root=out_root,
            group="backbone",
            label=f"{name}_off",
            patch=off_patch,
            **run_kwargs(args),
            extra={"backbone": name, "moe_state": "off"},
        )
        on = make_run(
            base_cfg,
            dataset=dataset,
            out_root=out_root,
            group="backbone",
            label=f"{name}_on",
            patch=on_patch,
            **run_kwargs(args),
            extra={"backbone": name, "moe_state": "on"},
        )
        row = {
            "dataset": dataset,
            "backbone": name,
            "status": "ok" if off["status"] == "ok" and on["status"] == "ok" else "failed",
            "moe_off_test_mse": off.get("test_mse", ""),
            "moe_on_test_mse": on.get("test_mse", ""),
            "moe_off_test_mae": off.get("test_mae", ""),
            "moe_on_test_mae": on.get("test_mae", ""),
            "moe_off_val_mse": off.get("val_mse", ""),
            "moe_on_val_mse": on.get("val_mse", ""),
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
    return rows


def seed_runs(
    base_cfg: dict[str, Any],
    *,
    dataset: str,
    out_root: Path,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    for seed in args.seeds:
        patch = {"exp": {"seed": int(seed)}, "cluster": {"random_state": int(seed)}}
        row = make_run(
            base_cfg,
            dataset=dataset,
            out_root=out_root,
            group="seeds",
            label=f"seed_{seed}",
            patch=patch,
            **run_kwargs(args),
            extra={"seed": seed},
        )
        rows.append(row)
    ok = [r for r in rows if r["status"] == "ok"]
    summary_rows = []
    if ok:
        mses = [float(r["test_mse"]) for r in ok]
        maes = [float(r["test_mae"]) for r in ok]
        summary_rows.append(
            {
                "dataset": dataset,
                "seeds": ",".join(str(s) for s in args.seeds),
                "count": len(ok),
                "test_mse_mean": sum(mses) / len(mses),
                "test_mse_std": statistics.stdev(mses) if len(mses) > 1 else 0.0,
                "test_mse_min": min(mses),
                "test_mse_max": max(mses),
                "test_mae_mean": sum(maes) / len(maes),
                "test_mae_std": statistics.stdev(maes) if len(maes) > 1 else 0.0,
                "test_mae_min": min(maes),
                "test_mae_max": max(maes),
            }
        )
    return rows, summary_rows


def run_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "device": args.device,
        "pred_len": args.pred_len,
        "input_len": args.input_len,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "reuse_existing": args.reuse_existing,
        "python": str(args.python),
    }


def add_gain(rows: list[dict[str, Any]], *, ref_label: str, gain_col: str = "gain_vs_ref_pct") -> None:
    ref = next((r for r in rows if r.get("label") == ref_label and r.get("status") == "ok"), None)
    if not ref:
        return
    ref_mse = float(ref["test_mse"])
    for row in rows:
        if row.get("status") == "ok":
            row[gain_col] = (ref_mse - float(row["test_mse"])) / ref_mse * 100.0


def config_path_for(dataset: str) -> Path:
    path = ROOT / "configs" / f"{dataset}.yaml"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def best_config_path_for(dataset: str, horizon: int, results_csv: Path) -> Path:
    csv_path = results_csv if results_csv.is_absolute() else ROOT / results_csv
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    candidates: list[dict[str, str]] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if str(row.get("dataset", "")) != str(dataset):
                continue
            if int(float(row.get("horizon", "nan"))) != int(horizon):
                continue
            if row.get("status", "ok") != "ok":
                continue
            candidates.append(row)

    if not candidates:
        raise ValueError(f"No ok best-result row for dataset={dataset}, horizon={horizon} in {csv_path}")

    best_rows = [r for r in candidates if str(r.get("is_best_for_cell", "")).lower() == "true"]
    row = best_rows[0] if best_rows else min(candidates, key=lambda r: float(r.get("test_mse", "inf")))
    path_text = row.get("config_path", "")
    if not path_text:
        raise ValueError(f"Selected best-result row has no config_path: {row}")
    path = Path(path_text)
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def source_config_path_for(dataset: str, args: argparse.Namespace) -> Path:
    if args.base_results_csv is not None:
        return best_config_path_for(dataset, args.pred_len, args.base_results_csv)
    return config_path_for(dataset)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["ETTm1", "ETTm2"])
    parser.add_argument("--groups", nargs="+", choices=["module", "detach", "backbone", "seeds"], default=["module"])
    parser.add_argument("--out-root", type=Path, default=ROOT / "outputs" / "current_module_ablation_rerun")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--input-len", type=int, default=336)
    parser.add_argument("--pred-len", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[2024, 2025, 2026, 2027, 2028])
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument(
        "--base-results-csv",
        type=Path,
        default=None,
        help="Use per-dataset best config_path from this best_results.csv instead of configs/{dataset}.yaml.",
    )
    parser.add_argument(
        "--preserve-config-moe",
        action="store_true",
        help="Use the YAML MoE section as-is instead of forcing the current cluster-prior MoE defaults.",
    )
    parser.add_argument("--prior-topk", type=int, default=1)
    parser.add_argument("--prior-temperature", type=float, default=0.7)
    parser.add_argument("--prior-smoothing", type=float, default=0.0)
    parser.add_argument("--prior-logit-strength", type=float, default=0.0)
    parser.add_argument("--fusion-init", type=float, default=-1.5)
    parser.add_argument("--intervention-init", type=float, default=-2.0)
    parser.add_argument("--skip-cost", type=float, default=0.15)
    parser.add_argument("--skip-init-bias", type=float, default=-2.0)
    args = parser.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    all_module: list[dict[str, Any]] = []
    all_detach: list[dict[str, Any]] = []
    all_backbone: list[dict[str, Any]] = []
    all_seed: list[dict[str, Any]] = []
    all_seed_summary: list[dict[str, Any]] = []

    for dataset in args.datasets:
        source_config_path = source_config_path_for(dataset, args)
        cfg = read_yaml(source_config_path)
        if not args.preserve_config_moe:
            deep_update(cfg, current_cluster_moe_patch(args))
        print(f"[dataset] {dataset} from {source_config_path}", flush=True)
        if "module" in args.groups:
            all_module.extend(module_runs(cfg, dataset=dataset, out_root=args.out_root, args=args))
        if "detach" in args.groups:
            all_detach.extend(detach_runs(cfg, dataset=dataset, out_root=args.out_root, args=args))
        if "backbone" in args.groups:
            all_backbone.extend(backbone_runs(cfg, dataset=dataset, out_root=args.out_root, args=args))
        if "seeds" in args.groups:
            seed_rows, summary_rows = seed_runs(cfg, dataset=dataset, out_root=args.out_root, args=args)
            all_seed.extend(seed_rows)
            all_seed_summary.extend(summary_rows)

    common_fields = [
        "dataset",
        "group",
        "label",
        "status",
        "input_len",
        "pred_len",
        "test_mse",
        "test_mae",
        "val_mse",
        "val_mae",
        "best_epoch",
        "penalties",
        "config_path",
        "out_dir",
        "returncode",
    ]
    if all_module:
        write_csv(
            args.out_root / "module_results.csv",
            all_module,
            common_fields
            + [
                "moe_enable",
                "dynamic_lambda_enable",
                "pred_side_residual_enable",
                "cluster_penalty_prior_enable",
                "cluster_penalty_prior_topk",
                "penalty_selector_enable",
                "fusion_gate_enable",
                "allow_skip",
                "gain_vs_ref_pct",
            ],
        )
    if all_detach:
        write_csv(
            args.out_root / "detach_results.csv",
            all_detach,
            common_fields
            + [
                "cluster_penalty_prior_enable",
                "cluster_penalty_prior_topk",
                "penalty_selector_enable",
                "fusion_gate_enable",
                "allow_skip",
                "detach_penalty_grad",
                "detach_routed_penalty_pred",
                "gain_vs_no_detach_pct",
            ],
        )
    if all_backbone:
        write_csv(
            args.out_root / "backbone_results.csv",
            all_backbone,
            [
                "dataset",
                "backbone",
                "status",
                "moe_off_test_mse",
                "moe_on_test_mse",
                "relative_gain_pct",
                "moe_off_test_mae",
                "moe_on_test_mae",
                "moe_off_val_mse",
                "moe_on_val_mse",
                "moe_off_config",
                "moe_on_config",
                "moe_off_run",
                "moe_on_run",
            ],
        )
    if all_seed:
        write_csv(args.out_root / "seed_results.csv", all_seed, common_fields + ["seed"])
    if all_seed_summary:
        write_csv(
            args.out_root / "seed_summary.csv",
            all_seed_summary,
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
    print(f"Saved current-module ablation outputs under {args.out_root}", flush=True)


if __name__ == "__main__":
    main()
