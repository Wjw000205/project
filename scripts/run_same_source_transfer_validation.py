from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]


DOMAIN_SPECS: dict[str, dict[str, Any]] = {
    "weather": {
        "base_config": ROOT / "configs" / "weather.yaml",
        "csv": ROOT / "data" / "weather.csv",
        "max_rows": 30000,
        "source_value_idx": list(range(0, 11)),
        "target_value_idx": list(range(11, 21)),
        "description": "Weather same-station variable groups",
    },
    "electricity": {
        "base_config": ROOT / "configs" / "electricity.yaml",
        "csv": ROOT / "data" / "electricity.csv",
        "max_rows": 26304,
        "source_value_idx": list(range(0, 32)),
        "target_value_idx": list(range(32, 64)),
        "description": "Electricity same-dataset customer groups",
    },
    "traffic": {
        "base_config": ROOT / "configs" / "traffic.yaml",
        "csv": ROOT / "data" / "traffic.csv",
        "max_rows": 17544,
        "source_value_idx": list(range(0, 32)),
        "target_value_idx": list(range(32, 64)),
        "description": "Traffic same-dataset sensor groups",
    },
}


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_yaml(path: Path, cfg: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "status",
        "domain",
        "description",
        "source_channels",
        "target_channels",
        "input_len",
        "pred_len",
        "epochs",
        "source_test_mse",
        "source_test_mae",
        "target_base_test_mse",
        "target_base_test_mae",
        "transfer_test_mse",
        "transfer_test_mae",
        "transfer_delta_mse_vs_target_base",
        "transfer_gain_mse_vs_target_base",
        "transfer_delta_mae_vs_target_base",
        "transfer_gain_mae_vs_target_base",
        "route_fit_scope",
        "normalize_train_only",
        "route_uses_train_only",
        "eval_uses_test_only",
        "predictor_variant",
        "corr_mode",
        "penalty_names",
        "cluster_counts",
        "corr_max_mean",
        "corr_max_min",
        "source_out_dir",
        "target_base_out_dir",
        "transfer_out_dir",
        "error",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def load_summary(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def metric(summary: dict[str, Any], key: str) -> float | None:
    if key in summary:
        return float(summary[key])
    test = summary.get("test", {}) or {}
    if key == "avg_mse" and "avg_mse" in test:
        return float(test["avg_mse"])
    if key == "avg_mae" and "avg_mae" in test:
        return float(test["avg_mae"])
    return None


def patch_train_cfg(
    cfg: dict[str, Any],
    *,
    name: str,
    csv_path: Path,
    out_dir: Path,
    input_len: int,
    pred_len: int,
    epochs: int,
    device: str,
    batch_size: int,
) -> dict[str, Any]:
    cfg = json.loads(json.dumps(cfg))
    cfg.setdefault("exp", {})
    cfg["exp"]["name"] = name
    cfg["exp"]["out_dir"] = str(out_dir)
    cfg["exp"]["device"] = device
    cfg["exp"]["deterministic"] = True

    cfg.setdefault("data", {})
    cfg["data"]["csv_path"] = str(csv_path)
    cfg["data"]["date_col"] = 0
    cfg["data"].pop("max_rows", None)
    cfg["data"]["train_ratio"] = 0.7
    cfg["data"]["val_ratio"] = 0.1
    cfg["data"]["test_ratio"] = 0.2

    cfg.setdefault("window", {})
    cfg["window"]["input_len"] = input_len
    cfg["window"]["pred_len"] = pred_len

    cfg.setdefault("normalize", {})
    cfg["normalize"]["global_zscore"] = True
    cfg["normalize"]["train_only"] = True

    cfg.setdefault("corr", {})
    cfg["corr"]["compute"] = True
    cfg["corr"]["save_path"] = str(out_dir / "corr.npy")

    cfg.setdefault("cluster", {})
    cfg["cluster"]["train_only"] = True
    cfg["cluster"]["n_clusters"] = min(3, max(1, cfg["cluster"].get("n_clusters", 3)))
    cfg["cluster"]["no_merge_if_channels_lt"] = 7

    cfg.setdefault("model", {})
    cfg["model"]["predictor"] = "mlp"
    cfg["model"]["hidden_dim"] = int(cfg["model"].get("hidden_dim", 256))
    cfg["model"]["dropout"] = float(cfg["model"].get("dropout", 0.2))

    cfg.setdefault("penalties", {})
    cfg["penalties"]["enabled"] = ["level", "delta", "d2_match", "diff_amp"]
    cfg["penalties"]["jump_threshold"] = 0.6

    cfg.setdefault("moe", {})
    cfg["moe"]["enable"] = True
    cfg["moe"]["topk"] = 1
    cfg["moe"]["gate_hidden_dim"] = 32
    cfg["moe"]["select_ranks"] = [1]
    cfg["moe"]["lambda_init"] = {
        "level": 0.1,
        "delta": 0.1,
        "d2_match": 0.1,
        "diff_amp": 0.1,
    }
    cfg["moe"]["lambda_min"] = {name: 0.0 for name in cfg["penalties"]["enabled"]}
    cfg["moe"]["lambda_schedule"] = {name: "none" for name in cfg["penalties"]["enabled"]}
    cfg["moe"]["gate_entropy_weight"] = 0.0
    cfg["moe"]["gate_balance_weight"] = 0.0
    cfg["moe"]["router_mode"] = "learned"
    cfg["moe"]["gate_route_on_penalty_only"] = True
    cfg["moe"]["allow_skip"] = True
    cfg["moe"].setdefault("dynamic_lambda", {})
    cfg["moe"]["dynamic_lambda"].update(
        {
            "enable": True,
            "mode": "multiscale",
            "hidden_dim": 32,
            "segment_bins": [4, 8],
            "max_factor": 1.5,
            "mix": 0.6,
            "dropout": 0.0,
            "reg_weight": 1.0e-4,
        }
    )
    cfg["moe"].setdefault("pred_side_residual", {})
    cfg["moe"]["pred_side_residual"].update(
        {
            "enable": True,
            "feature_mode": "legacy",
            "residual_clip": 0.0,
            "corrector_hidden": 32,
            "init_alpha": -3.0,
            "alpha_scale": 1.1,
            "specialization_weight": 0.1,
            "norm_weight": 0.0,
            "use_y_base_input": True,
            "selection_policy": "val_mse_gate",
            "selection_min_abs_improvement": 0.0,
            "selection_min_rel_improvement": 0.0,
            "gate_calibrator": {
                "loss": "mse",
                "selection_metric": "mse",
                "epochs": 20,
                "train_fraction": 0.7,
                "hidden_dim": 32,
                "batch_size": 256,
                "max_scale": 1.0,
                "init_scale": 0.8,
                "scale_reg": 1.0e-4,
            },
        }
    )

    cfg.setdefault("train", {})
    cfg["train"]["epochs"] = epochs
    cfg["train"]["batch_size"] = batch_size
    cfg["train"]["selection_metric"] = "val_mse"
    cfg["train"]["mse_weight"] = 0.9
    cfg["train"]["weight_decay"] = 1.0e-4
    cfg["train"]["penalty_warmup_epochs"] = min(10, max(1, epochs // 4))
    cfg["train"].setdefault("mae_objective", {})
    cfg["train"]["mae_objective"].update(
        {"enable": True, "kind": "l1", "weight": 0.6, "warmup_epochs": 5}
    )

    cfg.setdefault("early_stop", {})
    cfg["early_stop"]["patience"] = min(8, max(3, epochs // 3))
    cfg["early_stop"]["min_delta"] = 1.0e-6

    cfg.setdefault("plot", {})
    cfg["plot"]["enable"] = False
    cfg.setdefault("portrait", {})
    cfg["portrait"]["enable"] = False
    cfg.setdefault("knn_hybrid", {})
    cfg["knn_hybrid"]["enable"] = False
    cfg.setdefault("eval", {})
    cfg["eval"]["skip_test"] = False

    cfg.setdefault("memory", {})
    cfg["memory"]["enable"] = True
    cfg["memory"]["save_checkpoint"] = True
    cfg["memory"]["path"] = str(out_dir / "cluster_memory.pt")
    cfg["memory"]["checkpoint_path"] = str(out_dir / "best_checkpoint.pt")
    return cfg


def make_transfer_cfg(
    *,
    domain: str,
    source_out: Path,
    source_csv: Path,
    target_csv: Path,
    transfer_out: Path,
    input_len: int,
    pred_len: int,
    device: str,
    batch_size: int,
) -> dict[str, Any]:
    return {
        "exp": {
            "name": f"{domain}_same_source_transfer",
            "out_dir": str(transfer_out),
            "seed": 2026,
            "deterministic": True,
            "device": device,
        },
        "source": {
            "memory_path": str(source_out / "cluster_memory.pt"),
            "checkpoint_path": str(source_out / "best_checkpoint.pt"),
            "summary_path": str(source_out / "run_summary.json"),
            "csv_path": str(source_csv),
            "date_col": 0,
        },
        "data": {
            "csv_path": str(target_csv),
            "date_col": 0,
            "train_ratio": 0.7,
            "val_ratio": 0.1,
            "test_ratio": 0.2,
        },
        "window": {"input_len": input_len, "pred_len": pred_len},
        "normalize": {"global_zscore": True, "train_only": True},
        "transfer": {
            "corr_mode": "cycle_template",
            "route_fit_scope": "train",
            "use_pred_residual": True,
            "phase_bins": 64,
            "phase_max_shift": None,
            "period_min": None,
            "period_max": None,
            "period_min_hours": 12,
            "period_max_hours": 168,
            "corr_align": "head",
            "corr_threshold": None,
            "fallback_mode": "hard",
            "fallback_topk": 2,
            "fallback_temp": 1.0,
            "resample": {
                "enable": False,
                "target_step_minutes": None,
                "method": "linear",
            },
            "knn_hybrid": {
                "enable": False,
                "scope": "same_cluster",
                "bank_split": "train",
                "use_for_model_selection": False,
                "k": 16,
                "alpha": 0.1,
                "adaptive_alpha": "confidence",
                "confidence_floor": 0.0,
                "distance_sharpness": 1.0,
                "shape_bins": 24,
                "diff_bins": 12,
                "bank_stride": 4,
                "distance_weight": "inverse",
                "anchor_mode": "last",
            },
            "save_corr": True,
        },
        "eval": {"batch_size": batch_size},
    }


def run_cmd(cmd: list[str], *, cwd: Path, reuse_path: Path | None = None) -> tuple[int, float, str]:
    if reuse_path is not None and reuse_path.exists():
        return 0, 0.0, "reused"
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    start = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(cwd), env=env)
    return proc.returncode, time.perf_counter() - start, ""


def split_domain_data(domain: str, spec: dict[str, Any], out_root: Path) -> tuple[Path, Path, int, int]:
    df = pd.read_csv(spec["csv"])
    max_rows = int(spec.get("max_rows", 0) or 0)
    if max_rows > 0:
        df = df.iloc[:max_rows].copy()
    date_col = df.columns[0]
    value_cols = [c for c in df.columns if c != date_col]
    src_cols = [value_cols[i] for i in spec["source_value_idx"] if i < len(value_cols)]
    tgt_cols = [value_cols[i] for i in spec["target_value_idx"] if i < len(value_cols)]
    if len(src_cols) == 0 or len(tgt_cols) == 0:
        raise ValueError(f"{domain}: empty source or target channel selection")
    data_dir = out_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    source_csv = data_dir / f"{domain}_source.csv"
    target_csv = data_dir / f"{domain}_target.csv"
    df[[date_col, *src_cols]].to_csv(source_csv, index=False)
    df[[date_col, *tgt_cols]].to_csv(target_csv, index=False)
    return source_csv, target_csv, len(src_cols), len(tgt_cols)


def spec_with_overrides(
    spec: dict[str, Any],
    *,
    max_rows: int | None,
    channels_per_side: int | None,
) -> dict[str, Any]:
    spec = dict(spec)
    if max_rows is not None and max_rows > 0:
        spec["max_rows"] = int(max_rows)
    if channels_per_side is not None and channels_per_side > 0:
        n = int(channels_per_side)
        spec["source_value_idx"] = list(range(0, n))
        spec["target_value_idx"] = list(range(n, 2 * n))
    return spec


def collect_assignment_stats(path: Path) -> dict[str, Any]:
    assign_path = path / "cluster_assignment.csv"
    if not assign_path.exists():
        return {}
    df = pd.read_csv(assign_path)
    counts = df["cluster_id"].value_counts().sort_index().to_dict()
    return {
        "cluster_counts": json.dumps({int(k): int(v) for k, v in counts.items()}, ensure_ascii=False),
        "corr_max_mean": float(df["corr_max"].mean()),
        "corr_max_min": float(df["corr_max"].min()),
    }


def run_domain(
    domain: str,
    *,
    spec: dict[str, Any],
    out_root: Path,
    input_len: int,
    pred_len: int,
    epochs: int,
    device: str,
    batch_size: int,
    reuse_existing: bool,
    skip_target_base: bool,
) -> dict[str, Any]:
    domain_root = out_root / domain
    cfg_dir = domain_root / "configs"
    runs_dir = domain_root / "runs"
    source_csv, target_csv, source_channels, target_channels = split_domain_data(domain, spec, domain_root)
    base_cfg = read_yaml(Path(spec["base_config"]))

    source_out = runs_dir / "source_train"
    source_cfg_path = cfg_dir / "source_train.yaml"
    source_cfg = patch_train_cfg(
        base_cfg,
        name=f"{domain}_same_source_source",
        csv_path=source_csv,
        out_dir=source_out,
        input_len=input_len,
        pred_len=pred_len,
        epochs=epochs,
        device=device,
        batch_size=batch_size,
    )
    write_yaml(source_cfg_path, source_cfg)

    target_out = runs_dir / "target_base"
    target_cfg_path = cfg_dir / "target_base.yaml"
    if not skip_target_base:
        target_cfg = patch_train_cfg(
            base_cfg,
            name=f"{domain}_same_source_target_base",
            csv_path=target_csv,
            out_dir=target_out,
            input_len=input_len,
            pred_len=pred_len,
            epochs=epochs,
            device=device,
            batch_size=batch_size,
        )
        write_yaml(target_cfg_path, target_cfg)

    transfer_out = runs_dir / "source_to_target_transfer"
    transfer_cfg_path = cfg_dir / "source_to_target_transfer.yaml"
    transfer_cfg = make_transfer_cfg(
        domain=domain,
        source_out=source_out,
        source_csv=source_csv,
        target_csv=target_csv,
        transfer_out=transfer_out,
        input_len=input_len,
        pred_len=pred_len,
        device=device,
        batch_size=batch_size,
    )
    write_yaml(transfer_cfg_path, transfer_cfg)

    row: dict[str, Any] = {
        "status": "ok",
        "domain": domain,
        "description": spec["description"],
        "source_channels": source_channels,
        "target_channels": target_channels,
        "input_len": input_len,
        "pred_len": pred_len,
        "epochs": epochs,
        "source_out_dir": str(source_out),
        "target_base_out_dir": "" if skip_target_base else str(target_out),
        "transfer_out_dir": str(transfer_out),
    }
    try:
        rc, _, _ = run_cmd(
            [sys.executable, "-m", "src.train", "--config", str(source_cfg_path)],
            cwd=ROOT,
            reuse_path=(source_out / "run_summary.json") if reuse_existing else None,
        )
        if rc != 0:
            raise RuntimeError(f"source train failed with code {rc}")
        source_summary = load_summary(source_out / "run_summary.json")
        row["source_test_mse"] = metric(source_summary, "avg_mse")
        row["source_test_mae"] = metric(source_summary, "avg_mae")

        if not skip_target_base:
            rc, _, _ = run_cmd(
                [sys.executable, "-m", "src.train", "--config", str(target_cfg_path)],
                cwd=ROOT,
                reuse_path=(target_out / "run_summary.json") if reuse_existing else None,
            )
            if rc != 0:
                raise RuntimeError(f"target base train failed with code {rc}")
            target_summary = load_summary(target_out / "run_summary.json")
            row["target_base_test_mse"] = metric(target_summary, "avg_mse")
            row["target_base_test_mae"] = metric(target_summary, "avg_mae")

        rc, _, _ = run_cmd(
            [sys.executable, "-m", "src.transfer", "--config", str(transfer_cfg_path)],
            cwd=ROOT,
            reuse_path=(transfer_out / "transfer_summary.json") if reuse_existing else None,
        )
        if rc != 0:
            raise RuntimeError(f"transfer failed with code {rc}")
        transfer_summary = load_summary(transfer_out / "transfer_summary.json")
        row["transfer_test_mse"] = transfer_summary.get("avg_mse")
        row["transfer_test_mae"] = transfer_summary.get("avg_mae")
        row["route_fit_scope"] = transfer_summary.get("route_fit_scope")
        row["normalize_train_only"] = transfer_summary.get("normalize_train_only")
        row["route_uses_train_only"] = transfer_summary.get("route_uses_train_only")
        row["eval_uses_test_only"] = transfer_summary.get("eval_uses_test_only")
        row["predictor_variant"] = transfer_summary.get("predictor_variant")
        row["corr_mode"] = transfer_summary.get("corr_mode")
        row["penalty_names"] = json.dumps(transfer_summary.get("penalty_names", []), ensure_ascii=False)
        row.update(collect_assignment_stats(transfer_out))
        if row.get("target_base_test_mse") not in ("", None):
            row["transfer_delta_mse_vs_target_base"] = (
                float(row["transfer_test_mse"]) - float(row["target_base_test_mse"])
            )
            row["transfer_gain_mse_vs_target_base"] = -float(row["transfer_delta_mse_vs_target_base"])
        if row.get("target_base_test_mae") not in ("", None):
            row["transfer_delta_mae_vs_target_base"] = (
                float(row["transfer_test_mae"]) - float(row["target_base_test_mae"])
            )
            row["transfer_gain_mae_vs_target_base"] = -float(row["transfer_delta_mae_vs_target_base"])
    except Exception as exc:
        row["status"] = "failed"
        row["error"] = str(exc)
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default=str(ROOT / "outputs" / "same_source_transfer_validation"))
    ap.add_argument("--domains", nargs="+", default=["weather", "electricity", "traffic"])
    ap.add_argument("--input-len", type=int, default=336)
    ap.add_argument("--pred-len", type=int, default=96)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-rows", type=int, default=0, help="Override per-domain row limit for quick pilot runs.")
    ap.add_argument(
        "--channels-per-side",
        type=int,
        default=0,
        help="Override source/target channel count as first N vs next N value columns.",
    )
    ap.add_argument("--reuse-existing", action="store_true")
    ap.add_argument("--skip-target-base", action="store_true")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    rows: list[dict[str, Any]] = []
    for domain in args.domains:
        if domain not in DOMAIN_SPECS:
            raise ValueError(f"Unknown domain '{domain}'. Choices: {sorted(DOMAIN_SPECS)}")
        spec = spec_with_overrides(
            DOMAIN_SPECS[domain],
            max_rows=int(args.max_rows) if int(args.max_rows) > 0 else None,
            channels_per_side=int(args.channels_per_side) if int(args.channels_per_side) > 0 else None,
        )
        print(f"[same-source-transfer] {domain}", flush=True)
        row = run_domain(
            domain,
            spec=spec,
            out_root=out_root,
            input_len=int(args.input_len),
            pred_len=int(args.pred_len),
            epochs=int(args.epochs),
            device=str(args.device),
            batch_size=int(args.batch_size),
            reuse_existing=bool(args.reuse_existing),
            skip_target_base=bool(args.skip_target_base),
        )
        rows.append(row)
        write_rows(out_root / "transfer.csv", rows)
        print(
            f"[same-source-transfer] {domain} {row.get('status')} "
            f"transfer_mse={row.get('transfer_test_mse', '')}",
            flush=True,
        )
    write_rows(out_root / "transfer.csv", rows)
    print(f"Saved: {out_root / 'transfer.csv'}")


if __name__ == "__main__":
    main()
