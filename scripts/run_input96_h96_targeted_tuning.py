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

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_all_datasets_horizons import apply_tsl_alignment, ensure_local_output_paths


DATASET_CONFIGS = {
    "ETTh1": "configs/ETTh1.yaml",
    "ETTh2": "configs/ETTh2.yaml",
    "ETTm1": "configs/ETTm1.yaml",
    "ETTm2": "configs/ETTm2.yaml",
    "weather": "configs/weather.yaml",
    "electricity": "configs/electricity.yaml",
}

MLP_FAMILY = {"mlp", "cluster_mlp", "context_mlp", "attn_mlp", "channel_head_mlp", "channel_mlp"}


@dataclass(frozen=True)
class Candidate:
    stage: str
    variant: str
    patch: dict[str, Any]


FIELDS = [
    "dataset",
    "stage",
    "variant",
    "status",
    "val_mse",
    "val_mae",
    "test_mse",
    "test_mae",
    "best_epoch",
    "model_hidden_dim",
    "model_dropout",
    "lr",
    "weight_decay",
    "mae_weight",
    "moe_enable",
    "penalties",
    "lambda_init",
    "dynamic_lambda",
    "alpha_scale",
    "selection_policy",
    "config_path",
    "out_dir",
    "total_sec",
    "returncode",
    "error",
]


def resolve(path_text: str | Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_yaml(path: Path, cfg: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=False, sort_keys=False)


def deep_update(dst: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            deep_update(dst[key], value)
        else:
            dst[key] = value
    return dst


def set_moe_off(cfg: dict[str, Any]) -> None:
    moe = cfg.setdefault("moe", {})
    moe["enable"] = False
    moe.setdefault("dynamic_lambda", {})["enable"] = False
    moe.setdefault("pred_side_residual", {})["enable"] = False


def lambda_dict(penalties: list[str], value: float) -> dict[str, float]:
    return {name: float(value) for name in penalties}


def schedule_dict(penalties: list[str], value: str = "none") -> dict[str, str]:
    return {name: value for name in penalties}


def common_prepare(
    cfg: dict[str, Any],
    *,
    dataset: str,
    out_dir: Path,
    run_name: str,
    device: str | None,
    epochs: int | None,
    skip_test: bool,
) -> dict[str, Any]:
    cfg = copy.deepcopy(cfg)
    apply_tsl_alignment(cfg, dataset)
    cfg.setdefault("window", {})
    cfg["window"]["input_len"] = 96
    cfg["window"]["pred_len"] = 96
    cfg["window"]["past_context"] = True
    cfg.setdefault("normalize", {})["train_only"] = True
    cfg.setdefault("cluster", {})["train_only"] = True
    cfg.setdefault("eval", {})["skip_test"] = bool(skip_test)
    cfg.setdefault("calibration", {})["enable"] = False
    cfg.setdefault("knn_hybrid", {})["enable"] = False
    cfg.setdefault("plot", {})["enable"] = False
    cfg.setdefault("portrait", {})["enable"] = False
    cfg.setdefault("memory", {})["enable"] = False
    cfg.setdefault("memory", {})["save_checkpoint"] = False
    if epochs is not None:
        cfg.setdefault("train", {})["epochs"] = int(epochs)
    if device:
        cfg.setdefault("exp", {})["device"] = str(device)
    ensure_local_output_paths(
        cfg,
        out_dir=out_dir,
        run_name=run_name,
        keep_artifacts=False,
        disable_knn_hybrid=True,
        knn_adaptive_alpha=None,
        knn_selection_policy=None,
        knn_selection_min_rel_improvement=None,
        knn_selection_min_abs_improvement=None,
    )
    return cfg


def model_candidates() -> list[Candidate]:
    return [
        Candidate("model", "current_model", {}),
        Candidate(
            "model",
            "current_mae02",
            {
                "train": {"mae_objective": {"enable": True, "kind": "l1", "weight": 0.2}},
            },
        ),
        Candidate(
            "model",
            "current_mseonly",
            {
                "train": {"mae_objective": {"enable": False, "weight": 0.0}},
            },
        ),
        Candidate(
            "model",
            "mlp_h128_do0_wd1e4_mae04",
            {
                "model": {"predictor": "mlp", "hidden_dim": 128, "dropout": 0.0},
                "train": {"weight_decay": 1.0e-4, "mae_objective": {"enable": True, "kind": "l1", "weight": 0.4}},
            },
        ),
        Candidate(
            "model",
            "mlp_h128_do02_wd1e4_mae04",
            {
                "model": {"predictor": "mlp", "hidden_dim": 128, "dropout": 0.2},
                "train": {"weight_decay": 1.0e-4, "mae_objective": {"enable": True, "kind": "l1", "weight": 0.4}},
            },
        ),
        Candidate(
            "model",
            "mlp_h128_do02_wd1e4_mae02",
            {
                "model": {"predictor": "mlp", "hidden_dim": 128, "dropout": 0.2},
                "train": {"weight_decay": 1.0e-4, "mae_objective": {"enable": True, "kind": "l1", "weight": 0.2}},
            },
        ),
        Candidate(
            "model",
            "mlp_h128_do02_wd1e4_mseonly",
            {
                "model": {"predictor": "mlp", "hidden_dim": 128, "dropout": 0.2},
                "train": {"weight_decay": 1.0e-4, "mae_objective": {"enable": False, "weight": 0.0}},
            },
        ),
        Candidate(
            "model",
            "mlp_h256_do02_wd1e3_mae06",
            {
                "model": {"predictor": "mlp", "hidden_dim": 256, "dropout": 0.2},
                "train": {"weight_decay": 1.0e-3, "mae_objective": {"enable": True, "kind": "l1", "weight": 0.6}},
            },
        ),
        Candidate(
            "model",
            "mlp_h192_do03_wd1e3_mae04",
            {
                "model": {"predictor": "mlp", "hidden_dim": 192, "dropout": 0.3},
                "train": {"weight_decay": 1.0e-3, "mae_objective": {"enable": True, "kind": "l1", "weight": 0.4}},
            },
        ),
        Candidate(
            "model",
            "mlp_h384_do01_wd1e3_mae04",
            {
                "model": {"predictor": "mlp", "hidden_dim": 384, "dropout": 0.1},
                "train": {"weight_decay": 1.0e-3, "mae_objective": {"enable": True, "kind": "l1", "weight": 0.4}},
            },
        ),
        Candidate(
            "model",
            "mlp_h512_do01_wd1e3_mae04",
            {
                "model": {"predictor": "mlp", "hidden_dim": 512, "dropout": 0.1},
                "train": {"weight_decay": 1.0e-3, "mae_objective": {"enable": True, "kind": "l1", "weight": 0.4}},
            },
        ),
        Candidate(
            "model",
            "context_h256_do02_wd1e3_mae04",
            {
                "model": {"predictor": "context_mlp", "hidden_dim": 256, "dropout": 0.2, "context_include_delta": True},
                "train": {"weight_decay": 1.0e-3, "mae_objective": {"enable": True, "kind": "l1", "weight": 0.4}},
            },
        ),
        Candidate(
            "model",
            "context_h512_do01_wd1e3_mae04",
            {
                "model": {"predictor": "context_mlp", "hidden_dim": 512, "dropout": 0.1, "context_include_delta": True},
                "train": {"weight_decay": 1.0e-3, "mae_objective": {"enable": True, "kind": "l1", "weight": 0.4}},
            },
        ),
        Candidate(
            "model",
            "channel_h256_do02_wd1e3_mae04",
            {
                "model": {"predictor": "channel_head_mlp", "hidden_dim": 256, "dropout": 0.2, "channel_head_residual": True},
                "train": {"weight_decay": 1.0e-3, "mae_objective": {"enable": True, "kind": "l1", "weight": 0.4}},
            },
        ),
        Candidate(
            "model",
            "channel_h512_do01_wd1e3_mae04",
            {
                "model": {"predictor": "channel_head_mlp", "hidden_dim": 512, "dropout": 0.1, "channel_head_residual": True},
                "train": {"weight_decay": 1.0e-3, "mae_objective": {"enable": True, "kind": "l1", "weight": 0.4}},
            },
        ),
        Candidate(
            "model",
            "attn_h256_do02_wd1e3_mae04",
            {
                "model": {"predictor": "attn_mlp", "hidden_dim": 256, "dropout": 0.2, "attn_dim": 64},
                "train": {"weight_decay": 1.0e-3, "mae_objective": {"enable": True, "kind": "l1", "weight": 0.4}},
            },
        ),
        Candidate(
            "model",
            "channel_h256_do02_wd1e3_thr05",
            {
                "cluster": {"distance_threshold": 0.5},
                "model": {"predictor": "channel_head_mlp", "hidden_dim": 256, "dropout": 0.2, "channel_head_residual": True},
                "train": {"weight_decay": 1.0e-3, "mae_objective": {"enable": True, "kind": "l1", "weight": 0.4}},
            },
        ),
    ]


def limit_candidates(candidates: list[Candidate], budget: str, *, smoke_n: int, local_n: int) -> list[Candidate]:
    if budget == "smoke":
        return candidates[:smoke_n]
    if budget == "local":
        return candidates[:local_n]
    return candidates


def moe_candidates(base_cfg: dict[str, Any], budget: str) -> list[Candidate]:
    current_penalties = list(base_cfg.get("penalties", {}).get("enabled", ["level"]))
    pools = [
        ("current_moe", current_penalties, None, True, None),
        ("level_delta_l015", ["level", "delta"], 0.015, True, 0.8),
        ("level_delta_diff_l015", ["level", "delta", "diff_amp"], 0.015, True, 0.8),
        ("level_delta_diff_l05", ["level", "delta", "diff_amp"], 0.05, True, 1.1),
        ("level_delta_d2_diff_l05", ["level", "delta", "d2_match", "diff_amp"], 0.05, True, 1.1),
        ("amp_delta_diff_dir_l01", ["amp_under", "delta", "diff_amp", "direction"], 0.01, False, 0.6),
        ("trend_direction_l02", ["trend", "direction"], 0.02, True, 0.8),
    ]
    candidates: list[Candidate] = []
    for name, penalties, lam, dyn, alpha in pools:
        patch: dict[str, Any] = {
            "moe": {
                "enable": True,
                "dynamic_lambda": {"enable": bool(dyn)},
                "pred_side_residual": {
                    "enable": True,
                    "selection_policy": "val_mse_gate",
                    "alpha_scale": float(alpha) if alpha is not None else base_cfg.get("moe", {}).get("pred_side_residual", {}).get("alpha_scale", 0.8),
                    "gate_calibrator": {"epochs": 20, "batch_size": 256},
                },
                "gate_entropy_weight": 0.0,
                "gate_balance_weight": 0.0,
            },
            "penalties": {"enabled": penalties},
        }
        if lam is not None:
            patch["moe"]["lambda_init"] = lambda_dict(penalties, lam)
            patch["moe"]["lambda_min"] = lambda_dict(penalties, 0.0)
            patch["moe"]["lambda_schedule"] = schedule_dict(penalties, "none")
        candidates.append(Candidate("moe", name, patch))
    return limit_candidates(candidates, budget, smoke_n=2, local_n=4)


def run_train(config_path: Path, out_dir: Path, dry_run: bool) -> tuple[int, float]:
    if dry_run:
        return 0, 0.0
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "src.train", "--config", str(config_path)]
    env = dict(**os_environ_utf8())
    t0 = time.perf_counter()
    with (out_dir / "stdout.log").open("w", encoding="utf-8") as stdout_f, (out_dir / "stderr.log").open("w", encoding="utf-8") as stderr_f:
        completed = subprocess.run(cmd, cwd=str(REPO_ROOT), text=True, stdout=stdout_f, stderr=stderr_f, env=env)
    return int(completed.returncode), time.perf_counter() - t0


def os_environ_utf8() -> dict[str, str]:
    import os

    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("MOELOSS_PROGRESS_LEAVE", "0")
    return env


def read_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def row_from_summary(
    *,
    dataset: str,
    cand: Candidate,
    cfg: dict[str, Any],
    config_path: Path,
    out_dir: Path,
    returncode: int,
    total_sec: float,
    error: str = "",
) -> dict[str, Any]:
    summary = read_summary(out_dir / "run_summary.json")
    val = summary.get("val") or {}
    test = summary.get("test") or {}
    moe = cfg.get("moe", {}) or {}
    psr = moe.get("pred_side_residual", {}) or {}
    train = cfg.get("train", {}) or {}
    mae_obj = train.get("mae_objective", {}) or {}
    return {
        "dataset": dataset,
        "stage": cand.stage,
        "variant": cand.variant,
        "status": "ok" if returncode == 0 and summary else ("prepared" if returncode == 0 else "failed"),
        "val_mse": val.get("avg_mse", ""),
        "val_mae": val.get("avg_mae", ""),
        "test_mse": test.get("avg_mse", ""),
        "test_mae": test.get("avg_mae", ""),
        "best_epoch": ",".join(str(v) for v in summary.get("best_epoch", [])) if summary else "",
        "model_hidden_dim": cfg.get("model", {}).get("hidden_dim", ""),
        "model_dropout": cfg.get("model", {}).get("dropout", ""),
        "lr": train.get("lr", ""),
        "weight_decay": train.get("weight_decay", ""),
        "mae_weight": mae_obj.get("weight", ""),
        "moe_enable": moe.get("enable", ""),
        "penalties": ",".join(str(v) for v in cfg.get("penalties", {}).get("enabled", [])),
        "lambda_init": json.dumps(moe.get("lambda_init", ""), sort_keys=True),
        "dynamic_lambda": (moe.get("dynamic_lambda") or {}).get("enable", ""),
        "alpha_scale": psr.get("alpha_scale", ""),
        "selection_policy": psr.get("selection_policy", ""),
        "config_path": str(config_path),
        "out_dir": str(out_dir),
        "total_sec": total_sec,
        "returncode": returncode,
        "error": error,
    }


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDS})


def value(row: dict[str, Any], key: str = "val_mse") -> float:
    try:
        raw = row.get(key, "")
        if raw == "":
            return float("inf")
        return float(raw)
    except Exception:
        return float("inf")


def run_candidate(
    *,
    dataset: str,
    base_cfg: dict[str, Any],
    cand: Candidate,
    out_root: Path,
    device: str | None,
    epochs: int | None,
    skip_test: bool,
    dry_run: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    out_dir = out_root / "runs" / dataset / cand.stage / cand.variant
    cfg = common_prepare(
        base_cfg,
        dataset=dataset,
        out_dir=out_dir,
        run_name=f"{dataset}_input96_h96_{cand.stage}_{cand.variant}",
        device=device,
        epochs=epochs,
        skip_test=skip_test,
    )
    deep_update(cfg, copy.deepcopy(cand.patch))
    config_path = out_root / "configs" / dataset / cand.stage / f"{cand.variant}.yaml"
    write_yaml(config_path, cfg)
    if dry_run:
        returncode, total_sec = 0, 0.0
        error = ""
    else:
        returncode, total_sec = run_train(config_path, out_dir, dry_run=False)
        error = ""
        if returncode != 0:
            err_path = out_dir / "stderr.log"
            error = err_path.read_text(encoding="utf-8", errors="replace")[-2000:] if err_path.exists() else ""
    row = row_from_summary(
        dataset=dataset,
        cand=cand,
        cfg=cfg,
        config_path=config_path,
        out_dir=out_dir,
        returncode=returncode,
        total_sec=total_sec,
        error=error,
    )
    return row, cfg


def select_best(rows: list[dict[str, Any]], metric: str = "val_mse") -> dict[str, Any] | None:
    ok = [r for r in rows if r.get("status") == "ok" and r.get(metric) != ""]
    if not ok:
        return None
    tie_break = "test_mae" if metric == "test_mse" else "val_mae"
    return sorted(ok, key=lambda r: (value(r, metric), value(r, tie_break), value(r, "val_mse")))[0]


def main() -> None:
    ap = argparse.ArgumentParser(description="Input-96 H96 dataset-wise model and PKR-MoE targeted tuning.")
    ap.add_argument("--out-root", default="outputs/input96_h96_targeted_tuning")
    ap.add_argument("--datasets", nargs="+", default=list(DATASET_CONFIGS.keys()), choices=list(DATASET_CONFIGS.keys()))
    ap.add_argument("--device", default=None)
    ap.add_argument("--search-epochs", type=int, default=30)
    ap.add_argument("--final-epochs", type=int, default=100)
    ap.add_argument("--budget", choices=["smoke", "local", "compact"], default="local")
    ap.add_argument("--selection-metric", choices=["val_mse", "test_mse"], default="val_mse")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    search_skip_test = args.selection_metric != "test_mse"

    out_root = resolve(args.out_root)
    model_rows: list[dict[str, Any]] = []
    moe_rows: list[dict[str, Any]] = []
    final_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for dataset in args.datasets:
        print(f"=== {dataset}: model search ===", flush=True)
        base_cfg = load_yaml(resolve(DATASET_CONFIGS[dataset]))
        ds_model_rows: list[dict[str, Any]] = []
        ds_model_cfgs: dict[str, dict[str, Any]] = {}
        for cand in limit_candidates(model_candidates(), args.budget, smoke_n=2, local_n=3):
            off_cand = Candidate(cand.stage, cand.variant, copy.deepcopy(cand.patch))
            cfg_base = copy.deepcopy(base_cfg)
            set_moe_off(cfg_base)
            row, cfg = run_candidate(
                dataset=dataset,
                base_cfg=cfg_base,
                cand=off_cand,
                out_root=out_root,
                device=args.device,
                epochs=args.search_epochs,
                skip_test=search_skip_test,
                dry_run=bool(args.dry_run),
            )
            ds_model_rows.append(row)
            ds_model_cfgs[cand.variant] = cfg
            model_rows.append(row)
            write_rows(out_root / "model_results.csv", model_rows)
            print(
                f"[{dataset} model] {cand.variant}: {row['status']} "
                f"val_mse={row['val_mse']} test_mse={row['test_mse']}",
                flush=True,
            )

        best_model = select_best(ds_model_rows, args.selection_metric)
        if best_model is None:
            print(f"!!! {dataset}: no valid model candidate, skipping MoE search", flush=True)
            continue
        best_model_cfg = ds_model_cfgs[str(best_model["variant"])]

        print(f"=== {dataset}: MoE search on {best_model['variant']} ===", flush=True)
        ds_moe_rows: list[dict[str, Any]] = []
        ds_moe_cfgs: dict[str, dict[str, Any]] = {}
        for cand in moe_candidates(best_model_cfg, args.budget):
            row, cfg = run_candidate(
                dataset=dataset,
                base_cfg=best_model_cfg,
                cand=cand,
                out_root=out_root,
                device=args.device,
                epochs=args.search_epochs,
                skip_test=search_skip_test,
                dry_run=bool(args.dry_run),
            )
            ds_moe_rows.append(row)
            ds_moe_cfgs[cand.variant] = cfg
            moe_rows.append(row)
            write_rows(out_root / "moe_results.csv", moe_rows)
            print(
                f"[{dataset} moe] {cand.variant}: {row['status']} "
                f"val_mse={row['val_mse']} test_mse={row['test_mse']}",
                flush=True,
            )

        best_moe = select_best(ds_moe_rows, args.selection_metric)
        if best_moe is None:
            print(f"!!! {dataset}: no valid MoE candidate, using best model-off config", flush=True)
            best_cfg = best_model_cfg
            best_variant = str(best_model["variant"])
        else:
            best_cfg = ds_moe_cfgs[str(best_moe["variant"])]
            best_variant = str(best_moe["variant"])

        final_cand = Candidate("final_h96", best_variant, {})
        row, final_cfg = run_candidate(
            dataset=dataset,
            base_cfg=best_cfg,
            cand=final_cand,
            out_root=out_root,
            device=args.device,
            epochs=args.final_epochs,
            skip_test=False,
            dry_run=bool(args.dry_run),
        )
        final_rows.append(row)
        write_rows(out_root / "final_h96_results.csv", final_rows)

        best_config_path = out_root / "best_configs" / f"{dataset}.yaml"
        final_cfg.setdefault("eval", {})["skip_test"] = False
        write_yaml(best_config_path, final_cfg)
        summary_rows.append(
            {
                **row,
                "stage": "best_summary",
                "variant": best_variant,
                "config_path": str(best_config_path),
            }
        )
        write_rows(out_root / "best_summary.csv", summary_rows)
        print(f"=== {dataset}: selected {best_variant}, final test_mse={row['test_mse']} ===", flush=True)


if __name__ == "__main__":
    main()
