from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY = ROOT / "outputs" / "codex_table_target_20260614" / "input96_global_paired_backbone_moe_summary.csv"
PEMS_DEPTH_ROOT = ROOT / "outputs" / "pems_depth_rollout"
PEMS08_H96_DEPTH_CONFIG = ROOT / "outputs" / "pems08_h96_backbone_capacity" / "configs" / "MOE_on_hid192_b2.yaml"
ETTM2_H96_CFULL_CONFIG = (
    ROOT
    / "outputs"
    / "next11c_fair_stage2_audit"
    / "fair_test_once"
    / "configs"
    / "ETTm2_H96"
    / "c_full.yaml"
)
ETTH2_H96_ANCHORPATH_CONFIG = (
    ROOT
    / "outputs"
    / "pkr_moe_wiring_audit"
    / "configs"
    / "ETTh2_H96"
    / "full_anchorpath_trainanchor_baseline_test_once.yaml"
)

RESULT_FIELDS = [
    "status",
    "dataset",
    "horizon",
    "variant",
    "source_config",
    "strategy_name",
    "strategy_config",
    "config_path",
    "out_dir",
    "test_mse",
    "test_mae",
    "val_mse",
    "val_mae",
    "selected_mse",
    "selected_mae",
    "selected_variant",
    "train_stat_anchor",
    "train_residual_anchor",
    "history_anchor",
    "pred_side_residual",
    "best_epoch",
    "total_sec",
    "returncode",
    "error",
]


def resolve(path_text: str | Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return (ROOT / path).resolve()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def write_yaml(path: Path, cfg: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=False, sort_keys=False)


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


def write_results(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in RESULT_FIELDS})


def read_existing_results(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def apply_h96_strategy_overlay(cfg: dict[str, Any], template_cfg: dict[str, Any]) -> None:
    """Apply the H96-discovered stage-2 strategy onto a horizon-specific base config.

    The base keeps dataset/window/model/finetune/checkpoint wiring for its horizon; the
    template supplies the MoE-side strategy that we want to test for horizon transfer.
    """
    for key in ("moe", "penalties", "train", "early_stop", "memory"):
        if key in template_cfg:
            cfg[key] = copy.deepcopy(template_cfg[key])


def infer_anchor_period(dataset: str, horizon: int) -> int:
    if dataset.startswith("PEMS"):
        return 288
    if dataset.lower() == "weather":
        return 144 if horizon == 96 else 96
    return 96


def default_stat_anchor(dataset: str, horizon: int) -> dict[str, Any]:
    return {
        "enable": True,
        "period": infer_anchor_period(dataset, horizon),
        "alpha": 0.0,
        "mode": "phase_mean",
        "reference": "last",
        "blend_target": "prediction",
        "scale_selection": {
            "enable": True,
            "metric": "mse",
            "max_scale": 0.2,
            "steps": 9,
        },
    }


def default_residual_anchor(dataset: str, horizon: int) -> dict[str, Any]:
    return {
        "enable": True,
        "period": infer_anchor_period(dataset, horizon),
        "alpha": 0.0,
        "blend_target": "prediction",
        "scale_selection": {
            "enable": True,
            "metric": "mse",
            "max_scale": 1.2,
            "steps": 49,
            "horizon_segments": 7 if horizon >= 96 else 4,
        },
    }


def localize_paths(cfg: dict[str, Any], out_dir: Path, name: str, device: str | None, skip_test: bool) -> None:
    cfg.setdefault("exp", {})["name"] = name
    cfg["exp"]["out_dir"] = str(out_dir)
    if device:
        cfg["exp"]["device"] = str(device)
    cfg.setdefault("corr", {})["save_path"] = str(out_dir / "corr.npy")
    cfg.setdefault("plot", {})["enable"] = False
    cfg.setdefault("portrait", {})["enable"] = False
    cfg["portrait"]["out_dir"] = str(out_dir / "cluster_portraits")
    cfg.setdefault("eval", {})["skip_test"] = bool(skip_test)
    cfg.setdefault("memory", {})["path"] = str(out_dir / "cluster_memory.pt")
    cfg["memory"]["checkpoint_path"] = str(out_dir / "best_checkpoint.pt")


def force_anchor_on(cfg: dict[str, Any], dataset: str, horizon: int) -> None:
    moe = cfg.setdefault("moe", {})
    stat = moe.setdefault("train_stat_anchor_expert", default_stat_anchor(dataset, horizon))
    resid = moe.setdefault("train_residual_anchor_expert", default_residual_anchor(dataset, horizon))
    stat["enable"] = True
    resid["enable"] = True
    stat.setdefault("period", infer_anchor_period(dataset, horizon))
    resid.setdefault("period", infer_anchor_period(dataset, horizon))
    stat.setdefault("scale_selection", default_stat_anchor(dataset, horizon)["scale_selection"])
    resid.setdefault("scale_selection", default_residual_anchor(dataset, horizon)["scale_selection"])


def selected_rows(
    rows: list[dict[str, str]],
    datasets: list[str] | None,
    exclude_datasets: list[str],
    horizons: list[int] | None,
    limit: int,
) -> list[dict[str, str]]:
    dataset_filter = {d.lower() for d in datasets} if datasets else set()
    exclude_filter = {d.lower() for d in exclude_datasets}
    horizon_filter = {int(h) for h in horizons} if horizons else set()
    selected: list[dict[str, str]] = []
    for row in rows:
        dataset = str(row.get("dataset", "")).strip()
        if not dataset:
            continue
        if dataset.lower() in exclude_filter:
            continue
        if dataset_filter and dataset.lower() not in dataset_filter:
            continue
        horizon = int(row.get("horizon", "0") or 0)
        if horizon_filter and horizon not in horizon_filter:
            continue
        selected.append(row)
        if limit > 0 and len(selected) >= limit:
            break
    return selected


def strategy_override_config(dataset: str) -> tuple[str, Path] | None:
    if dataset == "ETTm2" and ETTM2_H96_CFULL_CONFIG.exists():
        return ("h96_cfull", ETTM2_H96_CFULL_CONFIG)
    if dataset == "ETTh2" and ETTH2_H96_ANCHORPATH_CONFIG.exists():
        return ("h96_anchorpath", ETTH2_H96_ANCHORPATH_CONFIG)
    return None


def better_override_config(dataset: str, horizon: int) -> Path | None:
    if dataset.startswith("PEMS"):
        if dataset == "PEMS08" and horizon == 96 and PEMS08_H96_DEPTH_CONFIG.exists():
            return PEMS08_H96_DEPTH_CONFIG
        path = PEMS_DEPTH_ROOT / "configs" / f"MOE_{dataset}_H{horizon}_b2.yaml"
        if path.exists():
            return path
    return None


def source_config_for(row: dict[str, str], use_better_overrides: bool) -> Path:
    dataset = str(row.get("dataset", "")).strip()
    horizon = int(row.get("horizon", "0") or 0)
    if use_better_overrides:
        override = better_override_config(dataset, horizon)
        if override is not None:
            return override
    return resolve(str(row.get("moe_config", "")))


def result_from_summary(
    *,
    row: dict[str, str],
    variant: str,
    source_config: Path,
    config_path: Path,
    out_dir: Path,
    cfg: dict[str, Any],
    strategy_name: str,
    strategy_config: Path | None,
    returncode: int,
    total_sec: float,
    error: str,
    prepared: bool = False,
) -> dict[str, Any]:
    summary = read_json(out_dir / "run_summary.json") if not prepared else {}
    val = summary.get("val") or {}
    test = summary.get("test") or {}
    selected = summary.get("selected") or {}
    moe = cfg.get("moe") or {}
    stat = bool((moe.get("train_stat_anchor_expert") or {}).get("enable", False))
    resid = bool((moe.get("train_residual_anchor_expert") or {}).get("enable", False))
    hist = bool((moe.get("history_anchor_expert") or {}).get("enable", False))
    pred_resid = bool((moe.get("pred_side_residual") or {}).get("enable", False))
    status = "prepared" if prepared else ("ok" if returncode == 0 and summary else "failed")
    return {
        "status": status,
        "dataset": row.get("dataset", ""),
        "horizon": row.get("horizon", ""),
        "variant": variant,
        "source_config": str(source_config),
        "strategy_name": strategy_name,
        "strategy_config": str(strategy_config) if strategy_config else "",
        "config_path": str(config_path),
        "out_dir": str(out_dir),
        "test_mse": test.get("avg_mse", ""),
        "test_mae": test.get("avg_mae", ""),
        "val_mse": val.get("avg_mse", ""),
        "val_mae": val.get("avg_mae", ""),
        "selected_mse": selected.get("avg_mse", ""),
        "selected_mae": selected.get("avg_mae", ""),
        "selected_variant": selected.get("variant", ""),
        "train_stat_anchor": stat,
        "train_residual_anchor": resid,
        "history_anchor": hist,
        "pred_side_residual": pred_resid,
        "best_epoch": ",".join(str(v) for v in summary.get("best_epoch", [])) if summary else "",
        "total_sec": total_sec,
        "returncode": returncode,
        "error": error,
    }


def upsert_result(rows: list[dict[str, Any]], result: dict[str, Any]) -> None:
    key = (str(result.get("dataset", "")), str(result.get("horizon", "")))
    for idx, row in enumerate(rows):
        if (str(row.get("dataset", "")), str(row.get("horizon", ""))) == key:
            rows[idx] = result
            return
    rows.append(result)


def run_one(
    *,
    row: dict[str, str],
    out_root: Path,
    device: str | None,
    skip_test: bool,
    force_anchor: bool,
    dry_run: bool,
    reuse_existing: bool,
    use_better_overrides: bool,
) -> dict[str, Any]:
    dataset = str(row["dataset"])
    horizon = int(row["horizon"])
    original_source = resolve(str(row.get("moe_config", "")))
    source_config = source_config_for(row, use_better_overrides)
    variant = str(row.get("moe_variant", "") or source_config.stem)
    if source_config != original_source:
        variant = source_config.stem
    strategy = strategy_override_config(dataset) if use_better_overrides else None
    strategy_name = ""
    strategy_config: Path | None = None
    if strategy is not None:
        strategy_name, strategy_config = strategy
        variant = f"{variant}_{strategy_name}"
    if not source_config.exists():
        return {
            "status": "missing_source_config",
            "dataset": dataset,
            "horizon": horizon,
            "variant": variant,
            "source_config": str(source_config),
            "strategy_name": strategy_name,
            "strategy_config": str(strategy_config) if strategy_config else "",
            "error": f"Source config not found: {source_config}",
        }

    cfg = load_yaml(source_config)
    if strategy_config is not None:
        apply_h96_strategy_overlay(cfg, load_yaml(strategy_config))
    run_name = f"{dataset}_input96_H{horizon}_{variant}"
    out_dir = out_root / "runs" / dataset / f"H{horizon}" / variant
    config_path = out_root / "configs" / dataset / f"H{horizon}" / f"{variant}.yaml"
    localize_paths(cfg, out_dir, run_name, device, skip_test)
    if force_anchor:
        force_anchor_on(cfg, dataset, horizon)
    write_yaml(config_path, cfg)

    if dry_run:
        return result_from_summary(
            row=row,
            variant=variant,
            source_config=source_config,
            config_path=config_path,
            out_dir=out_dir,
            cfg=cfg,
            strategy_name=strategy_name,
            strategy_config=strategy_config,
            returncode=0,
            total_sec=0.0,
            error="dry_run",
            prepared=True,
        )
    if reuse_existing and (out_dir / "run_summary.json").exists():
        return result_from_summary(
            row=row,
            variant=variant,
            source_config=source_config,
            config_path=config_path,
            out_dir=out_dir,
            cfg=cfg,
            strategy_name=strategy_name,
            strategy_config=strategy_config,
            returncode=0,
            total_sec=0.0,
            error="",
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = out_dir / "stdout.log"
    stderr_path = out_dir / "stderr.log"
    cmd = [sys.executable, "-u", "-m", "src.train", "--config", str(config_path)]
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env["MOELOSS_PROGRESS_LEAVE"] = "0"
    start = time.perf_counter()
    with stdout_path.open("w", encoding="utf-8") as stdout_f, stderr_path.open("w", encoding="utf-8") as stderr_f:
        completed = subprocess.run(cmd, cwd=str(ROOT), stdout=stdout_f, stderr=stderr_f, text=True, env=env)
    total_sec = time.perf_counter() - start
    error = ""
    if completed.returncode != 0:
        error = stderr_path.read_text(encoding="utf-8", errors="replace")[-3000:]
    return result_from_summary(
        row=row,
        variant=variant,
        source_config=source_config,
        config_path=config_path,
        out_dir=out_dir,
        cfg=cfg,
        strategy_name=strategy_name,
        strategy_config=strategy_config,
        returncode=int(completed.returncode),
        total_sec=total_sec,
        error=error,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rerun the current input-96 main-table MoE configs with anchors on, optionally excluding ECL/Electricity."
    )
    parser.add_argument("--summary-csv", default=str(DEFAULT_SUMMARY))
    parser.add_argument("--out-root", default="outputs/input96_main_table_anchor_on_no_ecl_20260619")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--exclude-datasets", nargs="*", default=["Electricity", "ECL", "Weather"])
    parser.add_argument("--horizons", nargs="*", type=int, default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("--no-force-anchor", action="store_true")
    parser.add_argument("--no-better-overrides", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_path = resolve(args.summary_csv)
    out_root = resolve(args.out_root)
    _, table_rows = read_rows(summary_path)
    rows = selected_rows(table_rows, args.datasets, args.exclude_datasets, args.horizons, int(args.limit))
    if not rows:
        raise SystemExit("No rows selected.")

    results_path = out_root / "results.csv"
    result_rows = read_existing_results(results_path) if results_path.exists() and not args.dry_run else []
    for idx, row in enumerate(rows, start=1):
        print(f"[{idx}/{len(rows)}] {row['dataset']} H{row['horizon']} {row.get('moe_variant', '')}", flush=True)
        result = run_one(
            row=row,
            out_root=out_root,
            device=str(args.device),
            skip_test=bool(args.skip_test),
            force_anchor=not bool(args.no_force_anchor),
            dry_run=bool(args.dry_run),
            reuse_existing=bool(args.reuse_existing),
            use_better_overrides=not bool(args.no_better_overrides),
        )
        upsert_result(result_rows, result)
        write_results(results_path, result_rows)
        print(
            json.dumps(
                {
                    "status": result.get("status"),
                    "dataset": result.get("dataset"),
                    "horizon": result.get("horizon"),
                    "variant": result.get("variant"),
                    "test_mse": result.get("test_mse"),
                    "test_mae": result.get("test_mae"),
                    "stat_anchor": result.get("train_stat_anchor"),
                    "resid_anchor": result.get("train_residual_anchor"),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        if result.get("status") not in {"ok", "prepared"} and args.stop_on_error:
            raise SystemExit(f"Stopped after failed row: {result}")
    print(f"Wrote: {results_path}", flush=True)


if __name__ == "__main__":
    main()
