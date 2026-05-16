import argparse
import copy
import csv
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]

DATASETS = ["ETTh1", "ETTh2", "ETTm1", "ETTm2"]
HORIZONS = [96, 192, 336, 720]
CONFIGS = {name: f"configs/{name}.yaml" for name in DATASETS}

PENALTY_POOLS: list[tuple[str, list[str]]] = [
    ("s1_level", ["level"]),
    ("s2_level_delta", ["level", "delta"]),
    ("s3_level_delta_diff", ["level", "delta", "diff_amp"]),
    ("s4_level_delta_d2_diff", ["level", "delta", "d2_match", "diff_amp"]),
    ("alt_trend_direction", ["trend", "direction"]),
    ("alt_level_trend_direction", ["level", "trend", "direction"]),
    ("alt_jump_amp_level_delta", ["jump", "amp_under", "level", "delta"]),
]

DEFAULT_COARSE_LAMBDAS = [0.005, 0.02, 0.05, 0.1, 0.2]
DEFAULT_FINE_MULTIPLIERS = [0.5, 0.75, 1.0, 1.25, 1.5]

RESULT_FIELDS = [
    "dataset",
    "stage",
    "rank",
    "variant",
    "penalties",
    "lambda_init",
    "lambda_min",
    "lambda_schedule",
    "pred_len",
    "val_mse",
    "val_mae",
    "test_mse_ref",
    "test_mae_ref",
    "gain_val_mse_vs_base",
    "best_epoch",
    "config_path",
    "out_dir",
    "status",
    "error",
]


@dataclass(frozen=True)
class Candidate:
    dataset: str
    stage: str
    variant: str
    penalties: tuple[str, ...]
    lambda_init: float
    lambda_min: float
    lambda_schedule: str
    pred_len: int = 96


def resolve_path(path_text: str | Path) -> Path:
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
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=False)


def safe_name(text: str) -> str:
    keep = []
    for ch in str(text):
        if ch.isalnum() or ch in {"_", "-"}:
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep).strip("_") or "item"


def lambda_text(value: float) -> str:
    return f"{value:.6g}".replace(".", "p").replace("-", "m")


def set_run_paths(cfg: dict[str, Any], out_dir: Path) -> None:
    cfg.setdefault("exp", {})["out_dir"] = str(out_dir)
    cfg.setdefault("corr", {})["save_path"] = str(out_dir / "corr.npy")
    cfg.setdefault("plot", {})["enable"] = False
    cfg.setdefault("portrait", {})["enable"] = False
    cfg["portrait"]["out_dir"] = str(out_dir / "cluster_portraits")
    cfg.setdefault("memory", {})["enable"] = False
    cfg["memory"]["save_checkpoint"] = False
    cfg["memory"]["path"] = str(out_dir / "cluster_memory.pt")
    cfg["memory"]["checkpoint_path"] = str(out_dir / "best_checkpoint.pt")


def configure_candidate(
    base_cfg: dict[str, Any],
    cand: Candidate,
    *,
    out_dir: Path,
    epochs: int,
    device: str | None,
) -> dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    set_run_paths(cfg, out_dir)
    cfg.setdefault("exp", {})["name"] = f"{cand.dataset}_{cand.stage}_{cand.variant}"
    if device:
        cfg["exp"]["device"] = str(device)

    cfg.setdefault("window", {})["input_len"] = 336
    cfg["window"]["pred_len"] = int(cand.pred_len)
    cfg.setdefault("normalize", {})["train_only"] = True
    cfg.setdefault("cluster", {})["train_only"] = True
    cfg.setdefault("eval", {})["skip_test"] = False
    cfg.setdefault("knn_hybrid", {})["enable"] = False
    cfg["knn_hybrid"]["use_for_model_selection"] = False
    cfg.setdefault("train", {})["epochs"] = int(epochs)

    penalties = list(cand.penalties)
    cfg.setdefault("penalties", {})["enabled"] = penalties
    moe_cfg = cfg.setdefault("moe", {})
    moe_cfg["enable"] = True
    moe_cfg["lambda_init"] = {name: float(cand.lambda_init) for name in penalties}
    moe_cfg["lambda_min"] = {name: float(cand.lambda_min) for name in penalties}
    moe_cfg["lambda_schedule"] = {name: str(cand.lambda_schedule) for name in penalties}
    moe_cfg.setdefault("dynamic_lambda", {})["enable"] = True
    moe_cfg.setdefault("pred_side_residual", {})["enable"] = True
    moe_cfg["pred_side_residual"].setdefault("selection_policy", "val_mse_gate")
    return cfg


def run_training(config_path: Path, out_dir: Path, *, reuse_existing: bool) -> tuple[str, str]:
    summary_path = out_dir / "run_summary.json"
    if reuse_existing and summary_path.exists():
        return "reused", ""
    out_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = out_dir / "stdout.log"
    stderr_path = out_dir / "stderr.log"
    cmd = [sys.executable, "-m", "src.train", "--config", str(config_path)]
    with stdout_path.open("w", encoding="utf-8") as stdout_f, stderr_path.open("w", encoding="utf-8") as stderr_f:
        done = subprocess.run(cmd, cwd=str(REPO_ROOT), text=True, stdout=stdout_f, stderr=stderr_f)
    if done.returncode != 0:
        err = stderr_path.read_text(encoding="utf-8", errors="replace")[-3000:]
        return "failed", err
    if not summary_path.exists():
        return "failed", "run_summary.json not found"
    return "ok", ""


def read_metrics(config_path: Path, cand: Candidate, status: str, error: str) -> dict[str, Any]:
    cfg = load_yaml(config_path)
    out_dir = Path(cfg["exp"]["out_dir"])
    row = {field: "" for field in RESULT_FIELDS}
    row.update(
        {
            "dataset": cand.dataset,
            "stage": cand.stage,
            "rank": "",
            "variant": cand.variant,
            "penalties": ",".join(cand.penalties),
            "lambda_init": cand.lambda_init,
            "lambda_min": cand.lambda_min,
            "lambda_schedule": cand.lambda_schedule,
            "pred_len": cand.pred_len,
            "config_path": str(config_path),
            "out_dir": str(out_dir),
            "status": status,
            "error": error,
        }
    )
    summary_path = out_dir / "run_summary.json"
    if not summary_path.exists():
        return row
    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)
    val = summary.get("val") or {}
    test = summary.get("test") or {}
    row.update(
        {
            "val_mse": val.get("avg_mse", ""),
            "val_mae": val.get("avg_mae", ""),
            "test_mse_ref": test.get("avg_mse", ""),
            "test_mae_ref": test.get("avg_mae", ""),
            "best_epoch": ",".join(str(v) for v in summary.get("best_epoch", [])),
        }
    )
    return row


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in RESULT_FIELDS})


def row_value(row: dict[str, Any], key: str, default: float = math.inf) -> float:
    try:
        value = row.get(key, "")
        if value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_float_list(text: str) -> list[float]:
    values = []
    for item in str(text).split(","):
        item = item.strip()
        if item:
            values.append(float(item))
    if not values:
        raise ValueError("Expected at least one float value.")
    return values


def rank_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = sorted([r for r in rows if r.get("status") in {"ok", "reused"}], key=lambda r: row_value(r, "val_mse"))
    for idx, row in enumerate(ranked, start=1):
        row["rank"] = idx
    return ranked


def make_penalty_candidates(dataset: str, base_cfg: dict[str, Any]) -> list[Candidate]:
    current_penalties = tuple(base_cfg.get("penalties", {}).get("enabled", []) or [])
    current_lambda = base_cfg.get("moe", {}).get("lambda_init", 0.1)
    if isinstance(current_lambda, dict) and current_penalties:
        vals = [float(current_lambda.get(name, 0.1)) for name in current_penalties]
        lam = sum(vals) / len(vals)
    elif current_lambda is None:
        lam = 0.1
    else:
        lam = float(current_lambda)
    candidates = []
    seen: set[tuple[str, ...]] = set()
    for name, penalties in PENALTY_POOLS:
        key = tuple(penalties)
        if key in seen:
            continue
        seen.add(key)
        candidates.append(
            Candidate(
                dataset=dataset,
                stage="penalty_step",
                variant=name,
                penalties=tuple(penalties),
                lambda_init=lam,
                lambda_min=0.0,
                lambda_schedule="none",
            )
        )
    if current_penalties and current_penalties not in seen:
        candidates.append(
            Candidate(
                dataset=dataset,
                stage="penalty_step",
                variant="current_penalties",
                penalties=current_penalties,
                lambda_init=lam,
                lambda_min=0.0,
                lambda_schedule="none",
            )
        )
    return candidates


def make_lambda_coarse_candidates(
    dataset: str,
    penalties: tuple[str, ...],
    *,
    coarse_lambdas: list[float],
    schedules: list[str],
) -> list[Candidate]:
    candidates = []
    for lam in coarse_lambdas:
        for schedule in schedules:
            candidates.append(
                Candidate(
                    dataset=dataset,
                    stage="lambda_coarse",
                    variant=f"lam{lambda_text(lam)}_{schedule}",
                    penalties=penalties,
                    lambda_init=float(lam),
                    lambda_min=0.0,
                    lambda_schedule=str(schedule),
                )
            )
    return candidates


def make_lambda_fine_candidates(
    dataset: str,
    best_row: dict[str, Any],
    *,
    fine_multipliers: list[float],
    min_ratios: list[float],
) -> list[Candidate]:
    penalties = tuple(str(best_row["penalties"]).split(","))
    base_lam = row_value(best_row, "lambda_init", 0.1)
    schedule = str(best_row.get("lambda_schedule") or "none")
    candidates = []
    seen = set()
    for multiplier in fine_multipliers:
        lam = max(1.0e-5, base_lam * float(multiplier))
        for min_ratio in min_ratios:
            key = (round(lam, 8), min_ratio)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(
                Candidate(
                    dataset=dataset,
                    stage="lambda_fine",
                    variant=f"lam{lambda_text(lam)}_min{lambda_text(lam * min_ratio)}_{schedule}",
                    penalties=penalties,
                    lambda_init=float(lam),
                    lambda_min=float(lam * min_ratio),
                    lambda_schedule=schedule,
                )
            )
    return candidates


def run_candidate(
    cand: Candidate,
    *,
    base_cfg: dict[str, Any],
    cfg_dir: Path,
    runs_dir: Path,
    epochs: int,
    device: str | None,
    reuse_existing: bool,
) -> dict[str, Any]:
    variant_dir = safe_name(cand.variant)
    config_path = cfg_dir / cand.dataset / cand.stage / f"{variant_dir}.yaml"
    out_dir = runs_dir / cand.dataset / cand.stage / variant_dir
    cfg = configure_candidate(base_cfg, cand, out_dir=out_dir, epochs=epochs, device=device)
    write_yaml(config_path, cfg)
    status, error = run_training(config_path, out_dir, reuse_existing=reuse_existing)
    return read_metrics(config_path, cand, status, error)


def plot_stage(path: Path, rows: list[dict[str, Any]], title: str) -> None:
    ok = [r for r in rows if r.get("status") in {"ok", "reused"}]
    if not ok:
        return
    ok = sorted(ok, key=lambda r: (str(r.get("dataset")), row_value(r, "val_mse")))
    labels = [f"{r['dataset']}\n{r['variant']}" for r in ok]
    vals = [row_value(r, "val_mse", math.nan) for r in ok]
    plt.figure(figsize=(max(9, len(ok) * 0.45), 5.0))
    plt.bar(range(len(ok)), vals, color="#4c78a8")
    plt.xticks(range(len(ok)), labels, rotation=45, ha="right", fontsize=8)
    plt.ylabel("Validation MSE")
    plt.title(title)
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def write_markdown(path: Path, summary_rows: list[dict[str, Any]]) -> None:
    fields = [
        "dataset",
        "best_penalty_variant",
        "best_penalties",
        "best_penalty_val_mse",
        "best_lambda_variant",
        "best_lambda",
        "best_lambda_schedule",
        "best_lambda_val_mse",
        "best_lambda_test_mse_ref",
    ]
    lines = ["# ETT H96 Penalty/Lambda Val Search", ""]
    lines.append("KNN is disabled. Selection uses validation MSE; test MSE is reference only.")
    lines.extend(["", "| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"])
    for row in summary_rows:
        vals = []
        for field in fields:
            value = row.get(field, "")
            if isinstance(value, float):
                vals.append(f"{value:.6g}")
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    lines.extend(
        [
            "",
            "Artifacts:",
            "",
            "- `penalty_results.csv`",
            "- `lambda_coarse_results.csv`",
            "- `lambda_fine_results.csv`",
            "- `summary.csv`",
            "- `penalty_val_ranked.png`",
            "- `lambda_val_ranked.png`",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="ETT H96 val-driven penalty selection and lambda coarse-to-fine search.")
    ap.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    ap.add_argument("--out-root", default="outputs/ett_penalty_lambda_val_search_h96")
    ap.add_argument("--search-epochs", type=int, default=30)
    ap.add_argument("--device", default=None)
    ap.add_argument("--reuse-existing", action="store_true")
    ap.add_argument("--skip-lambda", action="store_true")
    ap.add_argument("--coarse-lambdas", default=",".join(str(v) for v in DEFAULT_COARSE_LAMBDAS))
    ap.add_argument("--lambda-schedules", default="none,cosine")
    ap.add_argument("--fine-multipliers", default=",".join(str(v) for v in DEFAULT_FINE_MULTIPLIERS))
    ap.add_argument("--fine-min-ratios", default="0.0,0.1")
    args = ap.parse_args()

    out_root = resolve_path(args.out_root)
    cfg_dir = out_root / "configs"
    runs_dir = out_root / "runs"
    out_root.mkdir(parents=True, exist_ok=True)

    penalty_rows: list[dict[str, Any]] = []
    lambda_coarse_rows: list[dict[str, Any]] = []
    lambda_fine_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    coarse_lambdas = parse_float_list(args.coarse_lambdas)
    lambda_schedules = [item.strip().lower() for item in str(args.lambda_schedules).split(",") if item.strip()]
    fine_multipliers = parse_float_list(args.fine_multipliers)
    fine_min_ratios = parse_float_list(args.fine_min_ratios)

    for dataset in args.datasets:
        base_cfg = load_yaml(resolve_path(CONFIGS[dataset]))
        print(f"=== {dataset}: penalty step search ===")
        ds_penalty_rows = []
        for cand in make_penalty_candidates(dataset, base_cfg):
            print(f"[{dataset} penalty] {cand.variant}: {','.join(cand.penalties)}")
            row = run_candidate(
                cand,
                base_cfg=base_cfg,
                cfg_dir=cfg_dir,
                runs_dir=runs_dir,
                epochs=int(args.search_epochs),
                device=args.device,
                reuse_existing=bool(args.reuse_existing),
            )
            ds_penalty_rows.append(row)
            penalty_rows.append(row)
            write_rows(out_root / "penalty_results.csv", penalty_rows)

        ranked_penalty = rank_rows(ds_penalty_rows)
        best_penalty = ranked_penalty[0] if ranked_penalty else None
        if best_penalty is None:
            continue

        best_lambda_row = None
        if not args.skip_lambda:
            penalties = tuple(str(best_penalty["penalties"]).split(","))
            print(f"=== {dataset}: lambda coarse search on {','.join(penalties)} ===")
            ds_coarse_rows = []
            for cand in make_lambda_coarse_candidates(
                dataset,
                penalties,
                coarse_lambdas=coarse_lambdas,
                schedules=lambda_schedules,
            ):
                print(f"[{dataset} coarse] {cand.variant}")
                row = run_candidate(
                    cand,
                    base_cfg=base_cfg,
                    cfg_dir=cfg_dir,
                    runs_dir=runs_dir,
                    epochs=int(args.search_epochs),
                    device=args.device,
                    reuse_existing=bool(args.reuse_existing),
                )
                ds_coarse_rows.append(row)
                lambda_coarse_rows.append(row)
                write_rows(out_root / "lambda_coarse_results.csv", lambda_coarse_rows)
            ranked_coarse = rank_rows(ds_coarse_rows)
            best_coarse = ranked_coarse[0] if ranked_coarse else None

            if best_coarse is not None:
                print(f"=== {dataset}: lambda fine search around {best_coarse['variant']} ===")
                ds_fine_rows = []
                for cand in make_lambda_fine_candidates(
                    dataset,
                    best_coarse,
                    fine_multipliers=fine_multipliers,
                    min_ratios=fine_min_ratios,
                ):
                    print(f"[{dataset} fine] {cand.variant}")
                    row = run_candidate(
                        cand,
                        base_cfg=base_cfg,
                        cfg_dir=cfg_dir,
                        runs_dir=runs_dir,
                        epochs=int(args.search_epochs),
                        device=args.device,
                        reuse_existing=bool(args.reuse_existing),
                    )
                    ds_fine_rows.append(row)
                    lambda_fine_rows.append(row)
                    write_rows(out_root / "lambda_fine_results.csv", lambda_fine_rows)
                ranked_lambda = rank_rows(ds_coarse_rows + ds_fine_rows)
                best_lambda_row = ranked_lambda[0] if ranked_lambda else best_coarse

        if best_lambda_row is None:
            best_lambda_row = best_penalty

        summary_rows.append(
            {
                "dataset": dataset,
                "best_penalty_variant": best_penalty["variant"],
                "best_penalties": best_penalty["penalties"],
                "best_penalty_val_mse": row_value(best_penalty, "val_mse", math.nan),
                "best_penalty_test_mse_ref": row_value(best_penalty, "test_mse_ref", math.nan),
                "best_lambda_variant": best_lambda_row["variant"],
                "best_lambda": row_value(best_lambda_row, "lambda_init", math.nan),
                "best_lambda_min": row_value(best_lambda_row, "lambda_min", math.nan),
                "best_lambda_schedule": best_lambda_row.get("lambda_schedule", ""),
                "best_lambda_val_mse": row_value(best_lambda_row, "val_mse", math.nan),
                "best_lambda_test_mse_ref": row_value(best_lambda_row, "test_mse_ref", math.nan),
            }
        )
        write_rows(out_root / "penalty_results.csv", penalty_rows)
        write_rows(out_root / "lambda_coarse_results.csv", lambda_coarse_rows)
        write_rows(out_root / "lambda_fine_results.csv", lambda_fine_rows)

    summary_path = out_root / "summary.csv"
    if summary_rows:
        with summary_path.open("w", encoding="utf-8", newline="") as f:
            fields = list(summary_rows[0].keys())
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(summary_rows)
    (out_root / "summary.json").write_text(json.dumps(summary_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(out_root / "summary.md", summary_rows)
    plot_stage(out_root / "penalty_val_ranked.png", penalty_rows, "H96 Penalty Candidates Ranked by Validation MSE")
    plot_stage(out_root / "lambda_val_ranked.png", lambda_coarse_rows + lambda_fine_rows, "H96 Lambda Candidates Ranked by Validation MSE")

    print(f"Saved summary: {summary_path}")
    print(f"Saved markdown: {out_root / 'summary.md'}")
    print(json.dumps(summary_rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
