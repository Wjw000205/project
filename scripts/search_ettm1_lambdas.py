import argparse
import copy
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import yaml


def format_weights(weights: Dict[str, float], penalty_names: List[str]) -> str:
    return ", ".join(f"{name}={weights[name]:.2f}" for name in penalty_names)


class SearchProgress:
    def __init__(self, total_planned: int, penalty_names: List[str]) -> None:
        self.total_planned = max(int(total_planned), 1)
        self.penalty_names = penalty_names
        self.completed = 0

    def _bar(self) -> str:
        width = 28
        filled = int(width * self.completed / self.total_planned)
        return "#" * filled + "-" * (width - filled)

    def prefix(self) -> str:
        return f"[{self._bar()}] {self.completed}/{self.total_planned}"

    def log_start(self, run_id: int, stage: str, round_idx: int, target_penalty: str, weights: Dict[str, float]) -> None:
        current = self.completed + 1
        print(
            f"{self.prefix()} -> test {current}/{self.total_planned} | "
            f"run={run_id:04d} | stage={stage} | round={round_idx} | target={target_penalty}"
        )
        print(f"   weights: {format_weights(weights, self.penalty_names)}")

    def log_end(self, status: str, objective_name: str, objective_value, test_mae, total_sec) -> None:
        self.completed += 1
        if status == "ok":
            metric_text = f"{objective_name}={float(objective_value):.6f}"
            if test_mae not in ("", None):
                metric_text += f" | test_avg_mae={float(test_mae):.6f}"
        else:
            metric_text = "failed"
        time_text = ""
        if total_sec not in ("", None):
            time_text = f" | total_sec={float(total_sec):.2f}"
        print(f"{self.prefix()} <- {metric_text}{time_text}")


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def dump_yaml(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def build_grid(min_value: float, max_value: float, step: float) -> List[float]:
    if step <= 0:
        raise ValueError("step must be > 0")
    scale = round(1.0 / step)
    lo = int(round(min_value * scale))
    hi = int(round(max_value * scale))
    if hi < lo:
        raise ValueError("max_value must be >= min_value")
    return [round(v / scale, 2) for v in range(lo, hi + 1)]


def expand_penalty_setting(raw_value, penalty_names: List[str], default_value: float) -> Dict[str, float]:
    if isinstance(raw_value, dict):
        fallback = raw_value.get("default", default_value)
        return {name: float(raw_value.get(name, fallback)) for name in penalty_names}
    if isinstance(raw_value, (list, tuple)):
        if len(raw_value) != len(penalty_names):
            raise ValueError(f"Expected {len(penalty_names)} penalty weights, got {len(raw_value)}")
        return {name: float(value) for name, value in zip(penalty_names, raw_value)}
    value = default_value if raw_value is None else raw_value
    return {name: float(value) for name in penalty_names}


def make_run_config(
    base_cfg: dict,
    run_dir: Path,
    penalty_names: List[str],
    lambda_init: Dict[str, float],
    epochs_override: int = None,
) -> dict:
    cfg = copy.deepcopy(base_cfg)
    cfg["exp"]["out_dir"] = str(run_dir)
    cfg["moe"]["lambda_init"] = {name: float(lambda_init[name]) for name in penalty_names}

    corr_cfg = cfg.setdefault("corr", {})
    corr_cfg["save_path"] = str(run_dir / "corr.npy")

    plot_cfg = cfg.setdefault("plot", {})
    plot_cfg["enable"] = False

    portrait_cfg = cfg.setdefault("portrait", {})
    portrait_cfg["enable"] = False
    portrait_cfg["out_dir"] = str(run_dir / "cluster_portraits")

    memory_cfg = cfg.setdefault("memory", {})
    memory_cfg["enable"] = False
    memory_cfg["save_checkpoint"] = False
    memory_cfg["path"] = str(run_dir / "cluster_memory.pt")
    memory_cfg["checkpoint_path"] = str(run_dir / "best_checkpoint.pt")

    if epochs_override is not None:
        cfg["train"]["epochs"] = int(epochs_override)

    return cfg


def build_csv_row(
    run_id: int,
    status: str,
    stage: str,
    round_idx: int,
    target_penalty: str,
    weights: Dict[str, float],
    penalty_names: List[str],
) -> dict:
    row = {
        "run_id": run_id,
        "status": status,
        "stage": stage,
        "round_idx": round_idx,
        "target_penalty": target_penalty,
    }
    for name in penalty_names:
        row[f"lambda_{name}"] = float(weights[name])
    return row


def append_csv(csv_path: Path, row: dict, penalty_names: List[str]) -> None:
    fieldnames = [
        "run_id",
        "status",
        "stage",
        "round_idx",
        "target_penalty",
        "objective_name",
        "objective_value",
        "val_avg_loss",
        "val_avg_mse",
        "test_avg_mae",
        "test_avg_mse",
        "total_sec",
        "avg_epoch_sec",
        "out_dir",
        "config_path",
        "returncode",
        "error",
    ] + [f"lambda_{name}" for name in penalty_names]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists()
    with csv_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def evaluate_run(
    repo_root: Path,
    search_root: Path,
    base_cfg: dict,
    penalty_names: List[str],
    weights: Dict[str, float],
    run_id: int,
    stage: str,
    round_idx: int,
    target_penalty: str,
    epochs_override: int,
    objective_name: str,
    results_csv: Path,
    progress: SearchProgress,
) -> Tuple[float, dict]:
    run_dir = search_root / f"run_{run_id:04d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "config.yaml"
    cfg = make_run_config(base_cfg, run_dir, penalty_names, weights, epochs_override=epochs_override)
    dump_yaml(config_path, cfg)

    progress.log_start(run_id, stage, round_idx, target_penalty, weights)
    cmd = [sys.executable, "-m", "src.train", "--config", str(config_path)]
    completed = subprocess.run(
        cmd,
        cwd=repo_root,
        text=True,
        capture_output=True,
    )
    (run_dir / "stdout.log").write_text(completed.stdout, encoding="utf-8")
    (run_dir / "stderr.log").write_text(completed.stderr, encoding="utf-8")

    row = build_csv_row(run_id, "ok", stage, round_idx, target_penalty, weights, penalty_names)
    row["config_path"] = str(config_path)
    row["out_dir"] = str(run_dir)
    row["returncode"] = int(completed.returncode)
    row["error"] = ""

    if completed.returncode != 0:
        row["status"] = "failed"
        row["objective_name"] = objective_name
        row["objective_value"] = ""
        row["val_avg_loss"] = ""
        row["val_avg_mse"] = ""
        row["test_avg_mae"] = ""
        row["test_avg_mse"] = ""
        row["total_sec"] = ""
        row["avg_epoch_sec"] = ""
        row["error"] = (completed.stderr or completed.stdout).strip()[-2000:]
        append_csv(results_csv, row, penalty_names)
        progress.log_end(row["status"], row["objective_name"], None, None, row["total_sec"])
        return float("inf"), row

    summary_path = run_dir / "run_summary.json"
    if not summary_path.exists():
        row["status"] = "failed"
        row["objective_name"] = objective_name
        row["objective_value"] = ""
        row["val_avg_loss"] = ""
        row["val_avg_mse"] = ""
        row["test_avg_mae"] = ""
        row["test_avg_mse"] = ""
        row["total_sec"] = ""
        row["avg_epoch_sec"] = ""
        row["error"] = "run_summary.json not found"
        append_csv(results_csv, row, penalty_names)
        progress.log_end(row["status"], row["objective_name"], None, None, row["total_sec"])
        return float("inf"), row

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    val_info = summary.get("val") or {}
    test_info = summary.get("test") or {}
    timing = summary.get("timing") or {}
    objective_sources = {
        "val_avg_loss": val_info.get("avg_loss"),
        "val_avg_mse": val_info.get("avg_mse"),
        "test_avg_mae": test_info.get("avg_mae"),
        "test_avg_mse": test_info.get("avg_mse"),
    }
    objective_raw = objective_sources.get(objective_name)
    if objective_raw in ("", None):
        row["status"] = "failed"
        row["objective_name"] = objective_name
        row["objective_value"] = ""
        row["val_avg_loss"] = val_info.get("avg_loss", "")
        row["val_avg_mse"] = val_info.get("avg_mse", "")
        row["test_avg_mae"] = test_info.get("avg_mae", "")
        row["test_avg_mse"] = test_info.get("avg_mse", "")
        row["total_sec"] = timing.get("total_sec", "")
        row["avg_epoch_sec"] = timing.get("avg_epoch_sec", "")
        row["error"] = f"Objective '{objective_name}' missing in run_summary.json"
        append_csv(results_csv, row, penalty_names)
        progress.log_end(row["status"], row["objective_name"], None, row["test_avg_mae"], row["total_sec"])
        return float("inf"), row
    objective_value = float(objective_raw)

    row["objective_name"] = objective_name
    row["objective_value"] = objective_value
    row["val_avg_loss"] = val_info.get("avg_loss", "")
    row["val_avg_mse"] = val_info.get("avg_mse", "")
    row["test_avg_mae"] = test_info.get("avg_mae", "")
    row["test_avg_mse"] = test_info.get("avg_mse", "")
    row["total_sec"] = timing.get("total_sec", "")
    row["avg_epoch_sec"] = timing.get("avg_epoch_sec", "")
    append_csv(results_csv, row, penalty_names)
    progress.log_end(row["status"], row["objective_name"], row["objective_value"], row["test_avg_mae"], row["total_sec"])
    return objective_value, row


def search_best_lambdas(
    repo_root: Path,
    config_path: Path,
    output_root: Path,
    min_value: float,
    max_value: float,
    step: float,
    max_rounds: int,
    epochs_override: int,
    objective_name: str,
) -> None:
    base_cfg = load_yaml(config_path)
    penalty_names = list(base_cfg["penalties"]["enabled"])
    current = expand_penalty_setting(base_cfg["moe"].get("lambda_init", 0.05), penalty_names, 0.05)
    grid = build_grid(min_value, max_value, step)

    search_root = output_root
    search_root.mkdir(parents=True, exist_ok=True)
    results_csv = search_root / "search_results.csv"
    total_planned = 1 + max(0, max_rounds) * len(penalty_names) * len(grid)
    progress = SearchProgress(total_planned=total_planned, penalty_names=penalty_names)

    run_id = 1
    cache: Dict[Tuple[float, ...], float] = {}

    def weights_key(weights: Dict[str, float]) -> Tuple[float, ...]:
        return tuple(round(float(weights[name]), 2) for name in penalty_names)

    best_weights = dict(current)
    best_score, _ = evaluate_run(
        repo_root=repo_root,
        search_root=search_root,
        base_cfg=base_cfg,
        penalty_names=penalty_names,
        weights=best_weights,
        run_id=run_id,
        stage="baseline",
        round_idx=0,
        target_penalty="baseline",
        epochs_override=epochs_override,
        objective_name=objective_name,
        results_csv=results_csv,
        progress=progress,
    )
    cache[weights_key(best_weights)] = best_score
    run_id += 1

    for round_idx in range(1, max_rounds + 1):
        improved = False
        for penalty_name in penalty_names:
            local_best_score = best_score
            local_best_value = best_weights[penalty_name]
            for value in grid:
                candidate = dict(best_weights)
                candidate[penalty_name] = float(value)
                key = weights_key(candidate)
                if key in cache:
                    score = cache[key]
                else:
                    score, _ = evaluate_run(
                        repo_root=repo_root,
                        search_root=search_root,
                        base_cfg=base_cfg,
                        penalty_names=penalty_names,
                        weights=candidate,
                        run_id=run_id,
                        stage="coordinate_search",
                        round_idx=round_idx,
                        target_penalty=penalty_name,
                        epochs_override=epochs_override,
                        objective_name=objective_name,
                        results_csv=results_csv,
                        progress=progress,
                    )
                    cache[key] = score
                    run_id += 1
                if score < local_best_score:
                    local_best_score = score
                    local_best_value = float(value)
            if local_best_value != best_weights[penalty_name]:
                best_weights[penalty_name] = local_best_value
                best_score = local_best_score
                improved = True
        if not improved:
            break

    best_cfg = make_run_config(base_cfg, search_root / "best_run_template", penalty_names, best_weights, epochs_override=epochs_override)
    dump_yaml(search_root / "best_config.yaml", best_cfg)
    with (search_root / "best_params.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "objective_name": objective_name,
                "objective_value": best_score,
                "lambda_init": best_weights,
                "grid": grid,
                "config": str(config_path),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Search finished. Results CSV: {results_csv}")
    print(f"Progress summary: {progress.completed}/{progress.total_planned} tests executed")
    print(f"Best params: {search_root / 'best_params.json'}")
    print(f"Best config: {search_root / 'best_config.yaml'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/ETTm1.yaml")
    ap.add_argument("--output-root", type=str, default="outputs/ETTm1_lambda_search")
    ap.add_argument("--min", dest="min_value", type=float, default=0.00)
    ap.add_argument("--max", dest="max_value", type=float, default=0.10)
    ap.add_argument("--step", type=float, default=0.01)
    ap.add_argument("--max-rounds", type=int, default=2)
    ap.add_argument("--epochs-override", type=int, default=None)
    ap.add_argument(
        "--objective",
        type=str,
        default="val_avg_mse",
        choices=["val_avg_loss", "val_avg_mse", "test_avg_mae", "test_avg_mse"],
    )
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    search_best_lambdas(
        repo_root=repo_root,
        config_path=(repo_root / args.config).resolve(),
        output_root=(repo_root / args.output_root).resolve(),
        min_value=float(args.min_value),
        max_value=float(args.max_value),
        step=float(args.step),
        max_rounds=int(args.max_rounds),
        epochs_override=args.epochs_override,
        objective_name=str(args.objective),
    )


if __name__ == "__main__":
    main()
