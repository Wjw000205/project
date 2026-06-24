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

SOURCE_REF = {
    "memory_path": "outputs/ettm1_h96_transfer_no_leak/source/cluster_memory.pt",
    "checkpoint_path": "outputs/ettm1_val_refinement_base/runs/ETTm1/pred_96/best_checkpoint.pt",
    "summary_path": "outputs/ettm1_val_refinement_base/runs/ETTm1/pred_96/run_summary.json",
}

FIELDS = [
    "status",
    "name",
    "eval_split",
    "mse",
    "mae",
    "corr_mode",
    "corr_align",
    "phase_bins",
    "period_min_hours",
    "period_max_hours",
    "corr_max_lag",
    "use_pred_residual",
    "route_counts",
    "corr_mean",
    "corr_min",
    "out_dir",
    "config",
    "error",
]


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_yaml(path: Path, cfg: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in FIELDS})


def route_stats(path: Path) -> tuple[str, float | None, float | None]:
    counts: dict[int, int] = {}
    corr_vals: list[float] = []
    if not path.exists():
        return "", None, None
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cid = int(float(row["cluster_id"]))
            counts[cid] = counts.get(cid, 0) + 1
            if row.get("corr_max"):
                corr_vals.append(float(row["corr_max"]))
    return (
        json.dumps(dict(sorted(counts.items())), ensure_ascii=False),
        sum(corr_vals) / len(corr_vals) if corr_vals else None,
        min(corr_vals) if corr_vals else None,
    )


def make_variants() -> list[dict[str, Any]]:
    variants: list[dict[str, Any]] = []
    for align in ["head", "tail"]:
        for min_h, max_h in [(12, 168), (24, 24), (24, 168), (12, 48), (48, 168)]:
            variants.append(
                {
                    "name": f"cycle_res_{align}_{min_h}_{max_h}",
                    "corr_mode": "cycle_template",
                    "use_pred_residual": True,
                    "corr_align": align,
                    "phase_bins": 64,
                    "period_min_hours": min_h,
                    "period_max_hours": max_h,
                    "corr_max_lag": 0,
                }
            )
    for bins in [24, 48, 96, 128]:
        variants.append(
            {
                "name": f"cycle_res_bins{bins}",
                "corr_mode": "cycle_template",
                "use_pred_residual": True,
                "corr_align": "head",
                "phase_bins": bins,
                "period_min_hours": 12,
                "period_max_hours": 168,
                "corr_max_lag": 0,
            }
        )
    for lag in [0, 24, 48, 96]:
        variants.append(
            {
                "name": f"pearson_res_lag{lag}",
                "corr_mode": "pearson",
                "use_pred_residual": True,
                "corr_align": "head",
                "phase_bins": 64,
                "period_min_hours": None,
                "period_max_hours": None,
                "corr_max_lag": lag,
            }
        )
    return variants


def patch_cfg(base: dict[str, Any], variant: dict[str, Any], out_dir: Path, eval_split: str) -> dict[str, Any]:
    cfg = copy.deepcopy(base)
    cfg.setdefault("exp", {})
    cfg["exp"]["name"] = f"ETTm1_to_ETTm2_{variant['name']}_{eval_split}"
    cfg["exp"]["out_dir"] = str(out_dir)
    cfg.setdefault("source", {}).update(SOURCE_REF)
    cfg.setdefault("normalize", {})["train_only"] = True
    cfg.setdefault("eval", {})["split"] = eval_split
    cfg["eval"].setdefault("batch_size", 64)

    transfer = cfg.setdefault("transfer", {})
    transfer["route_fit_scope"] = "train"
    transfer["save_corr"] = True
    transfer.setdefault("resample", {})["enable"] = False
    for key in [
        "corr_mode",
        "use_pred_residual",
        "corr_align",
        "phase_bins",
        "period_min_hours",
        "period_max_hours",
        "corr_max_lag",
    ]:
        transfer[key] = variant[key]
    transfer["phase_max_shift"] = None
    transfer["corr_threshold"] = None
    transfer["fallback_mode"] = "hard"
    transfer["fallback_topk"] = 2
    return cfg


def run_transfer(py: list[str], cfg_path: Path) -> tuple[int, str]:
    proc = subprocess.run(
        [*py, "-u", "-m", "src.transfer", "--config", str(cfg_path)],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return proc.returncode, proc.stdout


def row_from_result(variant: dict[str, Any], out_dir: Path, cfg_path: Path, status: str, error: str = "") -> dict[str, Any]:
    row: dict[str, Any] = {
        "status": status,
        "name": variant["name"],
        "corr_mode": variant.get("corr_mode"),
        "corr_align": variant.get("corr_align"),
        "phase_bins": variant.get("phase_bins"),
        "period_min_hours": variant.get("period_min_hours"),
        "period_max_hours": variant.get("period_max_hours"),
        "corr_max_lag": variant.get("corr_max_lag"),
        "use_pred_residual": variant.get("use_pred_residual"),
        "out_dir": str(out_dir),
        "config": str(cfg_path),
        "error": error,
    }
    summary_path = out_dir / "transfer_summary.json"
    if summary_path.exists():
        summary = read_json(summary_path)
        row["eval_split"] = summary.get("eval_split")
        row["mse"] = summary.get("avg_mse")
        row["mae"] = summary.get("avg_mae")
    counts, corr_mean, corr_min = route_stats(out_dir / "cluster_assignment.csv")
    row["route_counts"] = counts
    row["corr_mean"] = corr_mean
    row["corr_min"] = corr_min
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-config", type=Path, default=ROOT / "configs" / "ETTm1ToETTm2.yaml")
    ap.add_argument("--out-root", type=Path, default=ROOT / "outputs" / "ettm1_to_ettm2_val_route_selection")
    ap.add_argument("--python", type=Path, default=None)
    args = ap.parse_args()

    py = [str(args.python)] if args.python else [sys.executable]
    base = read_yaml(args.base_config)

    val_rows: list[dict[str, Any]] = []
    for variant in make_variants():
        out_dir = args.out_root / "val_runs" / variant["name"]
        cfg_path = args.out_root / "configs" / "val" / f"{variant['name']}.yaml"
        cfg = patch_cfg(base, variant, out_dir, "val")
        write_yaml(cfg_path, cfg)
        code, output = run_transfer(py, cfg_path)
        if code != 0:
            row = row_from_result(variant, out_dir, cfg_path, "failed", output[-2000:])
            print(f"[val failed] {variant['name']}")
        else:
            row = row_from_result(variant, out_dir, cfg_path, "ok")
            print(f"[val ok] {variant['name']} mse={float(row['mse']):.6f} mae={float(row['mae']):.6f}")
        val_rows.append(row)
        write_rows(args.out_root / "val_results.csv", val_rows)

    ok_rows = [row for row in val_rows if row.get("status") == "ok" and row.get("mse") not in {None, ""}]
    if not ok_rows:
        raise RuntimeError("No valid val transfer runs completed.")
    ok_rows.sort(key=lambda r: (float(r["mse"]), float(r["mae"])))
    write_rows(args.out_root / "val_results_ranked.csv", ok_rows)

    winner_name = ok_rows[0]["name"]
    winner = next(v for v in make_variants() if v["name"] == winner_name)
    test_out = args.out_root / "test_winner" / winner_name
    test_cfg_path = args.out_root / "configs" / "test" / f"{winner_name}.yaml"
    test_cfg = patch_cfg(base, winner, test_out, "test")
    write_yaml(test_cfg_path, test_cfg)
    code, output = run_transfer(py, test_cfg_path)
    if code != 0:
        raise RuntimeError(output)
    test_row = row_from_result(winner, test_out, test_cfg_path, "ok")
    write_rows(args.out_root / "selected_test.csv", [test_row])

    summary = {
        "selection_metric": "val.avg_mse",
        "selected_name": winner_name,
        "selected_val_mse": ok_rows[0]["mse"],
        "selected_val_mae": ok_rows[0]["mae"],
        "selected_test_mse": test_row.get("mse"),
        "selected_test_mae": test_row.get("mae"),
        "selected_config": str(test_cfg_path),
        "selected_out_dir": str(test_out),
    }
    with (args.out_root / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
