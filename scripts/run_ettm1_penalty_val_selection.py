import argparse
import copy
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


CSV_FIELDS = [
    "rank",
    "stage",
    "variant",
    "penalties",
    "num_penalties",
    "val_mse",
    "val_mae",
    "test_mse",
    "test_mae",
    "gain_val_mse_vs_previous",
    "gain_val_mse_vs_level",
    "best_epoch",
    "out_dir",
    "config_path",
]


VARIANTS: list[dict[str, Any]] = [
    {"stage": "forward", "variant": "s1_level", "penalties": ["level"]},
    {"stage": "forward", "variant": "s2_level_delta", "penalties": ["level", "delta"]},
    {"stage": "forward", "variant": "s3_level_delta_d2", "penalties": ["level", "delta", "d2_match"]},
    {"stage": "forward", "variant": "s3_level_delta_diff", "penalties": ["level", "delta", "diff_amp"]},
    {
        "stage": "forward",
        "variant": "s4_level_delta_d2_diff",
        "penalties": ["level", "delta", "d2_match", "diff_amp"],
    },
    {"stage": "alternative", "variant": "alt_level_amp_under", "penalties": ["level", "amp_under"]},
    {"stage": "alternative", "variant": "alt_level_range_delta", "penalties": ["level", "range", "delta"]},
]


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


def configure_candidate(
    base_cfg: dict[str, Any],
    *,
    penalties: list[str],
    out_dir: Path,
    epochs: int,
    device: str | None,
) -> dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    cfg.setdefault("exp", {})["out_dir"] = str(out_dir)
    if device:
        cfg["exp"]["device"] = str(device)
    cfg.setdefault("data", {})["csv_path"] = "data/ETTm1.csv"
    cfg["data"]["max_rows"] = 57600
    cfg["data"]["train_ratio"] = 0.6
    cfg["data"]["val_ratio"] = 0.2
    cfg["data"]["test_ratio"] = 0.2
    cfg.setdefault("window", {})["input_len"] = 336
    cfg["window"]["pred_len"] = 96
    cfg.setdefault("model", {})["predictor"] = "mlp"
    cfg.setdefault("normalize", {})["train_only"] = True
    cfg.setdefault("cluster", {})["train_only"] = True
    cfg.setdefault("knn_hybrid", {})["enable"] = False
    cfg.setdefault("eval", {})["skip_test"] = False
    cfg.setdefault("plot", {})["enable"] = False
    cfg.setdefault("portrait", {})["enable"] = False
    cfg.setdefault("memory", {})["save_checkpoint"] = False
    cfg["memory"]["path"] = str(out_dir / "cluster_memory.pt")
    cfg["memory"]["checkpoint_path"] = str(out_dir / "best_checkpoint.pt")
    cfg.setdefault("corr", {})["save_path"] = str(out_dir / "corr.npy")
    cfg["portrait"]["out_dir"] = str(out_dir / "cluster_portraits")
    cfg.setdefault("train", {})["epochs"] = int(epochs)

    cfg.setdefault("penalties", {})["enabled"] = list(penalties)
    moe_cfg = cfg.setdefault("moe", {})
    moe_cfg["enable"] = True
    moe_cfg["lambda_init"] = {name: 0.1 for name in penalties}
    moe_cfg["lambda_min"] = {name: 0.0 for name in penalties}
    moe_cfg["lambda_schedule"] = {name: "none" for name in penalties}
    moe_cfg.setdefault("pred_side_residual", {})["enable"] = True
    moe_cfg["pred_side_residual"].setdefault("selection_policy", "val_mse_gate")
    return cfg


def run_training(config_path: Path, reuse_existing: bool) -> None:
    cfg = load_yaml(config_path)
    summary_path = Path(cfg["exp"]["out_dir"]) / "run_summary.json"
    if reuse_existing and summary_path.exists():
        print(f"[reuse] {summary_path}")
        return
    cmd = [sys.executable, "-m", "src.train", "--config", str(config_path)]
    print("[run] " + " ".join(cmd))
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)


def read_summary(config_path: Path, stage: str, variant: str, penalties: list[str]) -> dict[str, Any]:
    cfg = load_yaml(config_path)
    out_dir = Path(cfg["exp"]["out_dir"])
    summary_path = out_dir / "run_summary.json"
    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)
    val = summary.get("val") or {}
    test = summary.get("test") or {}
    best_epoch = ",".join(str(v) for v in summary.get("best_epoch", []))
    return {
        "rank": "",
        "stage": stage,
        "variant": variant,
        "penalties": ",".join(penalties),
        "num_penalties": len(penalties),
        "val_mse": float(val.get("avg_mse", float("nan"))),
        "val_mae": float(val.get("avg_mae", float("nan"))),
        "test_mse": float(test.get("avg_mse", float("nan"))) if test else "",
        "test_mae": float(test.get("avg_mae", float("nan"))) if test else "",
        "gain_val_mse_vs_previous": "",
        "gain_val_mse_vs_level": "",
        "best_epoch": best_epoch,
        "out_dir": str(out_dir),
        "config_path": str(config_path),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})


def plot_forward(path: Path, rows: list[dict[str, Any]]) -> None:
    forward = [r for r in rows if r["stage"] == "forward"]
    if not forward:
        return
    labels = [r["variant"].replace("_", "\n") for r in forward]
    val = [float(r["val_mse"]) for r in forward]
    test = [float(r["test_mse"]) for r in forward if r["test_mse"] != ""]
    plt.figure(figsize=(9.5, 4.8))
    plt.plot(labels, val, marker="o", linewidth=2, label="val_mse")
    if len(test) == len(forward):
        plt.plot(labels, test, marker="o", linewidth=2, label="test_mse reference")
    for idx, value in enumerate(val):
        plt.text(idx, value, f"{value:.6f}", ha="center", va="bottom", fontsize=8)
    plt.ylabel("MSE")
    plt.title("ETTm1 H96 Penalty Forward Selection")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_ranked(path: Path, rows: list[dict[str, Any]]) -> None:
    ranked = sorted(rows, key=lambda r: float(r["val_mse"]))
    labels = [r["variant"].replace("_", "\n") for r in ranked]
    vals = [float(r["val_mse"]) for r in ranked]
    colors = ["#4c78a8" if r["stage"] == "forward" else "#f58518" for r in ranked]
    plt.figure(figsize=(10.5, 5.0))
    plt.bar(labels, vals, color=colors)
    for idx, value in enumerate(vals):
        plt.text(idx, value, f"{value:.6f}", ha="center", va="bottom", fontsize=8)
    plt.ylabel("Validation MSE")
    plt.title("ETTm1 H96 Penalty Candidates Ranked by Validation MSE")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def markdown_table(rows: list[dict[str, Any]]) -> str:
    fields = [
        "rank",
        "stage",
        "variant",
        "penalties",
        "val_mse",
        "test_mse",
        "gain_val_mse_vs_previous",
        "gain_val_mse_vs_level",
    ]
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in rows:
        vals = []
        for field in fields:
            value = row.get(field, "")
            if isinstance(value, float):
                vals.append(f"{value:.6g}")
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Controlled ETTm1 H96 val-based penalty selection. Test MSE is reference only."
    )
    ap.add_argument("--base-config", default="outputs/ettm1_val_refinement_base/configs/ETTm1_pred_96.yaml")
    ap.add_argument("--out-root", default="outputs/ettm1_penalty_val_selection_h96")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--device", default=None)
    ap.add_argument("--reuse-existing", action="store_true")
    ap.add_argument("--skip-run", action="store_true")
    args = ap.parse_args()

    base_path = resolve_path(args.base_config)
    out_root = resolve_path(args.out_root)
    cfg_root = out_root / "configs"
    run_root = out_root / "runs"
    base_cfg = load_yaml(base_path)
    out_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for spec in VARIANTS:
        variant = str(spec["variant"])
        penalties = list(spec["penalties"])
        cfg_path = cfg_root / f"{variant}.yaml"
        run_dir = run_root / variant
        cfg = configure_candidate(
            base_cfg,
            penalties=penalties,
            out_dir=run_dir,
            epochs=int(args.epochs),
            device=args.device,
        )
        write_yaml(cfg_path, cfg)
        if not args.skip_run:
            run_training(cfg_path, reuse_existing=bool(args.reuse_existing))
        rows.append(read_summary(cfg_path, str(spec["stage"]), variant, penalties))

    by_variant = {r["variant"]: r for r in rows}
    level_mse = float(by_variant["s1_level"]["val_mse"])
    previous = None
    for row in rows:
        if row["stage"] == "forward":
            if previous is None:
                row["gain_val_mse_vs_previous"] = 0.0
            else:
                row["gain_val_mse_vs_previous"] = float(previous["val_mse"]) - float(row["val_mse"])
            previous = row
        row["gain_val_mse_vs_level"] = level_mse - float(row["val_mse"])

    ranked = sorted(rows, key=lambda r: float(r["val_mse"]))
    for idx, row in enumerate(ranked, start=1):
        row["rank"] = idx
    for row in rows:
        row["rank"] = by_variant[row["variant"]]["rank"]

    write_csv(out_root / "penalty_selection_results.csv", rows)
    write_csv(out_root / "penalty_selection_ranked.csv", ranked)
    plot_forward(out_root / "penalty_forward_trace.png", rows)
    plot_ranked(out_root / "penalty_val_ranked.png", rows)

    summary = {
        "base_config": str(base_path),
        "out_root": str(out_root),
        "dataset": "ETTm1",
        "pred_len": 96,
        "epochs": int(args.epochs),
        "selection_metric": "val_mse",
        "test_metric_role": "reference_only",
        "best_by_val": ranked[0] if ranked else None,
        "rows": rows,
    }
    (out_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    md = "\n".join(
        [
            "# ETTm1 H96 Penalty Selection",
            "",
            "Selection metric: `val_mse`. `test_mse` is reference only.",
            "",
            markdown_table(ranked),
            "",
            "Figures:",
            "",
            "- `penalty_forward_trace.png`",
            "- `penalty_val_ranked.png`",
            "",
        ]
    )
    (out_root / "summary.md").write_text(md, encoding="utf-8")

    print(f"Saved results: {out_root / 'penalty_selection_results.csv'}")
    print(f"Saved ranked: {out_root / 'penalty_selection_ranked.csv'}")
    print(f"Saved summary: {out_root / 'summary.md'}")
    print("Top rows:")
    for row in ranked[:5]:
        print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
