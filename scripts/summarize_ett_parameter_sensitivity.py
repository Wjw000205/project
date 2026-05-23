from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def _abs_md(path: Path) -> str:
    return path.resolve().as_posix()


def _fmt(v: Any, digits: int = 4) -> str:
    try:
        x = float(v)
    except Exception:
        return ""
    if not np.isfinite(x):
        return ""
    return f"{x:.{digits}f}"


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _maybe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _md_table(df: pd.DataFrame, cols: list[str], headers: list[str] | None = None, max_rows: int = 30) -> str:
    if df.empty:
        return "_No rows available._\n"
    headers = headers or cols
    sub = df.loc[:, cols].head(max_rows).copy()
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in sub.iterrows():
        vals: list[str] = []
        for col in cols:
            v = row[col]
            if isinstance(v, float) or isinstance(v, np.floating):
                vals.append(_fmt(v, 4))
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def _plot_dataset_bars(df: pd.DataFrame, out_path: Path, title: str, y_col: str = "val_mse", hue_col: str = "variant") -> None:
    if df.empty:
        return
    data = df.copy()
    data = data.sort_values(["dataset", y_col])
    datasets = list(data["dataset"].drop_duplicates())
    fig, axes = plt.subplots(len(datasets), 1, figsize=(10, max(3, 2.5 * len(datasets))), dpi=170, squeeze=False)
    for ax, dataset in zip(axes[:, 0], datasets):
        sub = data[data["dataset"] == dataset].head(12)
        labels = sub[hue_col].astype(str).str.replace("_", "\n", regex=False)
        ax.bar(np.arange(len(sub)), sub[y_col].astype(float), color="#4C78A8")
        ax.set_title(dataset)
        ax.set_ylabel(y_col)
        ax.set_xticks(np.arange(len(sub)))
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7)
        for i, v in enumerate(sub[y_col].astype(float)):
            ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    fig.suptitle(title, y=0.995, fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def _plot_lambda_curves(df: pd.DataFrame, out_path: Path, title: str) -> None:
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=170)
    for dataset, sub in df.groupby("dataset"):
        sub = sub.sort_values("lambda_init")
        ax.plot(sub["lambda_init"].astype(float), sub["val_mse"].astype(float), marker="o", label=str(dataset))
        for _, row in sub.iterrows():
            ax.text(float(row["lambda_init"]), float(row["val_mse"]), _fmt(row["test_mse_ref"], 3), fontsize=7, ha="left", va="bottom")
    ax.set_xlabel("lambda_init")
    ax.set_ylabel("val_mse")
    ax.set_title(title + " (point labels are test MSE reference)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def _plot_refinement_trace(df: pd.DataFrame, out_path: Path) -> None:
    if df.empty:
        return
    data = df.sort_values(["dataset", "stage_order", "val_rank"]).copy()
    fig, ax = plt.subplots(figsize=(8.0, 4.4), dpi=170)
    for dataset, sub in data.groupby("dataset"):
        top = sub.groupby("stage", as_index=False).first().sort_values("stage_order")
        ax.plot(top["stage"], top["val_mse"].astype(float), marker="o", label=str(dataset))
        for _, row in top.iterrows():
            ax.text(str(row["stage"]), float(row["val_mse"]), _fmt(row["test_mse_ref"], 3), fontsize=7)
    ax.set_ylabel("best val_mse")
    ax.set_title("Coarse-to-fine validation trace (labels: test MSE reference)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def _alignment_summary(df: pd.DataFrame, label: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if df.empty:
        return pd.DataFrame()
    usable = df.dropna(subset=["val_mse", "test_mse_ref"]).copy()
    for dataset, sub in usable.groupby("dataset"):
        sub = sub.copy()
        raw_n = len(sub)
        # Keep catastrophic failed runs out of rank-alignment statistics while
        # preserving them in the raw CSV. These rows are useful diagnostics but
        # not meaningful sensitivity points.
        cap = max(float(sub["test_mse_ref"].median()) * 5.0, float(sub["test_mse_ref"].quantile(0.9)) * 2.0)
        clean = sub[sub["test_mse_ref"] <= cap].copy()
        if clean.empty:
            continue
        pearson = clean[["val_mse", "test_mse_ref"]].corr(method="pearson").iloc[0, 1] if len(clean) >= 2 else np.nan
        spearman = clean[["val_mse", "test_mse_ref"]].corr(method="spearman").iloc[0, 1] if len(clean) >= 2 else np.nan
        best_val = clean.sort_values("val_mse").iloc[0]
        best_test = clean.sort_values("test_mse_ref").iloc[0]
        test_rank_of_best_val = int(clean["test_mse_ref"].rank(method="min").loc[best_val.name])
        rows.append(
            {
                "section": label,
                "dataset": dataset,
                "n_raw": raw_n,
                "n_used": len(clean),
                "pearson": pearson,
                "spearman": spearman,
                "best_val_variant": best_val.get("variant", ""),
                "best_val_mse": best_val["val_mse"],
                "best_val_test_ref": best_val["test_mse_ref"],
                "test_rank_of_best_val": test_rank_of_best_val,
                "best_test_variant": best_test.get("variant", ""),
                "best_test_mse_ref": best_test["test_mse_ref"],
                "best_test_val_mse": best_test["val_mse"],
            }
        )
    return pd.DataFrame(rows)


def _load_compact(compact_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    penalty = _maybe_read_csv(compact_root / "penalty_results.csv")
    coarse = _maybe_read_csv(compact_root / "lambda_coarse_results.csv")
    fine = _maybe_read_csv(compact_root / "lambda_fine_results.csv")
    for df in [penalty, coarse, fine]:
        if not df.empty:
            df["source"] = str(compact_root)
            df["status"] = df.get("status", "ok")
            df["pred_len"] = df.get("pred_len", 96)
            df["val_mse"] = pd.to_numeric(df["val_mse"], errors="coerce")
            df["test_mse_ref"] = pd.to_numeric(df["test_mse_ref"], errors="coerce")
    return penalty, coarse, fine


def _load_ettm2_manifest(root: Path) -> pd.DataFrame:
    manifest = root / "manifest.json"
    if not manifest.exists():
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for item in json.loads(manifest.read_text(encoding="utf-8")):
        if not item.get("on", False):
            continue
        run_dir = Path(item.get("run_dir", ""))
        summary = run_dir / "run_summary.json"
        if not summary.exists():
            continue
        data = _read_json(summary)
        rows.append(
            {
                "dataset": "ETTm2",
                "stage": item.get("stage", ""),
                "variant": item.get("name", ""),
                "preset": item.get("preset", ""),
                "penalties": ",".join(item.get("penalties", []) or data.get("penalty_names", [])),
                "pred_len": int(data.get("windowing", {}).get("pred_len", 96) or 96),
                "lambda_init": np.nan,
                "lambda_min": np.nan,
                "lambda_schedule": "",
                "val_mse": float(data.get("val", {}).get("avg_mse", np.nan)),
                "val_mae": float(data.get("val", {}).get("avg_mae", np.nan)),
                "test_mse_ref": float(data.get("test", {}).get("avg_mse", np.nan)),
                "test_mae_ref": float(data.get("test", {}).get("avg_mae", np.nan)),
                "config_path": item.get("config", ""),
                "out_dir": item.get("run_dir", ""),
                "status": "ok",
                "source": str(root),
            }
        )
    return pd.DataFrame(rows)


def _load_ettm2_lambda_gap(root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for summary in sorted((root / "runs").glob("td_lam*/run_summary.json")):
        name = summary.parent.name
        data = _read_json(summary)
        cfg_path = root / "configs" / f"{name}.yaml"
        lam = np.nan
        # The gap-search names encode the varied lambda scale, while the YAML keeps
        # per-penalty lambda values. We keep both the variant name and the parsed value.
        if "td_lam" in name:
            raw = name.split("td_lam", 1)[1].split("_", 1)[0].replace("p", ".")
            try:
                lam = float(raw)
            except ValueError:
                lam = np.nan
        rows.append(
            {
                "dataset": "ETTm2",
                "stage": "lambda_gap",
                "variant": name,
                "preset": "trend_direction",
                "penalties": ",".join(data.get("penalty_names", [])),
                "pred_len": int(data.get("windowing", {}).get("pred_len", 96) or 96),
                "lambda_init": lam,
                "lambda_min": np.nan,
                "lambda_schedule": "",
                "val_mse": float(data.get("val", {}).get("avg_mse", np.nan)),
                "val_mae": float(data.get("val", {}).get("avg_mae", np.nan)),
                "test_mse_ref": float(data.get("test", {}).get("avg_mse", np.nan)),
                "test_mae_ref": float(data.get("test", {}).get("avg_mae", np.nan)),
                "config_path": str(cfg_path) if cfg_path.exists() else "",
                "out_dir": str(summary.parent),
                "status": "ok",
                "source": str(root),
            }
        )
    return pd.DataFrame(rows)


def summarize(compact_root: Path, ettm2_root: Path, ettm2_gap_root: Path, out_root: Path) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    fig_dir = out_root / "figures"
    penalty, coarse, fine = _load_compact(compact_root)
    ettm2_manifest = _load_ettm2_manifest(ettm2_root)
    ettm2_gap = _load_ettm2_lambda_gap(ettm2_gap_root)

    penalty_rows = penalty.copy()
    if not ettm2_manifest.empty:
        stage2 = ettm2_manifest[ettm2_manifest["stage"].eq("stage2_coarse")].copy()
        penalty_rows = pd.concat([penalty_rows, stage2], ignore_index=True, sort=False)
    penalty_rows = penalty_rows[penalty_rows["status"].eq("ok")].copy()
    penalty_rows["val_rank"] = penalty_rows.groupby("dataset")["val_mse"].rank(method="first")
    penalty_rows = penalty_rows.sort_values(["dataset", "val_mse"])

    lambda_rows = pd.concat([coarse, fine, ettm2_gap], ignore_index=True, sort=False)
    lambda_rows = lambda_rows[lambda_rows["status"].eq("ok")].copy()
    lambda_rows["val_rank"] = lambda_rows.groupby(["dataset", "stage"])["val_mse"].rank(method="first")
    lambda_rows = lambda_rows.sort_values(["dataset", "stage", "val_mse"])

    stage_order = {"penalty_step": 0, "stage2_coarse": 0, "lambda_coarse": 1, "lambda_fine": 2, "lambda_gap": 2}
    trace_rows = pd.concat([penalty_rows, lambda_rows], ignore_index=True, sort=False)
    trace_rows["stage_order"] = trace_rows["stage"].map(stage_order).fillna(9).astype(int)
    trace_rows["val_rank"] = trace_rows.groupby(["dataset", "stage"])["val_mse"].rank(method="first")
    trace_rows = trace_rows.sort_values(["dataset", "stage_order", "val_rank"])

    penalty_rows.to_csv(out_root / "penalty_pool_sensitivity.csv", index=False)
    lambda_rows.to_csv(out_root / "lambda_sensitivity.csv", index=False)
    trace_rows.to_csv(out_root / "coarse_to_fine_trace.csv", index=False)
    alignment_rows = pd.concat(
        [
            _alignment_summary(penalty_rows, "penalty_pool"),
            _alignment_summary(lambda_rows, "lambda"),
            _alignment_summary(trace_rows, "coarse_to_fine"),
        ],
        ignore_index=True,
        sort=False,
    )
    alignment_rows.to_csv(out_root / "val_test_alignment_summary.csv", index=False)

    _plot_dataset_bars(
        penalty_rows,
        fig_dir / "penalty_pool_val_mse.png",
        "Penalty pool sensitivity ranked by validation MSE",
        y_col="val_mse",
        hue_col="variant",
    )
    _plot_lambda_curves(
        lambda_rows.dropna(subset=["lambda_init"]),
        fig_dir / "lambda_val_test_curves.png",
        "Lambda sensitivity",
    )
    _plot_refinement_trace(trace_rows, fig_dir / "coarse_to_fine_trace.png")

    best_penalty = penalty_rows.groupby("dataset", as_index=False).first()
    best_lambda = lambda_rows.dropna(subset=["lambda_init"]).groupby("dataset", as_index=False).first()
    top_penalty = penalty_rows.groupby("dataset", group_keys=False).head(6)
    top_lambda = lambda_rows.dropna(subset=["lambda_init"]).groupby("dataset", group_keys=False).head(8)

    md = [
        "# ETT Parameter Sensitivity Analysis",
        "",
        "This report summarizes existing ETT sensitivity artifacts. Configurations are ranked by validation MSE; test MSE/MAE are shown as aligned reference columns for the same candidate.",
        "",
        "Inputs:",
        f"- Compact H96 penalty/lambda search: `{compact_root}`",
        f"- ETTm2 H96 MoE search manifest: `{ettm2_root}`",
        f"- ETTm2 H96 lambda gap runs: `{ettm2_gap_root}`",
        "",
        "## Best Penalty Pool By Validation",
        "",
        _md_table(
            best_penalty,
            ["dataset", "stage", "variant", "penalties", "val_mse", "val_mae", "test_mse_ref", "test_mae_ref"],
            ["Dataset", "Stage", "Variant", "Penalty pool", "Val MSE", "Val MAE", "Test MSE ref", "Test MAE ref"],
            max_rows=20,
        ),
        "",
        f"![Penalty pool validation MSE](<{_abs_md(fig_dir / 'penalty_pool_val_mse.png')}>)",
        "",
        "## Top Penalty Pool Candidates",
        "",
        _md_table(
            top_penalty,
            ["dataset", "variant", "penalties", "val_mse", "test_mse_ref"],
            ["Dataset", "Variant", "Penalty pool", "Val MSE", "Test MSE ref"],
            max_rows=40,
        ),
        "",
        "## Lambda Sensitivity",
        "",
        _md_table(
            best_lambda,
            ["dataset", "stage", "variant", "penalties", "lambda_init", "val_mse", "test_mse_ref"],
            ["Dataset", "Stage", "Variant", "Penalty pool", "Lambda", "Val MSE", "Test MSE ref"],
            max_rows=20,
        ),
        "",
        f"![Lambda sensitivity](<{_abs_md(fig_dir / 'lambda_val_test_curves.png')}>)",
        "",
        "## Coarse-to-Fine Trace",
        "",
        "The trace keeps the best validation candidate from each stage. Point labels in the plot are the aligned test MSE reference for the same selected candidate.",
        "",
        f"![Coarse-to-fine trace](<{_abs_md(fig_dir / 'coarse_to_fine_trace.png')}>)",
        "",
        "## Validation-Test Alignment Check",
        "",
        "Validation and test are directionally aligned on most ETT sensitivity sweeps, but the ranking is not identical. This section reports rank/correlation diagnostics after excluding clearly failed catastrophic runs from the statistic only; raw rows are still kept in the CSV files.",
        "",
        _md_table(
            alignment_rows,
            [
                "section",
                "dataset",
                "n_used",
                "pearson",
                "spearman",
                "best_val_variant",
                "best_val_mse",
                "best_val_test_ref",
                "test_rank_of_best_val",
                "best_test_variant",
                "best_test_mse_ref",
            ],
            [
                "Section",
                "Dataset",
                "N",
                "Pearson",
                "Spearman",
                "Best-val variant",
                "Best val",
                "Best-val test ref",
                "Test rank",
                "Best-test variant",
                "Best test ref",
            ],
            max_rows=40,
        ),
        "",
        "Practical reading: ETTh1 and ETTm1 show acceptable validation-test agreement for the penalty/lambda sweeps. ETTh2 is partially aligned but the validation-best candidate is not always test-best. ETTm2 is weakly aligned in the lambda-gap sweep, so it should be described as sensitivity evidence rather than exact validation-to-test ranking.",
        "",
        "## Reading",
        "",
        "- ETTh1 and ETTm1 prefer a mid-size shape pool that includes `level`, `delta`, and `diff_amp` in the compact H96 search.",
        "- ETTh2 prefers the smaller `level,delta` pool, which supports the claim that more penalties are not automatically better.",
        "- ETTm2 is summarized from the broader H96 MoE search: the best validation rows favor `delta,trend,direction`-style pools, while the follow-up lambda gap shows a fairly narrow optimum around the trend/direction setting.",
        "- The validation-vs-test alignment is shown explicitly so this section can be used as a sensitivity analysis rather than a final model-selection claim.",
        "",
        "Generated CSV files:",
        "- `penalty_pool_sensitivity.csv`",
        "- `lambda_sensitivity.csv`",
        "- `coarse_to_fine_trace.csv`",
        "- `val_test_alignment_summary.csv`",
    ]
    (out_root / "ett_parameter_sensitivity_report.md").write_text("\n".join(md), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compact-root", type=Path, default=ROOT / "outputs" / "ett_penalty_lambda_val_search_h96_compact")
    parser.add_argument("--ettm2-root", type=Path, default=ROOT / "outputs" / "ettm2_96_moe_search")
    parser.add_argument("--ettm2-gap-root", type=Path, default=ROOT / "outputs" / "ettm2_96_moe_gap_search")
    parser.add_argument("--out-root", type=Path, default=ROOT / "outputs" / "ett_parameter_sensitivity_report")
    args = parser.parse_args()
    summarize(args.compact_root, args.ettm2_root, args.ettm2_gap_root, args.out_root)
    print(f"Saved ETT parameter sensitivity report to: {args.out_root / 'ett_parameter_sensitivity_report.md'}")


if __name__ == "__main__":
    main()
