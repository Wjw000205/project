from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def _abs_md(path: Path) -> str:
    return path.resolve().as_posix()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _fmt_float(value: Any, digits: int = 4) -> str:
    try:
        x = float(value)
    except Exception:
        return ""
    if not np.isfinite(x):
        return ""
    return f"{x:.{digits}f}"


def _pct(numer: float, denom: float) -> float:
    if not np.isfinite(numer) or not np.isfinite(denom) or abs(denom) < 1.0e-12:
        return float("nan")
    return 100.0 * numer / denom


def _safe_get(d: dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = d
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _short_run_id(dataset: str, horizon: int, cluster_id: int) -> str:
    return f"{dataset}-H{horizon}-C{cluster_id}"


def _plot_heatmap(
    df: pd.DataFrame,
    row_col: str,
    col_col: str,
    value_col: str,
    out_path: Path,
    title: str,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    figsize_scale: tuple[float, float] = (0.35, 0.28),
) -> None:
    if df.empty:
        return
    pivot = df.pivot_table(index=row_col, columns=col_col, values=value_col, aggfunc="mean")
    pivot = pivot.sort_index()
    height = max(4.0, min(24.0, 1.5 + figsize_scale[1] * len(pivot.index)))
    width = max(5.5, min(16.0, 2.5 + figsize_scale[0] * len(pivot.columns)))
    fig, ax = plt.subplots(figsize=(width, height), dpi=180)
    data = pivot.to_numpy(dtype=float)
    im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=7)
    ax.set_title(title, fontsize=11)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            x = data[i, j]
            if np.isfinite(x):
                ax.text(j, i, f"{x:.2f}", ha="center", va="center", fontsize=6, color="white" if x > np.nanmax(data) * 0.55 else "black")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def _plot_gain_heatmap(run_df: pd.DataFrame, out_path: Path, value_col: str, title: str) -> None:
    if run_df.empty:
        return
    pivot = run_df.pivot_table(index="dataset", columns="horizon", values=value_col, aggfunc="mean")
    pivot = pivot.reindex(index=["ETTh1", "ETTh2", "ETTm1", "ETTm2"])
    fig, ax = plt.subplots(figsize=(6.0, 3.2), dpi=180)
    data = pivot.to_numpy(dtype=float)
    finite = data[np.isfinite(data)]
    vmax = float(np.nanmax(finite)) if finite.size else 1.0
    im = ax.imshow(data, aspect="auto", cmap="YlGnBu", vmin=0.0, vmax=max(1.0, vmax))
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([f"H={int(c)}" for c in pivot.columns])
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title(title)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            x = data[i, j]
            if np.isfinite(x):
                ax.text(j, i, f"{x:.1f}%", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def _markdown_table(df: pd.DataFrame, cols: list[str], headers: list[str] | None = None, max_rows: int = 30) -> str:
    if df.empty:
        return "_No rows available._\n"
    sub = df.loc[:, cols].head(max_rows).copy()
    headers = headers or cols
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in sub.iterrows():
        values = []
        for col in cols:
            v = row[col]
            if isinstance(v, float):
                values.append(_fmt_float(v, 3 if "pct" in col or "rate" in col or "prob" in col else 4))
            else:
                values.append(str(v))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def summarize(best_results: Path, out_root: Path) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    fig_dir = out_root / "figures"
    best_df = pd.read_csv(best_results)
    best_df = best_df[(best_df["status"] == "ok") & (best_df["is_best_for_cell"].astype(str).str.lower() == "true")].copy()
    best_df["horizon"] = best_df["horizon"].astype(int)
    best_df = best_df.sort_values(["dataset", "horizon"])

    run_rows: list[dict[str, Any]] = []
    affinity_rows: list[dict[str, Any]] = []
    cluster_rows: list[dict[str, Any]] = []
    route_rows: list[dict[str, Any]] = []
    hit_rows: list[dict[str, Any]] = []

    for _, row in best_df.iterrows():
        dataset = str(row["dataset"])
        horizon = int(row["horizon"])
        run_dir = Path(str(row["out_dir"]))
        if not run_dir.is_absolute():
            run_dir = ROOT / run_dir
        summary_path = run_dir / "run_summary.json"
        if not summary_path.exists():
            continue
        summary = _read_json(summary_path)
        penalty_names = list(summary.get("penalty_names", []))
        selection = summary.get("moe_residual_selection", {}) or {}
        residual = summary.get("moe_residual", {}) or {}
        gate_hit = summary.get("moe_gate_penalty_hit", {}) or {}
        test_act = _safe_get(summary, "moe_residual_gate_calibrator", "test_activation", default={}) or {}
        val_act = _safe_get(summary, "moe_residual_gate_calibrator", "val_activation", default={}) or {}

        val_base = float(selection.get("val_pred_base_avg_mse", np.nan))
        val_resid = float(selection.get("val_residual_avg_mse", row.get("val_mse", np.nan)))
        val_scaled = float(selection.get("val_scaled_avg_mse", np.nan))
        test_base = float(test_act.get("base_mse", np.nan))
        test_raw = float(test_act.get("raw_residual_mse", np.nan))
        test_scaled = float(test_act.get("scaled_mse", row.get("test_mse", np.nan)))
        val_gain_pct = _pct(val_base - val_resid, val_base)
        test_gain_pct = _pct(test_base - test_scaled, test_base)

        run_rows.append(
            {
                "dataset": dataset,
                "horizon": horizon,
                "candidate": row.get("candidate", ""),
                "test_mse": float(row.get("test_mse", np.nan)),
                "test_mae": float(row.get("test_mae", np.nan)),
                "val_base_mse": val_base,
                "val_residual_mse": val_resid,
                "val_scaled_mse": val_scaled,
                "val_gain_pct_vs_base": val_gain_pct,
                "test_base_mse_from_activation": test_base,
                "test_raw_residual_mse": test_raw,
                "test_scaled_mse": test_scaled,
                "test_scaled_gain_pct_vs_base": test_gain_pct,
                "route_alpha_mean": float(residual.get("alpha_mean", np.nan)),
                "residual_base_rms_ratio": float(residual.get("residual_base_rms_ratio", np.nan)),
                "val_gate_hit_rate": float(_safe_get(gate_hit, "val", "top1_hit_rate_all", default=np.nan)),
                "test_gate_hit_rate": float(_safe_get(gate_hit, "test", "top1_hit_rate_all", default=np.nan)),
                "val_oracle_positive_rate": float(_safe_get(gate_hit, "val", "oracle_positive_rate", default=np.nan)),
                "test_oracle_positive_rate": float(_safe_get(gate_hit, "test", "oracle_positive_rate", default=np.nan)),
                "val_activation_start_hit_rate": float(val_act.get("start_hit_rate", np.nan)),
                "test_activation_start_hit_rate": float(test_act.get("start_hit_rate", np.nan)),
                "val_activation_target_positive_rate": float(val_act.get("target_positive_rate", np.nan)),
                "test_activation_target_positive_rate": float(test_act.get("target_positive_rate", np.nan)),
            }
        )

        for pname, value in (residual.get("effective_route_by_penalty", {}) or {}).items():
            route_rows.append(
                {
                    "dataset": dataset,
                    "horizon": horizon,
                    "penalty": pname,
                    "effective_route_rate": float(value),
                    "alpha": float((residual.get("alpha_by_penalty", {}) or {}).get(pname, np.nan)),
                }
            )

        for split, obj in [("val", gate_hit.get("val", {}) or {}), ("test", gate_hit.get("test", {}) or {})]:
            selected_count = obj.get("selected_count", {}) or {}
            oracle_count = obj.get("oracle_count", {}) or {}
            for pname in penalty_names:
                hit_rows.append(
                    {
                        "dataset": dataset,
                        "horizon": horizon,
                        "split": split,
                        "penalty": pname,
                        "selected_count": int(selected_count.get(pname, 0)),
                        "oracle_count": int(oracle_count.get(pname, 0)),
                        "top1_hit_rate_all": float(obj.get("top1_hit_rate_all", np.nan)),
                        "top1_hit_rate_on_positive_oracle": float(obj.get("top1_hit_rate_on_positive_oracle", np.nan)),
                        "oracle_positive_rate": float(obj.get("oracle_positive_rate", np.nan)),
                        "selected_top1_gain_pct_vs_base": float(obj.get("selected_top1_gain_pct_vs_base", np.nan)),
                    }
                )

        prob_path = run_dir / "cluster_penalty_probs.csv"
        prob_df = pd.DataFrame()
        if prob_path.exists():
            prob_df = pd.read_csv(prob_path)
            for _, prow in prob_df.iterrows():
                cid = int(prow["cluster_id"])
                penalty = str(prow["penalty"])
                affinity_rows.append(
                    {
                        "dataset": dataset,
                        "horizon": horizon,
                        "cluster_id": cid,
                        "row_id": _short_run_id(dataset, horizon, cid),
                        "penalty": penalty,
                        "avg_prob": float(prow.get("avg_prob", np.nan)),
                        "avg_lambda": float(prow.get("avg_lambda", np.nan)),
                        "rank": int(prow.get("rank", -1)),
                        "avg_skip_active": float(prow.get("avg_skip_active", np.nan)),
                        "skip_cost": float(prow.get("skip_cost", np.nan)),
                    }
                )

        per_channel = val_act.get("per_channel", []) or []
        base_per_ch = selection.get("val_pred_base_mse_per_channel", []) or []
        resid_per_ch = selection.get("val_residual_mse_per_channel", []) or []
        channel_clusters: dict[int, list[str]] = {}
        cluster_base: dict[int, list[float]] = {}
        cluster_resid: dict[int, list[float]] = {}
        for i, ch in enumerate(per_channel):
            cid = int(ch.get("cluster_id", -1))
            channel_clusters.setdefault(cid, []).append(str(ch.get("channel", i)))
            if i < len(base_per_ch):
                cluster_base.setdefault(cid, []).append(float(base_per_ch[i]))
            if i < len(resid_per_ch):
                cluster_resid.setdefault(cid, []).append(float(resid_per_ch[i]))
        top_by_cluster: dict[int, dict[str, Any]] = {}
        if not prob_df.empty:
            non_skip = prob_df[prob_df["penalty"].astype(str) != "skip"].copy()
            if not non_skip.empty:
                for cid, sub in non_skip.groupby("cluster_id"):
                    top = sub.sort_values(["avg_prob", "rank"], ascending=[False, True]).iloc[0].to_dict()
                    top_by_cluster[int(cid)] = top
        for cid, channels in sorted(channel_clusters.items()):
            b = float(np.mean(cluster_base.get(cid, [np.nan])))
            r = float(np.mean(cluster_resid.get(cid, [np.nan])))
            top = top_by_cluster.get(cid, {})
            cluster_rows.append(
                {
                    "dataset": dataset,
                    "horizon": horizon,
                    "cluster_id": cid,
                    "row_id": _short_run_id(dataset, horizon, cid),
                    "channels": ",".join(channels),
                    "num_channels": len(channels),
                    "top_penalty": str(top.get("penalty", "")),
                    "top_penalty_prob": float(top.get("avg_prob", np.nan)),
                    "top_penalty_lambda": float(top.get("avg_lambda", np.nan)),
                    "val_base_mse": b,
                    "val_residual_mse": r,
                    "val_gain_abs": b - r if np.isfinite(b) and np.isfinite(r) else np.nan,
                    "val_gain_pct": _pct(b - r, b),
                }
            )

    run_df = pd.DataFrame(run_rows)
    affinity_df = pd.DataFrame(affinity_rows)
    cluster_df = pd.DataFrame(cluster_rows)
    route_df = pd.DataFrame(route_rows)
    hit_df = pd.DataFrame(hit_rows)

    run_df.to_csv(out_root / "run_improvement_summary.csv", index=False)
    affinity_df.to_csv(out_root / "cluster_penalty_affinity.csv", index=False)
    cluster_df.to_csv(out_root / "cluster_improvement_summary.csv", index=False)
    route_df.to_csv(out_root / "penalty_participation_summary.csv", index=False)
    hit_df.to_csv(out_root / "gate_hit_summary.csv", index=False)

    if not affinity_df.empty:
        aff_plot = affinity_df[affinity_df["penalty"] != "skip"].copy()
        _plot_heatmap(
            aff_plot,
            "row_id",
            "penalty",
            "avg_prob",
            fig_dir / "cluster_penalty_affinity_heatmap.png",
            "Cluster-Penalty Affinity (avg gate probability)",
            cmap="magma",
            vmin=0.0,
            vmax=1.0,
        )
    if not route_df.empty:
        route_plot = route_df.copy()
        route_plot["run_id"] = route_plot["dataset"] + "-H" + route_plot["horizon"].astype(str)
        _plot_heatmap(
            route_plot,
            "run_id",
            "penalty",
            "effective_route_rate",
            fig_dir / "penalty_participation_heatmap.png",
            "Penalty Participation Rate (effective route share)",
            cmap="Blues",
            vmin=0.0,
            vmax=1.0,
        )
    _plot_gain_heatmap(run_df, fig_dir / "val_residual_gain_heatmap.png", "val_gain_pct_vs_base", "Validation residual gain vs base")
    _plot_gain_heatmap(run_df, fig_dir / "test_residual_gain_heatmap.png", "test_scaled_gain_pct_vs_base", "Test residual gain vs base")
    if not cluster_df.empty:
        gain_plot = cluster_df.loc[:, ["row_id", "top_penalty", "val_gain_pct"]].copy()
        gain_plot["metric"] = "val_gain_pct"
        _plot_heatmap(
            gain_plot.rename(columns={"metric": "column"}),
            "row_id",
            "column",
            "val_gain_pct",
            fig_dir / "cluster_val_gain_heatmap.png",
            "Cluster Validation Gain (%)",
            cmap="YlGnBu",
            vmin=0.0,
            vmax=float(np.nanmax(gain_plot["val_gain_pct"].to_numpy())) if len(gain_plot) else None,
            figsize_scale=(1.0, 0.28),
        )

    md_path = out_root / "ett_interpretability_report.md"
    route_summary = route_df.pivot_table(index=["dataset", "horizon"], columns="penalty", values="effective_route_rate", aggfunc="mean").reset_index()
    route_summary.columns = [str(c) for c in route_summary.columns]
    top_cluster = cluster_df.sort_values(["dataset", "horizon", "cluster_id"]).copy()
    run_top = run_df.sort_values(["dataset", "horizon"]).copy()
    hit_top = hit_df[hit_df["split"] == "test"].copy()
    if not hit_top.empty:
        hit_top = hit_top.groupby(["dataset", "horizon"], as_index=False).agg(
            test_hit_rate=("top1_hit_rate_all", "mean"),
            test_hit_on_positive=("top1_hit_rate_on_positive_oracle", "mean"),
            oracle_positive_rate=("oracle_positive_rate", "mean"),
            selected_gain_pct=("selected_top1_gain_pct_vs_base", "mean"),
        )

    md_lines = [
        "# ETT Interpretability Report",
        "",
        "This report summarizes the existing ETT best-run artifacts without refitting models. It uses `outputs/ett_horizon_specific_moe_tune/best_results.csv` to locate the selected ETTh1/ETTh2/ETTm1/ETTm2 runs.",
        "",
        "Protocol notes:",
        "- Cluster-penalty affinity is the average gate probability from `cluster_penalty_probs.csv`; it should be read as learned routing affinity, not as a causal Pearson coefficient.",
        "- Improvement is computed from the logged base-vs-residual MSE fields. Validation gain is the clean interpretation axis; test gain is reported as an aligned reference where available.",
        "- KNN/hybrid and calibration are not used for these interpretation numbers.",
        "",
        "## Global Residual Improvement",
        "",
        _markdown_table(
            run_top,
            ["dataset", "horizon", "val_base_mse", "val_residual_mse", "val_gain_pct_vs_base", "test_base_mse_from_activation", "test_scaled_mse", "test_scaled_gain_pct_vs_base"],
            ["Dataset", "H", "Val base", "Val residual", "Val gain %", "Test base", "Test residual/scaled", "Test gain %"],
            max_rows=40,
        ),
        "",
        f"![Validation residual gain heatmap](<{_abs_md(fig_dir / 'val_residual_gain_heatmap.png')}>)",
        "",
        f"![Test residual gain heatmap](<{_abs_md(fig_dir / 'test_residual_gain_heatmap.png')}>)",
        "",
        "## Cluster-Penalty Affinity",
        "",
        "For each cluster, the top penalty is the non-skip penalty with the largest average routing probability. This is the most direct answer to which penalty a cluster is associated with.",
        "",
        _markdown_table(
            top_cluster,
            ["dataset", "horizon", "cluster_id", "channels", "top_penalty", "top_penalty_prob", "top_penalty_lambda", "val_gain_pct"],
            ["Dataset", "H", "Cluster", "Channels", "Top penalty", "Affinity", "Lambda", "Val gain %"],
            max_rows=80,
        ),
        "",
        f"![Cluster penalty affinity heatmap](<{_abs_md(fig_dir / 'cluster_penalty_affinity_heatmap.png')}>)",
        "",
        "## Penalty Participation",
        "",
        "The participation heatmap uses `effective_route_by_penalty` from the residual MoE summary. It captures which penalty branch actually participates after routing/selection.",
        "",
        _markdown_table(route_summary.fillna(0.0), list(route_summary.columns), max_rows=40),
        "",
        f"![Penalty participation heatmap](<{_abs_md(fig_dir / 'penalty_participation_heatmap.png')}>)",
        "",
        "## Gate Hit / Oracle Alignment",
        "",
        "The hit table compares the selected penalty with the per-sample oracle best penalty logged during evaluation. The numbers are useful for explaining whether the gate is selecting penalties in places where residual correction is beneficial.",
        "",
        _markdown_table(
            hit_top,
            ["dataset", "horizon", "test_hit_rate", "test_hit_on_positive", "oracle_positive_rate", "selected_gain_pct"],
            ["Dataset", "H", "Hit rate", "Hit on positive oracle", "Oracle positive rate", "Selected gain %"],
            max_rows=40,
        ),
        "",
        "## Cluster-Level Optimization",
        "",
        "This heatmap shows the validation MSE reduction by cluster after residual correction. It links the cluster's selected penalty to how much the residual branch improves the base predictor on validation windows.",
        "",
        f"![Cluster validation gain heatmap](<{_abs_md(fig_dir / 'cluster_val_gain_heatmap.png')}>)",
        "",
        "## Reading",
        "",
        "- If a cluster has high affinity and high validation gain, the penalty route is interpretable and useful for that regime.",
        "- If affinity is concentrated but gain is weak, the route is stable but the selected penalty has limited corrective value for that horizon.",
        "- If test gain is lower than validation gain, the module is likely sensitive to regime shift; this is evidence for reporting adaptive participation as future work rather than as the current main claim.",
        "",
        "Generated files:",
        "- `run_improvement_summary.csv`",
        "- `cluster_penalty_affinity.csv`",
        "- `cluster_improvement_summary.csv`",
        "- `penalty_participation_summary.csv`",
        "- `gate_hit_summary.csv`",
    ]
    md_path.write_text("\n".join(md_lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--best-results", type=Path, default=ROOT / "outputs" / "ett_horizon_specific_moe_tune" / "best_results.csv")
    parser.add_argument("--out-root", type=Path, default=ROOT / "outputs" / "ett_interpretability_report")
    args = parser.parse_args()
    summarize(args.best_results, args.out_root)
    print(f"Saved interpretability report to: {args.out_root / 'ett_interpretability_report.md'}")


if __name__ == "__main__":
    main()
