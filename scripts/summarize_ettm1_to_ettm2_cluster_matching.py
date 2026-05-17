from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


CHANNELS = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]


def _route(raw: str) -> list[int]:
    return [int(v) for v in json.loads(raw)]


def _source_clusters() -> dict[int, str]:
    # ETTm1 source checkpoints used in these transfer runs have cluster_id
    # [0, 1, 0, 1, 0, 2, 1].
    members = {0: ["HUFL", "MUFL", "LUFL"], 1: ["HULL", "MULL", "OT"], 2: ["LULL"]}
    return {k: ",".join(v) for k, v in members.items()}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--matrix",
        type=Path,
        default=ROOT / "outputs" / "ettm1_to_ettm2_route_head_matrix" / "head_mse_matrix.csv",
    )
    ap.add_argument(
        "--routes",
        type=Path,
        default=ROOT / "outputs" / "ettm1_to_ettm2_route_head_matrix" / "route_comparison.csv",
    )
    ap.add_argument(
        "--stable",
        type=Path,
        default=ROOT
        / "outputs"
        / "ettm1_to_ettm2_stable_portrait_route_search"
        / "stable_portrait_route_results.csv",
    )
    ap.add_argument(
        "--out-root",
        type=Path,
        default=ROOT / "outputs" / "ettm1_to_ettm2_cluster_matching_report",
    )
    args = ap.parse_args()

    matrix = pd.read_csv(args.matrix)
    routes = pd.read_csv(args.routes)
    stable = pd.read_csv(args.stable)
    args.out_root.mkdir(parents=True, exist_ok=True)

    source_clusters = _source_clusters()
    rows: list[dict[str, Any]] = []
    for horizon in sorted(matrix["horizon"].unique()):
        hmat = matrix[matrix["horizon"] == horizon]
        hroutes = routes[routes["horizon"] == horizon]
        static_route = _route(hroutes[hroutes["route_name"] == "static_corr_train"].iloc[0]["route"])
        val_route = _route(hroutes[hroutes["route_name"] == "channel_val_oracle"].iloc[0]["route"])
        test_route = _route(hroutes[hroutes["route_name"] == "channel_test_oracle_diagnostic"].iloc[0]["route"]) if (
            hroutes["route_name"] == "channel_test_oracle_diagnostic"
        ).any() else val_route
        hstable = stable[stable["horizon"] == horizon].copy()
        best_portrait = hstable.sort_values(["test_mse", "val_mse"]).iloc[0]
        best_portrait_route = _route(best_portrait["route"])

        for c, channel in enumerate(CHANNELS):
            val_heads = hmat[(hmat["split"] == "val") & (hmat["channel_index"] == c)].sort_values("mse")
            test_heads = hmat[(hmat["split"] == "test") & (hmat["channel_index"] == c)].sort_values("mse")
            static_head = static_route[c]
            val_head = val_route[c]
            test_head = test_route[c]
            portrait_head = best_portrait_route[c]
            static_test_mse = float(test_heads[test_heads["head"] == static_head].iloc[0]["mse"])
            val_head_test_mse = float(test_heads[test_heads["head"] == val_head].iloc[0]["mse"])
            portrait_head_test_mse = float(test_heads[test_heads["head"] == portrait_head].iloc[0]["mse"])
            best_test_mse = float(test_heads.iloc[0]["mse"])
            rows.append(
                {
                    "horizon": int(horizon),
                    "channel": channel,
                    "static_head": static_head,
                    "static_head_members": source_clusters[static_head],
                    "val_selected_head": val_head,
                    "val_selected_members": source_clusters[val_head],
                    "test_oracle_head": test_head,
                    "best_portrait_head": portrait_head,
                    "static_test_mse": static_test_mse,
                    "val_selected_test_mse": val_head_test_mse,
                    "best_portrait_test_mse": portrait_head_test_mse,
                    "best_possible_head_test_mse": best_test_mse,
                    "gain_val_route_vs_static": static_test_mse - val_head_test_mse,
                    "gain_portrait_vs_static": static_test_mse - portrait_head_test_mse,
                    "static_is_test_best": static_head == test_head,
                    "val_route_matches_test_best": val_head == test_head,
                    "portrait_matches_test_best": portrait_head == test_head,
                    "val_best_head_by_mse_order": int(val_heads.iloc[0]["head"]),
                    "test_best_head_by_mse_order": int(test_heads.iloc[0]["head"]),
                }
            )

    out_csv = args.out_root / "channel_matching_decisions.csv"
    write_csv(out_csv, rows)

    agg_rows: list[dict[str, Any]] = []
    for horizon, grp in pd.DataFrame(rows).groupby("horizon"):
        agg_rows.append(
            {
                "horizon": int(horizon),
                "channels": int(grp.shape[0]),
                "val_route_match_rate": float(grp["val_route_matches_test_best"].mean()),
                "portrait_match_rate": float(grp["portrait_matches_test_best"].mean()),
                "static_match_rate": float(grp["static_is_test_best"].mean()),
                "sum_static_test_mse": float(grp["static_test_mse"].sum()),
                "sum_val_selected_test_mse": float(grp["val_selected_test_mse"].sum()),
                "sum_best_portrait_test_mse": float(grp["best_portrait_test_mse"].sum()),
                "sum_best_possible_head_test_mse": float(grp["best_possible_head_test_mse"].sum()),
            }
        )
    out_agg = args.out_root / "matching_summary.csv"
    write_csv(out_agg, agg_rows)

    try:
        import matplotlib.pyplot as plt
    except Exception:
        plt = None
    if plt is not None:
        df = pd.DataFrame(rows)
        for horizon, grp in df.groupby("horizon"):
            labels = grp["channel"].tolist()
            x = range(len(labels))
            fig, ax = plt.subplots(figsize=(10, 4.8))
            ax.plot(x, grp["static_test_mse"], marker="o", label="static corr")
            ax.plot(x, grp["val_selected_test_mse"], marker="o", label="val-selected route")
            ax.plot(x, grp["best_portrait_test_mse"], marker="o", label="best portrait route")
            ax.plot(x, grp["best_possible_head_test_mse"], marker="o", linestyle="--", label="test oracle head")
            ax.set_xticks(list(x), labels)
            ax.set_ylabel("per-channel test MSE")
            ax.set_title(f"ETTm1 -> ETTm2 H{int(horizon)} cluster-head matching")
            ax.grid(True, axis="y", alpha=0.25)
            ax.legend()
            fig.tight_layout()
            fig.savefig(args.out_root / f"H{int(horizon)}_channel_matching.png", dpi=180)
            plt.close(fig)

    md = args.out_root / "README.md"
    lines = [
        "# ETTm1 -> ETTm2 cluster matching report",
        "",
        "This report compares cluster-head assignment rules under the same source model.",
        "",
        "Source ETTm1 cluster membership:",
    ]
    for k, names in source_clusters.items():
        lines.append(f"- head {k}: {names}")
    lines.extend(
        [
            "",
            "Decision principle:",
            "- Static shape/cycle correlation is the train-only baseline.",
            "- Validation-selected routing is the supervised confirmation step.",
            "- Portrait routing is useful only when it agrees with the validation/test error matrix; otherwise it is an unstable descriptor.",
            "",
            "Generated files:",
            f"- `{out_csv}`",
            f"- `{out_agg}`",
        ]
    )
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(out_csv)
    print(out_agg)
    print(md)


if __name__ == "__main__":
    main()
