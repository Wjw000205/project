from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


FIELDS = [
    "horizon",
    "selected_candidate",
    "selected_family",
    "selection_reason",
    "val_mse",
    "test_mse_reference",
    "test_mae_reference",
    "target_self_mse",
    "source_test_mse",
    "route_or_weights",
]


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _as_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def collect_candidates(root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    hard = _read(root / "outputs" / "ettm1_to_ettm2_route_head_matrix" / "route_comparison.csv")
    if not hard.empty:
        for _, row in hard.iterrows():
            if "test_oracle" in str(row["route_name"]):
                continue
            family = "hard_val" if row["route_name"] == "channel_val_oracle" else "hard_static"
            complexity = 1 if family == "hard_val" else 0
            rows.append(
                {
                    "horizon": int(row["horizon"]),
                    "candidate": row["route_name"],
                    "family": family,
                    "complexity": complexity,
                    "val_mse": float(row["val_mse"]),
                    "test_mse": float(row["test_mse"]),
                    "test_mae": float(row["test_mae"]),
                    "target_self_mse": float(row["target_self_mse"]),
                    "source_test_mse": float(row["source_test_mse"]),
                    "route_or_weights": row["route"],
                }
            )

    portrait = _read(
        root
        / "outputs"
        / "ettm1_to_ettm2_stable_portrait_route_search"
        / "stable_portrait_route_results.csv"
    )
    if not portrait.empty:
        for _, row in portrait.iterrows():
            rows.append(
                {
                    "horizon": int(row["horizon"]),
                    "candidate": row["candidate"],
                    "family": "portrait",
                    "complexity": 2,
                    "val_mse": float(row["val_mse"]),
                    "test_mse": float(row["test_mse"]),
                    "test_mae": float(row["test_mae"]),
                    "target_self_mse": float(row["target_self_mse"]),
                    "source_test_mse": float(row["source_test_mse"]),
                    "route_or_weights": row["route"],
                }
            )

    soft = _read(root / "outputs" / "ettm1_to_ettm2_soft_cluster_matching" / "soft_cluster_results.csv")
    if not soft.empty:
        for _, row in soft.iterrows():
            val = _as_float(row.get("val_mse"))
            if val is None:
                continue
            rows.append(
                {
                    "horizon": int(row["horizon"]),
                    "candidate": row["candidate"],
                    "family": "soft",
                    "complexity": 3,
                    "val_mse": val,
                    "test_mse": float(row["test_mse"]),
                    "test_mae": float(row["test_mae"]),
                    "target_self_mse": float(row["target_self_mse"]),
                    "source_test_mse": float(row["source_test_mse"]),
                    "route_or_weights": row["route_or_weights"],
                }
            )

    dynamic = _read(
        root
        / "outputs"
        / "ettm1_to_ettm2_val_calibrated_dynamic_matching"
        / "dynamic_matching_results.csv"
    )
    if not dynamic.empty:
        for _, row in dynamic.iterrows():
            val = _as_float(row.get("val_mse"))
            if val is None:
                continue
            rows.append(
                {
                    "horizon": int(row["horizon"]),
                    "candidate": row["candidate"],
                    "family": "dynamic",
                    "complexity": 4,
                    "val_mse": val,
                    "test_mse": float(row["test_mse"]),
                    "test_mae": float(row["test_mae"]),
                    "target_self_mse": float(row["target_self_mse"]),
                    "source_test_mse": float(row["source_test_mse"]),
                    "route_or_weights": row["route_summary"],
                }
            )

    return pd.DataFrame(rows)


def write_rows(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=Path, default=ROOT / "outputs" / "ettm1_to_ettm2_robust_cluster_matching_policy")
    ap.add_argument("--val-tolerance", type=float, default=0.03)
    args = ap.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    candidates = collect_candidates(ROOT)
    if candidates.empty:
        raise RuntimeError("No cluster matching candidates found.")

    candidates = candidates.sort_values(["horizon", "val_mse", "complexity", "test_mse"])
    candidates.to_csv(args.out_root / "all_matching_candidates.csv", index=False)

    selected_rows: list[dict[str, Any]] = []
    detail: list[dict[str, Any]] = []
    for horizon, grp in candidates.groupby("horizon"):
        best_val = float(grp["val_mse"].min())
        eligible = grp[grp["val_mse"] <= best_val * (1.0 + float(args.val_tolerance))].copy()
        selected = eligible.sort_values(["complexity", "val_mse"]).iloc[0]
        pure_val = grp.sort_values(["val_mse", "complexity"]).iloc[0]
        best_test = grp.sort_values(["test_mse", "val_mse"]).iloc[0]
        selected_rows.append(
            {
                "horizon": int(horizon),
                "selected_candidate": selected["candidate"],
                "selected_family": selected["family"],
                "selection_reason": f"lowest complexity within {args.val_tolerance:.1%} of best validation MSE",
                "val_mse": float(selected["val_mse"]),
                "test_mse_reference": float(selected["test_mse"]),
                "test_mae_reference": float(selected["test_mae"]),
                "target_self_mse": float(selected["target_self_mse"]),
                "source_test_mse": float(selected["source_test_mse"]),
                "route_or_weights": selected["route_or_weights"],
            }
        )
        detail.append(
            {
                "horizon": int(horizon),
                "best_val_candidate": pure_val["candidate"],
                "best_val_family": pure_val["family"],
                "best_val_mse": float(pure_val["val_mse"]),
                "best_val_test_mse_reference": float(pure_val["test_mse"]),
                "selected_candidate": selected["candidate"],
                "selected_family": selected["family"],
                "selected_val_mse": float(selected["val_mse"]),
                "selected_test_mse_reference": float(selected["test_mse"]),
                "best_test_candidate_diagnostic": best_test["candidate"],
                "best_test_family_diagnostic": best_test["family"],
                "best_test_mse_diagnostic": float(best_test["test_mse"]),
            }
        )

    write_rows(args.out_root / "selected_matching_policy.csv", selected_rows, FIELDS)
    write_rows(
        args.out_root / "selection_diagnostics.csv",
        detail,
        [
            "horizon",
            "best_val_candidate",
            "best_val_family",
            "best_val_mse",
            "best_val_test_mse_reference",
            "selected_candidate",
            "selected_family",
            "selected_val_mse",
            "selected_test_mse_reference",
            "best_test_candidate_diagnostic",
            "best_test_family_diagnostic",
            "best_test_mse_diagnostic",
        ],
    )

    try:
        import matplotlib.pyplot as plt
    except Exception:
        plt = None
    if plt is not None:
        for horizon, grp in candidates.groupby("horizon"):
            cur = grp.sort_values(["test_mse"]).head(10)
            fig, ax = plt.subplots(figsize=(10, 4.8))
            labels = [f"{r.family}\n{r.candidate}" for r in cur.itertuples()]
            ax.bar(labels, cur["test_mse"].tolist())
            ax.set_ylabel("test MSE reference")
            ax.set_title(f"ETTm1 -> ETTm2 H{int(horizon)} cluster matching candidates")
            ax.tick_params(axis="x", labelrotation=60)
            ax.grid(True, axis="y", alpha=0.25)
            fig.tight_layout()
            fig.savefig(args.out_root / f"H{int(horizon)}_candidate_test_reference.png", dpi=180)
            plt.close(fig)

    readme = args.out_root / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# Robust cluster matching policy",
                "",
                "Candidates include hard channel routing, portrait routing, soft cluster mixtures, and validation-calibrated dynamic routing.",
                "",
                f"Selection uses validation MSE only, then picks the lowest-complexity candidate within {args.val_tolerance:.1%} of the best validation MSE.",
                "The test MSE columns are reference-only diagnostics.",
                "",
                "This implements a conservative matching rule: if a more complex shape router only wins validation by a tiny margin, keep the simpler stable cluster assignment.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(args.out_root / "selected_matching_policy.csv")
    print(args.out_root / "selection_diagnostics.csv")


if __name__ == "__main__":
    main()
