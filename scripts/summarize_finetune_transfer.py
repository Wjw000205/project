from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=Path, default=Path("outputs/ettm1_finetune_transfer"))
    args = ap.parse_args()

    out = args.out_root
    df = pd.read_csv(out / "finetune_vs_zero_shot.csv")
    numeric_cols = [
        "source_test_mse",
        "target_self_test_mse",
        "zero_shot_mse",
        "finetune_test_mse",
        "gain_mse_vs_zero_shot",
        "gain_pct_vs_zero_shot",
        "finetune_lr",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    paper_cols = [
        "target",
        "pred_len",
        "source_test_mse",
        "target_self_test_mse",
        "zero_shot_mse",
        "finetune_test_mse",
        "gain_mse_vs_zero_shot",
        "gain_pct_vs_zero_shot",
        "finetune_lr",
        "finetune_best_epoch",
        "zero_shot_route_uses_train_only",
        "normalize_train_only",
        "cluster_train_only",
        "resample_enable",
        "resample_method",
    ]
    paper = df[paper_cols].sort_values(["pred_len", "target"])
    paper.to_csv(out / "fine_tune_zero_shot_paper_table.csv", index=False)

    summary_target = (
        df.groupby("target")
        .agg(
            settings=("target", "count"),
            mean_zero_shot_mse=("zero_shot_mse", "mean"),
            mean_finetune_mse=("finetune_test_mse", "mean"),
            mean_gain_mse=("gain_mse_vs_zero_shot", "mean"),
            mean_gain_pct=("gain_pct_vs_zero_shot", "mean"),
            positive_settings=("gain_mse_vs_zero_shot", lambda s: int((s > 0).sum())),
        )
        .reset_index()
    )
    summary_target.to_csv(out / "summary_by_target.csv", index=False)

    summary_horizon = (
        df.groupby("pred_len")
        .agg(
            settings=("pred_len", "count"),
            mean_zero_shot_mse=("zero_shot_mse", "mean"),
            mean_finetune_mse=("finetune_test_mse", "mean"),
            mean_gain_mse=("gain_mse_vs_zero_shot", "mean"),
            mean_gain_pct=("gain_pct_vs_zero_shot", "mean"),
            positive_settings=("gain_mse_vs_zero_shot", lambda s: int((s > 0).sum())),
        )
        .reset_index()
    )
    summary_horizon.to_csv(out / "summary_by_horizon.csv", index=False)

    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=160)
        for target, group in df.sort_values("pred_len").groupby("target"):
            ax.plot(
                group["pred_len"],
                group["gain_pct_vs_zero_shot"],
                marker="o",
                linewidth=2,
                label=target,
            )
        ax.axhline(0, color="#444444", linewidth=1)
        ax.set_xlabel("Prediction horizon")
        ax.set_ylabel("Fine-tune gain vs zero-shot (%)")
        ax.set_title("ETTm1 source: fine-tune transfer vs zero-shot")
        ax.set_xticks([96, 192, 336, 720])
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(out / "fine_tune_gain_pct.png")
        plt.close(fig)
    except Exception as exc:
        print(f"Plot skipped: {exc}")

    print(f"Saved: {out / 'fine_tune_zero_shot_paper_table.csv'}")
    print("\nBy target:")
    print(summary_target.to_string(index=False))
    print("\nBy horizon:")
    print(summary_horizon.to_string(index=False))


if __name__ == "__main__":
    main()
