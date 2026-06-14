from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd


DATASETS = ("ETTh1", "ETTh2", "ETTm1", "ETTm2")


def _lag_corr(x: np.ndarray, lag: int) -> tuple[float, float]:
    lag = int(lag)
    if lag <= 0 or lag >= len(x) - 2:
        return float("nan"), float("nan")
    a = x[:-lag]
    b = x[lag:]
    values = []
    for c in range(x.shape[1]):
        ac = a[:, c]
        bc = b[:, c]
        if np.std(ac) < 1.0e-8 or np.std(bc) < 1.0e-8:
            values.append(np.nan)
        else:
            values.append(float(np.corrcoef(ac, bc)[0, 1]))
    arr = np.asarray(values, dtype=np.float64)
    return float(np.nanmean(arr)), float(np.nanmean(np.abs(arr)))


def main() -> None:
    repo = Path.cwd()
    out = repo / "outputs" / "h720_moe_route_mode_diagnostic"
    out.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float | str | int]] = []
    for dataset in DATASETS:
        df = pd.read_csv(repo / "data" / f"{dataset}.csv")
        date = pd.to_datetime(df.iloc[:, 0])
        deltas = date.diff().dropna().dt.total_seconds().to_numpy(dtype=np.float64) / 60.0
        step_minutes = float(np.median(deltas))
        horizon_days = 720.0 * step_minutes / (60.0 * 24.0)

        values = df.iloc[:, 1:].to_numpy(dtype=np.float64)
        train_n = int(len(values) * 0.6)
        train = values[:train_n]
        train = (train - train.mean(axis=0, keepdims=True)) / np.clip(train.std(axis=0, keepdims=True), 1.0e-6, None)

        daily_lag = int(round(24.0 * 60.0 / step_minutes))
        weekly_lag = int(round(7.0 * 24.0 * 60.0 / step_minutes))
        lag_7p5d = int(round(7.5 * 24.0 * 60.0 / step_minutes))

        h_mean, h_abs = _lag_corr(train, 720)
        d_mean, d_abs = _lag_corr(train, daily_lag)
        w_mean, w_abs = _lag_corr(train, weekly_lag)
        s_mean, s_abs = _lag_corr(train, lag_7p5d)

        rows.append(
            {
                "dataset": dataset,
                "step_minutes": step_minutes,
                "h720_days": horizon_days,
                "daily_lag_steps": daily_lag,
                "weekly_lag_steps": weekly_lag,
                "lag_7p5d_steps": lag_7p5d,
                "train_autocorr_lag720_mean": h_mean,
                "train_autocorr_lag720_abs_mean": h_abs,
                "train_autocorr_daily_mean": d_mean,
                "train_autocorr_daily_abs_mean": d_abs,
                "train_autocorr_weekly_mean": w_mean,
                "train_autocorr_weekly_abs_mean": w_abs,
                "train_autocorr_7p5d_mean": s_mean,
                "train_autocorr_7p5d_abs_mean": s_abs,
            }
        )

    csv_path = out / "h720_frequency_signal_diagnostic.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# H=720 Frequency and Signal Diagnostic",
        "",
        "| Dataset | Step | H=720 span | lag720 autocorr | daily autocorr | weekly autocorr | 7.5d autocorr |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {dataset} | {step:.0f} min | {span:.1f} d | {lag720:.3f} | {daily:.3f} | {weekly:.3f} | {seven:.3f} |".format(
                dataset=row["dataset"],
                step=float(row["step_minutes"]),
                span=float(row["h720_days"]),
                lag720=float(row["train_autocorr_lag720_mean"]),
                daily=float(row["train_autocorr_daily_mean"]),
                weekly=float(row["train_autocorr_weekly_mean"]),
                seven=float(row["train_autocorr_7p5d_mean"]),
            )
        )
    lines.extend(
        [
            "",
            "All autocorrelation values are computed on the train split only, averaged across channels after train-split z-score normalization.",
        ]
    )
    md_path = out / "h720_frequency_signal_diagnostic.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
