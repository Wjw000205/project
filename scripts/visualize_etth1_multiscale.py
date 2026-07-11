"""Create multi-scale visual diagnostics for ETTh1.

The figures are intended for model-debugging, not publication layout. They
focus on split shift, multi-scale level/volatility changes, and the H96 window
regime that the stage-2 anchor/MoE selector sees.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CHANNELS = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
KEY_CHANNELS = ["OT", "HULL", "MULL", "LULL"]
FEATURES = [
    "mean",
    "std",
    "last",
    "range",
    "slope",
    "diff_rms",
    "d2_rms",
    "last_minus_mean",
    "q10",
    "q90",
]


def _rolling_rms(values: pd.Series, window: int) -> pd.Series:
    return np.sqrt((values * values).rolling(window, min_periods=max(4, window // 4)).mean())


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _shade_splits(ax: plt.Axes, train_end: int, val_end: int, n: int) -> None:
    ax.axvspan(train_end, val_end, color="#f6c85f", alpha=0.13, lw=0)
    ax.axvspan(val_end, n, color="#6f9fd8", alpha=0.12, lw=0)
    ax.axvline(train_end, color="#9a6b00", lw=1.0, alpha=0.7)
    ax.axvline(val_end, color="#255f99", lw=1.0, alpha=0.7)


def _split_name(idx: int, train_end: int, val_end: int) -> str:
    if idx < train_end:
        return "train"
    if idx < val_end:
        return "val"
    return "test"


def _window_features(x: np.ndarray) -> np.ndarray:
    """Return features with shape (channels, features)."""
    t = np.arange(x.shape[0], dtype=np.float64)
    t = (t - t.mean()) / (t.std() + 1e-8)
    diff = np.diff(x, axis=0)
    d2 = np.diff(diff, axis=0)
    vals = [
        x.mean(axis=0),
        x.std(axis=0),
        x[-1],
        x.max(axis=0) - x.min(axis=0),
        ((x - x.mean(axis=0)) * t[:, None]).mean(axis=0) / (t.var() + 1e-8),
        np.sqrt(np.mean(diff * diff, axis=0)),
        np.sqrt(np.mean(d2 * d2, axis=0)),
        x[-1] - x.mean(axis=0),
        np.quantile(x, 0.10, axis=0),
        np.quantile(x, 0.90, axis=0),
    ]
    return np.stack(vals, axis=1)


def _build_window_feature_frame(
    z: np.ndarray, train_end: int, val_end: int, input_len: int, step: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    starts = np.arange(0, z.shape[0] - input_len + 1, step, dtype=np.int64)
    feats = np.stack([_window_features(z[s : s + input_len]) for s in starts], axis=0)
    split_ids = np.array([0 if s < train_end else 1 if s < val_end else 2 for s in starts])
    return starts, split_ids, feats


def _top_shifts(
    val_mean: np.ndarray,
    test_mean: np.ndarray,
    train_std: np.ndarray,
    limit: int = 20,
) -> list[dict[str, float | str]]:
    diff = (test_mean - val_mean) / np.maximum(train_std, 1e-8)
    flat = []
    for c_idx, channel in enumerate(CHANNELS):
        for f_idx, feature in enumerate(FEATURES):
            flat.append((abs(diff[c_idx, f_idx]), channel, feature, float(diff[c_idx, f_idx])))
    flat.sort(reverse=True)
    return [
        {"channel": channel, "feature": feature, "test_minus_val_trainstd": value}
        for _, channel, feature, value in flat[:limit]
    ]


def make_figures(csv_path: Path, out_dir: Path, max_rows: int, input_len: int, pred_len: int) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path, parse_dates=["date"]).iloc[:max_rows].copy()
    n = len(df)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)
    values = df[CHANNELS].to_numpy(dtype=np.float64)
    train_mean = values[:train_end].mean(axis=0)
    train_std = values[:train_end].std(axis=0)
    train_std[train_std < 1e-8] = 1.0
    z = (values - train_mean) / train_std
    dates = df["date"].astype(str).to_numpy()

    # 01 full normalized series.
    fig, axes = plt.subplots(len(CHANNELS), 1, figsize=(15, 10), sharex=True)
    x = np.arange(n)
    for i, (ax, channel) in enumerate(zip(axes, CHANNELS)):
        ax.plot(x, z[:, i], color="#222222", lw=0.45)
        _shade_splits(ax, train_end, val_end, n)
        ax.set_ylabel(channel, rotation=0, ha="right", va="center")
        lo, hi = np.quantile(z[:, i], [0.005, 0.995])
        ax.set_ylim(lo - 0.2, hi + 0.2)
        ax.grid(True, axis="y", color="#dddddd", lw=0.4)
    axes[-1].set_xticks([0, train_end, val_end, n - 1])
    axes[-1].set_xticklabels([dates[0][:10], dates[train_end][:10], dates[val_end][:10], dates[-1][:10]])
    fig.suptitle("ETTh1 full series, train-z normalized; yellow=val, blue=test", y=1.01)
    _save(fig, out_dir / "01_full_series_split_train_z.png")

    # 02 channel-time heatmap.
    fig, ax = plt.subplots(figsize=(15, 3.8))
    im = ax.imshow(np.clip(z.T, -3.0, 3.0), aspect="auto", cmap="coolwarm", vmin=-3, vmax=3)
    ax.axvline(train_end, color="black", lw=1.0)
    ax.axvline(val_end, color="black", lw=1.0)
    ax.set_yticks(np.arange(len(CHANNELS)))
    ax.set_yticklabels(CHANNELS)
    ax.set_xticks([0, train_end, val_end, n - 1])
    ax.set_xticklabels([dates[0][:10], dates[train_end][:10], dates[val_end][:10], dates[-1][:10]])
    ax.set_title("Channel-time heatmap, train-z clipped to [-3, 3]")
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    _save(fig, out_dir / "02_split_channel_heatmap_train_z.png")

    # 03 rolling level and volatility at multiple scales.
    zdf = pd.DataFrame(z, columns=CHANNELS)
    windows = [24, 168, 672]
    colors = {24: "#2f6fbb", 168: "#b25f00", 672: "#4c8b3c"}
    fig, axes = plt.subplots(len(KEY_CHANNELS), 2, figsize=(16, 10), sharex=True)
    for row, channel in enumerate(KEY_CHANNELS):
        series = zdf[channel]
        for window in windows:
            axes[row, 0].plot(x, series.rolling(window, min_periods=max(4, window // 4)).mean(), lw=0.9, color=colors[window], label=f"{window}h")
            axes[row, 1].plot(x, _rolling_rms(series.diff().fillna(0.0), window), lw=0.9, color=colors[window], label=f"{window}h")
        for col in range(2):
            _shade_splits(axes[row, col], train_end, val_end, n)
            axes[row, col].grid(True, color="#e2e2e2", lw=0.4)
        axes[row, 0].set_ylabel(channel, rotation=0, ha="right", va="center")
    axes[0, 0].set_title("rolling mean level")
    axes[0, 1].set_title("rolling diff RMS volatility")
    axes[0, 0].legend(loc="upper left", ncol=3, frameon=False)
    axes[-1, 0].set_xticks([0, train_end, val_end, n - 1])
    axes[-1, 0].set_xticklabels([dates[0][:10], dates[train_end][:10], dates[val_end][:10], dates[-1][:10]])
    axes[-1, 1].set_xticks([0, train_end, val_end, n - 1])
    axes[-1, 1].set_xticklabels([dates[0][:10], dates[train_end][:10], dates[val_end][:10], dates[-1][:10]])
    fig.suptitle("Multi-scale level and volatility, key channels", y=1.01)
    _save(fig, out_dir / "03_rolling_stats_multiscale.png")

    # 04 H96 local windows around split boundaries and midpoints.
    candidates = [
        ("train_tail", max(0, train_end - 2 * (input_len + pred_len))),
        ("val_start", train_end),
        ("val_mid", (train_end + val_end - input_len - pred_len) // 2),
        ("val_late", max(train_end, val_end - input_len - pred_len)),
        ("test_start", val_end),
        ("test_mid", (val_end + n - input_len - pred_len) // 2),
        ("test_late", max(val_end, n - input_len - pred_len)),
    ]
    span = input_len + pred_len
    fig, axes = plt.subplots(len(candidates), 1, figsize=(14, 11), sharex=True)
    offsets = np.arange(len(KEY_CHANNELS)) * 4.0
    for ax, (label, start) in zip(axes, candidates):
        seg = z[start : start + span, [CHANNELS.index(c) for c in KEY_CHANNELS]]
        local_x = np.arange(span)
        for j, channel in enumerate(KEY_CHANNELS):
            ax.plot(local_x, seg[:, j] + offsets[j], lw=0.8, label=channel)
        ax.axvspan(0, input_len, color="#d9d9d9", alpha=0.35, lw=0)
        ax.axvline(input_len, color="#b00020", lw=1.0)
        ax.set_yticks(offsets)
        ax.set_yticklabels(KEY_CHANNELS)
        ax.set_title(f"{label} | start={start} | {_split_name(start, train_end, val_end)} | {dates[start][:16]}")
        ax.grid(True, axis="x", color="#e0e0e0", lw=0.35)
    axes[-1].set_xlabel("hours in local window; grey=input96, right=forecast horizon96")
    _save(fig, out_dir / "04_h96_local_windows_val_test.png")

    # 05 feature-shift heatmap between val and test input windows.
    starts, split_ids, feats = _build_window_feature_frame(z, train_end, val_end, input_len, step=4)
    train_feats = feats[split_ids == 0]
    val_feats = feats[split_ids == 1]
    test_feats = feats[split_ids == 2]
    train_feat_mean = train_feats.mean(axis=0)
    train_feat_std = train_feats.std(axis=0)
    train_feat_std[train_feat_std < 1e-8] = 1.0
    val_mean = val_feats.mean(axis=0)
    test_mean = test_feats.mean(axis=0)
    shift = (test_mean - val_mean) / train_feat_std
    fig, ax = plt.subplots(figsize=(12, 5.0))
    im = ax.imshow(np.clip(shift, -3.0, 3.0), cmap="coolwarm", vmin=-3, vmax=3)
    ax.set_yticks(np.arange(len(CHANNELS)))
    ax.set_yticklabels(CHANNELS)
    ax.set_xticks(np.arange(len(FEATURES)))
    ax.set_xticklabels(FEATURES, rotation=35, ha="right")
    ax.set_title("Val-to-test shift by H96 input-window feature, units=train-window std")
    for i in range(shift.shape[0]):
        for j in range(shift.shape[1]):
            ax.text(j, i, f"{shift[i, j]:.1f}", ha="center", va="center", fontsize=7, color="black")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    _save(fig, out_dir / "05_val_test_feature_shift_heatmap.png")

    # 06 PCA of H96 window features, to visualize domain separation.
    flat = feats.reshape(feats.shape[0], -1)
    mu = flat[split_ids == 0].mean(axis=0)
    sig = flat[split_ids == 0].std(axis=0)
    sig[sig < 1e-8] = 1.0
    flat_z = (flat - mu) / sig
    centered = flat_z - flat_z.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    pc = centered @ vh[:2].T
    colors_split = np.array(["#555555", "#cc8a00", "#2d75bb"])
    labels_split = ["train", "val", "test"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for sid, label in enumerate(labels_split):
        mask = split_ids == sid
        axes[0].scatter(pc[mask, 0], pc[mask, 1], s=7, alpha=0.55, color=colors_split[sid], label=label)
        axes[1].plot(starts[mask], pc[mask, 0], ".", ms=3.2, alpha=0.5, color=colors_split[sid], label=label)
    axes[0].set_title("PCA of H96 input features")
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    axes[0].legend(frameon=False)
    axes[1].set_title("PC1 over time")
    axes[1].set_xlabel("window start")
    axes[1].set_ylabel("PC1")
    axes[1].axvline(train_end, color="black", lw=1.0)
    axes[1].axvline(val_end, color="black", lw=1.0)
    _save(fig, out_dir / "06_h96_feature_pca_domain_separation.png")

    # 07 frequency spectra by split.
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    split_ranges = {
        "train": (0, train_end, "#555555"),
        "val": (train_end, val_end, "#cc8a00"),
        "test": (val_end, n, "#2d75bb"),
    }
    for ax, channel in zip(axes.ravel(), KEY_CHANNELS):
        c_idx = CHANNELS.index(channel)
        for label, (lo, hi, color) in split_ranges.items():
            series = z[lo:hi, c_idx]
            series = series - np.mean(series)
            spec = np.abs(np.fft.rfft(series)) ** 2
            freq = np.fft.rfftfreq(series.shape[0], d=1.0)
            mask = freq > 0
            period = 1.0 / freq[mask]
            power = spec[mask] / (spec[mask].sum() + 1e-12)
            keep = (period >= 4) & (period <= 1000)
            ax.plot(period[keep], power[keep], lw=0.8, alpha=0.85, color=color, label=label)
        ax.axvline(24, color="#b00020", lw=0.8, ls="--")
        ax.axvline(168, color="#006b35", lw=0.8, ls="--")
        ax.set_xscale("log")
        ax.invert_xaxis()
        ax.set_title(channel)
        ax.grid(True, color="#e5e5e5", lw=0.4)
    axes[0, 0].legend(frameon=False)
    axes[-1, 0].set_xlabel("period in hours, log scale")
    axes[-1, 1].set_xlabel("period in hours, log scale")
    fig.suptitle("Normalized frequency spectra by split; dashed=24h/168h", y=1.01)
    _save(fig, out_dir / "07_frequency_spectra_by_split.png")

    # 08 row-standardized multi-scale level/energy heatmap.
    rows = []
    row_labels = []
    for channel in CHANNELS:
        series = zdf[channel]
        for window in [24, 96, 168, 336, 672]:
            rows.append(series.rolling(window, min_periods=max(4, window // 4)).mean().to_numpy())
            row_labels.append(f"{channel}:mean{window}")
        for window in [24, 96, 168, 336]:
            rows.append(_rolling_rms(series.diff().fillna(0.0), window).to_numpy())
            row_labels.append(f"{channel}:vol{window}")
    mat = np.vstack(rows)
    train_row_mean = np.nanmean(mat[:, :train_end], axis=1, keepdims=True)
    train_row_std = np.nanstd(mat[:, :train_end], axis=1, keepdims=True)
    train_row_std[train_row_std < 1e-8] = 1.0
    mat_z = (mat - train_row_mean) / train_row_std
    mat_z = np.nan_to_num(mat_z, nan=0.0, posinf=0.0, neginf=0.0)
    fig, ax = plt.subplots(figsize=(16, 12))
    im = ax.imshow(np.clip(mat_z, -3.0, 3.0), aspect="auto", cmap="coolwarm", vmin=-3, vmax=3)
    ax.axvline(train_end, color="black", lw=1.0)
    ax.axvline(val_end, color="black", lw=1.0)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=6)
    ax.set_xticks([0, train_end, val_end, n - 1])
    ax.set_xticklabels([dates[0][:10], dates[train_end][:10], dates[val_end][:10], dates[-1][:10]])
    ax.set_title("Multi-scale level/volatility energy heatmap, row standardized on train")
    fig.colorbar(im, ax=ax, fraction=0.018, pad=0.01)
    _save(fig, out_dir / "08_multiscale_energy_heatmap.png")

    # 09 hour-of-day and day-of-week signatures.
    fig, axes = plt.subplots(len(KEY_CHANNELS), 2, figsize=(13, 10), sharey="row")
    split_slices = {
        "train": slice(0, train_end),
        "val": slice(train_end, val_end),
        "test": slice(val_end, n),
    }
    split_colors = {"train": "#555555", "val": "#cc8a00", "test": "#2d75bb"}
    dt = pd.to_datetime(df["date"])
    for row, channel in enumerate(KEY_CHANNELS):
        for label, sl in split_slices.items():
            part = pd.DataFrame({"hour": dt.iloc[sl].dt.hour, "dow": dt.iloc[sl].dt.dayofweek, "v": zdf[channel].iloc[sl].to_numpy()})
            hod = part.groupby("hour")["v"].mean().reindex(range(24))
            dow = part.groupby("dow")["v"].mean().reindex(range(7))
            axes[row, 0].plot(range(24), hod, lw=1.1, color=split_colors[label], label=label)
            axes[row, 1].plot(range(7), dow, lw=1.1, color=split_colors[label], label=label)
        axes[row, 0].set_ylabel(channel, rotation=0, ha="right", va="center")
        axes[row, 0].grid(True, color="#e5e5e5", lw=0.4)
        axes[row, 1].grid(True, color="#e5e5e5", lw=0.4)
    axes[0, 0].set_title("mean by hour-of-day")
    axes[0, 1].set_title("mean by day-of-week")
    axes[0, 0].legend(frameon=False, ncol=3)
    axes[-1, 0].set_xlabel("hour")
    axes[-1, 1].set_xlabel("day of week")
    _save(fig, out_dir / "09_calendar_signature_by_split.png")

    # 10 contact sheet for quick review.
    sheet_files = [
        "01_full_series_split_train_z.png",
        "02_split_channel_heatmap_train_z.png",
        "03_rolling_stats_multiscale.png",
        "04_h96_local_windows_val_test.png",
        "05_val_test_feature_shift_heatmap.png",
        "06_h96_feature_pca_domain_separation.png",
        "07_frequency_spectra_by_split.png",
        "08_multiscale_energy_heatmap.png",
        "09_calendar_signature_by_split.png",
    ]
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    for ax, name in zip(axes.ravel(), sheet_files):
        img = mpimg.imread(out_dir / name)
        ax.imshow(img)
        ax.set_title(name, fontsize=8)
        ax.axis("off")
    _save(fig, out_dir / "10_contact_sheet.png")

    split_stats = {}
    for label, (lo, hi, _) in split_ranges.items():
        raw = values[lo:hi]
        zz = z[lo:hi]
        split_stats[label] = {
            "start": int(lo),
            "end": int(hi),
            "start_date": dates[lo],
            "end_date": dates[hi - 1],
            "train_z_mean": {channel: float(zz[:, i].mean()) for i, channel in enumerate(CHANNELS)},
            "train_z_std": {channel: float(zz[:, i].std()) for i, channel in enumerate(CHANNELS)},
            "raw_mean": {channel: float(raw[:, i].mean()) for i, channel in enumerate(CHANNELS)},
            "raw_std": {channel: float(raw[:, i].std()) for i, channel in enumerate(CHANNELS)},
        }
    summary = {
        "csv_path": str(csv_path),
        "out_dir": str(out_dir),
        "max_rows": max_rows,
        "n_rows": n,
        "split": {
            "train_end": train_end,
            "val_end": val_end,
            "test_end": n,
            "train": [0, train_end],
            "val": [train_end, val_end],
            "test": [val_end, n],
        },
        "split_stats": split_stats,
        "top_val_to_test_feature_shifts": _top_shifts(val_mean, test_mean, train_feat_std),
        "generated_files": sheet_files + ["10_contact_sheet.png"],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-path", type=Path, default=Path("data/ETTh1.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/etth1_multiscale_visual_diagnostic_20260710"))
    parser.add_argument("--max-rows", type=int, default=14400)
    parser.add_argument("--input-len", type=int, default=96)
    parser.add_argument("--pred-len", type=int, default=96)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = make_figures(
        csv_path=args.csv_path,
        out_dir=args.out_dir,
        max_rows=args.max_rows,
        input_len=args.input_len,
        pred_len=args.pred_len,
    )
    print(json.dumps({"out_dir": summary["out_dir"], "generated_files": summary["generated_files"]}, indent=2))


if __name__ == "__main__":
    main()
