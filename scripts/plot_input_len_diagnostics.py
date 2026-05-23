from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_case(root: Path, input_len: int):
    run_dir = root / f"input{input_len}"
    z = np.load(run_dir / "prediction_intermediates.npz")
    meta = json.loads((run_dir / "prediction_intermediates_meta.json").read_text(encoding="utf-8"))
    return z, meta


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot ETTh2 input-length diagnostic prediction intermediates.")
    ap.add_argument("--root", default="outputs/input_len_diagnostics/ETTh2_H96")
    ap.add_argument("--channels", nargs="+", default=["HULL", "MULL", "LUFL", "OT"])
    ap.add_argument("--out", default="outputs/input_len_diagnostics/ETTh2_H96/prediction_intermediate_examples.png")
    args = ap.parse_args()

    root = Path(args.root)
    cases = {336: load_case(root, 336), 96: load_case(root, 96)}
    channel_names = cases[336][1]["channel_names"]

    fig, axes = plt.subplots(len(args.channels), 2, figsize=(13, 3.0 * len(args.channels)), sharex=False)
    if len(args.channels) == 1:
        axes = np.asarray([axes])
    for r, channel in enumerate(args.channels):
        c = channel_names.index(channel)
        for col, input_len in enumerate([336, 96]):
            z, _ = cases[input_len]
            y = z["y_true"][:, c, :]
            base = z["y_base"][:, c, :]
            raw = z["y_residual_raw"][:, c, :]
            final = z["y_final"][:, c, :]
            mse_per_sample = ((final - y) ** 2).mean(axis=1)
            idx = int(mse_per_sample.argmax())
            ax = axes[r, col]
            ax.plot(y[idx], label="true", linewidth=2.0, color="black")
            ax.plot(base[idx], label="base", linewidth=1.5)
            ax.plot(raw[idx], label="raw residual", linewidth=1.2, alpha=0.8)
            ax.plot(final[idx], label="final", linewidth=1.5)
            ax.set_title(f"{channel} input={input_len} sample={idx} final_mse={mse_per_sample[idx]:.3f}")
            ax.grid(alpha=0.2)
            if r == 0 and col == 1:
                ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    print(out)


if __name__ == "__main__":
    main()
