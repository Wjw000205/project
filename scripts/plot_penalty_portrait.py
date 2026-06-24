"""Plot per-cluster '对症' penalty portraits from penalty_portrait.json.

Reads the diagnostic (frozen-backbone train-residual penalty values per cluster),
normalizes each penalty by its per-cell global mean (=> per-cluster relative
deficit, centered at 1.0), and renders clean heatmaps for the multi-cluster cells
(K=1 cells are degenerate under this normalization and are skipped).
"""
from __future__ import annotations
import json, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

JSON = r"outputs/penalty_diagnostic/penalty_portrait.json"
OUT_PNG = r"outputs/penalty_diagnostic/penalty_portrait_heatmap.png"
OUT_PDF = r"outputs/penalty_diagnostic/penalty_portrait_heatmap.pdf"

d = json.load(open(JSON, encoding="utf-8"))
P = d["penalty_names"]
cells = d["cells"]

# keep only informative multi-cluster cells (K>=3 → clean specialization)
panels = [(name, c) for name, c in cells.items() if int(c["n_clusters"]) >= 3]
# order: ETTm1 then Weather then any other multi-cluster
panels.sort(key=lambda kv: (0 if kv[0].startswith("ETT") else 1, kv[0]))

fig, axes = plt.subplots(
    1, len(panels), figsize=(1.0 + 5.2 * len(panels), 3.6),
    gridspec_kw={"width_ratios": [int(c["n_clusters"]) + 3 for _, c in panels]},
)
if len(panels) == 1:
    axes = [axes]

cmap = plt.get_cmap("RdBu_r")
norm = TwoSlopeNorm(vmin=0.0, vcenter=1.0, vmax=3.0)

for ax, (name, c) in zip(axes, panels):
    K = int(c["n_clusters"])
    g = np.array(c["penalty_global_mean"], dtype=float)
    raw = np.array(c["portrait_raw"], dtype=float)            # [K, P]
    M = raw / np.where(g == 0, 1.0, g)                        # [K, P] relative deficit
    Mc = np.clip(M, 0.0, 3.0)
    im = ax.imshow(Mc, aspect="auto", cmap=cmap, norm=norm)

    ax.set_xticks(range(len(P)))
    ax.set_xticklabels(P, rotation=45, ha="right", fontsize=8)
    sizes = c.get("cluster_sizes", list(range(K)))
    ax.set_yticks(range(K))
    ax.set_yticklabels([f"cluster {k}\n(n={sizes[k]})" for k in range(K)], fontsize=8)
    ax.set_title(name.replace("_", " "), fontsize=11, fontweight="bold")

    # annotate values; box the per-cluster top-3 (the 对症 selection)
    for k in range(K):
        order = np.argsort(-M[k])[:3]
        for j in range(len(P)):
            val = M[k, j]
            txt_color = "white" if (val > 2.0 or val < 0.35) else "black"
            ax.text(j, k, f"{val:.2f}", ha="center", va="center",
                    fontsize=6.5, color=txt_color)
            if j in order:
                ax.add_patch(plt.Rectangle((j - 0.5, k - 0.5), 1, 1, fill=False,
                                           edgecolor="lime", lw=2.0))

cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
cbar.set_label("per-cluster residual error / cell mean\n(>1 = backbone weaker here = on-target penalty)", fontsize=8)
fig.suptitle("PKR-MoE: per-cluster shape-residual diagnostic (frozen backbone, train) — green box = top-3 selected penalty",
             fontsize=12, fontweight="bold")
os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
fig.savefig(OUT_PDF, bbox_inches="tight")
print("saved:", OUT_PNG, "and", OUT_PDF)
print("panels:", [n for n, _ in panels])
