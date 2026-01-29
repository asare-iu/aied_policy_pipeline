#!/usr/bin/env python3
"""
Step 5.7d: Density/contour plots for chunk distributions in PCA space.

Inputs:
- data/derived/step5_pca_full/pca_full_scores.csv
- data/derived/step4_5_normativity_gate/chunks_normative_primary.jsonl

Outputs:
- figures/step5_pca_full/pack/chunk_density_primary_pc1_pc2.png
- figures/step5_pca_full/pack/chunk_density_excluded_pc1_pc2.png
- figures/step5_pca_full/pack/chunk_density_overlay_pc1_pc2.png

Method:
- 2D histogram density estimate (numpy histogram2d)
- contour lines plotted with matplotlib (default colormap)
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


OUT_DIR = Path("figures/step5_pca_full/pack")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SCORES_CSV = Path("data/derived/step5_pca_full/pca_full_scores.csv")
PRIMARY_JSONL = Path("data/derived/step4_5_normativity_gate/chunks_normative_primary.jsonl")


def load_primary_ids(path: Path) -> set[str]:
    ids = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            ids.add(r["chunk_id"])
    return ids


def hist2d_density(x: np.ndarray, y: np.ndarray, bins: int = 120):
    H, xedges, yedges = np.histogram2d(x, y, bins=bins)
    # Convert bin edges to bin centers for contour plotting
    xc = (xedges[:-1] + xedges[1:]) / 2
    yc = (yedges[:-1] + yedges[1:]) / 2
    return H.T, xc, yc  # transpose so H aligns with meshgrid(xc, yc)


def plot_contour(H, xc, yc, title: str, out_path: Path):
    X, Y = np.meshgrid(xc, yc)
    plt.figure(figsize=(9, 7))
    plt.contour(X, Y, H, levels=10)
    plt.xlabel("PC1 score")
    plt.ylabel("PC2 score")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    primary_ids = load_primary_ids(PRIMARY_JSONL)
    df = pd.read_csv(SCORES_CSV, usecols=["chunk_id", "pc1", "pc2"])
    df["is_primary"] = df["chunk_id"].isin(primary_ids)

    prim = df[df["is_primary"]]
    excl = df[~df["is_primary"]]

    Hp, xcp, ycp = hist2d_density(prim["pc1"].values, prim["pc2"].values)
    He, xce, yce = hist2d_density(excl["pc1"].values, excl["pc2"].values)

    plot_contour(Hp, xcp, ycp, "Chunk Density Contours (Normative Primary) in PC1–PC2", OUT_DIR / "chunk_density_primary_pc1_pc2.png")
    plot_contour(He, xce, yce, "Chunk Density Contours (Excluded by Gate) in PC1–PC2", OUT_DIR / "chunk_density_excluded_pc1_pc2.png")

    # Overlay
    Xp, Yp = np.meshgrid(xcp, ycp)
    Xe, Ye = np.meshgrid(xce, yce)

    plt.figure(figsize=(9, 7))
    plt.contour(Xe, Ye, He, levels=10)
    plt.contour(Xp, Yp, Hp, levels=10)
    plt.xlabel("PC1 score")
    plt.ylabel("PC2 score")
    plt.title("Chunk Density Contours Overlay (Excluded + Normative Primary) in PC1–PC2")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "chunk_density_overlay_pc1_pc2.png", dpi=200)
    plt.close()

    print(f"Wrote density figures to: {OUT_DIR}")


if __name__ == "__main__":
    main()
