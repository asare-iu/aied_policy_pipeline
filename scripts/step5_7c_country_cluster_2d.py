#!/usr/bin/env python3
"""
Step 5.7c: Country clustering visualization (2D projection of country vectors).

Inputs:
- data/derived/step5_6_pca_followthrough/country_pc_means.csv
- data/derived/step5_5_pca_interpretation/pca_pc_labels_final.csv

Output:
- figures/step5_pca_full/pack/country_clusters_2d.png

Method:
- Use selected PC mean columns as country vectors.
- Standardize.
- KMeans clustering in that space.
- PCA projection to 2D for visualization.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


OUT = Path("figures/step5_pca_full/pack/country_clusters_2d.png")
OUT.parent.mkdir(parents=True, exist_ok=True)

COUNTRY_CSV = Path("data/derived/step5_6_pca_followthrough/country_pc_means.csv")
LABELS_CSV = Path("data/derived/step5_5_pca_interpretation/pca_pc_labels_final.csv")


def main() -> None:
    df = pd.read_csv(COUNTRY_CSV)
    labels = pd.read_csv(LABELS_CSV)
    pc_label = {f"pc{int(r.pc)}": r.label for _, r in labels.iterrows()}

    pcs = ["pc13", "pc14", "pc17", "pc18", "pc19", "pc24", "pc25"]
    X = df[pcs].fillna(0.0).values
    Xs = StandardScaler().fit_transform(X)

    k = 6
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster = km.fit_predict(Xs)

    pca2 = PCA(n_components=2, random_state=42)
    Z = pca2.fit_transform(Xs)

    fig_title = "Country Clusters (KMeans) Visualized in 2D (PCA Projection)"
    subtitle = "Vector space: " + " | ".join([f"{p.upper()}={pc_label.get(p, p)}" for p in pcs])

    plt.figure(figsize=(10, 7))
    plt.scatter(Z[:, 0], Z[:, 1], c=cluster, s=(df["n_chunks"].astype(float) ** 0.5), alpha=0.8)

    # Label top countries by chunk count for readability
    top = df.sort_values("n_chunks", ascending=False).head(20).copy()
    top_idx = top.index.to_list()
    for i in top_idx:
        plt.text(Z[i, 0], Z[i, 1], str(df.loc[i, "country"]), fontsize=8)

    plt.xlabel("PC-Axis 1 (country vector projection)")
    plt.ylabel("PC-Axis 2 (country vector projection)")
    plt.title(fig_title)
    plt.suptitle(subtitle, fontsize=9, y=0.94)

    plt.tight_layout()
    plt.savefig(OUT, dpi=200)
    plt.close()

    print(f"Wrote: {OUT}")


if __name__ == "__main__":
    main()
