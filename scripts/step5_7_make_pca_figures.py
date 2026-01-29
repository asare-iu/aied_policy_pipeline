#!/usr/bin/env python3
"""
Step 5.7: Generate standard PCA figures for the full-corpus PCA round.

Inputs:
- data/derived/step5_pca_full/pca_full_metadata.json
- data/derived/step5_pca_full/pca_full_scores.csv
- data/derived/step4_5_normativity_gate/chunks_normative_primary.jsonl
- data/derived/step5_6_pca_followthrough/country_pc_means.csv
- data/derived/step5_5_pca_interpretation/pca_pc_labels_final.csv

Outputs:
- figures/step5_pca_full/pca_full_scree.png
- figures/step5_pca_full/pca_full_chunks_pc1_pc2_normativity.png
- figures/step5_pca_full/pca_full_country_pc14_pc19.png
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


FIG_DIR = Path("figures/step5_pca_full")
FIG_DIR.mkdir(parents=True, exist_ok=True)

META_JSON = Path("data/derived/step5_pca_full/pca_full_metadata.json")
SCORES_CSV = Path("data/derived/step5_pca_full/pca_full_scores.csv")
PRIMARY_JSONL = Path("data/derived/step4_5_normativity_gate/chunks_normative_primary.jsonl")
COUNTRY_CSV = Path("data/derived/step5_6_pca_followthrough/country_pc_means.csv")
LABELS_CSV = Path("data/derived/step5_5_pca_interpretation/pca_pc_labels_final.csv")


def load_explained_variance(meta_path: Path) -> list[float]:
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return list(meta.get("explained_variance_ratio", []))


def load_primary_chunk_ids(path: Path) -> set[str]:
    ids = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            ids.add(r["chunk_id"])
    return ids


def load_pc_labels(path: Path) -> dict[str, str]:
    df = pd.read_csv(path)
    return {f"pc{int(r.pc)}": r.label for _, r in df.iterrows()}


def make_scree_plot(explained: list[float]) -> Path:
    out = FIG_DIR / "pca_full_scree.png"
    xs = list(range(1, len(explained) + 1))
    cum = []
    s = 0.0
    for v in explained:
        s += float(v)
        cum.append(s)

    plt.figure()
    plt.plot(xs, explained, marker="o")
    plt.xlabel("Component")
    plt.ylabel("Explained variance ratio")
    plt.title("Full-corpus TF-IDF TruncatedSVD: Scree Plot")

    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()

    out2 = FIG_DIR / "pca_full_scree_cumulative.png"
    plt.figure()
    plt.plot(xs, cum, marker="o")
    plt.xlabel("Component")
    plt.ylabel("Cumulative explained variance ratio")
    plt.title("Full-corpus TF-IDF TruncatedSVD: Cumulative Explained Variance")
    plt.ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(out2, dpi=200)
    plt.close()

    return out


def make_chunk_scatter(primary_ids: set[str]) -> Path:
    out = FIG_DIR / "pca_full_chunks_pc1_pc2_normativity.png"

    usecols = ["chunk_id", "pc1", "pc2"]
    df = pd.read_csv(SCORES_CSV, usecols=usecols)
    df["is_primary"] = df["chunk_id"].isin(primary_ids)

    df_primary = df[df["is_primary"]]
    df_excl = df[~df["is_primary"]]

    plt.figure()
    plt.scatter(df_excl["pc1"], df_excl["pc2"], s=6, alpha=0.2, label="Excluded by gate")
    plt.scatter(df_primary["pc1"], df_primary["pc2"], s=6, alpha=0.2, label="Normative primary")

    plt.xlabel("PC1 score")
    plt.ylabel("PC2 score")
    plt.title("Chunks in PCA Space (PC1 vs PC2), Colored by Normativity Gate")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()
    return out


def make_country_centroid_plot(pc_x: str, pc_y: str, label_map: dict[str, str]) -> Path:
    out = FIG_DIR / f"pca_full_country_{pc_x}_{pc_y}.png"
    df = pd.read_csv(COUNTRY_CSV)

    # Size scaling for readability
    sizes = df["n_chunks"].apply(lambda n: 20.0 * math.sqrt(max(n, 1)) / 10.0)

    plt.figure()
    plt.scatter(df[pc_x], df[pc_y], s=sizes, alpha=0.6)

    # Label a small number of most-represented countries for readability
    top = df.sort_values("n_chunks", ascending=False).head(15)
    for _, r in top.iterrows():
        plt.text(r[pc_x], r[pc_y], str(r["country"]), fontsize=8)

    plt.xlabel(f"{pc_x.upper()}: {label_map.get(pc_x, pc_x)}")
    plt.ylabel(f"{pc_y.upper()}: {label_map.get(pc_y, pc_y)}")
    plt.title(f"Country Centroids in PCA Space ({pc_x.upper()} vs {pc_y.upper()})")

    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()
    return out


def main() -> None:
    explained = load_explained_variance(META_JSON)
    if not explained:
        raise SystemExit(f"No explained_variance_ratio found in {META_JSON}")

    primary_ids = load_primary_chunk_ids(PRIMARY_JSONL)
    label_map = load_pc_labels(LABELS_CSV)

    make_scree_plot(explained)
    make_chunk_scatter(primary_ids)
    make_country_centroid_plot("pc14", "pc19", label_map)

    print(f"Wrote figures to: {FIG_DIR}")


if __name__ == "__main__":
    main()
