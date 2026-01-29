#!/usr/bin/env python3
"""
Step 5.7b: Generate a thorough PCA figure pack for the full-corpus PCA round.

Inputs:
- data/derived/step5_pca_full/pca_full_metadata.json
- data/derived/step5_pca_full/pca_full_scores.csv
- data/derived/step5_pca_full/pca_full_top_terms.csv
- data/derived/step4_5_normativity_gate/chunks_normative_primary.jsonl
- data/derived/step5_6_pca_followthrough/country_pc_means.csv
- data/derived/step5_6_pca_followthrough/gate_skew_by_pc.csv
- data/derived/step5_5_pca_interpretation/pca_pc_labels_final.csv

Outputs (all under figures/step5_pca_full/pack/):
- scree + cumulative scree + elbow (difference)
- chunk scatters for key PC pairs (PC1-2, 1-3, 2-3, 13-19, 14-19, 18-19, 19-24)
  with normativity gate overlay (primary vs excluded)
- gate skew bar plot (absolute mean differences)
- country centroid scatters for key PC pairs, labeled by top n_chunks
- top terms bar charts for selected PCs (positive/negative)
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import matplotlib.pyplot as plt


OUT_DIR = Path("figures/step5_pca_full/pack")
OUT_DIR.mkdir(parents=True, exist_ok=True)

META_JSON = Path("data/derived/step5_pca_full/pca_full_metadata.json")
SCORES_CSV = Path("data/derived/step5_pca_full/pca_full_scores.csv")
TOP_TERMS_CSV = Path("data/derived/step5_pca_full/pca_full_top_terms.csv")
PRIMARY_JSONL = Path("data/derived/step4_5_normativity_gate/chunks_normative_primary.jsonl")
COUNTRY_CSV = Path("data/derived/step5_6_pca_followthrough/country_pc_means.csv")
GATE_SKEW_CSV = Path("data/derived/step5_6_pca_followthrough/gate_skew_by_pc.csv")
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


def savefig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def scree_plots(explained: list[float]) -> None:
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
    savefig(OUT_DIR / "scree.png")

    plt.figure()
    plt.plot(xs, cum, marker="o")
    plt.xlabel("Component")
    plt.ylabel("Cumulative explained variance ratio")
    plt.ylim(0, 1.0)
    plt.title("Full-corpus TF-IDF TruncatedSVD: Cumulative Explained Variance")
    savefig(OUT_DIR / "scree_cumulative.png")

    # Simple "elbow" visualization via first differences (heuristic)
    diffs = [explained[i] - explained[i + 1] for i in range(len(explained) - 1)]
    plt.figure()
    plt.plot(xs[:-1], diffs, marker="o")
    plt.xlabel("Component (difference index)")
    plt.ylabel("Explained variance drop (pc_i - pc_{i+1})")
    plt.title("Explained Variance Drop Between Successive Components")
    savefig(OUT_DIR / "scree_drop.png")


def chunk_scatter_with_gate(scores: pd.DataFrame, primary_ids: set[str], pcx: str, pcy: str, label_map: dict[str, str]) -> None:
    df = scores[["chunk_id", pcx, pcy]].copy()
    df["is_primary"] = df["chunk_id"].isin(primary_ids)

    df_primary = df[df["is_primary"]]
    df_excl = df[~df["is_primary"]]

    plt.figure()
    plt.scatter(df_excl[pcx], df_excl[pcy], s=6, alpha=0.15, label="Excluded by gate")
    plt.scatter(df_primary[pcx], df_primary[pcy], s=6, alpha=0.15, label="Normative primary")
    plt.xlabel(f"{pcx.upper()}: {label_map.get(pcx, pcx)}")
    plt.ylabel(f"{pcy.upper()}: {label_map.get(pcy, pcy)}")
    plt.title(f"Chunks in PCA Space ({pcx.upper()} vs {pcy.upper()}), Colored by Normativity Gate")
    plt.legend(loc="best")
    savefig(OUT_DIR / f"chunks_{pcx}_{pcy}_gate.png")


def gate_skew_plot(label_map: dict[str, str]) -> None:
    df = pd.read_csv(GATE_SKEW_CSV)
    df["abs_diff"] = df["mean_diff_primary_minus_excluded"].abs()
    df = df.sort_values("abs_diff", ascending=False).head(15).copy()
    df["pc_label"] = df["pc"].apply(lambda s: f"{s.upper()} | {label_map.get(s, s)}")

    plt.figure(figsize=(10, 6))
    plt.barh(df["pc_label"], df["mean_diff_primary_minus_excluded"])
    plt.xlabel("Mean score difference (primary − excluded)")
    plt.title("Normativity Gate Association by PC (Top 15 by absolute difference)")
    plt.gca().invert_yaxis()
    savefig(OUT_DIR / "gate_skew_top15.png")


def country_centroid_plot(pcx: str, pcy: str, label_map: dict[str, str]) -> None:
    df = pd.read_csv(COUNTRY_CSV)

    sizes = df["n_chunks"].apply(lambda n: 18.0 * math.sqrt(max(int(n), 1)) / 10.0)

    plt.figure()
    plt.scatter(df[pcx], df[pcy], s=sizes, alpha=0.6)

    top = df.sort_values("n_chunks", ascending=False).head(15)
    for _, r in top.iterrows():
        plt.text(r[pcx], r[pcy], str(r["country"]), fontsize=8)

    plt.xlabel(f"{pcx.upper()}: {label_map.get(pcx, pcx)}")
    plt.ylabel(f"{pcy.upper()}: {label_map.get(pcy, pcy)}")
    plt.title(f"Country Centroids ({pcx.upper()} vs {pcy.upper()})")
    savefig(OUT_DIR / f"country_{pcx}_{pcy}.png")


def load_top_terms(top_terms_path: Path, top_n: int = 12) -> Dict[int, Dict[str, List[Tuple[str, float]]]]:
    out: Dict[int, Dict[str, List[Tuple[str, float]]]] = {}
    with top_terms_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            pc = int(r["pc"])
            direction = r["direction"]
            rank = int(r["rank"])
            if rank > top_n:
                continue
            term = r["term"]
            loading = float(r["loading"])
            out.setdefault(pc, {}).setdefault(direction, []).append((term, loading))
    return out


def top_terms_bar(pc: int, terms: Dict[int, Dict[str, List[Tuple[str, float]]]], label_map: dict[str, str]) -> None:
    pos = terms.get(pc, {}).get("positive", [])
    neg = terms.get(pc, {}).get("negative", [])

    if not pos and not neg:
        return

    # Positive
    if pos:
        labels = [t for t, _ in pos][::-1]
        vals = [v for _, v in pos][::-1]
        plt.figure(figsize=(10, 5))
        plt.barh(labels, vals)
        plt.title(f"PC{pc} Top Positive Terms | {label_map.get(f'pc{pc}', f'pc{pc}')}")
        plt.xlabel("Loading")
        savefig(OUT_DIR / f"pc{pc:02d}_top_terms_positive.png")

    # Negative
    if neg:
        labels = [t for t, _ in neg][::-1]
        vals = [v for _, v in neg][::-1]
        plt.figure(figsize=(10, 5))
        plt.barh(labels, vals)
        plt.title(f"PC{pc} Top Negative Terms | {label_map.get(f'pc{pc}', f'pc{pc}')}")
        plt.xlabel("Loading")
        savefig(OUT_DIR / f"pc{pc:02d}_top_terms_negative.png")


def main() -> None:
    explained = load_explained_variance(META_JSON)
    if not explained:
        raise SystemExit(f"No explained_variance_ratio found in {META_JSON}")

    primary_ids = load_primary_chunk_ids(PRIMARY_JSONL)
    label_map = load_pc_labels(LABELS_CSV)

    scree_plots(explained)

    # Load scores once (subset columns needed for selected plots)
    scores = pd.read_csv(SCORES_CSV)

    # Chunk geometry plots
    chunk_pairs = [
        ("pc1", "pc2"),
        ("pc1", "pc3"),
        ("pc2", "pc3"),
        ("pc13", "pc19"),
        ("pc14", "pc19"),
        ("pc18", "pc19"),
        ("pc19", "pc24"),
    ]
    for pcx, pcy in chunk_pairs:
        if pcx in scores.columns and pcy in scores.columns:
            chunk_scatter_with_gate(scores, primary_ids, pcx, pcy, label_map)

    # Gate association plot
    gate_skew_plot(label_map)

    # Country centroid plots (key governance contrasts)
    country_pairs = [
        ("pc14", "pc19"),
        ("pc13", "pc19"),
        ("pc18", "pc19"),
        ("pc19", "pc24"),
        ("pc14", "pc10"),
    ]
    for pcx, pcy in country_pairs:
        country_centroid_plot(pcx, pcy, label_map)

    # Top-terms bar charts for a focused set of PCs (substantive + key artifacts)
    terms = load_top_terms(TOP_TERMS_CSV, top_n=15)
    pcs_to_plot = [1, 2, 10, 13, 14, 16, 18, 19, 24, 25]
    for pc in pcs_to_plot:
        top_terms_bar(pc, terms, label_map)

    print(f"Wrote PCA figure pack to: {OUT_DIR}")


if __name__ == "__main__":
    main()
