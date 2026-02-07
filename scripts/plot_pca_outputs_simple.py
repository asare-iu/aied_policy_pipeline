#!/usr/bin/env python3
"""
Minimal PCA plotting: reads PCA output CSVs from an existing model out_dir and writes figures.

Expected files in out_dir:
- explained_variance.csv  (columns: pc, explained_variance_ratio, cumulative)
- full_scores_from_models.csv (should contain PC score columns like PC1, PC2, ...)
- top_terms_long.csv (optional; columns typically: pc, term, loading, rank)
"""

import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def _find_pc_columns(df: pd.DataFrame):
    pc_cols = [c for c in df.columns if c.upper().startswith("PC")]
    # Sort PC columns numerically if possible
    def pc_key(c):
        try:
            return int("".join(ch for ch in c if ch.isdigit()))
        except Exception:
            return 10**9
    return sorted(pc_cols, key=pc_key)


def plot_variance(out_dir: Path, ev: pd.DataFrame, fig_dir: Path):
    # Expect: pc, explained_variance_ratio, cumulative
    ev = ev.copy()
    if "pc" not in ev.columns:
        raise ValueError("explained_variance.csv missing 'pc' column")
    if "explained_variance_ratio" not in ev.columns:
        raise ValueError("explained_variance.csv missing 'explained_variance_ratio' column")
    if "cumulative" not in ev.columns:
        # compute if absent
        ev["cumulative"] = ev["explained_variance_ratio"].cumsum()

    # Scree plot: variance ratio by PC
    plt.figure()
    plt.plot(ev["pc"], ev["explained_variance_ratio"], marker="o")
    plt.xticks(rotation=90)
    plt.xlabel("Principal Component")
    plt.ylabel("Explained variance ratio")
    plt.title(f"Scree plot: {out_dir.name}")
    plt.tight_layout()
    plt.savefig(fig_dir / "01_scree_variance_ratio.png", dpi=200)
    plt.close()

    # Cumulative variance plot
    plt.figure()
    plt.plot(ev["pc"], ev["cumulative"], marker="o")
    plt.xticks(rotation=90)
    plt.xlabel("Principal Component")
    plt.ylabel("Cumulative explained variance")
    plt.title(f"Cumulative variance: {out_dir.name}")
    plt.tight_layout()
    plt.savefig(fig_dir / "02_cumulative_variance.png", dpi=200)
    plt.close()


def plot_scores_scatter(out_dir: Path, scores: pd.DataFrame, fig_dir: Path):
    pc_cols = _find_pc_columns(scores)
    if len(pc_cols) < 2:
        raise ValueError("full_scores_from_models.csv does not appear to contain PC columns (PC1, PC2, ...).")

    x, y = pc_cols[0], pc_cols[1]

    # Optional coloring if a country-like column exists
    color_col = None
    for candidate in ["country", "doc_country", "iso3", "region"]:
        if candidate in scores.columns:
            color_col = candidate
            break

    plt.figure()
    if color_col is None:
        plt.scatter(scores[x], scores[y], s=10, alpha=0.6)
        plt.xlabel(x); plt.ylabel(y)
        plt.title(f"{x} vs {y}: {out_dir.name}")
    else:
        # Color by category (simple)
        cats = scores[color_col].fillna("NA").astype(str)
        unique = list(pd.unique(cats))[:30]  # cap to avoid insane legends
        for u in unique:
            mask = (cats == u)
            plt.scatter(scores.loc[mask, x], scores.loc[mask, y], s=10, alpha=0.6, label=u)
        plt.xlabel(x); plt.ylabel(y)
        plt.title(f"{x} vs {y} colored by {color_col}: {out_dir.name}")
        plt.legend(loc="best", fontsize=7, frameon=False, ncol=2)

    plt.tight_layout()
    plt.savefig(fig_dir / f"03_scores_{x}_{y}.png", dpi=200)
    plt.close()


def plot_top_terms(out_dir: Path, fig_dir: Path, top_terms_long_path: Path, n_pcs: int, top_n: int):
    if not top_terms_long_path.exists():
        return

    tt = pd.read_csv(top_terms_long_path)
    # Try to standardize expected columns
    pc_col = "pc" if "pc" in tt.columns else ("PC" if "PC" in tt.columns else None)
    term_col = "term" if "term" in tt.columns else ("token" if "token" in tt.columns else None)
    loading_col = "loading" if "loading" in tt.columns else ("weight" if "weight" in tt.columns else None)
    rank_col = "rank" if "rank" in tt.columns else None

    if pc_col is None or term_col is None:
        # Can't interpret the file format reliably; skip.
        return

    # If rank missing, create within each PC by abs(loading) descending if loading exists
    if rank_col is None:
        if loading_col is not None:
            tt["_abs"] = tt[loading_col].abs()
            tt = tt.sort_values([pc_col, "_abs"], ascending=[True, False])
            tt["_rank"] = tt.groupby(pc_col).cumcount() + 1
            rank_col = "_rank"
        else:
            tt = tt.sort_values([pc_col])
            tt["_rank"] = tt.groupby(pc_col).cumcount() + 1
            rank_col = "_rank"

    # Limit to first n_pcs and top_n terms per PC
    pcs = list(pd.unique(tt[pc_col]))[:n_pcs]
    tt = tt[tt[pc_col].isin(pcs)]
    tt = tt[tt[rank_col] <= top_n].copy()

    # Save a compact CSV for quick viewing
    tt.to_csv(fig_dir / "04_top_terms_subset.csv", index=False)

    # Simple text-style plot: one page per PC (PNG)
    for pc in pcs:
        sub = tt[tt[pc_col] == pc]
        terms = sub[term_col].astype(str).tolist()
        if loading_col and loading_col in sub.columns:
            vals = sub[loading_col].tolist()
        else:
            vals = list(range(len(terms), 0, -1))

        plt.figure(figsize=(8, max(3, 0.25 * len(terms))))
        plt.barh(list(reversed(terms)), list(reversed(vals)))
        plt.xlabel(loading_col if loading_col else "rank proxy")
        plt.title(f"Top terms for {pc}: {out_dir.name}")
        plt.tight_layout()
        safe_pc = str(pc).replace("/", "_")
        plt.savefig(fig_dir / f"04_top_terms_{safe_pc}.png", dpi=200)
        plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", help="Path to PCA model output directory (contains explained_variance.csv etc.)")
    ap.add_argument("--n_pcs_terms", type=int, default=10, help="How many PCs to export top-term plots for")
    ap.add_argument("--top_n_terms", type=int, default=20, help="How many top terms per PC")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    if not out_dir.exists():
        raise SystemExit(f"ERROR: out_dir not found: {out_dir}")

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    ev_path = out_dir / "explained_variance.csv"
    scores_path = out_dir / "full_scores_from_models.csv"
    tt_long_path = out_dir / "top_terms_long.csv"

    if not ev_path.exists():
        raise SystemExit(f"ERROR: missing {ev_path}")
    if not scores_path.exists():
        raise SystemExit(f"ERROR: missing {scores_path}")

    ev = pd.read_csv(ev_path)
    scores = pd.read_csv(scores_path)

    plot_variance(out_dir, ev, fig_dir)
    plot_scores_scatter(out_dir, scores, fig_dir)
    plot_top_terms(out_dir, fig_dir, tt_long_path, args.n_pcs_terms, args.top_n_terms)

    print(f"[OK] Wrote figures to: {fig_dir}")


if __name__ == "__main__":
    main()
