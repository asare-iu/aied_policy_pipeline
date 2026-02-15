#!/usr/bin/env python3
from __future__ import annotations

import sys
import argparse
from pathlib import Path
import pandas as pd

# ensure scripts/ is on PYTHONPATH
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from pca_plotting_utils import (
    load_models_outputs,
    load_terms_long,
    load_pc_interpretations,
    plot_explained_variance,
    plot_scores_scatter,
    plot_loadings_bar,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Create PCA visuals for interpretable PCs only.")
    ap.add_argument("--models-dir", required=True, help="Directory with explained_variance.csv, full_scores_from_models.csv, top_terms_long.csv")
    ap.add_argument("--out-dir", required=True, help="Where to write figures")
    ap.add_argument("--top-k", type=int, default=20, help="Top terms per side for loadings bar charts")
    args = ap.parse_args()

    model_dir = Path(args.models_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load PCA outputs
    df_var, df_scores, *_ = load_models_outputs(model_dir)
    df_terms_long = load_terms_long(model_dir / "top_terms_long.csv")


    # Load interpretations + select interpretable PCs
    df_interp = load_pc_interpretations(model_dir)
    interpretable_pcs = (
        df_interp[df_interp["interpretable"].astype(str).str.upper().eq("Y")]["pc"]
        .astype(str)
        .tolist()
    )

    # Always write variance plot (global diagnostic)
    plot_explained_variance(df_var, out_dir / "explained_variance.png")

    if not interpretable_pcs:
        print(f"[plot_pca_figures] No interpretable PCs marked 'Y' in {model_dir / 'pc_interpretations.csv'}")
        print(f"[plot_pca_figures] Wrote only explained_variance.png to {out_dir}")
        return

    # Score scatters: consecutive pairs among interpretable PCs (PCa vs PCb)
    for a, b in zip(interpretable_pcs, interpretable_pcs[1:]):
        plot_scores_scatter(df_scores, out_dir / f"{a.lower()}_{b.lower()}_scores.png", a, b)

    # Loadings bars: one per interpretable PC
    for pc in interpretable_pcs:
        plot_loadings_bar(df_terms_long, out_dir / f"{pc.lower()}_loadings.png", pc=pc, top_k=args.top_k)

    # Write a tiny manifest for traceability
    (out_dir / "manifest.txt").write_text(
        "models_dir: " + str(model_dir) + "\n" +
        "interpretable_pcs: " + ", ".join(interpretable_pcs) + "\n" +
        f"top_k: {args.top_k}\n"
    )

    print(f"[plot_pca_figures] Interpretable PCs: {interpretable_pcs}")
    print(f"[plot_pca_figures] Wrote figures to: {out_dir}")

if __name__ == "__main__":
    main()
