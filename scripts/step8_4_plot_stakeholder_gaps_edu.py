#!/usr/bin/env python3
"""
Step 8.4 — Plot stakeholder presence gaps (chunks vs institutional statements)

Inputs:
  - data/derived/step8_analysis/stakeholder_chunks_vs_statements_edu.csv
    columns expected:
      stakeholder, pct_chunks, pct_statements, gap_chunks_minus_statements

Outputs:
  - data/derived/step8_analysis/stakeholder_chunks_vs_statements_edu_bars.png
  - data/derived/step8_analysis/stakeholder_gap_chunks_minus_statements_edu.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in-csv",
        default="data/derived/step8_analysis/stakeholder_chunks_vs_statements_edu.csv",
        help="CSV produced by step8_4_stakeholder_chunks_vs_statements_edu.py",
    )
    ap.add_argument(
        "--out-dir",
        default="data/derived/step8_analysis",
        help="Directory to write figures",
    )
    ap.add_argument("--top-n", type=int, default=20, help="Plot top N stakeholders by pct_chunks")
    args = ap.parse_args()

    in_path = Path(args.in_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)

    needed = {"stakeholder", "pct_chunks", "pct_statements"}
    missing = sorted(list(needed - set(df.columns)))
    if missing:
        raise ValueError(f"Missing expected columns: {missing}. Found columns={list(df.columns)}")

    # Ensure numeric
    df["pct_chunks"] = pd.to_numeric(df["pct_chunks"], errors="coerce")
    df["pct_statements"] = pd.to_numeric(df["pct_statements"], errors="coerce")
    if "gap_chunks_minus_statements" not in df.columns:
        df["gap_chunks_minus_statements"] = df["pct_chunks"] - df["pct_statements"]
    else:
        df["gap_chunks_minus_statements"] = pd.to_numeric(df["gap_chunks_minus_statements"], errors="coerce")

    # Sort by pct_chunks so the plot reads as: most-discussed -> less-discussed
    plot_df = df.sort_values("pct_chunks", ascending=True).tail(args.top_n)

    # ---- Figure 1: Side-by-side bars for pct_chunks vs pct_statements (horizontal) ----
    fig1 = plt.figure(figsize=(10, 6))
    y = range(len(plot_df))

    # Two bars per stakeholder using small offsets
    # (no explicit colors requested; matplotlib defaults are fine)
    plt.barh([i - 0.2 for i in y], plot_df["pct_chunks"], height=0.35, label="% chunks")
    plt.barh([i + 0.2 for i in y], plot_df["pct_statements"], height=0.35, label="% institutional statements")

    plt.yticks(list(y), plot_df["stakeholder"])
    plt.xlabel("Percent")
    plt.title("Stakeholder Presence: Education chunks vs institutional statements")
    plt.legend(loc="lower right")
    plt.tight_layout()

    out1 = out_dir / "stakeholder_chunks_vs_statements_edu_bars.png"
    plt.savefig(out1, dpi=300)
    plt.close(fig1)
    print(f"Saved → {out1}")

    # ---- Figure 2: Gap-only bar plot ----
    plot_df2 = df.sort_values("gap_chunks_minus_statements", ascending=True).tail(args.top_n)

    fig2 = plt.figure(figsize=(10, 6))
    y2 = range(len(plot_df2))
    plt.barh(list(y2), plot_df2["gap_chunks_minus_statements"])
    plt.yticks(list(y2), plot_df2["stakeholder"])
    plt.xlabel("Gap (pct_chunks − pct_statements)")
    plt.title("Governance Drop-off: Stakeholder mentions in discourse vs explicit institutional targeting")
    plt.tight_layout()

    out2 = out_dir / "stakeholder_gap_chunks_minus_statements_edu.png"
    plt.savefig(out2, dpi=300)
    plt.close(fig2)
    print(f"Saved → {out2}")


if __name__ == "__main__":
    main()
