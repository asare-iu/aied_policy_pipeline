#!/usr/bin/env python3
"""
Step 8.3b postprocess — STRICT linkage + doc-level availability summary

Inputs
- statements_with_umbrella_o.parquet  (output of step8_3b_umbrella_o_extract_and_link.py)
- umbrella_o_blocks.parquet           (output of step8_3b_umbrella_o_extract_and_link.py)

Outputs (written to --out-dir)
- statements_with_umbrella_o_strict.parquet
- umbrella_o_doc_summary.csv
- umbrella_o_doc_types_long.csv
- umbrella_o_confidence_counts.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--statements",
        default="data/derived/step8_3b_umbrella_full/statements_with_umbrella_o.parquet",
        help="Path to statements_with_umbrella_o.parquet",
    )
    ap.add_argument(
        "--blocks",
        default="data/derived/step8_3b_umbrella_full/umbrella_o_blocks.parquet",
        help="Path to umbrella_o_blocks.parquet",
    )
    ap.add_argument(
        "--out-dir",
        default="data/derived/step8_3b_umbrella_full",
        help="Directory to write outputs",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stmts = pd.read_parquet(args.statements)
    blocks = pd.read_parquet(args.blocks)

    # ----------------------------
    # 1) STRICT linkage columns
    # ----------------------------
    # strict = only high/medium (credible inheritance)
    # (your current run has no medium counts, but keep logic future-proof)
    strict_conf = stmts["o_umbrella_confidence"].fillna("none").astype(str)
    stmts["o_umbrella_linked_strict"] = strict_conf.isin(["high", "medium"])
    stmts["o_umbrella_confidence_strict"] = strict_conf.where(
        strict_conf.isin(["high", "medium"]), other="none"
    )

    # availability = doc has any umbrella O block detected
    docs_with_blocks = set(blocks["doc_id"].astype(str).unique())
    stmts["o_umbrella_available_doc"] = stmts["doc_id"].astype(str).isin(docs_with_blocks)

    # keep doc-fallback flag explicit (interpretability)
    stmts["o_umbrella_is_doc_fallback"] = strict_conf.eq("low")

    strict_out = out_dir / "statements_with_umbrella_o_strict.parquet"
    stmts.to_parquet(strict_out, index=False)

    # ----------------------------
    # 2) Doc-level umbrella summary
    # ----------------------------
    # basic counts
    blk = blocks.copy()
    blk["doc_id"] = blk["doc_id"].astype(str)

    blocks_by_doc = (
        blk.groupby("doc_id")
           .agg(
               umbrella_blocks=("block_id", "count"),
               umbrella_heading_blocks=("o_umbrella_method", lambda s: (s == "heading_block").sum()),
               umbrella_cue_paragraphs=("o_umbrella_method", lambda s: (s == "cue_paragraph").sum()),
           )
           .reset_index()
    )

    # type distribution long + wide
    types_long = (
        blk.groupby(["doc_id", "o_umbrella_type"])
           .size()
           .reset_index(name="type_count")
    )
    types_long_out = out_dir / "umbrella_o_doc_types_long.csv"
    types_long.to_csv(types_long_out, index=False, escapechar="\\")

    types_wide = (
        types_long.pivot_table(index="doc_id", columns="o_umbrella_type", values="type_count", fill_value=0)
                 .reset_index()
    )

    # statement linkage summary per doc
    st = stmts.copy()
    st["doc_id"] = st["doc_id"].astype(str)

    link_by_doc = (
        st.groupby("doc_id")
          .agg(
              statements=("sentence_text", "count"),
              strict_linked=("o_umbrella_linked_strict", "sum"),
              doc_fallback=("o_umbrella_is_doc_fallback", "sum"),
              high=("o_umbrella_confidence", lambda s: (s == "high").sum()),
              medium=("o_umbrella_confidence", lambda s: (s == "medium").sum()),
              low=("o_umbrella_confidence", lambda s: (s == "low").sum()),
              none=("o_umbrella_confidence", lambda s: (s == "none").sum()),
          )
          .reset_index()
    )

    doc_summary = (
        link_by_doc
        .merge(blocks_by_doc, on="doc_id", how="left")
        .merge(types_wide, on="doc_id", how="left")
    )

    # percentages for readability
    doc_summary["pct_strict_linked"] = (100 * doc_summary["strict_linked"] / doc_summary["statements"]).round(2)
    doc_summary["pct_doc_fallback"] = (100 * doc_summary["doc_fallback"] / doc_summary["statements"]).round(2)
    doc_summary["pct_high"] = (100 * doc_summary["high"] / doc_summary["statements"]).round(2)
    doc_summary["pct_low"] = (100 * doc_summary["low"] / doc_summary["statements"]).round(2)

    doc_summary_out = out_dir / "umbrella_o_doc_summary.csv"
    doc_summary.to_csv(doc_summary_out, index=False, escapechar="\\")

    # ----------------------------
    # 3) Corpus-level confidence counts
    # ----------------------------
    conf_counts = (
        stmts["o_umbrella_confidence"].fillna("none").value_counts()
             .rename_axis("o_umbrella_confidence")
             .reset_index(name="count")
    )
    conf_out = out_dir / "umbrella_o_confidence_counts.csv"
    conf_counts.to_csv(conf_out, index=False, escapechar="\\")

    # quick audit prints
    total = len(stmts)
    pct_strict = round(100 * float(stmts["o_umbrella_linked_strict"].mean()), 2)
    pct_avail = round(100 * float(stmts["o_umbrella_available_doc"].mean()), 2)

    print("[ok] wrote:", strict_out)
    print("[ok] wrote:", doc_summary_out)
    print("[ok] wrote:", types_long_out)
    print("[ok] wrote:", conf_out)

    print("\n[audit] strict linked % (high/medium only):", pct_strict)
    print("[audit] doc has any umbrella block %:", pct_avail)
    print("[audit] confidence counts:", conf_counts.set_index("o_umbrella_confidence")["count"].to_dict())


if __name__ == "__main__":
    main()
