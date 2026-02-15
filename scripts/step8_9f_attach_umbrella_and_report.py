#!/usr/bin/env python3
"""
Step 8.9F — Attach umbrella-O (strict) to refined education closure and report.

Inputs
- data/derived/step8_9_regime_closure/edu_closure_statements_refined.parquet
- data/derived/step8_3b_umbrella_full/statements_with_umbrella_o_strict.parquet

Outputs
- data/derived/step8_9_regime_closure/reports/edu_closure_enforcement_attachment.csv
- data/derived/step8_9_regime_closure/reports/edu_direct_vs_indirect_empowerment_with_enforcement.csv
- data/derived/step8_9_regime_closure/reports/exemplars_directA.csv
- data/derived/step8_9_regime_closure/reports/exemplars_inherited_scope.csv
- data/derived/step8_9_regime_closure/reports/exemplars_umbrella_only.csv
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--closure",
        default=str(PROJECT_ROOT / "data/derived/step8_9_regime_closure/edu_closure_statements_refined.parquet"),
    )
    ap.add_argument(
        "--umbrella",
        default=str(PROJECT_ROOT / "data/derived/step8_3b_umbrella_full/statements_with_umbrella_o_strict.parquet"),
    )
    ap.add_argument(
        "--out-dir",
        default=str(PROJECT_ROOT / "data/derived/step8_9_regime_closure/reports"),
    )
    ap.add_argument("--top-n", type=int, default=250)
    args = ap.parse_args()

    t0 = time.time()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    closure = pd.read_parquet(args.closure)
    if len(closure) == 0:
        raise SystemExit("Refined closure is empty. Check step8_9e outputs.")

    umb = pd.read_parquet(args.umbrella)

    keys = ["doc_id", "chunk_id", "sentence_index_in_chunk"]
    for k in keys:
        if k not in closure.columns:
            raise ValueError(f"Closure missing key {k}")
        if k not in umb.columns:
            raise ValueError(f"Umbrella table missing key {k}")

    umb_cols = [
        "o_umbrella_present",
        "o_umbrella_linked_strict",
        "o_umbrella_confidence",
        "o_umbrella_confidence_strict",
        "o_umbrella_link_method",
        "o_umbrella_is_doc_fallback",
        "o_umbrella_available_doc",
        "o_umbrella_block_id",
        "o_umbrella_block_confidence",
        "o_umbrella_heading",
        "o_umbrella_type",
    ]
    umb_cols = [c for c in umb_cols if c in umb.columns]
    umb_small = umb[keys + umb_cols].copy()

    merged = closure.merge(umb_small, on=keys, how="left", validate="many_to_one")

    # Ensure booleans are sane
    for b in ["o_local_present", "o_umbrella_present", "o_umbrella_linked_strict", "o_umbrella_is_doc_fallback"]:
        if b in merged.columns:
            merged[b] = merged[b].fillna(False).astype(bool)

    # --- Report 1: doc-level enforcement attachment within education closure
    def _mean(series: pd.Series) -> float:
        if series is None or len(series) == 0:
            return 0.0
        return float(series.mean())

    doc_attachment = (
        merged.groupby("doc_id")
        .agg(
            closure_size=("doc_id", "size"),
            pct_o_local_present=("o_local_present", _mean),
            pct_o_umbrella_present=("o_umbrella_present", _mean),
            pct_o_umbrella_linked_strict=("o_umbrella_linked_strict", _mean),
            pct_umbrella_doc_fallback=("o_umbrella_is_doc_fallback", _mean) if "o_umbrella_is_doc_fallback" in merged.columns else ("doc_id", lambda x: 0.0),
            method=("closure_method", lambda x: x.iloc[0] if len(x) else ""),
        )
        .reset_index()
    )

    for c in [
        "pct_o_local_present",
        "pct_o_umbrella_present",
        "pct_o_umbrella_linked_strict",
        "pct_umbrella_doc_fallback",
    ]:
        if c in doc_attachment.columns:
            doc_attachment[c] = (doc_attachment[c] * 100).round(2)

    out1 = out_dir / "edu_closure_enforcement_attachment.csv"
    doc_attachment.sort_values(["pct_o_umbrella_present", "closure_size"], ascending=[False, False]).to_csv(
        out1, index=False, escapechar="\\", quoting=csv.QUOTE_MINIMAL
    )

    # --- Report 2: direct vs indirect empowerment, with enforcement context
    merged["direct_edu_A"] = merged["edu_actor_A_hit"].fillna(False).astype(bool)
    merged["indirect_scope_only"] = merged["edu_domain_hit"].fillna(False).astype(bool) & (~merged["direct_edu_A"])

    summary = (
        merged.groupby(["statement_type_candidate"])
        .agg(
            rows=("doc_id", "size"),
            pct_directA=("direct_edu_A", _mean),
            pct_indirect=("indirect_scope_only", _mean),
            pct_o_local=("o_local_present", _mean),
            pct_o_umbrella=("o_umbrella_present", _mean),
            pct_o_umbrella_strict=("o_umbrella_linked_strict", _mean),
        )
        .reset_index()
    )
    for c in ["pct_directA", "pct_indirect", "pct_o_local", "pct_o_umbrella", "pct_o_umbrella_strict"]:
        summary[c] = (summary[c] * 100).round(2)

    out2 = out_dir / "edu_direct_vs_indirect_empowerment_with_enforcement.csv"
    summary.to_csv(out2, index=False, escapechar="\\", quoting=csv.QUOTE_MINIMAL)

    # --- Exemplars for write-up
    keep_cols = [c for c in [
        "doc_id", "chunk_id", "sentence_index_in_chunk",
        "closure_method", "closure_anchors",
        "statement_type_candidate",
        "d_class", "d_surface", "d_lemma",
        "a_raw_text", "c_texts", "b_text",
        "edu_domain_hit", "edu_actor_any_hit", "edu_actor_A_hit",
        "o_local_present", "o_umbrella_present", "o_umbrella_linked_strict",
        "o_umbrella_confidence", "o_umbrella_confidence_strict",
        "o_umbrella_link_method", "o_umbrella_is_doc_fallback",
        "sentence_text"
    ] if c in merged.columns]

    direct = merged[merged["direct_edu_A"]].copy().sort_values(["doc_id", "chunk_id", "sentence_index_in_chunk"])
    inherited = merged[merged["indirect_scope_only"]].copy().sort_values(["doc_id", "chunk_id", "sentence_index_in_chunk"])
    umbrella_only = merged[(~merged["o_local_present"]) & (merged["o_umbrella_present"])].copy().sort_values(["doc_id", "chunk_id", "sentence_index_in_chunk"])

    (direct.head(args.top_n)[keep_cols]).to_csv(out_dir / "exemplars_directA.csv", index=False, escapechar="\\", quoting=csv.QUOTE_MINIMAL)
    (inherited.head(args.top_n)[keep_cols]).to_csv(out_dir / "exemplars_inherited_scope.csv", index=False, escapechar="\\", quoting=csv.QUOTE_MINIMAL)
    (umbrella_only.head(args.top_n)[keep_cols]).to_csv(out_dir / "exemplars_umbrella_only.csv", index=False, escapechar="\\", quoting=csv.QUOTE_MINIMAL)

    elapsed = time.time() - t0
    print(f"[step8_9f] wrote {out1}")
    print(f"[step8_9f] wrote {out2}")
    print(f"[step8_9f] wrote exemplars (directA / inherited_scope / umbrella_only)")
    print(f"[step8_9f] done elapsed_s={elapsed:.1f} rows={len(merged)} docs={merged['doc_id'].nunique()}")


if __name__ == "__main__":
    main()
