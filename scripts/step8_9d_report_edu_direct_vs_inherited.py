#!/usr/bin/env python3
"""
Step 8.9D — Direct vs inherited institutional strength for education.

Reads:
  - data/derived/step8_9_regime_closure/edu_closure_statements.parquet

Writes:
  - data/derived/step8_9_regime_closure/reports/edu_direct_vs_inherited_summary.csv
  - data/derived/step8_9_regime_closure/reports/edu_actor_as_attribute_rates.csv
  - data/derived/step8_9_regime_closure/reports/edu_closure_exemplars_directA.csv
  - data/derived/step8_9_regime_closure/reports/edu_closure_exemplars_inherited.csv
"""

from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--closure", default=str(PROJECT_ROOT / "data/derived/step8_9_regime_closure/edu_closure_statements.parquet"))
    ap.add_argument("--out-dir", default=str(PROJECT_ROOT / "data/derived/step8_9_regime_closure/reports"))
    ap.add_argument("--top-n", type=int, default=200)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    df = pd.read_parquet(args.closure)
    if len(df) == 0:
        raise SystemExit("Closure table is empty. Check step8_9c outputs.")

    # Direct = education actor appears in A (your “empowerment” criterion)
    df["direct_edu_A"] = df["edu_actor_A_hit"].astype(bool)
    # Inherited/domain governance = education is in scope (domain mention somewhere) but not in A
    df["inherited_edu_scope_only"] = df["edu_domain_hit"].astype(bool) & (~df["direct_edu_A"])

    # Summary at corpus level
    summary = []
    summary.append({"metric": "closure_rows_total", "value": int(len(df))})
    summary.append({"metric": "unique_docs_in_closure", "value": int(df["doc_id"].nunique())})
    summary.append({"metric": "rows_direct_edu_actor_as_A", "value": int(df["direct_edu_A"].sum())})
    summary.append({"metric": "rows_inherited_edu_scope_only", "value": int(df["inherited_edu_scope_only"].sum())})
    summary.append({"metric": "pct_directA_within_closure", "value": round(100 * df["direct_edu_A"].mean(), 2)})
    summary.append({"metric": "pct_inherited_scope_only_within_closure", "value": round(100 * df["inherited_edu_scope_only"].mean(), 2)})

    # Distribution by statement type
    by_type = (
        df.groupby(["statement_type_candidate"])
          .agg(
              rows=("doc_id","size"),
              pct_directA=("direct_edu_A","mean"),
              pct_inherited=("inherited_edu_scope_only","mean"),
          )
          .reset_index()
    )
    by_type["pct_directA"] = (by_type["pct_directA"] * 100).round(2)
    by_type["pct_inherited"] = (by_type["pct_inherited"] * 100).round(2)

    # Actor-as-Attribute rates by doc (this is the “education discussed but not empowered” lever)
    by_doc = (
        df.groupby("doc_id")
          .agg(
              closure_size=("doc_id","size"),
              directA=("direct_edu_A","sum"),
              inherited=("inherited_edu_scope_only","sum"),
              method=("closure_method", lambda x: x.iloc[0]),
          )
          .reset_index()
    )
    by_doc["directA_rate"] = (by_doc["directA"] / by_doc["closure_size"]).round(4)
    by_doc["inherited_rate"] = (by_doc["inherited"] / by_doc["closure_size"]).round(4)

    # Write outputs
    pd.DataFrame(summary).to_csv(out_dir / "edu_direct_vs_inherited_summary.csv", index=False)
    by_doc.sort_values(["directA_rate","closure_size"], ascending=[True, False]).to_csv(out_dir / "edu_actor_as_attribute_rates.csv", index=False)
    by_type.to_csv(out_dir / "edu_breakdown_by_statement_type.csv", index=False)

    # Exemplars for qualitative write-up
    # Direct exemplars: where edu actor is A and rule/norm candidate
    direct = df[df["direct_edu_A"]].copy()
    direct = direct.sort_values(["d_class","doc_id","chunk_id","sentence_index_in_chunk"])
    keep_cols = [c for c in [
        "doc_id","chunk_id","sentence_index_in_chunk",
        "closure_method","closure_anchors",
        "statement_type_candidate","d_lemma","d_class","d_polarity",
        "a_raw_text","edu_actor_A_hit_terms",
        "c_texts","b_text","i_phrase_text",
        "sentence_text"
    ] if c in df.columns]
    direct.head(args.top_n)[keep_cols].to_csv(out_dir / "edu_closure_exemplars_directA.csv", index=False, escapechar="\\")

    # Inherited exemplars: where education is domain only (not A), but rule_candidate
    inh = df[df["inherited_edu_scope_only"]].copy()
    inh = inh.sort_values(["closure_method","doc_id","chunk_id","sentence_index_in_chunk"])
    inh.head(args.top_n)[keep_cols].to_csv(out_dir / "edu_closure_exemplars_inherited.csv", index=False, escapechar="\\")

    print(f"[step8_9d] wrote reports to {out_dir}")

if __name__ == "__main__":
    main()
