#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Set

import pandas as pd


# -----------------------------
# Helpers
# -----------------------------

SANCTION_HEADING_KWS = [
    "sanction", "sanctions",
    "penalty", "penalties",
    "fine", "fines", "administrative fine", "administrative fines",
    "offence", "offences", "offense", "offenses",
    "enforcement",
    "liability", "liable",
    "revocation", "suspension", "withdrawal",
    "non-compliance", "noncompliance",
    "breach",
]

def heading_is_sanctionish(h: str) -> bool:
    h = (h or "").strip().lower()
    if not h:
        return False
    return any(kw in h for kw in SANCTION_HEADING_KWS)

def iter_doc_ids_from_chunks_jsonl(path: Path) -> Set[str]:
    doc_ids: Set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            doc_id = obj.get("doc_id")
            if doc_id is not None:
                doc_ids.add(str(doc_id))
    return doc_ids

def safe_bool_series(x: pd.Series) -> pd.Series:
    # normalizes truthy/falsey values in case the parquet has objects
    if x.dtype == bool:
        return x
    return x.fillna(False).astype(bool)


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full-igt", required=True, help="Full corpus 8.3 statements parquet")
    ap.add_argument("--umbrella-linked", required=True, help="8.3b linked statements parquet (full corpus)")
    ap.add_argument("--umbrella-blocks", required=True, help="8.3b umbrella blocks parquet (full corpus)")
    ap.add_argument("--edu-chunks-jsonl", required=True, help="step6 education chunks JSONL (doc filter)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--sample-n", type=int, default=200)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[load] edu doc_ids…")
    edu_doc_ids = iter_doc_ids_from_chunks_jsonl(Path(args.edu_chunks_jsonl))
    print(f"[ok] edu doc_ids: {len(edu_doc_ids):,}")

    print("[load] full IGT…")
    igt = pd.read_parquet(args.full_igt)

    print("[load] umbrella-linked statements…")
    linked = pd.read_parquet(args.umbrella_linked)

    print("[load] umbrella blocks…")
    blocks = pd.read_parquet(args.umbrella_blocks)

    # --- Base set: rule candidates from full IGT
    rules = igt[igt["statement_type_candidate"] == "rule_candidate"].copy()
    rules["doc_id"] = rules["doc_id"].astype(str)
    rules["chunk_id"] = rules["chunk_id"].astype(str)

    # --- Doc filter: docs that reference education (via edu chunks set)
    rules["doc_is_edu"] = rules["doc_id"].isin(edu_doc_ids)

    # --- Bring in umbrella link fields by joining on statement identity
    # Identity key: (doc_id, chunk_id, sentence_index_in_chunk, sentence_text)
    key_cols = ["doc_id", "chunk_id", "sentence_index_in_chunk", "sentence_text"]
    for c in key_cols:
        linked[c] = linked[c].astype(str) if c in ["doc_id", "chunk_id"] else linked[c]
    linked["doc_id"] = linked["doc_id"].astype(str)
    linked["chunk_id"] = linked["chunk_id"].astype(str)

    link_cols = [
        "doc_id","chunk_id","sentence_index_in_chunk","sentence_text",
        "o_umbrella_present","o_umbrella_confidence","o_umbrella_block_id",
        "o_umbrella_heading","o_umbrella_type","o_umbrella_method","o_umbrella_link_method",
    ]
    link_cols = [c for c in link_cols if c in linked.columns]
    link_small = linked[link_cols].copy()

    rules = rules.merge(link_small, on=key_cols, how="left", suffixes=("","_y"))

    # Normalize umbrella present
    if "o_umbrella_present" in rules.columns:
        rules["o_umbrella_present"] = safe_bool_series(rules["o_umbrella_present"])
    else:
        rules["o_umbrella_present"] = rules["o_umbrella_block_id"].notna()

    # --- Determine whether the linked umbrella block is in a “sanction section”
    # Prefer heading classification; if heading missing, fall back to type/method heuristics.
    blocks = blocks.copy()
    blocks["block_id"] = blocks["block_id"].astype(str)
    blocks["heading_text"] = blocks.get("heading_text", "").fillna("").astype(str)
    blocks["block_is_sanction_section"] = blocks["heading_text"].apply(heading_is_sanctionish)

    block_map = blocks[["block_id","block_is_sanction_section","heading_text","o_umbrella_type","o_umbrella_method"]].rename(
        columns={
            "block_id": "o_umbrella_block_id",
            "heading_text": "umbrella_heading_text",
            "o_umbrella_type": "umbrella_block_type",
            "o_umbrella_method": "umbrella_block_method",
        }
    )

    rules = rules.merge(block_map, on="o_umbrella_block_id", how="left")

    # For cases where heading_text is empty (very common in cue_paragraph),
    # allow type-based “sanction section” proxy (still deterministic):
    # If the block type is one of the enforcement types, treat it as sanctionish.
    sanction_types = {"fine","liability","offence","revocation_or_suspension","sanction_or_penalty"}
    rules["block_is_sanction_section"] = rules["block_is_sanction_section"].fillna(False)
    rules["block_is_sanction_section_proxy"] = rules["umbrella_block_type"].isin(sanction_types)
    # final flag: heading-based OR proxy
    rules["umbrella_is_sanctionish"] = rules["block_is_sanction_section"] | rules["block_is_sanction_section_proxy"]

    # --- Build the 3-way intersection
    rules["in_intersection"] = (
        rules["doc_is_edu"]
        & rules["o_umbrella_present"]
        & rules["umbrella_is_sanctionish"]
    )

    # --- Outputs
    summary = pd.DataFrame([{
        "total_rules_full": int(len(rules)),
        "rules_in_edu_docs": int((rules["doc_is_edu"]).sum()),
        "rules_with_any_umbrella_link": int((rules["o_umbrella_present"]).sum()),
        "rules_with_sanctionish_umbrella": int((rules["o_umbrella_present"] & rules["umbrella_is_sanctionish"]).sum()),
        "RULES_IN_3WAY_INTERSECTION": int(rules["in_intersection"].sum()),
        "pct_rules_in_3way_intersection_of_all_rules": round(100 * rules["in_intersection"].mean(), 3),
        "unique_docs_in_intersection": int(rules.loc[rules["in_intersection"], "doc_id"].nunique()),
    }])
    summary_path = out_dir / "rules_x_umbrellaSections_x_eduDocs_summary.csv"
    summary.to_csv(summary_path, index=False, escapechar="\\")
    print(f"[ok] wrote: {summary_path}")
    print(summary.to_string(index=False))

    # Full table (for analysis)
    keep = [
        "doc_id","chunk_id","sentence_index_in_chunk","sentence_text",
        "a_raw_text","a_class",
        "d_class","d_surface","d_polarity",
        "i_phrase_text","i_head_lemma",
        "b_text","b_found",
        "c_texts","c_count",
        "o_local_present","o_local_text","o_local_type",
        "doc_is_edu","o_umbrella_present","o_umbrella_confidence",
        "o_umbrella_block_id","umbrella_heading_text","umbrella_block_type","umbrella_block_method",
        "umbrella_is_sanctionish","in_intersection",
    ]
    keep = [c for c in keep if c in rules.columns]
    out_all = out_dir / "rules_with_umbrella_and_edu_flags.csv"
    rules[keep].to_csv(out_all, index=False, escapechar="\\")
    print(f"[ok] wrote: {out_all} (rows={len(rules):,})")

    # Intersection-only + sample for reading
    inter = rules[rules["in_intersection"]].copy()
    out_inter = out_dir / "RULES_IN_3WAY_INTERSECTION.csv"
    inter[keep].to_csv(out_inter, index=False, escapechar="\\")
    print(f"[ok] wrote: {out_inter} (rows={len(inter):,})")

    if len(inter) > 0:
        sample = inter.sample(n=min(args.sample_n, len(inter)), random_state=7)
        out_sample = out_dir / "RULES_IN_3WAY_INTERSECTION_SAMPLE.csv"
        sample[keep].to_csv(out_sample, index=False, escapechar="\\")
        print(f"[ok] wrote: {out_sample} (rows={len(sample):,})")


if __name__ == "__main__":
    main()
