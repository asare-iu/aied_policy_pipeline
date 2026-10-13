#!/usr/bin/env python3
"""
Create a reproducible manual audit sample for Education Gate v3.

The audit sample is drawn from the complete gate-decision population:
- 200 randomly selected V3 positives
- 200 randomly selected V3 negatives

The sample is intended to detect residual over- and under-inclusion after the
source-backed tiered vocabulary is frozen. It does not alter gate membership.
"""

import argparse
import csv
import json
import random
import re
from pathlib import Path

import pandas as pd


WS_RE = re.compile(r"\s+")
SEED = 20260823


def extract_text(record):
    for key in (
        "text_normalized",
        "text_norm",
        "text",
        "chunk_text",
        "chunk_text_normalized",
    ):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return WS_RE.sub(" ", value).strip()
    return ""


def extract_chunk_id(record, line_number):
    for key in ("chunk_id", "id"):
        value = record.get(key)
        if value is not None:
            return str(value)
    return f"line_{line_number}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        default=(
            "data/derived/step4_chunks_tagged/"
            "chunks_normalized_tagged.jsonl"
        ),
    )
    parser.add_argument(
        "--decisions",
        default=(
            "data/derived/step6_education_gate_v3/"
            "education_gate_v3_decisions.csv"
        ),
    )
    parser.add_argument(
        "--out",
        default=(
            "evidence/edu_gate_v3/"
            "education_gate_v3_manual_audit.csv"
        ),
    )
    parser.add_argument("--n-positive", type=int, default=200)
    parser.add_argument("--n-negative", type=int, default=200)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    source_path = Path(args.source)
    decisions_path = Path(args.decisions)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    decisions = pd.read_csv(
        decisions_path,
        dtype={"chunk_id": str, "doc_id": str},
        escapechar="\\",
    )

    decisions["edu_match_bool"] = (
        decisions["edu_match"].astype(str).str.lower().eq("true")
    )

    positives = decisions[decisions["edu_match_bool"]].copy()
    negatives = decisions[~decisions["edu_match_bool"]].copy()

    pos_n = min(args.n_positive, len(positives))
    neg_n = min(args.n_negative, len(negatives))

    pos_sample = positives.sample(n=pos_n, random_state=args.seed)
    neg_sample = negatives.sample(n=neg_n, random_state=args.seed + 1)

    sample = pd.concat([pos_sample, neg_sample], ignore_index=True)
    sample = sample.sample(frac=1, random_state=args.seed + 2).reset_index(drop=True)

    wanted_ids = set(sample["chunk_id"].astype(str))
    text_lookup = {}

    with source_path.open("r", encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            chunk_id = extract_chunk_id(record, line_number)
            if chunk_id in wanted_ids:
                text_lookup[chunk_id] = extract_text(record)

    sample["text"] = sample["chunk_id"].map(text_lookup).fillna("")

    # Manual coding fields. Gate result is retained so the audit can explicitly
    # identify false positives and false negatives.
    sample.insert(0, "review_id", [f"V3A{i:04d}" for i in range(1, len(sample) + 1)])
    sample["manual_education_relevant"] = ""
    sample["manual_confidence"] = ""
    sample["manual_reason"] = ""
    sample["manual_notes"] = ""

    keep_cols = [
        "review_id",
        "chunk_id",
        "doc_id",
        "edu_match",
        "edu_match_rule",
        "anchor_groups",
        "anchor_canonicals",
        "anchor_terms",
        "contextual_groups",
        "contextual_canonicals",
        "contextual_terms",
        "tag_only_groups",
        "tag_only_canonicals",
        "tag_only_terms",
        "k12_explicit_signal",
        "higher_ed_explicit_signal",
        "formal_vet_signal",
        "pedagogy_signal",
        "teacher_development_signal",
        "workforce_capacity_signal",
        "higher_ed_research_signal",
        "text",
        "manual_education_relevant",
        "manual_confidence",
        "manual_reason",
        "manual_notes",
    ]

    sample[keep_cols].to_csv(
        out_path,
        index=False,
        quoting=csv.QUOTE_ALL,
        escapechar="\\",
        doublequote=True,
    )

    print("EDUCATION GATE V3 AUDIT SAMPLE CREATED")
    print("--------------------------------------")
    print("Positive audit rows:", pos_n)
    print("Negative audit rows:", neg_n)
    print("Total audit rows:", len(sample))
    print("Seed:", args.seed)
    print("Wrote:", out_path)


if __name__ == "__main__":
    main()
