#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd


def safe_to_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(
        path,
        index=False,
        encoding="utf-8",
        quoting=csv.QUOTE_MINIMAL,
        escapechar="\\",
    )


BASE = Path("data/derived/step8_analysis/iad_rule_types_rules_only")

FULL_PATH = BASE / "full_corpus_iad_rule_types_rules_only.parquet"
EDU_PATH = BASE / "education_relevant_iad_rule_types_rules_only.parquet"
TITLE_PATH = BASE / "education_in_title_iad_rule_types_rules_only.parquet"

OUT_PATH = Path("data/derived/step8_analysis/rule_manual_coding_sheet.csv")


def main() -> None:
    full = pd.read_parquet(FULL_PATH)
    edu = pd.read_parquet(EDU_PATH)
    title = pd.read_parquet(TITLE_PATH)

    edu_keys = set(zip(edu["doc_id"].astype(str), edu["chunk_id"].astype(str), edu["sentence_text"].astype(str)))
    title_keys = set(zip(title["doc_id"].astype(str), title["chunk_id"].astype(str), title["sentence_text"].astype(str)))

    full = full.copy()
    full["_key"] = list(zip(full["doc_id"].astype(str), full["chunk_id"].astype(str), full["sentence_text"].astype(str)))

    full["is_education_relevant"] = full["_key"].isin(edu_keys).astype(int)
    full["is_education_in_title"] = full["_key"].isin(title_keys).astype(int)

    keep_cols = [
        "doc_id",
        "chunk_id",
        "sentence_text",
        "a_raw_text",
        "a_class",
        "d_surface",
        "d_class",
        "i_phrase_text",
        "c_texts",
        "b_text",
        "o_local_present",
        "o_local_text",
        "statement_type_candidate",
        "iad_rule_type_primary",
        "iad_rule_type_hits",
        "iad_rule_type_n_hits",
        "iad_boundary",
        "iad_position",
        "iad_choice",
        "iad_information",
        "iad_scope",
        "iad_aggregation",
        "iad_payoff",
        "is_education_relevant",
        "is_education_in_title",
    ]

    out = full[keep_cols].copy()

    # blank columns for manual coding
    out["manual_primary_rule_type"] = ""
    out["manual_secondary_rule_types"] = ""
    out["manual_is_unclear"] = ""
    out["manual_notes"] = ""
    out["manual_confidence"] = ""
    out["manual_reviewer"] = ""

    safe_to_csv(out, OUT_PATH)

    print(f"Saved → {OUT_PATH}")
    print(f"Rows  → {len(out):,}")
    print()
    print(out.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
