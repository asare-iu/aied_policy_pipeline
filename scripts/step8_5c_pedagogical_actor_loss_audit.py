#!/usr/bin/env python3
"""
Step 8.5c — Pedagogical actor loss audit for education-relevant IGT statements.

Purpose
-------
Diagnose *why* pedagogical actors (especially educators/teachers) disappear or
shrink sharply in the final education-relevant rules-only stakeholder analysis.

This script does not replace the stakeholder × rule-type analysis. It audits the
filters feeding that analysis and attributes loss across three main mechanisms:

1. The actor is mentioned in the sentence but not captured as the A component.
2. The actor is present in A but not marked as explicit.
3. The actor is present in A but the statement is not retained as a rule.

Research use
------------
This is a sensitivity / audit step intended to support claims like:
- pedagogical actors are discussed in education-relevant policy, but
- they weaken or disappear when the text is restricted to explicit-A rules.

Inputs
------
Primary input:
- data/derived/step8_igt_chunks_edu/igt_statements_full.parquet

Outputs
-------
Default directory:
- data/derived/step8_analysis/pedagogical_actor_loss_audit/

Files:
- run_metadata.json
- audit_stage_counts.csv
- pedagogical_statement_type_profile_sentence_mentions.csv
- pedagogical_statement_type_profile_a_mentions.csv
- pedagogical_loss_reasons_from_sentence_mentions.csv
- pedagogical_loss_reasons_from_a_mentions.csv
- pedagogical_loss_summary.md
- pedagogical_exemplars.csv
- pedagogical_examples_sentence_only_or_parser_miss.csv
- pedagogical_examples_nonexplicit_rules.csv
- pedagogical_examples_rules_cutoff.csv
- pedagogical_examples_final_retained.csv

Method defaults
---------------
- education-relevant IGT only
- pedagogical groups limited to: educators_teachers, students_learners,
  schools_institutions
- rule labels default to: rule_candidate, rule, rules
- outputs report both broad sentence-level mention and narrower A-level capture
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)



def maybe_read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv" or path.name.endswith(".csv.gz"):
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type: {path}")



def safe_to_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(
        path,
        index=False,
        encoding="utf-8",
        quoting=csv.QUOTE_MINIMAL,
        escapechar="\\",
    )



def textify(x) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x).strip()



def normalize_space(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip()



def normalize_a_text(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.lower()
        .str.replace(r"\bthe\b", "", regex=True)
        .str.replace(r"\ban\b", "", regex=True)
        .str.replace(r"\ba\b", "", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )



def build_markdown_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, row in df.iterrows():
        vals = [str(row[c]) for c in cols]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Schema / filters
# -----------------------------------------------------------------------------

EXPECTED_COLUMNS = {
    "doc_id",
    "chunk_id",
    "sentence_text",
    "a_raw_text",
    "a_class",
    "statement_type_candidate",
}

DEFAULT_RULE_LABELS = {"rule_candidate", "rule", "rules"}



def validate_minimum_schema(df: pd.DataFrame) -> None:
    missing = sorted(list(EXPECTED_COLUMNS - set(df.columns)))
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")



def normalize_stmt_type(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().str.lower()


# -----------------------------------------------------------------------------
# Pedagogical groups
# -----------------------------------------------------------------------------

PED_GROUP_PATTERNS: Dict[str, List[str]] = {
    "educators_teachers": [
        r"\beducator(s)?\b",
        r"\bteacher(s)?\b",
        r"\binstructor(s)?\b",
        r"\blecturer(s)?\b",
        r"\btrainer(s)?\b",
        r"\bfaculty\b",
        r"\bprofessor(s)?\b",
        r"\bteaching staff\b",
        r"\bpedagog(?:y|ical)\b",
    ],
    "students_learners": [
        r"\bstudent(s)?\b",
        r"\blearner(s)?\b",
        r"\bpupil(s)?\b",
        r"\bchild(ren)?\b",
        r"\bminor(s)?\b",
    ],
    "schools_institutions": [
        r"\bschool(s)?\b",
        r"\buniversity\b",
        r"\buniversities\b",
        r"\bcollege(s)?\b",
        r"\binstitution(s)?\b",
        r"\beducational institution(s)?\b",
        r"\bacademic institution(s)?\b",
        r"\btraining provider(s)?\b",
        r"\beducation provider(s)?\b",
        r"\bclassroom(s)?\b",
        r"\bcampus\b",
        r"\bcampuses\b",
    ],
}

PED_GROUP_ORDER = [
    "educators_teachers",
    "students_learners",
    "schools_institutions",
]



def match_groups(text: str, patterns: Dict[str, List[str]], ordered_groups: Sequence[str]) -> List[str]:
    s = textify(text).lower()
    if not s:
        return []
    hits: List[str] = []
    for grp in ordered_groups:
        grp_pats = patterns.get(grp, [])
        if any(re.search(pat, s, flags=re.IGNORECASE) for pat in grp_pats):
            hits.append(grp)
    return hits



def first_or_empty(groups: List[str]) -> str:
    return groups[0] if groups else ""

# -----------------------------------------------------------------------------
# Summaries
# -----------------------------------------------------------------------------


def build_stage_counts(df: pd.DataFrame, rule_labels: set[str]) -> pd.DataFrame:
    rows: List[dict] = []

    def add_stage(stage_name: str, mask: pd.Series) -> None:
        stage_df = df[mask].copy()
        for grp in PED_GROUP_ORDER:
            sent_n = int(stage_df[f"sent_has_{grp}"].sum())
            a_n = int(stage_df[f"a_has_{grp}"].sum())
            rows.append(
                {
                    "stage": stage_name,
                    "pedagogical_group": grp,
                    "n_rows_total_stage": int(len(stage_df)),
                    "n_sentence_mentions": sent_n,
                    "n_a_mentions": a_n,
                    "pct_stage_rows_sentence_mentions": round(100.0 * sent_n / max(len(stage_df), 1), 2),
                    "pct_stage_rows_a_mentions": round(100.0 * a_n / max(len(stage_df), 1), 2),
                }
            )

    stmt = df["_stmt_type_norm"]
    explicit = df["_a_is_explicit"]
    nonempty_a = df["_a_nonempty"]
    rule = stmt.isin(rule_labels)

    add_stage("all_edu_igt", pd.Series(True, index=df.index))
    add_stage("rule_only", rule)
    add_stage("explicit_a_only", explicit)
    add_stage("rule_only__explicit_a_only", rule & explicit)
    add_stage("nonempty_a_only", nonempty_a)
    add_stage("rule_only__nonempty_a_only", rule & nonempty_a)

    return pd.DataFrame(rows)



def build_statement_type_profiles(df: pd.DataFrame, base: str) -> pd.DataFrame:
    if base not in {"sentence", "a"}:
        raise ValueError("base must be 'sentence' or 'a'")

    rows: List[dict] = []
    for grp in PED_GROUP_ORDER:
        flag = f"{'sent' if base == 'sentence' else 'a'}_has_{grp}"
        sub = df[df[flag]].copy()
        if sub.empty:
            continue
        counts = (
            sub.groupby("_stmt_type_norm", dropna=False)
            .size()
            .reset_index(name="count")
            .sort_values(["count", "_stmt_type_norm"], ascending=[False, True])
            .reset_index(drop=True)
        )
        counts.insert(0, "pedagogical_group", grp)
        counts["percent_within_group"] = (100.0 * counts["count"] / len(sub)).round(2)
        counts["base"] = base
        rows.append(counts)

    if not rows:
        return pd.DataFrame(columns=["pedagogical_group", "_stmt_type_norm", "count", "percent_within_group", "base"])
    return pd.concat(rows, ignore_index=True)



def build_loss_reasons_from_sentence(df: pd.DataFrame, rule_labels: set[str]) -> pd.DataFrame:
    rows: List[dict] = []
    for grp in PED_GROUP_ORDER:
        sub = df[df[f"sent_has_{grp}"]].copy()
        if sub.empty:
            continue

        rule = sub["_stmt_type_norm"].isin(rule_labels)
        a_has = sub[f"a_has_{grp}"]
        explicit = sub["_a_is_explicit"]

        reason = np.where(
            a_has & explicit & rule,
            "retained_final_rule_explicit_A",
            np.where(
                a_has & (~explicit) & rule,
                "lost_nonexplicit_A_in_rules",
                np.where(
                    a_has & explicit & (~rule),
                    "lost_rules_only_cutoff",
                    np.where(
                        a_has & (~explicit) & (~rule),
                        "lost_rules_and_explicit_filters",
                        "sentence_only_or_parser_miss",
                    ),
                ),
            ),
        )
        sub["loss_reason"] = reason

        counts = (
            sub.groupby("loss_reason", dropna=False)
            .size()
            .reset_index(name="count")
            .sort_values(["count", "loss_reason"], ascending=[False, True])
            .reset_index(drop=True)
        )
        counts.insert(0, "pedagogical_group", grp)
        counts["percent_of_sentence_mentions"] = (100.0 * counts["count"] / len(sub)).round(2)
        rows.append(counts)

    if not rows:
        return pd.DataFrame(columns=["pedagogical_group", "loss_reason", "count", "percent_of_sentence_mentions"])
    return pd.concat(rows, ignore_index=True)



def build_loss_reasons_from_a(df: pd.DataFrame, rule_labels: set[str]) -> pd.DataFrame:
    rows: List[dict] = []
    for grp in PED_GROUP_ORDER:
        sub = df[df[f"a_has_{grp}"]].copy()
        if sub.empty:
            continue

        rule = sub["_stmt_type_norm"].isin(rule_labels)
        explicit = sub["_a_is_explicit"]

        reason = np.where(
            explicit & rule,
            "retained_final_rule_explicit_A",
            np.where(
                (~explicit) & rule,
                "lost_nonexplicit_A_in_rules",
                np.where(
                    explicit & (~rule),
                    "lost_rules_only_cutoff",
                    "lost_rules_and_explicit_filters",
                ),
            ),
        )
        sub["loss_reason"] = reason

        counts = (
            sub.groupby("loss_reason", dropna=False)
            .size()
            .reset_index(name="count")
            .sort_values(["count", "loss_reason"], ascending=[False, True])
            .reset_index(drop=True)
        )
        counts.insert(0, "pedagogical_group", grp)
        counts["percent_of_a_mentions"] = (100.0 * counts["count"] / len(sub)).round(2)
        rows.append(counts)

    if not rows:
        return pd.DataFrame(columns=["pedagogical_group", "loss_reason", "count", "percent_of_a_mentions"])
    return pd.concat(rows, ignore_index=True)


# -----------------------------------------------------------------------------
# Exemplars
# -----------------------------------------------------------------------------

KEEP_COLS = [
    "doc_id",
    "chunk_id",
    "sentence_index_in_chunk",
    "statement_type_candidate",
    "a_class",
    "a_raw_text",
    "a_norm",
    "sent_ped_groups",
    "a_ped_groups",
    "sentence_text",
]



def make_exemplars(df: pd.DataFrame, rule_labels: set[str], top_n: int, seed: int, sample_mode: str) -> pd.DataFrame:
    parts: List[pd.DataFrame] = []

    def take(sub: pd.DataFrame, bucket: str, grp: str) -> None:
        if sub.empty:
            return
        n = min(top_n, len(sub))
        if sample_mode == "random":
            out = sub.sample(n=n, random_state=seed)
        else:
            sort_cols = [c for c in ["doc_id", "chunk_id", "sentence_index_in_chunk"] if c in sub.columns]
            out = sub.sort_values(sort_cols).head(n) if sort_cols else sub.head(n)
        out = out[[c for c in KEEP_COLS if c in out.columns]].copy()
        out.insert(0, "pedagogical_group", grp)
        out.insert(0, "bucket", bucket)
        parts.append(out)

    for grp in PED_GROUP_ORDER:
        sent = df[f"sent_has_{grp}"]
        a_has = df[f"a_has_{grp}"]
        explicit = df["_a_is_explicit"]
        rule = df["_stmt_type_norm"].isin(rule_labels)

        take(df[sent & (~a_has)].copy(), "sentence_only_or_parser_miss", grp)
        take(df[a_has & rule & (~explicit)].copy(), "nonexplicit_rules", grp)
        take(df[a_has & explicit & (~rule)].copy(), "rules_only_cutoff", grp)
        take(df[a_has & explicit & rule].copy(), "final_retained", grp)

    if not parts:
        return pd.DataFrame(columns=["bucket", "pedagogical_group"] + KEEP_COLS)
    return pd.concat(parts, ignore_index=True)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--edu-igt",
        default="data/derived/step8_igt_chunks_edu/igt_statements_full.parquet",
        help="Education-relevant IGT parquet.",
    )
    ap.add_argument(
        "--out-dir",
        default="data/derived/step8_analysis/pedagogical_actor_loss_audit",
        help="Output directory.",
    )
    ap.add_argument(
        "--rule-labels",
        default="rule_candidate,rule,rules",
        help="Comma-separated statement-type labels treated as rules.",
    )
    ap.add_argument("--top-exemplars", type=int, default=20)
    ap.add_argument("--sample-mode", choices=["random", "head"], default="random")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    rule_labels = {x.strip().lower() for x in args.rule_labels.split(",") if x.strip()}
    if not rule_labels:
        rule_labels = set(DEFAULT_RULE_LABELS)

    edu_igt_path = Path(args.edu_igt)
    df = maybe_read_table(edu_igt_path)
    validate_minimum_schema(df)

    df = df.copy()
    df["_stmt_type_norm"] = normalize_stmt_type(df["statement_type_candidate"])
    df["_a_is_explicit"] = df["a_class"].fillna("").astype(str).str.strip().str.lower().eq("explicit")
    df["_a_nonempty"] = df["a_raw_text"].fillna("").astype(str).str.strip().ne("")
    df["a_norm"] = normalize_a_text(df["a_raw_text"])

    df["sent_ped_groups_list"] = df["sentence_text"].fillna("").astype(str).apply(
        lambda s: match_groups(s, PED_GROUP_PATTERNS, PED_GROUP_ORDER)
    )
    df["a_ped_groups_list"] = df["a_norm"].fillna("").astype(str).apply(
        lambda s: match_groups(s, PED_GROUP_PATTERNS, PED_GROUP_ORDER)
    )
    df["sent_ped_groups"] = df["sent_ped_groups_list"].apply(lambda xs: "|".join(xs))
    df["a_ped_groups"] = df["a_ped_groups_list"].apply(lambda xs: "|".join(xs))
    df["sent_ped_group_primary"] = df["sent_ped_groups_list"].apply(first_or_empty)
    df["a_ped_group_primary"] = df["a_ped_groups_list"].apply(first_or_empty)

    for grp in PED_GROUP_ORDER:
        df[f"sent_has_{grp}"] = df["sent_ped_groups_list"].apply(lambda xs, g=grp: g in xs)
        df[f"a_has_{grp}"] = df["a_ped_groups_list"].apply(lambda xs, g=grp: g in xs)

    stage_counts = build_stage_counts(df, rule_labels)
    stmt_profile_sentence = build_statement_type_profiles(df, base="sentence")
    stmt_profile_a = build_statement_type_profiles(df, base="a")
    loss_sentence = build_loss_reasons_from_sentence(df, rule_labels)
    loss_a = build_loss_reasons_from_a(df, rule_labels)
    exemplars = make_exemplars(df, rule_labels, top_n=args.top_exemplars, seed=args.seed, sample_mode=args.sample_mode)

    safe_to_csv(stage_counts, out_dir / "audit_stage_counts.csv")
    safe_to_csv(stmt_profile_sentence, out_dir / "pedagogical_statement_type_profile_sentence_mentions.csv")
    safe_to_csv(stmt_profile_a, out_dir / "pedagogical_statement_type_profile_a_mentions.csv")
    safe_to_csv(loss_sentence, out_dir / "pedagogical_loss_reasons_from_sentence_mentions.csv")
    safe_to_csv(loss_a, out_dir / "pedagogical_loss_reasons_from_a_mentions.csv")
    safe_to_csv(exemplars, out_dir / "pedagogical_exemplars.csv")
    safe_to_csv(exemplars[exemplars["bucket"] == "sentence_only_or_parser_miss"].copy(), out_dir / "pedagogical_examples_sentence_only_or_parser_miss.csv")
    safe_to_csv(exemplars[exemplars["bucket"] == "nonexplicit_rules"].copy(), out_dir / "pedagogical_examples_nonexplicit_rules.csv")
    safe_to_csv(exemplars[exemplars["bucket"] == "rules_only_cutoff"].copy(), out_dir / "pedagogical_examples_rules_cutoff.csv")
    safe_to_csv(exemplars[exemplars["bucket"] == "final_retained"].copy(), out_dir / "pedagogical_examples_final_retained.csv")

    summary_md_parts = [
        "# Pedagogical actor loss audit",
        "",
        "## Stage counts",
        build_markdown_table(stage_counts),
        "",
        "## Loss reasons from sentence mentions",
        build_markdown_table(loss_sentence),
        "",
        "## Loss reasons from A mentions",
        build_markdown_table(loss_a),
    ]
    (out_dir / "pedagogical_loss_summary.md").write_text("\n".join(summary_md_parts), encoding="utf-8")

    metadata = {
        "script": "step8_5c_pedagogical_actor_loss_audit.py",
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "input_path": str(edu_igt_path),
        "rule_labels": sorted(rule_labels),
        "n_rows_input": int(len(df)),
        "pedagogical_groups": PED_GROUP_ORDER,
        "top_exemplars": int(args.top_exemplars),
        "sample_mode": args.sample_mode,
        "seed": int(args.seed),
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("Saved:")
    for p in [
        out_dir / "run_metadata.json",
        out_dir / "audit_stage_counts.csv",
        out_dir / "pedagogical_statement_type_profile_sentence_mentions.csv",
        out_dir / "pedagogical_statement_type_profile_a_mentions.csv",
        out_dir / "pedagogical_loss_reasons_from_sentence_mentions.csv",
        out_dir / "pedagogical_loss_reasons_from_a_mentions.csv",
        out_dir / "pedagogical_loss_summary.md",
        out_dir / "pedagogical_exemplars.csv",
        out_dir / "pedagogical_examples_sentence_only_or_parser_miss.csv",
        out_dir / "pedagogical_examples_nonexplicit_rules.csv",
        out_dir / "pedagogical_examples_rules_cutoff.csv",
        out_dir / "pedagogical_examples_final_retained.csv",
    ]:
        print(f"  {p}")

    print("\nLoss reasons from sentence mentions:")
    if loss_sentence.empty:
        print("  [no pedagogical sentence mentions found]")
    else:
        print(loss_sentence.to_string(index=False))

    print("\nLoss reasons from A mentions:")
    if loss_a.empty:
        print("  [no pedagogical A mentions found]")
    else:
        print(loss_a.to_string(index=False))


if __name__ == "__main__":
    main()
