#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


# --------------------------------------------------
# Helpers
# --------------------------------------------------

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
    if pd.isna(x):
        return ""
    return str(x).strip()


def normalize_space(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def build_markdown_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, row in df.iterrows():
        vals = [str(row[c]) for c in cols]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


# --------------------------------------------------
# Repo schema detection
# --------------------------------------------------

def detect_columns(df: pd.DataFrame) -> Dict[str, str]:
    required = [
        "doc_id",
        "chunk_id",
        "sentence_text",
        "d_surface",
        "d_class",
        "i_phrase_text",
        "c_texts",
        "b_text",
        "a_raw_text",
        "a_class",
        "o_local_present",
        "o_local_text",
        "statement_type_candidate",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    return {
        "doc_id": "doc_id",
        "chunk_id": "chunk_id",
        "sentence": "sentence_text",
        "a_text": "a_raw_text",
        "a_class": "a_class",
        "d_text": "d_surface",
        "d_class": "d_class",
        "i_text": "i_phrase_text",
        "c_text": "c_texts",
        "b_text": "b_text",
        "o_text": "o_local_text",
        "o_present": "o_local_present",
        "stmt_type": "statement_type_candidate",
    }


# --------------------------------------------------
# Rules-only filter
# --------------------------------------------------

DEFAULT_RULE_LABELS = {"rule_candidate", "rule", "rules"}


def filter_to_rules_only(df: pd.DataFrame, stmt_type_col: str, rule_labels: set[str]) -> pd.DataFrame:
    out = df.copy()
    out["_stmt_type_norm"] = out[stmt_type_col].astype(str).str.strip().str.lower()
    return out[out["_stmt_type_norm"].isin(rule_labels)].copy()
# --------------------------------------------------
# Cleaned IAD rule-type patterns
# --------------------------------------------------

PATTERNS = {
    "boundary": re.compile(
        r"\b("
        r"eligible|eligibility|qualif(?:y|ies|ied|ication|ications)|criteria|criterion|"
        r"admi(?:t|ts|tted|ssion|ssions)|enrol(?:l|ls|led|ment)|enrolment|"
        r"register(?:ed|s|ing|ation)?|licen[sc](?:e|ed|ing|sure)|"
        r"certif(?:y|ied|ication|ications)|accredit(?:ed|ation|ations)?|"
        r"authorized user|authorised user|designation|designated|appoint(?:ed|ment)?|"
        r"applicant|applicants|entry|exit|membership|member state"
        r")\b",
        re.I,
    ),
    "position": re.compile(
        r"\b("
        r"ministry|minist(?:er|ers)|department|agency|authority|authorities|"
        r"commission|committee|council|board|office|officer|secretary|administrator|"
        r"director|regulator|controller|processor|operator|provider|coordinator|"
        r"school|schools|university|universities|college|colleges|institution|institutions|"
        r"teacher|teachers|educator|educators|student|students|learner|learners|"
        r"researcher|researchers|user|users|body|bodies|task force|working group"
        r")\b",
        re.I,
    ),
    "choice": re.compile(
        r"\b("
        r"shall|must|may|should|can|cannot|must not|may not|shall not|should not|"
        r"required to|is required to|are required to|"
        r"permitted to|allowed to|authorized to|authorised to|"
        r"prohibited from|forbidden from|ban(?:ned)?|obliged to|obligation"
        r")\b",
        re.I,
    ),
    "information": re.compile(
        r"\b("
        r"report|reports|reporting|reported|"
        r"disclos(?:e|ed|es|ure)|publish|published|publication|publicly available|"
        r"notify|notification|inform|information|"
        r"record|records|recordkeeping|document|documentation|"
        r"monitor|monitoring|audit|auditing|"
        r"assess|assessment|evaluate|evaluation|review|reviews|"
        r"transparency|submission|communicat(?:e|ion|ions)|"
        r"data sharing|share information"
        r")\b",
        re.I,
    ),
    "scope": re.compile(
        r"\b("
        r"goal|goals|objective|objectives|aim|aims|purpose|purposes|"
        r"outcome|outcomes|target|targets|vision|priority|priorities|"
        r"roadmap|framework|mission|milestone|"
        r"limit|limits|threshold|thresholds|"
        r"standard|standards|benchmark|benchmarks"
        r")\b",
        re.I,
    ),
    "aggregation": re.compile(
        r"\b("
        r"vote|voting|majority|minority|quorum|consensus|approve|approval|"
        r"adopt|adoption|joint decision|collective decision|"
        r"committee decision|board decision|panel decision|"
        r"consultation|consult|coordination mechanism|interministerial|inter-ministerial"
        r")\b",
        re.I,
    ),
    # tightened: removed generic cost/benefit/financial
    "payoff": re.compile(
        r"\b("
        r"fund|funding|budget|allocate|allocation|grant|grants|"
        r"subsidy|subsidies|incentive|incentives|penalty|penalties|fine|fines|"
        r"sanction|sanctions|reward|rewards|scholarship|scholarships|"
        r"liable|liability|compensation|fee|fees"
        r")\b",
        re.I,
    ),
}

#  specific institutional functions first; choice as fallback.
PRIMARY_PRIORITY = [
    "boundary",
    "aggregation",
    "payoff",
    "information",
    "position",
    "scope",
    "choice",
]

# --------------------------------------------------
# Classification
# --------------------------------------------------

def classify_statement(row: pd.Series, cols: Dict[str, str]) -> Dict[str, object]:
    sentence = textify(row[cols["sentence"]])
    a_text = textify(row[cols["a_text"]])
    d_text = textify(row[cols["d_text"]])
    d_class = textify(row[cols["d_class"]])
    i_text = textify(row[cols["i_text"]])
    c_text = textify(row[cols["c_text"]])
    b_text = textify(row[cols["b_text"]])
    o_text = textify(row[cols["o_text"]])

    full_text = normalize_space(" ".join(
        part for part in [sentence, a_text, d_text, d_class, i_text, c_text, b_text, o_text] if part
    ))
    actor_text = normalize_space(a_text)
    deontic_text = normalize_space(" ".join(part for part in [d_text, d_class] if part))

    hits = set()

    if PATTERNS["boundary"].search(full_text):
        hits.add("boundary")

    if actor_text and PATTERNS["position"].search(actor_text):
        hits.add("position")
    elif PATTERNS["position"].search(full_text):
        hits.add("position")

    if PATTERNS["aggregation"].search(full_text):
        hits.add("aggregation")

    if PATTERNS["information"].search(full_text):
        hits.add("information")

    if PATTERNS["payoff"].search(full_text):
        hits.add("payoff")

    if PATTERNS["scope"].search(full_text):
        hits.add("scope")

    # Because this is already rules-only, deontic presence implies a choice component.
    if deontic_text or PATTERNS["choice"].search(full_text):
        hits.add("choice")

    primary = next((rule for rule in PRIMARY_PRIORITY if rule in hits), "unclassified")

    return {
        "iad_boundary": int("boundary" in hits),
        "iad_position": int("position" in hits),
        "iad_choice": int("choice" in hits),
        "iad_information": int("information" in hits),
        "iad_scope": int("scope" in hits),
        "iad_aggregation": int("aggregation" in hits),
        "iad_payoff": int("payoff" in hits),
        "iad_rule_type_hits": "|".join(sorted(hits)) if hits else "",
        "iad_rule_type_n_hits": len(hits),
        "iad_rule_type_primary": primary,
    }


# --------------------------------------------------
# Summaries
# --------------------------------------------------

def summarize_primary(df: pd.DataFrame, corpus_name: str) -> pd.DataFrame:
    total = len(df)
    out = (
        df.groupby("iad_rule_type_primary", dropna=False)
        .size()
        .reset_index(name="n")
        .sort_values(["n", "iad_rule_type_primary"], ascending=[False, True])
        .reset_index(drop=True)
    )
    out["pct"] = (100.0 * out["n"] / total).round(2) if total else 0.0
    out.insert(0, "corpus", corpus_name)
    return out


def summarize_binary(df: pd.DataFrame, corpus_name: str) -> pd.DataFrame:
    cols = [
        "iad_boundary",
        "iad_position",
        "iad_choice",
        "iad_information",
        "iad_scope",
        "iad_aggregation",
        "iad_payoff",
    ]
    total = len(df)
    rows = []
    for c in cols:
        n = int(df[c].sum()) if total else 0
        rows.append({
            "corpus": corpus_name,
            "rule_type": c.replace("iad_", ""),
            "n_rule_statements_with_hit": n,
            "pct_rule_statements_with_hit": round(100.0 * n / total, 2) if total else 0.0,
        })
    return pd.DataFrame(rows)


def summarize_rule_subset_counts(corpus_name: str, total_before: int, total_rules_only: int) -> Dict[str, object]:
    return {
        "corpus": corpus_name,
        "total_statements_before_filter": total_before,
        "total_rule_statements_after_filter": total_rules_only,
        "pct_kept_as_rules": round(100.0 * total_rules_only / total_before, 2) if total_before else 0.0,
    }

# --------------------------------------------------
# QC export
# --------------------------------------------------

def export_qc_sample(
    df: pd.DataFrame,
    cols: Dict[str, str],
    out_path: Path,
    per_primary: int = 20,
    seed: int = 42,
) -> None:
    random.seed(seed)

    keep_cols = [
        cols["doc_id"],
        cols["chunk_id"],
        cols["sentence"],
        cols["a_text"],
        cols["a_class"],
        cols["d_text"],
        cols["d_class"],
        cols["i_text"],
        cols["c_text"],
        cols["b_text"],
        cols["o_text"],
        cols["stmt_type"],
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
    ]

    parts = []
    for label in ["boundary", "position", "choice", "information", "scope", "aggregation", "payoff", "unclassified"]:
        tmp = df[df["iad_rule_type_primary"] == label].copy()
        if tmp.empty:
            continue
        n = min(per_primary, len(tmp))
        tmp = tmp.sample(n=n, random_state=seed)
        tmp.insert(0, "qc_bucket", label)
        parts.append(tmp[["qc_bucket"] + keep_cols])

    if parts:
        out = pd.concat(parts, ignore_index=True)
        safe_to_csv(out, out_path)


# --------------------------------------------------
# Per-corpus runner
# --------------------------------------------------

def run_one_corpus(
    df: pd.DataFrame,
    corpus_name: str,
    out_prefix: Path,
    rule_labels: set[str],
    qc_per_primary: int,
    qc_seed: int,
) -> Tuple[pd.DataFrame, Dict[str, str], Dict[str, object]]:
    cols = detect_columns(df)
    stmt_type_col = cols["stmt_type"]

    print(f"\n[{corpus_name}] column map")
    for k, v in cols.items():
        print(f"  {k}: {v}")

    total_before = len(df)
    rules_only = filter_to_rules_only(df, stmt_type_col=stmt_type_col, rule_labels=rule_labels)
    total_after = len(rules_only)

    print(f"[{corpus_name}] kept {total_after:,} / {total_before:,} rows as rules")

    classified_rows = [classify_statement(row, cols) for _, row in rules_only.iterrows()]
    class_df = pd.DataFrame(classified_rows)

    out = pd.concat([rules_only.reset_index(drop=True), class_df], axis=1)
    out["corpus"] = corpus_name

    out_parquet = out_prefix.with_suffix(".parquet")
    out_csv = out_prefix.with_suffix(".csv")
    qc_csv = out_prefix.parent / f"{out_prefix.name}_qc_sample.csv"

    out.to_parquet(out_parquet, index=False)
    safe_to_csv(out, out_csv)
    export_qc_sample(out, cols, qc_csv, per_primary=qc_per_primary, seed=qc_seed)

    print(f"Saved → {out_parquet}")
    print(f"Saved → {out_csv}")
    print(f"Saved → {qc_csv}")

    subset_counts = summarize_rule_subset_counts(
        corpus_name=corpus_name,
        total_before=total_before,
        total_rules_only=total_after,
    )

    return out, cols, subset_counts


# --------------------------------------------------
# Main
# --------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full-igt", default="data/derived/step8_igt_full/igt_statements_full.parquet")
    ap.add_argument("--edu-igt", default="data/derived/step8_igt_chunks_edu/igt_statements_full.parquet")
    ap.add_argument("--title-igt", default="data/derived/step8_igt_title_edu/igt_statements_full.parquet")
    ap.add_argument("--out-dir", default="data/derived/step8_analysis/iad_rule_types_rules_only")
    ap.add_argument(
        "--rule-labels",
        default="rule_candidate,rule,rules",
        help="Comma-separated statement-type labels to treat as rules",
    )
    ap.add_argument("--qc-per-primary", type=int, default=20)
    ap.add_argument("--qc-seed", type=int, default=42)
    args = ap.parse_args()

    rule_labels = {x.strip().lower() for x in args.rule_labels.split(",") if x.strip()}
    if not rule_labels:
        rule_labels = set(DEFAULT_RULE_LABELS)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    corpora = [
        ("Full corpus", Path(args.full_igt), out_dir / "full_corpus_iad_rule_types_rules_only"),
        ("Education-relevant", Path(args.edu_igt), out_dir / "education_relevant_iad_rule_types_rules_only"),
        ("Education-in-title", Path(args.title_igt), out_dir / "education_in_title_iad_rule_types_rules_only"),
    ]

    primary_summaries = []
    binary_summaries = []
    subset_count_rows = []

    for corpus_name, in_path, out_prefix in corpora:
        df = maybe_read_table(in_path)
        typed_df, cols, subset_counts = run_one_corpus(
            df=df,
            corpus_name=corpus_name,
            out_prefix=out_prefix,
            rule_labels=rule_labels,
            qc_per_primary=args.qc_per_primary,
            qc_seed=args.qc_seed,
        )
        primary_summaries.append(summarize_primary(typed_df, corpus_name))
        binary_summaries.append(summarize_binary(typed_df, corpus_name))
        subset_count_rows.append(subset_counts)

    primary_long = pd.concat(primary_summaries, ignore_index=True)
    binary_long = pd.concat(binary_summaries, ignore_index=True)
    subset_counts_df = pd.DataFrame(subset_count_rows)

    primary_wide = (
        primary_long.pivot(index="iad_rule_type_primary", columns="corpus", values="pct")
        .reset_index()
        .fillna(0.0)
    )
    binary_wide = (
        binary_long.pivot(index="rule_type", columns="corpus", values="pct_rule_statements_with_hit")
        .reset_index()
        .fillna(0.0)
    )

    base = out_dir.parent

    primary_long_csv = base / "iad_rule_type_rules_only_primary_summary_long.csv"
    primary_wide_csv = base / "iad_rule_type_rules_only_primary_summary_wide.csv"
    primary_md = base / "iad_rule_type_rules_only_primary_summary_long.md"

    binary_long_csv = base / "iad_rule_type_rules_only_binary_hits_summary_long.csv"
    binary_wide_csv = base / "iad_rule_type_rules_only_binary_hits_summary_wide.csv"
    binary_md = base / "iad_rule_type_rules_only_binary_hits_summary_long.md"

    subset_counts_csv = base / "iad_rule_subset_counts_by_corpus.csv"
    subset_counts_md = base / "iad_rule_subset_counts_by_corpus.md"

    safe_to_csv(primary_long, primary_long_csv)
    safe_to_csv(primary_wide, primary_wide_csv)
    primary_md.write_text(build_markdown_table(primary_long), encoding="utf-8")

    safe_to_csv(binary_long, binary_long_csv)
    safe_to_csv(binary_wide, binary_wide_csv)
    binary_md.write_text(build_markdown_table(binary_long), encoding="utf-8")

    safe_to_csv(subset_counts_df, subset_counts_csv)
    subset_counts_md.write_text(build_markdown_table(subset_counts_df), encoding="utf-8")

    print("\nSaved summaries:")
    print(f"  {primary_long_csv}")
    print(f"  {primary_wide_csv}")
    print(f"  {primary_md}")
    print(f"  {binary_long_csv}")
    print(f"  {binary_wide_csv}")
    print(f"  {binary_md}")
    print(f"  {subset_counts_csv}")
    print(f"  {subset_counts_md}")

    print("\nRule subset counts:")
    print(subset_counts_df.to_string(index=False))

    print("\nPrimary rule type summary (rules only):")
    print(primary_long.to_string(index=False))

    print("\nBinary hit summary (rules only):")
    print(binary_long.to_string(index=False))


if __name__ == "__main__":
    main()
