#!/usr/bin/env python3
from __future__ import annotations

import csv
import re
import time
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

SOURCE_INPUT = Path("data/derived/step8_9_regime_closure/igt_statements_full_with_edu_flags.parquet")
LOOKUP_INPUT = Path("data/derived/step9_country_dataset/doc_country_lookup.csv")

TYPED_RULES_INPUTS = [
    Path("data/derived/step8_analysis/iad_rule_types_rules_only/education_relevant_iad_rule_types_rules_only.parquet"),
    Path("data/derived/step8_analysis/iad_rule_types_rules_only/education_relevant_iad_rule_types_rules_only.csv"),
]

OUTDIR = Path("data/derived/step10_education_dataset")
COUNTRY_COUNTS_OUT = OUTDIR / "education_country_rule_type_counts.csv"
COUNTRY_PRIMARY_OUT = OUTDIR / "education_country_rule_type_primary_counts.csv"
POOLED_OUT = OUTDIR / "education_rule_type_pooled_summary.csv"
ZERO_OUT = OUTDIR / "education_rule_type_zero_summary.csv"
DOC_OUT = OUTDIR / "education_country_rule_type_doc_summary.csv"


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

RULE_LABELS = {"rule_candidate", "rule", "rules"}

BINARY_RULE_COLS = [
    "iad_boundary",
    "iad_position",
    "iad_choice",
    "iad_information",
    "iad_scope",
    "iad_aggregation",
    "iad_payoff",
]

PRIMARY_PRIORITY = [
    "boundary",
    "aggregation",
    "payoff",
    "information",
    "position",
    "scope",
    "choice",
]

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


def maybe_read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type: {path}")


def first_existing(paths: Iterable[Path]) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None


def textify(x) -> str:
    if x is None:
        return ""
    if pd.isna(x):
        return ""
    return str(x).strip()


def normalize_space(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def pct_true(s: pd.Series) -> float:
    s = s.fillna(False).astype(bool)
    return float(s.mean()) if len(s) else np.nan


def safe_to_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(
        path,
        index=False,
        encoding="utf-8",
        quoting=csv.QUOTE_MINIMAL,
        escapechar="\\",
    )


def detect_required_columns(df: pd.DataFrame) -> Dict[str, str]:
    required = [
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
        "o_local_text",
        "o_local_present",
        "statement_type_candidate",
        "edu_any_hit",
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
        "edu_any_hit": "edu_any_hit",
    }


def classify_statement(row: pd.Series, cols: Dict[str, str]) -> Dict[str, object]:
    sentence = textify(row[cols["sentence"]])
    a_text = textify(row[cols["a_text"]])
    d_text = textify(row[cols["d_text"]])
    d_class = textify(row[cols["d_class"]])
    i_text = textify(row[cols["i_text"]])
    c_text = textify(row[cols["c_text"]])
    b_text = textify(row[cols["b_text"]])
    o_text = textify(row[cols["o_text"]])

    full_text = normalize_space(
        " ".join(part for part in [sentence, a_text, d_text, d_class, i_text, c_text, b_text, o_text] if part)
    )
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


def ensure_rule_type_columns(df: pd.DataFrame, cols: Dict[str, str]) -> pd.DataFrame:
    need = set(BINARY_RULE_COLS + ["iad_rule_type_hits", "iad_rule_type_n_hits", "iad_rule_type_primary"])
    if need.issubset(df.columns):
        out = df.copy()
        for c in BINARY_RULE_COLS:
            out[c] = out[c].fillna(0).astype(int)
        out["iad_rule_type_n_hits"] = out["iad_rule_type_n_hits"].fillna(0).astype(int)
        out["iad_rule_type_primary"] = out["iad_rule_type_primary"].fillna("unclassified").astype(str)
        out["iad_rule_type_hits"] = out["iad_rule_type_hits"].fillna("").astype(str)
        return out

    typed_rows = [classify_statement(row, cols) for _, row in df.iterrows()]
    typed = pd.DataFrame(typed_rows)
    return pd.concat([df.reset_index(drop=True), typed], axis=1)
# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    t0 = time.time()
    OUTDIR.mkdir(parents=True, exist_ok=True)

    typed_input = first_existing(TYPED_RULES_INPUTS)

    if typed_input is not None:
        print(f"[step10_4] loading pretyped rules file: {typed_input}")
        df = maybe_read_table(typed_input)

        # Bring country in if missing.
        if "country" not in df.columns:
            lookup = pd.read_csv(LOOKUP_INPUT)[["doc_id", "country"]].drop_duplicates()
            df = df.merge(lookup, on="doc_id", how="left")

        # Safety filter to education rows if the file contains extras.
        if "edu_any_hit" in df.columns:
            df = df[df["edu_any_hit"].fillna(False).astype(bool)].copy()

        # Safety filter to rules only.
        if "statement_type_candidate" in df.columns:
            df["_stmt_type_norm"] = df["statement_type_candidate"].astype(str).str.strip().str.lower()
            df = df[df["_stmt_type_norm"].isin(RULE_LABELS)].copy()
    else:
        print(f"[step10_4] loading source statement file: {SOURCE_INPUT}")
        df = pd.read_parquet(SOURCE_INPUT)
        cols = detect_required_columns(df)

        print("[step10_4] filtering to education-relevant statements")
        df = df[df[cols["edu_any_hit"]].fillna(False).astype(bool)].copy()

        print("[step10_4] filtering to rule statements only")
        df["_stmt_type_norm"] = df[cols["stmt_type"]].astype(str).str.strip().str.lower()
        df = df[df["_stmt_type_norm"].isin(RULE_LABELS)].copy()

        print("[step10_4] attaching country lookup")
        lookup = pd.read_csv(LOOKUP_INPUT)[["doc_id", "country"]].drop_duplicates()
        df = df.merge(lookup, on="doc_id", how="left")

        print("[step10_4] classifying IAD rule types")
        df = ensure_rule_type_columns(df, cols)

    # Ensure core cols exist after whichever path was used.
    needed_after = ["country", "doc_id", "chunk_id", "iad_rule_type_primary"] + BINARY_RULE_COLS
    missing_after = [c for c in needed_after if c not in df.columns]
    if missing_after:
        raise ValueError(f"[step10_4] missing required columns after preparation: {missing_after}")

    # Normalize binary cols
    for c in BINARY_RULE_COLS:
        df[c] = df[c].fillna(0).astype(int)

    df["country"] = df["country"].fillna("UNKNOWN_COUNTRY").astype(str)
    df["iad_rule_type_primary"] = df["iad_rule_type_primary"].fillna("unclassified").astype(str)

    print(f"[step10_4] education-relevant rule rows: {len(df):,}")
    print(f"[step10_4] countries: {df['country'].nunique():,}")
    print(f"[step10_4] docs: {df['doc_id'].nunique():,}")

    # -----------------------------------------------------------------
    # Country-level binary counts and shares
    # -----------------------------------------------------------------
    country_docs = (
        df.groupby("country")["doc_id"]
        .nunique()
        .rename("n_docs_with_edu_rules")
    )

    country_rules = (
        df.groupby("country")
        .size()
        .rename("n_edu_rule_statements")
    )

    country_binary = (
        df.groupby("country")[BINARY_RULE_COLS]
        .sum()
        .reset_index()
    )

    out = (
        country_binary
        .merge(country_docs.reset_index(), on="country", how="left")
        .merge(country_rules.reset_index(), on="country", how="left")
    )

    for c in BINARY_RULE_COLS:
        short = c.replace("iad_", "")
        out[f"{short}_share_within_edu_rules"] = out[c] / out["n_edu_rule_statements"].replace({0: np.nan})

    out = out.sort_values(["n_edu_rule_statements", "country"], ascending=[False, True]).reset_index(drop=True)

    # -----------------------------------------------------------------
    # Country-level primary rule type counts
    # -----------------------------------------------------------------
    primary = (
        df.groupby(["country", "iad_rule_type_primary"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    primary_type_order = ["boundary", "position", "choice", "information", "scope", "aggregation", "payoff", "unclassified"]
    for c in primary_type_order:
        if c not in primary.columns:
            primary[c] = 0

    primary = primary[["country"] + primary_type_order]
    primary = primary.merge(country_rules.reset_index(), on="country", how="left")

    for c in primary_type_order:
        primary[f"{c}_primary_share_within_edu_rules"] = primary[c] / primary["n_edu_rule_statements"].replace({0: np.nan})

    primary = primary.sort_values(["n_edu_rule_statements", "country"], ascending=[False, True]).reset_index(drop=True)
    # -----------------------------------------------------------------
    # Pooled summary
    # -----------------------------------------------------------------
    pooled_rows: List[Dict[str, object]] = []
    total_rules = len(df)

    for c in BINARY_RULE_COLS:
        short = c.replace("iad_", "")
        n = int(df[c].sum())
        pooled_rows.append({
            "summary_type": "binary_hit",
            "rule_type": short,
            "n": n,
            "share_of_edu_rules": (n / total_rules) if total_rules else np.nan,
        })

    primary_counts = df["iad_rule_type_primary"].value_counts(dropna=False)
    for c in primary_type_order:
        n = int(primary_counts.get(c, 0))
        pooled_rows.append({
            "summary_type": "primary_type",
            "rule_type": c,
            "n": n,
            "share_of_edu_rules": (n / total_rules) if total_rules else np.nan,
        })

    pooled = pd.DataFrame(pooled_rows)

    # -----------------------------------------------------------------
    # Zero-count / rarity summary
    # -----------------------------------------------------------------
    zero_rows: List[Dict[str, object]] = []
    n_countries = out["country"].nunique()

    for c in BINARY_RULE_COLS:
        short = c.replace("iad_", "")
        n_zero = int((out[c] == 0).sum())
        n_nonzero = int((out[c] > 0).sum())
        zero_rows.append({
            "rule_type": short,
            "n_countries_zero": n_zero,
            "n_countries_nonzero": n_nonzero,
            "pct_countries_zero": n_zero / n_countries if n_countries else np.nan,
            "pct_countries_nonzero": n_nonzero / n_countries if n_countries else np.nan,
        })

    zero_df = pd.DataFrame(zero_rows).sort_values(["n_countries_zero", "rule_type"], ascending=[False, True])

    # -----------------------------------------------------------------
    # Lightweight doc summary
    # -----------------------------------------------------------------
    doc_summary = pd.DataFrame([{
        "n_edu_rule_statements_total": len(df),
        "n_docs_with_edu_rules_total": df["doc_id"].nunique(),
        "n_countries_total": df["country"].nunique(),
        "typed_input_used": str(typed_input) if typed_input is not None else "",
        "source_input_used": str(SOURCE_INPUT),
        "lookup_input_used": str(LOOKUP_INPUT),
    }])

    safe_to_csv(out, COUNTRY_COUNTS_OUT)
    safe_to_csv(primary, COUNTRY_PRIMARY_OUT)
    safe_to_csv(pooled, POOLED_OUT)
    safe_to_csv(zero_df, ZERO_OUT)
    safe_to_csv(doc_summary, DOC_OUT)

    print(f"[step10_4] wrote: {COUNTRY_COUNTS_OUT}")
    print(f"[step10_4] wrote: {COUNTRY_PRIMARY_OUT}")
    print(f"[step10_4] wrote: {POOLED_OUT}")
    print(f"[step10_4] wrote: {ZERO_OUT}")
    print(f"[step10_4] wrote: {DOC_OUT}")
    print(f"[step10_4] done elapsed_s={round(time.time() - t0, 2)}")


if __name__ == "__main__":
    main()
