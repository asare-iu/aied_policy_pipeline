#!/usr/bin/env python3
"""
Step 8.6 — EDU chunks: O-scope attribution + actor tables + stratified rule samples

Reads:
- EDU institutional statements (Step 8.3 output):
  data/derived/step8_igt_chunks_edu/igt_statements_full.parquet
- Umbrella-linked statements (Step 8.3b output over FULL chunks):
  data/derived/step8_3b_umbrella_full/statements_with_umbrella_o.parquet

Writes:
- data/derived/step8_analysis/edu_o_scope_actor_tables.csv
- data/derived/step8_analysis/edu_o_scope_deontic_tables.csv
- data/derived/step8_analysis/edu_o_scope_statement_type_tables.csv
- data/derived/step8_analysis/edu_rule_samples_by_o_scope.csv
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# -----------------------
# Small helpers
# -----------------------

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def norm_text(s: Optional[str]) -> str:
    s = "" if s is None else str(s)
    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s

def make_a_norm(series: pd.Series) -> pd.Series:
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

def classify_o_scope(row: pd.Series) -> str:
    """
    O-scope categories (mutually exclusive):

    1) statement_scoped_local_o
       -> o_local_present == True

    2) umbrella_scoped_strict
       -> no local O, and umbrella link is high/medium

    3) umbrella_present_scope_unclear
       -> no local O, umbrella link is low

    4) no_o_detected
       -> no local O, umbrella none
    """
    local = bool(row.get("o_local_present", False))
    if local:
        return "statement_scoped_local_o"

    conf = str(row.get("o_umbrella_confidence", "none")).lower().strip()
    if conf in {"high", "medium"}:
        return "umbrella_scoped_strict"
    if conf == "low":
        return "umbrella_present_scope_unclear"
    return "no_o_detected"


# -----------------------
# Actor grouping (analysis-only)
# -----------------------

ACTOR_GROUP_PATTERNS: Dict[str, List[str]] = {
    # pedagogical
    "educators_teachers": [
        r"\beducator(s)?\b", r"\bteacher(s)?\b", r"\binstructor(s)?\b", r"\blecturer(s)?\b",
        r"\btrainer(s)?\b", r"\bfaculty\b", r"\bprofessor(s)?\b",
    ],
    "students_learners": [
        r"\bstudent(s)?\b", r"\blearner(s)?\b", r"\bpupil(s)?\b", r"\bchild(ren)?\b",
    ],
    "schools_institutions": [
        r"\bschool(s)?\b", r"\buniversity\b", r"\buniversities\b",
        r"\bcollege(s)?\b", r"\bacademic institution(s)?\b", r"\beducational institution(s)?\b",
        r"\btraining provider(s)?\b", r"\bprovider(s)?\b", r"\binstitution(s)?\b",
    ],

    # governance
    "policy_makers_ministries": [
        r"\bministry\b", r"\bminister(s)?\b", r"\bgovernment\b", r"\bpublic authority\b",
        r"\bpublic authorities\b", r"\bpolicy maker(s)?\b", r"\bcompetent authority\b",
        r"\bcompetent authorities\b", r"\bstate\b", r"\bnational\b",
    ],
    "regulators_supervisors": [
        r"\bregulator(s)?\b", r"\bsupervisory authority\b", r"\bsupervisory authorities\b",
        r"\benforcement authority\b", r"\bdata protection authority\b", r"\bdpa\b",
        r"\bcommission\b", r"\bagency\b",
    ],

    # tech supply chain
    "platforms_systems": [
        r"\bplatform(s)?\b", r"\bsystem(s)?\b", r"\btool(s)?\b", r"\bmodel(s)?\b",
        r"\balgorithm(s)?\b", r"\bai system(s)?\b", r"\bai model(s)?\b",
    ],
    "deployers_users": [
        r"\bdeployer(s)?\b", r"\boperator(s)?\b", r"\buser(s)?\b",
        r"\bimplementer(s)?\b", r"\badopter(s)?\b",
    ],
    "commercial_designers_vendors": [
        r"\bvendor(s)?\b", r"\bprovider(s)?\b", r"\bsupplier(s)?\b",
        r"\bdeveloper(s)?\b", r"\bmanufacturer(s)?\b", r"\bservice provider(s)?\b",
        r"\bcommercial\b",
    ],
    "researchers": [
        r"\bresearcher(s)?\b", r"\bresearch\b", r"\bacademics?\b", r"\bscientist(s)?\b",
    ],
    "data_subjects": [
        r"\bdata subject(s)?\b", r"\bindividual(s)?\b", r"\bperson(s)?\b", r"\bcitizen(s)?\b",
    ],
}

ACTOR_GROUP_ORDER = [
    "educators_teachers",
    "students_learners",
    "schools_institutions",
    "policy_makers_ministries",
    "regulators_supervisors",
    "platforms_systems",
    "deployers_users",
    "commercial_designers_vendors",
    "researchers",
    "data_subjects",
]


def assign_actor_group(a_norm: str) -> str:
    a = norm_text(a_norm)
    if not a:
        return "unclassified"
    for grp in ACTOR_GROUP_ORDER:
        pats = ACTOR_GROUP_PATTERNS[grp]
        for p in pats:
            if re.search(p, a, flags=re.IGNORECASE):
                return grp
    return "unclassified"


# -----------------------
# Main analysis
# -----------------------

def read_parquet(p: Path) -> pd.DataFrame:
    return pd.read_parquet(p)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--edu-statements", default="data/derived/step8_igt_chunks_edu/igt_statements_full.parquet")
    ap.add_argument("--umbrella-linked", default="data/derived/step8_3b_umbrella_full/statements_with_umbrella_o.parquet")
    ap.add_argument("--out-dir", default="data/derived/step8_analysis")
    ap.add_argument("--sample-n", type=int, default=50, help="Per O-scope stratum for rule_candidate sampling")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    edu_path = Path(args.edu_statements)
    umb_path = Path(args.umbrella_linked)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    # Load
    edu = read_parquet(edu_path)
    umb = read_parquet(umb_path)

    # Key alignment — use a stable join key
    # sentence_text can vary in whitespace; use doc_id+chunk_id+sentence_index_in_chunk primarily.
    key_cols = ["doc_id", "chunk_id", "sentence_index_in_chunk"]
    for c in key_cols:
        if c not in edu.columns or c not in umb.columns:
            raise SystemExit(f"Missing key column {c} in one of the inputs.")

    # Keep only umbrella fields needed
    umb_keep = umb[key_cols + [
        "o_umbrella_confidence",
        "o_umbrella_link_method",
        "o_umbrella_type",
        "o_umbrella_text",
        "o_umbrella_heading",
        "o_umbrella_present",
    ]].copy()

    merged = edu.merge(umb_keep, on=key_cols, how="left", validate="many_to_one")

    # Normalize/compute analysis fields
    merged["a_norm"] = make_a_norm(merged["a_raw_text"])
    merged["actor_group"] = merged["a_norm"].apply(assign_actor_group)

    merged["has_D"] = merged["d_lemma"].notna()
    merged["o_scope"] = merged.apply(classify_o_scope, axis=1)

    # ---------- Tables ----------
    def pct_within(df: pd.DataFrame, group_cols: List[str], count_col: str = "count") -> pd.DataFrame:
        out = df.copy()
        denom = out.groupby(group_cols[:-1])[count_col].transform("sum")
        out["percent_within"] = (out[count_col] / denom * 100).round(2)
        return out

    # 1) actor_group × o_scope
    t1 = (
        merged.groupby(["actor_group", "o_scope"])
        .size()
        .reset_index(name="count")
    )
    t1 = pct_within(t1, ["actor_group", "o_scope"])
    t1 = t1.sort_values(["actor_group", "count"], ascending=[True, False])

    out1 = out_dir / "edu_o_scope_actor_tables.csv"
    t1.to_csv(out1, index=False, escapechar="\\")

    # 2) actor_group × d_class × o_scope (only where D present and d_class not null)
    t2src = merged[merged["d_class"].notna()].copy()
    t2 = (
        t2src.groupby(["actor_group", "d_class", "o_scope"])
        .size()
        .reset_index(name="count")
    )
    # percent within actor_group+d_class
    denom = t2.groupby(["actor_group", "d_class"])["count"].transform("sum")
    t2["percent_within_actor_dclass"] = (t2["count"] / denom * 100).round(2)
    t2 = t2.sort_values(["actor_group", "d_class", "count"], ascending=[True, True, False])

    out2 = out_dir / "edu_o_scope_deontic_tables.csv"
    t2.to_csv(out2, index=False, escapechar="\\")

    # 3) actor_group × statement_type_candidate × o_scope
    t3 = (
        merged.groupby(["actor_group", "statement_type_candidate", "o_scope"])
        .size()
        .reset_index(name="count")
    )
    denom = t3.groupby(["actor_group", "statement_type_candidate"])["count"].transform("sum")
    t3["percent_within_actor_stmtType"] = (t3["count"] / denom * 100).round(2)
    t3 = t3.sort_values(["actor_group", "statement_type_candidate", "count"], ascending=[True, True, False])

    out3 = out_dir / "edu_o_scope_statement_type_tables.csv"
    t3.to_csv(out3, index=False, escapechar="\\")

    # ---------- Stratified rule samples ----------
    rng = np.random.default_rng(args.seed)
    rules = merged[merged["statement_type_candidate"] == "rule_candidate"].copy()

    strata = [
        "statement_scoped_local_o",
        "umbrella_scoped_strict",
        "umbrella_present_scope_unclear",
        "no_o_detected",
    ]

    sample_rows = []
    for s in strata:
        sub = rules[rules["o_scope"] == s].copy()
        if sub.empty:
            continue
        n = min(args.sample_n, len(sub))
        take = sub.sample(n=n, random_state=args.seed)
        take = take.assign(sample_stratum=s)
        sample_rows.append(take)

    if sample_rows:
        samp = pd.concat(sample_rows, axis=0, ignore_index=True)
        keep_cols = [
            "sample_stratum",
            "doc_id", "chunk_id", "sentence_index_in_chunk",
            "actor_group", "a_raw_text", "a_norm",
            "d_class", "d_surface", "d_polarity",
            "i_phrase_text",
            "b_text",
            "c_texts",
            "o_local_present", "o_local_type", "o_local_text",
            "o_umbrella_confidence", "o_umbrella_type", "o_umbrella_heading", "o_umbrella_text",
            "sentence_text",
        ]
        keep_cols = [c for c in keep_cols if c in samp.columns]
        samp = samp[keep_cols].copy()

        out4 = out_dir / "edu_rule_samples_by_o_scope.csv"
        samp.to_csv(out4, index=False, escapechar="\\")
    else:
        out4 = None

    # ---------- Console audit ----------
    print("[ok] wrote:", out1)
    print("[ok] wrote:", out2)
    print("[ok] wrote:", out3)
    if out4:
        print("[ok] wrote:", out4)

    # quick toplines
    print("\n[audit] EDU rows:", len(merged))
    print("[audit] O-scope distribution (%):")
    print((merged["o_scope"].value_counts(normalize=True) * 100).round(2).to_string())
    print("\n[audit] rules (rule_candidate) rows:", int((merged["statement_type_candidate"] == "rule_candidate").sum()))


if __name__ == "__main__":
    main()
