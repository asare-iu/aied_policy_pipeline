#!/usr/bin/env python3
"""
Step 8.5b — Stakeholder × rule type analysis (education-relevant rules only)

Purpose
-------
Move from stakeholder presence to stakeholder institutional positioning.
Using the education-relevant IGT statements, this script restricts the analysis
to rules only, assigns transparent stakeholder groups from the A component,
and cross-tabulates those stakeholder groups against IAD rule types.

This is designed to answer:
"When education-relevant policy becomes rule-like, what kinds of rules attach
 to different stakeholders, and do pedagogical actors receive meaningful
 boundary / information / aggregation access?"

Inputs 
------
Because at this point I do not exactly remember every pathway. This script is built with two options. 

Option A (preferred if it already exists):
- data/derived/step8_analysis/iad_rule_types_rules_only/
    education_relevant_iad_rule_types_rules_only.parquet

Option B (fallback):
- data/derived/step8_igt_chunks_edu/igt_statements_full.parquet
  In this case the script will filter to rules only and generate IAD rule-type
  columns internally using the same logic as the current rules-only classifier.

Outputs
-------
Directory (default):
- data/derived/step8_analysis/stakeholder_x_rule_type_edu_rules/

Files:
- run_metadata.json
- input_filter_summary.csv
- stakeholder_x_primary_rule_type_long.csv
- stakeholder_x_primary_rule_type_wide_counts.csv
- stakeholder_x_primary_rule_type_wide_pct_within_actor.csv
- stakeholder_x_binary_rule_hits_long.csv
- stakeholder_x_binary_rule_hits_wide_pct.csv
- pedagogical_contrast_rule_types.csv
- stakeholder_rule_type_exemplars.csv
- stakeholder_group_counts.csv
- stakeholder_x_primary_rule_type_long.md
- stakeholder_x_binary_rule_hits_long.md
- pedagogical_contrast_rule_types.md
- stakeholder_x_primary_rule_type_heatmap.png
- stakeholder_x_binary_rule_hits_heatmap.png

Method defaults
---------------
- education-relevant corpus only
- rules only
- explicit A only (default)
- stakeholder groups assigned from a_raw_text / A component only
- all seven IAD rule types retained from earlier analysis, but outputs include a pedagogical-contrast
  table emphasizing boundary / information / aggregation / choice
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Small helpers
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
# Column detection
# -----------------------------------------------------------------------------

EXPECTED_IGT_COLUMNS = {
    "doc_id",
    "chunk_id",
    "sentence_text",
    "a_raw_text",
    "a_class",
    "statement_type_candidate",
}

RULE_HIT_COLUMNS = [
    "iad_boundary",
    "iad_position",
    "iad_choice",
    "iad_information",
    "iad_scope",
    "iad_aggregation",
    "iad_payoff",
]

PRIMARY_COL = "iad_rule_type_primary"
HITS_COL = "iad_rule_type_hits"
N_HITS_COL = "iad_rule_type_n_hits"



def validate_minimum_schema(df: pd.DataFrame) -> None:
    missing = sorted(list(EXPECTED_IGT_COLUMNS - set(df.columns)))
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

# -----------------------------------------------------------------------------
# Rules-only filter
# -----------------------------------------------------------------------------

DEFAULT_RULE_LABELS = {"rule_candidate", "rule", "rules"}



def filter_to_rules_only(df: pd.DataFrame, stmt_type_col: str, rule_labels: set[str]) -> pd.DataFrame:
    out = df.copy()
    out["_stmt_type_norm"] = out[stmt_type_col].astype(str).str.strip().str.lower()
    return out[out["_stmt_type_norm"].isin(rule_labels)].copy()


# -----------------------------------------------------------------------------
# Stakeholder grouping (analysis-only, transparent)
# -----------------------------------------------------------------------------

ACTOR_GROUP_PATTERNS: Dict[str, List[str]] = {
    # pedagogical actors
    "educators_teachers": [
        r"\beducator(s)?\b",
        r"\bteacher(s)?\b",
        r"\binstructor(s)?\b",
        r"\blecturer(s)?\b",
        r"\btrainer(s)?\b",
        r"\bfaculty\b",
        r"\bprofessor(s)?\b",
        r"\bteaching staff\b",
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
    ],
    # governance actors
    "policy_makers_ministries": [
        r"\bministry\b",
        r"\bministries\b",
        r"\bminister(s)?\b",
        r"\bgovernment\b",
        r"\bpublic authority\b",
        r"\bpublic authorities\b",
        r"\bpolicy maker(s)?\b",
        r"\bcompetent authority\b",
        r"\bcompetent authorities\b",
        r"\bdepartment of education\b",
        r"\beducation department\b",
        r"\bnational authority\b",
        r"\bstate\b",
    ],
    "regulators_supervisors": [
        r"\bregulator(s)?\b",
        r"\bsupervisory authority\b",
        r"\bsupervisory authorities\b",
        r"\boversight body\b",
        r"\boversight bodies\b",
        r"\benforcement authority\b",
        r"\bdata protection authority\b",
        r"\binspectorate\b",
        r"\bcommission\b",
        r"\bagency\b",
        r"\bboard\b",
    ],
    # socio-technical actors
    "platforms_systems": [
        r"\bplatform(s)?\b",
        r"\bsystem(s)?\b",
        r"\btool(s)?\b",
        r"\bmodel(s)?\b",
        r"\balgorithm(s)?\b",
        r"\bai system(s)?\b",
        r"\bai model(s)?\b",
    ],
    "deployers_users": [
        r"\bdeployer(s)?\b",
        r"\boperator(s)?\b",
        r"\buser(s)?\b",
        r"\bimplementer(s)?\b",
        r"\badopter(s)?\b",
        r"\badministrator(s)?\b",
    ],
    "commercial_designers_vendors": [
        r"\bvendor(s)?\b",
        r"\bsupplier(s)?\b",
        r"\bdeveloper(s)?\b",
        r"\bmanufacturer(s)?\b",
        r"\bservice provider(s)?\b",
        r"\bprovider(s)?\b",
        r"\bcompany\b",
        r"\bcompanies\b",
        r"\bfirm(s)?\b",
        r"\bcommercial\b",
    ],
    "researchers": [
        r"\bresearcher(s)?\b",
        r"\bacademic(s)?\b",
        r"\bscientist(s)?\b",
        r"\bresearch community\b",
    ],
    "data_subjects": [
        r"\bdata subject(s)?\b",
        r"\bindividual(s)?\b",
        r"\bperson(s)?\b",
        r"\bcitizen(s)?\b",
        r"\blearner data subject(s)?\b",
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

PEDAGOGICAL_CONTRAST_ACTORS = [
    "educators_teachers",
    "students_learners",
    "schools_institutions",
    "policy_makers_ministries",
    "regulators_supervisors",
    "platforms_systems",
    "commercial_designers_vendors",
]

PEDAGOGICAL_CONTRAST_RULE_TYPES = [
    "boundary",
    "information",
    "aggregation",
    "choice",
]



def assign_actor_group(a_norm: str) -> str:
    s = textify(a_norm).lower()
    if not s:
        return "unclassified"
    for grp in ACTOR_GROUP_ORDER:
        for pat in ACTOR_GROUP_PATTERNS[grp]:
            if re.search(pat, s, flags=re.IGNORECASE):
                return grp
    return "unclassified"


# -----------------------------------------------------------------------------
# IAD rule typing 
# -----------------------------------------------------------------------------

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

PRIMARY_PRIORITY = [
    "boundary",
    "aggregation",
    "payoff",
    "information",
    "position",
    "scope",
    "choice",
]



def classify_iad_rule_type_from_row(row: pd.Series) -> Dict[str, object]:
    sentence = textify(row.get("sentence_text"))
    a_text = textify(row.get("a_raw_text"))
    d_text = textify(row.get("d_surface"))
    d_class = textify(row.get("d_class"))
    i_text = textify(row.get("i_phrase_text"))
    c_text = textify(row.get("c_texts"))
    b_text = textify(row.get("b_text"))
    o_text = textify(row.get("o_local_text"))

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



def ensure_rule_type_columns(df: pd.DataFrame) -> pd.DataFrame:
    has_all = set(RULE_HIT_COLUMNS + [PRIMARY_COL, HITS_COL, N_HITS_COL]).issubset(df.columns)
    if has_all:
        out = df.copy()
        for c in RULE_HIT_COLUMNS:
            out[c] = out[c].fillna(0).astype(int)
        out[N_HITS_COL] = out[N_HITS_COL].fillna(0).astype(int)
        out[PRIMARY_COL] = out[PRIMARY_COL].fillna("unclassified").astype(str)
        out[HITS_COL] = out[HITS_COL].fillna("").astype(str)
        return out

    typed = pd.DataFrame([classify_iad_rule_type_from_row(row) for _, row in df.iterrows()])
    out = pd.concat([df.reset_index(drop=True), typed], axis=1)
    return out
# -----------------------------------------------------------------------------
# Summaries
# -----------------------------------------------------------------------------


def summarize_primary(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    total_rules = len(df)
    long = (
        df.groupby(["actor_group", PRIMARY_COL], dropna=False)
        .size()
        .reset_index(name="count")
    )

    actor_totals = (
        df.groupby("actor_group")
        .size()
        .reset_index(name="n_actor_rules")
    )

    long = long.merge(actor_totals, on="actor_group", how="left")
    long["percent_within_actor"] = (100.0 * long["count"] / long["n_actor_rules"]).round(2)
    long["percent_of_all_rules"] = (100.0 * long["count"] / max(total_rules, 1)).round(2)
    long = long.sort_values(["actor_group", "count", PRIMARY_COL], ascending=[True, False, True]).reset_index(drop=True)

    wide_counts = (
        long.pivot(index="actor_group", columns=PRIMARY_COL, values="count")
        .fillna(0)
        .astype(int)
        .reset_index()
    )

    wide_pct = (
        long.pivot(index="actor_group", columns=PRIMARY_COL, values="percent_within_actor")
        .fillna(0.0)
        .reset_index()
    )

    return long, wide_counts, wide_pct



def summarize_binary_hits(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for actor_group, gdf in df.groupby("actor_group"):
        denom = len(gdf)
        for c in RULE_HIT_COLUMNS:
            rule_type = c.replace("iad_", "")
            n = int(gdf[c].fillna(0).astype(int).sum())
            rows.append(
                {
                    "actor_group": actor_group,
                    "rule_type": rule_type,
                    "n_statements_with_hit": n,
                    "n_actor_rules": denom,
                    "pct_actor_rules_with_hit": round(100.0 * n / max(denom, 1), 2),
                }
            )

    long = pd.DataFrame(rows).sort_values(["actor_group", "rule_type"]).reset_index(drop=True)
    wide = (
        long.pivot(index="actor_group", columns="rule_type", values="pct_actor_rules_with_hit")
        .fillna(0.0)
        .reset_index()
    )
    return long, wide



def build_pedagogical_contrast(binary_long: pd.DataFrame) -> pd.DataFrame:
    out = binary_long[
        binary_long["actor_group"].isin(PEDAGOGICAL_CONTRAST_ACTORS)
        & binary_long["rule_type"].isin(PEDAGOGICAL_CONTRAST_RULE_TYPES)
    ].copy()
    out = out.sort_values(["actor_group", "rule_type"]).reset_index(drop=True)
    return out



def summarize_actor_groups(df: pd.DataFrame) -> pd.DataFrame:
    out = (
        df.groupby("actor_group")
        .size()
        .reset_index(name="n_rules")
        .sort_values(["n_rules", "actor_group"], ascending=[False, True])
        .reset_index(drop=True)
    )
    out["pct_of_retained_rules"] = (100.0 * out["n_rules"] / max(len(df), 1)).round(2)
    return out


# -----------------------------------------------------------------------------
# Exemplars
# -----------------------------------------------------------------------------

EXEMPLAR_KEEP = [
    "doc_id",
    "chunk_id",
    "sentence_index_in_chunk",
    "actor_group",
    "a_class",
    "a_raw_text",
    "a_norm",
    "statement_type_candidate",
    PRIMARY_COL,
    HITS_COL,
    N_HITS_COL,
    "iad_boundary",
    "iad_position",
    "iad_choice",
    "iad_information",
    "iad_scope",
    "iad_aggregation",
    "iad_payoff",
    "d_surface",
    "d_class",
    "i_phrase_text",
    "b_text",
    "c_texts",
    "o_local_text",
    "sentence_text",
]



def build_exemplars(
    df: pd.DataFrame,
    top_exemplars: int,
    seed: int,
    sample_mode: str,
) -> pd.DataFrame:
    keep_cols = [c for c in EXEMPLAR_KEEP if c in df.columns]
    parts: List[pd.DataFrame] = []

    for actor_group, gdf in df.groupby("actor_group"):
        for primary_type, sdf in gdf.groupby(PRIMARY_COL):
            if sdf.empty:
                continue
            n = min(top_exemplars, len(sdf))
            if sample_mode == "random":
                out = sdf.sample(n=n, random_state=seed)
            else:
                sort_cols = [c for c in ["doc_id", "chunk_id", "sentence_index_in_chunk"] if c in sdf.columns]
                out = sdf.sort_values(sort_cols).head(n) if sort_cols else sdf.head(n)
            out = out[keep_cols].copy()
            out.insert(0, "exemplar_bucket", f"{actor_group}__{primary_type}")
            parts.append(out)

    if not parts:
        return pd.DataFrame(columns=["exemplar_bucket"] + keep_cols)
    return pd.concat(parts, ignore_index=True)


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------


def _prepare_heatmap_matrix(df_wide: pd.DataFrame, row_order: Optional[Sequence[str]] = None) -> Tuple[List[str], List[str], np.ndarray]:
    temp = df_wide.copy()
    if "actor_group" not in temp.columns:
        raise ValueError("Expected actor_group column in wide dataframe")

    rows = temp["actor_group"].astype(str).tolist()
    cols = [c for c in temp.columns if c != "actor_group"]

    if row_order is not None:
        ordered_rows = [r for r in row_order if r in rows] + [r for r in rows if r not in row_order]
        temp["_order"] = pd.Categorical(temp["actor_group"], categories=ordered_rows, ordered=True)
        temp = temp.sort_values("_order").drop(columns=["_order"])
        rows = temp["actor_group"].astype(str).tolist()

    mat = temp[cols].to_numpy(dtype=float)
    return rows, cols, mat



def save_heatmap(df_wide: pd.DataFrame, out_png: Path, title: str, value_fmt: str = ".1f") -> None:
    rows, cols, mat = _prepare_heatmap_matrix(df_wide, row_order=ACTOR_GROUP_ORDER)

    if mat.size == 0:
        return

    height = max(4, 0.45 * len(rows) + 1.5)
    width = max(7, 0.9 * len(cols) + 2.5)

    fig, ax = plt.subplots(figsize=(width, height))
    im = ax.imshow(mat, aspect="auto")
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels(rows)
    ax.set_title(title)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, format(mat[i, j], value_fmt), ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--typed-rules-parquet",
        default="data/derived/step8_analysis/iad_rule_types_rules_only/education_relevant_iad_rule_types_rules_only.parquet",
        help="Preferred pre-typed rules-only parquet. If missing, script falls back to --edu-igt.",
    )
    ap.add_argument(
        "--edu-igt",
        default="data/derived/step8_igt_chunks_edu/igt_statements_full.parquet",
        help="Education-relevant IGT parquet used if typed rules parquet is absent.",
    )
    ap.add_argument(
        "--out-dir",
        default="data/derived/step8_analysis/stakeholder_x_rule_type_edu_rules",
        help="Output directory",
    )
    ap.add_argument(
        "--rule-labels",
        default="rule_candidate,rule,rules",
        help="Comma-separated statement-type labels treated as rules when falling back to --edu-igt.",
    )
    ap.add_argument(
        "--explicit-a-only",
        action="store_true",
        default=True,
        help="Keep only rows where a_class == explicit (default: on).",
    )
    ap.add_argument(
        "--allow-nonexplicit-a",
        action="store_true",
        help="Disable the explicit-A-only restriction and keep all rows with non-empty A.",
    )
    ap.add_argument(
        "--include-unclassified-actors",
        action="store_true",
        help="Retain actor_group == unclassified in output tables.",
    )
    ap.add_argument("--top-exemplars", type=int, default=15, help="Examples per actor_group × primary_rule_type cell")
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

    typed_rules_path = Path(args.typed_rules_parquet)
    edu_igt_path = Path(args.edu_igt)

    # -------------------- Load --------------------
    source_mode = "typed_rules_parquet" if typed_rules_path.exists() else "edu_igt_fallback"
    in_path = typed_rules_path if typed_rules_path.exists() else edu_igt_path
    df = maybe_read_table(in_path)
    validate_minimum_schema(df)

    summary_rows = []
    summary_rows.append({"stage": "loaded", "n_rows": len(df), "source_mode": source_mode, "input_path": str(in_path)})

    # If we loaded raw edu IGT, filter to rules-only and add rule-type columns.
    if source_mode == "edu_igt_fallback":
        df = filter_to_rules_only(df, stmt_type_col="statement_type_candidate", rule_labels=rule_labels)
        summary_rows.append({"stage": "rules_only_filter", "n_rows": len(df), "source_mode": source_mode, "input_path": str(in_path)})
        df = ensure_rule_type_columns(df)
    else:
        # typed rules parquet may still contain rows that are not exactly rules-only if upstream changed,
        # so enforce the same filter defensively.
        df = filter_to_rules_only(df, stmt_type_col="statement_type_candidate", rule_labels=rule_labels)
        summary_rows.append({"stage": "rules_only_filter_defensive", "n_rows": len(df), "source_mode": source_mode, "input_path": str(in_path)})
        df = ensure_rule_type_columns(df)

    # -------------------- A filtering / actor groups --------------------
    if args.allow_nonexplicit_a:
        df = df[df["a_raw_text"].fillna("").astype(str).str.strip().ne("")].copy()
        summary_rows.append({"stage": "nonempty_A_only", "n_rows": len(df), "source_mode": source_mode, "input_path": str(in_path)})
    else:
        df = df[df["a_class"].astype(str).str.lower().eq("explicit")].copy()
        summary_rows.append({"stage": "explicit_A_only", "n_rows": len(df), "source_mode": source_mode, "input_path": str(in_path)})

    df["a_norm"] = normalize_a_text(df["a_raw_text"])
    df["actor_group"] = df["a_norm"].apply(assign_actor_group)
    summary_rows.append({"stage": "actor_group_assigned", "n_rows": len(df), "source_mode": source_mode, "input_path": str(in_path)})

    if not args.include_unclassified_actors:
        df = df[df["actor_group"].ne("unclassified")].copy()
        summary_rows.append({"stage": "drop_unclassified_actor_groups", "n_rows": len(df), "source_mode": source_mode, "input_path": str(in_path)})

    if df.empty:
        raise SystemExit("No rows remain after rules / A / actor-group filters.")

    # -------------------- Tables --------------------
    actor_counts = summarize_actor_groups(df)
    primary_long, primary_wide_counts, primary_wide_pct = summarize_primary(df)
    binary_long, binary_wide_pct = summarize_binary_hits(df)
    pedagogical_contrast = build_pedagogical_contrast(binary_long)
    exemplars = build_exemplars(df, top_exemplars=args.top_exemplars, seed=args.seed, sample_mode=args.sample_mode)
    summary_df = pd.DataFrame(summary_rows)

    # -------------------- Save --------------------
    safe_to_csv(summary_df, out_dir / "input_filter_summary.csv")
    safe_to_csv(actor_counts, out_dir / "stakeholder_group_counts.csv")
    safe_to_csv(primary_long, out_dir / "stakeholder_x_primary_rule_type_long.csv")
    safe_to_csv(primary_wide_counts, out_dir / "stakeholder_x_primary_rule_type_wide_counts.csv")
    safe_to_csv(primary_wide_pct, out_dir / "stakeholder_x_primary_rule_type_wide_pct_within_actor.csv")
    safe_to_csv(binary_long, out_dir / "stakeholder_x_binary_rule_hits_long.csv")
    safe_to_csv(binary_wide_pct, out_dir / "stakeholder_x_binary_rule_hits_wide_pct.csv")
    safe_to_csv(pedagogical_contrast, out_dir / "pedagogical_contrast_rule_types.csv")
    safe_to_csv(exemplars, out_dir / "stakeholder_rule_type_exemplars.csv")

    (out_dir / "stakeholder_x_primary_rule_type_long.md").write_text(build_markdown_table(primary_long), encoding="utf-8")
    (out_dir / "stakeholder_x_binary_rule_hits_long.md").write_text(build_markdown_table(binary_long), encoding="utf-8")
    (out_dir / "pedagogical_contrast_rule_types.md").write_text(build_markdown_table(pedagogical_contrast), encoding="utf-8")

    save_heatmap(
        primary_wide_pct,
        out_dir / "stakeholder_x_primary_rule_type_heatmap.png",
        title="Stakeholder × primary IAD rule type (% within actor; edu-relevant rules only)",
    )
    save_heatmap(
        binary_wide_pct,
        out_dir / "stakeholder_x_binary_rule_hits_heatmap.png",
        title="Stakeholder × IAD binary rule-type hits (% of actor rules; edu-relevant rules only)",
    )

    metadata = {
        "script": "step8_5b_stakeholder_x_rule_type_edu_rules.py",
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "source_mode": source_mode,
        "input_path": str(in_path),
        "typed_rules_parquet_exists": typed_rules_path.exists(),
        "edu_igt_exists": edu_igt_path.exists(),
        "rule_labels": sorted(rule_labels),
        "explicit_a_only": not args.allow_nonexplicit_a,
        "include_unclassified_actors": bool(args.include_unclassified_actors),
        "n_final_rows": int(len(df)),
        "actor_groups_present": sorted(df["actor_group"].dropna().astype(str).unique().tolist()),
        "top_exemplars": int(args.top_exemplars),
        "sample_mode": args.sample_mode,
        "seed": int(args.seed),
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("Saved:")
    for p in [
        out_dir / "run_metadata.json",
        out_dir / "input_filter_summary.csv",
        out_dir / "stakeholder_group_counts.csv",
        out_dir / "stakeholder_x_primary_rule_type_long.csv",
        out_dir / "stakeholder_x_primary_rule_type_wide_counts.csv",
        out_dir / "stakeholder_x_primary_rule_type_wide_pct_within_actor.csv",
        out_dir / "stakeholder_x_binary_rule_hits_long.csv",
        out_dir / "stakeholder_x_binary_rule_hits_wide_pct.csv",
        out_dir / "pedagogical_contrast_rule_types.csv",
        out_dir / "stakeholder_rule_type_exemplars.csv",
        out_dir / "stakeholder_x_primary_rule_type_long.md",
        out_dir / "stakeholder_x_binary_rule_hits_long.md",
        out_dir / "pedagogical_contrast_rule_types.md",
        out_dir / "stakeholder_x_primary_rule_type_heatmap.png",
        out_dir / "stakeholder_x_binary_rule_hits_heatmap.png",
    ]:
        print(f"  {p}")

    print("\nTop actor groups:")
    print(actor_counts.head(15).to_string(index=False))

    print("\nPedagogical contrast:")
    print(pedagogical_contrast.to_string(index=False))


if __name__ == "__main__":
    main()

