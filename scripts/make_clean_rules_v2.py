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
# I/O helpers
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


def truthy(x) -> bool:
    if isinstance(x, bool):
        return x
    s = textify(x).lower()
    return s in {"1", "true", "t", "yes", "y"}


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
# Schema
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

    cols = {
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
        "o_present": "o_local_present",
        "o_text": "o_local_text",
        "stmt_type": "statement_type_candidate",
    }
    if "c_count" in df.columns:
        cols["c_count"] = "c_count"
    return cols

# --------------------------------------------------
# Upstream rule-candidate filter
# --------------------------------------------------

DEFAULT_RULE_LABELS = {"rule_candidate", "rule", "rules"}


def filter_to_rule_candidates(df: pd.DataFrame, stmt_type_col: str, rule_labels: set[str]) -> pd.DataFrame:
    out = df.copy()
    out["_stmt_type_norm"] = out[stmt_type_col].astype(str).str.strip().str.lower()
    return out[out["_stmt_type_norm"].isin(rule_labels)].copy()


# --------------------------------------------------
# Heuristics
# --------------------------------------------------

STRONG_DEONTIC_RX = re.compile(
    r"\b("
    r"shall not|must not|shall|must|"
    r"required to|is required to|are required to|"
    r"prohibited from|prohibited|forbidden from|forbidden|"
    r"entitled to|subject to|only if"
    r")\b",
    re.I,
)

PERMISSION_RX = re.compile(
    r"\b("
    r"authorized to|authorised to|allowed to|permitted to|"
    r"may not|cannot|can not|"
    r"may|can|should"
    r")\b",
    re.I,
)

INSTITUTIONAL_ACTION_RX = re.compile(
    r"\b("
    r"report|reports|reporting|reported|"
    r"notify|notification|"
    r"disclos(?:e|ed|es|ure)|"
    r"publish|published|publication|"
    r"submit|submission|"
    r"register|registered|registration|"
    r"record|records|recordkeeping|"
    r"document|documentation|"
    r"monitor|monitoring|"
    r"audit|auditing|"
    r"assess|assessment|"
    r"evaluate|evaluation|"
    r"review|reviews|"
    r"adopt|adoption|"
    r"approve|approval|"
    r"consult|consultation|"
    r"require|requires|required|"
    r"maintain|maintained|"
    r"ensure|ensures|"
    r"implement|implemented|implementation"
    r")\b",
    re.I,
)

BOUNDARY_PROC_RX = re.compile(
    r"\b("
    r"eligible|eligibility|"
    r"qualif(?:y|ies|ied|ication|ications)|criteria|criterion|"
    r"admi(?:t|ts|tted|ssion|ssions)|"
    r"enrol(?:l|ls|led|ment)|enrolment|"
    r"license|licensed|licensing|licence|"
    r"certif(?:y|ied|ication|ications)|"
    r"accredit(?:ed|ation|ations)?|"
    r"applicant|applicants|"
    r"vote|voting|quorum|consensus|"
    r"committee decision|board decision|joint decision"
    r")\b",
    re.I,
)

PAYOFF_RX = re.compile(
    r"\b("
    r"fund|funding|budget|allocate|allocation|grant|grants|"
    r"subsidy|subsidies|"
    r"incentive|incentives|"
    r"penalty|penalties|fine|fines|"
    r"sanction|sanctions|"
    r"reward|rewards|"
    r"scholarship|scholarships|"
    r"liability|liable|"
    r"compensation|"
    r"fee|fees"
    r")\b",
    re.I,
)

ROLE_RX = re.compile(
    r"\b("
    r"ministry|minist(?:er|ers)|department|agency|authority|authorities|"
    r"commission|committee|council|board|office|officer|secretary|administrator|"
    r"director|regulator|controller|processor|operator|provider|coordinator|"
    r"school|schools|university|universities|college|colleges|institution|institutions|"
    r"teacher|teachers|educator|educators|student|students|learner|learners|"
    r"researcher|researchers|user|users|body|bodies|task force|working group|organisation|organization|organisations|organizations"
    r")\b",
    re.I,
)

# Strong false-positive patterns from QC
FATAL_NEGATIVE_RX = re.compile(
    r"\b("
    r"research has shown|"
    r"for example|e\.g\.|such as|"
    r"can lead to|may lead to|could lead to|"
    r"can help|may help|could help|"
    r"can support|may support|could support|"
    r"can improve|may improve|could improve|"
    r"can enhance|may enhance|could enhance|"
    r"can enable|may enable|could enable|"
    r"has potential to|"
    r"there is a need to|"
    r"it is important to"
    r")\b",
    re.I,
)

SOFT_NEGATIVE_RX = re.compile(
    r"\b("
    r"this can|these can|ai can|"
    r"can be used to|may be used to|"
    r"likely to|"
    r"question(?:s)? whether|"
    r"seems to|appear to"
    r")\b",
    re.I,
)

# Obvious junk / fragment / OCR indicators
FRAGMENT_JUNK_RX = re.compile(
    r"(\.{3,}|_{3,}|[□■▪]|�|^\W+$)",
    re.I,
)

def looks_fragment_or_ocr_noise(sentence: str) -> bool:
    s = normalize_space(sentence)
    if not s:
        return True
    if len(s) < 25:
        return True
    if FRAGMENT_JUNK_RX.search(s):
        return True
    # too few alphabetic chars
    alpha = sum(ch.isalpha() for ch in s)
    if alpha < 10:
        return True
    # odd punctuation density
    punct = sum(ch in ",;:()[]{}<>/\\|" for ch in s)
    if punct > max(12, len(s) * 0.2):
        return True
    return False


# --------------------------------------------------
# Cleaning classifier
# --------------------------------------------------

def classify_clean_rule_v2(row: pd.Series, cols: Dict[str, str]) -> Dict[str, object]:
    sentence = textify(row[cols["sentence"]])
    a_text = textify(row[cols["a_text"]])
    a_class = textify(row[cols["a_class"]]).lower()
    d_text = textify(row[cols["d_text"]])
    d_class = textify(row[cols["d_class"]]).lower()
    i_text = textify(row[cols["i_text"]])
    c_text = textify(row[cols["c_text"]])
    b_text = textify(row[cols["b_text"]])
    o_present = truthy(row[cols["o_present"]])
    o_text = textify(row[cols["o_text"]])

    c_count = 0
    if "c_count" in cols and cols["c_count"] in row.index:
        try:
            c_count = int(row[cols["c_count"]])
        except Exception:
            c_count = 0

    full_text = normalize_space(" ".join(
        part for part in [sentence, a_text, d_text, d_class, i_text, c_text, b_text, o_text] if part
    )).lower()

    positive: List[str] = []
    negative: List[str] = []
    score = 0

    fragment_noise = looks_fragment_or_ocr_noise(sentence)
    strong_deontic = bool(STRONG_DEONTIC_RX.search(full_text))
    permission_like = bool(PERMISSION_RX.search(full_text))
    institutional_action = bool(INSTITUTIONAL_ACTION_RX.search(full_text))
    boundary_proc = bool(BOUNDARY_PROC_RX.search(full_text))
    payoff = bool(PAYOFF_RX.search(full_text))
    role_actor = bool(ROLE_RX.search(a_text)) or bool(ROLE_RX.search(full_text))
    explicit_actor = (a_class == "explicit")
    has_condition = (c_count > 0) or (c_text != "")
    has_local_object = o_present or (o_text != "")
    fatal_negative = bool(FATAL_NEGATIVE_RX.search(full_text))
    soft_negative = bool(SOFT_NEGATIVE_RX.search(full_text))

    if strong_deontic:
        score += 4
        positive.append("strong_deontic")

    if d_class in {"obligation", "prohibition"}:
        score += 2
        positive.append(f"d_class:{d_class}")
    elif d_class == "permission":
        score += 1
        positive.append("d_class:permission")

    if institutional_action:
        score += 2
        positive.append("institutional_action")

    if boundary_proc:
        score += 1
        positive.append("boundary_or_procedure")

    if payoff:
        score += 1
        positive.append("payoff_signal")

    if explicit_actor:
        score += 1
        positive.append("explicit_actor")

    if role_actor:
        score += 1
        positive.append("role_actor")

    if has_condition:
        score += 1
        positive.append("has_condition")

    if has_local_object:
        score += 1
        positive.append("has_local_object")

    if fragment_noise:
        score -= 5
        negative.append("fragment_or_ocr_noise")

    # strong deontics should usually survive unless the sentence is junk
    if fatal_negative and not strong_deontic:
        score -= 4
        negative.append("fatal_negative")
    elif fatal_negative and strong_deontic:
        score -= 1
        negative.append("fatal_negative_but_overridden")

    if soft_negative and not strong_deontic:
        score -= 2
        negative.append("soft_negative")
    elif soft_negative and strong_deontic:
        score -= 1
        negative.append("soft_negative_but_overridden")

    institutional_cues = sum([
        1 if institutional_action else 0,
        1 if boundary_proc else 0,
        1 if payoff else 0,
        1 if explicit_actor else 0,
        1 if role_actor else 0,
        1 if has_condition else 0,
        1 if has_local_object else 0,
    ])

    keep = False
    reason = "drop:insufficient_rule_signals"

    if fragment_noise:
        keep = False
        reason = "drop:fragment_or_ocr_noise"
    elif strong_deontic and score >= 4:
        keep = True
        reason = "keep:strong_deontic_override"
    elif d_class in {"obligation", "prohibition"} and institutional_cues >= 2 and score >= 4:
        keep = True
        reason = "keep:obligation_or_prohibition_with_structure"
    elif institutional_action and explicit_actor and (has_local_object or has_condition) and score >= 4:
        keep = True
        reason = "keep:institutional_action_structure"
    elif permission_like and explicit_actor and institutional_action and institutional_cues >= 3 and score >= 5:
        keep = True
        reason = "keep:permission_with_clear_institutional_structure"
    elif boundary_proc and explicit_actor and score >= 4:
        keep = True
        reason = "keep:boundary_or_procedure_structure"

    return {
        "rule_clean_keep_v2": int(keep),
        "rule_clean_score_v2": score,
        "rule_clean_positive_signals_v2": "|".join(positive),
        "rule_clean_negative_signals_v2": "|".join(negative),
        "rule_clean_reason_v2": reason,
    }
# --------------------------------------------------
# QC / review samples
# --------------------------------------------------

def export_review_samples(
    annotated: pd.DataFrame,
    out_dir: Path,
    stem: str,
    keep_col: str,
    score_col: str,
    seed: int = 42,
    n_keep: int = 40,
    n_drop: int = 40,
    n_borderline: int = 40,
) -> None:
    random.seed(seed)

    keep_df = annotated[annotated[keep_col] == 1].copy()
    drop_df = annotated[annotated[keep_col] == 0].copy()

    def sample_df(df: pd.DataFrame, n: int) -> pd.DataFrame:
        if df.empty:
            return df.copy()
        n = min(n, len(df))
        return df.sample(n=n, random_state=seed)

    keep_sample = sample_df(keep_df, n_keep)

    borderline_pool = drop_df.sort_values(score_col, ascending=False).head(max(n_borderline * 3, 50)).copy()
    borderline_sample = sample_df(borderline_pool, n_borderline)

    drop_sample = sample_df(drop_df, n_drop)

    for label, df in [
        ("keep_sample", keep_sample),
        ("borderline_sample", borderline_sample),
        ("drop_sample", drop_sample),
    ]:
        out_path = out_dir / f"{stem}_{label}.csv"
        safe_to_csv(df, out_path)


# --------------------------------------------------
# Per-corpus runner
# --------------------------------------------------

def run_one_corpus(
    df: pd.DataFrame,
    corpus_name: str,
    out_dir: Path,
    stem: str,
    rule_labels: set[str],
    review_seed: int,
) -> Dict[str, object]:
    cols = detect_columns(df)
    stmt_type_col = cols["stmt_type"]

    print(f"\n[{corpus_name}] column map")
    for k, v in cols.items():
        print(f"  {k}: {v}")

    total_before = len(df)
    rule_candidates = filter_to_rule_candidates(df, stmt_type_col=stmt_type_col, rule_labels=rule_labels)
    total_rule_candidates = len(rule_candidates)

    print(f"[{corpus_name}] rule candidates kept from upstream: {total_rule_candidates:,} / {total_before:,}")

    classified_rows = [classify_clean_rule_v2(row, cols) for _, row in rule_candidates.iterrows()]
    class_df = pd.DataFrame(classified_rows)

    annotated = pd.concat([rule_candidates.reset_index(drop=True), class_df], axis=1)
    clean_kept = annotated[annotated["rule_clean_keep_v2"] == 1].copy()

    annotated_parquet = out_dir / f"{stem}_rule_candidates_annotated_cleaning_v2.parquet"
    annotated_csv = out_dir / f"{stem}_rule_candidates_annotated_cleaning_v2.csv"
    clean_parquet = out_dir / f"{stem}_rules_clean_v2.parquet"
    clean_csv = out_dir / f"{stem}_rules_clean_v2.csv"

    annotated.to_parquet(annotated_parquet, index=False)
    safe_to_csv(annotated, annotated_csv)
    clean_kept.to_parquet(clean_parquet, index=False)
    safe_to_csv(clean_kept, clean_csv)

    export_review_samples(
        annotated=annotated,
        out_dir=out_dir,
        stem=stem,
        keep_col="rule_clean_keep_v2",
        score_col="rule_clean_score_v2",
        seed=review_seed,
    )

    print(f"Saved → {annotated_parquet}")
    print(f"Saved → {annotated_csv}")
    print(f"Saved → {clean_parquet}")
    print(f"Saved → {clean_csv}")

    return {
        "corpus": corpus_name,
        "total_statements_before_filter": total_before,
        "total_rule_candidates_from_upstream": total_rule_candidates,
        "total_rules_clean_v2": int(len(clean_kept)),
        "pct_rule_candidates_kept_clean_v2": round(100.0 * len(clean_kept) / total_rule_candidates, 2) if total_rule_candidates else 0.0,
        "pct_of_all_statements_kept_clean_v2": round(100.0 * len(clean_kept) / total_before, 2) if total_before else 0.0,
        "mean_rule_clean_score_v2": round(float(annotated["rule_clean_score_v2"].mean()), 2) if len(annotated) else 0.0,
    }


# --------------------------------------------------
# Main
# --------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full-igt", default="data/derived/step8_igt_full/igt_statements_full.parquet")
    ap.add_argument("--edu-igt", default="data/derived/step8_igt_chunks_edu/igt_statements_full.parquet")
    ap.add_argument("--title-igt", default="data/derived/step8_igt_title_edu/igt_statements_full.parquet")
    ap.add_argument("--out-dir", default="data/derived/step8_analysis/rules_clean_v2")
    ap.add_argument(
        "--rule-labels",
        default="rule_candidate,rule,rules",
        help="Comma-separated statement-type labels to treat as upstream rule candidates",
    )
    ap.add_argument("--review-seed", type=int, default=42)
    args = ap.parse_args()

    rule_labels = {x.strip().lower() for x in args.rule_labels.split(",") if x.strip()}
    if not rule_labels:
        rule_labels = set(DEFAULT_RULE_LABELS)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    corpora: List[Tuple[str, Path, str]] = [
        ("Full corpus", Path(args.full_igt), "full_corpus"),
        ("Education-relevant", Path(args.edu_igt), "education_relevant"),
        ("Education-in-title", Path(args.title_igt), "education_in_title"),
    ]

    summary_rows: List[Dict[str, object]] = []

    for corpus_name, in_path, stem in corpora:
        df = maybe_read_table(in_path)
        row = run_one_corpus(
            df=df,
            corpus_name=corpus_name,
            out_dir=out_dir,
            stem=stem,
            rule_labels=rule_labels,
            review_seed=args.review_seed,
        )
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)

    summary_csv = out_dir.parent / "rules_clean_v2_summary_by_corpus.csv"
    summary_md = out_dir.parent / "rules_clean_v2_summary_by_corpus.md"

    safe_to_csv(summary, summary_csv)
    summary_md.write_text(build_markdown_table(summary), encoding="utf-8")

    print("\nSaved summaries:")
    print(f"  {summary_csv}")
    print(f"  {summary_md}")
    print()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
