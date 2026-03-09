#!/usr/bin/env python3
"""
Step 8.5i — Light semantic contrasts for pedagogical dropout cases.

Purpose
-------
Provide an interpretable semantic comparison for pedagogical actors who are
visible in the education corpus but drop out of the retained rules-only layer.

This is intentionally light. It uses:
1. keyness-style token / n-gram contrasts
2. small frame lexicons for interpretable discourse profiles

Default inputs
--------------
- data/derived/step8_analysis/pedagogical_actor_loss_audit/
- data/derived/step8_igt_chunks_edu/igt_statements_full.parquet

Default outputs
---------------
- data/derived/step8_analysis/semantic_contrasts_pedagogical_dropout_edu/
    - semantic_frame_contrast_overall.csv
    - semantic_frame_contrast_by_group.csv
    - semantic_keyness_all_lost_vs_retained.csv
    - semantic_keyness_all_lost_vs_all_rules.csv
    - semantic_keyness_students_lost_vs_retained.csv
    - semantic_keyness_schools_lost_vs_retained.csv
    - semantic_statement_type_contrast.csv
    - semantic_summary.md
    - run_metadata.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

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
# Tokenization / keyness
# -----------------------------------------------------------------------------

TOKEN_RE = re.compile(r"[a-z][a-z\-]{2,}")

STOPWORDS = {
    "the","and","for","with","that","this","from","into","onto","over","under","within","without",
    "shall","should","must","may","might","can","could","would","will","being","been","have","has",
    "had","were","was","are","is","be","it","its","their","them","they","these","those","there",
    "such","than","then","also","more","most","some","many","much","other","others","any","all",
    "each","every","one","two","three","not","but","if","or","as","at","by","of","on","to",
    "in","an","a","we","our","your","you","his","her","hers","him","who","whom","what","when",
    "where","which","while","because","through","during","after","before","between","about","across",
    "machine","translated","google",
}


def tokenize(text: str) -> List[str]:
    toks = [m.group(0).lower() for m in TOKEN_RE.finditer(textify(text).lower())]
    return [t for t in toks if t not in STOPWORDS and len(t) >= 3]



def ngrams(tokens: Sequence[str], n: int) -> Iterable[str]:
    if len(tokens) < n:
        return []
    return (" ".join(tokens[i:i+n]) for i in range(len(tokens) - n + 1))



def count_features(texts: Sequence[str], max_ngram: int = 2) -> tuple[Counter, int]:
    counts: Counter = Counter()
    total = 0
    for txt in texts:
        toks = tokenize(txt)
        for n in range(1, max_ngram + 1):
            feats = list(ngrams(toks, n))
            counts.update(feats)
            total += len(feats)
    return counts, total



def keyness_table(
    texts_a: Sequence[str],
    texts_b: Sequence[str],
    label_a: str,
    label_b: str,
    min_count: int = 3,
    max_terms: int = 150,
    alpha: float = 0.5,
) -> pd.DataFrame:
    counts_a, total_a = count_features(texts_a)
    counts_b, total_b = count_features(texts_b)
    vocab = {t for t in counts_a.keys() | counts_b.keys() if (counts_a[t] + counts_b[t]) >= min_count}
    if not vocab or total_a == 0 or total_b == 0:
        return pd.DataFrame(columns=["comparison","feature","count_a","count_b","rate_a_per_10k","rate_b_per_10k","log2_ratio","favours"])

    V = len(vocab)
    rows: List[dict] = []
    for feat in vocab:
        ca = counts_a.get(feat, 0)
        cb = counts_b.get(feat, 0)
        pa = (ca + alpha) / (total_a + alpha * V)
        pb = (cb + alpha) / (total_b + alpha * V)
        lr = math.log2(pa / pb)
        rows.append(
            {
                "comparison": f"{label_a}_vs_{label_b}",
                "feature": feat,
                "count_a": int(ca),
                "count_b": int(cb),
                "rate_a_per_10k": round(10000.0 * ca / total_a, 3) if total_a else 0.0,
                "rate_b_per_10k": round(10000.0 * cb / total_b, 3) if total_b else 0.0,
                "log2_ratio": round(lr, 4),
                "favours": label_a if lr > 0 else label_b,
            }
        )
    df = pd.DataFrame(rows).sort_values("log2_ratio", ascending=False)
    if max_terms:
        top = df.head(max_terms)
        bottom = df.tail(max_terms)
        df = pd.concat([top, bottom], ignore_index=True)
    return df


# -----------------------------------------------------------------------------
# Frame lexicons
# -----------------------------------------------------------------------------

FRAME_PATTERNS: Dict[str, List[str]] = {
    "capacity_training": [
        r"\btraining\b", r"\btrain\b", r"\bskills?\b", r"\bliteracy\b",
        r"\bcompetenc(e|ies)\b", r"\bprofessional development\b", r"\bcurricul(ar|um)\b",
        r"\bcapacity building\b", r"\bupskilling\b", r"\bawareness\b",
    ],
    "support_implementation": [
        r"\bsupport\b", r"\bimplement(ation|ed|ing)?\b", r"\bintegrat(e|ion|ing)\b",
        r"\badoption\b", r"\bdeployment\b", r"\buse\b", r"\bguidance\b",
        r"\baccompan(y|iment)\b", r"\bassist(ance)?\b",
    ],
    "ethics_safeguards": [
        r"\bethic(al|s)?\b", r"\bresponsible\b", r"\bfair(ness)?\b", r"\bbias\b",
        r"\bprivacy\b", r"\bsafety\b", r"\bwellbeing\b", r"\bprotection\b",
        r"\bhuman rights?\b", r"\btransparen(cy|t)\b",
    ],
    "beneficiary_access_inclusion": [
        r"\binclusion\b", r"\baccess\b", r"\bequity\b", r"\bvulnerable\b",
        r"\bdisadvantaged\b", r"\bspecial needs\b", r"\bopportunit(y|ies)\b",
    ],
    "oversight_compliance": [
        r"\baudit\b", r"\bcompliance\b", r"\bauthority\b", r"\bprovider\b", r"\bdeployer\b",
        r"\brisk management\b", r"\bconformity\b", r"\bobligation(s)?\b", r"\bsupervis(ion|ory)\b",
        r"\baccountability\b",
    ],
    "assessment_admission": [
        r"\badmission\b", r"\bassess(ment)?\b", r"\bgrading\b", r"\bevaluation\b",
        r"\bscore(s|ing)?\b", r"\bproctor(ing)?\b", r"\bselection\b",
    ],
    "workforce_innovation": [
        r"\bworkforce\b", r"\blabou?r\b", r"\bjobs?\b", r"\bemployment\b",
        r"\binnovation\b", r"\bcompetitiveness\b", r"\bproductivity\b",
    ],
}
COMPILED_FRAMES = {k: [re.compile(p, flags=re.IGNORECASE) for p in pats] for k, pats in FRAME_PATTERNS.items()}



def frame_summary(df: pd.DataFrame, text_col: str, set_name: str, ped_group: str = "ALL") -> pd.DataFrame:
    rows: List[dict] = []
    texts = df[text_col].fillna("").astype(str)
    n_rows = len(df)
    for frame_name, patterns in COMPILED_FRAMES.items():
        match_lists = texts.apply(lambda s: sum(1 for rx in patterns if rx.search(s)))
        rows.append(
            {
                "set_name": set_name,
                "pedagogical_group": ped_group,
                "frame": frame_name,
                "rows_with_frame": int((match_lists > 0).sum()),
                "pct_rows_with_frame": round(100.0 * (match_lists > 0).mean(), 2) if n_rows else 0.0,
                "total_pattern_hits": int(match_lists.sum()),
                "n_rows_total": int(n_rows),
            }
        )
    return pd.DataFrame(rows)

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit-dir", default="data/derived/step8_analysis/pedagogical_actor_loss_audit")
    ap.add_argument("--igt-parquet", default="data/derived/step8_igt_chunks_edu/igt_statements_full.parquet")
    ap.add_argument("--out-dir", default="data/derived/step8_analysis/semantic_contrasts_pedagogical_dropout_edu")
    ap.add_argument("--rule-labels", nargs="*", default=["rule_candidate", "rule", "rules"])
    ap.add_argument("--min-feature-count", type=int, default=3)
    ap.add_argument("--top-k", type=int, default=80)
    args = ap.parse_args()

    t0 = time.time()
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    audit_dir = Path(args.audit_dir)
    rule_labels = {str(x).strip().lower() for x in args.rule_labels}

    lost = maybe_read_table(audit_dir / "pedagogical_examples_rules_cutoff.csv")
    retained = maybe_read_table(audit_dir / "pedagogical_examples_final_retained.csv")
    parser_miss = maybe_read_table(audit_dir / "pedagogical_examples_sentence_only_or_parser_miss.csv")

    required = {"pedagogical_group", "sentence_text", "statement_type_candidate"}
    for name, df in [("lost", lost), ("retained", retained), ("parser_miss", parser_miss)]:
        missing = sorted(list(required - set(df.columns)))
        if missing:
            raise ValueError(f"Missing expected columns in {name}: {missing}")

    igt = maybe_read_table(Path(args.igt_parquet)).copy()
    if "sentence_text" not in igt.columns or "statement_type_candidate" not in igt.columns:
        raise ValueError("IGT parquet must contain sentence_text and statement_type_candidate")
    igt["_stmt_type_norm"] = igt["statement_type_candidate"].fillna("").astype(str).str.strip().str.lower()
    all_rules = igt.loc[igt["_stmt_type_norm"].isin(rule_labels)].copy()
    if "a_class" in all_rules.columns:
        all_rules = all_rules.loc[all_rules["a_class"].fillna("").astype(str).str.strip().str.lower().eq("explicit")].copy()

    # Frame contrasts
    frame_parts = [
        frame_summary(lost, "sentence_text", "lost_rules_only_cutoff", "ALL"),
        frame_summary(retained, "sentence_text", "final_retained", "ALL"),
        frame_summary(parser_miss, "sentence_text", "sentence_only_or_parser_miss", "ALL"),
        frame_summary(all_rules, "sentence_text", "all_explicit_A_rules", "ALL"),
    ]
    for group in sorted(set(lost["pedagogical_group"].dropna().astype(str))):
        frame_parts.append(frame_summary(lost.loc[lost["pedagogical_group"].astype(str).eq(group)], "sentence_text", "lost_rules_only_cutoff", group))
        if not retained.loc[retained["pedagogical_group"].astype(str).eq(group)].empty:
            frame_parts.append(frame_summary(retained.loc[retained["pedagogical_group"].astype(str).eq(group)], "sentence_text", "final_retained", group))
    frame_long = pd.concat(frame_parts, ignore_index=True)
    frame_overall = frame_long.loc[frame_long["pedagogical_group"].eq("ALL")].copy()
    safe_to_csv(frame_overall, out_dir / "semantic_frame_contrast_overall.csv")
    safe_to_csv(frame_long.loc[~frame_long["pedagogical_group"].eq("ALL")].copy(), out_dir / "semantic_frame_contrast_by_group.csv")

    # Statement type contrast
    stmt_parts = []
    for set_name, df in [
        ("lost_rules_only_cutoff", lost),
        ("final_retained", retained),
        ("sentence_only_or_parser_miss", parser_miss),
    ]:
        part = (
            df.groupby(["pedagogical_group", "statement_type_candidate"])
            .size()
            .reset_index(name="count")
        )
        part["set_name"] = set_name
        stmt_parts.append(part)
    stmt_df = pd.concat(stmt_parts, ignore_index=True)
    safe_to_csv(stmt_df.sort_values(["set_name", "pedagogical_group", "count"], ascending=[True, True, False]), out_dir / "semantic_statement_type_contrast.csv")

    # Keyness tables
    comparisons: List[tuple[str, pd.DataFrame]] = []
    comparisons.append(
        (
            "semantic_keyness_all_lost_vs_retained.csv",
            keyness_table(
                lost["sentence_text"].tolist(),
                retained["sentence_text"].tolist(),
                "lost_pedagogical",
                "retained_pedagogical",
                min_count=args.min_feature_count,
                max_terms=args.top_k,
            ),
        )
    )
    comparisons.append(
        (
            "semantic_keyness_all_lost_vs_all_rules.csv",
            keyness_table(
                lost["sentence_text"].tolist(),
                all_rules["sentence_text"].tolist(),
                "lost_pedagogical",
                "all_explicit_A_rules",
                min_count=args.min_feature_count,
                max_terms=args.top_k,
            ),
        )
    )

    for group, fname in [
        ("students_learners", "semantic_keyness_students_lost_vs_retained.csv"),
        ("schools_institutions", "semantic_keyness_schools_lost_vs_retained.csv"),
        ("educators_teachers", "semantic_keyness_educators_lost_vs_retained.csv"),
    ]:
        g_lost = lost.loc[lost["pedagogical_group"].astype(str).eq(group)].copy()
        g_ret = retained.loc[retained["pedagogical_group"].astype(str).eq(group)].copy()
        if len(g_lost) == 0 or len(g_ret) == 0:
            df = pd.DataFrame(columns=["comparison","feature","count_a","count_b","rate_a_per_10k","rate_b_per_10k","log2_ratio","favours"])
        else:
            df = keyness_table(
                g_lost["sentence_text"].tolist(),
                g_ret["sentence_text"].tolist(),
                f"{group}_lost",
                f"{group}_retained",
                min_count=max(2, args.min_feature_count - 1),
                max_terms=args.top_k,
            )
        comparisons.append((fname, df))

    for fname, df in comparisons:
        safe_to_csv(df, out_dir / fname)

    # Summary markdown
    overall_frames_pivot = frame_overall.pivot(index="frame", columns="set_name", values="pct_rows_with_frame").reset_index()
    overall_frames_pivot = overall_frames_pivot.fillna(0.0)
    overall_frames_pivot = overall_frames_pivot.sort_values(
        by=[c for c in overall_frames_pivot.columns if c != "frame"],
        ascending=False,
    )

    top_keyness = comparisons[0][1].head(20).copy() if not comparisons[0][1].empty else pd.DataFrame()
    bottom_keyness = comparisons[0][1].tail(20).copy() if not comparisons[0][1].empty else pd.DataFrame()

    md_lines = [
        "# Light semantic contrasts for pedagogical dropout cases",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "This is an interpretive companion to the dropout audit. It does not replace the actor analysis. It helps characterize what pedagogical actors are being asked to do or be when they are present in discourse but absent from retained rule-bearing A positions.",
        "",
        "## Frame contrast (overall)",
        build_markdown_table(overall_frames_pivot.head(20)) if not overall_frames_pivot.empty else "No frame rows.",
        "",
        "## Top features favouring lost pedagogical rows",
        build_markdown_table(top_keyness) if not top_keyness.empty else "No keyness rows.",
        "",
        "## Top features favouring retained pedagogical rows",
        build_markdown_table(bottom_keyness) if not bottom_keyness.empty else "No keyness rows.",
        "",
        "Interpretation target: if lost pedagogical rows over-index on training, support, ethics, inclusion, or implementation language while retained rows or general explicit-A rules over-index on obligations, authorities, providers, and compliance language, that supports the claim that pedagogical actors are present more as supported/protected subjects than as rule-bearing governors.",
    ]
    (out_dir / "semantic_summary.md").write_text("\n".join(md_lines), encoding="utf-8")

    metadata = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "audit_dir": args.audit_dir,
            "igt_parquet": args.igt_parquet,
        },
        "counts": {
            "lost_rows": int(len(lost)),
            "retained_rows": int(len(retained)),
            "parser_miss_rows": int(len(parser_miss)),
            "all_explicit_A_rules_rows": int(len(all_rules)),
        },
        "elapsed_seconds": round(time.time() - t0, 3),
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"[ok] wrote: {out_dir}")
    print(json.dumps(metadata["counts"], indent=2))


if __name__ == "__main__":
    main()
