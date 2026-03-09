#!/usr/bin/env python3
"""
Step 8.5e — Plain lexical audit for pedagogical actors in the education corpus.

Purpose
-------
Run an intentionally simple word/phrase search for pedagogical actors so I can compare raw lexical presence against parser-based and rules-based
outputs.

This is a diagnostic script, not the final institutional method. It helps answer:
- Are teacher/educator terms actually common in the education corpus?
- Do they remain common in education-relevant institutional statements?
- Do they remain common in rules-only statements?
- Are they common in sentence text but rare in A-fields?

Default inputs
--------------
- data/derived/step6_chunks_edu/chunks_edu.jsonl
- data/derived/step8_igt_chunks_edu/igt_statements_full.parquet

Default outputs
---------------
- data/derived/step8_analysis/lexical_pedagogical_audit_edu/
    - lexical_group_stage_summary.csv
    - lexical_term_stage_summary.csv
    - lexical_group_stage_pivot.csv
    - lexical_hit_examples.csv
    - lexical_a_vs_sentence_gap.csv
    - lexical_summary.md
    - run_metadata.json
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence

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
    df.to_csv(path, index=False, encoding="utf-8", quoting=csv.QUOTE_MINIMAL, escapechar="\\")



def textify(x) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x)



def build_markdown_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, row in df.iterrows():
        vals = [str(row[c]) for c in cols]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)



def iter_chunks(path: Path) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            chunk_text = (
                obj.get("chunk_text")
                or obj.get("text")
                or obj.get("text_normalized")
                or obj.get("chunk")
                or obj.get("content")
                or ""
            )
            rows.append(
                {
                    "doc_id": obj.get("doc_id") or obj.get("source_doc") or obj.get("document_id") or "",
                    "chunk_id": obj.get("chunk_id") or obj.get("chunk_uid") or obj.get("id") or str(i),
                    "text": str(chunk_text),
                }
            )
    return pd.DataFrame(rows)



def normalize_stmt_type(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().str.lower()

# -----------------------------------------------------------------------------
# Lexical specs
# -----------------------------------------------------------------------------

TERM_GROUPS: Dict[str, Dict[str, str]] = {
    "educators_teachers": {
        "teacher": r"\bteacher(s)?\b",
        "educator": r"\beducator(s)?\b",
        "faculty": r"\bfaculty\b",
        "instructor": r"\binstructor(s)?\b",
        "professor": r"\bprofessor(s)?\b",
        "lecturer": r"\blecturer(s)?\b",
        "teaching_staff": r"\bteaching staff\b",
    },
    "students_learners": {
        "student": r"\bstudent(s)?\b",
        "learner": r"\blearner(s)?\b",
        "pupil": r"\bpupil(s)?\b",
        "child": r"\bchild(ren)?\b",
        "minor": r"\bminor(s)?\b",
    },
    "schools_institutions": {
        "school": r"\bschool(s)?\b",
        "university": r"\buniversity\b|\buniversities\b",
        "college": r"\bcollege(s)?\b",
        "institution": r"\binstitution(s)?\b",
        "classroom": r"\bclassroom(s)?\b",
        "campus": r"\bcampus\b|\bcampuses\b",
    },
}

DEFAULT_RULE_LABELS = {"rule_candidate", "rule", "rules"}


# -----------------------------------------------------------------------------
# Main logic
# -----------------------------------------------------------------------------


def summarize_stage(df: pd.DataFrame, text_col: str, stage_name: str, term_groups: Dict[str, Dict[str, str]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    group_rows: List[dict] = []
    term_rows: List[dict] = []
    examples: List[dict] = []

    n_rows = len(df)
    texts = df[text_col].fillna("").astype(str)

    for group_name, term_spec in term_groups.items():
        group_row_mask = pd.Series(False, index=df.index)
        total_group_matches = 0
        for term_name, pattern in term_spec.items():
            rx = re.compile(pattern, flags=re.IGNORECASE)
            match_lists = texts.apply(lambda s: rx.findall(s))
            rows_with_term = match_lists.apply(lambda xs: len(xs) > 0)
            total_matches = int(match_lists.apply(len).sum())
            total_group_matches += total_matches
            group_row_mask = group_row_mask | rows_with_term

            term_rows.append(
                {
                    "stage": stage_name,
                    "group": group_name,
                    "term": term_name,
                    "rows_with_term": int(rows_with_term.sum()),
                    "pct_rows_with_term": round(100.0 * rows_with_term.mean(), 2) if n_rows else 0.0,
                    "total_matches": total_matches,
                    "n_rows_total": n_rows,
                }
            )

            if rows_with_term.any():
                ex_df = df.loc[rows_with_term, [c for c in ["doc_id", "chunk_id", text_col] if c in df.columns]].copy().head(3)
                for _, r in ex_df.iterrows():
                    examples.append(
                        {
                            "stage": stage_name,
                            "group": group_name,
                            "term": term_name,
                            "doc_id": r.get("doc_id", ""),
                            "chunk_id": r.get("chunk_id", ""),
                            "text_excerpt": textify(r.get(text_col, ""))[:500],
                        }
                    )

        group_rows.append(
            {
                "stage": stage_name,
                "group": group_name,
                "rows_with_any_group_term": int(group_row_mask.sum()),
                "pct_rows_with_any_group_term": round(100.0 * group_row_mask.mean(), 2) if n_rows else 0.0,
                "total_group_matches": int(total_group_matches),
                "n_rows_total": n_rows,
            }
        )

    return pd.DataFrame(group_rows), pd.DataFrame(term_rows), pd.DataFrame(examples)



def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks-jsonl", default="data/derived/step6_chunks_edu/chunks_edu.jsonl")
    ap.add_argument("--igt-parquet", default="data/derived/step8_igt_chunks_edu/igt_statements_full.parquet")
    ap.add_argument("--out-dir", default="data/derived/step8_analysis/lexical_pedagogical_audit_edu")
    ap.add_argument("--rule-labels", nargs="*", default=sorted(DEFAULT_RULE_LABELS))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    rule_labels = {str(x).strip().lower() for x in args.rule_labels}

    chunks = iter_chunks(Path(args.chunks_jsonl))
    igt = maybe_read_table(Path(args.igt_parquet))

    required = {"sentence_text", "a_raw_text", "a_class", "statement_type_candidate"}
    missing = sorted(list(required - set(igt.columns)))
    if missing:
        raise ValueError(f"Missing expected IGT columns: {missing}")

    igt = igt.copy()
    igt["_stmt_type_norm"] = normalize_stmt_type(igt["statement_type_candidate"])
    igt["_is_rule"] = igt["_stmt_type_norm"].isin(rule_labels)
    igt["_a_explicit"] = igt["a_class"].fillna("").astype(str).str.strip().str.lower().eq("explicit")

    stages = [
        ("education_chunks_text", chunks.rename(columns={"text": "stage_text"}), "stage_text"),
        ("igt_sentence_text", igt.rename(columns={"sentence_text": "stage_text"}), "stage_text"),
        ("igt_a_text_any", igt.rename(columns={"a_raw_text": "stage_text"}), "stage_text"),
        (
            "igt_a_text_explicit",
            igt.loc[igt["_a_explicit"]].rename(columns={"a_raw_text": "stage_text"}),
            "stage_text",
        ),
        (
            "igt_rules_sentence_text",
            igt.loc[igt["_is_rule"]].rename(columns={"sentence_text": "stage_text"}),
            "stage_text",
        ),
        (
            "igt_rules_a_text_explicit",
            igt.loc[igt["_is_rule"] & igt["_a_explicit"]].rename(columns={"a_raw_text": "stage_text"}),
            "stage_text",
        ),
    ]

    group_frames = []
    term_frames = []
    example_frames = []
    for stage_name, stage_df, text_col in stages:
        gdf, tdf, edf = summarize_stage(stage_df, text_col, stage_name, TERM_GROUPS)
        group_frames.append(gdf)
        term_frames.append(tdf)
        example_frames.append(edf)

    group_summary = pd.concat(group_frames, ignore_index=True)
    term_summary = pd.concat(term_frames, ignore_index=True)
    examples = pd.concat(example_frames, ignore_index=True).drop_duplicates()

    pivot = group_summary.pivot(index="group", columns="stage", values="pct_rows_with_any_group_term").fillna(0.0).reset_index()

    # Sentence vs A gap inside IGT
    sentence_stage = group_summary[group_summary["stage"] == "igt_sentence_text"][["group", "rows_with_any_group_term", "pct_rows_with_any_group_term"]].rename(
        columns={
            "rows_with_any_group_term": "sentence_rows_with_group_terms",
            "pct_rows_with_any_group_term": "pct_sentence_rows_with_group_terms",
        }
    )
    a_stage = group_summary[group_summary["stage"] == "igt_a_text_any"][["group", "rows_with_any_group_term", "pct_rows_with_any_group_term"]].rename(
        columns={
            "rows_with_any_group_term": "a_rows_with_group_terms",
            "pct_rows_with_any_group_term": "pct_a_rows_with_group_terms",
        }
    )
    rules_stage = group_summary[group_summary["stage"] == "igt_rules_sentence_text"][["group", "rows_with_any_group_term", "pct_rows_with_any_group_term"]].rename(
        columns={
            "rows_with_any_group_term": "rules_sentence_rows_with_group_terms",
            "pct_rows_with_any_group_term": "pct_rules_sentence_rows_with_group_terms",
        }
    )
    rules_a_stage = group_summary[group_summary["stage"] == "igt_rules_a_text_explicit"][["group", "rows_with_any_group_term", "pct_rows_with_any_group_term"]].rename(
        columns={
            "rows_with_any_group_term": "rules_explicit_a_rows_with_group_terms",
            "pct_rows_with_any_group_term": "pct_rules_explicit_a_rows_with_group_terms",
        }
    )

    gap_df = sentence_stage.merge(a_stage, on="group", how="outer").merge(rules_stage, on="group", how="outer").merge(rules_a_stage, on="group", how="outer").fillna(0)
    gap_df["gap_sentence_minus_a_pct"] = (gap_df["pct_sentence_rows_with_group_terms"] - gap_df["pct_a_rows_with_group_terms"]).round(2)
    gap_df["gap_rules_sentence_minus_rules_explicit_a_pct"] = (gap_df["pct_rules_sentence_rows_with_group_terms"] - gap_df["pct_rules_explicit_a_rows_with_group_terms"]).round(2)

    path_group = out_dir / "lexical_group_stage_summary.csv"
    path_term = out_dir / "lexical_term_stage_summary.csv"
    path_pivot = out_dir / "lexical_group_stage_pivot.csv"
    path_examples = out_dir / "lexical_hit_examples.csv"
    path_gap = out_dir / "lexical_a_vs_sentence_gap.csv"
    safe_to_csv(group_summary, path_group)
    safe_to_csv(term_summary, path_term)
    safe_to_csv(pivot, path_pivot)
    safe_to_csv(examples, path_examples)
    safe_to_csv(gap_df, path_gap)

    top_chunk = group_summary[group_summary["stage"] == "education_chunks_text"].sort_values("pct_rows_with_any_group_term", ascending=False)
    top_rules = group_summary[group_summary["stage"] == "igt_rules_sentence_text"].sort_values("pct_rows_with_any_group_term", ascending=False)

    md_lines = [
        "# Lexical pedagogical audit (education corpus)",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "This is a plain lexical diagnostic. It is intentionally simpler than the parser-based institutional method.",
        "",
        "## Group-level lexical presence in education chunks",
        build_markdown_table(top_chunk[["group", "rows_with_any_group_term", "pct_rows_with_any_group_term", "total_group_matches"]]),
        "",
        "## Group-level lexical presence in rules-only sentence text",
        build_markdown_table(top_rules[["group", "rows_with_any_group_term", "pct_rows_with_any_group_term", "total_group_matches"]]),
        "",
        "## Sentence-text vs A-text gap",
        build_markdown_table(gap_df[["group", "pct_sentence_rows_with_group_terms", "pct_a_rows_with_group_terms", "gap_sentence_minus_a_pct", "pct_rules_sentence_rows_with_group_terms", "pct_rules_explicit_a_rows_with_group_terms", "gap_rules_sentence_minus_rules_explicit_a_pct"]]),
    ]
    (out_dir / "lexical_summary.md").write_text("\n".join(md_lines), encoding="utf-8")

    metadata = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "chunks_jsonl": args.chunks_jsonl,
            "igt_parquet": args.igt_parquet,
        },
        "rule_labels": sorted(rule_labels),
        "term_groups": TERM_GROUPS,
        "n_chunks": int(len(chunks)),
        "n_igt_rows": int(len(igt)),
        "n_rule_rows": int(igt["_is_rule"].sum()),
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("Saved:")
    for p in [
        out_dir / "run_metadata.json",
        path_group,
        path_term,
        path_pivot,
        path_examples,
        path_gap,
        out_dir / "lexical_summary.md",
    ]:
        print(f"  {p}")

    print("\nGroup-level lexical presence in education chunks:")
    print(top_chunk[["group", "rows_with_any_group_term", "pct_rows_with_any_group_term", "total_group_matches"]].to_string(index=False))


if __name__ == "__main__":
    main()
