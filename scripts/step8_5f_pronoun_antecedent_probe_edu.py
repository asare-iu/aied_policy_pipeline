#!/usr/bin/env python3
"""
Step 8.5f — Pronoun / underspecified-A antecedent probe for rules-only statements.

Purpose
-------
Test a key methodological objection: some rule sentences may use pronouns or
underspecified A terms (for example, "they", "such institutions") where the
pedagogical actor was named in the immediately preceding sentence rather than in
that rule sentence itself.

This script estimates how often that happens in the education-relevant rules-only
subset and whether those antecedents are pedagogical.

Default inputs
--------------
- data/derived/step6_chunks_edu/chunks_edu.jsonl
- data/derived/step8_igt_chunks_edu/igt_statements_full.parquet

Default outputs
---------------
- data/derived/step8_analysis/pronoun_antecedent_probe_edu/
    - pronoun_a_reference_summary.csv
    - pronoun_a_pedagogical_link_summary.csv
    - pronoun_a_top_raw_texts.csv
    - pronoun_a_examples.csv
    - pronoun_a_summary.md
    - run_metadata.json
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import spacy


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
    return str(x).strip()



def normalize_stmt_type(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().str.lower()



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
                    "chunk_text": str(chunk_text),
                }
            )
    return pd.DataFrame(rows)
# -----------------------------------------------------------------------------
# Patterns
# -----------------------------------------------------------------------------

DEFAULT_RULE_LABELS = {"rule_candidate", "rule", "rules"}

PED_GROUP_PATTERNS: Dict[str, List[str]] = {
    "educators_teachers": [
        r"\beducator(s)?\b",
        r"\bteacher(s)?\b",
        r"\binstructor(s)?\b",
        r"\bfaculty\b",
        r"\bprofessor(s)?\b",
        r"\blecturer(s)?\b",
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
        r"\buniversity\b|\buniversities\b",
        r"\bcollege(s)?\b",
        r"\binstitution(s)?\b",
        r"\beducational institution(s)?\b",
        r"\bacademic institution(s)?\b",
        r"\bclassroom(s)?\b",
        r"\bcampus\b|\bcampuses\b",
    ],
}

PRONOUN_PATTERNS = [
    r"^they$",
    r"^them$",
    r"^their$",
    r"^those$",
    r"^these$",
    r"^such$",
    r"^it$",
    r"^its$",
]

UNDERSPECIFIED_PATTERNS = [
    r"^such (institution|institutions|body|bodies|entity|entities|actor|actors|stakeholder|stakeholders|provider|providers|user|users)$",
    r"^these (institutions|bodies|entities|actors|stakeholders|providers|users)$",
    r"^those (institutions|bodies|entities|actors|stakeholders|providers|users)$",
    r"^relevant (actors|stakeholders|institutions|authorities|bodies)$",
    r"^competent (authorities|bodies)$",
    r"^member states$",
    r"^stakeholders$",
    r"^actors$",
    r"^institutions$",
    r"^authorities$",
]


# -----------------------------------------------------------------------------
# Core logic
# -----------------------------------------------------------------------------


def classify_a_reference(a_text: str) -> str:
    s = textify(a_text).lower().strip()
    if not s:
        return "empty_or_missing_a"
    for pat in PRONOUN_PATTERNS:
        if re.match(pat, s, flags=re.IGNORECASE):
            return "pronoun_a"
    for pat in UNDERSPECIFIED_PATTERNS:
        if re.match(pat, s, flags=re.IGNORECASE):
            return "underspecified_a"
    return "named_or_other_a"



def get_sentencizer():
    nlp = spacy.blank("en")
    if "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")
    return nlp



def split_sentences(nlp, text: str) -> List[str]:
    if not text:
        return []
    doc = nlp(text)
    return [str(s.text).strip() for s in doc.sents if str(s.text).strip()]



def match_pedagogical_groups(text: str) -> List[str]:
    s = textify(text).lower()
    hits = []
    for group, pats in PED_GROUP_PATTERNS.items():
        if any(re.search(pat, s, flags=re.IGNORECASE) for pat in pats):
            hits.append(group)
    return hits



def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks-jsonl", default="data/derived/step6_chunks_edu/chunks_edu.jsonl")
    ap.add_argument("--igt-parquet", default="data/derived/step8_igt_chunks_edu/igt_statements_full.parquet")
    ap.add_argument("--out-dir", default="data/derived/step8_analysis/pronoun_antecedent_probe_edu")
    ap.add_argument("--rule-labels", nargs="*", default=sorted(DEFAULT_RULE_LABELS))
    ap.add_argument("--max-examples", type=int, default=200)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    rule_labels = {str(x).strip().lower() for x in args.rule_labels}

    igt = maybe_read_table(Path(args.igt_parquet))
    chunks = iter_chunks(Path(args.chunks_jsonl))

    required = {"doc_id", "chunk_id", "sentence_text", "sentence_index_in_chunk", "a_raw_text", "a_class", "statement_type_candidate"}
    missing = sorted(list(required - set(igt.columns)))
    if missing:
        raise ValueError(f"Missing expected IGT columns: {missing}")

    igt = igt.copy()
    igt["_stmt_type_norm"] = normalize_stmt_type(igt["statement_type_candidate"])
    rules = igt.loc[igt["_stmt_type_norm"].isin(rule_labels)].copy()
    rules["a_reference_type"] = rules["a_raw_text"].apply(classify_a_reference)

    nlp = get_sentencizer()
    chunk_map = {(str(r.doc_id), str(r.chunk_id)): r.chunk_text for r in chunks.itertuples(index=False)}

    enriched_rows: List[dict] = []
    for r in rules.itertuples(index=False):
        doc_id = str(r.doc_id)
        chunk_id = str(r.chunk_id)
        sent_idx = int(r.sentence_index_in_chunk) if pd.notna(r.sentence_index_in_chunk) else -1
        chunk_text = chunk_map.get((doc_id, chunk_id), "")
        chunk_sents = split_sentences(nlp, chunk_text) if chunk_text else []

        prev_sent = chunk_sents[sent_idx - 1] if sent_idx > 0 and sent_idx - 1 < len(chunk_sents) else ""
        curr_sent = chunk_sents[sent_idx] if sent_idx >= 0 and sent_idx < len(chunk_sents) else textify(r.sentence_text)
        next_sent = chunk_sents[sent_idx + 1] if sent_idx >= 0 and sent_idx + 1 < len(chunk_sents) else ""
        prior_sents = chunk_sents[:sent_idx] if sent_idx > 0 and sent_idx <= len(chunk_sents) else []

        prev_groups = match_pedagogical_groups(prev_sent)
        any_prior_groups = sorted({g for sent in prior_sents for g in match_pedagogical_groups(sent)})
        curr_groups = match_pedagogical_groups(curr_sent)

        enriched_rows.append(
            {
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "sentence_index_in_chunk": sent_idx,
                "statement_type_candidate": textify(r.statement_type_candidate),
                "a_raw_text": textify(r.a_raw_text),
                "a_class": textify(r.a_class),
                "a_reference_type": textify(r.a_reference_type),
                "sentence_text": textify(r.sentence_text),
                "prev_sentence": prev_sent,
                "next_sentence": next_sent,
                "prev_has_pedagogical_actor": bool(prev_groups),
                "prev_pedagogical_groups": "|".join(prev_groups) if prev_groups else "",
                "any_prior_has_pedagogical_actor": bool(any_prior_groups),
                "any_prior_pedagogical_groups": "|".join(any_prior_groups) if any_prior_groups else "",
                "current_sentence_pedagogical_groups": "|".join(curr_groups) if curr_groups else "",
            }
        )

    enriched = pd.DataFrame(enriched_rows)
    ref_summary = (
        enriched.groupby("a_reference_type")
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    ref_summary["pct_of_rules_only"] = (100.0 * ref_summary["count"] / max(len(enriched), 1)).round(2)

    ped_link_summary = (
        enriched.groupby(["a_reference_type", "prev_has_pedagogical_actor", "any_prior_has_pedagogical_actor"])
        .size()
        .reset_index(name="count")
        .sort_values(["a_reference_type", "count"], ascending=[True, False])
    )
    ped_link_summary["pct_within_reference_type"] = (
        ped_link_summary.groupby("a_reference_type")["count"]
        .transform(lambda x: (100.0 * x / x.sum()).round(2))
    )

    top_a_texts = (
        enriched["a_raw_text"].fillna("").astype(str).str.strip().replace("", "<EMPTY>")
        .value_counts()
        .reset_index()
    )
    top_a_texts.columns = ["a_raw_text", "count"]
    top_a_texts["pct_of_rules_only"] = (100.0 * top_a_texts["count"] / max(len(enriched), 1)).round(2)

    example_df = enriched.loc[
        enriched["a_reference_type"].isin(["pronoun_a", "underspecified_a", "empty_or_missing_a"])
    ].copy()
    example_df = example_df.sort_values(["a_reference_type", "prev_has_pedagogical_actor", "any_prior_has_pedagogical_actor"], ascending=[True, False, False])
    if len(example_df) > args.max_examples:
        example_df = example_df.groupby("a_reference_type", group_keys=False).head(max(1, args.max_examples // 3))

    path_ref = out_dir / "pronoun_a_reference_summary.csv"
    path_link = out_dir / "pronoun_a_pedagogical_link_summary.csv"
    path_top = out_dir / "pronoun_a_top_raw_texts.csv"
    path_examples = out_dir / "pronoun_a_examples.csv"
    safe_to_csv(ref_summary, path_ref)
    safe_to_csv(ped_link_summary, path_link)
    safe_to_csv(top_a_texts.head(100), path_top)
    safe_to_csv(example_df, path_examples)

    pronoun_subset = enriched[enriched["a_reference_type"] == "pronoun_a"]
    underspecified_subset = enriched[enriched["a_reference_type"] == "underspecified_a"]
    generic_subset = enriched[enriched["a_reference_type"].isin(["pronoun_a", "underspecified_a", "empty_or_missing_a"])]

    md_lines = [
        "# Pronoun / antecedent probe (education rules-only)",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        f"Rules-only rows examined: {len(enriched)}",
        "",
        "## A-reference type distribution",
        build_markdown_table(ref_summary),
        "",
        "## Generic A-forms and pedagogical antecedents",
        build_markdown_table(ped_link_summary),
        "",
        f"Pronoun-A rows: {len(pronoun_subset)}; of these, previous sentence has pedagogical actor = {int(pronoun_subset['prev_has_pedagogical_actor'].sum())}",
        f"Underspecified-A rows: {len(underspecified_subset)}; of these, previous sentence has pedagogical actor = {int(underspecified_subset['prev_has_pedagogical_actor'].sum())}",
        f"All generic/empty A rows: {len(generic_subset)}; of these, any prior sentence in chunk has pedagogical actor = {int(generic_subset['any_prior_has_pedagogical_actor'].sum())}",
    ]
    (out_dir / "pronoun_a_summary.md").write_text("\n".join(md_lines), encoding="utf-8")

    metadata = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "chunks_jsonl": args.chunks_jsonl,
            "igt_parquet": args.igt_parquet,
        },
        "n_rules_only_rows": int(len(enriched)),
        "rule_labels": sorted(rule_labels),
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("Saved:")
    for p in [
        out_dir / "run_metadata.json",
        path_ref,
        path_link,
        path_top,
        path_examples,
        out_dir / "pronoun_a_summary.md",
    ]:
        print(f"  {p}")

    print("\nA-reference type distribution:")
    print(ref_summary.to_string(index=False))


if __name__ == "__main__":
    main()
