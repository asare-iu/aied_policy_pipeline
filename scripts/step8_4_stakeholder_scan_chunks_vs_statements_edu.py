#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


def compile_patterns(spec: Dict[str, List[str]]) -> Dict[str, re.Pattern]:
    out = {}
    for k, terms in spec.items():
        # word-boundary for single words; allow spaces/hyphens inside phrases
        # we use a simple OR-regex; terms should be already lowercased
        escaped = [re.escape(t) for t in terms]
        pat = r"(" + r"|".join(escaped) + r")"
        out[k] = re.compile(pat, flags=re.IGNORECASE)
    return out


def iter_chunks_texts(jsonl_path: Path, text_field_candidates: List[str]) -> Tuple[int, List[str]]:
    """
    Returns: (n_chunks, list_of_texts)
    We stream-read JSONL but store texts in memory (ok for ~10k chunks).
    """
    texts = []
    n = 0
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            n += 1
            txt = None
            for field in text_field_candidates:
                if field in obj and obj[field]:
                    txt = obj[field]
                    break
            if txt is None:
                txt = ""
            texts.append(str(txt))
    return n, texts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--chunks-jsonl",
        default="data/derived/step6_chunks_edu/chunks_edu.jsonl",
        help="All education chunks JSONL (discourse baseline)",
    )
    ap.add_argument(
        "--igt-parquet",
        default="data/derived/step8_igt_chunks_edu/igt_statements_full.parquet",
        help="Edu-relevant institutional statements parquet",
    )
    ap.add_argument(
        "--out-csv",
        default="data/derived/step8_analysis/stakeholder_mentions_chunks_vs_statements_edu.csv",
        help="Output CSV path",
    )
    args = ap.parse_args()

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Stakeholder sets: tune freely, but keep transparent + deterministic.
    stakeholders: Dict[str, List[str]] = {
        # pedagogical actors
        "students": ["student", "students", "learner", "learners", "pupil", "pupils"],
        "educators": ["teacher", "teachers", "educator", "educators", "teaching staff", "faculty", "instructor", "instructors"],
        "schools_institutions": ["school", "schools", "university", "universities", "college", "colleges", "institution", "institutions"],

        # governance actors
        "policy_makers": ["ministry", "ministries", "government", "public authority", "public authorities", "education department", "department of education"],
        "regulators": ["regulator", "regulators", "supervisor", "supervisory authority", "oversight body", "inspectorate"],

        # ecosystem actors (socio-technical)
        "commercial_designers": ["vendor", "vendors", "provider", "providers", "supplier", "suppliers", "developer", "developers", "company", "companies", "platform company"],
        "deployers_users": ["deployer", "deployers", "operator", "operators", "user", "users", "administrator", "administrators"],

        # technical objects that often act like “actors” in discourse
        "platforms_systems": ["system", "systems", "platform", "platforms", "tool", "tools", "model", "models", "ai system", "ai systems", "algorithm", "algorithms"],
        "data_subjects": ["data subject", "data subjects", "child", "children", "minor", "minors"],
        "researchers": ["researcher", "researchers", "research community", "scientist", "scientists"],
    }
    pats = compile_patterns({k: [t.lower() for t in v] for k, v in stakeholders.items()})

    # ---- Scan CHUNKS (discourse baseline) ----
    chunks_path = Path(args.chunks_jsonl)
    # try common fields used across your pipeline
    n_chunks, chunk_texts = iter_chunks_texts(
        chunks_path,
        text_field_candidates=[
            "chunk_text", "text", "text_normalized", "chunk_text_normalized", "chunk_text_clean",
            "chunk_text_en", "chunk_text_translated",
        ],
    )

    chunk_hits = {k: 0 for k in stakeholders.keys()}
    for txt in chunk_texts:
        for k, pat in pats.items():
            if pat.search(txt):
                chunk_hits[k] += 1

    # ---- Scan STATEMENTS (institutional) ----
    igt_path = Path(args.igt_parquet)
    igt = pd.read_parquet(igt_path)

    if "sentence_text" not in igt.columns:
        raise ValueError(f"Expected column sentence_text in {igt_path}")

    n_statements = len(igt)
    stmt_hits = {k: 0 for k in stakeholders.keys()}

    # count % of statements where stakeholder appears anywhere in sentence_text
    st = igt["sentence_text"].fillna("").astype(str).tolist()
    for txt in st:
        for k, pat in pats.items():
            if pat.search(txt):
                stmt_hits[k] += 1

    rows = []
    for k in stakeholders.keys():
        rows.append(
            {
                "stakeholder": k,
                "chunks_with_mentions": chunk_hits[k],
                "pct_chunks": round(100 * chunk_hits[k] / max(n_chunks, 1), 2),
                "statements_with_mentions": stmt_hits[k],
                "pct_statements": round(100 * stmt_hits[k] / max(n_statements, 1), 2),
                "gap_chunks_minus_statements": round(
                    (100 * chunk_hits[k] / max(n_chunks, 1)) - (100 * stmt_hits[k] / max(n_statements, 1)),
                    2,
                ),
                "n_chunks_total": n_chunks,
                "n_statements_total": n_statements,
            }
        )

    out = pd.DataFrame(rows).sort_values("pct_chunks", ascending=False)
    out.to_csv(out_path, index=False)
    print(f"Saved → {out_path}")
    print(out[["stakeholder", "chunks_with_mentions", "pct_chunks", "statements_with_mentions", "pct_statements", "gap_chunks_minus_statements"]].head(25).to_string(index=False))


if __name__ == "__main__":
    main()
