#!/usr/bin/env python3
"""
Step 8.7 — Document-level umbrella sections (penalties/sanctions/enforcement) and rule intersection

This script extracts enforcement/sanction sections from document-level normalized text, assigns
character spans to each extracted section, and then links Institutional Grammar statements to these
sections via chunk character offsets. It supports three-way intersection analyses:
(1) rule_candidate statements, (2) statements located within umbrella sections, and (3) documents
containing education-relevant chunks.

Inputs
- data/derived/step1_texts/docs_normalized_text/<doc_id>.txt
- data/derived/step3_chunks_spacy/chunks_spacy.jsonl
- data/derived/step8_igt_full/igt_statements_full.parquet
- data/derived/step6_chunks_edu/chunks_edu.jsonl

Outputs (written to --out-dir)
- umbrella_sections.parquet
- rule_candidates_with_umbrella_flags.parquet
- rule_candidates_umbrella_INTERSECTION.parquet
- rule_candidates_umbrella_summary.csv
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


UMBRELLA_HEADING_KWS = [
    "penalt", "sanction", "enforcement", "offence", "offense",
    "liable", "liability", "administrative fine", "administrative fines",
    "fine", "fines", "punish", "prosecution", "breach",
    "non-compliance", "noncompliance", "revocation", "suspension", "withdrawal",
]

UMBRELLA_BODY_CUES = [
    "shall be liable", "shall be punished", "subject to", "administrative fine",
    "liable", "liability", "offence", "offense", "penalty", "penalties",
    "sanction", "sanctions", "non-compliance", "noncompliance",
    "failure to comply", "revocation", "suspension", "withdrawal",
    "may impose", "imposed by", "competent authority",
]

HEADING_MAX_CHARS = 140


def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def looks_like_heading(line: str) -> bool:
    s = line.strip()
    if not s or len(s) > HEADING_MAX_CHARS:
        return False
    if s.endswith(":"):
        return True
    if s.isupper() and len(s) >= 4:
        return True
    words = re.findall(r"[A-Za-z0-9]+", s)
    if 1 <= len(words) <= 12:
        caps = sum(1 for w in words if w and w[0].isupper())
        nums = sum(1 for w in words if w.isdigit() or re.fullmatch(r"[ivxlcdm]+", w.lower()))
        return (caps + nums) / max(len(words), 1) >= 0.6
    return False


def has_umbrella_heading_kw(heading: str) -> bool:
    h = heading.lower()
    return any(kw in h for kw in UMBRELLA_HEADING_KWS)


def has_umbrella_body_cue(body: str) -> bool:
    b = body.lower()
    return any(cue in b for cue in UMBRELLA_BODY_CUES)


def load_edu_doc_ids(edu_chunks_jsonl: Path) -> set[str]:
    edu_docs: set[str] = set()
    with edu_chunks_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            doc_id = str(obj.get("doc_id", "")).strip()
            if doc_id:
                edu_docs.add(doc_id)
    return edu_docs


def load_chunk_spans(chunks_jsonl: Path) -> pd.DataFrame:
    rows = []
    with chunks_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            rows.append(
                {
                    "doc_id": str(obj.get("doc_id")),
                    "chunk_id": str(obj.get("chunk_id")),
                    "char_start": int(obj.get("char_start", -1)),
                    "char_end": int(obj.get("char_end", -1)),
                }
            )
    df = pd.DataFrame(rows)
    df = df[(df["char_start"] >= 0) & (df["char_end"] >= 0)]
    return df


def extract_sections_from_doc_text(doc_id: str, text: str) -> pd.DataFrame:
    lines = text.splitlines(True)
    offsets: List[int] = []
    pos = 0
    for ln in lines:
        offsets.append(pos)
        pos += len(ln)

    headings: List[Tuple[int, str]] = []
    for i, ln in enumerate(lines):
        raw = ln.strip()
        if looks_like_heading(raw):
            headings.append((i, raw.rstrip(":").strip()))

    if not headings:
        return pd.DataFrame(
            columns=[
                "doc_id", "section_id", "heading_text",
                "section_char_start", "section_char_end",
                "section_text", "umbrella_flag",
                "umbrella_reason",
            ]
        )

    out_rows = []
    for h_idx, (line_i, heading) in enumerate(headings):
        start_line = line_i
        end_line = headings[h_idx + 1][0] if (h_idx + 1) < len(headings) else len(lines)
        section_char_start = offsets[start_line]
        section_char_end = offsets[end_line - 1] + len(lines[end_line - 1])
        section_text = "".join(lines[start_line:end_line]).strip("\n")
        body_only = normalize_ws(" ".join(ln.strip() for ln in lines[start_line + 1:end_line]))

        umbrella = False
        reason = "none"
        if has_umbrella_heading_kw(heading):
            if has_umbrella_body_cue(body_only) or any(k in heading.lower() for k in ["penalt", "sanction", "fine", "liabil", "offenc", "enforc"]):
                umbrella = True
                reason = "heading_kw"
            else:
                reason = "heading_kw_weak_body"
        else:
            if has_umbrella_body_cue(body_only):
                umbrella = True
                reason = "body_cue_no_heading_kw"

        out_rows.append(
            {
                "doc_id": doc_id,
                "section_id": f"{doc_id}::S{h_idx:04d}",
                "heading_text": heading,
                "section_char_start": section_char_start,
                "section_char_end": section_char_end,
                "section_text": section_text,
                "umbrella_flag": umbrella,
                "umbrella_reason": reason,
            }
        )

    return pd.DataFrame(out_rows)


def extract_umbrella_sections_for_corpus(doc_text_dir: Path, doc_ids: set[str] | None, report_every: int) -> pd.DataFrame:
    rows = []
    files = sorted(doc_text_dir.glob("*.txt"))
    t0 = time.time()
    n = 0
    for fp in files:
        doc_id = fp.stem
        if doc_ids is not None and doc_id not in doc_ids:
            continue
        try:
            text = fp.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        sec = extract_sections_from_doc_text(doc_id, text)
        if not sec.empty:
            sec = sec[sec["umbrella_flag"] == True]
            if not sec.empty:
                rows.append(sec)

        n += 1
        if report_every and n % report_every == 0:
            elapsed = (time.time() - t0) / 60.0
            print(f"[extract] docs={n:,} elapsed={elapsed:.1f}m")

    if not rows:
        return pd.DataFrame(
            columns=[
                "doc_id", "section_id", "heading_text",
                "section_char_start", "section_char_end",
                "section_text", "umbrella_flag",
                "umbrella_reason",
            ]
        )
    out = pd.concat(rows, ignore_index=True)
    return out


def interval_overlaps(a0: int, a1: int, b0: int, b1: int) -> bool:
    return (a0 < b1) and (b0 < a1)

def add_umbrella_flags_to_statements(
    statements: pd.DataFrame,
    chunk_spans: pd.DataFrame,
    umbrella_sections: pd.DataFrame,
    report_every: int,
) -> pd.DataFrame:
    st = statements.copy()

    st["doc_id"] = st["doc_id"].astype(str)
    st["chunk_id"] = st["chunk_id"].astype(str)

    chunk_spans = chunk_spans.copy()
    chunk_spans["doc_id"] = chunk_spans["doc_id"].astype(str)
    chunk_spans["chunk_id"] = chunk_spans["chunk_id"].astype(str)

    st = st.merge(chunk_spans, on=["doc_id", "chunk_id"], how="left")
    st["has_chunk_span"] = st["char_start"].notna() & st["char_end"].notna()

    umb_by_doc: Dict[str, List[Tuple[int, int, str, str]]] = {}
    for doc, g in umbrella_sections.groupby("doc_id"):
        umb_by_doc[str(doc)] = [
            (
                int(r["section_char_start"]),
                int(r["section_char_end"]),
                str(r["section_id"]),
                str(r["heading_text"]),
            )
            for _, r in g.iterrows()
        ]

    flags = []
    matched_section_id = []
    matched_heading = []

    t0 = time.time()
    for i, r in enumerate(st.itertuples(index=False), start=1):
        doc_id = getattr(r, "doc_id")
        cs = getattr(r, "char_start")
        ce = getattr(r, "char_end")
        if pd.isna(cs) or pd.isna(ce):
            flags.append(False)
            matched_section_id.append("")
            matched_heading.append("")
        else:
            c0 = int(cs)
            c1 = int(ce)
            hit = False
            sid = ""
            h = ""
            for (s0, s1, section_id, heading) in umb_by_doc.get(doc_id, []):
                if interval_overlaps(c0, c1, s0, s1):
                    hit = True
                    sid = section_id
                    h = heading
                    break
            flags.append(hit)
            matched_section_id.append(sid)
            matched_heading.append(h)

        if report_every and i % report_every == 0:
            elapsed = (time.time() - t0) / 60.0
            print(f"[link] statements={i:,}/{len(st):,} elapsed={elapsed:.1f}m")

    st["in_umbrella_section"] = flags
    st["umbrella_section_id"] = matched_section_id
    st["umbrella_heading"] = matched_heading
    return st


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, escapechar="\\")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc-text-dir", default="data/derived/step1_texts/docs_normalized_text")
    ap.add_argument("--chunks-jsonl", default="data/derived/step3_chunks_spacy/chunks_spacy.jsonl")
    ap.add_argument("--statements", default="data/derived/step8_igt_full/igt_statements_full.parquet")
    ap.add_argument("--edu-chunks-jsonl", default="data/derived/step6_chunks_edu/chunks_edu.jsonl")
    ap.add_argument("--out-dir", default="data/derived/step8_analysis/rules_x_umbrella_from_doc_sections")
    ap.add_argument("--report-every-docs", type=int, default=200)
    ap.add_argument("--report-every-statements", type=int, default=50000)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    doc_text_dir = Path(args.doc_text_dir)
    chunks_jsonl = Path(args.chunks_jsonl)
    statements_path = Path(args.statements)
    edu_chunks_jsonl = Path(args.edu_chunks_jsonl)

    edu_docs = load_edu_doc_ids(edu_chunks_jsonl)

    t0 = time.time()
    umbrella = extract_umbrella_sections_for_corpus(
        doc_text_dir=doc_text_dir,
        doc_ids=None,
        report_every=args.report_every_docs,
    )
    print(f"[extract] umbrella sections rows={len(umbrella):,} elapsed={(time.time()-t0)/60:.1f}m")
    umb_out = out_dir / "umbrella_sections.parquet"
    write_parquet(umbrella, umb_out)

    chunk_spans = load_chunk_spans(chunks_jsonl)

    st = pd.read_parquet(statements_path)
    st_rules = st[st["statement_type_candidate"] == "rule_candidate"].copy()

    t1 = time.time()
    st_rules_linked = add_umbrella_flags_to_statements(
        statements=st_rules,
        chunk_spans=chunk_spans,
        umbrella_sections=umbrella,
        report_every=args.report_every_statements,
    )
    print(f"[link] rules rows={len(st_rules_linked):,} elapsed={(time.time()-t1)/60:.1f}m")

    rules_out = out_dir / "rule_candidates_with_umbrella_flags.parquet"
    write_parquet(st_rules_linked, rules_out)

    inter = st_rules_linked[
        (st_rules_linked["in_umbrella_section"] == True)
        & (st_rules_linked["doc_id"].isin(edu_docs))
    ].copy()

    inter_out = out_dir / "rule_candidates_umbrella_INTERSECTION.parquet"
    write_parquet(inter, inter_out)

    summary = pd.DataFrame(
        {
            "total_docs_in_statements": [st["doc_id"].nunique()],
            "total_rule_candidates": [len(st_rules)],
            "total_umbrella_sections": [len(umbrella)],
            "rule_candidates_in_umbrella_sections": [int(st_rules_linked["in_umbrella_section"].sum())],
            "edu_docs_count": [len(edu_docs)],
            "three_way_intersection_rules": [len(inter)],
        }
    )
    summary_out = out_dir / "rule_candidates_umbrella_summary.csv"
    write_csv(summary, summary_out)

    print("[ok] wrote:", str(umb_out))
    print("[ok] wrote:", str(rules_out))
    print("[ok] wrote:", str(inter_out))
    print("[ok] wrote:", str(summary_out))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
