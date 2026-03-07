#!/usr/bin/env python3
"""
Step 8.9A — Document-level enforcement language index

This script scans document text for enforcement/penalty cues and produces a document-level index.

Inputs
- data/derived/step8_igt_full/chunks_full.parquet

Outputs
- data/derived/step8_9_regime_closure/doc_enforcement_index.parquet
- data/derived/step8_9_regime_closure/doc_enforcement_index.csv
- data/derived/step8_9_regime_closure/doc_enforcement_index_report.md
"""

from __future__ import annotations

import argparse
import csv
import re
import time
from pathlib import Path
from typing import Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Enforcement/penalty cues: broad enough for heterogeneous legal/policy texts.
ENFORCEMENT_RE = re.compile(
    r"\b("
    r"failure\s+to\s+comply|non[-\s]?compliance|non[-\s]?conformity|breach|violation|"
    r"liable|liability|penalt(?:y|ies)|sanction(?:s)?|fine(?:s)?|"
    r"administrative\s+fine(?:s)?|criminal|civil\s+penalt(?:y|ies)|"
    r"offen[cs]e(?:s)?|prosecution|"
    r"revocation|suspension|withdrawal|termination|"
    r"enforcement|investigation|inspection|audit|"
    r"cease\s+and\s+desist|injunction|"
    r"shall\s+be\s+punishable|punishable|"
    r"subject\s+to"
    r")\b",
    re.IGNORECASE,
)

TEXT_FIELDS = (
    "chunk_text",
    "text",
    "content",
    "chunk",
    "chunk_normalized",
    "text_normalized",
)

WS_RE = re.compile(r"\s+")

def pick_text_col(df: pd.DataFrame) -> str:
    for c in TEXT_FIELDS:
        if c in df.columns:
            return c
    # last resort: first string column (excluding doc_id/chunk_id)
    candidates = [
        c for c in df.columns
        if c not in ("doc_id", "chunk_id") and pd.api.types.is_string_dtype(df[c])
    ]
    if candidates:
        return candidates[0]
    raise ValueError(
        "No usable text column found. "
        f"Expected one of {TEXT_FIELDS}. Found columns: {list(df.columns)}"
    )

def sanitize_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s).replace("\x00", " ")
    s = WS_RE.sub(" ", s).strip()
    return s

def extract_snippet(text: str, match: re.Match, max_len: int) -> str:
    start = max(0, match.start() - 80)
    end = min(len(text), match.end() + 180)
    sn = sanitize_text(text[start:end])
    if len(sn) > max_len:
        sn = sn[:max_len] + "…"
    return sn

def write_report_md(
    path: Path,
    chunks_path: Path,
    text_col: str,
    n_docs: int,
    n_hit_docs: int,
    regex_pattern: str,
) -> None:
    md = []
    md.append("# Document-level enforcement language index\n")
    md.append("## Purpose\n")
    md.append(
        "This index identifies documents that contain language commonly associated with enforcement, "
        "penalties, sanctions, liability, and related compliance mechanisms. "
        "It is intended to support corpus-level analysis where enforcement provisions may be "
        "document-wide and not reliably recoverable through sentence-level linkage.\n"
    )
    md.append("## Method\n")
    md.append(
        f"- Unit of analysis: document (`doc_id`).\n"
        f"- Source text: chunked document text from `{chunks_path}`.\n"
        f"- Text field used: `{text_col}`.\n"
        "- Procedure: regex scan over all chunks in a document; count cue hits and retain one short evidence snippet.\n"
    )
    md.append("## Outputs\n")
    md.append(
        "- `doc_enforcement_index.parquet` and `doc_enforcement_index.csv` with fields:\n"
        "  - `doc_id`\n"
        "  - `chunk_count`\n"
        "  - `enforcement_cue_hits`\n"
        "  - `has_enforcement_language`\n"
        "  - `first_evidence_snippet`\n"
    )
    md.append("## Summary\n")
    md.append(f"- Documents scanned: {n_docs}\n")
    md.append(f"- Documents with ≥1 enforcement cue: {n_hit_docs}\n")
    md.append("## Cue pattern\n")
    md.append("The following regex was used:\n")
    md.append(f"```\n{regex_pattern}\n```\n")
    md.append("## Limitations\n")
    md.append(
        "- This is a cue-based indicator, not a legal classification.\n"
        "- False positives are possible (e.g., descriptive discussion of enforcement).\n"
        "- False negatives are possible where enforcement is expressed without the cue terms.\n"
        "- The index does not attribute penalties to specific obligations; it flags document-level presence.\n"
    )
    path.write_text("\n".join(md), encoding="utf-8")

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--chunks-parquet",
        default=str(PROJECT_ROOT / "data/derived/step8_igt_full/chunks_full.parquet"),
    )
    ap.add_argument(
        "--out-dir",
        default=str(PROJECT_ROOT / "data/derived/step8_9_regime_closure"),
    )
    ap.add_argument("--max-snippet-len", type=int, default=260)
    ap.add_argument(
        "--progress-every",
        type=int,
        default=200,
        help="Print progress every N documents",
    )
    ap.add_argument(
        "--max-chunks-per-doc",
        type=int,
        default=0,
        help="Optional cap for speed/debugging (0 = no cap). Uses earliest rows per doc.",
    )
    args = ap.parse_args()

    t0 = time.time()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    chunks_path = Path(args.chunks_parquet)
    chunks = pd.read_parquet(chunks_path)

    if "doc_id" not in chunks.columns:
        raise ValueError(f"Missing required column doc_id in {chunks_path}. Columns: {list(chunks.columns)}")

    text_col = pick_text_col(chunks)

    # Ensure deterministic ordering within docs
    sort_cols = [c for c in ["doc_id", "chunk_id"] if c in chunks.columns]
    if sort_cols:
        chunks = chunks.sort_values(sort_cols, kind="mergesort")

    rows = []
    doc_ids = chunks["doc_id"].dropna().unique().tolist()
    n_docs = len(doc_ids)

    last_print = t0
    for i, doc_id in enumerate(doc_ids, start=1):
        g = chunks[chunks["doc_id"] == doc_id]
        if args.max_chunks_per_doc and args.max_chunks_per_doc > 0:
            g = g.head(args.max_chunks_per_doc)

        texts = g[text_col].fillna("").astype(str).tolist()

        cue_hits = 0
        first_snip: Optional[str] = None

        for t in texts:
            t2 = sanitize_text(t)
            m = ENFORCEMENT_RE.search(t2)
            if m:
                # count hits in this chunk
                cue_hits += len(ENFORCEMENT_RE.findall(t2))
                if first_snip is None:
                    first_snip = extract_snippet(t2, m, args.max_snippet_len)

        rows.append(
            {
                "doc_id": doc_id,
                "chunk_count": int(len(g)),
                "enforcement_cue_hits": int(cue_hits),
                "has_enforcement_language": bool(cue_hits > 0),
                "first_evidence_snippet": first_snip,
            }
        )

        if (i % args.progress_every) == 0 or i == n_docs:
            now = time.time()
            elapsed = now - t0
            rate = i / elapsed if elapsed > 0 else 0.0
            hit_docs = sum(1 for r in rows if r["has_enforcement_language"])
            print(
                f"[step8_9a] docs={i}/{n_docs} "
                f"hit_docs={hit_docs} "
                f"elapsed_s={elapsed:.1f} "
                f"docs_per_s={rate:.2f}"
            )
            last_print = now

    out = pd.DataFrame(rows)

    out_path_pq = out_dir / "doc_enforcement_index.parquet"
    out_path_csv = out_dir / "doc_enforcement_index.csv"
    out_path_md = out_dir / "doc_enforcement_index_report.md"

    out.to_parquet(out_path_pq, index=False)
    out.to_csv(
        out_path_csv,
        index=False,
        escapechar="\\",
        quoting=csv.QUOTE_MINIMAL,
    )

    write_report_md(
        path=out_path_md,
        chunks_path=chunks_path,
        text_col=text_col,
        n_docs=n_docs,
        n_hit_docs=int(out["has_enforcement_language"].sum()) if len(out) else 0,
        regex_pattern=ENFORCEMENT_RE.pattern,
    )

    elapsed_total = time.time() - t0
    print(f"[step8_9a] wrote {out_path_pq}")
    print(f"[step8_9a] wrote {out_path_csv}")
    print(f"[step8_9a] wrote {out_path_md}")
    print(f"[step8_9a] done elapsed_s={elapsed_total:.1f} docs={n_docs}")

if __name__ == "__main__":
    main()
