#!/usr/bin/env python3
"""
Step 8.2 — Chunk re-aggregation (strict v2) over sentence substrate

Input:
  data/derived/step8_igt_full/sentences_full.parquet

Output:
  data/derived/step8_igt_full/chunks_full.parquet
  data/derived/step8_igt_full/_runtime_params.json (updated/created)

Purpose:
- Re-aggregate adjacent sentences into institutionally coherent "analysis chunks"
- Preserve full provenance: each chunk contains ordered sentence_ids + source chunk ids
- Deterministic behavior + explicit thresholds
- Uses frozen lexicons (resources/lexicons) as boundary cues (NOT as classifiers)

Notes:
- This is intentionally conservative: it errs toward splitting rather than over-merging.
- ADICO extraction happens in Step 8.3; this step only builds coherent units.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INPUT = PROJECT_ROOT / "data/derived/step8_igt_full/sentences_full.parquet"
DEFAULT_OUT_DIR = PROJECT_ROOT / "data/derived/step8_igt_full"
DEFAULT_OUTPUT = DEFAULT_OUT_DIR / "chunks_full.parquet"
DEFAULT_RUNTIME = DEFAULT_OUT_DIR / "_runtime_params.json"

LEX_DIR = PROJECT_ROOT / "resources/lexicons"


def get_git_hash() -> Optional[str]:
    import subprocess

    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT))
        return out.decode("utf-8").strip()
    except Exception:
        return None


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(block_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def update_runtime_params(runtime_path: Path, updates: Dict) -> None:
    if runtime_path.exists():
        try:
            data = json.loads(runtime_path.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    else:
        data = {}

    for k, v in updates.items():
        data[k] = v

    runtime_path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def fmt_hhmmss(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return time.strftime("%H:%M:%S", time.gmtime(seconds))


def load_lexicon_terms(path: Path) -> List[str]:
    """
    Load one term per line, ignore blanks and comments.
    Terms are treated as regex fragments (already curated).
    """
    terms: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith("#"):
                continue
            terms.append(s)
    return terms


def compile_or_regex(terms: Sequence[str], flags=re.IGNORECASE) -> re.Pattern:
    if not terms:
        # compile something that never matches
        return re.compile(r"a\A", flags)
    joined = r"(?:%s)" % "|".join(terms)
    return re.compile(joined, flags)


def is_heading_like(s: str) -> bool:
    """
    Conservative heading detector:
    - short
    - mostly non-lowercase
    - often ends with ':' or is ALL CAPS-ish
    """
    t = s.strip()
    if not t:
        return False
    if len(t) <= 80 and (t.endswith(":") or t.isupper()):
        return True
    # "TITLE CASE" / "Section 3" style headings, very short
    if len(t) <= 40 and re.match(r"^(?:\d+[\.\)]\s+)?[A-Z][A-Za-z0-9 \-]{0,38}$", t):
        # but avoid normal sentences ending with punctuation
        if not re.search(r"[\.!?]$", t):
            return True
    return False


LIST_ITEM_RE = re.compile(r"^\s*(?:[-•*]|\d+[\.\)]|[a-zA-Z][\.\)])\s+")
SOFT_CONTINUATION_RE = re.compile(
    r"^\s*(?:and|or|but|also|however|therefore|thus|in addition|furthermore|moreover|"
    r"with respect to|in accordance with|for example|for instance|this|that|these|those|"
    r"it|they|such|which|who|where|when)\b",
    re.IGNORECASE,
)


def should_start_new_chunk(
    prev_text: str,
    curr_text: str,
    prev_has_deontic: bool,
    curr_has_deontic: bool,
) -> bool:
    """
    Boundary rules (strict v2, conservative):
    Start a new chunk when:
    - current sentence looks like a heading
    - current sentence begins a list item
    - previous sentence ends with ':' and current is list-like/indented
    - current introduces a fresh deontic after a non-deontic run (often new rule)
    - current is not a soft continuation and looks like a new statement
    """
    ct = curr_text.strip()
    pt = prev_text.strip()

    if not ct:
        return True

    if is_heading_like(ct):
        return True

    if LIST_ITEM_RE.match(ct):
        return True

    # Colon-driven boundary (often introduces sub-requirements)
    if pt.endswith(":") and (LIST_ITEM_RE.match(ct) or ct[:1].islower() is False):
        return True

    # Deontic onset boundary (new rule/norm/strategy often begins)
    if curr_has_deontic and not prev_has_deontic:
        # unless clearly continuing the same sentence/idea
        if not SOFT_CONTINUATION_RE.match(ct):
            return True

    # If current doesn't look like a continuation, treat as new chunk.
    # (Conservative: only merge when we see continuation cues.)
    if not SOFT_CONTINUATION_RE.match(ct):
        return True

    return False


def make_chunk_id(doc_id: str, chunk_full_index: int) -> str:
    raw = f"{doc_id}::fullchunk::{chunk_full_index}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default=str(DEFAULT_INPUT))
    ap.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT))
    ap.add_argument("--runtime", type=str, default=str(DEFAULT_RUNTIME))
    ap.add_argument("--max-sentences-per-chunk", type=int, default=8)
    ap.add_argument("--max-chars-per-chunk", type=int, default=1800)
    ap.add_argument("--report-every-docs", type=int, default=25)
    ap.add_argument("--max-docs", type=int, default=0, help="Smoke test only. 0 = full run.")
    args = ap.parse_args()

    in_path = Path(args.input).resolve()
    out_path = Path(args.output).resolve()
    runtime_path = Path(args.runtime).resolve()

    if not in_path.exists():
        raise FileNotFoundError(f"Input not found: {in_path}")

    # Load sentence substrate
    df = pd.read_parquet(in_path)

    required_cols = {
        "sentence_id",
        "doc_id",
        "chunk_id",
        "chunk_index",
        "sentence_index_in_chunk",
        "sentence_text",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing required columns in {in_path}: {sorted(missing)}")

    # Deterministic ordering across entire corpus
    df = df.sort_values(
        ["doc_id", "chunk_index", "chunk_id", "sentence_index_in_chunk"],
        kind="mergesort",
    ).reset_index(drop=True)

    # Compile lexicon cues (frozen)
    deontic_terms = load_lexicon_terms(LEX_DIR / "deontic.txt")
    deontic_re = compile_or_regex(deontic_terms, flags=re.IGNORECASE)

    # I  keep enforcement/conditional compiled for future optional boundary cues,
    # but do not make chunking depend on them (conservative).
    enforcement_terms = load_lexicon_terms(LEX_DIR / "enforcement.txt")
    conditional_terms = load_lexicon_terms(LEX_DIR / "conditional.txt")
    enforcement_re = compile_or_regex(enforcement_terms, flags=re.IGNORECASE)
    conditional_re = compile_or_regex(conditional_terms, flags=re.IGNORECASE)

    # Precompute doc count for progress
    doc_ids = df["doc_id"].astype(str).unique().tolist()
    total_docs = len(doc_ids)
    if args.max_docs and args.max_docs < total_docs:
        doc_ids = doc_ids[: args.max_docs]
    run_docs = len(doc_ids)

    start_time = time.time()

    out_records: List[Dict] = []
    chunk_counter_total = 0

    # Iterate per doc to guarantee chunk indices restart per doc
    for di, doc_id in enumerate(doc_ids, start=1):
        sub = df[df["doc_id"].astype(str) == str(doc_id)]

        # Stream sentences in order
        current_sent_ids: List[str] = []
        current_src_chunk_ids: List[str] = []
        current_text_parts: List[str] = []
        current_chars = 0
        current_has_deontic = False

        chunk_full_index = 0

        rows = sub.to_dict("records")

        prev_sentence_text = ""

        for r_i, r in enumerate(rows):
            s_id = str(r["sentence_id"])
            s_text = str(r["sentence_text"] or "").strip()

            # Skip fully empty sentence spans
            if not s_text:
                continue

            s_has_deontic = bool(deontic_re.search(s_text))
            # (available for later; not used for boundary now)
            _s_has_enforcement = bool(enforcement_re.search(s_text))
            _s_has_conditional = bool(conditional_re.search(s_text))

            # Decide boundary
            start_new = False
            if not current_sent_ids:
                start_new = False
            else:
                start_new = should_start_new_chunk(
                    prev_text=prev_sentence_text,
                    curr_text=s_text,
                    prev_has_deontic=current_has_deontic,
                    curr_has_deontic=s_has_deontic,
                )

                # Hard caps (avoid giant chunks)
                if len(current_sent_ids) >= args.max_sentences_per_chunk:
                    start_new = True
                if current_chars >= args.max_chars_per_chunk:
                    start_new = True

            if start_new:
                # Emit chunk
                chunk_id_full = make_chunk_id(str(doc_id), chunk_full_index)
                chunk_text = " ".join(current_text_parts).strip()

                out_records.append(
                    {
                        "chunk_full_id": chunk_id_full,
                        "doc_id": str(doc_id),
                        "chunk_full_index": int(chunk_full_index),
                        "text": chunk_text,
                        "sentence_ids": list(current_sent_ids),
                        "source_chunk_ids": list(dict.fromkeys(current_src_chunk_ids)),  # preserve order, unique
                        "n_sentences": int(len(current_sent_ids)),
                        "n_chars": int(len(chunk_text)),
                        "has_deontic_cue": bool(current_has_deontic),
                    }
                )
                chunk_counter_total += 1
                chunk_full_index += 1

                # Reset
                current_sent_ids = []
                current_src_chunk_ids = []
                current_text_parts = []
                current_chars = 0
                current_has_deontic = False

            # Add current sentence to active chunk
            current_sent_ids.append(s_id)
            current_src_chunk_ids.append(str(r["chunk_id"]))
            current_text_parts.append(s_text)
            current_chars += len(s_text) + 1
            current_has_deontic = current_has_deontic or s_has_deontic

            prev_sentence_text = s_text

        # Flush last chunk for doc
        if current_sent_ids:
            chunk_id_full = make_chunk_id(str(doc_id), chunk_full_index)
            chunk_text = " ".join(current_text_parts).strip()

            out_records.append(
                {
                    "chunk_full_id": chunk_id_full,
                    "doc_id": str(doc_id),
                    "chunk_full_index": int(chunk_full_index),
                    "text": chunk_text,
                    "sentence_ids": list(current_sent_ids),
                    "source_chunk_ids": list(dict.fromkeys(current_src_chunk_ids)),
                    "n_sentences": int(len(current_sent_ids)),
                    "n_chars": int(len(chunk_text)),
                    "has_deontic_cue": bool(current_has_deontic),
                }
            )
            chunk_counter_total += 1

        # Progress
        if (args.report_every_docs and di % args.report_every_docs == 0) or di == run_docs:
            now = time.time()
            elapsed = now - start_time
            rate = di / elapsed if elapsed > 0 else 0.0
            remaining = run_docs - di
            eta = remaining / rate if rate > 0 else 0.0
            pct = (di / run_docs) * 100 if run_docs else 0.0
            print(
                f"[step8_2] docs {di:,}/{run_docs:,} ({pct:5.1f}%) | "
                f"chunks_out {chunk_counter_total:,} | elapsed {fmt_hhmmss(elapsed)} | ETA {fmt_hhmmss(eta)}",
                flush=True,
            )

    out_df = pd.DataFrame.from_records(out_records)

    # Deterministic ordering
    out_df = out_df.sort_values(["doc_id", "chunk_full_index"], kind="mergesort").reset_index(drop=True)

    out_df.to_parquet(out_path, index=False)

    runtime_updates = {
        "step8_2_chunking_v2": {
            "timestamp_utc": dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
            "git_hash": get_git_hash(),
            "input_path": str(in_path),
            "input_sha256": sha256_file(in_path),
            "output_path": str(out_path),
            "lexicons": {
                "deontic": str(LEX_DIR / "deontic.txt"),
                "enforcement": str(LEX_DIR / "enforcement.txt"),
                "conditional": str(LEX_DIR / "conditional.txt"),
            },
            "lexicon_sha256": {
                "deontic": sha256_file(LEX_DIR / "deontic.txt"),
                "enforcement": sha256_file(LEX_DIR / "enforcement.txt"),
                "conditional": sha256_file(LEX_DIR / "conditional.txt"),
            },
            "params": {
                "max_sentences_per_chunk": int(args.max_sentences_per_chunk),
                "max_chars_per_chunk": int(args.max_chars_per_chunk),
            },
            "n_docs_processed": int(run_docs),
            "n_chunks_emitted": int(len(out_df)),
            "columns": list(out_df.columns),
        }
    }
    update_runtime_params(runtime_path, runtime_updates)

    print(f"[step8_2] done. docs={run_docs:,} chunks={len(out_df):,}")
    print(f"[step8_2] wrote: {out_path}")
    print(f"[step8_2] runtime: {runtime_path}")


if __name__ == "__main__":
    main()
