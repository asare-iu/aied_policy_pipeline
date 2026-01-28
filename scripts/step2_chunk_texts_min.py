#!/usr/bin/env python3
"""
Step 2 (minimal): Segment normalized documents into chunks (units of analysis).

Input:
- data/derived/step1_texts/docs_normalized_text/*.txt

Outputs:
- data/derived/step2_chunks_raw/chunks_raw.jsonl
- data/derived/step2_chunks_raw/doc_chunk_counts.csv

Chunking policy (default):
- Split on paragraph boundaries (blank lines).
- Optionally enforce max_chars by further splitting long paragraphs.
- Write one JSON object per chunk (JSONL) with simple pointers.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterator, List, Tuple

from tqdm import tqdm


@dataclass(frozen=True)
class ChunkRow:
    chunk_id: str
    doc_id: str
    chunk_index: int
    char_start: int
    char_end: int
    n_chars: int
    text: str


def iter_paragraph_spans(text: str) -> Iterator[Tuple[int, int]]:
    """
    Yield (start, end) spans for paragraphs separated by one or more blank lines.
    Spans are character offsets into the original text.
    """
    n = len(text)
    i = 0
    while i < n:
        # Skip leading whitespace/newlines between paragraphs.
        while i < n and text[i] in "\n\t ":
            i += 1
        if i >= n:
            break

        start = i
        # Paragraph ends at the next blank line boundary.
        while i < n:
            if text[i] == "\n":
                j = i
                # Count consecutive newlines.
                while j < n and text[j] == "\n":
                    j += 1
                if j - i >= 2:
                    end = i
                    break
                i = j
                continue
            i += 1
        else:
            end = n

        yield start, end
        i = end


def split_long_span(text: str, start: int, end: int, max_chars: int) -> List[Tuple[int, int]]:
    """
    If a paragraph span exceeds max_chars, split it into smaller spans.
    Splitting is done on whitespace boundaries when possible.
    """
    if max_chars <= 0 or (end - start) <= max_chars:
        return [(start, end)]

    spans: List[Tuple[int, int]] = []
    i = start
    while i < end:
        j = min(i + max_chars, end)
        if j < end:
            # Back up to a whitespace boundary to avoid cutting words.
            k = j
            while k > i and text[k - 1] not in " \t\n":
                k -= 1
            if k > i + 50:  # avoid pathological tiny splits
                j = k
        spans.append((i, j))
        i = j
    return spans


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def doc_id_from_filename(path: Path) -> str:
    # Step 1 writes <sha16>.txt
    return path.stem


def main() -> int:
    parser = argparse.ArgumentParser(description="Step 2: chunk normalized texts into JSONL.")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/derived/step1_texts/docs_normalized_text",
        help="Directory containing normalized .txt files (one per doc_id).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/derived/step2_chunks_raw",
        help="Directory to write chunk outputs.",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=2500,
        help="Maximum characters per chunk; long paragraphs are split. Use 0 to disable.",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=50,
        help="Minimum characters to keep a chunk (filters headers/noise).",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    out_jsonl = output_dir / "chunks_raw.jsonl"
    out_counts = output_dir / "doc_chunk_counts.csv"

    docs = sorted(p for p in input_dir.glob("*.txt") if p.is_file())
    if not docs:
        raise SystemExit(f"No .txt files found in {input_dir}")

    doc_counts: List[Tuple[str, int, int]] = []  # (doc_id, doc_chars, n_chunks)
    total_chunks = 0

    with out_jsonl.open("w", encoding="utf-8") as jf:
        for doc_path in tqdm(docs, desc="Step 2: chunking", unit="doc"):
            doc_id = doc_id_from_filename(doc_path)
            text = load_text(doc_path)

            chunk_index = 0
            for p_start, p_end in iter_paragraph_spans(text):
                for s, e in split_long_span(text, p_start, p_end, args.max_chars):
                    chunk_text = text[s:e].strip()
                    if len(chunk_text) < args.min_chars:
                        continue

                    row = ChunkRow(
                        chunk_id=f"{doc_id}_{chunk_index:05d}",
                        doc_id=doc_id,
                        chunk_index=chunk_index,
                        char_start=s,
                        char_end=e,
                        n_chars=len(chunk_text),
                        text=chunk_text,
                    )
                    jf.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")
                    chunk_index += 1

            doc_counts.append((doc_id, len(text), chunk_index))
            total_chunks += chunk_index

    with out_counts.open("w", newline="", encoding="utf-8") as cf:
        w = csv.writer(cf)
        w.writerow(["doc_id", "doc_char_count", "n_chunks"])
        w.writerows(doc_counts)

    print(f"Step 2 complete | docs={len(docs)} chunks={total_chunks}")
    print(f"Wrote: {out_jsonl}")
    print(f"Wrote: {out_counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

