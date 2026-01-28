#!/usr/bin/env python3
"""
Step 3: spaCy annotate chunk texts.

Input:
- data/derived/step2_chunks_raw/chunks_raw.jsonl

Output:
- data/derived/step3_chunks_spacy/chunks_spacy.jsonl

Annotation includes:
- tokens: text, lemma, pos, tag, dep, head index, is_sent_start
- sentences: list of (start_token, end_token) spans

Notes:
- This is intentionally compact JSONL to keep size manageable.
- NER is disabled by default for speed; enable with --ner.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm
import spacy


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Step 3: spaCy annotate chunks.")
    parser.add_argument(
        "--input",
        type=str,
        default="data/derived/step2_chunks_raw/chunks_raw.jsonl",
        help="Path to chunks_raw.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/derived/step3_chunks_spacy",
        help="Directory for spaCy-annotated output.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="en_core_web_sm",
        help="spaCy model name.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Number of chunks per spaCy pipe batch.",
    )
    parser.add_argument(
        "--ner",
        action="store_true",
        help="Enable NER (disabled by default for speed).",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "chunks_spacy.jsonl"

    rows = read_jsonl(in_path)

    disable = ["ner"] if not args.ner else []
    nlp = spacy.load(args.model, disable=disable)
    nlp.max_length = 2_000_000  # safety for long chunks

    texts = [r["text"] for r in rows]

    with out_path.open("w", encoding="utf-8") as out:
        for r, doc in tqdm(
            zip(rows, nlp.pipe(texts, batch_size=args.batch_size)),
            total=len(rows),
            desc="Step 3: spaCy",
            unit="chunk",
        ):
            tokens = []
            for t in doc:
                tokens.append(
                    {
                        "i": t.i,
                        "text": t.text,
                        "lemma": t.lemma_,
                        "pos": t.pos_,
                        "tag": t.tag_,
                        "dep": t.dep_,
                        "head": t.head.i,
                        "is_sent_start": bool(t.is_sent_start),
                    }
                )

            sents = [{"start": s.start, "end": s.end} for s in doc.sents]

            out_row = {
                "chunk_id": r["chunk_id"],
                "doc_id": r["doc_id"],
                "chunk_index": r["chunk_index"],
                "char_start": r["char_start"],
                "char_end": r["char_end"],
                "n_chars": r["n_chars"],
                "text": r["text"],
                "tokens": tokens,
                "sentences": sents,
            }

            if args.ner:
                out_row["entities"] = [
                    {"start": e.start, "end": e.end, "label": e.label_, "text": e.text}
                    for e in doc.ents
                ]

            out.write(json.dumps(out_row, ensure_ascii=False) + "\n")

    print(f"Step 3 complete | chunks={len(rows)}")
    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
