#!/usr/bin/env python3
"""
Step 4 (lexicon-driven): Tag chunks with institutional signals using auditable term lists.

Input:
- data/derived/step3_chunks_spacy/chunks_spacy.jsonl

Outputs:
- data/derived/step4_chunks_tagged/chunks_normalized_tagged.jsonl

Lexicons:
- resources/lexicons/*.txt (one term/phrase per line; blank lines and # comments ignored)

Tagging:
- Each lexicon produces:
  - has_<name> (bool)
  - n_<name> (int count of matches)

Notes:
- Matching is case-insensitive.
- Word-boundaries are enforced for single words; phrases are matched as whitespace-normalized sequences.
- This stage does not change the original chunk text; it adds a whitespace-normalized copy for stable matching.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from tqdm import tqdm


LEXICON_SPECS: List[Tuple[str, str]] = [
    ("deontic", "resources/lexicons/deontic.txt"),
    ("conditional", "resources/lexicons/conditional.txt"),
    ("exception", "resources/lexicons/exception.txt"),
    ("definition", "resources/lexicons/definition.txt"),
    ("authority_delegation", "resources/lexicons/authority_delegation.txt"),
    ("enforcement", "resources/lexicons/enforcement.txt"),
    ("info_reporting", "resources/lexicons/info_reporting.txt"),
    ("monitoring_audit", "resources/lexicons/monitoring_audit.txt"),
    ("scope_applicability", "resources/lexicons/scope_applicability.txt"),
    ("education_terms", "resources/lexicons/education_terms.txt"),
]


_WS_RE = re.compile(r"\s+")


def normalize_for_match(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return _WS_RE.sub(" ", text).strip()


def load_terms(path: Path) -> List[str]:
    terms: List[str] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        terms.append(s)
    return terms


def compile_lexicon_regex(terms: List[str]) -> re.Pattern:
    """
    Compile a single regex that matches any term/phrase.
    - Single words use \\bword\\b boundaries.
    - Phrases are whitespace-normalized and matched with \\s+ between words.
    """
    patterns: List[str] = []
    for t in terms:
        t_norm = _WS_RE.sub(" ", t.strip())
        if " " in t_norm:
            # Phrase match: words separated by one or more spaces in normalized text.
            parts = [re.escape(p) for p in t_norm.split(" ")]
            patterns.append(r"(?:%s)" % r"\s+".join(parts))
        else:
            # Single token match with word boundaries.
            patterns.append(r"\b%s\b" % re.escape(t_norm))
    if not patterns:
        # Never match if lexicon is empty.
        return re.compile(r"(?!x)x")
    return re.compile(r"(?:%s)" % "|".join(patterns), flags=re.IGNORECASE)


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main() -> int:
    parser = argparse.ArgumentParser(description="Step 4: lexicon-driven regex tagging of chunks.")
    parser.add_argument(
        "--input",
        type=str,
        default="data/derived/step3_chunks_spacy/chunks_spacy.jsonl",
        help="Path to chunks_spacy.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/derived/step4_chunks_tagged",
        help="Directory for tagged output.",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "chunks_normalized_tagged.jsonl"

    # Load and compile lexicons once.
    lexicons: Dict[str, re.Pattern] = {}
    for name, rel_path in LEXICON_SPECS:
        p = Path(rel_path)
        terms = load_terms(p)
        lexicons[name] = compile_lexicon_regex(terms)

    rows = list(read_jsonl(in_path))

    with out_path.open("w", encoding="utf-8") as out:
        for r in tqdm(rows, desc="Step 4: lexicon tagging", unit="chunk"):
            text = r["text"]
            text_norm = normalize_for_match(text)

            tags: Dict[str, Any] = {}
            for name, rx in lexicons.items():
                matches = rx.findall(text_norm)
                tags[f"has_{name}"] = len(matches) > 0
                tags[f"n_{name}"] = len(matches)

            out_row = {
                "chunk_id": r["chunk_id"],
                "doc_id": r["doc_id"],
                "chunk_index": r["chunk_index"],
                "char_start": r["char_start"],
                "char_end": r["char_end"],
                "n_chars": r["n_chars"],
                "text": r["text"],
                "text_norm": text_norm,
                "tags": tags,
            }
            out.write(json.dumps(out_row, ensure_ascii=False) + "\n")

    print(f"Step 4 complete | chunks={len(rows)}")
    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
