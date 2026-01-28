#!/usr/bin/env python3
"""
Step 4.6: Build a random audit sample of chunks excluded by the primary normativity gate.

Inputs:
- data/derived/step4_chunks_tagged/chunks_normalized_tagged.jsonl
- data/derived/step4_5_normativity_gate/chunks_normative_primary.jsonl

Outputs:
- data/derived/step4_6_audit_sample/chunks_unmarked_audit_sample.jsonl
- data/derived/step4_6_audit_sample/sample_summary.json

Design:
- Primary set is defined by chunk_id membership in chunks_normative_primary.jsonl.
- Sample is drawn uniformly at random from the complement.
- Fixed RNG seed for reproducibility.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, Set

from tqdm import tqdm


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_primary_ids(path: Path) -> Set[str]:
    ids: Set[str] = set()
    for r in read_jsonl(path):
        ids.add(r["chunk_id"])
    return ids


def main() -> int:
    parser = argparse.ArgumentParser(description="Step 4.6: sample excluded chunks for recall audit.")
    parser.add_argument("--tagged", type=str, default="data/derived/step4_chunks_tagged/chunks_normalized_tagged.jsonl")
    parser.add_argument("--primary", type=str, default="data/derived/step4_5_normativity_gate/chunks_normative_primary.jsonl")
    parser.add_argument("--output-dir", type=str, default="data/derived/step4_6_audit_sample")
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()

    tagged_path = Path(args.tagged)
    primary_path = Path(args.primary)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_sample = out_dir / "chunks_unmarked_audit_sample.jsonl"
    out_summary = out_dir / "sample_summary.json"

    primary_ids = load_primary_ids(primary_path)

    excluded: list[Dict[str, Any]] = []
    total = 0
    for r in tqdm(read_jsonl(tagged_path), desc="Step 4.6: collecting excluded", unit="chunk"):
        total += 1
        if r["chunk_id"] not in primary_ids:
            excluded.append(r)

    rng = random.Random(args.seed)
    n = min(args.n, len(excluded))
    sample = rng.sample(excluded, n)

    with out_sample.open("w", encoding="utf-8") as out:
        for r in sample:
            out.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = {
        "total_chunks": total,
        "primary_chunks": len(primary_ids),
        "excluded_chunks": len(excluded),
        "sample_n": n,
        "seed": args.seed,
        "source_tagged": str(tagged_path),
        "source_primary": str(primary_path),
    }
    out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Step 4.6 complete | excluded={len(excluded)} sample_n={n}")
    print(f"Wrote: {out_sample}")
    print(f"Wrote: {out_summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
