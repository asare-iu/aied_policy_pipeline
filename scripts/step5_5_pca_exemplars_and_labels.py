#!/usr/bin/env python3
"""
Step 5.5: Prepare human interpretation artifacts for PCA components.

Inputs:
- data/derived/step5_pca_full/pca_full_scores.csv
- data/derived/step5_pca_full/pca_full_top_terms.csv
- data/derived/step4_chunks_tagged/chunks_normalized_tagged.jsonl

Outputs:
- data/derived/step5_5_pca_interpretation/pca_pc_exemplars.jsonl
- data/derived/step5_5_pca_interpretation/pca_pc_labels_template.csv

Design:
- For each PC:
    * take top N positive and top N negative scoring chunks
    * attach top positive/negative terms
- Human coder assigns labels ex post.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict

TOP_N = 15  # exemplars per direction per PC


def load_chunks(path: Path) -> Dict[str, Dict[str, Any]]:
    out = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            out[r["chunk_id"]] = r
    return out


def load_scores(path: Path) -> Dict[int, List[Dict[str, Any]]]:
    pcs: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            meta = {
                "chunk_id": r["chunk_id"],
                "doc_id": r["doc_id"],
                "chunk_index": int(r["chunk_index"]),
            }
            for k, v in r.items():
                if k.startswith("pc"):
                    pc = int(k.replace("pc", ""))
                    pcs[pc].append({**meta, "score": float(v)})
    return pcs


def load_top_terms(path: Path) -> Dict[int, Dict[str, List[Dict[str, Any]]]]:
    out: Dict[int, Dict[str, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            pc = int(r["pc"])
            out[pc][r["direction"]].append(r)
    return out


def main() -> None:
    base = Path("data/derived")
    scores_path = base / "step5_pca_full/pca_full_scores.csv"
    terms_path = base / "step5_pca_full/pca_full_top_terms.csv"
    chunks_path = base / "step4_chunks_tagged/chunks_normalized_tagged.jsonl"

    out_dir = base / "step5_5_pca_interpretation"
    out_dir.mkdir(parents=True, exist_ok=True)

    exemplars_out = out_dir / "pca_pc_exemplars.jsonl"
    labels_out = out_dir / "pca_pc_labels_template.csv"

    chunks = load_chunks(chunks_path)
    pcs = load_scores(scores_path)
    terms = load_top_terms(terms_path)

    with exemplars_out.open("w", encoding="utf-8") as out:
        for pc, rows in pcs.items():
            rows_sorted = sorted(rows, key=lambda x: x["score"])
            neg = rows_sorted[:TOP_N]
            pos = rows_sorted[-TOP_N:][::-1]

            record = {
                "pc": pc,
                "top_terms_positive": terms.get(pc, {}).get("positive", [])[:20],
                "top_terms_negative": terms.get(pc, {}).get("negative", [])[:20],
                "positive_exemplars": [
                    {
                        "chunk_id": r["chunk_id"],
                        "score": r["score"],
                        "text": chunks[r["chunk_id"]]["text"][:800],
                    }
                    for r in pos
                ],
                "negative_exemplars": [
                    {
                        "chunk_id": r["chunk_id"],
                        "score": r["score"],
                        "text": chunks[r["chunk_id"]]["text"][:800],
                    }
                    for r in neg
                ],
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")

    # CSV template for human labeling
    with labels_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "pc",
                "label",
                "short_description",
                "notes",
                "coder",
            ],
        )
        writer.writeheader()
        for pc in sorted(pcs.keys()):
            writer.writerow(
                {
                    "pc": pc,
                    "label": "",
                    "short_description": "",
                    "notes": "",
                    "coder": "",
                }
            )

    print("Step 5.5 complete")
    print(f"Wrote: {exemplars_out}")
    print(f"Wrote: {labels_out}")


if __name__ == "__main__":
    main()
