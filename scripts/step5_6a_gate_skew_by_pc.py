#!/usr/bin/env python3
"""
Step 5.6a: Compare mean PCA scores for normativity-gated vs excluded chunks.

Inputs:
- data/derived/step5_pca_full/pca_full_scores.csv
- data/derived/step4_5_normativity_gate/chunks_normative_primary.jsonl

Output:
- data/derived/step5_6_pca_followthrough/gate_skew_by_pc.csv
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Set


SCORES = Path("data/derived/step5_pca_full/pca_full_scores.csv")
PRIMARY = Path("data/derived/step4_5_normativity_gate/chunks_normative_primary.jsonl")
OUT = Path("data/derived/step5_6_pca_followthrough/gate_skew_by_pc.csv")


def load_primary_ids(path: Path) -> Set[str]:
    ids: Set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            ids.add(r["chunk_id"])
    return ids


def main() -> None:
    primary_ids = load_primary_ids(PRIMARY)

    with SCORES.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        pc_cols = [c for c in (reader.fieldnames or []) if c.startswith("pc")]

        sums_p = {pc: 0.0 for pc in pc_cols}
        sums_x = {pc: 0.0 for pc in pc_cols}
        n_p = 0
        n_x = 0

        for r in reader:
            is_primary = r["chunk_id"] in primary_ids
            if is_primary:
                n_p += 1
                for pc in pc_cols:
                    sums_p[pc] += float(r[pc])
            else:
                n_x += 1
                for pc in pc_cols:
                    sums_x[pc] += float(r[pc])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["pc", "mean_primary", "mean_excluded", "mean_diff_primary_minus_excluded", "n_primary", "n_excluded"],
        )
        w.writeheader()
        for pc in pc_cols:
            mp = sums_p[pc] / n_p if n_p else 0.0
            mx = sums_x[pc] / n_x if n_x else 0.0
            w.writerow(
                {
                    "pc": pc,
                    "mean_primary": mp,
                    "mean_excluded": mx,
                    "mean_diff_primary_minus_excluded": mp - mx,
                    "n_primary": n_p,
                    "n_excluded": n_x,
                }
            )

    print(f"Wrote: {OUT}")


if __name__ == "__main__":
    main()
