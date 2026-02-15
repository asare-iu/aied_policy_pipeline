#!/usr/bin/env python3
"""
Step 5.6b: Aggregate mean PCA scores by country folder.

Inputs:
- data/raw/ai_policies_raw (PDFs in country-organized folders)
- data/derived/step5_pca_full/pca_full_scores.csv

Output:
- data/derived/step5_6_pca_followthrough/country_pc_means.csv
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from collections import defaultdict
from typing import Dict


RAW = Path("data/raw/ai_policies_raw")
SCORES = Path("data/derived/step5_pca_full/pca_full_scores.csv")
OUT = Path("data/derived/step5_6_pca_followthrough/country_pc_means.csv")


def sha16(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def main() -> None:
    # doc_id -> country
    doc_to_country: Dict[str, str] = {}
    for p in RAW.rglob("*"):
        if p.is_file() and p.suffix.lower() == ".pdf":
            rel = p.relative_to(RAW)
            country = rel.parts[0] if len(rel.parts) >= 2 else ""
            doc_to_country[sha16(p)] = country

    with SCORES.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        pc_cols = [c for c in (reader.fieldnames or []) if c.startswith("pc")]

        sums = defaultdict(lambda: {pc: 0.0 for pc in pc_cols})
        counts = defaultdict(int)

        for r in reader:
            country = doc_to_country.get(r["doc_id"], "")
            if not country:
                continue
            counts[country] += 1
            for pc in pc_cols:
                sums[country][pc] += float(r[pc])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["country", "n_chunks"] + pc_cols)
        w.writeheader()
        for country in sorted(counts.keys()):
            row = {"country": country, "n_chunks": counts[country]}
            for pc in pc_cols:
                row[pc] = sums[country][pc] / counts[country]
            w.writerow(row)

    print(f"Wrote: {OUT}")


if __name__ == "__main__":
    main()
