#!/usr/bin/env python3
"""
Step 6.2a — Reconstruct doc_id → country from raw folder provenance.

Rationale
---------
Chunk artifacts (Step 4+) preserve doc_id but not country metadata. The raw
 corpus is organized under country folders:
  data/raw/corpus_raw/AI Policies/<Country>/...

This script rebuilds a deterministic mapping by re-extracting and normalizing
text from raw PDFs and hashing the normalized text using the same sha16 scheme
used in Step 1 normalized texts.

Output
------
data/derived/step6_chunks_edu/doc_id_to_country.csv
"""

from pathlib import Path
import csv
import hashlib
import re

from tqdm import tqdm
from pdfminer.high_level import extract_text

RAW_ROOT = Path("data/raw/corpus_raw/AI Policies")
OUT_CSV = Path("data/derived/step6_chunks_edu/doc_id_to_country.csv")

WS_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    return WS_RE.sub(" ", text.lower()).strip()


def sha16(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
def iter_pdfs(root: Path):
    # Skip non-country files at the root (e.g., AI on Education.xlsx)
    for p in root.rglob("*.pdf"):
        try:
            rel = p.relative_to(root)
        except Exception:
            continue
        if len(rel.parts) < 2:
            continue
        yield p


def main():
    pdfs = list(iter_pdfs(RAW_ROOT))

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    rows = 0
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["doc_id", "country", "raw_path"])

        for pdf in tqdm(pdfs, desc="Building doc_id→country map", unit="pdf"):
            # Country is the first folder under RAW_ROOT
            country = pdf.relative_to(RAW_ROOT).parts[0]

            try:
                raw_text = extract_text(str(pdf)) or ""
            except Exception:
                continue

            raw_text = raw_text.strip()
            if not raw_text:
                continue

            doc_id = sha16(normalize_text(raw_text))
            w.writerow([doc_id, country, str(pdf)])
            rows += 1

    print(f"Wrote {rows} rows → {OUT_CSV}")


if __name__ == "__main__":
    main()
