#!/usr/bin/env python3
"""
Step 6.2a — Reconstruct doc_id → country using raw-PDF byte hashing.
I made a mistake on the last one. I figured we needed to get country names from re-imported Pdfs. I was wrong. This is a simplified process. 

doc_id definition (frozen):
  doc_id = sha256(pdf_bytes)[:16]
This matches scripts/step1_min_pdf_to_txt.py exactly.

Country provenance:
  data/raw/corpus_raw/AI Policies/<Country>/.../*.pdf

Output:
  data/derived/step6_chunks_edu/doc_id_to_country.csv
"""

from pathlib import Path
import csv
import hashlib
from tqdm import tqdm

RAW_ROOT = Path("data/raw/corpus_raw/AI Policies")
OUT_CSV = Path("data/derived/step6_chunks_edu/doc_id_to_country.csv")


def sha16(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def iter_pdfs(root: Path):
    for p in root.rglob("*.pdf"):
        # ensure it's under a country folder (root/<country>/...)
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

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["doc_id", "country", "raw_path"])

        for pdf in tqdm(pdfs, desc="Hashing PDFs for doc_id→country", unit="pdf"):
            country = pdf.relative_to(RAW_ROOT).parts[0]
            doc_id = sha16(pdf)
            w.writerow([doc_id, country, str(pdf)])

    print(f"Wrote {len(pdfs)} mappings → {OUT_CSV}")


if __name__ == "__main__":
    main()
