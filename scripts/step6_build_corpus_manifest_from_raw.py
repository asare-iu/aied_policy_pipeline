#!/usr/bin/env python3
"""
Build full-corpus manifest for Step 6b.

Creates: data/manifests/corpus_manifest.csv
Columns: doc_id,title,country,raw_path

- title = PDF filename stem
- country = first directory under RAW_ROOT
- doc_id = sha256(normalized extracted text)[:16]
"""

import csv
import hashlib
import re
from pathlib import Path

from pdfminer.high_level import extract_text
from tqdm import tqdm

RAW_ROOT = Path("data/raw/corpus_raw/AI Policies")
OUT = Path("data/manifests/corpus_manifest.csv")
WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    return WS.sub(" ", (text or "").lower()).strip()


def sha16(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)

    pdfs = [p for p in RAW_ROOT.rglob("*.pdf") if len(p.relative_to(RAW_ROOT).parts) >= 2]

    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["doc_id", "title", "country", "raw_path"])

        for pdf in tqdm(pdfs, desc="Building corpus manifest"):
            country = pdf.relative_to(RAW_ROOT).parts[0]
            title = pdf.stem

            try:
                text = extract_text(str(pdf))
            except Exception:
                continue

            if not text.strip():
                continue

            doc_id = sha16(normalize(text))
            w.writerow([doc_id, title, country, str(pdf)])

    print(f"Wrote manifest → {OUT}")


if __name__ == "__main__":
    main()
