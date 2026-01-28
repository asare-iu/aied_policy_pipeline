
#!/usr/bin/env python3
"""
Minimal Step 1: Convert PDFs to normalized UTF-8 text with a progress bar.

Input:
- data/raw/ai_policies_raw/**/*.pdf

Output:
- data/derived/step1_texts/docs_normalized_text/<sha16>.txt
- data/derived/step1_texts/errors.csv
"""

from pathlib import Path
import hashlib
import unicodedata
import csv

from tqdm import tqdm
from pypdf import PdfReader


INPUT_ROOT = Path("data/raw/ai_policies_raw")
OUTPUT_ROOT = Path("data/derived/step1_texts")
TEXT_DIR = OUTPUT_ROOT / "docs_normalized_text"
ERRORS_CSV = OUTPUT_ROOT / "errors.csv"


def sha16(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.rstrip() for line in text.split("\n")).strip()


def extract_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def main() -> None:
    TEXT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(p for p in INPUT_ROOT.rglob("*.pdf") if p.is_file())

    with ERRORS_CSV.open("w", newline="", encoding="utf-8") as ef:
        writer = csv.writer(ef)
        writer.writerow(["rel_path", "error"])

        for p in tqdm(pdfs, desc="Step 1: PDFs → TXT", unit="file"):
            try:
                doc_id = sha16(p)
                text = normalize(extract_pdf(p))
                (TEXT_DIR / f"{doc_id}.txt").write_text(text, encoding="utf-8")
            except Exception as e:
                writer.writerow([str(p.relative_to(INPUT_ROOT)), str(e)])


if __name__ == "__main__":
    main()
