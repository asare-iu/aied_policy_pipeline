import argparse
from pathlib import Path
import textwrap

def chunk_text(text, max_words=1000, overlap=100):
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + max_words
        chunk = words[start:end]
        chunks.append(" ".join(chunk))
        start = end - overlap

    return chunks

def main(country, text_dir, out_dir, max_words, overlap):
    text_dir = Path(text_dir)
    out_dir = Path(out_dir) / country
    out_dir.mkdir(parents=True, exist_ok=True)

    for txt_file in text_dir.glob("*.txt"):
        text = txt_file.read_text(encoding="utf-8", errors="ignore")
        chunks = chunk_text(text, max_words, overlap)

        for i, chunk in enumerate(chunks, start=1):
            chunk_file = out_dir / f"{txt_file.stem}_chunk{i:03d}.txt"
            chunk_file.write_text(chunk, encoding="utf-8")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--country", required=True)
    parser.add_argument("--text_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--max_words", type=int, default=1000)
    parser.add_argument("--overlap", type=int, default=100)

    args = parser.parse_args()
    main(
        args.country,
        args.text_dir,
        args.out_dir,
        args.max_words,
        args.overlap
    )

