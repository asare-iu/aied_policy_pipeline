#!/usr/bin/env python3
"""
Step 7a: Build title-based education chunk subset.

Inputs:
- chunks_edu.jsonl (education-gated chunks)
- doc_ids_title_tier1plus2.txt (curated title-education doc IDs)

Output:
- chunks_title_edu.jsonl (only chunks whose doc_id is in title list)
"""

import argparse
import json
from tqdm import tqdm


def load_doc_ids(path):
    with open(path, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--chunks_path",
        default="data/derived/step4_chunks_tagged/chunks_normalized_tagged.jsonl",
        help="Path to education-gated chunks JSONL",
    )
    ap.add_argument(
        "--doc_ids_path",
        default="data/derived/step6b_title_edu/doc_ids_title_tier1plus2.txt",
        help="Curated title-education doc_id list",
    )
    ap.add_argument(
        "--out_path",
        default="data/derived/step7_chunks_title_edu/chunks_title_edu_allchunks.jsonl",
        help="Output JSONL path",
    )
    args = ap.parse_args()

    doc_ids = load_doc_ids(args.doc_ids_path)
    print(f"[step7] Loaded {len(doc_ids)} title-education doc_ids")

    kept = 0
    total = 0

    with open(args.chunks_path, "r", encoding="utf-8") as fin, \
         open(args.out_path, "w", encoding="utf-8") as fout:

        for line in tqdm(fin, desc="Filtering chunks"):
            total += 1
            if not line.strip():
                continue

            obj = json.loads(line)
            if str(obj.get("doc_id")) in doc_ids:
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                kept += 1

    print(f"[step7] Finished")
    print(f"[step7] Input chunks: {total}")
    print(f"[step7] Kept chunks: {kept}")
    print(f"[step7] Output written to: {args.out_path}")


if __name__ == "__main__":
    main()
