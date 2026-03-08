#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import pandas as pd


def load_doc_ids_from_jsonl(path: Path) -> set[str]:
    s: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            o = json.loads(line)
            doc_id = str(o.get("doc_id", "")).strip()
            if doc_id:
                s.add(doc_id)
    return s


def write_sections_as_jsonl(umbrella_sections_parquet: Path, out_jsonl: Path, edu_docs: set) -> int:
    import json
    import pandas as pd

    MAX_CHARS = 950_000  # always below spaCy default 1,000,000

    df = pd.read_parquet(umbrella_sections_parquet)

    if "section_text" in df.columns:
        text_col = "section_text"
    elif "text" in df.columns:
        text_col = "text"
    else:
        raise ValueError(f"Missing section text col. cols={df.columns.tolist()}")

    if "section_id" in df.columns:
        sec_id_col = "section_id"
    elif "block_id" in df.columns:
        sec_id_col = "block_id"
    else:
        sec_id_col = None

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with out_jsonl.open("w", encoding="utf-8") as f:
        for _, r in df.iterrows():
            doc_id = str(r["doc_id"])
            if doc_id not in edu_docs:
                continue

            raw = r.get(text_col, None)
            if raw is None:
                continue
            text = str(raw)

            base_section_id = str(r[sec_id_col]) if sec_id_col else "section"

            part = 0
            start = 0
            L = len(text)
            while start < L:
                piece = text[start : start + MAX_CHARS]
                chunk_id = f"{doc_id}__umbrella_{base_section_id}__part{part:03d}"
                obj = {
                    "doc_id": doc_id,
                    "chunk_id": chunk_id,
                    "text": piece,
                    "source": "umbrella_section",
                    "umbrella_section_id": base_section_id,
                    "umbrella_part_index": part,
                    "n_chars": len(piece),
                }
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
                n += 1
                part += 1
                start += MAX_CHARS

    return n


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--umbrella-sections", default="data/derived/step8_umbrella_sections_from_docs/umbrella_sections.parquet")
    ap.add_argument("--edu-chunks-jsonl", default="data/derived/step6_chunks_edu/chunks_edu.jsonl")
    ap.add_argument("--out-dir", default="data/derived/step8_umbrella_igt")
    ap.add_argument("--run-igt", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    edu_docs = load_doc_ids_from_jsonl(Path(args.edu_chunks_jsonl))
    print(f"[ok] edu_docs={len(edu_docs):,}")

    sections_jsonl = out_dir / "umbrella_sections_as_chunks.jsonl"
    n = write_sections_as_jsonl(Path(args.umbrella_sections), sections_jsonl, edu_docs)
    print(f"[ok] wrote sections_jsonl={sections_jsonl} rows={n:,}")

    if args.run_igt:
        subprocess.run(
    [
        "python", "scripts/step8_3_igt_parsing.py",
        "--input", str(out_dir / "sentences_full.parquet"),
        "--out-dir", str(out_dir),
        "--report-every", "200",
    ],
    check=True,
)


        subprocess.run(
            ["python", "scripts/step8_3_igt_parsing.py", "--in-dir", str(out_dir), "--out-dir", str(out_dir)],
            check=True,
        )

    igt_path = out_dir / "igt_statements_full.parquet"
    if not igt_path.exists():
        print(f"[stop] missing: {igt_path}")
        return

    igt = pd.read_parquet(igt_path)
    rules = igt[igt["statement_type_candidate"] == "rule_candidate"].copy()
    rules.to_parquet(out_dir / "umbrella_section_rule_candidates.parquet", index=False)

    summary = pd.DataFrame(
        {
            "umbrella_section_statements": [len(igt)],
            "umbrella_section_rule_candidates": [len(rules)],
            "umbrella_section_docs": [igt["doc_id"].nunique()],
        }
    )
    summary.to_csv(out_dir / "summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"[ok] wrote: {out_dir/'umbrella_section_rule_candidates.parquet'}")


if __name__ == "__main__":
    main()
