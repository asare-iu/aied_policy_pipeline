#!/usr/bin/env python3
"""
Step 8.1 — Sentence segmentation (FULL CORPUS baseline)

Input:
  data/derived/step4_chunks_tagged/chunks_normalized_tagged.jsonl

Output:
  data/derived/step8_igt_full/sentences_full.parquet
  data/derived/step8_igt_full/_runtime_params.json  (updated/created)

Deterministic + provenance-preserving.
Includes a progress/ETA tracker.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import spacy


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INPUT = PROJECT_ROOT / "data/derived/step4_chunks_tagged/chunks_normalized_tagged.jsonl"
DEFAULT_OUT_DIR = PROJECT_ROOT / "data/derived/step8_igt_full"
DEFAULT_OUTPUT = DEFAULT_OUT_DIR / "sentences_full.parquet"
DEFAULT_RUNTIME = DEFAULT_OUT_DIR / "_runtime_params.json"


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(block_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def get_git_hash() -> Optional[str]:
    import subprocess

    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT))
        return out.decode("utf-8").strip()
    except Exception:
        return None


def iter_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"JSON decode error in {path} at line {line_no}: {e}") from e


def stable_sentence_id(doc_id: str, chunk_id: str, sent_index_in_chunk: int) -> str:
    raw = f"{doc_id}::{chunk_id}::{sent_index_in_chunk}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def load_spacy(model: str) -> "spacy.language.Language":
    try:
        nlp = spacy.load(model, disable=["ner", "tagger", "lemmatizer", "attribute_ruler"])
    except OSError as e:
        raise RuntimeError(
            f"spaCy model '{model}' not found. Install it in the venv:\n"
            f"  python -m spacy download {model}"
        ) from e

    # Ensure sentence boundaries exist.
    if "parser" not in nlp.pipe_names and "senter" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")
    return nlp


def segment_sentences(nlp, text: str) -> List[Tuple[int, int, str]]:
    doc = nlp(text)
    out: List[Tuple[int, int, str]] = []
    for s in doc.sents:
        start = int(s.start_char)
        end = int(s.end_char)
        out.append((start, end, text[start:end]))
    return out


def ensure_out_dir(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)


def update_runtime_params(runtime_path: Path, updates: Dict) -> None:
    if runtime_path.exists():
        try:
            data = json.loads(runtime_path.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    else:
        data = {}

    for k, v in updates.items():
        data[k] = v

    runtime_path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def fmt_hhmmss(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return time.strftime("%H:%M:%S", time.gmtime(seconds))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default=str(DEFAULT_INPUT))
    ap.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT))
    ap.add_argument("--runtime", type=str, default=str(DEFAULT_RUNTIME))
    ap.add_argument("--spacy-model", type=str, default="en_core_web_sm")
    ap.add_argument("--max-rows", type=int, default=0, help="Smoke test only. 0 = full run.")
    ap.add_argument("--report-every", type=int, default=2000)
    args = ap.parse_args()

    in_path = Path(args.input).resolve()
    out_path = Path(args.output).resolve()
    runtime_path = Path(args.runtime).resolve()
    out_dir = out_path.parent

    if not in_path.exists():
        raise FileNotFoundError(f"Input not found: {in_path}")

    ensure_out_dir(out_dir)

    # Pre-count input lines for percent/ETA (fast enough; deterministic)
    with in_path.open("r", encoding="utf-8") as f:
        total_chunks = sum(1 for _ in f)

    nlp = load_spacy(args.spacy_model)

    records: List[Dict] = []
    n_in = 0
    n_sents = 0

    start_time = time.time()

    for row in iter_jsonl(in_path):
        n_in += 1
        if args.max_rows and n_in > args.max_rows:
            break

        doc_id = str(row.get("doc_id", ""))
        chunk_id = str(row.get("chunk_id", ""))
        chunk_index = int(row.get("chunk_index", -1))
        char_start = int(row.get("char_start", -1))
        char_end = int(row.get("char_end", -1))
        tags = row.get("tags", None)

        text = row.get("text", "")
        if not isinstance(text, str):
            text = "" if text is None else str(text)

        sents = segment_sentences(nlp, text)

        for i, (s_start, s_end, s_text) in enumerate(sents):
            sent_id = stable_sentence_id(doc_id, chunk_id, i)
            records.append(
                {
                    "sentence_id": sent_id,
                    "doc_id": doc_id,
                    "chunk_id": chunk_id,
                    "chunk_index": chunk_index,
                    "chunk_char_start": char_start,
                    "chunk_char_end": char_end,
                    "sentence_index_in_chunk": i,
                    "sentence_char_start_in_chunk": s_start,
                    "sentence_char_end_in_chunk": s_end,
                    "sentence_text": s_text,
                    "tags": tags,
                }
            )
            n_sents += 1

        if (args.report_every and n_in % args.report_every == 0) or n_in == total_chunks:
            now = time.time()
            elapsed = now - start_time
            rate = n_in / elapsed if elapsed > 0 else 0.0
            remaining = total_chunks - n_in
            eta = remaining / rate if rate > 0 else 0.0
            pct = (n_in / total_chunks) * 100 if total_chunks else 0.0
            print(
                f"[step8_1] {n_in:,}/{total_chunks:,} ({pct:5.1f}%) | "
                f"elapsed {fmt_hhmmss(elapsed)} | ETA {fmt_hhmmss(eta)}",
                flush=True,
            )

    df = pd.DataFrame.from_records(records)
    sort_cols = ["doc_id", "chunk_index", "chunk_id", "sentence_index_in_chunk"]
    df = df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

    df.to_parquet(out_path, index=False)

    runtime_updates = {
        "step8_1_sentence_segmentation": {
            "timestamp_utc": dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
            "git_hash": get_git_hash(),
            "input_path": str(in_path),
            "input_sha256": sha256_file(in_path),
            "output_path": str(out_path),
            "spacy_model": args.spacy_model,
            "spacy_version": spacy.__version__,
            "python": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
            "n_input_chunks": int(n_in),
            "n_output_sentences": int(len(df)),
            "columns": list(df.columns),
        }
    }
    update_runtime_params(runtime_path, runtime_updates)

    total_elapsed = time.time() - start_time
    print(f"[step8_1] done. chunks={n_in:,} sentences={len(df):,} | total time {fmt_hhmmss(total_elapsed)}")
    print(f"[step8_1] wrote: {out_path}")
    print(f"[step8_1] runtime: {runtime_path}")


if __name__ == "__main__":
    main()
