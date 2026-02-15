#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import pandas as pd


HEADING_MAX_CHARS = 140

UMBRELLA_HEADING_KWS = [
    "sanction", "sanctions",
    "penalty", "penalties",
    "fine", "fines",
    "administrative fine", "administrative fines",
    "offence", "offences", "offense", "offenses",
    "enforcement",
    "liability", "liable",
    "non-compliance", "noncompliance",
    "breach",
    "prosecution", "prosecutions",
    "revocation", "suspension", "withdrawal",
]

UMBRELLA_TEXT_CUES = [
    "shall be liable",
    "shall be punished",
    "subject to a fine",
    "failure to comply",
    "non-compliance",
    "administrative fine",
    "administrative fines",
    "liable",
    "liability",
    "penalty",
    "penalties",
    "sanction",
    "sanctions",
    "offence",
    "offense",
    "revocation",
    "suspension",
    "withdrawal",
]

KW_RE = re.compile("|".join(re.escape(k) for k in UMBRELLA_HEADING_KWS), flags=re.IGNORECASE)
CUE_RE = re.compile("|".join(re.escape(k) for k in UMBRELLA_TEXT_CUES), flags=re.IGNORECASE)


def looks_like_heading(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    if len(s) > HEADING_MAX_CHARS:
        return False
    if s.endswith(":"):
        return True
    if s.isupper() and len(s) >= 5:
        return True
    words = re.findall(r"[A-Za-z]+", s)
    if 1 <= len(words) <= 12:
        caps = sum(1 for w in words if w and w[0].isupper())
        return (caps / max(len(words), 1)) >= 0.7
    return False


def has_umbrella_kw(s: str) -> bool:
    return bool(KW_RE.search(s))


def extract_sections_from_text(doc_id: str, text: str) -> list[dict]:
    lines = [ln.rstrip("\n") for ln in text.splitlines()]
    headings = []
    for i, ln in enumerate(lines):
        if looks_like_heading(ln) and has_umbrella_kw(ln):
            headings.append((i, ln.strip().rstrip(":")))

    out = []
    for j, (i, htxt) in enumerate(headings):
        start = i + 1
        end = len(lines)
        if j + 1 < len(headings):
            end = headings[j + 1][0]
        body = "\n".join(lines[start:end]).strip()
        if not body:
            continue
        if not CUE_RE.search(body) and not has_umbrella_kw(htxt):
            continue
        out.append(
            {
                "doc_id": doc_id,
                "section_id": f"{doc_id}__U{j:03d}",
                "heading_text": htxt,
                "section_text": body,
                "start_line": i,
                "end_line": end - 1,
            }
        )
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs-dir", default="data/derived/step1_texts/docs_normalized_text")
    ap.add_argument("--out-dir", default="data/derived/step8_umbrella_sections_from_docs")
    ap.add_argument("--out-format", choices=["parquet", "csv"], default="parquet")
    ap.add_argument("--report-every", type=int, default=200)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.time()

    docs_dir = Path(args.docs_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    files = sorted(docs_dir.glob("*.txt"))
    n = 0
    for fp in files:
        doc_id = fp.stem
        text = fp.read_text(encoding="utf-8", errors="ignore")
        rows.extend(extract_sections_from_text(doc_id, text))
        n += 1
        if args.report_every and n % args.report_every == 0:
            print(f"[extract] docs={n:,}/{len(files):,} sections={len(rows):,} elapsed={(time.time()-t0)/60:.1f}m")

    df = pd.DataFrame(rows)
    out_base = out_dir / "umbrella_sections"
    if args.out_format == "parquet":
        df.to_parquet(out_base.with_suffix(".parquet"), index=False)
    else:
        df.to_csv(out_base.with_suffix(".csv"), index=False, escapechar="\\")

    print(f"[ok] docs={len(files):,} umbrella_sections={len(df):,} elapsed={(time.time()-t0)/60:.1f}m")
    print(f"[ok] wrote: {out_base.with_suffix('.' + args.out_format)}")


if __name__ == "__main__":
    main()
