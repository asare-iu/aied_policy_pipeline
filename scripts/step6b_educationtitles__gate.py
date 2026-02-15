#!/usr/bin/env python3
"""
Step 6b: Title-based "education" document subsets (Tier 1 and Tier 1+2).

Outputs (default out_dir: data/derived/step6b_title_edu/):
- doc_ids_title_tier1.txt
- doc_ids_title_tier1plus2.txt
- subset_report.md

Notes:
- Titles are assumed to be in English.
- This is intentionally title-only as an independent robustness check vs. chunk-based education gate.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Tuple

import pandas as pd
from tqdm import tqdm


# ----------------------------
# Regex tiers (English only)
# ----------------------------

TIER1_PATTERNS = [
    r"\beducation(al)?\b",
    r"\bschool(s|ing)?\b",
    r"\bteacher(s)?\b",
    r"\bteaching\b",
    r"\bstudent(s)?\b",
    r"\bcurricul(a|um|ar)\b",
    r"\bassess(ment|ments|ing)?\b",
    r"\bexam(s|ination|inations)?\b",
    r"\btest(ing|s)?\b",
    r"\buniversity|universities\b",
    r"\bhigher education\b",
    r"\btertiary education\b",
    r"\bcollege(s)?\b",
    r"\bvocational\b",
    r"\btechnical education\b",
    r"\btvet\b",
    r"\bapprentice(ship|ships)?\b",
    r"\bpedagog(y|ical)\b",
    r"\bclassroom(s)?\b",
    r"\blearning outcomes?\b",
]

TIER2_PATTERNS = [
    r"\bskills?\b",
    r"\bdigital skills?\b",
    r"\btraining\b",
    r"\bupskilling\b",
    r"\breskilling\b",
    r"\bworkforce development\b",
    r"\bcapacity building\b",
    r"\bai literacy\b",
    r"\bdata literacy\b",
    r"\bdigital literacy\b",
    r"\bmedia literacy\b",
    r"\bcompetenc(e|y|ies)\b",
    r"\bqualifications?\b",
    r"\btalent development\b",
    r"\blifelong learning\b",
    r"\badult learning\b",
    r"\bemployability\b",
]


def compile_patterns(patterns: List[str]) -> List[Tuple[str, re.Pattern]]:
    return [(p, re.compile(p, re.IGNORECASE)) for p in patterns]


def match_patterns(title: str, compiled: List[Tuple[str, re.Pattern]]) -> List[str]:
    hits: List[str] = []
    for label, rx in compiled:
        if rx.search(title):
            hits.append(label)
    return hits


# ----------------------------
# IO / Discovery
# ----------------------------

def find_titles_file(search_root: str = "data") -> Tuple[str, str, str]:
    """
    Try to auto-find a titles file under search_root that includes doc_id + title-like column.
    Returns: (path, doc_id_col, title_col)
    """
    candidates: List[str] = []
    for ext in ("csv", "tsv", "parquet"):
        candidates.extend(glob.glob(os.path.join(search_root, "**", f"*title*.{ext}"), recursive=True))
        candidates.extend(glob.glob(os.path.join(search_root, "**", f"*titles*.{ext}"), recursive=True))

    # If nothing matched "title" in filename, fall back to scanning all supported files (slower).
    if not candidates:
        for ext in ("csv", "tsv", "parquet"):
            candidates.extend(glob.glob(os.path.join(search_root, "**", f"*.{ext}"), recursive=True))

    scored: List[Tuple[int, str, str, str]] = []
    for p in candidates:
        try:
            if p.endswith(".parquet"):
                df_head = pd.read_parquet(p)
            elif p.endswith(".tsv"):
                df_head = pd.read_csv(p, sep="\t", nrows=50)
            else:
                df_head = pd.read_csv(p, nrows=50)
        except Exception:
            continue

        cols_lower = {c.lower(): c for c in df_head.columns}

        # Accept a few common variants
        doc_id_col = None
        for key in ("doc_id", "docid", "document_id"):
            if key in cols_lower:
                doc_id_col = cols_lower[key]
                break

        title_col = None
        for key in ("title", "document_title", "doc_title", "name"):
            if key in cols_lower:
                title_col = cols_lower[key]
                break

        if doc_id_col and title_col:
            base = os.path.basename(p).lower()
            score = 10
            if "title" in base or "titles" in base:
                score += 3
            scored.append((score, p, doc_id_col, title_col))

    if not scored:
        raise FileNotFoundError(
            "Could not auto-find a titles file. Provide --titles_path explicitly and (if needed) "
            "--doc_id_col and --title_col."
        )

    scored.sort(reverse=True, key=lambda x: x[0])
    _, best_path, best_doc_id_col, best_title_col = scored[0]
    return best_path, best_doc_id_col, best_title_col


def read_titles(path: str, doc_id_col: str, title_col: str) -> pd.DataFrame:
    if path.endswith(".parquet"):
        df = pd.read_parquet(path, columns=[doc_id_col, title_col])
    elif path.endswith(".tsv"):
        df = pd.read_csv(path, sep="\t", usecols=[doc_id_col, title_col])
    else:
        df = pd.read_csv(path, usecols=[doc_id_col, title_col])

    df = df.rename(columns={doc_id_col: "doc_id", title_col: "title"})
    df["doc_id"] = df["doc_id"].astype(str)
    df["title"] = df["title"].fillna("").astype(str)
    return df


def load_doc2country(map_path: str) -> Dict[str, str]:
    if not os.path.exists(map_path):
        # attempt fallback discovery
        maps = glob.glob("data/derived/**/doc_id_to_country.csv", recursive=True)
        if not maps:
            return {}
        map_path = maps[0]

    doc2country: Dict[str, str] = {}
    with open(map_path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            did = str(row.get("doc_id", "")).strip()
            c = (row.get("country") or "").strip()
            if did:
                doc2country[did] = c
    return doc2country


def load_edu_doc_ids(edu_chunks_path: str) -> Tuple[set, bool]:
    if not edu_chunks_path or not os.path.exists(edu_chunks_path):
        return set(), False
    doc_ids = set()
    with open(edu_chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                doc_ids.add(str(obj["doc_id"]))
            except Exception:
                continue
    return doc_ids, True


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    union = len(a | b)
    return (len(a & b) / union) if union else 0.0


# ----------------------------
# Reporting
# ----------------------------

def top_countries(ids: Iterable[str], doc2country: Dict[str, str], n: int = 15) -> List[Tuple[str, int]]:
    c = Counter()
    for did in ids:
        cc = doc2country.get(did, "")
        if cc:
            c[cc] += 1
    return c.most_common(n)


def write_doc_ids(path: str, ids: set) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for did in sorted(ids):
            f.write(did + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--titles_path", default="", help="Path to titles file (csv/tsv/parquet) with doc_id + title.")
    ap.add_argument("--doc_id_col", default="", help="doc_id column name (optional if auto-detected).")
    ap.add_argument("--title_col", default="", help="title column name (optional if auto-detected).")
    ap.add_argument("--search_root", default="data", help="Root to search for titles file if not provided.")
    ap.add_argument("--out_dir", default="data/derived/step6b_title_edu", help="Output directory.")
    ap.add_argument(
        "--map_path",
        default="data/derived/step6_chunks_edu/doc_id_to_country.csv",
        help="doc_id->country mapping CSV (optional but recommended).",
    )
    ap.add_argument(
        "--edu_chunks_path",
        default="data/derived/step6_chunks_edu/chunks_edu.jsonl",
        help="Education chunks JSONL to compute overlap (optional).",
    )
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Titles source
    if args.titles_path:
        titles_path = args.titles_path
        doc_id_col = args.doc_id_col or "doc_id"
        title_col = args.title_col or "title"
    else:
        titles_path, doc_id_col, title_col = find_titles_file(args.search_root)

    print(f"[step6b] Titles source: {titles_path}")
    print(f"[step6b] Using columns: doc_id='{doc_id_col}' title='{title_col}'")

    df = read_titles(titles_path, doc_id_col, title_col)
    print(f"[step6b] Loaded titles: {len(df)} rows")

    # Load mapping + edu subset ids (if present)
    doc2country = load_doc2country(args.map_path)
    edu_ids, has_edu = load_edu_doc_ids(args.edu_chunks_path)

    if doc2country:
        print(f"[step6b] Loaded doc->country map: {len(doc2country)} entries (map_path={args.map_path})")
    else:
        print("[step6b] WARNING: No doc->country map found/loaded. Country stats will be empty.")

    if has_edu:
        print(f"[step6b] Loaded edu-chunks doc_id set: {len(edu_ids)} docs (edu_chunks_path={args.edu_chunks_path})")
    else:
        print("[step6b] Edu-chunks file not found; overlap stats will be skipped.")

    tier1_re = compile_patterns(TIER1_PATTERNS)
    tier12_re = compile_patterns(TIER1_PATTERNS + TIER2_PATTERNS)

    tier1_ids: set = set()
    tier12_ids: set = set()

    tier1_term_counts = Counter()
    tier12_term_counts = Counter()

    tier1_examples = defaultdict(list)   # term -> [(doc_id, title), ...]
    tier12_examples = defaultdict(list)

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Scanning titles"):
        did = row["doc_id"]
        title = row["title"]

        hits1 = match_patterns(title, tier1_re)
        hits12 = match_patterns(title, tier12_re)

        if hits1:
            tier1_ids.add(did)
            tier12_ids.add(did)
            for h in hits1:
                tier1_term_counts[h] += 1
                if len(tier1_examples[h]) < 5:
                    tier1_examples[h].append((did, title))

        elif hits12:
            tier12_ids.add(did)

        if hits12:
            for h in hits12:
                tier12_term_counts[h] += 1
                if len(tier12_examples[h]) < 5:
                    tier12_examples[h].append((did, title))

    # Write doc_id lists
    tier1_path = os.path.join(args.out_dir, "doc_ids_title_tier1.txt")
    tier12_path = os.path.join(args.out_dir, "doc_ids_title_tier1plus2.txt")
    write_doc_ids(tier1_path, tier1_ids)
    write_doc_ids(tier12_path, tier12_ids)

    # Report
    report_path = os.path.join(args.out_dir, "subset_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Step 6b — Title-based Education Document Subsets\n\n")
        f.write(f"**Titles source:** `{titles_path}`\n\n")
        f.write(f"**Out dir:** `{args.out_dir}`\n\n")
        f.write(f"**Country map requested:** `{args.map_path}`\n\n")
        f.write(f"**Edu-chunks requested:** `{args.edu_chunks_path}`\n\n")

        f.write("## Sizes\n\n")
        f.write(f"- Tier 1 (formal education title matches): **{len(tier1_ids)} docs**\n")
        f.write(f"- Tier 1+2 (formal + skills/workforce title matches): **{len(tier12_ids)} docs**\n")
        if has_edu:
            f.write(f"- Edu-chunks subset (doc_id set): **{len(edu_ids)} docs**\n")
        f.write("\n")

        if has_edu:
            f.write("## Overlap with Edu-chunks (doc_id Jaccard)\n\n")
            f.write(f"- Jaccard(Tier1, Edu-chunks): **{jaccard(tier1_ids, edu_ids):.3f}**\n")
            f.write(f"- Jaccard(Tier1+2, Edu-chunks): **{jaccard(tier12_ids, edu_ids):.3f}**\n\n")

        if doc2country:
            f.write("## Top Countries (Tier 1)\n\n")
            for c, v in top_countries(tier1_ids, doc2country, 15):
                f.write(f"- {c}: {v}\n")
            f.write("\n## Top Countries (Tier 1+2)\n\n")
            for c, v in top_countries(tier12_ids, doc2country, 15):
                f.write(f"- {c}: {v}\n")
            f.write("\n")

        f.write("## Top Matched Terms (Tier 1)\n\n")
        for term, cnt in tier1_term_counts.most_common(25):
            f.write(f"- `{term}`: {cnt}\n")

        f.write("\n## Top Matched Terms (Tier 1+2)\n\n")
        for term, cnt in tier12_term_counts.most_common(25):
            f.write(f"- `{term}`: {cnt}\n")

        f.write("\n## Sanity Examples (Tier 1)\n\n")
        for term, _ in tier1_term_counts.most_common(10):
            f.write(f"### {term}\n")
            for did, title in tier1_examples.get(term, [])[:5]:
                f.write(f"- {did}: {title}\n")
            f.write("\n")

        f.write("\n## Sanity Examples (Tier 1+2)\n\n")
        for term, _ in tier12_term_counts.most_common(10):
            f.write(f"### {term}\n")
            for did, title in tier12_examples.get(term, [])[:5]:
                f.write(f"- {did}: {title}\n")
            f.write("\n")

    # Console summary
    print("\n[step6b] Wrote outputs:")
    print(" -", tier1_path)
    print(" -", tier12_path)
    print(" -", report_path)
    if has_edu:
        print(f"[step6b] Jaccard(Tier1, Edu): {jaccard(tier1_ids, edu_ids):.3f}")
        print(f"[step6b] Jaccard(Tier1+2, Edu): {jaccard(tier12_ids, edu_ids):.3f}")


if __name__ == "__main__":
    main()
