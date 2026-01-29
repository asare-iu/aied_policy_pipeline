#!/usr/bin/env python3
"""
Step 6.2 — Apply education gate to chunk corpus.

Input:
  - data/derived/step4_chunks_tagged/chunks_normalized_tagged.jsonl
  - resources/edu_lexicon_v1.yml

Output:
  - data/derived/step6_chunks_edu/chunks_edu.jsonl
  - data/derived/step6_chunks_edu/edu_subset_summary.csv
  - data/derived/step6_chunks_edu/edu_gate_audit_sample.csv

Gate definition:
  A chunk is included if it matches any seed-lexicon term from:
    education_levels OR education_institutions OR education_governance OR education_actors
  OR it matches education_technology AND also matches at least one of the above buckets.
  Weak terms are recorded for audit but are not sufficient for inclusion.

This step performs only education relevance selection. Normativity gating is applied later.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
from typing import Any, Dict, Iterable, List, Tuple

import yaml
from tqdm import tqdm

WS_RE = re.compile(r"\s+")

def normalize_text(text: str) -> str:
    return WS_RE.sub(" ", text.lower()).strip()

def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def extract_text(record: Dict[str, Any]) -> str:
    for key in ("text_normalized", "text", "chunk_text", "chunk_text_normalized"):
        val = record.get(key)
        if isinstance(val, str) and val.strip():
            return val
    raise KeyError("No recognized text field found in chunk record.")

def extract_doc_id(record: Dict[str, Any]) -> str:
    for k in ("doc_id", "document_id", "docid"):
        if k in record and record[k] is not None:
            return str(record[k])
    return "UNKNOWN_DOC"

def extract_country(record: Dict[str, Any]) -> str:
    for k in ("country", "jurisdiction", "country_name", "iso3", "iso"):
        if k in record and record[k] is not None:
            return str(record[k])
    md = record.get("metadata")
    if isinstance(md, dict):
        for k in ("country", "jurisdiction", "country_name", "iso3", "iso"):
            if k in md and md[k] is not None:
                return str(md[k])
    return "UNKNOWN_COUNTRY"

def compile_phrases(phrases: List[str]) -> List[Tuple[str, re.Pattern]]:
    out = []
    for p in phrases:
        p2 = normalize_text(p)
        pat = re.compile(rf"(?<!\w){re.escape(p2)}(?!\w)")
        out.append((p, pat))
    return out

def match_phrases(text_norm: str, compiled: List[Tuple[str, re.Pattern]]) -> List[str]:
    hits = []
    for raw, pat in compiled:
        if pat.search(text_norm):
            hits.append(raw)
    # dedupe, stable order
    seen = set()
    out = []
    for h in hits:
        if h not in seen:
            seen.add(h)
            out.append(h)
    return out
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", required=True)
    ap.add_argument("--lexicon", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--audit_n", type=int, default=500)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    random.seed(args.seed)

    lex = yaml.safe_load(open(args.lexicon, "r", encoding="utf-8"))
    version = lex.get("version", "v1")
    seed = lex["seed_lexicon"]

    # Buckets used for inclusion
    buckets_strong = ["education_levels", "education_institutions", "education_governance", "education_actors"]
    bucket_edtech = "education_technology"
    bucket_weak = "weak_terms"

    strong_terms = []
    for b in buckets_strong:
        strong_terms.extend(seed.get(b, {}).get("terms", []))

    edtech_terms = seed.get(bucket_edtech, {}).get("terms", [])
    weak_terms = seed.get(bucket_weak, {}).get("terms", [])

    strong_comp = compile_phrases(strong_terms)
    edtech_comp = compile_phrases(edtech_terms)
    weak_comp = compile_phrases(weak_terms)

    records = list(read_jsonl(args.infile))

    out_jsonl = os.path.join(args.outdir, "chunks_edu.jsonl")
    summary_csv = os.path.join(args.outdir, "edu_subset_summary.csv")
    audit_csv = os.path.join(args.outdir, "edu_gate_audit_sample.csv")

    kept = 0
    audit_pool = []
    doc_set = set()
    country_set = set()

    with open(out_jsonl, "w", encoding="utf-8") as out:
        for rec in tqdm(records, desc="Applying education gate", unit="chunks"):
            text = extract_text(rec)
            tnorm = normalize_text(text)

            hits_strong = match_phrases(tnorm, strong_comp)
            hits_edtech = match_phrases(tnorm, edtech_comp)
            hits_weak = match_phrases(tnorm, weak_comp)

            edu_match = False
            rule = "no_match"

            if hits_strong:
                edu_match = True
                rule = "strong_bucket_match"
            elif hits_edtech and hits_strong:
                edu_match = True
                rule = "edtech_plus_strong"
            elif hits_edtech:
                # edtech alone is not sufficient without strong buckets
                edu_match = False
                rule = "edtech_only_rejected"

            rec2 = dict(rec)
            rec2["edu_gate_version"] = version
            rec2["edu_match"] = edu_match
            rec2["edu_match_rule"] = rule
            rec2["edu_match_terms_strong"] = hits_strong[:12]
            rec2["edu_match_terms_edtech"] = hits_edtech[:12]
            rec2["edu_weak_hits"] = hits_weak[:12]

            if edu_match:
                kept += 1
                doc_set.add(extract_doc_id(rec2))
                country_set.add(extract_country(rec2))
                out.write(json.dumps(rec2, ensure_ascii=False) + "\n")

            # Audit pool includes: all kept + all nontrivial rejects (edtech-only or weak hits)
            if edu_match or hits_edtech or hits_weak:
                audit_pool.append(rec2)

    # Summary
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerow(["input_chunks", len(records)])
        w.writerow(["kept_chunks_edu", kept])
        w.writerow(["kept_pct", round((kept / len(records) * 100), 4) if records else 0.0])
        w.writerow(["unique_docs_in_edu", len(doc_set)])
        w.writerow(["unique_countries_in_edu", len(country_set)])

    # Audit sample
    sample_n = min(args.audit_n, len(audit_pool))
    sample = random.sample(audit_pool, k=sample_n) if sample_n else []

    with open(audit_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(
            f,
            quoting=csv.QUOTE_MINIMAL,
            escapechar="\\"
        )
        w.writerow([
            "edu_match",
            "edu_match_rule",
            "strong_terms",
            "edtech_terms",
            "weak_hits",
            "country",
            "doc_id",
            "text_preview"
        ])

        for r in sample:
            text = extract_text(r)
            w.writerow([
                r.get("edu_match"),
                r.get("edu_match_rule"),
                "|".join(r.get("edu_match_terms_strong", [])),
                "|".join(r.get("edu_match_terms_edtech", [])),
                "|".join(r.get("edu_weak_hits", [])),
                extract_country(r),
                extract_doc_id(r),
                (text[:260] + "…") if len(text) > 260 else text
            ])

    print("Step 6.2 complete.")
    print(f"Input chunks: {len(records)}")
    print(f"Edu chunks kept: {kept} ({(kept/len(records)*100):.2f}%)" if records else "Edu chunks kept: 0")
    print(f"Wrote: {out_jsonl}")
    print(f"Wrote: {summary_csv}")
    print(f"Wrote: {audit_csv}")

if __name__ == "__main__":
    main()
