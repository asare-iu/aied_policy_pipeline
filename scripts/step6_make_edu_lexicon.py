#!/usr/bin/env python3
"""
Step 6.1 — Construction of an education-relevance lexicon for policy content analysis.

Purpose
-------
This script constructs a versioned education lexicon used to identify
education-relevant policy statements within a large corpus of AI policy documents.

Design principles
-----------------
1. The lexicon is grounded in internationally recognized education
   classification systems and governance vocabularies (e.g., ISCED, OECD, UNESCO).
2. Lexical coverage is expanded empirically using statistical association
   with seed education terms observed within the corpus itself.
3. All terms are stored with explicit provenance to support auditability
   and methodological transparency.

The resulting lexicon is used solely for corpus subsetting and does not
perform normative or institutional classification.

Output
------
resources/edu_lexicon_v1.yml
"""

from __future__ import annotations
import argparse
import json
import re
import math
import time
from collections import Counter
from typing import Iterable, Dict, List, Tuple
import yaml
from tqdm import tqdm


TOKEN_RE = re.compile(r"[a-z][a-z\-]{2,}")
WS_RE = re.compile(r"\s+")

def normalize_text(text: str) -> str:
    """Lowercase and normalize whitespace for lexical matching."""
    return WS_RE.sub(" ", text.lower()).strip()

def extract_text(record: Dict) -> str:
    """
    Extract normalized text from a chunk record.

    This function is intentionally strict: failure indicates a schema
    inconsistency that should be corrected upstream.
    """
    for key in ("text_normalized", "text", "chunk_text", "chunk_text_normalized"):
        val = record.get(key)
        if isinstance(val, str) and val.strip():
            return val
    raise KeyError("No recognized text field found in chunk record.")

def read_jsonl(path: str) -> Iterable[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def tokenize(text: str) -> List[str]:
    """Tokenize text into alphabetic word units for frequency analysis."""
    return TOKEN_RE.findall(normalize_text(text))

def build_seed_lexicon() -> Dict[str, Dict]:
    """
    Construct seed education vocabularies derived from
    internationally standardized education systems and governance practice.

    Buckets reflect analytically distinct dimensions of education policy:
    - System levels (ISCED)
    - Institutions and sectors
    - Governance instruments and regulatory mechanisms
    - Actors and stakeholders
    - Education-specific digital infrastructure
    """

    return {
        "education_levels": {
            "terms": [
                "early childhood education", "pre-primary education",
                "primary education", "secondary education",
                "lower secondary", "upper secondary",
                "tertiary education", "higher education",
                "postsecondary education"
            ],
            "provenance": ["ISCED"]
        },
        "education_institutions": {
            "terms": [
                "ministry of education", "department of education",
                "school", "schools", "school system",
                "higher education institution", "university", "college",
                "vocational education", "technical education", "tvet"
            ],
            "provenance": ["ISCED", "OECD"]
        },
        "education_governance": {
            "terms": [
                "curriculum", "learning outcomes", "assessment",
                "examination", "accreditation", "quality assurance",
                "inspection", "school evaluation",
                "teacher certification", "qualification framework"
            ],
            "provenance": ["OECD", "UNESCO", "EHEA"]
        },
        "education_actors": {
            "terms": [
                "student", "students", "learner", "teacher", "educator",
                "school leader", "principal", "education authority",
                "accreditation body"
            ],
            "provenance": ["ERIC", "OECD"]
        },
        "education_technology": {
            "terms": [
                "learning management system", "digital learning",
                "online learning", "learning analytics",
                "virtual learning environment"
            ],
            "provenance": ["ERIC", "OECD"]
        },
        "weak_terms": {
            "terms": [
                "training", "upskilling", "reskilling", "capacity building"
            ],
            "provenance": ["disambiguation_guard"]
        }
    }

def expand_lexicon_from_corpus(
    infile: str,
    seed_terms: List[str],
    min_count: int,
    top_k: int
) -> List[Dict]:
    """
    Identify corpus-derived candidate terms associated with education using a
    foreground/background log-odds comparison.

    Foreground: chunks containing at least one seed term.
    Background: all chunks.

    Returns the top_k highest-scoring candidate tokens.
    """
    background = Counter()
    foreground = Counter()

    normalized_seeds = [normalize_text(t) for t in seed_terms]

    # Materialize records to enable an exact progress bar
    records = list(read_jsonl(infile))

    for rec in tqdm(
        records,
        desc="Scanning corpus for education signals",
        unit="chunks"
    ):
        text = extract_text(rec)
        tokens = tokenize(text)
        if not tokens:
            continue

        background.update(tokens)

        text_norm = normalize_text(text)
        if any(
            re.search(rf"(?<!\w){re.escape(s)}(?!\w)", text_norm)
            for s in normalized_seeds
        ):
            foreground.update(tokens)

    scored = []

    for term, fg_count in tqdm(
        foreground.items(),
        desc="Scoring corpus-derived education terms",
        unit="terms"
    ):
        if fg_count < min_count:
            continue

        bg_count = background.get(term, 0)
        score = math.log((fg_count + 1) / (bg_count + 1))

        scored.append({
            "term": term,
            "foreground_count": fg_count,
            "background_count": bg_count,
            "log_odds": score
        })

    return sorted(scored, key=lambda x: x["log_odds"], reverse=True)[:top_k]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--min_count", type=int, default=15)
    parser.add_argument("--top_k", type=int, default=400)
    args = parser.parse_args()

    seed_lexicon = build_seed_lexicon()

    seed_terms = []
    for bucket in seed_lexicon.values():
        seed_terms.extend(bucket["terms"])

    expansion = expand_lexicon_from_corpus(
        infile=args.infile,
        seed_terms=seed_terms,
        min_count=args.min_count,
        top_k=args.top_k
    )

    output = {
        "version": "v1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed_lexicon": seed_lexicon,
        "corpus_expansion": {
            "method": "log_odds_foreground_vs_background",
            "parameters": {
                "min_count": args.min_count,
                "top_k": args.top_k
            },
            "candidates": expansion
        }
    }

    with open(args.out, "w", encoding="utf-8") as f:
        yaml.safe_dump(output, f, sort_keys=False, allow_unicode=True)

if __name__ == "__main__":
    main()
