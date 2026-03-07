#!/usr/bin/env python3
"""
Step 8.9B — Add education flags to FULL IGT statements.

Reads:
  - data/derived/step8_igt_full/igt_statements_full.parquet
  - resources/edu_lexicon_v1.yml

Writes:
  - data/derived/step8_9_regime_closure/igt_statements_full_with_edu_flags.parquet

Flags:
  - edu_domain_hit: education domain mentioned anywhere (sentence/C/B), excluding weak_terms
  - edu_actor_any_hit: education actor term mentioned anywhere (sentence/C/B/A)
  - edu_actor_A_hit: education actor term specifically in A (and a_class == explicit by default)
"""

from __future__ import annotations
import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]

WS_RE = re.compile(r"\s+")
def norm(s: str) -> str:
    s = "" if s is None else str(s)
    s = s.lower()
    s = WS_RE.sub(" ", s).strip()
    return s

def compile_phrase_patterns(phrases: List[str]) -> List[Tuple[str, re.Pattern]]:
    out = []
    for p in phrases:
        p2 = norm(p)
        # boundary-ish match: avoid mid-word
        pat = re.compile(rf"(?<!\w){re.escape(p2)}(?!\w)")
        out.append((p, pat))
    return out

def match_any(text_norm: str, compiled: List[Tuple[str, re.Pattern]], max_hits: int = 12) -> List[str]:
    hits = []
    for raw, pat in compiled:
        if pat.search(text_norm):
            hits.append(raw)
            if len(hits) >= max_hits:
                break
    return hits

def get_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--igt", default=str(PROJECT_ROOT / "data/derived/step8_igt_full/igt_statements_full.parquet"))
    ap.add_argument("--edu-lex", default=str(PROJECT_ROOT / "resources/edu_lexicon_v1.yml"))
    ap.add_argument("--out", default=str(PROJECT_ROOT / "data/derived/step8_9_regime_closure/igt_statements_full_with_edu_flags.parquet"))
    ap.add_argument("--require-explicit-A", action="store_true", help="Only count edu_actor_A_hit when a_class == explicit (recommended).")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.igt)

    # Identify core fields (robust to small schema drift)
    sentence_col = get_col(df, ["sentence_text"])
    c_col = get_col(df, ["c_texts"])
    b_col = get_col(df, ["b_text"])
    a_col = get_col(df, ["a_raw_text"])
    aclass_col = get_col(df, ["a_class"])

    if sentence_col is None:
        raise ValueError(f"IGT missing sentence_text. Columns={list(df.columns)}")

    lex = yaml.safe_load(open(args.edu_lex, "r", encoding="utf-8"))
    seed = lex["seed_lexicon"]

    domain_buckets = ["education_levels", "education_institutions", "education_governance", "education_technology"]
    actor_bucket = "education_actors"
    weak_bucket = "weak_terms"

    domain_terms: List[str] = []
    for b in domain_buckets:
        domain_terms.extend(seed.get(b, {}).get("terms", []))
    actor_terms = seed.get(actor_bucket, {}).get("terms", [])
    weak_terms = seed.get(weak_bucket, {}).get("terms", [])

    domain_comp = compile_phrase_patterns(domain_terms)
    actor_comp = compile_phrase_patterns(actor_terms)
    weak_comp = compile_phrase_patterns(weak_terms)

    # Precompute normalized combined text fields
    sent_norm = df[sentence_col].fillna("").astype(str).map(norm)

    c_norm = df[c_col].fillna("").astype(str).map(norm) if c_col else pd.Series([""] * len(df))
    b_norm = df[b_col].fillna("").astype(str).map(norm) if b_col else pd.Series([""] * len(df))
    a_norm = df[a_col].fillna("").astype(str).map(norm) if a_col else pd.Series([""] * len(df))

    # Domain hit: sentence OR C OR B (exclude weak-only)
    combined_domain = (sent_norm + " " + c_norm + " " + b_norm).map(norm)
    combined_all = (combined_domain + " " + a_norm).map(norm)

    domain_hits = combined_domain.apply(lambda t: match_any(t, domain_comp))
    weak_hits = combined_domain.apply(lambda t: match_any(t, weak_comp))

    df["edu_domain_hit_terms"] = domain_hits.apply(lambda xs: "|".join(xs) if xs else "")
    df["edu_domain_hit"] = df["edu_domain_hit_terms"].ne("")

    # Actor hits anywhere (sentence/C/B/A)
    actor_any_hits = combined_all.apply(lambda t: match_any(t, actor_comp))
    df["edu_actor_any_hit_terms"] = actor_any_hits.apply(lambda xs: "|".join(xs) if xs else "")
    df["edu_actor_any_hit"] = df["edu_actor_any_hit_terms"].ne("")

    # Actor hits in A specifically
    actor_A_hits = a_norm.apply(lambda t: match_any(t, actor_comp))
    df["edu_actor_A_hit_terms"] = actor_A_hits.apply(lambda xs: "|".join(xs) if xs else "")
    df["edu_actor_A_hit_raw"] = df["edu_actor_A_hit_terms"].ne("")

    if args.require_explicit_A and aclass_col:
        df["edu_actor_A_hit"] = df["edu_actor_A_hit_raw"] & df[aclass_col].fillna("").astype(str).str.lower().eq("explicit")
    else:
        df["edu_actor_A_hit"] = df["edu_actor_A_hit_raw"]

    # Helpful composite flags
    df["edu_any_hit"] = df["edu_domain_hit"] | df["edu_actor_any_hit"]

    # Save
    df.to_parquet(out_path, index=False)
    print(f"[step8_9b] wrote {out_path}")
    print(f"[step8_9b] rows={len(df)} | edu_any_hit={(df['edu_any_hit'].sum() if len(df) else 0)} | edu_actor_A_hit={(df['edu_actor_A_hit'].sum() if len(df) else 0)}")

if __name__ == "__main__":
    main()
