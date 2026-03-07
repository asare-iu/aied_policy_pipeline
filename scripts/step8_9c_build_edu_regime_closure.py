#!/usr/bin/env python3
"""
Step 8.9C — Education regime closure (distributed-rule reconstruction)

This script constructs an "education-applicable closure set" of institutional statements
for each document. It aims to recover obligations that apply to education indirectly
(e.g., via category/regime language) even when education is not explicitly named
in the obligation sentence.

Inputs
- data/derived/step8_9_regime_closure/igt_statements_full_with_edu_flags.parquet
- (optional) data/derived/step8_9_regime_closure/doc_enforcement_index.parquet

Outputs
- data/derived/step8_9_regime_closure/edu_closure_statements.parquet
- data/derived/step8_9_regime_closure/edu_closure_doc_summary.csv
- data/derived/step8_9_regime_closure/edu_closure_audit_sample.csv

Closure methods (per doc, in order)
1) anchor_overlap: n-gram anchors shared by education-touched statements and rule/norm candidates
2) global_scope_clause: if education-touched statements contain scope/applicability cues, include all rule/norm candidates
3) semantic_similarity: TF-IDF similarity between rule/norm candidates and education centroid within the doc
"""

from __future__ import annotations

import argparse
import csv
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


PROJECT_ROOT = Path(__file__).resolve().parents[1]

WS_RE = re.compile(r"\s+")
PUNCT_RE = re.compile(r"[()\[\]{}<>.,;:!?\u201c\u201d\u2018\u2019\"'`~@#$%^&*_+=|\\/]+")

# Broad "scope/applicability" cues
GLOBAL_SCOPE_RE = re.compile(
    r"\b(scope|applicab(?:le|ility)|apply|applies|shall\s+apply|"
    r"application|in\s+all\s+sectors|across\s+the\s+board|including)\b",
    re.IGNORECASE,
)

# Tokenization for n-gram anchors
WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z\-]{1,}")

def norm(s: str) -> str:
    s = "" if s is None else str(s)
    s = s.lower()
    s = s.replace("\x00", " ")
    s = PUNCT_RE.sub(" ", s)
    s = WS_RE.sub(" ", s).strip()
    return s

def tokens(text_norm: str) -> List[str]:
    return [m.group(0) for m in WORD_RE.finditer(text_norm)]

def ngrams(tok: List[str], n: int) -> List[str]:
    if len(tok) < n:
        return []
    return [" ".join(tok[i:i+n]) for i in range(len(tok)-n+1)]

def build_ngram_df(texts: List[str], n_vals=(2, 3)) -> Dict[str, int]:
    """
    Document frequency of n-grams across a list of statements:
    df(term) = number of statements containing term at least once.
    """
    df: Dict[str, int] = {}
    for t in texts:
        tn = norm(t)
        tok = tokens(tn)
        seen = set()
        for n in n_vals:
            for g in ngrams(tok, n):
                if g not in seen:
                    df[g] = df.get(g, 0) + 1
                    seen.add(g)
    return df

def select_anchors(
    e_texts: List[str],
    r_texts: List[str],
    max_anchors: int = 6,
    min_df_r: int = 3,
    min_cov_r: float = 0.05,
    max_cov_r: float = 0.85,
) -> List[str]:
    """
    Choose n-gram anchors that:
    - appear in education-touched statements (E)
    - appear in R statements with doc-frequency >= min_df_r
    - are neither too rare nor too universal within R

    Returns a short ordered list of anchor phrases.
    """
    if not e_texts or not r_texts:
        return []

    e_df = build_ngram_df(e_texts)
    r_df = build_ngram_df(r_texts)

    if not e_df or not r_df:
        return []

    n_r = max(1, len(r_texts))
    candidates: List[Tuple[float, str]] = []
    for term in set(e_df.keys()).intersection(r_df.keys()):
        rdf = r_df.get(term, 0)
        if rdf < min_df_r:
            continue
        cov = rdf / n_r
        if cov < min_cov_r or cov > max_cov_r:
            continue
        # Score prefers terms that are common enough in R to be regime-like but not universal,
        # and present in E to provide an education bridge.
        score = (1.0 - cov) * float(rdf) * float(e_df.get(term, 1))
        candidates.append((score, term))

    candidates.sort(key=lambda x: x[0], reverse=True)
    return [t for _, t in candidates[:max_anchors]]

def contains_any_anchor(text: str, anchors: List[str]) -> bool:
    tn = norm(text)
    return any(a in tn for a in anchors)

def safe_bool_series(s: pd.Series) -> pd.Series:
    # Handles weird dtype issues robustly
    return s.fillna(False).astype(bool)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--igt",
        default=str(PROJECT_ROOT / "data/derived/step8_9_regime_closure/igt_statements_full_with_edu_flags.parquet"),
    )
    ap.add_argument(
        "--enforcement",
        default=str(PROJECT_ROOT / "data/derived/step8_9_regime_closure/doc_enforcement_index.parquet"),
    )
    ap.add_argument(
        "--out-dir",
        default=str(PROJECT_ROOT / "data/derived/step8_9_regime_closure"),
    )
    ap.add_argument("--similarity-threshold", type=float, default=0.18)
    ap.add_argument("--min-closure", type=int, default=12)
    ap.add_argument("--progress-every", type=int, default=200)
    ap.add_argument("--audit-n-per-method", type=int, default=75)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    t0 = time.time()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.igt)

    # Required minimal schema
    required = {
        "doc_id",
        "chunk_id",
        "sentence_index_in_chunk",
        "sentence_text",
        "statement_type_candidate",
        "edu_any_hit",
        "edu_domain_hit",
        "edu_actor_any_hit",
        "edu_actor_A_hit",
    }
    missing = sorted(list(required - set(df.columns)))
    if missing:
        raise ValueError(
            f"Missing required columns in IGT+edu flags: {missing}\n"
            f"Columns present: {list(df.columns)}"
        )

    # Optional columns used for audit output (only included if present)
    optional_cols = [c for c in [
        "d_lemma", "d_class", "d_polarity",
        "a_raw_text", "a_class",
        "c_texts", "b_text", "i_phrase_text",
        "edu_actor_A_hit_terms"
    ] if c in df.columns]

    # Define R pool: rule/norm candidates
    df["is_R"] = df["statement_type_candidate"].isin(["rule_candidate", "norm_candidate"])

    # Optional: join doc enforcement index (doc-level, robust)
    enf = None
    enf_path = Path(args.enforcement)
    if enf_path.exists():
        enf = pd.read_parquet(enf_path)
        if "doc_id" in enf.columns:
            keep = [c for c in ["doc_id", "has_enforcement_language", "enforcement_cue_hits"] if c in enf.columns]
            enf = enf[keep].copy()
        else:
            enf = None

    doc_ids = df["doc_id"].dropna().unique().tolist()
    n_docs = len(doc_ids)

    closure_parts: List[pd.DataFrame] = []
    doc_rows: List[Dict[str, object]] = []

    rng = np.random.default_rng(args.seed)

    for i, doc_id in enumerate(doc_ids, start=1):
        g = df[df["doc_id"] == doc_id].copy()
        if len(g) == 0:
            continue

        E = g[safe_bool_series(g["edu_any_hit"])].copy()
        R = g[safe_bool_series(g["is_R"])].copy()

        if len(E) == 0 or len(R) == 0:
            continue

        e_texts = E["sentence_text"].fillna("").astype(str).tolist()
        r_texts = R["sentence_text"].fillna("").astype(str).tolist()

        anchors = select_anchors(e_texts, r_texts)

        method = ""
        chosen = pd.DataFrame()

        # Method 1: anchor overlap
        if anchors:
            mask = R["sentence_text"].astype(str).apply(lambda t: contains_any_anchor(t, anchors))
            chosen = R[mask].copy()
            if len(chosen) > 0:
                method = "anchor_overlap"

        # Method 2: global scope clause
        if method == "":
            # if education statements mention scope/applicability terms, treat as doc-wide applicability cue
            if any(GLOBAL_SCOPE_RE.search(norm(t)) for t in e_texts[: min(len(e_texts), 50)]):
                chosen = R.copy()
                method = "global_scope_clause"

        # Method 3: semantic similarity fallback (safe centroid, no np.matrix)
        if method == "":
            # Vectorize within doc only
            all_texts = [norm(t) for t in (e_texts + r_texts)]
            vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=8000)
            X = vec.fit_transform(all_texts)

            XE = X[: len(E), :]
            XR = X[len(E) :, :]

            # centroid as ndarray with shape (1, n_features)
            centroid = np.asarray(XE.mean(axis=0)).reshape(1, -1)

            sims = cosine_similarity(XR, centroid).reshape(-1)

            R2 = R.copy()
            R2["closure_similarity"] = sims

            chosen = R2[R2["closure_similarity"] >= args.similarity_threshold].copy()
            if len(chosen) < args.min_closure:
                chosen = R2.sort_values("closure_similarity", ascending=False).head(args.min_closure).copy()

            method = "semantic_similarity"

        if len(chosen) == 0:
            continue

        chosen["closure_method"] = method
        chosen["closure_anchors"] = "|".join(anchors) if anchors else ""
        chosen["closure_E_statement_count"] = int(len(E))
        chosen["closure_R_statement_count"] = int(len(R))
        chosen["closure_size"] = int(len(chosen))
        chosen["doc_has_edu_actor_A"] = bool(safe_bool_series(g["edu_actor_A_hit"]).sum() > 0)

        if enf is not None:
            chosen = chosen.merge(enf, on="doc_id", how="left")

        closure_parts.append(chosen)

        doc_rows.append({
            "doc_id": doc_id,
            "closure_method": method,
            "closure_size": int(len(chosen)),
            "E_statements": int(len(E)),
            "R_statements": int(len(R)),
            "doc_has_edu_actor_A": bool(safe_bool_series(g["edu_actor_A_hit"]).sum() > 0),
            "closure_anchors": "|".join(anchors) if anchors else "",
        })

        if (i % args.progress_every) == 0 or i == n_docs:
            elapsed = time.time() - t0
            rate = i / elapsed if elapsed > 0 else 0.0
            print(f"[step8_9c] docs={i}/{n_docs} elapsed_s={elapsed:.1f} docs_per_s={rate:.2f}")

    closure = pd.concat(closure_parts, ignore_index=True) if closure_parts else pd.DataFrame()
    doc_summary = pd.DataFrame(doc_rows)

    out_pq = out_dir / "edu_closure_statements.parquet"
    out_csv = out_dir / "edu_closure_doc_summary.csv"
    out_audit = out_dir / "edu_closure_audit_sample.csv"

    closure.to_parquet(out_pq, index=False)
    doc_summary.to_csv(out_csv, index=False, escapechar="\\", quoting=csv.QUOTE_MINIMAL)

    print(f"[step8_9c] wrote {out_pq} rows={len(closure)}")
    print(f"[step8_9c] wrote {out_csv} docs={len(doc_summary)}")
    if len(doc_summary):
        print(doc_summary["closure_method"].value_counts().to_string())
    # Audit sample stratified by closure_method
    if len(closure) == 0:
        # still write an empty audit file for pipeline stability
        pd.DataFrame().to_csv(out_audit, index=False)
        print(f"[step8_9c] wrote {out_audit} rows=0")
        print(f"[step8_9c] done elapsed_s={time.time() - t0:.1f}")
        return

    audit_parts: List[pd.DataFrame] = []
    for m, sub in closure.groupby("closure_method"):
        take = min(args.audit_n_per_method, len(sub))
        samp = sub.sample(n=take, random_state=args.seed)
        cols = [
            "doc_id",
            "chunk_id",
            "sentence_index_in_chunk",
            "closure_method",
            "closure_anchors",
            "edu_domain_hit",
            "edu_actor_any_hit",
            "edu_actor_A_hit",
            "statement_type_candidate",
        ] + optional_cols + ["sentence_text"]
        cols = [c for c in cols if c in samp.columns]
        audit_parts.append(samp[cols].copy())

    audit = pd.concat(audit_parts, ignore_index=True) if audit_parts else pd.DataFrame()
    audit.to_csv(out_audit, index=False, escapechar="\\", quoting=csv.QUOTE_MINIMAL)

    print(f"[step8_9c] wrote {out_audit} rows={len(audit)}")
    print(f"[step8_9c] done elapsed_s={time.time() - t0:.1f}")

if __name__ == "__main__":
    main()
