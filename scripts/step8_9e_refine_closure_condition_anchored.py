#!/usr/bin/env python3
"""
Step 8.9E — Refined education regime closure (condition-anchored, conservative)

Inputs
- data/derived/step8_9_regime_closure/igt_statements_full_with_edu_flags.parquet

Outputs
- data/derived/step8_9_regime_closure/edu_closure_statements_refined.parquet
- data/derived/step8_9_regime_closure/edu_closure_doc_summary_refined.csv
- data/derived/step8_9_regime_closure/edu_closure_audit_sample_refined.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# --- Text normalization helpers
WS_RE = re.compile(r"\s+")
PUNCT_RE = re.compile(r"[()\[\]{}<>.,;:!?\u201c\u201d\u2018\u2019\"'`~@#$%^&*_+=|\\/]+")

WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z\-]{1,}")  # anchor tokenization

# Conservative stopword list (enough to kill "of the / in the / should be" anchors)
STOPWORDS = {
    "a","an","and","are","as","at","be","been","being","but","by",
    "can","could","did","do","does","doing","done",
    "for","from","had","has","have","having","he","her","here","hers",
    "him","his","how","i","if","in","into","is","it","its",
    "may","might","more","most","must","my",
    "no","not","of","on","or","our","out","over","own",
    "shall","should","so","such",
    "than","that","the","their","them","then","there","these","they","this","those","through","to","too",
    "under","until","up","upon","us",
    "was","we","were","what","when","where","which","who","whom","why","will","with","within","without","would","you","your",
}

# Scope/applicability cues (used for "global scope" closure method)
GLOBAL_SCOPE_RE = re.compile(
    r"\b(scope|applicab(?:le|ility)|apply|applies|shall\s+apply|application|"
    r"in\s+all\s+sectors|across\s+the\s+board|including|in\s+general|generally)\b",
    re.IGNORECASE,
)

# Heuristic TOC / heading noise cues
TOC_RE = re.compile(r"\b(contents|table\s+of\s+contents)\b", re.IGNORECASE)
DOT_LEADER_RE = re.compile(r"\.{8,}")  # dotted leader lines
ARTICLE_LIST_RE = re.compile(r"^(article|section|chapter|annex)\s+\w+", re.IGNORECASE)


def norm(s: str) -> str:
    s = "" if s is None else str(s)
    s = s.replace("\x00", " ").lower()
    s = PUNCT_RE.sub(" ", s)
    s = WS_RE.sub(" ", s).strip()
    return s


def is_noise_sentence(sentence_text: str) -> bool:
    """
    Filters common junk that tends to contaminate closure:
    - Table of contents lines
    - dotted leader lists
    - pure heading/index fragments
    """
    t = "" if sentence_text is None else str(sentence_text)
    tn = t.lower()

    if TOC_RE.search(tn) and DOT_LEADER_RE.search(t):
        return True

    if t.count(".") >= 40 and DOT_LEADER_RE.search(t):
        return True

    letters = sum(ch.isalpha() for ch in t)
    if len(t) >= 120 and letters <= 25 and t.count(".") >= 20:
        return True

    if ARTICLE_LIST_RE.match(t.strip()) and len(t.strip().split()) <= 6:
        return True

    return False


def tokens(text_norm: str) -> List[str]:
    return [m.group(0) for m in WORD_RE.finditer(text_norm)]


def content_tokens(text_norm: str) -> List[str]:
    toks = tokens(text_norm)
    return [t for t in toks if t not in STOPWORDS and len(t) >= 3]


def ngrams(tok: List[str], n: int) -> List[str]:
    if len(tok) < n:
        return []
    return [" ".join(tok[i:i+n]) for i in range(len(tok)-n+1)]


def build_ngram_df(texts: List[str], n_vals=(2, 3)) -> Dict[str, int]:
    """
    Document frequency of *content-bearing* n-grams across statements.
    Excludes stopword-only/low-information n-grams.
    """
    df: Dict[str, int] = {}
    for t in texts:
        tn = norm(t)
        tok = content_tokens(tn)
        if not tok:
            continue
        seen = set()
        for n in n_vals:
            for g in ngrams(tok, n):
                # require at least 2 content tokens (conservative)
                if len(g.split()) < 2:
                    continue
                if g not in seen:
                    df[g] = df.get(g, 0) + 1
                    seen.add(g)
    return df


def regime_text(row: pd.Series) -> str:
    """
    Use conditions as regime-defining text whenever present; otherwise fall back to sentence text.
    """
    c = row.get("c_texts", None)
    if c is not None:
        c_str = str(c).strip()
        if c_str and c_str.lower() != "none":
            return c_str
    return str(row.get("sentence_text", ""))


def select_anchors_conservative(
    e_texts: List[str],
    r_texts: List[str],
    max_anchors: int = 5,
    min_df_r: int = 4,
    min_cov_r: float = 0.08,
    max_cov_r: float = 0.65,
) -> List[str]:
    """
    Conservative anchor selection:
    - Anchors must appear in E and in R.
    - In R, anchors must be frequent enough (min_df_r) but not universal (max_cov_r).
    - Prefer anchors with higher "specificity" via IDF-like weighting over R.
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

        # IDF-like weight (higher when term is less ubiquitous)
        idf = math.log((n_r + 1.0) / (rdf + 1.0)) + 1.0

        # Require the anchor to contain at least one "substantive" token
        # (already content-only, but keep a guard)
        parts = term.split()
        if len(parts) < 2:
            continue

        # Score: reward presence in E, specificity in R, and moderate prevalence in R
        score = float(e_df.get(term, 1)) * idf * (1.0 - cov)
        candidates.append((score, term))

    candidates.sort(key=lambda x: x[0], reverse=True)
    return [t for _, t in candidates[:max_anchors]]


def contains_any_anchor(text: str, anchors: List[str]) -> bool:
    tn = norm(text)
    ct = " ".join(content_tokens(tn))
    return any(a in ct for a in anchors)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--igt-edu",
        default=str(PROJECT_ROOT / "data/derived/step8_9_regime_closure/igt_statements_full_with_edu_flags.parquet"),
    )
    ap.add_argument(
        "--out-dir",
        default=str(PROJECT_ROOT / "data/derived/step8_9_regime_closure"),
    )
    ap.add_argument("--similarity-threshold", type=float, default=0.24)
    ap.add_argument("--min-closure", type=int, default=10)
    ap.add_argument("--progress-every", type=int, default=200)
    ap.add_argument("--audit-n-per-method", type=int, default=75)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    t0 = time.time()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.igt_edu)

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
        "c_texts",
    }
    missing = sorted(list(required - set(df.columns)))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Noise filter
    df["is_noise"] = df["sentence_text"].fillna("").astype(str).apply(is_noise_sentence)

    # R pool = rule/norm candidates
    df["is_R"] = df["statement_type_candidate"].isin(["rule_candidate", "norm_candidate"])
    # E pool = any education signal
    df["is_E"] = df["edu_any_hit"].fillna(False).astype(bool)

    # Filter out noise for closure construction
    df2 = df[~df["is_noise"]].copy()

    doc_ids = df2["doc_id"].dropna().unique().tolist()
    n_docs = len(doc_ids)

    closure_parts: List[pd.DataFrame] = []
    doc_rows: List[Dict[str, object]] = []

    rng = np.random.default_rng(args.seed)

    for i, doc_id in enumerate(doc_ids, start=1):
        g = df2[df2["doc_id"] == doc_id].copy()
        if len(g) == 0:
            continue

        E = g[g["is_E"]].copy()
        R = g[g["is_R"]].copy()

        if len(E) == 0 or len(R) == 0:
            continue

        # Build texts for anchor selection using regime_text (condition-first)
        E_texts = E.apply(regime_text, axis=1).tolist()
        R_texts = R.apply(regime_text, axis=1).tolist()

        anchors = select_anchors_conservative(E_texts, R_texts)

        method = ""
        chosen = pd.DataFrame()

        # Method 1: anchor overlap (match anchors against regime_text)
        if anchors:
            mask = R.apply(lambda r: contains_any_anchor(regime_text(r), anchors), axis=1)
            chosen = R[mask].copy()
            if len(chosen) > 0:
                method = "anchor_overlap"

        # Method 2: global scope clause (detect in E using sentence_text + c_texts)
        if method == "":
            E_scope_texts = (
                E["sentence_text"].fillna("").astype(str)
                + " "
                + E["c_texts"].fillna("").astype(str)
            ).tolist()
            if any(GLOBAL_SCOPE_RE.search(norm(t)) for t in E_scope_texts[: min(len(E_scope_texts), 60)]):
                chosen = R.copy()
                method = "global_scope_clause"

        # Method 3: semantic similarity fallback (TF-IDF over regime_text; safe centroid)
        if method == "":
            all_texts = [norm(t) for t in (E_texts + R_texts)]
            vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=12000)
            X = vec.fit_transform(all_texts)

            XE = X[: len(E_texts), :]
            XR = X[len(E_texts) :, :]

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
        chosen["doc_has_edu_actor_A"] = bool(E["edu_actor_A_hit"].fillna(False).astype(bool).sum() > 0)

        closure_parts.append(chosen)

        doc_rows.append({
            "doc_id": doc_id,
            "closure_method": method,
            "closure_size": int(len(chosen)),
            "E_statements": int(len(E)),
            "R_statements": int(len(R)),
            "doc_has_edu_actor_A": bool(E["edu_actor_A_hit"].fillna(False).astype(bool).sum() > 0),
            "closure_anchors": "|".join(anchors) if anchors else "",
        })

        if (i % args.progress_every) == 0 or i == n_docs:
            elapsed = time.time() - t0
            rate = i / elapsed if elapsed > 0 else 0.0
            print(f"[step8_9e] docs={i}/{n_docs} elapsed_s={elapsed:.1f} docs_per_s={rate:.2f}")

    closure = pd.concat(closure_parts, ignore_index=True) if closure_parts else pd.DataFrame()
    doc_summary = pd.DataFrame(doc_rows)

    out_pq = out_dir / "edu_closure_statements_refined.parquet"
    out_csv = out_dir / "edu_closure_doc_summary_refined.csv"
    out_audit = out_dir / "edu_closure_audit_sample_refined.csv"

    closure.to_parquet(out_pq, index=False)
    doc_summary.to_csv(out_csv, index=False, escapechar="\\", quoting=csv.QUOTE_MINIMAL)

    print(f"[step8_9e] wrote {out_pq} rows={len(closure)}")
    print(f"[step8_9e] wrote {out_csv} docs={len(doc_summary)}")
    if len(doc_summary):
        print(doc_summary["closure_method"].value_counts().to_string())

    # Audit sample by method
    if len(closure) == 0:
        pd.DataFrame().to_csv(out_audit, index=False)
        print(f"[step8_9e] wrote {out_audit} rows=0")
        print(f"[step8_9e] done elapsed_s={time.time() - t0:.1f}")
        return

    cols = [
        "doc_id", "chunk_id", "sentence_index_in_chunk",
        "closure_method", "closure_anchors",
        "statement_type_candidate", "d_class", "d_surface", "d_lemma",
        "a_raw_text", "c_texts", "b_text",
        "edu_domain_hit", "edu_actor_any_hit", "edu_actor_A_hit",
        "sentence_text",
    ]
    cols = [c for c in cols if c in closure.columns]

    audit_parts: List[pd.DataFrame] = []
    for m, sub in closure.groupby("closure_method"):
        take = min(args.audit_n_per_method, len(sub))
        # deterministic sample for reproducibility
        samp = sub.sample(n=take, random_state=args.seed)
        audit_parts.append(samp[cols].copy())

    audit = pd.concat(audit_parts, ignore_index=True) if audit_parts else pd.DataFrame()
    audit.to_csv(out_audit, index=False, escapechar="\\", quoting=csv.QUOTE_MINIMAL)

    print(f"[step8_9e] wrote {out_audit} rows={len(audit)}")
    print(f"[step8_9e] done elapsed_s={time.time() - t0:.1f}")


if __name__ == "__main__":
    main()
