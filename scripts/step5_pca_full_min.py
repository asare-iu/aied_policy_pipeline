#!/usr/bin/env python3
"""
Step 5: TF-IDF vectorization + PCA-equivalent decomposition (TruncatedSVD) on all chunks.

Input:
- data/derived/step4_chunks_tagged/chunks_normalized_tagged.jsonl
  (uses text_norm for stable whitespace normalization)

Outputs:
- data/derived/step5_pca_full/pca_full_scores.csv
- data/derived/step5_pca_full/pca_full_top_terms.csv
- data/derived/step5_pca_full/pca_full_metadata.json

Notes:
- TruncatedSVD on TF-IDF is commonly used as "PCA for sparse text data".
- This is deterministic given random_state.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


def read_chunks(path: Path) -> Tuple[List[Dict[str, Any]], List[str]]:
    rows: List[Dict[str, Any]] = []
    texts: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            rows.append(
                {
                    "chunk_id": r["chunk_id"],
                    "doc_id": r["doc_id"],
                    "chunk_index": r["chunk_index"],
                }
            )
            # Prefer text_norm if present; fallback to raw text.
            texts.append(r.get("text_norm") or r.get("text") or "")
    return rows, texts


def write_scores_csv(path: Path, rows: List[Dict[str, Any]], scores, n_components: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["chunk_id", "doc_id", "chunk_index"] + [f"pc{i}" for i in range(1, n_components + 1)]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for meta, vec in zip(rows, scores):
            out = dict(meta)
            for i in range(n_components):
                out[f"pc{i+1}"] = float(vec[i])
            w.writerow(out)


def top_terms_per_component(
    feature_names: List[str],
    components,
    top_n: int,
) -> List[Dict[str, Any]]:
    """
    Return a list of rows with top positive and top negative terms per component.
    """
    out: List[Dict[str, Any]] = []
    for i, comp in enumerate(components, start=1):
        # comp is shape (n_features,)
        idx_sorted = comp.argsort()
        neg_idx = idx_sorted[:top_n]
        pos_idx = idx_sorted[-top_n:][::-1]

        for rank, j in enumerate(pos_idx, start=1):
            out.append({"pc": i, "direction": "positive", "rank": rank, "term": feature_names[j], "loading": float(comp[j])})
        for rank, j in enumerate(neg_idx, start=1):
            out.append({"pc": i, "direction": "negative", "rank": rank, "term": feature_names[j], "loading": float(comp[j])})
    return out


def write_top_terms_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["pc", "direction", "rank", "term", "loading"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> int:
    parser = argparse.ArgumentParser(description="Step 5: TF-IDF + TruncatedSVD (PCA-equivalent) on all chunks.")
    parser.add_argument(
        "--input",
        type=str,
        default="data/derived/step4_chunks_tagged/chunks_normalized_tagged.jsonl",
        help="Tagged chunks JSONL (uses text_norm).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/derived/step5_pca_full",
        help="Output directory for PCA artifacts.",
    )
    parser.add_argument("--n-components", type=int, default=25, help="Number of components (PCs).")
    parser.add_argument("--max-features", type=int, default=50000, help="Max vocabulary size.")
    parser.add_argument("--min-df", type=int, default=2, help="Min document frequency.")
    parser.add_argument("--max-df", type=float, default=0.95, help="Max document frequency proportion.")
    parser.add_argument("--top-terms", type=int, default=30, help="Top terms per direction per PC to export.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for SVD.")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scores_csv = out_dir / "pca_full_scores.csv"
    terms_csv = out_dir / "pca_full_top_terms.csv"
    meta_json = out_dir / "pca_full_metadata.json"

    print("Loading chunks...")
    meta_rows, texts = read_chunks(in_path)
    n_docs = len(texts)
    print(f"Chunks loaded: {n_docs}")

    print("Vectorizing (TF-IDF)...")
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        max_features=args.max_features,
        min_df=args.min_df,
        max_df=args.max_df,
        ngram_range=(1, 2),
    )
    X = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out().tolist()
    print(f"TF-IDF shape: {X.shape} | vocab: {len(feature_names)}")

    print("Decomposing (TruncatedSVD)...")
    svd = TruncatedSVD(n_components=args.n_components, random_state=args.random_state)
    scores = svd.fit_transform(X)

    explained = svd.explained_variance_ratio_.tolist()
    explained_sum = float(sum(explained))

    print("Writing outputs...")
    write_scores_csv(scores_csv, meta_rows, scores, args.n_components)

    top_rows = top_terms_per_component(feature_names, svd.components_, top_n=args.top_terms)
    write_top_terms_csv(terms_csv, top_rows)

    metadata = {
        "input": str(in_path),
        "n_chunks": n_docs,
        "tfidf_shape": [int(X.shape[0]), int(X.shape[1])],
        "vocab_size": len(feature_names),
        "params": {
            "n_components": args.n_components,
            "max_features": args.max_features,
            "min_df": args.min_df,
            "max_df": args.max_df,
            "ngram_range": [1, 2],
            "stop_words": "english",
            "random_state": args.random_state,
        },
        "explained_variance_ratio": explained,
        "explained_variance_ratio_sum": explained_sum,
    }
    meta_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Step 5 complete | PCs={args.n_components} explained_sum={explained_sum:.3f}")
    print(f"Wrote: {scores_csv}")
    print(f"Wrote: {terms_csv}")
    print(f"Wrote: {meta_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
