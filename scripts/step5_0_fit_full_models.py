#!/usr/bin/env python3
"""
Fit and persist canonical full-corpus text models:

- TF-IDF vectorizer (fit on full corpus)
- TruncatedSVD

Writes:
  data/derived/step5_models_full/
    vectorizer.joblib
    svd.joblib
    models_metadata.json
    row_alignment.csv
    X_full_tfidf.npz
    full_scores_from_models.csv
    explained_variance.csv
    top_terms.csv
"""

import argparse
from pathlib import Path

import os
import json
import hashlib
from datetime import datetime

import pandas as pd
from tqdm import tqdm
import joblib
import scipy.sparse as sp

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


CHUNKS_PATH = "data/derived/step4_chunks_tagged/chunks_normalized_tagged.jsonl"
OUT_DIR = "data/derived/step5_models_full_40pc"
META_IN = "data/derived/step5_pca_full/pca_full_metadata.json"

# Default params (will be overridden by META_IN if present)
DEFAULT_MAX_FEATURES = 50000
DEFAULT_N_COMPONENTS = 40
TOP_N_TERMS = 30


def iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def file_sha256(path: str, block_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(block_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    params = {}
    if os.path.exists(META_IN):
        with open(META_IN, "r", encoding="utf-8") as f:
            params = json.load(f)

    max_features = int(params.get("tfidf_max_features", DEFAULT_MAX_FEATURES))
    n_components = int(params.get("n_components", DEFAULT_N_COMPONENTS))
    n_components = args.n_components


    texts = []
    doc_ids = []
    chunk_ids = []

    missing = {"doc_id": 0, "chunk_id": 0, "text_norm": 0}

    for rec in tqdm(iter_jsonl(CHUNKS_PATH), desc="[step5.0] loading chunks"):
        if "doc_id" not in rec or not str(rec["doc_id"]).strip():
            missing["doc_id"] += 1
            continue
        if "chunk_id" not in rec or not str(rec["chunk_id"]).strip():
            missing["chunk_id"] += 1
            continue

        # Prefer normalized text; fall back to raw text only
        t = rec.get("text_norm")
        if t is None or not str(t).strip():
            missing["text_norm"] += 1
            t = rec.get("text", "")

        texts.append(str(t))
        doc_ids.append(str(rec["doc_id"]))
        chunk_ids.append(str(rec["chunk_id"]))

    if missing["doc_id"] or missing["chunk_id"]:
        raise RuntimeError(f"[step5.0] Missing required fields; counts={missing}")

    if not texts:
        raise RuntimeError("[step5.0] No texts loaded; cannot fit models")

    # Ensure IDs align
    if any(not x for x in doc_ids) or any(not x for x in chunk_ids):
        raise RuntimeError("[step5.0] Empty doc_id/chunk_id found after loading")

    print(f"[step5.0] loaded {len(texts)} chunks")
    if missing["text_norm"]:
        print(f"[step5.0] note: {missing['text_norm']} rows lacked text_norm and used text fallback")

    vec = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        max_features=max_features,
        strip_accents="unicode",
    )

    X = vec.fit_transform(texts)
    joblib.dump(vec, f"{OUT_DIR}/vectorizer.joblib")
    sp.save_npz(f"{OUT_DIR}/X_full_tfidf.npz", X)

    svd = TruncatedSVD(n_components=n_components, random_state=42)
    Z = svd.fit_transform(X)
    joblib.dump(svd, f"{OUT_DIR}/svd.joblib")

    pd.DataFrame({"row": range(len(texts)), "doc_id": doc_ids, "chunk_id": chunk_ids}).to_csv(
        f"{OUT_DIR}/row_alignment.csv", index=False
    )

    out_meta = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "chunks_path": CHUNKS_PATH,
        "chunks_sha256": file_sha256(CHUNKS_PATH),
        "n_chunks": len(texts),
        "missing_counts": missing,
        "vectorizer": {
            "stop_words": "english",
            "ngram_range": [1, 2],
            "min_df": 2,
            "max_df": 0.95,
            "max_features": max_features,
        },
        "reducer": {"model": "TruncatedSVD", "n_components": n_components, "random_state": 42},
    }
    with open(f"{OUT_DIR}/models_metadata.json", "w", encoding="utf-8") as f:
        json.dump(out_meta, f, indent=2)

    pc_cols = [f"PC{i+1}" for i in range(n_components)]
    df_scores = pd.DataFrame(Z, columns=pc_cols)
    df_scores.insert(0, "chunk_id", chunk_ids)
    df_scores.insert(1, "doc_id", doc_ids)
    df_scores.to_csv(f"{OUT_DIR}/full_scores_from_models.csv", index=False)

    ev = pd.DataFrame(
        {
            "pc": pc_cols,
            "explained_variance_ratio": svd.explained_variance_ratio_,
            "cumulative": pd.Series(svd.explained_variance_ratio_).cumsum(),
        }
    )
    ev.to_csv(f"{OUT_DIR}/explained_variance.csv", index=False)

    feature_names = vec.get_feature_names_out()
    comps = svd.components_
    rows = []
    for i in range(n_components):
        w = comps[i]
        top_pos = feature_names[w.argsort()[-TOP_N_TERMS:]][::-1]
        top_neg = feature_names[w.argsort()[:TOP_N_TERMS]]
        rows.append(
            {
                "pc": f"PC{i+1}",
                "top_positive_terms": "; ".join(top_pos),
                "top_negative_terms": "; ".join(top_neg),
            }
        )
    pd.DataFrame(rows).to_csv(f"{OUT_DIR}/top_terms.csv", index=False)

    print("[step5.0] wrote models to:", OUT_DIR)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--chunks_jsonl",
        required=True,
        help="Input chunks JSONL (e.g., full corpus or education subset).",
    )
    ap.add_argument(
        "--out_dir",
        required=True,
        help="Output directory for models and term tables.",
    )
    ap.add_argument(
        "--n_components",
        type=int,
        default=40,
        help="Number of components for SVD/PCA-style decomposition.",
    )
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Override hardcoded defaults at runtime ( args exists here)
    CHUNKS_PATH = args.chunks_jsonl
    OUT_DIR = args.out_dir
    os.makedirs(OUT_DIR, exist_ok=True)

    # Ensure CLI controls n_components (avoid META_IN overriding)
    main_params = {"n_components": args.n_components}

    # Run
    main_params_json = json.dumps(main_params)

    tmp_meta = f"{OUT_DIR}/_runtime_params.json"
    with open(tmp_meta, "w", encoding="utf-8") as f:
        f.write(main_params_json)

    META_IN = tmp_meta
    main()

