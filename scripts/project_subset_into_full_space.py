#!/usr/bin/env python3
"""
Project a subset JSONL of chunks into the  full-corpus TF-IDF+SVD space.

Requires:
  data/derived/step5_models_full/vectorizer.joblib
  data/derived/step5_models_full/svd.joblib

Writes:
  out_csv with columns: chunk_id, doc_id, PC1..PCk
"""
import argparse, json
import pandas as pd
from tqdm import tqdm
import joblib


def iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset_jsonl", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--models_dir", default="data/derived/step5_models_full")
    args = ap.parse_args()

    vec = joblib.load(f"{args.models_dir}/vectorizer.joblib")
    svd = joblib.load(f"{args.models_dir}/svd.joblib")
    k = svd.n_components

    texts, doc_ids, chunk_ids = [], [], []

    for rec in tqdm(iter_jsonl(args.subset_jsonl), desc="[project] load subset"):
        if "doc_id" not in rec or "chunk_id" not in rec:
            raise RuntimeError("subset rows must include doc_id and chunk_id")
        t = rec.get("text_norm")
        if t is None or not str(t).strip():
            t = rec.get("text", "")
        texts.append(str(t))
        doc_ids.append(str(rec["doc_id"]))
        chunk_ids.append(str(rec["chunk_id"]))

    X = vec.transform(texts)
    Z = svd.transform(X)

    pc_cols = [f"PC{i+1}" for i in range(k)]
    df = pd.DataFrame(Z, columns=pc_cols)
    df.insert(0, "chunk_id", chunk_ids)
    df.insert(1, "doc_id", doc_ids)
    df.to_csv(args.out_csv, index=False)
    print("[project] wrote:", args.out_csv)


if __name__ == "__main__":
    main()
