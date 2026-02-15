#!/usr/bin/env python3
"""
Compute dispersion/coherence metrics from chunk-level PC score CSV.

Inputs:
  --scores_csv: chunk_id, doc_id, PC1..PCk
  --doc2country_csv: doc_id,country

Outputs:
  --out_doc_csv: per-doc variance of chunk PCs (coherence proxy)
  --out_country_csv: variance across docs within each country
"""
import argparse
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores_csv", required=True)
    ap.add_argument("--doc2country_csv", required=True)
    ap.add_argument("--out_doc_csv", required=True)
    ap.add_argument("--out_country_csv", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.scores_csv)
    d2c = pd.read_csv(args.doc2country_csv)

    if "doc_id" not in df.columns:
        raise RuntimeError("scores_csv must include doc_id")

    m = dict(zip(d2c["doc_id"].astype(str), d2c["country"]))
    df["doc_id"] = df["doc_id"].astype(str)
    df["country"] = df["doc_id"].map(m)

    pc_cols = [c for c in df.columns if c.startswith("PC")]
    if not pc_cols:
        raise RuntimeError("No PC columns found in scores_csv")

    doc_var = df.groupby("doc_id", as_index=False)[pc_cols].var()
    doc_var["country"] = doc_var["doc_id"].map(m)
    doc_var.to_csv(args.out_doc_csv, index=False)

    doc_mean = df.groupby("doc_id", as_index=False)[pc_cols].mean()
    doc_mean["country"] = doc_mean["doc_id"].map(m)
    country_var = doc_mean.groupby("country", as_index=False)[pc_cols].var()
    country_var.to_csv(args.out_country_csv, index=False)

    print("[dispersion] wrote:", args.out_doc_csv)
    print("[dispersion] wrote:", args.out_country_csv)


if __name__ == "__main__":
    main()

