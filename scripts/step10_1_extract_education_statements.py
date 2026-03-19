#!/usr/bin/env python3
import pandas as pd
from pathlib import Path

INPUT = "data/derived/step8_9_regime_closure/igt_statements_full_with_edu_flags.parquet"
LOOKUP = "data/derived/step9_country_dataset/doc_country_lookup.csv"
OUTDIR = Path("data/derived/step10_education_dataset")
OUTDIR.mkdir(parents=True, exist_ok=True)

print("[step10_1] loading:", INPUT)
df = pd.read_parquet(INPUT)

if "edu_any_hit" not in df.columns:
    raise ValueError("[step10_1] expected column 'edu_any_hit' not found")

edu = df[df["edu_any_hit"].fillna(False).astype(bool)].copy()

lookup = pd.read_csv(LOOKUP)[["doc_id", "country"]].drop_duplicates()
edu = edu.merge(lookup, on="doc_id", how="left")

print("[step10_1] rows:", len(edu))
print("[step10_1] docs:", edu["doc_id"].nunique())
print("[step10_1] countries:", edu["country"].nunique())
print("[step10_1] missing countries:", edu["country"].isna().sum())

out = OUTDIR / "education_igt_statements.parquet"
edu.to_parquet(out, index=False)

print("[step10_1] wrote:", out)
