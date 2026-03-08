import pandas as pd
from pathlib import Path

IGT_PATH = "data/derived/step8_igt_full/igt_statements_full.parquet"
LOOKUP_PATH = "data/derived/step9_country_dataset/doc_country_lookup.csv"
OUT_PATH = "data/derived/step9_country_dataset/igt_with_country.parquet"

print("[step9_1] loading datasets")

igt = pd.read_parquet(IGT_PATH)
lookup = pd.read_csv(LOOKUP_PATH)

print("[step9_1] IGT rows:", len(igt))
print("[step9_1] lookup rows:", len(lookup))

lookup = lookup[["doc_id", "country"]].drop_duplicates()

merged = igt.merge(lookup, on="doc_id", how="left")

missing = merged["country"].isna().sum()

print("[step9_1] missing countries:", missing)
print("[step9_1] country coverage:", 1 - missing / len(merged))

Path("data/derived/step9_country_dataset").mkdir(parents=True, exist_ok=True)
merged.to_parquet(OUT_PATH)

print("[step9_1] wrote:", OUT_PATH)
print("[step9_1] countries:", merged["country"].nunique())
