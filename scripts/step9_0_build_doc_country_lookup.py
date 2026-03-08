from pathlib import Path
import pandas as pd

MANIFEST = "data/manifests/corpus_manifest.csv"
OUT = "data/derived/step9_country_dataset/doc_country_lookup.csv"

df = pd.read_csv(MANIFEST)

# extract country from path
df["country"] = df["filepath"].apply(
    lambda x: Path(x).parts[-2]
)

lookup = df[["doc_id", "country"]].drop_duplicates()

Path(OUT).parent.mkdir(parents=True, exist_ok=True)
lookup.to_csv(OUT, index=False)

print("rows:", len(lookup))
print("countries:", lookup["country"].nunique())
print("wrote:", OUT)
