import pandas as pd
from pathlib import Path

INPUT = "data/derived/education_igt/education_igt_statements.parquet"
OUTDIR = Path("data/derived/step10_education_dataset")
OUTDIR.mkdir(parents=True, exist_ok=True)

print("Loading education institutional statements...")
df = pd.read_parquet(INPUT)

print("Statements:", len(df))
print("Countries:", df["country"].nunique())

df.to_parquet(OUTDIR / "education_igt_statements.parquet")

print("Saved:", OUTDIR / "education_igt_statements.parquet")
