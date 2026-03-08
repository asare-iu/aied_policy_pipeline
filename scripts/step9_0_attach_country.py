python - <<'PY'
import pandas as pd
from pathlib import Path

BASE = Path("data/derived/step8_igt_full")

INPUT = BASE / "igt_statements.parquet"
OUTPUT = Path("data/derived/step9_country_dataset/igt_with_country.parquet")

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(INPUT)

def extract_country(path):
    p = Path(path)
    return p.parts[-2] if len(p.parts) > 1 else "unknown"

if "source_file" in df.columns:
    df["country"] = df["source_file"].apply(extract_country)
elif "file_path" in df.columns:
    df["country"] = df["file_path"].apply(extract_country)
else:
    raise ValueError("No path column found to extract country.")

df.to_parquet(OUTPUT)

print("wrote", OUTPUT, "rows=", len(df))
print("countries:", df["country"].nunique())
PY
