from pathlib import Path
import hashlib
import pandas as pd

RAW = Path("data/raw/ai_policies_raw")
OUT = Path("data/derived/step9_country_dataset/doc_country_lookup.csv")

def sha16(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:16]

rows = []

for pdf in sorted(RAW.rglob("*.pdf")):
    rel = pdf.relative_to(RAW)
    parts = rel.parts

    # prefer folder name; fallback to filename prefix before " - "
    if len(parts) >= 2:
        country = parts[0]
    else:
        name = pdf.stem
        country = name.split(" - ")[0].strip() if " - " in name else "unknown"

    rows.append({
        "doc_id": sha16(pdf),
        "country": country,
        "raw_path": str(pdf),
        "filename": pdf.name,
    })

df = pd.DataFrame(rows).drop_duplicates(subset=["doc_id"])

OUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT, index=False)

print("wrote:", OUT)
print("rows:", len(df))
print("countries:", df["country"].nunique())
print(df.head(5))
