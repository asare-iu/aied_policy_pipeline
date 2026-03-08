from pathlib import Path
import hashlib
import pandas as pd
import pdfminer.high_level

RAW = Path("data/raw/corpus_raw/AI Policies")
OUT = Path("data/derived/step9_country_dataset/doc_country_lookup.csv")

def normalize(text):
    return " ".join(text.split()).lower()

rows = []

for country_dir in RAW.iterdir():
    if not country_dir.is_dir():
        continue

    country = country_dir.name

    for pdf in country_dir.glob("*.pdf"):

        try:
            text = pdfminer.high_level.extract_text(str(pdf))
        except Exception:
            continue

        norm = normalize(text)
        doc_id = hashlib.sha256(norm.encode()).hexdigest()[:16]

        rows.append({
            "doc_id": doc_id,
            "country": country,
            "filename": pdf.name
        })

df = pd.DataFrame(rows)

OUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT, index=False)

print("wrote:", OUT)
print("rows:", len(df))
print("countries:", df["country"].nunique())
