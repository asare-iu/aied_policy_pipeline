import os
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

IN_PATH = "evidence/egypt_pilot/04_edu_filter/egypt_sentences_education.tsv"
OUT_DIR = "methods/embeddings"
OUT_PATH = os.path.join(OUT_DIR, "egypt_edu_sentences_embeddings.parquet")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 32

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_csv(IN_PATH, sep="\t", dtype=str).fillna("")
    df = df[df["sentence"].str.strip() != ""].copy()

    print("Rows loaded:", len(df))

    model = SentenceTransformer(MODEL_NAME)

    texts = df["sentence"].tolist()
    embs = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Batches"):
        batch = texts[i:i+BATCH_SIZE]
        e = model.encode(batch, show_progress_bar=False, normalize_embeddings=True)
        embs.extend(list(e))

    out = df[["country","chunk_id","sent_id","source_doc","chunk_file","chunk_path","sentence","bytes"]].copy()
    out["embedding"] = embs
    out.to_parquet(OUT_PATH, index=False)

    print("Wrote embeddings to:", OUT_PATH)
    print("Embedding dimension:", len(embs[0]) if embs else "NA")
    print("Duplicate sent_ids:", out["sent_id"].duplicated().sum())

if __name__ == "__main__":
    main()
