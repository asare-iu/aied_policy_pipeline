import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

IN_PATH = "data/corpus/egypt_chunks.tsv"
OUT_PATH = "methods/embeddings/egypt_chunks_embeddings.parquet"

ID_COL = "chunk_id"
DOC_COL = "source_doc"
TEXT_COL = "text"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    df = pd.read_csv(IN_PATH, sep="\t", dtype=str)
    print("Rows loaded:", len(df))

    # basic column check
    for c in [ID_COL, DOC_COL, TEXT_COL]:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    df[TEXT_COL] = df[TEXT_COL].fillna("").astype(str)
    df = df[df[TEXT_COL].str.strip() != ""].copy()
    print("Rows after dropping empty text:", len(df))

    model = SentenceTransformer(MODEL_NAME)

    embeddings = model.encode(
        df[TEXT_COL].tolist(),
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    df_out = df[["country", ID_COL, DOC_COL, "chunk_file", "path", "bytes"]].copy()
    df_out["embedding"] = [e.astype(np.float32) for e in embeddings]

    df_out.to_parquet(OUT_PATH, index=False)

    print("Wrote embeddings to:", OUT_PATH)
    print("Embedding dimension:", len(df_out["embedding"].iloc[0]))
    print("Duplicate chunk_ids:", df_out.duplicated(subset=[ID_COL]).sum())

if __name__ == "__main__":
    main()
