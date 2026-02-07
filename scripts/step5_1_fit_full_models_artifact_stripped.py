#!/usr/bin/env python3
import os, re, json
import pandas as pd
from tqdm import tqdm
import joblib
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

CHUNKS_PATH = "data/derived/step4_chunks_tagged/chunks_normalized_tagged.jsonl"
OUT_DIR = "data/derived/step5_models_full_artifact_stripped_40pc"
ARTIFACT_TOKENS_PATH = "config/artifact_tokens.txt"
META_IN = "data/derived/step5_pca_full/pca_full_metadata.json"

def iter_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def load_artifact_pattern(path):
    toks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            t = line.strip()
            if t and not t.startswith("#"):
                toks.append(re.escape(t))
    if not toks:
        return None
    # whole-word match, case-insensitive
    return re.compile(r"\b(" + "|".join(toks) + r")\b", flags=re.IGNORECASE)

def strip_artifacts(text, pat):
    if not pat:
        return text
    # replace with space, then collapse whitespace
    text = pat.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    params = {}
    if os.path.exists(META_IN):
        with open(META_IN, "r", encoding="utf-8") as f:
            params = json.load(f)

    max_features = int(params.get("tfidf_max_features", 50000))
    n_components = int(params.get("n_components", 40))

    pat = load_artifact_pattern(ARTIFACT_TOKENS_PATH)

    texts, doc_ids, chunk_ids = [], [], []
    n_stripped = 0

    for rec in tqdm(iter_jsonl(CHUNKS_PATH), desc="[step5.1] loading + stripping"):
        text = rec.get("text_norm") or rec.get("text") or ""
        before = text
        after = strip_artifacts(before, pat)
        if after != before:
            n_stripped += 1

        doc_id = str(rec.get("doc_id") or "")
        chunk_id = str(rec.get("chunk_id") or "")
        texts.append(after)
        doc_ids.append(doc_id)
        chunk_ids.append(chunk_id)

    print(f"[step5.1] loaded {len(texts)} chunks; stripped artifacts in {n_stripped} chunks")

    vec = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1,2),
        min_df=2,
        max_df=0.95,
        max_features=max_features,
        strip_accents="unicode"
    )
    X = vec.fit_transform(texts)
    joblib.dump(vec, f"{OUT_DIR}/vectorizer.joblib")
    sp.save_npz(f"{OUT_DIR}/X_full_tfidf.npz", X)

    svd = TruncatedSVD(n_components=n_components, random_state=42)
    Z = svd.fit_transform(X)
    joblib.dump(svd, f"{OUT_DIR}/svd.joblib")

    pd.DataFrame({"row": range(len(texts)), "doc_id": doc_ids, "chunk_id": chunk_ids}).to_csv(
        f"{OUT_DIR}/row_alignment.csv", index=False
    )

    pc_cols = [f"PC{i+1}" for i in range(n_components)]
    df_scores = pd.DataFrame(Z, columns=pc_cols)
    df_scores.insert(0, "chunk_id", chunk_ids)
    df_scores.insert(1, "doc_id", doc_ids)
    df_scores.to_csv(f"{OUT_DIR}/full_scores_from_models.csv", index=False)

    ev = pd.DataFrame({
        "pc": pc_cols,
        "explained_variance_ratio": svd.explained_variance_ratio_,
        "cumulative": pd.Series(svd.explained_variance_ratio_).cumsum()
    })
    ev.to_csv(f"{OUT_DIR}/explained_variance.csv", index=False)

    feature_names = vec.get_feature_names_out()
    comps = svd.components_
    rows = []
    top_n = 30
    for i in range(n_components):
        w = comps[i]
        top_pos = feature_names[w.argsort()[-top_n:]][::-1]
        top_neg = feature_names[w.argsort()[:top_n]]
        rows.append({
            "pc": f"PC{i+1}",
            "top_positive_terms": "; ".join(top_pos),
            "top_negative_terms": "; ".join(top_neg),
        })
    pd.DataFrame(rows).to_csv(f"{OUT_DIR}/top_terms.csv", index=False)

    with open(f"{OUT_DIR}/models_metadata.json", "w", encoding="utf-8") as f:
        json.dump({
            "chunks_path": CHUNKS_PATH,
            "n_chunks": len(texts),
            "n_components": n_components,
            "tfidf_max_features": max_features,
            "artifact_tokens_path": ARTIFACT_TOKENS_PATH,
            "n_chunks_artifacts_stripped": n_stripped
        }, f, indent=2)

    print("[step5.1] wrote:", OUT_DIR)

if __name__ == "__main__":
    main()
