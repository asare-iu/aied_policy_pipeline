#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity


def load_sparse_matrix(path: Path):
    # supports .npz saved via scipy.sparse.save_npz
    return sparse.load_npz(path)


def top_terms_from_components(components, feature_names, topn=50):
    # components: (n_components, n_features)
    out = []
    for k in range(components.shape[0]):
        w = components[k]
        idx = np.argsort(np.abs(w))[::-1][:topn]
        terms = [feature_names[i] for i in idx]
        out.append(terms)
    return out


def jaccard(a, b):
    sa, sb = set(a), set(b)
    return len(sa & sb) / max(1, len(sa | sb))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True, help="e.g. data/derived/step5_models_full_40pc")
    ap.add_argument("--x-npz", default=None, help="Path to X_full_tfidf.npz (defaults to <model-dir>/X_full_tfidf.npz)")
    ap.add_argument("--svd", default=None, help="Path to svd.joblib (defaults to <model-dir>/svd.joblib)")
    ap.add_argument("--vectorizer", default=None, help="Path to vectorizer.joblib (defaults to <model-dir>/vectorizer.joblib)")
    ap.add_argument("--pcs", type=int, default=40)
    ap.add_argument("--subsample", type=float, default=0.8)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--topn", type=int, default=50)
    ap.add_argument("--outdir", default="data/derived/step5_8_pca_validation", help="Output folder")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    x_path = Path(args.x_npz) if args.x_npz else model_dir / "X_full_tfidf.npz"
    svd_path = Path(args.svd) if args.svd else model_dir / "svd.joblib"
    vec_path = Path(args.vectorizer) if args.vectorizer else model_dir / "vectorizer.joblib"

    for p in [x_path, svd_path, vec_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    print(f"[load] X={x_path}")
    X = load_sparse_matrix(x_path)
    n_rows = X.shape[0]
    print(f"[load] X shape = {X.shape}")

    svd0 = joblib.load(svd_path)
    vec = joblib.load(vec_path)
    feature_names = np.array(getattr(vec, "get_feature_names_out")())

    comps0 = svd0.components_[: args.pcs]
    top0 = top_terms_from_components(comps0, feature_names, topn=args.topn)

    # store run params
    (outdir / "pca_validation_params.json").write_text(json.dumps({
        "model_dir": str(model_dir),
        "x_path": str(x_path),
        "svd_path": str(svd_path),
        "vectorizer_path": str(vec_path),
        "pcs": args.pcs,
        "subsample": args.subsample,
        "seeds": args.seeds,
        "topn": args.topn,
        "n_rows": int(n_rows),
        "n_features": int(X.shape[1]),
    }, indent=2))

    # per-seed results
    diag_rows = []
    jacc_rows = []

    # We’ll align each refit PC to the best-matching original PC via max abs cosine.
    for seed in range(args.seeds):
        rng = np.random.default_rng(seed)
        m = int(n_rows * args.subsample)
        idx = rng.choice(n_rows, size=m, replace=False)
        Xs = X[idx]

        svd = TruncatedSVD(n_components=args.pcs, random_state=seed)
        svd.fit(Xs)
        comps = svd.components_

        # abs cosine similarities (pcs x pcs)
        S = np.abs(cosine_similarity(comps0, comps))
        # greedy matching (good enough; avoids heavy Hungarian)
        used = set()
        match = {}
        for i in range(args.pcs):
            j = int(np.argmax(S[i]))
            while j in used:
                S[i, j] = -1  # force new choice
                j = int(np.argmax(S[i]))
            used.add(j)
            match[i] = j

        # diagonal summary after matching
        diag = [float(np.abs(cosine_similarity(comps0[i:i+1], comps[match[i]:match[i]+1])[0,0])) for i in range(args.pcs)]
        diag_rows.append({
            "seed": seed,
            **{f"pc{i+1:02d}": diag[i] for i in range(args.pcs)}
        })

        # top-term Jaccard after matching
        top = top_terms_from_components(comps, feature_names, topn=args.topn)
        for i in range(min(10, args.pcs)):
            j = match[i]
            jac = jaccard(top0[i], top[j])
            jacc_rows.append({
                "seed": seed,
                "pc": f"PC{i+1}",
                "matched_pc": f"PC{j+1}",
                "jaccard_top_terms": float(jac),
            })

        print(f"[seed {seed}] done | median abs-cos(PC1-10)={np.median(diag[:10]):.3f} | mean Jacc(PC1-10)={pd.DataFrame(jacc_rows).query('seed==@seed')['jaccard_top_terms'].mean():.3f}")

    df_diag = pd.DataFrame(diag_rows)
    df_diag.to_csv(outdir / "pca_component_abs_cosine_by_seed.csv", index=False)

    df_j = pd.DataFrame(jacc_rows)
    df_j.to_csv(outdir / "pca_top_terms_jaccard_pc1_10_by_seed.csv", index=False)

    # summary: median across seeds for each PC
    pc_cols = [c for c in df_diag.columns if c.startswith("pc")]
    summary = pd.DataFrame({
        "pc": [f"PC{i+1}" for i in range(args.pcs)],
        "median_abs_cosine": [float(df_diag[f"pc{i+1:02d}"].median()) for i in range(args.pcs)],
        "p10_abs_cosine": [float(df_diag[f"pc{i+1:02d}"].quantile(0.10)) for i in range(args.pcs)],
    })
    summary.to_csv(outdir / "pca_component_abs_cosine_summary.csv", index=False)

    print(f"[ok] wrote: {outdir}/pca_component_abs_cosine_by_seed.csv")
    print(f"[ok] wrote: {outdir}/pca_component_abs_cosine_summary.csv")
    print(f"[ok] wrote: {outdir}/pca_top_terms_jaccard_pc1_10_by_seed.csv")


if __name__ == "__main__":
    main()
