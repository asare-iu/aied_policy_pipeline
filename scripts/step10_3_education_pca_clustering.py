#!/usr/bin/env python3
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

INPUT = Path("data/derived/step10_education_dataset/education_country_dataset.csv")
OUTDIR = Path("data/derived/step10_education_dataset")

def zscore(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    mu = s.mean()
    sd = s.std(ddof=0)
    if pd.isna(sd) or sd == 0:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - mu) / sd

def main():
    t0 = time.time()
    OUTDIR.mkdir(parents=True, exist_ok=True)

    print("[step10_3] loading:", INPUT)
    df = pd.read_csv(INPUT)

    out = df.copy()
    out["regulatory_intensity_index"] = (
        zscore(out["n_rule_share"]) +
        zscore(out["rule_to_norm_ratio"]) +
        zscore(out["strong_deontic_share"]) +
        zscore(out["pct_o_local_present"])
    ) / 4.0

    out["strategic_orientation_index"] = zscore(out["n_strategy_share"])
    out["institutional_complexity_index"] = (
        zscore(out["mean_c_count"]) +
        zscore(out["pct_a_explicit"]) +
        zscore(out["pct_b_found"]) +
        zscore(out["pct_c_text_present"])
    ) / 4.0

    out["education_actor_centrality_index"] = (
        zscore(out["pct_edu_actor_A"]) +
        zscore(out["pct_edu_domain"])
    ) / 2.0

    out["governance_density_index"] = (
        zscore(out["n_statements"]) +
        zscore(out["statements_per_doc"]) +
        zscore(out["n_docs"])
    ) / 3.0

    feature_candidates = [
        "n_docs", "n_statements", "statements_per_doc",
        "n_strategy_share", "n_norm_share", "n_rule_share",
        "rule_to_norm_ratio", "rule_to_strategy_ratio",
        "pct_a_explicit", "pct_a_inferred", "pct_a_conjoined",
        "mean_c_count", "pct_b_found", "pct_o_local_present",
        "pct_i_conjoined", "pct_a_raw_present", "pct_b_text_present",
        "pct_c_text_present", "pct_edu_actor_A", "pct_edu_domain",
        "strong_deontic_share",
    ]
    features = [c for c in feature_candidates if c in out.columns]

    X = out[features].copy()
    prep = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    Xs = prep.fit_transform(X)

    pca = PCA(n_components=min(5, Xs.shape[1]), random_state=7)
    coords = pca.fit_transform(Xs)

    pca_df = pd.DataFrame({
        "country": out["country"],
        "pca1": coords[:, 0] if coords.shape[1] >= 1 else np.nan,
        "pca2": coords[:, 1] if coords.shape[1] >= 2 else np.nan,
        "pca3": coords[:, 2] if coords.shape[1] >= 3 else np.nan,
        "pca4": coords[:, 3] if coords.shape[1] >= 4 else np.nan,
        "pca5": coords[:, 4] if coords.shape[1] >= 5 else np.nan,
    })

    explained = pd.DataFrame({
        "component": [f"PC{i+1}" for i in range(pca.n_components_)],
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "cumulative_variance_ratio": np.cumsum(pca.explained_variance_ratio_),
    })

    diag_rows = []
    best_k = None
    best_sil = -np.inf
    n = len(out)
    for k in range(2, min(6, n - 1) + 1):
        km = KMeans(n_clusters=k, random_state=7, n_init=25)
        labels = km.fit_predict(Xs)
        sil = silhouette_score(Xs, labels) if len(set(labels)) > 1 else np.nan
        ch = calinski_harabasz_score(Xs, labels) if len(set(labels)) > 1 else np.nan
        db = davies_bouldin_score(Xs, labels) if len(set(labels)) > 1 else np.nan
        diag_rows.append({"k": k, "silhouette": sil, "calinski_harabasz": ch, "davies_bouldin": db})
        if np.isfinite(sil) and sil > best_sil:
            best_sil = sil
            best_k = k

    diagnostics = pd.DataFrame(diag_rows)
    final_km = KMeans(n_clusters=best_k, random_state=7, n_init=25)
    out["cluster_id"] = final_km.fit_predict(Xs)

    indices = out[[
        "country",
        "regulatory_intensity_index",
        "strategic_orientation_index",
        "institutional_complexity_index",
        "education_actor_centrality_index",
        "governance_density_index",
        "cluster_id",
    ]].copy()

    assign = out[["country", "cluster_id"]].merge(pca_df, on="country", how="left").merge(indices, on=["country", "cluster_id"], how="left")

    indices.to_csv(OUTDIR / "education_governance_indices.csv", index=False)
    pca_df.to_csv(OUTDIR / "education_pca_coordinates.csv", index=False)
    explained.to_csv(OUTDIR / "education_pca_explained_variance.csv", index=False)
    assign.to_csv(OUTDIR / "education_cluster_assignments.csv", index=False)
    diagnostics.to_csv(OUTDIR / "education_cluster_diagnostics.csv", index=False)

    print("[step10_3] best_k=", best_k, "silhouette=", round(best_sil, 4))
    print("[step10_3] wrote outputs to", OUTDIR)
    print("[step10_3] done elapsed_s=", round(time.time() - t0, 2))

if __name__ == "__main__":
    main()
