#!/usr/bin/env python3
"""
Step 9.3 — Governance indices + PCA + clustering

Input
-----
data/derived/step9_country_dataset/country_governance_dataset.csv

Outputs
-------
data/derived/step9_country_dataset/country_governance_indices.csv
data/derived/step9_country_dataset/country_pca_coordinates.csv
data/derived/step9_country_dataset/country_cluster_assignments.csv
data/derived/step9_country_dataset/country_cluster_centroids_standardized.csv
data/derived/step9_country_dataset/country_cluster_diagnostics.csv

Design principle
----------------
This script does not assume any particular governance typology.
Indices are transparent summaries.
Clustering is performed on standardized raw features, not on hand-labeled outcomes.
"""

from __future__ import annotations

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


INPUT = Path("data/derived/step9_country_dataset/country_governance_dataset.csv")
OUTDIR = Path("data/derived/step9_country_dataset")


def zscore(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    mu = s.mean()
    sd = s.std(ddof=0)
    if pd.isna(sd) or sd == 0:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - mu) / sd


def main() -> None:
    t0 = time.time()
    OUTDIR.mkdir(parents=True, exist_ok=True)

    print("[step9_3] loading:", INPUT)
    df = pd.read_csv(INPUT)

    if "country" not in df.columns:
        raise ValueError("[step9_3] expected column 'country' not found")

    print("[step9_3] countries:", len(df))
    print("[step9_3] columns:", len(df.columns))

    out = df.copy()

    # ---------------------------------------------------------
    # Transparent governance indices
    # ---------------------------------------------------------
    # These are descriptive summaries only.
    # Clustering below uses standardized raw features separately.

    # 1. Regulatory intensity
    reg_components = []
    for c in ["n_rule_share", "rule_to_norm_ratio", "strong_deontic_share", "pct_o_local_present"]:
        if c in out.columns:
            reg_components.append(zscore(out[c]))
    out["regulatory_intensity_index"] = sum(reg_components) / len(reg_components) if reg_components else np.nan

    # 2. Strategic orientation
    strat_components = []
    for c in ["n_strategy_share"]:
        if c in out.columns:
            strat_components.append(zscore(out[c]))
    out["strategic_orientation_index"] = sum(strat_components) / len(strat_components) if strat_components else np.nan

    # 3. Institutional complexity
    complexity_components = []
    for c in ["mean_c_count", "pct_a_explicit", "pct_b_found", "pct_c_text_present"]:
        if c in out.columns:
            complexity_components.append(zscore(out[c]))
    out["institutional_complexity_index"] = sum(complexity_components) / len(complexity_components) if complexity_components else np.nan

    # 4. Governance density
    density_components = []
    for c in ["n_statements", "statements_per_doc", "n_docs"]:
        if c in out.columns:
            density_components.append(zscore(out[c]))
    out["governance_density_index"] = sum(density_components) / len(density_components) if density_components else np.nan

    # 5. Normativity / obligation orientation
    norm_components = []
    for c in ["n_norm_share", "dclass_obligation", "pct_a_explicit"]:
        if c in out.columns:
            norm_components.append(zscore(out[c]))
    out["normative_orientation_index"] = sum(norm_components) / len(norm_components) if norm_components else np.nan

    # Keep index output
    index_cols = [
        "country",
        "regulatory_intensity_index",
        "strategic_orientation_index",
        "institutional_complexity_index",
        "governance_density_index",
        "normative_orientation_index",
    ]
    index_cols += [c for c in out.columns if c.startswith("z_")]
    indices = out[[c for c in index_cols if c in out.columns]].copy()

    # ---------------------------------------------------------
    # PCA + clustering use standardized raw features
    # ---------------------------------------------------------
    feature_candidates = [
        "n_docs",
        "n_statements",
        "statements_per_doc",
        "n_strategy_share",
        "n_norm_share",
        "n_rule_share",
        "rule_to_norm_ratio",
        "rule_to_strategy_ratio",
        "pct_a_explicit",
        "pct_a_inferred",
        "pct_a_conjoined",
        "mean_c_count",
        "pct_b_found",
        "pct_o_local_present",
        "pct_i_conjoined",
        "pct_a_raw_present",
        "pct_b_text_present",
        "pct_c_text_present",
        "dclass_obligation",
        "dclass_permission",
        "dclass_prohibition",
        "strong_deontic_share",
    ]
    feature_cols = [c for c in feature_candidates if c in out.columns]

    print("[step9_3] clustering features:", feature_cols)

    X = out[feature_cols].copy()

    prep = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    Xs = prep.fit_transform(X)

    # ---------------------------------------------------------
    # PCA
    # ---------------------------------------------------------
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

    loadings = pd.DataFrame(
        pca.components_.T,
        index=feature_cols,
        columns=[f"PC{i+1}" for i in range(pca.n_components_)]
    )
    explained = pd.DataFrame({
        "component": [f"PC{i+1}" for i in range(pca.n_components_)],
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "cumulative_variance_ratio": np.cumsum(pca.explained_variance_ratio_),
    })

    # ---------------------------------------------------------
    # K-means diagnostics across candidate k
    # ---------------------------------------------------------
    diag_rows = []
    best_k = None
    best_sil = -np.inf
    best_labels = None

    n_countries = len(out)
    k_min = 2
    k_max = min(6, n_countries - 1)

    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=7, n_init=25)
        labels = km.fit_predict(Xs)

        if len(set(labels)) < 2:
            sil = np.nan
            ch = np.nan
            db = np.nan
        else:
            sil = silhouette_score(Xs, labels)
            ch = calinski_harabasz_score(Xs, labels)
            db = davies_bouldin_score(Xs, labels)

        diag_rows.append({
            "k": k,
            "silhouette": sil,
            "calinski_harabasz": ch,
            "davies_bouldin": db,
        })

        if np.isfinite(sil) and sil > best_sil:
            best_sil = sil
            best_k = k
            best_labels = labels

    diagnostics = pd.DataFrame(diag_rows)

    if best_k is None:
        raise RuntimeError("[step9_3] no valid cluster solution found")

    # final clustering
    final_km = KMeans(n_clusters=best_k, random_state=7, n_init=25)
    final_labels = final_km.fit_predict(Xs)

    assign = out[["country"]].copy()
    assign["cluster_id"] = final_labels
    assign = assign.merge(pca_df, on="country", how="left")

    centroids = pd.DataFrame(final_km.cluster_centers_, columns=feature_cols)
    centroids.insert(0, "cluster_id", range(best_k))

    # join indices onto assignments for convenience
    assign = assign.merge(indices, on="country", how="left")

    # ---------------------------------------------------------
    # Write outputs
    # ---------------------------------------------------------
    indices_out = OUTDIR / "country_governance_indices.csv"
    pca_out = OUTDIR / "country_pca_coordinates.csv"
    pca_loadings_out = OUTDIR / "country_pca_loadings.csv"
    pca_explained_out = OUTDIR / "country_pca_explained_variance.csv"
    assign_out = OUTDIR / "country_cluster_assignments.csv"
    centroids_out = OUTDIR / "country_cluster_centroids_standardized.csv"
    diagnostics_out = OUTDIR / "country_cluster_diagnostics.csv"

    indices.to_csv(indices_out, index=False)
    pca_df.to_csv(pca_out, index=False)
    loadings.to_csv(pca_loadings_out)
    explained.to_csv(pca_explained_out, index=False)
    assign.sort_values(["cluster_id", "country"]).to_csv(assign_out, index=False)
    centroids.to_csv(centroids_out, index=False)
    diagnostics.to_csv(diagnostics_out, index=False)

    print("[step9_3] wrote:", indices_out)
    print("[step9_3] wrote:", pca_out)
    print("[step9_3] wrote:", pca_loadings_out)
    print("[step9_3] wrote:", pca_explained_out)
    print("[step9_3] wrote:", assign_out)
    print("[step9_3] wrote:", centroids_out)
    print("[step9_3] wrote:", diagnostics_out)
    print(f"[step9_3] best_k={best_k} silhouette={best_sil:.4f}")
    print("[step9_3] done elapsed_s=", round(time.time() - t0, 2))


if __name__ == "__main__":
    main()
