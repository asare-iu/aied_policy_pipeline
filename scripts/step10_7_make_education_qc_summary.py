#!/usr/bin/env python3
import pandas as pd
from pathlib import Path

DIAG = Path("data/derived/step10_education_dataset/education_cluster_diagnostics.csv")
EXP = Path("data/derived/step10_education_dataset/education_pca_explained_variance.csv")
ASSIGN = Path("data/derived/step10_education_dataset/education_cluster_assignments.csv")
OUTDIR = Path("data/derived/step10_education_dataset/qc")
OUTDIR.mkdir(parents=True, exist_ok=True)

diag = pd.read_csv(DIAG)
exp = pd.read_csv(EXP)
assign = pd.read_csv(ASSIGN)

best = diag.sort_values("silhouette", ascending=False).iloc[0]

summary = pd.DataFrame([
    {"metric": "best_k", "value": best["k"]},
    {"metric": "best_silhouette", "value": best["silhouette"]},
    {"metric": "pc1_variance", "value": exp.loc[exp["component"]=="PC1", "explained_variance_ratio"].iloc[0]},
    {"metric": "pc2_variance", "value": exp.loc[exp["component"]=="PC2", "explained_variance_ratio"].iloc[0]},
    {"metric": "pc1_pc2_cumulative", "value": exp.loc[exp["component"]=="PC2", "cumulative_variance_ratio"].iloc[0]},
    {"metric": "n_clusters_realized", "value": assign["cluster_id"].nunique()},
    {"metric": "largest_cluster_size", "value": assign["cluster_id"].value_counts().max()},
    {"metric": "smallest_cluster_size", "value": assign["cluster_id"].value_counts().min()},
])

summary.to_csv(OUTDIR / "education_pca_cluster_qc_summary.csv", index=False)
print("wrote", OUTDIR / "education_pca_cluster_qc_summary.csv")
