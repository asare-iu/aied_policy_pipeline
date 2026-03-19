#!/usr/bin/env python3
import pandas as pd
from pathlib import Path

ASSIGN = Path("data/derived/step10_education_dataset/education_cluster_assignments.csv")
DATA = Path("data/derived/step10_education_dataset/education_country_dataset.csv")
IND = Path("data/derived/step10_education_dataset/education_governance_indices.csv")
OUTDIR = Path("data/derived/step10_education_dataset/qc")
OUTDIR.mkdir(parents=True, exist_ok=True)

assign = pd.read_csv(ASSIGN)
data = pd.read_csv(DATA)
ind = pd.read_csv(IND)

df = assign.merge(ind, on="country", how="left", suffixes=("", "_ind"))
df = df.merge(data, on="country", how="left", suffixes=("", "_data"))

if "cluster_id" not in df.columns and "cluster_id_ind" in df.columns:
    df = df.rename(columns={"cluster_id_ind": "cluster_id"})

agg_spec = {"country": ("country", "nunique")}

for col in [
    "regulatory_intensity_index",
    "strategic_orientation_index",
    "institutional_complexity_index",
    "education_actor_centrality_index",
    "governance_density_index",
    "n_docs",
    "n_statements",
    "n_strategy_share",
    "n_norm_share",
    "n_rule_share",
    "strong_deontic_share",
    "pct_o_local_present",
]:
    if col in df.columns:
        agg_spec[col] = (col, "mean")

profiles = (
    df.groupby("cluster_id")
      .agg(**agg_spec)
      .reset_index()
      .rename(columns={"country": "n_countries"})
)

member_cols = [c for c in [
    "country", "cluster_id", "pca1", "pca2",
    "regulatory_intensity_index", "strategic_orientation_index",
    "institutional_complexity_index", "education_actor_centrality_index",
    "governance_density_index"
] if c in df.columns]

members = df.sort_values(["cluster_id", "country"])[member_cols]

profiles.to_csv(OUTDIR / "education_cluster_profiles.csv", index=False)
members.to_csv(OUTDIR / "education_cluster_members.csv", index=False)

print("wrote", OUTDIR / "education_cluster_profiles.csv")
print("wrote", OUTDIR / "education_cluster_members.csv")
print(profiles)
