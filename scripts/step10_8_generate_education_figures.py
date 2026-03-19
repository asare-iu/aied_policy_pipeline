#!/usr/bin/env python3
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = Path("data/derived/step10_education_dataset")
QCDIR = BASE / "qc"
FIGDIR = BASE / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

def save_fig(fig, stem):
    fig.tight_layout()
    fig.savefig(FIGDIR / f"{stem}.png", dpi=220)
    fig.savefig(FIGDIR / f"{stem}.pdf")
    plt.close(fig)
    print("wrote", FIGDIR / f"{stem}.png")
    print("wrote", FIGDIR / f"{stem}.pdf")

assign = pd.read_csv(BASE / "education_cluster_assignments.csv")
ind = pd.read_csv(BASE / "education_governance_indices.csv")
data = pd.read_csv(BASE / "education_country_dataset.csv")
exp = pd.read_csv(BASE / "education_pca_explained_variance.csv")
profiles = pd.read_csv(QCDIR / "education_cluster_profiles.csv")
comp = pd.read_csv(QCDIR / "overall_vs_education_country_comparison.csv")
top_rule = pd.read_csv(QCDIR / "education_top_rule_intensity.csv")

df = assign.merge(ind, on=["country", "cluster_id"], how="left").merge(data, on="country", how="left")

# fig1 PCA scatter
pc1 = exp.loc[exp["component"]=="PC1", "explained_variance_ratio"].iloc[0]
pc2 = exp.loc[exp["component"]=="PC2", "explained_variance_ratio"].iloc[0]
fig, ax = plt.subplots(figsize=(10,7))
for cid, g in df.groupby("cluster_id"):
    ax.scatter(g["pca1"], g["pca2"], label=f"Cluster {cid}", alpha=0.8)
for _, r in df.iterrows():
    ax.text(r["pca1"], r["pca2"], str(r["country"]), fontsize=7)
ax.set_title("Education Governance Profiles in PCA Space")
ax.set_xlabel(f"PC1 ({pc1:.1%} variance)")
ax.set_ylabel(f"PC2 ({pc2:.1%} variance)")
ax.legend(loc="best", fontsize=9)
save_fig(fig, "fig1_education_pca_scatter")

# fig2 regulatory vs strategic
fig, ax = plt.subplots(figsize=(10,7))
for cid, g in df.groupby("cluster_id"):
    ax.scatter(g["regulatory_intensity_index"], g["strategic_orientation_index"], label=f"Cluster {cid}", alpha=0.8)
for _, r in df.iterrows():
    ax.text(r["regulatory_intensity_index"], r["strategic_orientation_index"], str(r["country"]), fontsize=7)
ax.set_title("Education Regulatory Intensity vs Strategic Orientation")
ax.set_xlabel("Regulatory intensity index")
ax.set_ylabel("Strategic orientation index")
ax.legend(loc="best", fontsize=9)
save_fig(fig, "fig2_education_regulatory_vs_strategic")

# fig3 cluster profile heatmap
cols = [
    "regulatory_intensity_index", "strategic_orientation_index",
    "institutional_complexity_index", "education_actor_centrality_index",
    "governance_density_index", "n_strategy_share", "n_norm_share",
    "n_rule_share", "strong_deontic_share", "pct_o_local_present"
]
cols = [c for c in cols if c in profiles.columns]
plot_df = profiles.sort_values("cluster_id")
data_m = plot_df[cols].to_numpy()
fig, ax = plt.subplots(figsize=(10,5))
im = ax.imshow(data_m, aspect="auto")
ax.set_title("Education Cluster Profiles")
ax.set_yticks(np.arange(len(plot_df)))
ax.set_yticklabels([f"Cluster {int(x)}" for x in plot_df["cluster_id"]])
ax.set_xticks(np.arange(len(cols)))
ax.set_xticklabels(cols, rotation=30, ha="right", fontsize=9)
for i in range(data_m.shape[0]):
    for j in range(data_m.shape[1]):
        ax.text(j, i, f"{data_m[i,j]:.2f}", ha="center", va="center", fontsize=8)
fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
save_fig(fig, "fig3_education_cluster_heatmap")

# fig4 overall vs education mean deltas
delta_cols = [c for c in comp.columns if c.startswith("delta_")]
means = comp[delta_cols].mean(numeric_only=True).sort_values()
fig, ax = plt.subplots(figsize=(9,6))
ax.barh(means.index, means.values)
ax.set_title("Mean Difference: Education Governance minus Overall AI Governance")
ax.set_xlabel("Mean delta")
save_fig(fig, "fig4_overall_vs_education_mean_deltas")

# fig5 top rule intensity countries
fig, ax = plt.subplots(figsize=(9,7))
ax.barh(top_rule["country"][::-1], top_rule["n_rule_share"][::-1])
ax.set_title("Top Countries by Education Rule Share")
ax.set_xlabel("Education rule share")
save_fig(fig, "fig5_top_education_rule_share")
