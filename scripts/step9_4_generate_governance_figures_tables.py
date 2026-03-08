#!/usr/bin/env python3
"""
Step 9.4 — Generate governance figures and summary tables
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE = Path("data/derived/step9_country_dataset")
FIGDIR = BASE / "figures"
TABDIR = BASE / "tables"


def ensure_dirs() -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    TABDIR.mkdir(parents=True, exist_ok=True)


def save_fig(fig, stem: str) -> None:
    png = FIGDIR / f"{stem}.png"
    pdf = FIGDIR / f"{stem}.pdf"
    fig.tight_layout()
    fig.savefig(png, dpi=220)
    fig.savefig(pdf)
    plt.close(fig)
    print(f"[fig] wrote {png}")
    print(f"[fig] wrote {pdf}")


def fig1_cluster_diagnostics(diag: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(diag["k"], diag["silhouette"], marker="o", label="Silhouette")
    ax.plot(diag["k"], diag["calinski_harabasz"], marker="o", label="Calinski-Harabasz")
    ax.plot(diag["k"], diag["davies_bouldin"], marker="o", label="Davies-Bouldin")
    ax.set_title("Country Clustering Diagnostics")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Score")
    ax.legend(loc="best", fontsize=9)
    save_fig(fig, "fig1_cluster_diagnostics")


def fig2_pca_scatter(assign: pd.DataFrame, explained: pd.DataFrame) -> None:
    pc1_var = explained.loc[explained["component"] == "PC1", "explained_variance_ratio"].iloc[0]
    pc2_var = explained.loc[explained["component"] == "PC2", "explained_variance_ratio"].iloc[0]

    fig, ax = plt.subplots(figsize=(10, 7))
    for cid, g in assign.groupby("cluster_id"):
        ax.scatter(g["pca1"], g["pca2"], label=f"Cluster {cid}", alpha=0.8)

    for _, r in assign.iterrows():
        ax.text(r["pca1"], r["pca2"], str(r["country"]), fontsize=7)

    ax.set_title("Country Governance Profiles in PCA Space")
    ax.set_xlabel(f"PC1 ({pc1_var:.1%} variance)")
    ax.set_ylabel(f"PC2 ({pc2_var:.1%} variance)")
    ax.legend(loc="best", fontsize=9)
    save_fig(fig, "fig2_country_pca_scatter")


def fig3_indices_scatter(assign: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))
    for cid, g in assign.groupby("cluster_id"):
        ax.scatter(
            g["regulatory_intensity_index"],
            g["strategic_orientation_index"],
            label=f"Cluster {cid}",
            alpha=0.8,
        )

    for _, r in assign.iterrows():
        ax.text(
            r["regulatory_intensity_index"],
            r["strategic_orientation_index"],
            str(r["country"]),
            fontsize=7,
        )

    ax.set_title("Regulatory Intensity vs Strategic Orientation")
    ax.set_xlabel("Regulatory intensity index")
    ax.set_ylabel("Strategic orientation index")
    ax.legend(loc="best", fontsize=9)
    save_fig(fig, "fig3_regulatory_vs_strategic")


def fig4_complexity_density(assign: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))
    for cid, g in assign.groupby("cluster_id"):
        ax.scatter(
            g["institutional_complexity_index"],
            g["governance_density_index"],
            label=f"Cluster {cid}",
            alpha=0.8,
        )

    for _, r in assign.iterrows():
        ax.text(
            r["institutional_complexity_index"],
            r["governance_density_index"],
            str(r["country"]),
            fontsize=7,
        )

    ax.set_title("Institutional Complexity vs Governance Density")
    ax.set_xlabel("Institutional complexity index")
    ax.set_ylabel("Governance density index")
    ax.legend(loc="best", fontsize=9)
    save_fig(fig, "fig4_complexity_vs_density")


def fig5_cluster_profile_heatmap(cluster_profiles: pd.DataFrame) -> None:
    cols = [
        "regulatory_intensity_index",
        "strategic_orientation_index",
        "institutional_complexity_index",
        "governance_density_index",
        "normative_orientation_index",
        "n_strategy_share",
        "n_norm_share",
        "n_rule_share",
        "strong_deontic_share",
        "pct_o_local_present",
    ]
    cols = [c for c in cols if c in cluster_profiles.columns]

    plot_df = cluster_profiles.sort_values("cluster_id").copy()
    data = plot_df[cols].to_numpy()

    fig, ax = plt.subplots(figsize=(10, 4.8))
    im = ax.imshow(data, aspect="auto")
    ax.set_title("Cluster Governance Profiles")
    ax.set_yticks(np.arange(len(plot_df)))
    ax.set_yticklabels([f"Cluster {int(x)}" for x in plot_df["cluster_id"]])
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(cols, rotation=30, ha="right", fontsize=9)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    save_fig(fig, "fig5_cluster_profile_heatmap")

def main() -> None:
    t0 = time.time()
    ensure_dirs()

    assign = pd.read_csv(BASE / "country_cluster_assignments.csv")
    centroids = pd.read_csv(BASE / "country_cluster_centroids_standardized.csv")
    diagnostics = pd.read_csv(BASE / "country_cluster_diagnostics.csv")
    explained = pd.read_csv(BASE / "country_pca_explained_variance.csv")
    dataset = pd.read_csv(BASE / "country_governance_dataset.csv")

    # assign already contains the governance indices from step 9.3
    assign_full = assign.merge(dataset, on="country", how="left")

    # --- Tables
    cluster_profiles = (
        assign_full.groupby("cluster_id")
        .agg(
            n_countries=("country", "nunique"),
            regulatory_intensity_index=("regulatory_intensity_index", "mean"),
            strategic_orientation_index=("strategic_orientation_index", "mean"),
            institutional_complexity_index=("institutional_complexity_index", "mean"),
            governance_density_index=("governance_density_index", "mean"),
            normative_orientation_index=("normative_orientation_index", "mean"),
            n_docs=("n_docs", "mean"),
            n_statements=("n_statements", "mean"),
            n_strategy_share=("n_strategy_share", "mean"),
            n_norm_share=("n_norm_share", "mean"),
            n_rule_share=("n_rule_share", "mean"),
            strong_deontic_share=("strong_deontic_share", "mean"),
            pct_o_local_present=("pct_o_local_present", "mean"),
        )
        .reset_index()
    )

    top_regulatory = assign_full.sort_values("regulatory_intensity_index", ascending=False)[
        ["country", "cluster_id", "regulatory_intensity_index", "n_rule_share", "rule_to_norm_ratio", "strong_deontic_share", "pct_o_local_present"]
    ].head(15)

    top_strategic = assign_full.sort_values("strategic_orientation_index", ascending=False)[
        ["country", "cluster_id", "strategic_orientation_index", "n_strategy_share", "n_norm_share", "n_rule_share"]
    ].head(15)

    top_complexity = assign_full.sort_values("institutional_complexity_index", ascending=False)[
        ["country", "cluster_id", "institutional_complexity_index", "mean_c_count", "pct_a_explicit", "pct_b_found", "pct_c_text_present"]
    ].head(15)

    cluster_members = assign_full.sort_values(["cluster_id", "country"])[
        ["country", "cluster_id", "pca1", "pca2",
         "regulatory_intensity_index", "strategic_orientation_index",
         "institutional_complexity_index", "governance_density_index"]
    ]

    cluster_profiles.to_csv(TABDIR / "cluster_profiles.csv", index=False)
    top_regulatory.to_csv(TABDIR / "top_regulatory_countries.csv", index=False)
    top_strategic.to_csv(TABDIR / "top_strategic_countries.csv", index=False)
    top_complexity.to_csv(TABDIR / "top_complexity_countries.csv", index=False)
    cluster_members.to_csv(TABDIR / "cluster_members.csv", index=False)

    print(f"[table] wrote {TABDIR / 'cluster_profiles.csv'}")
    print(f"[table] wrote {TABDIR / 'top_regulatory_countries.csv'}")
    print(f"[table] wrote {TABDIR / 'top_strategic_countries.csv'}")
    print(f"[table] wrote {TABDIR / 'top_complexity_countries.csv'}")
    print(f"[table] wrote {TABDIR / 'cluster_members.csv'}")

    # --- Figures
    fig1_cluster_diagnostics(diagnostics)
    fig2_pca_scatter(assign_full, explained)
    fig3_indices_scatter(assign_full)
    fig4_complexity_density(assign_full)
    fig5_cluster_profile_heatmap(cluster_profiles)

    print(f"[step9_4] done elapsed_s={time.time() - t0:.2f}")


if __name__ == "__main__":
    main()
