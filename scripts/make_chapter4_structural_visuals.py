#!/usr/bin/env python3
"""
Create Chapter 4 structural visuals for AIED governance.

Outputs five main figures:
  4.4 Two-gate structure of education-relevant AI governance
  4.5 Cumulative jurisdictional concentration curve
  4.6 Rank-shift ribbons from overall to education-relevant governance
  4.7 Institutional density cliff
  4.8 Education governance cluster constellation

Run from the repository root:
  cd /home/isakntiasare/aied_policy_pipeline
  python scripts/make_chapter4_structural_visuals.py

"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch, Rectangle


# -----------------------------
# Configuration and helpers
# -----------------------------

PALETTE = {
    "full": "#4C566A",
    "edu": "#5E81AC",
    "direct": "#2E7D6B",
    "cross": "#C77D32",
    "context": "#7B68A6",
    "outside": "#B0B7C3",
    "grid": "#D8DEE9",
    "text": "#20242A",
    "muted": "#667085",
}


def set_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#333333",
        "axes.labelcolor": PALETTE["text"],
        "xtick.color": PALETTE["text"],
        "ytick.color": PALETTE["text"],
        "text.color": PALETTE["text"],
        "font.size": 11,
        "axes.titlesize": 15,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.bbox": "tight",
        "savefig.dpi": 320,
    })


def first_existing(root: Path, candidates: Iterable[str], label: str, required: bool = True) -> Path | None:
    for candidate in candidates:
        p = root / candidate
        if p.exists():
            return p
    if required:
        msg = [f"Missing required input for {label}.", "Looked for:"]
        msg.extend([f"  - {root / c}" for c in candidates])
        raise FileNotFoundError("\n".join(msg))
    return None


def read_jsonl_alignment(path: Path) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            obj = json.loads(line)
            doc_id = obj.get("doc_id") or obj.get("document_id") or obj.get("source_doc_id")
            chunk_id = obj.get("chunk_id") or obj.get("id") or obj.get("chunk_uid")
            if chunk_id is None and doc_id is not None:
                chunk_id = f"{doc_id}_{i:05d}"
            rows.append({"row": i, "doc_id": doc_id, "chunk_id": chunk_id})
    df = pd.DataFrame(rows)
    if "chunk_id" not in df.columns or df["chunk_id"].isna().all():
        raise ValueError(f"Could not recover chunk_id from {path}")
    return df


def read_alignment(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    elif path.suffix.lower() in {".jsonl", ".json"}:
        df = read_jsonl_alignment(path)
    else:
        raise ValueError(f"Unsupported alignment format: {path}")

    if "chunk_id" not in df.columns:
        if "id" in df.columns:
            df = df.rename(columns={"id": "chunk_id"})
        else:
            raise ValueError(f"{path} does not contain a chunk_id column")
    if "doc_id" not in df.columns:
        df["doc_id"] = np.nan
    return df[[c for c in ["row", "doc_id", "chunk_id"] if c in df.columns]].dropna(subset=["chunk_id"])


def save_figure(fig: plt.Figure, outdir: Path, stem: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f"{stem}.png")
    fig.savefig(outdir / f"{stem}.pdf")
    plt.close(fig)


def pct(part: float, whole: float) -> float:
    return (part / whole * 100.0) if whole else float("nan")


def fmt_pct(value: float) -> str:
    return f"{value:.1f}%"


def add_caption_note(fig: plt.Figure, text: str) -> None:
    fig.text(0.01, 0.01, text, ha="left", va="bottom", fontsize=9, color=PALETTE["muted"])

# -----------------------------
# Input loaders
# -----------------------------


def load_gate_alignments(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Path]]:
    full_path = first_existing(root, [
        "data/derived/step5_models_full_40pc/row_alignment.csv",
        "data/derived/step5_models_full/row_alignment.csv",
        "review_bundle/derived/step5_models_full_40pc/row_alignment.csv",
        "data/derived/step4_chunks_tagged/chunks_normalized_tagged.jsonl",
        "data/derived/step3_chunks_spacy/chunks_spacy.jsonl",
    ], "full corpus row alignment")

    edu_path = first_existing(root, [
        "data/derived/step5_models_edu_embedded_40pc/row_alignment.csv",
        "review_bundle/derived/step5_models_edu_embedded_40pc/row_alignment.csv",
        "data/derived/step6_chunks_edu/chunks_edu.jsonl",
    ], "education chunk-gated row alignment")

    title_path = first_existing(root, [
        "data/derived/step5_models_title_edu_40pc_clean/row_alignment.csv",
        "data/derived/step7_chunks_title_edu/chunks_title_edu_allchunks_clean.jsonl",
    ], "cleaned title-gated row alignment")

    paths = {"full": full_path, "education": edu_path, "title_clean": title_path}
    return read_alignment(full_path), read_alignment(edu_path), read_alignment(title_path), paths


def load_country_data(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Path]]:
    full_path = first_existing(root, [
        "data/derived/step9_country_dataset/country_governance_dataset.csv",
    ], "full country governance dataset")
    edu_path = first_existing(root, [
        "data/derived/step10_education_dataset/education_country_dataset.csv",
    ], "education country governance dataset")
    cluster_profile_path = first_existing(root, [
        "data/derived/step10_education_dataset/qc/education_cluster_profiles.csv",
    ], "education cluster profiles")

    full = pd.read_csv(full_path)
    edu = pd.read_csv(edu_path)
    clusters = pd.read_csv(cluster_profile_path)

    for label, df in [("full", full), ("education", edu)]:
        required = {"country", "n_docs", "n_statements", "statements_per_doc"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{label} country dataset is missing required columns: {sorted(missing)}")

    return full, edu, clusters, {"full_country": full_path, "edu_country": edu_path, "cluster_profiles": cluster_profile_path}


# -----------------------------
# Figure 4.4: two-gate flow
# -----------------------------


def bezier_ribbon(ax, x0, x1, y0a, y0b, y1a, y1b, color, alpha=0.78, zorder=1):
    dx = x1 - x0
    c = dx * 0.45
    verts = [
        (x0, y0a),
        (x0 + c, y0a),
        (x1 - c, y1a),
        (x1, y1a),
        (x1, y1b),
        (x1 - c, y1b),
        (x0 + c, y0b),
        (x0, y0b),
        (x0, y0a),
    ]
    codes = [
        MplPath.MOVETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.LINETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CLOSEPOLY,
    ]
    patch = PathPatch(MplPath(verts, codes), facecolor=color, edgecolor="none", alpha=alpha, zorder=zorder)
    ax.add_patch(patch)


def stacked_intervals(order: list[tuple[str, int]], total: int, gap: float = 0.002) -> dict[str, tuple[float, float]]:
    intervals = {}
    y = 0.0
    for label, count in order:
        h = count / total
        intervals[label] = (y + gap, y + h - gap)
        y += h
    return intervals


def make_figure_44_gate_flow(full_rows: pd.DataFrame, edu_rows: pd.DataFrame, title_rows: pd.DataFrame, outdir: Path) -> dict[str, float]:
    full_ids = set(full_rows["chunk_id"].astype(str))
    edu_ids = set(edu_rows["chunk_id"].astype(str))
    title_ids = set(title_rows["chunk_id"].astype(str))

    edu_ids = edu_ids & full_ids
    title_ids = title_ids & full_ids

    direct_ids = edu_ids & title_ids
    cross_ids = edu_ids - title_ids
    context_ids = title_ids - edu_ids
    outside_ids = full_ids - (edu_ids | title_ids)

    counts = {
        "direct": len(direct_ids),
        "cross_domain": len(cross_ids),
        "contextual_title": len(context_ids),
        "outside_both": len(outside_ids),
        "full_total": len(full_ids),
        "edu_total": len(edu_ids),
        "title_total": len(title_ids),
    }
    counts["direct_share_of_edu"] = pct(counts["direct"], counts["edu_total"])
    counts["cross_share_of_edu"] = pct(counts["cross_domain"], counts["edu_total"])

    pd.DataFrame([
        {"population": "Direct education-relevant governance", "count": counts["direct"], "share_full": pct(counts["direct"], counts["full_total"]), "share_education_relevant": counts["direct_share_of_edu"]},
        {"population": "Cross-domain education-relevant governance", "count": counts["cross_domain"], "share_full": pct(counts["cross_domain"], counts["full_total"]), "share_education_relevant": counts["cross_share_of_edu"]},
        {"population": "Contextual material in education-framed documents", "count": counts["contextual_title"], "share_full": pct(counts["contextual_title"], counts["full_total"]), "share_education_relevant": np.nan},
        {"population": "Outside both education gates", "count": counts["outside_both"], "share_full": pct(counts["outside_both"], counts["full_total"]), "share_education_relevant": np.nan},
    ]).to_csv(outdir / "figure_4_4_two_gate_counts.csv", index=False)

    total = counts["full_total"]
    left_order = [
        ("outside_both", counts["outside_both"]),
        ("contextual_title", counts["contextual_title"]),
        ("cross_domain", counts["cross_domain"]),
        ("direct", counts["direct"]),
    ]
    right_order = left_order[:]
    li = stacked_intervals(left_order, total)
    ri = stacked_intervals(right_order, total)

    colors = {
        "direct": PALETTE["direct"],
        "cross_domain": PALETTE["cross"],
        "contextual_title": PALETTE["context"],
        "outside_both": PALETTE["outside"],
    }
    labels = {
        "direct": "Direct education-relevant\n703 chunks\n6.7% of education-relevant",
        "cross_domain": "Cross-domain education-relevant\n9,854 chunks\n93.3% of education-relevant",
        "contextual_title": "Contextual material in\neducation-titled documents\n1,092 chunks",
        "outside_both": "Outside both\neducation gates\n35,863 chunks",
    }

    fig, ax = plt.subplots(figsize=(13, 7.5))
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(0, 1)
    ax.axis("off")

    for key, _count in left_order:
        y0, y1 = li[key]
        z = 3 if key in {"direct", "cross_domain"} else 2
        bezier_ribbon(ax, 0.18, 0.82, y0, y1, ri[key][0], ri[key][1], colors[key], alpha=0.84 if key != "outside_both" else 0.45, zorder=z)

    # Gate bars
    bar_w = 0.035
    ax.add_patch(Rectangle((0.14, 0), bar_w, 1, facecolor="#E5E7EB", edgecolor="none", zorder=5))
    ax.add_patch(Rectangle((0.825, 0), bar_w, 1, facecolor="#E5E7EB", edgecolor="none", zorder=5))

    # Content-gate and title-gate marks
    edu_height = counts["edu_total"] / total
    title_height = counts["title_total"] / total
    ax.add_patch(Rectangle((0.14, 1 - edu_height), bar_w, edu_height, facecolor=PALETTE["edu"], edgecolor="none", zorder=6))
    ax.add_patch(Rectangle((0.825, 1 - title_height), bar_w, title_height, facecolor=PALETTE["direct"], edgecolor="none", zorder=6))

    ax.text(0.1575, 1.035, "Chunk-level\neducation gate", ha="center", va="bottom", fontsize=12, weight="bold")
    ax.text(0.8425, 1.035, "Document-title\neducation gate", ha="center", va="bottom", fontsize=12, weight="bold")
    ax.text(0.1575, 1 - edu_height / 2, f"Education content\n{counts['edu_total']:,} chunks\n{fmt_pct(pct(counts['edu_total'], total))}", ha="center", va="center", fontsize=10, color="white", weight="bold", rotation=90)
    ax.text(0.8425, 1 - title_height / 2, f"Education-titled\n{counts['title_total']:,} chunks\n{fmt_pct(pct(counts['title_total'], total))}", ha="center", va="center", fontsize=9, color="white", weight="bold", rotation=90)

    # Labels for the two central populations
    ax.text(0.50, (li["cross_domain"][0] + li["cross_domain"][1]) / 2, labels["cross_domain"], ha="center", va="center", fontsize=13, weight="bold", color="#4A2A05")
    ax.text(0.50, (li["direct"][0] + li["direct"][1]) / 2 + 0.02, labels["direct"], ha="center", va="center", fontsize=11, weight="bold", color="#0F3B31")
    ax.text(0.50, (li["contextual_title"][0] + li["contextual_title"][1]) / 2, labels["contextual_title"], ha="center", va="center", fontsize=9, color="#3C315C")
    ax.text(0.50, 0.08, labels["outside_both"], ha="center", va="center", fontsize=10, color=PALETTE["muted"])

    ax.set_title("Two-gate structure of education-relevant AI governance", pad=20, weight="bold")
    ax.text(0.5, -0.04, "Most education-relevant chunks sit outside explicitly education-titled documents.", ha="center", va="top", fontsize=11, color=PALETTE["muted"])

    save_figure(fig, outdir, "figure_4_4_two_gate_structure")
    return counts
# -----------------------------
# Figure 4.5: cumulative concentration
# -----------------------------


def cumulative_curve(df: pd.DataFrame, value_col: str = "n_statements") -> pd.DataFrame:
    d = df.sort_values(value_col, ascending=False).reset_index(drop=True).copy()
    d["rank"] = np.arange(1, len(d) + 1)
    total = d[value_col].sum()
    d["cumulative_statements"] = d[value_col].cumsum()
    d["cumulative_share"] = d["cumulative_statements"] / total * 100
    return d


def make_figure_45_concentration(full: pd.DataFrame, edu: pd.DataFrame, outdir: Path) -> dict[str, float]:
    full_c = cumulative_curve(full)
    edu_c = cumulative_curve(edu)
    full_c.assign(layer="Full corpus").to_csv(outdir / "figure_4_5_full_cumulative_concentration.csv", index=False)
    edu_c.assign(layer="Education-relevant").to_csv(outdir / "figure_4_5_education_cumulative_concentration.csv", index=False)

    def share_at(d: pd.DataFrame, n: int) -> float:
        return float(d.loc[d["rank"] == min(n, len(d)), "cumulative_share"].iloc[0])

    stats = {
        "full_top3": share_at(full_c, 3),
        "edu_top3": share_at(edu_c, 3),
        "full_top5": share_at(full_c, 5),
        "edu_top5": share_at(edu_c, 5),
        "full_top10": share_at(full_c, 10),
        "edu_top10": share_at(edu_c, 10),
        "full_top20": share_at(full_c, 20),
        "edu_top20": share_at(edu_c, 20),
        "full_n_for_90": int(full_c.loc[full_c["cumulative_share"] >= 90, "rank"].iloc[0]),
        "edu_n_for_90": int(edu_c.loc[edu_c["cumulative_share"] >= 90, "rank"].iloc[0]),
    }
    pd.DataFrame([stats]).to_csv(outdir / "figure_4_5_concentration_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    ax.plot(full_c["rank"], full_c["cumulative_share"], lw=3, color=PALETTE["full"], label="Full corpus")
    ax.plot(edu_c["rank"], edu_c["cumulative_share"], lw=3, color=PALETTE["edu"], label="Education-relevant subset")
    ax.scatter([3], [stats["full_top3"]], color=PALETTE["full"], s=60, zorder=4)
    ax.scatter([3], [stats["edu_top3"]], color=PALETTE["edu"], s=60, zorder=4)
    ax.annotate(f"Top 3: {stats['full_top3']:.1f}%", xy=(3, stats["full_top3"]), xytext=(8, stats["full_top3"] - 8), arrowprops=dict(arrowstyle="-", color=PALETTE["full"]), fontsize=10, color=PALETTE["full"])
    ax.annotate(f"Top 3: {stats['edu_top3']:.1f}%", xy=(3, stats["edu_top3"]), xytext=(8, stats["edu_top3"] + 4), arrowprops=dict(arrowstyle="-", color=PALETTE["edu"]), fontsize=10, color=PALETTE["edu"])
    ax.axhline(90, color=PALETTE["grid"], lw=1.2, ls="--")
    ax.text(max(len(full_c), len(edu_c)) * 0.72, 91.5, "90% of statements", color=PALETTE["muted"], fontsize=9)
    ax.set_title("Cumulative concentration of institutional statements by jurisdiction", weight="bold")
    ax.set_xlabel("Jurisdictions ranked by statement volume")
    ax.set_ylabel("Cumulative share of institutional statements")
    ax.set_xlim(1, max(len(full_c), len(edu_c)))
    ax.set_ylim(0, 101)
    ax.grid(axis="y", color=PALETTE["grid"], lw=0.8)
    ax.legend(frameon=False, loc="lower right")
    add_caption_note(fig, "Source: country_governance_dataset.csv and education_country_dataset.csv")
    save_figure(fig, outdir, "figure_4_5_cumulative_concentration")
    return stats


# -----------------------------
# Figure 4.6: rank shift ribbons
# -----------------------------


def make_figure_46_rank_shift(full: pd.DataFrame, edu: pd.DataFrame, outdir: Path, top_n: int = 20) -> pd.DataFrame:
    full_rank = full.sort_values("n_statements", ascending=False).reset_index(drop=True)
    full_rank["overall_rank"] = np.arange(1, len(full_rank) + 1)
    edu_rank = edu.sort_values("n_statements", ascending=False).reset_index(drop=True)
    edu_rank["education_rank"] = np.arange(1, len(edu_rank) + 1)

    merged = edu_rank[["country", "education_rank", "n_statements"]].rename(columns={"n_statements": "education_statements"})
    merged = merged.merge(full_rank[["country", "overall_rank", "n_statements"]].rename(columns={"n_statements": "overall_statements"}), on="country", how="left")
    plot_df = merged.head(top_n).copy()
    plot_df["rank_change"] = plot_df["overall_rank"] - plot_df["education_rank"]
    plot_df.to_csv(outdir / "figure_4_6_rank_shift_top20.csv", index=False)

    fig, ax = plt.subplots(figsize=(11.5, 10))
    ax.set_xlim(-0.25, 1.25)
    ax.set_ylim(top_n + 1, 0)
    ax.axis("off")

    ax.text(0, 0.2, "Overall AI governance rank", ha="center", va="bottom", fontsize=12, weight="bold")
    ax.text(1, 0.2, "Education-relevant rank", ha="center", va="bottom", fontsize=12, weight="bold")
    ax.plot([0, 0], [1, top_n], color=PALETTE["grid"], lw=2)
    ax.plot([1, 1], [1, top_n], color=PALETTE["grid"], lw=2)

    for _, row in plot_df.iterrows():
        y0 = row["overall_rank"]
        y1 = row["education_rank"]
        color = PALETTE["edu"] if row["rank_change"] > 0 else (PALETTE["cross"] if row["rank_change"] < 0 else PALETTE["full"])
        lw = 3.2 if row["education_rank"] <= 5 else 1.7
        alpha = 0.88 if row["education_rank"] <= 10 else 0.55
        ax.plot([0, 1], [y0, y1], color=color, lw=lw, alpha=alpha)
        ax.scatter([0], [y0], s=42, color=color, alpha=alpha, zorder=3)
        ax.scatter([1], [y1], s=42, color=color, alpha=alpha, zorder=3)
        ax.text(-0.03, y0, f"{int(row['overall_rank'])}. {row['country']}", ha="right", va="center", fontsize=9.5)
        ax.text(1.03, y1, f"{int(row['education_rank'])}. {row['country']} ({int(row['education_statements']):,})", ha="left", va="center", fontsize=9.5)

    ax.set_title("Jurisdictional rank shifts from overall to education-relevant AI governance", weight="bold", pad=22)
    ax.text(0.5, top_n + 0.6, "Rising lines mark jurisdictions that become more prominent in the education-relevant layer.", ha="center", va="top", fontsize=10, color=PALETTE["muted"])
    add_caption_note(fig, "Source: country_governance_dataset.csv and education_country_dataset.csv")
    save_figure(fig, outdir, "figure_4_6_rank_shift_ribbons")
    return plot_df


# -----------------------------
# Figure 4.7: density cliff
# -----------------------------


def make_figure_47_density_cliff(full: pd.DataFrame, edu: pd.DataFrame, outdir: Path) -> dict[str, float]:
    rng = np.random.default_rng(42)
    full_vals = full["statements_per_doc"].dropna().to_numpy(float)
    edu_vals = edu["statements_per_doc"].dropna().to_numpy(float)

    summary = {
        "full_mean": float(np.mean(full_vals)),
        "edu_mean": float(np.mean(edu_vals)),
        "full_median": float(np.median(full_vals)),
        "edu_median": float(np.median(edu_vals)),
        "full_q25": float(np.quantile(full_vals, 0.25)),
        "edu_q25": float(np.quantile(edu_vals, 0.25)),
        "full_q75": float(np.quantile(full_vals, 0.75)),
        "edu_q75": float(np.quantile(edu_vals, 0.75)),
        "full_max": float(np.max(full_vals)),
        "edu_max": float(np.max(edu_vals)),
    }
    pd.DataFrame([summary]).to_csv(outdir / "figure_4_7_density_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 6.8))
    positions = [1, 2]
    bp = ax.boxplot([full_vals, edu_vals], positions=positions, widths=0.42, patch_artist=True, showfliers=False)
    for patch, color in zip(bp["boxes"], [PALETTE["full"], PALETTE["edu"]]):
        patch.set_facecolor(color)
        patch.set_alpha(0.28)
        patch.set_edgecolor(color)
    for med in bp["medians"]:
        med.set_color("#111111")
        med.set_linewidth(2.2)
    for whisker in bp["whiskers"] + bp["caps"]:
        whisker.set_color(PALETTE["muted"])

    x_full = rng.normal(1, 0.045, size=len(full_vals))
    x_edu = rng.normal(2, 0.045, size=len(edu_vals))
    ax.scatter(x_full, full_vals, s=28, color=PALETTE["full"], alpha=0.45, edgecolor="none")
    ax.scatter(x_edu, edu_vals, s=28, color=PALETTE["edu"], alpha=0.55, edgecolor="none")

    ax.plot([1, 2], [summary["full_median"], summary["edu_median"]], color=PALETTE["cross"], lw=3)
    ax.text(1, summary["full_median"] * 1.13, f"Median {summary['full_median']:.1f}", ha="center", va="bottom", fontsize=10, color=PALETTE["full"], weight="bold")
    ax.text(2, summary["edu_median"] * 1.18, f"Median {summary['edu_median']:.1f}", ha="center", va="bottom", fontsize=10, color=PALETTE["edu"], weight="bold")

    ax.set_yscale("log")
    ax.set_xticks(positions)
    ax.set_xticklabels(["Overall corpus", "Education-relevant subset"])
    ax.set_ylabel("Institutional statements per document, log scale")
    ax.set_title("Institutional density cliff", weight="bold")
    ax.grid(axis="y", color=PALETTE["grid"], lw=0.8, which="both")
    ax.text(1.5, max(full_vals) * 0.8, "Education-relevant governance is much thinner\nper document than the corpus as a whole.", ha="center", va="top", fontsize=11, color=PALETTE["muted"])
    add_caption_note(fig, "Source: country_governance_dataset.csv and education_country_dataset.csv")
    save_figure(fig, outdir, "figure_4_7_institutional_density_cliff")
    return summary


# -----------------------------
# Figure 4.8: cluster constellation
# -----------------------------


def make_figure_48_cluster_constellation(clusters: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    required = {"cluster_id", "n_countries", "strategic_orientation_index", "governance_density_index", "n_rule_share", "n_strategy_share", "n_norm_share"}
    missing = required - set(clusters.columns)
    if missing:
        raise ValueError(f"Cluster profile file is missing required columns: {sorted(missing)}")

    profiles = {
        0: "More regulatory,\nmodest density",
        1: "Norm-heavy,\nlow density",
        2: "Strategy-heavy,\nlow density",
        3: "Modal cluster,\nminimal rules",
        4: "US/EU\nhigh-density outlier",
    }
    plot_df = clusters.copy()
    plot_df["profile_label"] = plot_df["cluster_id"].map(profiles).fillna("Cluster")
    plot_df.to_csv(outdir / "figure_4_8_cluster_constellation_data.csv", index=False)

    fig, ax = plt.subplots(figsize=(10.5, 7.5))
    sizes = 180 + plot_df["n_countries"] * 90
    scatter = ax.scatter(
        plot_df["strategic_orientation_index"],
        plot_df["governance_density_index"],
        s=sizes,
        c=plot_df["cluster_id"],
        cmap="viridis",
        alpha=0.72,
        edgecolor="white",
        linewidth=1.5,
    )

    for _, row in plot_df.iterrows():
        label = f"Cluster {int(row['cluster_id'])}\n{int(row['n_countries'])} jurisdictions\n{row['profile_label']}"
        ax.annotate(
            label,
            (row["strategic_orientation_index"], row["governance_density_index"]),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=9.5,
            weight="bold" if int(row["cluster_id"]) in {3, 4} else "normal",
        )

    ax.axhline(0, color=PALETTE["grid"], lw=1.2)
    ax.axvline(0, color=PALETTE["grid"], lw=1.2)
    ax.set_xlabel("Strategic orientation index")
    ax.set_ylabel("Governance density index")
    ax.set_title("Cluster constellation of education-relevant AI governance profiles", weight="bold")
    ax.grid(color=PALETTE["grid"], lw=0.8, alpha=0.9)
    ax.text(0.02, 0.98, "Bubble size = number of jurisdictions", transform=ax.transAxes, ha="left", va="top", fontsize=10, color=PALETTE["muted"])
    add_caption_note(fig, "Source: education_cluster_profiles.csv")
    save_figure(fig, outdir, "figure_4_8_cluster_constellation")
    return plot_df

# -----------------------------
# Manifest
# -----------------------------


def write_manifest(outdir: Path, sources: dict[str, Path], gate_counts: dict[str, float], concentration: dict[str, float], density: dict[str, float]) -> None:
    lines = []
    lines.append("# Chapter 4 structural visuals manifest")
    lines.append("")
    lines.append("## Source files")
    for k, v in sources.items():
        lines.append(f"- {k}: `{v}`")
    lines.append("")
    lines.append("## Key computed values")
    lines.append(f"- Full chunk total: {gate_counts['full_total']:,}")
    lines.append(f"- Education content-gated chunks: {gate_counts['edu_total']:,} ({pct(gate_counts['edu_total'], gate_counts['full_total']):.1f}% of full corpus)")
    lines.append(f"- Clean title-gated chunks: {gate_counts['title_total']:,} ({pct(gate_counts['title_total'], gate_counts['full_total']):.1f}% of full corpus)")
    lines.append(f"- Direct education-relevant chunks, chunk gate and title gate: {gate_counts['direct']:,} ({gate_counts['direct_share_of_edu']:.1f}% of education-relevant chunks)")
    lines.append(f"- Cross-domain education-relevant chunks, chunk gate outside title gate: {gate_counts['cross_domain']:,} ({gate_counts['cross_share_of_edu']:.1f}% of education-relevant chunks)")
    lines.append(f"- Full corpus top-three statement share: {concentration['full_top3']:.1f}%")
    lines.append(f"- Education-relevant top-three statement share: {concentration['edu_top3']:.1f}%")
    lines.append(f"- Full median statements per document: {density['full_median']:.1f}")
    lines.append(f"- Education-relevant median statements per document: {density['edu_median']:.1f}")
    lines.append("")
    lines.append("## Figure captions")
    lines.append("- Figure 4.4. Two-gate structure of education-relevant AI governance. The figure cross-classifies chunks by the chunk-level education content gate and the cleaned document-title education gate.")
    lines.append("- Figure 4.5. Cumulative concentration of institutional statements by jurisdiction. Jurisdictions are ranked by statement volume for the full corpus and the education-relevant subset.")
    lines.append("- Figure 4.6. Jurisdictional rank shifts from overall AI governance to education-relevant AI governance. The education-relevant subset changes the hierarchy while retaining a concentrated head.")
    lines.append("- Figure 4.7. Institutional density cliff. Country-level statements per document are substantially lower in the education-relevant subset than in the full corpus.")
    lines.append("- Figure 4.8. Cluster constellation of education-relevant AI governance profiles. Bubble size represents the number of jurisdictions in each cluster.")
    lines.append("")
    outdir.joinpath("figure_manifest.md").write_text("\n".join(lines), encoding="utf-8")


# -----------------------------
# Main
# -----------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Create Chapter 4 structural visuals for AIED governance.")
    parser.add_argument("--root", default=".", help="Repository root. Default: current directory.")
    parser.add_argument("--outdir", default="data/derived/chapter4_structural_visuals", help="Output directory, relative to root unless absolute.")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    outdir = Path(args.outdir)
    if not outdir.is_absolute():
        outdir = root / outdir
    outdir.mkdir(parents=True, exist_ok=True)

    set_style()

    print(f"Repository root: {root}")
    print(f"Output directory: {outdir}")

    full_rows, edu_rows, title_rows, gate_sources = load_gate_alignments(root)
    full_country, edu_country, cluster_profiles, country_sources = load_country_data(root)
    sources = {**gate_sources, **country_sources}

    gate_counts = make_figure_44_gate_flow(full_rows, edu_rows, title_rows, outdir)
    concentration = make_figure_45_concentration(full_country, edu_country, outdir)
    make_figure_46_rank_shift(full_country, edu_country, outdir, top_n=20)
    density = make_figure_47_density_cliff(full_country, edu_country, outdir)
    make_figure_48_cluster_constellation(cluster_profiles, outdir)

    write_manifest(outdir, sources, gate_counts, concentration, density)

    print("\nDone. Created:")
    for p in sorted(outdir.glob("figure_4_*.*")):
        print(f"  {p}")
    print(f"  {outdir / 'figure_manifest.md'}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
