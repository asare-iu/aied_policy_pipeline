#!/usr/bin/env python3
"""
Generate Figure 4.1 for Chapter 4.

Figure purpose:
    Substantive governance components ranked by principal component number.

Data source:
    data/derived/step5_5_pca_interpretation/pca_pc_labels_final.csv

Default outputs:
    figures/chapter4/figure_4_1_substantive_governance_components.png
    figures/chapter4/figure_4_1_substantive_governance_components.pdf
    figures/chapter4/figure_4_1_substantive_governance_components.svg

Notes:
    This script intentionally keeps the dissertation caption and source note
    outside the figure image. Add the formal caption in the dissertation
    manuscript below the inserted figure.
"""

from __future__ import annotations

import argparse
import re
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_LABELS_CSV = Path("data/derived/step5_5_pca_interpretation/pca_pc_labels_final.csv")
DEFAULT_OUTPUT_DIR = Path("figures/chapter4")
DEFAULT_STEM = "figure_4_1_substantive_governance_components"

EXPECTED_SUBSTANTIVE_PCS = [1, 2, 4, 5, 7, 13, 14, 15, 16, 17, 18, 19, 21, 23, 24, 25]
EDUCATION_RELEVANT_PCS = {19, 23, 24}


def clean_pc(value: object) -> int | None:
    """Return the integer component number from values such as 1, '1', or 'PC1'."""
    if pd.isna(value):
        return None

    match = re.search(r"\d+", str(value))
    if not match:
        return None

    return int(match.group())


def wrap_label(label: object, width: int = 42) -> str:
    """Wrap long component labels for readable plotting."""
    return "\n".join(textwrap.wrap(str(label), width=width))


def load_substantive_components(labels_csv: Path) -> pd.DataFrame:
    """Load and validate the substantive component labels."""
    if not labels_csv.exists():
        raise FileNotFoundError(
            f"Could not find the labels CSV: {labels_csv}\n"
            "Run this script from the repository root, or pass --labels-csv."
        )

    df = pd.read_csv(labels_csv)

    required_columns = {"pc", "label", "category"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"Missing required columns in {labels_csv}: {missing_columns}. "
            f"Columns found: {list(df.columns)}"
        )

    df["pc_num"] = df["pc"].apply(clean_pc)

    substantive = df[df["category"].astype(str).str.lower().eq("substantive")].copy()
    substantive = substantive.dropna(subset=["pc_num"])
    substantive["pc_num"] = substantive["pc_num"].astype(int)
    substantive = substantive.sort_values("pc_num").reset_index(drop=True)

    actual_pcs = substantive["pc_num"].tolist()
    if actual_pcs != EXPECTED_SUBSTANTIVE_PCS:
        print("WARNING: Substantive component list differs from the expected Chapter 4 list.")
        print(f"Expected: {EXPECTED_SUBSTANTIVE_PCS}")
        print(f"Actual:   {actual_pcs}")

    substantive["highlight"] = substantive["pc_num"].isin(EDUCATION_RELEVANT_PCS)
    substantive["label_wrapped"] = substantive["label"].apply(wrap_label)
    substantive["y"] = list(range(len(substantive)))[::-1]

    return substantive


def set_dissertation_style() -> None:
    """Configure conservative dissertation-style matplotlib settings."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
            "font.size": 10.5,
            "axes.labelsize": 11,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def make_figure(components: pd.DataFrame, output_dir: Path, stem: str) -> None:
    """Create and save Figure 4.1 in PNG, PDF, and SVG formats."""
    output_dir.mkdir(parents=True, exist_ok=True)

    set_dissertation_style()

    base_color = "#2C2C2C"
    text_color = "#1F1F1F"
    line_color = "#B8BDC4"
    grid_color = "#E3E5E8"
    highlight_color = "#123A73"
    highlight_fill = "#EEF3FA"
    background_color = "white"

    fig, ax = plt.subplots(figsize=(11.2, 7.2), dpi=300)
    fig.patch.set_facecolor(background_color)
    ax.set_facecolor(background_color)

    # Soft background band for education-relevant components.
    for _, row in components[components["highlight"]].iterrows():
        ax.axhspan(
            row["y"] - 0.43,
            row["y"] + 0.43,
            color=highlight_fill,
            zorder=0,
        )

    # Horizontal lollipop stems.
    for _, row in components.iterrows():
        ax.hlines(
            y=row["y"],
            xmin=1,
            xmax=row["pc_num"],
            color=line_color,
            linewidth=1.15,
            zorder=1,
        )

    # Dots and component labels.
    for _, row in components.iterrows():
        is_highlight = bool(row["highlight"])
        dot_color = highlight_color if is_highlight else base_color
        label_color = highlight_color if is_highlight else text_color
        label_weight = "bold" if is_highlight else "normal"

        ax.scatter(
            row["pc_num"],
            row["y"],
            s=78,
            color=dot_color,
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )

        ax.text(
            row["pc_num"] + 0.45,
            row["y"],
            f"PC{int(row['pc_num'])}: {row['label_wrapped']}",
            va="center",
            ha="left",
            fontsize=10.3,
            color=label_color,
            fontweight=label_weight,
            linespacing=1.08,
            zorder=4,
        )

    # Axis formatting.
    ax.set_xlim(0.8, 27.2)
    ax.set_ylim(-0.85, len(components) - 0.15)
    ax.set_yticks([])
    ax.set_xticks(range(1, 26))
    ax.set_xlabel("Principal Component Number", labelpad=9)

    ax.grid(
        axis="x",
        linestyle=":",
        linewidth=0.6,
        color=grid_color,
        zorder=0,
    )

    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)

    ax.spines["bottom"].set_color("#333333")
    ax.spines["bottom"].set_linewidth(0.9)
    ax.tick_params(axis="x", length=4, width=0.8, color="#333333")

    # No title, caption, or source inside the figure.
    # Those belong below the figure in the dissertation manuscript.
    plt.tight_layout()

    outputs = {
        "PNG": output_dir / f"{stem}.png",
        "PDF": output_dir / f"{stem}.pdf",
        "SVG": output_dir / f"{stem}.svg",
    }

    fig.savefig(outputs["PNG"], dpi=300, bbox_inches="tight", facecolor=background_color)
    fig.savefig(outputs["PDF"], bbox_inches="tight", facecolor=background_color)
    fig.savefig(outputs["SVG"], bbox_inches="tight", facecolor=background_color)
    plt.close(fig)

    print("Saved Figure 4.1 outputs:")
    for label, path in outputs.items():
        print(f"  {label}: {path.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate dissertation-style Figure 4.1 from pca_pc_labels_final.csv."
    )
    parser.add_argument(
        "--labels-csv",
        type=Path,
        default=DEFAULT_LABELS_CSV,
        help=f"Path to pca_pc_labels_final.csv. Default: {DEFAULT_LABELS_CSV}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for PNG/PDF/SVG outputs. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--stem",
        default=DEFAULT_STEM,
        help=f"Output filename stem. Default: {DEFAULT_STEM}",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    components = load_substantive_components(args.labels_csv)
    make_figure(components, args.output_dir, args.stem)


if __name__ == "__main__":
    main()
