#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def maybe_read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv" or path.name.endswith(".csv.gz"):
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type: {path}")


def safe_to_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(
        path,
        index=False,
        encoding="utf-8",
        quoting=csv.QUOTE_MINIMAL,
        escapechar="\\",
    )


def build_markdown_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, row in df.iterrows():
        vals = [str(row[c]) for c in cols]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


# --------------------------------------------------
# Config
# --------------------------------------------------

PRIMARY_ORDER = [
    "boundary",
    "position",
    "choice",
    "information",
    "aggregation",
    "payoff",
    "scope",
]

CORPUS_SPECS: List[Tuple[str, str]] = [
    ("Full corpus", "full_corpus"),
    ("Education-relevant", "education_relevant"),
    ("Education-in-title", "education_in_title"),
]


# --------------------------------------------------
# Build summaries from typed rule files
# --------------------------------------------------

def summarize_primary(df: pd.DataFrame, corpus_name: str) -> pd.DataFrame:
    total = len(df)
    out = (
        df.groupby("iad_rule_type_primary", dropna=False)
        .size()
        .reset_index(name="n")
        .rename(columns={"iad_rule_type_primary": "rule_type"})
    )
    out["pct"] = (100.0 * out["n"] / total).round(2) if total else 0.0
    out["corpus"] = corpus_name
    return out


def summarize_binary(df: pd.DataFrame, corpus_name: str) -> pd.DataFrame:
    rows = []
    total = len(df)
    for rt in PRIMARY_ORDER:
        col = f"iad_{rt}"
        if col not in df.columns:
            continue
        n = int(df[col].sum())
        rows.append({
            "corpus": corpus_name,
            "rule_type": rt,
            "n_rule_statements_with_hit": n,
            "pct_rule_statements_with_hit": round(100.0 * n / total, 2) if total else 0.0,
        })
    return pd.DataFrame(rows)


def build_primary_wide(primary_long: pd.DataFrame) -> pd.DataFrame:
    count_wide = (
        primary_long.pivot(index="rule_type", columns="corpus", values="n")
        .reset_index()
        .fillna(0)
    )
    pct_wide = (
        primary_long.pivot(index="rule_type", columns="corpus", values="pct")
        .reset_index()
        .fillna(0.0)
    )

    for corpus, _ in CORPUS_SPECS:
        if corpus not in count_wide.columns:
            count_wide[corpus] = 0
        if corpus not in pct_wide.columns:
            pct_wide[corpus] = 0.0

    pct_wide = pct_wide.rename(columns={c: f"{c} %" for c, _ in CORPUS_SPECS})

    wide = count_wide.merge(pct_wide, on="rule_type", how="left")
    wide["sort_order"] = wide["rule_type"].apply(lambda x: PRIMARY_ORDER.index(x) if x in PRIMARY_ORDER else 999)
    wide = wide.sort_values(["sort_order", "rule_type"]).drop(columns=["sort_order"]).reset_index(drop=True)

    ordered_cols = ["rule_type"]
    for corpus, _ in CORPUS_SPECS:
        ordered_cols.extend([corpus, f"{corpus} %"])
    wide = wide[ordered_cols]

    return wide


def build_binary_wide(binary_long: pd.DataFrame) -> pd.DataFrame:
    wide = (
        binary_long.pivot(index="rule_type", columns="corpus", values="pct_rule_statements_with_hit")
        .reset_index()
        .fillna(0.0)
    )

    for corpus, _ in CORPUS_SPECS:
        if corpus not in wide.columns:
            wide[corpus] = 0.0

    wide["sort_order"] = wide["rule_type"].apply(lambda x: PRIMARY_ORDER.index(x) if x in PRIMARY_ORDER else 999)
    wide = wide.sort_values(["sort_order", "rule_type"]).drop(columns=["sort_order"]).reset_index(drop=True)

    return wide

# --------------------------------------------------
# QC adjudication artefacts
# --------------------------------------------------

def build_qc_adjudication_sheet(
    dfs: Dict[str, pd.DataFrame],
    n_per_primary_per_corpus: int,
    seed: int,
) -> pd.DataFrame:
    parts = []

    keep_cols_base = [
        "doc_id",
        "chunk_id",
        "sentence_text",
        "a_raw_text",
        "a_class",
        "d_surface",
        "d_class",
        "i_phrase_text",
        "c_texts",
        "b_text",
        "o_local_present",
        "o_local_text",
        "statement_type_candidate",
        "iad_rule_type_primary",
        "iad_rule_type_hits",
        "iad_rule_type_n_hits",
        "iad_boundary",
        "iad_position",
        "iad_choice",
        "iad_information",
        "iad_scope",
        "iad_aggregation",
        "iad_payoff",
    ]

    for corpus_name, _stem in CORPUS_SPECS:
        df = dfs[corpus_name].copy()
        for rule_type in PRIMARY_ORDER:
            tmp = df[df["iad_rule_type_primary"] == rule_type].copy()
            if tmp.empty:
                continue
            n = min(n_per_primary_per_corpus, len(tmp))
            tmp = tmp.sample(n=n, random_state=seed)

            keep_cols = [c for c in keep_cols_base if c in tmp.columns]
            tmp = tmp[keep_cols].copy()
            tmp.insert(0, "corpus", corpus_name)
            tmp.insert(1, "sample_bucket_primary", rule_type)

            # Blank adjudication columns
            tmp["manual_keep_as_rule"] = ""
            tmp["manual_primary_rule_type"] = ""
            tmp["manual_secondary_rule_types"] = ""
            tmp["manual_agrees_with_model"] = ""
            tmp["manual_confidence"] = ""
            tmp["manual_notes"] = ""
            tmp["manual_reviewer"] = ""

            parts.append(tmp)

    if not parts:
        return pd.DataFrame()

    out = pd.concat(parts, ignore_index=True)
    return out


def build_qc_sample_summary(adjudication_df: pd.DataFrame) -> pd.DataFrame:
    if adjudication_df.empty:
        return pd.DataFrame(columns=["corpus", "sample_bucket_primary", "n"])
    return (
        adjudication_df.groupby(["corpus", "sample_bucket_primary"])
        .size()
        .reset_index(name="n")
        .sort_values(["corpus", "sample_bucket_primary"])
        .reset_index(drop=True)
    )


def build_qc_codebook() -> str:
    return """# QC Adjudication Codebook

## Purpose
This sheet is used to manually review a stratified sample of cleaned rule statements before final dissertation reporting.

## Manual fields

### manual_keep_as_rule
- `yes` = the statement is genuinely rule-like in form
- `no` = the statement is not genuinely a rule
- `unclear` = borderline / fragment / too ambiguous to decide confidently

### manual_primary_rule_type
Choose one:
- `boundary`
- `position`
- `choice`
- `information`
- `aggregation`
- `payoff`
- `scope`
- `unclear`

### manual_secondary_rule_types
Optional additional rule types separated by `|` if the rule clearly performs multiple functions.

### manual_agrees_with_model
- `yes`
- `partial`
- `no`

### manual_confidence
- `high`
- `medium`
- `low`

### manual_notes
Short explanation of the decision.

## Rule-type reminders
- **Boundary**: who may enter, remain in, or leave a position; eligibility, accreditation, admission, licensing.
- **Position**: roles or offices that exist in the action situation.
- **Choice**: what actors must, may, or must not do.
- **Information**: what must be reported, disclosed, documented, assessed, monitored, or communicated.
- **Aggregation**: how collective decisions are made; consultation, approval, voting, committee or board decisions.
- **Payoff**: incentives, funding, grants, sanctions, penalties, fines, fees, liability, rewards.
- **Scope**: required or desired outcomes, targets, objectives, or state constraints.

## Important note
A rule can perform more than one IAD function. The primary label is the dominant function used for summary comparison.
"""

# --------------------------------------------------
# Optional country-level meso tables
# --------------------------------------------------

def maybe_build_country_meso(df: pd.DataFrame, out_dir: Path, stem: str) -> List[Path]:
    created: List[Path] = []
    country_col = None
    for c in ["country", "country_name"]:
        if c in df.columns:
            country_col = c
            break

    if country_col is None:
        return created

    # Primary shares by country
    primary = (
        df.groupby([country_col, "iad_rule_type_primary"])
        .size()
        .reset_index(name="n")
        .rename(columns={"iad_rule_type_primary": "rule_type"})
    )
    totals = df.groupby(country_col).size().reset_index(name="total_rules")
    primary = primary.merge(totals, on=country_col, how="left")
    primary["pct"] = (100.0 * primary["n"] / primary["total_rules"]).round(2)

    primary_wide = (
        primary.pivot(index=country_col, columns="rule_type", values="pct")
        .reset_index()
        .fillna(0.0)
    )

    out1 = out_dir / f"{stem}_country_primary_rule_type_shares.csv"
    safe_to_csv(primary_wide, out1)
    created.append(out1)

    # Binary shares by country
    rows = []
    for country, g in df.groupby(country_col):
        total = len(g)
        row = {country_col: country}
        for rt in PRIMARY_ORDER:
            col = f"iad_{rt}"
            if col in g.columns:
                row[rt] = round(100.0 * g[col].sum() / total, 2) if total else 0.0
        rows.append(row)
    binary_wide = pd.DataFrame(rows)

    out2 = out_dir / f"{stem}_country_binary_rule_type_shares.csv"
    safe_to_csv(binary_wide, out2)
    created.append(out2)

    return created


# --------------------------------------------------
# Figures
# --------------------------------------------------

def make_primary_stacked_bar(primary_wide: pd.DataFrame, out_path: Path) -> None:
    # corpora on x axis, stacked by rule type
    corpora = [c for c, _ in CORPUS_SPECS]
    fig, ax = plt.subplots(figsize=(10, 6))

    bottoms = [0.0] * len(corpora)
    for rule_type in PRIMARY_ORDER:
        row = primary_wide[primary_wide["rule_type"] == rule_type]
        if row.empty:
            vals = [0.0] * len(corpora)
        else:
            vals = [float(row[f"{corpus} %"].iloc[0]) for corpus in corpora]
        ax.bar(corpora, vals, bottom=bottoms, label=rule_type)
        bottoms = [b + v for b, v in zip(bottoms, vals)]

    ax.set_ylabel("Percent of cleaned rules")
    ax.set_title("Primary IAD rule-type distribution across corpora")
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    plt.xticks(rotation=0)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def make_binary_heatmap(binary_wide: pd.DataFrame, out_path: Path) -> None:
    corpora = [c for c, _ in CORPUS_SPECS]
    plot_df = binary_wide.set_index("rule_type").reindex(PRIMARY_ORDER).fillna(0.0)
    vals = plot_df[corpora].values

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(vals, aspect="auto")

    ax.set_xticks(range(len(corpora)))
    ax.set_xticklabels(corpora)
    ax.set_yticks(range(len(plot_df.index)))
    ax.set_yticklabels(plot_df.index)

    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            ax.text(j, i, f"{vals[i, j]:.1f}", ha="center", va="center")

    ax.set_title("Percent of cleaned rules with each IAD rule-type hit")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# --------------------------------------------------
# Main
# --------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--iad-dir",
        default="data/derived/step8_analysis/iad_rule_types_rules_only",
        help="Directory containing the cleaned-v2 IAD typed rule files",
    )
    ap.add_argument(
        "--clean-summary",
        default="data/derived/step8_analysis/rules_clean_v2_summary_by_corpus.csv",
        help="Cleaning summary CSV from make_clean_rules_v2.py",
    )
    ap.add_argument(
        "--out-dir",
        default="data/derived/step8_analysis/dissertation_artifacts",
        help="Output directory for QC and meso artefacts",
    )
    ap.add_argument(
        "--qc-n-per-primary-per-corpus",
        type=int,
        default=8,
        help="Rows to sample per primary rule type per corpus for manual adjudication",
    )
    ap.add_argument("--qc-seed", type=int, default=42)
    args = ap.parse_args()

    iad_dir = Path(args.iad_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dfs: Dict[str, pd.DataFrame] = {}
    primary_parts = []
    binary_parts = []

    for corpus_name, stem in CORPUS_SPECS:
        p = iad_dir / f"{stem}_iad_rule_types_rules_only.parquet"
        df = maybe_read_table(p)
        dfs[corpus_name] = df
        primary_parts.append(summarize_primary(df, corpus_name))
        binary_parts.append(summarize_binary(df, corpus_name))

    primary_long = pd.concat(primary_parts, ignore_index=True)
    binary_long = pd.concat(binary_parts, ignore_index=True)

    primary_wide = build_primary_wide(primary_long)
    binary_wide = build_binary_wide(binary_long)

    # Save meso tables
    primary_long_csv = out_dir / "meso_primary_rule_type_long.csv"
    primary_wide_csv = out_dir / "meso_primary_rule_type_wide.csv"
    primary_wide_md = out_dir / "meso_primary_rule_type_wide.md"

    binary_long_csv = out_dir / "meso_binary_rule_type_long.csv"
    binary_wide_csv = out_dir / "meso_binary_rule_type_wide.csv"
    binary_wide_md = out_dir / "meso_binary_rule_type_wide.md"

    safe_to_csv(primary_long, primary_long_csv)
    safe_to_csv(primary_wide, primary_wide_csv)
    primary_wide_md.write_text(build_markdown_table(primary_wide), encoding="utf-8")

    safe_to_csv(binary_long, binary_long_csv)
    safe_to_csv(binary_wide, binary_wide_csv)
    binary_wide_md.write_text(build_markdown_table(binary_wide), encoding="utf-8")

    # Cleaning summary copy
    clean_summary_df = maybe_read_table(Path(args.clean_summary))
    clean_summary_csv = out_dir / "rule_cleaning_summary_by_corpus.csv"
    clean_summary_md = out_dir / "rule_cleaning_summary_by_corpus.md"
    safe_to_csv(clean_summary_df, clean_summary_csv)
    clean_summary_md.write_text(build_markdown_table(clean_summary_df), encoding="utf-8")

    # QC adjudication sheet
    adjudication_df = build_qc_adjudication_sheet(
        dfs=dfs,
        n_per_primary_per_corpus=args.qc_n_per_primary_per_corpus,
        seed=args.qc_seed,
    )
    adjudication_csv = out_dir / "qc_adjudication_sheet.csv"
    safe_to_csv(adjudication_df, adjudication_csv)

    qc_summary_df = build_qc_sample_summary(adjudication_df)
    qc_summary_csv = out_dir / "qc_adjudication_sample_summary.csv"
    qc_summary_md = out_dir / "qc_adjudication_sample_summary.md"
    safe_to_csv(qc_summary_df, qc_summary_csv)
    qc_summary_md.write_text(build_markdown_table(qc_summary_df), encoding="utf-8")

    qc_codebook_md = out_dir / "qc_codebook.md"
    qc_codebook_md.write_text(build_qc_codebook(), encoding="utf-8")

    # Figures
    stacked_bar_png = out_dir / "meso_primary_rule_type_stacked_bar.png"
    heatmap_png = out_dir / "meso_binary_rule_type_heatmap.png"
    make_primary_stacked_bar(primary_wide, stacked_bar_png)
    make_binary_heatmap(binary_wide, heatmap_png)

    # Optional country-level meso tables
    country_created: List[Path] = []
    country_created += maybe_build_country_meso(dfs["Education-relevant"], out_dir, "education_relevant")
    country_created += maybe_build_country_meso(dfs["Education-in-title"], out_dir, "education_in_title")

    # README / notes
    note_text = """# Dissertation artefacts: QC evidence and meso presentation

## Main caution
In the binary-hit tables, `choice = 100%` is structural because the dataset has already been restricted to cleaned rules. Do not treat that as a substantive finding. Use the **primary-rule-type tables** for the main meso comparison.

## Recommended use in dissertation

### Main text
- `rule_cleaning_summary_by_corpus.md`
- `meso_primary_rule_type_wide.md`
- `meso_primary_rule_type_stacked_bar.png`
- optionally `meso_binary_rule_type_heatmap.png`

### Appendix
- `qc_codebook.md`
- `qc_adjudication_sheet.csv`
- `qc_adjudication_sample_summary.md`
- selected rows from the adjudication sheet as examples of retained / disputed cases

## Interpretation frame
The cleaned-v2 rule layer should be presented as a higher-precision subset derived from the upstream IGT rule-candidate artefact, not as a perfect gold-standard legal coding layer.
"""
    readme_md = out_dir / "README_qc_and_meso.md"
    readme_md.write_text(note_text, encoding="utf-8")

    print("Saved artefacts:")
    for p in [
        clean_summary_csv, clean_summary_md,
        primary_long_csv, primary_wide_csv, primary_wide_md,
        binary_long_csv, binary_wide_csv, binary_wide_md,
        adjudication_csv, qc_summary_csv, qc_summary_md, qc_codebook_md,
        stacked_bar_png, heatmap_png, readme_md,
        *country_created,
    ]:
        print(f"  {p}")

    print("\nQC adjudication rows:", len(adjudication_df))
    print("QC sample summary:")
    print(qc_summary_df.to_string(index=False))

    print("\nPrimary meso table:")
    print(primary_wide.to_string(index=False))

    print("\nBinary meso table:")
    print(binary_wide.to_string(index=False))


if __name__ == "__main__":
    main()
