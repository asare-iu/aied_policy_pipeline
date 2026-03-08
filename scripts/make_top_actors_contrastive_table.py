#!/usr/bin/env python3
from __future__ import annotations

"""
Build a contrastive top-actors table across:
  1) Full corpus
  2) Education-relevant
  3) Education-in-title

Outputs:
- data/derived/step8_analysis/top_actors_contrastive_wide.csv
- data/derived/step8_analysis/top_actors_contrastive_long.csv
- data/derived/step8_analysis/top_actors_contrastive.md

Key features:
- prefers cleaner actor columns if available
- filters out pronouns / discourse placeholders
- normalizes obvious surface variants
- compares top-N actors across the three corpora
"""

import argparse
import re
from pathlib import Path
from typing import List, Tuple

import pandas as pd


# -----------------------------
# Config
# -----------------------------

PREFERRED_ACTOR_COLUMNS = [
    "a_canonical",
    "a_normalized",
    "a_lexicon_label",
    "actor_category",
    "actor_group",
    "a_raw_text",
]

STOP_ACTORS = {
    "it", "we", "they", "them", "their", "theirs",
    "this", "that", "these", "those",
    "who", "which", "what",
    "you", "your", "yours",
    "i", "me", "my", "mine",
    "he", "him", "his", "she", "her", "hers",
    "one", "ones", "someone", "somebody", "anyone", "anybody",
    "everyone", "everybody", "no one", "nobody",
    "each", "all", "some", "many", "most", "others", "other",
    "there", "here", "such",
}

# Optional ultra-generic terms that are often not very helpful as actors.
# Keep "ai" out by default because it frequently behaves like a pseudo-actor.
BAD_SINGLE_TOKENS = {
    "ai",
}

PHRASE_MAP = {
    "higher education institutions": "higher education institution",
    "higher education institution": "higher education institution",
    "educational institutions": "educational institution",
    "education institutions": "educational institution",
    "education institution": "educational institution",
    "schools and universities": "schools / universities",
    "teachers and students": "teachers / students",
    "public authorities": "public authority",
    "competent authorities": "competent authority",
    "member states": "member state",
}

TOKEN_MAP = {
    "teachers": "teacher",
    "educators": "educator",
    "students": "student",
    "learners": "learner",
    "schools": "school",
    "universities": "university",
    "colleges": "college",
    "institutions": "institution",
    "governments": "government",
    "ministries": "ministry",
    "authorities": "authority",
    "agencies": "agency",
    "companies": "company",
    "businesses": "business",
    "citizens": "citizen",
    "researchers": "researcher",
    "developers": "developer",
    "providers": "provider",
    "vendors": "vendor",
    "systems": "system",
    "platforms": "platform",
    "commissions": "commission",
    "departments": "department",
    "organizations": "organization",
    "organisations": "organization",
}


# -----------------------------
# Helpers
# -----------------------------

def maybe_read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv" or path.name.endswith(".csv.gz"):
        return pd.read_csv(path)
    raise ValueError(f"Unsupported format for file: {path}")


def choose_actor_column(df: pd.DataFrame, label: str) -> str:
    for col in PREFERRED_ACTOR_COLUMNS:
        if col in df.columns:
            return col
    raise ValueError(
        f"{label}: could not find a usable actor column. "
        f"Tried: {PREFERRED_ACTOR_COLUMNS}"
    )


def normalize_actor(text: str) -> str:
    s = "" if text is None else str(text).strip().lower()

    # remove leading determiners
    s = re.sub(r"^(the|a|an)\s+", "", s)

    # strip possessive apostrophes and normalize punctuation
    s = re.sub(r"[\u2018\u2019']", "", s)
    s = re.sub(r"[^a-z0-9\s\-/&]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    if not s:
        return ""

    # collapse known multiword variants first
    if s in PHRASE_MAP:
        return PHRASE_MAP[s]

    # singularize / normalize simple single-token variants
    if s in TOKEN_MAP:
        s = TOKEN_MAP[s]

    return s


def is_substantive_actor(s: str) -> bool:
    if not s:
        return False
    if s in STOP_ACTORS:
        return False
    if s in BAD_SINGLE_TOKENS:
        return False
    if len(s) <= 2 and s not in {"uk", "eu", "us", "un"}:
        return False
    if re.fullmatch(r"[0-9]+", s):
        return False
    return True


def build_markdown_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, row in df.iterrows():
        vals = [str(row[c]) for c in cols]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def top_actor_counts(
    df: pd.DataFrame,
    label: str,
    explicit_only: bool = True,
    min_docs: int = 1,
) -> pd.DataFrame:
    if "doc_id" not in df.columns:
        raise ValueError(f"{label}: missing required column: doc_id")

    sub = df.copy()

    if explicit_only:
        if "a_class" not in sub.columns:
            raise ValueError(f"{label}: explicit_only=True but no a_class column found")
        sub = sub[sub["a_class"].astype(str).str.lower().eq("explicit")].copy()

    actor_col = choose_actor_column(sub, label=label)
    sub["actor_source_col"] = actor_col
    sub["actor_norm"] = sub[actor_col].apply(normalize_actor)
    sub = sub[sub["actor_norm"].apply(is_substantive_actor)].copy()

    if sub.empty:
        return pd.DataFrame(
            columns=[
                "actor_norm",
                "statements",
                "unique_docs",
                "pct_of_actor_statements",
                "corpus",
                "actor_source_col",
            ]
        )

    out = (
        sub.groupby("actor_norm")
        .agg(
            statements=("actor_norm", "size"),
            unique_docs=("doc_id", "nunique"),
        )
        .reset_index()
    )

    out = out[out["unique_docs"] >= min_docs].copy()

    total_statements = int(out["statements"].sum()) if len(out) else 0
    out["pct_of_actor_statements"] = (
        (100.0 * out["statements"] / total_statements).round(2)
        if total_statements
        else 0.0
    )

    out["corpus"] = label
    out["actor_source_col"] = actor_col

    out = out.sort_values(
        ["statements", "unique_docs", "actor_norm"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    return out


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--full-igt",
        default="data/derived/step8_igt_full/igt_statements_full.parquet",
    )
    ap.add_argument(
        "--edu-igt",
        default="data/derived/step8_igt_chunks_edu/igt_statements_full.parquet",
    )
    ap.add_argument(
        "--title-igt",
        default="data/derived/step8_igt_title_edu/igt_statements_full.parquet",
    )
    ap.add_argument(
        "--out-dir",
        default="data/derived/step8_analysis",
    )
    ap.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Number of actors to keep in the final comparison",
    )
    ap.add_argument(
        "--selection-mode",
        choices=["combined", "union_of_topn"],
        default="combined",
        help=(
            "combined = take top actors by total statements across all corpora combined; "
            "union_of_topn = union of each corpus's top-N actors"
        ),
    )
    ap.add_argument(
        "--include-implicit",
        action="store_true",
        help="Include implicit A rows instead of explicit-A-only",
    )
    ap.add_argument(
        "--min-docs",
        type=int,
        default=1,
        help="Keep only actors appearing in at least this many documents per corpus",
    )
    args = ap.parse_args()

    explicit_only = not args.include_implicit
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    specs: List[Tuple[str, Path]] = [
        ("Full corpus", Path(args.full_igt)),
        ("Education-relevant", Path(args.edu_igt)),
        ("Education-in-title", Path(args.title_igt)),
    ]

    long_parts: List[pd.DataFrame] = []

    for label, path in specs:
        df = maybe_read_table(path)
        counts = top_actor_counts(
            df=df,
            label=label,
            explicit_only=explicit_only,
            min_docs=args.min_docs,
        )
        long_parts.append(counts)

    long_df = pd.concat(long_parts, ignore_index=True)

    if long_df.empty:
        raise ValueError("No actor rows survived filtering. Try lowering --min-docs or checking the actor columns.")

    if args.selection_mode == "combined":
        keep = (
            long_df.groupby("actor_norm", as_index=False)["statements"]
            .sum()
            .sort_values(["statements", "actor_norm"], ascending=[False, True])
            .head(args.top_n)["actor_norm"]
            .tolist()
        )
    else:
        keep_set = set()
        for corpus_name in long_df["corpus"].dropna().unique():
            tmp = long_df[long_df["corpus"] == corpus_name].sort_values(
                ["statements", "unique_docs", "actor_norm"],
                ascending=[False, False, True],
            )
            keep_set.update(tmp.head(args.top_n)["actor_norm"].tolist())
        keep = sorted(keep_set)

    long_keep = long_df[long_df["actor_norm"].isin(keep)].copy()

    # statements pivot
    wide_counts = (
        long_keep.pivot(index="actor_norm", columns="corpus", values="statements")
        .fillna(0)
        .astype(int)
        .reset_index()
    )

    # pct pivot
    wide_pct = (
        long_keep.pivot(index="actor_norm", columns="corpus", values="pct_of_actor_statements")
        .fillna(0.0)
        .reset_index()
    )

    # docs pivot
    wide_docs = (
        long_keep.pivot(index="actor_norm", columns="corpus", values="unique_docs")
        .fillna(0)
        .astype(int)
        .reset_index()
    )

    # source column pivot (first non-null value per actor/corpus)
    source_df = (
        long_keep.groupby(["actor_norm", "corpus"], as_index=False)["actor_source_col"]
        .first()
    )
    wide_source = (
        source_df.pivot(index="actor_norm", columns="corpus", values="actor_source_col")
        .reset_index()
    )

    # ensure expected columns exist
    corpora = ["Full corpus", "Education-relevant", "Education-in-title"]
    for c in corpora:
        if c not in wide_counts.columns:
            wide_counts[c] = 0
        if c not in wide_pct.columns:
            wide_pct[c] = 0.0
        if c not in wide_docs.columns:
            wide_docs[c] = 0
        if c not in wide_source.columns:
            wide_source[c] = ""

    wide_pct = wide_pct.rename(columns={c: f"{c} %" for c in corpora})
    wide_docs = wide_docs.rename(columns={c: f"{c} docs" for c in corpora})
    wide_source = wide_source.rename(columns={c: f"{c} actor_col" for c in corpora})

    wide = (
        wide_counts
        .merge(wide_pct, on="actor_norm", how="left")
        .merge(wide_docs, on="actor_norm", how="left")
        .merge(wide_source, on="actor_norm", how="left")
    )

    wide["combined_total"] = (
        wide["Full corpus"]
        + wide["Education-relevant"]
        + wide["Education-in-title"]
    )

    wide = wide.sort_values(
        ["combined_total", "Education-relevant", "Education-in-title", "Full corpus", "actor_norm"],
        ascending=[False, False, False, False, True],
    ).reset_index(drop=True)

    for c in wide.columns:
        if c.endswith(" %"):
            wide[c] = wide[c].round(2)

    wide = wide[
        [
            "actor_norm",
            "combined_total",
            "Full corpus",
            "Full corpus %",
            "Full corpus docs",
            "Full corpus actor_col",
            "Education-relevant",
            "Education-relevant %",
            "Education-relevant docs",
            "Education-relevant actor_col",
            "Education-in-title",
            "Education-in-title %",
            "Education-in-title docs",
            "Education-in-title actor_col",
        ]
    ].copy()

    wide_csv = out_dir / "top_actors_contrastive_wide.csv"
    long_csv = out_dir / "top_actors_contrastive_long.csv"
    md_path = out_dir / "top_actors_contrastive.md"

    wide.to_csv(wide_csv, index=False)
    long_keep.sort_values(
        ["corpus", "statements", "unique_docs", "actor_norm"],
        ascending=[True, False, False, True],
    ).to_csv(long_csv, index=False)
    md_path.write_text(build_markdown_table(wide), encoding="utf-8")

    print(f"Saved → {wide_csv}")
    print(f"Saved → {long_csv}")
    print(f"Saved → {md_path}")
    print()
    print(wide.to_string(index=False))


if __name__ == "__main__":
    main()
