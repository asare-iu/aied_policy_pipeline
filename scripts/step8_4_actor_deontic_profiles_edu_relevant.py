#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd
import matplotlib.pyplot as plt


def make_a_norm(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.lower()
        .str.replace(r"\bthe\b", "", regex=True)
        .str.replace(r"\ban\b", "", regex=True)
        .str.replace(r"\ba\b", "", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--igt-parquet",
        default="data/derived/step8_igt_chunks_edu/igt_statements_full.parquet",
        help="Edu-relevant institutional statements parquet",
    )
    ap.add_argument(
        "--out-dir",
        default="data/derived/step8_analysis",
        help="Output directory",
    )
    ap.add_argument("--top-exemplars", type=int, default=25, help="Exemplars per actor group (sorted for traceability)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.igt_parquet)

    req = {"doc_id", "chunk_id", "sentence_index_in_chunk", "sentence_text", "a_raw_text", "a_class", "d_lemma", "d_class", "d_polarity"}
    missing = sorted(list(req - set(df.columns)))
    if missing:
        raise ValueError(f"Missing columns: {missing}. Found={list(df.columns)}")

    df["a_norm"] = make_a_norm(df["a_raw_text"])

    # actor group lexicon (transparent)
    actor_groups: Dict[str, List[str]] = {
        "educators_teachers": ["educator", "educators", "teacher", "teachers", "teaching staff", "faculty", "instructor", "instructors"],
        "students_learners": ["student", "students", "learner", "learners", "pupil", "pupils"],
    }

    def group_for_a_norm(a_norm: str) -> str:
        s = str(a_norm)
        for g, keys in actor_groups.items():
            if any(k in s for k in keys):
                return g
        return ""

    # Explicit A only (your methodological standard)
    sub = df[df["a_class"].eq("explicit")].copy()
    sub["actor_group"] = sub["a_norm"].apply(group_for_a_norm)
    sub = sub[sub["actor_group"].ne("")].copy()

    sub["d_present"] = sub["d_lemma"].notna()

    # --- Summary table: D present vs absent by actor_group ---
    d_presence = (
        sub.groupby(["actor_group", "d_present"])
           .size()
           .reset_index(name="count")
    )
    d_presence["percent_within_actor"] = d_presence.groupby("actor_group")["count"].transform(lambda x: 100 * x / x.sum())

    out_csv = out_dir / "edu_relevant_deontic_presence_by_actor.csv"
    d_presence.to_csv(out_csv, index=False)
    print(f"Saved → {out_csv}")

    # --- Plot: stacked bar (100%) D present vs absent ---
    pivot = (
        d_presence.pivot(index="actor_group", columns="d_present", values="percent_within_actor")
                  .fillna(0.0)
    )
    for col in [True, False]:
        if col not in pivot.columns:
            pivot[col] = 0.0
    pivot = pivot[[True, False]]

    fig = plt.figure(figsize=(8, 5))
    pivot.plot(kind="bar", stacked=True, ax=plt.gca())
    plt.ylabel("Percent of statements")
    plt.title("Deontic presence by actor group (Edu-relevant; explicit A only)")
    plt.legend(["D present", "No D"], bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    out_png = out_dir / "edu_relevant_deontic_presence_by_actor.png"
    plt.savefig(out_png, dpi=300)
    plt.close(fig)
    print(f"Saved → {out_png}")

    # --- When D present: breakdown by d_class ---
    sub_d = sub[sub["d_present"]].copy()
    dclass = (
        sub_d.groupby(["actor_group", "d_class"])
             .size()
             .reset_index(name="count")
    )
    dclass["percent_within_actor_Dpresent"] = dclass.groupby("actor_group")["count"].transform(lambda x: 100 * x / x.sum())
    out_dclass = out_dir / "edu_relevant_deontic_class_by_actor.csv"
    dclass.to_csv(out_dclass, index=False)
    print(f"Saved → {out_dclass}")

    # --- Exemplars: where educators/students are explicitly targeted ---
    keep = [
        "doc_id", "chunk_id", "sentence_index_in_chunk",
        "actor_group", "a_raw_text", "a_norm",
        "d_lemma", "d_class", "d_polarity",
        "i_phrase_text", "b_text", "c_texts",
        "statement_type_candidate",
        "sentence_text",
    ]
    keep = [c for c in keep if c in sub.columns]

    ex = sub.sort_values(["actor_group", "d_present", "doc_id", "chunk_id", "sentence_index_in_chunk"]).copy()

    rows = []
    for g, gdf in ex.groupby("actor_group"):
        rows.append(gdf.head(args.top_exemplars)[keep])

    exemplars = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=keep)

    out_ex = out_dir / "edu_relevant_actor_exemplars_explicitA.csv"
    exemplars.to_csv(out_ex, index=False, escapechar="\\")
    print(f"Saved → {out_ex}")

    print("\nDone.")


if __name__ == "__main__":
    main()
