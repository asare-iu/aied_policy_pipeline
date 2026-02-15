#!/usr/bin/env python3
from __future__ import annotations

import pandas as pd
from pathlib import Path

# INPUT
IGT_PATH = "data/derived/step8_igt_chunks_edu/igt_statements_full.parquet"
OUT_DIR  = Path("data/derived/step8_analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Actor definitions (transparent, theory-aligned)
ACTOR_GROUPS = {
    "educators_teachers": [
        "educator", "educators", "teacher", "teachers",
        "teaching staff", "faculty", "instructor", "instructors"
    ],
    "schools_institutions": [
        "school", "schools", "university", "universities",
        "college", "colleges", "institution", "institutions"
    ],
}

def normalize_a(s: str) -> str:
    return (
        str(s).lower()
        .replace("the ", "")
        .replace("an ", "")
        .replace("a ", "")
        .strip()
    )

def actor_group(a_norm: str) -> str:
    for g, terms in ACTOR_GROUPS.items():
        if any(t in a_norm for t in terms):
            return g
    return ""

# ---- Load ----
df = pd.read_parquet(IGT_PATH)

# Explicit A only (your methodological rule)
df = df[df["a_class"] == "explicit"].copy()

df["a_norm"] = df["a_raw_text"].apply(normalize_a)
df["actor_group"] = df["a_norm"].apply(actor_group)
df = df[df["actor_group"] != ""]

# ---- Statement type distribution ----
stmt_types = (
    df.groupby(["actor_group", "statement_type_candidate"])
      .size()
      .reset_index(name="count")
)

stmt_types["percent_within_actor"] = (
    stmt_types.groupby("actor_group")["count"]
    .transform(lambda x: 100 * x / x.sum())
)

out_csv = OUT_DIR / "actor_statement_types_edu_relevant.csv"
stmt_types.to_csv(out_csv, index=False)

print(f"\nSaved → {out_csv}")
print(stmt_types.sort_values(["actor_group","count"], ascending=[True,False]))

# ---- Deontic class distribution (when D is present) ----
df_d = df[df["d_lemma"].notna()].copy()

d_classes = (
    df_d.groupby(["actor_group", "d_class"])
        .size()
        .reset_index(name="count")
)

d_classes["percent_within_actor"] = (
    d_classes.groupby("actor_group")["count"]
    .transform(lambda x: 100 * x / x.sum())
)

out_csv2 = OUT_DIR / "actor_deontic_classes_edu_relevant.csv"
d_classes.to_csv(out_csv2, index=False)

print(f"\nSaved → {out_csv2}")
print(d_classes.sort_values(["actor_group","count"], ascending=[True,False]))
