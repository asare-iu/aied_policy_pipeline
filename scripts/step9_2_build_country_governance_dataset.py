#!/usr/bin/env python3
"""
Step 9.2 — Build country-level governance dataset from IGT statements with country

Input
-----
data/derived/step9_country_dataset/igt_with_country.parquet

Output
------
data/derived/step9_country_dataset/country_governance_dataset.csv
data/derived/step9_country_dataset/country_governance_dataset.parquet

Purpose
-------
Aggregate statement-level IGT features to the country level without imposing any
typology or expected result.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd


INPUT = Path("data/derived/step9_country_dataset/igt_with_country.parquet")
OUTDIR = Path("data/derived/step9_country_dataset")
OUT_CSV = OUTDIR / "country_governance_dataset.csv"
OUT_PQ = OUTDIR / "country_governance_dataset.parquet"


def pct_true(series: pd.Series) -> float:
    s = series.fillna(False).astype(bool)
    return float(s.mean()) if len(s) else np.nan


def share_of_value(series: pd.Series, value: str) -> float:
    s = series.fillna("").astype(str)
    return float((s == value).mean()) if len(s) else np.nan


def safe_ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    d = den.replace({0: np.nan})
    return num / d


def main() -> None:
    t0 = time.time()
    OUTDIR.mkdir(parents=True, exist_ok=True)

    print("[step9_2] loading:", INPUT)
    df = pd.read_parquet(INPUT)

    required = ["doc_id", "country", "statement_type_candidate"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"[step9_2] missing required columns: {missing}")

    print("[step9_2] rows:", len(df))
    print("[step9_2] countries:", df["country"].nunique())
    print("[step9_2] docs:", df["doc_id"].nunique())

    # --- normalize booleans
    for c in ["o_local_present", "b_found", "a_is_conjoined", "i_has_conj"]:
        if c in df.columns:
            df[c] = df[c].fillna(False).astype(bool)

    # --- document counts by country
    doc_counts = (
        df[["country", "doc_id"]]
        .drop_duplicates()
        .groupby("country")
        .size()
        .rename("n_docs")
    )

    # --- statement counts by country
    stmt_counts = (
        df.groupby("country")
        .size()
        .rename("n_statements")
    )

    # --- statement type composition
    statement_type_table = (
        df.groupby(["country", "statement_type_candidate"])
        .size()
        .unstack(fill_value=0)
    )

    # Ensure common columns exist
    for col in ["strategy_candidate", "norm_candidate", "rule_candidate"]:
        if col not in statement_type_table.columns:
            statement_type_table[col] = 0

    statement_type_table = statement_type_table.rename(columns={
        "strategy_candidate": "n_strategy",
        "norm_candidate": "n_norm",
        "rule_candidate": "n_rule",
    })

    # --- deontic composition
    if "d_class" in df.columns:
        dclass = (
            df.groupby(["country", "d_class"])
            .size()
            .unstack(fill_value=0)
        )
        dclass.columns = [f"dclass_{str(c)}" for c in dclass.columns]
    else:
        dclass = pd.DataFrame(index=sorted(df["country"].dropna().unique()))

    # --- attribute quality
    aclass = (
        df.groupby("country")
        .agg(
            pct_a_explicit=("a_class", lambda s: share_of_value(s, "explicit")),
            pct_a_inferred=("a_class", lambda s: share_of_value(s, "inferred")),
            pct_a_conjoined=("a_is_conjoined", pct_true) if "a_is_conjoined" in df.columns else ("doc_id", lambda s: np.nan),
        )
    )

    # --- conditions / objects / sanctions
    structural = (
        df.groupby("country")
        .agg(
            mean_c_count=("c_count", "mean") if "c_count" in df.columns else ("doc_id", lambda s: np.nan),
            pct_b_found=("b_found", pct_true) if "b_found" in df.columns else ("doc_id", lambda s: np.nan),
            pct_o_local_present=("o_local_present", pct_true) if "o_local_present" in df.columns else ("doc_id", lambda s: np.nan),
            pct_i_conjoined=("i_has_conj", pct_true) if "i_has_conj" in df.columns else ("doc_id", lambda s: np.nan),
        )
    )

    # --- lexical / actor proxy richness
    actor_proxy = (
        df.groupby("country")
        .agg(
            pct_a_raw_present=("a_raw_text", lambda s: float(s.fillna("").astype(str).str.strip().ne("").mean())),
            pct_b_text_present=("b_text", lambda s: float(s.fillna("").astype(str).str.strip().ne("").mean())) if "b_text" in df.columns else ("doc_id", lambda s: np.nan),
            pct_c_text_present=("c_texts", lambda s: float(s.fillna("").astype(str).str.strip().ne("").mean())) if "c_texts" in df.columns else ("doc_id", lambda s: np.nan),
        )
    )

    # --- combine
    out = pd.concat(
        [
            doc_counts,
            stmt_counts,
            statement_type_table,
            dclass,
            aclass,
            structural,
            actor_proxy,
        ],
        axis=1,
    ).reset_index()

    # --- derived rates
    out["statements_per_doc"] = out["n_statements"] / out["n_docs"].replace({0: np.nan})

    for base in ["n_strategy", "n_norm", "n_rule"]:
        out[f"{base}_share"] = out[base] / out["n_statements"].replace({0: np.nan})

    out["rule_to_norm_ratio"] = out["n_rule"] / out["n_norm"].replace({0: np.nan})
    out["rule_to_strategy_ratio"] = out["n_rule"] / out["n_strategy"].replace({0: np.nan})

    # If common d_class columns exist, create a couple interpretable groupings
    dclass_cols = [c for c in out.columns if c.startswith("dclass_")]
    if dclass_cols:
        # crude but transparent summaries
        strong_cols = [c for c in dclass_cols if any(k in c.lower() for k in ["must", "shall", "required", "prohibit", "forbid"])]
        weak_cols = [c for c in dclass_cols if any(k in c.lower() for k in ["should", "may", "recommend", "encourage"])]

        if strong_cols:
            out["n_strong_deontic"] = out[strong_cols].sum(axis=1)
            out["strong_deontic_share"] = out["n_strong_deontic"] / out["n_statements"].replace({0: np.nan})
        else:
            out["n_strong_deontic"] = np.nan
            out["strong_deontic_share"] = np.nan

        if weak_cols:
            out["n_weak_deontic"] = out[weak_cols].sum(axis=1)
            out["weak_deontic_share"] = out["n_weak_deontic"] / out["n_statements"].replace({0: np.nan})
        else:
            out["n_weak_deontic"] = np.nan
            out["weak_deontic_share"] = np.nan

    # clean sort
    out = out.sort_values(["n_docs", "n_statements", "country"], ascending=[False, False, True])

    out.to_csv(OUT_CSV, index=False)
    out.to_parquet(OUT_PQ, index=False)

    print("[step9_2] wrote:", OUT_CSV)
    print("[step9_2] wrote:", OUT_PQ)
    print("[step9_2] countries:", len(out))
    print("[step9_2] done elapsed_s=", round(time.time() - t0, 2))


if __name__ == "__main__":
    main()
