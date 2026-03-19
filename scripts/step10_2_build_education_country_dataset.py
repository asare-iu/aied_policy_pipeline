#!/usr/bin/env python3
import time
from pathlib import Path
import numpy as np
import pandas as pd

INPUT = Path("data/derived/step10_education_dataset/education_igt_statements.parquet")
OUTDIR = Path("data/derived/step10_education_dataset")
OUT_CSV = OUTDIR / "education_country_dataset.csv"
OUT_PQ = OUTDIR / "education_country_dataset.parquet"

def pct_true(series: pd.Series) -> float:
    s = series.fillna(False).astype(bool)
    return float(s.mean()) if len(s) else np.nan

def share_of_value(series: pd.Series, value: str) -> float:
    s = series.fillna("").astype(str)
    return float((s == value).mean()) if len(s) else np.nan

def main():
    t0 = time.time()
    OUTDIR.mkdir(parents=True, exist_ok=True)

    print("[step10_2] loading:", INPUT)
    df = pd.read_parquet(INPUT)

    required = ["country", "doc_id", "statement_type_candidate"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"[step10_2] missing required columns: {missing}")

    for c in ["o_local_present", "b_found", "a_is_conjoined", "i_has_conj"]:
        if c in df.columns:
            df[c] = df[c].fillna(False).astype(bool)

    doc_counts = df[["country", "doc_id"]].drop_duplicates().groupby("country").size().rename("n_docs")
    stmt_counts = df.groupby("country").size().rename("n_statements")

    st = df.groupby(["country", "statement_type_candidate"]).size().unstack(fill_value=0)
    for col in ["strategy_candidate", "norm_candidate", "rule_candidate", "other_low_confidence"]:
        if col not in st.columns:
            st[col] = 0
    st = st.rename(columns={
        "strategy_candidate": "n_strategy",
        "norm_candidate": "n_norm",
        "rule_candidate": "n_rule",
        "other_low_confidence": "other_low_confidence",
    })

    structural = df.groupby("country").agg(
        pct_a_explicit=("a_class", lambda s: share_of_value(s, "explicit")),
        pct_a_inferred=("a_class", lambda s: share_of_value(s, "inferred")),
        pct_a_conjoined=("a_is_conjoined", pct_true) if "a_is_conjoined" in df.columns else ("doc_id", lambda s: np.nan),
        mean_c_count=("c_count", "mean") if "c_count" in df.columns else ("doc_id", lambda s: np.nan),
        pct_b_found=("b_found", pct_true) if "b_found" in df.columns else ("doc_id", lambda s: np.nan),
        pct_o_local_present=("o_local_present", pct_true) if "o_local_present" in df.columns else ("doc_id", lambda s: np.nan),
        pct_i_conjoined=("i_has_conj", pct_true) if "i_has_conj" in df.columns else ("doc_id", lambda s: np.nan),
        pct_a_raw_present=("a_raw_text", lambda s: float(s.fillna("").astype(str).str.strip().ne("").mean())),
        pct_b_text_present=("b_text", lambda s: float(s.fillna("").astype(str).str.strip().ne("").mean())) if "b_text" in df.columns else ("doc_id", lambda s: np.nan),
        pct_c_text_present=("c_texts", lambda s: float(s.fillna("").astype(str).str.strip().ne("").mean())) if "c_texts" in df.columns else ("doc_id", lambda s: np.nan),
        pct_edu_actor_A=("edu_actor_A_hit", pct_true) if "edu_actor_A_hit" in df.columns else ("doc_id", lambda s: np.nan),
        pct_edu_domain=("edu_domain_hit", pct_true) if "edu_domain_hit" in df.columns else ("doc_id", lambda s: np.nan),
    )

    dclass = df.groupby(["country", "d_class"]).size().unstack(fill_value=0) if "d_class" in df.columns else pd.DataFrame()
    if len(dclass.columns):
        dclass.columns = [f"dclass_{c}" for c in dclass.columns]

    out = pd.concat([doc_counts, stmt_counts, st, structural, dclass], axis=1).reset_index()

    out["statements_per_doc"] = out["n_statements"] / out["n_docs"].replace({0: np.nan})
    for base in ["n_strategy", "n_norm", "n_rule"]:
        out[f"{base}_share"] = out[base] / out["n_statements"].replace({0: np.nan})

    out["rule_to_norm_ratio"] = out["n_rule"] / out["n_norm"].replace({0: np.nan})
    out["rule_to_strategy_ratio"] = out["n_rule"] / out["n_strategy"].replace({0: np.nan})

    dcols = [c for c in out.columns if c.startswith("dclass_")]
    strong_cols = [c for c in dcols if any(k in c.lower() for k in ["obligation", "must", "shall", "required", "prohibition", "forbid"])]
    if strong_cols:
        out["n_strong_deontic"] = out[strong_cols].sum(axis=1)
        out["strong_deontic_share"] = out["n_strong_deontic"] / out["n_statements"].replace({0: np.nan})
    else:
        out["n_strong_deontic"] = np.nan
        out["strong_deontic_share"] = np.nan

    out = out.sort_values(["n_docs", "n_statements", "country"], ascending=[False, False, True])

    out.to_csv(OUT_CSV, index=False)
    out.to_parquet(OUT_PQ, index=False)

    print("[step10_2] wrote:", OUT_CSV)
    print("[step10_2] wrote:", OUT_PQ)
    print("[step10_2] countries:", len(out))
    print("[step10_2] done elapsed_s=", round(time.time() - t0, 2))

if __name__ == "__main__":
    main()
