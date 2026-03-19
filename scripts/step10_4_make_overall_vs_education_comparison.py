#!/usr/bin/env python3
import pandas as pd
from pathlib import Path

OVERALL = Path("data/derived/step9_country_dataset/country_governance_dataset.csv")
EDU = Path("data/derived/step10_education_dataset/education_country_dataset.csv")
OUTDIR = Path("data/derived/step10_education_dataset/qc")
OUTDIR.mkdir(parents=True, exist_ok=True)

overall = pd.read_csv(OVERALL)
edu = pd.read_csv(EDU)

keep = [
    "country",
    "n_docs", "n_statements", "statements_per_doc",
    "n_strategy_share", "n_norm_share", "n_rule_share",
    "rule_to_norm_ratio", "rule_to_strategy_ratio",
    "pct_a_explicit", "mean_c_count", "pct_b_found",
    "pct_o_local_present", "strong_deontic_share"
]
keep_overall = [c for c in keep if c in overall.columns]
keep_edu = [c for c in keep if c in edu.columns]

overall2 = overall[keep_overall].copy()
edu2 = edu[keep_edu].copy()

overall2 = overall2.add_prefix("overall_")
edu2 = edu2.add_prefix("edu_")

overall2 = overall2.rename(columns={"overall_country": "country"})
edu2 = edu2.rename(columns={"edu_country": "country"})

comp = overall2.merge(edu2, on="country", how="outer")

# simple deltas where possible
pairs = [
    ("n_strategy_share", "strategy_share"),
    ("n_norm_share", "norm_share"),
    ("n_rule_share", "rule_share"),
    ("strong_deontic_share", "strong_deontic_share"),
    ("pct_o_local_present", "o_local_present"),
    ("statements_per_doc", "statements_per_doc"),
]
for base, outname in pairs:
    oc = f"overall_{base}"
    ec = f"edu_{base}"
    if oc in comp.columns and ec in comp.columns:
        comp[f"delta_{outname}"] = comp[ec] - comp[oc]

comp.to_csv(OUTDIR / "overall_vs_education_country_comparison.csv", index=False)

summary_rows = []
for c in comp.columns:
    if c.startswith("delta_"):
        s = pd.to_numeric(comp[c], errors="coerce")
        summary_rows.append({
            "metric": c,
            "mean_delta": s.mean(),
            "median_delta": s.median(),
            "min_delta": s.min(),
            "max_delta": s.max(),
        })

pd.DataFrame(summary_rows).to_csv(OUTDIR / "overall_vs_education_delta_summary.csv", index=False)

print("wrote", OUTDIR / "overall_vs_education_country_comparison.csv")
print("wrote", OUTDIR / "overall_vs_education_delta_summary.csv")

