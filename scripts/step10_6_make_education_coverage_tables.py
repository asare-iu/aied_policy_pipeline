#!/usr/bin/env python3
import pandas as pd
from pathlib import Path

DATA = Path("data/derived/step10_education_dataset/education_country_dataset.csv")
OUTDIR = Path("data/derived/step10_education_dataset/qc")
OUTDIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA)

top_stmt = df.sort_values("n_statements", ascending=False)[["country", "n_docs", "n_statements", "statements_per_doc"]].head(25)
top_rule = df.sort_values("n_rule_share", ascending=False)[["country", "n_rule_share", "rule_to_norm_ratio", "strong_deontic_share"]].head(25)
top_strategy = df.sort_values("n_strategy_share", ascending=False)[["country", "n_strategy_share", "n_norm_share", "n_rule_share"]].head(25)

top_stmt.to_csv(OUTDIR / "education_top_statement_volume.csv", index=False)
top_rule.to_csv(OUTDIR / "education_top_rule_intensity.csv", index=False)
top_strategy.to_csv(OUTDIR / "education_top_strategy_orientation.csv", index=False)

print("wrote", OUTDIR / "education_top_statement_volume.csv")
print("wrote", OUTDIR / "education_top_rule_intensity.csv")
print("wrote", OUTDIR / "education_top_strategy_orientation.csv")
