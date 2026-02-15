import pandas as pd
from pathlib import Path

# Paths
BASE = Path("data/derived/step5_models_edu_embedded_40pc")
IN_FILE = BASE / "pc_interpretations.csv"
OUT_FILE = BASE / "pc_interpretations_deep.csv"

# Load existing interpretation layer
df = pd.read_csv(IN_FILE)

# Add new-interpretation columns (strings only)
df["deep_frame"] = ""
df["policy_genre"] = ""
df["education_role"] = ""
df["signal_type"] = ""
df["dissertation_relevance"] = ""
df["analytic_note"] = ""

# Explicit column order
df = df[
    [
        "pc",
        "explained_variance_ratio",
        "cumulative",
        "label_expanded",
        "admissible",
        "interpretable",
        "deep_frame",
        "policy_genre",
        "education_role",
        "signal_type",
        "dissertation_relevance",
        "analytic_note",
        "note",
    ]
]

# Write new interpretation file
df.to_csv(OUT_FILE, index=False)

print(f"Deep interpretation scaffold written to: {OUT_FILE}")
print(df.head())
