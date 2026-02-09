import pandas as pd
import matplotlib.pyplot as plt

# load data
df = pd.read_parquet(
    "data/derived/step8_igt_title_edu/igt_statements_full.parquet"
)

# analysis-only normalization
df["a_norm"] = (
    df["a_raw_text"]
      .astype(str)
      .str.lower()
      .str.replace(r"\bthe\b","",regex=True)
      .str.replace(r"\ban\b","",regex=True)
      .str.strip()
)

df = df[df["a_norm"].str.len() > 0]

# top attributes
top = df["a_norm"].value_counts().head(10).index
sub = df[df["a_norm"].isin(top)].copy()

sub["has_D"] = sub["d_lemma"].notna()

pivot = (
    sub.groupby(["a_norm","has_D"])
       .size()
       .unstack(fill_value=0)
)

# force both columns to exist
pivot = pivot.reindex(columns=[True, False], fill_value=0)

pct = pivot.div(pivot.sum(axis=1), axis=0) * 100
pct = pct.rename(columns={True: "D present", False: "No D"})

ax = pct[["D present","No D"]].plot(
    kind="bar",
    stacked=True,
    figsize=(9,5)
)

plt.ylabel("Percent of statements")
plt.title("Deontic Force by Attribute (Education-Title)")
plt.tight_layout()

out_path = "data/derived/step8_analysis/a_by_d_top_attributes_edu_title.png"
plt.savefig(out_path, dpi=300)
print("Saved →", out_path)
