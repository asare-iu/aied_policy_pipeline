import re
import pandas as pd
from pathlib import Path

# ------------------ Paths ------------------
CHUNKS_FREQ = "data/derived/step8_analysis/stakeholder_presence_chunks_edu.csv"
IGT_PATH = "data/derived/step8_igt_chunks_edu/igt_statements_full.parquet"
OUT_DIR = Path("data/derived/step8_analysis")

# ------------------ Same stakeholder patterns ------------------
STAKEHOLDERS = {
    "students": [r"\bstudent(s)?\b", r"\blearner(s)?\b"],
    "educators": [r"\bteacher(s)?\b", r"\beducator(s)?\b"],
    "schools_institutions": [r"\bschool(s)?\b", r"\buniversity\b", r"\bcollege\b"],
    "policy_makers": [r"\bgovernment\b", r"\bministry\b", r"\bpublic authority\b"],
    "regulators": [r"\bregulator(s)?\b", r"\bsupervisory authority\b"],
    "commercial_designers": [r"\bprovider(s)?\b", r"\bdeveloper(s)?\b", r"\bvendor(s)?\b"],
    "deployers_users": [r"\boperator(s)?\b", r"\buser(s)?\b"],
    "researchers": [r"\bresearcher(s)?\b", r"\bacademic(s)?\b"],
    "platforms_systems": [r"\bplatform(s)?\b", r"\bsystem(s)?\b"],
    "data_subjects": [r"\bpersonal data\b", r"\bstudent data\b"],
}

PATTERNS = {
    k: [re.compile(p, flags=re.I) for p in pats]
    for k, pats in STAKEHOLDERS.items()
}

def matches(text, regexes):
    return any(rx.search(text) for rx in regexes)

# ------------------ Load data ------------------
chunks_df = pd.read_csv(CHUNKS_FREQ)
igt = pd.read_parquet(IGT_PATH)

sentences = igt["sentence_text"].astype(str)

rows = []
for stakeholder, regexes in PATTERNS.items():
    hit = sentences.apply(lambda s: matches(s, regexes))
    rows.append({
        "stakeholder": stakeholder,
        "statements_with_mentions": int(hit.sum()),
        "pct_statements": round(100 * hit.mean(), 2)
    })

stmt_df = pd.DataFrame(rows)

# ------------------ Merge & gap ------------------
merged = (
    chunks_df
      .merge(stmt_df, on="stakeholder", how="left")
      .fillna(0)
)

merged["gap_chunks_minus_statements"] = (
    merged["pct_chunks"] - merged["pct_statements"]
).round(2)

merged = merged.sort_values("gap_chunks_minus_statements", ascending=False)

out_path = OUT_DIR / "stakeholder_chunks_vs_statements_edu.csv"
merged.to_csv(out_path, index=False)

print("Saved →", out_path)
print(merged)
