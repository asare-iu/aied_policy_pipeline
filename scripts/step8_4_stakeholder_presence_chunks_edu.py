import json
import re
from pathlib import Path
import pandas as pd

# ------------------ Paths ------------------
CHUNKS_JSONL = "data/derived/step6_chunks_edu/chunks_edu.jsonl"
OUT_DIR = Path("data/derived/step8_analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------ Socio-technical stakeholders (H&D grounded) ------------------
STAKEHOLDERS = {
    "students": [r"\bstudent(s)?\b", r"\blearner(s)?\b", r"\bpupil(s)?\b"],
    "educators": [r"\bteacher(s)?\b", r"\beducator(s)?\b", r"\binstructor(s)?\b"],
    "schools_institutions": [r"\bschool(s)?\b", r"\buniversity\b", r"\bcollege\b", r"\bcampus\b"],
    "policy_makers": [r"\bgovernment\b", r"\bministry\b", r"\bpublic authority\b", r"\bminister(s)?\b"],
    "regulators": [r"\bregulator(s)?\b", r"\bsupervisory authority\b", r"\benforcement\b"],
    "commercial_designers": [r"\bprovider(s)?\b", r"\bdeveloper(s)?\b", r"\bvendor(s)?\b", r"\bmanufacturer(s)?\b"],
    "deployers_users": [r"\bdeployer(s)?\b", r"\boperator(s)?\b", r"\buser(s)?\b"],
    "researchers": [r"\bresearcher(s)?\b", r"\bacademic(s)?\b"],
    "platforms_systems": [r"\bplatform(s)?\b", r"\bsystem(s)?\b", r"\btool(s)?\b", r"\bLMS\b"],
    "data_subjects": [r"\bpersonal data\b", r"\bstudent data\b", r"\blearner data\b"],
}

PATTERNS = {
    k: [re.compile(p, flags=re.I) for p in pats]
    for k, pats in STAKEHOLDERS.items()
}

def matches(text, regexes):
    return any(rx.search(text) for rx in regexes)

# ------------------ Load chunks ------------------
texts = []
with open(CHUNKS_JSONL, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        t = obj.get("chunk_text") or obj.get("text") or obj.get("content")
        if t:
            texts.append(str(t))

series = pd.Series(texts, dtype="string")
n = len(series)

# ------------------ Scan ------------------
rows = []
for stakeholder, regexes in PATTERNS.items():
    hit = series.apply(lambda s: matches(s, regexes))
    rows.append({
        "stakeholder": stakeholder,
        "chunks_with_mentions": int(hit.sum()),
        "pct_chunks": round(100 * hit.mean(), 2)
    })

out = (
    pd.DataFrame(rows)
      .sort_values("pct_chunks", ascending=False)
)

out_path = OUT_DIR / "stakeholder_presence_chunks_edu.csv"
out.to_csv(out_path, index=False)

print("Saved →", out_path)
print(out)
