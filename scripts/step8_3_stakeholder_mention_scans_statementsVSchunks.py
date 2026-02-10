
import json, re
from pathlib import Path
import pandas as pd

# ---------- Inputs ----------
IGT_PATH = "data/derived/step8_igt_title_edu/igt_statements_full.parquet"
CHUNKS_JSONL = "data/derived/step7_chunks_title_edu/chunks_title_edu_allchunks.jsonl"
OUT_DIR = Path("data/derived/step8_analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Expected stakeholder dictionary (editable) ----------
# Keep this small + defensible; you're testing "presence", not ontology.
EXPECTED = {
    "students": [r"\bstudent(s)?\b", r"\blearner(s)?\b", r"\bpupil(s)?\b"],
    "teachers": [r"\bteacher(s)?\b", r"\beducator(s)?\b", r"\binstructor(s)?\b", r"\bfaculty\b"],
    "schools": [r"\bschool(s)?\b", r"\bclassroom(s)?\b", r"\bcampus\b", r"\buniversity\b", r"\bcollege\b"],
    "administrators": [r"\bprincipal(s)?\b", r"\bheadteacher(s)?\b", r"\bdean(s)?\b", r"\badministrator(s)?\b"],
    "education_ministry": [r"\bministry of education\b", r"\beducation ministry\b", r"\bdepartment of education\b"],
    "government": [r"\bgovernment\b", r"\bstate\b", r"\bpublic authority\b", r"\bcompetent authority\b", r"\bminister(s)?\b"],
    "regulators": [r"\bregulator(s)?\b", r"\bsupervisory authorit(y|ies)\b", r"\benforcement\b"],
    "providers_vendors": [r"\bprovider(s)?\b", r"\bdeveloper(s)?\b", r"\bvendor(s)?\b", r"\bsupplier(s)?\b", r"\bmanufacturer(s)?\b"],
    "deployers_operators": [r"\bdeployer(s)?\b", r"\boperator(s)?\b", r"\buser(s)?\b"],
    "platforms_edtech": [r"\bedtech\b", r"\bplatform(s)?\b", r"\blearning management system(s)?\b|\bLMS\b", r"\btool(s)?\b"],
    "parents_guardians": [r"\bparent(s)?\b", r"\bguardian(s)?\b"],
    "researchers": [r"\bresearcher(s)?\b", r"\bacademic(s)?\b", r"\bresearch institution(s)?\b"],
    "civil_society": [r"\bNGO(s)?\b", r"\bcivil society\b", r"\badvocacy\b"],
    "data_protection": [r"\bdata protection\b", r"\bDPA\b", r"\bprivacy\b", r"\bpersonal data\b"],
}

def compile_expected(d):
    out={}
    for k, pats in d.items():
        out[k] = [re.compile(p, flags=re.I) for p in pats]
    return out

PATTERNS = compile_expected(EXPECTED)

def match_any(text, compiled_list):
    if text is None: 
        return False
    s = str(text)
    return any(rx.search(s) for rx in compiled_list)

def scan_series(series: pd.Series, label: str) -> pd.DataFrame:
    n = len(series)
    rows=[]
    for stake, rxs in PATTERNS.items():
        hits = series.apply(lambda x: match_any(x, rxs))
        rows.append({
            "corpus": label,
            "stakeholder": stake,
            "hits": int(hits.sum()),
            "pct": round(100*float(hits.mean()), 2),
        })
    return pd.DataFrame(rows).sort_values(["pct","hits"], ascending=False)

# ---------- Load corpora ----------
igt = pd.read_parquet(IGT_PATH)
igt_text = igt["sentence_text"].astype(str)

# chunks jsonl (scan chunk_text; if your field differs, we’ll auto-detect)
chunk_texts=[]
with open(CHUNKS_JSONL, "r", encoding="utf-8") as f:
    for line in f:
        obj=json.loads(line)
        # common field fallbacks
        t = obj.get("chunk_text") or obj.get("text") or obj.get("chunk") or obj.get("content")
        if t is not None:
            chunk_texts.append(str(t))
chunks_text = pd.Series(chunk_texts, dtype="string")

# ---------- Scan ----------
df_igt = scan_series(igt_text, "edu_title_institutional_statements")
df_chunks = scan_series(chunks_text, "edu_title_all_chunks")

out = pd.concat([df_igt, df_chunks], ignore_index=True)

# ---------- Gap / contrast view ----------
# "Observed more in chunks than statements" suggests discourse mention not institutionalized.
pivot = out.pivot_table(index="stakeholder", columns="corpus", values="pct", aggfunc="first").fillna(0.0)
pivot["gap_chunks_minus_statements_pct"] = (
    pivot.get("edu_title_all_chunks", 0.0) - pivot.get("edu_title_institutional_statements", 0.0)
).round(2)
pivot = pivot.sort_values("gap_chunks_minus_statements_pct", ascending=False).reset_index()

# ---------- Save ----------
out_path1 = OUT_DIR / "stakeholder_mentions_edu_title_statements_vs_chunks.csv"
out.to_csv(out_path1, index=False)

out_path2 = OUT_DIR / "stakeholder_gap_chunks_minus_statements_edu_title.csv"
pivot.to_csv(out_path2, index=False)

print("Saved →", out_path1)
print("Saved →", out_path2)
print("\nTop stakeholders in institutional statements:")
print(df_igt.head(12).to_string(index=False))
print("\nBiggest positive gaps (more in chunks than statements):")
print(pivot.head(12).to_string(index=False))
