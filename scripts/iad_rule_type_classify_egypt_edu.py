import re
import pandas as pd
from pathlib import Path

IN_PATH = "evidence/egypt_pilot/05_igt_iad/02_adico_extraction/egypt_edu_adico_autopass.tsv"
OUT_DIR = Path("evidence/egypt_pilot/05_igt_iad/03_rule_type_classification")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# IAD rule types (Crawford & Ostrom tradition) — heuristic keyword patterns.
# These are deliberately transparent; you'll refine after inspection.
PATTERNS = [
    ("position", re.compile(r"\b(role|position|committee|council|unit|center of excellence|working group|task force|board|secretariat)\b", re.I)),
    ("boundary", re.compile(r"\b(eligible|eligibility|qualification|criteria|admission|enroll|enrol|register|certification|accreditation|licensed|license)\b", re.I)),
    ("authority", re.compile(r"\b(shall|must|required|require|mandate|authorize|may|prohibit|ban|forbid|obliged)\b", re.I)),
    ("aggregation", re.compile(r"\b(quorum|vote|voting|majority|consensus|approve|approval|committee decision|collective decision)\b", re.I)),
    ("information", re.compile(r"\b(report|reporting|publish|disclose|notify|information|transparency|records|data sharing|communicate|submission|monitoring|evaluate|assessment)\b", re.I)),
    ("payoff", re.compile(r"\b(fund|funding|finance|budget|allocate|incentive|grant|subsidy|penalty|fine|sanction|reward|scholarship)\b", re.I)),
    ("scope", re.compile(r"\b(goal|objective|aim|purpose|outcome|target|vision|strategy|roadmap|framework)\b", re.I)),
]

def classify(sentence: str, deontic: str, obj: str, cond: str):
    text = " ".join([sentence or "", deontic or "", obj or "", cond or ""]).strip()
    hits = []
    for label, rx in PATTERNS:
        if rx.search(text):
            hits.append(label)

    # If multiple, choose a priority order that tends to separate "rules" from "descriptions"
    # (authority/information/payoff are usually the most rule-like)
    priority = ["authority","information","payoff","boundary","position","aggregation","scope"]
    chosen = ""
    for p in priority:
        if p in hits:
            chosen = p
            break

    return chosen, "|".join(hits)

def main():
    df = pd.read_csv(IN_PATH, sep="\t", dtype=str).fillna("")
    df["iad_rule_type_guess"] = ""
    df["iad_rule_type_hits"] = ""

    for i, r in df.iterrows():
        chosen, hits = classify(
            r.get("sentence",""),
            r.get("D_deontic",""),
            r.get("O_object",""),
            r.get("C_condition",""),
        )
        df.at[i, "iad_rule_type_guess"] = chosen
        df.at[i, "iad_rule_type_hits"] = hits

    out_path = OUT_DIR / "egypt_edu_iad_rule_types_autopass.tsv"
    df.to_csv(out_path, sep="\t", index=False)

    summary = {
        "rows": len(df),
        "with_guess": int((df["iad_rule_type_guess"].str.strip() != "").sum()),
        "blank_guess": int((df["iad_rule_type_guess"].str.strip() == "").sum()),
    }
    pd.DataFrame([summary]).to_csv(OUT_DIR / "iad_rule_type_summary.tsv", sep="\t", index=False)

    print("SUMMARY:", summary)
    print("WROTE:", out_path)

if __name__ == "__main__":
    main()
