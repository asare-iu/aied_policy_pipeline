import re
import pandas as pd
from pathlib import Path

IN_PATH = "evidence/egypt_pilot/05_igt_iad/00_inputs/egypt_sentences_education.tsv"
OUT_DIR = Path("evidence/egypt_pilot/05_igt_iad/01_rule_candidates")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Strong deontics + policy verbs
DEONTIC = re.compile(r"\b(shall|must|required|require|may|should|prohibit|prohibited|ban|forbid|obliged|mandatory)\b", re.I)

# Governance/action verbs that often indicate institutional statements even if "shall" is missing
GOV_ACTION = re.compile(r"\b(establish|create|adopt|issue|develop|implement|ensure|monitor|evaluate|report|publish|train|certify|accredit|mandate|regulate|govern|oversee|coordinate|allocate|fund|finance)\b", re.I)

# Condition markers (useful for ADICO "C")
COND = re.compile(r"\b(if|when|where|unless|until|in order to|provided that|subject to)\b", re.I)

def main():
    df = pd.read_csv(IN_PATH, sep="\t", dtype=str).fillna("")
    s = df["sentence"].astype(str)

    df["has_deontic"] = s.apply(lambda x: bool(DEONTIC.search(x)))
    df["has_gov_action"] = s.apply(lambda x: bool(GOV_ACTION.search(x)))
    df["has_condition_marker"] = s.apply(lambda x: bool(COND.search(x)))

    # Candidate rules = has deontic OR has governance action verb
    cand = df[df["has_deontic"] | df["has_gov_action"]].copy()

    df.to_csv(OUT_DIR / "egypt_edu_sentences_with_flags.tsv", sep="\t", index=False)
    cand.to_csv(OUT_DIR / "egypt_edu_rule_candidates.tsv", sep="\t", index=False)

    summary = {
        "edu_sentences_total": len(df),
        "rule_candidates": len(cand),
        "pct_candidates": round((len(cand)/len(df))*100, 2) if len(df) else 0,
        "has_deontic": int(df["has_deontic"].sum()),
        "has_gov_action": int(df["has_gov_action"].sum()),
        "has_condition_marker": int(df["has_condition_marker"].sum()),
    }
    pd.DataFrame([summary]).to_csv(OUT_DIR / "rule_candidate_summary.tsv", sep="\t", index=False)
    print("SUMMARY:", summary)
    print("WROTE:", OUT_DIR / "egypt_edu_rule_candidates.tsv")

if __name__ == "__main__":
    main()
