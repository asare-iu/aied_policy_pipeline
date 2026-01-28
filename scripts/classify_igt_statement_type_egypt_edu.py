import re
import pandas as pd
from pathlib import Path
from datetime import datetime

IN_PATH = "evidence/egypt_pilot/05_igt_iad/02_adico_extraction/egypt_edu_adico_autopass.tsv"

OUT_BASE = Path("evidence/egypt_pilot/05_igt_iad")
stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = OUT_BASE / f"02b_statement_type_{stamp}"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Heuristic lexicons (tune later; keep simple for pilot)
RULE_DEONTICS = re.compile(r"\b(must|shall|required|prohibited|forbidden|may not|must not|shall not)\b", re.I)
NORM_DEONTICS = re.compile(r"\b(should|encouraged|expected|recommended)\b", re.I)
STRATEGY_CUES = re.compile(r"\b(we will|will\s+establish|plans?\s+to|aims?\s+to|seeks?\s+to|will\s+develop|will\s+launch|will\s+support)\b", re.I)
PRINCIPLE_CUES = re.compile(r"\b(principle|values?|ethical|ethics|dignity|equity|inclusion|fairness|transparency|accountability|human[- ]?centric|rights?)\b", re.I)

def classify(sentence: str, D_deontic: str):
    s = (sentence or "").strip()
    d = (D_deontic or "").strip()

    # If explicit strong deontic -> rule candidate
    if RULE_DEONTICS.search(s) or (d.lower() in ["must","shall","required","prohibited","forbidden","must not","shall not","may not"]):
        return "rule"

    # Soft deontic -> norm
    if NORM_DEONTICS.search(s) or (d.lower() in ["should","encouraged","expected","recommended","may","can"]):
        # "may/can" can be individual strategy; keep as strategy-lite unless clearly normative
        if re.search(r"\b(may|can)\b", s, re.I) and not NORM_DEONTICS.search(s):
            return "strategy"
        return "norm"

    # No deontic: strategy cues
    if STRATEGY_CUES.search(s):
        return "strategy"

    # Principles are often abstract, value-laden, non-action
    if PRINCIPLE_CUES.search(s) and not re.search(r"\b(implement|establish|create|develop|launch|train|teach)\b", s, re.I):
        return "principle"

    return "other"

def main():
    df = pd.read_csv(IN_PATH, sep="\t", dtype=str).fillna("")
    if "sentence" not in df.columns:
        raise ValueError("Expected column 'sentence' in ADICO autopass TSV")
    if "D_deontic" not in df.columns:
        df["D_deontic"] = ""

    df["igt_statement_type_guess"] = [
        classify(s, d) for s, d in zip(df["sentence"], df["D_deontic"])
    ]

    out_path = OUT_DIR / "egypt_edu_statement_type_autopass.tsv"
    df.to_csv(out_path, sep="\t", index=False)

    summary = df["igt_statement_type_guess"].value_counts(dropna=False).to_dict()
    summ_path = OUT_DIR / "statement_type_summary.tsv"
    pd.DataFrame([summary]).to_csv(summ_path, sep="\t", index=False)

    print("WROTE:", out_path)
    print("WROTE:", summ_path)
    print("SUMMARY:", summary)

if __name__ == "__main__":
    main()
