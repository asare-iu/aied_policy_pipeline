import re
import pandas as pd
from pathlib import Path

IN_PATH = "evidence/egypt_pilot/05_igt_iad/regexish_chunks_from_tsv_20251231_161819/egypt_edu_adibco_autopass.tsv"
OUT_DIR = Path("evidence/egypt_pilot/05_igt_iad/regexish_chunks_from_tsv_20251231_161819")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# You can tune these lists, but start simple and explicit.
STRONG_DEONTICS = {"must", "shall", "must not", "shall not", "may not", "prohibited", "forbidden", "required", "mandatory"}
WEAK_DEONTICS   = {"should", "recommended", "encouraged", "expected"}  # keep "may" out (too noisy)

def norm_deontic(d: str) -> str:
    d = (d or "").strip().lower()
    d = re.sub(r"\s+", " ", d)
    return d

def has_text(x: str) -> bool:
    return bool((x or "").strip())

def classify_row(A, D, I, B, C, O):
    """
    Extended-ADIBCO structural typing (aligned to your stated mapping):

    - Rule: strong deontic + or-else present  (ADIBCO)
    - Norm: weak deontic + no or-else        (ADIC, weak D)
    - Strategy: no deontic, but has A + I    (AIC)
    - Principle: no A, no D, no B, no O but value-ish language (fallback heuristic)
    - Other: everything else
    """

    A_ok = has_text(A)
    I_ok = has_text(I)
    B_ok = has_text(B)
    O_ok = has_text(O)
    d = norm_deontic(D)
# RULE = strong D + O present (you explicitly want O = or-else)
    if d in STRONG_DEONTICS and O_ok:
        return "rule"

    # NORM = weak D and no or-else
    if d in WEAK_DEONTICS and not O_ok:
        return "norm"

    # STRATEGY = no deontic, but actor + aim
    if not d and A_ok and I_ok:
        return "strategy"

    # PRINCIPLE: structurally thin + value language
    # (Because IC isn't directly observable from your extractor, we use value-cue fallback.)
    s = " ".join([str(A), str(I), str(B), str(C), str(O)])
    if (not d) and (not B_ok) and (not O_ok):
        if re.search(r"\b(principle|values?|ethic|ethical|equity|inclusion|fairness|transparency|accountability|rights?|privacy|safety|trust|human[- ]?centric)\b", s, re.I):
            return "principle"

    return "other"

def main():
    df = pd.read_csv(IN_PATH, sep="\t", dtype=str).fillna("")

    required = ["A_attribute","D_deontic","I_aim","B_object","C_condition","O_or_else","sentence"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {IN_PATH}: {missing}")

    df["igt_statement_type_guess"] = [
        classify_row(a, d, i, b, c, o)
        for a, d, i, b, c, o in zip(
            df["A_attribute"],
            df["D_deontic"],
            df["I_aim"],
            df["B_object"],
            df["C_condition"],
            df["O_or_else"],
        )
    ]

    out_path = OUT_DIR / "egypt_edu_statement_type_from_adibco.tsv"
    df.to_csv(out_path, sep="\t", index=False)

    summary = df["igt_statement_type_guess"].value_counts().to_dict()
    summ_path = OUT_DIR / "statement_type_summary_from_adibco.tsv"
    pd.DataFrame([summary]).to_csv(summ_path, sep="\t", index=False)

    print("WROTE:", out_path)
    print("WROTE:", summ_path)
    print("SUMMARY:", summary)

if __name__ == "__main__":
    main()
