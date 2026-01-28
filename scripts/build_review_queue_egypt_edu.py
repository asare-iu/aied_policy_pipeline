import pandas as pd
from pathlib import Path
from datetime import datetime

IN_PATH = "evidence/egypt_pilot/05_igt_iad/03_rule_type_classification/egypt_edu_iad_rule_types_autopass.tsv"

def main():
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"evidence/egypt_pilot/05_igt_iad/04_review_queue_{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("IN_PATH:", IN_PATH)
    print("OUT_DIR:", out_dir)

    df = pd.read_csv(IN_PATH, sep="\t", dtype=str).fillna("")
    print("Loaded rows:", len(df))
    print("Columns:", list(df.columns))

    # Column names in YOUR file:
    # - A_attribute (not A_actor)
    # - D_deontic
    # - iad_rule_type_guess
    # - adico_flags (we'll treat any non-empty as "flagged")
    a_attr = df["A_attribute"].astype(str).str.strip() if "A_attribute" in df.columns else pd.Series([""] * len(df))
    d_deon = df["D_deontic"].astype(str).str.strip() if "D_deontic" in df.columns else pd.Series([""] * len(df))
    rt_guess = df["iad_rule_type_guess"].astype(str).str.strip() if "iad_rule_type_guess" in df.columns else pd.Series([""] * len(df))
    flags = df["adico_flags"].astype(str).str.strip() if "adico_flags" in df.columns else pd.Series([""] * len(df))

    # Review criteria:
    # - any ADICO flags present
    # - missing Deontic
    # - missing Attribute
    # - missing rule type guess
    needs_review = (flags != "") | (d_deon == "") | (a_attr == "") | (rt_guess == "")

    review = df[needs_review].copy()
    clean  = df[~needs_review].copy()

    # Add human/LLM fields for resolution
    for col in ["iad_rule_type_final", "adico_fix_note", "rule_type_fix_note", "reviewer"]:
        if col not in review.columns:
            review[col] = ""

    review_path = out_dir / "egypt_edu_adico_iad_review_queue.tsv"
    clean_path  = out_dir / "egypt_edu_adico_iad_clean_autopass.tsv"
    summary_path = out_dir / "review_queue_summary.tsv"

    review.to_csv(review_path, sep="\t", index=False)
    clean.to_csv(clean_path, sep="\t", index=False)

    summary = {
        "total_rows": len(df),
        "clean_autopass_rows": len(clean),
        "needs_review_rows": len(review),
        "pct_needs_review": round((len(review) / len(df)) * 100, 2) if len(df) else 0,
        "missing_deontic_rows": int((d_deon == "").sum()),
        "missing_attribute_rows": int((a_attr == "").sum()),
        "blank_rule_type_guess_rows": int((rt_guess == "").sum()),
        "flagged_rows": int((flags != "").sum()),
    }
    pd.DataFrame([summary]).to_csv(summary_path, sep="\t", index=False)

    print("WROTE:", review_path, "rows=", len(review))
    print("WROTE:", clean_path, "rows=", len(clean))
    print("WROTE:", summary_path)
    print("SUMMARY:", summary)

if __name__ == "__main__":
    main()
