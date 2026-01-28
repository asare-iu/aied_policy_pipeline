import pandas as pd
from pathlib import Path
from datetime import datetime

IN_PATH = "evidence/egypt_pilot/05_igt_iad/03_rule_type_classification/egypt_edu_iad_rule_types_autopass.tsv"

# Your pipeline's raw column names (DO NOT CHANGE THESE)
RAW = {
    "actor":     "A_attribute",
    "deontic":   "D_deontic",
    "aim":       "Aim_verb",
    "object":    "O_object",
    "condition": "C_condition",
    "or_else":   "I_or_else",
}

def s(df, col):
    return df[col].astype(str).fillna("").str.strip() if col in df.columns else pd.Series([""]*len(df))

def main():
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"evidence/egypt_pilot/05_igt_iad/04_review_queue_{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(IN_PATH, sep="\t", dtype=str).fillna("")
    print("Loaded rows:", len(df))
    print("Columns:", list(df.columns))

    # Create canonical ADIBCO slot columns (COPY values; do NOT rename raw columns)
    for slot, rawcol in RAW.items():
        df[f"adibco_{slot}"] = s(df, rawcol)

    # Flags / guesses
    flags   = s(df, "adico_flags")  # keep your existing flag logic
    rt_guess = s(df, "iad_rule_type_guess")

    # Missingness over canonical slots
    missing = {slot: (df[f"adibco_{slot}"] == "") for slot in RAW.keys()}
    any_missing = None
    for m in missing.values():
        any_missing = m if any_missing is None else (any_missing | m)

    # Review rule: needs review if flagged OR missing any slot OR blank rule-type guess
    needs_review = (flags != "") | (rt_guess == "") | any_missing

    review = df[needs_review].copy()
    clean  = df[~needs_review].copy()

    # Human resolution columns (separate from the raw + canonical)
    for col in [
        "human_actor_final",
        "human_deontic_final",
        "human_aim_final",
        "human_object_final",
        "human_condition_final",
        "human_or_else_final",
        "iad_rule_type_final",
        "reviewer",
        "review_note",
    ]:
        if col not in review.columns:
            review[col] = ""

    # Outputs
    review_path  = out_dir / "egypt_edu_adibco_review_queue.tsv"
    clean_path   = out_dir / "egypt_edu_clean_autopass.tsv"
    summary_path = out_dir / "review_queue_summary.tsv"
    mapping_path = out_dir / "adibco_mapping_manifest.tsv"

    review.to_csv(review_path, sep="\t", index=False)
    clean.to_csv(clean_path, sep="\t", index=False)

    summary = {
        "total_rows": len(df),
        "clean_autopass_rows": len(clean),
        "needs_review_rows": len(review),
        "pct_needs_review": round((len(review)/len(df))*100, 2) if len(df) else 0,
        "flagged_rows": int((flags != "").sum()),
        "blank_rule_type_guess_rows": int((rt_guess == "").sum()),
    }
    for slot in RAW.keys():
        summary[f"missing_{slot}_rows"] = int(missing[slot].sum())

    pd.DataFrame([summary]).to_csv(summary_path, sep="\t", index=False)

    # Evidence manifest: explicitly documents how your raw columns map to ADIBCO slots
    mapping = pd.DataFrame([{
        "adibco_slot": slot,
        "raw_column_in_pipeline": rawcol,
        "canonical_column_written": f"adibco_{slot}",
        "note": "Canonical ADIBCO slot column copies values from raw pipeline column; raw columns preserved unchanged."
    } for slot, rawcol in RAW.items()])
    mapping.to_csv(mapping_path, sep="\t", index=False)

    print("WROTE:", review_path, "rows=", len(review))
    print("WROTE:", clean_path,  "rows=", len(clean))
    print("WROTE:", summary_path)
    print("WROTE:", mapping_path)
    print("SUMMARY:", summary)

if __name__ == "__main__":
    main()
