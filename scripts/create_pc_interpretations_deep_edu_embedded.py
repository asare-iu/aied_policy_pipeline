import pandas as pd
from pathlib import Path

BASE = Path("data/derived/step5_models_edu_embedded_40pc")
IN_FILE = BASE / "pc_interpretations.csv"
OUT_FILE = BASE / "pc_interpretations_deep.csv"

# Controlled vocabularies (documented for consistent use)
EDUCATION_ROLE = {"object", "instrument", "context", "incidental"}
POLICY_INSTRUMENT = {
    "strategy_plan",
    "law_regulation",
    "standards_guidance",
    "funding_budget",
    "program_implementation",
    "institutional_governance",
    "procurement_infrastructure",
}
SIGNAL_CLASS = {
    "substantive",
    "genre_boilerplate",
    "formatting_artefact",
    "translation_noise",
    "mixed",
}
REPORTING_PRIORITY = {"primary", "secondary", "exclude"}

def main() -> None:
    if not IN_FILE.exists():
        raise FileNotFoundError(f"Missing input file: {IN_FILE}")

    df = pd.read_csv(IN_FILE)

    # Add deep interpretation columns as strings (avoids pandas dtype warnings)
    df["governance_frame"] = pd.Series([""] * len(df), dtype="string")
    df["policy_instrument"] = pd.Series([""] * len(df), dtype="string")
    df["education_role"] = pd.Series([""] * len(df), dtype="string")
    df["signal_class"] = pd.Series([""] * len(df), dtype="string")
    df["reporting_priority"] = pd.Series([""] * len(df), dtype="string")
    df["evidence_terms"] = pd.Series([""] * len(df), dtype="string")
    df["analytic_note"] = pd.Series([""] * len(df), dtype="string")

    # Stable, readable order
    desired_cols = [
        "pc",
        "explained_variance_ratio",
        "cumulative",
        "label_expanded",
        "admissible",
        "interpretable",
        "governance_frame",
        "policy_instrument",
        "education_role",
        "signal_class",
        "reporting_priority",
        "evidence_terms",
        "analytic_note",
        "note",
    ]

    missing = [c for c in desired_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Input file is missing expected columns: {missing}")

    df = df[desired_cols]
    df.to_csv(OUT_FILE, index=False)

    print(f"Wrote deep-interpretation CSV scaffold: {OUT_FILE}")
    print("Controlled vocabularies (for your use when filling rows):")
    print("  education_role:", sorted(EDUCATION_ROLE))
    print("  policy_instrument:", sorted(POLICY_INSTRUMENT))
    print("  signal_class:", sorted(SIGNAL_CLASS))
    print("  reporting_priority:", sorted(REPORTING_PRIORITY))

if __name__ == "__main__":
    main()
