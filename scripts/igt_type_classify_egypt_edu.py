import re
import pandas as pd
from pathlib import Path
from datetime import datetime

IN_PATH = "evidence/egypt_pilot/05_igt_iad/03_rule_type_classification/egypt_edu_iad_rule_types_autopass.tsv"
OUT_DIR = Path("evidence/egypt_pilot/05_igt_iad/01_igt_type_classification")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Signals ---
# Strong deontics (rule-ish)
DEONTIC_STRONG = re.compile(r"\b(must|shall|required|require(d)? to|prohibit(ed)?|must not|may not|shall not)\b", re.I)
# Weak deontics (norm-ish)
DEONTIC_WEAK = re.compile(r"\b(should|encouraged|expected|recommended)\b", re.I)
# Strategy framing
STRATEGY = re.compile(r"\b(will|plans? to|intend(s)? to|aim(s)? to|seek(s)? to|commit(s)? to|strategy|roadmap|initiative|program(me)?|framework)\b", re.I)
# Permissive / discretionary
PERMISSIVE = re.compile(r"\b(may|can|could|might)\b", re.I)

# Explicit sanction / enforcement language (Or-else proxy)
SANCTION = re.compile(r"\b(penalt(y|ies)|fine(s)?|sanction(s)?|disciplin(e|ary)|liable|subject to|revok(e|ed)|suspend(ed)?|criminal|civil)\b", re.I)

# Principle/value statements (context-setting)
PRINCIPLE = re.compile(r"\b(principle(s)?|value(s)?|ethic(s|al)?|human dignity|equity|inclusion|fair(ness)?|transparen(cy|t)|accountab(le|ility)|trust|rights?)\b", re.I)

# Collective actor-ish cues (shared vs individual strategy)
COLLECTIVE_ACTOR = re.compile(r"\b(ministry|government|state|council|authority|institut(ion|ions)|universit(y|ies)|schools?|we|our|national|public sector)\b", re.I)
INDIVIDUAL_ACTOR = re.compile(r"\b(teacher(s)?|student(s)?|educator(s)?|parent(s)?|learner(s)?)\b", re.I)

def guess_type(sentence: str, D_deontic: str, I_or_else: str) -> str:
    s = (sentence or "").strip()
    d = (D_deontic or "").strip()
    orelse = (I_or_else or "").strip()

    strong = bool(d) or bool(DEONTIC_STRONG.search(s))
    weak = bool(DEONTIC_WEAK.search(s))
    sanction = bool(orelse) or bool(SANCTION.search(s))

    if strong:
        # rule vs norm: sanction present => rule; else could be rule w/ implicit sanction missing or norm written strongly
        if sanction:
            return "rule"
        return "rule_or_norm_uncertain"

    if weak:
        return "norm"

    # Principle if no action modality but value framing dominates
    # (We keep it simple: if it has principle language AND lacks strong strategy/permissive cues)
    if PRINCIPLE.search(s) and not STRATEGY.search(s) and not PERMISSIVE.search(s):
        return "principle"

    # Strategies (shared vs individual)
    if STRATEGY.search(s) or PERMISSIVE.search(s):
        # permissive => likely individual strategy, but could still be institutional "may" (discretionary policy)
        if PERMISSIVE.search(s) and INDIVIDUAL_ACTOR.search(s) and not COLLECTIVE_ACTOR.search(s):
            return "individual_strategy"
        # default to shared strategy when actor cues are collective
        if COLLECTIVE_ACTOR.search(s) and not INDIVIDUAL_ACTOR.search(s):
            return "shared_strategy"
        # fallback
        return "strategy_uncertain"

    return "other_uncertain"

def main():
    df = pd.read_csv(IN_PATH, sep="\t", dtype=str).fillna("")
    df["igt_type_guess"] = df.apply(lambda r: guess_type(r.get("sentence",""), r.get("D_deontic",""), r.get("I_or_else","")), axis=1)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUT_DIR / f"egypt_edu_igt_type_guess_{stamp}.tsv"
    df.to_csv(out_path, sep="\t", index=False)

    summary = df["igt_type_guess"].value_counts().reset_index()
    summary.columns = ["igt_type_guess", "rows"]
    summary_path = OUT_DIR / f"igt_type_guess_summary_{stamp}.tsv"
    summary.to_csv(summary_path, sep="\t", index=False)

    print("WROTE:", out_path)
    print("WROTE:", summary_path)
    print(summary.to_string(index=False))

if __name__ == "__main__":
    main()
