import glob
import pandas as pd
from pathlib import Path
from datetime import datetime
import re

# Use latest IGT type file
files = sorted(glob.glob("evidence/egypt_pilot/05_igt_iad/01_igt_type_classification/egypt_edu_igt_type_guess_*.tsv"))
if not files:
    raise SystemExit("No IGT type guess file found. Run scripts/igt_type_classify_egypt_edu.py first.")
IN_PATH = files[-1]

OUT_DIR = Path("evidence/egypt_pilot/05_igt_iad/04_review_queue_igt_type_aware")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SANCTION = re.compile(r"\b(penalt(y|ies)|fine(s)?|sanction(s)?|disciplin(e|ary)|liable|subject to|revok(e|ed)|suspend(ed)?|criminal|civil)\b", re.I)

df = pd.read_csv(IN_PATH, sep="\t", dtype=str).fillna("")
s = df.get("sentence","").astype(str)

# helpers
A_blank = df.get("A_attribute","").astype(str).str.strip().eq("")
D_blank = df.get("D_deontic","").astype(str).str.strip().eq("")
Aim_blank = df.get("Aim_verb","").astype(str).str.strip().eq("")
Orelse_blank = df.get("I_or_else","").astype(str).str.strip().eq("")
implicit_sanction = s.apply(lambda x: bool(SANCTION.search(x)))

t = df["igt_type_guess"].astype(str).str.strip()

# Expected-slot logic
is_rule = t.eq("rule")
is_norm = t.eq("norm")
is_rule_unc = t.eq("rule_or_norm_uncertain")
is_shared = t.eq("shared_strategy")
is_indiv = t.eq("individual_strategy")
is_strat_unc = t.eq("strategy_uncertain")
is_principle = t.eq("principle")
is_other = t.eq("other_uncertain")

needs_review = (
    # RULE: require A,D,Aim; and require either explicit or implicit sanction cue
    (is_rule & (A_blank | D_blank | Aim_blank | ((Orelse_blank) & (~implicit_sanction)))) |

    # NORM: require A,D,Aim
    (is_norm & (A_blank | D_blank | Aim_blank)) |

    # RULE_OR_NORM_UNCERTAIN: review (classification ambiguity)
    (is_rule_unc) |

    # STRATEGIES: require A and Aim; no D expected
    ((is_shared | is_indiv) & (A_blank | Aim_blank)) |

    # Strategy_uncertain / other_uncertain: review
    (is_strat_unc | is_other)
)

review = df[needs_review].copy()
accounted = df[~needs_review].copy()

for col in [
    "igt_type_final",
    "human_actor_final","human_deontic_final","human_aim_final","human_object_final","human_condition_final","human_or_else_final",
    "iad_rule_type_final",
    "reviewer","review_note"
]:
    if col not in review.columns:
        review[col] = ""

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
review_path = OUT_DIR / f"egypt_edu_review_queue_{stamp}.tsv"
acc_path = OUT_DIR / f"egypt_edu_accounted_{stamp}.tsv"
summary_path = OUT_DIR / f"review_queue_summary_{stamp}.tsv"

review.to_csv(review_path, sep="\t", index=False)
accounted.to_csv(acc_path, sep="\t", index=False)

summary = pd.DataFrame([{
    "total_rows": len(df),
    "review_rows": len(review),
    "accounted_rows": len(accounted),
    "pct_review": round((len(review)/len(df))*100, 2) if len(df) else 0,
    "rules": int(is_rule.sum()),
    "norms": int(is_norm.sum()),
    "shared_strategies": int(is_shared.sum()),
    "individual_strategies": int(is_indiv.sum()),
    "principles": int(is_principle.sum()),
    "uncertain_type_rows": int((is_rule_unc | is_strat_unc | is_other).sum()),
}])
summary.to_csv(summary_path, sep="\t", index=False)

print("IN_PATH:", IN_PATH)
print("WROTE:", review_path, "rows=", len(review))
print("WROTE:", acc_path, "rows=", len(accounted))
print("WROTE:", summary_path)
print(summary.to_string(index=False))
