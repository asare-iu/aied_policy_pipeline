from pathlib import Path
import re
from collections import defaultdict

SCRIPTS = Path("scripts")

# Artifacts / fields invalidated by the V3 migration.
SEEDS = [
    "step8_igt_chunks_edu",
    "chunks_edu.parquet",

    "step5_models_edu_embedded_40pc",

    "igt_statements_full_with_edu_flags.parquet",

    "step10_education_dataset",
    "education_igt_statements.parquet",

    "edu_any_hit",
    "edu_domain_hit",
    "edu_actor_A_hit",
]

files = sorted(SCRIPTS.rglob("*.py"))

hits = defaultdict(list)

for path in files:
    try:
        text = path.read_text(
            encoding="utf-8",
            errors="replace"
        )
    except Exception:
        continue

    lines = text.splitlines()

    for seed in SEEDS:
        for i, line in enumerate(lines, 1):
            if seed.lower() in line.lower():
                hits[str(path)].append(
                    (seed, i, line.strip())
                )

print()
print("DIRECT CONSUMERS OF INVALIDATED EDUCATION ARTIFACTS")
print("==================================================")
print()

for path in sorted(hits):
    print(path)

    seen = set()

    for seed, line_no, text in hits[path]:
        key = (seed, line_no, text)

        if key in seen:
            continue

        seen.add(key)

        print(
            f"  line {line_no:4}: "
            f"[{seed}] {text}"
        )

    print()

print("================================")
print("FILES REQUIRING DIRECT INSPECTION")
print("================================")

for path in sorted(hits):
    print(path)

print()
print("TOTAL:", len(hits))
