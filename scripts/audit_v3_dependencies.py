from pathlib import Path
import csv
import re
from collections import Counter, defaultdict

ROOTS = [Path("scripts")]
EXTENSIONS = {".py", ".sh", ".R", ".r"}

PATTERNS = {
    "old_gate_or_flag": [
        r"edu_lexicon_v1",
        r"edu_any_hit",
        r"edu_domain_hit",
        r"edu_actor_A_hit",
        r"strong_bucket_match",
        r"step6_chunks_edu",
        r"chunks_edu\.parquet",
        r"step8_igt_chunks_edu",
        r"education_igt_statements\.parquet",
    ],
    "old_education_dataset_path": [
        r"step10_education_dataset(?!_v3)",
        r"step5_models_edu_embedded_40pc",
        r"step5_models_edu_embedded(?!_v3)",
        r"education_relevant_iad_rule_types",
        r"education_statements",
        r"education_country_dataset",
    ],
    "education_pca": [
        r"edu_embedded",
        r"education.*pca",
        r"pca.*education",
        r"education.*svd",
        r"svd.*education",
        r"pc_interpretations.*edu",
        r"edu.*pc_interpretations",
        r"models_edu",
    ],
    "stakeholder_actor": [
        r"stakeholder.*edu",
        r"edu.*stakeholder",
        r"actor.*edu",
        r"edu.*actor",
        r"attrition.*edu",
        r"edu.*attrition",
    ],
    "rules_iad_regime": [
        r"rule.*edu",
        r"edu.*rule",
        r"iad.*edu",
        r"edu.*iad",
        r"regime.*edu",
        r"edu.*regime",
        r"umbrella.*edu",
        r"edu.*umbrella",
    ],
    "country_analysis": [
        r"country.*edu",
        r"edu.*country",
        r"education.*delta",
        r"delta.*education",
        r"education.*cluster",
        r"cluster.*education",
        r"overall.*education",
        r"education.*overall",
    ],
    "title_gate_comparison": [
        r"title.*edu",
        r"edu.*title",
        r"title_gate",
        r"title-gat",
    ],
    "old_numeric_result": [
        r"\b10557\b",
        r"10,557",
        r"\b7893\b",
        r"7,893",
        r"\b671\b",
        r"\b22\.2\b",
        r"\b77\.8\b",
        r"\b17\.8\b",
        r"\b93\.3\b",
        r"\b6\.7\b",
        r"\b42\.7\b",
        r"\b2\.23\b",
        r"\b44\.1\b",
        r"\b143\.9\b",
        r"\b2261\b",
        r"2,261",
        r"\b759\b",
        r"\b460\b",
        r"\b28\.6\b",
        r"\b9\.6\b",
        r"\b5\.8\b",
        r"74 jurisdictions",
    ],
    "generic_education_dependency": [
        r"education_chunks",
        r"education_igt",
        r"edu_chunks",
        r"edu_statements",
        r"education_relevant",
        r"education_subset",
        r"edu_subset",
    ],
}

compiled = {
    category: [re.compile(pat, re.I) for pat in pats]
    for category, pats in PATTERNS.items()
}

rows = []
files = []

for root in ROOTS:
    if not root.exists():
        continue

    for path in root.rglob("*"):
        if path.is_file() and path.suffix in EXTENSIONS:
            files.append(path)

for path in sorted(files):
    try:
        text = path.read_text(
            encoding="utf-8",
            errors="replace",
        )
    except Exception:
        continue

    for lineno, line in enumerate(text.splitlines(), start=1):
        for category, regexes in compiled.items():
            matches = [
                rx.pattern
                for rx in regexes
                if rx.search(line)
            ]

            if matches:
                rows.append(
                    {
                        "file": str(path),
                        "line": lineno,
                        "category": category,
                        "patterns": " | ".join(matches),
                        "text": line.strip(),
                    }
                )

outdir = Path("evidence/edu_gate_v3/dependency_audit")
outdir.mkdir(parents=True, exist_ok=True)

manifest = outdir / "education_v3_dependency_manifest.csv"

with manifest.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "file",
            "line",
            "category",
            "patterns",
            "text",
        ],
    )
    writer.writeheader()
    writer.writerows(rows)

by_file = defaultdict(set)

for row in rows:
    by_file[row["file"]].add(row["category"])

summary = outdir / "education_v3_dependency_files.csv"

with summary.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["file", "categories", "n_categories"])

    for file, categories in sorted(by_file.items()):
        writer.writerow(
            [
                file,
                "|".join(sorted(categories)),
                len(categories),
            ]
        )

print()
print("EDUCATION V3 DEPENDENCY AUDIT")
print("=============================")
print("Script files scanned:", len(files))
print("Matching lines:", len(rows))
print("Files requiring inspection:", len(by_file))

print()
print("MATCHES BY CATEGORY")
print("-------------------")

counts = Counter(row["category"] for row in rows)

for category, count in sorted(
    counts.items(),
    key=lambda x: (-x[1], x[0]),
):
    print(f"{category:32} {count:5}")

print()
print("FILES BY DEPENDENCY CATEGORY")
print("----------------------------")

for category in PATTERNS:
    relevant = sorted(
        {
            row["file"]
            for row in rows
            if row["category"] == category
        }
    )

    if not relevant:
        continue

    print()
    print(f"[{category}]")

    for file in relevant:
        print(" ", file)

print()
print("OUTPUTS")
print("-------")
print(manifest)
print(summary)
