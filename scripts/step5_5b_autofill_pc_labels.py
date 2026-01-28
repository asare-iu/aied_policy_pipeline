#!/usr/bin/env python3
"""
Step 5.5b: Auto-fill PCA PC label template with proposed labels.

Inputs:
- data/derived/step5_5_pca_interpretation/pca_pc_labels_template.csv

Outputs:
- data/derived/step5_5_pca_interpretation/pca_pc_labels_filled.csv

This script populates labels based on my current working interpretation.

"""

from __future__ import annotations

import csv
from pathlib import Path


TEMPLATE = Path("data/derived/step5_5_pca_interpretation/pca_pc_labels_template.csv")
OUT = Path("data/derived/step5_5_pca_interpretation/pca_pc_labels_final.csv")


LABELS = {
    1: (
        "General AI Strategy & Public-Sector Modernization",
        "High-level discourse framing AI as a public-sector and societal development priority, largely detached from sector-specific regulation.",
        "substantive",
    ),
    2: (
        "Operational AI Systems & Deployment",
        "Language focused on the deployment, use, and functioning of AI systems, contrasted with formal legal or administrative structuring text.",
        "substantive",
    ),
    3: (
        "Spanish-Language Text",
        "Spanish-language policy narrative distinct from English-dominant AI governance discourse.",
        "language_artifact",
    ),
    4: (
        "AI Systems Governance vs Data Protection",
        "Regulatory discourse contrasting AI systems governance with traditional personal data protection and privacy frameworks.",
        "substantive",
    ),
    5: (
        "Personal Data Processing & Compliance",
        "Emphasis on personal data processing, compliance mechanisms, and administrative controls, contrasted with security or defense-oriented statutory language.",
        "substantive",
    ),
    6: (
        "Bulgarian Education & Training Text",
        "Education- and training-related policy discourse expressed primarily in Bulgarian, distinct from English AI governance text.",
        "language_artifact",
    ),
    7: (
        "AI & Data Protection Legal Drafting Register",
        "Formal statutory and regulatory drafting language governing AI and personal data, contrasted with innovation- and skills-oriented policy strategy.",
        "substantive",
    ),
    8: (
        "German-Language Text",
        "German-language policy narrative distinct from English-language data governance and privacy regulation text.",
        "language_artifact",
    ),
    9: (
        "Research & Citation Register",
        "AI policy language anchored in references to reports, research documents, and external sources, contrasted with implementation-oriented discourse.",
        "documentary_artifact",
    ),
    10: (
        "Legislative Amendment Mechanics",
        "Formal legislative amendment language indicating how statutes are modified, independent of substantive policy domains.",
        "drafting_artifact",
    ),
    11: (
        "National Statutory Form vs EU Regulatory Form",
        "A contrast between national legislative amendment mechanics and EU-style regulatory drafting and institutional authority language.",
        "drafting_artifact",
    ),
    12: (
        "UK Web & Publication Provenance",
        "Language indicating UK government publication sources and web-based document references rather than substantive policy content.",
        "documentary_artifact",
    ),
    13: (
        "Digital Government & Service Transformation",
        "Policy discourse focused on digital transformation of government services and rights, contrasted with research and innovation funding language.",
        "substantive",
    ),
    14: (
        "EU AI Strategy & Regulatory Framing",
        "Strategic and regulatory framing of artificial intelligence at the European Union level.",
        "substantive",
    ),
    15: (
        "Operational ML Systems & Digital Transformation",
        "Discourse emphasizing operational machine-learning systems within broader digital transformation initiatives.",
        "substantive",
    ),
    16: (
        "Rights-Based Data Protection vs ML Systems",
        "A dimension contrasting rights-based data protection law with machine-learning system development and deployment.",
        "substantive",
    ),
    17: (
        "Public-Sector AI Decision-Making",
        "AI governance language focused on public-sector decision-making, institutional authority, and administrative use of AI.",
        "substantive",
    ),
    18: (
        "Defense & National Security AI Governance",
        "Governance discourse related to defense, national security, and strategic AI programs.",
        "substantive",
    ),
    19: (
        "Education, Skills & Workforce Development",
        "Policy language addressing education, skills development, and workforce preparation in relation to AI.",
        "substantive",
    ),
    20: (
        "Program-Specific Education Initiative (SmartLabs/PNRR)",
        "References to specific national or EU-funded education initiatives (e.g., SmartLabs, PNRR), distinct from general education policy.",
        "program_specific",
    ),
    21: (
        "EU Security & Sensitive Data Governance",
        "EU-level governance discourse concerning security, sensitive data categories, and protected information domains.",
        "substantive",
    ),
    22: (
        "Program-Specific Education Initiative (SmartLabs/PNRR)",
        "Additional component capturing program-specific education initiatives tied to national recovery or investment plans.",
        "program_specific",
    ),
    23: (
        "EU Education, Research & Strategic Decision Frameworks",
        "European policy discourse linking education, research, and strategic decision-making frameworks.",
        "substantive",
    ),
    24: (
        "Education, Training & Skills Policy (Generative AI)",
        "Education and training policy language explicitly engaging with generative AI technologies.",
        "substantive",
    ),
    25: (
        "Health & Care Generative AI Applications",
        "Policy discourse on generative AI applications in health and care contexts.",
        "substantive",
    ),
}


def main() -> None:
    if not TEMPLATE.exists():
        raise SystemExit(f"Template not found: {TEMPLATE}")

    rows = []
    with TEMPLATE.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        for r in reader:
            rows.append(r)

    # adds a "category" column without breaking anything (I Hope).
    out_fieldnames = list(fieldnames)
    if "category" not in out_fieldnames:
        out_fieldnames.append("category")

    for r in rows:
        pc = int(r["pc"])
        label, desc, cat = LABELS.get(pc, ("", "", ""))
        r["label"] = label
        r["short_description"] = desc
        r["coder"] = "INA"
        r["notes"] = ""
        r["category"] = cat

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=out_fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"Wrote: {OUT}")
    print("Tip: open this CSV in a spreadsheet editor and revise labels/notes/coder as needed.")


if __name__ == "__main__":
    main()
