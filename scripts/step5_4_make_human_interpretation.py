#!/usr/bin/env python3
"""
Create Human_interpretation.csv for the 40-PC artifact-stripped PCA run.

- One row per PC
- Explicit admissibility decision
- Evidence-based notes only
- No confidence language
"""

import argparse
import pandas as pd
from pathlib import Path


PC_INTERPRETATION = {
    "PC1": {
        "pc_type": "structural",
        "admissibility": "inadmissible",
        "tags": "structural_language,structural_masking,structural_multilingual",
        "short_label": "English AI vocab vs masked/non-English",
        "long_label": "English AI governance vocabulary contrasted with masked tokens and non-English language artifacts.",
        "pos_summary": "English AI governance and public-sector framing.",
        "neg_summary": "Masked/unmasked tokens and non-English fragments.",
    },
    "PC2": {
        "pc_type": "mixed",
        "admissibility": "mixed",
        "tags": "substantive_ai_governance,structural_us_statute",
        "short_label": "AI governance vs US defense code",
        "long_label": "AI governance discourse contrasted with US defense statutory drafting form.",
        "pos_summary": "AI systems, ethics, governance, generative AI.",
        "neg_summary": "US code and defense statute scaffolding.",
    },
    "PC3": {
        "pc_type": "structural",
        "admissibility": "inadmissible",
        "tags": "structural_language,language_spanish",
        "short_label": "Spanish language axis",
        "long_label": "Spanish-language surface form dominates variance.",
        "pos_summary": "Spanish policy language.",
        "neg_summary": "English AI and statutory vocabulary.",
    },
    "PC4": {
        "pc_type": "admissible",
        "admissibility": "admissible",
        "tags": "privacy_data_protection,rights,legal_governance",
        "short_label": "Data protection governance",
        "long_label": "Personal data protection and privacy governance contrasted with AI research and systems framing.",
        "pos_summary": "Controller, consent, data subject, privacy architecture.",
        "neg_summary": "AI systems, research, generative AI framing.",
    },
    "PC5": {
        "pc_type": "structural",
        "admissibility": "inadmissible",
        "tags": "structural_language,language_bulgarian",
        "short_label": "Bulgarian language axis",
        "long_label": "Bulgarian-language surface form dominates variance.",
        "pos_summary": "Bulgarian education and digital transformation language.",
        "neg_summary": "English statutory and cross-lingual boilerplate.",
    },
    "PC6": {
        "pc_type": "structural",
        "admissibility": "inadmissible",
        "tags": "structural_ids_dates,structural_boilerplate,privacy_data_protection",
        "short_label": "IDs and boilerplate residue",
        "long_label": "Document identifiers and boilerplate dominate variance.",
        "pos_summary": "IDs, dates, headers with privacy terms.",
        "neg_summary": "Clean strategy-oriented development language.",
    },
    "PC7": {
        "pc_type": "structural",
        "admissibility": "inadmissible",
        "tags": "structural_ids_dates,structural_headers",
        "short_label": "Document IDs and dates",
        "long_label": "Identifiers and date fragments dominate variance.",
        "pos_summary": "Document IDs and dates.",
        "neg_summary": "Congressional drafting markup.",
    },
    "PC8": {
        "pc_type": "structural",
        "admissibility": "inadmissible",
        "tags": "structural_language,language_german",
        "short_label": "German stopword axis",
        "long_label": "German-language stopwords dominate variance.",
        "pos_summary": "German stopwords.",
        "neg_summary": "Data sharing and privacy language.",
    },
    "PC9": {
        "pc_type": "admissible",
        "admissibility": "admissible",
        "tags": "ai_ethics,research_framing,digital_transformation",
        "short_label": "Ethics vs digital delivery",
        "long_label": "AI ethics and research framing contrasted with digital transformation and service delivery.",
        "pos_summary": "Ethics, research, reports, national AI framing.",
        "neg_summary": "Digital transformation, services, skills.",
},
    "PC10": {
        "pc_type": "structural",
        "admissibility": "inadmissible",
        "tags": "structural_legislative,structural_markup",
        "short_label": "Legislative mechanics",
        "long_label": "Statutory drafting mechanics dominate variance.",
        "pos_summary": "Insert, strike, amend, redesignate.",
        "neg_summary": "Conference report scaffolding.",
    },
    "PC11": {
        "pc_type": "structural",
        "admissibility": "inadmissible",
        "tags": "structural_legislative,structural_corrupted_tokens",
        "short_label": "Statutory scaffold and corrupted tokens",
        "long_label": "Statutory scaffolding and corrupted tokenization dominate variance.",
        "pos_summary": "Drafting scaffolding with AI terms embedded.",
        "neg_summary": "Corrupted regulation fragments.",
    },
    "PC12": {
        "pc_type": "admissible",
        "admissibility": "admissible",
        "tags": "uk_policy,online_targeting,algorithmic_bias",
        "short_label": "UK targeting and bias oversight",
        "long_label": "UK oversight of online targeting and algorithmic bias.",
        "pos_summary": "Gov.uk publications, targeting, bias review.",
        "neg_summary": "Implementation and processing language.",
    },
    "PC13": {
        "pc_type": "admissible",
        "admissibility": "admissible",
        "tags": "human_rights,automated_decision_making,algorithmic_bias",
        "short_label": "Rights and ADM vs R&D",
        "long_label": "Human rights and ADM governance contrasted with research and innovation framing.",
        "pos_summary": "Rights, decision-making, bias.",
        "neg_summary": "Research funding and innovation.",
    },
    "PC14": {
        "pc_type": "admissible",
        "admissibility": "admissible",
        "tags": "eu_policy,governance_architecture,liability_risk",
        "short_label": "Strategy vs liability",
        "long_label": "Governance architecture contrasted with liability and risk framing.",
        "pos_summary": "Strategy, data sharing, public sector.",
        "neg_summary": "Liability, risk, automated harm.",
    },
    "PC15": {
        "pc_type": "admissible",
        "admissibility": "admissible",
        "tags": "skills_training,applied_programs",
        "short_label": "Skills and applied programs",
        "long_label": "Skills and training programs contrasted with research and strategy framing.",
        "pos_summary": "Training, skills, applied models.",
        "neg_summary": "Research and governance framing.",
    },
    "PC16": {
        "pc_type": "admissible",
        "admissibility": "admissible",
        "tags": "privacy_data_protection,human_rights,legal_framework",
        "short_label": "Legal privacy vs ML quality",
        "long_label": "Legal privacy and rights governance contrasted with ML and quality management.",
        "pos_summary": "Rights, privacy, legal authority.",
        "neg_summary": "ML models and quality management.",
    },
    "PC17": {
        "pc_type": "admissible",
        "admissibility": "admissible",
        "tags": "public_sector,automated_decision_making,regulatory_oversight",
        "short_label": "Public-sector ADM oversight",
        "long_label": "Public-sector automated decision-making oversight.",
        "pos_summary": "Public sector decision-making and bias review.",
        "neg_summary": "Research and education artifacts.",
    },
    "PC18": {
        "pc_type": "mixed",
        "admissibility": "mixed",
        "tags": "defense_science_programs,cyber_security,structural_institutional",
        "short_label": "Defense science programs",
        "long_label": "Defense and science program governance contrasted with market and EU framing.",
        "pos_summary": "Defense, committees, science programs.",
        "neg_summary": "Market and liability framing.",
    },
    "PC19": {
        "pc_type": "admissible",
        "admissibility": "admissible",
        "tags": "education,skills_training,privacy_data_protection",
        "short_label": "Education capacity and data rights",
        "long_label": "Education capacity-building combined with data protection.",
        "pos_summary": "Schools, students, teachers, data protection.",
        "neg_summary": "Infrastructure and administrative framing.",
    },
    "PC20": {
        "pc_type": "admissible",
        "admissibility": "exclude",
        "tags": "education_infrastructure,procurement_program",
        "short_label": "SmartLabs infrastructure program",
        "long_label": "Education infrastructure procurement program.",
        "pos_summary": "SmartLabs and laboratory modernization.",
        "neg_summary": "Research and ethics framing.",
    },
"PC21": {
        "pc_type": "mixed",
        "admissibility": "mixed",
        "tags": "eu_safety,cyber_security,structural_markup",
        "short_label": "EU safety vs markup",
        "long_label": "EU safety and cyber framing contrasted with conference markup.",
        "pos_summary": "Safety and cyber policy.",
        "neg_summary": "Conference markup language.",
    },
    "PC22": {
        "pc_type": "admissible",
        "admissibility": "exclude",
        "tags": "education_infrastructure,public_sector",
        "short_label": "SmartLabs public-sector buildout",
        "long_label": "Public-sector education infrastructure buildout.",
        "pos_summary": "Schools and laboratories.",
        "neg_summary": "Labor market framing.",
    },
    "PC23": {
        "pc_type": "admissible",
        "admissibility": "admissible",
        "tags": "service_delivery,generative_ai,public_administration",
        "short_label": "Service delivery and deployment",
        "long_label": "AI deployment and service delivery contrasted with rights-heavy governance.",
        "pos_summary": "Services, applications, deployment.",
        "neg_summary": "Rights and regulatory framing.",
    },
    "PC24": {
        "pc_type": "admissible",
        "admissibility": "admissible",
        "tags": "education,skills_training,algorithmic_bias",
        "short_label": "Education and bias oversight",
        "long_label": "Education and skills governance with algorithmic bias oversight.",
        "pos_summary": "Education, training, bias review.",
        "neg_summary": "Cyber and industrial framing.",
    },
    "PC25": {
        "pc_type": "admissible",
        "admissibility": "admissible",
        "tags": "health,health_data,public_services",
        "short_label": "Health and health data",
        "long_label": "Healthcare delivery and health data systems.",
        "pos_summary": "Healthcare services and patient data.",
        "neg_summary": "Economic and industrial framing.",
    },
    "PC26": {
        "pc_type": "admissible",
        "admissibility": "admissible",
        "tags": "liability_harm,damage,legal_framework",
        "short_label": "Civil liability and damage",
        "long_label": "Civil liability and damage in automated systems.",
        "pos_summary": "Liability and damage.",
        "neg_summary": "Governance and rights framing.",
    },
    "PC27": {
        "pc_type": "admissible",
        "admissibility": "admissible",
        "tags": "foundation_models,generative_ai,model_governance",
        "short_label": "Foundation model ecosystem",
        "long_label": "Foundation and generative model ecosystem governance.",
        "pos_summary": "Foundation models and projects.",
        "neg_summary": "Standards and safety framing.",
    },
    "PC28": {
        "pc_type": "admissible",
        "admissibility": "admissible",
        "tags": "foundation_models,rights_security_privacy",
        "short_label": "Foundation model rights and security",
        "long_label": "Rights, security, and privacy in foundation model governance.",
        "pos_summary": "Rights and security framing.",
        "neg_summary": "Sectoral application framing.",
    },
    "PC29": {
        "pc_type": "admissible",
        "admissibility": "admissible",
        "tags": "liability_harm,automated_systems,cyber_security",
        "short_label": "Automated system liability",
        "long_label": "Liability and harm in automated systems.",
        "pos_summary": "Liability and damage language.",
        "neg_summary": "Digital transformation framing.",
    },
    "PC30": {
        "pc_type": "admissible",
        "admissibility": "admissible",
        "tags": "automated_decision_making,human_rights,algorithmic_bias",
        "short_label": "ADM and rights",
        "long_label": "Algorithmic decision-making and human rights protections.",
        "pos_summary": "ADM and bias protections.",
        "neg_summary": "Platform targeting governance.",
    },"PC31": {
        "pc_type": "admissible",
        "admissibility": "admissible",
        "tags": "foundation_models,health,liability",
        "short_label": "Models and health governance",
        "long_label": "Model-centric governance intersecting with health and liability.",
        "pos_summary": "Models and healthcare framing.",
        "neg_summary": "Systems and transport framing.",
    },
    "PC32": {
        "pc_type": "admissible",
        "admissibility": "admissible",
        "tags": "online_targeting,platform_regulation,risk_governance",
        "short_label": "Platform targeting regulation",
        "long_label": "Regulation of online targeting and platform risk.",
        "pos_summary": "Targeting and regulatory oversight.",
        "neg_summary": "Liability and damage regime.",
    },
    "PC33": {
        "pc_type": "structural",
        "admissibility": "inadmissible",
        "tags": "structural_language,language_french",
        "short_label": "French legal language axis",
        "long_label": "French legal surface form dominates variance.",
        "pos_summary": "French legal language.",
        "neg_summary": "English service discourse.",
    },
    "PC34": {
        "pc_type": "structural",
        "admissibility": "inadmissible",
        "tags": "structural_artifact_lowprop,defense_appropriations",
        "short_label": "Defense and artifact axis",
        "long_label": "Defense appropriations and artifact tokens dominate variance.",
        "pos_summary": "Defense and fiscal language.",
        "neg_summary": "Legal and maritime terms.",
    },
    "PC35": {
        "pc_type": "admissible",
        "admissibility": "admissible",
        "tags": "public_administration,standards,agency_governance",
        "short_label": "Agency standards governance",
        "long_label": "Public administration standards and agency governance.",
        "pos_summary": "Agencies and standards.",
        "neg_summary": "Targeting and translation noise.",
    },
    "PC36": {
        "pc_type": "admissible",
        "admissibility": "admissible",
        "tags": "aviation_safety,transportation,automation",
        "short_label": "Aviation and transport safety",
        "long_label": "Safety governance in aviation and transport automation.",
        "pos_summary": "Aviation and vehicle safety.",
        "neg_summary": "Defense and cyber framing.",
    },
    "PC37": {
        "pc_type": "structural",
        "admissibility": "inadmissible",
        "tags": "structural_language,language_kinyarwanda",
        "short_label": "Kinyarwanda language axis I",
        "long_label": "Kinyarwanda surface form dominates variance.",
        "pos_summary": "Kinyarwanda language.",
        "neg_summary": "Governance residue.",
    },
    "PC38": {
        "pc_type": "structural",
        "admissibility": "inadmissible",
        "tags": "structural_language,language_kinyarwanda",
        "short_label": "Kinyarwanda language axis II",
        "long_label": "Kinyarwanda surface form dominates variance.",
        "pos_summary": "Kinyarwanda language.",
        "neg_summary": "Lowprop and governance residue.",
    },
    "PC39": {
        "pc_type": "structural",
        "admissibility": "inadmissible",
        "tags": "structural_language,language_kinyarwanda",
        "short_label": "Kinyarwanda language axis III",
        "long_label": "Kinyarwanda surface form dominates variance.",
        "pos_summary": "Kinyarwanda language.",
        "neg_summary": "Generative and security framing.",
    },
    "PC40": {
        "pc_type": "mixed",
        "admissibility": "mixed",
        "tags": "foundation_models,defense_space,structural_language",
        "short_label": "Foundation models vs language axis",
        "long_label": "Foundation model and defense/space governance contrasted with language artifacts.",
        "pos_summary": "Foundation and defense models.",
        "neg_summary": "Language artifacts.",
    },
}
def collect_terms(df, pc, side, n):
    sub = df[(df["pc"] == pc) & (df["side"] == side)].sort_values("rank")
    return sub["term"].astype(str).tolist()[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--top_terms_long_csv",
        default="data/derived/step5_models_full_artifact_stripped_40pc/top_terms_long.csv",
    )
    ap.add_argument(
        "--out_csv",
        default="data/derived/step5_models_full_artifact_stripped_40pc/Human_interpretation.csv",
    )
    ap.add_argument("--evidence_terms_per_side", type=int, default=12)
    args = ap.parse_args()

    df = pd.read_csv(args.top_terms_long_csv)

    rows = []
    pcs = sorted(df["pc"].unique(), key=lambda x: int(x.replace("PC", "")))

    for pc in pcs:
        meta = PC_INTERPRETATION.get(pc)
        if not meta:
            continue

        if meta["admissibility"] == "exclude":
            continue

        pos_terms = collect_terms(df, pc, "pos", args.evidence_terms_per_side)
        neg_terms = collect_terms(df, pc, "neg", args.evidence_terms_per_side)

        admissibility_tag = f"admissibility_{meta['admissibility']}"
        tags = ",".join([meta["tags"], admissibility_tag])

        notes = (
            f"POS evidence: {', '.join(pos_terms)} | "
            f"NEG evidence: {', '.join(neg_terms)}"
        )

        rows.append({
            "pc": pc,
            "admissibility": meta["admissibility"],
            "pc_type": meta["pc_type"],
            "tags": tags,
            "short_label": meta["short_label"],
            "long_label": meta["long_label"],
            "pos_summary": meta["pos_summary"],
            "neg_summary": meta["neg_summary"],
            "notes": notes,
        })

    out_df = pd.DataFrame(rows)
    out_df["pc_num"] = out_df["pc"].str.replace("PC", "", regex=False).astype(int)
    out_df = out_df.sort_values("pc_num").drop(columns=["pc_num"])

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)
    print(f"Wrote {args.out_csv} ({len(out_df)} rows)")


if __name__ == "__main__":
    main()
