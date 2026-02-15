#!/usr/bin/env python3
from pathlib import Path
import pandas as pd

BASE = Path("data/derived/step5_models_edu_embedded_40pc")
IN_FILE = BASE / "pc_interpretations_deep.csv"
OUT_FILE = BASE / "pc_interpretations_deep.csv"  # in-place update

# Controlled vocabularies
EDU_ROLE = {"object", "instrument", "context", "incidental"}
POLICY_INSTR = {
    "funding_budget",
    "institutional_governance",
    "law_regulation",
    "procurement_infrastructure",
    "program_implementation",
    "standards_guidance",
    "strategy_plan",
}
SIGNAL_CLASS = {"formatting_artefact", "genre_boilerplate", "mixed", "substantive", "translation_noise"}
PRIORITY = {"exclude", "primary", "secondary"}

CODES = {
    "PC1": dict(
        education_role="context",
        policy_instrument="strategy_plan",
        signal_class="mixed",
        reporting_priority="primary",
        signal_type="Generic AI-strategy lexicon vs multilingual fragments/numeric artefacts",
        dissertation_relevance="Baseline axis for the corpus: separates core AI-policy vocabulary from multilingual/noise tails; useful for normalization and robustness checks.",
        analytic_note="Positive loadings reflect generic AI governance discourse (AI/data/digital/research/education/skills). Negative side is dominated by multilingual fragments and numeric patterns, indicating residual language/format variance rather than a policy mechanism.",
    ),
    "PC2": dict(
        education_role="object",
        policy_instrument="institutional_governance",
        signal_class="mixed",
        reporting_priority="primary",
        signal_type="AI systems + skills/education framing vs US defense/congressional header boilerplate",
        dissertation_relevance="Captures a key education-facing governance frame (skills, AI systems, ethics) while flagging contamination from defense/legal document genres.",
        analytic_note="Positive side bundles AI systems, skills, education, ethics, sector/industry and ministry language—substantive. Negative side is dominated by defense/congressional formatting tokens (xml/section/subsection/secretary), indicating genre artefacts.",
    ),
    "PC3": dict(
        education_role="incidental",
        policy_instrument="institutional_governance",
        signal_class="formatting_artefact",
        reporting_priority="exclude",
        signal_type="XML/export/OS-path residue component",
        dissertation_relevance="Quality-control dimension: identifies extraction/export contamination that should be excluded from substantive interpretation.",
        analytic_note="Positive loadings are xml/rcp/appdata/user-path strings typical of exports; negative includes defense/legal boilerplate. Treat as formatting noise (filter/strip or downweight).",
    ),
    "PC4": dict(
        education_role="object",
        policy_instrument="procurement_infrastructure",
        signal_class="substantive",
        reporting_priority="secondary",
        signal_type="School lab/infrastructure modernization programmes (SmartLabs / PNRR-style equipment investment)",
        dissertation_relevance="Concrete ‘education-as-infrastructure’ mechanism: modernization via equipment/labs and capital investment programmes.",
        analytic_note="Strong association with high school/SmartLabs/technology/equipment framing. Negative side contrasts with US defense/legal boilerplate, reinforcing this as an education-infrastructure programme cluster.",
    ),
    "PC5": dict(
        education_role="context",
        policy_instrument="law_regulation",
        signal_class="substantive",
        reporting_priority="secondary",
        signal_type="Privacy/data protection + AI risk governance packaged with education programme implementation language",
        dissertation_relevance="Shows how education appears inside broader AI-risk and data-protection governance, often via programme packaging rather than pedagogy.",
        analytic_note="Positive side links AI systems, risk, assessment, personal data, rights—classic privacy/risk governance. Education terms appear alongside programme markers (e.g., SmartLabs) indicating implementation bundles where education is a venue for governance.",
    ),
    "PC6": dict(
        education_role="incidental",
        policy_instrument="funding_budget",
        signal_class="genre_boilerplate",
        reporting_priority="exclude",
        signal_type="US defense R&D/talent pipeline appropriations genre with incidental education mentions",
        dissertation_relevance="Primarily a non-education governance genre; keep only as a document-type contaminant check.",
        analytic_note="Dominated by US defense/appropriations/talent-program language. Education appears as incidental (students/programs) within defense R&D context; not an education-governance mechanism.",
    ),
    "PC7": dict(
        education_role="incidental",
        policy_instrument="law_regulation",
        signal_class="formatting_artefact",
        reporting_priority="exclude",
        signal_type="Token-splitting/OCR artefact over EU AI Act compliance vocabulary",
        dissertation_relevance="Extraction artefact dimension; warns that compliance terms have been fragmented and should not be treated as clean conceptual signals.",
        analytic_note="Fragmented tokens (ar ticle, cybersecur ity, mity) indicate OCR/tokenization splitting. Underlying theme is compliance/regulation, but text corruption makes it non-interpretable.",
    ),
    "PC8": dict(
        education_role="incidental",
        policy_instrument="standards_guidance",
        signal_class="genre_boilerplate",
        reporting_priority="exclude",
        signal_type="Web-citation / academic-bibliographic footprint vs implementation vocabulary",
        dissertation_relevance="Document-format axis; separates heavily cited/URL-heavy texts from implementation discourse; use for cleaning not theory.",
        analytic_note="Positive side dominated by https/www/years/institutional citation patterns, suggesting bibliographic sections or scraped pages. Negative side aligns with classroom/skills terms. Treat as format/genre.",
    ),
    "PC9": dict(
        education_role="context",
        policy_instrument="funding_budget",
        signal_class="substantive",
        reporting_priority="secondary",
        signal_type="Research funding totals, programme counts, allocation/accounting governance",
        dissertation_relevance="Shows budgeting and resource-allocation as a governance mechanism; education appears via research/innovation system funding and capacity metrics.",
        analytic_note="Positive loadings are funding/program totals, counts, allocations; negative contrasts against AI-strategy lexicon. Interpretable as ‘governance through metrics and budgets.’",
    ),
    "PC10": dict(
        education_role="incidental",
        policy_instrument="law_regulation",
        signal_class="genre_boilerplate",
        reporting_priority="exclude",
        signal_type="Statutory drafting/amendment mechanics vs policy vocabulary",
        dissertation_relevance="Non-substantive legislative drafting axis; keep only for genre control.",
        analytic_note="Positive side has ‘section/subsection/committee/striking/inserting’—drafting mechanics. This is structural text, not an education or AI governance construct.",
    ),"PC11": dict(
        education_role="instrument",
        policy_instrument="standards_guidance",
        signal_class="substantive",
        reporting_priority="primary",
        signal_type="AI assurance/quality frameworks (external quality, learning elements, safety/requirements)",
        dissertation_relevance="Directly relevant to AI-in-education governance: quality assurance, evaluation elements, and safety requirements are central institutional mechanisms.",
        analytic_note="Positive loadings cluster around external quality, learning elements, requirements, safety—suggesting assurance regimes. Negative side is mainly publication/URL footprints; still substantively coherent overall.",
    ),
    "PC12": dict(
        education_role="context",
        policy_instrument="institutional_governance",
        signal_class="substantive",
        reporting_priority="secondary",
        signal_type="Science/innovation ministry programme governance (Serbia-style research administration)",
        dissertation_relevance="Institutional arrangement signal: governance by ministries and research programmes where education is embedded via universities/research systems.",
        analytic_note="Positive side emphasizes ministry, scientific research, funding, programmes, universities. Education is present as part of national research/innovation governance rather than schooling policy.",
    ),
    "PC13": dict(
        education_role="instrument",
        policy_instrument="standards_guidance",
        signal_class="mixed",
        reporting_priority="secondary",
        signal_type="Guidance + QA language with strong UK/Australia web/document fingerprints",
        dissertation_relevance="Useful for identifying ‘guidance-led governance’ (standards, QA) though partially mixed with web artefacts.",
        analytic_note="Positive side mixes QA/guidance terms (external quality, safety, review, requirements) with URL-heavy fingerprints. Treat as interpretable but mixed; cross-check with document sources.",
    ),
    "PC14": dict(
        education_role="incidental",
        policy_instrument="law_regulation",
        signal_class="genre_boilerplate",
        reporting_priority="exclude",
        signal_type="Congressional conference ‘contained provision’ reconciliation boilerplate",
        dissertation_relevance="Document-genre contaminant; not a conceptual governance dimension for education.",
        analytic_note="Dominated by conference report mechanics (‘contained provision’, ‘recedes’, ‘conferees’). Exclude from substantive education interpretation.",
    ),
    "PC15": dict(
        education_role="context",
        policy_instrument="law_regulation",
        signal_class="mixed",
        reporting_priority="secondary",
        signal_type="Education/skills + ML terms blended into statutory drafting and compliance language",
        dissertation_relevance="Illustrates how education and skills are legislated indirectly via broader governance/legal frameworks.",
        analytic_note="Positive side shows education/students/ML/skills combined with drafting tokens; negative side is rights/liability/risk/legal framing. Interpretable as legal embedding of education/skills rather than education policy per se.",
    ),
    "PC16": dict(
        education_role="context",
        policy_instrument="law_regulation",
        signal_class="substantive",
        reporting_priority="primary",
        signal_type="Education institutional context within privacy/data-protection and rights-based governance",
        dissertation_relevance="High relevance: demonstrates education’s governance coupling to privacy/processing/rights regimes (data protection in schools/universities).",
        analytic_note="Strong blend of education/schools/university with personal data, processing, protection, rights, legal terms. This is a substantive ‘rights-based governance’ axis where education is a regulated setting.",
    ),
    "PC17": dict(
        education_role="context",
        policy_instrument="strategy_plan",
        signal_class="mixed",
        reporting_priority="secondary",
        signal_type="Digital transformation + EU QA/security vocabulary with translation and citation noise",
        dissertation_relevance="Captures transformation framing that often drives education reform, but requires caution due to translation/URL contamination.",
        analytic_note="Positive side includes transformation, EU/QA/security, translated-google markers; negative side includes decision-making/bias/children and education actors. Treat as mixed: transformation discourse + noisy extraction.",
    ),
    "PC18": dict(
        education_role="incidental",
        policy_instrument="institutional_governance",
        signal_class="genre_boilerplate",
        reporting_priority="exclude",
        signal_type="Institutional bios/affiliations (cyber/privacy/human rights) rather than policy text",
        dissertation_relevance="Not a governance construct; indicates biography/attribution sections contaminating the corpus.",
        analytic_note="Dominated by professor/director/institute/department bios and submissions. Exclude from substantive PCA interpretation; strip author/affiliation blocks upstream if possible.",
    ),
    "PC19": dict(
        education_role="instrument",
        policy_instrument="program_implementation",
        signal_class="substantive",
        reporting_priority="secondary",
        signal_type="Ministry-led pilots and programme implementation (education/ministry execution mechanisms)",
        dissertation_relevance="Implementation mechanism signal: pilots/programmes as governance tools affecting education systems.",
        analytic_note="Positive side emphasizes ministry, implementation, pilot program, activities, assessment, public administration. Education appears as a governed sector through implementation structures.",
    ),
    "PC20": dict(
        education_role="context",
        policy_instrument="institutional_governance",
        signal_class="substantive",
        reporting_priority="secondary",
        signal_type="Public services governance with ministry-education coupling and GenAI packaging",
        dissertation_relevance="Shows cross-sector packaging: generative AI appears within public-service governance where education is an administrative portfolio.",
        analytic_note="Positive side includes government/public/services/health plus ministry education and generative AI. Negative side is recognition/assessment jargon—suggesting a distinct administrative cluster.",
    ),"PC21": dict(
        education_role="incidental",
        policy_instrument="institutional_governance",
        signal_class="mixed",
        reporting_priority="exclude",
        signal_type="Official recognition/assessment frameworks (IDPS/TARF/TAAF) with weak education linkage",
        dissertation_relevance="Too domain-specific without clear education mechanism; keep only if later mapped to known countries/documents where this is central.",
        analytic_note="Dominated by recognition/assessment-level jargon and authority markers; education appears mainly as a generic sector reference. Exclude for dissertation claims unless re-grounded with document provenance.",
    ),
    "PC22": dict(
        education_role="incidental",
        policy_instrument="institutional_governance",
        signal_class="mixed",
        reporting_priority="exclude",
        signal_type="University/health/services administrative mix with GenAI references",
        dissertation_relevance="Cross-domain administrative cluster; not cleanly education-governance unless filtered to specific subcorpus.",
        analytic_note="Mixes university/health/services/center/admin and GenAI. Education is present as institutional setting but not the object; treat as mixed and low-precision.",
    ),
    "PC23": dict(
        education_role="incidental",
        policy_instrument="law_regulation",
        signal_class="genre_boilerplate",
        reporting_priority="exclude",
        signal_type="Transport/maritime/coast-guard risk-reporting genre contamination",
        dissertation_relevance="Non-education domain signal; exclude from education interpretation.",
        analytic_note="Dominated by coast/guard/vessel/transport/risk/reporting language. This is a sectoral contamination axis rather than education governance.",
    ),
    "PC24": dict(
        education_role="context",
        policy_instrument="standards_guidance",
        signal_class="mixed",
        reporting_priority="secondary",
        signal_type="EU-style methods/models/tools governance (risk, liability, scientific methods) with cross-sector embedding",
        dissertation_relevance="Relevant as ‘governance of AI methods/tools’ that can apply to education; but education is not consistently the object.",
        analytic_note="Emphasizes models/methods/tools/risk/liability and EU/horizon-like vocabulary. Education appears embedded; interpret as general AI governance tooling rather than education-specific policy.",
    ),
    "PC25": dict(
        education_role="context",
        policy_instrument="law_regulation",
        signal_class="substantive",
        reporting_priority="primary",
        signal_type="Automated decision-making accountability (liability/decision-making) + skills framing",
        dissertation_relevance="Strong relevance: accountability for automated decisions intersects schooling (student decisions), higher ed admissions, and educational services procurement.",
        analytic_note="Pairs decision-making/liability/standards with digital skills and institutional actors. This is a substantive accountability axis; education is a high-stakes venue where these rules land.",
    ),
    "PC26": dict(
        education_role="incidental",
        policy_instrument="institutional_governance",
        signal_class="mixed",
        reporting_priority="exclude",
        signal_type="Health + recognition/assessment + GenAI + defense blending (cross-domain cluster)",
        dissertation_relevance="Too blended to support clean education claims; use only if later stratified by document type/country.",
        analytic_note="Combines health/authority/assessment frameworks with defense and GenAI references. Education appears via ministry education but not coherently; classify as mixed and exclude for primary reporting.",
    ),
    "PC27": dict(
        education_role="context",
        policy_instrument="strategy_plan",
        signal_class="substantive",
        reporting_priority="secondary",
        signal_type="Spanish-language digital transformation + decision/liability governance framing",
        dissertation_relevance="Shows multilingual substantive governance: transformation and accountability discourse that can structure education policy reform.",
        analytic_note="Coherent Spanish governance lexicon (transformación digital, decisión, responsabilidad) indicating non-English but interpretable policy framing. Education is contextual rather than central.",
    ),
    "PC28": dict(
        education_role="incidental",
        policy_instrument="law_regulation",
        signal_class="substantive",
        reporting_priority="secondary",
        signal_type="Health/cybersecurity harms + liability/damages accountability framing",
        dissertation_relevance="Accountability mechanism relevant to education when education is treated as a service sector exposed to cyber risk and harm frameworks.",
        analytic_note="Liability/damage/security/cyber/healthcare cluster. Education appears indirectly (skills, institutions) but the mechanism is accountability for harm; keep as secondary.",
    ),
    "PC29": dict(
        education_role="object",
        policy_instrument="law_regulation",
        signal_class="substantive",
        reporting_priority="primary",
        signal_type="Liability/damages + school/teacher standards (education embedded in accountability regimes)",
        dissertation_relevance="Directly relevant: governance of schooling via standards, fault, and liability regimes affecting teachers/schools and decision systems.",
        analytic_note="Strong coupling of liability/damages with schooling/teachers/standards. This is education-relevant accountability: how rules assign responsibility when harms occur in educational settings.",
    ),
    "PC30": dict(
        education_role="instrument",
        policy_instrument="strategy_plan",
        signal_class="substantive",
        reporting_priority="primary",
        signal_type="Cybersecurity + jobs/workforce + GenAI + ministry packaging (skills pipeline governance)",
        dissertation_relevance="Core ‘education-as-workforce pipeline’ mechanism: training/skills policy tied to cyber and GenAI labor markets.",
        analytic_note="Clusters ministry, skills, training, jobs/workers, cybersecurity, GenAI. Education appears as the instrument for workforce capacity and economic/security objectives.",
    ),"PC31": dict(
        education_role="instrument",
        policy_instrument="standards_guidance",
        signal_class="substantive",
        reporting_priority="primary",
        signal_type="GenAI tools governance + innovation + privacy/protection (online/learning references present)",
        dissertation_relevance="High relevance: governance of GenAI tools, protections, and guidance intersects education adoption and acceptable use regimes.",
        analytic_note="Pairs generative AI tools with policy/innovation and privacy/protection vocabulary; includes programme/learning/online cues. Interpretable as ‘tool-governance with protective constraints.’",
    ),
    "PC32": dict(
        education_role="instrument",
        policy_instrument="funding_budget",
        signal_class="mixed",
        reporting_priority="secondary",
        signal_type="National pilots + risk + funding infrastructure (NAIRR/NSF-like) with processing governance",
        dissertation_relevance="Capacity-building mechanism: national AI infrastructure and pilots that can enable education-sector experimentation and research.",
        analytic_note="Dominated by pilot programs, funding, national infrastructure terms (NAIRR/NSF) plus processing/risk. Education not central but present via institutional participation; mixed but usable.",
    ),
    "PC33": dict(
        education_role="instrument",
        policy_instrument="strategy_plan",
        signal_class="mixed",
        reporting_priority="secondary",
        signal_type="Skills/training + ethics/privacy + security/armed forces workforce framing",
        dissertation_relevance="Education-as-skills pipeline under security/ethics framing; relevant for how states justify training initiatives affecting curricula.",
        analytic_note="Combines skills/training, ethics/privacy, workforce and security/armed forces tokens. Education is instrumental (training) rather than the object; mixed due to defense presence.",
    ),
    "PC34": dict(
        education_role="context",
        policy_instrument="law_regulation",
        signal_class="substantive",
        reporting_priority="secondary",
        signal_type="Liability/damages + scientific/technological development programme governance (Serbia-like)",
        dissertation_relevance="Accountability + development-program governance: relevant where education institutions participate in national innovation systems and liability regimes.",
        analytic_note="Links liability/damages with scientific development programmes and implementation cues. Education is contextual through national development governance rather than schooling.",
    ),
    "PC35": dict(
        education_role="context",
        policy_instrument="standards_guidance",
        signal_class="mixed",
        reporting_priority="secondary",
        signal_type="Risk/impact assessment + bias framing tied to country/programme context",
        dissertation_relevance="Supports dissertation themes on evaluation and bias as governance concerns that often surface in education adoption debates.",
        analytic_note="Risk/impact/bias/assessment language with country/programme fingerprints. Education appears via institutional actors; treat as mixed (substantive evaluation theme + contextual noise).",
    ),
    "PC36": dict(
        education_role="incidental",
        policy_instrument="law_regulation",
        signal_class="formatting_artefact",
        reporting_priority="exclude",
        signal_type="Token-split compliance/product requirements (aviation cyber) with severe text corruption",
        dissertation_relevance="Extraction artefact; exclude from substantive claims.",
        analytic_note="Heavily fragmented tokens (‘digital elements’, ‘cybersecur ity’, split words) indicate OCR/token splitting. Underlying is product/compliance requirements, but corrupted beyond reliable interpretation.",
    ),
    "PC37": dict(
        education_role="context",
        policy_instrument="law_regulation",
        signal_class="mixed",
        reporting_priority="primary",
        signal_type="GenAI + privacy + facial recognition + programme governance (surveillance/accountability intersection)",
        dissertation_relevance="High relevance: surveillance/privacy governance intersects education as a sensitive environment (students, campuses) and often drives restrictive policies.",
        analytic_note="Combines generative AI programme language with privacy and facial recognition. Some compliance token-splitting appears, but substantive surveillance/privacy governance is clear enough for primary reporting.",
    ),
    "PC38": dict(
        education_role="incidental",
        policy_instrument="institutional_governance",
        signal_class="mixed",
        reporting_priority="secondary",
        signal_type="Geopolitics + liability/damages + processing/security governance (cross-national framing)",
        dissertation_relevance="Contextual governance axis: geopolitical/security framing that shapes sectoral policy narratives including education indirectly.",
        analytic_note="Mixes cross-national/security actors with liability/processing and damage framing. Education is not central; treat as secondary context about how security geopolitics co-travels with governance language.",
    ),
    "PC39": dict(
        education_role="instrument",
        policy_instrument="program_implementation",
        signal_class="substantive",
        reporting_priority="primary",
        signal_type="Cybersecurity programmes/pilots/projects + institutionalization mechanisms",
        dissertation_relevance="Direct mechanism for education-sector capacity building: training programmes, pilots, and institutional infrastructure for cyber/AI governance.",
        analytic_note="Strong program/pilot/project/institute vocabulary with cybersecurity/security. Education appears through training/programmes and institutional capacity-building—highly usable.",
    ),
    "PC40": dict(
        education_role="instrument",
        policy_instrument="institutional_governance",
        signal_class="substantive",
        reporting_priority="primary",
        signal_type="Task forces/strategy/action plans + national AI infrastructure coordination",
        dissertation_relevance="Core governance mechanism: coordination structures (task forces, action plans) that drive cross-sector implementation including education.",
        analytic_note="Positive side emphasizes task force, strategy, action, policy, national resources/infrastructure. Negative side includes higher-education/risk/innovation terms—indicating a coordination axis where education is a governed portfolio.",
    ),
}

def validate_patch(pc, patch):
    er = patch.get("education_role", "")
    pi = patch.get("policy_instrument", "")
    sc = patch.get("signal_class", "")
    rp = patch.get("reporting_priority", "")
    if er and er not in EDU_ROLE:
        raise ValueError(f"{pc}: invalid education_role={er}")
    if pi and pi not in POLICY_INSTR:
        raise ValueError(f"{pc}: invalid policy_instrument={pi}")
    if sc and sc not in SIGNAL_CLASS:
        raise ValueError(f"{pc}: invalid signal_class={sc}")
    if rp and rp not in PRIORITY:
        raise ValueError(f"{pc}: invalid reporting_priority={rp}")

def main():
    df = pd.read_csv(IN_FILE, dtype="string")

    # Columns we will ensure exist, then fill
    deep_cols = [
        "education_role",
        "policy_instrument",
        "signal_class",
        "reporting_priority",
        "signal_type",
        "dissertation_relevance",
        "analytic_note",
    ]
    for col in deep_cols:
        if col not in df.columns:
            df[col] = pd.Series([""] * len(df), dtype="string")

    # Apply codes
    missing = []
    for pc in df["pc"].tolist():
        if pc not in CODES:
            missing.append(pc)
            continue
        patch = CODES[pc]
        validate_patch(pc, patch)
        idx = df.index[df["pc"] == pc]
        if len(idx) != 1:
            raise ValueError(f"Expected exactly one row for {pc}, found {len(idx)}")
        i = idx[0]
        for k, v in patch.items():
            df.at[i, k] = v

    if missing:
        raise ValueError(f"Missing codes for PCs: {missing}")

    df.to_csv(OUT_FILE, index=False)
    print(f"Wrote deep interpretations (filled PC1..PC40): {OUT_FILE}")

if __name__ == "__main__":
    main()
