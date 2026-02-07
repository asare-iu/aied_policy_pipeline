import pandas as pd
from pathlib import Path

BASE = Path("data/derived/step5_models_edu_embedded_40pc")
CSV_PATH = BASE / "pc_interpretations.csv"

df = pd.read_csv(CSV_PATH)

for c in ["label_expanded", "admissible", "interpretable", "note"]:
    if c in df.columns:
        df[c] = df[c].astype("string")

ANNOTATIONS = {
    "PC1": ("English AI-policy backbone (AI/data/digital/research/education) vs multilingual/noise tail", "Y", "Y",
            "Dominant policy vocabulary; negative side contains multilingual fragments and numeric artefacts"),
    "PC2": ("AI systems + skills/education framing vs US defense header artefacts", "Y", "Y",
            "Substantive skills/education governance contrasted against US email/header-like boilerplate"),
    "PC3": ("XML/appdata/formatting artefact component", "Y", "N",
            "Markup/export noise (xml, rcp, appdata)"),
    "PC4": ("School infrastructure modernization via SmartLabs/PnRR programmes", "Y", "Y",
            "Education modernization framed through equipment/lab investment programmes"),
    "PC5": ("AI risk + personal data protection + SmartLabs programme packaging", "Y", "Y",
            "Education appears inside risk/privacy governance and programme implementation language"),
    "PC6": ("US defense R&D/talent pipeline discourse with AI education references", "Y", "N",
            "Primarily US defense/appropriations genre; education appears as incidental"),
    "PC7": ("EU AI Act-style compliance vocabulary artefact (tokenization)", "Y", "N",
            "Fragmented compliance terms (ar ticle, cybersecur ity) suggest OCR/token splitting"),
    "PC8": ("Academic/scientific web-citation footprint vs classroom implementation vocabulary", "Y", "N",
            "Dominated by URLs/citations/years; weak substantive interpretability"),
    "PC9": ("Research funding/accounting metrics and programme totals", "Y", "Y",
            "Governance via funding allocations, counts, programme growth"),
    "PC10": ("Mixed statutory drafting vocabulary vs AI/digital policy vocabulary", "Y", "N",
            "Legislative amendment structure dominates; treat as genre"),
    "PC11": ("AI assurance/quality framework (external quality, learning elements)", "Y", "Y",
            "Quality assurance framing applicable to education-related AI systems"),
    "PC12": ("Serbia-style science/innovation ministry programme governance", "Y", "Y",
            "Scientific research and ministry programme administration"),
    "PC13": ("UK/Australia guidance + external quality + web footprints", "Y", "Y",
            "Guidance and QA language present though mixed with URLs"),
    "PC14": ("US congressional conference/contained-provision boilerplate", "Y", "N",
            "Legislative reconciliation genre; not conceptually meaningful for education"),
    "PC15": ("Education + machine learning + statutory drafting blend", "Y", "Y",
            "Education and skills embedded in broader governance/legal framing"),
    "PC16": ("Education + data protection/legal rights (privacy law embedding)", "Y", "Y",
            "Education embedded via privacy, processing, rights and institutional contexts"),
    "PC17": ("Digital transformation + EU QA + security + translation artefacts", "Y", "Y",
            "Transformation/QA/security governance with some translation noise"),
    "PC18": ("Cybersecurity + human rights/privacy institutional bios", "Y", "N",
            "Bio/affiliation style text dominates; limited education interpretability"),
    "PC19": ("Ministry/implementation/pilot programme administration", "Y", "Y",
            "Policy enacted through pilots, programmes, ministry implementation"),
    "PC20": ("Public services + ministry education + generative AI governance packaging", "Y", "Y",
            "Generative AI appears inside public-service and ministry governance language"),
    "PC21": ("Official recognition/assessment frameworks (IDPS/TARF/TAAF)", "Y", "N",
            "Highly domain-specific recognition jargon; education link unclear"),
    "PC22": ("University/health/services/admin mix incl. GenAI", "Y", "N",
            "Cross-domain mix (health/services/university); treat cautiously"),
    "PC23": ("Risk/reporting/security + coast guard genre contamination", "Y", "N",
            "Transport/maritime/coast guard dominates; not education"),
    "PC24": ("EU health/tech methods + model/tool governance", "Y", "Y",
            "Governance language around models/methods/tools; education appears embedded"),
    "PC25": ("Decision-making + liability + digital skills (governance of automated decisions)", "Y", "Y",
            "Substantive: liability and decision-making governance intersects education"),
    "PC26": ("Health + recognition/assessment + GenAI + defense mix", "Y", "N",
            "Cross-domain cluster; interpret only if you later subset by country/doc type"),
    "PC27": ("Spanish-language digital transformation + decision-making/liability", "Y", "Y",
            "Non-English but substantively coherent governance theme"),
    "PC28": ("Health/cybersecurity + liability/damages (accountability governance)", "Y", "Y",
            "Accountability and harm/liability framing; education appears indirectly"),
    "PC29": ("Liability/damages + schooling/teachers standards", "Y", "Y",
            "Education embedded in liability/standards discourse"),
    "PC30": ("Cybersecurity + jobs/workforce + GenAI + ministry packaging", "Y", "Y",
            "Workforce and labor-market framing where education is embedded"),
    "PC31": ("Generative AI tools + policy innovation + protection framing", "Y", "Y",
            "GenAI governance with privacy/protection and online learning references"),
    "PC32": ("National pilot + risk + funding (NAIRR/NSF) + processing", "Y", "Y",
            "Capacity-building governance via pilots and national infrastructure"),
    "PC33": ("Skills/training + ethics/privacy + armed forces/workforce", "Y", "Y",
            "Workforce + ethics governance; education appears as skills pipeline"),
    "PC34": ("Serbia liability/damages + scientific development programmes", "Y", "Y",
            "Accountability + development-programme governance"),
    "PC35": ("Country risk/impact assessments + digital tech + bias", "Y", "Y",
            "Risk framing with bias/assessment language; education embedded"),
    "PC36": ("Token-split compliance/product requirements + aviation cyber", "Y", "N",
            "Severe token splitting suggests OCR artefact; not cleanly interpretable"),
    "PC37": ("GenAI + privacy + facial recognition + programme governance", "Y", "Y",
            "Surveillance/privacy governance intersects education via programme contexts"),
    "PC38": ("Geopolitics + liability/damages + processing governance", "Y", "Y",
            "Cross-national governance framing; education not central but embedded"),
    "PC39": ("Cybersecurity programmes/pilots/projects + institutionalization", "Y", "Y",
            "Education embedded through training/programmes and institutional capacity"),
    "PC40": ("Task force/strategy/action plan + national AI infrastructure + aviation", "Y", "Y",
            "Coordination mechanisms and strategy governance; education appears as subtheme"),
}

for i, row in df.iterrows():
    pc = row["pc"]
    if pc in ANNOTATIONS:
        label, adm, interp, note = ANNOTATIONS[pc]
        df.at[i, "label_expanded"] = label
        df.at[i, "admissible"] = adm
        df.at[i, "interpretable"] = interp
        df.at[i, "note"] = note

df.to_csv(CSV_PATH, index=False)
