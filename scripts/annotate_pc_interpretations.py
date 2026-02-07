import pandas as pd
from pathlib import Path

# -------- CONFIG --------
BASE = Path("data/derived/step5_models_title_edu_40pc")
CSV_PATH = BASE / "pc_interpretations.csv"

# -------- LOAD --------
df = pd.read_csv(CSV_PATH)

for c in ["label_expanded","admissible","interpretable","note"]:
    if c in df.columns:
        df[c] = df[c].astype("string")


# -------- AUTHORITATIVE ANNOTATIONS --------
ANNOTATIONS = {
    "PC1":  ("Language dominance + education transformation discourse vs numeric artefacts", "Y", "N",
             "Separates Bulgarian-language education text from numeric/token noise; structural not substantive"),
    "PC2":  ("Global AI and digital skills policy frame (skills pipeline)", "Y", "Y",
             "Canonical OECD/EU human-capital framing of AI in education; language-confounded but meaningful"),
    "PC3":  ("Quality assurance and safety governance applied to education systems", "Y", "Y",
             "Education treated as a regulated system using assurance, reliability, and guideline logics"),
    "PC4":  ("Generative AI governance in education (ethics, literacy, privacy, leadership)", "Y", "Y",
             "Post-2022 reactive governance focused on GenAI risks and institutional responses"),
    "PC5":  ("Biometrics and face-recognition evaluation artefact", "Y", "N",
             "Topic contamination from biometric testing reports"),
    "PC6":  ("Classroom digitalization and AI literacy vs legal-policy boilerplate", "Y", "Y",
             "Separates pedagogical implementation language from institutional legal scaffolding"),
    "PC7":  ("EU and regional policy boilerplate vs national accreditation discourse", "Y", "N",
             "Primarily genre and language separation"),
    "PC8":  ("Regional and local development policy embedding education", "Y", "Y",
             "Education framed within territorial and social policy contexts"),
    "PC9":  ("Higher education law and R&D governance vs evaluation annex artefacts", "Y", "Y",
             "Legislative and institutional governance of higher education systems"),
    "PC10": ("Education–science–innovation ministry ecosystem", "Y", "Y",
             "State-centric framing of education within national innovation systems"),
    "PC11": ("National innovation and skills governance vs school-level implementation", "Y", "Y",
             "Contrasts innovation-ministry logic with education-ministry operations"),
    "PC12": ("Copyright and webpage boilerplate artefact", "Y", "N",
             "Web scrape footer text with no policy substance"),
    "PC13": ("Numeric templates and measurement bins artefact", "Y", "N",
             "Formatting and template noise"),
    "PC14": ("National accreditation and evaluation discourse (language-bound)", "Y", "Y",
             "Substantive accreditation and QA framing within Bulgarian context"),
    "PC15": ("STEM and cyber-physical systems vision linked to education strategy", "Y", "Y",
             "Education framed as pipeline to strategic industrial sectors"),
    "PC16": ("Data quality and measurement frameworks vs institutional QA", "Y", "Y",
             "Distinguishes technical data assurance from policy QA"),
    "PC17": ("National strategy and budget packaging of AI and education", "Y", "Y",
             "Administrative bundling of AI, education, youth, and budget"),
    "PC18": ("Academic and project-based smart education discourse", "Y", "Y",
             "Education framed through projects, programmes, and scholarly dissemination"),
    "PC19": ("Principles-based generative AI consultation frameworks", "Y", "Y",
             "Soft-law governance of GenAI in schools"),
    "PC20": ("Programmatic funding instruments for AI and education", "Y", "Y",
             "Education governed through calls, projects, and industrial policy tools"),
    "PC21": ("Automated driving and transport safety testing artefact", "Y", "N",
             "Topic contamination unrelated to education policy"),
    "PC22": ("Smart education mixed with transport testing artefact", "Y", "N",
             "Contaminated component combining unrelated domains"),
    "PC23": ("Mixed scrape artefact (URLs, standards, youth sports)", "Y", "N",
             "Non-coherent scrape noise"),
    "PC24": ("Education innovation programmes and teacher-driven edtech", "Y", "Y",
             "Institutionalized innovation programmes within education systems"),
    "PC25": ("Face recognition and biometric surveillance technologies", "Y", "N",
             "Off-target surveillance technology content"),
    "PC26": ("Leadership and task-force governance for generative AI", "Y", "Y",
             "Organizational response mechanisms to GenAI"),
    "PC27": ("Internships and education-to-work pipeline governance", "Y", "Y",
             "Education framed through employability and transition to work"),
    "PC28": ("Project-based and transnational education pilots", "Y", "Y",
             "Education as economic development and project instrument"),
    "PC29": ("Industrial maintenance and engineering artefact", "Y", "N",
             "Non-education technical contamination"),
    "PC30": ("STEM, vocational training, and employment pipeline", "Y", "Y",
             "Human-capital framing from early learning to labor market"),
    "PC31": ("Industrial diagnostics and engineering research artefact", "Y", "N",
             "Non-education technical contamination"),
    "PC32": ("STEM metrics and schooling vs digital learning resources", "Y", "Y",
             "Within-language contrast of education approaches"),
    "PC33": ("Safety and reliability engineering intersecting education governance", "Y", "Y",
             "AI assurance ecosystem intersecting education via national agencies"),
    "PC34": ("Distance and online learning operations vs higher education governance", "Y", "Y",
             "Operational education practice contrasted with institutional governance"),
    "PC35": ("Administrative modernization of educational processes", "Y", "Y",
             "Implementation and managerial transformation framing"),
    "PC36": ("EU-level regulatory governance vs national implementation", "Y", "Y",
             "Multi-level governance and regulatory layering"),
    "PC37": ("Programme administration and implementation capacity", "Y", "Y",
             "Operational state capacity for delivery"),
    "PC38": ("Curriculum and instructional core vs ethical governance", "Y", "Y",
             "Pedagogical design contrasted with governance frameworks"),
    "PC39": ("Task-force and action-plan governance mechanisms", "Y", "Y",
             "Institutional coordination distinct from funding instruments"),
    "PC40": ("Project monitoring, baselines, and evaluation governance", "Y", "Y",
             "Education enacted through project management and M&E"),
}

# -------- APPLY --------
for idx, row in df.iterrows():
    pc = row["pc"]
    if pc in ANNOTATIONS:
        label, admissible, interpretable, note = ANNOTATIONS[pc]
        df.at[idx, "label_expanded"] = label
        df.at[idx, "admissible"] = admissible
        df.at[idx, "interpretable"] = interpretable
        df.at[idx, "note"] = note

# -------- SAVE --------
df.to_csv(CSV_PATH, index=False)
print(f"Updated PC interpretations written to {CSV_PATH}")
