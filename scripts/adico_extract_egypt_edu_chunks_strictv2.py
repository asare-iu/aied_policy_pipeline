import re
import pandas as pd
from pathlib import Path
import spacy

# Input created earlier from strict-v2 chunks
IN_PATH = "evidence/egypt_pilot/05_igt_iad/regexish_chunks_from_tsv_20251231_161819/03_edu_chunks_for_adico.tsv"

# Write outputs next to the run artifacts
OUT_DIR = Path("evidence/egypt_pilot/05_igt_iad/regexish_chunks_from_tsv_20251231_161819")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Minimal deontic lexicon (extend later if needed)
DEONTIC_TERMS = {
    "shall","must","may","should",
    "required","require",
    "prohibited","prohibit",
    "mandatory","forbidden",
    "cannot","can't",
    "mustn't","shalln't"
}

OR_ELSE_CUES = re.compile(
    r"\b("
    r"or else|"
    r"penalt(y|ies)|sanction(s)?|fine(s)?|imprisonment|"
    r"liable to|punishable|"
    r"shall be punished|shall be liable|"
    r"revocation|suspension|"
    r"failure to comply|non-?compliance"
    r")\b",
    re.I
)


def extract_adibco(nlp, text: str):
    """
    Extended ADIBCO (Siddiki et al., 2011 style):
      A = Attribute
      D = Deontic
      I = Aim
      B = Object
      C = Condition
      O = Or else (sanction)
    Returns: A, D, I, B, C, O, flags
    """
    text = text or ""
    doc = nlp(text)

    # D: first matching deontic token
    D = ""
    for t in doc:
        tl = t.text.lower()
        if tl in DEONTIC_TERMS:
            D = t.text
            break

    # Root verb as Aim (I)
    root = next((t for t in doc if t.dep_ == "ROOT"), None)
    A = ""
    I = ""
    B = ""
    C = ""
    O = ""
    flags = ""

    if root is not None:
        I = root.lemma_

        # A: nominal subject of ROOT
        for ch in root.children:
            if ch.dep_ in ("nsubj", "nsubjpass"):
                A = doc[ch.left_edge.i: ch.right_edge.i + 1].text
                break

        # B: object/complement of ROOT
        for ch in root.children:
            if ch.dep_ in ("dobj","obj","pobj","attr","acomp","xcomp","ccomp"):
                B = doc[ch.left_edge.i: ch.right_edge.i + 1].text
                break

    # Fallback A: first noun chunk if no subject
    if not A:
        ncs = list(doc.noun_chunks) if hasattr(doc, "noun_chunks") else []
        if ncs:
            A = ncs[0].text

    # C: condition heuristic (if/when/unless/until/provided/subject to...)
    cond_spans = []
    for t in doc:
        if t.text.lower() in ("if","when","unless","until","provided","subject"):
            cond_spans.append(doc[t.left_edge.i: t.right_edge.i + 1].text)
    if cond_spans:
        C = " | ".join(dict.fromkeys(cond_spans))

    # O: or-else heuristic (sanction cues)
    m = OR_ELSE_CUES.search(text)
    if m:
        start = max(0, m.start() - 60)
        end = min(len(text), m.end() + 120)
        O = text[start:end].strip()

    # Flags (lightweight diagnostics)
    if not I:
        flags += "no_aim;"
    if not A:
        flags += "no_attribute;"
    if not B:
        flags += "no_object;"
    if D and D.lower() == "may":
        flags += "deontic_may;"
    if O:
        flags += "has_or_else;"

    return A, D, I, B, C, O, flags.strip(";")

def main():
    df = pd.read_csv(IN_PATH, sep="\t", dtype=str).fillna("")
    if "sentence" not in df.columns:
        raise ValueError(f"Expected a 'sentence' column in {IN_PATH}, got columns: {df.columns.tolist()}")

    # Load spaCy model (fallback if model missing)
    try:
        nlp = spacy.load("en_core_web_sm")
    except Exception:
        nlp = spacy.blank("en")

    out_rows = []
    for r in df.itertuples(index=False):
        sent = getattr(r, "sentence", "")
        A, D, I, B, C, O, flags = extract_adibco(nlp, sent)

        out_rows.append({
            "country": getattr(r, "country", ""),
            "doc_id": getattr(r, "doc_id", ""),
            "source_doc": getattr(r, "source_doc", ""),
            "chunk_id": getattr(r, "chunk_id", ""),
            "sent_id": getattr(r, "sent_id", ""),
            "sentence": sent,

            # Extended ADIBCO fields (correct semantics)
            "A_attribute": A,
            "D_deontic": D,
            "I_aim": I,
            "B_object": B,
            "C_condition": C,
            "O_or_else": O,

            "adico_flags": flags
        })

    out = pd.DataFrame(out_rows)
    out_path = OUT_DIR / "egypt_edu_adibco_autopass.tsv"
    out.to_csv(out_path, sep="\t", index=False)

    summary = {
        "rows": len(out),
        "missing_deontic": int(out["D_deontic"].eq("").sum()),
        "missing_attribute": int(out["A_attribute"].eq("").sum()),
        "missing_aim": int(out["I_aim"].eq("").sum()),
        "missing_object": int(out["B_object"].eq("").sum()),
        "has_or_else": int(out["O_or_else"].ne("").sum()),
        "flagged_rows": int(out["adico_flags"].ne("").sum()),
    }
    summ_path = OUT_DIR / "adibco_autopass_summary.tsv"
    pd.DataFrame([summary]).to_csv(summ_path, sep="\t", index=False)

    print("SUMMARY:", summary)
    print("WROTE:", out_path)
    print("WROTE:", summ_path)

if __name__ == "__main__":
    main()
