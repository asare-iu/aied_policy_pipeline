import pandas as pd
from pathlib import Path
import spacy

IN_PATH = "evidence/egypt_pilot/05_igt_iad/01_rule_candidates/egypt_edu_rule_candidates.tsv"
OUT_DIR = Path("evidence/egypt_pilot/05_igt_iad/02_adico_extraction")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEONTIC_TERMS = {"shall","must","may","should","required","require","prohibited","prohibit","mandatory","forbid","ban","obliged"}

def extract_adico(nlp, text: str):
    doc = nlp(text)

    # Deontic: first matching deontic token (lower)
    deontic = ""
    for t in doc:
        tl = t.text.lower()
        if tl in DEONTIC_TERMS:
            deontic = t.text
            break

    # Attribute: prefer nominal subject of the main verb (ROOT), else first noun chunk
    root = next((t for t in doc if t.dep_ == "ROOT"), None)
    attr = ""
    aim = ""
    obj = ""
    cond = ""

    if root is not None:
        aim = root.lemma_
        # subject
        for ch in root.children:
            if ch.dep_ in ("nsubj", "nsubjpass"):
                attr = ch.text
                # expand subject span (subtree) if longer
                attr = doc[ch.left_edge.i: ch.right_edge.i+1].text
                break
        # object / complement
        for ch in root.children:
            if ch.dep_ in ("dobj","obj","pobj","attr","acomp","xcomp","ccomp"):
                obj = doc[ch.left_edge.i: ch.right_edge.i+1].text
                break

    if not attr:
        # fallback: first noun chunk
        ncs = list(doc.noun_chunks)
        if ncs:
            attr = ncs[0].text

    # Condition: capture leading subordinate clauses and "if/when/unless" spans
    # Simple heuristic: any token with dep_ "mark" or text triggers, take its subtree sentence span
    cond_spans = []
    for t in doc:
        if t.text.lower() in ("if","when","unless","until","provided","subject"):
            span = doc[t.left_edge.i: t.right_edge.i+1].text
            cond_spans.append(span)
    cond = " | ".join(sorted(set(cond_spans)))

    # I (or else): leave blank on first pass; typically requires deeper parsing
    imp = ""

    # Confidence: low if missing aim or attribute
    flags = []
    if not aim: flags.append("missing_aim")
    if not attr: flags.append("missing_attribute")
    if not deontic: flags.append("missing_deontic")

    return attr.strip(), deontic.strip(), aim.strip(), imp.strip(), cond.strip(), obj.strip(), ";".join(flags)

def main():
    df = pd.read_csv(IN_PATH, sep="\t", dtype=str).fillna("")
    nlp = spacy.load("en_core_web_sm")

    out_rows = []
    for r in df.itertuples(index=False):
        sent = getattr(r, "sentence")
        A,D,Aim,I,C,O,flags = extract_adico(nlp, sent)
        out_rows.append({
            "country": getattr(r, "country", ""),
            "source_doc": getattr(r, "source_doc", ""),
            "chunk_id": getattr(r, "chunk_id", ""),
            "sent_id": getattr(r, "sent_id", ""),
            "sentence": sent,
            "A_attribute": A,
            "D_deontic": D,
            "I_or_else": I,
            "C_condition": C,
            "O_object": O,
            "Aim_verb": Aim,
            "adico_flags": flags
        })

    out = pd.DataFrame(out_rows)
    out_path = OUT_DIR / "egypt_edu_adico_autopass.tsv"
    out.to_csv(out_path, sep="\t", index=False)

    # quick stats
    summary = {
        "rows": len(out),
        "missing_deontic": int(out["D_deontic"].eq("").sum()),
        "missing_attribute": int(out["A_attribute"].eq("").sum()),
        "missing_aim": int(out["Aim_verb"].eq("").sum()),
        "flagged_rows": int(out["adico_flags"].ne("").sum()),
    }
    pd.DataFrame([summary]).to_csv(OUT_DIR / "adico_autopass_summary.tsv", sep="\t", index=False)
    print("SUMMARY:", summary)
    print("WROTE:", out_path)

if __name__ == "__main__":
    main()
