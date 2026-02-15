import os, json, time
import pandas as pd
from openai import OpenAI

ADJDIR = os.environ["ADJDIR"]
INP = os.path.join(ADJDIR, "input_chunks.tsv")
OUT_JSONL = os.path.join(ADJDIR, "llm_adjudicated_spans.jsonl")
OUT_TSV = os.path.join(ADJDIR, "llm_adjudicated_queue.tsv")
MODEL = os.environ.get("OPENAI_MODEL", "gpt-5-mini")

def is_substring(span: str, text: str) -> bool:
    span = "" if span is None else str(span)
    return (span == "") or (span in text)

def classify_exact(A, D, I, B, C, O):
    A = (A or "").strip()
    D = (D or "").strip()
    I = (I or "").strip()
    B = (B or "").strip()
    C = (C or "").strip()
    O = (O or "").strip()

    hasA = A != ""
    hasD = D != ""
    hasI = I != ""
    hasB = B != ""
    hasC = C != ""
    hasO = O != ""

    # Deterministic, hierarchical: most-specific match wins
    if hasA and hasD and hasI and hasB and hasC and hasO:
        return "rule"       # ADIBCO
    if hasA and hasD and hasI and hasC:
        return "norm"       # ADIC
    if hasA and hasI and hasC:
        return "strategy"   # AIC
    if hasI and hasC:
        return "principle"  # IC
    return "other"


INSTRUCTIONS = """
You are adjudicating ONE policy chunk (education-relevant).

Return GOLD ADIBCO spans as EXACT VERBATIM SUBSTRINGS of the chunk text.
Do NOT paraphrase. Do NOT invent text. If absent, output "".

DEFINITIONS / RULES:
- A (Attribute) must be a concrete institutional actor named in the text (e.g., Government, Ministry, Council, universities, schools, agencies, teachers, regulators). Generic groups like “countries” only count if the text clearly assigns them responsibility.
- D (Deontic) must be an explicit modal/obligation phrase in the text (e.g., must, should, shall, required to, need to).
- I (Aim) is the governed action (verb phrase), usually near D; if no D, choose the main action being described.
- B (Object) is the direct object/complement of I (what the action applies to).

- C (Condition) INCLUDES BOTH:
  (1) triggers/constraints (if/when/unless/in case/during/by [date]/prior to/as part of/under), AND
  (2) SCOPE/LEVEL/DOMAIN framing phrases (e.g., “On the national level”, “At the regional level”, “In the education sector”, “Within universities”, “In schools”).
  If there is no clear trigger/constraint, you MAY still output a scope phrase as C.

- O (Or-else) must be an explicit sanction/consequence/penalty (penalty, fine, revoked, liable, subject to sanctions, etc.). Otherwise "".

SPAN RULES:
- Each span must be an exact substring of chunk_text.
- Use MINIMAL spans (prefer 2–12 words). Avoid long sentences.
- If you cannot find a clean substring for a component, output "".

OUTPUT: JSON ONLY in exactly this shape:
{
  "chunk_id": "...",
  "doc_id": "...",
  "present": {"A":true,"D":true,"I":true,"B":true,"C":true,"O":false},
  "adibco_gold": {"A":"","D":"","I":"","B":"","C":"","O":""},
  "notes": "1–2 short sentences explaining any tricky choices"
}"""

def main():
    df = pd.read_csv(INP, sep="\t", dtype=str).fillna("")
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    done = set()
    if os.path.exists(OUT_JSONL):
        with open(OUT_JSONL, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    done.add(json.loads(line).get("chunk_id",""))
                except Exception:
                    continue

    out = open(OUT_JSONL, "a", encoding="utf-8")

    for _, r in df.iterrows():
        cid = r["chunk_id"]
        if cid in done:
            continue 

        print(f"[LLM] {len(done)+1}/{len(df)} chunk_id={cid}", flush=True)


        payload = {
            "chunk_id": cid,
            "doc_id": r.get("doc_id",""),
            "chunk_text": r.get("sentence",""),
            "baseline_adibco": {
                "A": r.get("A_attribute",""),
                "D": r.get("D_deontic",""),
                "I": r.get("I_aim",""),
                "B": r.get("B_object",""),
                "C": r.get("C_condition",""),
                "O": r.get("O_or_else","")
            }
        }

        resp = client.responses.create(
            model=MODEL,
            instructions=INSTRUCTIONS,
            input=json.dumps(payload, ensure_ascii=False),
        )

        raw = (resp.output_text or "").strip()

        try:
            obj = json.loads(raw)
        except Exception:
            obj = {
                "chunk_id": cid,
                "doc_id": r.get("doc_id",""),
                "adibco_gold": {"A":"","D":"","I":"","B":"","C":"","O":""},
                "reason": "non_json_output"
            }

        spans = obj.get("adibco_gold") or {"A":"","D":"","I":"","B":"","C":"","O":""}
        for k in ["A","D","I","B","C","O"]:
            if k not in spans:
                spans[k] = ""
            if not is_substring(spans[k], payload["chunk_text"]):
                spans[k] = ""

        obj["adibco_gold"] = spans
        out.write(json.dumps(obj, ensure_ascii=False) + "\n")
        out.flush()
        time.sleep(0.2)

    out.close()

    rows = []
    with open(OUT_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    gdf = pd.DataFrame(rows)
    for k in ["A","D","I","B","C","O"]:
        gdf[f"{k}_gold"] = gdf["adibco_gold"].apply(lambda d: (d or {}).get(k,"") if isinstance(d, dict) else "")

    gdf["statement_type_gold"] = gdf.apply(lambda x: classify_exact(
        x.get("A_gold",""), x.get("D_gold",""), x.get("I_gold",""),
        x.get("B_gold",""), x.get("C_gold",""), x.get("O_gold","")
    ), axis=1)

    merged = df.merge(
        gdf[["chunk_id","doc_id","statement_type_gold","A_gold","D_gold","I_gold","B_gold","C_gold","O_gold","reason"]],
        on=["chunk_id","doc_id"],
        how="left"
    )

    merged.to_csv(OUT_TSV, sep="\t", index=False)
    print("WROTE:", OUT_JSONL)
    print("WROTE:", OUT_TSV)
    print("Rows:", len(merged))

if __name__ == "__main__":
    main()
