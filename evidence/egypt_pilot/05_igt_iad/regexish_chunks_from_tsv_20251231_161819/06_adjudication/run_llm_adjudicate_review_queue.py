import os, json, time
import pandas as pd
from openai import OpenAI

ADJDIR = os.environ["ADJDIR"]
INP = os.path.join(ADJDIR, "input_chunks.tsv")
OUT_JSONL = os.path.join(ADJDIR, "llm_adjudicated_spans.jsonl")
OUT_TSV   = os.path.join(ADJDIR, "llm_adjudicated_queue.tsv")

# Choose a model via env var if you want:
# export OPENAI_MODEL="gpt-5-mini" (or whatever your account supports)
MODEL = os.environ.get("OPENAI_MODEL", "gpt-5-mini")

# --- Deterministic statement-type mapping (your strict logic) ---
def classify(A, D, I, B, C, O):
    hasA = A != ""
    hasD = D != ""
    hasI = I != ""
    hasB = B != ""
    hasC = C != ""
    hasO = O != ""
    # precedence resolves overlaps deterministically
    if hasA and hasD and hasI and hasB and hasC and hasO:
        return "rule"        # ADIBCO
    if hasA and hasD and hasI and hasC:
        return "norm"        # ADIC (B and O may or may not be present; rule already caught above)
    if hasA and hasI and hasC:
        return "strategy"    # AIC
    if hasI and hasC:
        return "principle"   # IC
    return "other"

def is_substring(span: str, text: str) -> bool:
    span = "" if span is None else str(span)
    return (span == "") or (span in text)
# --- JSON Schema for Structured Outputs ---
SCHEMA = {
  "name": "adjudication",
  "schema": {
    "type": "object",
    "additionalProperties": False,
    "properties": {
      "chunk_id": {"type": "string"},
      "doc_id": {"type": "string"},
      "adibco_gold": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
          "A": {"type": "string"},
          "D": {"type": "string"},
          "I": {"type": "string"},
          "B": {"type": "string"},
          "C": {"type": "string"},
          "O": {"type": "string"}
        },
        "required": ["A","D","I","B","C","O"]
      },
      "reason": {"type": "string"},
      "fix_note": {"type": "string"}
    },
    "required": ["chunk_id","doc_id","adibco_gold","reason","fix_note"]
  }
}

INSTRUCTIONS = """
You are adjudicating ONE policy CHUNK (education-relevant).

Goal:
Return corrected ADIBCO spans as EXACT VERBATIM SUBSTRINGS of the chunk text.
No inference. Do not invent words.

ADIBCO definitions:
A Attribute (actor), D Deontic, I Aim, B Object, C Condition, O Or-else (sanction).

Rules:
- Spans MUST appear exactly in the chunk text.
- Use minimal spans that support the component.
- If absent, output "".
- Output must conform to the provided JSON schema.

You will be given:
- chunk_id, doc_id
- chunk_text
- baseline ADIBCO spans (may be wrong)
""".strip()

def main():
    df = pd.read_csv(INP, sep="\t", dtype=str).fillna("")
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    # Resume support
    done = set()
    if os.path.exists(OUT_JSONL):
        with open(OUT_JSONL, "r", encoding="utf-8") as f:
            for line in f:
                line=line.strip()
                if not line: 
                    continue
                try:
                    done.add(json.loads(line).get("chunk_id",""))
                except Exception:
                    continue

    out = open(OUT_JSONL, "a", encoding="utf-8")

    for _, r in df.iterrows():
        chunk_id = r["chunk_id"]
        if chunk_id in done:
            continue

        chunk_text = r.get("sentence","")
        payload = {
            "chunk_id": chunk_id,
            "doc_id": r.get("doc_id",""),
            "chunk_text": chunk_text,
            "baseline_adibco": {
                "A_attribute": r.get("A_attribute",""),
                "D_deontic": r.get("D_deontic",""),
                "I_aim": r.get("I_aim",""),
                "B_object": r.get("B_object",""),
                "C_condition": r.get("C_condition",""),
                "O_or_else": r.get("O_or_else","")
            }
        }

cat > "$ADJDIR/run_llm_adjudicate_review_queue.py" <<'PY'
import os, json, time
import pandas as pd
from openai import OpenAI

ADJDIR = os.environ["ADJDIR"]
INP = os.path.join(ADJDIR, "input_chunks.tsv")
OUT_JSONL = os.path.join(ADJDIR, "llm_adjudicated_spans.jsonl")
OUT_TSV   = os.path.join(ADJDIR, "llm_adjudicated_queue.tsv")

# Choose a model via env var if you want:
# export OPENAI_MODEL="gpt-5-mini" (or whatever your account supports)
MODEL = os.environ.get("OPENAI_MODEL", "gpt-5-mini")

# --- Deterministic statement-type mapping (your strict logic) ---
def classify(A, D, I, B, C, O):
    hasA = A != ""
    hasD = D != ""
    hasI = I != ""
    hasB = B != ""
    hasC = C != ""
    hasO = O != ""
    # precedence resolves overlaps deterministically
    if hasA and hasD and hasI and hasB and hasC and hasO:
        return "rule"        # ADIBCO
    if hasA and hasD and hasI and hasC:
        return "norm"        # ADIC (B and O may or may not be present; rule already caught above)
    if hasA and hasI and hasC:
        return "strategy"    # AIC
    if hasI and hasC:
        return "principle"   # IC
    return "other"

def is_substring(span: str, text: str) -> bool:
    span = "" if span is None else str(span)
    return (span == "") or (span in text)
# --- JSON Schema for Structured Outputs ---
SCHEMA = {
  "name": "adjudication",
  "schema": {
    "type": "object",
    "additionalProperties": False,
    "properties": {
      "chunk_id": {"type": "string"},
      "doc_id": {"type": "string"},
      "adibco_gold": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
          "A": {"type": "string"},
          "D": {"type": "string"},
          "I": {"type": "string"},
          "B": {"type": "string"},
          "C": {"type": "string"},
          "O": {"type": "string"}
        },
        "required": ["A","D","I","B","C","O"]
      },
      "reason": {"type": "string"},
      "fix_note": {"type": "string"}
    },
    "required": ["chunk_id","doc_id","adibco_gold","reason","fix_note"]
  }
}

INSTRUCTIONS = """
You are adjudicating ONE policy CHUNK (education-relevant).

Goal:
Return corrected ADIBCO spans as EXACT VERBATIM SUBSTRINGS of the chunk text.
No inference. Do not invent words.

ADIBCO definitions:
A Attribute (actor), D Deontic, I Aim, B Object, C Condition, O Or-else (sanction).

Rules:
- Spans MUST appear exactly in the chunk text.
- Use minimal spans that support the component.
- If absent, output "".
- Output must conform to the provided JSON schema.

You will be given:
- chunk_id, doc_id
- chunk_text
- baseline ADIBCO spans (may be wrong)
""".strip()

def main():
    df = pd.read_csv(INP, sep="\t", dtype=str).fillna("")
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    # Resume support
    done = set()
    if os.path.exists(OUT_JSONL):
        with open(OUT_JSONL, "r", encoding="utf-8") as f:
            for line in f:
                line=line.strip()
                if not line: 
                    continue
                try:
                    done.add(json.loads(line).get("chunk_id",""))
                except Exception:
                    continue

    out = open(OUT_JSONL, "a", encoding="utf-8")

    for _, r in df.iterrows():
        chunk_id = r["chunk_id"]
        if chunk_id in done:
            continue

        chunk_text = r.get("sentence","")
        payload = {
            "chunk_id": chunk_id,
            "doc_id": r.get("doc_id",""),
            "chunk_text": chunk_text,
            "baseline_adibco": {
                "A_attribute": r.get("A_attribute",""),
                "D_deontic": r.get("D_deontic",""),
                "I_aim": r.get("I_aim",""),
                "B_object": r.get("B_object",""),
                "C_condition": r.get("C_condition",""),
                "O_or_else": r.get("O_or_else","")
            }
        }




# --- JSON Schema for Structured Outputs ---
SCHEMA = {
  "name": "adjudication",
  "schema": {
    "type": "object",
    "additionalProperties": False,
    "properties": {
      "chunk_id": {"type": "string"},
      "doc_id": {"type": "string"},
      "adibco_gold": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
          "A": {"type": "string"},
          "D": {"type": "string"},
          "I": {"type": "string"},
          "B": {"type": "string"},
          "C": {"type": "string"},
          "O": {"type": "string"}
        },
        "required": ["A","D","I","B","C","O"]
      },
      "reason": {"type": "string"},
      "fix_note": {"type": "string"}
    },
    "required": ["chunk_id","doc_id","adibco_gold","reason","fix_note"]
  }
}

INSTRUCTIONS = """
You are adjudicating ONE policy CHUNK (education-relevant).

Goal:
Return corrected ADIBCO spans as EXACT VERBATIM SUBSTRINGS of the chunk text.
No inference. Do not invent words.

ADIBCO definitions:
A Attribute (actor), D Deontic, I Aim, B Object, C Condition, O Or-else (sanction).

Rules:
- Spans MUST appear exactly in the chunk text.
- Use minimal spans that support the component.
- If absent, output "".
- Output must conform to the provided JSON schema.

You will be given:
- chunk_id, doc_id
- chunk_text
- baseline ADIBCO spans (may be wrong)
""".strip()

def main():
    df = pd.read_csv(INP, sep="\t", dtype=str).fillna("")
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    # Resume support
    done = set()
    if os.path.exists(OUT_JSONL):
        with open(OUT_JSONL, "r", encoding="utf-8") as f:
            for line in f:
                line=line.strip()
                if not line: 
                    continue
                try:
                    done.add(json.loads(line).get("chunk_id",""))
                except Exception:
                    continue

    out = open(OUT_JSONL, "a", encoding="utf-8")

    for _, r in df.iterrows():
        chunk_id = r["chunk_id"]
        if chunk_id in done:
            continue

        chunk_text = r.get("sentence","")
        payload = {
            "chunk_id": chunk_id,
            "doc_id": r.get("doc_id",""),
            "chunk_text": chunk_text,
            "baseline_adibco": {
                "A_attribute": r.get("A_attribute",""),
                "D_deontic": r.get("D_deontic",""),
                "I_aim": r.get("I_aim",""),
                "B_object": r.get("B_object",""),
                "C_condition": r.get("C_condition",""),
                "O_or_else": r.get("O_or_else","")
            }
        }
cat > "$ADJDIR/run_llm_adjudicate_review_queue.py" <<'PY'
import os, json, time
import pandas as pd
from openai import OpenAI

ADJDIR = os.environ["ADJDIR"]
INP = os.path.join(ADJDIR, "input_chunks.tsv")
OUT_JSONL = os.path.join(ADJDIR, "llm_adjudicated_spans.jsonl")
OUT_TSV   = os.path.join(ADJDIR, "llm_adjudicated_queue.tsv")

# Choose a model via env var if you want:
# export OPENAI_MODEL="gpt-5-mini" (or whatever your account supports)
MODEL = os.environ.get("OPENAI_MODEL", "gpt-5-mini")

# --- Deterministic statement-type mapping (your strict logic) ---
def classify(A, D, I, B, C, O):
    hasA = A != ""
    hasD = D != ""
    hasI = I != ""
    hasB = B != ""
    hasC = C != ""
    hasO = O != ""
    # precedence resolves overlaps deterministically
    if hasA and hasD and hasI and hasB and hasC and hasO:
        return "rule"        # ADIBCO
    if hasA and hasD and hasI and hasC:
        return "norm"        # ADIC (B and O may or may not be present; rule already caught above)
    if hasA and hasI and hasC:
        return "strategy"    # AIC
    if hasI and hasC:
        return "principle"   # IC
    return "other"

def is_substring(span: str, text: str) -> bool:
    span = "" if span is None else str(span)
    return (span == "") or (span in text)

# --- JSON Schema for Structured Outputs ---
SCHEMA = {
  "name": "adjudication",
  "schema": {
    "type": "object",
    "additionalProperties": False,
    "properties": {
      "chunk_id": {"type": "string"},
      "doc_id": {"type": "string"},
      "adibco_gold": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
          "A": {"type": "string"},
          "D": {"type": "string"},
          "I": {"type": "string"},
          "B": {"type": "string"},
          "C": {"type": "string"},
          "O": {"type": "string"}
        },
        "required": ["A","D","I","B","C","O"]
      },
      "reason": {"type": "string"},
      "fix_note": {"type": "string"}
    },
    "required": ["chunk_id","doc_id","adibco_gold","reason","fix_note"]
  }
}

INSTRUCTIONS = """
You are adjudicating ONE policy CHUNK (education-relevant).

Goal:
Return corrected ADIBCO spans as EXACT VERBATIM SUBSTRINGS of the chunk text.
No inference. Do not invent words.

ADIBCO definitions:
A Attribute (actor), D Deontic, I Aim, B Object, C Condition, O Or-else (sanction).

Rules:
- Spans MUST appear exactly in the chunk text.
- Use minimal spans that support the component.
- If absent, output "".
- Output must conform to the provided JSON schema.

You will be given:
- chunk_id, doc_id
- chunk_text
- baseline ADIBCO spans (may be wrong)
""".strip()

def main():
    df = pd.read_csv(INP, sep="\t", dtype=str).fillna("")
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    # Resume support
    done = set()
    if os.path.exists(OUT_JSONL):
        with open(OUT_JSONL, "r", encoding="utf-8") as f:
            for line in f:
                line=line.strip()
                if not line: 
                    continue
                try:
                    done.add(json.loads(line).get("chunk_id",""))
                except Exception:
                    continue

    out = open(OUT_JSONL, "a", encoding="utf-8")

    for _, r in df.iterrows():
        chunk_id = r["chunk_id"]
        if chunk_id in done:
            continue

        chunk_text = r.get("sentence","")
        payload = {
            "chunk_id": chunk_id,
            "doc_id": r.get("doc_id",""),
            "chunk_text": chunk_text,
            "baseline_adibco": {
                "A_attribute": r.get("A_attribute",""),
                "D_deontic": r.get("D_deontic",""),
                "I_aim": r.get("I_aim",""),
                "B_object": r.get("B_object",""),
                "C_condition": r.get("C_condition",""),
                "O_or_else": r.get("O_or_else","")
            }
        }

        resp = client.responses.create(
            model=MODEL,
            instructions=INSTRUCTIONS,
            input=json.dumps(payload, ensure_ascii=False),
            # Structured outputs
            response_format={
                "type": "json_schema",
                "json_schema": SCHEMA
            }
        )

        raw = (resp.output_text or "").strip()
        try:
            obj = json.loads(raw)
        except Exception:
            obj = {
                "chunk_id": chunk_id,
                "doc_id": r.get("doc_id",""),
                "adibco_gold": {"A":"","D":"","I":"","B":"","C":"","O":""},
                "reason": "non_json_output",
                "fix_note": raw[:5000]
            }

        # Enforce substring constraint (clear any invalid spans)
        ag = obj.get("adibco_gold") or {}
        for k in ["A","D","I","B","C","O"]:
            span = ag.get(k, "")
            if not is_substring(span, chunk_text):
                ag[k] = ""
                obj["fix_note"] = (obj.get("fix_note","") + f" | cleared non-substring {k}").strip(" |")
        obj["adibco_gold"] = ag

        out.write(json.dumps(obj, ensure_ascii=False) + "\n")
        out.flush()
        time.sleep(0.2)

    out.close()

    # Build TSV with deterministic statement type
    rows = []
    with open(OUT_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line: 
                continue
            rows.append(json.loads(line))

    gdf = pd.DataFrame(rows)
    for k in ["A","D","I","B","C","O"]:
        gdf[f"{k}_gold"] = gdf["adibco_gold"].apply(lambda d: (d or {}).get(k,"") if isinstance(d, dict) else "")

    gdf["statement_type_gold"] = gdf.apply(lambda x: classify(
        x.get("A_gold",""), x.get("D_gold",""), x.get("I_gold",""),
        x.get("B_gold",""), x.get("C_gold",""), x.get("O_gold","")
    ), axis=1)

    keep = ["chunk_id","doc_id","statement_type_gold","A_gold","D_gold","I_gold","B_gold","C_gold","O_gold","reason","fix_note"]
    merged = df.merge(gdf[keep], on=["chunk_id","doc_id"], how="left")
    merged.to_csv(OUT_TSV, sep="\t", index=False)

    print("WROTE:", OUT_JSONL)
    print("WROTE:", OUT_TSV)
    print("Rows:", len(merged))
    print(merged[["chunk_id","statement_type_gold","A_gold","D_gold","I_gold","B_gold","C_gold","O_gold"]].head(10).to_string(index=False))

if __name__ == "__main__":
    main()
