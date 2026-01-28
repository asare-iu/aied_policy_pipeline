import os, json, csv, time
from pathlib import Path

def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line:
                yield json.loads(line)

def write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def flatten_to_tsv(out_tsv: Path, results):
    # Collect union of keys from result['output'] objects for stable TSV
    base_cols = [
        "chunk_id","baseline_statement_type","final_statement_type",
        "recoverable_from_other","other_reason_bucket","confidence",
        "evidence_span","missing_elements","notes_for_humans","recommended_pipeline_change"
    ]
    # Write TSV
    with out_tsv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(base_cols)
        for r in results:
            o = r.get("output") or {}
            row = []
            for c in base_cols:
                v = o.get(c, "")
                if isinstance(v, list):
                    v = json.dumps(v, ensure_ascii=False)
                elif v is None:
                    v = ""
                row.append(v)
            w.writerow(row)

def main():
    run_dir = Path(os.environ.get("RUN_DIR","")).expanduser()
    if not run_dir or not run_dir.exists():
        raise SystemExit("Set RUN_DIR to your run folder, e.g. export RUN_DIR=.../run_YYYYMMDD_HHMMSS")

    req_path = run_dir / "requests.jsonl"
    if not req_path.exists():
        raise SystemExit(f"Missing {req_path}")

    # Load schema + instructions from the request objects (authoritative per request)
    reqs = list(load_jsonl(req_path))
    if not reqs:
        raise SystemExit("requests.jsonl is empty")

    model = os.environ.get("OPENAI_MODEL") or os.environ.get("MODEL") or "gpt-4o-mini"
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is not set")

    responses_path   = run_dir / "responses.jsonl"
    diagnostics_path = run_dir / "diagnostics.json"
    preds_path       = run_dir / "predictions.tsv"

    # Try OpenAI python client (preferred)
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
    except Exception as e:
        raise SystemExit(f"Could not import/use OpenAI client. Install openai or fix env. Error: {e}")

    results = []
    errors = []

    # Small helper: robustly extract JSON object from a string if model returns text
    def try_parse_json(s: str):
        try:
            return json.loads(s)
        except Exception:
            # attempt to find first {...} block
            start = s.find("{")
            end = s.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(s[start:end+1])
                except Exception:
                    return None
            return None

    for i, req in enumerate(reqs, start=1):
        custom_id = req.get("custom_id", f"row{i}")
        instructions_md = req.get("instructions_md","")
        schema = req.get("schema", None)
        payload = req.get("payload", {})

        # Build a single user message that includes everything needed
        # (We keep the schema enforcement via response_format when possible.)
        user_content = json.dumps(payload, ensure_ascii=False)

        t0 = time.time()
        try:
            # Prefer Responses API with JSON Schema if available in this client/version
            out_obj = None
            used = None

            # Attempt responses.create with json_schema response_format
            try:
                resp = client.responses.create(
                    model=model,
                    input=[
                        {"role":"system","content":instructions_md},
                        {"role":"user","content":user_content}
                    ],
                    response_format={
                        "type":"json_schema",
                        "json_schema":{
                            "name":"stage1_statement_type",
                            "schema": schema if schema else {"type":"object"},
                            "strict": True
                        }
                    }
                )
                # resp.output_text may contain JSON, but the SDK also exposes parsed output in some versions
                text = getattr(resp, "output_text", None)
                if text:
                    out_obj = try_parse_json(text)
                used = "responses.json_schema"
            except Exception:
                # Fallback: chat.completions with strong instruction "ONLY JSON"
                resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role":"system","content":instructions_md},
                        {"role":"user","content":user_content + "\n\nReturn ONLY valid JSON per the schema."}
                    ],
                    temperature=0
                )
                text = resp.choices[0].message.content or ""
                out_obj = try_parse_json(text)
                used = "chat.fallback"

            if not isinstance(out_obj, dict):
                raise ValueError("Model did not return a JSON object")

            results.append({
                "custom_id": custom_id,
                "model": model,
                "api_mode": used,
                "latency_s": round(time.time()-t0, 3),
                "output": out_obj
            })

        except Exception as e:
            errors.append({
                "custom_id": custom_id,
                "model": model,
                "error": str(e),
                "latency_s": round(time.time()-t0, 3)
            })

    # Write outputs
    write_jsonl(responses_path, results)

    # Diagnostics summary
    diag = {
        "run_dir": str(run_dir),
        "requests": len(reqs),
        "success": len(results),
        "errors": len(errors),
        "error_examples": errors[:5],
        "model": model
    }
    diagnostics_path.write_text(json.dumps(diag, indent=2), encoding="utf-8")

    # Flatten TSV for quick diffing
    flatten_to_tsv(preds_path, results)

    print("WROTE:", responses_path)
    print("WROTE:", diagnostics_path)
    print("WROTE:", preds_path)
    print("SUMMARY:", diag)

if __name__ == "__main__":
    main()
