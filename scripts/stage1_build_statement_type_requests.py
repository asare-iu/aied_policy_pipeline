import json, csv, os, sys
from pathlib import Path

def main():
    stage1_dir = Path(os.environ["STAGE1"])
    inp = stage1_dir / "input_baseline.tsv"
    task_md = (stage1_dir / "stage1_task.md").read_text(encoding="utf-8")
    schema = json.loads((stage1_dir / "stage1_schema.json").read_text(encoding="utf-8"))

    run = stage1_dir / f"run_{os.popen('date +%Y%m%d_%H%M%S').read().strip()}"
    run.mkdir(parents=True, exist_ok=True)

    out_jsonl = run / "requests.jsonl"
    manifest = run / "run_manifest.json"

    # Parse TSV with possible multiline quoted fields
    with inp.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)

    # Find fields
    # chunk text column might be 'sentence' even though it actually contains chunk text in your pipeline
    # We accept either 'sentence' or 'chunk_text' if present.
    def pick(colnames, candidates):
        for c in candidates:
            if c in colnames:
                return c
        return None

    col_chunk_id = pick(reader.fieldnames, ["chunk_id"])
    col_text     = pick(reader.fieldnames, ["chunk_text", "sentence"])
    col_label    = pick(reader.fieldnames, ["igt_statement_type_guess", "baseline_statement_type", "statement_type"])
    if col_chunk_id is None or col_text is None or col_label is None:
        raise SystemExit(f"Missing required columns. Have={reader.fieldnames}. Need chunk_id + (chunk_text|sentence) + label")

    # Collect ADIBCO fields if present
    adibco_cols = [c for c in reader.fieldnames if c.startswith(("A_","D_","I_","B_","C_","O_"))]

    with out_jsonl.open("w", encoding="utf-8") as w:
        for r in rows:
            payload = {
                "chunk_id": r.get(col_chunk_id, ""),
                "baseline_statement_type": (r.get(col_label, "") or "").strip().lower(),
                "chunk_text": r.get(col_text, "") or "",
                "adibco": {c: r.get(c, "") for c in adibco_cols}
            }

            req = {
                "custom_id": payload["chunk_id"],
                "instructions_md": task_md,
                "schema": schema,
                "payload": payload
            }
            w.write(json.dumps(req, ensure_ascii=False) + "\n")

    manifest.write_text(json.dumps({
        "input_tsv": str(inp),
        "requests_jsonl": str(out_jsonl),
        "rows": len(rows),
        "chunk_id_col": col_chunk_id,
        "text_col": col_text,
        "label_col": col_label,
        "adibco_cols": adibco_cols
    }, indent=2), encoding="utf-8")

    print("WROTE:", out_jsonl)
    print("WROTE:", manifest)

if __name__ == "__main__":
    if "STAGE1" not in os.environ:
        print("ERROR: export STAGE1=... first", file=sys.stderr)
        sys.exit(2)
    main()
