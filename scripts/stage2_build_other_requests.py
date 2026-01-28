import os, csv, json
from pathlib import Path
from datetime import datetime

STAGE2 = Path(os.environ["STAGE2"])
QUEUE  = Path(os.environ["STAGE1_FINAL"]) / "stage2_queue_other.tsv"
BASELINE = Path(os.environ["BASELINE"])

run = STAGE2 / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
run.mkdir(parents=True, exist_ok=True)

# Load task + schema
task = (STAGE2 / "stage2_task.md").read_text()
schema = json.loads((STAGE2 / "stage2_schema.json").read_text())

# Map chunk_id → chunk_text
chunk_text = {}
with BASELINE.open() as f:
    rows = list(csv.DictReader(f, delimiter="\t"))
text_col = "sentence"
for r in rows:
    chunk_text[r["chunk_id"]] = r.get(text_col, "")

out = run / "requests.jsonl"

with QUEUE.open() as f, out.open("w") as w:
    for r in csv.DictReader(f, delimiter="\t"):
        cid = r["chunk_id"]
        w.write(json.dumps({
            "custom_id": cid,
            "schema": schema,
            "task_md": task,
            "input": {
                "chunk_id": cid,
                "chunk_text": chunk_text.get(cid, ""),
                "stage1": r
            }
        }) + "\n")

(run / "run_manifest.json").write_text(json.dumps({
    "run_dir": str(run),
    "n_requests": sum(1 for _ in open(QUEUE)) - 1
}, indent=2))

print("WROTE:", out)
print("RUN_DIR:", run)
