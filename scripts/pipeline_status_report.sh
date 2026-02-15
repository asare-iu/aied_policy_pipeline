set -euo pipefail

echo "=============================="
echo " PIPELINE SNAPSHOT (FULL) "
echo "=============================="
echo "PWD: $(pwd)"
echo "DATE: $(date)"
echo

echo "=== GIT: STATUS ==="
git status --porcelain=v1 || true
echo

echo "=== GIT: RECENT COMMITS (last 20) ==="
git --no-pager log -20 --oneline --decorate || true
echo

echo "=== TOP-LEVEL TREE (depth 2) ==="
find . -maxdepth 2 -type d \
  ! -path "./.git*" ! -path "./.venv*" ! -path "./__pycache__*" \
  | sed "s|^\./||" | sort
echo

echo "=== CONFIG / METHODS / RESOURCES / MANIFESTS ==="
for d in config methods resources data/manifests evidence; do
  if [ -d "$d" ]; then
    echo "--- $d ---"
    find "$d" -maxdepth 2 -type f | sort | head -n 200
    echo
  fi
done

echo "=== SCRIPTS: INVENTORY (step0..step9 + helpers) ==="
ls -1 scripts | sort | head -n 400
echo

echo "=== DATA DERIVED: STEP DIRECTORY SUMMARY ==="
if [ -d data/derived ]; then
  find data/derived -maxdepth 2 -type d -name "step*" | sort
else
  echo "NO data/derived directory found"
fi
echo

echo "=== DATA DERIVED: FILE COUNTS + SIZE BY STEP (depth 1 under step*) ==="
python - <<'PY'
from pathlib import Path
import os

root = Path("data/derived")
if not root.exists():
    print("NO data/derived found")
    raise SystemExit

steps = sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith("step")])
def human(n):
    for u in ["B","KB","MB","GB","TB"]:
        if n < 1024: return f"{n:.1f}{u}"
        n /= 1024
    return f"{n:.1f}PB"

for s in steps:
    total = 0
    files = 0
    for p in s.rglob("*"):
        if p.is_file():
            files += 1
            try:
                total += p.stat().st_size
            except OSError:
                pass
    print(f"{s.name:35} files={files:7d}  size={human(total)}")
PY
echo

echo "=== KEY ARTIFACT CHECKLIST (exists + size) ==="
# Add/remove lines here as your pipeline evolves
ls -lh \
  data/derived/step0_document_inventory.csv \
  data/derived/step0_document_inventory_deduped.csv \
  data/derived/step2_chunks_raw/chunks_raw.jsonl \
  data/derived/step3_chunks_spacy/chunks_spacy.jsonl \
  data/derived/step4_5_normativity_gate/chunks_normative_primary.jsonl \
  data/derived/step5_pca_full 2>/dev/null || true
echo
ls -lh \
  data/derived/step8_igt_full/igt_statements_full.parquet \
  data/derived/step8_igt_chunks_edu/igt_statements_full.parquet \
  data/derived/step8_igt_title_edu/igt_statements_full.parquet \
  data/derived/step8_3b_umbrella_full/umbrella_o_blocks.parquet \
  data/derived/step8_3b_umbrella_full/statements_with_umbrella_o.parquet \
  2>/dev/null || true
echo

echo "=== QUICK SHAPE CHECKS (rows/cols for major tables) ==="
python - <<'"'"'PY'"'"'
import pandas as pd
from pathlib import Path

def show_csv(path):
    df = pd.read_csv(path)
    print(f"{path}: rows={len(df)} cols={len(df.columns)}")

def show_parquet(path):
    df = pd.read_parquet(path)
    print(f"{path}: rows={len(df)} cols={len(df.columns)}")

candidates = [
  ("csv", "data/derived/step0_document_inventory.csv"),
  ("csv", "data/derived/step0_document_inventory_deduped.csv"),
]

for kind, p in candidates:
    if Path(p).exists():
        try:
            show_csv(p)
        except Exception as e:
            print(f"{p}: ERROR {e}")

parq = [
  "data/derived/step8_igt_full/igt_statements_full.parquet",
  "data/derived/step8_igt_chunks_edu/igt_statements_full.parquet",
  "data/derived/step8_igt_title_edu/igt_statements_full.parquet",
  "data/derived/step8_3b_umbrella_full/statements_with_umbrella_o.parquet",
  "data/derived/step8_3b_umbrella_full/umbrella_o_blocks.parquet",
]

for p in parq:
    if Path(p).exists():
        try:
            show_parquet(p)
        except Exception as e:
            print(f"{p}: ERROR {e}")
PY
echo

echo "=== OPTIONAL: RUFF SUMMARY (scripts only; non-fatal) ==="
ruff check scripts 2>/dev/null || true
echo

echo "=============================="
echo " END SNAPSHOT "
echo "=============================="

