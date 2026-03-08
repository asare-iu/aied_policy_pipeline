#!/usr/bin/env bash
set -euo pipefail

# Make a light review bundle for ChatGPT inspection:
# - code/config/lexicons/manifests
# - key small derived outputs (PCA interpretations, metadata, tables)
# - small gzipped CSV samples from big parquet tables (schema + content)
#
# Output: review_bundle_light_YYYYmmdd_HHMMSS.tgz in repo root

REPO_ROOT="$(pwd)"
OUTROOT="${REPO_ROOT}/review_bundle"

# ---- helpers ----
have() { command -v "$1" >/dev/null 2>&1; }

copytree() {
  # copytree SRC DEST  (best effort)
  local src="$1" dest="$2"
  if [ -d "$src" ]; then
    mkdir -p "$dest"
    if have rsync; then
      rsync -a "$src"/ "$dest"/ 2>/dev/null || true
    else
      cp -a "$src"/. "$dest"/ 2>/dev/null || true
    fi
  fi
}

copyfile() {
  # copyfile SRC DESTDIR (best effort)
  local src="$1" destdir="$2"
  if [ -f "$src" ]; then
    mkdir -p "$destdir"
    cp -a "$src" "$destdir"/ 2>/dev/null || true
  fi
}

# ---- 0) clean workspace ----
rm -rf "$OUTROOT"
mkdir -p "$OUTROOT"/{meta,config,resources,methods,manifests,scripts,derived,analysis,figures,evidence,samples}

# ---- 1) environment + git metadata ----
{
  echo "DATE_UTC: $(date -u)"
  echo "PWD:      ${REPO_ROOT}"
  echo
  echo "GIT_HEAD: $(git rev-parse HEAD 2>/dev/null || true)"
  echo
  echo "GIT_STATUS_PORCELAIN:"
  git status --porcelain 2>/dev/null || true
  echo
  echo "UNAME: $(uname -a)"
  echo
  echo "PYTHON: $(python -V 2>&1 || true)"
} > "$OUTROOT/meta/environment.txt"

pip freeze > "$OUTROOT/meta/pip_freeze.txt" 2>/dev/null || true

# ---- 2) copy method + metadata (safe/inspectable) ----
copytree "config"    "$OUTROOT/config"
copytree "resources" "$OUTROOT/resources"
copytree "methods"   "$OUTROOT/methods"
copytree "data/manifests" "$OUTROOT/manifests"

# scripts (exclude caches/backups)
mkdir -p "$OUTROOT/scripts"
if have rsync; then
  rsync -a \
    --exclude='__pycache__' \
    --exclude='*.save' \
    --exclude='*.save.*' \
    --exclude='*~' \
    scripts/ "$OUTROOT/scripts/" 2>/dev/null || true
else
  # fallback
  find scripts -type f \
    ! -name '*.save' ! -name '*.save.*' ! -name '*~' \
    -print0 2>/dev/null | while IFS= read -r -d '' f; do
      d="$OUTROOT/scripts/$(dirname "${f#scripts/}")"
      mkdir -p "$d"
      cp -a "$f" "$d/" 2>/dev/null || true
    done
fi

# ---- 3) key derived outputs (small/meaningful) ----
copyfile "data/derived/step0_document_inventory.csv" "$OUTROOT/derived"
copyfile "data/derived/step0_document_inventory_deduped.csv" "$OUTROOT/derived"

# PCA human interpretation artifacts
copytree "data/derived/step5_5_pca_interpretation" "$OUTROOT/derived/step5_5_pca_interpretation"

# PCA model metadata + top terms for key model dirs (copy only small files)
for d in \
  data/derived/step5_models_full_40pc \
  data/derived/step5_models_full_artifact_stripped_40pc \
  data/derived/step5_models_title_edu_40pc \
  data/derived/step5_models_edu_embedded_40pc
do
  if [ -d "$d" ]; then
    bn="$(basename "$d")"
    mkdir -p "$OUTROOT/derived/$bn"
    copyfile "$d/models_metadata.json"     "$OUTROOT/derived/$bn"
    copyfile "$d/_runtime_params.json"     "$OUTROOT/derived/$bn"
    copyfile "$d/explained_variance.csv"   "$OUTROOT/derived/$bn"
    copyfile "$d/top_terms.csv"            "$OUTROOT/derived/$bn"
    copyfile "$d/top_terms_long.csv"       "$OUTROOT/derived/$bn"
    copyfile "$d/pc_interpretations_deep.csv" "$OUTROOT/derived/$bn"
    copyfile "$d/row_alignment.csv"        "$OUTROOT/derived/$bn"
  fi
done

# Step 8 analysis outputs (usually small + highly relevant)
copytree "data/derived/step8_analysis" "$OUTROOT/analysis"

# evidence folder (exclude big/binary artifacts)
if [ -d "evidence" ]; then
  mkdir -p "$OUTROOT/evidence"
  if have rsync; then
    rsync -a \
      --exclude='*.parquet' --exclude='*.joblib' --exclude='*.npz' --exclude='*.jsonl' --exclude='*.tar.gz' \
      evidence/ "$OUTROOT/evidence/" 2>/dev/null || true
  else
    # fallback: copy only small text-ish files
    find evidence -type f \
      ! -name '*.parquet' ! -name '*.joblib' ! -name '*.npz' ! -name '*.jsonl' ! -name '*.tar.gz' \
      -print0 2>/dev/null | while IFS= read -r -d '' f; do
        d="$OUTROOT/evidence/$(dirname "${f#evidence/}")"
        mkdir -p "$d"
        cp -a "$f" "$d/" 2>/dev/null || true
      done
  fi
fi

# ---- 4) sample big parquet tables into compact CSVs (for inspection) ----
python - <<'PY'
import pandas as pd
from pathlib import Path

out = Path("review_bundle/samples")
out.mkdir(parents=True, exist_ok=True)

targets = [
    ("igt_full", "data/derived/step8_igt_full/igt_statements_full.parquet"),
    ("igt_chunks_edu", "data/derived/step8_igt_chunks_edu/igt_statements_full.parquet"),
    ("igt_title_edu", "data/derived/step8_igt_title_edu/igt_statements_full.parquet"),
    ("umbrella_statements", "data/derived/step8_3b_umbrella_full/statements_with_umbrella_o.parquet"),
    ("umbrella_blocks", "data/derived/step8_3b_umbrella_full/umbrella_o_blocks.parquet"),
]

def sample_parquet(tag, path, n=2000):
    p = Path(path)
    if not p.exists():
        return
    df = pd.read_parquet(p)

    # schema
    schema_path = out / f"{tag}_schema.txt"
    schema_path.write_text("\n".join([f"{c}\t{df[c].dtype}" for c in df.columns]))

    # deterministic sample
    s = df.sample(min(n, len(df)), random_state=42)
    out_path = out / f"{tag}_sample_{min(n,len(df))}.csv.gz"
    import csv
    s.to_csv(
        out_path,
        index=False,
        compression="gzip",
        escapechar="\\",
        quoting=csv.QUOTE_MINIMAL,
    )


for tag, path in targets:
    sample_parquet(tag, path)

print("Wrote parquet samples + schema to:", out)
PY

# ---- 5) pack it ----
stamp="$(date -u +%Y%m%d_%H%M%S)"
tar -czf "review_bundle_light_${stamp}.tgz" review_bundle

echo
ls -lh "review_bundle_light_${stamp}.tgz"
echo
echo "Upload this file here:"
echo "  review_bundle_light_${stamp}.tgz"
