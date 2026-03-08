#!/usr/bin/env bash
set -e

EXPORT_NAME="dissertation_review_export"
DATE=$(date +"%Y%m%d_%H%M%S")
ARCHIVE="${EXPORT_NAME}_${DATE}.tar.gz"

echo "Creating dissertation review export..."

tar --exclude='.venv' \
    --exclude='.git' \
    --exclude='.ruff_cache' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='*.pyo' \
    --exclude='*.pyd' \
    --exclude='export.log' \
    --exclude='egypt_pilot_export_*.tar.gz' \
    --exclude='data/derived/step2_chunks_raw' \
    --exclude='data/derived/step3_chunks_spacy' \
    --exclude='data/derived/step4_chunks_tagged' \
    --exclude='data/derived/step4_5_normativity_gate' \
    --exclude='data/derived/step5_models_full_40pc' \
    --exclude='data/derived/step5_models_full_artifact_stripped_40pc' \
    -czf "$ARCHIVE" .

echo "Export created:"
echo "$ARCHIVE"

