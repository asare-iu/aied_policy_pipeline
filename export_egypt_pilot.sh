#!/usr/bin/env bash
set -euo pipefail

TS=$(date +"%Y%m%d_%H%M%S")
OUT="egypt_pilot_export_${TS}.tar.gz"

echo "Creating $OUT in $(pwd) at $(date)"

tar --exclude='.venv' \
    --exclude='.git' \
    --exclude='data/raw' \
    -czf "$OUT" \
    scripts methods evidence data docs logs .env .gitignore 2>/dev/null || true

ls -lah "$OUT"
echo "DONE $(date)"
