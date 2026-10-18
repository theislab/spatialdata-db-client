#!/usr/bin/env bash
# Re-execute the guide notebooks against the live catalog (RTD does not execute them).
# Run from the repo root:  bash docs/execute_notebooks.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export SDDB_CACHE_DIR="${SDDB_CACHE_DIR:-$(mktemp -d)}"
echo "cache: $SDDB_CACHE_DIR" >&2
uv run --group doc jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=1800 docs/guides/*.ipynb
