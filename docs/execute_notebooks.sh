#!/usr/bin/env bash
# Re-execute the tutorial and use-case notebooks against the dated catalog (the docs build does not run them).
# Run from the repo root on a machine with network access and about 2 GB of free disk:
#   bash docs/execute_notebooks.sh
# 01_find_and_filter downloads the gene index (about 900 MB) and needs about 40 GB of RAM to query it, so run
# this on an HPC node rather than a laptop. The Visium atlas is a GPU run with its own environment and is not
# executed here (see docs/use_cases/visium_atlas/pixi.toml).
set -euo pipefail
cd "$(dirname "$0")/.."
export SDDB_CACHE_DIR="${SDDB_CACHE_DIR:-$(mktemp -d)}"
echo "cache: $SDDB_CACHE_DIR" >&2
uv run --group doc --extra viz jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=1800 docs/tutorials/*.ipynb docs/use_cases/training_cohort.ipynb
