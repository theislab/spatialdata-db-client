#!/usr/bin/env bash
# Execute one primer notebook in its own environment and check its size. Run on a GPU node from the repo root:
#   bash docs/primers/run_primer.sh <slug>
# The primer lives in docs/primers/<slug>/ with primer.ipynb, pixi.toml and pixi.lock. The executed notebook,
# outputs included, must stay at or below 20 MB; above that the run fails and the notebook is left unchanged.
set -euo pipefail
slug="${1:?usage: run_primer.sh <slug>}"
dir="$(cd "$(dirname "$0")" && pwd)/$slug"
[ -f "$dir/primer.ipynb" ] && [ -f "$dir/pixi.toml" ] && [ -f "$dir/pixi.lock" ] || { echo "$dir needs primer.ipynb, pixi.toml and pixi.lock" >&2; exit 1; }
export SDDB_CACHE_DIR="${SDDB_CACHE_DIR:-$(mktemp -d)}"
out="$(mktemp -d)/primer.ipynb"
(cd "$dir" && pixi run --frozen jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 --output "$out" primer.ipynb)
size=$(wc -c < "$out")
max=$((20 * 1024 * 1024))
if [ "$size" -gt "$max" ]; then
  echo "executed notebook is $((size / 1024 / 1024)) MB, above the 20 MB cap: $out (not copied)" >&2
  exit 1
fi
mv "$out" "$dir/primer.ipynb"
echo "$slug: $((size / 1024)) kB" >&2
