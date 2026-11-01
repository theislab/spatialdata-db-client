# Sourced by the sbatch scripts. WORK holds the code checkout, the env and the outputs; caches go to the
# node's local disk so no HPC path reaches a notebook.
set -euo pipefail
WORK=${WORK:?set WORK to the primers workspace (code/, bundles/, logs/)}
ENV=$WORK/code/primers/env
# ~/lustre links to group storage, so the shared pixi cache stays off the small home directory
export PIXI_CACHE_DIR=$HOME/lustre/pixi-cache RATTLER_CACHE_DIR=$HOME/lustre/pixi-cache CONDA_OVERRIDE_CUDA=13.3
# node-local scratch; when a node's local disk is full, fall back to the workspace on lustre
LOCAL=$(mktemp -d "${TMPDIR:-/tmp}/sddb-primers.XXXXXX" 2>/dev/null || { mkdir -p "$WORK/tmp" && mktemp -d "$WORK/tmp/sddb-primers.XXXXXX"; })
trap 'rm -rf "$LOCAL"' EXIT
export SDDB_CACHE_DIR=$LOCAL/sddb XDG_CACHE_HOME=$LOCAL/xdg MPLCONFIGDIR=$WORK/mpl-cache NUMBA_CACHE_DIR=$LOCAL/numba UV_CACHE_DIR=$WORK/uv-cache
export PYTHONPATH=$WORK/code/primers${PYTHONPATH:+:$PYTHONPATH}
run() { ~/.pixi/bin/pixi run --manifest-path "$ENV/pixi.toml" "$@"; }
