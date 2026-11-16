#!/bin/bash
# Copy this checkout to the HPC workspace and record which commit it is, because the copy has no .git.
# Run on the machine with the git checkout: primers/hpc/sync.sh <ssh host> <workspace>
set -euo pipefail
HOST=${1:?usage: sync.sh <ssh host> <workspace>}
WORK=${2:?usage: sync.sh <ssh host> <workspace>}
ROOT=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
git -C "$ROOT" fetch -q origin || true
REV=$(git -C "$ROOT" rev-parse HEAD)
DIRTY=$(git -C "$ROOT" status --porcelain -- primers src pyproject.toml | head -1)
PUBLIC=$(git -C "$ROOT" branch -r --contains HEAD | head -1)
rsync -a --delete --exclude .venv --exclude .pixi --exclude __pycache__ --exclude .git --exclude '*.pyc' --exclude uv.lock "$ROOT/" "$HOST:$WORK/code/"
printf 'PRIMERS_REF=%s\nDIRTY=%s\nPUBLIC=%s\n' "$REV" "${DIRTY:+yes}" "${PUBLIC:+yes}" | ssh "$HOST" "cat > $WORK/code/REVISION"
echo "synced $REV (dirty: ${DIRTY:+yes}${DIRTY:-no}, on a remote branch: ${PUBLIC:+yes}${PUBLIC:-no})"
