#!/bin/bash
#SBATCH -J primers-primer
#SBATCH -p gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100_80gb|h100_80gb|b300
#SBATCH -c 16
#SBATCH --mem=200G
#SBATCH -t 04:00:00
#SBATCH -o logs/primers-primer-%j.log
# Usage: sbatch --export=ALL,SLUG=human-colon[,SMOKE=3][,RELEASE=2026-10-28][,PUBLISH=1] hpc/run_primer.sbatch
# Runs the primer notebook for one cohort and writes primer.ipynb and primer.html into its bundle.
# PRIMERS_REF is the commit of the code checkout, so the notebook's install line and source links name the
# code that ran. A development run (the default) may use a dirty checkout or a commit that no remote branch
# holds; the notebook then shows a banner saying so. PUBLISH=1 refuses both.
# submit from the workspace (sbatch code/primers/hpc/<job>.sbatch), or set WORK
source "${WORK:=$SLURM_SUBMIT_DIR}/code/primers/hpc/common.sh"
# The workspace copy has no .git: hpc/sync.sh writes REVISION (PRIMERS_REF, DIRTY, PUBLIC) when it copies.
if [ -d "$WORK/code/.git" ]; then
  PRIMERS_REF=$(git -C "$WORK/code" rev-parse HEAD)
  DIRTY=$(git -C "$WORK/code" status --porcelain)
  PUBLIC=$(git -C "$WORK/code" branch -r --contains HEAD)
elif [ -f "$WORK/code/REVISION" ]; then
  source "$WORK/code/REVISION"
else
  echo "no .git and no REVISION in $WORK/code; copy the code with hpc/sync.sh" >&2
  exit 1
fi
DEV_NOTE=""
if [ -z "$PUBLIC" ]; then
  DEV_NOTE="Development run: the source links point to commit $PRIMERS_REF, which is not public yet."
fi
if [ -n "$DIRTY" ]; then
  DEV_NOTE="${DEV_NOTE:-Development run.} The checkout had uncommitted changes, so the code that ran may differ from commit $PRIMERS_REF."
fi
if [ "${PUBLISH:-0}" = 1 ] && [ -n "$DEV_NOTE" ]; then
  echo "PUBLISH=1 needs a clean checkout whose HEAD is on a remote branch: $DEV_NOTE" >&2
  exit 1
fi
BUNDLE=$WORK/bundles/$SLUG${SMOKE:+-smoke}  # export_bundle names a smoke run's bundle <slug>-smoke
mkdir -p "$BUNDLE"
# papermill runs in node-local scratch; bundles/ there links to the workspace, so the notebook only ever
# sees the relative path "bundles"
ln -s "$WORK/bundles" "$LOCAL/bundles"
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 2000 > "$LOCAL/gpu_mem.txt" &
SMI=$!
run papermill "$WORK/code/primers/notebooks/primer.ipynb" "$BUNDLE/primer.ipynb" --cwd "$LOCAL" \
  -p slug "$SLUG" -p release "${RELEASE:-2026-10-28}" -p out_dir bundles -p primers_ref "$PRIMERS_REF" \
  -p dev_note "$DEV_NOTE" ${SMOKE:+-p smoke $SMOKE}
run python "$WORK/code/primers/scripts/clean_notebook.py" "$BUNDLE/primer.ipynb"  # papermill's absolute paths
# cells tagged hide-input (the title, the banner and the install line) show only their output
run jupyter nbconvert --to html --TagRemovePreprocessor.remove_input_tags=hide-input "$BUNDLE/primer.ipynb"
kill $SMI
echo "peak GPU memory MiB: $(sort -n "$LOCAL/gpu_mem.txt" | tail -1)"
