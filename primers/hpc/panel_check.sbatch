#!/bin/bash
#SBATCH -J primers-panel
#SBATCH -p cpu_p
#SBATCH --qos=cpu_normal
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH -o logs/primers-panel-%j.log
# submit from the workspace (sbatch code/primers/hpc/<job>.sbatch), or set WORK
source "${WORK:=$SLURM_SUBMIT_DIR}/code/primers/hpc/common.sh"
mkdir -p "$WORK/panel"
cd "$WORK/code/primers"
run python scripts/panel_check.py cohorts.yaml --release "${RELEASE:-2026-10-28}" --out "$WORK/panel/panels"
