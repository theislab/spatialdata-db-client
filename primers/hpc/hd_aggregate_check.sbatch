#!/bin/bash
#SBATCH -J primers-hdagg
#SBATCH -p cpu_p
#SBATCH --qos=cpu_normal
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 01:00:00
#SBATCH -o logs/primers-hdagg-%j.log
# submit from the workspace (sbatch code/primers/hpc/<job>.sbatch), or set WORK
source "${WORK:=$SLURM_SUBMIT_DIR}/code/primers/hpc/common.sh"
cd "$WORK/code/primers"
run python scripts/hd_aggregate_check.py "${UID_CHECK:-iAZtgjxeI9UbeR4w0000}" --release "${RELEASE:-2026-10-28}"
