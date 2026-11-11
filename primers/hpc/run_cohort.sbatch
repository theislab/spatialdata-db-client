#!/bin/bash
#SBATCH -J primers-run
#SBATCH -p gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100_80gb|h100_80gb|b300
#SBATCH -c 16
#SBATCH --mem=200G
#SBATCH -t 08:00:00
#SBATCH -o logs/primers-run-%j.log
# Usage: sbatch --export=ALL,SLUG=human-colon[,SMOKE=3][,RELEASE=2026-10-28] hpc/run_cohort.sbatch
# submit from the workspace (sbatch code/primers/hpc/<job>.sbatch), or set WORK
source "${WORK:=$SLURM_SUBMIT_DIR}/code/primers/hpc/common.sh"
mkdir -p "$WORK/bundles"
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 2000 > "$LOCAL/gpu_mem.txt" &
SMI=$!
cd "$WORK/code/primers"
run python scripts/run_cohort.py cohorts.yaml "$SLUG" --release "${RELEASE:-2026-10-28}" --out "$WORK/bundles" ${SMOKE:+--smoke $SMOKE}
kill $SMI
echo "peak GPU memory MiB: $(sort -n "$LOCAL/gpu_mem.txt" | tail -1)"
