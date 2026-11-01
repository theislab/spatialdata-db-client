#!/bin/bash
#SBATCH -J primers-hdbin
#SBATCH -p gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100_80gb|h100_80gb|b300
#SBATCH -c 16
#SBATCH --mem=160G
#SBATCH -t 04:00:00
#SBATCH -o logs/primers-hdbin-%j.log
# submit from the workspace (sbatch code/primers/hpc/<job>.sbatch), or set WORK
source "${WORK:=$SLURM_SUBMIT_DIR}/code/primers/hpc/common.sh"
mkdir -p "$WORK/hd_bin_size"
cd "$WORK/code/primers"
run python scripts/hd_bin_size.py oliveira-2024-colorectal colon --release "${RELEASE:-2026-10-28}" --out "$WORK/hd_bin_size/hd_bin_size.tsv"
