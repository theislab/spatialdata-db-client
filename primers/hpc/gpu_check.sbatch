#!/bin/bash
#SBATCH -J primers-gpu
#SBATCH -p gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 00:30:00
#SBATCH -o logs/primers-gpu-%j.log
# submit from the workspace (sbatch code/primers/hpc/<job>.sbatch), or set WORK
source "${WORK:=$SLURM_SUBMIT_DIR}/code/primers/hpc/common.sh"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
run python "$WORK/code/primers/scripts/gpu_check.py"
