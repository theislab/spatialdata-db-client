#!/bin/bash
#SBATCH -J primers-install
#SBATCH -p cpu_p
#SBATCH --qos=cpu_normal
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH -o logs/primers-install-%j.log
# submit from the workspace (sbatch code/primers/hpc/<job>.sbatch), or set WORK
source "${WORK:=$SLURM_SUBMIT_DIR}/code/primers/hpc/common.sh"
(cd "$ENV" && ~/.pixi/bin/pixi install)
run python -c "import sddb, primers; print(\"sddb\", sddb.__version__)"
