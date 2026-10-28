#!/bin/bash
# CPU-only job on grace (no GPU). Compute goes through SLURM, not ada (Harsh, 2026-10-20); this project
# runs on grace only, mira and vega are for other projects (Harsh, 2026-10-28).
#   sbatch scripts/cpu.sbatch <python module> <args...>
#SBATCH --job-name=cpu
#SBATCH --partition=ashton
#SBATCH --qos=ashton
#SBATCH --nodelist=grace
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/%x_%j.log
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
pwd; hostname; date
set +u
eval "$(/ada1/u/harsh/miniconda3/bin/conda shell.bash hook)"
conda activate sote
set -u
cd "${CODE:-/ada1/u/harsh/assist-adapt}"
MOD=$1
shift
python -u -m "$MOD" "$@" --workers "${SLURM_CPUS_PER_TASK:-32}"
date
