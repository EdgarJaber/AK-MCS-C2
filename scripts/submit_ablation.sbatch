#!/bin/bash
#SBATCH --job-name=akmcs-abl
#SBATCH --output=logs/abl_%A_%a.out
#SBATCH --error=logs/abl_%A_%a.err
#SBATCH --array=0-39
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=24:00:00
# Resumable: finished tasks are skipped on resubmission. Add --fixed-budget
# to the python line for the STOP=none column.
set -euo pipefail
mkdir -p logs results_ablation
module purge; module load python/3.11 2>/dev/null || true
source "${VENV:-$PWD/venv}/bin/activate"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
python scripts/run_ablation.py --out-dir results_ablation \
    --shard "${SLURM_ARRAY_TASK_ID}" --n-shards "${SLURM_ARRAY_TASK_COUNT}"
