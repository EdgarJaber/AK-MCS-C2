#!/bin/bash
#SBATCH --job-name=akmcs-c2
#SBATCH --output=logs/akmcs-c2_%A_%a.out
#SBATCH --error=logs/akmcs-c2_%A_%a.err
#SBATCH --array=0-15
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=12:00:00
# Uncomment for GPU nodes (then add --use-cuda below):
##SBATCH --gres=gpu:1

# Overnight benchmark for the AK-MCS-C2 article.
# 4 problems x 4 methods x 50 seeds = 800 tasks, sharded over the array.
# Each task writes its own .npz on completion: resubmitting the array after
# a timeout resumes exactly where it stopped (existing files are skipped).

set -euo pipefail
mkdir -p logs results

module purge
module load python/3.11 2>/dev/null || true
source "${VENV:-$HOME/venvs/akmcsc2}/bin/activate"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}

python scripts/run_experiments.py \
    --out-dir results \
    --shard "${SLURM_ARRAY_TASK_ID}" \
    --n-shards "${SLURM_ARRAY_TASK_COUNT}"

# After ALL array tasks finish, build the article figures with:
#   python scripts/make_figures.py --results results --out figures
# Or submit it as a dependent job:
#   sbatch --dependency=afterok:<jobid> --wrap "python scripts/make_figures.py"
