#!/usr/bin/env bash
# =============================================================================
# Overnight AK-MCS-C2 benchmark on a single workstation (no SLURM).
#
# Target: 2x Xeon Gold 6442Y (48 physical cores), 512 GB RAM,
#         2x RTX 6000 Ada (optional, off by default -- GP designs are small
#         enough that CPU task-parallelism dominates).
#
# Usage:
#   ./scripts/run_overnight.sh                 # CPU only, auto worker count
#   ./scripts/run_overnight.sh --workers 40    # explicit CPU worker count
#   ./scripts/run_overnight.sh --with-gpus     # + one extra shard per GPU
#
# Detach-safe: everything runs under nohup; close the SSH session freely.
#   tail -f logs/overnight_*.log               # watch progress
#   ls results/*.npz | wc -l                   # 800 when complete
#
# Fully resumable: each (problem, method, seed) writes its own .npz on
# completion; relaunching skips existing files.
#
# When finished, figures + summary.csv for the article:
#   python scripts/make_figures.py --results results --out figures
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

WORKERS=""
WITH_GPUS=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --workers)   WORKERS="$2"; shift 2 ;;
    --with-gpus) WITH_GPUS=1;  shift   ;;
    *) echo "unknown option: $1"; exit 1 ;;
  esac
done

# Physical cores (HT gives no speedup for dense linear algebra); leave a few
# cores for the OS and the figure run.
PHYS=$(lscpu -p=Core,Socket 2>/dev/null | grep -v '^#' | sort -u | wc -l)
PHYS=${PHYS:-48}
WORKERS=${WORKERS:-$(( PHYS - 4 ))}

mkdir -p logs results
STAMP=$(date +%Y%m%d_%H%M%S)

# One worker = one task = one torch process: pin BLAS/torch to 1 thread each
# to avoid 40 workers x 48 threads oversubscription.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TORCH_NUM_THREADS=1

PIDS=()

if [[ "$WITH_GPUS" -eq 1 ]]; then
  # 3-way shard: GPU0, GPU1, CPU pool. The GPU shards are sequential
  # single-process runs (per-task GPU utilisation is low for these GP sizes;
  # do not oversubscribe a GPU with multiple workers without MPS).
  echo "Launching: 2 GPU shards + ${WORKERS} CPU workers"
  CUDA_VISIBLE_DEVICES=0 nohup python scripts/run_experiments.py \
      --out-dir results --use-cuda --shard 0 --n-shards 3 \
      > "logs/overnight_${STAMP}_gpu0.log" 2>&1 &
  PIDS+=($!)
  CUDA_VISIBLE_DEVICES=1 nohup python scripts/run_experiments.py \
      --out-dir results --use-cuda --shard 1 --n-shards 3 \
      > "logs/overnight_${STAMP}_gpu1.log" 2>&1 &
  PIDS+=($!)
  CUDA_VISIBLE_DEVICES="" nohup python scripts/run_experiments.py \
      --out-dir results --n-jobs "$WORKERS" --shard 2 --n-shards 3 \
      > "logs/overnight_${STAMP}_cpu.log" 2>&1 &
  PIDS+=($!)
else
  echo "Launching: ${WORKERS} CPU workers (GPUs unused; pass --with-gpus to change)"
  CUDA_VISIBLE_DEVICES="" nohup python scripts/run_experiments.py \
      --out-dir results --n-jobs "$WORKERS" \
      > "logs/overnight_${STAMP}_cpu.log" 2>&1 &
  PIDS+=($!)
fi

echo "${PIDS[@]}" > "logs/overnight_${STAMP}.pids"
echo "PIDs: ${PIDS[*]}  (saved to logs/overnight_${STAMP}.pids)"
echo "Monitor:  tail -f logs/overnight_${STAMP}_*.log"
echo "Progress: watch -n 60 'ls results/*.npz 2>/dev/null | wc -l'   # target: 800"
echo "Kill:     kill \$(cat logs/overnight_${STAMP}.pids)"

# Detached watcher: polls the workers and builds the article figures when all
# are done. setsid + nohup: survives SSH disconnection and session teardown.
POLL=${POLL:-60}
setsid nohup bash -c '
  for pid in '"${PIDS[*]}"'; do
    while kill -0 "$pid" 2>/dev/null; do sleep '"$POLL"'; done
  done
  python scripts/make_figures.py --results results --out figures
  echo "[watcher] figures + summary.csv written to figures/"
' > "logs/figures_${STAMP}.log" 2>&1 &
echo "Watcher PID: $!  (auto-runs make_figures.py at completion -> logs/figures_${STAMP}.log)"
