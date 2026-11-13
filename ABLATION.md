# Ablation: selection rule × stopping rule (R2.4, R3.1, R4.7)

Which component produces the improvement: the conformal **selection**
(diameter learning function) or the conformal **stopping** rule?

|            | stop **U** | stop **EFF** | stop **C2** (relative gap ≤ ε) |
|------------|-----------|--------------|--------------------------------|
| select **U**   | U-U (classical) | U-EFF | U-C2-{jp,mm} |
| select **EFF** | EFF-U | EFF-EFF (classical, δ = 2σ) | EFF-C2-{jp,mm} |
| select **C2**  | C2-U-{jp,mm} | C2-EFF-{jp,mm} | C2-C2-{jp,mm} (AK-MCS-C²) |

* Off-diagonal cells isolate the contribution. `--fixed-budget` adds a
  `STOP=none` column (all methods run to `max_calls`): fixed-budget accuracy,
  and pf_hat at any budget is read from the saved history.
* jp = J+GP, mm = J-minmax-GP, shared by selection and stopping.
* Same seed ⇒ same MC population and same initial design in every cell.
* U stop: min U ≥ 2; EFF stop: max EFF ≤ 1e-3 (δ = 2σ); C2 stop:
  (P⁺−P⁻)/max(P̂_f, 1/N) ≤ 0.1. If C2 selection finds no ambiguous point while
  the stop rule is not C2, selection falls back to U (run continues).
* Code: `Ablation` in `src/akmcsc2/learning.py`; driver `scripts/run_ablation.py`;
  summary `scripts/summarize_ablation.py`.

## Run

```bash
# 0. environment (python >= 3.11)
python3.11 -m venv venv && source venv/bin/activate
pip install --upgrade pip
pip install torch --extra-index-url https://download.pytorch.org/whl/cpu
pip install numpy scipy scikit-learn joblib gpytorch pytest -e .

# 1. sanity (≈1 min)
pytest tests -q -k "eff or ablation"
python scripts/run_ablation.py --list [--fixed-budget]

# 2. pilot: 3 seeds, one problem, to time it (extrapolate: 50 seeds ≈ 17× pilot)
time python scripts/run_ablation.py --problems branch2d_k6 --n-seeds 3 --n-jobs 4 --out-dir pilot

# 3a. full run on one node, detached (no SLURM)
nohup env OMP_NUM_THREADS=1 python scripts/run_ablation.py --n-jobs 10 \
      --out-dir results_ablation > run_ablation.log 2>&1 &

# 3b. SLURM job array (resumable: resubmit after a timeout, done files are skipped)
sbatch scripts/submit_ablation.sbatch
```

Add `--fixed-budget` to 3a/3b for the fixed-budget column (these runs go to
`max_calls`, so they are the most expensive; run them last or with fewer seeds,
e.g. `--n-seeds 20`). Use `--problems`, `--cells`, `--n-seeds` to split work.
Sizes without fixed-budget: 14 cells × 4 problems × 50 seeds = 2800 runs.

## Monitor / finish

```bash
ls results_ablation/*.npz | wc -l          # 2800 (3600+ with --fixed-budget) when complete
pgrep -f run_ablation | wc -l              # 0 when finished (nohup case)
ls results_ablation/*FAILED* 2>/dev/null   # tracebacks of crashed seeds
tail -2 run_ablation.log                   # "All done."
```

## Retrieve and summarize

```bash
python scripts/summarize_ablation.py --results results_ablation   # ablation_summary.csv + grids
# from the laptop:
scp -r <host>:<path>/results_ablation .
```
