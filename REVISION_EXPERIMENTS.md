# Revision experiments: ingredients and order

| # | Experiment | Reviewers | Script → output | Needs |
|---|------------|-----------|-----------------|-------|
| A | Ablation selection × stopping (+ fixed budget) | R4.7, R3.1, R2.4 | `run_ablation.py` → `results_ablation/` | corrected EFF δ=2σ, relative stop, `Ablation` |
| B | Near-boundary coverage (supplies Prop. 1) | R3.2, R4.2 | same runs as A → `summarize_diagnostics.py` | per-iteration `diag` (written by A) |
| C | Gaussian vs conformal intervals, same designs | R3.3 | same runs as A → `summarize_diagnostics.py` | `diag` (gp_* columns) |
| D | Fixed-hyperparameter vs refitted LOO | R4.5 | `run_loo_comparison.py` → `loo_comparison/` | `loo_mode="refit"` |
| E | Cost table: wall-clock, GP fit, conformal, memory | R4.8 | `run_timing.py` → `timing_results/` | quiet node; no coverage tracking |

## Order

1. **Freeze the code** (this version): EFF δ=2σ, relative stop, `Ablation`,
   diagnostics, timers, honest evaluation accounting. Run `pytest tests`.
2. **A+B+C in one campaign** (`run_ablation.py`; every run saves `diag`, so
   B and C cost nothing extra and use exactly the designs of A).
   Pilot first (3 seeds), then the 14 cells; fixed-budget column last.
   The EFF-EFF cell supersedes the standalone EFF study.
3. **D** (small: 3 benchmarks × few seeds; n GP fits per interval).
4. **E last, alone on an idle node**: timings are corrupted by a loaded
   cluster. Do not run it concurrently with A.
5. Summaries: `summarize_ablation.py`, `summarize_diagnostics.py`; then update
   manuscript tables/figures and fill the red TODOs in the response letter.

## Evaluation accounting (R4.8): what changed
* `n_calls` = size of the final design. `n_calls_total` additionally counts the
  evaluations of designs discarded by the `min_iterations` restart guard; this
  is the honest cost and must be used in the cost comparisons
  (`n_restarts` says how often it happened).
* The one-off evaluation of g on the whole cloud (`track_coverage`) is a
  validation device, never counted, and excluded from timings (E disables it).
* Candidate screening only restricts the *selection* step. The bounds
  [P⁻, P⁺] need the conformal intervals on the **full** cloud every iteration
  (`conf_bounds` timer; `O(N·n)` per iteration, `O(predict_batch_size·n)`
  memory). Set `conformal_every=k` to amortise. Report `conf_bounds` as a
  separate column.
* Empirical coverage: the original `emp_coverage` column counted evaluated
  points as covered (they are in the cloud). The new `*_cov_*` diagnostics use
  the unevaluated points D∖D_t, as Eq. (32) states. Use the new ones in the
  paper (values will be slightly lower).

## Caveats
* Refit mode uses the GP's `predict` std (includes the likelihood noise);
  fixed mode uses the latent LOO formulas at test points: expect a tiny
  systematic std difference, which experiment D will show.
* Peak memory is per-process maximum RSS; meaningful only in `run_timing.py`
  (one fresh process per run).
