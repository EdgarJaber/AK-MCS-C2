# AK-MCS-C²

**Active Kriging Monte Carlo Simulation with conformal prediction intervals for failure probability estimation.**

Reference implementation accompanying the paper:

> E. Jaber, V. Chabridon, M. Mougeot, *AK-MCS-C²: Active Kriging Monte Carlo Simulation method with conformal prediction intervals for failure probability estimation* (revised version, 2026).

This Python package provides a class-oriented AK-MCS framework with pluggable learning functions:
  - the classical **U-function** (Echard et al., 2011);
  - the **EFF** function (Bichon et al., 2008), with tolerance δ = 2σ;
  - the **SUR** strategy (stepwise uncertainty reduction; Bect et al., 2012; Chevalier et al., 2014), see [How SUR works](#how-sur-works);
  - the proposed **conformal C²** criterion built on the **J+GP** and **J-minmax-GP** cross-conformal estimators (Jaber et al., 2025).

The C² layer turns the Kriging surrogate into conformal response intervals
`[m_t(x), M_t(x)]` on the whole Monte Carlo population. These intervals drive
three decisions:
  - **selection:** add the widest interval among the ambiguous points `{m_t ≤ 0 ≤ M_t}`;
  - **bounds:** `P⁻ = P_N(M_t ≤ 0)` and `P⁺ = P_N(m_t ≤ 0)` on the failure probability;
  - **stopping:** stop when `(P⁺ − P⁻) / max(P̂_f, 1/N) ≤ ε_stop`.

An exploration safeguard prevents the vacuous stop when no failure has been
found yet: while `P⁺ = 0`, the point with the smallest lower bound `m_t` is
added instead. The coverage guarantee of the conformal intervals holds for
the initial (exchangeable) design only. On the enriched designs the intervals
and bounds are *empirically calibrated*, and their reliability is assessed
numerically in the paper.

## Installation

```bash
git clone https://github.com/EdgarJaber/AK-MCS-C2.git
cd AK-MCS-C2
pip install -e .                  # core
pip install -e ".[experiments]"   # + matplotlib, pandas, joblib, tqdm
```

Requires Python ≥ 3.10, PyTorch and GPyTorch (installed automatically).

## Quick start

```python
from akmcsc2 import AKMCS, ConformalC2, LOOGPRegressor, four_branch

algo = AKMCS(
    problem=four_branch(k=6),
    learning=ConformalC2(rel_gap_tol=0.1, batch_q=1),
    conformal="j-mm-gp",          # or "j+gp"
    gp_factory=lambda: LOOGPRegressor(kernel="matern52", noise=1e-4),
    n_mc=100_000,
    n_init=20,
    max_calls=200,
    track_coverage=True,           # cheap analytic g only
)
result = algo.run(seed=0)

print(f"Pf = {result.pf:.3e}  in  [{result.pf_lower:.3e}, {result.pf_upper:.3e}]"
      f"  with {result.n_calls} limit-state evaluations")
```

`result.history` records, per iteration:
  - `pf_hat` and the conformal bounds `pf_lower`/`pf_upper`;
  - the value of the stopping statistic;
  - the empirical coverage diagnostic;
  - the wall-clock time.

`result.history_array()` returns it as a `(n_iter, 7)` array. `n_calls` is
the size of the final design, while `n_calls_total` also counts the
evaluations of designs discarded by the restart guard.

Classical baselines:

```python
AKMCS(problem=four_branch(6), learning="U").run(seed=0)     # min U >= 2 stopping
AKMCS(problem=four_branch(6), learning="EFF").run(seed=0)   # max EFF <= 1e-3
```

## Decoupled selection and stopping (ablation, SUR)

`Ablation` combines any selection rule with any stopping rule. The rules are
`"U"`, `"EFF"`, `"C2"`, `"SUR"`, plus `"none"` for stopping only, which runs
to a fixed budget:

```python
from akmcsc2 import AKMCS, Ablation, LOOGPRegressor, make_conformal, get_problem

algo = AKMCS(
    problem=get_problem("rastrigin2d"),
    learning=Ablation(select_with="SUR", stop_with="SUR", sur_tol=0.1),
    gp_factory=lambda: LOOGPRegressor(kernel="matern52", noise=1e-4),
    n_mc=10_000, n_init=20, max_calls=650,
)
# conformal stopping on top of the SUR selection:
algo_c2 = AKMCS(problem=get_problem("rastrigin2d"),
                learning=Ablation(select_with="SUR", stop_with="C2"),
                conformal=make_conformal("j-mm-gp", alpha=0.1),
                gp_factory=lambda: LOOGPRegressor(kernel="matern52", noise=1e-4),
                n_mc=10_000, n_init=20, max_calls=650)
```

`make_conformal("gaussian")` gives the plain Gaussian interval `μ ± z σ` with
the same interface. Plugging it into the C² rules is the control experiment
that answers the question: does the conformal calibration matter?

## How SUR works

SUR is a one-step look-ahead criterion on the Kriging posterior
(`SURCriterion` in `src/akmcsc2/learning.py`). Let `g̃_t` and `σ̃_t` be the
posterior mean and standard deviation of the GP trained on the current design
`D_t`, and `D` the Monte Carlo population.

1. **Failure probability of each point.**
   `p_t(u) = Φ(−g̃_t(u) / σ̃_t(u))` is the posterior probability that `u` fails.
   `min(p_t, 1 − p_t) = Φ(−U_t)` is the probability that the sign of `g̃_t`
   misclassifies `u`.
2. **Uncertainty on the failure set.**
   `H_t = Σ_{u ∈ D} p_t(u) (1 − p_t(u))` measures the uncertainty over the
   whole population.
3. **Look-ahead.** Adding a candidate `x` with the not yet observed response
   `Y ~ N(g̃_t(x), σ̃_t²(x))`, with the hyperparameters kept fixed, gives the
   Kriging update:
   - the variance `σ̃²_{t+1}(u) = σ̃_t²(u) − c_t(u,x)² / σ̃_t²(x)` does not depend on `Y`;
   - the mean `g̃_{t+1}(u)` is affine in `Y`, with slope `c_t(u,x) / σ̃_t²(x)`;
   - `c_t` is the posterior covariance.
4. **Criterion.** `J_t(x) = E_Y[ Σ_u p_{t+1}(u) (1 − p_{t+1}(u)) ]` is the
   expected uncertainty after one more evaluation. It has a closed form:
   `J_t(x) = Σ_u 2 T(h_u, √((1 − ρ_u)/(1 + ρ_u)))`, where
   - `h_u = −g̃_t(u)/σ̃_t(u)`,
   - `ρ_u = c_t(u,x)² / (σ̃_t²(u) σ̃_t²(x))`,
   - `T` is Owen's T function.

   This is the bivariate-normal expression of Chevalier et al. (2014). A
   Gauss–Hermite quadrature over `Y` is available with `exact=False`.
5. **Selection.** `x_{t+1} = argmin J_t(x)` over a candidate set made of the
   200 unevaluated points with the smallest `U_t`. To keep the cost linear in
   the population size, only the active points `|g̃_t| ≤ 6 σ̃_t` enter the sums.
6. **Stopping** (`Ablation(stop_with="SUR")`). The posterior expected number of
   misclassified points, relative to the estimated number of failures, must
   fall below a threshold:
   `Σ_{u ∉ D_t} min(p_t, 1 − p_t) / max(N P̂_f, 1) ≤ sur_tol` (default `0.1`).
   This is the Gaussian analogue of the relative conformal gap.

The implementation reproduces the `sur_optim_parallel` criterion of the R
package KrigInv to 1e-15 on a test model and selects the same point on all
tested designs (see `validation/`).

## Benchmarks

`get_problem(name)` with `name` in:
  - `branch2d_k6`, `branch2d_k7`: four-branch series system;
  - `rastrigin2d`: modified Rastrigin, disconnected failure domain;
  - `oscillator6d`: nonlinear oscillator of Echard et al. (2011), P_f ≈ 2.86e-2;
  - `oscillator6d_variant`: the version used in the revised article, whose
    sine argument is ω₀²t₁/2 instead of ω₀t₁/2 (P_f ≈ 3.9e-2). Use it to
    reproduce the published oscillator results;
  - `truss23_tau120`, `truss23_tau134`, `truss23_tau145`: the 23-bar truss of
    Blatman & Sudret (2011), d = 10, with failure when the mid-span deflection
    exceeds τ = 12, 13.4 or 14.5 cm (reference P_f ≈ 1.5e-3, 1.1e-4, 1.2e-5).
    The deflection is computed by a direct-stiffness solver. It coincides with
    the closed form of the UQWorld benchmark (Hlobilová, Marelli & Sudret,
    2024) to machine precision.

## Adding a custom learning function

Subclass `LearningFunction` and implement `select` / `converged` over the
per-iteration `IterationState`. The state holds the GP posterior on the Monte
Carlo population, the conformal intervals if requested, and the current
bounds:

```python
from akmcsc2 import LearningFunction, Selection, register_learning
import numpy as np

@register_learning("my-rule")
class MyRule(LearningFunction):
    requires_conformal = False     # True -> conformal regressor + Pf bounds provided

    def select(self, state):
        score = -np.abs(state.mu)              # whatever you like
        score[~state.mask_unused] = -np.inf
        j = int(np.argmax(score))
        return Selection(batch_idx=[j], criterion=float(score[j]))

    def converged(self, state, selection):
        return selection.criterion < -5.0
```

Then `AKMCS(problem, learning="my-rule")` or pass an instance directly.
Batch enrichment for parallel limit-state evaluation is available through the
greedy farthest-radius selector (`ConformalC2(batch_q=q, batch_delta=δ)`).

## Reproducing the article

Every `(problem, method, seed)` run is written to its own `.npz` upon
completion. Rerunning skips existing files, so interrupted campaigns resume
for free.

```bash
# main comparison (U, EFF, C2-J+GP, C2-J-mm-GP), Table 1 and Figures 2-5
python scripts/run_experiments.py --n-jobs 16          # or: sbatch scripts/submit_overnight.sbatch

# selection x stopping ablation (+ --fixed-budget for the fixed-budget table)
python scripts/run_ablation.py --n-jobs 16 --out-dir results_ablation
python scripts/run_ablation.py --sur   --out-dir results_ablation   # SUR comparison
python scripts/run_ablation.py --gauss --out-dir results_gauss      # C2 rules with Gaussian intervals
python scripts/run_ablation.py --problems truss23_tau120 truss23_tau134 --n-seeds 10 --out-dir results_truss

# other studies
python scripts/run_sensitivity.py        # one-at-a-time sensitivity
python scripts/run_loo_comparison.py     # fixed vs refitted leave-one-out
python scripts/run_timing.py             # cost table (run alone on an idle node)

# final designs of each method on the 2D benchmarks (same seed for all methods)
python scripts/run_design_figure.py run  --problems branch2d_k7 rastrigin2d --seed 0 --n-jobs 4
python scripts/run_design_figure.py plot --problems branch2d_k7 rastrigin2d --out designs_2d.pdf

# summaries and figures
python scripts/summarize_ablation.py; python scripts/summarize_diagnostics.py
python scripts/make_ablation_figures.py --results results_ablation --out figs
python scripts/make_paper_assets_gauss.py --results results_ablation --out paper_assets_gauss
python scripts/make_paper_assets_gauss.py --results results_truss --out paper_assets_gauss --truss
```

`REVISION_EXPERIMENTS.md` lists which experiment answers which reviewer
comment, and how the evaluations are counted.

## Citing

```bibtex
@misc{Jaber2026,
  title   = {AK-MCS-C$^2$: Active Kriging Monte Carlo Simulation method with
             conformal prediction intervals for failure probability estimation},
  author  = {Jaber, Edgar and Chabridon, Vincent and Mougeot, Mathilde},
  year    = {2026}
}
```

## License

MIT — see [LICENSE](LICENSE).
