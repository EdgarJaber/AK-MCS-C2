# AK-MCS-C²

**Active Kriging Monte Carlo Simulation with conformal certification for failure probability estimation.**

Reference implementation accompanying the paper:

> E. Jaber, V. Chabridon, M. Mougeot, *AK-MCS-C²: Active Kriging Monte Carlo Simulation method with conformal certification for failure probability estimation*.

This Python package provides a class-oriented AK-MCS framework with pluggable learning functions:
  - the classical **U-function** (Echard et al., 2011) 
  - the **EFF** function (Bichon et al., 2008)
  - the proposed **conformal C²** criterion built on the **J+GP** and **J-minmax-GP** cross-conformal estimators (Jaber et al., 2025). 

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
    learning=ConformalC2(pf_gap_tol=1e-3, batch_q=1),
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

`result.history` records, per iteration: `pf_hat`, the certified conformal bounds `pf_lower`/`pf_upper`, the learning criterion, the empirical coverage diagnostic, and wall-clock time. `result.history_array()` returns it as a `(n_iter, 7)` array.

Classical baselines:

```python
AKMCS(problem=four_branch(6), learning="U").run(seed=0)     # min U >= 2 stopping
AKMCS(problem=four_branch(6), learning="EFF").run(seed=0)   # max EFF <= 1e-3
```

## Adding a custom learning function

Subclass `LearningFunction` and implement `select` / `converged` over the
per-iteration `IterationState` (GP posterior on the Monte Carlo population,
conformal intervals if requested, current bounds):

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

## Reproducing the article figures

```bash
# overnight on a workstation (~hours, 800 runs):
python scripts/run_experiments.py --n-jobs 16

# or on a SLURM cluster:
sbatch scripts/submit_overnight.sbatch

# then:
python scripts/make_figures.py --results results --out figures
```

Every `(problem, method, seed)` run is written to its own `.npz` upon
completion; rerunning skips existing files, so interrupted campaigns resume
for free. `figures/summary.csv` is the direct source of the summary table in
the paper.

## Citing

```bibtex
@misc{Jaber2026,
  title   = {AK-MCS-C$^2$: Active Kriging Monte Carlo Simulation method with
             conformal certification for failure probability estimation},
  author  = {Jaber, Edgar and Chabridon, Vincent and Mougeot, Mathilde},
  arxiv   = {...},
  year    = {2026}
}
```

## License

MIT — see [LICENSE](LICENSE).
