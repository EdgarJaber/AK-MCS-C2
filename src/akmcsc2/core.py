"""AK-MCS orchestrator.

One GP fit per iteration is shared between (i) the posterior mean/std used
by the learning function and the plug-in estimate ``pf_hat``, (ii) the
conformal regressor (``prefit=True``) used for the C^2 criterion and the
certified bounds, and (iii) the empirical-coverage diagnostic.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Union

import numpy as np
from sklearn.preprocessing import StandardScaler

from .conformal import ConformalGPRegressor, make_conformal
from .gp import LOOGPRegressor
from .learning import IterationState, LearningFunction, Selection, make_learning
from .problems import ReliabilityProblem

__all__ = ["AKMCS", "AKMCSResult", "HistoryRecord"]


@dataclass
class HistoryRecord:
    it: int
    n_calls: int
    pf_hat: float
    pf_lower: Optional[float]
    pf_upper: Optional[float]
    pf_gap: Optional[float]
    criterion: float
    batch_size: int
    emp_coverage: Optional[float]
    elapsed_s: float


@dataclass
class AKMCSResult:
    pf: float
    pf_lower: Optional[float]
    pf_upper: Optional[float]
    n_calls: int
    n_iterations: int
    converged: bool
    history: List[HistoryRecord]
    X_train: np.ndarray
    y_train: np.ndarray
    S_raw: np.ndarray
    pf_mc_ref: float                      # crude MC on the same cloud (analytic g only)
    gp: Optional[LOOGPRegressor] = None
    conformal: Optional[ConformalGPRegressor] = None

    def history_array(self) -> np.ndarray:
        """(n_iter, 7): n_calls, pf_hat, pf_lower, pf_upper, pf_gap,
        emp_coverage, elapsed_s. None -> NaN."""
        def f(v):
            return np.nan if v is None else float(v)
        return np.array(
            [
                [
                    h.n_calls, h.pf_hat, f(h.pf_lower), f(h.pf_upper),
                    f(h.pf_gap), f(h.emp_coverage), h.elapsed_s,
                ]
                for h in self.history
            ],
            dtype=float,
        )

    HISTORY_COLUMNS = (
        "n_calls", "pf_hat", "pf_lower", "pf_upper",
        "pf_gap", "emp_coverage", "elapsed_s",
    )


class AKMCS:
    """Active Kriging Monte Carlo Simulation with pluggable learning functions.

    Parameters
    ----------
    problem : ReliabilityProblem
    learning : LearningFunction or str
        ``"U"``, ``"EFF"``, ``"C2"`` (see :mod:`akmcsc2.learning`), or any
        custom :class:`LearningFunction` instance.
    conformal : ConformalGPRegressor, str or None
        Required when ``learning.requires_conformal``; ``"j+gp"`` or
        ``"j-mm-gp"``, or a prototype instance. ``alpha`` is taken from it.
    gp_factory : callable -> LOOGPRegressor
        Builds the surrogate refitted at each iteration.
    n_mc : int
        Monte Carlo population size.
    n_init : int
        Initial design size (drawn uniformly from the MC population).
    max_calls : int
        Budget of limit-state evaluations (including the initial design).
    min_iterations : int
        Convergence before this many iterations triggers a design restart
        (guards against degenerate early stops on tiny designs).
    conformal_every : int
        Recompute the full-population conformal bounds every k iterations
        (the dominant cost); cached in between.
    track_coverage : bool
        Evaluate g on the whole population once and record the empirical
        coverage of the conformal intervals at each (recomputed) iteration.
        Only meaningful for cheap analytic benchmarks.
    """

    def __init__(
        self,
        problem: ReliabilityProblem,
        learning: Union[LearningFunction, str] = "U",
        conformal: Union[ConformalGPRegressor, str, None] = None,
        gp_factory: Optional[Callable[[], LOOGPRegressor]] = None,
        n_mc: int = 100_000,
        n_init: int = 20,
        max_calls: int = 400,
        min_iterations: int = 20,
        conformal_every: int = 1,
        track_coverage: bool = False,
        predict_batch_size: int = 65536,
        verbose: bool = False,
    ) -> None:
        self.problem = problem
        self.learning = (
            make_learning(learning) if isinstance(learning, str) else learning
        )
        if isinstance(conformal, str):
            conformal = make_conformal(conformal)
        if self.learning.requires_conformal and conformal is None:
            conformal = make_conformal("j+gp")
        self.conformal_proto = conformal
        self.gp_factory = gp_factory or (
            lambda: LOOGPRegressor(kernel="matern52", noise=1e-4, training_iter=100)
        )
        self.n_mc = n_mc
        self.n_init = n_init
        self.max_calls = max_calls
        self.min_iterations = min_iterations
        self.conformal_every = conformal_every
        self.track_coverage = track_coverage
        self.predict_batch_size = predict_batch_size
        self.verbose = verbose

    # ------------------------------------------------------------------
    def run(self, seed: int = 0) -> AKMCSResult:
        rng = np.random.default_rng(seed)
        problem, learning = self.problem, self.learning
        use_conf = learning.requires_conformal

        # 0) MC cloud --------------------------------------------------
        S_raw = problem.sample(self.n_mc, rng)
        scaler = StandardScaler()
        S = scaler.fit_transform(S_raw)
        N = S.shape[0]

        y_true_full = None
        pf_mc_ref = np.nan
        if self.track_coverage:
            y_true_full = problem(S_raw)             # evaluated ONCE
            pf_mc_ref = float(np.mean(y_true_full <= 0.0))

        # 1) initial design ---------------------------------------------
        def init_design():
            idx0 = rng.choice(N, size=self.n_init, replace=False)
            y0 = problem(S_raw[idx0])
            cache = np.full(N, np.nan)
            cache[idx0] = y0
            return S[idx0].copy(), y0.copy(), set(int(i) for i in idx0), cache

        X_train, y_train, used, y_cache = init_design()

        history: List[HistoryRecord] = []
        cached_bounds = None       # (pf_lower, pf_upper, emp_cov)
        gp = conf = None
        converged = False
        t0 = time.perf_counter()
        it = 0

        while True:
            # -- fit ONE surrogate -------------------------------------
            gp = self.gp_factory()
            gp.fit(X_train, y_train)

            used_idx = np.fromiter(used, dtype=int)
            mask_unused = np.ones(N, dtype=bool)
            mask_unused[used_idx] = False

            mu = np.empty(N)
            std = np.empty(N)
            if np.any(mask_unused):
                mu[mask_unused], std[mask_unused] = gp.predict(
                    S[mask_unused], return_std=True,
                    batch_size=self.predict_batch_size,
                )
            mu[used_idx] = y_cache[used_idx]
            std[used_idx] = 0.0
            pf_hat = float(np.mean(mu <= 0.0))

            # -- conformal layer (shared fit) ---------------------------
            pf_lower = pf_upper = pf_gap = emp_cov = None
            conf = None
            if use_conf:
                conf = type(self.conformal_proto)(
                    estimator=gp,
                    alpha=self.conformal_proto.alpha,
                    eps=self.conformal_proto.eps,
                    prefit=True,
                )
                conf.fit(X_train, y_train)

                if it % self.conformal_every == 0 or cached_bounds is None:
                    interval = conf.predict_interval(
                        S, batch_size=self.predict_batch_size
                    )
                    lo, up = interval[:, 0].copy(), interval[:, 1].copy()
                    lo[used_idx] = y_cache[used_idx]
                    up[used_idx] = y_cache[used_idx]
                    pf_lower = float(np.mean(up <= 0.0))
                    pf_upper = float(np.mean(lo <= 0.0))
                    if y_true_full is not None:
                        emp_cov = float(np.mean(
                            (lo <= y_true_full) & (y_true_full <= up)
                        ))
                    cached_bounds = (pf_lower, pf_upper, emp_cov)
                else:
                    pf_lower, pf_upper, emp_cov = cached_bounds
                pf_gap = pf_upper - pf_lower

            # -- learning function --------------------------------------
            state = IterationState(
                it=it, S_scaled=S, mu=mu, std=std, mask_unused=mask_unused,
                gp=gp, conformal=conf, pf_hat=pf_hat,
                pf_lower=pf_lower, pf_upper=pf_upper, pf_gap=pf_gap,
                n_calls=len(y_train), rng=rng,
            )
            selection = learning.select(state)
            stop = learning.converged(state, selection)

            history.append(HistoryRecord(
                it=it, n_calls=len(y_train), pf_hat=pf_hat,
                pf_lower=pf_lower, pf_upper=pf_upper, pf_gap=pf_gap,
                criterion=selection.criterion,
                batch_size=len(selection.batch_idx),
                emp_coverage=emp_cov,
                elapsed_s=time.perf_counter() - t0,
            ))
            if self.verbose:
                gap = f"{pf_gap:.2e}" if pf_gap is not None else "  NA  "
                print(
                    f"[{learning.name:>3s}] it={it:4d} ncall={len(y_train):4d} "
                    f"pf={pf_hat:.3e} gap={gap} crit={selection.criterion:.3e}"
                )

            # -- stop / restart / budget --------------------------------
            if stop or len(y_train) >= self.max_calls or selection.exhausted:
                if stop and it < self.min_iterations and len(y_train) < self.max_calls:
                    X_train, y_train, used, y_cache = init_design()
                    cached_bounds = None
                    it += 1
                    continue
                converged = stop
                break

            # -- enrich --------------------------------------------------
            batch = np.asarray(selection.batch_idx, dtype=int)
            y_new = problem(S_raw[batch])
            X_train = np.vstack([X_train, S[batch]])
            y_train = np.append(y_train, y_new)
            for j, yv in zip(batch, y_new):
                used.add(int(j))
                y_cache[int(j)] = float(yv)
            it += 1

        return AKMCSResult(
            pf=pf_hat, pf_lower=pf_lower, pf_upper=pf_upper,
            n_calls=len(y_train), n_iterations=it + 1, converged=converged,
            history=history, X_train=X_train, y_train=y_train, S_raw=S_raw,
            pf_mc_ref=pf_mc_ref, gp=gp, conformal=conf,
        )
