"""Learning functions for active enrichment.

A learning function consumes the per-iteration :class:`IterationState` and
returns a :class:`Selection` (batch of Monte Carlo indices to evaluate, the
criterion value, and whether its native stopping rule fires).

To add a custom learning function, subclass :class:`LearningFunction`,
implement :meth:`select` (and optionally :meth:`converged`), then either pass
an instance to :class:`akmcsc2.AKMCS` or register it::

    @register_learning("my-rule")
    class MyRule(LearningFunction):
        ...
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Type

import numpy as np
from scipy.stats import norm

__all__ = [
    "IterationState",
    "Selection",
    "LearningFunction",
    "UFunction",
    "ExpectedFeasibility",
    "ConformalC2",
    "register_learning",
    "make_learning",
]


@dataclass
class IterationState:
    """Snapshot handed to the learning function at each iteration."""

    it: int
    S_scaled: np.ndarray              # (N, d) standardized MC cloud
    mu: np.ndarray                    # (N,) GP posterior mean (true y at used pts)
    std: np.ndarray                   # (N,) GP posterior std (0 at used pts)
    mask_unused: np.ndarray           # (N,) bool
    gp: object                        # fitted LOOGPRegressor
    conformal: Optional[object]       # fitted ConformalGPRegressor or None
    pf_hat: float
    pf_lower: Optional[float] = None  # conformal bounds (C2 only)
    pf_upper: Optional[float] = None
    pf_gap: Optional[float] = None
    n_calls: int = 0
    rng: Optional[np.random.Generator] = None


@dataclass
class Selection:
    batch_idx: List[int] = field(default_factory=list)
    criterion: float = np.nan
    exhausted: bool = False           # no admissible candidate left


class LearningFunction(ABC):
    """Base class. Set ``requires_conformal=True`` to have the AK-MCS loop
    fit a conformal regressor and maintain the bounds (pf_lower, pf_upper)."""

    requires_conformal: bool = False
    name: str = "base"

    @abstractmethod
    def select(self, state: IterationState) -> Selection:
        ...

    @abstractmethod
    def converged(self, state: IterationState, selection: Selection) -> bool:
        ...


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------

_LEARNING: Dict[str, Type[LearningFunction]] = {}


def register_learning(name: str):
    def deco(cls: Type[LearningFunction]) -> Type[LearningFunction]:
        cls.name = name
        _LEARNING[name.lower()] = cls
        return cls
    return deco


def make_learning(name: str, **kwargs) -> LearningFunction:
    try:
        return _LEARNING[name.lower()](**kwargs)
    except KeyError as e:
        raise KeyError(
            f"Unknown learning function '{name}'. Available: {sorted(_LEARNING)}"
        ) from e


# ---------------------------------------------------------------------------
# classical criteria
# ---------------------------------------------------------------------------

@register_learning("U")
class UFunction(LearningFunction):
    """U-function of Echard et al. (2011); stops when min U >= threshold."""

    def __init__(self, threshold: float = 2.0) -> None:
        self.threshold = threshold

    def select(self, state: IterationState) -> Selection:
        U = np.abs(state.mu) / np.maximum(state.std, 1e-12)
        U[~state.mask_unused] = np.inf
        j = int(np.argmin(U))
        return Selection(batch_idx=[j], criterion=float(U[j]),
                         exhausted=not np.any(state.mask_unused))

    def converged(self, state, selection) -> bool:
        return selection.criterion >= self.threshold


@register_learning("EFF")
class ExpectedFeasibility(LearningFunction):
    """Expected Feasibility Function (Bichon et al., 2008), threshold a=0,
    tolerance eps = 2*sigma^2; stops when max EFF <= tol."""

    def __init__(self, tol: float = 1e-3) -> None:
        self.tol = tol

    @staticmethod
    def _eff(mu: np.ndarray, sigma: np.ndarray, a: float = 0.0) -> np.ndarray:
        """Closed form of E[(eps - |G - a|)^+], G ~ N(mu, sigma^2),
        eps = 2 sigma^2 (Bichon et al., 2008). Note the MINUS sign on the
        sigma term; validated against quadrature in tests/test_learning.py."""
        eps = 2.0 * sigma**2
        s = np.maximum(sigma, 1e-12)
        zm, z, zp = (a - eps - mu) / s, (a - mu) / s, (a + eps - mu) / s
        Phi, phi = norm.cdf, norm.pdf
        eff = (
            (mu - a) * (2 * Phi(z) - Phi(zm) - Phi(zp))
            - s * (2 * phi(z) - phi(zm) - phi(zp))
            + eps * (Phi(zp) - Phi(zm))
        )
        return np.maximum(eff, 0.0)

    def select(self, state: IterationState) -> Selection:
        eff = self._eff(state.mu, state.std)
        eff[~state.mask_unused] = -np.inf
        j = int(np.argmax(eff))
        return Selection(batch_idx=[j], criterion=float(eff[j]),
                         exhausted=not np.any(state.mask_unused))

    def converged(self, state, selection) -> bool:
        return selection.criterion <= self.tol


# ---------------------------------------------------------------------------
# conformal C^2 criterion
# ---------------------------------------------------------------------------

def greedy_farthest_radius(
    candidate_idx: np.ndarray,
    scores: np.ndarray,
    S: np.ndarray,
    q: int = 1,
    delta: Optional[float] = None,
) -> List[int]:
    """Greedy batch: maximize interval radius subject to pairwise distance
    >= delta. Vectorized O(q * |candidates|) instead of the naive O(q^2 *
    |candidates|) inner re-scan."""
    if len(candidate_idx) == 0 or q <= 0:
        return []
    order = np.argsort(-scores)
    idx = np.asarray(candidate_idx)[order]
    if q == 1 or delta is None or delta <= 0:
        return [int(i) for i in idx[:q]]

    pts = S[idx]                                   # (c, d), sorted by score
    selected: List[int] = [int(idx[0])]
    dmin = np.linalg.norm(pts - pts[0], axis=1)    # running min distance
    alive = dmin >= delta
    alive[0] = False
    while len(selected) < q and np.any(alive):
        j = int(np.argmax(alive))                  # best remaining score
        selected.append(int(idx[j]))
        d = np.linalg.norm(pts - pts[j], axis=1)
        dmin = np.minimum(dmin, d)
        alive &= dmin >= delta
        alive[j] = False
    # top-up without the distance constraint if the batch is short
    if len(selected) < q:
        chosen = set(selected)
        for i in idx:
            if int(i) not in chosen:
                selected.append(int(i))
                chosen.add(int(i))
            if len(selected) >= q:
                break
    return selected


@register_learning("C2")
class ConformalC2(LearningFunction):
    """Conformal C^2 learning function (this paper).

    Selects, among Monte Carlo points whose conformal interval straddles 0,
    the one(s) of largest interval diameter, with an optional greedy
    farthest-radius batch (``batch_q``, ``batch_delta``). Stops when the
    certified bound gap ``pf_upper - pf_lower <= pf_gap_tol``.

    Parameters
    ----------
    candidate_size : int
        Number of candidates pre-screened by the (free) U-proxy before the
        conformal intervals are computed; the conformal pass is the dominant
        cost, so this bounds it independently of the MC population size.
    """

    requires_conformal = True

    def __init__(
        self,
        candidate_size: int = 5000,
        batch_q: int = 1,
        batch_delta: Optional[float] = None,
        pf_gap_tol: float = 1e-3,
    ) -> None:
        self.candidate_size = candidate_size
        self.batch_q = batch_q
        self.batch_delta = batch_delta
        self.pf_gap_tol = pf_gap_tol

    def select(self, state: IterationState) -> Selection:
        if state.conformal is None:
            raise RuntimeError("ConformalC2 requires a fitted conformal regressor.")
        n_unused = int(np.sum(state.mask_unused))
        if n_unused == 0:
            return Selection(exhausted=True)

        Uproxy = np.abs(state.mu) / np.maximum(state.std, 1e-12)
        Uproxy[~state.mask_unused] = np.inf
        c = min(self.candidate_size, n_unused)
        cand = np.argpartition(Uproxy, c - 1)[:c]
        cand = cand[state.mask_unused[cand]]
        if cand.size == 0:
            return Selection(exhausted=True)

        interval = state.conformal.predict_interval(state.S_scaled[cand])
        lower, upper = interval[:, 0], interval[:, 1]
        uncertain = (lower <= 0.0) & (upper >= 0.0)
        if not np.any(uncertain):
            return Selection(exhausted=True)

        u = np.where(uncertain)[0]
        scores = upper[u] - lower[u]
        batch = greedy_farthest_radius(
            cand[u], scores, state.S_scaled, q=self.batch_q, delta=self.batch_delta
        )
        return Selection(batch_idx=batch, criterion=float(np.max(scores)))

    def converged(self, state, selection) -> bool:
        if selection.exhausted:
            return True
        return state.pf_gap is not None and state.pf_gap <= self.pf_gap_tol
