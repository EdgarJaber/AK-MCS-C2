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
    exploring: bool = False           # C2 exploration fallback was used


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
    tolerance eps = 2*sigma; stops when max EFF <= tol."""

    def __init__(self, tol: float = 1e-3) -> None:
        self.tol = tol

    @staticmethod
    def _eff(mu: np.ndarray, sigma: np.ndarray, a: float = 0.0) -> np.ndarray:
        """Closed form of E[(eps - |G - a|)^+], G ~ N(mu, sigma^2),
        eps = 2 sigma (Bichon et al., 2008). Note the MINUS sign on the
        sigma term; validated against quadrature in tests/test_learning.py."""
        eps = 2.0 * sigma
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
def _no_possible_failure(state: "IterationState") -> bool:
    """True iff the conformal upper failure probability P_f^+ is zero, i.e. no
    Monte Carlo point has a lower response bound <= 0 (no failure found)."""
    return state.pf_upper is not None and state.pf_upper <= 0.0


class ConformalC2(LearningFunction):
    """Conformal C^2 learning function (this paper).

    Selects, among Monte Carlo points whose conformal interval straddles 0,
    the one(s) of largest interval diameter, with an optional greedy
    farthest-radius batch (``batch_q``, ``batch_delta``). Stops when the
    relative bound gap
    ``(pf_upper - pf_lower) / max(pf_hat, 1/N) <= rel_gap_tol``.

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
        rel_gap_tol: float = 0.1,
        explore: bool = True,
    ) -> None:
        self.explore = explore
        self.candidate_size = candidate_size
        self.batch_q = batch_q
        self.batch_delta = batch_delta
        self.rel_gap_tol = rel_gap_tol

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
            if self.explore and _no_possible_failure(state):
                # Exploration safeguard: no pool point is classified as possibly
                # failing (P_f^+ = 0), so the conformal gap is vacuously closed.
                # Add the candidate closest to failure (smallest lower bound).
                j = int(np.argmin(lower))
                return Selection(batch_idx=[int(cand[j])], criterion=float(lower[j]),
                                 exploring=True)
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
        if state.pf_gap is None:
            return False
        if self.explore and _no_possible_failure(state):
            return False                  # never stop before a failure is found
        denom = max(state.pf_hat, 1.0 / len(state.mu))
        return state.pf_gap / denom <= self.rel_gap_tol


# ---------------------------------------------------------------------------
# SUR (stepwise uncertainty reduction), Bect et al. (2012), Chevalier et al. (2014)
# ---------------------------------------------------------------------------

def sur_expected_H(m, s2, C, sx2, exact=True, z=None, w=None):
    """Expected posterior uncertainty J(x) = E[ sum_u p'(u)(1-p'(u)) ] after one
    noise-free observation at each candidate x (SUR criterion).

    m, s2 : (M,) posterior mean and latent variance at the integration points;
    C     : (M, K) posterior covariances c(u, x_k);  sx2 : (K,) variances at x_k.
    exact=True uses the closed form  J_k = sum_u 2 T(h_u, sqrt((1-rho)/(1+rho))),
    h = -m/s, rho = c(u,x)^2/(s^2(u) s^2(x)), T = Owen's T function
    (equivalently  Phi(h) - Phi_2(h,h;rho), the bivariate-normal expression of
    Chevalier et al. 2014); otherwise Gauss-Hermite quadrature over the
    future observation (z, w: probabilists' nodes and normalised weights).
    """
    from scipy.stats import norm
    s2 = np.maximum(s2, 1e-14); sx2 = np.maximum(sx2, 1e-14)
    if exact:
        from scipy.special import owens_t
        h = -m / np.sqrt(s2)
        rho = np.clip(C ** 2 / (s2[:, None] * sx2[None, :]), 0.0, 1.0 - 1e-12)
        return np.sum(2.0 * owens_t(h[:, None], np.sqrt((1.0 - rho) / (1.0 + rho))), axis=0)
    a = C / np.sqrt(sx2)[None, :]
    sn = np.sqrt(np.maximum(s2[:, None] - a ** 2, 1e-14))
    J = np.zeros(C.shape[1])
    for zq, wq in zip(z, w):
        p = norm.cdf(-(m[:, None] + a * zq) / sn)
        J += wq * np.sum(p * (1.0 - p), axis=0)
    return J


class SURCriterion(LearningFunction):
    """One-step-lookahead SUR for the excursion set {g <= 0} of a GP.

    Uncertainty measure H = sum_u p(u)(1-p(u)) over the Monte Carlo cloud, with
    p = Phi(-m/s) the posterior failure probability of a point. For a candidate
    x the future observation changes m(u) -> m(u) + c(u,x)/s(x) Z, Z~N(0,1), and
    s^2(u) -> s^2(u) - c(u,x)^2/s^2(x) (noise-free update), c = latent posterior
    covariance; J(x) = E_Z[H_{n+1}] is evaluated by Gauss-Hermite quadrature and
    the minimiser over a candidate set (lowest U) is added. Only the active
    points (|m| <= z_max s) enter the sums.  Own implementation on the same GP
    as the other criteria; stopping is handled by ``Ablation(stop_with="SUR")``.
    """

    requires_conformal = False

    def __init__(self, candidate_size: int = 200, n_gh: int = 15, z_max: float = 6.0,
                 max_active: int = 5000, exact: bool = True) -> None:
        self.exact = exact
        self.candidate_size, self.n_gh = candidate_size, n_gh
        self.z_max, self.max_active = z_max, max_active
        x, w = np.polynomial.hermite_e.hermegauss(n_gh)
        self._z, self._w = x, w / np.sqrt(2.0 * np.pi)

    def criterion_values(self, state: IterationState):
        """Returns (candidate indices, J(x) per candidate, H_n)."""
        from scipy.stats import norm
        un = state.mask_unused
        S = state.S_scaled
        U = np.abs(state.mu) / np.maximum(state.std, 1e-12); U[~un] = np.inf
        c = min(self.candidate_size, int(un.sum()))
        cand = np.argpartition(U, c - 1)[:c]; cand = cand[un[cand]]
        # active integration points
        act = np.where(un & (U < self.z_max))[0]
        if act.size > self.max_active:
            act = act[np.argsort(U[act])[: self.max_active]]
        gp = state.gp
        if act.size == 0:                       # every pool point is |mu| >= z_max sigma
            return cand[:0], np.empty(0), 0.0
        m = state.mu[act]
        s2 = np.maximum(gp.latent_var(S[act]), 1e-14)
        p0 = norm.cdf(-m / np.sqrt(s2))
        H0 = float(np.sum(p0 * (1.0 - p0)))
        sx2 = np.maximum(gp.latent_var(S[cand]), 1e-14)
        C = gp.latent_cross_cov(S[act], S[cand])                     # (M, K)
        J = sur_expected_H(m, s2, C, sx2, exact=self.exact, z=self._z, w=self._w)
        return cand, J, H0

    def select(self, state: IterationState) -> Selection:
        if int(state.mask_unused.sum()) == 0:
            return Selection(exhausted=True)
        cand, J, H0 = self.criterion_values(state)
        if cand.size == 0 or H0 < 1e-12:
            return Selection(exhausted=True)
        j = int(np.argmin(J))
        return Selection(batch_idx=[int(cand[j])], criterion=float(J[j]))

    def converged(self, state, selection) -> bool:      # used only standalone
        return selection.exhausted


# ---------------------------------------------------------------------------
# ablation: selection rule x stopping rule
# ---------------------------------------------------------------------------

@register_learning("ABL")
class Ablation(LearningFunction):
    """Decoupled learning function for the selection x stopping ablation.

    ``select_with`` chooses *which point is added* ("U", "EFF" or "C2");
    ``stop_with`` chooses *when to stop* ("U": min U >= u_threshold,
    "EFF": max EFF <= eff_tol, "C2": (pf_upper - pf_lower) /
    max(pf_hat, 1/N) <= rel_gap_tol, "none": never, i.e. fixed budget
    ``max_calls``). The stopping statistic is always computed over the
    unevaluated MC points and written to ``selection.criterion`` so that the
    history records the value of the stopping rule actually in force.

    If C2 selection finds no ambiguous point while the stopping rule is not
    C2, the selection falls back to the U-function so that the run goes on
    under the chosen stopping rule instead of ending without having stopped.
    The conformal layer is fitted whenever either rule is "C2".
    """

    _RULES = ("U", "EFF", "C2", "SUR")

    def __init__(
        self,
        select_with: str = "C2",
        stop_with: str = "C2",
        u_threshold: float = 2.0,
        eff_tol: float = 1e-3,
        rel_gap_tol: float = 0.1,
        candidate_size: int = 5000,
        batch_q: int = 1,
        batch_delta: Optional[float] = None,
        sur_tol: float = 0.1,
        explore: bool = True,
    ) -> None:
        self.explore = explore
        if select_with not in self._RULES:
            raise ValueError(f"select_with must be in {self._RULES}")
        if stop_with not in self._RULES + ("none",):
            raise ValueError(f"stop_with must be in {self._RULES + ('none',)}")
        self.select_with = select_with
        self.stop_with = stop_with
        self.u_threshold = u_threshold
        self.eff_tol = eff_tol
        self.rel_gap_tol = rel_gap_tol
        self._sel = {
            "U": UFunction(u_threshold),
            "EFF": ExpectedFeasibility(eff_tol),
            "C2": ConformalC2(candidate_size, batch_q, batch_delta, rel_gap_tol, explore),
            "SUR": SURCriterion(),
        }
        self.sur_tol = sur_tol
        self.requires_conformal = "C2" in (select_with, stop_with)

    def select(self, state: IterationState) -> Selection:
        sel = self._sel[self.select_with].select(state)
        if (sel.exhausted and self.select_with in ("C2", "SUR") and self.stop_with not in ("C2", "SUR")
                and np.any(state.mask_unused)):
            sel = self._sel["U"].select(state)
        return sel

    def _stop_value(self, state: IterationState):
        un = state.mask_unused
        if self.stop_with == "U":
            U = np.abs(state.mu[un]) / np.maximum(state.std[un], 1e-12)
            v = float(U.min()) if U.size else np.inf
            return v, v >= self.u_threshold
        if self.stop_with == "EFF":
            e = ExpectedFeasibility._eff(state.mu[un], state.std[un])
            v = float(e.max()) if e.size else 0.0
            return v, v <= self.eff_tol
        if self.stop_with == "C2":
            if state.pf_gap is None:
                return np.nan, False
            v = state.pf_gap / max(state.pf_hat, 1.0 / len(state.mu))
            ok = v <= self.rel_gap_tol
            if self.explore and _no_possible_failure(state):
                ok = False                # vacuous gap: no failure found yet
            return float(v), bool(ok)
        if self.stop_with == "SUR":
            # posterior expected number of misclassified pool points relative
            # to the estimated number of failure points (Gaussian analogue of
            # the relative conformal gap): sum min(p, 1-p) / max(N pf_hat, 1)
            from scipy.stats import norm
            sd = state.std[un]; ok = sd > 0
            p = norm.cdf(-state.mu[un][ok] / sd[ok])
            delta = float(np.sum(np.minimum(p, 1.0 - p)))
            v = delta / max(state.pf_hat * len(state.mu), 1.0)
            return v, v <= self.sur_tol
        return np.nan, False

    def converged(self, state, selection) -> bool:
        v, stop = self._stop_value(state)
        selection.criterion = v          # history records the stopping statistic
        if self.stop_with in ("C2", "SUR") and selection.exhausted:
            return True
        return stop
