"""Reliability benchmark problems.

A :class:`ReliabilityProblem` bundles a limit-state function ``g`` (failure
iff ``g(x) <= 0``), a sampler from the input distribution, the input
dimension and an optional reference failure probability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import numpy as np

__all__ = [
    "ReliabilityProblem",
    "four_branch",
    "modified_rastrigin",
    "linear_oscillator",
    "truss23",
    "get_problem",
    "PROBLEMS",
]


@dataclass
class ReliabilityProblem:
    name: str
    dim: int
    g: Callable[[np.ndarray], np.ndarray]
    sampler: Callable[[int, np.random.Generator], np.ndarray]
    pf_ref: Optional[float] = None
    description: str = ""

    def sample(self, n: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        rng = np.random.default_rng() if rng is None else rng
        return self.sampler(n, rng)

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.g(np.atleast_2d(X)), dtype=float).reshape(-1)


# ---------------------------------------------------------------------------
# 2D four-branch series system
# ---------------------------------------------------------------------------

def four_branch(k: float = 6.0) -> ReliabilityProblem:
    """Two-dimensional four-branch series system (Waarts, 2000)."""

    def g(X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        x1, x2 = X[:, 0], X[:, 1]
        g1 = 3.0 + 0.1 * (x1 - x2) ** 2 - (x1 + x2) / np.sqrt(2.0)
        g2 = 3.0 + 0.1 * (x1 - x2) ** 2 + (x1 + x2) / np.sqrt(2.0)
        g3 = (x1 - x2) + k / np.sqrt(2.0)
        g4 = (x2 - x1) + k / np.sqrt(2.0)
        return np.minimum.reduce([g1, g2, g3, g4])

    def sampler(n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.standard_normal((n, 2))

    pf_ref = {6.0: 4.46e-3, 7.0: 2.23e-3}.get(float(k))
    return ReliabilityProblem(
        name=f"branch2d_k{int(k)}",
        dim=2,
        g=g,
        sampler=sampler,
        pf_ref=pf_ref,
        description=f"2D four-branch series system, k={k}",
    )


# ---------------------------------------------------------------------------
# 2D modified Rastrigin
# ---------------------------------------------------------------------------

def modified_rastrigin() -> ReliabilityProblem:
    """Modified Rastrigin function: disconnected, multimodal failure domain."""

    def g(X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        return 10.0 - np.sum(X**2 - 5.0 * np.cos(2.0 * np.pi * X), axis=1)

    def sampler(n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.standard_normal((n, 2))

    return ReliabilityProblem(
        name="rastrigin2d",
        dim=2,
        g=g,
        sampler=sampler,
        pf_ref=7.3e-2,
        description="2D modified Rastrigin, disjoint failure islands",
    )


# ---------------------------------------------------------------------------
# 6D oscillator
# ---------------------------------------------------------------------------

def linear_oscillator(variant: bool = False) -> ReliabilityProblem:
    """Single-DOF oscillator under transient excitation (Echard et al., 2011):

        g = 3 r - | 2 F1 / (m w0^2) * sin(w0 t1 / 2) |,   w0 = sqrt((c1 + c2) / m).

    Reference P_f = 2.86e-2 (2e7 Monte Carlo samples, seed 123, CoV 0.13%;
    Echard et al. report 2.834e-2).

    ``variant=True`` gives the version used in the revised manuscript, whose
    sine argument is w0^2 t1 / 2 instead of w0 t1 / 2 (P_f = 3.9e-2 with the
    same inputs). It is kept so that the published results can be reproduced
    (problem name ``oscillator6d_variant``).
    """

    def g(X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        c1, c2, m, r, t1, F1 = (X[:, j] for j in range(6))
        w0sq = (c1 + c2) / m
        arg = w0sq * t1 / 2.0 if variant else np.sqrt(w0sq) * t1 / 2.0
        return 3.0 * r - np.abs(2.0 * F1 / (m * w0sq) * np.sin(arg))

    means = np.array([1.0, 0.1, 1.0, 0.5, 1.0, 1.0])
    stds = np.array([0.1, 0.01, 0.05, 0.05, 0.2, 0.2])
    # ordering: c1, c2, m, r, t1, F1

    def sampler(n: int, rng: np.random.Generator) -> np.ndarray:
        return means[None, :] + stds[None, :] * rng.standard_normal((n, 6))

    return ReliabilityProblem(
        name="oscillator6d_variant" if variant else "oscillator6d",
        dim=6,
        g=g,
        sampler=sampler,
        pf_ref=3.9e-2 if variant else 2.86e-2,
        description="6D nonlinear oscillator" + (" (variant: sin(w0^2 t1/2))" if variant else ""),
    )


# ---------------------------------------------------------------------------
# 23-bar planar truss (10 random variables), FE model
# ---------------------------------------------------------------------------

def _lognormal(mean: float, std: float, n: int, rng: np.random.Generator) -> np.ndarray:
    s2 = np.log1p((std / mean) ** 2)
    return np.exp(np.log(mean) - 0.5 * s2 + np.sqrt(s2) * rng.standard_normal(n))


def _gumbel_max(mean: float, std: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """Gumbel (maximum) with given mean and std: scale = std*sqrt(6)/pi."""
    beta = std * np.sqrt(6.0) / np.pi
    return (mean - np.euler_gamma * beta) - beta * np.log(-np.log(rng.uniform(size=n)))


def _truss23_geometry():
    """Warren truss: 7 lower nodes (x=0,4,...,24 m, y=0) and 6 upper nodes
    (x=2,6,...,22 m, y=2 m) = 13 nodes; 6 lower-chord bars, 5 upper-chord bars
    (horizontal: E1, A1) and 12 diagonals (E2, A2) = 23 bars."""
    lower = [(4.0 * i, 0.0) for i in range(7)]
    upper = [(2.0 + 4.0 * i, 2.0) for i in range(6)]
    nodes = np.array(lower + upper)
    bars = [(i, i + 1) for i in range(6)] + [(7 + i, 8 + i) for i in range(5)]
    horizontal = np.array([True] * len(bars))
    for i in range(6):
        bars += [(i, 7 + i), (i + 1, 7 + i)]
    horizontal = np.concatenate([horizontal, np.zeros(12, bool)])
    return nodes, np.array(bars), horizontal


# Reference P_f from 3e7 Monte Carlo samples (seed 123); numbers of failures:
# tau=0.120: 45670, 0.134: 3181 (CoV 1.8%), 0.145: 353 (CoV 5.3%).
_TRUSS_PF_REF = {0.12: 1.52e-3, 0.134: 1.06e-4, 0.145: 1.18e-5}


def truss23(tau: float = 0.12) -> ReliabilityProblem:
    """23-bar truss of Blatman & Sudret (2011), used in Marelli & Sudret (2018):
    failure iff the mid-span vertical deflection exceeds ``tau`` (m).

    Inputs (independent): E1, E2 ~ LN(2.1e11, 2.1e10) Pa; A1 ~ LN(2e-3, 2e-4),
    A2 ~ LN(1e-3, 1e-4) m^2; P1..P6 ~ Gumbel(5e4, 7.5e3) N (vertical, downward,
    applied at the upper nodes). Order: E1, E2, A1, A2, P1..P6.
    g = tau - u_mid.  Geometry as in ``_truss23_geometry`` (span 24 m, height 2 m).
    """
    nodes, bars, horiz = _truss23_geometry()
    nb, nn = len(bars), len(nodes)
    d = nodes[bars[:, 1]] - nodes[bars[:, 0]]
    L = np.hypot(d[:, 0], d[:, 1]); c, s_ = d[:, 0] / L, d[:, 1] / L
    dofs = np.stack([2 * bars[:, 0], 2 * bars[:, 0] + 1, 2 * bars[:, 1], 2 * bars[:, 1] + 1], 1)
    fixed = [0, 1, 2 * 6 + 1]                       # pin at node 0, roller at node 6
    free = np.array([k for k in range(2 * nn) if k not in fixed])
    mid = 2 * 3 + 1                                  # vertical dof of lower mid-span node
    load_dof = np.array([2 * (7 + i) + 1 for i in range(6)])
    # unit direction matrices per bar: k_loc = (EA/L) * v v^T, v = [-c,-s,c,s]
    V = np.stack([-c, -s_, c, s_], 1)                # (nb, 4)
    outer = V[:, :, None] * V[:, None, :]           # (nb, 4, 4)

    def g(X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        out = np.empty(X.shape[0])
        for a in range(0, X.shape[0], 20000):
            B = X[a : a + 20000]; m = B.shape[0]
            E = np.where(horiz[None, :], B[:, [0]], B[:, [1]])
            A = np.where(horiz[None, :], B[:, [2]], B[:, [3]])
            k = (E * A / L[None, :])[:, :, None, None] * outer[None]   # (m, nb, 4, 4)
            K = np.zeros((m, 2 * nn, 2 * nn))
            for b in range(nb):
                idx = dofs[b]
                K[:, idx[:, None], idx[None, :]] += k[:, b]
            F = np.zeros((m, 2 * nn)); F[:, load_dof] = -B[:, 4:10]
            u = np.linalg.solve(K[:, free][:, :, free], F[:, free][..., None])[..., 0]
            full = np.zeros((m, 2 * nn)); full[:, free] = u
            out[a : a + m] = tau + full[:, mid]       # u_mid is negative downwards
        return out

    def sampler(n: int, rng: np.random.Generator) -> np.ndarray:
        cols = [_lognormal(2.1e11, 2.1e10, n, rng), _lognormal(2.1e11, 2.1e10, n, rng),
                _lognormal(2.0e-3, 2.0e-4, n, rng), _lognormal(1.0e-3, 1.0e-4, n, rng)]
        cols += [_gumbel_max(5.0e4, 7.5e3, n, rng) for _ in range(6)]
        return np.column_stack(cols)

    return ReliabilityProblem(
        name=f"truss23_tau{int(round(tau * 1000))}", dim=10, g=g, sampler=sampler,
        pf_ref=_TRUSS_PF_REF.get(round(float(tau), 3)),
        description=f"23-bar truss, tau={tau} m, FE model",
    )


PROBLEMS: Dict[str, Callable[[], ReliabilityProblem]] = {
    "branch2d_k6": lambda: four_branch(6.0),
    "branch2d_k7": lambda: four_branch(7.0),
    "rastrigin2d": modified_rastrigin,
    "oscillator6d": linear_oscillator,
    "oscillator6d_variant": lambda: linear_oscillator(variant=True),
    "truss23_tau120": lambda: truss23(0.12),
    "truss23_tau134": lambda: truss23(0.134),
    "truss23_tau145": lambda: truss23(0.145),
}


def get_problem(name: str) -> ReliabilityProblem:
    try:
        return PROBLEMS[name]()
    except KeyError as e:
        raise KeyError(
            f"Unknown problem '{name}'. Available: {sorted(PROBLEMS)}"
        ) from e
