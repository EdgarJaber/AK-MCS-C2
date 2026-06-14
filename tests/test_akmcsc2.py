"""Smoke + correctness tests.

Run with:  pytest tests/ -q
"""

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import norm

from akmcsc2 import (
    AKMCS,
    ConformalC2,
    ExpectedFeasibility,
    JMinmaxGP,
    JPlusGP,
    LOOGPRegressor,
    four_branch,
)


# ---------------------------------------------------------------------------
# EFF closed form vs quadrature
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mu,sigma", [(0.0, 1.0), (1.0, 0.7), (-2.0, 1.5), (0.3, 0.2)])
def test_eff_matches_quadrature(mu, sigma):
    eps = 2.0 * sigma**2
    val = ExpectedFeasibility._eff(np.array([mu]), np.array([sigma]))[0]
    ref, _ = quad(
        lambda g: max(eps - abs(g), 0.0) * norm.pdf(g, loc=mu, scale=sigma),
        mu - 10 * sigma, mu + 10 * sigma,
    )
    assert val == pytest.approx(ref, rel=1e-6, abs=1e-12)


# ---------------------------------------------------------------------------
# LOO-at-test identities vs brute-force retraining
# ---------------------------------------------------------------------------

def test_loo_predict_at_matches_retrained_models():
    rng = np.random.default_rng(0)
    X = rng.uniform(-2, 2, size=(15, 2))
    y = np.sin(X[:, 0]) + 0.5 * X[:, 1] ** 2
    Xt = rng.uniform(-2, 2, size=(7, 2))

    # fixed hyperparameters so that the n retrained models share them
    common = dict(
        kernel="rbf", ard=False, noise=1e-3, learn_noise=False,
        standardize_y=False, training_iter=0,
    )

    full = LOOGPRegressor(**common).fit(X, y)
    mu_loo, std_loo = full.loo_predict_at(Xt)

    for i in range(len(y)):
        keep = np.ones(len(y), dtype=bool)
        keep[i] = False
        gp_i = LOOGPRegressor(**common).fit(X[keep], y[keep])
        m_i, s_i = gp_i.predict(Xt, return_std=True)
        # gp_i.predict includes likelihood noise in the variance; compare
        # latent variances by removing the (fixed) nugget from both sides
        assert np.allclose(mu_loo[:, i], m_i, atol=1e-6), f"mean mismatch, i={i}"
        lat_i = np.sqrt(np.maximum(s_i**2 - 1e-3, 0.0))
        assert np.allclose(std_loo[:, i], lat_i, atol=1e-5), f"std mismatch, i={i}"

    # Dubrule formulas at the training points
    mu_tr, std_tr = full.loo_predict(return_std=True)
    mu_loo_tr, _ = full.loo_predict_at(X)
    assert np.allclose(np.diagonal(mu_loo_tr), mu_tr, atol=1e-6)


# ---------------------------------------------------------------------------
# Conformal marginal coverage (exchangeable data)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls,floor", [(JPlusGP, 0.75), (JMinmaxGP, 0.85)])
def test_marginal_coverage(cls, floor):
    """Guarantees are 1-2*alpha (J+) and 1-alpha (minmax) in expectation
    over exchangeable draws; floors include Monte Carlo slack for the
    finite number of (correlated) replications."""
    rng = np.random.default_rng(1)
    alpha = 0.1

    def f(X):
        return np.sin(3 * X[:, 0]) + X[:, 1]

    covered = []
    for rep in range(20):
        X = rng.uniform(-1, 1, size=(30, 2))
        y = f(X)
        Xt = rng.uniform(-1, 1, size=(200, 2))
        yt = f(Xt)
        conf = cls(
            estimator=LOOGPRegressor(kernel="rbf", noise=1e-4, training_iter=50),
            alpha=alpha,
        ).fit(X, y)
        itv = conf.predict_interval(Xt)
        covered.append(np.mean((itv[:, 0] <= yt) & (yt <= itv[:, 1])))
    cov = float(np.mean(covered))
    assert cov >= floor, f"{cls.__name__}: coverage {cov:.3f} < {floor}"


def test_minmax_contains_jplus():
    rng = np.random.default_rng(2)
    X = rng.uniform(-1, 1, size=(25, 2))
    y = np.cos(2 * X[:, 0]) * X[:, 1]
    Xt = rng.uniform(-1, 1, size=(50, 2))
    gp = LOOGPRegressor(kernel="rbf", noise=1e-4, training_iter=50).fit(X, y)
    jp = JPlusGP(estimator=gp, alpha=0.1, prefit=True).fit(X, y)
    mm = JMinmaxGP(estimator=gp, alpha=0.1, prefit=True).fit(X, y)
    a, b = jp.predict_interval(Xt), mm.predict_interval(Xt)
    assert np.all(b[:, 0] <= a[:, 0] + 1e-9)
    assert np.all(b[:, 1] >= a[:, 1] - 1e-9)


# ---------------------------------------------------------------------------
# End-to-end smoke runs
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "learning,conformal",
    [("U", None), ("EFF", None),
     (ConformalC2(candidate_size=500, pf_gap_tol=2e-3), "j+gp"),
     (ConformalC2(candidate_size=500, batch_q=3, batch_delta=0.3,
                  pf_gap_tol=2e-3), "j-mm-gp")],
)
def test_end_to_end_four_branch(learning, conformal):
    algo = AKMCS(
        problem=four_branch(6.0),
        learning=learning,
        conformal=conformal,
        gp_factory=lambda: LOOGPRegressor(
            kernel="matern52", noise=1e-4, training_iter=30
        ),
        n_mc=3000,
        n_init=12,
        max_calls=40,
        min_iterations=5,
        track_coverage=True,
    )
    res = algo.run(seed=0)
    assert res.n_calls <= 40
    assert len(res.history) >= 1
    assert 0.0 <= res.pf <= 1.0
    arr = res.history_array()
    assert arr.shape[1] == 7
    if conformal is not None:
        h = res.history[-1]
        assert h.pf_lower is not None and h.pf_upper is not None
        assert h.pf_lower <= h.pf_upper + 1e-12
        assert h.emp_coverage is not None
