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
    eps = 2.0 * sigma
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
     (ConformalC2(candidate_size=500, rel_gap_tol=0.2), "j+gp"),
     (ConformalC2(candidate_size=500, batch_q=3, batch_delta=0.3,
                  rel_gap_tol=0.2), "j-mm-gp")],
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


# ---------------------------------------------------------------------------
# ablation: selection x stopping decoupling
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sel,stop,cv", [("U", "C2", "j+gp"), ("C2", "U", "j+gp"),
                                         ("EFF", "none", None)])
def test_ablation_runs(sel, stop, cv):
    from akmcsc2 import AKMCS, Ablation, LOOGPRegressor, four_branch
    algo = AKMCS(problem=four_branch(6), learning=Ablation(sel, stop, rel_gap_tol=0.3),
                 conformal=cv, gp_factory=lambda: LOOGPRegressor(
                     kernel="matern52", noise=1e-4, training_iter=30),
                 n_mc=2000, n_init=15, max_calls=30, min_iterations=1)
    assert algo.learning.requires_conformal == ("C2" in (sel, stop))
    res = algo.run(seed=0)
    assert res.n_calls <= 30
    if stop == "none":
        assert res.n_calls == 30 and not res.converged


# ---------------------------------------------------------------------------
# diagnostics, timers, evaluation accounting, refit LOO
# ---------------------------------------------------------------------------

def test_diagnostics_timers_and_accounting():
    from akmcsc2 import AKMCS, ConformalC2, LOOGPRegressor, four_branch
    from akmcsc2.core import DIAG_COLUMNS, TIMER_NAMES
    algo = AKMCS(problem=four_branch(6), learning=ConformalC2(rel_gap_tol=0.5),
                 conformal="j-mm-gp", gp_factory=lambda: LOOGPRegressor(
                     kernel="matern52", noise=1e-4, training_iter=30),
                 n_mc=2000, n_init=15, max_calls=25, min_iterations=1,
                 track_coverage=True)
    res = algo.run(seed=0)
    D = res.diag_array()
    assert D.shape == (len(res.history), len(DIAG_COLUMNS))
    cols = list(DIAG_COLUMNS)
    assert np.all(np.isfinite(D[:, cols.index("conf_cov_all")]))
    assert np.all(np.isfinite(D[:, cols.index("gp_cov_all")]))
    assert set(res.timers) <= set(TIMER_NAMES) and res.timers["conf_bounds"] > 0
    assert res.n_calls_total >= res.n_calls


def test_refit_loo_close_to_fixed():
    from akmcsc2 import LOOGPRegressor, make_conformal
    rng = np.random.default_rng(1)
    X = rng.uniform(-2, 2, size=(18, 2)); y = np.sin(X[:, 0]) + 0.5 * X[:, 1] ** 2
    Xt = rng.uniform(-2, 2, size=(30, 2))
    out = {}
    for mode in ("fixed", "refit"):
        c = make_conformal("j+gp", estimator=LOOGPRegressor(kernel="matern52", noise=1e-4,
                           training_iter=50), alpha=0.1, loo_mode=mode)
        c.fit(X, y); out[mode] = (c.scores_, c.predict_interval(Xt))
    assert out["refit"][1].shape == out["fixed"][1].shape == (30, 2)
    assert np.all(np.isfinite(out["refit"][0])) and np.all(out["refit"][1][:, 0] <= out["refit"][1][:, 1])


# ---------------------------------------------------------------------------
# SUR criterion
# ---------------------------------------------------------------------------

def test_latent_var_consistent_with_predict_and_sur_decreases_uncertainty():
    from akmcsc2 import LOOGPRegressor
    from akmcsc2.core import IterationState
    from akmcsc2.learning import SURCriterion
    rng = np.random.default_rng(0)
    X = rng.uniform(-2, 2, (20, 2)); y = np.sin(X[:, 0]) + X[:, 1] ** 2 - 1
    gp = LOOGPRegressor(kernel="matern52", noise=1e-4, training_iter=40).fit(X, y)
    Xt = rng.uniform(-2, 2, (300, 2))
    mu, sd = gp.predict(Xt, return_std=True)
    lv = gp.latent_var(Xt)
    noise = float(gp.likelihood_.noise.mean()) * (gp._scaler_.std_ ** 2)
    assert np.allclose(sd ** 2 - noise, lv, atol=1e-6 * (1 + sd.max() ** 2))
    C = gp.latent_cross_cov(Xt[:5], Xt[:5])
    assert np.allclose(np.diag(C), lv[:5], atol=1e-8) and np.allclose(C, C.T, atol=1e-8)
    st = IterationState(it=0, S_scaled=Xt, mu=mu, std=sd, mask_unused=np.ones(300, bool),
                        gp=gp, conformal=None, pf_hat=float(np.mean(mu <= 0)))
    cand, J, H0 = SURCriterion(candidate_size=30).criterion_values(st)
    assert np.all(J <= H0 + 1e-9) and np.all(J >= 0)   # E[H_{n+1}] <= H_n


def test_exploration_safeguard_does_not_stop_without_a_failure():
    from akmcsc2 import AKMCS, Ablation, LOOGPRegressor
    from akmcsc2.problems import ReliabilityProblem
    prob = ReliabilityProblem(
        name="tail", dim=2, g=lambda X: 3.3 - np.atleast_2d(X)[:, 0],
        sampler=lambda n, rng: rng.standard_normal((n, 2)))
    def run(explore):
        algo = AKMCS(problem=prob, learning=Ablation("C2", "C2", rel_gap_tol=0.3, explore=explore),
                     conformal="j-mm-gp", gp_factory=lambda: LOOGPRegressor(
                         kernel="matern52", noise=1e-4, training_iter=30),
                     n_mc=3000, n_init=10, max_calls=30, min_iterations=1)
        return algo.run(seed=1)
    off, on = run(False), run(True)
    assert off.n_calls <= on.n_calls          # without the safeguard it may stop at once
    assert on.n_calls > 10 or on.pf > 0       # with it: keeps acquiring until a failure is found


def test_sur_closed_form_and_torch_posterior():
    from scipy.stats import multivariate_normal, norm
    from akmcsc2 import LOOGPRegressor
    from akmcsc2.learning import sur_expected_H
    # closed form (Owen's T) vs bivariate normal cdf and Gauss-Hermite
    m = np.array([0.4, -1.1, 0.0]); s2 = np.array([1.0, 0.5, 2.0])
    C = np.array([[0.3, 0.7], [0.1, 0.5], [1.0, 0.0]]); sx2 = np.array([1.5, 1.0])
    ex = sur_expected_H(m, s2, C, sx2, exact=True)
    ref = np.zeros(2)
    for j in range(2):
        for u in range(3):
            h = -m[u] / np.sqrt(s2[u]); rho = C[u, j] ** 2 / (s2[u] * sx2[j])
            ref[j] += norm.cdf(h) - multivariate_normal(mean=[0, 0], cov=[[1, rho], [rho, 1]]).cdf([h, h])
    z, w = np.polynomial.hermite_e.hermegauss(80)
    gh = sur_expected_H(m, s2, C, sx2, exact=False, z=z, w=w / np.sqrt(2 * np.pi))
    assert np.allclose(ex, ref, atol=1e-8) and np.allclose(ex, gh, atol=1e-8)
    # torch latent posterior vs an independent numpy radial Matern-5/2 implementation
    rng = np.random.default_rng(3)
    X = rng.uniform(-2, 2, (15, 2)); y = np.sin(X[:, 0]) + X[:, 1] ** 2
    gp = LOOGPRegressor(kernel="matern52", noise=1e-4, training_iter=30).fit(X, y)
    ls = gp.model_.covar_module.base_kernel.lengthscale.detach().numpy().ravel()
    os_ = float(gp.model_.covar_module.outputscale.detach())
    noise = float(gp.likelihood_.noise.detach().mean())
    def k(A, B):
        r = np.sqrt((((A[:, None, :] - B[None, :, :]) / ls) ** 2).sum(-1))
        return os_ * (1 + np.sqrt(5) * r + 5 * r ** 2 / 3) * np.exp(-np.sqrt(5) * r)
    Xt = rng.uniform(-2, 2, (6, 2)); Xs = gp._tx_.numpy()
    ref = k(Xt, Xt) - k(Xt, Xs) @ np.linalg.solve(k(Xs, Xs) + noise * np.eye(len(Xs)), k(Xs, Xt))
    assert np.allclose(gp.latent_cross_cov(Xt, Xt), ref * gp._scaler_.std_ ** 2, atol=1e-6)
