"""Cross-conformal estimators for GP surrogates (J+GP and J-minmax-GP).

Implements the constructions of Jaber et al. (2025): normalized LOO
non-conformity scores

    R_i = |y_i - mu_{-i}(x_i)| / max(eps, sigma_{-i}(x_i)),

and, for a test point x:

J+GP (1 - 2*alpha marginal guarantee, ~1 - alpha empirically)::

    C(x) = [ q^-_{n,alpha}{ mu_{-i}(x) - R_i * max(eps, sigma_{-i}(x)) },
             q^+_{n,alpha}{ mu_{-i}(x) + R_i * max(eps, sigma_{-i}(x)) } ]

J-minmax-GP (1 - alpha marginal guarantee, conservative)::

    C(x) = [ min_i mu_{-i}(x) - q^+_{n,alpha}{ R_i * max(eps, sigma_{-i}(x)) },
             max_i mu_{-i}(x) + q^+_{n,alpha}{ R_i * max(eps, sigma_{-i}(x)) } ]

Both use the genuine LOO predictors *at the test point* obtained in closed
form from :meth:`akmcsc2.gp.LOOGPRegressor.loo_predict_at` -- not the
full-model predictor with a calibrated radius (which would be a split-style
approximation, not Jackknife+).
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone

from .gp import LOOGPRegressor

__all__ = [
    "ConformalGPRegressor",
    "JPlusGP",
    "JMinmaxGP",
    "make_conformal",
]


def _plus_rank(n: int, alpha: float) -> int:
    """0-based index of the ceil((1-alpha)(n+1))-th smallest of n values."""
    k = int(math.ceil((1.0 - alpha) * (n + 1)))
    return min(max(k, 1), n) - 1


def _minus_rank(n: int, alpha: float) -> int:
    """0-based index of the floor(alpha(n+1))-th smallest of n values."""
    k = int(math.floor(alpha * (n + 1)))
    return min(max(k, 1), n) - 1


class ConformalGPRegressor(BaseEstimator, RegressorMixin, ABC):
    """Base class for GP cross-conformal regressors.

    Parameters
    ----------
    estimator : LOOGPRegressor, optional
        Prototype base GP (cloned at ``fit``). A default is built if None.
    alpha : float
        Miscoverage level.
    eps : float
        Score-normalization floor, ``max(eps, sigma)``.
    prefit : bool
        If True, ``estimator`` is assumed already fitted on the same data and
        is reused without cloning/refitting. This is the hook used by the
        AK-MCS loop to share a single GP fit per iteration between the
        learning function, the stopping bounds and the point estimator.
    """

    def __init__(
        self,
        estimator: Optional[LOOGPRegressor] = None,
        alpha: float = 0.1,
        eps: float = 1e-12,
        prefit: bool = False,
    ) -> None:
        self.estimator = estimator
        self.alpha = alpha
        self.eps = eps
        self.prefit = prefit

    # -- fitting -----------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> "ConformalGPRegressor":
        if self.prefit:
            if self.estimator is None or not getattr(self.estimator, "trained_", False):
                raise RuntimeError("prefit=True requires a fitted estimator.")
            self.estimator_ = self.estimator
        else:
            self.estimator_ = (
                clone(self.estimator) if self.estimator is not None else LOOGPRegressor()
            )
            self.estimator_.fit(X, y)

        loo_mean, loo_std = self.estimator_.loo_predict(return_std=True)
        y = np.asarray(y, dtype=float).reshape(-1)
        self.scores_ = np.abs(y - loo_mean) / np.maximum(loo_std, self.eps)
        self.n_ = len(self.scores_)
        self.trained_ = True
        return self

    # -- prediction --------------------------------------------------------
    def predict(self, X: np.ndarray, batch_size: int = 8192) -> np.ndarray:
        self._check_fitted()
        return self.estimator_.predict(X, batch_size=batch_size)

    def predict_interval(
        self, X: np.ndarray, batch_size: int = 8192
    ) -> np.ndarray:
        """Conformal prediction intervals, shape (m, 2)."""
        self._check_fitted()
        X = np.atleast_2d(np.asarray(X, dtype=float))
        out = np.empty((X.shape[0], 2), dtype=float)
        for i in range(0, X.shape[0], batch_size):
            mu_loo, std_loo = self.estimator_.loo_predict_at(
                X[i : i + batch_size], batch_size=batch_size
            )
            out[i : i + mu_loo.shape[0]] = self._interval_from_loo(mu_loo, std_loo)
        return out

    @abstractmethod
    def _interval_from_loo(
        self, mu_loo: np.ndarray, std_loo: np.ndarray
    ) -> np.ndarray:
        """(m, n) LOO predictors -> (m, 2) intervals."""

    def _check_fitted(self):
        if not getattr(self, "trained_", False):
            raise RuntimeError("Call fit() first.")

    def __sklearn_is_fitted__(self):
        return bool(getattr(self, "trained_", False))


class JPlusGP(ConformalGPRegressor):
    """Jackknife+ GP estimator (J+GP) of Jaber et al. (2025)."""

    def _interval_from_loo(self, mu_loo, std_loo):
        w = self.scores_[None, :] * np.maximum(std_loo, self.eps)  # (m, n)
        lo_vals = mu_loo - w
        up_vals = mu_loo + w
        n = self.n_
        k_lo = _minus_rank(n, self.alpha)
        k_up = _plus_rank(n, self.alpha)
        lower = np.partition(lo_vals, k_lo, axis=1)[:, k_lo]
        upper = np.partition(up_vals, k_up, axis=1)[:, k_up]
        return np.column_stack([lower, upper])


class JMinmaxGP(ConformalGPRegressor):
    """Jackknife-minmax GP estimator (J-minmax-GP), 1-alpha guarantee."""

    def _interval_from_loo(self, mu_loo, std_loo):
        w = self.scores_[None, :] * np.maximum(std_loo, self.eps)  # (m, n)
        k_up = _plus_rank(self.n_, self.alpha)
        radius = np.partition(w, k_up, axis=1)[:, k_up]
        lower = mu_loo.min(axis=1) - radius
        upper = mu_loo.max(axis=1) + radius
        return np.column_stack([lower, upper])


_CONFORMAL = {"j+gp": JPlusGP, "j-mm-gp": JMinmaxGP, "jminmax": JMinmaxGP}


def make_conformal(variant: str, **kwargs) -> ConformalGPRegressor:
    try:
        return _CONFORMAL[variant.lower()](**kwargs)
    except KeyError as e:
        raise KeyError(
            f"Unknown conformal variant '{variant}'. Available: {sorted(_CONFORMAL)}"
        ) from e
