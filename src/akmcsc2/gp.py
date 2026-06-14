"""GPyTorch exact-GP regressors with closed-form leave-one-out predictors.

The key extension over a vanilla GP wrapper is
:meth:`LOOGPRegressor.loo_predict_at`, which returns the *n* leave-one-out
posterior means and standard deviations evaluated at arbitrary test points,
vectorized, from a single Cholesky factorization. These are the quantities
required by the Jackknife+ family of conformal estimators.

LOO identities (Sherman--Morrison on the deleted row/column of
``C = K + noise*I``, with ``alpha = C^{-1}(y - m)`` and
``q_i = (C^{-1})_{ii}``, ``c_i(x) = [C^{-1} k_n(x)]_i``)::

    mu_{-i}(x)    = mu(x)    - c_i(x) * alpha_i / q_i
    sigma2_{-i}(x) = sigma2(x) + c_i(x)**2 / q_i

At a training point ``x = x_i`` these reduce to the classical Dubrule
formulas ``mu_{-i}(x_i) = y_i - alpha_i / q_i`` and
``sigma2_{-i}(x_i) = 1 / q_i``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import torch
import gpytorch
from sklearn.base import BaseEstimator, RegressorMixin

__all__ = ["GPRegressor", "LOOGPRegressor"]


def _as_2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return x.reshape(-1, 1) if x.ndim == 1 else x


@dataclass
class _YScaler:
    mean_: float
    std_: float

    @staticmethod
    def fit(y: np.ndarray, eps: float = 1e-12) -> "_YScaler":
        s = float(np.std(y))
        return _YScaler(mean_=float(np.mean(y)), std_=s if s > eps else 1.0)

    def transform(self, y):
        return (y - self.mean_) / self.std_

    def inv_mean(self, y):
        return y * self.std_ + self.mean_

    def inv_std(self, s):
        return s * self.std_


_KERNELS = {
    "rbf": lambda ard_dims: gpytorch.kernels.RBFKernel(ard_num_dims=ard_dims),
    "matern12": lambda ard_dims: gpytorch.kernels.MaternKernel(nu=0.5, ard_num_dims=ard_dims),
    "matern32": lambda ard_dims: gpytorch.kernels.MaternKernel(nu=1.5, ard_num_dims=ard_dims),
    "matern52": lambda ard_dims: gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=ard_dims),
}


class _ExactGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, mean, kernel, ard):
        super().__init__(train_x, train_y, likelihood)
        if mean == "constant":
            self.mean_module = gpytorch.means.ConstantMean()
        elif mean == "linear":
            self.mean_module = gpytorch.means.LinearMean(train_x.size(-1))
        else:
            raise ValueError("mean must be 'constant' or 'linear'")
        try:
            base = _KERNELS[kernel](train_x.size(-1) if ard else None)
        except KeyError as e:
            raise ValueError(f"kernel must be one of {sorted(_KERNELS)}") from e
        self.covar_module = gpytorch.kernels.ScaleKernel(base)

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x)
        )


class GPRegressor(BaseEstimator, RegressorMixin):
    """Exact GP regressor (GPyTorch backend), scikit-learn style.

    Parameters
    ----------
    mean : {'constant', 'linear'}
    kernel : {'rbf', 'matern12', 'matern32', 'matern52'}
    ard : bool
        Anisotropic lengthscales.
    noise : float
        Initial (or fixed, if ``learn_noise=False``) Gaussian-likelihood
        noise variance. Acts as the nugget regularization of the paper.
    learn_noise : bool
    standardize_y : bool
    training_iter, lr : Adam MLE optimization of hyperparameters.
    use_cuda : bool
    dtype : torch dtype, float64 by default (LOO algebra is
        ill-conditioned in float32 for small nuggets).
    """

    def __init__(
        self,
        mean: str = "constant",
        kernel: str = "matern52",
        ard: bool = True,
        noise: Optional[float] = 1e-4,
        learn_noise: bool = True,
        standardize_y: bool = True,
        training_iter: int = 100,
        lr: float = 0.1,
        use_cuda: bool = False,
        dtype: torch.dtype = torch.float64,
        seed: int = 0,
    ) -> None:
        self.mean = mean
        self.kernel = kernel
        self.ard = ard
        self.noise = noise
        self.learn_noise = learn_noise
        self.standardize_y = standardize_y
        self.training_iter = training_iter
        self.lr = lr
        self.use_cuda = use_cuda
        self.dtype = dtype
        self.seed = seed

    # -- fitting -----------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> "GPRegressor":
        X = _as_2d(X)
        y = np.asarray(y, dtype=float).reshape(-1)
        self.X_train_, self.y_train_ = X.copy(), y.copy()

        torch.manual_seed(self.seed)
        self._scaler_ = _YScaler.fit(y) if self.standardize_y else None
        y_used = self._scaler_.transform(y) if self._scaler_ else y

        self.device_ = torch.device(
            "cuda" if (self.use_cuda and torch.cuda.is_available()) else "cpu"
        )
        tx = torch.as_tensor(X, dtype=self.dtype, device=self.device_)
        ty = torch.as_tensor(y_used, dtype=self.dtype, device=self.device_)

        lik = gpytorch.likelihoods.GaussianLikelihood().to(self.device_, self.dtype)
        if self.noise is not None:
            lik.noise = torch.tensor(float(self.noise), dtype=self.dtype, device=self.device_)
        if not self.learn_noise:
            lik.raw_noise.requires_grad_(False)

        model = _ExactGP(tx, ty, lik, self.mean, self.kernel, self.ard).to(
            self.device_, self.dtype
        )
        model.train(); lik.train()
        opt = torch.optim.Adam(model.parameters(), lr=self.lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(lik, model)
        with gpytorch.settings.cholesky_jitter(1e-8):
            for _ in range(self.training_iter):
                opt.zero_grad(set_to_none=True)
                loss = -mll(model(tx), ty)
                loss.backward()
                opt.step()

        self.model_, self.likelihood_ = model, lik
        self._tx_, self._ty_ = tx, ty
        self._post_fit()
        self.trained_ = True
        return self

    def _post_fit(self) -> None:  # hook for subclasses
        pass

    # -- prediction --------------------------------------------------------
    def predict(
        self, X: np.ndarray, return_std: bool = False, batch_size: int = 65536
    ):
        self._check_fitted()
        X = _as_2d(X)
        means, stds = [], []
        self.model_.eval(); self.likelihood_.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            for i in range(0, X.shape[0], batch_size):
                x = torch.as_tensor(
                    X[i : i + batch_size], dtype=self.dtype, device=self.device_
                )
                pred = self.likelihood_(self.model_(x))
                means.append(pred.mean.cpu().numpy())
                stds.append(np.sqrt(np.maximum(pred.variance.cpu().numpy(), 0.0)))
        mean = np.concatenate(means)
        std = np.concatenate(stds)
        if self._scaler_:
            mean, std = self._scaler_.inv_mean(mean), self._scaler_.inv_std(std)
        return (mean, std) if return_std else mean

    def _check_fitted(self):
        if not getattr(self, "trained_", False):
            raise RuntimeError("Call fit() first.")

    def __sklearn_is_fitted__(self):
        return bool(getattr(self, "trained_", False))


class LOOGPRegressor(GPRegressor):
    """GP regressor exposing analytic LOO predictors, including at test points."""

    def _post_fit(self) -> None:
        """Cache the Cholesky factor and LOO quantities at training points."""
        self.model_.eval(); self.likelihood_.eval()
        tx, ty = self._tx_, self._ty_
        n = tx.shape[0]
        with torch.no_grad():
            m = self.model_.mean_module(tx)
            K = self.model_.covar_module(tx).evaluate()
            noise = self.likelihood_.noise
            noise = noise.mean() if torch.numel(noise) > 1 else noise
            C = K + noise * torch.eye(n, dtype=self.dtype, device=self.device_)
            L = torch.linalg.cholesky(C)
            resid = (ty - m).unsqueeze(-1)
            alpha = torch.cholesky_solve(resid, L).squeeze(-1)
            Cinv = torch.cholesky_inverse(L)
            q = torch.diag(Cinv).clamp_min(1e-14)

        self._L_, self._alpha_, self._q_ = L, alpha, q

        loo_mean_u = (ty - alpha / q).cpu().numpy()
        loo_std_u = torch.sqrt(1.0 / q).cpu().numpy()
        if self._scaler_:
            self.loo_mean_ = self._scaler_.inv_mean(loo_mean_u)
            self.loo_std_ = self._scaler_.inv_std(loo_std_u)
        else:
            self.loo_mean_, self.loo_std_ = loo_mean_u, loo_std_u
        self.loo_residuals_ = self.y_train_ - self.loo_mean_

    def loo_predict(self, return_std: bool = False):
        """LOO predictors at the *training* points (classical Dubrule formulas)."""
        self._check_fitted()
        if return_std:
            return self.loo_mean_.copy(), self.loo_std_.copy()
        return self.loo_mean_.copy()

    def loo_predict_at(
        self, X: np.ndarray, batch_size: int = 8192
    ) -> Tuple[np.ndarray, np.ndarray]:
        """All n LOO posterior means/stds at test points X.

        Returns
        -------
        mu_loo, std_loo : ndarray of shape (m, n)
            ``mu_loo[j, i] = mu_{-i}(x_j)``, in the original (unscaled) y units.

        Notes
        -----
        Memory is O(batch_size * n); with n <= a few hundred design points and
        batch_size ~ 8k this stays well below 100 MB in float64.
        """
        self._check_fitted()
        X = _as_2d(X)
        m_total = X.shape[0]
        n = self._tx_.shape[0]
        out_mu = np.empty((m_total, n), dtype=float)
        out_std = np.empty((m_total, n), dtype=float)

        alpha_over_q = (self._alpha_ / self._q_).unsqueeze(0)  # (1, n)
        inv_q = (1.0 / self._q_).unsqueeze(0)                   # (1, n)

        self.model_.eval()
        with torch.no_grad():
            for i in range(0, m_total, batch_size):
                x = torch.as_tensor(
                    X[i : i + batch_size], dtype=self.dtype, device=self.device_
                )
                # latent full posterior (mean module + kernel algebra by hand,
                # so that mean/var and cross-covariances are consistent)
                k_xt = self.model_.covar_module(self._tx_, x).evaluate()  # (n, m)
                mean_x = self.model_.mean_module(x)
                kxx = self.model_.covar_module(x).diagonal()              # prior var
                c = torch.cholesky_solve(k_xt, self._L_)                  # C^{-1} k (n, m)
                mu_full = mean_x + (k_xt * self._alpha_.unsqueeze(-1)).sum(0)
                var_full = (kxx - (k_xt * c).sum(0)).clamp_min(0.0)

                cT = c.transpose(0, 1)                                    # (m, n)
                mu_loo = mu_full.unsqueeze(-1) - cT * alpha_over_q        # (m, n)
                var_loo = var_full.unsqueeze(-1) + cT.pow(2) * inv_q
                std_loo = torch.sqrt(var_loo.clamp_min(0.0))

                out_mu[i : i + x.shape[0]] = mu_loo.cpu().numpy()
                out_std[i : i + x.shape[0]] = std_loo.cpu().numpy()

        if self._scaler_:
            out_mu = self._scaler_.inv_mean(out_mu)
            out_std = self._scaler_.inv_std(out_std)
        return out_mu, out_std
