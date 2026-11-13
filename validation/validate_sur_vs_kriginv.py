#!/usr/bin/env python
"""Compare the Python SUR criterion with KrigInv's, on the CSVs written by
kriginv_reference.R (same DiceKriging model with fixed hyper-parameters).

Checks (max relative differences are printed):
  A. criterion  J(x)  from R's intermediates (post. mean/sd at U, cov(U,x), var(x))
     -> exact closed form (Owen's T), Gauss-Hermite quadrature, vs R's value;
  B. posterior ingredients recomputed independently in numpy (product
     Matern-5/2, universal kriging as in DiceKriging) vs R's mean/sd/kn/sx2.
  python validation/validate_sur_vs_kriginv.py kriginv_ref
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from akmcsc2.learning import sur_expected_H  # noqa: E402

D = Path(sys.argv[1] if len(sys.argv) > 1 else "kriginv_ref")
ld = lambda n: np.loadtxt(D / n, delimiter=",", ndmin=2)
X, y, U, Xc = ld("X.csv"), ld("y.csv").ravel(), ld("U.csv"), ld("Xc.csv")
post = ld("post_U.csv"); KN = ld("KN.csv"); sx2 = ld("sx2.csv").ravel()
crit_all = ld("crit.csv").ravel(); H_R, crit_R = crit_all[0], crit_all[1:]
theta, sig2, mu0, nug = ld("params.csv").ravel()[[0, 1]], *ld("params.csv").ravel()[2:]
m, s = post[:, 0], post[:, 1]
rel = lambda a, b: float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-12)))

# A. criterion math from R's own intermediates -------------------------------
zq, wq = np.polynomial.hermite_e.hermegauss(60); wq = wq / np.sqrt(2 * np.pi)
Jex = sur_expected_H(m, s ** 2, KN, sx2, exact=True) / len(m)
Jgh = sur_expected_H(m, s ** 2, KN, sx2, exact=False, z=zq, w=wq) / len(m)
print(f"A. exact  vs KrigInv: max rel diff = {rel(Jex, crit_R):.2e}")
print(f"A. GH(60) vs KrigInv: max rel diff = {rel(Jgh, crit_R):.2e}")
from scipy.stats import norm
H_py = float(np.mean(norm.cdf(m / s) * (1 - norm.cdf(m / s))))
print(f"A. H_n: python {H_py:.8f}  KrigInv {H_R:.8f}")
print(f"A. argmin agrees: {int(np.argmin(Jex)) == int(np.argmin(crit_R))}  (candidate {int(np.argmin(crit_R))})")

# B. posterior ingredients, independent numpy implementation -------------------
def k(A, B):
    h = np.abs(A[:, None, :] - B[None, :, :]) / theta[None, None, :]
    return sig2 * np.prod((1 + np.sqrt(5) * h + 5 * h ** 2 / 3) * np.exp(-np.sqrt(5) * h), axis=2)
n = len(y); Kxx = k(X, X) + nug * np.eye(n); Ki = np.linalg.inv(Kxx); F = np.ones((n, 1))
A = float((F.T @ Ki @ F)[0, 0])
def uk_cov(P, Q):
    cP, cQ = k(X, P), k(X, Q)
    return k(P, Q) - cP.T @ Ki @ cQ + np.outer(1 - (F.T @ Ki @ cP).ravel(), 1 - (F.T @ Ki @ cQ).ravel()) / A
c_u = k(X, U)
mean_py = mu0 + c_u.T @ Ki @ (y - mu0)
var_py = sig2 + nug - np.sum(c_u * (Ki @ c_u), 0) + (1 - (F.T @ Ki @ c_u).ravel()) ** 2 / A
print(f"B. mean  max abs diff = {np.max(np.abs(mean_py - m)):.2e}   sd max rel diff = {rel(np.sqrt(var_py), s):.2e}")
KN_py = uk_cov(U, Xc)
print(f"B. cov(U,x) max abs diff = {np.max(np.abs(KN_py - KN)):.2e}   var(x) max rel diff = {rel(np.diag(uk_cov(Xc, Xc)), sx2):.2e}")
Jpy = sur_expected_H(mean_py, np.diag(uk_cov(U, U)) if False else (var_py - nug), KN_py, np.diag(uk_cov(Xc, Xc)), exact=True) / len(m)
print(f"B. criterion fully recomputed in numpy vs KrigInv: max rel diff = {rel(Jpy, crit_R):.2e}")
