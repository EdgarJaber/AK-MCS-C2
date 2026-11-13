#!/usr/bin/env python
"""How much do the practical restrictions of the Python SUR (candidate subset,
active-point truncation, Gauss-Hermite quadrature) change the selected point?

Reference: exact closed form, ALL unevaluated pool points as candidates, ALL
points as integration points. Variants change one restriction at a time.
Reports, over designs of several sizes/seeds: agreement of the selected point
with the reference, and the regret (J(chosen) - J_ref_min)/(H_n - J_ref_min).
  python validation/sur_restrictions_study.py --problems branch2d_k6 oscillator6d --out sur_restrictions.csv
"""
import argparse, csv, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from sklearn.preprocessing import StandardScaler
from akmcsc2 import AKMCS, LOOGPRegressor, get_problem
from akmcsc2.core import IterationState
from akmcsc2.learning import SURCriterion
from run_experiments import CONFIG

p = argparse.ArgumentParser()
p.add_argument("--problems", nargs="+", default=["branch2d_k6", "oscillator6d"])
p.add_argument("--sizes", type=int, nargs="+", default=[30, 50, 80])
p.add_argument("--seeds", type=int, default=3)
p.add_argument("--n-mc", type=int, default=5000)
p.add_argument("--out", default="sur_restrictions.csv")
a = p.parse_args()

def state_at(pb, n, seed):
    gp_cfg = dict(CONFIG["gp"])
    algo = AKMCS(problem=get_problem(pb), learning="U", gp_factory=lambda: LOOGPRegressor(**gp_cfg),
                 n_mc=a.n_mc, n_init=20, max_calls=n, min_iterations=1, predict_batch_size=8192)
    res = algo.run(seed=seed)
    S = StandardScaler().fit_transform(res.S_raw)
    d = ((S[:, None, :] - res.X_train[None, :, :]) ** 2).sum(-1).min(1)
    unused = d > 1e-18
    mu, std = res.gp.predict(S, return_std=True)
    y = np.full(len(S), np.nan); 
    mu = mu.copy(); std = std.copy(); mu[~unused] = 0.0; std[~unused] = 0.0   # evaluated: excluded
    return IterationState(it=0, S_scaled=S, mu=mu, std=std, mask_unused=unused, gp=res.gp,
                          conformal=None, pf_hat=float(np.mean(mu <= 0)))

ref_crit = SURCriterion(candidate_size=10 ** 9, max_active=10 ** 9, z_max=1e9, exact=True)
variants = {
    "cand200":    SURCriterion(candidate_size=200, max_active=10 ** 9, z_max=1e9, exact=True),
    "cand1000":   SURCriterion(candidate_size=1000, max_active=10 ** 9, z_max=1e9, exact=True),
    "active_z6":  SURCriterion(candidate_size=10 ** 9, max_active=5000, z_max=6.0, exact=True),
    "gh15":       SURCriterion(candidate_size=10 ** 9, max_active=10 ** 9, z_max=1e9, exact=False, n_gh=15),
    "gh40":       SURCriterion(candidate_size=10 ** 9, max_active=10 ** 9, z_max=1e9, exact=False, n_gh=40),
    "pilot(200,z6,gh15)": SURCriterion(candidate_size=200, max_active=5000, z_max=6.0, exact=False, n_gh=15),
    "default(200,z6,exact)": SURCriterion(candidate_size=200, max_active=5000, z_max=6.0, exact=True),
}
rows = []
for pb in a.problems:
    for n in a.sizes:
        for seed in range(a.seeds):
            st = state_at(pb, n, seed)
            cand, J, H0 = ref_crit.criterion_values(st)
            jref = int(np.argmin(J)); Jmin = float(J[jref]); best = int(cand[jref])
            lookup = dict(zip(cand.tolist(), J.tolist()))
            for name, vc in variants.items():
                c2, J2, _ = vc.criterion_values(st)
                ch = int(c2[int(np.argmin(J2))])
                regret = (lookup[ch] - Jmin) / max(H0 - Jmin, 1e-300)
                rows.append(dict(problem=pb, n=n, seed=seed, variant=name, same_point=int(ch == best),
                                 regret=regret, H0=H0, Jref_min=Jmin, n_cand_ref=len(cand)))
            print(pb, n, seed, "H0=%.3f Jref=%.3f" % (H0, Jmin), flush=True)
with open(a.out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print("\nvariant: fraction same selected point | median / max regret (fraction of the reference gain H_n - J_min)")
for name in variants:
    r = [x for x in rows if x["variant"] == name]
    print(f"  {name:24s} {np.mean([x['same_point'] for x in r]):.2f} | {np.median([x['regret'] for x in r]):.2e} / {np.max([x['regret'] for x in r]):.2e}")
