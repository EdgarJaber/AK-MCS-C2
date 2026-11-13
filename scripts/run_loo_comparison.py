#!/usr/bin/env python
"""Fixed-hyperparameter LOO vs fully refitted LOO (R4.5).

For each small benchmark and seed: run one AK-MCS-C2 (fixed LOO, J-mm-GP) to
obtain a design sequence (prefixes of X_train are the designs of earlier
iterations), then for each design size n and each variant (J+GP, J-mm-GP)
build the conformal regressor twice -- loo_mode="fixed" and "refit" (full GP
learning, incl. hyper-parameters and standardisation, re-run on each of the n
datasets) -- and compare, on the unevaluated MC cloud:
  scores R_i, interval endpoints, ambiguous set U_t agreement, P_f^-/P_f^+,
  and empirical coverage of each.
Output: loo_comparison/loo_comparison.csv  (one row per problem/seed/n/variant).

  python scripts/run_loo_comparison.py --seeds 5 --sizes 20 30 40 60
"""
import argparse, csv, os, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_experiments import CONFIG, build_algo  # noqa: E402
from akmcsc2 import LOOGPRegressor, make_conformal  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--problems", nargs="+", default=["branch2d_k6", "branch2d_k7", "rastrigin2d"])
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--sizes", type=int, nargs="+", default=[20, 30, 40, 60])
    p.add_argument("--n-mc", type=int, default=CONFIG["n_mc"])
    p.add_argument("--out-dir", default="loo_comparison")
    a = p.parse_args()
    out = Path(a.out_dir); out.mkdir(exist_ok=True)
    rows = []
    for pb in a.problems:
        for seed in range(a.seeds):
            algo = build_algo(pb, "C2-mm", use_cuda=False)
            res = algo.run(seed=seed)
            X_all, y_all = res.X_train, res.y_train
            # same cloud/scaler as the run
            S_raw = res.S_raw
            S = StandardScaler().fit_transform(S_raw)
            g_true = algo.problem(S_raw)
            for n in [k for k in a.sizes if k <= len(y_all)]:
                X, y = X_all[:n], y_all[:n]
                # unevaluated points = cloud minus the n design points
                d = ((S[:, None, :] - X[None, :, :]) ** 2).sum(-1).min(1)
                mask = d > 1e-18
                St, gt = S[mask], g_true[mask]
                gp_cfg = dict(CONFIG["gp"])
                for var in ("j+gp", "j-mm-gp"):
                    res_v = {}
                    for mode in ("fixed", "refit"):
                        base = LOOGPRegressor(**gp_cfg)
                        conf = make_conformal(var, estimator=base,
                                              alpha=CONFIG["alpha"], loo_mode=mode)
                        conf.fit(X, y)
                        iv = conf.predict_interval(St)
                        res_v[mode] = (conf.scores_, iv)
                    (sf, ivf), (sr, ivr) = res_v["fixed"], res_v["refit"]
                    amb = lambda iv: (iv[:, 0] <= 0) & (iv[:, 1] >= 0)
                    af, ar = amb(ivf), amb(ivr)
                    w = lambda iv: float(np.mean(iv[:, 1] - iv[:, 0]))
                    cov = lambda iv: float(np.mean((iv[:, 0] <= gt) & (gt <= iv[:, 1])))
                    rows.append(dict(
                        problem=pb, seed=seed, n=n, variant=var,
                        score_med_absdiff=float(np.median(np.abs(sf - sr))),
                        score_max_absdiff=float(np.max(np.abs(sf - sr))),
                        score_med_reldiff=float(np.median(np.abs(sf - sr) / np.maximum(sr, 1e-12))),
                        lo_med_absdiff=float(np.median(np.abs(ivf[:, 0] - ivr[:, 0]))),
                        up_med_absdiff=float(np.median(np.abs(ivf[:, 1] - ivr[:, 1]))),
                        width_fixed=w(ivf), width_refit=w(ivr),
                        cov_fixed=cov(ivf), cov_refit=cov(ivr),
                        amb_fixed=int(af.sum()), amb_refit=int(ar.sum()),
                        amb_jaccard=float((af & ar).sum() / max((af | ar).sum(), 1)),
                        pf_lower_fixed=float(np.mean(ivf[:, 1] <= 0)), pf_lower_refit=float(np.mean(ivr[:, 1] <= 0)),
                        pf_upper_fixed=float(np.mean(ivf[:, 0] <= 0)), pf_upper_refit=float(np.mean(ivr[:, 0] <= 0)),
                    ))
                    print(pb, seed, n, var, "cov fixed/refit",
                          round(rows[-1]["cov_fixed"], 3), round(rows[-1]["cov_refit"], 3), flush=True)
                    keys = list(rows[0])
                    with open(out / "loo_comparison.csv", "w", newline="") as fh:
                        wr = csv.DictWriter(fh, fieldnames=keys); wr.writeheader(); wr.writerows(rows)
    print("wrote", out / "loo_comparison.csv")


if __name__ == "__main__":
    main()
