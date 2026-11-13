#!/usr/bin/env python
"""Summarize results_sensitivity/ -> sensitivity_summary.csv + printed table.
Per (problem, setting): median/q75 |rel. err.| vs the MC reference on the same
population, median n_calls_total, converged fraction, conformal coverage at
stopping, frequency of P_ref in [P-, P+].  python scripts/summarize_sensitivity.py"""
import argparse, csv, glob
from collections import defaultdict
from pathlib import Path
import numpy as np
p = argparse.ArgumentParser(); p.add_argument("--results", default="results_sensitivity"); a = p.parse_args()
R = defaultdict(list)
for f in sorted(glob.glob(str(Path(a.results) / "*.npz"))):
    pb, st, _ = Path(f).stem.split("__")
    with np.load(f, allow_pickle=True) as d:
        R[(pb, st)].append({k: d[k] for k in d.files})
rows = []
for (pb, st), rs in sorted(R.items()):
    ref = np.array([float(r["pf_mc_ref"]) for r in rs]); pf = np.array([float(r["pf"]) for r in rs])
    err = 100 * np.abs(pf - ref) / np.where(ref > 0, ref, np.nan)
    cols = list(rs[0]["diag_cols"]); cov = np.array([r["diag"][cols.index("conf_cov_all")] for r in rs])
    pl = np.array([float(r["pf_lower"]) for r in rs]); pu = np.array([float(r["pf_upper"]) for r in rs])
    rows.append(dict(problem=pb, setting=st, n_runs=len(rs), err_med=np.nanmedian(err), err_q75=np.nanpercentile(err, 75),
                     ncalls_med=np.median([int(r["n_calls_total"]) for r in rs]),
                     conv=np.mean([bool(r["converged"]) for r in rs]), cov_med=np.nanmedian(cov),
                     ref_in_bounds=float(np.mean((pl <= ref) & (ref <= pu)))))
out = Path(a.results) / "sensitivity_summary.csv"
with open(out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
for pb in sorted({r["problem"] for r in rows}):
    print(f"== {pb}: setting | n | |err|% med/q75 | calls | conv | cov | ref in bounds")
    for r in rows:
        if r["problem"] == pb:
            print(f"  {r['setting']:14s} {r['n_runs']:3d} | {r['err_med']:6.2f}/{r['err_q75']:6.2f} | {r['ncalls_med']:5.0f} | {r['conv']:.2f} | {r['cov_med']:.3f} | {r['ref_in_bounds']:.2f}")
print("wrote", out)
