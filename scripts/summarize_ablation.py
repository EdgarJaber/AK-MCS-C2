#!/usr/bin/env python
"""Summarize results_ablation/ into ablation_summary.csv and print, per
problem, the SEL x STOP grids (median rel. error vs the MC reference on the
same population, median n_calls, fraction converged). With fixed-budget
("none") cells present, also reports the median error at given budgets.

  python scripts/summarize_ablation.py --results results_ablation
"""
import argparse, csv, glob, re
from collections import defaultdict
from pathlib import Path
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--results", default="results_ablation")
p.add_argument("--budgets", type=int, nargs="*", default=[40, 60, 100, 200])
a = p.parse_args()

R = defaultdict(list)
for f in sorted(glob.glob(str(Path(a.results) / "*.npz"))):
    pb, cell, _ = Path(f).stem.split("__")
    with np.load(f, allow_pickle=True) as d:      # close the file (ulimit -n)
        R[(pb, cell)].append({k: d[k] for k in d.files})

rows = []
for (pb, cell), runs in sorted(R.items()):
    ref = np.array([float(r["pf_mc_ref"]) for r in runs])
    pf = np.array([float(r["pf"]) for r in runs])
    err = 100 * (pf - ref) / np.where(ref > 0, ref, np.nan)
    nc = np.array([int(r["n_calls"]) for r in runs])
    cv = np.array([bool(r["converged"]) for r in runs])
    row = dict(problem=pb, cell=cell, n_runs=len(runs),
               pf_median=np.median(pf), err_pct_median=np.nanmedian(err),
               abs_err_pct_median=np.nanmedian(np.abs(err)),
               abs_err_pct_q75=np.nanpercentile(np.abs(err), 75),
               ncalls_median=np.median(nc), converged_frac=cv.mean())
    if cell.split("-")[1] == "none":   # fixed-budget accuracy from histories
        for b in a.budgets:
            e = []
            for r, rf in zip(runs, ref):
                h = r["history"]; k = np.where(h[:, 0] <= b)[0]
                if len(k) and rf > 0:
                    e.append(abs(100 * (h[k[-1], 1] - rf) / rf))
            row[f"abs_err_pct_at_{b}"] = np.median(e) if e else np.nan
    rows.append(row)

keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in
        ("problem", "cell", "n_runs"), k))
out = Path(a.results) / "ablation_summary.csv"
with open(out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
print("wrote", out)

for pb in sorted({r["problem"] for r in rows}):
    print(f"\n=== {pb}: median |rel.err| % / median n_calls / converged ===")
    for r in rows:
        if r["problem"] == pb:
            print(f"  {r['cell']:>10s}  err={r['abs_err_pct_median']:6.2f}  "
                  f"ncalls={r['ncalls_median']:5.0f}  conv={r['converged_frac']:.2f}  n={r['n_runs']}")
