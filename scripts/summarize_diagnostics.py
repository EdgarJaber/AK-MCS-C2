#!/usr/bin/env python
"""Near-boundary coverage (R3.2, R4.2) and GP-vs-conformal intervals (R3.3).

Reads results_ablation/*.npz (files written with diagnostics) and writes
  <results>/diagnostics_final.csv : median of every DIAG column at stopping,
        + frequency (over seeds) of  P_ref in [P-,P+]  (raw bounds),
          P_ref in [P- - beta, P+ + beta] (Proposition 1, beta = conf_miscov),
          P_ref in Gaussian-interval bounds [gp_pf_lower, gp_pf_upper].
  <results>/figures/<problem>__<cell>_coverage.pdf : median coverage vs n_calls,
        global vs near-boundary (|mu|<=2 sigma), conformal vs Gaussian.
P_ref is the crude MC probability on the same cloud (pf_mc_ref).

  python scripts/summarize_diagnostics.py --results results_ablation \
         --cells C2-C2-jp C2-C2-mm U-U EFF-EFF
"""
import argparse, csv, glob
from collections import defaultdict
from pathlib import Path
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--results", default="results_ablation")
p.add_argument("--cells", nargs="+", default=["C2-C2-jp", "C2-C2-mm", "U-U", "EFF-EFF"])
a = p.parse_args()

runs = defaultdict(list)
for f in sorted(glob.glob(str(Path(a.results) / "*.npz"))):
    pb, cell, _ = Path(f).stem.split("__")
    if cell in a.cells:
        with np.load(f, allow_pickle=True) as d:
            runs[(pb, cell)].append({k: d[k] for k in d.files})

rows = []
for (pb, cell), rs in sorted(runs.items()):
    cols = list(rs[0]["diag_cols"])
    fin = np.array([r["diag"][-1] for r in rs])          # (seeds, K) at stopping
    ref = np.array([float(r["pf_mc_ref"]) for r in rs])
    row = dict(problem=pb, cell=cell, n_runs=len(rs))
    for j, c in enumerate(cols):
        row[c] = float(np.nanmedian(fin[:, j])) if np.isfinite(fin[:, j]).any() else np.nan
    pl = np.array([float(r["pf_lower"]) for r in rs]); pu = np.array([float(r["pf_upper"]) for r in rs])
    beta = fin[:, cols.index("conf_miscov")]
    ok = lambda lo, hi: (np.nan if not np.isfinite(lo).any() else float(np.mean((lo <= ref) & (ref <= hi))))
    row["freq_ref_in_raw_bounds"] = ok(pl, pu)
    row["freq_ref_in_inflated_bounds"] = ok(pl - beta, pu + beta)
    row["freq_ref_in_gp_bounds"] = ok(fin[:, cols.index("gp_pf_lower")], fin[:, cols.index("gp_pf_upper")])
    rows.append(row)

out = Path(a.results) / "diagnostics_final.csv"
keys = list(rows[0]) if rows else []
with open(out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
print("wrote", out)
for r in rows:
    print(f"{r['problem']:14s} {r['cell']:9s} cov_all conf/gp={r['conf_cov_all']:.3f}/{r['gp_cov_all']:.3f}  "
          f"cov_c2 conf/gp={r['conf_cov_c2']:.3f}/{r['gp_cov_c2']:.3f}  "
          f"in raw/infl/gp bounds={r['freq_ref_in_raw_bounds']:.2f}/{r['freq_ref_in_inflated_bounds']:.2f}/{r['freq_ref_in_gp_bounds']:.2f}")

try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
except ImportError:
    raise SystemExit("matplotlib not installed: CSV written, figures skipped")
fig_dir = Path(a.results) / "figures"; fig_dir.mkdir(exist_ok=True)
for (pb, cell), rs in sorted(runs.items()):
    cols = list(rs[0]["diag_cols"]); L = min(len(r["diag"]) for r in rs)
    nc = np.nanmedian([r["history"][:L, 0] for r in rs], 0)
    D = np.array([r["diag"][:L] for r in rs])
    fig, ax = plt.subplots(figsize=(5, 3.2))
    for key, ls, lab in [("conf_cov_all", "-", "conformal, global"), ("conf_cov_c2", "--", "conformal, $|\\mu|\\leq2\\sigma$"),
                         ("gp_cov_all", "-.", "Gaussian, global"), ("gp_cov_c2", ":", "Gaussian, $|\\mu|\\leq2\\sigma$")]:
        v = D[:, :, cols.index(key)]
        if np.isfinite(v).any():
            ax.plot(nc, np.nanmedian(v, 0), ls, label=lab)
    ax.axhline(0.9, color="gray", lw=0.6); ax.set_xlabel("$n_{calls}$"); ax.set_ylabel("coverage")
    ax.set_title(f"{pb} / {cell}", fontsize=9); ax.legend(fontsize=7); fig.tight_layout()
    fig.savefig(fig_dir / f"{pb}__{cell}_coverage.pdf"); plt.close(fig)
print("figures ->", fig_dir)
