#!/usr/bin/env python
"""Figures for the ablation appendix.
  ablation_grid.pdf     : 3x3 (selection x stopping) heatmaps, J-minmax-GP, per
                          problem: median n_calls and median |rel. err.| (%).
  ablation_coverage.pdf : median coverage vs n_calls, global and near the true
                          boundary ({|g| <= 5%-quantile of |g|}), conformal
                          (C2-C2-mm) vs Gaussian intervals on the same designs
                          (C2-C2-mm, U-U, EFF-EFF).
  python scripts/make_ablation_figures.py --results results_ablation --out figs
"""
import argparse, csv, glob
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

PB = ["branch2d_k6", "branch2d_k7", "rastrigin2d", "oscillator6d"]
TITLE = {"branch2d_k6": "4-branch $k=6$", "branch2d_k7": "4-branch $k=7$",
         "rastrigin2d": "Rastrigin", "oscillator6d": "Oscillator 6D"}
S = ["U", "EFF", "C2"]

p = argparse.ArgumentParser()
p.add_argument("--results", default="results_ablation")
p.add_argument("--out", default=".")
p.add_argument("--variant", default="mm")
a = p.parse_args()
out = Path(a.out); out.mkdir(exist_ok=True, parents=True)

rows = {(r["problem"], r["cell"]): r for r in csv.DictReader(open(Path(a.results) / "ablation_summary.csv"))}
def cell(sel, stop): return f"{sel}-{stop}" + (f"-{a.variant}" if "C2" in (sel, stop) else "")

fig, ax = plt.subplots(2, 4, figsize=(12, 5.4))
for j, pb in enumerate(PB):
    for i, (key, lab, fmt) in enumerate([("ncalls_median", "median $n_{calls}$", "{:.0f}"),
                                         ("abs_err_pct_median", "median $|$rel. err.$|$ (%)", "{:.2f}")]):
        M = np.array([[float(rows[(pb, cell(s, t))][key]) for t in S] for s in S])
        im = ax[i, j].imshow(M, cmap="Blues" if i == 0 else "Reds",
                             norm=matplotlib.colors.LogNorm(vmin=max(M[M > 0].min(), 1e-2), vmax=M.max()) if (M > 0).any() else None)
        for u in range(3):
            for v in range(3):
                conv = float(rows[(pb, cell(S[u], S[v]))]["converged_frac"])
                ax[i, j].text(v, u, fmt.format(M[u, v]) + ("*" if conv < 0.99 else ""),
                              ha="center", va="center", fontsize=8)
        ax[i, j].set_xticks(range(3)); ax[i, j].set_xticklabels(S); ax[i, j].set_yticks(range(3)); ax[i, j].set_yticklabels(S)
        if i == 1: ax[i, j].set_xlabel("stopping rule")
        if j == 0: ax[i, j].set_ylabel("selection rule\n" + lab)
        else: ax[i, j].set_ylabel(lab, fontsize=8)
        ax[i, j].set_title(TITLE[pb] if i == 0 else "", fontsize=9)
fig.suptitle("Selection $\\times$ stopping ablation (C$^2$ cells: J-minmax-GP). * : not all runs converged within the budget", fontsize=9)
fig.tight_layout(); fig.savefig(out / "ablation_grid.pdf"); plt.close(fig)

# --- coverage curves
runs = defaultdict(list)
for f in sorted(glob.glob(str(Path(a.results) / "*.npz"))):
    pb, c, _ = Path(f).stem.split("__")
    if c in ("C2-C2-mm", "U-U", "EFF-EFF"):
        with np.load(f, allow_pickle=True) as d:
            runs[(pb, c)].append((d["history"][:, 0].copy(), d["diag"].copy(), list(d["diag_cols"])))
fig, ax = plt.subplots(1, 4, figsize=(13, 3.3), sharey=True)
for j, pb in enumerate(PB):
    for c, items in [("C2-C2-mm", [("conf_cov_all", "-", "C0", "conformal, global"), ("gp_cov_all", "-", "C3", "Gaussian, global"), ]),
                     ("U-U", [("gp_cov_all", ":", "C2", "Gaussian, $U$-$U$")]),
                     ("EFF-EFF", [("gp_cov_all", ":", "C1", "Gaussian, EFF-EFF")])]:
        R = runs[(pb, c)]
        if not R: continue
        grid = np.arange(20, int(max(r[0].max() for r in R)) + 1)
        for key, ls, col, lab in items:
            M = np.full((len(R), len(grid)), np.nan)
            for k, (nc, D, cols) in enumerate(R):
                v = D[:, cols.index(key)]
                for gi, g in enumerate(grid):
                    idx = np.where(nc <= g)[0]
                    if len(idx) and g <= nc.max(): M[k, gi] = v[idx[-1]]
            ok = np.sum(np.isfinite(M), 0) >= max(5, len(R) // 3)
            if ok.any(): ax[j].plot(grid[ok], np.nanmedian(M[:, ok], 0), ls, color=col, label=lab, lw=1.3)
    ax[j].axhline(0.9, color="gray", lw=0.6); ax[j].set_title(TITLE[pb], fontsize=9); ax[j].set_xlabel("$n_{calls}$")
    ax[j].set_ylim(0.4, 1.02)
ax[0].set_ylabel("coverage (median over seeds)"); ax[0].legend(fontsize=6.5, loc="lower right")
fig.tight_layout(); fig.savefig(out / "ablation_coverage.pdf"); plt.close(fig)
print("wrote", out / "ablation_grid.pdf", out / "ablation_coverage.pdf")
