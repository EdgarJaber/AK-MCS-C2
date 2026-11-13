#!/usr/bin/env python
"""Figures 1-3 and truss figures with Gaussian coverage added.

Per problem: <problem>_pf_vs_iter.pdf (as make_paper_assets.py) and
<problem>_coverage_vs_iter.pdf with, in addition to the conformal coverage of
the two C2 variants (solid, markers, IQR band), the empirical coverage of the
Gaussian 90% intervals mu +- 1.645 sigma on each method's own design (dotted,
same colour, no band). All coverages are on the unevaluated points D \\ D_t.
  python scripts/make_paper_assets_gauss.py --results results_ablation --out paper_assets_gauss
  python scripts/make_paper_assets_gauss.py --results results_truss --out paper_assets_gauss --truss
"""
import argparse, glob
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

PB = {"branch2d_k6": ("2D four-branch (k=6)", 4.46e-3), "branch2d_k7": ("2D four-branch (k=7)", 2.23e-3),
      "rastrigin2d": ("2D modified Rastrigin", 7.30e-2), "oscillator6d": ("6D nonlinear oscillator", 3.90e-2)}
TR = {"truss23_tau120": ("23-bar truss, $\\tau=12$ cm", 1.52e-3), "truss23_tau134": ("23-bar truss, $\\tau=13.4$ cm", 1.06e-4)}
METHODS = [("U-U", "U", "C0", "o"), ("EFF-EFF", "EFF", "C1", "s"),
           ("C2-C2-jp", r"$C^2$-J+GP", "C2", "^"), ("C2-C2-mm", r"$C^2$-J+minmax-GP", "C3", "D")]
p = argparse.ArgumentParser(); p.add_argument("--results", default="results_ablation")
p.add_argument("--out", default="paper_assets_gauss"); p.add_argument("--truss", action="store_true")
a = p.parse_args(); out = Path(a.out); out.mkdir(exist_ok=True, parents=True)

def load(pb, cell):
    R = []
    for f in sorted(glob.glob(str(Path(a.results) / f"{pb}__{cell}__seed*.npz"))):
        with np.load(f, allow_pickle=True) as d:
            R.append(dict(h=d["history"], diag=d["diag"], cols=list(d["diag_cols"])))
    return R

def band(curves, min_frac=0.5):
    L = max(len(c) for c in curves); M = np.full((len(curves), L), np.nan)
    for i, c in enumerate(curves): M[i, :len(c)] = c
    alive = np.sum(np.isfinite(M), 0) >= min_frac * len(curves)
    n = int(np.argmin(alive)) if not alive.all() else L
    M = M[:, :n]
    return np.nanmedian(M, 0), np.nanpercentile(M, 25, 0), np.nanpercentile(M, 75, 0)

for pb, (title, ref) in (TR if a.truss else PB).items():
    fig1, ax1 = plt.subplots(figsize=(6, 3.7)); fig2, ax2 = plt.subplots(figsize=(6, 3.7))
    for cell, lab, col, mk in METHODS:
        R = load(pb, cell)
        if not R: continue
        m, lo, hi = band([r["h"][:, 1] for r in R]); x = np.arange(len(m))
        ax1.plot(x, m, color=col, marker=mk, ms=3.5, markevery=2, lw=1.2, label=lab); ax1.fill_between(x, lo, hi, color=col, alpha=0.15)
        if cell.startswith("C2"):
            j = R[0]["cols"].index("conf_cov_all")
            m, lo, hi = band([r["diag"][:, j] for r in R]); x = np.arange(len(m))
            ax2.plot(x, m, color=col, marker=mk, ms=3.5, markevery=2, lw=1.2, label=lab + " (conformal)"); ax2.fill_between(x, lo, hi, color=col, alpha=0.12)
        j = R[0]["cols"].index("gp_cov_all")
        m, _, _ = band([r["diag"][:, j] for r in R]); x = np.arange(len(m))
        ax2.plot(x, m, color=col, ls=":", lw=1.5, label=lab + " (Gaussian)")
    ax1.axhline(ref, color="k", ls="--", lw=1, label=r"$P_f^{\mathrm{ref}}$"); ax1.set_xlabel("Number of iterations"); ax1.set_ylabel(r"$\widehat{P}_f$")
    if a.truss: ax1.set_yscale("log")
    ax1.set_title(title + "\n" + r"$\widehat{P}_f$ vs iterations", fontsize=10); ax1.legend(fontsize=8); ax1.grid(alpha=0.2)
    ax2.axhline(0.9, color="k", ls="--", lw=1, label=r"$1-\alpha=0.90$"); ax2.set_ylim(0, 1.05); ax2.set_xlabel("Number of iterations"); ax2.set_ylabel("Empirical coverage")
    ax2.set_title(title + "\nEmpirical coverage vs iterations", fontsize=10); ax2.legend(fontsize=6.5, loc="lower right", ncol=2); ax2.grid(alpha=0.2)
    fig1.tight_layout(); fig1.savefig(out / f"{pb}_pf_vs_iter.pdf"); fig2.tight_layout(); fig2.savefig(out / f"{pb}_coverage_vs_iter.pdf")
    plt.close("all"); print("ok", pb)
