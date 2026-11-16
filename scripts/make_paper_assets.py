#!/usr/bin/env python
"""Main-text figures and Table-2 rows from the 50-seed ablation campaign.

Per problem: <problem>_pf_vs_iter.pdf (median + IQR of P_f^(t) over seeds for
U-U, EFF-EFF, C2-C2-jp, C2-C2-mm; reference line) and
<problem>_coverage_vs_iter.pdf (empirical coverage of the conformal intervals on
the unevaluated points D \\ D_t, same definition as Eq. (32)); plus
table2_rows.tex. Curves stop when fewer than half of the runs are still active;
a run that has stopped is not extended.
  python scripts/make_paper_assets.py --results results_ablation --out paper_assets
"""
import argparse, glob
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

PB = {"branch2d_k6": ("2D four-branch (k=6)", 4.46e-3), "branch2d_k7": ("2D four-branch (k=7)", 2.23e-3),
      "rastrigin2d": ("2D modified Rastrigin", 7.30e-2), "oscillator6d": ("6D nonlinear oscillator", 2.86e-2),
      "oscillator6d_variant": ("6D nonlinear oscillator (variant)", 3.90e-2)}
METHODS = [("U-U", "U", "C0", "o"), ("EFF-EFF", "EFF", "C1", "s"),
           ("C2-C2-jp", r"$C^2$-J+GP", "C2", "^"), ("C2-C2-mm", r"$C^2$-J+minmax-GP", "C3", "D")]
p = argparse.ArgumentParser(); p.add_argument("--results", default="results_ablation"); p.add_argument("--out", default="paper_assets")
a = p.parse_args(); out = Path(a.out); out.mkdir(exist_ok=True, parents=True)

def load(pb, cell):
    R = []
    for f in sorted(glob.glob(str(Path(a.results) / f"{pb}__{cell}__seed*.npz"))):
        with np.load(f, allow_pickle=True) as d:
            R.append(dict(h=d["history"], diag=d["diag"], cols=list(d["diag_cols"]), pf=float(d["pf"]),
                          ncalls=int(d["n_calls"]), ntot=int(d["n_calls_total"])))
    return R

def band(curves, min_frac=0.5):
    L = max(len(c) for c in curves); M = np.full((len(curves), L), np.nan)
    for i, c in enumerate(curves): M[i, :len(c)] = c
    alive = np.sum(np.isfinite(M), 0) >= min_frac * len(curves)
    n = int(np.argmin(alive)) if not alive.all() else L
    M = M[:, :n]
    return np.nanmedian(M, 0), np.nanpercentile(M, 25, 0), np.nanpercentile(M, 75, 0)

rows = []

def data_ref(pb, default):
    """Tabulated reference P_f, checked against the run files: if the median
    crude MC estimate on the populations (pf_mc_ref) differs from it by more
    than 10% (e.g. old oscillator6d runs made with the sin(w0^2 t1/2) variant),
    the data value is used and a warning is printed."""
    refs = []
    for f in sorted(glob.glob(str(Path(a.results) / f"{pb}__*__seed*.npz"))):
        with np.load(f, allow_pickle=True) as d:
            if "pf_mc_ref" in d.files and np.isfinite(float(d["pf_mc_ref"])):
                refs.append(float(d["pf_mc_ref"]))
    if not refs:
        return default
    med = float(np.median(refs))
    if abs(med - default) > 0.1 * default:
        print(f"[warning] {pb}: tabulated P_f_ref={default:.3e} but runs give {med:.3e}; using the runs")
        return med
    return default

for pb, (title, ref) in PB.items():
    ref = data_ref(pb, ref)
    fig1, ax1 = plt.subplots(figsize=(6, 3.7)); fig2, ax2 = plt.subplots(figsize=(6, 3.7))
    for cell, lab, col, mk in METHODS:
        R = load(pb, cell)
        m, lo, hi = band([r["h"][:, 1] for r in R]); x = np.arange(len(m))
        ax1.plot(x, m, color=col, marker=mk, ms=3.5, markevery=2, lw=1.2, label=lab); ax1.fill_between(x, lo, hi, color=col, alpha=0.15)
        if cell.startswith("C2"):
            j = R[0]["cols"].index("conf_cov_all")
            m, lo, hi = band([r["diag"][:, j] for r in R]); x = np.arange(len(m))
            ax2.plot(x, m, color=col, marker=mk, ms=3.5, markevery=2, lw=1.2, label=lab); ax2.fill_between(x, lo, hi, color=col, alpha=0.15)
        pf = np.array([r["pf"] for r in R]); fin = np.array([r["diag"][-1][R[0]["cols"].index("conf_cov_all")] for r in R])
        rows.append((pb, cell, float(np.median(pf)), 100 * (np.median(pf) - ref) / ref, 100 * pf.std() / pf.mean(),
                     int(np.median([r["ntot"] for r in R])), float(np.nanmedian(fin)) if cell.startswith("C2") else None))
    ax1.axhline(ref, color="k", ls="--", lw=1, label=r"$P_f^{\mathrm{ref}}$"); ax1.set_xlabel("Number of iterations"); ax1.set_ylabel(r"$\widehat{P}_f$")
    ax1.set_title(title + "\n" + r"$\widehat{P}_f$ vs iterations", fontsize=10); ax1.legend(fontsize=8); ax1.grid(alpha=0.2)
    ax2.axhline(0.9, color="k", ls="--", lw=1, label=r"$1-\alpha=0.90$"); ax2.set_ylim(0, 1.05); ax2.set_xlabel("Number of iterations"); ax2.set_ylabel("Empirical coverage")
    ax2.set_title(title + "\nEmpirical coverage vs iterations", fontsize=10); ax2.legend(fontsize=8, loc="lower right"); ax2.grid(alpha=0.2)
    fig1.tight_layout(); fig1.savefig(out / f"{pb}_pf_vs_iter.pdf"); fig2.tight_layout(); fig2.savefig(out / f"{pb}_coverage_vs_iter.pdf")
    plt.close("all")
with open(out / "table2_rows.tsv", "w") as fh:
    for r in rows:
        fh.write("\t".join("" if v is None else (f"{v:.6g}" if isinstance(v, float) else str(v)) for v in r) + "\n")
print("wrote", out)
