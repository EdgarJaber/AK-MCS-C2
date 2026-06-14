#!/usr/bin/env python
"""Generate the article figures and summary.csv from results/*.npz.

For each problem produces (matching the manuscript's figure style):
  figures/<problem>_pf_vs_iter.pdf        : median + IQR band of pf_hat,
                                            all methods, with Pf_ref line
  figures/<problem>_coverage_vs_iter.pdf  : empirical coverage of the C2
                                            variants with the 1-alpha line
and a single figures/summary.csv with terminal medians, relative errors,
model-call counts and final coverages -- the source for Table `tab:summary`
of the paper.

Usage:  python scripts/make_figures.py --results results --out figures
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# history columns
NCALL, PF, PFLO, PFUP, GAP, COV, TIME = range(7)

STYLE = {
    "U":     dict(label=r"U",                 color="tab:blue",   marker="o", ls="-"),
    "EFF":   dict(label=r"EFF",               color="tab:orange", marker="s", ls="--"),
    "C2-jp": dict(label=r"$C^2$-J+GP",        color="tab:green",  marker="^", ls="-."),
    "C2-mm": dict(label=r"$C^2$-J+minmax-GP", color="tab:red",    marker="D", ls=":"),
}
TITLES = {
    "branch2d_k6": "2D four-branch (k=6)",
    "branch2d_k7": "2D four-branch (k=7)",
    "rastrigin2d": "2D modified Rastrigin",
    "oscillator6d": "6D nonlinear oscillator",
}
ALPHA = 0.10


def load(results: Path):
    """data[problem][method] = list of (history, scalars-dict)."""
    data: dict = defaultdict(lambda: defaultdict(list))
    for f in sorted(results.glob("*__*__seed*.npz")):
        problem, method, _ = f.stem.split("__")
        with np.load(f) as z:
            data[problem][method].append(
                (z["history"], {k: float(z[k]) for k in
                                ("pf", "pf_ref", "pf_mc_ref", "n_calls")})
            )
    return data


def med_iqr(runs, col, n_iter_max):
    """Median and IQR across seeds, per iteration; NaN-padded after each
    run's own termination so that shorter runs simply drop out."""
    M = np.full((len(runs), n_iter_max), np.nan)
    for r, (h, _) in enumerate(runs):
        M[r, : h.shape[0]] = h[:, col]
    with np.errstate(all="ignore"):
        med = np.nanmedian(M, axis=0)
        q1 = np.nanpercentile(M, 25, axis=0)
        q3 = np.nanpercentile(M, 75, axis=0)
    valid = np.sum(~np.isnan(M), axis=0) >= max(3, len(runs) // 4)
    return med, q1, q3, valid


def pf_ref_of(per_method) -> float:
    for runs in per_method.values():
        for _, s in runs:
            if np.isfinite(s["pf_ref"]):
                return s["pf_ref"]
            if np.isfinite(s["pf_mc_ref"]):
                return s["pf_mc_ref"]
    return np.nan


def fig_pf(problem, per_method, out: Path):
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    n_max = max(h.shape[0] for runs in per_method.values() for h, _ in runs)
    for m, st in STYLE.items():
        if m not in per_method:
            continue
        med, q1, q3, valid = med_iqr(per_method[m], PF, n_max)
        it = np.arange(n_max)
        ax.plot(it[valid], med[valid], marker=st["marker"], ls=st["ls"],
                color=st["color"], label=st["label"], markevery=max(1, n_max // 25),
                ms=5, lw=1.4)
        ax.fill_between(it[valid], q1[valid], q3[valid], color=st["color"], alpha=0.15)
    ref = pf_ref_of(per_method)
    if np.isfinite(ref):
        ax.axhline(ref, color="k", ls="--", lw=1.2, label=r"$P_f^{\mathrm{ref}}$")
    ax.set_xlabel("Number of iterations")
    ax.set_ylabel(r"$\widehat{P}_f$")
    ax.set_title(f"{TITLES.get(problem, problem)}\n" + r"$\widehat{P}_f$ vs iterations")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3, ls=":")
    fig.tight_layout()
    fig.savefig(out / f"{problem}_pf_vs_iter.pdf")
    plt.close(fig)


def fig_coverage(problem, per_method, out: Path):
    c2 = {m: per_method[m] for m in ("C2-jp", "C2-mm") if m in per_method}
    if not c2:
        return
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    n_max = max(h.shape[0] for runs in c2.values() for h, _ in runs)
    for m, runs in c2.items():
        st = STYLE[m]
        med, q1, q3, valid = med_iqr(runs, COV, n_max)
        it = np.arange(n_max)
        ax.plot(it[valid], med[valid], marker=st["marker"], ls=st["ls"],
                color=st["color"], label=st["label"], markevery=max(1, n_max // 25),
                ms=5, lw=1.4)
        ax.fill_between(it[valid], q1[valid], q3[valid], color=st["color"], alpha=0.15)
    ax.axhline(1 - ALPHA, color="k", ls="--", lw=1.2,
               label=rf"$1-\alpha = {1-ALPHA:.2f}$")
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("Number of iterations")
    ax.set_ylabel("Empirical coverage")
    ax.set_title(f"{TITLES.get(problem, problem)}\nEmpirical coverage vs iterations")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3, ls=":")
    fig.tight_layout()
    fig.savefig(out / f"{problem}_coverage_vs_iter.pdf")
    plt.close(fig)


def summary(data) -> pd.DataFrame:
    rows = []
    for problem, per_method in data.items():
        ref = pf_ref_of(per_method)
        for m, runs in per_method.items():
            pf = np.array([s["pf"] for _, s in runs])
            nc = np.array([s["n_calls"] for _, s in runs])
            cov = np.array([h[-1, COV] for h, _ in runs])
            t = np.array([h[-1, TIME] for h, _ in runs])
            pf_med = float(np.median(pf))
            rows.append(dict(
                problem=problem,
                method=m,
                n_seeds=len(runs),
                pf_ref=ref,
                pf_median=pf_med,
                pf_q25=float(np.percentile(pf, 25)),
                pf_q75=float(np.percentile(pf, 75)),
                pf_cov_pct=float(100 * np.std(pf, ddof=1) / max(np.mean(pf), 1e-300)),
                rel_err_pct=float(100 * (pf_med - ref) / ref) if np.isfinite(ref) else np.nan,
                n_calls_median=float(np.median(nc)),
                final_coverage_median=(float(np.nanmedian(cov))
                                       if np.any(np.isfinite(cov)) else np.nan),
                walltime_median_s=float(np.median(t)),
            ))
    return pd.DataFrame(rows).sort_values(["problem", "method"])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, default=Path("results"))
    p.add_argument("--out", type=Path, default=Path("figures"))
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({"font.size": 11, "axes.titlesize": 12})

    data = load(args.results)
    if not data:
        raise SystemExit(f"No .npz results found in {args.results}/")

    for problem, per_method in data.items():
        fig_pf(problem, per_method, args.out)
        fig_coverage(problem, per_method, args.out)
        print(f"[figs] {problem}: methods={sorted(per_method)}")

    df = summary(data)
    df.to_csv(args.out / "summary.csv", index=False)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print(f"\nWrote figures + summary.csv to {args.out}/")


if __name__ == "__main__":
    main()
