#!/usr/bin/env python
"""Final designs of each method on the 2D benchmarks (figure of the paper).

The standard result files store only summaries, not the designs, so this script
re-runs ONE seed per (problem, method) with exactly the ablation protocol
(``run_ablation.build``: same population, initial design, GP and stopping
rules as the diagonal cells of Table 9 and SUR-SUR of Table 6). It saves the
final designs, then draws one panel per method: failure domain, limit state
g = 0, initial design and added points coloured by order of addition.

Seed choice (stated in the caption): the same seed for all methods, namely
the run whose relative error for the reference cell (default C2-C2-mm) is the
median over the seeds of an existing ablation directory. Ties are broken by
the smallest seed. Pass --seed to fix it explicitly.

Usage:
  # 1) runs (needs torch/gpytorch, like run_ablation.py)
  python scripts/run_design_figure.py run --problems branch2d_k7 rastrigin2d \
      --pick-from results_ablation --n-jobs 5
  # 2) figure (numpy/matplotlib only)
  python scripts/run_design_figure.py plot --problems branch2d_k7 rastrigin2d \
      --out ../main-paper-v0/designs_2d.pdf
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_CELLS = ["U-U", "EFF-EFF", "SUR-SUR", "C2-C2-jp", "C2-C2-mm"]
LABELS = {"U-U": "$U$", "EFF-EFF": "EFF", "SUR-SUR": "SUR",
          "C2-C2-jp": "C$^2$ (J+GP)", "C2-C2-mm": "C$^2$ (J-mm-GP)",
          "C2-C2-g": "C$^2$ (Gaussian)"}
TITLES = {"branch2d_k6": "Four-branch, $k=6$", "branch2d_k7": "Four-branch, $k=7$",
          "rastrigin2d": "Modified Rastrigin"}


def load_problems_module():
    """Import akmcsc2/problems.py alone (numpy only), so that plotting does
    not require torch/gpytorch."""
    spec = importlib.util.spec_from_file_location("akmcs_problems",
                                                  ROOT / "src" / "akmcsc2" / "problems.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod          # needed by @dataclass
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- seed choice
def pick_median_seed(results_dir: Path, problem: str, cell: str) -> int:
    rows = []
    for f in sorted(results_dir.glob(f"{problem}__{cell}__seed*.npz")):
        d = np.load(f)
        ref = float(d["pf_mc_ref"])
        if np.isfinite(ref) and ref > 0:
            rows.append((abs(float(d["pf"]) - ref) / ref, int(d["seed"])))
    if not rows:
        sys.exit(f"no {problem}__{cell}__seed*.npz in {results_dir}; pass --seed")
    rows.sort()
    errs = np.array([r[0] for r in rows])
    med = np.median(errs)
    best = min(rows, key=lambda r: (abs(r[0] - med), r[1]))
    return best[1]


# ----------------------------------------------------------------------- run
def run_one(problem, cell, seed, out_dir: Path, rel_gap_tol, use_cuda):
    sys.path.insert(0, str(HERE))
    from run_ablation import build  # noqa: E402  (needs torch)

    out = out_dir / f"{problem}__{cell}__seed{seed:03d}__design.npz"
    if out.exists():
        return f"[skip] {out.name}"
    t0 = time.perf_counter()
    try:
        algo = build(problem, cell, rel_gap_tol, use_cuda)
        res = algo.run(seed=seed)
        # the design is stored in standardized coordinates: undo the
        # StandardScaler fitted on the Monte Carlo cloud S_raw
        mu = res.S_raw.mean(axis=0)
        sd = res.S_raw.std(axis=0)
        X_raw = res.X_train * sd + mu
        np.savez_compressed(
            out, X=X_raw, y=res.y_train, pf=res.pf, pf_mc_ref=res.pf_mc_ref,
            pf_lower=np.nan if res.pf_lower is None else res.pf_lower,
            pf_upper=np.nan if res.pf_upper is None else res.pf_upper,
            n_calls=res.n_calls, n_calls_total=res.n_calls_total,
            n_restarts=res.n_restarts, n_init=algo.n_init, converged=res.converged,
            seed=seed, cell=cell, problem=problem,
            lo=np.quantile(res.S_raw, 0.0005, axis=0), hi=np.quantile(res.S_raw, 0.9995, axis=0),
        )
        return f"[done] {out.name} pf={res.pf:.3e} ncall={res.n_calls} ({time.perf_counter()-t0:.0f}s)"
    except Exception:
        (out_dir / (out.stem + ".FAILED.txt")).write_text(traceback.format_exc())
        return f"[FAIL] {out.name}"


def cmd_run(a):
    sys.path.insert(0, str(HERE))
    from run_experiments import CONFIG  # noqa: E402
    out_dir = Path(a.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    seeds = {}
    for pb in a.problems:
        seeds[pb] = a.seed if a.seed is not None else pick_median_seed(
            Path(a.pick_from), pb, a.reference_cell)
    (out_dir / "seeds.json").write_text(json.dumps(
        dict(seeds=seeds, reference_cell=a.reference_cell,
             rule="median relative error of the reference cell" if a.seed is None else "fixed"),
        indent=2))
    print("seeds:", seeds, flush=True)
    tasks = [(pb, c, seeds[pb]) for pb in a.problems for c in a.cells]
    rel = a.rel_gap_tol if a.rel_gap_tol is not None else CONFIG["rel_gap_tol"]
    if a.n_jobs > 1:
        from joblib import Parallel, delayed
        for m in Parallel(n_jobs=a.n_jobs, verbose=5)(
                delayed(run_one)(pb, c, s, out_dir, rel, a.use_cuda) for pb, c, s in tasks):
            print(m, flush=True)
    else:
        for pb, c, s in tasks:
            print(run_one(pb, c, s, out_dir, rel, a.use_cuda), flush=True)


# ---------------------------------------------------------------------- plot
def cmd_plot(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    probs = load_problems_module()
    in_dir = Path(a.out_dir)
    seeds = json.loads((in_dir / "seeds.json").read_text())["seeds"] if a.seed is None \
        else {pb: a.seed for pb in a.problems}
    nr, nc = len(a.problems), len(a.cells)
    fig, axs = plt.subplots(nr, nc, figsize=(2.35 * nc, 2.45 * nr + 0.5), squeeze=False)
    plt.rcParams.update({"font.size": 8})
    for r, pb in enumerate(a.problems):
        prob = probs.PROBLEMS[pb]()
        for c, cell in enumerate(a.cells):
            ax = axs[r, c]
            f = in_dir / f"{pb}__{cell}__seed{int(seeds[pb]):03d}__design.npz"
            if not f.exists():
                ax.text(0.5, 0.5, "missing run", ha="center", va="center", transform=ax.transAxes)
                ax.set_xticks([]); ax.set_yticks([])
                continue
            d = np.load(f)
            lo, hi = d["lo"], d["hi"]
            L = float(max(np.max(np.abs(lo)), np.max(np.abs(hi))))
            t = np.linspace(-L, L, 400)
            G1, G2 = np.meshgrid(t, t)
            G = prob(np.column_stack([G1.ravel(), G2.ravel()])).reshape(G1.shape)
            ax.contourf(G1, G2, (G <= 0).astype(float), levels=[0.5, 1.5],
                        cmap=ListedColormap(["#f4c7c3"]))
            ax.contour(G1, G2, G, levels=[0.0], colors="k", linewidths=0.7)
            X, n0 = d["X"], int(d["n_init"])
            order = np.arange(len(X) - n0)
            ax.scatter(X[n0:, 0], X[n0:, 1], c=order, cmap="viridis", s=7, lw=0, zorder=3)
            ax.scatter(X[:n0, 0], X[:n0, 1], marker="x", c="k", s=12, lw=0.8, zorder=4)
            ref = float(d["pf_mc_ref"])
            err = 100 * (float(d["pf"]) - ref) / ref
            nct = int(d["n_calls_total"]) if "n_calls_total" in d else int(d["n_calls"])
            ax.set_title(f"{LABELS.get(cell, cell)}\n$n_{{\\mathrm{{calls}}}}={nct}$, "
                         f"err $={err:+.1f}\\%$", fontsize=8)
            ax.set_xlim(-L, L); ax.set_ylim(-L, L); ax.set_aspect("equal")
            ax.tick_params(labelsize=6)
            if c == 0:
                ax.set_ylabel(f"{TITLES.get(pb, pb)}\n$x_2$", fontsize=8)
            if r == nr - 1:
                ax.set_xlabel("$x_1$", fontsize=8)
    fig.tight_layout()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print("wrote", out, "seeds:", seeds)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("run", "plot"):
        q = sub.add_parser(name)
        q.add_argument("--problems", nargs="+", default=["branch2d_k7", "rastrigin2d"])
        q.add_argument("--cells", nargs="+", default=DEFAULT_CELLS)
        q.add_argument("--seed", type=int, default=None)
        q.add_argument("--out-dir", default="results_designs")
    r = sub.choices["run"]
    r.add_argument("--pick-from", default="results_ablation",
                   help="ablation directory used to pick the median seed")
    r.add_argument("--reference-cell", default="C2-C2-mm")
    r.add_argument("--rel-gap-tol", type=float, default=None)
    r.add_argument("--n-jobs", type=int, default=1)
    r.add_argument("--use-cuda", action="store_true")
    sub.choices["plot"].add_argument("--out", default="results_designs/designs_2d.pdf")
    a = p.parse_args()
    cmd_run(a) if a.cmd == "run" else cmd_plot(a)


if __name__ == "__main__":
    main()
