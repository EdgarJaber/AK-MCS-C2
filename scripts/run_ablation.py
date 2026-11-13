#!/usr/bin/env python
"""Ablation: selection rule x stopping rule (answers R2.4, R3.1, R4.7).

Each cell (SEL, STOP) adds points with SEL in {U, EFF, C2} and stops with
STOP in {U, EFF, C2} (+ optional "none" = fixed budget, ``--fixed-budget``).
Diagonal cells (U-U, EFF-EFF, C2-C2) are the original methods; off-diagonal
cells isolate the contribution of the conformal intervals to the *selection*
(C2-U, C2-EFF) and to the *stopping decision* (U-C2, EFF-C2).
The conformal variant (jp = J+GP, mm = J-minmax-GP) is shared by selection
and stopping, and is only part of the cell name when C2 is involved.

Cell names: "<SEL>-<STOP>" or "<SEL>-<STOP>-<jp|mm>", e.g. "C2-U-mm".
Same seed => same MC population and same initial design in every cell.

One .npz per (problem, cell, seed), skipped if already present (resumable,
shardable). Usage:
  python scripts/run_ablation.py --list                      # show the cells
  python scripts/run_ablation.py --n-seeds 3 --n-jobs 4      # pilot
  python scripts/run_ablation.py --n-jobs 10                 # full, one node
  python scripts/run_ablation.py --shard $I --n-shards $N    # job array
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_experiments import CONFIG  # noqa: E402  (shared protocol)
from akmcsc2.core import DIAG_COLUMNS  # noqa: E402
from akmcsc2 import AKMCS, Ablation, LOOGPRegressor, get_problem, make_conformal  # noqa: E402

SELECTIONS = ["U", "EFF", "C2"]
STOPS = ["U", "EFF", "C2"]


def sur_cells():
    """SUR comparison (Chevalier et al. 2014 criterion, own implementation):
    SUR-SUR is the full SUR method; the mixed cells cross it with C2."""
    return ["SUR-SUR", "SUR-C2-jp", "SUR-C2-mm", "C2-SUR-jp", "C2-SUR-mm"]


def cells(fixed_budget: bool):
    stops = STOPS + (["none"] if fixed_budget else [])
    out = []
    for sel in SELECTIONS:
        for stop in stops:
            if "C2" in (sel, stop):
                out += [f"{sel}-{stop}-jp", f"{sel}-{stop}-mm"]
            else:
                out.append(f"{sel}-{stop}")
    return out


def build(problem_name: str, cell: str, rel_gap_tol: float, use_cuda: bool,
          predict_batch: int = 8192):
    parts = cell.split("-")
    sel, stop = parts[0], parts[1]
    cv = parts[2] if len(parts) > 2 else None
    learning = Ablation(
        select_with=sel, stop_with=stop, rel_gap_tol=rel_gap_tol,
        candidate_size=CONFIG["candidate_size"], batch_q=CONFIG["batch_q"],
        batch_delta=CONFIG["batch_delta"],
    )
    conformal = None
    if cv is not None:
        conformal = make_conformal("j+gp" if cv == "jp" else "j-mm-gp",
                                   alpha=CONFIG["alpha"])
    gp_cfg = dict(CONFIG["gp"])
    return AKMCS(
        problem=get_problem(problem_name), learning=learning, conformal=conformal,
        gp_factory=lambda: LOOGPRegressor(use_cuda=use_cuda, **gp_cfg),
        n_mc=CONFIG.get("n_mc_by_problem", {}).get(problem_name, CONFIG["n_mc"]),
        n_init=CONFIG["n_init"],
        max_calls=CONFIG["max_calls"][problem_name],
        min_iterations=CONFIG["min_iterations"],
        conformal_every=CONFIG["conformal_every"],
        track_coverage=True,   # cheap analytic g: gives pf_mc_ref for every cell
        predict_batch_size=predict_batch,   # memory only; does not change results
    )


def run_one(pb, cell, seed, out_dir, rel_gap_tol, use_cuda, predict_batch=8192):
    out = out_dir / f"{pb}__{cell}__seed{seed:03d}.npz"
    if out.exists():
        return f"[skip] {out.name}"
    t0 = time.perf_counter()
    try:
        algo = build(pb, cell, rel_gap_tol, use_cuda, predict_batch)
        res = algo.run(seed=seed)
        np.savez_compressed(
            out, history=res.history_array(), pf=res.pf,
            pf_lower=np.nan if res.pf_lower is None else res.pf_lower,
            pf_upper=np.nan if res.pf_upper is None else res.pf_upper,
            n_calls=res.n_calls, converged=res.converged,
            pf_mc_ref=res.pf_mc_ref,
            pf_ref=np.nan if algo.problem.pf_ref is None else algo.problem.pf_ref,
            seed=seed,
            diag=res.diag_array(), diag_cols=np.array(DIAG_COLUMNS),
            n_calls_total=res.n_calls_total, n_restarts=res.n_restarts,
            timer_names=np.array(list(res.timers)), timer_values=np.array(list(res.timers.values())),
        )
        return f"[done] {out.name} pf={res.pf:.3e} ncall={res.n_calls} ({time.perf_counter()-t0:.0f}s)"
    except Exception:
        err = out_dir / (out.stem + ".FAILED.txt")
        err.write_text(traceback.format_exc())
        return f"[FAIL] {out.name} -> {err.name}"


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", default="results_ablation")
    p.add_argument("--problems", nargs="+", default=CONFIG["problems"],
                   choices=CONFIG["problems"] + CONFIG.get("extra_problems", []))
    p.add_argument("--cells", nargs="+", default=None, help="subset of cell names")
    p.add_argument("--fixed-budget", action="store_true",
                   help="add STOP=none cells (run to max_calls; fixed-budget accuracy)")
    p.add_argument("--n-seeds", type=int, default=CONFIG["n_seeds"])
    p.add_argument("--rel-gap-tol", type=float, default=CONFIG["rel_gap_tol"])
    p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument("--use-cuda", action="store_true")
    p.add_argument("--shard", type=int, default=None)
    p.add_argument("--n-shards", type=int, default=None)
    p.add_argument("--predict-batch", type=int, default=8192,
                   help="MC-cloud prediction batch (memory only)")
    p.add_argument("--order", choices=["seed-major", "problem-major"], default="seed-major",
                   help="seed-major: all cells/problems for seed 0, then seed 1, ... "
                        "(partial runs stay balanced)")
    p.add_argument("--sur", action="store_true", help="run ONLY the SUR comparison cells")
    p.add_argument("--list", action="store_true")
    a = p.parse_args()

    all_cells = sur_cells() if a.sur else cells(a.fixed_budget)
    if a.list:
        print("\n".join(all_cells)); print(len(all_cells), "cells"); return
    sel_cells = a.cells or all_cells
    for c in sel_cells:
        if c not in all_cells:
            sys.exit(f"unknown cell {c}; use --list (add --fixed-budget for 'none')")

    out_dir = Path(a.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    meta = dict(CONFIG, rel_gap_tol=a.rel_gap_tol, cells=sel_cells,
                n_seeds=a.n_seeds, eff_delta="2*sigma", u_threshold=2.0, eff_tol=1e-3)
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str))

    if a.order == "seed-major":
        tasks = [(pb, c, s) for s in range(a.n_seeds) for pb in a.problems for c in sel_cells]
    else:
        tasks = [(pb, c, s) for pb in a.problems for c in sel_cells for s in range(a.n_seeds)]
    if a.shard is not None:
        if not a.n_shards:
            sys.exit("--shard requires --n-shards")
        tasks = tasks[a.shard::a.n_shards]
    print(f"{len(tasks)} tasks -> {out_dir}/ (n_jobs={a.n_jobs})", flush=True)

    if a.n_jobs > 1:
        from joblib import Parallel, delayed
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        for m in Parallel(n_jobs=a.n_jobs, verbose=5)(
                delayed(run_one)(pb, c, s, out_dir, a.rel_gap_tol, a.use_cuda, a.predict_batch)
                for pb, c, s in tasks):
            print(m)
    else:
        for pb, c, s in tasks:
            print(run_one(pb, c, s, out_dir, a.rel_gap_tol, a.use_cuda, a.predict_batch), flush=True)
    print("All done.")


if __name__ == "__main__":
    main()
