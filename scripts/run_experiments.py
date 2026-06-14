#!/usr/bin/env python
"""Overnight cluster benchmark for the AK-MCS-C2 article.

Runs every (problem, method, seed) combination, each saved as a standalone
.npz the moment it finishes -- safe to interrupt and resume, safe to shard
across a SLURM job array.

Methods
-------
  U          : classical U-function
  EFF        : Expected Feasibility Function
  C2-jp      : conformal C^2 with J+GP
  C2-mm      : conformal C^2 with J-minmax-GP

Output (one file per task)
--------------------------
results/<problem>__<method>__seed<seed>.npz with
  history : (n_iter, 7) [n_calls, pf_hat, pf_lower, pf_upper, pf_gap,
                          emp_coverage, elapsed_s]
  pf, pf_lower, pf_upper, n_calls, converged, pf_mc_ref, pf_ref, seed
plus a meta.json with the full configuration (written once).

Usage
-----
  # everything, 8 local workers
  python scripts/run_experiments.py --n-jobs 8

  # SLURM array sharding (see scripts/submit_overnight.sbatch):
  python scripts/run_experiments.py --shard $SLURM_ARRAY_TASK_ID \
                                    --n-shards $SLURM_ARRAY_TASK_COUNT
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

from akmcsc2 import AKMCS, ConformalC2, LOOGPRegressor, get_problem

# ---------------------------------------------------------------------------
# configuration (edit here, recorded in meta.json)
# ---------------------------------------------------------------------------

CONFIG = dict(
    problems=["branch2d_k6", "branch2d_k7", "rastrigin2d", "oscillator6d"],
    methods=["U", "EFF", "C2-jp", "C2-mm"],
    n_seeds=50,
    n_mc=int(1e4),
    n_init=20,
    min_iterations=20,
    alpha=0.1,
    pf_gap_tol=1e-3,
    candidate_size=5000,
    batch_q=1,
    batch_delta=None,
    conformal_every=1,
    gp=dict(kernel="matern52", noise=1e-4, learn_noise=True, training_iter=100),
    # per-problem evaluation budget (Rastrigin needs ~600 for minmax/EFF)
    max_calls={"branch2d_k6": 200, "branch2d_k7": 200,
               "rastrigin2d": 650, "oscillator6d": 250},
)


def build_algo(problem_name: str, method: str, use_cuda: bool) -> AKMCS:
    problem = get_problem(problem_name)
    gp_cfg = dict(CONFIG["gp"])
    gp_factory = lambda: LOOGPRegressor(use_cuda=use_cuda, **gp_cfg)  # noqa: E731

    if method in ("U", "EFF"):
        learning, conformal, track = method, None, False
    elif method in ("C2-jp", "C2-mm"):
        learning = ConformalC2(
            candidate_size=CONFIG["candidate_size"],
            batch_q=CONFIG["batch_q"],
            batch_delta=CONFIG["batch_delta"],
            pf_gap_tol=CONFIG["pf_gap_tol"],
        )
        from akmcsc2 import make_conformal
        conformal = make_conformal(
            "j+gp" if method == "C2-jp" else "j-mm-gp", alpha=CONFIG["alpha"]
        )
        track = True
    else:
        raise ValueError(method)

    return AKMCS(
        problem=problem,
        learning=learning,
        conformal=conformal,
        gp_factory=gp_factory,
        n_mc=CONFIG["n_mc"],
        n_init=CONFIG["n_init"],
        max_calls=CONFIG["max_calls"][problem_name],
        min_iterations=CONFIG["min_iterations"],
        conformal_every=CONFIG["conformal_every"],
        track_coverage=track,
    )


def run_one(problem_name: str, method: str, seed: int,
            out_dir: Path, use_cuda: bool) -> str:
    out = out_dir / f"{problem_name}__{method}__seed{seed:03d}.npz"
    if out.exists():
        return f"[skip] {out.name}"
    t0 = time.perf_counter()
    try:
        algo = build_algo(problem_name, method, use_cuda)
        res = algo.run(seed=seed)
        np.savez_compressed(
            out,
            history=res.history_array(),
            pf=res.pf,
            pf_lower=np.nan if res.pf_lower is None else res.pf_lower,
            pf_upper=np.nan if res.pf_upper is None else res.pf_upper,
            n_calls=res.n_calls,
            converged=res.converged,
            pf_mc_ref=res.pf_mc_ref,
            pf_ref=np.nan if algo.problem.pf_ref is None else algo.problem.pf_ref,
            seed=seed,
        )
        return (f"[done] {out.name}  pf={res.pf:.3e} "
                f"ncall={res.n_calls} ({time.perf_counter()-t0:.0f}s)")
    except Exception:
        # one bad seed must not kill the overnight run
        err = out_dir / (out.stem + ".FAILED.txt")
        err.write_text(traceback.format_exc())
        return f"[FAIL] {out.name} -> {err.name}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", type=str, default="results")
    p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument("--use-cuda", action="store_true")
    p.add_argument("--problem", choices=CONFIG["problems"], default=None)
    p.add_argument("--method", choices=CONFIG["methods"], default=None)
    p.add_argument("--shard", type=int, default=None,
                   help="0-based shard index for job arrays")
    p.add_argument("--n-shards", type=int, default=None)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "meta.json").write_text(json.dumps(CONFIG, indent=2, default=str))

    problems = [args.problem] if args.problem else CONFIG["problems"]
    methods = [args.method] if args.method else CONFIG["methods"]
    tasks = [
        (pb, m, s)
        for pb in problems
        for m in methods
        for s in range(CONFIG["n_seeds"])
    ]
    if args.shard is not None:
        if not args.n_shards:
            sys.exit("--shard requires --n-shards")
        tasks = tasks[args.shard :: args.n_shards]

    print(f"{len(tasks)} tasks -> {out_dir}/  (n_jobs={args.n_jobs}, "
          f"cuda={args.use_cuda})", flush=True)

    if args.n_jobs > 1:
        from joblib import Parallel, delayed
        # avoid n_jobs x torch-threads oversubscription
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        msgs = Parallel(n_jobs=args.n_jobs, verbose=5)(
            delayed(run_one)(pb, m, s, out_dir, args.use_cuda)
            for pb, m, s in tasks
        )
        for msg in msgs:
            print(msg)
    else:
        for pb, m, s in tasks:
            print(run_one(pb, m, s, out_dir, args.use_cuda), flush=True)

    print("All done.")


if __name__ == "__main__":
    main()
