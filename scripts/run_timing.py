#!/usr/bin/env python
"""Cost table (R4.8): wall-clock, component times, peak memory per method.

Each (problem, method, seed) runs in a FRESH subprocess (so peak RSS is that of
the run alone), single-threaded, WITHOUT coverage tracking (diagnostics are not
part of the method's cost). Run on a quiet node, after the heavy campaigns.

Methods: U, EFF, C2-jp, C2-mm (original methods; C2 uses the relative stop).
Output: timing_results/timing.csv with, per run, total wall-clock, the
TIMER_NAMES components, n_calls (final design), n_calls_total (incl. restarts),
n_restarts, peak_rss_mb.

  python scripts/run_timing.py --n-seeds 5
  python scripts/run_timing.py --problems branch2d_k6 --methods C2-mm --n-seeds 2
"""
import argparse, csv, multiprocessing as mp, os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def _worker(args, q):
    os.environ["OMP_NUM_THREADS"] = "1"; os.environ["MKL_NUM_THREADS"] = "1"
    import torch; torch.set_num_threads(1)
    from run_experiments import build_algo, CONFIG
    pb, m, seed, rel_gap_tol = args
    CONFIG["rel_gap_tol"] = rel_gap_tol
    algo = build_algo(pb, m, use_cuda=False)
    algo.track_coverage = False
    t0 = time.perf_counter()
    res = algo.run(seed=seed)
    wall = time.perf_counter() - t0
    row = dict(problem=pb, method=m, seed=seed, wall_s=wall, pf=res.pf,
               n_calls=res.n_calls, n_calls_total=res.n_calls_total,
               n_restarts=res.n_restarts, n_iter=res.n_iterations,
               converged=res.converged, peak_rss_mb=res.peak_rss_mb)
    row.update({f"t_{k}": v for k, v in res.timers.items()})
    q.put(row)


def main():
    from run_experiments import CONFIG
    p = argparse.ArgumentParser()
    p.add_argument("--problems", nargs="+", default=CONFIG["problems"])
    p.add_argument("--methods", nargs="+", default=CONFIG["methods"])
    p.add_argument("--n-seeds", type=int, default=5)
    p.add_argument("--rel-gap-tol", type=float, default=CONFIG["rel_gap_tol"])
    p.add_argument("--out-dir", default="timing_results")
    a = p.parse_args()
    out = Path(a.out_dir); out.mkdir(exist_ok=True)
    ctx = mp.get_context("spawn"); rows = []
    for pb in a.problems:
        for m in a.methods:
            for s in range(a.n_seeds):
                q = ctx.Queue()
                pr = ctx.Process(target=_worker, args=((pb, m, s, a.rel_gap_tol), q))
                pr.start(); row = q.get(); pr.join()
                rows.append(row)
                print(f"{pb:14s} {m:6s} seed={s} wall={row['wall_s']:.0f}s "
                      f"rss={row['peak_rss_mb']:.0f}MB calls={row['n_calls_total']}", flush=True)
                keys = sorted({k for r in rows for k in r})
                with open(out / "timing.csv", "w", newline="") as fh:
                    w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print("wrote", out / "timing.csv")


if __name__ == "__main__":
    main()
