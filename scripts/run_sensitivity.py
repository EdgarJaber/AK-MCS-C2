#!/usr/bin/env python
"""One-at-a-time sensitivity of AK-MCS-C2 (C2 selection + conformal stop,
J-minmax-GP unless --variant jp) around the baseline protocol (R4.9).

Factors and levels (baseline in bold in the article):
  alpha     : conformal level            0.05, **0.10**, 0.20
  n_init    : initial design size        10, **20**, 40
  nugget    : fixed GP noise in standardised-y units (learn_noise=False)  1e-6, 1e-4, 1e-2
              (baseline learns the noise from the 1e-4 initialisation)
  n_mc      : Monte Carlo population     5000, **10000**, 20000
  eps_stop  : relative gap tolerance     0.02, 0.05, **0.1**, 0.2

One .npz per (problem, setting, seed); resumable/shardable.
  python scripts/run_sensitivity.py --list
  python scripts/run_sensitivity.py --n-seeds 20 --n-jobs 10
"""
import argparse, os, sys, time, traceback
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_experiments import CONFIG  # noqa: E402
from akmcsc2.core import DIAG_COLUMNS  # noqa: E402
from akmcsc2 import AKMCS, Ablation, LOOGPRegressor, get_problem, make_conformal  # noqa: E402

BASE = dict(alpha=0.1, n_init=20, n_mc=10000, eps_stop=0.1, nugget=None)
SETTINGS = {"baseline": {}}
for v in (0.05, 0.2): SETTINGS[f"alpha{v}"] = dict(alpha=v)
for v in (10, 40): SETTINGS[f"ninit{v}"] = dict(n_init=v)
for v in (1e-6, 1e-4, 1e-2): SETTINGS[f"nugget{v:g}"] = dict(nugget=v)
for v in (5000, 20000): SETTINGS[f"nmc{v}"] = dict(n_mc=v)
for v in (0.02, 0.05, 0.2): SETTINGS[f"eps{v}"] = dict(eps_stop=v)


def run_one(pb, setting, seed, out_dir, variant, predict_batch):
    out = out_dir / f"{pb}__{setting}-{variant}__seed{seed:03d}.npz"
    if out.exists():
        return f"[skip] {out.name}"
    cfg = dict(BASE, **SETTINGS[setting]); t0 = time.perf_counter()
    try:
        gp_cfg = dict(CONFIG["gp"])
        if cfg["nugget"] is not None:
            gp_cfg.update(noise=cfg["nugget"], learn_noise=False,
                          noise_lower=min(1e-4, cfg["nugget"]))
        algo = AKMCS(
            problem=get_problem(pb),
            learning=Ablation("C2", "C2", rel_gap_tol=cfg["eps_stop"],
                              candidate_size=CONFIG["candidate_size"],
                              batch_q=CONFIG["batch_q"], batch_delta=CONFIG["batch_delta"]),
            conformal=make_conformal("j+gp" if variant == "jp" else "j-mm-gp", alpha=cfg["alpha"]),
            gp_factory=lambda: LOOGPRegressor(**gp_cfg),
            n_mc=cfg["n_mc"], n_init=cfg["n_init"], max_calls=CONFIG["max_calls"][pb],
            min_iterations=CONFIG["min_iterations"], conformal_every=1,
            track_coverage=True, predict_batch_size=predict_batch,
            gp_interval_alpha=cfg["alpha"],
        )
        res = algo.run(seed=seed)
        np.savez_compressed(
            out, pf=res.pf, pf_lower=res.pf_lower, pf_upper=res.pf_upper,
            n_calls=res.n_calls, n_calls_total=res.n_calls_total, n_restarts=res.n_restarts,
            converged=res.converged, pf_mc_ref=res.pf_mc_ref,
            pf_ref=np.nan if algo.problem.pf_ref is None else algo.problem.pf_ref,
            seed=seed, diag=res.diag_array()[-1], diag_cols=np.array(DIAG_COLUMNS),
            wall_s=time.perf_counter() - t0,
        )
        return f"[done] {out.name} pf={res.pf:.3e} ncall={res.n_calls_total} ({time.perf_counter()-t0:.0f}s)"
    except Exception:
        (out_dir / (out.stem + ".FAILED.txt")).write_text(traceback.format_exc())
        return f"[FAIL] {out.name}"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", default="results_sensitivity")
    p.add_argument("--problems", nargs="+", default=["branch2d_k7", "oscillator6d"])
    p.add_argument("--settings", nargs="+", default=None)
    p.add_argument("--variant", choices=["mm", "jp"], default="mm")
    p.add_argument("--n-seeds", type=int, default=20)
    p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument("--predict-batch", type=int, default=8192)
    p.add_argument("--list", action="store_true")
    a = p.parse_args()
    if a.list:
        print("\n".join(SETTINGS)); print(len(SETTINGS), "settings"); return
    sets = a.settings or list(SETTINGS)
    out_dir = Path(a.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    tasks = [(pb, st, s) for s in range(a.n_seeds) for pb in a.problems for st in sets]
    print(f"{len(tasks)} tasks -> {out_dir}/", flush=True)
    if a.n_jobs > 1:
        from joblib import Parallel, delayed
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        for m in Parallel(n_jobs=a.n_jobs, verbose=5)(
                delayed(run_one)(pb, st, s, out_dir, a.variant, a.predict_batch) for pb, st, s in tasks):
            print(m)
    else:
        for pb, st, s in tasks:
            print(run_one(pb, st, s, out_dir, a.variant, a.predict_batch), flush=True)
    print("All done.")


if __name__ == "__main__":
    main()
