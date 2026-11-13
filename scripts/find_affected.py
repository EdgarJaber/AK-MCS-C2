#!/usr/bin/env python
"""List (and optionally move away) the runs that the exploration safeguard changes.

The safeguard only acts at iterations where the conformal upper failure
probability P_f^+ is zero. A saved run of a cell involving C2 whose history
never has pf_upper == 0 therefore follows exactly the same trajectory with the
safeguard (same seed), and need not be rerun. Affected runs are moved to
<dir>_prefix/ so that run_ablation.py (resumable) recomputes only those.

  python scripts/find_affected.py results_ablation          # report
  python scripts/find_affected.py results_ablation --move   # move affected away
"""
import argparse, glob, shutil
from collections import Counter
from pathlib import Path
import numpy as np
p = argparse.ArgumentParser(); p.add_argument("results"); p.add_argument("--move", action="store_true"); a = p.parse_args()
res = Path(a.results); dst = res.with_name(res.name + "_prefix")
tot = Counter(); aff = Counter(); moved = []
for f in sorted(glob.glob(str(res / "*.npz"))):
    pb, cell, _ = Path(f).stem.split("__")
    if "C2" not in cell:
        continue
    tot[(pb, cell)] += 1
    with np.load(f, allow_pickle=True) as d:
        pu = d["history"][:, 3]
    if np.any(pu[np.isfinite(pu)] <= 0.0):
        aff[(pb, cell)] += 1; moved.append(f)
for k in sorted(tot):
    if aff[k]: print(f"{k[0]:15s} {k[1]:10s} affected {aff[k]:2d}/{tot[k]}")
print(f"TOTAL affected {len(moved)}/{sum(tot.values())} runs in cells involving C2")
if a.move and moved:
    dst.mkdir(exist_ok=True)
    for f in moved: shutil.move(f, dst / Path(f).name)
    print("moved to", dst)
