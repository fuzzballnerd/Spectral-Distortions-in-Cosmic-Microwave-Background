#!/usr/bin/env python3
"""
Find which points clamp rho_e / fail Newton, and test whether dtau_max or den_fac changes it.
Uses cmb_scan's own point runners and warning capture; writes NOTHING to the shard folder.

Run from the repo folder with the venv active:

    python clamp_scan.py --dm 75 --families beta0.5 cascade --logtau 5.5 6.5 7.5 8.5 9.5 11 13
    python clamp_scan.py --dm 75 --families cascade --logtau 6.5 --dtau-max 1.0
    python clamp_scan.py --dm 75 --families cascade --logtau 6.5 --den-fac 1e-4

Columns: rho_e_clamps and newton_fail come from the solver warnings of that point; mu/den and y/den are the
raw (floor-included) sums divided by den_fac, handy for comparing runs with different den_fac.
"""
import argparse, time, warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import cmb_scan as c

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--dm", type=float, nargs="+", default=[75.0])
ap.add_argument("--families", nargs="+", default=["beta0.5", "cascade"], choices=c.RUN_FAMILIES)
ap.add_argument("--logtau", type=float, nargs="+", default=[5.5, 6.5, 7.5, 8.5, 9.5, 11.0, 13.0])
ap.add_argument("--dtau-max", type=float, default=c.DTAU_MAX)
ap.add_argument("--den-fac", type=float, default=c.DEN_FAC)
ap.add_argument("--workers", type=int, default=4)
a = ap.parse_args()

c.DTAU_MAX, c.DEN_FAC = a.dtau_max, a.den_fac            # cmb_scan's runners read these module globals
warnings.simplefilter("always")
warnings.showwarning = c._showwarning                    # per-thread capture, as in the scan

todo = [(f, dm, lt) for f in a.families for dm in a.dm for lt in a.logtau]
print(f"{len(todo)} points, {min(a.workers, len(todo))} workers, dtau_max={c.DTAU_MAX:g}, den_fac={c.DEN_FAC:g}, "
      f"x_cut={c.X_CUT:g}", flush=True)

rows = []
t0 = time.time()
with ThreadPoolExecutor(max_workers=min(a.workers, len(todo))) as ex:
    futs = {ex.submit(c.run_point_logged, *p): p for p in todo}
    for fut in as_completed(futs):
        f, dm, lt = futs[fut]
        try:
            p = fut.result()
            row = (f, dm, lt, p.get("rho_e_clamps", 0), p.get("newton_fail", 0), p["mu"], p["y"],
                   ",".join(p.get("flags", [])) or "-", p.get("wall_s"), p.get("z_decay"))
        except Exception as e:
            row = (f, dm, lt, None, None, float("nan"), float("nan"), f"FAILED {e!r}"[:60], None, None)
        rows.append(row)
        print(f"  done {len(rows)}/{len(todo)}: {f} dm={dm:g} lt={lt:g}  clamps={row[3]} newton_fail={row[4]}  "
              f"({(time.time()-t0)/60:.1f} min)", flush=True)

print("\nfamily    dm     lt    z_dec  clamps  newton    mu/den      y/den     wall_s  flags")
for f, dm, lt, cl, nf, mu, y, fl, w, zd in sorted(rows, key=lambda r: (r[0], r[1], r[2])):
    print(f"{f:8s} {dm:5g} {lt:6.2f} {zd if zd is None else format(zd,'.3g'):>8} {str(cl):>7s} {str(nf):>7s} {mu/c.DEN_FAC:10.3e} {y/c.DEN_FAC:10.3e} "
          f"{str(w):>8s}  {fl}")
