#!/usr/bin/env python3
"""
Diagnose ONE cascade point (grid-size mismatch + rho_e clamping).

Run from the repo folder, with the venv active (same place you run cmb_scan.py):

    source .venv/bin/activate
    python diag_point.py --dm 6 --logtau 7.5
    python diag_point.py --dm 6 --logtau 7.5 --dtau-max 1.0
    python diag_point.py --dm 6 --logtau 7.5 --den-fac 1e-4

Nothing is written to the shard folder; this only prints.
"""
import argparse, re, sys, warnings
import numpy as np
import cmb_scan as c
import spectroxide
from Cascades.cascade.mckeen_bridge import (
    omega_inj_and_Tn_eV, make_mckeen_dq_dz, make_mckeen_photon_source)

ap = argparse.ArgumentParser()
ap.add_argument("--dm", type=float, required=True)
ap.add_argument("--logtau", type=float, required=True)
ap.add_argument("--den-fac", type=float, default=c.DEN_FAC)
ap.add_argument("--dtau-max", type=float, default=c.DTAU_MAX)
ap.add_argument("--xcut", type=float, default=c.X_CUT, help="photon-line cut (bridge x_max_factor); must be < PDE x_max=60")
ap.add_argument("--old-table", action="store_true", help="old behaviour: log table on [X_MIN, X_MAX], no custom x grid")
a = ap.parse_args()

# cmb_scan's helpers read these module-level values, so overriding them here is enough
c.DEN_FAC, c.DTAU_MAX, c.X_CUT = a.den_fac, a.dtau_max, a.xcut
print(f"dm={a.dm} MeV  log10(tau)={a.logtau}  den_fac={c.DEN_FAC:g}  dtau_max={c.DTAU_MAX:g}  x_cut={c.X_CUT:g}  table={'old logspace' if a.old_table else 'X_TABLE'}")

omega, _ = omega_inj_and_Tn_eV(a.dm * 1e6)
gam = 10.0 ** (-a.logtau)
kw = dict(method="pde", cosmo=c.COSMO_AK19, z_start=c.Z_START, z_end=c.Z_END,
          z_min=c.Z_END, z_max=c.Z_START, n_z=c.N_Z, x_min=c.X_MIN, x_max=c.X_MAX,
          n_x=c.N_X, timeout=c.TIMEOUT_S, dtau_max=c.DTAU_MAX)

def run(label, **src):
    extra = {} if (a.old_table or 'photon_source' not in src) else dict(x=c.X_TABLE)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        r = spectroxide.solve(**src, **extra, **kw)
    msgs = [str(x.message) for x in w]
    clamps = [int(m.group(1)) for m in (re.search(r"clamped (\d+) times", s) for s in msgs) if m]
    newton = sum("Newton iteration did not converge" in s for s in msgs)
    print(f"[{label}] mu={r.mu:.4e} y={r.y:.4e} grid_points={len(r.x)} "
          f"x_range=({r.x[0]:.3g}, {r.x[-1]:.3g}) rho_e_clamps={sum(clamps)} newton_fail={newton}")
    return r

r_h = run("heat  ", dq_dz=make_mckeen_dq_dz(omega, gam, c.DEN_FAC, x_max_factor=c.X_CUT))
src = make_mckeen_photon_source(omega, gam, c.DEN_FAC, x_max_factor=c.X_CUT, x_min=c.X_MIN)
r_p = run("photon", photon_source=src)

xr, xe = c.line_position(omega, c.X_CUT)
print(f"line placement: x_raw={xr:.1f} -> x_eff={xe:.1f}  (PDE grid ends at x={r_p.x[-1]:.0f})")
print(f"\ngrid sizes: heat={len(r_h.x)} photon={len(r_p.x)} difference={len(r_p.x)-len(r_h.x)}")
if len(r_h.x) != len(r_p.x):
    extra = np.setdiff1d(np.round(r_p.x, 10), np.round(r_h.x, 10))
    print(f"  points only in photon grid: {len(extra)}  range [{extra.min():.4g}, {extra.max():.4g}]")

# where does the tabulated photon line actually sit, and is it inside the PDE grid?
xt = np.logspace(np.log10(c.X_MIN), np.log10(c.X_MAX), c.N_X) if a.old_table else c.X_TABLE
for z in (1e6, 3e5, 1e5):
    s = np.asarray(src(xt, z), float)
    if s.max() > 0:
        w = xt**3 * s
        w_tot = np.trapezoid(w, xt) if hasattr(np, "trapezoid") else np.trapz(w, xt)
        above = xt > r_p.x[-1]
        print(f"  z={z:.0e}: line peak at x={xt[np.argmax(s)]:.1f}; PDE grid ends at x={r_p.x[-1]:.1f}; "
              f"fraction of tabulated energy above PDE x_max = "
              f"{(np.trapezoid(w[above], xt[above]) / w_tot if above.sum() > 1 else 0.0):.3f}")
    else:
        print(f"  z={z:.0e}: photon source is identically zero")

print(f"\nmu/den_fac = {(r_h.mu + r_p.mu)/c.DEN_FAC:.4e}   y/den_fac = {(r_h.y + r_p.y)/c.DEN_FAC:.4e}")
print("Compare these two numbers between runs with different --den-fac / --dtau-max.")
