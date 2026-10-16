#!/usr/bin/env python3
"""
CMB (mu, y) parameter-space sweep -- stand-alone version of the notebook's COMPUTE cell
(section 6, "CMB parameter space ... COMPUTE CELL").  Same physics, grids and settings; only the
driver differs:

  * you choose which Delta_m values this machine runs (--dm ...), so 4 machines can split the grid;
  * every finished point is appended to a JSONL file and fsync'ed to disk immediately
    (a power cut loses at most the points that were still being computed);
  * re-running the same command resumes: points already on disk are skipped.

Run from the folder that contains Cascades/ (or install your modules), e.g.

    python cmb_scan.py --dm 0.5 1 3 6
    python cmb_scan.py --list                    # show grid + number of points

Merge the shard files from all machines afterwards with merge_cmb.py.
"""
import argparse, os, sys, json, hashlib, time, math, socket, signal
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

# make `Cascades` importable when running from this folder
for _p in (str(Path(__file__).resolve().parent), os.getcwd()):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from spectroxide.solver import solve as solve_heat           # cell 68 interface
import spectroxide                                           # cell 82 interface (spectroxide.solve)
from Cascades.cascade.mckeen_bridge import (
    omega_inj_and_Tn_eV, make_mckeen_dq_dz, make_mckeen_photon_source,
    validate_omega_inj_for_high_energy, validate_pde_z_range,
)

# ── grids (identical to the notebook) ──────────────────────────────────────
DM_GRID_CMB = np.array([0.5, 1, 3, 6, 10, 15, 25, 40, 75, 100, 120], float)   # MeV
BETAS_CMB   = [0.0, 0.5, 0.9]
LOGTAU_GRID = np.arange(5.5, 16.5 + 1e-9, 0.5)                                # log10(tau/s)
RUN_FAMILIES = [f"beta{b:g}" for b in BETAS_CMB] + ["cascade"]

# ── physics / solver settings (identical to the notebook; they define TAG) ──
DEN_FAC   = 0.01
M_N_EV    = 939.565379e6
E_HEAT_EV = 0.301791e6
Z_START, Z_END = 5.0e6, 5.0e2
X_MIN, X_MAX, N_X, N_Z = 1.5e-4, 80.0, 1000, 500
TIMEOUT_S, DTAU_MAX = 3600, 5.0
COSMO_AK19 = dict(h=0.6688, omega_b=0.0221/0.6688**2, omega_m=(0.0221+0.1206)/0.6688**2,
                  y_p=0.24, t_cmb=2.726, n_eff=3.046)
validate_pde_z_range(Z_START, Z_END, Z_END, Z_START)

SETTINGS = dict(den_fac=DEN_FAC, e_heat_ev=E_HEAT_EV, z=(Z_START, Z_END), x=(X_MIN, X_MAX, N_X),
                n_z=N_Z, dtau_max=DTAU_MAX, cosmo_cascade=COSMO_AK19, version=1)
TAG = hashlib.md5(json.dumps(SETTINGS, sort_keys=True).encode()).hexdigest()[:8]

def pkey(family, dm, logtau):
    return f"{family}|dm={dm:.6g}|lt={logtau:.3f}"

# ── kinematics + one PDE point (verbatim from the notebook) ────────────────
def two_body_recoil(dm_eV, m_gp_eV, m1_eV=M_N_EV):
    M  = m1_eV + dm_eV
    E1 = (M**2 + m1_eV**2 - m_gp_eV**2) / (2*M)
    return E1 - m1_eV

def run_heat_point(beta, dm_MeV, logtau):
    dm_eV = dm_MeV*1e6
    F_X = DEN_FAC * (E_HEAT_EV + two_body_recoil(dm_eV, beta*dm_eV))
    r = solve_heat(
        injection={'type': 'decaying_particle', 'f_x': F_X, 'gamma_x': 10.0**(-logtau)},
        z_start=Z_START, z_end=Z_END, x_min=X_MIN, x_max=X_MAX, n_x=N_X,
        method='pde', production_grid=False, number_conserving=True,
        timeout=TIMEOUT_S, dtau_max=DTAU_MAX)
    return dict(mu=float(r.mu), y=float(r.y), status="ok")

def preflight_cascade(dm_MeV):
    """The bridge's dq_dz / photon_source closures swallow *every* exception and return 0.0, so a broken
    cascade module silently yields mu = y = 0.  Call the underlying pieces directly (no try/except) so the
    real error is raised here, once, before any hours of compute."""
    from Cascades.cascade import mckeen_bridge as mb
    omega_inj_eV, _ = omega_inj_and_Tn_eV(dm_MeV*1e6)
    if not validate_omega_inj_for_high_energy(omega_inj_eV):
        return
    mb._cached_budget("gamma", omega_inj_eV, 1.0e5, high_energy=True)          # raises the real error if broken
    v = make_mckeen_dq_dz(omega_inj_eV, 1e-10, DEN_FAC, x_max_factor=X_MAX)(1.0e5)
    if not (np.isfinite(v) and v != 0.0):
        raise RuntimeError(f"cascade dq_dz(z=1e5) returned {v!r} for dm={dm_MeV:g} (bridge is failing silently)")

def run_cascade_point(dm_MeV, logtau):
    dm_eV, gamma_x = dm_MeV*1e6, 10.0**(-logtau)
    omega_inj_eV, _ = omega_inj_and_Tn_eV(dm_eV)
    if not validate_omega_inj_for_high_energy(omega_inj_eV):
        return dict(mu=float("nan"), y=float("nan"), status="skip_below_pair_threshold")
    kw = dict(method="pde", cosmo=COSMO_AK19, z_start=Z_START, z_end=Z_END, z_min=Z_END, z_max=Z_START,
              n_z=N_Z, x_min=X_MIN, x_max=X_MAX, n_x=N_X, timeout=TIMEOUT_S, dtau_max=DTAU_MAX)
    dq_dz = make_mckeen_dq_dz(omega_inj_eV, gamma_x, DEN_FAC, x_max_factor=X_MAX)
    r_h = spectroxide.solve(dq_dz=dq_dz, **kw)
    src = make_mckeen_photon_source(omega_inj_eV, gamma_x, DEN_FAC, x_max_factor=X_MAX, x_min=X_MIN)
    r_p = spectroxide.solve(photon_source=src, **kw)
    if not np.allclose(r_h.x, r_p.x):
        raise RuntimeError("heat/photon x grids differ")
    mu, y = float(r_h.mu + r_p.mu), float(r_h.y + r_p.y)
    if not (math.isfinite(mu) and math.isfinite(y)) or (mu == 0.0 and y == 0.0):
        raise RuntimeError(f"cascade point gave mu={mu}, y={y} (zero/non-finite => bridge failed silently)")
    return dict(mu=mu, y=y, status="ok")

def run_point(family, dm, logtau):
    if family == "cascade":
        return run_cascade_point(dm, logtau)
    return run_heat_point(float(family.removeprefix("beta")), dm, logtau)

# ── crash-safe storage: one JSON line per finished point, fsync'ed ─────────
def load_done(shard_dir):
    """All points already stored by any shard file of this TAG in shard_dir."""
    done = {}
    for f in sorted(shard_dir.glob(f"cmb_{TAG}_*.jsonl")):
        with open(f, "r") as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue                      # torn last line after a power cut: that point is simply redone
                if "key" in rec:
                    done[rec["key"]] = rec["point"]
    return done

class ShardWriter:
    def __init__(self, path):
        existed = path.exists() and path.stat().st_size > 0
        self.f = open(path, "ab")
        if existed:                               # repair a torn last line so new records start on a fresh line
            with open(path, "rb") as r:
                r.seek(-1, os.SEEK_END)
                if r.read(1) != b"\n":
                    self.f.write(b"\n")
        else:
            self._write({"_tag": TAG, "_settings": SETTINGS})
            dfd = os.open(path.parent, os.O_RDONLY)       # make the new file's directory entry durable too
            try: os.fsync(dfd)
            finally: os.close(dfd)
    def _write(self, rec):
        self.f.write((json.dumps(rec) + "\n").encode())
        self.f.flush(); os.fsync(self.f.fileno())
    def add(self, key, point):
        self._write(dict(key=key, point=point))

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dm", type=float, nargs="+", help="Delta_m values (MeV) for this machine; must be in the grid")
    ap.add_argument("--families", nargs="+", default=RUN_FAMILIES, choices=RUN_FAMILIES)
    ap.add_argument("--workers", type=int, default=os.cpu_count(), help="parallel PDE solves (default: all cores)")
    ap.add_argument("--shard-dir", default="Parameter-Space-data/shards")
    ap.add_argument("--name", default=socket.gethostname(), help="label used in the output file name")
    ap.add_argument("--list", action="store_true", help="print the grid and exit")
    args = ap.parse_args()

    n_per_dm = len(args.families) * len(LOGTAU_GRID)
    if args.list or not args.dm:
        print(f"TAG = {TAG}")
        print(f"dm grid ({len(DM_GRID_CMB)}): {' '.join(f'{d:g}' for d in DM_GRID_CMB)}")
        print(f"log10 tau grid: {len(LOGTAU_GRID)} points ({LOGTAU_GRID[0]:g}..{LOGTAU_GRID[-1]:g})")
        print(f"families: {args.families}  ->  {n_per_dm} points per dm, {n_per_dm*len(DM_GRID_CMB)} in total")
        return

    # snap requested dm to the grid values (so the cache keys match the notebook exactly)
    dms = []
    for d in args.dm:
        i = int(np.argmin(np.abs(DM_GRID_CMB - d)))
        if not np.isclose(DM_GRID_CMB[i], d):
            sys.exit(f"dm={d:g} is not in the grid: {[float(x) for x in DM_GRID_CMB]}")
        dms.append(float(DM_GRID_CMB[i]))

    shard_dir = Path(args.shard_dir); shard_dir.mkdir(parents=True, exist_ok=True)
    out = shard_dir / f"cmb_{TAG}_{args.name}.jsonl"
    done_pts = load_done(shard_dir)
    todo = [(f, dm, float(lt)) for f in args.families for dm in dms for lt in LOGTAU_GRID
            if pkey(f, dm, lt) not in done_pts]
    n_all = n_per_dm * len(dms)
    print(f"[{args.name}] TAG={TAG}  dm={dms}  families={args.families}", flush=True)
    print(f"[{args.name}] {n_all-len(todo)}/{n_all} already on disk, {len(todo)} to run, "
          f"{min(args.workers, max(len(todo),1))} workers -> {out}", flush=True)
    if not todo:
        return

    if "cascade" in {f for f, _, _ in todo}:
        for dm in sorted({d for f, d, _ in todo if f == "cascade"}):
            preflight_cascade(dm)                 # aborts with the real traceback if the cascade module is broken
        print(f"[{args.name}] cascade pre-flight OK", flush=True)

    writer = ShardWriter(out)
    fail_log = shard_dir / f"failed_{args.name}.log"

    def _term(signum, frame):                     # SIGTERM (e.g. `pkill`) behaves like Ctrl+C
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, _term)

    t0 = time.time(); ok = failed = 0
    ex = ThreadPoolExecutor(max_workers=min(args.workers, len(todo)))
    try:
        futs = {ex.submit(run_point, *p): p for p in todo}
        for fut in as_completed(futs):
            f, dm, lt = futs[fut]
            try:
                writer.add(pkey(f, dm, lt), fut.result())
                ok += 1
            except Exception as e:
                failed += 1
                msg = f"FAILED {pkey(f, dm, lt)}: {e!r}"
                print("  " + msg, flush=True)
                with open(fail_log, "a") as fl:
                    fl.write(msg + "\n")
            n = ok + failed; el = time.time() - t0
            eta = el/n*(len(todo)-n)
            print(f"  {n}/{len(todo)}  ok={ok} failed={failed}  last={f} dm={dm:g} lt={lt:g}  "
                  f"elapsed {el/60:.1f} min, ETA {eta/60:.1f} min", flush=True)
    except KeyboardInterrupt:
        print("interrupted - every finished point is already on disk; rerun the same command to resume", flush=True)
        ex.shutdown(wait=False, cancel_futures=True)
        os._exit(130)                              # don't wait for PDE solves that are still running
    ex.shutdown()
    print(f"[{args.name}] finished: {ok} ok, {failed} failed (failed points are retried on the next run)", flush=True)

if __name__ == "__main__":
    main()
