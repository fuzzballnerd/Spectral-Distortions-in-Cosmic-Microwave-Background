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

Cascade line placement (version 2): the photon line is pinned below X_CUT = 35 (inside the PDE grid, which ends at
x = 60) and tabulated on a fine grid X_TABLE.  Cascade points store x_cut/x_raw/x_eff/line_pinned/z_decay; old
cascade points made with another x_cut are recomputed automatically, beta points are reused (TAG unchanged).

Logging / flags (added):
  * every stored point has `flagged` (bool) and `flags` (list) -- e.g. rho_e_clamped, newton_not_converged,
    solver_warning, photon_line_beyond_pde_grid, photon_source_zero, decay_outside_pde_window.  Flagged points are still stored
    (status stays "ok"); filter on `flagged` / `flags` when merging.
  * warnings_<name>.log : one line per flagged point;  failed_<name>.log : message + traceback per failed point.
  * cascade points also store mu_heat/y_heat/mu_phot/y_phot, grid sizes, and where the photon line sits.
"""
import argparse, os, sys, json, hashlib, time, math, socket, signal, re, threading, traceback, warnings
from collections import Counter
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

# ── photon-line placement for the cascade (NEW, version 2) ──────────────────
# The PDE integrates on spectroxide's production grid, x in [1e-5, 60] -- NOT on [X_MIN, X_MAX].  The bridge pins
# the line below x_cut, so x_cut must sit well inside 60.  35 is the bridge's own default (and the paper's cap
# for the non-relativistic kernel).  With the old value (80) the line sat at x ~ 75-79, entirely off the grid, and
# its energy was lost.  The photon table gets its own grid: log below 25, fine linear spacing (~0.025, matching the
# PDE grid) from 25 to 40 so the narrow Gaussian is resolved.
X_CUT = 35.0
_xt = np.r_[np.logspace(np.log10(X_MIN), np.log10(25.0), 500, endpoint=False), np.linspace(25.0, 40.0, 600)]
# spectroxide's CSV writer prints each x with 10 significant digits and Rust requires a strictly ascending grid,
# so de-duplicate *after* rounding to that precision (np.unique on raw floats left 25.0 twice).
X_TABLE = np.array(sorted({float(f"{v:.10e}") for v in _xt}))
assert np.all(np.diff(X_TABLE) > 0)
CASCADE_SETTINGS = dict(version=2, x_cut=X_CUT, x_table=(float(X_TABLE[0]), float(X_TABLE[-1]), int(len(X_TABLE))))
M_E_EV, E_CMB_MEAN_FACTOR = 0.51099895e6, 2.701     # only used to record where the line is placed

def line_position(omega_inj_eV, x_cut=X_CUT):
    """Mirror of the bridge's x_eff construction, for the record (x_raw, x_eff)."""
    E_each = max(omega_inj_eV - 2.0*M_E_EV, 0.0) / 2.0
    g = 1.0 + E_each / M_E_EV
    raw = (4.0/3.0) * g**2 * (1.0 - 1.0/g**2) * E_CMB_MEAN_FACTOR
    anchor = 0.9 * x_cut
    if raw <= anchor:
        return raw, max(raw, 1.0)
    u = math.log(raw / anchor)
    return raw, anchor + (x_cut - anchor) * u / (1.0 + u)

# ── decay redshift: z at which t(z) = tau (matter + radiation + Lambda, cosmology = COSMO_AK19) ──────
def _build_t_of_z():
    h, Om = COSMO_AK19["h"], COSMO_AK19["omega_m"]
    Or = (1 + 0.2271*COSMO_AK19["n_eff"]) * 2.473e-5 * (COSMO_AK19["t_cmb"]/2.7255)**4 / h**2
    OL = 1.0 - Om - Or
    H0 = 100.0*h*1e5/3.0857e24
    u = np.linspace(math.log(2.0), math.log(1e10 + 1.0), 20001)        # u = ln(1+z), z from 1 to 1e10
    a = np.exp(u); f = 1.0 / (H0*np.sqrt(Or*a**4 + Om*a**3 + OL))
    seg = 0.5*(f[1:] + f[:-1]) * np.diff(u)
    return u, np.concatenate([np.cumsum(seg[::-1])[::-1], [0.0]])      # t(z) = int_z^inf dz'/((1+z')H)
_U_T, _T_Z = _build_t_of_z()

def z_at_time(t_s):
    """Redshift at cosmic time t_s (seconds)."""
    return float(math.exp(np.interp(-t_s, -_T_Z, _U_T)) - 1.0)

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
        method='pde', production_grid=True, number_conserving=True,
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
    v = make_mckeen_dq_dz(omega_inj_eV, 1e-10, DEN_FAC, x_max_factor=X_CUT)(1.0e5)
    if not (np.isfinite(v) and v != 0.0):
        raise RuntimeError(f"cascade dq_dz(z=1e5) returned {v!r} for dm={dm_MeV:g} (bridge is failing silently)")

# ── per-point warning capture + flags ──────────────────────────────────────
# spectroxide re-emits the Rust solver's diagnostics via warnings.warn() in the *calling thread*.
# We install one global hook that stores them in a thread-local buffer, so every worker thread
# collects exactly the warnings of the point it is computing (warnings.catch_warnings is NOT thread-safe).
_tls = threading.local()
_orig_showwarning = warnings.showwarning

def _showwarning(message, category, filename, lineno, file=None, line=None):
    text = str(message)
    buf = getattr(_tls, "buf", None)
    if buf is not None:                        # inside a worker: keep it for the point's record/log, don't spam the terminal
        buf.append(text)
        return
    if "Progress:" in text:
        return
    _orig_showwarning(message, category, filename, lineno, file, line)

_CLAMP_RE  = re.compile(r"clamped (\d+) times")

def classify_warnings(msgs):
    """Turn raw solver warnings into (flags, info).  'Progress:' lines are ignored."""
    flags, info, other = [], dict(rho_e_clamps=0, newton_fail=0), []
    seen = set()
    for m in msgs:
        if "Progress:" in m or m in seen:
            continue
        seen.add(m)
        mc = _CLAMP_RE.search(m)
        if mc:
            info["rho_e_clamps"] += int(mc.group(1))
        elif "Newton iteration did not converge" in m:
            info["newton_fail"] += 1
        else:
            other.append(m)
    if info["rho_e_clamps"]:
        flags.append("rho_e_clamped")          # injection energy was silently discarded
    if info["newton_fail"]:
        flags.append("newton_not_converged")
    if other:
        flags.append("solver_warning")
    info["warnings"] = [m[:300] for m in other[:5]]
    return flags, info

def _trapz(y, x):
    return float((getattr(np, "trapezoid", None) or np.trapz)(y, x))

def probe_photon_line(src, logtau, x_pde_max):
    """Where does the tabulated photon line sit, and how much of its energy lies beyond the PDE grid?
    (z-position of the line does not matter much; probe at the decay redshift, where the source is non-zero)."""
    xt = X_TABLE
    z_probe = float(np.clip(math.sqrt(2.39e19 / 10.0**logtau), 1.0e3, 0.98 * Z_START))
    s = np.asarray(src(xt, z_probe), float)
    w = xt**3 * s
    tot = _trapz(w, xt)
    if not np.isfinite(tot) or tot <= 0.0:
        return dict(line_peak_x=None, line_frac_beyond_grid=None, z_probe=z_probe, zero=True)
    above = xt > x_pde_max
    frac = _trapz(w[above], xt[above]) / tot if above.sum() > 1 else 0.0
    return dict(line_peak_x=float(xt[int(np.argmax(s))]), line_frac_beyond_grid=float(frac),
                z_probe=z_probe, zero=False)

def run_cascade_point(dm_MeV, logtau):
    dm_eV, gamma_x = dm_MeV*1e6, 10.0**(-logtau)
    omega_inj_eV, _ = omega_inj_and_Tn_eV(dm_eV)
    if not validate_omega_inj_for_high_energy(omega_inj_eV):
        return dict(mu=float("nan"), y=float("nan"), status="skip_below_pair_threshold")
    kw = dict(method="pde", cosmo=COSMO_AK19, z_start=Z_START, z_end=Z_END, z_min=Z_END, z_max=Z_START,
              n_z=N_Z, x_min=X_MIN, x_max=X_MAX, n_x=N_X, timeout=TIMEOUT_S, dtau_max=DTAU_MAX)
    dq_dz = make_mckeen_dq_dz(omega_inj_eV, gamma_x, DEN_FAC, x_max_factor=X_CUT)   # x_max_factor is ignored here
    r_h = spectroxide.solve(dq_dz=dq_dz, **kw)
    src = make_mckeen_photon_source(omega_inj_eV, gamma_x, DEN_FAC, x_max_factor=X_CUT, x_min=X_MIN)
    r_p = spectroxide.solve(photon_source=src, x=X_TABLE, **kw)

    # ── RELAXED GRID CHECK (this replaces the old `if not np.allclose(r_h.x, r_p.x): raise ...`) ──────────
    # The photon run gets an automatic refinement zone (~399 extra x points); mu and y are scalars computed
    # on each run's own grid, so adding them is fine.  We only record whether the grids matched.
    grid_ok = bool(len(r_h.x) == len(r_p.x) and np.allclose(r_h.x, r_p.x))

    mu, y = float(r_h.mu + r_p.mu), float(r_h.y + r_p.y)
    if not (math.isfinite(mu) and math.isfinite(y)) or (mu == 0.0 and y == 0.0):
        raise RuntimeError(f"cascade point gave mu={mu}, y={y} (zero/non-finite => bridge failed silently)")

    flags, extra = [], {}
    try:                                       # a failing probe must never kill a finished point
        pr = probe_photon_line(src, logtau, float(r_p.x[-1]))
        extra.update(line_peak_x=pr["line_peak_x"], line_frac_beyond_grid=pr["line_frac_beyond_grid"])
        if pr["zero"]:
            flags.append("photon_source_zero")
        elif pr["line_frac_beyond_grid"] > 0.05:
            flags.append("photon_line_beyond_pde_grid")   # photon channel contributes ~nothing
    except Exception as e:
        flags.append("line_probe_failed"); extra["line_probe_error"] = repr(e)[:200]

    x_raw, x_eff = line_position(omega_inj_eV)
    z_decay = math.sqrt(2.39e19 * gamma_x)   # rough z at t = tau (radiation era)
    extra.update(x_raw=x_raw, x_eff=x_eff, line_pinned=bool(x_raw > 0.9 * X_CUT))
    return dict(mu=mu, y=y, status="ok", x_cut=X_CUT, cascade_version=CASCADE_SETTINGS["version"], _flags=flags,
                mu_heat=float(r_h.mu), y_heat=float(r_h.y), mu_phot=float(r_p.mu), y_phot=float(r_p.y),
                grid_h=len(r_h.x), grid_p=len(r_p.x), grid_ok=grid_ok, **extra)

def run_point(family, dm, logtau):
    if family == "cascade":
        return run_cascade_point(dm, logtau)
    return run_heat_point(float(family.removeprefix("beta")), dm, logtau)

def run_point_logged(family, dm, logtau):
    """Run one point, collecting this thread's solver warnings.  Returns a record with `flagged`/`flags`.
    On failure the exception carries the captured warnings and the traceback."""
    _tls.buf = []
    t0 = time.time()
    try:
        point = run_point(family, dm, logtau)
        msgs = list(_tls.buf)
    except Exception as e:
        e.captured_warnings = [m for m in _tls.buf if "Progress:" not in m]
        e.tb = traceback.format_exc()
        raise
    finally:
        _tls.buf = None
    flags = list(point.pop("_flags", []))
    if point.get("status") == "ok":
        z_dec = z_at_time(10.0**logtau)
        point["z_decay"] = z_dec
        if z_dec > Z_START or z_dec < Z_END:      # most of the decay happens before z_start / after z_end: result is truncated
            flags.append("decay_outside_pde_window")
    wflags, info = classify_warnings(msgs)
    flags = wflags + flags
    point.update(info)
    point.update(flagged=bool(flags), flags=flags, wall_s=round(time.time() - t0, 1))
    return point

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
                    p = rec["point"]
                    if rec["key"].startswith("cascade|") and p.get("status") == "ok" \
                            and p.get("x_cut") != X_CUT:
                        continue                  # computed with a different line placement: redo it
                    done[rec["key"]] = p
    return done

class ShardWriter:
    """Append-only JSONL.  Re-opens the file for every record and re-creates the directory if it has
    disappeared, so a deleted/moved folder can never make finished points vanish into an unlinked inode."""
    def __init__(self, path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.stat().st_size > 0:
            with open(path, "rb") as r:        # repair a torn last line so new records start on a fresh line
                r.seek(-1, os.SEEK_END)
                torn = r.read(1) != b"\n"
            if torn:
                with open(path, "ab") as f:
                    f.write(b"\n")
        else:
            self._write({"_tag": TAG, "_settings": SETTINGS, "_cascade": CASCADE_SETTINGS})
    def _write(self, rec):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "ab") as f:
            f.write((json.dumps(rec) + "\n").encode())
            f.flush(); os.fsync(f.fileno())
    def add(self, key, point):
        self._write(dict(key=key, point=point))

def safe_append(path, text):
    """Append to a log file; never raises (a broken log must not kill the scan)."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as fl:
            fl.write(text if text.endswith("\n") else text + "\n")
    except OSError as e:
        print(f"  (could not write {path.name}: {e!r})", flush=True)

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
        print(f"TAG = {TAG}   cascade: x_cut={X_CUT:g}, photon table {len(X_TABLE)} pts in [{X_TABLE[0]:.2g}, {X_TABLE[-1]:g}]")
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

    shard_dir = Path(args.shard_dir).expanduser().resolve(); shard_dir.mkdir(parents=True, exist_ok=True)
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

    warnings.simplefilter("always")               # don't let Python de-duplicate repeated solver warnings
    warnings.showwarning = _showwarning           # per-thread capture (see run_point_logged)
    writer = ShardWriter(out)
    fail_log = shard_dir / f"failed_{args.name}.log"          # points that raised: message + traceback + warnings
    warn_log = shard_dir / f"warnings_{args.name}.log"        # points that finished but were flagged

    def _term(signum, frame):                     # SIGTERM (e.g. `pkill`) behaves like Ctrl+C
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, _term)

    t0 = time.time(); ok = failed = nflag = 0
    flag_counts = Counter()
    ex = ThreadPoolExecutor(max_workers=min(args.workers, len(todo)))
    try:
        futs = {ex.submit(run_point_logged, *p): p for p in todo}
        for fut in as_completed(futs):
            f, dm, lt = futs[fut]
            key = pkey(f, dm, lt)
            try:
                point = fut.result()
            except Exception as e:                 # the point itself failed: log everything, carry on
                failed += 1
                print(f"  FAILED {key}: {e!r}", flush=True)
                safe_append(fail_log, f"FAILED {key}: {e!r}\n"
                            + "".join(f"  solver warning: {m[:300]}\n" for m in getattr(e, "captured_warnings", []))
                            + getattr(e, "tb", "") + "-" * 60)
            else:
                writer.add(key, point)             # an OSError here is fatal on purpose (see except below)
                ok += 1
                if point.get("flagged"):
                    nflag += 1; flag_counts.update(point["flags"])
                    msg = f"FLAGGED {key}: {point['flags']} clamps={point.get('rho_e_clamps')} " \
                          f"newton_fail={point.get('newton_fail')} warnings={point.get('warnings')}"
                    print("  " + msg, flush=True)
                    safe_append(warn_log, msg)
            n = ok + failed; el = time.time() - t0
            eta = el/n*(len(todo)-n)
            print(f"  {n}/{len(todo)}  ok={ok} (flagged={nflag}) failed={failed}  last={f} dm={dm:g} lt={lt:g}  "
                  f"elapsed {el/60:.1f} min, ETA {eta/60:.1f} min", flush=True)
    except KeyboardInterrupt:
        print("interrupted - every finished point is already on disk; rerun the same command to resume", flush=True)
        ex.shutdown(wait=False, cancel_futures=True)
        os._exit(130)                              # don't wait for PDE solves that are still running
    except BaseException as e:                     # e.g. the shard file cannot be written: stop *everything*
        print(f"FATAL in main loop: {e!r} -- stopping all workers", flush=True)
        traceback.print_exc()
        ex.shutdown(wait=False, cancel_futures=True)
        os._exit(1)                                # otherwise worker threads keep computing results nobody saves
    ex.shutdown()
    print(f"[{args.name}] finished: {ok} ok ({nflag} flagged), {failed} failed (failed points are retried on the next run)", flush=True)
    if flag_counts:
        print(f"[{args.name}] flag summary: " + ", ".join(f"{k}={v}" for k, v in flag_counts.most_common()), flush=True)
        print(f"[{args.name}] details: {warn_log}", flush=True)
    if failed:
        print(f"[{args.name}] failure details (with tracebacks): {fail_log}", flush=True)

if __name__ == "__main__":
    main()
