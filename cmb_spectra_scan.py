#!/usr/bin/env python3
"""
Electron-cascade spectra scan: stores the FULL Delta n(x) spectrum of every point, not just (mu, y).

Physics, grids, solver settings, cosmology, photon-line placement and the electron-energy convention are all
imported from cmb_scan.py (same folder), so the results are identical to the (mu, y) scan -- only the output
differs.  Only the electron cascade (chi -> n + X, neutron beta-decay electron cascaded through
mckeen_electron) is run, for the four betas in BETAS (default 0, 0.1, 1/3, 0.9).

Per point two PDE solves are made (as in cmb_scan.run_electron_point):
    heat run   : dq_dz  -> r_h.x, r_h.delta_n
    photon run : photon_source on X_TABLE -> r_p.x, r_p.delta_n   (its grid has extra refinement points)
Both raw spectra are stored (lossless), plus `x_tot`/`dn_tot` = the sum on the union of the two grids
(each spectrum linearly interpolated in ln x onto the union).  Do the statistics on the raw pair if the
narrow line matters.

Storage: one JSON line per finished point, fsync'ed (same scheme as cmb_scan).  Re-running the same command
resumes: points on disk are skipped.  Numbers are stored with 9 significant digits.

    python cmb_spectra_electron.py --list
    python cmb_spectra_electron.py                          # everything, all cores
    python cmb_spectra_electron.py --dm 100 --logtau-min 8 --logtau-max 10 --workers 8
    tmux new -s spec 'python cmb_spectra_electron.py 2>&1 | tee -a spectra_run.log'

Read the data back:
    from cmb_spectra_electron import load_spectra
    pts = load_spectra("Parameter-Space-data/spectra")     # {key: record}
    rec = pts["electron|beta=0.333333|dm=100|lt=9.000"]
    x, dn = np.array(rec["x_tot"]), np.array(rec["dn_tot"])
"""
import argparse, os, sys, json, time, math, socket, signal, warnings, traceback
from collections import Counter
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

for _p in (str(Path(__file__).resolve().parent), os.getcwd()):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cmb_scan as cs
import spectroxide
from Cascades.cascade.mckeen_electron import (
    make_mckeen_electron_dq_dz, make_mckeen_electron_photon_source, electron_line_position,
)

BETAS = [0.0, 0.5, 0.9]
SPEC_VERSION = 1


def pkey(beta, dm, logtau):
    return f"electron|beta={beta:.6g}|dm={dm:.6g}|lt={logtau:.3f}"


def _arr(a):
    return [float(f"{v:.9g}") for v in np.asarray(a, dtype=float)]


def run_point_spectrum(beta, dm_MeV, logtau):
    E_e, E_rest, T_n = cs.electron_energies_eV(beta, dm_MeV)
    if not cs._electron_active(E_e):
        return dict(status="skip_no_electron_energy", E_e_eV=E_e, T_n_eV=T_n)
    gamma_x, kw = 10.0**(-logtau), cs._pde_kw()

    r_h = spectroxide.solve(dq_dz=make_mckeen_electron_dq_dz(E_e, gamma_x, cs.DEN_FAC, E_rest_eV=E_rest), **kw)
    src = make_mckeen_electron_photon_source(E_e, gamma_x, cs.DEN_FAC, x_max_factor=cs.X_CUT, x_min=cs.X_MIN)
    r_p = spectroxide.solve(photon_source=src, x=cs.X_TABLE, **kw)

    xh, dh = np.asarray(r_h.x, float), np.asarray(r_h.delta_n, float)
    xp, dp = np.asarray(r_p.x, float), np.asarray(r_p.delta_n, float)
    mu, y = float(r_h.mu + r_p.mu), float(r_h.y + r_p.y)
    if not (np.all(np.isfinite(dh)) and np.all(np.isfinite(dp)) and math.isfinite(mu) and math.isfinite(y)):
        raise RuntimeError(f"non-finite spectrum/mu/y (mu={mu}, y={y}) => bridge failed silently")
    if mu == 0.0 and y == 0.0 and not (np.any(dh) or np.any(dp)):
        raise RuntimeError("all-zero spectrum => bridge failed silently")

    # total spectrum on the union grid (interpolate in ln x; both grids span ~the same range)
    xt = np.unique(np.r_[xh, xp])
    lx = np.log(xt)
    dn_tot = np.interp(lx, np.log(xh), dh) + np.interp(lx, np.log(xp), dp)

    flags, extra = [], {}
    try:
        pr = cs.probe_photon_line(src, logtau, float(xp[-1]))
        extra.update(line_peak_x=pr["line_peak_x"], line_frac_beyond_grid=pr["line_frac_beyond_grid"])
        if pr["zero"]:
            flags.append("photon_source_zero")
        elif pr["line_frac_beyond_grid"] > 0.05:
            flags.append("photon_line_beyond_pde_grid")
    except Exception as e:
        flags.append("line_probe_failed"); extra["line_probe_error"] = repr(e)[:200]
    x_raw, x_eff = electron_line_position(E_e, cs.X_CUT)
    extra.update(x_raw=x_raw, x_eff=x_eff, line_pinned=bool(x_raw > 0.9*cs.X_CUT))

    return dict(status="ok", spec_version=SPEC_VERSION, beta=beta, dm_MeV=dm_MeV, logtau=logtau,
                E_e_eV=E_e, E_rest_eV=E_rest, T_n_eV=T_n,
                elec_mode=cs.ELECTRON_MODE, x_cut=cs.X_CUT,
                mu=mu, y=y, mu_heat=float(r_h.mu), y_heat=float(r_h.y),
                mu_phot=float(r_p.mu), y_phot=float(r_p.y),
                x_heat=_arr(xh), dn_heat=_arr(dh), x_phot=_arr(xp), dn_phot=_arr(dp),
                x_tot=_arr(xt), dn_tot=_arr(dn_tot),
                grid_h=len(xh), grid_p=len(xp), _flags=flags, **extra)


def run_point_logged(beta, dm, logtau):
    cs._tls.buf = []
    t0 = time.time()
    try:
        point = run_point_spectrum(beta, dm, logtau)
        msgs = list(cs._tls.buf)
    except Exception as e:
        e.captured_warnings = [m for m in cs._tls.buf if "Progress:" not in m]
        e.tb = traceback.format_exc()
        raise
    finally:
        cs._tls.buf = None
    flags = list(point.pop("_flags", []))
    if point.get("status") == "ok":
        z_dec = cs.z_at_time(10.0**logtau)
        point["z_decay"] = z_dec
        if z_dec > cs.Z_START:
            flags.append("decay_before_pde_start")
        elif z_dec < cs.Z_END:
            point["decay_after_pde_end"] = True          # informational only
    wflags, info = cs.classify_warnings(msgs)
    flags = wflags + flags
    point.update(info)
    point.update(flagged=bool(flags), flags=flags, wall_s=round(time.time() - t0, 1),
                 rho_e_cap=os.environ.get("SPECTROXIDE_RHO_E_CAP"))
    return point


# ── crash-safe storage ──────────────────────────────────────────────────────
def load_spectra(shard_dir, tag=None):
    """All finished points of this TAG in shard_dir -> {key: record}.  A torn last line is ignored."""
    shard_dir, tag, done = Path(shard_dir), tag or cs.TAG, {}
    for f in sorted(shard_dir.glob(f"spec_{tag}_*.jsonl")):
        with open(f, "r") as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "key" not in rec:
                    continue
                p = rec["point"]
                if p.get("status") == "ok" and (
                        p.get("elec_mode") != cs.ELECTRON_MODE or p.get("x_cut") != cs.X_CUT
                        or p.get("spec_version") != SPEC_VERSION):
                    continue                              # other settings: redo
                cap = os.environ.get("SPECTROXIDE_RHO_E_CAP")
                if cap and "rho_e_clamped" in (p.get("flags") or []) and p.get("rho_e_cap") != cap:
                    continue
                done[rec["key"]] = p
    return done


class ShardWriter:
    def __init__(self, path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.stat().st_size > 0:
            with open(path, "rb") as r:
                r.seek(-1, os.SEEK_END)
                torn = r.read(1) != b"\n"
            if torn:
                with open(path, "ab") as f:
                    f.write(b"\n")
        else:
            self._write({"_tag": cs.TAG, "_settings": cs.SETTINGS, "_cascade": cs.CASCADE_SETTINGS,
                         "_betas": BETAS, "_spec_version": SPEC_VERSION})

    def _write(self, rec):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "ab") as f:
            f.write((json.dumps(rec) + "\n").encode())
            f.flush(); os.fsync(f.fileno())

    def add(self, key, point):
        self._write(dict(key=key, point=point))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dm", type=float, nargs="+", default=None, help="Delta_m values (MeV), must be in the grid; default: all")
    ap.add_argument("--betas", type=float, nargs="+", default=BETAS, help="default: 0 0.1 1/3 0.5 0.9")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--shard-dir", default="Parameter-Space-data/spectra")
    ap.add_argument("--name", default=socket.gethostname())
    ap.add_argument("--rho-e-cap", type=float, default=None)
    ap.add_argument("--logtau-min", type=float, default=float(cs.LOGTAU_GRID[0]))
    ap.add_argument("--logtau-max", type=float, default=float(cs.LOGTAU_GRID[-1]))
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    if args.rho_e_cap:
        os.environ["SPECTROXIDE_RHO_E_CAP"] = repr(float(args.rho_e_cap))

    for v in (args.logtau_min, args.logtau_max):
        if abs(v*2 - round(v*2)) > 1e-9:
            sys.exit(f"--logtau-min/max must be multiples of 0.5 (got {v})")
    if args.logtau_max < args.logtau_min:
        sys.exit("--logtau-max < --logtau-min")
    lt_grid = np.arange(args.logtau_min, args.logtau_max + 1e-9, 0.5)

    dms = []
    for d in (args.dm if args.dm else cs.DM_GRID_CMB):
        i = int(np.argmin(np.abs(cs.DM_GRID_CMB - d)))
        if not np.isclose(cs.DM_GRID_CMB[i], d):
            sys.exit(f"dm={d:g} is not in the grid: {[float(x) for x in cs.DM_GRID_CMB]}")
        dms.append(float(cs.DM_GRID_CMB[i]))
    betas = [float(b) for b in args.betas]
    n_all = len(betas) * len(dms) * len(lt_grid)

    if args.list:
        print(f"TAG = {cs.TAG}  electron_mode={cs.ELECTRON_MODE}")
        print(f"betas: {betas}\ndm: {dms}\nlog10 tau: {lt_grid[0]:g}..{lt_grid[-1]:g} ({len(lt_grid)} pts)")
        print(f"total points: {n_all}")
        return

    shard_dir = Path(args.shard_dir).expanduser().resolve(); shard_dir.mkdir(parents=True, exist_ok=True)
    out = shard_dir / f"spec_{cs.TAG}_{args.name}.jsonl"
    done_pts = load_spectra(shard_dir)
    todo = [(b, dm, float(lt)) for b in betas for dm in dms for lt in lt_grid
            if pkey(b, dm, lt) not in done_pts]
    print(f"[{args.name}] TAG={cs.TAG}  betas={betas}", flush=True)
    print(f"[{args.name}] {n_all-len(todo)}/{n_all} already on disk, {len(todo)} to run, "
          f"{min(args.workers, max(len(todo),1))} workers -> {out}", flush=True)
    if not todo:
        return

    for b in sorted({b for b, _, _ in todo}):
        for dm in sorted({d for bb, d, _ in todo if bb == b}):
            cs.preflight_electron(b, dm)             # aborts with the real traceback if the cascade is broken
        print(f"[{args.name}] beta={b:.4g} pre-flight OK", flush=True)

    warnings.simplefilter("always")
    warnings.showwarning = cs._showwarning
    writer = ShardWriter(out)
    fail_log = shard_dir / f"failed_{args.name}.log"
    warn_log = shard_dir / f"warnings_{args.name}.log"

    def _term(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, _term)

    t0 = time.time(); ok = failed = nflag = 0
    flag_counts = Counter()
    ex = ThreadPoolExecutor(max_workers=min(args.workers, len(todo)))
    try:
        futs = {ex.submit(run_point_logged, *p): p for p in todo}
        for fut in as_completed(futs):
            b, dm, lt = futs[fut]
            key = pkey(b, dm, lt)
            try:
                point = fut.result()
            except Exception as e:
                failed += 1
                print(f"  FAILED {key}: {e!r}", flush=True)
                cs.safe_append(fail_log, f"FAILED {key}: {e!r}\n"
                               + "".join(f"  solver warning: {m[:300]}\n" for m in getattr(e, "captured_warnings", []))
                               + getattr(e, "tb", "") + "-" * 60)
            else:
                writer.add(key, point)
                ok += 1
                if point.get("flagged"):
                    nflag += 1; flag_counts.update(point["flags"])
                    msg = f"FLAGGED {key}: {point['flags']} clamps={point.get('rho_e_clamps')} " \
                          f"newton_fail={point.get('newton_fail')} warnings={point.get('warnings')}"
                    print("  " + msg, flush=True)
                    cs.safe_append(warn_log, msg)
            n = ok + failed; el = time.time() - t0
            print(f"  {n}/{len(todo)}  ok={ok} (flagged={nflag}) failed={failed}  last: beta={b:.4g} dm={dm:g} lt={lt:g}  "
                  f"elapsed {el/60:.1f} min, ETA {el/n*(len(todo)-n)/60:.1f} min", flush=True)
    except KeyboardInterrupt:
        print("interrupted - every finished point is on disk; rerun the same command to resume", flush=True)
        ex.shutdown(wait=False, cancel_futures=True)
        os._exit(130)
    except BaseException as e:
        print(f"FATAL in main loop: {e!r} -- stopping all workers", flush=True)
        traceback.print_exc()
        ex.shutdown(wait=False, cancel_futures=True)
        os._exit(1)
    ex.shutdown()
    print(f"[{args.name}] finished: {ok} ok ({nflag} flagged), {failed} failed (failed points are retried on rerun)", flush=True)
    if flag_counts:
        print(f"[{args.name}] flag summary: " + ", ".join(f"{k}={v}" for k, v in flag_counts.most_common()), flush=True)
    if failed:
        print(f"[{args.name}] failure details: {fail_log}", flush=True)


if __name__ == "__main__":
    main()
