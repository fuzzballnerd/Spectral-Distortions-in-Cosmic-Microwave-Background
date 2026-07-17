"""
fig6_spectrum_via_spectroxide.py — actually reproduce AK19 Fig. 6

WHY THIS FILE EXISTS
---------------------
fig6_via_spectroxide.py calls spx_solve() twice per (species, energy) point
(once for the heat channel, once for the photon channel) and then does

    mu = result_heat.mu + result_photon.mu
    y  = result_heat.y  + result_photon.y

and returns *only* those two scalars. Both SolverResult objects also carry
the full frequency grid `x` and the spectral distortion `delta_n` (and a
`.delta_I` convenience property) but the script never reads them out — it
throws the arrays away and keeps two numbers. run_sweep() then plots those
numbers as "y vs injection energy", one point per (species, energy) pair.

AK19 Fig. 6 is NOT y vs energy. It is delta_I_nu(x) vs x: a frequency
spectrum, one curve per (species, energy) combination, all drawn together
on the same axes (see Fig. 6d: 10 keV e-, 10 keV gamma, 100 keV e-, 100 keV
gamma, 1 MeV e-, 1 MeV gamma, six curves on one panel, x on the horizontal
axis, delta_I_nu in units of 1e-23 W/m^2/Hz/sr on the vertical axis, plus a
scaled y-type curve for comparison). So:

  - the old script's output (a scalar per point, plotted against energy) is
    a different quantity than what Fig. 6 shows, which is why the shape
    looked nothing like the paper — it isn't supposed to;
  - the "different points" you saw were N_POINTS=1 per species, i.e. one
    smoke-test energy each (deliberately, per the docstring, because each
    point costs ~25 min) rather than the several energies drawn per panel
    in the paper;
  - nothing needs to change physics-wise. Both SolverResult objects already
    contain the spectrum; this script just keeps `x` and `delta_n` instead
    of discarding them, sums the two channels' delta_n arrays (safe because
    both calls use spectroxide's default x grid — logspace(x_min, x_max,
    n_x) — so the grids are identical), converts to delta_I, and plots that
    against x. It also saves every point's raw arrays to disk, which the
    original script never did (its results only ever lived in a matplotlib
    figure).

USAGE
-----
    /home/batman/Documents/Work/.venv/bin/python fig6_spectrum_via_spectroxide.py

OUTPUT
------
  - fig6_spectrum_data.npz   -- every point's (x, delta_n_heat,
    delta_n_photon, delta_n_total, nu_ghz, delta_I_paper_units, mu, y),
    plus the run config, so the plot can be regenerated or re-analyzed
    without rerunning the (expensive) sweep.
  - fig6_spectrum_via_spectroxide.png -- delta_I_nu vs x, one curve per
    (species, energy) point, styled after AK19 Fig. 6d.

This keeps the parallelization and NaN-guarding from fig6_via_spectroxide.py
(those parts were fine) and only changes what is *extracted and saved* from
each spectroxide call, and what gets plotted.
"""

import sys
import os
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cascade import solve_burst as sb
from cascade import interface

try:
    from spectroxide import solve as spx_solve
    from spectroxide.greens import delta_n_to_delta_I, y_shape
    HAVE_SPECTROXIDE = True
except ImportError:
    raise ImportError(
        "spectroxide not found. Run this script with:\n"
        "  /home/batman/Documents/Work/.venv/bin/python fig6_spectrum_via_spectroxide.py\n"
        "The venv at /home/batman/Documents/Work/.venv has spectroxide installed."
    )

# ---------------------------------------------------------------------------
# Configuration — matches AK19 Fig. 6d (10 keV - 1 MeV electrons/photons)
# ---------------------------------------------------------------------------
Z_INJ = 20000.0
INJECTED_FRAC = 1.0e-5
TAU_X_S = 1.0e11
TARGET_Y_ALL_HEAT = 1.5e-5

# The six curves AK19 Fig. 6d actually draws on one panel.
POINTS = [
    ("e", 1.0e4),      # 10 keV e-
    ("gamma", 1.0e4),  # 10 keV gamma
    ("e", 1.0e5),      # 100 keV e-
    ("gamma", 1.0e5),  # 100 keV gamma
    ("e", 1.0e6),      # 1 MeV e-
    ("gamma", 1.0e6),  # 1 MeV gamma
]

Z_START_PDE = 1.5e5
Z_END_PDE = 1.0e3

# BUG FIX (x_max/x_cut mismatch, previously flagged in review): spectroxide's
# solve() defaults x_max to 30.0, but interface.py's make_photon_source
# hardcodes its own tabulated-range cutoff (x_cut) to 35.0, and neither
# spx_solve() call below used to pass x_max= explicitly -- so the photon
# source could place its narrow line as high as x=35 while spectroxide's own
# grid silently stopped at x=30, spilling energy off the actual solve grid.
# Passing x_max=X_MAX_SPX=35.0 explicitly to both calls (and to
# make_photon_source's x_max_factor=, so the two stay in lockstep) closes
# that gap.
X_MAX_SPX = 35.0

# Jy/sr -> 1e-23 W m^-2 Hz^-1 sr^-1 (AK19's plotted units): Jy/sr is
# 1e-26 W m^-2 Hz^-1 sr^-1, so multiply by 1e-26/1e-23 = 1e-3.
JY_TO_PAPER_UNITS = 1.0e-3

N_WORKERS = 6  # min(len(POINTS), 6) is plenty since there are only 6 tasks

DATA_FILE = "fig6_spectrum_data.npz"
PLOT_FILE = "fig6_spectrum_via_spectroxide.png"


# ---------------------------------------------------------------------------
# Safe wrappers (unchanged from fig6_via_spectroxide.py)
# ---------------------------------------------------------------------------
def _safe_dq_dz_wrapper(raw_dq_dz):
    def safe_dq_dz(z):
        try:
            val = raw_dq_dz(z)
            return float(val) if np.isfinite(val) else 0.0
        except Exception:
            return 0.0
    return safe_dq_dz


def _safe_photon_source_wrapper(raw_photon_source):
    def safe_photon_source(x, z):
        try:
            val = raw_photon_source(x, z)
            result = np.where(np.isfinite(val), val, 0.0)
            if np.ndim(result) == 0:
                return float(result)
            return result
        except Exception:
            if np.ndim(x) == 0:
                return 0.0
            return np.zeros_like(np.asarray(x, dtype=float))
    return safe_photon_source


# ---------------------------------------------------------------------------
# Core: run spectroxide for one (species, energy) point, keep the *spectrum*
# ---------------------------------------------------------------------------
def spectrum_via_spectroxide_pde(species, E0_eV):
    """Run the heat-channel and photon-channel PDE solves for one point and
    return the combined spectrum plus scalars.

    Two separate solve() calls are still required (spx_solve's PDE dispatch
    is an if/elif chain; passing dq_dz and photon_source together silently
    drops photon_source -- see fig6_via_spectroxide.py's docstring for the
    full explanation). What's new here: instead of discarding result.x and
    result.delta_n, we keep them and sum the two calls' delta_n arrays
    (Delta rho/rho ~ 1e-5, so the PDE evolution is linear in the source at
    this order -- this superposition is exact at that order, the same
    argument the old script already relied on when summing mu and y).
    """
    high_energy = E0_eV > 2.0 * sb.M_E_EV

    raw_dq_dz, f_X = interface.make_dq_dz(
        species, E0_eV, Z_INJ, TAU_X_S, TARGET_Y_ALL_HEAT,
        high_energy=high_energy
    )
    safe_dq = _safe_dq_dz_wrapper(raw_dq_dz)

    result_heat = spx_solve(
        method="pde",
        dq_dz=safe_dq,
        z_start=Z_START_PDE,
        z_end=Z_END_PDE,
        x_max=X_MAX_SPX,
        debug=True,
    )

    raw_ps = interface.make_photon_source(
        species, E0_eV, Z_INJ, TAU_X_S, f_X,
        high_energy=high_energy, x_max_factor=X_MAX_SPX,
    )
    safe_ps = _safe_photon_source_wrapper(raw_ps)

    result_photon = spx_solve(
        method="pde",
        photon_source=safe_ps,
        z_start=Z_START_PDE,
        z_end=Z_END_PDE,
        x_max=X_MAX_SPX,
        debug=True,
    )

    # Both calls use spectroxide's default x grid (no custom `x` passed in
    # either call above), so the grids are identical and delta_n arrays can
    # be summed directly.
    x = result_heat.x
    if not np.allclose(x, result_photon.x):
        raise RuntimeError(
            "heat and photon solves returned different x grids -- cannot "
            "sum delta_n directly; pass an explicit shared `x=` to both "
            "spx_solve() calls."
        )

    delta_n_heat = result_heat.delta_n
    delta_n_photon = result_photon.delta_n
    delta_n_total = delta_n_heat + delta_n_photon

    nu_ghz, delta_I_jy = delta_n_to_delta_I(x, delta_n_total)
    delta_I_paper = delta_I_jy * JY_TO_PAPER_UNITS

    mu = result_heat.mu + result_photon.mu
    y = result_heat.y + result_photon.y

    return {
        "x": x,
        "delta_n_heat": delta_n_heat,
        "delta_n_photon": delta_n_photon,
        "delta_n_total": delta_n_total,
        "nu_ghz": nu_ghz,
        "delta_I_paper_units": delta_I_paper,
        "mu": mu,
        "y": y,
    }


def _compute_point(task):
    """Top-level, picklable unit of work for ProcessPoolExecutor."""
    species, E0 = task
    t0 = time.time()
    try:
        spec = spectrum_via_spectroxide_pde(species, E0)
        err = None
    except Exception as e:
        spec = None
        err = str(e)
    dt = time.time() - t0
    return {"species": species, "E0": E0, "spectrum": spec, "err": err, "dt": dt}


def _ensure_rust_binary_built():
    """One-time, sequential warm-up so worker processes don't race on
    `cargo build --release` (see fig6_via_spectroxide.py for the full
    explanation of the threading.Lock-vs-multiprocessing hazard)."""
    print("Ensuring spectroxide Rust binary is built (one-time, sequential warm-up)...")
    t0 = time.time()
    dummy_dq = lambda z: 0.0
    try:
        spx_solve(method="pde", dq_dz=dummy_dq, z_start=Z_START_PDE,
                   z_end=Z_END_PDE, debug=True)
    except Exception as e:
        print(f"  (warm-up call raised, this is fine as long as the build "
              f"itself succeeded: {e})")
    print(f"  done in {time.time() - t0:.1f}s\n")


# ---------------------------------------------------------------------------
# Sweep, save, plot
# ---------------------------------------------------------------------------
def run_sweep(n_workers=N_WORKERS):
    results = []
    total = len(POINTS)
    print(f"\n{'='*60}\nRunning {total} (species, energy) points across "
          f"{n_workers} worker processes\n{'='*60}")

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        future_to_task = {executor.submit(_compute_point, t): t for t in POINTS}
        done = 0
        for future in as_completed(future_to_task):
            species, E0 = future_to_task[future]
            r = future.result()
            results.append(r)
            done += 1
            note = f"  FAILED: {r['err']}" if r["err"] else ""
            y_str = f"{r['spectrum']['y']:.3e}" if r["spectrum"] else "nan"
            print(f"  [{done}/{total}] species={species:5s} E={E0:.1e} eV: "
                  f"y={y_str}  ({r['dt']:.1f}s){note}")
    return results


def save_data(results, fname=DATA_FILE):
    """Save every point's raw spectrum + scalars to a single .npz file, so
    the sweep never has to be rerun just to replot or reanalyze."""
    save_dict = {
        "z_inj": Z_INJ,
        "injected_frac": INJECTED_FRAC,
        "tau_x_s": TAU_X_S,
        "species_list": np.array([r["species"] for r in results]),
        "E0_list": np.array([r["E0"] for r in results]),
        "err_list": np.array([r["err"] or "" for r in results]),
    }
    for i, r in enumerate(results):
        prefix = f"pt{i:02d}_{r['species']}_{r['E0']:.0e}"
        if r["spectrum"] is not None:
            for key, val in r["spectrum"].items():
                save_dict[f"{prefix}_{key}"] = val
    np.savez(fname, **save_dict)
    print(f"\nWrote {fname}")


def plot_results(results, fname=PLOT_FILE):
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "figure.dpi": 130, "savefig.dpi": 150, "font.size": 11,
        "axes.titlesize": 12, "axes.titleweight": "bold",
        "axes.labelsize": 11, "legend.fontsize": 9,
        "legend.frameon": False, "lines.linewidth": 1.8,
    })

    fig, ax = plt.subplots(figsize=(8, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    y_ref = None
    for (r, c) in zip(results, colors):
        spec = r["spectrum"]
        if spec is None:
            continue
        label = f"{r['E0']:.0f} eV {'e-' if r['species'] == 'e' else 'gamma'}"
        ax.plot(spec["x"], spec["delta_I_paper_units"], color=c, label=label)
        if y_ref is None:
            y_ref = spec["y"]  # use one point's y to draw the comparison shape

    # Scaled y-type distortion for shape comparison, as AK19 does explicitly
    # ("we have only shown a scaled y-distortion spectrum for comparison").
    if y_ref is not None:
        x_ref = results[0]["spectrum"]["x"]
        nu_ghz_ref, _ = delta_n_to_delta_I(x_ref, np.ones_like(x_ref))
        # y-type Delta n(x) = y * y_shape(x); reuse delta_n_to_delta_I for
        # correct intensity normalization instead of hand-deriving it.
        delta_n_y = y_ref * y_shape(x_ref)
        _, delta_I_y_jy = delta_n_to_delta_I(x_ref, delta_n_y)
        ax.plot(x_ref, delta_I_y_jy * JY_TO_PAPER_UNITS, "k--", lw=1.2,
                label=f"y-distortion shape (y={y_ref:.2e})")

    ax.set_xscale("linear")
    ax.set_xlabel("x")
    ax.set_ylabel(r"$\delta I_\nu$  [$10^{-23}$ W m$^{-2}$ Hz$^{-1}$ sr$^{-1}$]")
    ax.set_title(f"Fig 6-style spectrum (via spectroxide) at z={Z_INJ:.0f}")
    ax.legend(fontsize=8, loc="best")
    ax.axhline(0, color="gray", lw=0.6)

    fig.tight_layout()
    fig.savefig(fname)
    print(f"Wrote {fname}")


if __name__ == "__main__":
    mp.freeze_support()
    t_start = time.time()

    _ensure_rust_binary_built()
    results = run_sweep()
    save_data(results)
    plot_results(results)

    dt_total = time.time() - t_start
    print(f"\nTotal runtime: {dt_total:.0f}s ({dt_total/60:.1f} min)")
