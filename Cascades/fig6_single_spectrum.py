"""
fig6_single_spectrum.py -- reproduce one AK19 Fig. 6 curve (delta_I_nu vs x)
by calling spectroxide directly, for a single (species, energy) case.

This is a stripped-down version of fig6_spectrum_via_spectroxide.py: same
physics (interface.make_dq_dz / make_photon_source feeding two spx_solve()
PDE calls -- heat channel and photon channel -- summed together), but no
sweep over multiple points and no multiprocessing. Edit SPECIES/E0_EV below
and run.

Requires spectroxide's compiled Rust binary (`cargo build --release` in the
spectroxide crate root) -- see spx_solve()'s own error message if it's not
built.
"""

import os
import sys

# Make sure `cascade` is importable regardless of the caller's cwd (matches
# fig6_spectrum_via_spectroxide.py; this file previously omitted it and only
# worked when run from inside Cascades/Cascades/).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cascade import solve_burst as sb
from cascade import interface

from spectroxide import solve as spx_solve
from spectroxide.greens import delta_n_to_delta_I, y_shape

# ---------------------------------------------------------------------------
# Configuration -- edit these for the case you want
# ---------------------------------------------------------------------------
SPECIES = "e"        # "e" or "gamma"
E0_EV = 1.0e5        # injected kinetic (electron) or total (photon) energy, eV

Z_INJ = 20000.0
TAU_X_S = 1.0e11
TARGET_Y_ALL_HEAT = 1.5e-5

Z_START_PDE = 1.5e5
Z_END_PDE = 1.0e3

# spectroxide's solve() defaults x_max to 30.0, but interface.py's
# make_photon_source's own tabulated-range cutoff is 35.0 -- pass x_max
# explicitly to both spx_solve() calls (and to make_photon_source's
# x_max_factor=) so they stay in lockstep; see review notes for why this
# matters (a mismatch here silently truncates part of the injected line).
X_MAX_SPX = 35.0

# Jy/sr -> 1e-23 W m^-2 Hz^-1 sr^-1 (AK19's plotted units): Jy/sr is
# 1e-26 W m^-2 Hz^-1 sr^-1, so multiply by 1e-26/1e-23 = 1e-3.
JY_TO_PAPER_UNITS = 1.0e-3

# BUG FIX: PLOT_FILE/DATA_FILE used to be hardcoded constants
# ("fig6_single_spectrum.png"/"_data.npz"), not derived from SPECIES/E0_EV.
# Fig. 6d needs 6 points (10 keV/100 keV/1 MeV x e-/gamma) plotted together;
# the intended workflow is to edit SPECIES/E0_EV and rerun this script once
# per point. With hardcoded names, each rerun silently overwrote the
# previous point's saved .npz/.png with no warning -- after 6 runs only the
# last point's data would exist on disk. Filenames are now derived from
# (species, E0_eV) and written into POINTS_DIR, so every point gets its own
# file and all 6 accumulate side by side, ready for fig6_combine_points.py
# to stitch into the final overlay.
POINTS_DIR = "fig6_points"


def _point_tag(species, E0_eV):
    """Filesystem-safe identifier for one (species, E0_eV) point."""
    species_tag = "e" if species == "e" else "gamma"
    return f"{species_tag}_{E0_eV:.3e}eV".replace("+", "")


def point_filenames(species, E0_eV, points_dir=POINTS_DIR):
    os.makedirs(points_dir, exist_ok=True)
    tag = _point_tag(species, E0_eV)
    plot_file = os.path.join(points_dir, f"fig6_point_{tag}.png")
    data_file = os.path.join(points_dir, f"fig6_point_{tag}_data.npz")
    return plot_file, data_file


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


def spectrum_via_spectroxide_pde(species, E0_eV):
    """
    Run the heat-channel and photon-channel PDE solves for one
    (species, E0_eV) point and return the combined spectrum plus mu/y.

    Two separate spx_solve() calls are required (its PDE dispatch is an
    if/elif chain: passing dq_dz and photon_source together silently drops
    photon_source). Both calls use spectroxide's default x grid, so their
    delta_n arrays can be summed directly (Delta rho/rho ~ 1e-5, so the PDE
    is linear in the source at this order -- exact superposition).
    """
    high_energy = E0_eV > 2.0 * sb.M_E_EV

    raw_dq_dz, f_X = interface.make_dq_dz(
        species, E0_eV, Z_INJ, TAU_X_S, TARGET_Y_ALL_HEAT,
        high_energy=high_energy
    )
    safe_dq = _safe_dq_dz_wrapper(raw_dq_dz)

    print(f"Running heat-channel PDE solve (species={species}, E0={E0_eV:.2e} eV)...")
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

    print("Running photon-channel PDE solve...")
    result_photon = spx_solve(
        method="pde",
        photon_source=safe_ps,
        z_start=Z_START_PDE,
        z_end=Z_END_PDE,
        x_max=X_MAX_SPX,
        debug=True,
    )

    x = result_heat.x
    if not np.allclose(x, result_photon.x):
        raise RuntimeError(
            "heat and photon solves returned different x grids -- cannot "
            "sum delta_n directly; pass an explicit shared x= to both "
            "spx_solve() calls."
        )

    delta_n_total = result_heat.delta_n + result_photon.delta_n
    nu_ghz, delta_I_jy = delta_n_to_delta_I(x, delta_n_total)
    delta_I_paper = delta_I_jy * JY_TO_PAPER_UNITS

    mu = result_heat.mu + result_photon.mu
    y = result_heat.y + result_photon.y

    return {
        "x": x,
        "nu_ghz": nu_ghz,
        "delta_n_heat": result_heat.delta_n,
        "delta_n_photon": result_photon.delta_n,
        "delta_n_total": delta_n_total,
        "delta_I_paper_units": delta_I_paper,
        "mu": mu,
        "y": y,
    }


def plot_spectrum(spec, species, E0_eV, fname=None):
    if fname is None:
        fname, _ = point_filenames(species, E0_eV)
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "figure.dpi": 130, "savefig.dpi": 150, "font.size": 11,
        "axes.titlesize": 12, "axes.titleweight": "bold",
        "axes.labelsize": 11, "legend.fontsize": 9,
        "legend.frameon": False, "lines.linewidth": 1.8,
    })

    fig, ax = plt.subplots(figsize=(7, 5.5))

    label = f"{E0_eV:.0f} eV {'e-' if species == 'e' else 'gamma'}"
    ax.plot(spec["x"], spec["delta_I_paper_units"], color="C0", label=label)

    # Scaled y-type distortion shape for comparison, as AK19 does ("we have
    # only shown a scaled y-distortion spectrum for comparison").
    delta_n_y = spec["y"] * y_shape(spec["x"])
    _, delta_I_y_jy = delta_n_to_delta_I(spec["x"], delta_n_y)
    ax.plot(spec["x"], delta_I_y_jy * JY_TO_PAPER_UNITS, "k--", lw=1.2,
            label=f"y-distortion shape (y={spec['y']:.2e})")

    ax.set_xlabel("x")
    ax.set_ylabel(r"$\delta I_\nu$  [$10^{-23}$ W m$^{-2}$ Hz$^{-1}$ sr$^{-1}$]")
    ax.set_title(f"Fig. 6-style spectrum (via spectroxide), z={Z_INJ:.0f}")
    ax.legend(fontsize=9, loc="best")
    ax.axhline(0, color="gray", lw=0.6)

    fig.tight_layout()
    fig.savefig(fname)
    print(f"Wrote {fname}")


if __name__ == "__main__":
    plot_file, data_file = point_filenames(SPECIES, E0_EV)

    if os.path.exists(data_file):
        print(f"[note] {data_file} already exists for this exact "
              f"(species={SPECIES}, E0_eV={E0_EV:.3e}) point and will be "
              f"overwritten by this run. Different points get different "
              f"filenames automatically -- this only fires on an exact "
              f"repeat.")

    spec = spectrum_via_spectroxide_pde(SPECIES, E0_EV)

    print(f"\nmu = {spec['mu']:.4e}")
    print(f"y  = {spec['y']:.4e}")

    np.savez(data_file, species=SPECIES, E0_eV=E0_EV, z_inj=Z_INJ, **spec)
    print(f"Wrote {data_file}")

    plot_spectrum(spec, SPECIES, E0_EV, fname=plot_file)
    print(f"\nRun this once per (species, E0_eV) point (edit the constants "
          f"above and rerun) -- each point is saved separately under "
          f"{POINTS_DIR}/. Once all 6 Fig. 6d points are collected, run "
          f"fig6_combine_points.py to stitch them into the final overlay.")
