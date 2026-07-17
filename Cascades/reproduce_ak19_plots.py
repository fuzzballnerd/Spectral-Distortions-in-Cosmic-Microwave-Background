"""
reproduce_ak19_plots.py  (v2)

Run this from the root of the extracted Cascades/ folder (the directory
that directly contains `cascade/` and `tests/`):

    cd Cascades
    pip install numpy scipy matplotlib --break-system-packages   # if needed
    python3 reproduce_ak19_plots.py

Produces PNG files in the current directory. Compare them directly against
the corresponding figures in acharya2019-2.pdf.

WHAT CHANGED FROM v1 (why v1 looked ugly):

1. Fig 2 (cooling ratios): v1 swept electron energy from 10 eV to 100 GeV
   on a log-log plot. t_cool/t_H falls to ~1e-10-1e-14 at the low-energy
   end and the whole point of interest (the ~2.5 keV crossover) got
   squeezed into a sliver a few pixels wide, with ~15 empty decades of
   axis around it. Fixed by keeping a narrower, still-generous energy
   window and explicitly clamping the y-axis around the crossover instead
   of letting matplotlib autoscale to the full data range.

2. Fig 13 (heat fraction) and Fig 6 (y vs energy): v1 used only 30-60
   points on the log grid. The heat fraction genuinely has sharp features
   (pair production turning on, successive IC generations) -- with too
   few samples these render as jagged sawtooth kinks with "-o" markers
   rather than smooth curves. Fixed by increasing sample density and
   removing the marker clutter. NOTE: this makes the script noticeably
   slower (each point is a recursive cascade evaluation) -- expect this
   to take a couple of minutes; turn N_HEAT_FRACTION_POINTS down if you
   want faster iteration.

3. Fig 14a (injection profile): v1 scanned z from z_X/30 to z_X*30 with a
   fixed multiplicative window. The exp(-cosmic_time(z)/tau_X) decay
   factor can fall by *100+ orders of magnitude* across that range once z
   drops much below z_X (verified: it goes from O(1) at z_X down to
   ~1e-61 by z=1000 for the tau_X=1e11 s benchmark) -- so most of that
   z-range was numerically zero, again leaving the plot as a tiny spike
   in an enormous empty log-log frame. Fixed with an adaptive window
   finder that scans first, then only plots the actual support of the
   profile (where it's above a floor relative to its own peak), padded
   a bit on each side.

4. Fig 6 comparison ("current vs patched"): this panel compared the
   pre-fix and post-fix y_distortion_estimate. Since the fix has now been
   applied directly in solve_burst.py (see the diff you already have),
   the two lines are now numerically identical -- the "comparison" plot
   was just two overlapping lines, uninformative at best. Replaced with a
   single curve (the fixed result) against the AK19 reference line, with
   an annotation explaining why it comes out nearly flat: with f_heat
   correctly included, the model now conserves total injected energy
   (f_heat+f_ic+f_subthreshold ~ 1) almost independent of species/energy
   in this range, so the y-vs-energy *trend* AK19 report must come from
   spectral-shape effects this scalar-only model still can't capture --
   that's a real, separate limitation, not something this fix addresses.

5. General: applied a consistent plot style (gridlines, larger figure/DPI,
   sane legends) across all five figures instead of bare matplotlib
   defaults.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt

from cascade import rates
from cascade import cosmology as cosmo
from cascade import solve_burst as sb

# Turn this down (e.g. to 40) for faster iteration; turn it up for smoother
# curves. Each point costs one recursive cascade evaluation -- on this
# machine, ~0.5 s/point, so 100 points x 2 species ~ 100 s per plot.
N_HEAT_FRACTION_POINTS = 1000
N_Y_VS_ENERGY_POINTS = 400


def _apply_style():
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 150,
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "legend.frameon": False,
        "lines.linewidth": 1.8,
    })


# ----------------------------------------------------------------------
# Helper: background conditions at a given redshift
# ----------------------------------------------------------------------
def _bg(z):
    n_e_cm3 = cosmo.n_e_total(z) / 1.0e6
    T_K = cosmo.T_gamma(z)
    U_CMB_eV_cm3 = cosmo.A_RAD * T_K ** 4 * 6.241509074e18 / 1.0e6
    E_e_bg_eV = 1.5 * rates.K_B_EV_PER_K * T_K
    return n_e_cm3, U_CMB_eV_cm3, E_e_bg_eV


# ----------------------------------------------------------------------
# 1. AK19 ANALOG -- Fig. 2: cooling time / Hubble time vs electron energy
# ----------------------------------------------------------------------
def plot_cooling_ratios(fname="fig2_cooling_ratios.png"):
    E = np.logspace(2, 7, 400)  # 100 eV - 10 MeV: brackets the ~2.5 keV
                                 # crossover with margin on both sides
                                 # without dragging in the deep-sub-eV or
                                 # ultra-relativistic tails.
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for z, style in [(1000.0, "-"), (20000.0, "--")]:
        n_e, U_CMB, E_bg = _bg(z)
        t_H = 1.0 / cosmo.H(z)
        t_coulomb = E / rates.coulomb_loss_rate(E, n_e, E_bg)
        t_ic = E / rates.ic_thomson_loss_rate(E, U_CMB)
        r_c, r_ic = t_coulomb / t_H, t_ic / t_H
        ax.plot(E, r_c, style, color="C0", label=f"Coulomb, z={z:.0f}")
        ax.plot(E, r_ic, style, color="C1", label=f"IC (Thomson), z={z:.0f}")

        # mark where the two channels cross for this z (just the closest
        # sampled point in log-space -- fine for an annotation, the actual
        # precise crossover is what tests/test_cp1_rates.py bisects for)
        i_cross = np.argmin(np.abs(np.log(r_c) - np.log(r_ic)))
        ax.plot(E[i_cross], r_c[i_cross], "o", color="k", ms=5, zorder=5)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 1e2)   # focus the visible window around the crossover
    ax.axhline(1.0, color="gray", lw=0.8, ls=":")
    ax.axvline(2500.0, color="gray", lw=0.8, ls=":")
    ax.text(2500.0, 5e1, " AK19: ~2.5 keV", fontsize=8, color="gray")
    ax.set_xlabel("electron kinetic energy [eV]")
    ax.set_ylabel(r"$t_{\rm cool}/t_H$")
    ax.set_title("AK19 Fig. 2 analog: cooling time vs Hubble time")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(fname)
    print(f"wrote {fname}")


# ----------------------------------------------------------------------
# 2. AK19 ANALOG -- Fig. 13a/b: fraction of injected energy going to heat
# ----------------------------------------------------------------------
def _energy_grid_with_threshold_refinement(lo_dex=3, hi_dex=12, n_base=None,
                                            threshold_eV=None, n_dense=60,
                                            half_width_dex=0.6):
    """Log grid over [lo_dex, hi_dex] (in eV, as powers of 10), plus a dense
    cluster of extra points around `threshold_eV`.

    burst_energy_budget switches its recursion branch at E0 = 2*m_e (the
    pair-production threshold for photons; also where the electron-side
    high_energy flag flips), and the physical cross-section itself turns on
    over a narrow window there. A uniform log grid with ~100 points across
    9 decades puts only a handful of samples anywhere near that window, so
    the kink gets rendered as a jagged 2-3 point zigzag instead of the
    smooth-but-steep turn-on it actually is. Concentrating extra points in
    a +/- half_width_dex band around the threshold (in log10 space) fixes
    this without paying for density we don't need out in the smooth tails.
    """
    if n_base is None:
        n_base = N_HEAT_FRACTION_POINTS
    base = np.logspace(lo_dex, hi_dex, n_base)
    if threshold_eV is None:
        return np.sort(base)
    dense = np.logspace(np.log10(threshold_eV) - half_width_dex,
                         np.log10(threshold_eV) + half_width_dex, n_dense)
    return np.unique(np.sort(np.concatenate([base, dense])))


def plot_heat_fraction_vs_energy(z=20000.0, fname="fig13_heat_fraction.png"):
    threshold_eV = 2.0 * sb.M_E_EV  # pair-production / high_energy branch point
    E_grid = _energy_grid_with_threshold_refinement(
        lo_dex=3, hi_dex=12, threshold_eV=threshold_eV)  # 1 keV - 1 TeV

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # --- Left panel: f_heat only, both species (as before, now smoother) ---
    ax = axes[0]
    budgets = {}  # cache so the right panel doesn't recompute
    for species, color in [("e", "C0"), ("gamma", "C1")]:
        rows = [sb.burst_energy_budget(species, E0, z,
                                        high_energy=(E0 > threshold_eV))
                for E0 in E_grid]
        budgets[species] = rows
        ax.plot(E_grid, [r["f_heat"] for r in rows], "-", color=color, lw=1.6,
                label=species)
    ax.axvline(threshold_eV, color="gray", lw=0.8, ls=":")
    ax.text(threshold_eV, 0.95, "  2 m_e", fontsize=8, color="gray")
    ax.set_xscale("log")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("injected particle energy [eV]")
    ax.set_ylabel("fraction of injected energy -> heat")
    ax.set_title(f"AK19 Fig. 13 analog: heat fraction at z={z:.0f}")
    ax.legend()

    # --- Right panel: stacked f_heat / f_ic / f_subthreshold for one
    # species, as a visual conservation check (should always sum to 1) ---
    ax2 = axes[1]
    species_check = "gamma"
    rows = budgets[species_check]
    f_heat = np.array([r["f_heat"] for r in rows])
    f_ic = np.array([r["f_ic"] for r in rows])
    f_sub = np.array([r["f_subthreshold"] for r in rows])
    ax2.stackplot(E_grid, f_heat, f_ic, f_sub,
                  labels=["f_heat", "f_ic", "f_subthreshold"],
                  colors=["C0", "C1", "C2"], alpha=0.85)
    total = f_heat + f_ic + f_sub
    if not np.allclose(total, 1.0, atol=1e-6):
        bad = np.argmax(np.abs(total - 1.0))
        print(f"  WARNING: energy budget for {species_check} does not sum to "
              f"1 at E0={E_grid[bad]:.2e} eV (sum={total[bad]:.6f}) -- "
              f"check burst_energy_budget for a leak/double-count")
    ax2.axvline(threshold_eV, color="gray", lw=0.8, ls=":")
    ax2.set_xscale("log")
    ax2.set_ylim(0, 1.02)
    ax2.set_xlabel("injected particle energy [eV]")
    ax2.set_ylabel("fraction of injected energy")
    ax2.set_title(f"Energy-budget conservation check ({species_check})")
    ax2.legend(fontsize=8, loc="lower left")

    fig.tight_layout()
    fig.savefig(fname)
    print(f"wrote {fname}")


# ----------------------------------------------------------------------
# 3. DIAGNOSTIC -- pair-production cross section near threshold
# ----------------------------------------------------------------------
def plot_pair_production_threshold_behavior(fname="diagnostic_pair_production.png"):
    E_MeV = np.linspace(0.8, 6.0, 2000)
    sigma = rates.pair_production_xsec(E_MeV * 1.0e6, Z=1.0)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(E_MeV, sigma, "-", color="C2")
    ax.axvline(1.022, color="r", ls="--", lw=1, label="2 m_e c^2 = 1.022 MeV (threshold)")
    ax.axvspan(1.022, 3.42, color="r", alpha=0.08,
               label="previously a dead zone (sigma=0) here")
    ax.set_xlabel("photon energy [MeV]")
    ax.set_ylabel(r"$\sigma_{\rm pair}$ [cm$^2$]")
    ax.set_title("DIAGNOSTIC: pair-production cross section near threshold\n"
                  "(smootherstep bridge replacing the old hard cutoff)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fname)
    print(f"wrote {fname}")


# ----------------------------------------------------------------------
# 4. AK19 ANALOG -- Fig. 14a: dark-matter decay energy-injection profile
# ----------------------------------------------------------------------
def _find_injection_z_window(z_X, tau_X_s, f_X, rel_floor=1e-4,
                              z_scan_lo_factor=200.0, z_scan_hi_factor=50.0,
                              n_scan=300, pad_dex=0.4):
    """
    exp(-cosmic_time(z)/tau_X_s) can fall by well over a hundred orders of
    magnitude across even a factor-few change in z once z drops below
    z_X, so a fixed multiplicative window like [z_X/30, z_X*30] can extend
    deep into a region that's numerically zero. Scan first, then return a
    padded window around wherever the actual profile is non-negligible.
    """
    z_scan = np.logspace(np.log10(z_X / z_scan_lo_factor),
                          np.log10(z_X * z_scan_hi_factor), n_scan)
    rate = np.array([sb.decay_injection_rate_eV_cm3_s(zp, f_X, tau_X_s) for zp in z_scan])
    dt_dz = 1.0 / ((1.0 + z_scan) * cosmo.H(z_scan))
    profile = rate * dt_dz / sb.rho_cmb_eV_cm3(z_scan)
    peak = profile.max()
    support = z_scan[profile > rel_floor * peak]
    z_lo = support.min() / (10.0 ** pad_dex)
    z_hi = support.max() * (10.0 ** pad_dex)
    return z_lo, z_hi


def plot_injection_profile(z_X=2.0e4, tau_X_s=1.0e11, target_y_all_heat=1.5e-5,
                            fname="fig14a_injection_profile.png"):
    f_X = sb.f_X_for_target_all_heat_y(z_X, target_y_all_heat)
    z_lo, z_hi = _find_injection_z_window(z_X, tau_X_s, f_X)
    z_grid = np.logspace(np.log10(z_lo), np.log10(z_hi), 2000)

    rate = np.array([sb.decay_injection_rate_eV_cm3_s(zp, f_X, tau_X_s) for zp in z_grid])
    dt_dz = 1.0 / ((1.0 + z_grid) * cosmo.H(z_grid))
    dE_dz_over_rho_cmb = rate * dt_dz / sb.rho_cmb_eV_cm3(z_grid)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(z_grid, dE_dz_over_rho_cmb, color="C3")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.axvline(z_X, color="k", ls="--", lw=0.8, label=f"z_X = {z_X:.0e}")
    ax.set_xlabel("redshift z")
    ax.set_ylabel(r"$(dE/dz)/\rho_{\rm CMB}(z)$")
    ax.set_title("AK19 Fig. 14a analog: DM-decay injection profile")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fname)
    print(f"wrote {fname}")


# ----------------------------------------------------------------------
# 5. AK19 ANALOG -- Fig. 6d/7a-style: y vs injection energy, 10 keV - 1 MeV
#    (now using the already-fixed sb.y_distortion_estimate directly --
#    see the module docstring for why the old "current vs patched"
#    comparison panel no longer makes sense)
# ----------------------------------------------------------------------
def plot_y_vs_energy(z=20000.0, injected_frac=1.0e-5,
                      fname="fig6_y_vs_energy.png"):
    E_grid = np.logspace(4, 6, N_Y_VS_ENERGY_POINTS)  # 10 keV - 1 MeV
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for species, color in [("e", "C0"), ("gamma", "C1")]:
        y_vals = [sb.y_distortion_estimate(species, E0, z, injected_frac)[0]
                  for E0 in E_grid]
        ax.plot(E_grid, y_vals, "-", color=color, label=species)
    ax.axhline(1.85e-6, color="k", lw=0.8, ls="--",
               label="AK19 quoted reference, y=1.85e-6")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("injected particle energy [eV]")
    ax.set_ylabel("y")
    ax.set_title(f"AK19 Fig. 6d/7a analog at z={z:.0f}")
    ax.legend(fontsize=8)
    ax.text(0.02, 0.02,
            "nearly flat here because f_heat+f_ic+f_sub~1 (energy is\n"
            "conserved) -- AK19's real energy-dependence is a spectral-\n"
            "shape effect this scalar-only model doesn't capture yet.",
            transform=ax.transAxes, fontsize=7, color="gray", va="bottom")
    fig.tight_layout()
    fig.savefig(fname)
    print(f"wrote {fname}")


if __name__ == "__main__":
    _apply_style()
    plot_cooling_ratios()
    plot_heat_fraction_vs_energy()
    plot_pair_production_threshold_behavior()
    plot_injection_profile()
    plot_y_vs_energy()
    print("\nDone. Open the PNGs and compare side-by-side against "
          "acharya2019-2.pdf Figs. 2, 13, 14a, and 6d/7a.")
