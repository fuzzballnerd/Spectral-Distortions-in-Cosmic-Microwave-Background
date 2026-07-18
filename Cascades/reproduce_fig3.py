"""
reproduce_fig3.py -- AK19 Fig. 3 (fix #6 from review_of_cascade_corrections_analysis.md)

WHY THIS IS ITS OWN SCRIPT, NOT PART OF Fig-6-STYLE GROUP B
--------------------------------------------------------------
The amended priority list's item 6 re-scopes Fig. 3 as its own task rather
than assuming it "falls out for free" once Figs. 6/7/8/10's fixes land
(cascade_corrections_all_figures.md originally filed it as "the simplest
member of Group B"). Two things make it genuinely different:

  1. Fig. 3 is electron-injection only -- no photon-injection curve, unlike
     Fig. 6. So the `make_photon_source` narrow-line proxy (interface.py)
     and its x_eff gamma-species degeneracy bug are simply not in play here.
  2. AK19 state explicitly that Fig. 3's curves come from a different, more
     exact formula than the rest of Group B: Eq. A.5 (Jones 1968's analytic
     IC frequency-redistribution function for a monoenergetic relativistic
     electron off the full CMB blackbody), not the ad hoc multi-point-
     per-decade characteristic-photon-energy proxy used elsewhere. Eq. A.6
     (ultra-relativistic Klein-Nishina) would matter above ~GeV, but Fig. 3's
     electrons (10-70 keV) never get there, so this script only exercises
     Eq. A.5 -- see cascade/ic_spectrum.py for both.

Also unlike Fig. 6 (which sums the heat and photon channels through a
spectroxide PDE relaxation), Fig. 3 is the *instantaneous*, single-burst,
photon-channel-ONLY result -- AK19 explicitly exclude the heat-channel
y-distortion ("The y-distortion from the energy dissipated as heat is not
added to any of the curves") and there is no subsequent PDE relaxation:
the caption describes it as the raw post-scattering shape ("This initial
spectrum will get modified due to the Compton scattering at subsequent
redshifts" -- i.e. Fig. 3 is *before* that modification). So this script
does not touch spectroxide/interface.py at all -- it computes delta_I_nu(x)
directly from cascade/ic_spectrum.py's electron_ic_spectrum, and overlays
a scaled y-distortion curve (Eq. C.1) for shape comparison only, exactly as
AK19's own caption does ("y-distortion curve is shown just for reference").

CONFIG -- matches AK19 Fig. 3's caption exactly: z=20000, 10/30/50/70 keV
electrons, each injecting 1e-5 * rho_CMB(z) of energy.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cascade import ic_spectrum as ics
from cascade import cosmology as cosmo
from cascade import solve_burst as sb
from cascade import rates

Z_INJ = 20000.0
INJECTED_FRAC = 1.0e-5
ELECTRON_ENERGIES_EV = [1.0e4, 3.0e4, 5.0e4, 7.0e4]  # 10/30/50/70 keV

X_MIN, X_MAX, N_X = 0.05, 20.0, 600
E_MIN_EV = 100.0  # cooling-track floor (well below all four injection energies)

PLOT_FILE = "fig3_reproduce.png"
DATA_FILE = "fig3_reproduce_data.npz"

# Same eV/cm^3 -> W/m^2/Hz/sr chain fig6_spectrum_via_spectroxide.py uses for
# its unit conversion (Jy/sr -> 1e-23 W/m^2/Hz/sr): here delta_I from
# ic_spectrum.fig3_style_delta_I already comes out in SI (W/m^2/Hz/sr), so
# just rescale to AK19's plotted units directly.
SI_TO_PAPER_UNITS = 1.0e23  # W/m^2/Hz/sr -> 1e-23 W/m^2/Hz/sr


def y_shape_eq_C1(x):
    """AK19 Eq. C.1's bracket (the y-distortion spectral shape, before the
    A_y=y amplitude): x*e^x/(e^x-1)^2 * [x*(e^x+1)/(e^x-1) - 4]."""
    x = np.asarray(x, dtype=float)
    ex = np.exp(np.clip(x, -700, 700))
    return (x * ex / (ex - 1.0) ** 2) * (x * (ex + 1.0) / (ex - 1.0) - 4.0)


def y_distortion_delta_I_paper_units(y, x_grid):
    """Eq. C.1 y-distortion delta_I_nu(x), in AK19's 1e-23 W/m^2/Hz/sr units.
    Uses TODAY's T_CMB0 (x is redshift-invariant -- see module docstring /
    ic_spectrum.fig3_style_delta_I's matching note) so this is directly
    comparable to the electron-injection curves."""
    EV_TO_J = 1.602176634e-19
    H_PLANCK = 6.62607015e-34
    C_LIGHT_SI = 2.99792458e8
    kT_eV = rates.K_B_EV_PER_K * cosmo.T_CMB0
    kT_J = kT_eV * EV_TO_J
    nu = x_grid * kT_J / H_PLANCK
    prefactor_SI = 2.0 * H_PLANCK * nu ** 3 / C_LIGHT_SI ** 2
    return y * prefactor_SI * y_shape_eq_C1(x_grid) * SI_TO_PAPER_UNITS


def run():
    x_grid = np.linspace(X_MIN, X_MAX, N_X)
    T_K = cosmo.T_gamma(Z_INJ)

    results = []
    for E0 in ELECTRON_ENERGIES_EV:
        delta_I_SI, budget = ics.fig3_style_delta_I(E0, Z_INJ, INJECTED_FRAC, x_grid, E_min_eV=E_MIN_EV)
        delta_I_paper = delta_I_SI * SI_TO_PAPER_UNITS
        results.append({"E0": E0, "delta_I": delta_I_paper, "budget": budget})
        print(f"  E0={E0/1e3:5.0f} keV  f_heat={budget['f_heat']:.3f}  f_ic={budget['f_ic']:.3f}  "
              f"peak={delta_I_paper.max():.3g}  dip={delta_I_paper.min():.3g}")

    # Reference y-distortion curve (shape comparison only, per AK19's own
    # caption), scaled via the existing (heat+IC-inclusive) y estimate for
    # the lowest injection energy -- same "one point's y sets the reference
    # curve normalization" approach fig6_spectrum_via_spectroxide.py uses.
    y_ref, _ = sb.y_distortion_estimate("e", ELECTRON_ENERGIES_EV[0], Z_INJ, INJECTED_FRAC)
    y_curve = y_distortion_delta_I_paper_units(y_ref, x_grid)

    np.savez(DATA_FILE, x=x_grid, z_inj=Z_INJ, injected_frac=INJECTED_FRAC,
             E0_list=np.array(ELECTRON_ENERGIES_EV),
             delta_I_list=np.array([r["delta_I"] for r in results]),
             f_heat_list=np.array([r["budget"]["f_heat"] for r in results]),
             f_ic_list=np.array([r["budget"]["f_ic"] for r in results]),
             y_ref=y_ref, y_curve=y_curve)
    print(f"Wrote {DATA_FILE}")

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "figure.dpi": 130, "savefig.dpi": 150, "font.size": 11,
        "axes.titlesize": 12, "axes.titleweight": "bold",
        "axes.labelsize": 11, "legend.fontsize": 9,
        "legend.frameon": False, "lines.linewidth": 1.8,
    })
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(results)))
    for r, c in zip(results, colors):
        ax.plot(x_grid, r["delta_I"], color=c, label=f"{r['E0']/1e3:.0f} keV e-")
    ax.plot(x_grid, y_curve, "k--", lw=1.2, label=f"y-distortion shape (y={y_ref:.2e})")
    ax.axhline(0, color="gray", lw=0.6)
    ax.set_xlabel("x")
    ax.set_ylabel(r"$\delta I_\nu$  [$10^{-23}$ W m$^{-2}$ Hz$^{-1}$ sr$^{-1}$]")
    ax.set_title(f"Fig 3 reproduction (electron-only, Eq. A.5) at z={Z_INJ:.0f}")
    ax.legend(fontsize=9, loc="best")
    fig.tight_layout()
    fig.savefig(PLOT_FILE)
    print(f"Wrote {PLOT_FILE}")


if __name__ == "__main__":
    run()
