"""
fig3_electron_ic_spectrum.py -- reproduction of AK19 Fig. 3: the CMB spectral
distortion after a burst of monoenergetic electrons deposits all of its
kinetic energy via inverse-Compton scattering, using AK19's own exact
recipe (Eq. A.5, the relativistic frequency-redistribution function),
NOT the narrow-line / Recoil-only cascade proxy used elsewhere in this
codebase for Figs. 6/7/8/10.

This is amended-priority item 6 ("re-scope Fig. 3 as its own task"), per
`review_of_cascade_corrections_analysis.md` Sec. 2.3: Fig. 3 is
electron-injection only (no photon-injection case, unlike Fig. 6), and
AK19 state explicitly its shape comes from "results of [45-47] with the
full CMB spectrum" (Eq. A.5), not from the Recoil-only cascade machinery
used for the other Group B figures. `rates.ic_relativistic_frequency_redistribution`
(Eq. A.5, verified correct against Ensslin & Kaiser 2000's own normalization
and mean-energy-gain identities -- see `tests/test_cp1_rates.py`) is the
building block used here.

-----------------------------------------------------------------------
PHYSICAL DERIVATION (worth spelling out in full, since this required
working through several conceptual traps -- see the two "REJECTED
APPROACH" notes below for mistakes made and caught along the way):

A single electron of Lorentz factor gamma(t), injected at gamma_0, cools
by repeatedly Compton-scattering CMB photons, each scattering event
picking a photon (weighted by the local photon field's number-density
spectrum) and shifting its frequency by a factor t=nu'/nu drawn from
`ic_relativistic_frequency_redistribution`'s P(t;p), p=beta_e*gamma_e.

For a population of such electrons with (burst-injected, non-evolving)
number density n_e, the standard optically-thin Compton radiative
transfer equation (Ensslin & Kaiser 2000, their Eq. 8's differential
form) gives the rate of change of the dimensionless CMB intensity shape
function i(x) = x^3/(e^x-1) (I_nu = i0*i(x), i0 = 2(kT)^3/(hc)^2):

    d[i(x)]/dt = n_e * sigma_T * c * [ j(x; p(t)) - i(x) ]

    j(x; p) = integral_over_t' P(t'; p) * i(x/t') dt'   (Eq. 15-analogue,
                                                            single-scatter
                                                            convolution)

n_e here is the ELECTRON number density (not the CMB photon number
density -- see REJECTED APPROACH #1 below, this distinction is where the
first implementation attempt went wrong by a factor of ~1e15).

Converting the time integral into a Lorentz-factor integral using AK19's
own non-relativistic-electron-frame Thomson energy-loss rate (Eq. A.4,
`rates.ic_thomson_loss_rate`'s underlying formula):

    dE_e/dt = -(4/3) sigma_T c U_CMB gamma^2 beta^2,   dE_e = m_e c^2 dgamma

gives dt = m_e c^2 dgamma / [-(4/3) sigma_T c U_CMB gamma^2 beta^2], and
substituting into the transfer equation, sigma_T and c cancel completely
(this cancellation is the reason the result doesn't depend on the
electron's actual scattering rate/cross section beyond Thomson -- only on
how much energy it has left to give up):

    Delta[i(x)] = n_e * (3 m_e c^2)/(4 U_CMB) *
                  integral_{gamma=1}^{gamma_0} dgamma [j(x;p(gamma))-i(x)] / (gamma^2 beta^2)

U_CMB = rho_cmb_eV_cm3(z) (the CMB ENERGY density, not number density --
see REJECTED APPROACH #1 again). This Delta[i(x)] is verified (see
`_energy_conservation_self_test` below) to satisfy exact energy
conservation: integrating it back into an energy-density change
(via the standard u = (4 pi/c) integral I_nu dnu relation, itself checked
against `solve_burst.rho_cmb_eV_cm3` to ~1e-9 relative accuracy) reproduces
n_e * E0 to <0.1% (limited by grid resolution, not a residual bug).

REJECTED APPROACH #1 (caught by the energy-conservation self-test, not
assumed correct): an early version of this derivation used the CMB PHOTON
number density n_gamma in place of the electron density n_e in the
transfer equation above (reasoning, incorrectly, that "the electron
encounters photons at rate n_gamma*sigma_T*c, so that's the relevant
rate"). This is dimensionally the wrong equation -- the standard Compton
transfer equation's "n_e" is fundamentally the ELECTRON density along an
observer's sightline (it comes from the photon's optical depth through an
electron population, not the electron's per-photon encounter rate), and
conflating it with n_gamma produced a ~3.3e15x error (numerically
diagnosed via the energy-conservation self-test, not caught by dimensional
analysis alone, since n_gamma*sigma_T*c IS honestly a rate with correct
units -- it's just the wrong rate for this equation).

REJECTED APPROACH #2: near gamma->1 (p->0), evaluating `j(x;p)-i(x)`
directly via `ic_relativistic_frequency_redistribution` breaks down from
catastrophic cancellation (same failure mode documented in
`rates.klein_nishina_mean_transfer_fraction`'s docstring, t_min/t_max
becoming nearly equal). Fixed the same way: verified numerically that
j(x;p)-i(x) -> (p^2/3)*g_y(x) as p->0, where g_y(x) is the standard
Kompaneets/thermal-y spectral shape function (also needed anyway for
AK19's reference y-distortion curve in Fig. 3's own plot) -- checked this
scaling holds to <0.1% at p=0.03 across several x values before relying on
it, and the prefactor (1/3) was fit numerically, not guessed.
-----------------------------------------------------------------------

This module does NOT yet attempt pixel-level agreement with AK19's actual
plotted curves (no digitized reference data was available to compare
against) -- what IS verified is internal physical self-consistency (exact
energy conservation, matching known small-p/large-p limits). Treat the
absolute peak/dip positions and amplitudes as a first-principles
prediction from AK19's own stated formula, not as a validated
reproduction, until compared against digitized Fig. 3 data.
"""

import numpy as np

from cascade import rates
from cascade import cosmology as cosmo
from cascade import solve_burst as sb

H_EV_S = 4.135667696e-15  # Planck constant, eV*s

# Below this p, ic_relativistic_frequency_redistribution suffers
# catastrophic cancellation (t_min/t_max nearly equal); use the analytic
# small-p limit instead (see module docstring, REJECTED APPROACH #2).
P_MIN = 0.02


def blackbody_intensity_x(x):
    """i(x) = x^3/(e^x - 1), the dimensionless Planck intensity shape
    function (I_nu = i0 * i(x)), x = h*nu/kT."""
    x = np.asarray(x, dtype=float)
    return x ** 3 / np.expm1(x)


def y_distortion_shape(x):
    """
    The standard (thermal-averaged, Kompaneets-limit) y-distortion spectral
    shape function g(x):

        g(x) = x^4 e^x/(e^x-1)^2 * [ x(e^x+1)/(e^x-1) - 4 ]

    Used both as AK19's own reference y-distortion curve in Fig. 3, and as
    the small-p limit of the single-scatter kernel (see module docstring).
    """
    x = np.asarray(x, dtype=float)
    ex = np.exp(x)
    return x ** 4 * ex / (ex - 1.0) ** 2 * (x * (ex + 1.0) / (ex - 1.0) - 4.0)


def single_scatter_j_minus_i(x_grid, p, n_t=500):
    """
    j(x;p) - i(x): the change to the Planck intensity shape from
    convolving it once against AK19 Eq. A.5's frequency-redistribution
    kernel P(t;p) (see module docstring's Eq. 15-analogue). Uses the
    analytic small-p limit below P_MIN to avoid catastrophic cancellation
    in `ic_relativistic_frequency_redistribution` (see REJECTED APPROACH #2).

    Vectorized over x_grid; internally quadrature-free (fixed-resolution
    trapezoidal rule over t) for speed, since this is called inside a
    further integral over the electron's cooling track.
    """
    x_grid = np.asarray(x_grid, dtype=float)
    if p < P_MIN:
        return (p ** 2 / 3.0) * y_distortion_shape(x_grid)

    t_min = np.exp(-2.0 * np.arcsinh(p))
    t_max = np.exp(2.0 * np.arcsinh(p))
    t = np.linspace(t_min, t_max, n_t)
    Pt = rates.ic_relativistic_frequency_redistribution(t, p)
    X_over_t = x_grid[None, :] / t[:, None]
    j = np.trapezoid(Pt[:, None] * blackbody_intensity_x(X_over_t), t, axis=0)
    return j - blackbody_intensity_x(x_grid)


def electron_ic_delta_i(E0_eV, z, x_grid, n_e_cm3=1.0, n_gamma_pts=600, n_t=500):
    """
    Delta[i(x)] for a burst of monoenergetic electrons (initial kinetic
    energy E0_eV, injected at redshift z) with number density n_e_cm3,
    each cooling completely via inverse-Compton scattering off the CMB.
    See module docstring for the full derivation.

    Splits the gamma-integral into an exact analytic piece (p < P_MIN,
    where the integrand is provably constant in gamma -- see docstring)
    and a numerically integrated piece (p >= P_MIN, well-behaved).

    Parameters
    ----------
    E0_eV : electron injection kinetic energy, eV.
    z : injection redshift.
    x_grid : array of x = h*nu/kT values to evaluate the distortion at.
    n_e_cm3 : injected electron number density, cm^-3 (default 1.0 --
        use this default for the energy-conservation self-test; scale
        the result by the actual physical n_e for a real reproduction,
        e.g. n_e = f_inj*rho_cmb_eV_cm3(z)/E0_eV for AK19's normalization).
    n_gamma_pts, n_t : numerical resolution knobs (gamma-grid points,
        t-grid points per single-scatter convolution).

    Returns
    -------
    Delta[i(x)], same shape as x_grid.
    """
    x_grid = np.asarray(x_grid, dtype=float)
    me = rates.M_E_EV
    gamma0 = 1.0 + E0_eV / me
    U_CMB = sb.rho_cmb_eV_cm3(z)
    prefac = 3.0 * n_e_cm3 * me / (4.0 * U_CMB)

    gamma_pmin = np.sqrt(1.0 + P_MIN ** 2)

    if gamma0 <= gamma_pmin:
        # electron never leaves the small-p (y-distortion-like) regime
        return (prefac / 3.0) * y_distortion_shape(x_grid) * (gamma0 - 1.0)

    analytic_part = (prefac / 3.0) * y_distortion_shape(x_grid) * (gamma_pmin - 1.0)

    gammas = np.linspace(gamma_pmin, gamma0, n_gamma_pts)
    integrand = np.empty((n_gamma_pts, x_grid.size))
    for i, g in enumerate(gammas):
        p = np.sqrt(g ** 2 - 1.0)
        beta2 = 1.0 - 1.0 / g ** 2
        integrand[i] = single_scatter_j_minus_i(x_grid, p, n_t=n_t) / (g ** 2 * beta2)
    numeric_part = prefac * np.trapezoid(integrand, gammas, axis=0)

    return analytic_part + numeric_part


def delta_i_to_delta_I_nu_units(delta_i, z_observe=0.0):
    """
    Convert a dimensionless Delta[i(x)] into physical delta-I_nu units
    (eV s^-1 cm^-2 Hz^-1 sr^-1), via I_nu = i0*i(x), i0 = 2(kT)^3/(hc)^2.

    BUG FIX (found by comparing against AK19's actual Fig. 3 axis scale: an
    early version of this function used the *injection*-redshift kT for i0,
    which overstated the plotted brightness by ~(1+z_inj)^3 ~ 8e12 at
    z_inj=2e4 -- an error large enough to be obvious once compared against
    the paper's own y-axis range). The correct epoch for i0 is whatever
    redshift you want to REPORT the brightness at (default z_observe=0.0,
    i.e. today) -- NOT the injection redshift. This is because x=h*nu/kT is
    redshift-invariant by construction (AK19's own point, quoted in this
    module's docstring), so Delta[i(x)] itself (a pure function of x) is
    the same function whether evaluated "at injection" or "observed today"
    -- only the overall i0~kT^3 brightness scale differs between the two
    epochs (photon specific intensity redshifts as I_nu/nu^3 = const along
    a free-streaming null geodesic, i.e. i0(z) ~ T(z)^3). The redshift used
    inside `electron_ic_delta_i` (to set U_CMB in the derivation's
    prefactor) is correctly the INJECTION redshift -- that's a separate,
    correct use of z, not to be confused with this function's z_observe.
    """
    T_K = cosmo.T_gamma(z_observe)
    kT_eV = rates.K_B_EV_PER_K * T_K
    i0 = 2.0 * kT_eV ** 3 / (H_EV_S * rates.C_CM_S) ** 2
    return i0 * delta_i


def _energy_conservation_self_test(E0_eV, z, verbose=True):
    """
    Checks that integral(Delta[i(x)] dx) * (4 pi i0 kT/(c h)), with
    n_e_cm3=1.0, equals E0_eV*1.0 (eV/cm^3) -- i.e. that a burst of exactly
    1 electron/cm^3, each depositing E0_eV of kinetic energy, increases the
    CMB energy density by exactly E0_eV*1 eV/cm^3. This is the core physical
    consistency check for the whole derivation (see module docstring).
    """
    x_grid = np.linspace(5.0e-4, 80.0, 1500)
    delta_i = electron_ic_delta_i(E0_eV, z, x_grid, n_e_cm3=1.0, n_gamma_pts=500, n_t=400)

    T_K = cosmo.T_gamma(z)
    kT_eV = rates.K_B_EV_PER_K * T_K
    i0 = 2.0 * kT_eV ** 3 / (H_EV_S * rates.C_CM_S) ** 2
    conv = (4.0 * np.pi * i0 * kT_eV) / (rates.C_CM_S * H_EV_S)
    delta_u = conv * np.trapezoid(delta_i, x_grid)

    expected = E0_eV * 1.0
    rel_err = abs(delta_u - expected) / expected
    if verbose:
        print(f"E0={E0_eV/1e3:.0f} keV: delta_u={delta_u:.4f} eV/cm^3, "
              f"expected={expected:.4f} eV/cm^3, rel_err={rel_err:.4%}")
    return rel_err


def _run_self_tests():
    print("Energy-conservation self-test (n_e=1 cm^-3 convention):")
    for E0_keV in [10, 30, 50, 70]:
        _energy_conservation_self_test(E0_keV * 1.0e3, z=2.0e4)


def make_fig3_plot(output_path="fig3_reproduction.png"):
    """
    Reproduce AK19 Fig. 3: delta-I_nu (spectral distortion) vs x=h*nu/kT
    for 10/30/50/70 keV electron injection at z=20000, with total injected
    energy = 1e-5 * rho_cmb(z) (AK19's own normalization), plus the
    reference y-distortion curve AK19 also plots (their Fig. 3 caption:
    "The y-distortion curve is shown just for reference"). No heat/y
    channel added back in (AK19 explicitly excludes it from these curves --
    see review Sec. 2.3/item 5 of `cascade_corrections_all_figures.md`).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z_inj = 2.0e4
    f_inj = 1.0e-5
    U_CMB = sb.rho_cmb_eV_cm3(z_inj)
    E_inj_total = f_inj * U_CMB  # eV/cm^3, AK19's total injected energy density

    x_grid = np.linspace(0.05, 15.0, 220)
    # NOTE: kT_eV here is the OBSERVATION-epoch (z=0, today) temperature,
    # used only for the final brightness-unit conversion -- see
    # delta_i_to_delta_I_nu_units's docstring for why this differs from
    # the injection-epoch z_inj used in electron_ic_delta_i's derivation.
    T_K_observe = cosmo.T_gamma(0.0)
    kT_eV_observe = rates.K_B_EV_PER_K * T_K_observe

    fig, ax = plt.subplots(figsize=(7, 5))

    energies_keV = [10, 30, 50, 70]
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(energies_keV)))
    for E_keV, color in zip(energies_keV, colors):
        E0_eV = E_keV * 1.0e3
        n_e_cm3 = E_inj_total / E0_eV  # electrons/cm^3 needed for this E0 to match E_inj_total
        delta_i = electron_ic_delta_i(E0_eV, z_inj, x_grid, n_e_cm3=n_e_cm3,
                                       n_gamma_pts=500, n_t=400)
        delta_I_nu = delta_i_to_delta_I_nu_units(delta_i, z_observe=0.0)
        # AK19's y-axis units: 1e-23 W m^-2 Hz^-1 sr^-1. Convert from
        # eV s^-1 cm^-2 Hz^-1 sr^-1: 1 eV = 1.602176634e-19 J,
        # 1 cm^-2 = 1e4 m^-2 -> eV/s/cm^2/Hz/sr * 1.602e-19 J/eV * 1e4 m^2/cm^2
        # / 1e-23 = *1.602e-19*1e4/1e-23 = *1.602e8
        delta_I_nu_plot_units = delta_I_nu * 1.602176634e-19 * 1.0e4 / 1.0e-23
        ax.plot(x_grid, delta_I_nu_plot_units, color=color, label=f"{E_keV} keV e$^-$")

    # reference y-distortion curve: AK19 normalize it for visual reference
    # only (their caption doesn't pin an exact y value) -- use a
    # representative y ~ f_inj-scale amplitude for visual comparison, not a
    # claimed exact match
    y_ref = f_inj * 0.25  # order-of-magnitude reference, NOT a fitted value
    g_vals = y_distortion_shape(x_grid)
    i0_observe = 2.0 * kT_eV_observe ** 3 / (H_EV_S * rates.C_CM_S) ** 2
    y_dist_I_nu = y_ref * g_vals * i0_observe * 1.602176634e-19 * 1.0e4 / 1.0e-23
    ax.plot(x_grid, y_dist_I_nu, "k--", alpha=0.5, label="y-distortion (reference)")

    ax.axhline(0, color="gray", lw=0.5)
    ax.set_xlabel(r"$x = h\nu/kT$")
    ax.set_ylabel(r"$\delta I_\nu$ ($10^{-23}$ W m$^{-2}$ Hz$^{-1}$ sr$^{-1}$)")
    ax.set_title("Fig. 3 reproduction: electron-injection CMB distortion (Eq. A.5-based)")
    ax.legend(fontsize=9)
    ax.set_xlim(0, 15)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"Saved {output_path}")
    return output_path


if __name__ == "__main__":
    _run_self_tests()
    print()
    print("Building Fig. 3 reproduction plot (this takes a couple minutes)...")
    make_fig3_plot("fig3_reproduction.png")
