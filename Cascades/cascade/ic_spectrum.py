"""
ic_spectrum.py -- AK19 Eq. A.5 (relativistic) / Eq. A.6 (ultra-relativistic
Klein-Nishina) analytic IC frequency-redistribution functions, and the
electron-only, single-generation photon spectrum built from them.

WHY THIS MODULE EXISTS
-----------------------
Every other spectral-shape figure in this codebase (Fig. 6 and friends, see
fig6_spectrum_via_spectroxide.py) goes through the cascade's per-decade
*scalar* energy-fraction bookkeeping (electron_cascade/photon_cascade_full
in solve_burst.py), which by construction only ever reports a single
characteristic photon energy per decade -- never a resolved spectral shape.
review_of_cascade_corrections_analysis.md Sec 2.3 (amended-plan item 6)
flags that Fig. 3 should NOT be treated as "falls out for free" once that
machinery is fixed: AK19 state explicitly (Sec 2, right after introducing
Fig. 3) that the electron-injection spectral shape is computed from a
different, more exact formula -- Eq. A.5 for relativistic electrons (the
Jones 1968 analytic frequency-redistribution function for one IC scattering
of a monoenergetic electron off the *full* Planck blackbody), and Eq. A.6
(double-differential Klein-Nishina) once the electron rest-frame CMB photon
energy becomes comparable to m_e c^2 (electron energy >~ GeV, AK19 Sec 2/A.4
discussion -- not reached by Fig. 3's 10-70 keV electrons, but implemented
here too since the amended plan groups A.5/A.6 together as "the paper-given
recipe" for the electron-side spectrum generally).

This module implements those two formulas directly (not the phenomenological
x/(1+x) proxy solve_burst.py uses for Compton recoil) and builds the
photon-channel-only spectrum Fig. 3 needs.

EQ. A.5 -- WHAT "t" AND "nu" MEAN HERE, AND WHY IT'S A CONVOLUTION:

This is the Jones (1968, ApJ 151, 545) closed-form result for one Compton
scattering of a monoenergetic electron (momentum p = beta*gamma) off an
*isotropic blackbody* radiation field. "t = nu'/nu" is meant literally per
AK19's own text: nu is the frequency of the SPECIFIC unscattered (seed)
photon actually hit, nu' the frequency it leaves at -- confirmed
numerically here, not just by the literal text: mean_t_relativistic(p) -> 1
as p -> 0 (negligible electron momentum gives negligible boost for *any*
seed photon, which only holds if t is measured relative to whatever photon
was actually hit; if t were instead referenced to a fixed scale like kT/h,
p -> 0 would have to collapse P(t;p) onto whatever x that fixed scale is,
not onto t=1 universally). The remarkable simplification Jones (1968) found
is that, after integrating over an isotropic blackbody's angular and
frequency distribution, the RATIO t alone (not nu separately) determines
the outcome -- so P(t;p) is a genuine ratio-only kernel, and getting the
actual output spectrum on a fixed dimensionless-x grid requires convolving
it against the seed (Planck) spectrum's own shape:

    dN_out/dx'(x') = integral dx0 n0(x0) P(x'/x0; p) / x0

(see _convolved_output_shape below), where n0(x0) is the normalized Planck
photon-NUMBER spectral shape. P(t;p) itself (photon-number-normalized:
integral over its domain is 1) is the distribution of the OUTPUT/INPUT
frequency RATIO for one scattering, valid for |ln t| < 2 arcsinh(p).

Units/convention note: unlike rates.py's eV/cm/s convention, everything here
is expressed directly in the dimensionless x = h*nu/(kT) variable, matching
spectroxide's own convention (interface.py's G2_0/x^2 usage) and AK19's own
plots.
"""

import numpy as np
from scipy import integrate

from . import rates
from . import cosmology as cosmo
from . import solve_burst as sb

M_E_EV = rates.M_E_EV
E_CMB_MEAN_FACTOR = sb.E_CMB_MEAN_FACTOR  # <x> = 2.70 for a Planck NUMBER spectrum (3*zeta(4)/zeta(3))
G2_0 = 2.0 * 1.2020569031595943           # 2*zeta(3), Planck-spectrum photon-number normalization

# Standard CMB photon number density today (Fixsen 2009 / standard value),
# used to scale n_gamma(z) = N_GAMMA0_CM3 * (1+z)^3.
N_GAMMA0_CM3 = 410.7


def n_gamma_cm3(z):
    """Physical CMB photon number density at redshift z, cm^-3."""
    return N_GAMMA0_CM3 * (1.0 + z) ** 3


# ----------------------------------------------------------------------
# Eq. A.5 -- Jones (1968) relativistic single-scattering redistribution
# ----------------------------------------------------------------------
def ic_relativistic_frequency_dist(t, p):
    """
    AK19 Eq. A.5. t is the (dimensionless, x-convention -- see module
    docstring) frequency of the scattered photon; p = beta_e*gamma_e is the
    electron's momentum in units of m_e*c. Returns 0 outside the kinematic
    domain |ln t| < 2*arcsinh(p) (no single scattering can reach those t).

    TRANSCRIPTION NOTE: the second bracket's first term has denominator
    sqrt(1+p^2), not (1+p^2) -- this PDF's plain-text extraction drops the
    square root (same fraction-layout garbling rates.py's B.3 docstring
    already flags), verified against the actual arXiv:1808.02897 source
    text directly. Using (1+p^2) here made P(t;p) negative everywhere (all
    p, all t), which is unphysical for a probability density -- caught by
    that sign check, not by a units/dimensional-analysis check.
    """
    t = np.asarray(t, dtype=float)
    p = np.asarray(p, dtype=float)
    t_safe = np.where(t > 0, t, 1.0)

    with np.errstate(all="ignore"):
        lnt = np.log(t_safe)
        bound = 2.0 * np.arcsinh(p)
        domain = (t > 0) & (np.abs(lnt) < bound) & (p > 0)

        term1 = (-3.0 * np.abs(1.0 - t_safe) / (32.0 * p ** 6 * t_safe)) * (
            1.0 + (10.0 + 8.0 * p ** 2 + 4.0 * p ** 4) * t_safe + t_safe ** 2
        )
        term2 = (3.0 * (1.0 + t_safe) / (8.0 * p ** 5)) * (
            (3.0 + 3.0 * p ** 2 + p ** 4) / np.sqrt(1.0 + p ** 2)
            - (3.0 + 2.0 * p ** 2) / (2.0 * p) * (2.0 * np.arcsinh(p) - np.abs(lnt))
        )
        val = term1 + term2

    val = np.where(np.isfinite(val), val, 0.0)
    return np.where(domain, np.maximum(val, 0.0), 0.0)


def _t_domain(p):
    """(t_min, t_max) kinematic bounds for Eq. A.5 at momentum p."""
    bound = 2.0 * np.arcsinh(p)
    return np.exp(-bound), np.exp(bound)


def mean_t_relativistic(p, n=30000):
    """
    <t> = integral t*P(t;p) dt / integral P(t;p) dt over the kinematic
    domain, i.e. the mean dimensionless output-photon frequency for one
    Eq. A.5 scattering at momentum p. Used to convert an energy budget
    (eV transferred to the photon channel) into a photon *count*, the same
    way E_CMB_MEAN_FACTOR=2.70 already converts energy<->number for the
    unscattered Planck spectrum elsewhere in this codebase.
    """
    t_min, t_max = _t_domain(p)
    t_grid = np.geomspace(max(t_min, 1e-8), t_max, n)
    P = ic_relativistic_frequency_dist(t_grid, p)
    norm = np.trapezoid(P, t_grid)
    if norm <= 0 or not np.isfinite(norm):
        return E_CMB_MEAN_FACTOR  # degenerate p -> no net boost; caller skips this decade anyway
    return np.trapezoid(t_grid * P, t_grid) / norm


def _planck_number_shape(x):
    """Normalized (integral = 1) Planck photon-NUMBER spectral shape,
    x^2/(e^x-1) / (2*zeta(3)) -- both the seed-photon "pick" probability
    (Thomson cross section is frequency-independent, so any given seed
    photon's probability of being scattered is just proportional to how
    many photons of that frequency exist) and, per module docstring, what
    gets subtracted at each decade (the Fig. 3 "dip")."""
    x = np.asarray(x, dtype=float)
    shape = np.where(x > 0, x ** 2 / np.expm1(np.clip(x, 1e-8, 700.0)), 0.0)
    norm = np.trapezoid(shape, x)
    return shape / norm if norm > 0 else shape


def _convolved_output_shape(x_out, p, x0_grid):
    """
    Normalized (integral = 1) output photon-number shape dNout/dx' at
    momentum p, found by convolving the Planck seed shape n0(x0) with
    Eq. A.5's P(t;p) over the actual (self-similar) seed frequency each
    photon started at:

        dNout/dx'(x') = integral dx0 n0(x0) P(x'/x0; p) / x0

    (x0_grid is the seed-frequency integration grid; x_out is the
    output-frequency grid the result is evaluated on -- independent grids,
    typically x0_grid narrower/denser around the Planck peak.)
    """
    n0 = _planck_number_shape(x0_grid)
    # matrix: rows = x_out, cols = x0_grid
    t_mat = x_out[:, None] / x0_grid[None, :]
    P_mat = ic_relativistic_frequency_dist(t_mat, p)
    integrand = P_mat * n0[None, :] / x0_grid[None, :]
    out = np.trapezoid(integrand, x0_grid, axis=1)
    norm = np.trapezoid(out, x_out)
    return out / norm if norm > 0 else out


# ----------------------------------------------------------------------
# Eq. A.6 -- ultra-relativistic (ambient-photon-energy-comparable-to-m_e c^2)
# double-differential Klein-Nishina spectrum. Not reached by Fig. 3's 10-70
# keV electrons (gamma_e ~ 1.02-1.14) -- included per the amended plan's
# "A.5/A.6-based" framing for completeness / future high-energy figures.
# ----------------------------------------------------------------------
def klein_nishina_double_diff(E1_eV, eps_eV, gamma_e):
    """
    AK19 Eq. A.6: d^2N/(dt dE1) for a photon of unscattered energy eps_eV
    Compton up-scattered to E1_eV by an electron of Lorentz factor gamma_e,
    per unit incident-photon number density n(eps) (i.e. this returns the
    kernel that still needs to be integrated against n(eps)deps -- callers
    wanting a rate must supply that themselves; not needed for Fig. 3).

    Gamma = 4*eps*gamma_e/m_e, q = E1/(Gamma*(gamma_e*m_e*c^2 - E1)).
    Kinematically valid for 1/(4 gamma_e^2) <= q <= 1.
    """
    eps = np.asarray(eps_eV, dtype=float)
    E1 = np.asarray(E1_eV, dtype=float)
    ge = np.asarray(gamma_e, dtype=float)
    me = M_E_EV

    Gamma = 4.0 * eps * ge / me
    denom = Gamma * (ge * me - E1)
    with np.errstate(all="ignore"):
        q = np.where(denom > 0, E1 / denom, 0.0)
        domain = (q >= 1.0 / (4.0 * ge ** 2)) & (q <= 1.0) & (denom > 0)
        bracket = 2.0 * q * np.log(np.where(q > 0, q, 1.0)) + (1.0 + 2.0 * q) * (1.0 - q) \
            + 0.5 * (Gamma * q) ** 2 / (1.0 + Gamma * q) * (1.0 - q)
        val = (2.0 * np.pi * rates.R0_CM ** 2 * me ** 3) / (ge * eps ** 2) * bracket
    return np.where(domain & np.isfinite(val), np.maximum(val, 0.0), 0.0)


# ----------------------------------------------------------------------
# Electron-only, single-burst, photon-channel-only spectrum (Fig. 3)
# ----------------------------------------------------------------------
def electron_ic_spectrum(E0_eV, z, x_grid, E_min_eV=100.0, min_decades_per_e_fold=4):
    """
    Build the AK19 Fig. 3-style spectrum for a single injected electron of
    kinetic energy E0_eV, fully depositing its energy at redshift z: the
    *photon-channel-only* excess d(dE)/dx over the electron's whole cooling
    track (E0 -> E_min_eV), using Eq. A.5 for the up-scattered shape at each
    decade (NOT electron_cascade's characteristic-energy-per-decade proxy,
    and NOT the heat/y-channel -- AK19 explicitly excludes heat's
    y-distortion from Fig. 3: "The y-distortion from the energy dissipated
    as heat is not added to any of the curves").

    Reuses the same Coulomb-vs-IC decade branching as electron_cascade
    (solve_burst.py) to get how much energy goes to the IC channel in each
    decade -- that part of the physics (A.1 vs A.4 competition) is unrelated
    to fix #6 and is already validated by test_cp2/cp3. What's new here is
    *where in x* that IC-channel energy lands, via Eq. A.5's actual shape
    instead of a single characteristic photon energy.

    Per decade, with local electron momentum p = beta*gamma and IC-channel
    energy budget dE_ic (eV):
      - convert dE_ic to a photon COUNT via dE_ic / (kT*(<t>_p - 2.70)),
        i.e. energy transferred per scattered photon is kT*(<t>_p - 2.70):
        2.70 = E_CMB_MEAN_FACTOR is what the photon carried away from the
        Planck seed spectrum (number-weighted mean of a Planck photon
        number spectrum, x^2/(e^x-1): 3*zeta(4)/zeta(3) = 2.70, matching
        the constant already used elsewhere in this codebase for the same
        quantity); <t>_p*kT is what it carries after the boost.
      - add that many photons at Eq. A.5's shape P(x;p) (the "peak"/boosted
        part of the distortion)
      - remove that many photons from the Planck seed shape itself, at
        x^2/(e^x-1)/(2*zeta(3)) (the "dip" AK19 describe as sitting at the
        Planck peak, since that's number-weighted where most background
        photons are)
    Both pieces are photon-number-conserving by construction (elastic
    Compton scattering), so total returned d(dE)/dx integrates to exactly
    the burst's photon-channel energy budget (f_ic*E0_eV) minus whatever
    falls off the supplied x_grid.

    Returns dict with x_grid, "d_dE_dx" (eV per unit x, the raw per-electron
    spectrum), "f_heat"/"f_ic" (this function's own, un-reprocessed decade
    totals), and "shape_reliable"/"f_ic_corrected" -- see the validity
    cross-check below.

    VALIDITY CROSS-CHECK (added after finding this the hard way): this
    function's decade loop books every IC-channel photon straight to
    distortion via Eq. A.5 with NO further reprocessing -- unlike
    solve_burst.electron_cascade, which recursively feeds its own
    characteristic IC photon through photon_cascade_full so that a
    sufficiently energetic daughter photon can itself pair-produce or
    gg-scatter (items #2/#3/#5's fixes all live inside that reprocessing
    chain). For Fig. 3 (10-70 keV) and Fig. 10 (10-100 keV electrons) the
    characteristic IC photon energy never gets remotely close to the
    pair-production threshold, so the two approaches agree and this gap is
    invisible. But Figs. 11/12 use the SAME electron_ic_spectrum machinery
    (via dual_solver_harness.py) up to 1 GeV/100 GeV, where it very much
    does NOT agree -- checked directly: at E0=1e9 eV, z=1e4, this
    function's own totals say f_heat=0.000/f_ic=1.000, while
    solve_burst.burst_energy_budget('e', 1e9, 1e4, high_energy=True) (the
    correctly-reprocessed answer) says f_heat=0.267/f_ic=0.716 -- a large,
    silent discrepancy, not a rounding-level one.

    So: this function now cross-checks its own f_ic against
    solve_burst.burst_energy_budget's reprocessed answer. If they agree
    (shape_reliable=True), d_dE_dx is returned as-is -- this is the regime
    Eq. A.5's single-generation shape is actually valid in. If they
    disagree beyond a loose tolerance, shape_reliable=False and d_dE_dx is
    rescaled so its integral matches the correctly-reprocessed f_ic total
    (f_ic_corrected) -- this fixes the ENERGY bookkeeping (so callers who
    only need f_heat/f_ic, like dual_solver_harness's heat channel, get the
    right answer) but the detailed Eq. A.5 SHAPE is not reliable in that
    regime (it's missing whatever a real pair-production/gg-scattering
    sub-cascade would actually do to the spectrum) -- callers plotting the
    shape itself (not just the total) should treat shape_reliable=False
    results as qualitative only, or restrict themselves to Fig. 3/10's
    keV-scale range where this doesn't trigger.
    """
    x_grid = np.asarray(x_grid, dtype=float)
    n_e_cm3 = cosmo.n_e_total(z) / 1.0e6
    T_K = cosmo.T_gamma(z)
    kT_eV = rates.K_B_EV_PER_K * T_K
    U_CMB_J_m3 = cosmo.A_RAD * T_K ** 4
    U_CMB_eV_cm3 = U_CMB_J_m3 * 6.241509074e18 / 1.0e6
    E_e_bg_eV = 1.5 * rates.K_B_EV_PER_K * T_K

    if E0_eV <= E_min_eV:
        return {"x_grid": x_grid, "d_dE_dx": np.zeros_like(x_grid), "f_heat": 1.0, "f_ic": 0.0,
                "shape_reliable": True, "f_ic_corrected": 0.0}

    n_decades = max(4, int(np.ceil(np.log10(E0_eV / E_min_eV) * min_decades_per_e_fold)))
    E_edges = np.geomspace(E0_eV, E_min_eV, n_decades + 1)

    seed_shape = _planck_number_shape(x_grid)
    x0_grid = np.geomspace(1e-3, 30.0, 4000)  # seed-frequency integration grid for the convolution

    d_dE_dx = np.zeros_like(x_grid)
    heat_energy = 0.0
    ic_energy = 0.0

    for i in range(n_decades):
        E_hi, E_lo = E_edges[i], E_edges[i + 1]
        E_mid = np.sqrt(E_hi * E_lo)
        dE = E_hi - E_lo

        c = rates.coulomb_loss_rate(E_mid, n_e_cm3, E_e_bg_eV)
        ic = rates.ic_thomson_loss_rate(E_mid, U_CMB_eV_cm3)
        f_heat_local = c / (c + ic) if (c + ic) > 0 else 1.0
        dE_ic = (1.0 - f_heat_local) * dE

        heat_energy += f_heat_local * dE
        if dE_ic <= 0:
            continue

        gamma_e = 1.0 + E_mid / M_E_EV
        beta_e = np.sqrt(max(1.0 - 1.0 / gamma_e ** 2, 0.0))
        p = beta_e * gamma_e
        if p <= 1e-6:
            ic_energy += dE_ic  # negligible boost; book as distortion, no shape to speak of
            continue

        t_bar = mean_t_relativistic(p)
        # <x'_out> = <t>_p * <x0>_seed (seed drawn from the Planck NUMBER
        # shape, mean E_CMB_MEAN_FACTOR=2.70 -- see mean_t_relativistic and
        # _convolved_output_shape docstrings for why t is referenced per
        # actual seed photon, not a fixed kT/h scale).
        boost_per_photon_kT = E_CMB_MEAN_FACTOR * (t_bar - 1.0)
        if boost_per_photon_kT <= 0:
            ic_energy += dE_ic
            continue

        n_scatter = dE_ic / (kT_eV * boost_per_photon_kT)  # photon count (common arbitrary normalization)
        ic_energy += dE_ic

        out_shape = _convolved_output_shape(x_grid, p, x0_grid)
        d_dE_dx += n_scatter * kT_eV * x_grid * out_shape  # added (boosted) photons; dE = x*kT*dN
        d_dE_dx -= n_scatter * kT_eV * x_grid * seed_shape  # removed from the seed (the "dip")

    f_heat = np.clip(heat_energy / E0_eV, 0.0, 1.0)
    f_ic = np.clip(ic_energy / E0_eV, 0.0, 1.0)

    # Validity cross-check -- see docstring. Only worth the (cheap relative
    # to the convolution loop above) extra call above a few hundred keV;
    # below that, reprocessing physically cannot matter (characteristic IC
    # photon energies are far below any pair-production/gg threshold) so
    # skip it to keep the common Fig. 3/10 case fast and dependency-light.
    shape_reliable = True
    f_ic_corrected = f_ic
    if E0_eV > 3.0e5:
        try:
            ref = sb.burst_energy_budget("e", E0_eV, z, high_energy=True)
            f_ic_ref = ref["f_ic"] + ref.get("f_subthreshold", 0.0)
            if abs(f_ic_ref - f_ic) > 0.05:  # loose: flag real divergence, not float noise
                shape_reliable = False
                f_ic_corrected = f_ic_ref
                total = np.trapezoid(d_dE_dx, x_grid)
                if total > 0:
                    d_dE_dx = d_dE_dx * (f_ic_corrected * E0_eV / total)
        except Exception:
            pass  # cross-check is best-effort; never let it break the primary result

    return {"x_grid": x_grid, "d_dE_dx": d_dE_dx, "f_heat": f_heat, "f_ic": f_ic,
            "shape_reliable": shape_reliable, "f_ic_corrected": f_ic_corrected}


def electron_ic_delta_n(species_E0_eV, z, injected_frac_rho_cmb, x_grid, E_min_eV=100.0, min_decades_per_e_fold=4):
    """
    Occupation-number perturbation delta_n(x) from a single electron burst,
    in spectroxide's own convention (delta_n(x) = [G2_0/x^2] * d(deltaN/N)/dx
    -- see interface.py's make_photon_source, which builds this same
    quantity via the narrow-line proxy; this is the Eq. A.5-resolved
    version). Photon-channel only, per electron_ic_spectrum's docstring.
    Also passes through electron_ic_spectrum's shape_reliable/f_ic_corrected
    fields -- see that function's docstring for the GeV-scale caveat.
    """
    spec = electron_ic_spectrum(species_E0_eV, z, x_grid, E_min_eV=E_min_eV, min_decades_per_e_fold=min_decades_per_e_fold)
    T_K = cosmo.T_gamma(z)
    kT_eV = rates.K_B_EV_PER_K * T_K

    rho_cmb_eV_cm3 = sb.rho_cmb_eV_cm3(z)
    scale = injected_frac_rho_cmb * rho_cmb_eV_cm3 / species_E0_eV
    d_dE_dx_physical = spec["d_dE_dx"] * scale  # eV/cm^3 per unit x

    n_gamma = n_gamma_cm3(z)  # cm^-3
    with np.errstate(divide="ignore", invalid="ignore"):
        delta_n = d_dE_dx_physical / (kT_eV * x_grid * (n_gamma * x_grid ** 2 / G2_0))
    delta_n = np.where(np.isfinite(delta_n), delta_n, 0.0)
    return delta_n, {"f_heat": spec["f_heat"], "f_ic": spec["f_ic"],
                      "shape_reliable": spec["shape_reliable"], "f_ic_corrected": spec["f_ic_corrected"]}


def fig3_style_delta_I(species_E0_eV, z, injected_frac_rho_cmb, x_grid, E_min_eV=100.0, min_decades_per_e_fold=4):
    """
    Convert electron_ic_spectrum's per-electron d(dE)/dx into AK19 Fig. 3's
    plotted quantity: delta_I_nu(x) = (2h nu^3/c^2) * delta_n(x), photon-
    channel only (heat/y excluded, per AK19's explicit statement -- see
    electron_ic_spectrum docstring), for a burst injecting
    injected_frac_rho_cmb * rho_CMB(z) of energy as electrons of kinetic
    energy species_E0_eV.

    Returns (delta_I_nu_SI, budget) where delta_I_nu_SI is in
    W m^-2 Hz^-1 sr^-1 (full physical unit conversion done here -- see
    electron_ic_delta_n for the delta_n(x) step -- followed by the standard
    I_nu = 2h nu^3/c^2 * n(x) intensity relation with nu = x*kT/h, using
    TODAY's T_CMB0 since x is redshift-invariant) and budget carries
    f_heat/f_ic (plus shape_reliable/f_ic_corrected) for the cross-check.
    """
    delta_n, budget = electron_ic_delta_n(species_E0_eV, z, injected_frac_rho_cmb, x_grid, E_min_eV=E_min_eV, min_decades_per_e_fold=min_decades_per_e_fold)

    # I_nu = 2h nu^3/c^2 * n(x), nu = x*kT/h -- full SI unit conversion.
    # Use TODAY's T_CMB0 here (not T(z), used above only to get delta_n(x)
    # itself correctly from the physics actually happening at z) -- see
    # docstring note above the call site in reproduce_fig3.py for why.
    EV_TO_J = 1.602176634e-19
    H_PLANCK = 6.62607015e-34
    C_LIGHT_SI = 2.99792458e8
    kT0_eV = rates.K_B_EV_PER_K * cosmo.T_CMB0
    kT0_J = kT0_eV * EV_TO_J
    delta_I = 2.0 * kT0_J ** 3 * x_grid ** 3 * delta_n / (H_PLANCK ** 2 * C_LIGHT_SI ** 2)

    return delta_I, budget
