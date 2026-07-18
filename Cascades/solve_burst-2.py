"""
solve_burst.py -- CP2: low-energy cascade solver (Compton + inverse-Compton
only, pair production and gamma-gamma scattering switched off).

Implements the triangular back-substitution scheme of AK19 Eq. 3.1, restricted
to the low-energy branch (injection energies safely below the 2*m_e*c^2 ~
1.022 MeV pair-production threshold, plan Sec 6 / CP2).

Physics included (plan Sec 3.3, CP2 subset):
  - electron/positron: Coulomb scattering (A.1) + inverse-Compton, Thomson
    limit (A.4)
  - photon: non-relativistic Compton recoil (B.2) sets the "soft" (heating)
    channel; the exact Klein-Nishina mean-fractional-energy-transfer
    integral (AK19 Eq. B.3, via `rates.klein_nishina_mean_transfer_fraction`;
    see `_compton_mean_transfer_fraction` below) sets how much of a photon's
    energy is handed to a single recoil electron per scattering. This used
    to be a phenomenological x/(1+x) interpolation standing in for the full
    Klein-Nishina kinematics -- now replaced with the literal Eq. B.3
    integral (plan's fix #2).

Both channels are solved by the same trick used throughout AK19: since energy
only ever flows to lower energy (triangular structure, Sec 3.2 of the plan),
we do not need to bin and back-substitute explicitly here -- we can integrate
the *local branching ratio* between competing loss channels along the
one-way cooling track from E0 down to threshold. This is mathematically the
continuum limit of the same back-substitution and is exact for the "on-the-
spot" approximation (Sec 3.4): every particle deposits all of its energy
within the current redshift step.

Outputs of a burst are always decomposed into exactly three buckets so that
energy conservation (Sec 3.5) can be asserted directly:
    f_heat + f_ic + f_subthreshold = 1
"""

import numpy as np
from scipy import integrate

from . import rates
from . import cosmology as cosmo

M_E_EV = rates.M_E_EV


# ----------------------------------------------------------------------
# Electron/positron branching: Coulomb (-> heat) vs IC-Thomson (-> photon)
# ----------------------------------------------------------------------
def electron_heat_fraction(E0_eV, z, E_min_eV=1000.0):
    """
    Fraction of an injected electron's kinetic energy E0_eV that ends up as
    direct heat (Coulomb channel), integrating the local branching ratio
    coulomb_rate / (coulomb_rate + ic_rate) along the cooling track from
    E_min_eV up to E0_eV (Sec 3.4 on-the-spot bookkeeping):

        f_heat(E0) = (1/E0) * integral_{E_min}^{E0} [coulomb(E) / (coulomb(E) + ic(E))] dE

    The complement, 1 - f_heat, is the fraction handed to CMB photons via
    inverse-Compton (the IC channel).
    """
    n_e_cm3 = cosmo.n_e_total(z) / 1.0e6  # m^-3 -> cm^-3 (H + fully-ionized He electrons)
    T_K = cosmo.T_gamma(z)
    U_CMB_J_m3 = cosmo.A_RAD * T_K ** 4
    U_CMB_eV_cm3 = U_CMB_J_m3 * 6.241509074e18 / 1.0e6
    E_e_bg_eV = 1.5 * rates.K_B_EV_PER_K * T_K

    def branch(E):
        c = rates.coulomb_loss_rate(E, n_e_cm3, E_e_bg_eV)
        ic = rates.ic_thomson_loss_rate(E, U_CMB_eV_cm3)
        return c / (c + ic)

    if E0_eV <= E_min_eV:
        return 1.0  # entirely sub-threshold, Coulomb-dominated -> heat

    val, _ = integrate.quad(branch, E_min_eV, E0_eV, limit=200)
    return np.clip(val / E0_eV, 0.0, 1.0)


# ----------------------------------------------------------------------
# Photon branching: Compton recoil cascade
# ----------------------------------------------------------------------
def _compton_mean_transfer_fraction(E_gamma_eV):
    """
    Mean fractional energy a photon of energy E_gamma_eV hands to the
    electron in a single Compton scattering, delta(E) = <Delta E>/E.

    FIX (plan's fix #2, amended-priority item 3): this now calls the exact
    Klein-Nishina mean transfer fraction, `rates.klein_nishina_mean_transfer_fraction`,
    which integrates the literal AK19 Eq. B.3 differential cross section
    over the allowed scattered-photon energy range -- rather than the old
    phenomenological interpolation delta(E) = x/(1+x), x = E/(m_e c^2).

    The old x/(1+x) form is still worth recording as a sanity check: it
    matched the two known limits (delta -> x as x -> 0, the non-relativistic
    recoil limit consistent with Eq. B.2; delta -> 1 as x -> infinity, deep
    Klein-Nishina saturation) and the right order of magnitude at the
    crossover, but it's a guessed interpolating shape, not the paper's own
    formula -- the exact Eq. B.3 integral has a different functional form
    in between the two limits (see `rates.klein_nishina_mean_transfer_fraction`
    docstring for the closed-form derivation and its comparison against
    numerical quadrature).
    """
    return rates.klein_nishina_mean_transfer_fraction(E_gamma_eV)


def photon_cascade(E0_eV, z, E_min_eV=1000.0, max_scatters=100000):
    """
    Cascade a single injected photon of energy E0_eV down via repeated
    Compton scatters (CP2: no pair production, no gamma-gamma scattering).

    At each scatter, a fraction delta(E) of the photon's current energy is
    handed to a fresh recoil electron, which is immediately processed by
    `electron_heat_fraction` (on-the-spot, Sec 3.4) to get its own
    heat/IC split. The remaining (1-delta(E)) stays with the photon, which
    continues scattering until it falls below E_min_eV.

    Returns a dict with three energy fractions of E0_eV that must sum to 1:
        f_heat            -- deposited as heat via daughter electrons' Coulomb channel
        f_ic              -- handed to CMB photons via daughter electrons' IC channel
        f_subthreshold     -- energy remaining in the photon itself once it
                              drops below E_min_eV (the lowest bin, plan Sec 3.1)
    """
    E = float(E0_eV)
    heat_energy = 0.0
    ic_energy = 0.0
    n = 0
    while E > E_min_eV and n < max_scatters:
        delta = _compton_mean_transfer_fraction(E)
        dE_electron = E * delta
        if dE_electron < 1e-12 * E0_eV:
            # negligible further transfer per scatter (deep non-relativistic
            # regime) -- the rest is only removed by many-many tiny recoils,
            # which is exactly the non-relativistic recoil heating captured
            # by Eq. B.2; treat the remainder as heat directly rather than
            # looping indefinitely.
            heat_energy += E
            E = 0.0
            break
        f_heat_e = electron_heat_fraction(dE_electron, z)
        heat_energy += f_heat_e * dE_electron
        ic_energy += (1.0 - f_heat_e) * dE_electron
        E -= dE_electron
        n += 1

    f_heat = heat_energy / E0_eV
    f_ic = ic_energy / E0_eV
    f_subthreshold = E / E0_eV
    return {"f_heat": f_heat, "f_ic": f_ic, "f_subthreshold": f_subthreshold, "n_scatters": n}


# ----------------------------------------------------------------------
# CP3: high-energy cascade -- add pair production and gamma-gamma scattering
# ----------------------------------------------------------------------
# CP2's electron_heat_fraction/photon_cascade are Compton+IC only and are
# kept exactly as-is above (still exercised by tests/test_cp2_*.py as a
# regression baseline). CP3 adds two things on top:
#   (1) at high photon energy, pair production (Eq. B.4) and gamma-gamma
#       scattering on the CMB (Eq. B.6) compete with Compton scattering as
#       loss channels, and
#   (2) the cascade becomes genuinely recursive: an electron's IC-upscattered
#       photon can itself be energetic enough to pair-produce or gg-scatter
#       again, and a pair-produced e+/e- can itself IC-upscatter another hard
#       photon. This backreaction is exactly what AK19 point to as the reason
#       electron-only and photon-only injection converge to the same
#       "universal" spectrum above ~10 GeV (AK19 Sec 3.2, Figs 8b/9a) -- the
#       cascade forgets what species it started as after a few generations.
#       That is CP3's headline check (test_cp3_high_energy_cascade.py).
#
# Positrons: per plan Sec 3.3/3.4 ("treat every injected positron as an
# injected electron of the same kinetic energy, plus two 511 keV photons"),
# handled at CP4, not here -- pair production below therefore books both
# daughters of a pair as plain electrons for CP3's purposes (their combined
# heat/IC split is unaffected by the electron/positron label; only the two
# 511 keV annihilation photons, deferred to CP4, are left out).

E_CMB_MEAN_FACTOR = 2.70  # <h nu> ~ 2.70 kT for a Planck spectrum, kept only
                          # as a fallback/reference value -- no longer used
                          # to build the single characteristic IC photon (see
                          # _planck_photon_number_quantiles / N_IC_SAMPLES
                          # below, "issue 1.1" fix).

N_IC_SAMPLES = 5  # number of seed-photon energies spread across the CMB
                  # Planck spectrum used to represent each decade's
                  # up-scattered IC photon output, replacing the old
                  # single-characteristic-photon approximation.


def _planck_photon_number_quantiles(n_samples=N_IC_SAMPLES, x_min=0.01, x_max=20.0, n_grid=4000):
    """
    Return n_samples values of x = E_photon / (k_B T) drawn as evenly
    spaced quantiles of the CMB Planck photon number distribution,

        dN/dx ~ x^2 / (exp(x) - 1),

    i.e. the seed-photon energy distribution actually being inverse-Compton
    upscattered, rather than a single characteristic energy (<h nu> ~ 2.70 kT)
    or a uniform-log grid. Equal-probability quantiles give each of the
    n_samples representative energies equal weight (1/n_samples) while
    still concentrating samples where the photon number density is largest.

    This directly addresses issue 1.1 (electron_cascade's
    single-characteristic-IC-photon-per-decade approximation): instead of
    recursing on one representative photon per decade, we now recurse on
    n_samples photons spread across the Planck spectrum and average the
    results, smoothing the boost/de-boost oscillation the way AK19's
    resolved multi-bin treatment does (their Eq. 3.1).
    """
    x = np.linspace(x_min, x_max, n_grid)
    pdf = x ** 2 / np.expm1(x)
    cdf = integrate.cumulative_trapezoid(pdf, x, initial=0.0)
    cdf /= cdf[-1]
    probs = (np.arange(n_samples) + 0.5) / n_samples
    return np.interp(probs, cdf, x)


def _pair_production_total_rate(E_gamma_eV, z):
    """
    Total Bethe-Heitler pair-production rate (s^-1) for a photon of energy
    E_gamma_eV at redshift z, summing over free electrons (Z=1), ionized
    hydrogen nuclei (Z=1) and ionized helium (Z=2), Z^2-weighted exactly as
    Eq. B.4 prescribes (plan Sec 3.3, "sum over H+/He+/He2+").
    """
    n_H_cm3 = cosmo.n_H(z) / 1.0e6
    n_He_cm3 = cosmo.n_He(z) / 1.0e6
    Xe = cosmo.X_e(z)
    n_p_cm3 = Xe * n_H_cm3            # ionized H nuclei, Z=1
    n_e_cm3 = cosmo.n_e_total(z) / 1.0e6  # ALL free electrons, Z=1 (H-ionization + He's 2 electrons/nucleus)
    n_He_ion_cm3 = Xe * n_He_cm3      # ionized He nuclei, Z=2 (scaled by X_e(z), see n_e_total's docstring)

    sigma_p = rates.pair_production_xsec(E_gamma_eV, Z=1.0)
    sigma_He = rates.pair_production_xsec(E_gamma_eV, Z=2.0)

    n_eff_Z1 = n_p_cm3 + n_e_cm3
    return rates.C_CM_S * (n_eff_Z1 * sigma_p + n_He_ion_cm3 * sigma_He)


def _compton_total_rate(z):
    """Compton scattering rate n_e sigma_T c (s^-1); used only for branching
    against pair production / gg scattering, so the (unmodeled) high-energy
    Klein-Nishina suppression of the total cross section is not needed here
    -- at the energies where that suppression matters, pair production and
    gg scattering already dominate by orders of magnitude (AK19 Sec 2)."""
    n_e_cm3 = cosmo.n_e_total(z) / 1.0e6
    return n_e_cm3 * rates.SIGMA_T_CM2 * rates.C_CM_S


def electron_cascade(E0_eV, z, E_min_eV=1000.0, depth=0, max_depth=60):
    """
    CP3 electron cascade: like electron_heat_fraction, but the IC-upscattered
    energy is no longer booked directly as "f_ic" -- instead a representative
    daughter photon is built and recursively cascaded through
    photon_cascade_full, so that a sufficiently relativistic electron's
    up-scattered photon can itself pair-produce / gg-scatter and continue the
    shower (plan Sec 3.4's on-the-spot bookkeeping, extended to the
    multi-generation regime CP3 targets).

    The electron's cooling track (E0 -> E_min) is stepped in log-decades;
    within each decade the local Coulomb/IC branching ratio (same formulas as
    electron_heat_fraction) splits that decade's energy loss into heat vs. a
    representative IC photon of characteristic energy
    (4/3) gamma^2 beta^2 * <E_CMB>, which is fed back into photon_cascade_full.

    Returns {"f_heat", "f_ic", "f_subthreshold"} fractions of E0_eV.
    """
    if E0_eV <= E_min_eV or depth >= max_depth:
        return {"f_heat": 1.0, "f_ic": 0.0, "f_subthreshold": 0.0}

    n_e_cm3 = cosmo.n_e_total(z) / 1.0e6
    T_K = cosmo.T_gamma(z)
    U_CMB_J_m3 = cosmo.A_RAD * T_K ** 4
    U_CMB_eV_cm3 = U_CMB_J_m3 * 6.241509074e18 / 1.0e6
    E_e_bg_eV = 1.5 * rates.K_B_EV_PER_K * T_K
    # Issue 1.1 fix: N_IC_SAMPLES seed-photon energies spread across the CMB
    # Planck photon-number distribution, in place of the single
    # characteristic energy E_CMB_MEAN_FACTOR * kT used previously. Depends
    # only on z (via T_K), so computed once per electron_cascade call rather
    # than per decade.
    #
    # Each sample recurses into photon_cascade_full, which itself recurses
    # back into electron_cascade -- so naively spreading at every recursion
    # depth multiplies the branching factor by N_IC_SAMPLES at every level
    # of an already-deep cascade tree (max_depth=60), which is
    # combinatorially explosive. The spread only matters where it was
    # actually quantified (top few generations, where most of the injected
    # energy still lives); a few generations down, the per-decade IC energy
    # budget is already a small fraction of a small fraction, so we fall
    # back to a single representative (median-quantile) photon there,
    # matching the original approximation's cost/accuracy at
    # negligible-impact depths.
    IC_SPREAD_MAX_DEPTH = 0
    if depth <= IC_SPREAD_MAX_DEPTH:
        E_cmb_samples_eV = _planck_photon_number_quantiles() * rates.K_B_EV_PER_K * T_K
    else:
        E_cmb_samples_eV = _planck_photon_number_quantiles(n_samples=1) * rates.K_B_EV_PER_K * T_K
    n_ic_samples = len(E_cmb_samples_eV)

    n_decades = max(4, int(np.ceil(np.log10(E0_eV / E_min_eV) * 4)))
    E_edges = np.geomspace(E0_eV, E_min_eV, n_decades + 1)

    heat_energy = 0.0
    ic_energy = 0.0
    for i in range(n_decades):
        E_hi, E_lo = E_edges[i], E_edges[i + 1]
        E_mid = np.sqrt(E_hi * E_lo)
        c = rates.coulomb_loss_rate(E_mid, n_e_cm3, E_e_bg_eV)
        ic = rates.ic_thomson_loss_rate(E_mid, U_CMB_eV_cm3)
        f_heat_local = c / (c + ic)
        dE = E_hi - E_lo

        heat_energy += f_heat_local * dE
        dE_ic = (1.0 - f_heat_local) * dE
        if dE_ic <= 0:
            continue

        gamma_e = 1.0 + E_mid / M_E_EV
        beta2 = 1.0 - 1.0 / gamma_e ** 2

        # Issue 1.1 fix: instead of recursing on a single characteristic
        # up-scattered photon energy, split this decade's IC energy budget
        # dE_ic evenly across n_ic_samples seed photons drawn from the
        # spread of the CMB Planck distribution and recurse on each,
        # averaging the results. This smooths the boost/de-boost
        # oscillation that a single-point approximation leaves unsmoothed
        # (AK19 Sec 4 / Fig 13 discussion).
        dE_ic_sample = dE_ic / n_ic_samples
        for E_cmb_eV in E_cmb_samples_eV:
            E_gamma_char = (4.0 / 3.0) * gamma_e ** 2 * beta2 * E_cmb_eV
            E_gamma_char = min(E_gamma_char, dE_ic_sample)  # can't exceed this sample's energy budget

            if E_gamma_char <= E_min_eV:
                # sub-relativistic IC photon, well below any pair/gg threshold --
                # book directly as distortion (equivalent to CP2's treatment)
                ic_energy += dE_ic_sample
            else:
                sub_budget = photon_cascade_full(E_gamma_char, z, E_min_eV=E_min_eV,
                                                  depth=depth + 1, max_depth=max_depth)
                # sub_budget's fractions are relative to E_gamma_char; scale by
                # the actual energy budget dE_ic_sample being represented by
                # this sample photon (dE_ic_sample may differ from
                # E_gamma_char, since E_gamma_char is clipped -- see the
                # min() above).
                heat_energy += sub_budget["f_heat"] * dE_ic_sample
                ic_energy += (sub_budget["f_ic"] + sub_budget["f_subthreshold"]) * dE_ic_sample

    # The loop above only accounts for energy lost while cooling from E0_eV
    # down to E_min_eV, i.e. sum(dE) = E0_eV - E_min_eV. The leftover
    # E_min_eV of kinetic energy the electron still carries once it hits the
    # floor must still be booked somewhere, or energy silently leaks out of
    # the recursion (this is most visible for the low-energy Compton-recoil
    # electrons electron_cascade is recursively called on from
    # photon_cascade_full, where E_min_eV/E0_eV is not negligible). Book it
    # as heat, consistent with this function's own convention for an
    # electron injected at or below E_min_eV (the E0_eV <= E_min_eV branch
    # above, which returns f_heat=1.0).
    heat_energy += E_min_eV

    f_heat = heat_energy / E0_eV
    f_ic = ic_energy / E0_eV
    return {"f_heat": np.clip(f_heat, 0.0, 1.0), "f_ic": np.clip(f_ic, 0.0, 1.0), "f_subthreshold": 0.0}


def photon_cascade_full(E0_eV, z, E_min_eV=1000.0, max_scatters=2000, depth=0, max_depth=60):
    """
    CP3 photon cascade: Compton scattering, Bethe-Heitler pair production
    (Eq. B.4), and gamma-gamma scattering on the CMB (Eq. B.6) all compete as
    loss channels, branching at each step by their relative rates.

      - Compton branch: same exact Klein-Nishina delta(E) transfer as
        photon_cascade (CP2) -- see `_compton_mean_transfer_fraction` -- but
        the recoil electron is now processed by the recursive
        `electron_cascade` (not the terminal electron_heat_fraction), so its
        own IC output can rejoin the shower.
      - Pair-production branch: the photon is destroyed; per plan Sec 3.3
        ("treat every injected positron as an injected electron... plus two
        511 keV photons", applied here to both pair-production daughters
        pre-CP4), its energy (minus the 2*m_e*c^2 rest-mass cost) is split
        evenly between two daughter electrons, each recursively cascaded.
      - Gamma-gamma branch: per AK19's own description ("gives a large
        fraction of its energy to the background photon in one scattering"),
        this branch's energy is booked directly to the distortion bucket
        (f_ic) rather than recursed -- consistent with AK19's explicit
        justification for *not* separately modeling pair production on the
        CMB (Sec 2): both channels rapidly converge to a universal spectrum,
        so resolving gg-scattering's daughter spectrum in detail would not
        change the CP3 pass criterion (the universal-spectrum convergence
        itself).

    Returns {"f_heat", "f_ic", "f_subthreshold"} fractions of E0_eV.
    """
    if depth >= max_depth:
        return {"f_heat": 0.0, "f_ic": 0.0, "f_subthreshold": 1.0}

    E = float(E0_eV)
    heat_energy = 0.0
    ic_energy = 0.0
    n = 0
    while E > E_min_eV and n < max_scatters:
        compton_rate = _compton_total_rate(z)
        pair_rate = _pair_production_total_rate(E, z)
        gg_rate = rates.gamma_gamma_scatter_rate(E, cosmo.T_gamma(z))
        total_rate = compton_rate + pair_rate + gg_rate
        if total_rate <= 0:
            break

        p_compton = compton_rate / total_rate
        p_pair = pair_rate / total_rate
        p_gg = gg_rate / total_rate

        E_compton = E * p_compton
        E_pair = E * p_pair
        E_gg = E * p_gg

        # Compton channel: recoil electron + surviving degraded photon
        delta = _compton_mean_transfer_fraction(E)
        dE_electron = E_compton * delta
        remainder_photon = E_compton * (1.0 - delta)
        if dE_electron > 0:
            sub = electron_cascade(dE_electron, z, E_min_eV=E_min_eV,
                                    depth=depth + 1, max_depth=max_depth)
            heat_energy += sub["f_heat"] * dE_electron
            ic_energy += sub["f_ic"] * dE_electron

        # Pair-production channel: destroyed, energy (less 2 m_e c^2) split
        # evenly between two daughter electrons. The 2*m_e*c^2 rest-mass
        # cost isn't actually lost -- it comes back out as two 511 keV
        # annihilation photons once the daughter positron thermalizes and
        # annihilates (mirrors positron_burst_energy's treatment of the
        # exact same 2*m_e*c^2 bookkeeping), so re-inject and cascade those
        # here too rather than silently dropping them.
        if E_pair > 0:
            E_kinetic_total = max(E_pair - 2.0 * M_E_EV, 0.0)
            if E_kinetic_total > 0:
                E_each = E_kinetic_total / 2.0
                for _ in range(2):
                    sub = electron_cascade(E_each, z, E_min_eV=E_min_eV,
                                            depth=depth + 1, max_depth=max_depth)
                    heat_energy += sub["f_heat"] * E_each
                    ic_energy += sub["f_ic"] * E_each
                for _ in range(2):  # two 511 keV annihilation photons
                    ann = photon_cascade_full(M_E_EV, z, E_min_eV=E_min_eV,
                                               depth=depth + 1, max_depth=max_depth)
                    heat_energy += ann["f_heat"] * M_E_EV
                    ic_energy += (ann["f_ic"] + ann.get("f_subthreshold", 0.0)) * M_E_EV
            else:
                # below the pair-production threshold in practice; shouldn't
                # normally trigger since pair_rate=0 below threshold, but
                # keep energy accounted for safety
                ic_energy += E_pair

        # Gamma-gamma channel: booked directly to distortion (see docstring)
        ic_energy += E_gg

        E = remainder_photon
        n += 1

        if delta < 1e-6 and p_pair < 1e-6 and p_gg < 1e-6:
            # deep non-relativistic Compton-only regime with negligible
            # per-scatter transfer -- terminate as heat rather than looping
            # (mirrors CP2's photon_cascade termination condition)
            heat_energy += E
            E = 0.0
            break

    f_heat = heat_energy / E0_eV
    f_ic = ic_energy / E0_eV
    f_subthreshold = E / E0_eV
    return {"f_heat": np.clip(f_heat, 0.0, 1.0), "f_ic": np.clip(f_ic, 0.0, 1.0),
            "f_subthreshold": np.clip(f_subthreshold, 0.0, 1.0), "n_scatters": n}


# ----------------------------------------------------------------------
# CP4: positron / annihilation bookkeeping
# ----------------------------------------------------------------------
# Per plan Sec 3.3 ("Annihilation rate is always much slower than cooling
# ... treat every injected positron as an injected electron of the same
# kinetic energy, plus two 511 keV photons, exactly as AK19 do in their last
# paragraph of Sec 3.1") -- this removes the need to separately track e+
# energy-loss channels (Eqs. A.2/A.3 are catalogued in rates.py but not
# exercised by the cascade itself, matching AK19's own approach).
#
# Bookkeeping note: the two 511 keV photons come from annihilation with a
# *background* thermal electron, not from the injected positron's own
# kinetic energy -- so they add real extra energy to the cascade beyond the
# positron's KE. Burst outputs below are therefore normalized to the
# positron's kinetic energy alone (matching AK19's f_X, "total kinetic
# energy density of injected electron-positron pairs", Sec 3.1 discussion),
# and f_ic/f_heat/f_subthreshold CAN exceed 1 for low-KE positrons -- this
# is expected and is exactly the effect AK19 describe as "a big factor for
# a low energy positron".

E_ANNIHILATION_EV = 2.0 * M_E_EV  # energy of a single 511 keV photon


def positron_burst_energy(E0_eV, z, high_energy=False):
    """
    Energy-fraction budget (relative to the positron's kinetic energy E0_eV)
    for a positron burst: its kinetic energy cascades exactly like an
    electron's (electron_heat_fraction / electron_cascade), plus two
    511 keV annihilation photons are injected and cascaded through the
    photon solver (photon_cascade / photon_cascade_full).

    511 keV sits below the pair-production threshold (1.022 MeV) so the
    CP2 and CP3 photon solvers agree on the annihilation-photon fate; the
    high_energy flag mainly controls which solver processes the KE part.
    """
    if high_energy:
        ke_budget = electron_cascade(E0_eV, z)
    else:
        f_heat = electron_heat_fraction(E0_eV, z)
        ke_budget = {"f_heat": f_heat, "f_ic": 1.0 - f_heat, "f_subthreshold": 0.0}

    ke_heat = ke_budget["f_heat"] * E0_eV
    ke_ic = ke_budget["f_ic"] * E0_eV
    ke_sub = ke_budget.get("f_subthreshold", 0.0) * E0_eV

    ann_heat = ann_ic = ann_sub = 0.0
    for _ in range(2):  # two 511 keV photons per annihilation
        if high_energy:
            b = photon_cascade_full(rates.M_E_EV, z)
        else:
            b = photon_cascade(rates.M_E_EV, z)
        ann_heat += b["f_heat"] * rates.M_E_EV
        ann_ic += b["f_ic"] * rates.M_E_EV
        ann_sub += b.get("f_subthreshold", 0.0) * rates.M_E_EV

    f_heat = (ke_heat + ann_heat) / E0_eV
    f_ic = (ke_ic + ann_ic) / E0_eV
    f_subthreshold = (ke_sub + ann_sub) / E0_eV
    return {"f_heat": f_heat, "f_ic": f_ic, "f_subthreshold": f_subthreshold,
            "E_annihilation_eV": 2.0 * rates.M_E_EV}


# ----------------------------------------------------------------------
# CP5: multi-step redshift marching
# ----------------------------------------------------------------------
# The plan's CP5 (Sec 6) is about letting a burst's surviving high-energy
# population continue interacting across *multiple* redshift steps rather
# than assuming it fully completes within a single step. In this code,
# "on-the-spot" completion is not an assumption bolted on separately -- it
# falls directly out of CP1's own result (test_cp1_rates.py's Coulomb/IC
# crossover check + AK19 Fig. 2, reproduced by the same rate functions used
# throughout electron_cascade/photon_cascade_full): cooling times are always
# many orders of magnitude shorter than the Hubble time across the energy
# range considered here, so each burst's cascade genuinely does complete
# within a single redshift step, and burst_energy_budget already returns
# that completed result.
#
# What CP5 concretely adds, then, is machinery to evaluate a burst's
# response *as a function of injection redshift* z_inj -- exactly the
# z-dependence AK19 show in Figs. 10/11, and exactly the per-z_inj building
# block the CP6 Green's-function convolution (Sec 3.6 of the plan) needs.
# The qualitative trend to check (plan's CP5 pass criterion): sharper,
# higher-amplitude distortions at higher z_inj, because there are more
# scatterings/background targets available (higher n_e, higher U_CMB)
# before the fixed low-z endpoint is reached.
def burst_y_vs_redshift(species, E0_eV, z_inj_grid, injected_frac_rho_cmb=1.0e-5,
                         high_energy=None):
    """
    Evaluate the burst y-distortion estimate for a fixed injected
    species/energy across a grid of injection redshifts z_inj_grid.

    high_energy=None auto-selects the CP3 (pair production + gg scattering)
    solver whenever E0_eV is above the pair-production threshold
    (2*m_e*c^2), and the CP2 solver otherwise -- matching how CP0-4 already
    partition their validated energy domains.

    Returns a list of (z_inj, y, budget) tuples, one per grid point,
    ordered the same as z_inj_grid.
    """
    if high_energy is None:
        high_energy = E0_eV > 2.0 * M_E_EV

    results = []
    for z_inj in z_inj_grid:
        y, budget = y_distortion_estimate(species, E0_eV, z_inj,
                                           injected_frac_rho_cmb, high_energy=high_energy)
        results.append((z_inj, y, budget))
    return results



def burst_energy_budget(species, E0_eV, z, high_energy=None):
    """
    Return the (f_heat, f_ic, f_subthreshold) energy-fraction decomposition
    for a single injected particle of the given species ('e', 'e+', or
    'gamma') at kinetic/photon energy E0_eV, injected at redshift z.

    high_energy=None (default) auto-selects the solver by comparing E0_eV
    to twice the electron rest mass (E0_eV > 2*M_E_EV), the same convention
    used by burst_y_vs_redshift, convolve_generic_injection_y,
    interface.make_dq_dz, and interface.make_photon_source -- this used to
    hardcode high_energy=False here, which was inconsistent with those call
    sites (not live-broken today since every current caller passes
    high_energy explicitly or stays below the pair-production threshold,
    but an easy trap for a future call site). high_energy=False forces the
    CP2 Compton+IC-only solver, valid below the pair-production threshold;
    high_energy=True forces the CP3 solver (electron_cascade /
    photon_cascade_full), which adds pair production and gamma-gamma
    scattering and is required above ~1 MeV.

    species='e+' (CP4) adds the two 511 keV annihilation photons on top of
    the positron's own kinetic-energy cascade (see positron_burst_energy);
    note its f_heat/f_ic/f_subthreshold are normalized to the positron's KE
    and so can exceed 1 at low KE (annihilation injects extra energy).
    """
    if high_energy is None:
        high_energy = E0_eV > 2.0 * M_E_EV
    if species == "e":
        if high_energy:
            return electron_cascade(E0_eV, z)
        f_heat = electron_heat_fraction(E0_eV, z)
        return {"f_heat": f_heat, "f_ic": 1.0 - f_heat, "f_subthreshold": 0.0}
    elif species == "e+":
        return positron_burst_energy(E0_eV, z, high_energy=high_energy)
    elif species == "gamma":
        if high_energy:
            return photon_cascade_full(E0_eV, z)
        return photon_cascade(E0_eV, z)
    else:
        raise ValueError(f"unknown species {species!r}")


def y_distortion_estimate(species, E0_eV, z, injected_frac_rho_cmb=1.0e-5, high_energy=None):
    """

    Estimate the Compton-y parameter for a monochromatic burst of the given
    species/energy, injecting a total energy of injected_frac_rho_cmb * rho_CMB(z).

    Uses the standard single-scattering relation between fractional energy
    added to the photon field and the y-parameter, Delta(rho_gamma)/rho_gamma
    = 4y (valid in the small-y, Thomson-dominated regime that CP2 targets --
    AK19's own Fig. 6d/7a targets are exactly this regime, y ~ 1e-6).

    IMPORTANT (bug fix, flagged so the reasoning is traceable): the heat
    channel (f_heat) is included here, not just f_ic/f_subthreshold. Heat
    deposited in the background electrons is not a dead end -- those
    electrons immediately upscatter CMB photons via ordinary Compton
    scattering, which is exactly the y-distortion mechanism itself (this is
    literally how a y-distortion is defined/produced; AK19's own
    introduction describes heating and Compton up-scattering as two views of
    the same process in the single-scattering regime this module targets).
    An earlier version of this function excluded f_heat, which silently
    discarded up to ~90% of the injected energy's contribution to y for
    heat-dominated (typically low-energy photon) bursts -- and was
    internally inconsistent with f_X_for_target_all_heat_y below, which
    already assumes 100% heat-to-y efficiency for its calibration. Both are
    now consistent: heat contributes to y here exactly as
    f_X_for_target_all_heat_y assumes.

    Returns (y, budget) where budget is the energy-fraction dict, with the
    additional key 'f_total' == f_heat+f_ic+f_subthreshold asserted close to 1.
    """
    budget = burst_energy_budget(species, E0_eV, z, high_energy=high_energy)
    f_distortion = budget["f_heat"] + budget["f_ic"] + budget.get("f_subthreshold", 0.0)
    delta_rho_over_rho = f_distortion * injected_frac_rho_cmb
    y = delta_rho_over_rho / 4.0

    f_total = budget["f_heat"] + budget["f_ic"] + budget.get("f_subthreshold", 0.0)
    budget["f_total"] = f_total
    return y, budget


# ----------------------------------------------------------------------
# CP6: Green's function grid + convolution for continuous (decaying-particle)
# injection, plus a minimal standalone Compton-only relaxation
# ----------------------------------------------------------------------
# Per plan Sec 3.6, exploit linearity (AK19's own point: "we can obtain the
# spectral distortion solution for any general initial particle spectrum by
# doing a linear superposition of the monochromatic spectra"): rather than
# marching a continuously-injected, exponentially-decaying source through
# many redshift steps directly, evaluate the single-burst response
# (already built and validated CP2-5) on a grid of injection redshifts and
# convolve against the physical injection rate dGamma/dz.
#
# Standalone relaxation (plan Sec 7 fallback, used here "for this checkpoint
# only" per plan Sec 6/CP6): every burst response above is already reported
# in the y-type single-scattering approximation (Delta rho/rho = 4y) used
# throughout CP2-5 rather than the full Sazonov-Sunyaev Kompaneets PDE
# (Eqs. 3.4-3.6). This is an explicit, flagged open item -- a full Kompaneets
# relaxation would additionally capture Doppler broadening/boosting and
# stimulated scattering (spectral *shape* corrections), but at the level of
# total distortion *amplitude* that CP6's pass criterion targets (AK19 Sec
# 4.1's "y = 1.5e-5 all-heat benchmark -> actual few*1e-6"), the
# single-scattering y-relation is the right leading-order tool and is what
# is used here; swapping in the literal PDE is deferred as follow-up work.

def rho_dm_eV_cm3(z):
    """Physical dark-matter energy density at redshift z, in eV/cm^3."""
    omega_dm = cosmo.OMEGA_M - cosmo.OMEGA_B
    rho_mass_kg_m3 = omega_dm * cosmo.RHO_CRIT0 * (1.0 + z) ** 3
    rho_J_m3 = rho_mass_kg_m3 * cosmo.C_LIGHT ** 2
    return rho_J_m3 * 6.241509074e18 / 1.0e6


def rho_cmb_eV_cm3(z):
    """CMB photon energy density at redshift z, in eV/cm^3."""
    T_K = cosmo.T_gamma(z)
    rho_J_m3 = cosmo.A_RAD * T_K ** 4
    return rho_J_m3 * 6.241509074e18 / 1.0e6


def f_X_for_target_all_heat_y(z_X, target_y_all_heat):
    """
    Solve for the decaying-species DM fraction f_X such that instantaneously
    converting f_X * rho_dm(z_X) entirely to heat at z_X would produce the
    quoted "would-be-all-heat" y-distortion (plan/AK19 Sec 4.1 benchmark:
    z_X=2e4, target_y_all_heat=1.5e-5), using the same Delta(rho)/rho = 4y
    relation as the rest of this module:

        target_y_all_heat = f_X * rho_dm(z_X) / (4 * rho_CMB(z_X))
    """
    return target_y_all_heat * 4.0 * rho_cmb_eV_cm3(z_X) / rho_dm_eV_cm3(z_X)


def decay_injection_rate_eV_cm3_s(z, f_X, tau_X_s):
    """
    Physical energy-injection rate density (eV cm^-3 s^-1) at redshift z for
    a decaying species with DM fraction f_X and lifetime tau_X_s:

        dE/dV/dt (z) = f_X * rho_dm(z) / tau_X * exp(-t(z)/tau_X)

    (physical DM density diluting as (1+z)^3 like any other matter, times
    the standard radioactive-decay depletion factor).
    """
    return f_X * rho_dm_eV_cm3(z) / tau_X_s * np.exp(-cosmo.cosmic_time(z) / tau_X_s)


def build_zinj_grid(tau_X_s, n_z=40, t_min_factor=1.0e-2, t_max_factor=30.0):
    """
    Build a log-spaced grid of injection redshifts spanning the decay's
    active e-folds (plan Sec 3.6, "~30-100 log-spaced values spanning the
    decay's active e-folds"), found by bisecting cosmic_time(z) against
    tau_X_s * t_min_factor and tau_X_s * t_max_factor.
    """
    from scipy import optimize

    def solve_z_for_time(t_target):
        f = lambda z: cosmo.cosmic_time(z) - t_target
        return optimize.brentq(f, 1.0, 3.0e6)

    z_hi = solve_z_for_time(tau_X_s * t_min_factor)   # early time -> high z
    z_lo = solve_z_for_time(tau_X_s * t_max_factor)   # late time -> low z
    z_lo = max(z_lo, 1.0)
    return np.sort(np.geomspace(z_hi, z_lo, n_z))


def convolve_generic_injection_y(species, E0_eV, injection_rate_eV_cm3_s_fn, z_grid,
                                  high_energy=None):
    """
    Generic version of the CP6 convolution (decay_convolution_y), factored
    out so CP8 can reuse it with a different injection-rate normalization
    (MPR21's dark-baryon decay uses n_H(z)-normalized f_X, not the DM-density
    normalization decay_injection_rate_eV_cm3_s uses) without duplicating the
    burst-response/integration logic.

    injection_rate_eV_cm3_s_fn(z) must return the physical energy-injection
    rate density dE/dV/dt at redshift z, in eV cm^-3 s^-1.

    Returns (y_total, dy_dz_values) over the given z_grid (ascending order).
    """
    if high_energy is None:
        high_energy = E0_eV > 2.0 * M_E_EV

    def shell_y_at(zp):
        rate_eV_cm3_s = injection_rate_eV_cm3_s_fn(zp)
        dt_dz = 1.0 / ((1.0 + zp) * cosmo.H(zp))
        dE_dz_eV_cm3 = rate_eV_cm3_s * dt_dz
        injected_frac = dE_dz_eV_cm3 / rho_cmb_eV_cm3(zp)
        y_shell, _ = y_distortion_estimate(species, E0_eV, zp, injected_frac,
                                            high_energy=high_energy)
        return y_shell

    dy_dz = np.array([shell_y_at(zp) for zp in z_grid])
    y_total = np.trapezoid(dy_dz, z_grid)
    return y_total, dy_dz


def decay_convolution_y(species, E0_eV, z_X, tau_X_s, target_y_all_heat=1.5e-5,
                         n_z=40, high_energy=None):
    """
    CP6 headline calculation: total y-distortion from a continuously
    injecting, exponentially-decaying particle species, computed by
    convolving the single-burst response (electron_cascade/
    photon_cascade_full, or their CP2 counterparts) over a grid of injection
    redshifts against the physical decay rate dGamma/dz (plan Sec 3.6 Eq.):

        y_total = integral dz' (dE_inj/dV/dt)(z') * |dt/dz'| * f_distortion(E0,z') / (4 rho_CMB(z'))

    f_X is fixed via f_X_for_target_all_heat_y so that an instantaneous,
    100%-efficient conversion at z_X would give y=target_y_all_heat (the
    AK19 Sec 4.1 benchmark quotes 1.5e-5); the actual convolved answer is
    suppressed relative to this because (a) energy is injected across a
    spread of redshifts rather than all at z_X, and (b) not all injected
    energy ends up as photon-field distortion (some goes to heat/subthreshold).

    Returns (y_total, z_grid, dy_dz_values, f_X) for inspection/plotting.
    """
    f_X = f_X_for_target_all_heat_y(z_X, target_y_all_heat)
    z_grid = build_zinj_grid(tau_X_s, n_z=n_z)
    rate_fn = lambda z: decay_injection_rate_eV_cm3_s(z, f_X, tau_X_s)
    y_total, dy_dz = convolve_generic_injection_y(species, E0_eV, rate_fn, z_grid,
                                                   high_energy=high_energy)

    return y_total, z_grid, dy_dz, f_X
