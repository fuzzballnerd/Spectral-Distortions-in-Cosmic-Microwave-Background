"""
rates.py -- collision rates and cross sections for the EM cascade code (CP1).

Every formula here is transcribed directly from Acharya & Khatri 2019
(arXiv:1808.02897, "AK19"), Appendices A (electron/positron) and B (photon).
Equation numbers in each docstring refer to AK19.

Units convention in this module (matching AK19's own presentation): energies
in eV, number densities in cm^-3, rates in eV/s or s^-1, lengths in cm, time
in s. This is a deliberate departure from cosmology.py's SI convention, to
keep every formula a literal transcription of the paper (minimizing
transcription-error risk) -- conversions happen at the call site.
"""

import numpy as np

# ----------------------------------------------------------------------
# Physical constants (eV / cm / s units)
# ----------------------------------------------------------------------
M_E_EV = 510998.95          # electron rest mass energy, eV
ALPHA = 7.2973525693e-3     # fine structure constant
R0_CM = 2.8179403227e-13    # classical electron radius, cm
SIGMA_T_CM2 = 6.6524587321e-25   # Thomson cross section, cm^2
C_CM_S = 2.99792458e10      # speed of light, cm/s
K_B_EV_PER_K = 8.617333262e-5    # Boltzmann constant, eV/K


# ----------------------------------------------------------------------
# A.1 -- Coulomb scattering (Swartz-Nisbet-Green fit)
# ----------------------------------------------------------------------
def coulomb_loss_rate(E_eV, n_e_cm3, E_e_bg_eV):
    """
    AK19 Eq. A.1: -dE/dt for an incident electron of kinetic energy E_eV (eV)
    losing energy to Coulomb scattering off background electrons of density
    n_e_cm3 (cm^-3) and characteristic background electron energy E_e_bg_eV (eV).
    Returns eV/s (positive = loss rate).
    """
    E = np.asarray(E_eV, dtype=float)
    Ee = np.asarray(E_e_bg_eV, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(E > Ee, (E - Ee) / (E - 0.53 * Ee), 0.0)
        result = np.where(E > Ee, 2.0e-4 * (n_e_cm3 ** 0.97) / (E ** 0.44) * (np.maximum(ratio, 0.0) ** 2.36), 0.0)
    if np.ndim(E_eV) == 0 and np.ndim(n_e_cm3) == 0 and np.ndim(E_e_bg_eV) == 0:
        return float(result)
    return result


# ----------------------------------------------------------------------
# A.4 -- Inverse Compton scattering, Thomson limit
# ----------------------------------------------------------------------
def ic_thomson_loss_rate(E_kin_eV, U_CMB_eV_cm3):
    """
    AK19 Eq. A.4: dE/dt = (4/3) sigma_T c U_CMB gamma_e^2 beta_e^2.
    E_kin_eV: electron kinetic energy (eV). U_CMB_eV_cm3: CMB energy density
    (eV/cm^3). Returns eV/s.
    """
    E = np.asarray(E_kin_eV, dtype=float)
    gamma_e = 1.0 + E / M_E_EV
    beta_e2 = 1.0 - 1.0 / gamma_e**2
    return (4.0 / 3.0) * SIGMA_T_CM2 * C_CM_S * U_CMB_eV_cm3 * gamma_e**2 * beta_e2


# ----------------------------------------------------------------------
# A.2/A.3 -- e+e- annihilation cross sections (catalogued; not exercised by
# CP1's numeric checks, which focus on the Coulomb/IC crossover and pair
# production threshold, but included for completeness / later checkpoints).
# ----------------------------------------------------------------------
def annihilation_xsec_nonrel(beta):
    """AK19 Eq. A.2: non-relativistic (Coulomb-corrected) annihilation cross section, cm^2."""
    beta = np.asarray(beta, dtype=float)
    x = 2 * np.pi * ALPHA / beta
    return np.pi * R0_CM**2 * x / (1 - np.exp(-x))


def annihilation_xsec_rel(gamma):
    """AK19 Eq. A.3: relativistic (plane-wave) annihilation cross section, cm^2."""
    gamma = np.asarray(gamma, dtype=float)
    g2m1 = gamma**2 - 1
    return (np.pi * R0_CM**2 / (gamma + 1)) * (
        (gamma**2 + 4 * gamma + 1) / g2m1 * np.log(gamma + np.sqrt(g2m1))
        - (gamma + 3) / np.sqrt(g2m1)
    )


# ----------------------------------------------------------------------
# B.2 -- Compton scattering, non-relativistic recoil energy-loss rate
# ----------------------------------------------------------------------
def compton_recoil_loss_rate(hnu_eV, n_e_cm3):
    """
    AK19 Eq. B.2: dE/dt = n_e sigma_T c (h nu)^2 / m_e.
    hnu_eV: photon energy (eV). n_e_cm3: electron density (cm^-3). Returns eV/s.
    """
    hnu = np.asarray(hnu_eV, dtype=float)
    return n_e_cm3 * SIGMA_T_CM2 * C_CM_S * hnu**2 / M_E_EV


# ----------------------------------------------------------------------
# B.3 -- Klein-Nishina differential cross section (catalogued)
# ----------------------------------------------------------------------
def klein_nishina_dsigma_deps(hnu_eV, eps_eV):
    """
    AK19 Eq. B.3: differential Klein-Nishina cross section d(sigma)/d(eps),
    valid for m_e/(m_e+2 hnu) * hnu <= eps <= hnu. Returns cm^2/eV.

    dsigma/deps = [3 sigma_T m_e / (8 (hnu)^2 eps)] *
                  [hnu/eps + eps/hnu + (m_e/eps - m_e/hnu)^2 - 2(m_e/eps - m_e/hnu)]

    BUG FIX (previously flagged in review): the prefactor is 3*sigma_T*m_e /
    (8*(hnu)^2*eps) -- i.e. it has a 1/eps factor in addition to the 1/hnu^2.
    An earlier version of this function omitted the 1/eps factor from the
    prefactor (computing 3*sigma_T*m_e/(8*hnu^2) instead), which is a plain
    transcription error against AK19's own denominator 8*(hnu)^2*eps. The
    bracket term was already correct; only the prefactor needed the fix.
    """
    hnu = np.asarray(hnu_eV, dtype=float)
    eps = np.asarray(eps_eV, dtype=float)
    me = M_E_EV
    return (3.0 * SIGMA_T_CM2 / 8.0) * (me / (hnu**2 * eps)) * (
        hnu / eps + eps / hnu + (me / eps - me / hnu) ** 2 - 2 * (me / eps - me / hnu)
    )


def klein_nishina_mean_transfer_fraction(hnu_eV):
    """
    Exact Klein-Nishina mean fractional energy transfer to the recoil
    electron, delta(hnu) = <hnu - eps>/hnu, obtained by integrating the
    literal AK19 Eq. B.3 differential cross section (`klein_nishina_dsigma_deps`)
    over the allowed scattered-photon energy range
    m_e/(m_e+2 hnu) * hnu <= eps <= hnu:

        <eps> = [ integral eps * dsigma/deps deps ] / [ integral dsigma/deps deps ]
        delta = 1 - <eps>/hnu

    This replaces the old `x/(1+x)` phenomenological interpolation
    (`_compton_mean_transfer_fraction` in solve_burst.py) with the paper's
    own recipe, per the amended plan's fix #2.

    Implementation note: `dsigma/deps` is a rational function of eps (once
    hnu is fixed). Writing x = hnu/m_e c^2 and non-dimensionalizing
    eps = u*hnu (u in [1/(1+2x), 1]), both integral(dsigma/deps) and
    integral(eps*dsigma/deps) reduce to elementary closed forms in x alone
    (rational terms plus a single log(1+2x) term) -- this is used instead of
    numerical quadrature purely for speed: this function is called inside
    `photon_cascade`'s per-scatter loop (up to ~1e5 times per photon), so a
    `scipy.integrate.quad` call per evaluation would be prohibitively slow.

    The naive version of this closed form (evaluating the antiderivatives
    directly in eps at eps_min and eps_max and subtracting) suffers
    catastrophic cancellation for x << 1, because eps_min and eps_max become
    nearly equal there (eps_min/eps_max = 1/(1+2x) -> 1) -- it was checked
    against `scipy.integrate.quad` directly on `klein_nishina_dsigma_deps`
    and silently returns garbage (e.g. delta > 1, or negative) below
    x ~ 1e-3 in double precision, well inside the energy range CP2/CP3
    actually use (photon injection down to ~keV, i.e. x ~ 1e-8). The
    dimensionless log(1+2x)-based form used below is far more stable but
    still loses precision as x -> 0 (log(1+2x) -> 2x is itself a
    cancellation), so for x below `_X_SERIES_SWITCH` a direct Taylor series
    in x is used instead. Both branches were checked against a 50-digit
    mpmath evaluation of the same log-based expression and against
    `scipy.integrate.quad` in the regime where quad itself stays accurate;
    they agree to <1e-8 relative error at and near the switch point (see
    `tests/test_cp1_rates.py`).

    Returns delta (dimensionless), same shape as hnu_eV. Correctly recovers
    the two known limits: delta -> hnu/m_e c^2 as hnu -> 0 (matching the
    non-relativistic recoil rate, Eq. B.2) and delta -> 1 as hnu -> infinity
    (deep Klein-Nishina saturation).
    """
    h = np.asarray(hnu_eV, dtype=float)
    m = M_E_EV
    x = h / m
    scalar_input = (x.ndim == 0)
    x = np.atleast_1d(x)

    _X_SERIES_SWITCH = 1.0e-2

    # -- Branch 1: log-based closed form (accurate for x >= _X_SERIES_SWITCH) --
    with np.errstate(divide="ignore", invalid="ignore"):
        log1p2x = np.log1p(2.0 * x)
        sigma_rel = (4 * x**4 - 6 * x**2 - 2 * x + (4 * x**2 + 4 * x + 1) * log1p2x) / (x**2 * (2 * x + 1))
        mean_rel = (log1p2x + 0.5 - 1.0 / (2.0 * (2 * x + 1) ** 2)
                    - 2.0 * log1p2x / x + 4.0 / x - 2.0 * log1p2x / x**2)
        delta_closed = 1.0 - mean_rel / sigma_rel

    # -- Branch 2: Taylor series about x=0 (accurate for x < _X_SERIES_SWITCH) --
    # delta(x) = x - (9/5)x^2 + (39/10)x^3 - (2943/350)x^4 + (1221/70)x^5 + O(x^6)
    delta_series = x - (9.0 / 5.0) * x**2 + (39.0 / 10.0) * x**3 \
        - (2943.0 / 350.0) * x**4 + (1221.0 / 70.0) * x**5

    delta = np.where(x < _X_SERIES_SWITCH, delta_series, delta_closed)
    delta = np.clip(delta, 0.0, 1.0)

    return float(delta[0]) if scalar_input else delta


# ----------------------------------------------------------------------
# A.5 -- Inverse Compton spectrum of a monoenergetic relativistic electron
# off the full CMB (exact frequency-redistribution function)
# ----------------------------------------------------------------------
def ic_relativistic_frequency_redistribution(t, p):
    """
    AK19 Eq. A.5: the frequency redistribution function P(t;p) for a single
    inverse-Compton scattering of a monoenergetic relativistic electron
    (Lorentz factor gamma_e, p = beta_e*gamma_e) with the CMB, giving the
    spectral shape of the scattered photon frequency nu' relative to the
    unscattered (incident CMB) photon frequency nu, t = nu'/nu:

        P(t;p) = -3|1-t| / (32 p^6 t) * [1 + (10+8p^2+4p^4) t + t^2]
                 + 3(1+t) / (8 p^5) * [ (3+3p^2+p^4)/sqrt(1+p^2)
                                         - (3+2p^2)/(2p) * (2 arcsinh(p) - |ln t|) ]

    valid for |ln t| < 2*arcsinh(p) (zero outside that range -- this is the
    kinematic bound on the frequency shift for a single scatter off an
    electron of momentum p).

    BUG FIX (previously flagged in review as an unresolved sign caveat, now
    resolved): an earlier version of this transcription grouped the second
    bracket's terms as [A - B] * (2 arcsinh(p) - |ln t|), applying the
    (2 arcsinh(p) - |ln t|) factor to the *entire* bracket. That was wrong --
    it only multiplies the second term, B = (3+2p^2)/(2p), i.e. the correct
    grouping is A - B*(2 arcsinh(p) - |ln t|). This was tracked down by
    fetching AK19's own cited source for this formula (ref [47]: Enßlin &
    Kaiser 2000, A&A 360, 417, arXiv:astro-ph/0001429, their Eq. 21 -- AK19's
    Eq. A.5 is a direct transcription of it) and checking the mis-grouped
    version against the two closed-form identities Enßlin & Kaiser state
    P(t;p) itself must satisfy: integral(P(t;p) dt) = 1 (proper
    normalization) and integral(P(t;p)*(t-1) dt) = (4/3)*p^2 (mean fractional
    energy gain of the scattered photon, the relativistic generalization of
    Eq. A.4's Thomson energy-loss rate). The mis-grouped version satisfied
    neither (and was sign-indeterminate depending on p); the correctly
    grouped version below reproduces both identities to floating-point
    precision for p in [0.01, 10] (checked numerically, not just at one
    point) -- see `tests/test_cp1_rates.py`.

    This is the redistribution function for scattering off a *fixed*
    incident photon frequency nu -- AK19 state it is used "with the full CMB
    spectrum" (Fig. 3's caption), meaning the actual electron -> photon
    spectrum requires convolving P(t;p) against the Planck spectrum n_CMB(nu)
    over the incident frequency nu, weighted appropriately. That convolution
    (and its normalization) is NOT implemented here -- this function only
    provides the literal Eq. A.5 kernel, now verified correct in isolation.
    Wiring it into an actual Fig. 3 reproduction is the re-scoped task,
    amended-priority item 6.

    Parameters
    ----------
    t : frequency ratio nu'/nu (dimensionless), scalar or array.
    p : beta_e * gamma_e (dimensionless electron momentum in units of m_e c),
        scalar or array (broadcastable against t).

    Returns
    -------
    P(t;p), same broadcast shape as t/p. Zero outside the kinematically
    allowed range |ln t| < 2*arcsinh(p). Non-negative and normalized to 1
    when integrated over dt across the full valid range (verified).
    """
    t = np.asarray(t, dtype=float)
    p = np.asarray(p, dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        term1 = (-3.0 * np.abs(1.0 - t) / (32.0 * p ** 6 * t)) * (
            1.0 + (10.0 + 8.0 * p ** 2 + 4.0 * p ** 4) * t + t ** 2
        )
        A = (3.0 + 3.0 * p ** 2 + p ** 4) / np.sqrt(1.0 + p ** 2)
        B = (3.0 + 2.0 * p ** 2) / (2.0 * p)
        two_arcsinh_p = 2.0 * np.arcsinh(p)
        log_t = np.log(t)
        bracket = A - B * (two_arcsinh_p - np.abs(log_t))
        term2 = (3.0 * (1.0 + t) / (8.0 * p ** 5)) * bracket
        result = term1 + term2

    valid = np.abs(log_t) < two_arcsinh_p
    return np.where(valid, result, 0.0)


# ----------------------------------------------------------------------
# A.6 -- Double-differential Klein-Nishina spectrum, ultra-relativistic
# electron (needed once electron energy reaches ~GeV, per AK19 Sec A.4)
# ----------------------------------------------------------------------
def ic_klein_nishina_double_diff(E1_eV, eps_eV, gamma_e, n_eps_per_eV):
    """
    AK19 Eq. A.6: the double-differential inverse-Compton scattering rate
    per unit scattered-photon energy E1, for an ultra-relativistic electron
    (Lorentz factor gamma_e) scattering off a photon field of unscattered
    energy eps with number density (per unit energy) n(eps):

        d^2N/(dt dE1) = [2 pi r0^2 m_e c^3 n(eps) deps] / (gamma_e eps)
                        * [2q ln(q) + (1+2q)(1-q) + 0.5 (Gamma q)^2 (1-q)/(1+Gamma q)]

        Gamma = 4*eps*gamma_e / (m_e c^2)
        q = E1 / [Gamma * (gamma_e*m_e*c^2 - E1)]

    Transcribed directly from AK19 via PDF page rasterization (same
    garbling issue as Eq. A.5). Structurally this is the standard
    Blumenthal & Gould (1970) extreme-relativistic Compton formula (their
    Eq. 2.48), which AK19 cite as ref [48] for this equation.

    OPEN ITEM: AK19's prefactor as printed is "2 pi r0^2 m_e c^3 n(eps) deps
    / (gamma_e eps)". This function passes through n_eps_per_eV (n(eps), a
    number density *per unit photon energy*, so the deps factor is already
    absorbed into it by the caller) and literally multiplies by
    m_e_eV * C_CM_S rather than m_e*c^3 with an independently-verified unit
    conversion -- the overall normalization has NOT been checked against a
    known total rate (e.g. integrating d^2N/(dt dE1) over E1 should recover
    a rate consistent with `ic_thomson_loss_rate`'s Thomson-limit rate at
    low gamma_e, which has not been verified here). This is flagged
    explicitly as unverified rather than asserted correct -- do not treat
    the numeric normalization as trustworthy until that cross-check is
    done, as part of re-scoped item 6 (Fig. 3 as its own task).

    Parameters
    ----------
    E1_eV : scattered photon energy (eV).
    eps_eV : unscattered (incident) photon energy (eV).
    gamma_e : electron Lorentz factor (dimensionless).
    n_eps_per_eV : incident photon number density per unit energy at eps_eV
        (i.e. n(eps)*deps folded in already by the caller -- pass the
        appropriately normalized differential number density), cm^-3 eV^-1.

    Returns
    -------
    d^2N/(dt dE1), in the same (unverified) units as described above.
    Zero (not NaN) outside the kinematically valid range
    1/(4*gamma_e^2) <= q < 1 (E1 < gamma_e*m_e*c^2) implied by AK19's own q
    definition -- the standard Blumenthal & Gould kinematic floor, matching
    `ic_spectrum.klein_nishina_double_diff`'s domain. (An earlier version of
    this function only required q > 0, which let it return nonzero values
    in the kinematically-forbidden region 0 < q < 1/(4*gamma_e^2) where the
    true rate is zero.)
    """
    E1 = np.asarray(E1_eV, dtype=float)
    eps = np.asarray(eps_eV, dtype=float)
    ge = np.asarray(gamma_e, dtype=float)
    n_eps = np.asarray(n_eps_per_eV, dtype=float)
    me = M_E_EV

    Gamma = 4.0 * eps * ge / me
    denom = Gamma * (ge * me - E1)

    with np.errstate(divide="ignore", invalid="ignore"):
        q = E1 / denom
        bracket = 2.0 * q * np.log(q) + (1.0 + 2.0 * q) * (1.0 - q) + \
            0.5 * (Gamma * q) ** 2 * (1.0 - q) / (1.0 + Gamma * q)
        prefactor = 2.0 * np.pi * R0_CM ** 2 * (C_CM_S * me) * n_eps / (ge * eps)
        result = prefactor * bracket

    valid = (q >= 1.0 / (4.0 * ge ** 2)) & (q < 1.0) & (E1 < ge * me)
    return np.where(valid, result, 0.0)


# ----------------------------------------------------------------------
# B.4 -- Pair production on matter (Bethe-Heitler, high-energy log form)
# ----------------------------------------------------------------------
def pair_production_xsec(Egamma_eV, Z):
    """
    AK19 Eq. B.4: sigma = alpha r0^2 Z^2 [ (28/9) ln(2 Egamma) - 218/27 ],
    with Egamma in units of m_e c^2 in the original formula.

    This is the standard high-energy "complete screening" asymptotic
    Bethe-Heitler form quoted in AK19 -- it is NOT itself valid near
    threshold: taken literally, the bare log expression is negative from
    E_gamma = 2*m_e*c^2 (the physical threshold) up to E_gamma ~ 6.7*m_e*c^2
    ~ 3.42 MeV (where the log expression's own zero crossing sits), even
    though pair production is of course physically allowed as soon as
    E_gamma > 2*m_e*c^2.

    BUG FIX (previously flagged in review): an earlier version of this
    function clamped the negative region to exactly zero via
    np.maximum(raw, 0.0), which does correctly enforce the physical
    threshold at 2*m_e*c^2, but ALSO silently zeroes out the ~2.4 MeV-wide
    window from threshold to ~3.42 MeV where the cross section should be
    small but positive and rising -- not exactly zero. This matters in
    practice: MPR21's own Delta_m range (~100 keV-50+ MeV) and the plan's
    own stated omega_inj window (1 keV-100 MeV, Sec 1) both pass directly
    through this gap.

    FIX: since AK19 do not give (and this codebase does not derive from
    scratch) an exact near-threshold Bethe-Heitler cross section, the region
    from threshold to X_MATCH*m_e*c^2 (chosen safely above the log-formula's
    own zero crossing, where the asymptotic form is unambiguously valid) is
    bridged with a smooth (C2-continuous, monotonically increasing)
    "smootherstep" ramp from 0 at threshold to the exact AK19 Eq. B.4 value
    at X_MATCH. This is an explicit approximation, not a first-principles
    near-threshold derivation -- it removes the unphysical dead zone and the
    equally unphysical discontinuous "abrupt turn-on" at the old formula's
    zero crossing, while leaving the AK19 formula completely unchanged
    everywhere it's actually being used as intended (E_gamma >~ 5 MeV).
    Replacing this with a literal near-threshold Born-approximation Bethe-
    Heitler formula is flagged as a follow-up open item if more precision is
    needed close to threshold specifically.

    Returns cm^2 (0 at/below the physical threshold E_gamma = 2*m_e*c^2).
    """
    Eg_eV = np.asarray(Egamma_eV, dtype=float)
    x = Eg_eV / M_E_EV  # Egamma in units of m_e c^2, as the formula expects

    def raw_form(xx):
        return ALPHA * R0_CM ** 2 * Z ** 2 * ((28.0 / 9.0) * np.log(2 * xx) - 218.0 / 27.0)

    X_THRESHOLD = 2.0
    X_MATCH = 10.0  # safely above the log-formula's own zero crossing (~6.7)
    sigma_match = raw_form(X_MATCH)

    t = np.clip((x - X_THRESHOLD) / (X_MATCH - X_THRESHOLD), 0.0, 1.0)
    smootherstep = t ** 3 * (t * (t * 6.0 - 15.0) + 10.0)  # C2-continuous, 0->1 monotonic

    sigma_ramp = smootherstep * sigma_match       # used for threshold <= E < X_MATCH*m_e*c^2
    sigma_asymptotic = raw_form(x)                # used for E >= X_MATCH*m_e*c^2 (unchanged AK19 formula)

    sigma = np.where(x >= X_MATCH, sigma_asymptotic, sigma_ramp)
    return np.where(x >= X_THRESHOLD, np.maximum(sigma, 0.0), 0.0)


# ----------------------------------------------------------------------
# B.6 -- Photon-photon elastic scattering rate on the CMB
# ----------------------------------------------------------------------
def gamma_gamma_scatter_rate(Egamma_eV, T_CMB_K):
    """
    AK19 Eq. B.6: p(Egamma) = 3.33e11 * (T_CMB/m_e)^6 * (Egamma/m_e)^3  [s^-1],
    with T_CMB and Egamma both in units of m_e c^2 (T_CMB here means k_B*T_CMB).
    Returns s^-1.
    """
    Eg = np.asarray(Egamma_eV, dtype=float)
    kT_eV = K_B_EV_PER_K * T_CMB_K
    return 3.33e11 * (kT_eV / M_E_EV) ** 6 * (Eg / M_E_EV) ** 3


if __name__ == "__main__":
    # Quick smoke tests (dimensional sanity), full checkpoint tests live in
    # tests/checkpoints/test_cp1_rates.py
    print("Coulomb loss rate at E=1 keV, n_e=1 cm^-3, E_e_bg=1 eV:",
          coulomb_loss_rate(1000.0, 1.0, 1.0), "eV/s")
    print("IC Thomson loss rate at E=1 keV, U_CMB=0.26 eV/cm^3:",
          ic_thomson_loss_rate(1000.0, 0.26), "eV/s")
    print("Compton recoil loss rate at hnu=1 eV, n_e=1 cm^-3:",
          compton_recoil_loss_rate(1.0, 1.0), "eV/s")
    print("Pair production xsec at 0.5 MeV (below threshold):",
          pair_production_xsec(0.5e6, 1.0), "cm^2 (expect 0)")
    print("Pair production xsec at 10 MeV:",
          pair_production_xsec(10e6, 1.0), "cm^2")
    print("gamma-gamma scatter rate at Egamma=1 GeV, T=2.726K:",
          gamma_gamma_scatter_rate(1e9, 2.726), "s^-1")
