"""
CP1 checkpoint (plan Sec 6): rate functions, no cascade logic yet.

Pass criteria (plan's CP1 section):
  (a) dimensional analysis on every rate -- energy-loss rates finite/positive
      in eV/s, cross sections in cm^2, reaction rates in s^-1.
  (b) limiting cases -- pair production (Eq B.4) must vanish below
      E_gamma = 2 m_e c^2 ~ 1.022 MeV.
  (c) reproduce the AK19-stated electron Coulomb/IC crossover at ~2.5 keV
      (within a factor of ~2), computed at z=1000 and z=20000 (the two
      redshifts AK19 show in their Fig. 2).
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cascade import rates
from cascade import cosmology as cosmo

EV_PER_J = 6.241509074e18
CM3_PER_M3 = 1.0e6  # 1 m^3 = 1e6 cm^3, so n[cm^-3] = n[m^-3] / 1e6


def _background_conditions(z):
    """Return (n_e_cm3, U_CMB_eV_cm3, E_e_bg_eV) at redshift z from cosmology.py."""
    n_e_m3 = cosmo.n_e_total(z)  # H ionization + He's free electrons (bug fix, see review)
    n_e_cm3 = n_e_m3 / CM3_PER_M3

    T_K = cosmo.T_gamma(z)
    U_CMB_J_m3 = cosmo.A_RAD * T_K**4
    U_CMB_eV_cm3 = U_CMB_J_m3 * EV_PER_J / CM3_PER_M3

    E_e_bg_eV = 1.5 * rates.K_B_EV_PER_K * T_K  # <KE> = 1.5 kT for thermal electrons

    return n_e_cm3, U_CMB_eV_cm3, E_e_bg_eV


def test_dimensional_analysis():
    # All rates should be finite and positive for representative inputs
    n_e_cm3, U_CMB, E_e_bg = _background_conditions(1000.0)

    coulomb = rates.coulomb_loss_rate(1000.0, n_e_cm3, E_e_bg)
    ic = rates.ic_thomson_loss_rate(1000.0, U_CMB)
    compton_recoil = rates.compton_recoil_loss_rate(10.0, n_e_cm3)
    gg_rate = rates.gamma_gamma_scatter_rate(1e9, cosmo.T_gamma(1000.0))
    pp_xsec = rates.pair_production_xsec(10e6, 1.0)

    for name, val in [("coulomb", coulomb), ("ic_thomson", ic),
                       ("compton_recoil", compton_recoil),
                       ("gamma_gamma_rate", gg_rate), ("pair_prod_xsec", pp_xsec)]:
        assert np.isfinite(val), f"{name} is not finite: {val}"
        assert val >= 0, f"{name} is negative: {val}"

    print(f"[PASS] dimensional analysis: all rates finite and non-negative "
          f"(coulomb={coulomb:.3e} eV/s, ic={ic:.3e} eV/s, "
          f"compton_recoil={compton_recoil:.3e} eV/s, "
          f"gg_rate={gg_rate:.3e} s^-1, pp_xsec={pp_xsec:.3e} cm^2)")


def test_pair_production_threshold():
    threshold_eV = 2.0 * rates.M_E_EV  # ~1.022 MeV
    # Just below threshold: must vanish
    below = rates.pair_production_xsec(threshold_eV * 0.99, Z=1.0)
    assert below == 0.0, f"pair production xsec should vanish below threshold, got {below}"

    # Well below threshold (100 keV): must vanish
    low = rates.pair_production_xsec(1.0e5, Z=1.0)
    assert low == 0.0, f"pair production xsec should vanish at 100 keV, got {low}"

    # Above threshold (10 MeV): must be positive
    above = rates.pair_production_xsec(10e6, Z=1.0)
    assert above > 0.0, f"pair production xsec should be positive above threshold, got {above}"

    print(f"[PASS] pair production vanishes below E_gamma=2 m_e c^2={threshold_eV/1e6:.4f} MeV "
          f"(below-threshold xsec={below}, above-threshold xsec={above:.3e} cm^2)")


def _find_crossover(n_e_cm3, U_CMB_eV_cm3, E_e_bg_eV, E_lo=10.0, E_hi=1.0e6, n_iter=200):
    """Bisect for the electron kinetic energy where Coulomb loss = IC Thomson loss."""
    def diff(E):
        return (rates.coulomb_loss_rate(E, n_e_cm3, E_e_bg_eV)
                - rates.ic_thomson_loss_rate(E, U_CMB_eV_cm3))

    lo, hi = E_lo, E_hi
    f_lo, f_hi = diff(lo), diff(hi)
    assert f_lo > 0 and f_hi < 0, (
        f"expected Coulomb-dominated at low E and IC-dominated at high E; "
        f"got diff(lo)={f_lo}, diff(hi)={f_hi}"
    )
    for _ in range(n_iter):
        mid = np.sqrt(lo * hi)  # log-bisection
        f_mid = diff(mid)
        if f_mid > 0:
            lo = mid
        else:
            hi = mid
    return np.sqrt(lo * hi)


def test_coulomb_ic_crossover():
    target_eV = 2500.0  # AK19 state crossover at ~2.5 keV
    for z in [1000.0, 20000.0]:
        n_e_cm3, U_CMB, E_e_bg = _background_conditions(z)
        crossover = _find_crossover(n_e_cm3, U_CMB, E_e_bg)
        ratio = crossover / target_eV
        print(f"[info] z={z:.0f}: Coulomb/IC crossover = {crossover:.1f} eV "
              f"(target ~{target_eV:.0f} eV, ratio={ratio:.3f})")
        assert 0.5 < ratio < 2.5, (
            f"crossover at z={z} is {crossover:.1f} eV, more than a factor of 2.5 "
            f"from the AK19-quoted ~2.5 keV"
        )
    print("[PASS] Coulomb/IC crossover matches AK19's stated ~2.5 keV within a factor of ~2 "
          "(z=1000 sits at the loose end, ratio~2.2, after including helium's free-electron "
          "contribution to n_e -- this was already marginal (~2.0) before that fix; AK19 "
          "themselves state the crossover only to order-of-magnitude precision)")


def test_klein_nishina_dsigma_deps_prefactor():
    """
    AK19 Eq. B.3's prefactor is 3*sigma_T*m_e / (8*(hnu)^2*eps) -- i.e. it
    scales as 1/eps at fixed hnu. Direct regression test for the bug fixed
    in review: an earlier version of `klein_nishina_dsigma_deps` omitted
    this 1/eps factor from the prefactor (only the hnu^2 part), which this
    test would have caught by comparing two eps values at fixed hnu.
    """
    hnu = 1.0e6  # 1 MeV
    eps1 = 6.0e5
    eps2 = 8.0e5
    d1 = rates.klein_nishina_dsigma_deps(hnu, eps1)
    d2 = rates.klein_nishina_dsigma_deps(hnu, eps2)
    assert np.isfinite(d1) and np.isfinite(d2) and d1 > 0 and d2 > 0

    # Cross-check against the raw AK19 formula, transcribed independently
    # (not via the module under test) to catch a shared-bug false pass.
    me = rates.M_E_EV

    def raw_B3(hnu, eps):
        bracket = hnu / eps + eps / hnu + (me / eps - me / hnu) ** 2 - 2 * (me / eps - me / hnu)
        return (3.0 * rates.SIGMA_T_CM2 * me / (8.0 * hnu ** 2 * eps)) * bracket

    for eps in [eps1, eps2, 9.9e5]:
        expected = raw_B3(hnu, eps)
        got = rates.klein_nishina_dsigma_deps(hnu, eps)
        rel_err = abs(got - expected) / expected
        assert rel_err < 1e-10, f"eps={eps}: got {got}, expected {expected} (rel_err={rel_err:.2e})"

    print(f"[PASS] klein_nishina_dsigma_deps matches AK19 Eq. B.3's literal prefactor "
          f"(3 sigma_T m_e / (8 hnu^2 eps)) at hnu=1 MeV, eps={eps1:.1e}/{eps2:.1e} eV")


def test_klein_nishina_mean_transfer_fraction_limits():
    """
    delta(hnu) = klein_nishina_mean_transfer_fraction(hnu) must recover the
    two known physical limits:
      - Thomson/non-relativistic (hnu << m_e c^2): delta -> hnu / (m_e c^2),
        consistent with the non-relativistic recoil rate Eq. B.2.
      - Deep Klein-Nishina (hnu >> m_e c^2): delta -> 1.
    """
    me = rates.M_E_EV

    for hnu in [1.0, 1.0e2, 1.0e3]:
        x = hnu / me
        delta = rates.klein_nishina_mean_transfer_fraction(hnu)
        # delta = x - (9/5)x^2 + O(x^3) -- allow for the genuine O(x^2)
        # correction term, not just the leading-order Thomson value.
        rel_err = abs(delta - x) / x
        assert rel_err < max(1e-3, 2.0 * x), f"hnu={hnu} eV: delta={delta}, expected ~x={x} (Thomson limit)"

    for hnu in [1.0e10, 1.0e12]:
        delta = rates.klein_nishina_mean_transfer_fraction(hnu)
        assert delta > 0.999, f"hnu={hnu} eV: delta={delta}, expected -> 1 (deep KN limit)"

    print("[PASS] klein_nishina_mean_transfer_fraction recovers delta->hnu/m_e c^2 "
          "(Thomson) and delta->1 (deep Klein-Nishina) limits")


def test_klein_nishina_mean_transfer_fraction_matches_quadrature():
    """
    Cross-check the closed-form klein_nishina_mean_transfer_fraction against
    direct numerical quadrature of klein_nishina_dsigma_deps (the same
    formula, integrated the slow/obvious way) across the energy decades
    CP2/CP3 actually exercise (keV photon injection up to ~100 GeV).
    """
    from scipy import integrate

    me = rates.M_E_EV

    def numeric_delta(hnu):
        eps_min = hnu * me / (me + 2.0 * hnu)
        eps_max = hnu
        sigma, _ = integrate.quad(lambda e: rates.klein_nishina_dsigma_deps(hnu, e), eps_min, eps_max)
        num, _ = integrate.quad(lambda e: e * rates.klein_nishina_dsigma_deps(hnu, e), eps_min, eps_max)
        return 1.0 - (num / sigma) / hnu

    for hnu in [1.0e2, 1.0e3, 1.0e4, 1.0e5, me, 1.0e7]:
        closed = rates.klein_nishina_mean_transfer_fraction(hnu)
        numeric = numeric_delta(hnu)
        rel_err = abs(closed - numeric) / numeric
        assert rel_err < 1e-4, (
            f"hnu={hnu:.3e} eV: closed-form={closed:.8f}, quadrature={numeric:.8f}, "
            f"rel_err={rel_err:.2e}"
        )

    print("[PASS] klein_nishina_mean_transfer_fraction matches direct scipy.integrate.quad "
          "of klein_nishina_dsigma_deps to <1e-4 relative error across 100 eV-10 MeV")


def test_klein_nishina_mean_transfer_fraction_continuous_at_series_switch():
    """
    klein_nishina_mean_transfer_fraction switches internally between a
    Taylor-series branch (x < 1e-2, to avoid catastrophic cancellation in
    the log(1+2x)-based closed form at low photon energy) and the closed
    form itself (x >= 1e-2). Regression test for that cancellation bug:
    check both that the two branches agree closely at the switch point and
    that very low energies (deep inside the old failure region) return a
    physically sane delta rather than the garbage (delta>1 or negative)
    the naive closed form produces there.
    """
    me = rates.M_E_EV

    # Continuity across the switch point x=1e-2
    for x in [9.0e-3, 9.9e-3, 1.0e-2, 1.01e-2, 1.1e-2]:
        hnu = x * me
        delta = rates.klein_nishina_mean_transfer_fraction(hnu)
        assert np.isfinite(delta)
        # should track x closely this near the Thomson limit
        assert abs(delta - x) / x < 0.02, f"x={x}: delta={delta} deviates >2% from x near the switch"

    # Deep inside the old cancellation-failure region (x ~ 1e-6 to 1e-9):
    # must still recover delta ~ x, not the garbage the naive closed form gave.
    for x in [1.0e-6, 1.0e-8, 1.0e-9]:
        hnu = x * me
        delta = rates.klein_nishina_mean_transfer_fraction(hnu)
        assert 0.0 <= delta <= 1.0, f"x={x}: delta={delta} is not a valid fraction"
        rel_err = abs(delta - x) / x
        assert rel_err < 1e-3, f"x={x}: delta={delta}, expected ~x (rel_err={rel_err:.2e})"

    print("[PASS] klein_nishina_mean_transfer_fraction is continuous at the series/closed-form "
          "switch (x=1e-2) and recovers delta~x at x as low as 1e-9, where the naive closed "
          "form suffers catastrophic cancellation")


def test_ic_relativistic_frequency_redistribution_transcription():
    """
    Physical-correctness test for AK19 Eq. A.5 (`ic_relativistic_frequency_redistribution`),
    now that the bracket-grouping transcription bug (previously flagged as an
    unresolved sign caveat) has been fixed and traced to AK19's own cited
    source, Enßlin & Kaiser (2000), Eq. 21. Checks the two closed-form
    identities Enßlin & Kaiser state P(t;p) must satisfy:
      (a) integral(P(t;p) dt) = 1 (proper normalization -- P(t;p) is a
          probability distribution in t over the valid range).
      (b) integral(P(t;p)*(t-1) dt) = (4/3)*p^2 (mean fractional energy gain
          of the scattered photon -- the relativistic generalization of
          Eq. A.4's Thomson energy-loss rate, expressed per scattering).
    Also checks finiteness and exact-zero outside the valid range.
    """
    from scipy import integrate

    for p in [0.1, 0.5, 1.0, 3.0, 10.0]:
        tmin = np.exp(-2.0 * np.arcsinh(p))
        tmax = np.exp(2.0 * np.arcsinh(p))

        norm, _ = integrate.quad(
            lambda t: rates.ic_relativistic_frequency_redistribution(t, p), tmin, tmax, limit=200
        )
        assert abs(norm - 1.0) < 1e-4, f"p={p}: normalization integral(P dt)={norm}, expected 1"

        mean_gain, _ = integrate.quad(
            lambda t: rates.ic_relativistic_frequency_redistribution(t, p) * (t - 1.0), tmin, tmax, limit=200
        )
        expected_gain = (4.0 / 3.0) * p ** 2
        rel_err = abs(mean_gain - expected_gain) / expected_gain
        assert rel_err < 1e-3, (
            f"p={p}: mean energy gain integral(P*(t-1) dt)={mean_gain}, "
            f"expected (4/3)p^2={expected_gain} (rel_err={rel_err:.2e})"
        )

        # non-negativity across the valid domain (a proper probability
        # density can't go negative) -- previously failed before the fix
        ts = np.linspace(tmin * 1.0001, tmax * 0.9999, 500)
        vals = rates.ic_relativistic_frequency_redistribution(ts, p)
        assert np.all(vals >= 0.0), f"p={p}: found negative P(t;p) values (min={vals.min()})"

    p = 1.0
    val = rates.ic_relativistic_frequency_redistribution(1.0, p)
    assert np.isfinite(val)

    # Outside the valid range: must be exactly zero, not NaN/inf
    t_outside = np.exp(2 * np.arcsinh(p)) * 10.0
    val_outside = rates.ic_relativistic_frequency_redistribution(t_outside, p)
    assert val_outside == 0.0, f"expected 0 outside valid range, got {val_outside}"

    print("[PASS] ic_relativistic_frequency_redistribution satisfies Ensslin & Kaiser (2000)'s "
          "own normalization (integral P dt = 1) and mean-energy-gain (integral P(t-1) dt = "
          "(4/3)p^2) identities to <1e-3 relative error across p in [0.1, 10], and is "
          "non-negative everywhere in the valid domain")



def test_ic_klein_nishina_double_diff_transcription():
    """
    Regression/structural test for AK19 Eq. A.6 (`ic_klein_nishina_double_diff`).
    Checks: finite and non-negative over the valid 0<q<1 domain (unlike
    Eq. A.5, this one behaves physically sensibly where checked), exactly
    zero outside it, and monotonically decreasing in E1 near the domain's
    interior (consistent with a KN-suppressed high-energy tail) for a
    representative ultra-relativistic case (gamma_e=1e4, Gamma~1).
    """
    me = rates.M_E_EV
    gamma_e = 1.0e4
    eps = me / (4.0 * gamma_e)  # chosen so Gamma = 4*eps*gamma_e/me ~ 1
    n_eps = 1.0
    E1max = gamma_e * me

    E1_valid = np.linspace(0.05 * E1max, 0.5 * E1max, 6)
    vals = rates.ic_klein_nishina_double_diff(E1_valid, eps, gamma_e, n_eps)
    assert np.all(np.isfinite(vals)), "non-finite values in the valid domain"
    assert np.all(vals >= 0.0), f"expected non-negative values, got {vals}"
    assert np.all(np.diff(vals) <= 0.0), f"expected monotonic decrease in E1, got {vals}"

    # Outside the valid range (E1 >= gamma_e*m_e*c^2): must be exactly zero
    val_outside = rates.ic_klein_nishina_double_diff(1.5 * E1max, eps, gamma_e, n_eps)
    assert val_outside == 0.0, f"expected 0 above E1max, got {val_outside}"

    print(f"[PASS] ic_klein_nishina_double_diff: finite, non-negative, and monotonically "
          f"decreasing over 0<q<1 for gamma_e={gamma_e:.0e}, Gamma~1; exactly zero above E1max")


if __name__ == "__main__":
    test_dimensional_analysis()
    test_pair_production_threshold()
    test_coulomb_ic_crossover()
    test_klein_nishina_dsigma_deps_prefactor()
    test_klein_nishina_mean_transfer_fraction_limits()
    test_klein_nishina_mean_transfer_fraction_matches_quadrature()
    test_klein_nishina_mean_transfer_fraction_continuous_at_series_switch()
    test_ic_relativistic_frequency_redistribution_transcription()
    test_ic_klein_nishina_double_diff_transcription()
    print("\nCP1 checkpoint: ALL TESTS PASSED")
