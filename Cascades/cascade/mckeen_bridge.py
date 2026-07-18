"""
mckeen_bridge.py -- MPR21 (chi -> n + gamma) bridge from the validated
Tier-1 cascade machinery into spectroxide, per
`mckeen2021_cascade_implementation_plan_v3.md`.

This module is being built incrementally, function by function, per the
plan's §7 file layout:

    1. omega_inj_and_Tn_eV(delta_m_eV)        <- this piece
    2. _cached_budget(species, omega_inj_eV, z)
    3. make_mckeen_dq_dz(omega_inj_eV, Gamma_X_s, den_fac, x_max_factor=...)
    4. make_mckeen_photon_source(omega_inj_eV, Gamma_X_s, den_fac, x_max_factor=...)

Only (1)-(2) are implemented so far.
"""

from functools import lru_cache

import numpy as np

from . import mpr21
from . import solve_burst as sb
from . import cosmology as cosmo
from . import rates

# MPR21 Eq. 7's two-body treatment stops describing the physics once
# m_chi - m_n exceeds the pi0 mass -- hadronic final states open up with
# "similar cosmological signatures" that this formula does not capture
# (plan v3 Sec 1, Sec 5; flagged as out-of-scope in the superseded
# pre-v1 plan, Sec 8).
M_PI0_MEV = 134.9768
HADRONIC_THRESHOLD_EV = M_PI0_MEV * 1.0e6  # ~1.34977e8 eV


def omega_inj_and_Tn_eV(delta_m_eV, m_n_MeV=mpr21.M_N_MEV):
    """
    Two-body chi -> n + gamma kinematics (MPR21 Eq. 7, chi at rest), given
    in energy units directly consumable by the rest of this bridge (eV),
    with the two validity guards plan v3 Sec 5 calls for:

      - delta_m_eV <= 0: kinematically forbidden (chi lighter than, or
        equal to, the neutron) -- omega_inj_MeV would otherwise silently
        return a negative number that propagates as a bogus "energy" into
        the cascade code with no error.
      - delta_m_eV >= HADRONIC_THRESHOLD_EV (~m_pi0): the two-body-only
        formula no longer describes the physics (hadronic channels open).
        This is a hard error here, not a warning, per plan Sec 5's
        guardrail table -- the two-body formula would otherwise still
        return a numerically plausible-looking answer with nothing to
        flag that the physical scenario has changed underneath it.

    Returns
    -------
    omega_inj_eV : float
        Photon injection energy in the chi rest frame (redshift-
        independent at production).
    Tn_eV : float
        Neutron recoil kinetic energy, computed as delta_m_eV - omega_inj_eV
        (i.e. total energy release minus what the photon carries away).
        Not injected into the photon bath anywhere downstream (documented
        approximation, plan v3 Sec 6 item 1) -- returned here only for
        bookkeeping/reporting (e.g. the summary-table T_n column the
        attached script already prints).
    """
    if delta_m_eV <= 0:
        raise ValueError(
            f"delta_m_eV={delta_m_eV:.6g} <= 0: chi -> n + gamma is "
            "kinematically forbidden (chi must be heavier than the neutron)."
        )
    if delta_m_eV >= HADRONIC_THRESHOLD_EV:
        raise ValueError(
            f"delta_m_eV={delta_m_eV:.6g} eV ({delta_m_eV/1.0e6:.3f} MeV) is "
            f"at or above the hadronic threshold (~m_pi0 = {M_PI0_MEV} MeV, "
            f"{HADRONIC_THRESHOLD_EV:.6g} eV): two-body chi -> n + gamma "
            "kinematics no longer applies once hadronic final states open up "
            "(MPR21's own note). This bridge does not model that regime."
        )

    m_chi_MeV = m_n_MeV + delta_m_eV / 1.0e6
    omega_inj_MeV_val = mpr21.omega_inj_MeV(m_chi_MeV, m_n_MeV=m_n_MeV)
    omega_inj_eV = omega_inj_MeV_val * 1.0e6
    Tn_eV = delta_m_eV - omega_inj_eV
    return omega_inj_eV, Tn_eV


@lru_cache(maxsize=None)
def _cached_budget(species, omega_inj_eV, z, high_energy=True):
    """
    Memoized wrapper around `solve_burst.burst_energy_budget`, keyed on
    (species, omega_inj_eV, z, high_energy).

    Mandatory per plan v3 Sec 4.6, not an optional speedup: a single
    `burst_energy_budget('gamma', ...)` call takes ~0.18-0.5 s (recursive
    photon/electron cascade walk), and `photon_source(x, z)` -- as written
    in `interface.py`'s pattern this bridge's photon builder reuses -- is
    called once per (x, z) *pair* even though the budget only depends on
    z (and the fixed omega_inj for a given Delta_m), not on x. SPX's
    tabulated-photon path samples up to `min(n_z, 500)` z-values times
    `n_x` (e.g. 1000) x-values per solve() call (Sec 4.6) -- unmemoized,
    that is up to 500,000 calls, tens of hours, for a single Delta_m point.
    Caching on (species, omega_inj_eV, z) collapses that back down to (at
    most) ~500 unique evaluations, shared between the heat and photon
    channels for the same Delta_m point since both call this helper with
    the same omega_inj_eV.

    `high_energy` is included in the cache key (rather than hardcoded)
    so this helper stays reusable outside the gamma-only, high-energy-only
    context of the current sweep, but the McKeen builders below always
    pass `high_energy=True` explicitly (plan Sec 5's guard on `omega_inj`:
    all Delta_m = 5-50 MeV points here give omega_inj in the ~5-49 MeV
    range, safely above the 2*m_e pair-production threshold, so relying on
    auto-detect is unnecessary and Sec 5 recommends being explicit instead).

    Note: `z` must be exactly the same float each call for the cache to
    hit -- true here since both dq_dz(z) and photon_source(x, z) are only
    ever invoked by spectroxide on the z-values *it* samples internally
    (the same z_grid feeds both channels' tabulation, per plan Sec 4.6),
    not on independently-rounded z's from different call sites.
    """
    return sb.burst_energy_budget(species, omega_inj_eV, z, high_energy=high_energy)


# Redshift ceiling for anything touching cascade.cosmology / solve_burst
# (plan v3 Sec 2.2, Sec 5): cosmic_time(z)'s upper integration bound is
# hardcoded to z=1e7 as its "t~0" reference point, so for z > 1e7 the
# integration bounds invert and the function silently returns a wrong-
# signed time with no exception. Sec 3 recommends operating well below
# this (z_start <~ 5e6, ~2x margin above the active window's own top for
# tau_X=1e9 s), so this bridge enforces the tighter Sec 3 operating rule
# (5e6), not just the harder Sec 2.2 domain boundary (1e7), in every
# per-z closure built below.
Z_COSMOLOGY_CEILING = 5.0e6


def make_mckeen_dq_dz(omega_inj_eV, Gamma_X_s, den_fac, x_max_factor=None,
                       force_f_heat=None):
    """
    Build the heat-channel dq_dz(z) callable for the McKeen chi -> n + gamma
    scenario, in spectroxide's exact convention (d(Delta rho/rho)/dz,
    fractional heating rate relative to rho_gamma(z)).

    This is `interface.py::make_dq_dz`'s body, with the two changes plan
    v3 Sec 4.1 calls for (not one -- doing only the first leaves dead
    AK19-convention machinery silently shadowed):

      1. The rate line is swapped from the AK19 "all-heat y-target"
         convention (`solve_burst.decay_injection_rate_eV_cm3_s`, DM
         mass-fraction f_X) to the MPR21 number-ratio convention
         (`mpr21.mpr21_energy_injection_rate_eV_cm3_s`, den_fac =
         n_{chi,0}/n_{H,0} -- a fixed physical input, not solved for).
      2. The `f_X_for_target_all_heat_y(z_X, target_y_all_heat)` call
         (and its z_X/target_y_all_heat parameters) is deleted entirely,
         not just left uncalled -- there is nothing left to solve for
         once den_fac is a direct input. `f_X_for_target_all_heat_y` and
         `mpr21.f_X_baryon_normalized` (which double-counts omega_inj/m_chi,
         plan Sec 2.1, up to ~190x suppression) are both never called from
         this module.

    `x_max_factor` is accepted (matching the photon builder's signature,
    Sec 4.4's shared-constant pattern) but unused here -- the heat channel
    has no x-dependence to cut off; kept only so callers can pass the same
    keyword set to both builders without special-casing.

    Parameters
    ----------
    omega_inj_eV : float
        Photon injection energy per decay (from omega_inj_and_Tn_eV).
    Gamma_X_s : float
        Decay rate, Gamma_X = 1/tau_X, in s^-1.
    den_fac : float
        n_{chi,0}/n_{H,0}, the MPR21 number-density-ratio convention
        (plan Sec 2.1/Sec 4.1) -- NOT a dark-matter mass fraction.
    force_f_heat : float or None
        If given, bypasses the cascade budget lookup and uses this value
        for f_heat directly (plan Sec 8 item 3's regression check: set to
        1.0, and skip the corresponding photon_source solve() call
        entirely, to compare against the attached script's 100%-heat
        approximation). None (default) uses the real budget['f_heat'].
    """

    def dq_dz(z):
        try:
            if z > Z_COSMOLOGY_CEILING:
                return 0.0
            rate_eV_cm3_s = mpr21.mpr21_energy_injection_rate_eV_cm3_s(
                z, den_fac, Gamma_X_s, omega_inj_eV
            )
            dt_dz = 1.0 / ((1.0 + z) * cosmo.H(z))
            dE_dz_eV_cm3 = rate_eV_cm3_s * dt_dz
            if force_f_heat is not None:
                f_heat = force_f_heat
            else:
                budget = _cached_budget("gamma", omega_inj_eV, z, high_energy=True)
                f_heat = budget["f_heat"]
            val = f_heat * dE_dz_eV_cm3 / sb.rho_cmb_eV_cm3(z)
            if np.isfinite(val):
                return float(val)
            return 0.0
        except Exception:
            return 0.0

    return dq_dz


# G_2^(0) = 2*zeta(3), SPX Eq. 1 (same constant interface.py uses).
G2_0 = 2.0 * 1.2020569031595943


def make_mckeen_photon_source(omega_inj_eV, Gamma_X_s, den_fac, x_max_factor=35.0,
                               x_min=1.0e-3):
    """
    Build the photon/distortion-channel photon_source(x, z) callable for
    the McKeen chi -> n + gamma scenario, in spectroxide's occupation-
    number convention.

    This is `interface.py::make_photon_source`'s body (species='gamma'
    branch only -- chi -> n + gamma is always direct photon injection, so
    the e/e+ Lorentz-factor branch that function also handles never
    applies here and is dropped), with the same two changes as
    `make_mckeen_dq_dz` above (plan Sec 4.1):

      1. Rate line swapped to `mpr21.mpr21_energy_injection_rate_eV_cm3_s`
         (MPR21 number-ratio convention, den_fac = n_{chi,0}/n_{H,0}).
      2. No `f_X_for_target_all_heat_y` solve -- den_fac is a direct
         input, nothing computed via the AK19 all-heat-y benchmark.

    Everything downstream of the rate line -- the f_ic+f_subthreshold
    split, the x_eff log-compression placement (plan Sec 4.4's
    x_max_factor/x_cut shared-constant pattern, unchanged here), the
    number-fraction conversion, the Gaussian line width/normalization, and
    the G2_0/x^2 conversion -- is convention-agnostic and carried over
    verbatim from `interface.py`, per plan Sec 4.1's closing note.

    Also carries over the Sec 2.2/Sec 5 z <= Z_COSMOLOGY_CEILING (5e6)
    guard, on top of the existing try/except Exception: return 0.0 safety
    net (which protects numerical corner cases inside the closure, not a
    bad z_start/z_max choice on the solve() call itself -- Sec 2.3).

    Parameters
    ----------
    omega_inj_eV : float
        Photon injection energy per decay.
    Gamma_X_s : float
        Decay rate, Gamma_X = 1/tau_X, in s^-1.
    den_fac : float
        n_{chi,0}/n_{H,0} (MPR21 convention, NOT a DM mass fraction).
    x_max_factor : float
        Shared cutoff -- MUST match the x_max= passed to the
        corresponding spectroxide.solve() call for this channel (plan
        Sec 4.4), or the line can land almost entirely outside the PDE's
        actual grid and its energy is lost.
    """

    def photon_source(x, z):
        try:
            if z > Z_COSMOLOGY_CEILING:
                if np.ndim(x) == 0:
                    return 0.0
                return np.zeros_like(np.asarray(x, dtype=float))

            T_K = cosmo.T_gamma(z)
            kT_eV = rates.K_B_EV_PER_K * T_K
            x_cut = x_max_factor

            budget = _cached_budget("gamma", omega_inj_eV, z, high_energy=True)
            f_distortion = budget["f_ic"] + budget.get("f_subthreshold", 0.0)
            if f_distortion <= 0:
                if np.ndim(x) == 0:
                    return 0.0
                return np.zeros_like(np.asarray(x, dtype=float))

            rate_eV_cm3_s = mpr21.mpr21_energy_injection_rate_eV_cm3_s(
                z, den_fac, Gamma_X_s, omega_inj_eV
            )
            dt_dz = 1.0 / ((1.0 + z) * cosmo.H(z))
            dE_dz_eV_cm3 = rate_eV_cm3_s * dt_dz
            dE_over_rho_dz = f_distortion * dE_dz_eV_cm3 / sb.rho_cmb_eV_cm3(z)

            # Characteristic x for the line. chi -> n + gamma is direct
            # photon injection, but at these (McKeen-scale, MeV) energies
            # the dominant loss channels feeding f_ic/f_subthreshold are
            # pair production and Compton recoil (photon_cascade_full,
            # solve_burst.py), both of which hand the energy off to
            # daughter electrons that are then run back through
            # electron_cascade -- i.e. there *is* an electron Lorentz
            # factor to speak of, one level down in the recursion.
            # Anchoring on the raw, un-cascaded omega_inj/kT therefore
            # places the line at the wrong energy scale (pins it against
            # x_cut for the whole sweep). Fix: reuse interface.py's own
            # e/e+ formula (electron_cascade's characteristic-photon
            # energy, (4/3) gamma^2 beta^2 <E_CMB>), evaluated at the
            # pair-production daughter-electron energy (omega_inj -
            # 2*m_e)/2 -- the same split solve_burst.py:359-361 uses --
            # as a first-generation proxy for where the degraded photons
            # actually land.
            E_each_eV = max(omega_inj_eV - 2.0 * rates.M_E_EV, 0.0) / 2.0
            gamma_e = 1.0 + E_each_eV / rates.M_E_EV
            beta2 = 1.0 - 1.0 / gamma_e ** 2
            x_eff_raw = (4.0 / 3.0) * gamma_e ** 2 * beta2 * sb.E_CMB_MEAN_FACTOR

            x_anchor = 0.9 * x_cut
            if np.ndim(x_eff_raw) == 0:
                if x_eff_raw <= x_anchor:
                    x_eff = max(x_eff_raw, 1.0)
                else:
                    u = np.log(x_eff_raw / x_anchor)
                    x_eff = x_anchor + (x_cut - x_anchor) * u / (1.0 + u)
            else:
                u = np.log(np.maximum(x_eff_raw, x_anchor) / x_anchor)
                x_eff = np.where(
                    x_eff_raw <= x_anchor,
                    np.maximum(x_eff_raw, 1.0),
                    x_anchor + (x_cut - x_anchor) * u / (1.0 + u),
                )

            dN_over_N_dz = dE_over_rho_dz * (sb.E_CMB_MEAN_FACTOR / x_eff)

            sigma_x = min(0.1 * x_eff, (x_cut - x_eff) / 4.0)
            x_arr = np.asarray(x, dtype=float)
            if np.any(x_arr > x_cut) or np.any(x_arr < x_min):
                line = np.where((x_arr <= x_cut) & (x_arr >= x_min),
                                 np.exp(-0.5 * ((x_arr - x_eff) / sigma_x) ** 2), 0.0)
            else:
                line = np.exp(-0.5 * ((x_arr - x_eff) / sigma_x) ** 2)
            norm = sigma_x * np.sqrt(2.0 * np.pi)
            d_deltaN_N_dx_dz = dN_over_N_dz * line / norm

            val = (G2_0 / x_arr ** 2) * d_deltaN_N_dx_dz
            result = np.where(np.isfinite(val), val, 0.0)
            if np.ndim(x) == 0:
                return float(result)
            return result
        except Exception:
            if np.ndim(x) == 0:
                return 0.0
            return np.zeros_like(np.asarray(x, dtype=float))

    return photon_source


# ── §5 guardrail rows 4, 5, 7: not enforced anywhere until now ──────────────

# Row 7: rates.pair_production_xsec's own docstring (rates.py, the
# BUG FIX note on the near-threshold bridging ramp) names MPR21's own
# Delta_m range as ~100 keV-50+ MeV and the plan's own stated omega_inj
# window as 1 keV-100 MeV -- both pass through the ~2*m_e threshold
# region, which is why that function bridges it with a smooth ramp rather
# than leaving a dead zone. high_energy=True (hardcoded in _cached_budget's
# default and every call site above) is the right solver selection only
# when omega_inj is safely above 2*m_e -- below that, high_energy=False's
# Compton+IC-only solver is what's actually valid. This check makes that
# an explicit, checkable assumption instead of a silent one.
TWO_M_E_EV = 2.0 * rates.M_E_EV  # ~1.022 MeV, pair-production threshold


def validate_omega_inj_for_high_energy(omega_inj_eV):
    """
    Row 7 guard: confirm omega_inj_eV sits safely above the pair-production
    threshold, i.e. that hardcoding high_energy=True (as every builder
    above does) is actually the right solver-branch choice for this point.

    Does not raise -- omega_inj values below 2*m_e are not a physics error
    (chi -> n + gamma is still valid kinematics there), just a solver-
    selection mismatch this bridge doesn't currently handle (it always
    requests the high-energy cascade solver). Returns True/False and lets
    the caller decide whether to warn or hard-fail, per plan Sec 5's own
    "if you extend Delta_m down near/below ~2 MeV, double-check the low-
    energy branch's own docstring before trusting it" language -- i.e.
    this is a documented boundary to watch, not (yet) a hard stop.
    """
    return omega_inj_eV > TWO_M_E_EV


# Rows 4-5: z_start operating ceiling and z_min/z_max <-> z_start/z_end
# bracket consistency. Row 4's actual hard-error/soft-warning boundary is
# enforced by the Rust binary itself (src/solver.rs, Sec 2.3) -- this
# Python-side check is a cheap pre-flight net so a bad choice surfaces
# immediately, with a clear message, before a 3600s solve() call is even
# launched, rather than after the Rust subprocess eventually errors out.
Z_START_SOFT_CEILING = 5.0e6   # plan Sec 3's recommended operating rule
Z_START_HARD_CEILING = 1.0e7   # Rust binary's own hard-error boundary (Sec 2.3)


def validate_pde_z_range(z_start, z_end, z_min, z_max):
    """
    Pre-flight check for every spectroxide.solve() call this bridge feeds
    (plan Sec 3's consistency requirement + Sec 5's z_start row), run
    Python-side before launching the (expensive, timeout=3600) Rust
    subprocess:

      - z_start > Z_START_HARD_CEILING: raises here directly, matching
        what the Rust binary would itself return as a RuntimeError
        (Sec 2.3) -- same failure, caught earlier and cheaper.
      - z_start > Z_START_SOFT_CEILING: raises ValueError too. The Rust
        binary only *soft-warns* in this band (Sec 2.3), but this bridge's
        own operating rule (Sec 3) is to stay below it with margin, so a
        violation here is treated as a configuration bug, not just logged.
      - z_max < z_start or z_min > z_end: raises -- tabulation range must
        bracket the PDE walk (Sec 3), otherwise the PDE queries dq_dz/
        photon_source outside their tabulated range (undefined/
        extrapolation behavior, not independently verified anywhere in
        this codebase -- treat as untested, per Sec 5's own table entry).
    """
    if z_start > Z_START_HARD_CEILING:
        raise ValueError(
            f"z_start={z_start:.3g} exceeds the Rust binary's own hard-error "
            f"ceiling ({Z_START_HARD_CEILING:.3g}); the Kompaneets Fokker-"
            "Planck approximation is invalid there (theta_e too large)."
        )
    if z_start > Z_START_SOFT_CEILING:
        raise ValueError(
            f"z_start={z_start:.3g} exceeds this bridge's operating ceiling "
            f"({Z_START_SOFT_CEILING:.3g}, plan Sec 3); the Rust binary would "
            "only soft-warn here, but this bridge treats it as a config error."
        )
    if z_max < z_start:
        raise ValueError(
            f"z_max={z_max:.3g} < z_start={z_start:.3g}: tabulation range "
            "does not bracket the PDE walk's start."
        )
    if z_min > z_end:
        raise ValueError(
            f"z_min={z_min:.3g} > z_end={z_end:.3g}: tabulation range does "
            "not bracket the PDE walk's end."
        )
