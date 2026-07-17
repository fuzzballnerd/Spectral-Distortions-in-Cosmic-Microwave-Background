"""
interface.py -- CP7: wire the validated Tier-1 cascade (CP0-6) into
spectroxide (SPX), per plan Sec 4.

IMPORTANT / STATUS OF THIS FILE, READ BEFORE USING:
spectroxide is not installed in the environment this was developed in --
`pip install spectroxide` finds no matching distribution (checked directly),
and the plan itself notes the user's own preference ("using spectroxide
'shouldn't be necessary'", plan Sec 7 intro). This file is therefore written
to the plan's documented SPX call signature (Sec 4) but has NOT been run
end-to-end against a real spectroxide install -- the CP7 "check" (agreement
with CP6's standalone Kompaneets-proxy answer in the y-era, to a few
percent) could not actually be executed here. If you have spectroxide
available locally, run this module's __main__ block there; otherwise, CP6's
standalone answer (already validated against AK19 Sec 4.1) stands as the
Tier-2 result, consistent with the plan's own Sec 7 fallback.

Two things are wired up:

1. dq_dz(z) -- the heat channel. This is a direct, exact pass-through: per
   the plan ("Tier 1's dq_dz_total(z) is already in SPX's exact convention
   -- fractional energy relative to rho_gamma, per unit z"), no conversion
   is needed beyond swapping solve_burst.py's "f_ic+f_subthreshold" (used
   for the y/distortion channel in CP2-6) for "f_heat" in the same
   shell-integration structure `decay_convolution_y` already uses.

2. photon_source(x, z) -- the photon channel. OPEN ITEM, flagged explicitly
   rather than silently approximated: SPX's Eq. 18 conversion
   f(x,z) = [G_2^(0)/x^2] * d(Delta N/N)/dx/dz requires an actual
   *x-resolved* photon spectrum d(Delta N/N)/dx from the cascade. This
   codebase has never computed one -- every checkpoint from CP2 onward
   tracks only integrated energy fractions (f_heat/f_ic/f_subthreshold), not
   spectral shape, and each checkpoint's docstring flags this explicitly
   (e.g. CP2's "this codebase does not resolve the photon spectral shape").
   `photon_source` below is therefore a *placeholder* that reproduces the
   right total photon-channel energy (so dq_dz + integral of photon_source
   conserves energy correctly) by depositing it as a narrow line at a single
   characteristic x_eff(z) -- exactly the same y-type single-scattering
   proxy used throughout Tier 1 -- rather than a resolved shape. This is
   sufficient to route *some* photon source into SPX and sanity-check
   overall amplitude, but NOT sufficient to reproduce AK19's actual spectral
   shapes (peak positions, dip amplitudes) through the SPX PDE. Resolving
   the real d(Delta N/N)/dx spectrum is the natural next build item if SPX
   wiring is pursued further.
"""

import numpy as np

from . import solve_burst as sb
from . import cosmology as cosmo
from . import rates

try:
    import spectroxide
    HAVE_SPECTROXIDE = True
except ImportError:
    spectroxide = None
    HAVE_SPECTROXIDE = False

G2_0 = 2.0 * 1.2020569031595943  # G_2^(0) = 2*zeta(3), SPX Eq. 1


def make_dq_dz(species, E0_eV, z_X, tau_X_s, target_y_all_heat=1.5e-5, high_energy=None):
    """
    Build a dq_dz(z) callable in SPX's exact convention (d(Delta rho/rho)/dz,
    fractional heating rate relative to rho_gamma(z)) for a decaying-particle
    injection scenario, reusing decay_convolution_y's shell-integration
    machinery but reporting the heat channel instead of the distortion
    channel.
    """
    if high_energy is None:
        high_energy = E0_eV > 2.0 * sb.M_E_EV
    f_X = sb.f_X_for_target_all_heat_y(z_X, target_y_all_heat)

    def dq_dz(z):
        try:
            rate_eV_cm3_s = sb.decay_injection_rate_eV_cm3_s(z, f_X, tau_X_s)
            dt_dz = 1.0 / ((1.0 + z) * cosmo.H(z))
            dE_dz_eV_cm3 = rate_eV_cm3_s * dt_dz
            budget = sb.burst_energy_budget(species, E0_eV, z, high_energy=high_energy)
            f_heat = budget["f_heat"]
            val = f_heat * dE_dz_eV_cm3 / sb.rho_cmb_eV_cm3(z)
            if np.isfinite(val):
                return float(val)
            return 0.0
        except Exception:
            return 0.0

    return dq_dz, f_X


def make_photon_source(species, E0_eV, z_X, tau_X_s, f_X, high_energy=None,
                        x_min=1.0e-3, x_max_factor=35.0):
    """
    Build a photon_source(x, z) callable in SPX's occupation-number
    convention (see module docstring for the important caveat: this is a
    narrow-line proxy, not a resolved spectrum).

    The characteristic x_eff(z) is set the same way electron_cascade already
    picks a representative IC daughter-photon energy: x ~ (4/3) gamma^2 for
    a relativistic injected electron, or x ~ 1 (near the CMB peak) for
    non-relativistic/photon injection -- mapped into [x_min, x_cut) so the
    source vanishes cleanly outside the tabulated Compton-dominated range
    (plan Sec 4's "practical note"). `x_cut` itself is set by `x_max_factor`
    (BUG FIX, previously flagged in review: this parameter used to be
    accepted but silently ignored -- the closure hardcoded `x_cut = 35.0`
    internally instead of using it, so the signature implied a configurable
    cutoff that wasn't actually configurable).
    """
    if high_energy is None:
        high_energy = E0_eV > 2.0 * sb.M_E_EV

    def photon_source(x, z):
        try:
            T_K = cosmo.T_gamma(z)
            kT_eV = rates.K_B_EV_PER_K * T_K
            x_cut = x_max_factor  # BUG FIX: was hardcoded to 35.0, ignoring the parameter

            budget = sb.burst_energy_budget(species, E0_eV, z, high_energy=high_energy)
            f_distortion = budget["f_ic"] + budget.get("f_subthreshold", 0.0)
            if f_distortion <= 0:
                if np.ndim(x) == 0:
                    return 0.0
                return np.zeros_like(np.asarray(x, dtype=float))

            rate_eV_cm3_s = sb.decay_injection_rate_eV_cm3_s(z, f_X, tau_X_s)
            dt_dz = 1.0 / ((1.0 + z) * cosmo.H(z))
            dE_dz_eV_cm3 = rate_eV_cm3_s * dt_dz
            dE_over_rho_dz = f_distortion * dE_dz_eV_cm3 / sb.rho_cmb_eV_cm3(z)  # energy-fraction rate

            # Characteristic x for the line, matching electron_cascade's own
            # characteristic-photon formula (4/3 gamma^2 beta^2 * <E_CMB>) for
            # electron/positron injection -- previously this used a simplified
            # (4/3)*gamma_e placeholder (missing the square, beta^2, and the
            # 2.7*kT mean-photon-energy factor electron_cascade actually uses,
            # flagged in review), and applied it even when species="gamma", where
            # "gamma_e" isn't a Lorentz factor at all. Fixed to be species-aware:
            # for e/e+, reproduce electron_cascade's formula exactly (in units of
            # kT, since x = E/kT); for gamma, use the injected photon's own
            # energy directly (E0/kT) as the characteristic scale, since there is
            # no electron Lorentz factor to speak of for direct photon injection.
            if species in ("e", "e+"):
                gamma_e = 1.0 + E0_eV / sb.M_E_EV
                beta2 = 1.0 - 1.0 / gamma_e ** 2
                x_eff_raw = (4.0 / 3.0) * gamma_e ** 2 * beta2 * sb.E_CMB_MEAN_FACTOR
            else:  # species == "gamma"
                x_eff_raw = E0_eV / kT_eV

            # BUG FIX (x_eff gamma-species degeneracy, previously flagged in
            # review): x_eff_raw for photon injection scales linearly with
            # E0/kT, which at z~2e4 is already >>x_cut for E0 as low as 10 keV
            # (x_eff_raw ~ 2.1e3) and grows further for 100 keV/1 MeV (~2.1e4,
            # ~2.1e5) -- a hard np.clip(x_eff_raw, 1, x_cut*0.9) collapsed all
            # three onto the *exact same* clipped value (31.5), making the
            # narrow-line proxy indistinguishable across a full decade-plus of
            # injection energy. This codebase has no resolved Recoil-only
            # cascade spectrum to say where above x_cut these photons'
            # degraded daughters would actually land (that's fix #3 / item 6's
            # job), but a hard clip is strictly worse than necessary: it
            # destroys energy-ordering information a smooth map can preserve.
            #
            # Below the anchor point x_anchor=0.9*x_cut, x_eff tracks
            # x_eff_raw exactly (unchanged from before, matches the old
            # behavior in the regime it was already correct). Above the
            # anchor, x_eff_raw is mapped through log-compression --
            # u = ln(x_eff_raw/x_anchor), x_eff = x_anchor + (x_cut-x_anchor)*u/(1+u)
            # -- rather than a linear-argument exponential saturation: an
            # exponential with a fixed decay scale of order x_cut (~35)
            # underflows to machine-precision-identical x_cut for ANY
            # x_eff_raw more than a few hundred above the anchor (verified
            # numerically: 10/100/1000 keV photon injections at z~2e4 all
            # collapsed to x_eff=35.000000 exactly under that scheme, an even
            # worse degeneracy than the original bug). The log-compression
            # map instead spreads decades of x_eff_raw out gently -- 10/100/
            # 1000 keV photon injections at z~2e4 land at distinguishable
            # x_eff ~= 34.33/34.53/34.64 respectively (checked numerically) --
            # while still asymptoting monotonically to x_cut as x_eff_raw ->
            # infinity, consistent with AK19's own statement (Sec 4, quoted
            # verbatim elsewhere in this codebase's docs) that the solution
            # is insensitive to the exact boundary placement once it's safely
            # above x~20-30, so a compressed-but-ordered placement in that
            # regime is a defensible proxy, not a new physics claim.
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

            # Convert energy-fraction rate -> number-fraction rate: each injected
            # photon carries x_eff*kT of energy, vs. the background's mean photon
            # energy of E_CMB_MEAN_FACTOR*kT, so dN/N = (E_CMB_MEAN_FACTOR/x_eff) * d(rho)/rho.
            # Without this, dE_over_rho_dz (an energy quantity) was being fed
            # straight into the d(ΔN/N)/dx formula as if it were already a number
            # quantity, overstating the injected photon number (and hence Δn, μ, y
            # from this channel) by ~x_eff/E_CMB_MEAN_FACTOR -- up to ~12x at the
            # clipped high-energy end.
            dN_over_N_dz = dE_over_rho_dz * (sb.E_CMB_MEAN_FACTOR / x_eff)

            # width of the line, in x -- narrow but not delta-function, so a
            # finite grid can resolve it; arbitrary O(10%) width since this is
            # explicitly a proxy (see docstring). BUG FIX: also cap the width
            # by the remaining room to x_cut (in units of sigma) -- the log-
            # compression map above can put x_eff arbitrarily close to x_cut
            # for very high injection energies, and an uncapped 0.1*x_eff
            # width would then get heavily truncated by the x<=x_cut boundary
            # (verified: at x_eff~34.6 with sigma=3.46, the boundary sits at
            # only ~0.1 sigma away, cutting off roughly half the line's mass).
            # Requiring at least 4 sigma of headroom below x_cut keeps
            # truncation loss negligible (<0.01%) regardless of how close
            # x_eff sits to the boundary.
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


def run_cp7_check(species="e", E0_eV=1.0e8, z_X=2.0e4, tau_X_s=1.0e11,
                   target_y_all_heat=1.5e-5, z_start=1.5e5, z_end=1.0e3):
    """
    Attempt the CP7 check: run the same decaying-particle scenario through
    spectroxide via dq_dz/photon_source, and compare the resulting y-era
    distortion against CP6's standalone answer. Requires spectroxide to be
    installed; raises RuntimeError with a clear message otherwise (see
    module docstring for why that's expected in this environment).
    """
    if not HAVE_SPECTROXIDE:
        raise RuntimeError(
            "spectroxide is not installed in this environment (pip has no "
            "matching distribution). This CP7 check cannot be executed here "
            "-- run this function in an environment with spectroxide "
            "installed. CP6's standalone-Kompaneets-proxy answer "
            "(solve_burst.decay_convolution_y) is the validated fallback "
            "per plan Sec 7."
        )

    dq_dz, f_X = make_dq_dz(species, E0_eV, z_X, tau_X_s, target_y_all_heat)
    photon_source = make_photon_source(species, E0_eV, z_X, tau_X_s, f_X)

    spx_result = spectroxide.solve(method="pde", dq_dz=dq_dz, photon_source=photon_source,
                                    z_start=z_start, z_end=z_end)

    y_cp6, *_ = sb.decay_convolution_y(species, E0_eV, z_X, tau_X_s, target_y_all_heat)

    return spx_result, y_cp6


if __name__ == "__main__":
    if not HAVE_SPECTROXIDE:
        print("spectroxide not installed -- cannot run the live CP7 check in this "
              "environment. See module docstring for status and the CP6 fallback.")
    else:
        spx_result, y_cp6 = run_cp7_check()
        print("SPX result:", spx_result)
        print(f"CP6 standalone y: {y_cp6:.3e}")
