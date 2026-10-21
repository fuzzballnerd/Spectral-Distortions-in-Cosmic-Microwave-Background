"""
mckeen_electron.py -- electron-injection counterpart of mckeen_bridge's chi -> n + gamma builders.

mckeen_bridge starts every cascade from a PHOTON (budget species "gamma", line placed from the pair-production
daughters).  Here the cascade starts from an ELECTRON of kinetic energy E_e (the beta-decay electron of the recoil
neutron): budget species "e" (solve_burst.electron_cascade), line placed at the electron's own gamma
(interface.py's e/e+ formula).  Everything else -- rate convention (MPR21 number ratio), budget cache,
z ceiling, G2_0, closure structure -- is taken from mckeen_bridge, which is not modified.

Both builders return closures in spectroxide's convention; like the bridge's, they return 0.0 on any exception,
so pre-flight them (see cmb_scan.preflight_electron).
"""
import numpy as np

from . import mpr21
from . import solve_burst as sb
from . import cosmology as cosmo
from . import rates
from .mckeen_bridge import _cached_budget, Z_COSMOLOGY_CEILING, G2_0


def _log_compress_x_eff(x_eff_raw, x_cut):
    """Same log-compression the bridge uses to keep the line inside the PDE grid."""
    x_anchor = 0.9 * x_cut
    if x_eff_raw <= x_anchor:
        return max(x_eff_raw, 1.0)
    u = np.log(x_eff_raw / x_anchor)
    return x_anchor + (x_cut - x_anchor) * u / (1.0 + u)


def electron_line_position(E_e_eV, x_cut=35.0):
    """(x_raw, x_eff) of the IC line for an electron of kinetic energy E_e_eV:
    x_raw = (4/3) gamma^2 beta^2 <E_CMB>/kT  (electron_cascade's characteristic photon, interface.py's e/e+ branch)."""
    gamma_e = 1.0 + E_e_eV / rates.M_E_EV
    beta2 = 1.0 - 1.0 / gamma_e ** 2
    raw = (4.0 / 3.0) * gamma_e ** 2 * beta2 * sb.E_CMB_MEAN_FACTOR
    return raw, _log_compress_x_eff(raw, x_cut)


def make_mckeen_electron_dq_dz(E_e_eV, Gamma_X_s, den_fac, E_rest_eV=0.0):
    """
    Heat channel: f_heat(E_e, z) x rate(E_e)  [+ E_rest_eV deposited as plain heat, e.g. the part of T_n + E_mean
    not carried by the electron in 'kinematic' mode; 0 in 'max' mode].

    E_e_eV : electron KINETIC energy per decay (burst_energy_budget('e', E0) takes kinetic energy and returns
             fractions of it); also the energy per decay fed to the MPR21 rate.
    """
    def dq_dz(z):
        try:
            if z > Z_COSMOLOGY_CEILING:
                return 0.0
            dt_dz = 1.0 / ((1.0 + z) * cosmo.H(z))
            f_heat = _cached_budget("e", E_e_eV, z, high_energy=True)["f_heat"]
            dE = f_heat * mpr21.mpr21_energy_injection_rate_eV_cm3_s(z, den_fac, Gamma_X_s, E_e_eV) * dt_dz
            if E_rest_eV > 0.0:
                dE += mpr21.mpr21_energy_injection_rate_eV_cm3_s(z, den_fac, Gamma_X_s, E_rest_eV) * dt_dz
            val = dE / sb.rho_cmb_eV_cm3(z)
            return float(val) if np.isfinite(val) else 0.0
        except Exception:
            return 0.0
    return dq_dz


def make_mckeen_electron_photon_source(E_e_eV, Gamma_X_s, den_fac, x_max_factor=35.0, x_min=1.0e-3):
    """
    Photon channel: (f_ic + f_subthreshold) x rate(E_e) as a narrow Gaussian line at the electron's own x_eff
    (z-independent for a fixed E_e), converted to spectroxide's occupation-number source exactly as the bridge does
    (number fraction = energy fraction x <E_CMB>/(x_eff kT), G2_0/x^2 factor).

    x_max_factor must equal the x_cut used by the caller (cmb_scan.X_CUT), as in the bridge.
    """
    x_cut = x_max_factor
    _, x_eff = electron_line_position(E_e_eV, x_cut)
    sigma_x = min(0.1 * x_eff, (x_cut - x_eff) / 4.0)

    def photon_source(x, z):
        x_arr = np.asarray(x, dtype=float)
        zero = np.zeros_like(x_arr)
        try:
            if z > Z_COSMOLOGY_CEILING:
                return float(zero) if np.ndim(x) == 0 else zero
            budget = _cached_budget("e", E_e_eV, z, high_energy=True)
            f_dist = budget["f_ic"] + budget.get("f_subthreshold", 0.0)
            if f_dist <= 0:
                return float(zero) if np.ndim(x) == 0 else zero
            dt_dz = 1.0 / ((1.0 + z) * cosmo.H(z))
            dE_over_rho_dz = (f_dist * mpr21.mpr21_energy_injection_rate_eV_cm3_s(z, den_fac, Gamma_X_s, E_e_eV)
                              * dt_dz / sb.rho_cmb_eV_cm3(z))
            dN_over_N_dz = dE_over_rho_dz * (sb.E_CMB_MEAN_FACTOR / x_eff)
            line = np.where((x_arr <= x_cut) & (x_arr >= x_min),
                            np.exp(-0.5 * ((x_arr - x_eff) / sigma_x) ** 2), 0.0)
            val = (G2_0 / x_arr ** 2) * dN_over_N_dz * line / (sigma_x * np.sqrt(2.0 * np.pi))
            result = np.where(np.isfinite(val), val, 0.0)
            return float(result) if np.ndim(x) == 0 else result
        except Exception:
            return float(zero) if np.ndim(x) == 0 else zero
    return photon_source
