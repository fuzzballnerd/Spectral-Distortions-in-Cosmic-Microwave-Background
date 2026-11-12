"""
Photon lines born ABOVE x_cut (outside the PDE grid / above PIXIE's 6 THz, x = 105.6).

A photon of frequency x >> 4 in the plasma loses energy to the electrons by Compton recoil.  Without diffusion
(n << 1, T_e = T_gamma) the Kompaneets drift is
        dx/dR = -x (x - 4),        dR = sigma_T n_e c (k T_gamma / m_e c^2) dt,
which has the exact solution  x(R) = 4 / (1 - A e^{-4R}),  A = (x0 - 4)/x0   (-> 1/x = 1/x0 + R for x >> 4).
The energy the photon loses goes to the electrons and from them to the CMB: it is HEAT (mu / y).

overflow_split(x_raw, z, x_cut) does the part of the trajectory above x_cut analytically:
    x_raw <= x_cut                  -> (x_raw, 1, 0)       the line is inside the grid, the PDE does everything
    reaches x_cut before today      -> (x_cut, x_cut/x_raw, 1 - x_cut/x_raw)
                                       same photon NUMBER, re-injected at x_cut; the energy lost on the way is heat.
                                       (approximation: injected at the birth redshift z, not at the slightly later
                                        z at which it actually reaches x_cut)
    never reaches x_cut             -> (None, 0, 1 - x_today/x_raw)
                                       the lost energy is heat; the photons themselves stay above x_cut, i.e. above
                                       PIXIE's band, and are not put into the spectrum.
Thomson limit: x * kT_gamma/(m_e c^2) << 1 is assumed (see thomson_parameter()).
"""
from functools import lru_cache
import numpy as np
from . import cosmology as cosmo

SIGMA_T_M2 = 6.6524587321e-29                       # m^2
_ME_C2_J = cosmo.M_E * cosmo.C_LIGHT**2
_LNZ = np.linspace(0.0, np.log1p(3.0e7), 30001)      # ln(1+z) grid, z = 0 .. 3e7


@lru_cache(maxsize=1)
def _R_table():
    """R(z) = int_0^z sigma_T n_e c theta_gamma dz'/((1+z')H)   (cosmology.py: SI units, n_e incl. helium)."""
    z = np.expm1(_LNZ)
    theta = cosmo.K_B * cosmo.T_gamma(z) / _ME_C2_J
    f = SIGMA_T_M2 * cosmo.n_e_total(z) * cosmo.C_LIGHT * theta / cosmo.H(z)        # dR / d ln(1+z)
    return np.concatenate([[0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(_LNZ))])


def R_of_z(z):
    """Recoil 'optical depth' accumulated from redshift z down to today."""
    return float(np.interp(np.log1p(z), _LNZ, _R_table()))


def x_after(x0, R):
    """Frequency after a recoil depth R (exact solution of dx/dR = -x(x-4))."""
    if x0 <= 4.0:
        return float(x0)
    return float(4.0 / (1.0 - (x0 - 4.0) / x0 * np.exp(-4.0 * R)))


def thomson_parameter(x, z):
    """x kT_gamma / m_e c^2: the fractional energy loss per scattering.  << 1 for the drift picture to hold."""
    return float(x * cosmo.K_B * cosmo.T_gamma(z) / _ME_C2_J)


def overflow_split(x_raw, z, x_cut):
    """(x_inject or None, f_phot, f_heat): fractions of the line's ENERGY."""
    if x_raw <= x_cut:
        return float(x_raw), 1.0, 0.0
    x_today = x_after(x_raw, R_of_z(z))
    if x_today < x_cut:
        return float(x_cut), x_cut / x_raw, 1.0 - x_cut / x_raw
    return None, 0.0, 1.0 - x_today / x_raw
