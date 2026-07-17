"""
mpr21.py -- CP8: plug in the actual chi -> n gamma kinematics from
McKeen, Pospelov & Raj 2021 (MPR21, Phys. Rev. D 103, 115002), "Cosmological
and astrophysical probes of dark baryons," and sanity-check the resulting
spectral-distortion amplitude against the regime where MPR21's own
CMB-anisotropy/COBE constraints bite.

Formulas used (transcribed directly from the paper, not the plan's
paraphrase, to avoid propagating a transcription error twice):

  - Photon injection energy (MPR21 Eq. 7, the m_chi > m_n branch):
        omega_inj = (m_chi/2) * (1 - m_n^2/m_chi^2)
    (redshift-independent at production -- two-body kinematics.)

  - Decay rate (MPR21 Eq. 7, numerical form given in-text):
        Gamma_{chi->n gamma} ~= (1/220 s) * (theta/1e-10)^2 * (Delta_m/10 MeV)^3
    where Delta_m = m_chi - m_n.

  - Injection rate density (plan Sec 1, matching MPR21/SPX's "Decaying
    particle" convention -- note this is n_H(z)-normalized, NOT
    n_DM(z)-normalized like CP6's dark-matter-decay benchmark):
        dE/(dt dV) = f_X * Gamma_X * n_H(z) * exp(-Gamma_X * t)
        f_X = epsilon_X * m_X * n_{X,0} / n_{H,0}
    with epsilon_X = omega_inj / m_chi (all injected energy is
    electromagnetic -- the neutron recoil is EM-inert to leading order, so
    this is treated as pure single-photon injection, plan Sec 1's explicit
    simplification).

  - MPR21's own quoted CMB-anisotropy sensitivity floor (Eq. 19):
        f_chi / tau_chi <~ 1e-25 s^-1
    where f_chi is "the ratio of the chi energy density to that measured in
    DM" -- i.e. f_chi plays the role of CP6's DM-fraction f_X, NOT this
    module's baryon-normalized f_X above (MPR21 use f_X for the BBN/CMB
    baryon-density bookkeeping and f_chi separately for the DM-fraction
    language in Eq. 19 -- both symbols are reproduced here exactly as MPR21
    use them, despite the unfortunate collision, to keep the numbers
    traceable back to the paper).
"""

import numpy as np

from . import solve_burst as sb
from . import cosmology as cosmo

M_N_MEV = 939.565  # neutron mass, MeV
M_P_MEV = 938.272  # proton mass, MeV


def omega_inj_MeV(m_chi_MeV, m_n_MeV=M_N_MEV):
    """Photon injection energy in the chi rest frame (MPR21 Eq. 7, m_chi > m_n branch)."""
    return (m_chi_MeV / 2.0) * (1.0 - (m_n_MeV / m_chi_MeV) ** 2)


def gamma_chi_to_ngamma(theta, delta_m_MeV):
    """
    chi -> n gamma decay rate (MPR21 Eq. 7 numerical form), in s^-1.
    delta_m_MeV = m_chi - m_n, in MeV.
    """
    return (1.0 / 220.0) * (theta / 1.0e-10) ** 2 * (delta_m_MeV / 10.0) ** 3


def f_X_baryon_normalized(omega_inj_MeV_val, m_chi_MeV, n_chi0_over_nH0):
    """
    f_X = epsilon_X * m_X * n_{X,0}/n_{H,0} (plan Sec 1), with
    epsilon_X = omega_inj/m_chi. Note m_X cancels against the 1/m_chi in
    epsilon_X, leaving f_X = omega_inj * n_chi0_over_nH0 / m_chi -- i.e.
    f_X is just the injected-energy-weighted number-density ratio. Written
    out in full (not pre-cancelled) so the formula stays checkable against
    the plan's stated form term-by-term.
    """
    epsilon_X = omega_inj_MeV_val / m_chi_MeV
    return epsilon_X * m_chi_MeV * n_chi0_over_nH0 / m_chi_MeV


def mpr21_energy_injection_rate_eV_cm3_s(z, f_X, Gamma_X_s, omega_inj_eV):
    """
    Physical energy-injection rate density at redshift z, in eV cm^-3 s^-1:

        dE/(dt dV) = omega_inj * f_X * Gamma_X * n_H(z) * exp(-Gamma_X * t(z))

    (f_X * Gamma_X * n_H(z) * exp(-Gamma_X t) is the *number* of decays per
    unit time per unit volume; multiplying by omega_inj converts this to an
    energy rate, since each decay releases one photon of energy omega_inj.)
    """
    n_H_cm3 = cosmo.n_H(z) / 1.0e6
    t_s = cosmo.cosmic_time(z)
    decays_per_cm3_per_s = f_X * Gamma_X_s * n_H_cm3 * np.exp(-Gamma_X_s * t_s)
    return omega_inj_eV * decays_per_cm3_per_s


def run_benchmark(m_chi_MeV=None, delta_m_MeV=10.0, theta=None, tau_chi_s=1.0e11,
                   n_chi0_over_nH0=5.4, n_z=40):
    """
    CP8 headline calculation: pick a (m_chi, theta, Delta_m) benchmark point
    drawn from MPR21's own explored parameter space (default: their
    Delta_m=10 MeV BBN benchmark, n_chi0=5.4*n_baryon0 "chi is all of DM"
    scenario), compute the resulting spectral-distortion amplitude via the
    same Tier-1 machinery validated in CP0-6, and report it alongside
    MPR21's own CMB-sensitivity language for an order-of-magnitude sanity
    check (plan's CP8 pass criterion -- no formal numeric target).

    If theta is not given, it is solved so that the decay lifetime equals
    tau_chi_s (default 1e11 s, matching CP6's own benchmark redshift window
    z ~ 2e4, so the result is directly comparable to CP6's answer).
    """
    if m_chi_MeV is None:
        m_chi_MeV = M_N_MEV + delta_m_MeV

    omega_inj = omega_inj_MeV(m_chi_MeV)  # MeV

    if theta is None:
        # solve gamma_chi_to_ngamma(theta, delta_m_MeV) = 1/tau_chi_s
        target_Gamma = 1.0 / tau_chi_s
        theta = 1.0e-10 * np.sqrt(target_Gamma / gamma_chi_to_ngamma(1.0e-10, delta_m_MeV))

    Gamma_X_s = gamma_chi_to_ngamma(theta, delta_m_MeV)
    tau_chi_actual_s = 1.0 / Gamma_X_s

    f_X = f_X_baryon_normalized(omega_inj, m_chi_MeV, n_chi0_over_nH0)
    omega_inj_eV = omega_inj * 1.0e6  # MeV -> eV

    rate_fn = lambda z: mpr21_energy_injection_rate_eV_cm3_s(z, f_X, Gamma_X_s, omega_inj_eV)

    z_grid = sb.build_zinj_grid(tau_chi_actual_s, n_z=n_z)
    y_total, dy_dz = sb.convolve_generic_injection_y("gamma", omega_inj_eV, rate_fn, z_grid)

    # MPR21's own CMB-sensitivity floor (Eq. 19), in their f_chi/tau_chi
    # language: f_chi is chi's DM-density fraction, which for the "chi is
    # all of DM" benchmark (n_chi0=5.4*n_baryon0) is f_chi=1 by construction.
    f_chi = 1.0 if n_chi0_over_nH0 >= 5.0 else n_chi0_over_nH0 / 5.4
    f_chi_over_tau_chi = f_chi / tau_chi_actual_s

    return {
        "m_chi_MeV": m_chi_MeV,
        "delta_m_MeV": delta_m_MeV,
        "theta": theta,
        "omega_inj_MeV": omega_inj,
        "Gamma_X_s": Gamma_X_s,
        "tau_chi_s": tau_chi_actual_s,
        "f_X_baryon_normalized": f_X,
        "f_chi_over_tau_chi_s": f_chi_over_tau_chi,
        "y_total": y_total,
        "z_grid": z_grid,
        "dy_dz": dy_dz,
    }
