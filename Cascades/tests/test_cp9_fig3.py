"""
CP9 checkpoint: fig3_electron_ic_spectrum.py -- the Fig. 3 (electron-only,
Eq. A.5-based) reproduction pipeline, amended-priority item 6.

The central check is energy conservation (see that module's docstring for
the full derivation and the two conceptual bugs it caught along the way):
a burst of n_e=1 cm^-3 electrons, each injected with kinetic energy E0 and
depositing all of it via inverse-Compton scattering, must increase the CMB
energy density by exactly n_e*E0 -- this is not a loose sanity check, it's
an exact identity the derivation is built to satisfy, so any implementation
bug that breaks the physics shows up here as a large (not small) violation.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fig3_electron_ic_spectrum import (
    electron_ic_delta_i,
    blackbody_intensity_x,
    y_distortion_shape,
    single_scatter_j_minus_i,
    _energy_conservation_self_test,
)
from cascade import rates
from cascade import solve_burst as sb


def test_blackbody_and_y_distortion_shapes_are_finite_and_sane():
    x = np.linspace(0.01, 30, 200)
    i_vals = blackbody_intensity_x(x)
    g_vals = y_distortion_shape(x)
    assert np.all(np.isfinite(i_vals)) and np.all(i_vals >= 0.0)
    assert np.all(np.isfinite(g_vals))
    # y-distortion shape's well-known zero crossing is near x~3.83
    sign_changes = np.where(np.diff(np.sign(g_vals)) != 0)[0]
    assert len(sign_changes) >= 1
    x_cross = x[sign_changes[0]]
    assert 3.0 < x_cross < 4.5, f"y-distortion zero crossing at x={x_cross}, expected near 3.83"
    print(f"[PASS] blackbody_intensity_x and y_distortion_shape are finite and sane "
          f"(y-distortion zero crossing at x={x_cross:.2f}, expected near 3.83)")


def test_single_scatter_kernel_small_p_matches_y_distortion_limit():
    # cross-check the P_MIN branch switch is continuous: just above P_MIN
    # (direct numerical convolution) should closely match the analytic
    # small-p formula (p^2/3)*g_y(x) used just below P_MIN
    from fig3_electron_ic_spectrum import P_MIN
    x = np.array([1.0, 2.0, 5.0, 8.0])
    p_above = P_MIN * 1.5
    direct = single_scatter_j_minus_i(x, p_above, n_t=800)
    analytic = (p_above ** 2 / 3.0) * y_distortion_shape(x)
    rel_err = np.abs(direct - analytic) / np.abs(analytic)
    assert np.all(rel_err < 0.05), f"branch mismatch near P_MIN: rel_err={rel_err}"
    print(f"[PASS] single_scatter_j_minus_i is continuous across the P_MIN branch switch "
          f"(max rel_err={rel_err.max():.2%})")


def test_electron_ic_delta_i_conserves_energy():
    """
    The core physical check (see module docstring): integral(Delta[i(x)] dx)
    converted to an energy density via the standard u=(4pi/c) int I_nu dnu
    relation must equal n_e*E0 to within grid-resolution numerical error.
    """
    for E0_keV in [10, 30, 50, 70]:
        rel_err = _energy_conservation_self_test(E0_keV * 1.0e3, z=2.0e4, verbose=False)
        assert rel_err < 0.005, f"E0={E0_keV} keV: energy conservation violated by {rel_err:.2%}"
    print("[PASS] electron_ic_delta_i conserves energy to <0.5% (grid-resolution-limited) "
          "for E0 in {10,30,50,70} keV at z=2e4, matching AK19 Fig. 3's own parameters")


def test_electron_ic_delta_i_shape_sanity():
    """
    Basic shape sanity: the distortion should show AK19's described
    qualitative pattern -- a deficit (negative Delta[i]) near the Planck
    peak (x~2.8) and an excess (positive Delta[i]) at higher x (Wien tail),
    since IC scattering moves photons from low to high frequency.
    """
    x_grid = np.linspace(0.01, 20, 100)
    di = electron_ic_delta_i(30.0e3, 2.0e4, x_grid, n_e_cm3=1.0, n_gamma_pts=300, n_t=300)
    assert np.isfinite(di).all()
    i_peak_idx = np.argmin(np.abs(x_grid - 2.8))
    i_high_idx = np.argmin(np.abs(x_grid - 10.0))
    assert di[i_peak_idx] < 0, f"expected a deficit near the Planck peak, got {di[i_peak_idx]}"
    assert di[i_high_idx] > 0, f"expected an excess in the Wien tail, got {di[i_high_idx]}"
    print(f"[PASS] electron_ic_delta_i shows AK19's described qualitative pattern: "
          f"deficit near x~2.8 (got {di[i_peak_idx]:.4f}), excess near x~10 (got {di[i_high_idx]:.4f})")


def test_delta_I_nu_units_use_observation_epoch_not_injection_epoch():
    """
    Regression test for a units bug found while building the actual Fig. 3
    plot: an early version of delta_i_to_delta_I_nu_units used the
    INJECTION redshift's kT for the i0~kT^3 brightness normalization,
    overstating the plotted brightness by ~(1+z_inj)^3 (~8e12 at z_inj=2e4)
    -- caught by comparing the output magnitude against AK19's actual Fig. 3
    y-axis range (order 1-10, in units of 1e-23 W/m^2/Hz/sr), which the
    injection-epoch version missed by 13 orders of magnitude. The fix: the
    i0 conversion should use the epoch you want to REPORT brightness at
    (default z_observe=0, today), not the injection epoch -- x is
    redshift-invariant so Delta[i(x)] itself doesn't need to change, only
    the overall i0 scale.
    """
    from fig3_electron_ic_spectrum import delta_i_to_delta_I_nu_units

    delta_i_shape = np.array([1.0])  # arbitrary unit shape value
    val_today = delta_i_to_delta_I_nu_units(delta_i_shape, z_observe=0.0)
    val_at_z2e4 = delta_i_to_delta_I_nu_units(delta_i_shape, z_observe=2.0e4)

    ratio = val_at_z2e4[0] / val_today[0]
    expected_ratio = (1.0 + 2.0e4) ** 3  # i0 ~ T^3 ~ (1+z)^3
    rel_err = abs(ratio - expected_ratio) / expected_ratio
    assert rel_err < 0.01, f"i0(z=2e4)/i0(z=0) = {ratio:.3e}, expected ~(1+z)^3={expected_ratio:.3e}"
    print(f"[PASS] delta_i_to_delta_I_nu_units's i0 scales as (1+z)^3 with the *reporting*"
          f" epoch, confirming the units bug (originally using z_inj instead of z_observe) "
          f"is fixed (ratio={ratio:.3e} vs expected {expected_ratio:.3e})")


def test_fig3_amplitude_decreases_with_electron_energy():
    """
    AK19's own qualitative description (Sec. after Fig. 3's introduction):
    "This results in a progressive reduction of the amplitude of the dip
    and the peak of the spectrum for increasing electron energy" -- because
    higher-energy electrons need fewer, larger-shift scatterings to deposit
    the same fixed total energy, spreading/smoothing the distortion rather
    than concentrating it. Check this ordering holds for AK19's own energies
    (30/50/70 keV -- 10 keV is excluded here since AK19 separately notes its
    amplitude is further suppressed by Coulomb heat losses, a channel this
    module deliberately doesn't model -- see module docstring).
    """
    from fig3_electron_ic_spectrum import electron_ic_delta_i, delta_i_to_delta_I_nu_units

    z_inj = 2.0e4
    f_inj = 1.0e-5
    U_CMB = sb.rho_cmb_eV_cm3(z_inj)
    E_inj_total = f_inj * U_CMB
    x_grid = np.linspace(0.05, 15.0, 120)

    peak_amplitudes = []
    for E_keV in [30, 50, 70]:
        E0_eV = E_keV * 1.0e3
        n_e_cm3 = E_inj_total / E0_eV
        di = electron_ic_delta_i(E0_eV, z_inj, x_grid, n_e_cm3=n_e_cm3, n_gamma_pts=300, n_t=300)
        d_I_nu = delta_i_to_delta_I_nu_units(di, z_observe=0.0)
        peak_amplitudes.append(np.max(d_I_nu))

    assert peak_amplitudes[0] > peak_amplitudes[1] > peak_amplitudes[2], (
        f"expected peak amplitude to decrease with electron energy (30>50>70 keV), "
        f"got {peak_amplitudes}"
    )
    print(f"[PASS] Fig. 3 peak amplitude decreases with electron energy as AK19 describes: "
          f"30/50/70 keV peaks = {[f'{a:.3f}' for a in peak_amplitudes]}")


if __name__ == "__main__":
    test_blackbody_and_y_distortion_shapes_are_finite_and_sane()
    test_single_scatter_kernel_small_p_matches_y_distortion_limit()
    test_electron_ic_delta_i_conserves_energy()
    test_electron_ic_delta_i_shape_sanity()
    test_delta_I_nu_units_use_observation_epoch_not_injection_epoch()
    test_fig3_amplitude_decreases_with_electron_energy()
    print("\nCP9 checkpoint: ALL TESTS PASSED")
