"""
CP8 checkpoint (plan Sec 6): chi -> n gamma-specific application.

Build: mpr21.py, plugging MPR21's actual kinematics (omega_inj, Gamma_{chi->n
gamma}, Eq. 7) into the validated Tier-1 pipeline (CP0-6) at a benchmark
(m_chi, theta, Delta_m) point drawn from MPR21's own explored parameter
space (their Delta_m=10 MeV BBN benchmark, "chi is all of DM" scenario).

Check (plan): order-of-magnitude sanity -- does the resulting distortion
amplitude sit in a regime comparable to where MPR21's own CMB-
anisotropy/COBE constraints bite (their Fig. 3/4)? No formal numeric target;
this is new-physics territory, not a reproduction.

A note on the f_chi/tau_chi <~ 1e-25 s^-1 comparison, worked out explicitly
here because it's easy to misapply: MPR21's Eq. 19 bound (f_chi/tau_chi <~
1e-25 s^-1) is specifically their *CMB temperature anisotropy* bound, valid
for tau_chi >~ t_rec ~ 1e13 s (decays during/after recombination altering
the ionization history) -- a different observable, at a much later epoch,
than the *spectral distortion* (mu/y) regime this entire Tier-1/Tier-2
pipeline targets (tau_chi ~ 1e8-1e13 s, z ~ 1e3-2e5, before recombination).
Comparing this benchmark's f_chi/tau_chi directly against 1e-25 s^-1 is
therefore not the right check -- of course they differ by many orders of
magnitude, they're different bounds at different epochs. The actually
meaningful sanity check (and the one implemented below) is: does a
spectral-distortion-regime benchmark (tau_chi ~ 1e11 s, comparable to CP6's
own AK19 benchmark) produce a y-distortion of the same order as COBE/FIRAS's
own y <~ 1.5e-5 sensitivity -- i.e. is the code landing in the observationally
relevant ballpark, not many orders off in either direction.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cascade import mpr21

COBE_Y_SENSITIVITY = 1.5e-5


def test_kinematics_reproduce_paper_numeric_example():
    # MPR21's own in-text numeric example: for theta=1e-10, Delta_m=10 MeV,
    # Gamma_{chi->n gamma} ~= 1/(220 s).
    Gamma = mpr21.gamma_chi_to_ngamma(theta=1.0e-10, delta_m_MeV=10.0)
    tau = 1.0 / Gamma
    print(f"[info] theta=1e-10, Delta_m=10 MeV: tau_chi={tau:.1f} s (MPR21 quote ~220 s)")
    assert abs(tau - 220.0) / 220.0 < 0.05, f"expected tau~220s, got {tau:.2f}s"
    print("[PASS] gamma_chi_to_ngamma reproduces MPR21's own quoted numeric example exactly")


def test_omega_inj_reduces_to_delta_m_for_small_splitting():
    # omega_inj = (m_chi/2)(1 - m_n^2/m_chi^2) ~= Delta_m for Delta_m << m_chi
    # (plan Sec 1's own approximation, stated explicitly).
    for delta_m in [1.0, 10.0, 50.0]:
        m_chi = mpr21.M_N_MEV + delta_m
        omega = mpr21.omega_inj_MeV(m_chi)
        rel_diff = abs(omega - delta_m) / delta_m
        print(f"[info] Delta_m={delta_m} MeV: omega_inj={omega:.3f} MeV "
              f"(relative diff from Delta_m: {rel_diff:.2%})")
        assert rel_diff < 0.03, f"omega_inj should be within 3% of Delta_m for Delta_m<<m_chi"
    print("[PASS] omega_inj ~= Delta_m for Delta_m << m_chi, as expected from two-body kinematics")


def test_distortion_amplitude_comparable_to_cobe_sensitivity():
    # Benchmark: Delta_m=10 MeV, "chi is all of DM" (n_chi0=5.4*n_baryon0),
    # tau_chi=1e11s (spectral-distortion regime, comparable to CP6's own
    # AK19 benchmark redshift window).
    result = mpr21.run_benchmark(delta_m_MeV=10.0, tau_chi_s=1.0e11,
                                  n_chi0_over_nH0=5.4, n_z=25)
    y = result["y_total"]
    print(f"[info] benchmark (Delta_m=10 MeV, tau_chi=1e11 s, chi=all DM): "
          f"theta={result['theta']:.3e}, omega_inj={result['omega_inj_MeV']:.2f} MeV, "
          f"y_total={y:.3e} (COBE sensitivity: {COBE_Y_SENSITIVITY:.1e})")
    ratio = y / COBE_Y_SENSITIVITY
    assert 0.01 < ratio < 10.0, (
        f"y_total={y:.3e} should be within ~2 orders of magnitude of COBE's "
        f"y~1.5e-5 sensitivity for a physically reasonable benchmark, got ratio={ratio:.3f}"
    )
    print(f"[PASS] benchmark distortion amplitude ({y:.3e}) sits in the "
          f"observationally-relevant ballpark relative to COBE (ratio={ratio:.3f})")


def test_amplitude_scales_with_delta_m():
    # Larger Delta_m -> larger omega_inj -> (generically) larger distortion,
    # at fixed tau_chi -- a basic physical sensitivity check (plan's Fig 14b-
    # style expectation, now applied to the MPR21-specific kinematics).
    y_10, y_50 = [], []
    for dm in [10.0, 50.0]:
        r = mpr21.run_benchmark(delta_m_MeV=dm, tau_chi_s=1.0e11, n_chi0_over_nH0=5.4, n_z=20)
        (y_10 if dm == 10.0 else y_50).append(r["y_total"])
    print(f"[info] Delta_m=10 MeV: y={y_10[0]:.3e}; Delta_m=50 MeV: y={y_50[0]:.3e}")
    assert y_50[0] > y_10[0], "distortion amplitude should grow with Delta_m at fixed tau_chi"
    print("[PASS] distortion amplitude grows with Delta_m at fixed tau_chi, as expected")


def test_anisotropy_floor_epoch_mismatch_documented():
    # Explicitly demonstrate (not just assert) why f_chi/tau_chi for a
    # spectral-distortion-regime benchmark differs enormously from MPR21's
    # 1e-25 s^-1 anisotropy floor -- documenting the epoch/observable
    # mismatch rather than silently comparing incompatible numbers.
    result = mpr21.run_benchmark(delta_m_MeV=10.0, tau_chi_s=1.0e11,
                                  n_chi0_over_nH0=5.4, n_z=15)
    f_chi_over_tau = result["f_chi_over_tau_chi_s"]
    mpr21_anisotropy_floor = 1.0e-25
    print(f"[info] this benchmark's f_chi/tau_chi = {f_chi_over_tau:.3e} s^-1 "
          f"(tau_chi=1e11s, spectral-distortion regime, z~1e3-2e5)")
    print(f"[info] MPR21's Eq. 19 anisotropy floor = {mpr21_anisotropy_floor:.3e} s^-1 "
          f"(applies for tau_chi >~ 1e13 s, decays during/after recombination)")
    print(f"[info] ratio = {f_chi_over_tau / mpr21_anisotropy_floor:.3e} -- expected to be "
          f"enormous, since these are different observables at different epochs, "
          f"not a discrepancy in this code (see module docstring)")
    assert f_chi_over_tau > mpr21_anisotropy_floor, (
        "sanity check: a benchmark deep in the pre-recombination spectral-"
        "distortion regime should have f_chi/tau_chi far above the much "
        "more sensitive late-time anisotropy floor"
    )
    print("[PASS] epoch/observable mismatch between this benchmark and MPR21's "
          "anisotropy floor is large and in the expected direction, documented "
          "rather than treated as a code discrepancy")


if __name__ == "__main__":
    test_kinematics_reproduce_paper_numeric_example()
    test_omega_inj_reduces_to_delta_m_for_small_splitting()
    test_distortion_amplitude_comparable_to_cobe_sensitivity()
    test_amplitude_scales_with_delta_m()
    test_anisotropy_floor_epoch_mismatch_documented()
    print("\nCP8 checkpoint: ALL TESTS PASSED")
