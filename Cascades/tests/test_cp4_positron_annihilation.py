"""
CP4 checkpoint (plan Sec 6): positron / annihilation bookkeeping.

Build: the "positron = electron of same KE + two 511 keV photons"
substitution (plan Sec 3.3/3.4), implemented as `positron_burst_energy` in
solve_burst.py.

Check against AK19 Fig. 9b, comparing electron-positron-pair injection
against electron-only injection at 10 keV and 100 GeV:
  - at 10 keV, the spectrum should be visibly dominated by the 511 keV line
    (annihilation energy, 2*511 keV = 1.022 MeV, vastly exceeds the 10 keV
    kinetic energy, so the positron's *normalized* distortion should be
    roughly two orders of magnitude larger than an electron-only burst of
    the same KE);
  - at 100 GeV, electron-only and e+e- pair injection should be
    indistinguishable (annihilation energy is now a ~1e-5 correction to the
    100 GeV kinetic energy).

Per the plan's own diagnosis for this checkpoint ("this is purely a
bookkeeping/data-flow check... a failure here is almost certainly in how the
511 keV photons are injected into the bins"), the tests below are aimed
squarely at that bookkeeping: energy placement (exactly 2*m_e*c^2 total) and
no double-counting against the positron's own kinetic energy.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cascade import solve_burst as sb
from cascade import rates

Z_INJ = 20000.0
INJECTED_FRAC_RHO_CMB = 1.0e-5


def test_annihilation_energy_bookkeeping():
    # Exactly two 511 keV (m_e c^2) photons per annihilation, and this
    # shouldn't depend on the positron's kinetic energy.
    for E0 in [1.0e4, 1.0e8]:
        budget = sb.positron_burst_energy(E0, Z_INJ, high_energy=(E0 > 1e7))
        assert abs(budget["E_annihilation_eV"] - 2.0 * rates.M_E_EV) < 1.0, (
            f"annihilation energy should be exactly 2*m_e*c^2, "
            f"got {budget['E_annihilation_eV']} eV"
        )
    print(f"[PASS] annihilation energy is exactly 2*m_e*c^2 = {2*rates.M_E_EV:.1f} eV, "
          f"independent of positron KE")


def test_10kev_positron_dominated_by_511_line():
    # At 10 keV KE, the 1.022 MeV of annihilation energy dwarfs the
    # positron's own kinetic energy -- normalized distortion should be
    # roughly 2 orders of magnitude larger than an electron-only burst.
    E0 = 1.0e4
    y_e, budget_e = sb.y_distortion_estimate("e", E0, Z_INJ, INJECTED_FRAC_RHO_CMB)
    y_ep, budget_ep = sb.y_distortion_estimate("e+", E0, Z_INJ, INJECTED_FRAC_RHO_CMB)
    ratio = y_ep / y_e
    print(f"[info] 10 keV: y_e={y_e:.3e}, y_e+={y_ep:.3e}, ratio={ratio:.1f}")
    print(f"[info] positron budget: {budget_ep}")
    assert ratio > 20.0, (
        f"10 keV positron distortion should be dominated by the 511 keV "
        f"line (~2 orders of magnitude above electron-only), got ratio={ratio:.1f}"
    )
    print("[PASS] 10 keV positron burst is dominated by the 511 keV annihilation line "
          f"(y_e+ / y_e = {ratio:.1f})")


def test_100gev_positron_indistinguishable_from_electron():
    # At 100 GeV KE, annihilation energy is a ~1e-5 relative correction --
    # electron-only and e+e- injection should be indistinguishable.
    E0 = 1.0e11
    y_e, _ = sb.y_distortion_estimate("e", E0, Z_INJ, INJECTED_FRAC_RHO_CMB, high_energy=True)
    y_ep, _ = sb.y_distortion_estimate("e+", E0, Z_INJ, INJECTED_FRAC_RHO_CMB, high_energy=True)
    ratio = y_ep / y_e
    print(f"[info] 100 GeV: y_e={y_e:.6e}, y_e+={y_ep:.6e}, ratio={ratio:.6f}")
    assert 0.99 < ratio < 1.01, (
        f"100 GeV electron-only and positron bursts should be indistinguishable "
        f"(annihilation is a ~1e-5 correction), got ratio={ratio:.6f}"
    )
    print(f"[PASS] 100 GeV positron burst indistinguishable from electron-only "
          f"(ratio={ratio:.6f})")


def test_positron_ke_channel_matches_electron_alone():
    # The positron's KE-only contribution (subtracting the annihilation
    # photons back out) should reduce exactly to electron_heat_fraction /
    # electron_cascade -- i.e. no cross-contamination between the KE and
    # annihilation bookkeeping.
    for E0, high_energy in [(1.0e5, False), (1.0e9, True)]:
        if high_energy:
            ke_only = sb.electron_cascade(E0, Z_INJ)
        else:
            f_heat = sb.electron_heat_fraction(E0, Z_INJ)
            ke_only = {"f_heat": f_heat, "f_ic": 1.0 - f_heat, "f_subthreshold": 0.0}

        positron = sb.positron_burst_energy(E0, Z_INJ, high_energy=high_energy)
        ann_heat, ann_ic, ann_sub = 0.0, 0.0, 0.0
        for _ in range(2):
            b = sb.photon_cascade_full(rates.M_E_EV, Z_INJ) if high_energy \
                else sb.photon_cascade(rates.M_E_EV, Z_INJ)
            ann_heat += b["f_heat"] * rates.M_E_EV
            ann_ic += b["f_ic"] * rates.M_E_EV
            ann_sub += b.get("f_subthreshold", 0.0) * rates.M_E_EV

        recon_f_heat = (ke_only["f_heat"] * E0 + ann_heat) / E0
        recon_f_ic = (ke_only["f_ic"] * E0 + ann_ic) / E0
        assert abs(recon_f_heat - positron["f_heat"]) < 1e-9
        assert abs(recon_f_ic - positron["f_ic"]) < 1e-9
    print("[PASS] positron budget decomposes cleanly into KE (matches electron alone) "
          "+ two independently-cascaded 511 keV photons, no cross-contamination")


if __name__ == "__main__":
    test_annihilation_energy_bookkeeping()
    test_10kev_positron_dominated_by_511_line()
    test_100gev_positron_indistinguishable_from_electron()
    test_positron_ke_channel_matches_electron_alone()
    print("\nCP4 checkpoint: ALL TESTS PASSED")
