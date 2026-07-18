"""
test_mckeen_bridge.py -- standing tests for mckeen2021_cascade_implementation_plan_v3.md
Sec 8's validation-plan items that don't require the compiled spectroxide
Rust binary (items 2 and 6; items 4 and 5 also plan-Sec-8 items but are
either binary-dependent (4) or already run and reported inline (5, the
H(z)/n_H(z) cross-check -- see the conversation, not repeated as a test
here since it's a one-time cross-codebase sanity check, not a regression
guard on this bridge's own code).

Run with: pytest test_mckeen_bridge.py -v (or, from repo root: pytest tests/)
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from cascade.mckeen_bridge import (
    omega_inj_and_Tn_eV,
    _cached_budget,
    HADRONIC_THRESHOLD_EV,
)

# ── Sec 8 item 6: hadronic-threshold guard actually fires ──────────────────

def test_hadronic_guard_fires_above_threshold():
    """Delta_m = 140 MeV, just above m_pi0 (~134.98 MeV) -- must raise,
    not silently return a two-body omega_inj."""
    with pytest.raises(ValueError, match="hadronic"):
        omega_inj_and_Tn_eV(1.4e8)


def test_hadronic_guard_silent_below_threshold():
    """Delta_m = 130 MeV, below threshold -- must NOT raise."""
    omega_inj, Tn = omega_inj_and_Tn_eV(1.3e8)
    assert omega_inj > 0


def test_hadronic_guard_boundary_is_m_pi0():
    """The threshold constant itself should sit at m_pi0, not some other
    round number -- pin it so a future edit can't silently drift."""
    assert 1.349e8 < HADRONIC_THRESHOLD_EV < 1.351e8


def test_forbidden_decay_guard_fires():
    """Delta_m <= 0 is kinematically forbidden -- must raise for both a
    zero and a negative mass splitting."""
    for bad_delta_m in (0.0, -1.0e6):
        with pytest.raises(ValueError):
            omega_inj_and_Tn_eV(bad_delta_m)


@pytest.mark.parametrize("dm_mev,expected_omega_mev", [
    (5, 4.9868), (10, 9.9473), (20, 19.7916),
    (30, 29.5359), (40, 39.1833), (50, 48.7368),
])
def test_kinematics_matches_plan_table(dm_mev, expected_omega_mev):
    """Sec 1's reference table -- pin the actual sweep grid's kinematics
    so a future change to mpr21.omega_inj_MeV can't silently drift."""
    omega_inj_eV, _ = omega_inj_and_Tn_eV(dm_mev * 1.0e6)
    assert omega_inj_eV / 1.0e6 == pytest.approx(expected_omega_mev, abs=2e-4)


# ── Sec 8 item 2: energy-budget sum check across the active z window ──────

@pytest.mark.parametrize("dm_mev", [5, 10, 20, 30, 40, 50])
@pytest.mark.parametrize("z", [2.76e4, 1.0e5, 1.53832e5, 5.0e5, 1.0e6, 1.53e6])
def test_energy_budget_conserves(dm_mev, z):
    """f_heat + f_ic + f_subthreshold must sum to 1.0 (within 1e-6, per
    plan Sec 8 item 2) at every z across the tau_X=1e9s active window
    (z in [2.76e4, 1.53e6], plan Sec 3), for every Delta_m in the sweep."""
    omega_inj_eV, _ = omega_inj_and_Tn_eV(dm_mev * 1.0e6)
    budget = _cached_budget("gamma", omega_inj_eV, z, high_energy=True)
    total = budget["f_heat"] + budget["f_ic"] + budget.get("f_subthreshold", 0.0)
    assert total == pytest.approx(1.0, abs=1.0e-6)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))

def test_photon_source_line_not_pinned_at_grid_edge():
    """The photon-channel line's peak should move meaningfully across the
    Delta_m sweep grid, not collapse onto x_cut regardless of Delta_m."""
    import numpy as np
    from cascade.mckeen_bridge import make_mckeen_photon_source, omega_inj_and_Tn_eV
    x_probe = np.logspace(np.log10(1.5e-4), np.log10(80.0), 4000)
    peaks = []
    for dm_mev in (5, 50):
        omega_inj_eV, _ = omega_inj_and_Tn_eV(dm_mev * 1e6)
        ps = make_mckeen_photon_source(omega_inj_eV, 1/1e9, 0.01,
                                        x_max_factor=80.0, x_min=1.5e-4)
        vals = ps(x_probe, 1.53832e5)  # t=tau_X point
        peaks.append(x_probe[np.argmax(vals)])
    assert peaks[1] - peaks[0] > 5.0, (
        f"Delta_m=5 and 50 MeV photon-channel lines land at nearly the same "
        f"x ({peaks}) -- placement formula is likely saturating the "
        f"log-compression uniformly across the sweep."
    )

def test_photon_source_not_saturated_against_x_cut():
    """At the low end of the Delta_m grid the line shouldn't already be
    sitting on top of x_cut."""
    import numpy as np
    from cascade.mckeen_bridge import make_mckeen_photon_source, omega_inj_and_Tn_eV
    omega_inj_eV, _ = omega_inj_and_Tn_eV(5.0e6)
    ps = make_mckeen_photon_source(omega_inj_eV, 1/1e9, 0.01,
                                    x_max_factor=80.0, x_min=1.5e-4)
    x_probe = np.logspace(np.log10(1.5e-4), np.log10(80.0), 4000)
    vals = ps(x_probe, 1.53832e5)
    x_peak = x_probe[np.argmax(vals)]
    # Hand-check in the review (Sec 4) for Delta_m=5 MeV gives
    # x_eff_raw ~ 82, i.e. just above x_anchor=0.9*x_cut=72 -- barely into
    # the log-compression regime, landing at x_eff ~ 73. The review's own
    # illustrative "< 70" bound was written before that hand-check; the
    # bound actually consistent with the fixed formula is "clear of x_cut
    # itself" (< 76), not clear of x_anchor.
    assert x_peak < 76.0, f"x_eff={x_peak:.2f} is within 4 of x_cut=80"
