"""
dual_solver_harness.py -- AK19 Figs. 10/11/12 (fix #7 from
review_of_cascade_corrections_analysis.md)

SCOPE CORRECTION THIS FILE IMPLEMENTS
---------------------------------------
cascade_corrections_all_figures.md originally filed Figs. 7, 9, 11, 12 as
"Group C: figures that need a second solver". Reading every caption in the
PDF directly (review_of_cascade_corrections_analysis.md Part 1) shows that's
wrong: Figs. 7, 8, 9 are single-code ("with Kompaneets kernel code" only);
only **Figs. 10, 11, 12** actually overlay two independently-run solvers
("Kompaneets kernel" vs "Recoil-only") on the same injection. Fig. 9 in
particular is NOT dual-solver, but per the review it still isn't purely a
"validates spectroxide wiring alone" figure either -- its panel (b) legend
includes 100 GeV injections, and every Kompaneets-kernel figure (Fig. 9
included) feeds its x>=35 boundary condition from the Recoil-only cascade,
so a 100 GeV run's low-x result still depends on cascade fixes #1/#2. Fig. 9
is therefore not built here; it only needs the existing (single-solver)
spectroxide wiring in interface.py, correctly scoped as "validates wiring
at LOW injection energy only" (where the x>=35 handoff barely matters).

WHAT THIS FILE BUILDS
-----------------------
Both curves for a genuine Figs. 10/11/12-style side-by-side comparison, for
a *single one-time burst* injection (NOT the continuous decaying-particle
scenario interface.py's make_dq_dz/make_photon_source were built for --
confirmed directly against the captions: "Comparison of spectral
distortions... for one-time energy injection of 1e-5 x rho_CMB", same
scenario as Fig. 3, just carried further through a low-z PDE relaxation):

  - "Recoil-only": the cascade's own approximate treatment, run standalone
    across the FULL spectrum (both x<35 and x>=35) -- this is exactly
    cascade/ic_spectrum.py's electron_ic_spectrum/fig3_style_delta_I output
    (fix #6), i.e. the pre-relaxation, single-generation spectrum, with NO
    further Compton reprocessing. Runnable in this sandbox (no Rust needed).

  - "Kompaneets kernel": that same photon-channel spectrum (plus the heat
    channel) fed as a source into spectroxide's real Kompaneets+DC/BR PDE
    (x<x_max) and relaxed from z_inj down to a low observation redshift --
    i.e. what AK19's "full Kompaneets kernel" code actually solves. This
    requires spectroxide's Rust binary (`cargo build --release` in the
    crate root); NOT buildable in this sandbox -- no Cargo.toml/Cargo.lock
    or vendored crates.io dependencies were included in src.zip, and
    network access is disabled here, so a build cannot be attempted at all
    (a stronger blocker than interface.py's original "pip has no matching
    distribution" note -- that suggested a plain `pip install` might work
    elsewhere; a from-scratch Rust build additionally needs the crate's
    actual dependency manifest). Structured to run correctly in an
    environment that has the binary built (same pattern as
    fig6_spectrum_via_spectroxide.py) -- see run_kompaneets_side's
    docstring.

Also implements test_x_cut_sensitivity(), which directly tests AK19's own
cross-check claim ("we have varied the boundary point to x=30 and 25 and
checked that it does not affect the solution for x <~ 20") against this
codebase's actual handoff point -- currently hardcoded at x_cut=35 in
interface.make_photon_source, flagged separately as fix #5's x_max/x_cut
mismatch (not fixed by this file; this harness's own spx_solve calls always
pass x_max explicitly so its own sensitivity test is done correctly
regardless of that open bug elsewhere).

Neither of the two "Recoil-only-only" building blocks above needed fixing
for this file -- fix #6 (Eq. A.5-based electron_ic_spectrum) and CP0-6's
heat-channel budget already existed and are reused as-is; what's new here
is purely the burst-style dq_dz(z)/photon_source(x,z) construction (a
narrow pulse at z_inj, since spectroxide's PDE wants continuous-in-z
callables but AK19's scenario is an instantaneous burst -- CP5 already
established these injections complete in << 1 Hubble time) and the
side-by-side comparison/sensitivity-test plumbing around them.
"""

import numpy as np

from cascade import ic_spectrum as ics
from cascade import solve_burst as sb
from cascade import cosmology as cosmo
from cascade import rates

try:
    from spectroxide import solve as spx_solve
    HAVE_SPECTROXIDE = True
except ImportError:
    spx_solve = None
    HAVE_SPECTROXIDE = False

G2_0 = ics.G2_0
X_MIN_DEFAULT, X_MAX_DEFAULT, N_X_DEFAULT = 0.05, 35.0, 700


def _burst_z_pulse(z_inj, rel_width=3.0e-3):
    """
    Normalized (integral over z = 1) narrow pulse in z representing an
    instantaneous burst at z_inj -- CP5 (test_cp5_redshift_marching.py)
    already established injection/cooling completes in << 1 Hubble time
    across the whole 10 keV-100 GeV range this harness covers, i.e. the
    burst really is a delta function on any z-grid spacing spectroxide's
    PDE would resolve; rel_width just needs to be narrow enough for the PDE
    grid to resolve it as such; not itself a physical scale, matching the
    same "the actual width doesn't matter physically, only that the pulse
    integrates to the right total" logic solve_burst.py's docstrings use
    elsewhere for burst-completion arguments.
    """
    width = rel_width * z_inj

    def pulse(z):
        return np.exp(-0.5 * ((np.asarray(z, dtype=float) - z_inj) / width) ** 2) / (width * np.sqrt(2.0 * np.pi))

    return pulse


def make_burst_dq_dz(species, E0_eV, z_inj, injected_frac_rho_cmb=1.0e-5, high_energy=None):
    """Heat-channel dq_dz(z) for a one-time burst at z_inj (SPX convention:
    d(Delta rho/rho)/dz). f_heat*injected_frac deposited as a narrow pulse
    at z_inj -- see _burst_z_pulse."""
    if high_energy is None:
        high_energy = E0_eV > 2.0 * sb.M_E_EV
    budget = sb.burst_energy_budget(species, E0_eV, z_inj, high_energy=high_energy)
    f_heat = budget["f_heat"]
    pulse = _burst_z_pulse(z_inj)

    def dq_dz(z):
        try:
            val = f_heat * injected_frac_rho_cmb * pulse(z)
            return float(val) if np.isfinite(val) else 0.0
        except Exception:
            return 0.0

    return dq_dz, budget


def make_burst_photon_source(species, E0_eV, z_inj, injected_frac_rho_cmb=1.0e-5,
                              x_grid=None, high_energy=None, E_min_eV=100.0):
    """
    Photon-channel source(x, z) for a one-time burst at z_inj, in SPX's
    [G2_0/x^2]*d(deltaN/N)/dx/dz convention: the Eq. A.5-resolved
    electron_ic_delta_n(x) shape (fix #6 -- replacing interface.py's
    narrow-line proxy with the actual resolved spectrum, since we now have
    it) times the same narrow z-pulse used for the heat channel.

    Only supports species in ("e", "e+") -- electron_ic_spectrum is
    electron-only (Eq. A.5 is specifically the electron/positron IC
    formula); photon injection would need a different (as-yet-unbuilt)
    source and isn't attempted here (Figs. 10/11/12 are electron-only
    anyway, per their captions).
    """
    if species not in ("e", "e+"):
        raise NotImplementedError(
            "make_burst_photon_source only supports electron/positron "
            "injection (electron_ic_spectrum is Eq. A.5-based, electron-"
            "only); Figs. 10/11/12 are electron-only, so this isn't a "
            "restriction for this harness's actual use case."
        )
    if x_grid is None:
        x_grid = np.linspace(X_MIN_DEFAULT, X_MAX_DEFAULT, N_X_DEFAULT)

    delta_n_x, budget = ics.electron_ic_delta_n(E0_eV, z_inj, injected_frac_rho_cmb, x_grid, E_min_eV=E_min_eV)
    pulse = _burst_z_pulse(z_inj)

    def photon_source(x, z):
        try:
            x_arr = np.asarray(x, dtype=float)
            shape = np.interp(x_arr, x_grid, delta_n_x, left=0.0, right=0.0)
            val = shape * pulse(z)
            result = np.where(np.isfinite(val), val, 0.0)
            if np.ndim(x) == 0:
                return float(result)
            return result
        except Exception:
            if np.ndim(x) == 0:
                return 0.0
            return np.zeros_like(np.asarray(x, dtype=float))

    return photon_source, budget


# ----------------------------------------------------------------------
# "Recoil-only" side -- runnable standalone, no Rust needed
# ----------------------------------------------------------------------
def run_recoil_only_side(species, E0_eV, z_inj, injected_frac_rho_cmb=1.0e-5, x_grid=None):
    """
    The Recoil-only curve: cascade/ic_spectrum.py's raw (pre-relaxation)
    electron IC spectrum, exactly as reproduce_fig3.py computes it (fix
    #6) -- "the cascade-style approximate treatment... run standalone
    across the full spectrum" per cascade_corrections_all_figures.md's own
    description of what Recoil-only-standalone means.
    """
    if x_grid is None:
        x_grid = np.linspace(X_MIN_DEFAULT, X_MAX_DEFAULT, N_X_DEFAULT)
    delta_I, budget = ics.fig3_style_delta_I(E0_eV, z_inj, injected_frac_rho_cmb, x_grid)
    return {"x": x_grid, "delta_I": delta_I, "budget": budget}


# ----------------------------------------------------------------------
# "Kompaneets kernel" side -- needs the real spectroxide Rust binary
# ----------------------------------------------------------------------
def run_kompaneets_side(species, E0_eV, z_inj, injected_frac_rho_cmb=1.0e-5,
                         x_max=35.0, z_end=1.0e3, x_grid=None):
    """
    Run the actual spectroxide Kompaneets+DC/BR PDE on the burst-style
    dq_dz/photon_source above, from z_inj (times a small headroom factor,
    so the PDE's z_start is strictly above the injection pulse) down to
    z_end.

    Requires spectroxide's Rust binary to be built (see module docstring
    for why that's not possible in this sandbox). Raises RuntimeError with
    a clear message if unavailable, matching interface.py's existing
    pattern -- run this function in an environment where
    `cargo build --release` has been run in the spectroxide crate root.
    """
    if not HAVE_SPECTROXIDE:
        raise RuntimeError(
            "spectroxide is not importable in this environment. If it's "
            "importable but the Rust binary isn't built, this will instead "
            "fail inside spx_solve() with a 'build the Rust binary' error "
            "-- run `cargo build --release` in the spectroxide crate root "
            "first. Neither is possible in this sandbox (no network, and "
            "src.zip has no Cargo.toml/vendored dependencies to build "
            "from) -- see module docstring."
        )
    if x_grid is None:
        x_grid = np.linspace(X_MIN_DEFAULT, min(X_MAX_DEFAULT, x_max), N_X_DEFAULT)

    dq_dz, heat_budget = make_burst_dq_dz(species, E0_eV, z_inj, injected_frac_rho_cmb)
    photon_source, photon_budget = make_burst_photon_source(
        species, E0_eV, z_inj, injected_frac_rho_cmb, x_grid=x_grid)

    z_start = z_inj * 1.2  # headroom above the injection pulse (rel_width=3e-3*z_inj is far narrower)

    result_heat = spx_solve(method="pde", dq_dz=dq_dz, z_start=z_start, z_end=z_end,
                             x_max=x_max, debug=True)
    result_photon = spx_solve(method="pde", photon_source=photon_source, z_start=z_start,
                               z_end=z_end, x_max=x_max, debug=True)

    if not np.allclose(result_heat.x, result_photon.x):
        raise RuntimeError("heat/photon spx_solve() calls returned different x grids.")

    delta_n_total = result_heat.delta_n + result_photon.delta_n
    return {"x": result_heat.x, "delta_n": delta_n_total,
            "y": result_heat.y + result_photon.y,
            "mu": result_heat.mu + result_photon.mu,
            "budget": {"f_heat": heat_budget["f_heat"], "f_ic": photon_budget["f_ic"]}}


# ----------------------------------------------------------------------
# Side-by-side comparison + the paper's own x_cut sensitivity claim
# ----------------------------------------------------------------------
def compare_solvers(species, E0_eV, z_inj_list, injected_frac_rho_cmb=1.0e-5, x_max=35.0):
    """
    Fig. 10/11/12-style comparison: Recoil-only (always runnable here) vs
    Kompaneets kernel (needs spectroxide's binary -- skipped with a clear
    note if unavailable, rather than silently omitted) for the same
    species/energy across several injection redshifts.
    """
    out = {"recoil_only": [], "kompaneets": []}
    for z_inj in z_inj_list:
        out["recoil_only"].append({"z_inj": z_inj, **run_recoil_only_side(
            species, E0_eV, z_inj, injected_frac_rho_cmb)})
        if HAVE_SPECTROXIDE:
            try:
                out["kompaneets"].append({"z_inj": z_inj, **run_kompaneets_side(
                    species, E0_eV, z_inj, injected_frac_rho_cmb, x_max=x_max)})
            except Exception as e:
                out["kompaneets"].append({"z_inj": z_inj, "error": str(e)})
        else:
            out["kompaneets"].append({"z_inj": z_inj, "error": "spectroxide unavailable"})
    return out


def test_x_cut_sensitivity(species="e", E0_eV=1.0e4, z_inj=1.0e4,
                            injected_frac_rho_cmb=1.0e-5,
                            x_max_values=(35.0, 30.0, 25.0), x_check_max=20.0):
    """
    AK19's own cross-check, run against this codebase: "we have varied the
    boundary point to x=30 and 25 and checked that it does not affect the
    solution for x <~ 20." Reruns run_kompaneets_side at each x_max and
    reports the max relative difference in delta_n(x) for x <= x_check_max
    against the x_max=35 baseline. Needs spectroxide's binary (see
    run_kompaneets_side); raises the same RuntimeError otherwise.
    """
    if not HAVE_SPECTROXIDE:
        raise RuntimeError(
            "spectroxide is not importable in this environment -- see "
            "run_kompaneets_side's docstring; this sensitivity test cannot "
            "be executed in this sandbox."
        )
    results = {}
    for x_max in x_max_values:
        results[x_max] = run_kompaneets_side(species, E0_eV, z_inj, injected_frac_rho_cmb, x_max=x_max)

    baseline = results[x_max_values[0]]
    x_base = baseline["x"]
    mask = x_base <= x_check_max
    report = {}
    for x_max in x_max_values[1:]:
        r = results[x_max]
        dn_interp = np.interp(x_base, r["x"], r["delta_n"])
        denom = np.where(np.abs(baseline["delta_n"][mask]) > 0, np.abs(baseline["delta_n"][mask]), np.nan)
        rel_diff = np.abs(dn_interp[mask] - baseline["delta_n"][mask]) / denom
        report[x_max] = {"max_rel_diff_x_le_20": float(np.nanmax(rel_diff))}
    return report


if __name__ == "__main__":
    # Runs the part of this harness that doesn't need spectroxide's binary
    # (Recoil-only side, both electron energies Fig. 10 actually uses), so
    # this file is at least partially verifiable in this sandbox; the
    # Kompaneets-kernel side and the x_cut sensitivity test need to be run
    # where the binary is built (see module docstring).
    print(f"HAVE_SPECTROXIDE = {HAVE_SPECTROXIDE}\n")
    for E0 in (1.0e4, 1.0e5):
        for z_inj in (1.0e4, 2.0e4, 3.0e4, 4.0e4):
            r = run_recoil_only_side("e", E0, z_inj)
            print(f"  Recoil-only  E0={E0/1e3:6.0f} keV  z_inj={z_inj:.0e}  "
                  f"f_heat={r['budget']['f_heat']:.3f}  f_ic={r['budget']['f_ic']:.3f}  "
                  f"peak={r['delta_I'].max()*1e23:.3g}  dip={r['delta_I'].min()*1e23:.3g} (1e-23 W/m^2/Hz/sr)")

    if not HAVE_SPECTROXIDE:
        print("\nspectroxide's Rust binary is not buildable in this sandbox "
              "(see module docstring) -- Kompaneets-kernel side and the "
              "x_cut sensitivity test were not run here.")
