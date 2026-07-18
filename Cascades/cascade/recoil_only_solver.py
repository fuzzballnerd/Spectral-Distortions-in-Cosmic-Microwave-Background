"""
recoil_only_solver.py -- AK19's "Recoil-only" code (Sec. 3.2/4, Eq. 3.3),
built as the second solver needed for the Figs. 10/11/12 dual-solver
comparison (amended-priority item 7 -- "Group C: figures that need a
second solver").

WHAT "RECOIL-ONLY" ACTUALLY MEANS (per AK19's own text, Sec. 3.2 and the
paragraph introducing Fig. 6): this is NOT the same thing as Fig. 3's
Eq. A.5 kernel (that's a single-scattering redistribution function used
only for the injection-instant spectrum). "Recoil-only" is AK19's *second
code*, used to evolve a photon spectrum over many subsequent redshift
steps using only the *average* (deterministic, no stochastic Doppler
broadening, no stimulated/induced-scattering term) recoil formula:

    Delta(nu)/nu = (4kT - h*nu) / (m_e c^2)                       (Eq. 3.3)

applied n_e*sigma_T*c*dt times per timestep (AK19's own stated collision
count). This is explicitly the code AK19 contrast against the "Kompaneets
kernel" code (spectroxide's PDE solve, which adds Doppler broadening from
the electrons' thermal spread and induced/stimulated scattering) in
Figs. 10, 11, 12 -- confirmed by rasterizing and reading AK19 Sec. 3.2-4.1
directly (search for "Recoil-only"/"Kompaneets kernel" in the PDF text
turned up nothing with plain extraction; the actual wording only appears
cleanly once OCR'd from the raw page images together with the surrounding
prose, which is where Eq. 3.3 and the flowchart in Fig. 5 come from).

DERIVATION -- A PURE ADVECTION EQUATION IN x-SPACE
-----------------------------------------------------
Since Eq. 3.3 gives a *deterministic* fractional frequency shift per
scattering (no stochastic spread), and the number of scatterings per unit
time is n_e(z)*sigma_T*c (AK19's own stated collision-rate formula, same
one already used elsewhere in this codebase as `solve_burst._compton_total_rate`),
each photon's dimensionless frequency x=h*nu/kT follows a deterministic
ODE trajectory (in x=h*nu/kT(z) units, using kT(z) at the CURRENT redshift
-- see the "x is NOT literally redshift-invariant under Recoil-only
evolution" note below):

    dx/dt = [n_e(z) sigma_T c] * x * (4 - x) * [kT(z)/(m_e c^2)]

This makes the Recoil-only evolution a pure ADVECTION (no diffusion) of
the photon occupation-number perturbation Delta_n(x) in x-space, governed
by the continuity equation

    d(Delta_n)/dt = -d[v(x,z)*Delta_n]/dx,   v(x,z) as above.

This module solves that continuity equation with a conservative
(photon-number-preserving) upwind finite-volume scheme, stepping in z from
an injection redshift z_inj down to a final z_end. This is DELIBERATELY
simpler than a full stochastic/Monte-Carlo cascade -- it's the direct,
literal implementation of what AK19 say their Recoil-only code does
(average recoil formula, explicit collision count, no Doppler diffusion).

IMPORTANT CAVEAT -- "x is redshift-invariant" ONLY UNDER PURE REDSHIFTING:
the rest of this codebase (rates.py, fig3_electron_ic_spectrum.py) relies
on x=h*nu/kT(z) being invariant under the Hubble expansion alone (photons
redshift exactly with the CMB temperature, so x doesn't change from pure
expansion). That invariance is NOT violated here -- what this module
integrates is the *additional* drift *on top of* that invariance, from
Compton recoil scattering, which is a real physical process that moves
photons in x-space (that's the whole point: recoil is what generates the
distortion). The dz-stepping loop below works entirely in the x-frame
already, so no separate redshift bookkeeping is needed for the ambient
spectrum -- only the recoil-driven drift matters here.

SEEDING: takes `fig3_electron_ic_spectrum.py`'s already-validated Eq. A.5
Delta[i(x)] as the injection-instant (z=z_inj) seed spectrum, converts it
to occupation-number perturbation Delta_n(x)=Delta[i(x)]/x^3, and evolves
that via the advection equation above down to z_end. This matches AK19's
own Fig. 5 flowchart (the electron-injection spectrum feeds into whichever
of the two codes handles its x range), and avoids re-deriving/duplicating
item 6's already-validated electron -> photon spectrum construction.

NOT ATTEMPTED HERE: the "Kompaneets kernel" side needs spectroxide's
compiled Rust binary (`cargo build --release` in the crate root) -- this
sandbox has neither a Cargo.toml/vendored dependencies to build from, nor
`cargo`/`rustc` installed at all (checked directly: `which cargo rustc`
returns nothing), a strictly stronger blocker than a missing prebuilt
binary. `run_kompaneets_side` below is structured to call spectroxide's
real solver when it IS available (same calling convention as
fig6_spectrum_via_spectroxide.py) and raises a clear, specific
RuntimeError otherwise -- it does not attempt to reimplement the
Kompaneets/DC/BR physics itself, since that would just be a second,
unvalidated implementation of what spectroxide is already supposed to
provide.
"""

import numpy as np

from cascade import rates
from cascade import cosmology as cosmo
from cascade import solve_burst as sb
from fig3_electron_ic_spectrum import electron_ic_delta_i, H_EV_S

try:
    from spectroxide import solve as spx_solve
    HAVE_SPECTROXIDE = True
except ImportError:
    spx_solve = None
    HAVE_SPECTROXIDE = False


def recoil_velocity(x, z):
    """
    dx/dt for a photon at dimensionless frequency x, at redshift z, under
    AK19's Eq. 3.3 average recoil formula and the standard n_e*sigma_T*c
    collision rate. Returns cm... actually returns 1/s * dimensionless =
    1/s (dx/dt has units of x per second, x itself dimensionless).
    """
    T_K = cosmo.T_gamma(z)
    kT_eV = rates.K_B_EV_PER_K * T_K
    collision_rate = sb._compton_total_rate(z)  # n_e sigma_T c, s^-1
    return collision_rate * x * (4.0 - x) * (kT_eV / rates.M_E_EV)


def _dt_dz(z):
    """dt/dz = -1/[(1+z)H(z)] (negative; z decreases as t increases).
    Returns |dt/dz| (always integrate with explicit sign handling in the
    stepping loop, not embedded here, to keep this function's sign
    unambiguous to callers)."""
    return 1.0 / ((1.0 + z) * cosmo.H(z))


def evolve_recoil_only(delta_n_seed, x_grid, z_start, z_end, max_dz_frac=0.02):
    """
    Evolve an occupation-number perturbation Delta_n(x) from z_start to
    z_end under pure Eq. 3.3 recoil advection (see module docstring),
    using a conservative upwind finite-volume scheme with adaptive
    (CFL-limited) step size in z.

    Parameters
    ----------
    delta_n_seed : Delta_n(x) at z_start, same shape as x_grid.
    x_grid : dimensionless frequency grid (uniform spacing assumed).
    z_start, z_end : integrate from z_start down to z_end (z_start > z_end).
    max_dz_frac : cap each step's |dz|/z at this fraction, in addition to
        the CFL condition, for smooth resolution of the evolution.

    Returns
    -------
    Delta_n(x) at z_end, same shape as x_grid.
    """
    x_grid = np.asarray(x_grid, dtype=float)
    dx = x_grid[1] - x_grid[0]
    n_now = np.array(delta_n_seed, dtype=float, copy=True)
    z = z_start

    n_steps = 0
    while z > z_end:
        v = recoil_velocity(x_grid, z)
        v_max = np.max(np.abs(v))
        if v_max <= 0:
            break
        # CFL: |v|*dt/dx <= 1 for the upwind scheme's stability. Convert to
        # a dz step via dt = |dt/dz|*|dz|.
        dt_dz_now = _dt_dz(z)
        dz_cfl = dx / (v_max * dt_dz_now) if v_max * dt_dz_now > 0 else np.inf
        dz = min(dz_cfl * 0.5, max_dz_frac * z, z - z_end)
        dz = max(dz, 1.0e-8 * z)  # guard against a pathological zero step
        dt = dz * dt_dz_now

        # Conservative upwind flux: F_{i+1/2} = v_{i+1/2}*n_i if v>=0 else v_{i+1/2}*n_{i+1}
        v_faces = 0.5 * (v[:-1] + v[1:])
        n_left = n_now[:-1]
        n_right = n_now[1:]
        flux = np.where(v_faces >= 0, v_faces * n_left, v_faces * n_right)
        # zero-flux (reflective-free, i.e. photons don't spuriously enter)
        # boundaries -- append zero flux at both ends
        flux_full = np.concatenate(([0.0], flux, [0.0]))
        dn_dt = -(flux_full[1:] - flux_full[:-1]) / dx
        n_now = n_now + dt * dn_dt

        z -= dz
        n_steps += 1
        if n_steps > 200000:
            raise RuntimeError("evolve_recoil_only: step count exceeded 200000, "
                                "CFL step size is not converging -- check inputs.")

    return n_now


def run_recoil_only_side(species, E0_eV, z_inj, z_end=1.0e3,
                          injected_frac_rho_cmb=1.0e-5, x_grid=None,
                          n_gamma_pts=400, n_t=350):
    """
    Full "Recoil-only" pipeline for a burst injection: build the Eq. A.5
    injection-instant spectrum (item 6's electron_ic_delta_i), convert to
    Delta_n(x), evolve it via pure recoil advection from z_inj to z_end,
    and convert back to delta_I_nu units (today's brightness scale, same
    convention as fig3_electron_ic_spectrum.delta_i_to_delta_I_nu_units).

    Electron/positron injection only (Eq. A.5 is electron-only) -- matches
    AK19's own Figs. 10/11/12, which are all monochromatic-electron cases.
    """
    if species not in ("e", "e+"):
        raise NotImplementedError(
            "run_recoil_only_side only supports electron/positron injection "
            "(Eq. A.5, this module's seed spectrum, is electron-only); "
            "AK19's Figs. 10/11/12 are electron-only, so this isn't a "
            "restriction for this harness's actual use case."
        )
    if x_grid is None:
        x_grid = np.linspace(0.05, 60.0, 500)

    U_CMB = sb.rho_cmb_eV_cm3(z_inj)
    E_inj_total = injected_frac_rho_cmb * U_CMB
    n_e_cm3 = E_inj_total / E0_eV

    delta_i_seed = electron_ic_delta_i(E0_eV, z_inj, x_grid, n_e_cm3=n_e_cm3,
                                        n_gamma_pts=n_gamma_pts, n_t=n_t)
    delta_n_seed = delta_i_seed / x_grid ** 3

    delta_n_final = evolve_recoil_only(delta_n_seed, x_grid, z_inj, z_end)

    delta_i_final = delta_n_final * x_grid ** 3
    T0_K = cosmo.T_gamma(0.0)
    kT0_eV = rates.K_B_EV_PER_K * T0_K
    i0_today = 2.0 * kT0_eV ** 3 / (H_EV_S * rates.C_CM_S) ** 2
    delta_I_nu = i0_today * delta_i_final  # eV s^-1 cm^-2 Hz^-1 sr^-1

    return {"x": x_grid, "delta_I_nu": delta_I_nu, "delta_n": delta_n_final,
            "n_e_cm3": n_e_cm3, "z_inj": z_inj, "z_end": z_end}


def run_kompaneets_side(species, E0_eV, z_inj, z_end=1.0e3,
                         injected_frac_rho_cmb=1.0e-5, x_max=35.0):
    """
    The "Kompaneets kernel" side -- requires spectroxide's compiled Rust
    binary. See module docstring for why that's unavailable in this
    sandbox (no cargo/rustc, no Cargo.toml/vendored deps to build from).
    Raises RuntimeError with a specific, actionable message otherwise;
    does NOT attempt an independent reimplementation of the Kompaneets/
    DC/BR physics.
    """
    if not HAVE_SPECTROXIDE:
        raise RuntimeError(
            "spectroxide is not importable in this environment, and even if "
            "it were, its 'pde' solve method shells out to a compiled Rust "
            "binary that isn't built here -- this sandbox has neither "
            "`cargo`/`rustc` installed nor a Cargo.toml/vendored "
            "dependencies to build from (checked directly). Run this "
            "function in an environment with the spectroxide crate built "
            "(`cargo build --release` in its root) to get the Kompaneets-"
            "kernel comparison curve."
        )
    # Structured for when the binary IS available -- see
    # fig6_spectrum_via_spectroxide.py for the calling convention this
    # would use (burst dq_dz/photon_source callables, spx_solve(method="pde",...)).
    raise NotImplementedError(
        "spectroxide is importable but the burst-injection PDE wiring "
        "(dq_dz/photon_source callables for a one-time burst, matching "
        "fig6_spectrum_via_spectroxide.py's continuous-injection pattern) "
        "has not been built out in this module -- only the Recoil-only "
        "side was completed, since the Kompaneets side cannot be tested "
        "in this sandbox regardless. Build the burst dq_dz/photon_source "
        "wiring here (following fig6_spectrum_via_spectroxide.py's pattern, "
        "adapted for a narrow burst pulse in z rather than continuous "
        "decay injection) once you're in an environment where you can "
        "actually verify it runs."
    )


def compare_solvers(species, E0_eV, z_inj, z_end=1.0e3, injected_frac_rho_cmb=1.0e-5):
    """
    Fig. 10/11/12-style comparison. Recoil-only always runs (no external
    dependencies); Kompaneets-kernel is attempted and reported as
    unavailable with a clear reason if spectroxide's binary isn't built.
    """
    out = {"recoil_only": run_recoil_only_side(species, E0_eV, z_inj, z_end, injected_frac_rho_cmb)}
    try:
        out["kompaneets"] = run_kompaneets_side(species, E0_eV, z_inj, z_end, injected_frac_rho_cmb)
    except (RuntimeError, NotImplementedError) as e:
        out["kompaneets"] = {"error": str(e)}
    return out



def make_fig10_style_plot(output_path="recoil_only_fig10_style.png"):
    """
    Reproduce AK19 Fig. 10's right-hand ("Recoil-only") panels: delta_I_nu
    vs x for 10 keV and 100 keV electron injection, at four injection
    redshifts (1e4, 2e4, 3e4, 4e4), evolved to z_end=1000. The Kompaneets-
    kernel (left-hand) panels are not produced here -- see module
    docstring for why.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    z_list = [1.0e4, 2.0e4, 3.0e4, 4.0e4]
    colors = plt.cm.plasma(np.linspace(0.15, 0.85, len(z_list)))

    for ax, E0_keV in zip(axes, (10, 100)):
        for z_inj, color in zip(z_list, colors):
            r = run_recoil_only_side("e", E0_keV * 1.0e3, z_inj, z_end=1.0e3)
            dI = r["delta_I_nu"] * 1.602176634e-19 * 1.0e4 / 1.0e-23
            ax.plot(r["x"], dI, color=color, label=f"z={z_inj:.0e}")
        ax.axhline(0, color="gray", lw=0.5)
        ax.set_xlabel(r"$x = h\nu/kT$")
        ax.set_xlim(0, 20)
        ax.set_title(f"Recoil-only, {E0_keV} keV e$^-$")
        ax.legend(fontsize=8)
    axes[0].set_ylabel(r"$\delta I_\nu$ ($10^{-23}$ W m$^{-2}$ Hz$^{-1}$ sr$^{-1}$)")

    fig.suptitle("Recoil-only solver (item 7): AK19 Fig. 10-style redshift comparison\n"
                  "(Kompaneets-kernel panels not available -- needs spectroxide's Rust binary)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"Saved {output_path}")
    return output_path


if __name__ == "__main__":
    print(f"HAVE_SPECTROXIDE = {HAVE_SPECTROXIDE}\n")
    print("Recoil-only side (AK19 Fig. 10-style: 10 keV and 100 keV electrons, "
          "z_inj=1e4..4e4, evolved to z_end=1e3):\n")
    for E0_keV in (10, 100):
        for z_inj in (1.0e4, 2.0e4, 3.0e4, 4.0e4):
            r = run_recoil_only_side("e", E0_keV * 1.0e3, z_inj, z_end=1.0e3)
            dI = r["delta_I_nu"] * 1.602176634e-19 * 1.0e4 / 1.0e-23
            print(f"  E0={E0_keV:4d} keV  z_inj={z_inj:.0e}  "
                  f"peak={dI.max():7.3f}  dip={dI.min():7.3f}  (1e-23 W/m^2/Hz/sr)")

    if not HAVE_SPECTROXIDE:
        print("\nspectroxide unavailable in this sandbox -- Kompaneets-kernel "
              "side not run here (see module docstring / run_kompaneets_side).")

    print()
    print("Building Fig. 10-style Recoil-only plot...")
    make_fig10_style_plot("/mnt/user-data/outputs/recoil_only_fig10_style.png")
