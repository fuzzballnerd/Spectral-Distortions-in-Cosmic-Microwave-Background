"""
sweep_deltam_mckeen_corrected.py -- corrected McKeen 2021 (chi -> n + gamma)
spectral-distortion sweep, per mckeen2021_cascade_implementation_plan_v3.md.

Forked per plan Sec 7's file-layout table:
  - skeleton (npz resume-cache, ThreadPoolExecutor sweep, two-panel +
    SED plot, summary table) comes from the attached
    sweep_deltam_photon_tau1e9.py, since that's the target output format.
  - the inner per-Delta_m solve function (run_one) comes from
    fig6_spectrum_via_spectroxide.py's spectrum_via_spectroxide_pde
    pattern (two solve() calls, x-grid-match assert, delta_n/mu/y sum),
    since that plumbing is already correct and reused verbatim.
  - the only new code is the two builder-function calls inside run_one
    (cascade.mckeen_bridge.make_mckeen_dq_dz / make_mckeen_photon_source),
    plus cosmo=cosmo_ak19 (Sec 4.5), timeout=3600 (Sec 4.3/Sec 7), and the
    z_start/z_end/z_min/z_max/n_z choices from Sec 3/Sec 4.6.

This is a FORK, not a rewrite of the attached script -- everything not
called out in the table above (imports, the resume-cache dict shape, the
two-panel spectrum+residual plot, the SED plot, the summary table) is
carried over with only the minimum changes needed to plug in the new
run_one.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import spectroxide

from cascade.mckeen_bridge import (
    omega_inj_and_Tn_eV,
    make_mckeen_dq_dz,
    make_mckeen_photon_source,
    validate_omega_inj_for_high_energy,
    validate_pde_z_range,
)

# ── save location ───────────────────────────────────────────────────────────
SAVE_DIR  = "../spectroxide/RUNS"
SAVE_FILE = "sweep_deltam_mckeen_corrected_tau1e9"
save_path = Path(SAVE_DIR)
save_path.mkdir(parents=True, exist_ok=True)
npz_file  = save_path / f"{SAVE_FILE}.npz"

# ── physical constants (SI) ───────────────────────────────────────────────
h_PLANCK = 6.62607015e-34     # J s
k_B      = 1.380649e-23       # J/K
c_LIGHT  = 2.99792458e8       # m/s
T_CMB0   = 2.7255             # K, present-day CMB temperature

def x_to_freq_GHz(x):
    """x = h*nu / (k_B*T_CMB) -> nu in GHz"""
    nu_Hz = x * k_B * T_CMB0 / h_PLANCK
    return nu_Hz / 1e9

def delta_n_to_intensity(x, delta_n):
    """|Delta I_nu| = (2 h nu^3 / c^2) * |delta_n|, same convention as the
    attached script (plots the magnitude on a log axis)."""
    nu_Hz = x * k_B * T_CMB0 / h_PLANCK
    prefac = 2.0 * h_PLANCK * nu_Hz**3 / c_LIGHT**2
    return prefac * delta_n

# ── physical parameters ──────────────────────────────────────────────────────
tau_X   = 1e9                 # [s]
GAMMA_X = 1.0 / tau_X
den_fac = 0.01                 # n_X,0/n_H,0 (McKeen 2021 benchmark)

DELTA_M_MEV = [5, 10, 20, 30, 40, 50]          # sweep grid [MeV]
DELTA_M_EV  = np.array(DELTA_M_MEV) * 1e6       # [eV]

# ── redshift range (plan Sec 3): pulled in from the attached script's
# z_start=1e7 (which sat with zero margin at the Rust binary's hard-error
# boundary, Sec 2.3, and at cascade.cosmology.cosmic_time's own domain
# break, Sec 2.2) down to a value with margin below the 5e6 soft-warning
# line, comfortably above the active window's own top (z~1.53e6 for
# tau_X=1e9 s) ──
Z_START, Z_END = 5.0e6, 5.0e2
# tabulation range must bracket the PDE walk (Sec 3's consistency
# requirement) -- z_max >= z_start, z_min <= z_end, set explicitly rather
# than relying on solve()'s defaults (z_min=1e3, z_max=3e6) to happen to
# bracket the choice above.
Z_MIN, Z_MAX = Z_END, Z_START
# N_Z=500 matches the photon channel's internal tabulation cap exactly
# (Sec 4.6) -- avoids the heat/photon channel resolution mismatch that
# would otherwise waste heat-channel resolution above 500 with no
# photon-channel benefit.
N_Z = 500

X_MAX  = 80.0     # matches the attached script's grid; fed to BOTH
                   # make_mckeen_photon_source's x_max_factor= AND every
                   # solve() call's x_max= (Sec 4.4's shared-constant
                   # pattern -- the only failure mode left is forgetting
                   # to reuse this same constant in both places)
X_MIN  = 1.5e-4
N_X    = 1000

# cosmology consistency (plan Sec 4.5): cascade.cosmology hardcodes AK19's
# own Planck values; pass the equivalent explicitly to spectroxide.solve()
# so the Python-side dq_dz/photon_source rate functions and the Rust PDE
# evolve under the same H(z)/n_H(z), rather than spectroxide's own
# (different) defaults.
COSMO_AK19 = dict(
    h=0.6688,
    omega_b=0.0221 / 0.6688**2,
    omega_m=(0.0221 + 0.1206) / 0.6688**2,
    y_p=0.24,
    t_cmb=2.726,
    n_eff=3.046,
)

TIMEOUT_S = 3600  # matches the attached script's own value (Sec 4.3/Sec 7);
                   # solve()'s own internal default is 600s, and this plan
                   # issues two PDE calls per Delta_m point instead of one.

# §5 rows 4-5: fail fast, before launching any (expensive) solve() call,
# rather than discovering a bad z-range choice from a Rust subprocess
# error 3600s into a run.
validate_pde_z_range(Z_START, Z_END, Z_MIN, Z_MAX)


# ── per-Delta_m solve: fork of fig6_spectrum_via_spectroxide.py's
#    spectrum_via_spectroxide_pde, with the two builder calls swapped for
#    the McKeen-specific ones and timeout=3600 added to both solve() calls ──
def run_one(dm_eV, force_all_heat=False):
    """
    force_all_heat=True implements plan Sec 8 item 3's regression check:
    "temporarily force f_heat=1, f_ic=f_subthreshold=0 (100% of omega_inj
    into heat, skip the photon solve() call). The result should be close
    to -- not identical to, since the attached script uses Delta_m as the
    energy scale and this forces only omega_inj -- the attached script's
    output." This isolates "did I wire the heat/photon split correctly"
    from "does the cascade module work" (already validated against AK19's
    figures separately). Not run automatically as part of the sweep --
    call run_one(dm, force_all_heat=True) by hand and diff against the
    attached script's own npz output for the same Delta_m point.
    """
    omega_inj_eV, Tn_eV = omega_inj_and_Tn_eV(dm_eV)

    # §5 row 7: every builder below hardcodes high_energy=True; confirm
    # that's actually the right solver branch for this point rather than
    # assuming it (documented boundary, not a hard physics limit -- see
    # validate_omega_inj_for_high_energy's docstring).
    if not validate_omega_inj_for_high_energy(omega_inj_eV):
        raise ValueError(
            f"omega_inj_eV={omega_inj_eV:.4g} (Δm={dm_eV/1e6:.1f} MeV) is at "
            "or below 2*m_e -- the hardcoded high_energy=True cascade solver "
            "is not validated in that regime (plan Sec 5); would need "
            "high_energy=False wired in before trusting this point."
        )

    dq_dz = make_mckeen_dq_dz(
        omega_inj_eV, GAMMA_X, den_fac, x_max_factor=X_MAX,
        force_f_heat=1.0 if force_all_heat else None,
    )

    r_heat = spectroxide.solve(
        method="pde", dq_dz=dq_dz, cosmo=COSMO_AK19,
        z_start=Z_START, z_end=Z_END,
        z_min=Z_MIN, z_max=Z_MAX, n_z=N_Z,
        x_min=X_MIN, x_max=X_MAX, n_x=N_X,
        timeout=TIMEOUT_S,
    )

    if force_all_heat:
        # Sec 8 item 3: skip the photon solve() call entirely -- 100% of
        # omega_inj is being forced into heat, so there is nothing left
        # for the photon channel to carry.
        return dm_eV, (r_heat.x, r_heat.delta_n, r_heat.mu, r_heat.y, Tn_eV)

    photon_source = make_mckeen_photon_source(omega_inj_eV, GAMMA_X, den_fac, x_max_factor=X_MAX, x_min=X_MIN)

    r_phot = spectroxide.solve(
        method="pde", photon_source=photon_source, cosmo=COSMO_AK19,
        z_start=Z_START, z_end=Z_END,
        z_min=Z_MIN, z_max=Z_MAX, n_z=N_Z,
        x_min=X_MIN, x_max=X_MAX, n_x=N_X,
        timeout=TIMEOUT_S,
    )

    # Both calls share x_min/x_max/n_x explicitly (not just relying on
    # matching defaults), but verify rather than assume before summing
    # (same safety check fig6_spectrum_via_spectroxide.py / interface.py's
    # run_cp7_check use).
    if not np.allclose(r_heat.x, r_phot.x):
        raise RuntimeError(
            "heat and photon spectroxide.solve() calls returned different "
            "x grids -- cannot sum delta_n directly."
        )

    delta_n = r_heat.delta_n + r_phot.delta_n
    mu = r_heat.mu + r_phot.mu
    y  = r_heat.y  + r_phot.y

    return dm_eV, (r_heat.x, delta_n, mu, y, Tn_eV)


# ── analytic templates (occupation-number convention), unchanged from the
#    attached script ──
def G(x):
    return x * np.exp(x) / (np.exp(x) - 1)**2

def Ysz(x):
    return G(x) * (x / np.tanh(x / 2) - 4)

def tempshift(x):
    return G(x)

def mudist(x):
    return 1.401 * G(x) * (0.4561 - 1 / x)

def brightness_temp_exact(x, delta_n):
    xc  = np.clip(np.asarray(x, dtype=float), 1e-12, None)
    n0  = 1.0 / np.expm1(np.clip(xc, None, 500.0))
    n_x = n0 + delta_n
    return xc / np.log1p(1.0 / n_x) - 1.0

def fit_mu_y_T(x, delta_n, x_lo=1e-3, x_hi=40.0):
    mask = (x > x_lo) & (x < x_hi) & np.isfinite(delta_n)
    A = np.vstack([mudist(x[mask]), Ysz(x[mask]), tempshift(x[mask])]).T
    b = delta_n[mask]
    coeffs, *_ = np.linalg.lstsq(A, b, rcond=None)
    a_mu, a_y, a_T = coeffs
    model = a_mu * mudist(x) + a_y * Ysz(x) + a_T * tempshift(x)
    return delta_n - model, a_mu, a_y, a_T


# ── load cache (resume support), unchanged pattern from the attached script ──
results = {}
if npz_file.exists():
    print(f"Found existing data at '{npz_file}', loading...")
    existing = np.load(npz_file, allow_pickle=True)
    loaded_keys = set(existing.files)
    for dm in DELTA_M_EV:
        kx, kdn, kmu, ky, kTn = (
            f"dm{dm:.6g}_{s}" for s in ("x", "delta_n", "mu", "y", "Tn")
        )
        if all(k in loaded_keys for k in (kx, kdn, kmu, ky, kTn)):
            results[dm] = (existing[kx], existing[kdn], float(existing[kmu]),
                           float(existing[ky]), float(existing[kTn]))
            print(f"  loaded Δm = {dm/1e6:.0f} MeV")

def save_results():
    flat = {"tau_X": np.array(tau_X), "den_fac": np.array(den_fac),
            "delta_m_ev": DELTA_M_EV}
    for dm, (x_arr, dn, mu_r, y_r, Tn_r) in results.items():
        flat[f"dm{dm:.6g}_x"]       = x_arr
        flat[f"dm{dm:.6g}_delta_n"] = dn
        flat[f"dm{dm:.6g}_mu"]      = np.array(mu_r)
        flat[f"dm{dm:.6g}_y"]       = np.array(y_r)
        flat[f"dm{dm:.6g}_Tn"]      = np.array(Tn_r)
    np.savez(npz_file, **flat)
    print(f"  → saved to '{npz_file}'")


# ── run all missing Δm points in parallel, same pattern as the attached
#    script (per-point try/except so one failure doesn't kill the sweep,
#    plan Sec 5's "outer sweep loop" guardrail) ──
to_run = [dm for dm in DELTA_M_EV if dm not in results]
if to_run:
    print(f"Launching {len(to_run)} runs in parallel ...")
    with ThreadPoolExecutor(max_workers=min(len(to_run), 10)) as ex:
        futures = {ex.submit(run_one, dm): dm for dm in to_run}
        for fut in as_completed(futures):
            dm = futures[fut]
            try:
                dm_done, payload = fut.result()
                results[dm_done] = payload
                print(f"  Δm = {dm_done/1e6:.0f} MeV ... done")
            except Exception as e:
                print(f"  Δm = {dm/1e6:.0f} MeV ... FAILED: {e}")
    save_results()
else:
    print("All Δm points already cached.")


if __name__ == "__main__":
    # ── colour gradient over Δm ─────────────────────────────────────────
    cmap = plt.cm.viridis
    norm = mcolors.Normalize(vmin=min(DELTA_M_MEV), vmax=max(DELTA_M_MEV))
    sm   = plt.cm.ScalarMappable(norm=norm, cmap=cmap)

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(9, 9), gridspec_kw={"height_ratios": [1.4, 1], "hspace": 0.32}
    )
    fig_sed, ax = plt.subplots(figsize=(9, 6))

    table_rows = []
    for dm_mev, dm in zip(DELTA_M_MEV, DELTA_M_EV):
        x_arr, dn, mu_r, y_r, Tn_eV = results[dm]
        c = cmap(norm(dm_mev))

        dT = brightness_temp_exact(x_arr, dn)
        ax_top.semilogx(x_arr, dT, color=c, lw=1.6, label=fr'$\Delta m={dm_mev}$ MeV')

        resid, a_mu, a_y, a_T = fit_mu_y_T(x_arr, dn)
        mask_plot = (x_arr > 0.05) & (x_arr < 20)
        ax_bot.semilogx(x_arr[mask_plot], resid[mask_plot], color=c, lw=1.4)

        table_rows.append(dict(
            delta_m_MeV = dm_mev,
            den_fac     = den_fac,
            T_n_keV     = Tn_eV / 1e3,
            mu          = mu_r,
            y           = y_r,
            a_mu_resid  = a_mu,
            a_y_resid   = a_y,
            a_T_resid   = a_T,
            resid_rms   = np.sqrt(np.mean(resid[mask_plot]**2)),
        ))

        nu_GHz = x_to_freq_GHz(x_arr)
        I_nu = delta_n_to_intensity(x_arr, dn)
        pos = I_nu > 0
        neg = ~pos
        ax.plot(nu_GHz[pos], I_nu[pos], color=c, lw=1.8, label=fr'$\Delta m={dm_mev}$ MeV')
        ax.plot(nu_GHz[neg], -I_nu[neg], color=c, lw=1.8, ls='--')

    ax_top.axhline(0, color='k', lw=0.5, ls=':')
    ax_top.set_xlim(2e-4, 50)
    ax_top.set_ylim(-2.5e-5, 0.5e-5)
    ax_top.set_xlabel(r'$x = h\nu / kT_z$', fontsize=12)
    ax_top.set_ylabel(r'$T(x)/T_{\rm CMB} - 1$', fontsize=12)
    ax_top.set_title(r'Corrected CMB spectral distortion — $\chi \to n + \gamma$  '
                      r'($\tau_X=10^9$ s, heat+photon channels split)', fontsize=11)
    ax_top.legend(fontsize=8, frameon=False, loc='lower right', ncol=2)

    ax_bot.axhline(0, color='k', lw=0.5, ls=':')
    ax_bot.set_xlim(0.05, 20)
    ax_bot.set_ylim(-3e-5, 3e-5)
    ax_bot.set_xlabel(r'$x = h\nu / kT_z$', fontsize=12)
    ax_bot.set_ylabel(r'Residual $\Delta n - (\mu M + y Y_{\rm SZ} + a_T G_{\rm bb})$', fontsize=10.5)
    ax_bot.set_title('Residual after subtracting best-fit µ/y/T-shift templates', fontsize=10.5)

    cbar = fig.colorbar(sm, ax=[ax_top, ax_bot], pad=0.02)
    cbar.set_label(r'$\Delta m$ [MeV]', fontsize=11)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(1e-4, 1e4)
    ax.set_ylim(1e-33, 1.1e-23)
    xlim = ax.get_xlim()
    pixie_sens = 5e-26
    ax.plot([3e-4, xlim[1]], [pixie_sens, pixie_sens], color='black', lw=1.5, ls=':', alpha=0.5)
    ax.annotate(fr'PIXIE sensitivity $\sim5\,\mathrm{{Jy\,sr^{{-1}}}}$',
                xy=(3e-2, pixie_sens), xytext=(1e-4, pixie_sens), fontsize=9, color='black')
    ax.set_xlabel(r'Frequency, $\nu$ [GHz]', fontsize=12)
    ax.set_ylabel(r'Intensity, $|\Delta I_\nu|$ [J s$^{-1}$ m$^{-2}$ Hz$^{-1}$ sr$^{-1}$]', fontsize=12)
    ax.set_title(r'Corrected spectral distortion intensity — $\chi \to n + \gamma$'
                 r'  (solid = positive, dashed = negative)', fontsize=11)
    ax.legend(fontsize=9, frameon=False, loc='best')
    ax.grid(True, which='both', alpha=0.15)
    fig_sed.colorbar(sm, ax=ax, pad=0.02, label=r'$\Delta m$ [MeV]')

    fig.savefig(save_path / "sweep_deltam_mckeen_corrected_spectrum.png", dpi=150)
    fig_sed.savefig(save_path / "sweep_deltam_mckeen_corrected_sed.png", dpi=150)
    plt.show()

    def print_table(rows, columns):
        sep = "+" + "+".join("-" * (w + 2) for _, _, _, w in columns) + "+"
        header = "| " + " | ".join(f"{h:^{w}}" for _, h, _, w in columns) + " |"
        print(sep); print(header); print(sep)
        for row in rows:
            line = "| " + " | ".join(
                f"{format(row[k], fmt):>{w}}" for k, h, fmt, w in columns
            ) + " |"
            print(line)
        print(sep)

    columns = [
        ("delta_m_MeV", "Δm [MeV]",      ".0f",  9),
        ("den_fac",     "n_X0/n_H0",     ".3e", 11),
        ("T_n_keV",     "T_n [keV]",     ".4f", 11),
        ("mu",          "μ",             ".3e", 11),
        ("y",           "y",             ".3e", 11),
        ("a_mu_resid",  "a_μ (resid)",   ".3e", 13),
        ("a_y_resid",   "a_y (resid)",   ".3e", 13),
        ("a_T_resid",   "a_T (resid)",   ".3e", 13),
        ("resid_rms",   "resid RMS",     ".3e", 11),
    ]

    print(f"\nCorrected photon case (χ → n + γ)   τ_X = {tau_X:.0e} s   "
          f"n_X,0/n_H,0 = {den_fac}\n")
    print_table(table_rows, columns)
