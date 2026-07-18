# McKeen 2021 (χ → n + γ) cascade-corrected spectrum — implementation plan v3

**Supersedes `mckeen2021_cascade_implementation_plan_v2.md`.** Scope is unchanged from v2
(Tier 2 / `src/cascade.rs` / `cascade_photon_source.rs` stay dropped; only
`Cascades/cascade/*.py` and the public `spectroxide.solve(...)` API are touched). This
revision does **not** re-derive v2 from scratch — every numeric claim in v2 was re-checked
directly against the code in this repo (by reading the relevant source, and, where the
computation is pure Python, by actually running it), and v2 held up almost entirely. What
changed here are the handful of places that didn't:

| # | What changed | Where |
|---|---|---|
| 1 | **Corrected**: the `z_start` hard-error (>1e7) / soft-warning (>5e6) guardrail is enforced by the **Rust binary** (`src/solver.rs::SolverConfig::validate()`/`soft_warnings()`), not by Python's `_validation.py::warn_z_max_regime` — and that Python function doesn't even run on the code path this plan uses (see §2.3) | §2.3 (new), §5 |
| 2 | **Added**: a genuine physics-validity boundary on Δm itself (hadronic threshold, Δm ≳ m_π⁰) that v2's guardrail table omitted even though the *superseded* pre-v1 plan had flagged it | §5 |
| 3 | **Made explicit**: §4.1's "swap the rate line" instruction also requires deleting the `f_X_for_target_all_heat_y(z_X, target_y_all_heat)` call and its `z_X`/`target_y_all_heat` parameters entirely — v2's prose only mentioned the rate line, and left this implicit in a code snippet | §4.1 |
| 4 | **Added**: explicit `timeout=` on every `solve()` call — default is 600 s, this plan now issues two PDE calls per Δm point instead of one, and the original script itself used 3600 s | §4.3, §7 |
| 5 | **Added**: a note that `production_grid=True`/`number_conserving=True` don't need to be passed explicitly — the defaults already match what the original script set explicitly, so dropping them (as the forked skeleton does) is not a silent regression | §4.7 |
| 6 | **Added**: a minor z-grid tabulation asymmetry between the two channels worth knowing about (not a bug, a shared-cache caveat) | §4.6 |
| 7 | **Corrected wording**: v2 called `DecayingParticle`'s `f_x` "joules per baryon" — the Rust struct's own doc-comment and the conversion line both say **eV** (converted to joules internally); no numeric impact, just a units label | §0 |

Everything else below — the kinematics, the two real bugs (§2.1, §2.2), the redshift-window
analysis (§3), the heat/photon split (§4.2), the already-correct two-`solve()`-calls pattern
(§4.3/4.4), the cosmology mismatch (§4.5), the caching requirement (§4.6), the production-
resolution knobs (§4.7), the file layout (§7), and the known approximations (§6) — was
independently re-verified against the actual repo (not just re-read from v2) and is carried
forward unchanged in substance. Where a number is quoted, it was recomputed here directly
from `cascade/*.py`, not copied from v2.

**Short answer to "do you agree with it": yes, on everything that's checkable by reading or
running the code — v2's kinematics table, both bug reports, and every "already correct,
reuse as-is" claim reproduced exactly on direct execution. The seven items above are the
corrections/additions; none of them overturn v2's core plan, but #1 and #2 are worth reading
before you write the guard code, since #1 changes *which layer* is actually responsible for
catching a bad `z_start`, and #2 is a validity boundary v2 genuinely dropped.**

---

## 0. What the attached script actually does, and why it's only an approximation

`sweep_deltam_photon_tau1e9.py` calls `spectroxide.solve()` with
`injection={'type': 'decaying_particle', 'f_x': F_X, 'gamma_x': GAMMA_X}` and
`F_X = Δm · den_fac`. Tracing this into `src/energy_injection.rs` (`InjectionScenario::
DecayingParticle`'s `heating_rate()` arm, confirmed by direct read):

```rust
InjectionScenario::DecayingParticle { f_x, gamma_x } => {
    let f_x_joules = f_x * EV_IN_JOULES;   // f_x is in eV, not joules
    f_x_joules * gamma_x * n_h * (-gamma_x * t).exp() / rho_gamma
}
```

This is the *only* place `DecayingParticle` is consumed anywhere in `src/` — it feeds
`heating_rate()` alone; there is no photon channel in this injection type at all. So the
script is implicitly assuming the entire mass splitting Δm, for every χ decay, thermalizes
as heat.

Physically, for χ → n + γ, a single decay releases a photon of energy `ω_inj` (MPR21 Eq. 7)
plus a recoiling neutron of kinetic energy `T_n = Δm − ω_inj`. The photon does not instantly
become "heat" — it Compton-scatters, pair-produces, and inverse-Compton-upscatters through
the plasma first (that redistribution is exactly what the validated `cascade` module
computes), and only part of its energy ends up as smooth heating; the rest stays as a
photon-number/spectral perturbation (the μ/y-type distortion channel). The fix: split the
injected energy into a heat channel and a photon/distortion channel using
`solve_burst.burst_energy_budget`, and feed each into `spectroxide.solve()` separately.

---

## 1. χ → n + γ kinematics (verified by direct computation, unchanged from v2)

From `cascade/mpr21.py`, MPR21 Eq. 7 (χ at rest, two-body decay):

```
ω_inj(m_χ) = (m_χ / 2) · (1 − (m_n / m_χ)²)      # photon energy, χ rest frame
T_n        = Δm − ω_inj                            # neutron recoil KE, Δm = m_χ − m_n
```

Recomputed directly from `mpr21.omega_inj_MeV` (`M_N_MEV = 939.565`; the attached script
uses `939.565379`, a 0.4 keV difference, negligible — pick one and use it consistently
rather than mixing the two source files' constants):

| Δm [MeV] | ω_inj [MeV] | T_n [keV] | T_n / Δm |
|---:|---:|---:|---:|
| 5  | 4.9868  | 13.23  | 0.265% |
| 10 | 9.9473  | 52.66  | 0.527% |
| 20 | 19.7916 | 208.43 | 1.042% |
| 30 | 29.5359 | 464.13 | 1.547% |
| 40 | 39.1833 | 816.69 | 2.042% |
| 50 | 48.7368 | 1263.18 | 2.526% |

**The attached script's `injection['f_x'] = Δm · den_fac` uses Δm.** It should use
`ω_inj · den_fac` as the energy scale for the photon channel. `T_n` is not injected into the
photon bath by any channel in this codebase — that's a deliberate, documented MPR21-level
approximation (§6), not an oversight; it's sub-3% across this whole grid.

**New in this revision — the kinematics formula's own validity ceiling.** `ω_inj_MeV` is a
pure two-body-decay formula; it silently returns a numerically plausible answer for *any*
`m_chi_MeV` you give it, including ones where the physics it encodes has stopped applying.
MPR21 themselves note that once `m_χ > m_n + m_π⁰` (i.e. `Δm ≳ m_π⁰ ≈ 135` MeV), hadronic
final states open up "with similar cosmological signatures" that this two-body-only formula
does not describe — a real physical branch point, not a numerical edge case. This is called
out explicitly as a non-goal in this repo's own *superseded* pre-v1 plan
(`em_cascade_implementation_plan-old.md`, §8: *"Hadronic final states (m_χ > m_n + m_π⁰) —
out of scope"*), but v2 dropped it from the guardrail table when it rewrote §5. Nothing in
`mpr21.py` enforces this bound — add it explicitly (§5).

For this specific sweep (Δm = 5–50 MeV) it's moot — you're a factor of 3–27 below the
threshold — but since a "sweep" script is exactly the kind of code that gets its grid
extended later, guard it now rather than rediscover it as a silent physics error if someone
pushes Δm up toward 100+ MeV.

---

## 2. Two real bugs found while tracing this (both still present, both matter)

### 2.1 `f_X_baryon_normalized` double-counts `ω_inj`

`mpr21_energy_injection_rate_eV_cm3_s(z, f_X, Γ_X, ω_inj_eV)` computes

```
dE/dV/dt = ω_inj_eV · f_X · Γ_X · n_H(z) · exp(−Γ_X t(z))
```

i.e. it expects `f_X = n_{χ,0}/n_{H,0}` (a pure number ratio — its own docstring says so).
But `f_X_baryon_normalized(ω_inj, m_χ, n_χ0/n_H0)` returns `epsilon_X · n_χ0/n_H0` where
`epsilon_X = ω_inj/m_χ` — i.e. it *pre-multiplies* by `ω_inj/m_χ`. Feeding that into the rate
function above double-counts the energy factor and suppresses the rate by an extra factor of
`epsilon_X ≈ ω_inj/m_χ`. Re-verified numerically (not copied from v2):

```
Δm=5 MeV:  epsilon_X = 0.0052794  →  rate suppressed ~189.4×
Δm=50 MeV: epsilon_X = 0.0492508  →  rate suppressed ~20.3×
```

`mpr21.run_benchmark()` (and hence `test_cp8_mpr21.py`) calls `f_X_baryon_normalized` and is
affected, but that test only asserts `0.01 < y/y_COBE < 10` (a **3-orders-of-magnitude**
window) and a monotonicity check — both loose enough that a 20–190× error doesn't necessarily
fail them (confirmed by reading the test file directly: `test_distortion_amplitude_
comparable_to_cobe_sensitivity` and `test_amplitude_scales_with_delta_m` are the only
numeric assertions in that file). **Do not call `f_X_baryon_normalized` or `run_benchmark`
from the new code.** Pass `den_fac` (i.e. `n_{χ,0}/n_{H,0}`) directly as the `f_X` argument of
`mpr21_energy_injection_rate_eV_cm3_s`.

### 2.2 `cascade.cosmology.cosmic_time(z)` silently returns the wrong sign above z ≈ 1e7

```python
val, _ = integrate.quad(integrand_log, np.log1p(z), np.log1p(1.0e7), limit=500, epsrel=1e-10)
```

The upper integration bound is hardcoded to `z = 1e7` (used as a "t ≈ 0" reference point).
For `z > 1e7` the bounds invert and the function returns a negative time with no warning or
exception. Reproduced by direct execution in this environment:

```
cosmic_time(1e6)   =  2.3578593897e7 s   (correct)
cosmic_time(1e7)   =  0.0 s              (reference point, correct by construction)
cosmic_time(1.5e7) = -1.32448e5 s        (wrong sign, no error)
cosmic_time(5e7)   = -2.28876e5 s        (wrong sign, no error)
```

This is exactly the failure mode you said you want to avoid: not a crash, a plausible-looking
wrong number. For this τ_X = 1e9 s / Δm = 5–50 MeV sweep the magnitude is harmless
(`|t| ≪ τ_X` near z = 1e7, so `exp(−Γ_X t)` barely moves), but guard `z ≤ 1e7` explicitly
wherever the new code passes a redshift into anything under `cascade.cosmology` or
`cascade.solve_burst` (§5). Note: this is a *different* issue from the `IntegrationWarning`
fix already applied to this same line (the `limit=500, epsrel=1e-10` you see above is the
already-fixed version, per `Cascades/README.md` item 20) — that fix improved numerical
precision of the integral; it does nothing about the hardcoded `1e7` upper bound, which is a
domain problem, not a precision problem.

### 2.3 New: what actually enforces the z_start ceiling, and where (correction to v2)

v2 attributed the PDE's `z_start` hard-error/soft-warning to
`spectroxide._validation.py::warn_z_max_regime`. That's not quite right, and the distinction
matters for where you'd look if a run ever fails or behaves unexpectedly:

- `warn_z_max_regime(z_max)` is real, and its thresholds (`>1e7` → warning, `>5e6` → milder
  warning) do match what v2 quoted — but grepping its call sites shows it is only invoked
  from `greens.py` (four call sites, all inside the analytic Green's-function / table code
  path), and it is keyed on `z_max`, a Python-side `warnings.warn` (non-fatal either way).
  **It is never called anywhere in the `method="pde"` + `dq_dz`/`photon_source` path this
  plan uses.**
- The actual guardrail for that path — a **hard error** at `z_start > 1e7` and a **soft
  warning** at `z_start > 5e6` — lives in the **Rust binary**, `src/solver.rs`'s
  `SolverConfig::validate()` / `soft_warnings()`:

  ```rust
  if self.z_start > 1.0e7 {
      return Err(format!(
          "z_start={:.1e} implies theta_e ~ {:.1e}; Kompaneets Fokker-Planck \
           approximation is invalid for theta_e > ~0.005. Max z_start ~ 1e7.", ...));
  }
  ```

  This `Err` surfaces to Python as a `RuntimeError` from `_run_rust_binary` (non-zero exit
  code, stderr captured into the message) — a real exception you can `try/except` around, not
  a hang and not a silent wrong number, confirmed by reading `solver.py::_run_rust_binary`
  directly.

Practical upshot: your Python-side `try/except Exception: return 0.0` wrapper around each
`dq_dz`/`photon_source` closure (§5, carried over from `interface.py`) protects the *tabulation*
step (building the CSV that gets handed to the Rust binary). It does **not**, and cannot,
protect against a bad `z_start`/`z_max`/`z_end` choice on the `solve()` call itself — that's a
one-time argument to the whole PDE run, not something queried per-(x,z) inside the callables,
and the Rust binary's own `validate()` is what actually catches it. Keep `z_start ≤ 5e6` (§3)
as your operating rule; the Rust binary is your safety net if that's ever violated, not the
Python validators.

---

## 3. Where the attached script's redshift range actually sits

`build_zinj_grid(τ_X)` (bisects `cosmic_time(z) = t_min_factor·τ_X` and
`t_max_factor·τ_X`, defaults `1e-2`/`30`) gives, for τ_X = 1e9 s:

```
active window:  z ∈ [2.76e4, 1.53e6]     (t = 1e7 s  to  3e10 s)
z at t = τ_X:   z ≈ 1.538e5   (confirmed by direct bisection: 1.53832e5)
```

This matches the attached script's `z_start=1e7` — but that's the *cosmic time in seconds*
grid point (`build_zinj_grid`'s own lower `t` bound), not literally `z=1e7`; the script's
`z_start=1e7` is a redshift, comfortably above the z≈1.53e6 top of the active window, so the
original run was not truncating early. Keep this in mind since it's easy to conflate
"z_start" and "t_min_factor·τ_X" — they're both `1e7` here by coincidence of units, not
because one determines the other.

The Rust binary hard-errors above `z_start > 1e7` and soft-warns above `z_start > 5e6` (§2.3)
— the attached script's `z_start=1e7` sits exactly at that hard-error boundary with zero
margin, and `cascade.cosmology.cosmic_time` independently breaks at the same value (§2.2).
Two unrelated parts of the stack agreeing on the same ceiling is a useful cross-check, but it
means you should pull `z_start` in rather than run right at the wall:

```
z_start ≈ 3e6 – 5e6   (below the 5e6 soft-warning line, ~2× margin above the z≈1.53e6
                        active-window top)
z_end   ≈ 500 – 1e3   (comfortably below the z≈2.76e4 active-window bottom)
```

**Consistency requirement:** with the tabulated `dq_dz`/`photon_source` route (§4.3),
`spectroxide.solve()` also takes separate `z_min`/`z_max`/`n_z` arguments (defaults
`1e3`/`3e6`/`5000`) that control the grid the callables get *sampled* on before being handed
to the Rust PDE — independent of `z_start`/`z_end`. If `z_start` exceeds `z_max`, the PDE will
query the callables outside their tabulated range. Set `z_max ≥ z_start` and `z_min ≤ z_end`
explicitly on every `solve()` call rather than relying on the defaults to happen to bracket
whatever `z_start`/`z_end` you chose.

---

## 4. The fix, task by task

| § | Item | Status |
|---|---|---|
| 4.1 | MPR21-convention rate function | **New** — nothing McKeen-specific exists yet |
| 4.2 | Heat/photon energy split via `burst_energy_budget` | **New** wiring (split function itself validated; McKeen-specific call is new) |
| 4.3 | Two `solve()` calls, summed | **Already correct** — `fig6_spectrum_via_spectroxide.py`, reuse verbatim (plus add `timeout=`, see below) |
| 4.4 | `x_max`/`x_cut` grid consistency | **Already correct** — same file, reuse verbatim |
| 4.5 | Cosmology (`cosmo=`) consistency | **New** — grepped, not passed anywhere in the repo today |
| 4.6 | Caching `burst_energy_budget` | **New** — grepped, no `lru_cache`/memoization anywhere today |
| 4.7 | Production-resolution knobs | **New** (config only, no code); defaults already match original script's `production_grid=True` |

### 4.1 New rate function — and what to remove, not just what to swap

`interface.py`'s existing `make_dq_dz`/`make_photon_source` call
`sb.decay_injection_rate_eV_cm3_s(z, f_X, τ_X_s)`, where `f_X` is a **dark-matter
mass-density fraction**, solved via `f_X = sb.f_X_for_target_all_heat_y(z_X,
target_y_all_heat)` — AK19's "all-heat" y-target benchmark convention. It is not the MPR21
convention: χ → n + γ only releases `ω_inj ≪ m_χ` per decay, and the natural parameter is
`n_{χ,0}/n_{H,0}` (`den_fac`), a fixed physical input, not something solved for from a target
distortion amplitude.

**Two changes are needed, not one** (v2's prose only stated the first; the function
signature it gave in §7 implied the second but never said so explicitly):

1. Swap the rate line:
   ```python
   # interface.py's version (AK19 all-heat convention — wrong for this model):
   rate_eV_cm3_s = sb.decay_injection_rate_eV_cm3_s(z, f_X, tau_X_s)
   # new version (MPR21 number-ratio convention — correct for chi -> n + gamma):
   rate_eV_cm3_s = mpr21.mpr21_energy_injection_rate_eV_cm3_s(z, den_fac, Gamma_X_s, omega_inj_eV)
   ```
2. **Delete the `f_X_for_target_all_heat_y(z_X, target_y_all_heat)` call and its `z_X`/
   `target_y_all_heat` parameters entirely.** `den_fac` is a direct input to the new builder
   functions (§7's `make_mckeen_dq_dz(omega_inj_eV, Gamma_X_s, den_fac, ...)` signature has no
   `z_X`/`target_y_all_heat` at all) — there is nothing left to solve for. If you copy
   `interface.py`'s function bodies and only touch the rate line, you'll be left with dead
   `z_X`/`target_y_all_heat` parameters and an `f_X` that's still being computed the AK19 way
   before being silently shadowed — delete that machinery, don't just stop calling it.

Everything downstream of the rate line in `interface.py` — `dt_dz`, the
`burst_energy_budget(...)` call, `f_heat`/`f_ic`/`f_subthreshold` extraction, the `x_eff`
log-compression placement, the `sigma_x` width cap, the `G2_0/x²` conversion, the
`np.isfinite` guards and try/except wrapping — is convention-agnostic and already reviewed
(`cascade_analysis_review_v2.md`, `cascade_corrections_for_fig6.md`); reuse verbatim.

**Naming note:** `f_X` means three different things across this codebase (a DM
mass-fraction in `interface.py`/AK19's benchmark, a baryon-number-ratio in
`mpr21.f_X_baryon_normalized`, and MPR21's own `f_chi` DM-fraction in their anisotropy bound
— `mpr21.py`'s own docstring flags this collision). To avoid re-creating the confusion in the
new bridge module, don't call the new function's parameter `f_X` at all — call it `den_fac`
throughout (matching the attached script's own name for it), so it's never ambiguous which
convention is in play.

### 4.2 Two energy channels, not one

```python
budget = sb.burst_energy_budget('gamma', omega_inj_eV, z, high_energy=True)
# budget['f_heat'], budget['f_ic'], budget.get('f_subthreshold', 0.0)  — sum to 1.0000
```

`dq_dz(z)` uses `budget['f_heat']`; `photon_source(x, z)` uses
`budget['f_ic'] + budget.get('f_subthreshold', 0.0)`. Re-verified by direct execution at
`z ≈ 1.53832e5` (the `t = τ_X` point for this sweep, τ_X = 1e9 s):

| Δm [MeV] | f_heat | f_ic | f_subthreshold | sum |
|---:|---:|---:|---:|---:|
| 5  | 0.43586 | 0.56394 | 2.002e-4 | 1.0000 |
| 50 | 0.29622 | 0.70376 | 2.051e-5 | 1.0000 |

**Headline physical result:** only ~30–44% of the injected energy ends up as smooth heating;
the rest stays as a photon-number/spectral perturbation. The attached script's 100%-heat
approximation should therefore visibly *overpredict* the Y_SZ/temperature-shift-like
component and *underpredict* the μ/y-type component relative to the corrected version — the
qualitative check to look for once you run both.

### 4.3 Two `spectroxide.solve()` calls per Δm point, summed — already correct, reuse as-is

`solve()`'s PDE dispatch is a strict if/elif/elif chain, confirmed by direct read of
`solver.py` (~line 1602): `injection is not None` → PDE injection path; `elif dq_dz is not
None` → tabulated-heating path; `elif photon_source is not None` → tabulated-photon path.
Passing both `dq_dz=` and `photon_source=` in one call silently drops `photon_source`. The
fix — call `solve()` once with `dq_dz=` and once with `photon_source=`, check the returned
`x` grids match (`np.allclose`) before summing, then sum `delta_n`/`mu`/`y` directly (exact at
this order: linear superposition of two independent injections into the same PDE at
`Δρ/ρ ~ f_X·ω_inj·den_fac ≪ 1`) — is already written and correct, in
`fig6_spectrum_via_spectroxide.py::spectrum_via_spectroxide_pde`. No new code needed here;
fork this function body directly.

**One addition:** pass `timeout=3600` (matching the attached script's own value) explicitly
to both `solve()` calls. `solve()`'s internal default is 600 s
(`kwargs.pop("timeout", 600.0)`, confirmed by reading `solver.py`); this plan doubles the
per-Δm PDE cost (two calls instead of one), and §4.7's production-resolution pass raises the
*cascade budget* cost independently of the PDE's own cost — a 600 s ceiling has less margin
than it did for the original single-call script. A timeout here raises a catchable
`subprocess.TimeoutExpired` (not a silent truncation), but there's no reason to let it fire
when the original script's own choice of 3600 s is right there to reuse.

### 4.4 `x_max`/`x_cut` consistency — already fixed, reuse as-is

`photon_source`'s narrow line is placed at `x_eff`, log-compressed to asymptote at
`x_cut = x_max_factor`. If `x_max_factor` (photon builder) and `x_max` (the `solve()` call
for the photon channel) disagree, the line can sit almost entirely outside the actual PDE
grid and its energy is lost. `fig6_spectrum_via_spectroxide.py` already closes this with a
single shared constant, `X_MAX_SPX = 35.0`, passed to *both* `spx_solve()` calls' `x_max=`
*and* `make_photon_source`'s `x_max_factor=`. Carry the same one-constant-in-lockstep pattern
into the new script — the attached script used `x_max=80.0`; if you keep that,
`X_MAX_SPX = 80.0` feeds both places. The only failure mode left is *forgetting* to reuse the
shared constant when copying the function body.

### 4.5 Cosmology consistency

`cascade/cosmology.py` hardcodes AK19's own Planck values: `h=0.6688, Ω_b h²=0.0221,
Ω_c h²=0.1206` (→ `Ω_m ≈ 0.31907`), `Y_p=0.24`, `T_CMB0=2.726`, `N_eff=3.046`.
`spectroxide.Cosmology`'s default (`Cosmology.default()`, confirmed by reading
`spectroxide/cosmology.py`) is `h=0.71, Ω_b=0.044, Ω_m=0.26, Y_p=0.24, T_cmb=2.726,
N_eff=3.046` — a 6% difference in `h` and ~19% difference in `Ω_m`. Since the new
`dq_dz`/`photon_source` callables compute rates using `cascade.cosmology`'s `H(z)`/`n_H(z)`/
`t(z)` while the Rust PDE solver evolves the spectrum using its own, differently-parametrized
`H(z)`, pass an explicit `cosmo=` to every `spectroxide.solve()` call:

```python
cosmo_ak19 = dict(h=0.6688, omega_b=0.0221/0.6688**2, omega_m=(0.0221+0.1206)/0.6688**2,
                   y_p=0.24, t_cmb=2.726, n_eff=3.046)
```

Confirmed against `spectroxide/cosmology.py`'s `Cosmology.to_dict()` directly: the accepted
keys are exactly `h, omega_b, omega_m, y_p, t_cmb, n_eff` — the dict above matches field for
field (`n_eff` defaults to 3.046 in both codebases anyway, so it's optional but harmless to
include explicitly). `solve(cosmo=...)` accepts a plain dict or a `Cosmology` instance
(converted via `.to_dict()` internally) — either works.

### 4.6 Mandatory: cache `burst_energy_budget` per (species, ω_inj, z)

Directly measured in this environment: `burst_energy_budget('gamma', ω_inj_eV, z,
high_energy=True)` at `z ≈ 1.5e5`, `ω_inj` in the 5–49 MeV range, takes ~0.18–0.5 s per call
(recursive photon/electron cascade walk). `dq_dz(z)` calls it once per z — `n_z` (a few
hundred) points × ~0.3 s ≈ 1–3 minutes, fine. `photon_source(x, z)`, as written in
`interface.py`, calls it once per `(x, z)` pair even though the budget doesn't depend on `x`
at all. Confirmed by reading `spectroxide/solver.py::_run_tabulated_photon` directly: it
builds its CSV with a nested loop, `for z_val in z_grid: for xi in x_grid: photon_source(xi,
z_val)`, where `z_grid = np.logspace(..., min(n_z, 500))` — **capped at 500 regardless of the
`n_z` you pass.** Unmemoized, that's up to `500 × n_x` (e.g. 1000) calls × ~0.3 s ≈ tens of
hours for a single Δm point.

Fix: wrap the budget lookup in an `lru_cache` keyed on `(species, omega_inj_eV, z)`:

```python
from functools import lru_cache

@lru_cache(maxsize=None)
def _cached_budget(species, omega_inj_eV, z):
    return sb.burst_energy_budget(species, omega_inj_eV, z, high_energy=True)
```

Share one `_cached_budget` between `dq_dz` and `photon_source` for a given Δm point — the
~n_z calls needed for the heat channel also warm the cache for the photon channel, since both
query the same `(species, ω_inj_eV, z)` pairs.

**One asymmetry worth knowing about** (not a bug, but easy to misread): `_run_tabulated_
heating` (the `dq_dz` path) uses the *full* `n_z` you pass for its z-grid — no 500-point cap
— while `_run_tabulated_photon` always caps at 500. If you set `N_Z` above 500 (the global
`solve()` default is 5000), the heat channel gets tabulated at a finer z resolution than the
photon channel ever will be, for no benefit — the photon channel's *input* z-grid is
structurally capped at 500 regardless. Setting `N_Z = 500` (as this plan's §7 constants do)
sidesteps the asymmetry entirely by matching the cap exactly; if you ever raise `N_Z` for the
heat channel specifically, know that the photon channel silently won't follow.

### 4.7 Production-resolution knobs

Per `Cascades/README.md` (confirmed by direct read, exact line numbers match), the cascade's
own resolution defaults are dev-speed, not production:

| function | parameter | current default | production value (README) |
|---|---|---|---|
| `electron_cascade` (internal `n_decades` multiplier, hardcoded, not a kwarg) | multiplier in `n_decades = max(4, ceil(log10(E0/E_min)*4))` | `4` | `60` |
| `photon_cascade_full` | `max_scatters` | `2000` | `20000` |
| `photon_cascade_full` / `electron_cascade` | `max_depth` | `60` | `120` |

These only affect the *resolution* of the cascade energy-budget calculation (already
verified above to conserve energy to 1.0000 at these defaults for Δm=5 and 50 MeV), not
correctness — but for a plot you intend to treat as final, edit the source (these are
hardcoded literals inside `solve_burst.py`, not `solve_burst.*` kwargs you can pass in) to the
README's production values and expect roughly `60/4 × 20000/2000 × 120/60 ≈ 300×` slower
`burst_energy_budget` calls (the measured 0.2–0.5 s/call above is already at the fast
defaults). Do a default-resolution sweep first for plumbing/shape, then re-run at production
resolution — §4.6's caching is what makes the second pass tractable at all.

**No change needed for `production_grid`/`number_conserving` on the `spectroxide.solve()`
side** (distinct from the cascade-resolution knobs above): `solve()`'s own defaults already
match what the attached script set explicitly. `number_conserving` defaults to `True`
internally (`kwargs.pop("number_conserving", True)`, confirmed in `solver.py`), and
`production_grid` is `True` whenever `debug=False` (`solve()`'s own default) via the
`PRODUCTION` quality preset — confirmed by reading the preset dict directly
(`"production_grid": True` under `PRODUCTION`, `False` under `DEBUG`). So simply *not* passing
either kwarg in the forked script (as `fig6_spectrum_via_spectroxide.py`'s pattern does)
reproduces the attached script's explicit `production_grid=True, number_conserving=True` —
not a regression, just implicit. (`fig6_spectrum_via_spectroxide.py` itself passes
`debug=True` to get *faster* dev-resolution PDE runs for that figure's purposes — don't copy
that particular flag into the mckeen script unless you deliberately want the same speed/
accuracy trade for a first smoke-test pass.)

---

## 5. Validity ranges and crash-avoidance guardrails

None of the functions below raise on bad input by default — they either return a nonsense
number silently or, in a couple of cases, run very slowly without limit. Add explicit guards
in the new bridging code rather than relying on the sweep grid always being "reasonable":

| Quantity | Valid range | What happens outside it | Guard to add |
|---|---|---|---|
| `Δm = m_χ − m_n` | `Δm > 0` (else decay is kinematically forbidden) | `omega_inj_MeV` returns a negative number; propagates as a negative "energy" into the cascade code with no error | `if delta_m_eV <= 0: raise ValueError(...)` before anything else |
| `Δm` (upper end, **new in v3**) | `Δm ≲ m_π⁰ ≈ 134.98` MeV — above this, hadronic final states open (MPR21's own note; two-body `ω_inj` formula no longer describes the physics) | `omega_inj_MeV` still returns a numerically valid-looking two-body answer; nothing flags that the *physical scenario* has changed | `if delta_m_eV >= 1.3e8: raise ValueError("Δm above the hadronic threshold (~m_pi0); two-body chi->n+gamma kinematics no longer applies")` — or at minimum a `warnings.warn` if you want to explore Δm beyond this in future work |
| `z` passed into `cascade.cosmology.*` or `solve_burst.*` | `z ≤ 1e7` (hardcoded reference point in `cosmic_time`) | silently returns a wrong-signed time, no exception (§2.2) | clamp/assert `z ≤ 1e7` (recommend `≤ 5e6`, see §3) at the top of every new `dq_dz`/`photon_source` closure |
| `z_start` (PDE `solve()` call) | `≤ 5e6` soft, `≤ 1e7` hard — enforced by the **Rust binary** (`src/solver.rs`, §2.3), not a Python pre-check | `>1e7`: Rust returns `Err`, surfaces as a Python `RuntimeError` (catchable, not a hang); `5e6`–`1e7`: Rust logs a soft warning, ~1% systematic | keep `z_start ≤ 5e6` (§3) as the operating rule; you can `try/except RuntimeError` around the whole `solve()` call as a last-resort net, but the real fix is just not choosing `z_start` above the line |
| `z_min`/`z_max` (tabulation) vs `z_start`/`z_end` (PDE walk) | tabulation range must bracket the PDE range | undefined/extrapolation behavior if not (not independently verified here — treat as untested) | always set `z_max ≥ z_start`, `z_min ≤ z_end` explicitly on every `solve()` call (§3) |
| `n_z` for the photon channel specifically | tabulated at `min(n_z, 500)` regardless of what you pass (§4.6) | no error — just wasted heat-channel resolution above 500 with no photon-channel benefit | set `N_Z = 500` for both channels to avoid the mismatch, not a hard requirement but avoids confusion |
| `ω_inj` (photon energy fed to the cascade) | designed/verified for ~100 keV – 50+ MeV (`rates.pair_production_xsec`'s own docstring names this range explicitly, matching MPR21's parameter space) | below `2·m_e = 1.022 MeV`: `high_energy` auto-detect correctly switches to the low-energy branch, not a crash; well above ~50 MeV: asymptotic formula, not independently re-verified here | force `high_energy=True` explicitly for this sweep (all points are 4.99–48.7 MeV, safely above threshold) rather than relying on auto-detect; if you extend Δm down near/below ~2 MeV, double-check the low-energy branch's own docstring before trusting it |
| `E0_eV` fed to `photon_cascade_full`/`electron_cascade` | `E0_eV > E_min_eV` (default `1000` eV) for any actual cascade processing | `E0_eV ≤ E_min_eV`: returns `f_subthreshold=1.0` cleanly, no crash — correct-ish, just means the injected energy never gets a heat/photon split | none needed — already safe; very small Δm (sub-keV `ω_inj`) shows up as 100% `f_subthreshold`, effectively invisible to both channels |
| `n_decades` multiplier, `max_scatters`, `max_depth` recursion caps | see §4.7 | capped, so no infinite recursion/hang regardless of input — just less accurate at low settings | none needed for stability; §4.7 covers accuracy |
| `f_X_baryon_normalized`, `mpr21.run_benchmark` | — | known-wrong by up to ~190× (§2.1) | don't call these from new code |
| every `dq_dz`/`photon_source` closure | any `z`, any `x` `spectroxide` chooses to query | `interface.py`'s existing closures already wrap the whole body in `try/except Exception: return 0.0`, and check `np.isfinite` before returning | keep this pattern in the new builders — it's what stops one bad numerical corner (e.g. a `scipy.integrate.quad` convergence failure at some odd `z`) from crashing the whole PDE tabulation step. Note this does **not** cover a bad `z_start`/`z_max` choice on the `solve()` call itself (§2.3) — that's a separate, one-time argument, guarded by the Rust binary, not by these per-point closures |
| the outer sweep loop (per-Δm) | — | one Δm point failing shouldn't kill the other five | wrap each `run_one(dm)` in try/except and report `FAILED` per-point, same pattern the attached script already uses for its `ThreadPoolExecutor` loop — keep it |

---

## 6. Known, deliberate approximations — document these on the plot, don't hide them

1. **Neutron recoil energy `T_n` is not injected anywhere.** MPR21's own treatment assumes
   the neutron is EM-inert at leading order; this codebase has no channel for a free neutron
   to deposit its ~13 keV–1.3 MeV of kinetic energy into the photon bath (e.g. via elastic
   n–p scattering feeding the Compton-coupled baryon–photon fluid). `T_n/Δm` is 0.27%–2.5%
   across this grid (§1) — small, but it means `dq_dz + photon_source` together conserve
   `ω_inj`, not `Δm`. Worth a one-line note on the final plot/table rather than a silent gap.
2. **The photon channel's spectral shape is a proxy, not a resolved cascade spectrum.**
   `solve_burst.py` (and everything built on it) tracks only scalar energy fractions
   (`f_heat`/`f_ic`/`f_subthreshold`), never an actual `dN/dE` spectrum.
   `interface.make_photon_source`'s docstring says this directly: it deposits the photon
   channel's energy as a single narrow (Gaussian, ~10% width) line at one characteristic
   `x_eff(z)`, not a resolved continuum. For this Δm range the practical impact is limited —
   `f_subthreshold` (the bucket most sensitive to the shape-proxy's placement) is negligible
   here (2.0e-4 at Δm=5 MeV down to 2.1e-5 at Δm=50 MeV, §4.2) — so the *amplitude* split
   (f_heat vs. photon-channel total) should be reasonably solid, but the SED plot's shape in
   the region where the photon channel dominates is a proxy placement, not a physical
   prediction of where that energy actually piles up in frequency space. Label it as such in
   the plot legend/caption.
3. **Two independently-coded cosmologies** (`cascade.cosmology`'s Python `H(z)` vs.
   `spectroxide`'s Rust `H(z)`) are only as consistent as the parameters you pass to each
   (§4.5) — not independently cross-validated against each other at the function level in
   this repo.
4. **The `dq_dz`/`photon_source` → `spectroxide.solve()` bridging code in this repo was
   written without a local `spectroxide` build to test against** (`interface.py`'s module
   docstring and `dual_solver_harness.py` both say `spectroxide` wasn't installed in that
   environment; `test_cp7_interface.py` skips/falls back accordingly). §4.3/§4.4 are
   confirmed correct here by inspection (grid-shape arithmetic, a shared constant used
   consistently) and by tracing the exact `solve()` dispatch chain against `solver.py`
   directly — but the bridge as a whole hasn't been run against a real Rust binary from
   inside *this* environment either. You have a working `spectroxide` build (the attached
   script runs it successfully), so the rest of this plan is directly testable on your end —
   budget time for a first-run debugging pass on the remaining plumbing (grid shapes,
   keyword names), not just the physics.
5. **`Δm ≳ m_π⁰` is out of scope** (§1, §5) — not an approximation so much as a hard boundary
   of what MPR21's own two-body treatment (and hence this whole pipeline) describes at all.

---

## 7. File / function layout

Don't modify `Cascades/cascade/interface.py` (leave the AK19 CP7 check it's written for
intact). Two source files, one genuinely new, one a fork:

```
Cascades/cascade/mckeen_bridge.py     (new, ~120 lines)
    omega_inj_and_Tn_eV(delta_m_eV)
        # section 1: Delta_m > 0 guard AND the new Delta_m < ~135 MeV hadronic-threshold guard
    _cached_budget(species, omega_inj_eV, z)      # lru_cache wrapper, section 4.6
    make_mckeen_dq_dz(omega_inj_eV, Gamma_X_s, den_fac, x_max_factor=...)
        # copy of interface.make_dq_dz's body, WITHOUT the f_X_for_target_all_heat_y call
        # (section 4.1 -- den_fac is a direct parameter, nothing is solved for),
        # rate line swapped per section 4.1, z<=1e7 guard added per section 5
    make_mckeen_photon_source(omega_inj_eV, Gamma_X_s, den_fac, x_max_factor=...)
        # same changes as above, for the photon channel

<your analysis script>.py     (FORK of fig6_spectrum_via_spectroxide.py, not a rewrite)
    run_one(dm):
        # = fig6_spectrum_via_spectroxide.py's spectrum_via_spectroxide_pde(), with:
        #   - the two builder-function calls swapped (section 4.1/4.2)
        #   - timeout=3600 added to both solve() calls (section 4.3)
        # everything else (the two spx_solve() calls, the x-grid-match assert, the
        # delta_n/mu/y sum) is section 4.3/4.4, already correct -- copy verbatim
        omega_inj_eV, Tn_eV = omega_inj_and_Tn_eV(dm)
        dq_dz         = make_mckeen_dq_dz(omega_inj_eV, GAMMA_X, den_fac, x_max_factor=X_MAX)
        photon_source = make_mckeen_photon_source(omega_inj_eV, GAMMA_X, den_fac, x_max_factor=X_MAX)
        r_heat  = spectroxide.solve(method='pde', dq_dz=dq_dz, cosmo=cosmo_ak19,
                                     z_start=Z_START, z_end=Z_END,
                                     z_min=Z_MIN, z_max=Z_MAX, n_z=N_Z,
                                     x_min=X_MIN, x_max=X_MAX, n_x=N_X,
                                     timeout=3600)
        r_phot  = spectroxide.solve(method='pde', photon_source=photon_source, cosmo=cosmo_ak19,
                                     z_start=Z_START, z_end=Z_END,
                                     z_min=Z_MIN, z_max=Z_MAX, n_z=N_Z,
                                     x_min=X_MIN, x_max=X_MAX, n_x=N_X,
                                     timeout=3600)
        assert np.allclose(r_heat.x, r_phot.x)
        delta_n = r_heat.delta_n + r_phot.delta_n
        mu, y   = r_heat.mu + r_phot.mu, r_heat.y + r_phot.y
        return dm, (r_heat.x, delta_n, mu, y, Tn_eV)
```

What to keep unchanged from each source when forking:

| from `fig6_spectrum_via_spectroxide.py` | from `sweep_deltam_photon_tau1e9.py` (attached) |
|---|---|
| `_safe_dq_dz_wrapper` / `_safe_photon_source_wrapper` | the npz resume-cache (`results` dict + `save_results()`) |
| the `X_MAX_SPX` single-constant pattern (§4.4) | `ThreadPoolExecutor` sweep over Δm |
| `spectrum_via_spectroxide_pde`'s two-`solve()`-calls-summed body (§4.3) | the two-panel (spectrum + μ/y/T-shift residual) plot |
| `_ensure_rust_binary_built()` (one-time sequential warm-up before the pool) | the SED plot, the summary table |

i.e. the new script's *skeleton* (sweep loop, caching, plotting) comes from the attached
script since that's the target output format; the *inner per-Δm solve function* comes from
`fig6_spectrum_via_spectroxide.py` since that's where the already-correct SPX plumbing lives;
only the two builder-function calls inside that inner function are new, plus `timeout=3600`.

Suggested shared constants for the sweep (§3, §4.4, §4.6):

```python
X_MAX  = 80.0     # match the attached script's grid; pass to BOTH make_mckeen_photon_source
                   # (x_max_factor) and every solve() call (x_max) -- section 4.4
X_MIN  = 1.5e-4
N_X    = 1000
Z_START, Z_END = 5.0e6, 5.0e2     # section 3
Z_MIN,  Z_MAX  = Z_END, Z_START   # tabulation range must bracket the PDE walk -- section 3
N_Z    = 500      # matches the internal photon-channel cap exactly -- section 4.6
```

---

## 8. Validation plan (do these before trusting the sweep output)

1. **Unit-check the rate fix.** Confirm
   `mpr21.mpr21_energy_injection_rate_eV_cm3_s(z, den_fac, GAMMA_X, omega_inj_eV)` at some z
   gives `omega_inj_eV × den_fac × GAMMA_X × n_H(z) × exp(-Γ_X t(z))` by hand for one point.
2. **Energy-budget sum check.** For each Δm in the sweep, assert
   `f_heat + f_ic + f_subthreshold` from `_cached_budget` is `1.0 ± 1e-6` at a few z values
   (verified above at `z ≈ 1.53832e5`; spot-check a couple more z across the active window).
3. **Regression check against the attached script.** Temporarily force `f_heat = 1, f_ic =
   f_subthreshold = 0` in the budget lookup (100% of `ω_inj` into heat, skip the photon
   `solve()` call). The result should be *close to* — not identical to, since the attached
   script uses `Δm` as the energy scale and this forces only `ω_inj` — the attached script's
   output. This isolates "did I wire the heat/photon split correctly" from "does the cascade
   module work" (the latter is already validated against AK19's figures).
4. **Single-Δm smoke test** before the full sweep, at default (fast) cascade resolution
   (§4.7), to catch grid/keyword-plumbing issues cheaply.
5. **Cross-check `H(z)`/`n_H(z)` between `cascade.cosmology` and `spectroxide.cosmology`** at
   the same `cosmo=` parameters (§4.5) at a couple of z values.
6. **New — hadronic-threshold guard actually fires.** Call `omega_inj_and_Tn_eV(1.4e8)`
   (Δm = 140 MeV, just above `m_π⁰`) and confirm it raises rather than silently returning a
   two-body `ω_inj` — a one-line test that the §5 guard you added actually works, not just
   that you wrote it.

## 9. Runtime estimate

Per Δm point, per channel: `n_z ~ 500` unique `(ω_inj, z)` budget evaluations × ~0.3 s (dev
resolution) = ~2.5 min, shared via the §4.6 cache across both channels → ~2.5 min/Δm point
(not ~5, since the cache is shared) at dev resolution, plus the actual Rust PDE solve time
(two calls instead of the attached script's one — budget roughly 2× the attached script's own
per-point PDE cost, hence the `timeout=3600` in §4.3/§7). For 6 Δm points in parallel
(`ThreadPoolExecutor`, as the attached script already does): dominated by the slowest single
point, so still a small number of minutes wall-clock — provided the §4.6 cache fix is in
place. At production cascade resolution (§4.7, ~300× slower per `burst_energy_budget` call),
re-estimate before committing to a full sweep; consider running the production pass only on
the Δm points you'll actually put in the final figure.

## 10. Explicitly out of scope (per your instructions)

- `src/energy_injection.rs`'s `DecayingParticlePhoton`/`MonochromaticPhoton` canned types —
  these inject a narrow line straight into the Kompaneets PDE with no cascade pre-processing
  at all (confirmed by reading `energy_injection.rs` directly); not usable standalone for a
  decay that needs cascade reprocessing first, and not part of this plan.
- `cascade.rs`/`cascade_photon_source.rs` and anything under `src/` beyond confirming what the
  canned injection types above do — dropped per your instruction, not touched.
- Building an actual resolved `dN/dE` cascade spectrum (§6, item 2) — would remove the
  narrow-line-proxy limitation, but is new physics-code work beyond "obvious changes" to the
  existing validated Tier-1 module, and isn't part of this plan.
- Hadronic final states for `Δm ≳ m_π⁰` (§1, §5, §6) — guard against, don't attempt to model.
