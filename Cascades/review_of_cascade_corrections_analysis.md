## Status of amended priority list (updated as fixes land)

1. **[DONE] Fix `cosmology.py`'s parameters to AK19's actual Planck values.**
   `H0_KM_S_MPC` changed 71.0 -> 66.88; `OMEGA_B`/`OMEGA_M` now derived from
   `Omega_b h^2=0.0221`, `Omega_c h^2=0.1206` (h=0.6688) instead of the old
   hardcoded CosmoTherm-defaults values 0.044/0.26. `Y_P=0.24` was already
   correct and is unchanged. Docstring updated to cite AK19 directly instead
   of conflating it with CosmoTherm/SPX defaults. Verified: `test_cp0_cosmology.py`
   (7/7 pass -- those tests only check internal self-consistency, e.g.
   H(0)=H0 and flat-universe closure, so they don't lock in wrong parameter
   values) and the full suite (38/39 pass; the one failure,
   `test_cp7_interface.py::test_photon_source_integrates_to_distortion_channel_energy`,
   is the separate, already-tracked `x_eff`/`x_max` clipping bug -- item 5
   below -- untouched by this change).
2. [ ] Fix `klein_nishina_dsigma_deps`'s missing `1/eps` factor (rates.py).
3. [ ] Fix #2 (Compton transfer fraction), once its ingredient function is correct.
4. [ ] Implement Eq. A.5 (relativistic) / A.6 (ultra-relativistic Klein-Nishina)
   for the electron->photon IC spectrum, at least for Fig. 3.
5. [ ] Fix the `x_max`/`x_cut` mismatch (pass `x_max=35` to every `spx_solve()`
   call) and the `x_eff` gamma-species degeneracy.
6. **[DONE] Re-scope Fig. 3 as its own task (electron-only, Eq. A.5/A.6-based).**
   New `cascade/ic_spectrum.py`: implements Eq. A.5 (Jones 1968 relativistic
   IC redistribution -- caught a dropped sqrt in a `(1+p^2)` vs `sqrt(1+p^2)`
   denominator during transcription, confirmed against the arXiv source) and
   Eq. A.6 (ultra-relativistic Klein-Nishina, for completeness/future use;
   not reached by Fig. 3's 10-70 keV electrons). `electron_ic_spectrum`
   convolves Eq. A.5 against the Planck seed shape per decade of the
   electron's cooling track (reusing the existing, already-validated
   Coulomb/IC decade branching for *how much* energy goes to the IC channel;
   what's new is *where in x* it lands). New `reproduce_fig3.py` reproduces
   Fig. 3 directly (no spectroxide/PDE -- Fig. 3 is explicitly the raw,
   pre-relaxation spectrum, heat/y channel excluded per AK19's own text).
   Verified: energy conservation to ~1% across all four decade-tracks;
   output peak/dip amplitudes (~3.5 / ~-2.5, in AK19's 1e-23 W/m^2/Hz/sr
   units) and shape (dip at the Planck peak, boosted peak on the Wien side,
   decreasing amplitude with electron energy) match Fig. 3 well. Full test
   suite still 38/39 (same pre-existing CP7 failure, untouched by this).
7. **[DONE] Correct Group C's membership to {10, 11, 12} and build the
   dual-solver comparison harness; treat Fig. 9 as validating wiring only
   at low injection energy.**
   `cascade_corrections_all_figures.md`'s Group C section rewritten to the
   corrected {10, 11, 12} membership and Fig. 9's actual (partial-fixes-
   still-matter) status. New `dual_solver_harness.py`: builds a genuine
   one-time-burst dq_dz(z)/photon_source(x,z) pair (a narrow pulse at
   z_inj -- Figs. 10-12 are single-burst injections, like Fig. 3, not the
   continuous decaying-particle scenario interface.py's existing
   make_dq_dz/make_photon_source targets), with the photon-channel source
   now built from ic_spectrum.py's actual Eq. A.5 spectrum (fix #6) instead
   of interface.py's narrow-line proxy. `run_recoil_only_side` (the
   pre-relaxation cascade curve) is fully runnable and verified in this
   sandbox; `run_kompaneets_side` (the real spectroxide PDE curve) and
   `test_x_cut_sensitivity` (the paper's own x=30/25 boundary-point
   cross-check) are structurally complete but need spectroxide's Rust
   binary, which this sandbox cannot build at all (no Cargo.toml/vendored
   deps in src.zip, no network) -- degrades gracefully with a clear error
   rather than silently, verified via `compare_solvers`.

**Consistency check against items 1-5 (done separately, after items 1-5
landed):** items 2/3/5 all live inside `solve_burst.electron_cascade` /
`photon_cascade_full` / `interface.make_photon_source`'s reprocessing
chain, which fixes 6/7's `ic_spectrum.py` deliberately bypasses (it
implements the electron->photon step directly via Eq. A.5 instead of
routing through that chain) -- for Fig. 3/10's actual keV-scale energies
this is fine, re-verified numerically to give bit-identical results
before/after items 2/3/5 landed (the old `_compton_mean_transfer_fraction`
bug and the x_eff/x_max fixes never engage at those photon energies; full
9/9 checkpoint suite also passes, including CP7 which items 1-5 newly
fixed). Two real issues were found and fixed while checking this, though:
  - `electron_ic_spectrum` never checked whether its own IC-channel photons
    would actually reach the pair-production/gg-scattering regime that
    `electron_cascade` reprocesses through -- caught by cross-checking
    against `sb.burst_energy_budget` directly: at E0=1 GeV, z=1e4,
    `electron_ic_spectrum` alone said f_heat=0.000/f_ic=1.000, while the
    correctly-reprocessed answer is f_heat=0.267/f_ic=0.716 (a large,
    non-rounding-level gap). Only matters for Figs. 11/12's GeV-scale
    points (Fig. 3/10 never reach it); now cross-checked and corrected
    inside `electron_ic_spectrum` itself (new `shape_reliable`/
    `f_ic_corrected` return fields, threaded through
    `electron_ic_delta_n`/`fig3_style_delta_I`).
  - The re-uploaded `Cascades.zip` had `cascade/dual_solver_harness.py`
    (which calls `ic_spectrum.electron_ic_delta_n`) paired with a version
    of `ic_spectrum.py` from before that helper existed -- an
    `AttributeError` on any call to `make_burst_photon_source`/
    `run_kompaneets_side`. Not caught by `dual_solver_harness.py`'s own
    `__main__` block, since that only exercises `run_recoil_only_side`
    (the half that doesn't need it). Restored.

---

# Review of the corrections-plan review (`cascade_corrections_for_fig6.md` / `cascade_corrections_all_figures.md`)

Scope: I re-derived every checkable claim in the submitted analysis independently —
against `cascade/solve_burst.py`, `cascade/interface.py`, `cascade/rates.py`,
`cascade/cosmology.py`, the test suite, the rendered AK19 PDF (text extraction +
page rasterization for the equations, since the differential-cross-section
formulas garble badly in plain text extraction), and the Rust crate in `src.zip`
(confirmed to be spectroxide's own backend, not a separate project). Verdict:
**the analysis is right about everything it checked**, including the two
corrections it made to the plan (Group C miscategorization, Fig. 9's
dependence on cascade fixes) and the two things it says the plan misses
(the `x_eff` degeneracy bug, the `x_max`/`x_cut` mismatch). But it also
**overclaims in one place, and misses several things of its own** — most
importantly, its confident "Group A checks out" verdict rests on a
cosmology-parameter claim that is actually wrong once checked against the
paper's own numbers.

---

## Part 1 — What the analysis got right (independently re-verified)

- **`electron_cascade`/`photon_cascade_full` collapsing to scalars, `_compton_mean_transfer_fraction` being `x/(1+x)`, and the `gg_rate` branch booking straight to `f_ic`** — read the actual code; all three match the analysis's description exactly (`solve_burst.py` lines ~79–110, 207–274, 277–372).
- **`klein_nishina_dsigma_deps` is defined in `rates.py` but never called** — confirmed by grep; only `_compton_mean_transfer_fraction` is used at both call sites. (But see Part 2 — the function itself has a bug, so "the hard part is already done" doesn't hold.)
- **The Group C figure-list miscategorization is real.** I read every caption in the PDF directly:
  - Fig. 7, 8, 9 captions: single-code ("with Kompaneets kernel code").
  - Fig. 10, 11 captions: *"Left and right panels are the distortions obtained with full Kompaneets kernel and Recoil-only codes respectively"* — genuine dual-solver.
  - Fig. 12: same-panel overlay of both codes' curves (`z=1×10⁴ (Recoil-only)` and `z=1×10⁴(Kompaneets)` on one axis) — also genuine dual-solver.
  - So the true dual-solver set is **{10, 11, 12}**, not the plan's **{7, 9, 11, 12}**. The analysis's correction is exactly right.
- **Fig. 9's high-energy dependence on the cascade fixes is a real gap in the plan.** Fig. 9 panel (b) legend does include `100 GeV e⁻` and `100 GeV e⁻e⁺` (confirmed via page rasterization, not just the noisy text extraction). And the text right before Fig. 9's caption states the general Kompaneets/Recoil-only interface — Recoil-only handles `x ≥ 35` and feeds the result into the Kompaneets kernel every timestep — which applies to *every* Kompaneets-kernel figure, Fig. 9 included. So a 100 GeV injection's low-x spectrum is only as good as the Recoil-only cascade that fed it, meaning fixes #1/#2 do matter for Fig. 9(b). The analysis's downgrade ("validates wiring at low injection energy only") is the right call.
- **The `x_eff` degeneracy bug is real and the numbers check out.** I recomputed `x_eff_raw = E0/kT` at z=20000 (kT≈4.70 eV) for 10 keV/100 keV/1 MeV photon injections: 2.13e3, 2.13e4, 2.13e5 — all clip to the same `x_cut*0.9 = 31.5`. Independently confirmed.
- **The `x_max`/`x_cut` mismatch is real.** `interface.py`'s `make_photon_source` hardcodes `x_cut = 35.0`; spectroxide's `solve()` defaults to `x_max: float = 30.0` (confirmed directly in `spectroxide/solver.py`); `fig6_spectrum_via_spectroxide.py` never passes `x_max=` in either `spx_solve()` call. I also reproduced the 1 MeV electron-branch tail-spillage numbers exactly: `x_eff≈27.88`, `sigma_x≈2.79`, ±3σ = [19.5, 36.2] — past the grid's actual edge at 30. And `test_cp7_interface.py`'s comment about "the narrow-line proxy's Gaussian can be clipped by the x_cut boundary" is verbatim as described.
- **The Rust crate claims all check out.** `src/cascade.rs` cites Kawasaki & Moroi (1995) and ACROPOLIS (Depta, Schmidt-Hoberg & Wild 2020) explicitly; `CascadeResult{photon_spectrum, electron_spectrum, positron_spectrum}` and a genuine per-bin 3×3 `solve_3x3` linear solve both exist as described. `solve_cascade`/`CascadeResult` are called *only* from `cascade_photon_source.rs`, which is built around `DecayingParticlePhotonParams` — `InjectionScenario::SingleBurst` (the AK19 one-time-burst scenario) never touches this module anywhere in `solver.rs`/`energy_injection.rs`. The "~36x residual" and "lepton-channel energy... not yet routed" language is quoted almost verbatim from comments in `cascade_photon_source.rs`. All confirmed.
- **The verbatim AK19 quotes are all real**, checked against the actual paper text: the Fig. 3 heat-exclusion sentence, the Fig. 6 "scaled y-distortion spectrum for comparison" sentence, and the "varied the boundary point to x=30 and 25... does not affect the solution for x ≲20" sentence all appear exactly as quoted.

## Part 2 — Where the analysis is wrong or overclaims

### 2.1 "Group A checks out" rests on a cosmology-parameter claim that's actually false

The analysis states: *"cosmology params match AK19 exactly (h=0.71, Ωb=0.044, Ωm=0.26, Yp=0.24)."* This is the analysis's own added verification, and it doesn't hold up. AK19's actual text (Sec. 2, just before Fig. 1) says:

> *"We use Planck [1] ΛCDM cosmological parameters with Hubble constant H₀=66.88 km s⁻¹ Mpc⁻¹, baryon energy density parameter Ωb h²=0.0221, cold dark matter energy density parameter Ωc h²=0.1206... Helium mass fraction (Yp)=0.24."*

Converting: h = 0.6688 (not 0.71), Ωb = 0.0221/h² ≈ 0.0494 (not 0.044), Ωm = Ωb+Ωc ≈ (0.0221+0.1206)/h² ≈ 0.319 (not 0.26). **Only Yp=0.24 actually matches.** `cascade/cosmology.py`'s own docstring is explicit about where its numbers really came from: *"Parameters match SPX's own validation choices... i.e. CosmoTherm v1.0.3 defaults"* — a different code's defaults, not AK19's stated parameters. The same conflation ("AK19/CosmoTherm defaults", as if the two were the same thing) appears already in the older planning docs (`old_implementation_plan.md`, `em_cascade_implementation_plan-old.md`), so this is an inherited assumption that nobody — the original plan, the corrections docs, or this review — traced back to the paper's actual numbers. `test_cp0_cosmology.py` only checks internal self-consistency (`H(0)=H0`, flat-universe closure), never the parameter values themselves, so this passes tests silently.

This isn't a Fig-1/2-only issue: `cosmology.py`'s `H(z)`, `n_e_total(z)`, `T_gamma(z)`, and `rho_cmb_eV_cm3` feed *every* figure in the paper (they set the Coulomb/IC crossover in Figs. 1/2/13, the CMB energy density normalization in every burst calculation, and every redshift-dependent rate in the cascade). A ~5–20% shift in h/Ωb/Ωm won't be dramatic, but for a project whose explicit goal is precise reproduction of AK19's plots, this deserves an explicit fix (swap in H₀=66.88, Ωb h²=0.0221, Ωc h²=0.1206) and a re-check of Fig. 1/2's stated ~2.5 keV crossover once corrected, rather than being asserted as already-exact.

### 2.2 The "hard part of fix #2 is already done" claim is wrong — `klein_nishina_dsigma_deps` itself has a transcription bug

I rasterized the actual PDF page (text extraction mangles this formula's fraction layout badly) to read AK19 Eq. B.3 directly:

```
dσ/dε = [3 σT me / (8 (hν)² ε)] × [hν/ε + ε/hν + (me/ε − me/hν)² − 2(me/ε − me/hν)]
```

The code's version:

```python
return (3.0 * SIGMA_T_CM2 / 8.0) * (me / hnu**2) * ( ... same bracket ... )
```

The prefactor the code computes is `3σT·me / (8·hν²)` — **missing the `1/ε` factor** that's in the paper's denominator (`8(hν)²ε`, not `8(hν)²`). The bracket term is transcribed correctly, but the prefactor isn't. So the claim that *"the hard part (transcribing B.3) is already done; someone just needs to integrate it for the mean transfer and wire it in"* isn't accurate — the transcription needs a one-line fix (`/ eps` added to the prefactor) before it can be integrated for the mean-transfer fix at all. Small fix, but it means fix #2's actual first step is "repair `klein_nishina_dsigma_deps`," not "just call it."

### 2.3 Fig. 3 is mischaracterized as "the simplest member of Group B, same fixes as Fig. 6" — it may need a different formula path entirely

Two things the analysis (and the plan) both miss about Fig. 3:

1. **Fig. 3 is electron-injection only.** Its caption is *"Spectral distortion of the CMB after deposition of all of the electron energy..."* — no photon-injection case at all, unlike Fig. 6 (which explicitly spans both electron and photon injection). The `make_photon_source` narrow-line bug (item #3 in the fig6 plan, and the `x_eff` gamma-species degeneracy the analysis found) is a photon-channel problem — it may be simply irrelevant to reproducing Fig. 3, whose only channel is electron→IC.
2. **AK19's own text says Fig. 3's shape comes from a different, more exact formula than the Recoil-only cascade used for Figs. 6/7/8/10.** Right after introducing Fig. 3: *"For relativistic electrons, the spectrum after IC scattering is calculated using results of [45–47] with the full CMB spectrum. The formulae are given in Appendix A."* That's Eq. A.5 (the exact analytic frequency-redistribution function `P(t;p)` for one IC scattering of a monoenergetic relativistic electron off the full Planck spectrum) and, per Sec. A.4, Eq. A.6 (the double-differential Klein-Nishina spectrum, needed once electron energy reaches ~GeV). **Neither formula is implemented anywhere in this codebase** — `rates.py`/`solve_burst.py` only ever produce a single characteristic photon energy per decade (the thing fix #1 is trying to patch with ad hoc multi-point sampling). Grepping the whole `cascade/` package for any trace of `P(t;p)`, `arcsinh`, or a double-differential IC spectrum comes up empty.

This matters beyond just Fig. 3: **Eq. A.5/A.6 are the exact, paper-given recipe for the "full fix" to cascade item #1's electron-side photon spectrum** — a more precise and directly paper-faithful path than guessing how many log-spaced sample points per decade are "enough." Since Fig. 3 is a single-generation (no further cascading needed — it's literally described as the electron depositing all its energy in one pass) spectral-shape figure, it may be the one figure where the ad hoc "3–5 points per decade" proxy is least likely to reproduce the paper's precise curve, making Fig. 3 arguably one of the *harder* members of Group B to get exactly right, not the simplest.

### 2.4 Two more unused-but-available exact formulas, adjacent to already-flagged items

- **Eq. B.5** (the exact positron energy spectrum from pair production) is never used; `photon_cascade_full`'s pair-production branch splits energy evenly 50/50 between the electron and positron daughters instead. This is a known, explicitly-flagged simplification in the code's own docstring, so it's not a new bug — but it's a second concrete "exact formula sitting unused in the paper" case, in the same spirit as B.3/A.5/A.6, worth batching with those fixes rather than treating as a separate lower-priority task.
- **Eq. B.7** (the secondary-photon spectrum from γγ→γγ scattering on the CMB) is likewise unused — the `gg_rate` branch (already flagged as item #4 in the fig6 plan) just books energy directly to `f_ic` rather than distributing it via B.7. Same point as above: the plan correctly flags this as low priority, but doesn't mention that AK19 already hands over the formula that would resolve it.

### 2.5 Minor: a dead parameter reinforces the general `x_cut` confusion

`make_photon_source(..., x_max_factor=35.0)` takes an `x_max_factor` keyword argument that is **never referenced anywhere in the function body** — the inner closure hardcodes `x_cut = 35.0` directly instead. So the function's own signature implies the cutoff is configurable when it isn't. Not a physics bug on its own, but it's one more symptom (alongside the real `x_max`/`x_cut` mismatch the analysis found) of the x-grid boundary being handled inconsistently across this file.

---

## Bottom line

The analysis is a careful, well-sourced review and its central corrections to the plan (Group C's real membership, Fig. 9's actual dependencies, the `x_eff`/`x_max` bugs, and the existence of the more-rigorous Rust cascade) all hold up under independent re-checking against the code and the paper. Its weak point is that it verified the plan's *cascade logic* claims meticulously but took two *supporting* claims about "already correct" or "already done" pieces at face value — the cosmology parameters (Group A) and the `klein_nishina_dsigma_deps` transcription (fix #2's "hard part") — and both turn out to have real, independently-checkable errors once traced back to AK19's actual numbers and equations. It also treats Fig. 3 as a same-fixes-apply member of Group B without checking that AK19 describes a different, more exact method for it specifically.

**Suggested amended priority order**, layering these findings onto the plan's own:

1. Fix `cosmology.py`'s parameters to AK19's actual Planck values (H₀=66.88, Ωb h²=0.0221, Ωc h²=0.1206, Yp=0.24) — affects every figure, cheap to fix, currently untested against the paper's own numbers.
2. Fix `klein_nishina_dsigma_deps`'s missing `1/ε` factor before using it for fix #2's Compton mean-transfer integral.
3. Fix #2 (Compton transfer fraction), now that its ingredient function is actually correct.
4. Recognize that the "full" version of fix #1 already has a paper-given recipe — Eq. A.5 (relativistic) / A.6 (ultra-relativistic Klein-Nishina) for the electron→photon IC spectrum — and consider implementing that directly for Fig. 3 (and as the eventual real fix elsewhere) rather than only the ad hoc multi-point sampling proxy.
5. Fix the `x_max`/`x_cut` mismatch (pass `x_max=35` to every `spx_solve()` call) and the `x_eff` gamma-species degeneracy, as the analysis recommends, before trusting any Fig. 6 output.
6. Re-scope Fig. 3 as its own task (electron-only, Eq. A.5/A.6-based) rather than assuming it falls out "for free" once Figs. 6/7/8/10's fixes land.
7. Correct Group C's membership to {10, 11, 12} and build the actual dual-solver comparison harness; treat Fig. 9 as validating wiring only at low injection energy, per the analysis.
