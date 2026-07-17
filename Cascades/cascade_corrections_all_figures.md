# Corrections needed in `cascade/` to reproduce AK19's plots (Figs. 1–15)

This extends the Fig. 6-specific notes to the whole paper. AK19's figures
fall into three groups, and each group depends on the cascade code
differently — so "fix the cascade" means different things depending on
which figure you're chasing. Everything below is confirmed from actual
code (control flow, variable types, struct fields), same standard as
before.

---

## Group A — pure cascade output, no spectroxide needed

**Figs. 1, 2** (photon and electron/positron cooling times vs Hubble
time), **Fig. 13** (heat fraction vs injection energy).

These only need `rates.py` (Coulomb A.1, IC-Thomson A.4, and — for photons
— the Compton/pair-production/gg rates) and the scalar energy-budget
functions in `solve_burst.py`. No spectral shape is involved; these are
already the easiest group and the repo already has working analogs
(`fig2_cooling_ratios.png`, `fig13_heat_fraction.png`).

- **Figs. 1/2 should already be close to correct.** `rates.py`'s formulas
  are direct transcriptions of AK19 Eqs. A.1/A.4/B.2, and cooling-time
  ratios don't touch the cascade recursion at all — there's nothing in
  the cascade-specific issues below that affects these two.
- **Fig. 13 is the one place the single-characteristic-energy problem
  (see Group B, item 1) shows up *without* needing spectroxide at all** —
  it's a plot of `f_heat` itself, and `f_heat` for high-energy electrons
  comes directly out of `electron_cascade`'s per-decade
  single-characteristic-photon recursion. This is already root-caused
  independently (the boost/de-boost oscillation that should damp out with
  cascade depth doesn't, because there's no spread of daughter-photon
  energies to average over) — same fix as Group B, item 1 below.

---

## Group B — spectral-shape figures (the big group)

**Figs. 3, 6, 7, 8, 10** (species/energy/redshift comparisons of
`δI_ν(x)` or `T(x)/T_CMB−1`), **Fig. 14b** (spectral distortions from a
DM decay scenario), **Fig. 15** (residual after subtracting best-fit
thermal templates, i.e. the "ntr-distortion").

All of these are literally the same measurement as Fig. 6 — a photon
spectrum, not a scalar — just sliced along a different axis (energy,
redshift, or decay lifetime instead of species). So every correction
identified for Fig. 6 applies to all of them simultaneously, and fixing
these once fixes this entire group:

1. **Single characteristic energy per cascade generation**
   (`electron_cascade`, `photon_cascade_full`) — the cascade can only
   report *how much* energy landed in heat vs. photons, never *at what
   frequencies*. This caps every one of these figures at "right total
   amplitude, wrong/missing shape," regardless of which axis is being
   swept. Cheap fix: sample several (3–5) log-spaced daughter energies
   per decade instead of one, recurse on each, sum weighted results —
   already shown to move amplitudes in the right direction without
   restructuring anything. Full fix: replace the scalar bookkeeping with
   an actual array/histogram carried through the recursion.

2. **Phenomenological Compton transfer fraction**
   (`_compton_mean_transfer_fraction`, the `x/(1+x)` interpolation) — sets
   the recoil kinematics at *every* scatter, so it directly shifts where
   the peak/zero-crossing sit at *every* energy these figures scan across
   (10 keV–1 MeV for Figs. 6/7, up to 10 MeV for Fig. 8, 1 GeV–100 GeV for
   Fig. 11 [Group C]). Cheap, local, self-contained fix: swap in the
   literal AK19 Eq. B.3 Klein-Nishina mean-transfer integral.

3. **`make_photon_source`'s narrow-line proxy** (`interface.py`) — this is
   the wiring between the cascade and spectroxide, and it's what actually
   determines every one of these figures' x-axis dependence once fed
   through the PDE. It needs rewriting regardless of #1's status, since
   even a fixed multi-point cascade still needs a function that turns
   "several (energy, weight) pairs" into "several lines summed into a
   `d(ΔN/N)/dx` array" instead of collapsing back to one `x_eff`.

**Figure-specific notes on top of the three shared fixes:**

- **Fig. 3** ("after the electrons have deposited all of their energy," a
  single burst, no redshift evolution) is actually the *simplest* member
  of this group — it's one z, one burst, and AK19 explicitly excludes the
  heat-channel y-distortion from these curves ("The y-distortion from the
  energy dissipated as heat is not added to any of the curves"). That
  last point is a wiring detail worth flagging: reproducing Fig. 3 means
  plotting the **photon-channel-only** spectrum (no heat/y-channel added
  back in), which is different from how `fig6_via_spectroxide.py`/
  `fig6_spectrum_via_spectroxide.py` build their curves (heat + photon
  summed). Don't sum the two channels when targeting Fig. 3 specifically.
- **Fig. 10** sweeps *redshift* (`z=1e4...4e4`) at fixed energy rather
  than energy at fixed redshift — same three fixes apply, just swept
  along `Z_INJ` instead of `E_GRID` in whatever script builds it.
- **Fig. 14b / Fig. 15** are for a *continuous* (decaying-particle)
  source rather than a single burst — these need the convolution machinery
  (`decay_convolution_y`, `build_zinj_grid`) feeding into the same
  cascade→spectroxide route, still limited by fixes #1–#3 exactly the
  same way. Fig. 15 additionally needs the μ/y/i/T-shift template
  best-fit-and-subtract step (present already in the neutron-decay script
  we discussed, `fit_mu_y_T`) — AK19 calls the residual after that fit the
  "ntr-distortion," which is the real object of interest for this figure.

---

## Group C — figures that need a second solver, not (only) a cascade fix

[CORRECTED per review_of_cascade_corrections_analysis.md, amended-plan item
7 -- verified directly against every caption in the PDF: Figs. 7, 8, 9 are
single-code ("with Kompaneets kernel code" only). Only **Figs. 10, 11, 12**
actually overlay two independently-run solvers on the same injection
("Left and right panels are the distortions obtained with full Kompaneets
kernel and Recoil-only codes respectively" / same-panel overlay for Fig.
12). The true dual-solver set is **{10, 11, 12}**, not {7, 9, 11, 12}.
Built in `dual_solver_harness.py`.]

**Figs. 10-12** compare two independent numerical methods AK19
built: a **"Kompaneets kernel" code** (full Kompaneets PDE, x < 35,
1000 energy bins, tracks each individual photon-electron collision) and
a **"Recoil-only" code** (the cascade-style approximate treatment, used
for x ≥ 35, and also run standalone across the *full* spectrum including
x < 35 for the Fig. 6-style comparisons). Fig. 5 in the paper is literally
a flowchart of how these two hand off to each other.

**What this repo currently has:** the "Kompaneets kernel" side is exactly
what spectroxide's Rust PDE solver already *is* — Kompaneets + DC/BR,
low-x, exact. The "Recoil-only" side is what `cascade/` approximates via
`burst_energy_budget` + (as of fix #6) `cascade/ic_spectrum.py`'s actual
Eq. A.5-resolved spectrum instead of a single characteristic-energy proxy.
So the building blocks exist, and `dual_solver_harness.py` now runs
**both** solvers on the *same* one-time-burst injection and plots them
side by side, the way Figs. 10/11/12 do (previously everything went
through the cascade→spectroxide handoff as one combined pipeline, not two
independently-computed curves to compare) -- though the Kompaneets-kernel
side needs spectroxide's Rust binary built, which this sandbox cannot do
(no Cargo.toml/vendored deps in src.zip, no network) -- see that file's
module docstring.

The paper's own cross-check criterion — "we have varied the boundary
point to x=30 and 25 and checked that it does not affect the solution
for x ≲ 20" — is a testable claim about *this* codebase's handoff point
(currently hardcoded at `x_cut = 35` in `make_photon_source`, the broader
fix #5 mismatch -- still open) that `dual_solver_harness.py`'s
`test_x_cut_sensitivity()` now directly tests (also needs the Rust binary
to actually execute).

**Fig. 9 is NOT dual-solver** (single Kompaneets-kernel code only) but
isn't a pure "no cascade fixes needed" validation target either: its panel
(b) legend includes 100 GeV injections, and the text describing the
Kompaneets/Recoil-only interface — Recoil-only handles `x ≥ 35` and feeds
the result into the Kompaneets kernel every timestep — applies to *every*
Kompaneets-kernel figure, Fig. 9 included. So a 100 GeV injection's low-x
spectrum is only as good as the Recoil-only cascade that fed it, meaning
fixes #1/#2 do matter for Fig. 9(b). Correctly scoped as "validates
spectroxide wiring alone at LOW injection energy only" -- not built here;
only needs interface.py's existing (single-solver) wiring.

---

## Suggested order of attack

1. **Fig. 9** — validates the spectroxide wiring alone (Group C shortcut,
   no cascade fixes needed).
2. **Fig. 2/1** — should already be fine; quick sanity check, no fixes
   expected.
3. **Fix #2** (Compton transfer fraction) — cheap, local, improves every
   Group B figure's peak/crossing accuracy immediately.
4. **Fix #1's cheap version** (multi-point-per-decade sampling) —
   improves Fig. 13 directly and broadens what Group B figures have to
   work with.
5. **Fix #3** (rewrite `make_photon_source` to consume a real spectrum) —
   unlocks Figs. 3/6/7/8/10/14b/15 together once #1 supplies something
   richer than one number.
6. **Figs. 7/11/12** (dual-solver comparison) — build the actual
   side-by-side Kompaneets-vs-Recoil-only comparison harness, and use it
   to test the `x_cut` handoff-point sensitivity claim from the paper.
7. **Figs. 14b/15** — wire the continuous-decay convolution through the
   same (by-then-fixed) cascade→spectroxide route.
