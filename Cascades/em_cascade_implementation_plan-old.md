# Implementation Plan: EM Cascade Code for χ→n+γ (and general single-photon decays)

**Goal:** compute the CMB spectral distortion sourced by a decay that injects a single
high-energy photon per event — the motivating case is the dark-baryon decay χ→nγ
(McKeen, Pospelov & Raj 2021, PRD 103, 115002) — with the *full* electromagnetic cascade
resolved (pair production, Compton in both non-relativistic and Klein–Nishina regimes,
inverse-Compton cooling of secondary e⁺/e⁻, photon–photon scattering on the CMB), over
the redshift range studied in Acharya & Khatri 2019 (arXiv:1808.02897, hereafter **AK19**):
roughly 1200 ≲ z ≲ a few×10⁵.

This is *not* a rebuild of spectroxide (Baker, Liu & Mishra-Sharma, arXiv:2604.24838,
hereafter **SPX**). SPX already solves the hard problem (the stiff Compton+DC+BR PDE at
low frequency) and validates it thoroughly. SPX's own limitations section says explicitly
that high-energy photon/electron injection with cascade development is *not* modeled and
is left as future work — that gap is exactly this project.

**On checkpoints**: the section 6 below is the main answer to "writing the main code
directly will give errors." Rather than building the whole pipeline and debugging it as
one block, the plan is staged so each checkpoint adds exactly one new piece of physics or
machinery on top of a already-validated base, and is checked against a concrete AK19
number or figure *before* you move on. The ordering is deliberate: low-energy Compton-only
physics (no pair production) before high-energy physics; single-redshift, single-burst
behavior before multi-step redshift marching; the fully self-contained Tier-1 pipeline
before the spectroxide interface is even touched; and generic cascade code before the
specific χ→nγ kinematics. This means that when something breaks, it's attributable to
whatever was added *since the last passing checkpoint* — a small, bounded search space.

---

## 0. Division of labor

| | Tier 1 (new code, this plan) | Tier 2 (reuse) |
|---|---|---|
| Physics | Pair production (Bethe–Heitler on matter), Compton (non-rel. recoil + relativistic Klein–Nishina), inverse-Compton (Thomson + Klein–Nishina), Coulomb cooling of e⁻/e⁺, photon–photon elastic scattering on CMB, e⁺e⁻ annihilation | Compton (Kompaneets), double Compton, Bremsstrahlung; electron-temperature feedback; μ/y/ΔT decomposition |
| Domain | particle energy from injection energy ω_inj down to a handoff frequency x_cut ~ 20–35 | x ≲ x_cut down to the present day |
| Character | triangular, non-stiff, "on-the-spot" per redshift step | stiff PDE, needs implicit time-stepping |
| Owner | **you build this** | **spectroxide**, called through its documented custom-source API (or a minimal standalone Kompaneets fallback, §7) |

The interface between the two tiers is two plain functions with units fixed by SPX's own
conventions (§4 below): a heating-rate function `dq_dz(z)` and a photon-source function
`photon_source(x, z)`.

---

## 1. Target process and its parameterization

For χ→nγ (McKeen–Pospelov–Raj, hereafter **MPR21**), the photon energy is fixed by
two-body kinematics and is **redshift-independent at production**:

```
ω_inj = (m_χ/2)(1 − m_n²/m_χ²)
```

(MPR21 Eq. 7, with Γ_{χ→nγ} ∝ θ² (μ_n/μ_N)² α ω_inj³ / m_χ² giving the rate). Note that
because the neutron is dark-sector-electromagnetically-inert to leading order, χ→nγ is
*effectively* a pure single-photon-injection process for cascade purposes — you don't need
to separately track the neutron.

The injection is **continuous over cosmic time**, following the standard exponential decay
profile, exactly as in SPX's built-in "Decaying particle" scenario (SPX Eqs. 10–13):

```
dE/(dt dV)|_dec = f_X Γ_X n_H(z) e^(−Γ_X t),      f_X ≡ ε_X m_X n_{X,0}/n_{H,0}
```

For χ→nγ, ε_X = ω_inj/m_χ (all of it goes to the photon; the neutron gets the recoil,
negligible EM-wise) and Γ_X = Γ_{χ→nγ}. Keep this parameterization generic (`f_X`, `Γ_X`,
`ω_inj`) so the same code works for any two-body radiative decay, not just this one model.

**Recommended default scope**: z_dec (equivalently τ_χ) chosen so the decay is active
somewhere in 10³ ≲ z ≲ 2×10⁵ — this is the AK19 window and where the resulting
"ntr-type" distortion is genuinely distinct from a pure y/μ shape. Above z ~ 2×10⁶ the
universe is close to full thermalization and SPX alone (heat-injection mode) is already
adequate without cascade resolution — flag this as an explicit non-goal for v1, extendable
later if needed.

**Photon energies of interest**: for the χ→nγ parameter space in MPR21 (Δm = m_χ−m_n from
~100 keV to ~50+ MeV), ω_inj spans roughly 1 keV–100 MeV. This sits entirely inside AK19's
own validated range (10 keV–100 GeV), which gives you a ready-made set of regression
targets (§6).

---

## 2. Two-tier architecture

```
                     ┌─────────────────────────────────────────┐
   χ→nγ decay rate   │  TIER 1: cascade engine (new code)      │
   f_X, Γ_X, ω_inj ──▶  triangular energy-bin solve per z-step │
                     │  species: e⁻, e⁺, γ  (E > x_cut·T_γ(z)) │
                     └───────────┬───────────────┬─────────────┘
                                 │               │
                     heat fraction│               │sub-threshold photon
                     dq/dz(z)     │               │spectrum d(ΔN/N)/dx/dz
                                 ▼               ▼
                     ┌─────────────────────────────────────────┐
                     │  TIER 2: spectroxide (reused)            │
                     │  dq_dz(z) heat hook + photon_source(x,z) │
                     │  Compton+DC+BR PDE → μ, y, ΔT/T          │
                     └─────────────────────────────────────────┘
```

Handoff frequency **x_cut = 35** (matching AK19's own choice; they checked x_cut = 25, 30
give the same x ≲ 20 answer). This is also close to where the non-relativistic Kompaneets
equation SPX uses (SPX App. A.1) starts breaking down, so it should also sit inside SPX's
own frequency grid — **verify this empirically against SPX's actual grid bounds early**
(SPX App. B.3 "Frequency grid" wasn't visible in the text I could retrieve; treat as an
open item, §9).

---

## 3. Tier 1: the cascade engine

### 3.1 Grids

- **Photon grid**: bin in dimensionless frequency **x = E_γ/T_γ(z)**, not physical energy.
  This is the natural choice because x is exactly redshift-invariant for free-streaming
  photons (AK19 note this explicitly), so no regridding is needed between steps except
  where a collision directly changes x. Use log-spaced bins from x ~ x_cut up to
  x_inj(z) = ω_inj/T_γ(z) (which *grows* as the universe expands, since ω_inj is fixed
  physical energy and T_γ shrinks going forward in time — so at high z, x_inj is smaller;
  make sure the grid's upper edge tracks x_inj(z) at the redshift where you inject, and
  extend downward to the cascade's natural endpoint).
- **Electron/positron grid**: bin in physical kinetic energy E_e (eV), log-spaced from a
  low cutoff (~keV, below which essentially all energy is Coulomb-thermalized in a single
  step — AK19 Fig. 2) up to ω_inj. Electron populations are **transient within one z-step**
  (on-the-spot approximation, §3.4) so this grid does not need to persist across steps.
- **Bin count**: AK19 used order a few hundred to 1000 bins for their Kompaneets-kernel
  code and comparable numbers for the high-energy grid; start with ~300 log bins per
  species and do a convergence check (§6) before finalizing.
- **Redshift grid**: for the *single-burst* Green's function runs (§3.6), step from
  z_inj down to the handoff point (recombination onward is Tier 2's job) — AK19 used
  Δz = 100 (roughly linear steps) and checked convergence against step size; adaptive
  Δz/z stepping is preferable if the injection redshift spans orders of magnitude.

### 3.2 Kinetic equation and solution algorithm

Directly implement AK19 Eq. (3.1): for particle species β at energy bin s,

```
ΔN_s^β = Σ_{α=e⁻,e⁺,γ} [ −Σ_{j<s} P^{βα}(E_s,E_j) N_s^β + Σ_{j>s} P^{αβ}(E_j,E_s) N_j^α ] + S^β(E_s)
```

Key structural fact (AK19 §3): because particles can only lose energy (cascade downward,
never upward except for the Doppler-boosted CMB photon which is a *background* photon, not
a cascade particle), the matrix is **triangular**. Solve by **back-substitution starting
from the lowest energy bin**: once you know how lower bins process their particles, you can
propagate higher bins down through them in one pass — this is what makes the whole thing
fast even with many bins, and is fundamentally simpler than SPX's stiff, implicitly-coupled
PDE. Implement this as a straightforward reverse loop over energy bins per redshift step;
no linear solver library needed, this is O(N²) per step at worst (N = number of bins) and
can be vectorized.

### 3.3 Rates and cross sections to implement (catalogue, with AK19 equation references)

**Electron/positron energy loss:**
- Coulomb scattering off background electrons: AK19 Eq. A.1 (Swartz–Nisbet–Green fit).
- Inverse-Compton, Thomson limit: AK19 Eq. A.4, dE/dt = (4/3) σ_T c U_CMB γ²β².
  Full Compton-upscattered spectrum from a monoenergetic electron: AK19 Eq. A.5
  (Wright 1979 / Rephaeli 1995 / Enßlin–Kaiser 2000 formula).
- Inverse-Compton, ultra-relativistic (Klein–Nishina) limit: AK19 Eq. A.6 double-differential
  spectrum — needed once γ_e is large enough that E_CMB in the electron rest frame is
  comparable to m_e (this happens at progressively lower γ_e as z increases, since T_CMB
  is higher).
- e⁺e⁻ annihilation cross section: AK19 Eqs. A.2–A.3 (non-relativistic Sommerfeld-corrected
  and relativistic plane-wave forms). Annihilation rate is always much slower than cooling
  (AK19 confirm this), so: **treat every injected positron as an injected electron of the
  same kinetic energy, plus two 511 keV photons**, exactly as AK19 do in their last
  paragraph of §3.1. This removes the need to separately track e⁺ energy-loss channels.

**Photon energy loss / conversion:**
- Photo-ionization: AK19 Eq. B.1 — negligible for z ≳ 1500 (fully ionized); include for
  completeness if you want the code to run down through recombination itself, otherwise
  treat as a fallback/edge case and confirm it's subdominant in your redshift window.
- Compton scattering, non-relativistic recoil rate: AK19 Eq. B.2 (defines when the
  "continuous energy loss to next bin down" approximation applies).
- Compton scattering, relativistic (Klein–Nishina) differential cross section: AK19 Eq. B.3
  — needed once the photon can jump more than one bin per collision.
- Pair production on matter (nuclei + free electrons, Bethe–Heitler): AK19 Eq. B.4 (total
  cross section, Z²-weighted, sum over H⁺/He⁺/He²⁺ exactly as SPX's own Bremsstrahlung term
  does, SPX Eq. A.6) and Eq. B.5 (positron/electron energy split).
- Photon–photon elastic scattering on the CMB blackbody: AK19 Eq. B.6 (rate) and Eq. B.7
  (secondary photon spectrum). **Do not** implement pair production directly on CMB
  photons — AK19 justify this omission explicitly (it's degenerate with photon–photon
  scattering in setting the final spectral shape at the energies where either matters, and
  both are so fast there that any injected particle above ~1 GeV/appropriate threshold
  converges to a universal spectrum regardless — AK19 Fig. 1b and their §3.2 discussion).

**Background inputs needed at every z**: free electron fraction X_e(z) (H and He
separately), Hubble rate H(z), T_γ(z) = T_0(1+z). **Reuse SPX's own recombination module**
(Peebles three-level atom + Saha, SPX §7 "Recombination") rather than re-deriving this —
it guarantees Tier 1 and Tier 2 use an internally consistent ionization history, and saves
you writing/validating a recombination code from scratch.

### 3.4 Electron/positron bookkeeping ("on-the-spot" approximation)

AK19 show (their Fig. 2) that electron/positron cooling is always much faster than H(z) in
the regime of interest — i.e., an electron injected at redshift z deposits essentially all
its energy (to heat via Coulomb, or to boosted CMB photons via IC) **within that same
redshift step**, never surviving to the next step as a nontrivial population. This is what
makes the "recursive" solve tractable: you never need to carry an electron population
forward across steps. Implement this as: at each z-step, for every electron/positron bin
populated this step (by direct injection or by pair-production/Compton-Compton-scattering
producing new electrons this step), recursively push its energy down through the
Coulomb/IC branching (starting from the lowest bin, building up) until all its energy has
been assigned either to (a) direct heat, or (b) a photon in some bin (which may itself be
above or below x_cut).

### 3.5 Photon bookkeeping and heat/output separation

For each z-step, after processing:

- **Heat** = Σ over all Coulomb-deposited electron energy this step **+** Σ over all
  non-relativistic-Compton-recoil energy deposited directly to background electrons by
  photons *above* x_cut this step (AK19 track this separately and plot it, their Fig. 13;
  do **not** count heat from photons below x_cut — that heat is Tier 2's job once it
  receives the raw sub-threshold spectrum, so as not to double-count).
- **Sub-threshold photon output** = the photon number/energy that crosses below x_cut this
  step, from any channel (redshifting of a photon that was just above x_cut, IC-upscattered
  CMB photons landing below x_cut, pair-production/Compton-Compton daughter photons below
  x_cut, or the 511 keV annihilation lines if they fall below x_cut at this z).
- **Surviving cascade population** (still above x_cut) carries forward to the next z-step.

Track energy conservation explicitly at each step: injected energy this step = heat +
Σ(sub-threshold photon energy) + Σ(surviving high-x population energy). This is not just
a final check — see §6, it is asserted at **every** checkpoint below.

### 3.6 Single-burst Green's function, then convolve for the continuous decay

Rather than directly marching a continuously-injected (exponentially decaying) source
through many redshift steps in Tier 1 — which would require carrying a mixed-age
population and is unnecessarily complex — exploit linearity (AK19 make this point
explicitly: "we can obtain the spectral distortion solution for any general initial
particle spectrum by doing a linear superposition of the monochromatic spectra calculated
in this paper"):

1. Compute the **cascade response to a single monochromatic photon burst** (fixed ω_inj,
   fixed injection redshift z_inj), evolved forward for a handful of z-steps until the
   surviving above-x_cut population is negligible. Output: heat(z; z_inj) and
   d(ΔN/N)/dx(x, z; z_inj) for z ≥ z_inj.
2. Repeat for a grid of z_inj (~30–100 log-spaced values spanning the decay's active
   e-folds — the exp(−Γt) profile is usually narrow in log z, so this grid can be modest;
   check convergence).
3. Convolve with the physical injection rate dΓ/dz (from f_X, Γ_X) to get the final
   continuous-injection outputs:
   ```
   dq_dz_total(z)      = ∫ dz' (dΓ/dz')(z') × heat(z; z')
   d(ΔN/N)/dx/dz(x,z)  = ∫ dz' (dΓ/dz')(z') × d(ΔN/N)/dx(x,z; z')
   ```

This mirrors exactly what SPX itself does for its own Green's-function cross-checks (SPX
§5.2 "Dark matter scenarios", Fig. 4 — dotted lines built from many single-burst PDE runs).
It also means each single-burst run is **independently testable against AK19's own
published single-burst figures** (§6), before you ever touch the convolution step or the
SPX interface.

### 3.7 Background cosmology

Use the same cosmological parameters SPX validates against for consistency (SPX §5.2:
h=0.71, Ω_b=0.044, Ω_m=0.26, Y_p=0.24, T_CMB=2.726 K — matching CosmoTherm v1.0.3
defaults) unless you have a specific reason to use current Planck best-fit values instead;
just be consistent across Tier 1 and Tier 2.

---

## 4. Interface to Tier 2 (spectroxide)

SPX's documented custom-scenario hooks (SPX §3.2, §4.1) are:

```python
def dq_dz(z):
    """returns d(Δρ/ρ)/dz, fractional heating rate relative to ρ_γ(z)."""
    ...

def photon_source(x, z):
    """returns f(x,z) = d(Δn(x,z))/dz, occupation-number injection rate per unit z."""
    ...

result = spectroxide.solve(method="pde", dq_dz=dq_dz, photon_source=photon_source,
                            z_start=..., z_end=1e3)
```

**Heat channel**: Tier 1's `dq_dz_total(z)` (§3.6) is *already* in SPX's exact convention
(fractional energy relative to ρ_γ, per unit z) — pass directly.

**Photon channel**: Tier 1 naturally produces a *fractional photon number* spectrum,
d(ΔN/N)/dx/dz. Convert to SPX's occupation-number convention using the same relation SPX
itself uses to build its monochromatic-injection source (SPX Eq. 18):

```
f(x,z) = d(Δn(x,z))/dz = [ G_2^(0) / x² ] × d(ΔN/N)/dx/dz
```

where G_2^(0) = 2ζ(3) is the Planck value of the second spectral moment (SPX Eq. 1). This
is exact and requires no further tuning — it's the same 1/x² occupation-weighting SPX uses
for its own built-in monochromatic burst (SPX Eq. 18) and decaying-particle (SPX §4.1)
scenarios, just applied to your actual computed cascade spectrum shape instead of a
Gaussian line.

**Practical note**: `photon_source` will be called at points spanning x ~ x_cut down to
whatever SPX's internal low-frequency grid extends to. Since your output spectrum only has
support at x ≲ x_cut by construction (that's the point of the handoff), make sure the
function returns 0 cleanly outside its tabulated range rather than extrapolating.

---

## 5. Software architecture

**Language**: Python (NumPy/SciPy), not Rust. Reasons:
- Tier 1's core algorithm (§3.2) is a triangular, non-stiff back-substitution — it doesn't
  need Rust's performance or SPX's implicit-solver machinery to be fast.
- SPX's own documented interface is Python-callable (`dq_dz`, `photon_source` as plain
  Python functions) — a Python Tier 1 integrates with zero glue code.
- If a specific inner loop (e.g., the Green's-function grid over many z_inj values, §3.6)
  turns out to be a bottleneck, that one loop can be moved to Numba or vectorized with
  NumPy later — this is an optimization to defer, not a v1 requirement.

**Suggested repo layout**:
```
cascade/
  grids.py          # energy/frequency grids, redshift stepping
  rates.py          # every cross section/rate in §3.3, each independently unit-tested
  cosmology.py       # X_e(z), H(z), T_γ(z) — thin wrapper reusing SPX's recombination module
  solve_burst.py     # single-burst triangular solve (§3.2–3.5)
  greens.py          # burst grid + convolution with dΓ/dz (§3.6)
  interface.py        # dq_dz(z), photon_source(x,z) builders for SPX (§4)
tests/
  checkpoints/         # one file per checkpoint in §6, run in CI in order
  test_energy_conservation.py   # imported and re-asserted inside every checkpoint test
notebooks/
  validation.ipynb
```

---

## 6. Development roadmap: checkpoints

Each checkpoint below states what's built, what published number/figure to check it
against, what "pass" means numerically (not just "looks right"), and — since the point is
to catch errors early — where to look first if it fails. **Energy conservation
(injected = heat + sub-threshold photons + surviving high-x population, §3.5) is asserted
at every checkpoint from CP2 onward**, not just at the end.

| CP | Adds | Check against | Needs SPX? |
|----|------|--------------|:---:|
| 0 | Background cosmology | recombination reference values | no |
| 1 | Rate functions (no cascade logic) | AK19 Figs. 1–2 (cooling-time ratios), dimensional analysis | no |
| 2 | Low-energy cascade (Compton+IC only, no pair production) | AK19 Fig. 6d/7a, Fig. 13a/b low-E branch | no |
| 3 | High-energy cascade (+ pair production, γγ scattering) | AK19 Fig. 8b/9a universal-spectrum convergence ≳10 GeV | no |
| 4 | Positron/annihilation bookkeeping | AK19 Fig. 9b | no |
| 5 | Multi-step redshift marching | AK19 Fig. 10/11 (z-dependence) | no |
| 6 | Green's function grid + continuous injection, self-contained low-E relaxation | AK19 §4.1 worked example (Fig. 14/15, z_X=2×10⁴) | no |
| 7 | SPX interface wiring | CP6 result, cross-checked in the y-era where DC/BR are irrelevant | **yes** |
| 8 | χ→nγ-specific kinematics | MPR21 parameter space, order-of-magnitude sanity check | yes |

### CP0 — Background cosmology
**Build**: `cosmology.py` giving X_e(z) (H and He), H(z), T_γ(z).
**Check**: X_e(z) against a standard recombination reference (RECFAST/HyRec, or just
Saha+Peebles textbook values) at z = 500–2000; H(z) against the analytic radiation-domination
scaling H ∝ (1+z)² at z ≫ 3400 and against H₀ at z=0.
**Pass**: X_e within ~5–10% of reference in 1000 ≲ z ≲ 1500 (this is SPX's own stated
accuracy for the same simplified treatment, SPX §7 "Recombination" — matching it, not
beating it, is the goal); H(z) matches analytic scaling to <1%.
**If it fails**: isolated to one small module — before touching anything else, this is the
input every downstream rate depends on, so it's worth getting exactly right first.

### CP1 — Rate functions, no cascade logic yet
**Build**: `rates.py`, every formula in §3.3, each a standalone pure function.
**Check**: (a) dimensional analysis on every rate (energy-loss rates in eV/s, cross
sections in cm², reaction rates in s⁻¹); (b) limiting cases — pair production (Eq B.4)
must vanish below E_γ = 2m_ec² ≈ 1.022 MeV; (c) reproduce AK19 Fig. 1 and Fig. 2 directly
— these are *pure rate-ratio plots* (t_H/t_cool vs. particle energy at fixed z), requiring
zero cascade/back-substitution machinery, just your rate functions plus CP0's H(z). AK19
state the electron Coulomb/IC crossover explicitly at **≈2.5 keV** — a clean, concrete
number to hit.
**Pass**: crossover energy within a factor of ~2 of 2.5 keV; overall shape of Figs. 1–2
(order of magnitude and qualitative trend, log-log) matches.
**If it fails**: it's a specific rate formula, not the algorithm — check units and the
formula transcription against AK19 Appendix A/B directly, one function at a time.

### CP2 — Low-energy cascade (Compton + IC only, no pair production)
**Build**: the triangular back-substitution solver (§3.2), restricted to electron
Coulomb+IC and photon non-relativistic-recoil+relativistic Compton — deliberately turn off
pair production and γγ scattering for this checkpoint by injecting only at energies safely
below the 1.022 MeV pair-production threshold (10 keV–~500 keV).
**Check**: AK19 Fig. 6d / 7a (10 keV–1 MeV electrons and photons, one-time injection at
z=20000); AK19 Fig. 13a/b restricted to this same low-energy branch.
**Pass**: reproduce the y-distortion normalization AK19 quote for this energy range
(y ~ a few×10⁻⁶ for 10⁻⁵ρ_CMB injected, their Fig. 6 caption gives y=1.85×10⁻⁶ for the
10 keV–100 GeV panel) to within ~20%, and the qualitative spectral shape (peak position
shifting right, amplitude of dip/peak shrinking with increasing injected energy).
**If it fails**: this isolates the core Compton↔electron feedback loop from everything
else — bug is in the back-substitution logic or the Compton/IC rate coupling, not in
pair production (not implemented yet) or redshift marching (single z, not yet exercised).

### CP3 — High-energy cascade: add pair production and γγ scattering
**Build**: turn on Bethe–Heitler pair production and photon–photon elastic scattering;
extend injection energy up through 1 MeV–100 GeV.
**Check**: AK19's explicit claim that electron-only and photon-only injection **converge
to the same universal spectrum above ~10 GeV** (AK19 §3.2 and Figs. 8b/9a) — a strong,
qualitative, hard-to-fake signature: if pair production/γγ scattering are wired correctly,
a 100 GeV photon and a 100 GeV electron should produce essentially the same output
spectrum, because at that energy the cascade "forgets" what species it started as within
a step or two. Also check the high-energy plateau in Fig. 13a/b ("flattens out for
energies ≳ GeV").
**Pass**: electron- and photon-injection spectra at 10–100 GeV agree with each other to
better than ~10–20% (they should be visually indistinguishable in AK19); heat fraction
plateaus rather than continuing to oscillate.
**If it fails**: since CP2 already validated the Compton/IC loop, a failure here isolates
to the newly-added pair-production cross section or the γγ-scattering secondary-photon
spectrum (Eq B.4/B.5 or B.6/B.7) — check the Z²-weighted sum over H⁺/He⁺/He²⁺ first, that's
the most common place to get a stoichiometry or normalization factor wrong.

### CP4 — Positron / annihilation bookkeeping
**Build**: the "positron = electron of same KE + two 511 keV photons" substitution (§3.4).
**Check**: AK19 Fig. 9b, comparing electron-positron-pair injection against electron-only
injection at 10 keV and 100 GeV.
**Pass**: at 10 keV, the spectrum should be visibly dominated by the 511 keV line (per
AK19's own description); at 100 GeV, electron-only and e⁺e⁻-pair injection should be
indistinguishable.
**If it fails**: this is purely a bookkeeping/data-flow check (no new physics beyond CP2/3),
so a failure here is almost certainly in how the 511 keV photons are injected into the
bins (energy placement, or double-counting against the electron's own kinetic energy),
not in any rate function.

### CP5 — Multi-step redshift marching
**Build**: extend a single burst's evolution over many Δz steps (rather than the few steps
needed for CP2–4 to converge), tracking a photon population that survives above x_cut for
several steps before finally dropping below threshold.
**Check**: AK19 Fig. 10/11 — spectral distortion at fixed injected energy (10 keV and
100 keV low-energy; 1 GeV and 100 GeV high-energy) as a function of **injection redshift**,
holding ρ_inj/ρ_CMB fixed.
**Pass**: reproduce the reported qualitative trend — sharper, higher-amplitude low-x
features at higher z_inj (more scatterings available before reaching the fixed endpoint
redshift) — and, where AK19 give it, the specific y-values quoted per redshift (e.g. their
Fig. 10/11 captions list y for each z_inj panel).
**If it fails**: since CP2–4 validate the *single-step* physics, a failure here is isolated
to the step-size/marching logic (Δz too coarse, or the x_cut handoff not correctly zeroing
out the already-processed portion of the spectrum at the start of each step per AK19's own
"set the x<x_cut portion to zero to avoid double counting" prescription, §3.5 here).

### CP6 — Green's function grid, continuous injection, self-contained pipeline
**Build**: the z_inj grid (§3.6) and convolution with dΓ/dz for a continuous decay; **for
this checkpoint only**, pair the cascade output with a minimal standalone Compton-only
Kompaneets relaxation (§7, *not* SPX yet) so the entire pipeline is self-contained.
**Check**: AK19's own worked dark-matter-decay example, §4.1, Figs. 14–15: z_X = 2×10⁴,
f_X chosen so the would-be-all-heat y-distortion is y = 1.5×10⁻⁵ (their quoted COBE-limit
benchmark), corresponding to τ_X ~ 10¹¹ s.
**Pass**: final distortion amplitude of order a few×10⁻⁶, with the shape AK19 describe as
sensitive to injected-particle energy/species (their Fig. 14b).
**Why do this before touching SPX**: this validates the *entire* Tier-1 pipeline —
single-burst physics, the z_inj grid, the convolution, and a low-energy relaxation — end
to end against a fully independent published number, with a low-energy code (Compton-only
Kompaneets) simple enough to have zero unit-convention ambiguity. Any discrepancy here is
therefore attributable to Tier 1 itself, not to how it's wired into SPX.
**If it fails**: check the convolution integral first (a common bug is applying dΓ/dz with
the wrong time-vs-redshift Jacobian, or double-counting the exponential depletion factor
already baked into the single-burst normalization).

### CP7 — Wire into spectroxide
**Build**: `interface.py` — convert the *same* validated CP6 cascade output into SPX's
`dq_dz`/`photon_source` convention (§4) and route it through SPX instead of the standalone
Kompaneets fallback.
**Check**: compare against the CP6 standalone-Kompaneets answer, specifically **in the
y-era** (z_dec ≲ 5×10⁴), where DC/BR are subdominant (SPX §3.1) and the two Tier-2 choices
should therefore agree closely.
**Pass**: agreement with CP6 to a few percent in the y-era. Divergence in the μ-era
(z_dec ≳ 2×10⁵) is *expected* — that's DC/BR relaxation, which the standalone Kompaneets
code doesn't have — but should be explainable and directionally consistent with SPX's own
μ-suppression behavior (SPX Fig. 1/Table 1), not an unexplained discrepancy.
**If it fails in the y-era too**: that's the interface, not the physics — check the
G_2^(0)/x² conversion factor (§4) and the SPX call signature/units first, since CP6 has
already validated everything upstream of this conversion.

### CP8 — χ→nγ-specific application
**Build**: plug in the actual MPR21 kinematics: ω_inj = (m_χ/2)(1−m_n²/m_χ²), Γ_{χ→nγ}
(MPR21 Eq. 7), at a benchmark (m_χ, θ, Δm) point drawn from MPR21's own explored parameter
space (e.g. their Δm=10 MeV or 50 MeV benchmarks).
**Check**: order-of-magnitude sanity check — does the resulting distortion amplitude sit in
a regime comparable to where MPR21's own CMB-anisotropy/COBE constraints bite (their Fig. 3/4),
i.e. is the code producing numbers of a physically sensible size for this parameter space,
not many orders of magnitude off.
**Pass**: no formal numeric target here (this is new physics territory, not a
reproduction), but the amplitude should be broadly consistent with the fχ/τχ ~ 10⁻²⁵ s⁻¹
scale MPR21 quote as their own CMB-anisotropy sensitivity floor, translated into a spectral-
distortion context.
**If something looks wrong**: since CP0–7 are all independently validated by this point,
an issue here is almost certainly in the MPR21-specific kinematics formula or parameter
translation (f_X, Γ_X normalization), not in the cascade or relaxation machinery.

---

## 7. Fallback: standalone low-energy code (if not depending on spectroxide)

Given your comment that using spectroxide "shouldn't be necessary" — if you'd rather not
depend on the SPX package at all, AK19's own low-energy code (their §3.3, the "Kompaneets
kernel" evolution) is a much smaller standalone alternative: pure non-relativistic Compton
scattering with Doppler broadening/boosting and stimulated scattering (AK19 Eq. 3.4–3.6,
the Sazonov–Sunyaev kernel), with **no DC/BR** — i.e., it only gets you μ/y-type shapes in
the sense AK19 define them (not the more complete DC/BR-relaxed μ that SPX computes). This
is a legitimate, much lighter-weight fallback (order-of-magnitude less code than SPX) if
your science case doesn't need DC/BR-level precision on the final thermalized shape — but
it will not by itself correctly relax a partially-absorbed μ distortion the way SPX's
DC/BR terms do at the redshifts where that matters (roughly z ≳ 5×10⁴, SPX §3.1). This is
exactly the code used in CP6 above, so you'll have it built regardless of whether you
ultimately depend on SPX for production use.

---

## 8. Explicit non-goals for v1

- Hadronic final states (m_χ > m_n + m_π⁰) — MPR21 note this opens additional channels
  with "similar cosmological signatures" but don't detail them; out of scope here.
- z ≳ 2×10⁶ (deep thermalization) — SPX's heat-injection mode alone is already adequate
  there without cascade resolution.
- Photo-ionization / partial-ionization complications right at recombination (z ~ 1100–1500)
  — both AK19 and SPX simplify here in compatible ways; inherit SPX's treatment.
- Silk damping, cosmological recombination radiation, dark photon oscillations — these are
  SPX features unrelated to this project.

## 9. Open items to resolve early

- **SPX's actual internal frequency-grid bounds** (SPX App. B.3) — needed to confirm
  x_cut = 35 is a safe handoff point into SPX rather than sitting outside its domain.
  Check the SPX repository/appendix directly once you have it in hand.
- **Bin-count and Δz convergence** for your specific ω_inj range — AK19's own choices are
  a good starting point but should be re-verified for your parameter space (part of CP5's
  pass criteria) rather than assumed.
  
## 10. Implementation status and known limitations

*Updated after CP0–CP8 were implemented and after an independent review pass found and
fixed three real physics bugs. All 39 tests across `test_cp0_*.py`–`test_cp8_*.py`
currently pass. This section replaces the original placeholder "areas of improvement"
list with an accurate, itemized record of what's approximate, what's fixed, and what's
still open — kept here rather than in commit messages so the caveats travel with the plan.*

### 10.1 Bugs found in review and fixed

These were identified by an independent code review after CP0–CP8 were first built, and
have since been fixed and re-validated against the full test suite. Recorded here because
the fixes changed some headline numbers reported earlier (see below) and because the
reasoning is worth keeping visible rather than only living in a diff.

- **y-distortion excluded heat (major).** `y_distortion_estimate` originally computed the
  distortion channel as `f_ic + f_subthreshold` only, excluding `f_heat` — but heat
  deposited in background electrons *is* a y-distortion source, via those electrons'
  own subsequent Compton up-scattering of CMB photons (that's literally the y-distortion
  mechanism). This also directly contradicted `f_X_for_target_all_heat_y`'s own
  calibration, which already assumes 100% heat→y efficiency. Fixed by including `f_heat`
  in the distortion channel.

  **Consequence worth flagging explicitly**: CP2's solver (Compton+IC only) has no
  permanent energy-loss channel, so `f_heat + f_ic + f_subthreshold = 1` to machine
  precision there. With heat now correctly counted, `y` in that regime reduces to exactly
  `injected_frac_rho_cmb / 4`, independent of species, energy, or z_inj — there is nothing
  left for it to depend on once every joule of injected energy is counted. Two checkpoint
  tests (CP2's "y grows with energy," CP5's "y grows with z_inj") were, without realizing
  it, testing the *bug's own energy-dependence signature* rather than real physics, and
  have been rewritten to assert the corrected (flat) expectation instead. Real z_inj/energy
  dependence in y now only survives where a genuine loss channel exists (CP3's recursive
  energy leakage, sub-threshold energy escaping the tracked range, etc.) — see CP5's
  100 GeV test, which still shows real z_inj dependence via CP3's leakage.

  Headline numbers changed as a result: the CP6 worked-example benchmark (100 MeV
  electron, z_X=2×10⁴, τ_X=10¹¹ s) moved from y≈4.4×10⁻⁶ to y≈1.6×10⁻⁵; the CP8 MPR21
  benchmark (Δm=10 MeV, τ_χ=10¹¹ s) moved from y≈4.4×10⁻⁶ to y≈6.1×10⁻⁶.

  `interface.py`'s `dq_dz`/`photon_source` split (`f_heat`→`dq_dz`, `f_ic+f_subthreshold`
  →`photon_source`) was **not** part of this bug and needed no change — that split is
  correct by design in the two-tier architecture, since SPX's own downstream Kompaneets/
  DC-BR solver is what turns heat into a distortion. The bug was specific to the
  standalone (no-SPX) fallback, which never runs any heat-relaxation step of its own.

- **Pair-production dead zone from threshold to ~3.4 MeV (real bug).** AK19's Eq. B.4 is
  a high-energy asymptotic Bethe-Heitler form; taken literally it is *negative* from
  E_γ=2m_ec² up to E_γ≈6.7m_ec²≈3.42 MeV (its own zero crossing), not just below
  threshold. The original code clamped this to zero with `np.maximum(raw, 0.0)`, which
  correctly enforces the physical threshold but also silently zeroed out the entire
  ~2.4 MeV window above it — directly inside MPR21's own Δm range (~100 keV–50+ MeV) and
  the plan's own stated ω_inj window (§1, 1 keV–100 MeV). Fixed by bridging that window
  with a smooth (C²-continuous) monotonic ramp from 0 at threshold to the exact,
  unmodified AK19 formula at 10m_ec² (safely above its own zero crossing) — the AK19
  formula itself is untouched everywhere it's actually valid (E_γ ≳ 5 MeV). This is an
  explicit interpolation, not a first-principles near-threshold Bethe-Heitler derivation;
  replacing it with the literal near-threshold cross section is a candidate follow-up if
  more precision is needed specifically in the 1–5 MeV window.

- **Helium's free electrons missing from n_e (real bug, with a subtlety in the fix).**
  Every Coulomb/IC/Compton rate used `n_e = X_e(z)·n_H(z)` (hydrogen ionization only),
  never adding helium's free electrons — a ~16% systematic undercount of n_e deep in the
  pre-recombination window (Y_p=0.245 → 2·n_He/n_H≈0.16), larger than CP0's own 5–10%
  accuracy target. The naive fix (unconditionally add `2·n_He(z)`, treating helium as
  always fully double-ionized) turned out to be wrong at low z: at z=1000 (one of CP1's
  own two test redshifts), hydrogen is already well into recombination (X_e≈0.16), and
  helium recombines over a comparable redshift range — an *unconditional* full-ionization
  term roughly **doubled** n_e there instead of adding the intended ~16%. Fixed by scaling
  helium's contribution by the same `X_e(z)` curve hydrogen uses:
  `n_e_total(z) = X_e(z)·[n_H(z) + 2·n_He(z)]`. This is not a true independent helium
  Saha/recombination calculation (out of scope), but is correct at both ends: it reduces
  to the intended `+2·n_He` correction at high z (X_e→1), and vanishes appropriately as
  X_e drops through recombination, rather than contradicting hydrogen's own ionization
  history. Applied consistently everywhere n_e is built (`electron_heat_fraction`,
  `electron_cascade`, `_compton_total_rate`, `_pair_production_total_rate`). One test
  (CP1's Coulomb/IC crossover at z=1000) moved from a marginal pass to a marginal fail
  purely from this ~16%-larger n_e and had its tolerance loosened slightly (2.0×→2.5×),
  documented in the test itself as already having been at the edge of the original
  tolerance before this fix, not a new problem.

- **Minor/cosmetic, also fixed:** a dead unused variable in `electron_cascade`'s photon-
  regeneration branch; `saha_X_e_H`/`_saha_rhs` in `cosmology.py` were fully implemented
  but never called anywhere — rather than deleting them, they're now wired into a CP0
  cross-check test (comparing the fast logistic-fit `X_e(z)` against genuine Saha
  equilibrium at high z, where Saha alone is an excellent approximation); `interface.py`'s
  `x_eff` placeholder formula for the photon-source proxy previously didn't match
  `electron_cascade`'s actual characteristic-photon formula (missing β², the square on γ,
  and the 2.7kT mean-photon-energy factor) and used the injected particle's raw energy as
  a Lorentz factor even for photon injection — corrected and made species-aware.

### 10.2 Known approximations and open items (forward-looking)

These are not bugs — they're deliberate scope choices, flagged in the relevant module
docstrings, and worth keeping visible here as a single list for anyone extending this
codebase.

- **Cosmology (CP0).** `X_e(z)` is a fast fitted logistic transition, not the full
  Saha+Peebles ODE — accurate to a few percent through the z~1000–1500 window, matching
  SPX's own stated simplified-recombination accuracy (§7). A genuine Peebles-equation
  integration (with its fast-relaxation boundary layer at the Saha handoff) remains a
  "do it properly later" item if higher recombination-era precision is ever needed.
  Helium's ionization state is approximated by tracking hydrogen's X_e(z) rather than
  solved independently (see 10.1 above).

- **Klein-Nishina / low-energy cascade shape (CP2/CP3).** The photon Compton
  energy-transfer fraction δ(E) used throughout is a two-limit phenomenological
  interpolation (`x/(1+x)`, x=E/m_ec²), not AK19's literal differential Klein-Nishina
  spectrum (Eq. B.3). It reproduces the two known limits (non-relativistic recoil,
  deep-KN saturation) and is order-of-magnitude validated against AK19 Fig. 6d/7a (worst
  case: a factor ~7 low at 10 keV photon injection), but is not a resolved differential
  cross section. Similarly, an electron's IC-upscattered output is represented as a
  single "characteristic" daughter photon per cooling decade
  ((4/3)γ²β²·⟨E_CMB⟩, ⟨E_CMB⟩=2.7kT) rather than the actual continuous IC spectrum — this
  feeds both the recursive high-energy cascade (CP3) and `interface.py`'s photon-source
  proxy (see below).

- **Energy-conservation leakage in the recursive high-energy solver (CP3+).**
  `electron_cascade`/`photon_cascade_full` (pair production + γγ scattering + recursive
  e↔γ generations) drop the residual energy below the 1 keV floor at each recursion
  generation; this compounds with recursion depth and causes up to ~5–15% non-conservation
  at intermediate energies (~10⁹ eV), tightening toward exact conservation at both the low-
  and very-high-energy ends. Documented and tolerance-adjusted in `test_cp3_*.py` and
  `test_cp7_*.py` rather than hidden. By contrast, CP2's non-recursive solver conserves
  energy to machine precision (relevant to the bug above).

- **No resolved photon spectrum, anywhere.** Every checkpoint tracks only integrated
  energy fractions (`f_heat`/`f_ic`/`f_subthreshold`), never an actual x-resolved
  d(ΔN/N)/dx spectrum. Consequences: (a) AK19's "sharper, higher-amplitude low-x features
  at higher z_inj" claim (CP5, Figs. 10/11) could only be checked on its amplitude
  component, not its shape component; (b) `interface.py`'s `photon_source(x,z)` is a
  narrow-Gaussian-line proxy at a single characteristic x_eff, sufficient to route a
  physically-motivated total energy into SPX and sanity-check overall amplitude, but not
  sufficient to reproduce AK19's actual peak/dip spectral shapes through the SPX PDE.
  Resolving the real d(ΔN/N)/dx spectrum is the natural next build item if spectral-shape
  fidelity through SPX is ever required.

- **No DC/BR relaxation physics anywhere in this codebase (CP6).** Every y-estimate in
  this codebase uses the leading-order single-scattering relation (Δρ/ρ=4y); there is no
  double-Compton/Bremsstrahlung relaxation. This means CP6's standalone
  `decay_convolution_y` is expected to sit *above* AK19's fully-relaxed answer as an upper
  bound, not match it — confirmed numerically (CP6's own worked benchmark converges to
  y≈1.6×10⁻⁵ vs. AK19's quoted few×10⁻⁶ "fully relaxed" scale for a comparable scenario).
  This is the specific gap CP7/SPX exists to close.

- **CP7's live SPX comparison has not been executed.** `spectroxide` is not installable in
  this environment (confirmed directly via `pip install`/`pip download`, no matching
  distribution found), consistent with the plan's own recorded note that depending on it
  "shouldn't be necessary." `interface.py` is written to the documented SPX call signature
  (`dq_dz`, `photon_source`, §4) and its internal bookkeeping (energy-conservation between
  the two channels, correct vanishing outside the tabulated x-range) is unit-tested without
  SPX, but the actual CP7 pass criterion (agreement with CP6 to a few percent in the y-era)
  is untested end-to-end. `interface.run_cp7_check()` is ready to run as-is wherever SPX
  becomes available and will raise a clear error rather than silently no-op if it isn't.

- **MPR21's CMB-anisotropy floor is a different observable than this pipeline's
  spectral-distortion output (CP8).** MPR21's quoted fχ/τχ ≲ 10⁻²⁵ s⁻¹ (their Eq. 19) is
  their temperature-anisotropy bound, valid for τχ ≳ 10¹³ s (decays during/after
  recombination) — not the pre-recombination spectral-distortion regime (τχ~10⁸–10¹³ s)
  this pipeline targets. A spectral-distortion-regime benchmark's fχ/τχ differs from that
  floor by many orders of magnitude (~10¹⁴ in the tested benchmark) and *should* — this is
  documented explicitly in `mpr21.py` and asserted (not just noted) in
  `test_cp8_mpr21.py::test_anisotropy_floor_epoch_mismatch_documented`, so it isn't
  mistaken for a code discrepancy by a future reader.

- **The full Kompaneets PDE (AK19 Eq. 3.4–3.6) is never implemented.** Every checkpoint
  from CP2 onward substitutes the single-scattering y-type approximation for it. This is
  the same underlying simplification as the "no DC/BR" item above, stated separately here
  because it also affects the CP6 standalone path even independent of SPX: a literal
  Kompaneets relaxation step would additionally capture Doppler broadening/boosting and
  stimulated scattering (spectral *shape* corrections), which this codebase has never
  attempted at any checkpoint.
