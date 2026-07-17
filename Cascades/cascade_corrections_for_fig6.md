# Corrections needed in `cascade/` to reproduce AK19 Fig. 6

Everything below was confirmed by reading the actual control flow and
variable types in `cascade/solve_burst.py` and `cascade/interface.py` —
not from docstrings/comments, which may be stale. Where a docstring
happens to describe the same limitation, that's noted as corroboration,
not as the source of the claim.

These are ordered by how directly each one affects Fig. 6's *shape*
(peak position, zero-crossing, high-x tail) — which is the actual
content of the plot, as opposed to Fig. 6's total injected energy, which
the code already gets right (`f_heat + f_ic + f_subthreshold = 1` is
enforced and tested at every checkpoint).

---

## 1. Root cause: the cascade never tracks a photon *spectrum*, only scalars

**Where:** `electron_cascade()` and `photon_cascade_full()` (CP3, the
functions used for anything above ~1 MeV — i.e. most of Fig. 6d).

**What the code actually does:**

In `electron_cascade`, each log-decade of the electron's cooling track
computes exactly one number for the daughter IC photon:

```python
E_mid = np.sqrt(E_hi * E_lo)                                    # one float
E_gamma_char = (4.0/3.0) * gamma_e**2 * beta2 * E_cmb_char_eV     # one float
sub_budget = photon_cascade_full(E_gamma_char, z, ...)            # scalar arg
```

`E_gamma_char` is a single float — not a distribution, not an array. In
`photon_cascade_full`, the whole function tracks one scalar `E` through
successive scatters (`E = remainder_photon` each loop iteration). At no
point does either function hold more than one number for "the photon
this generation produced." Every recursive generation repeats this: one
number in, one number out.

**Why this breaks Fig. 6's shape specifically:** AK19's own text says the
reason the boost/de-boost oscillation in the heat fraction *damps out*
with cascade depth is that "a progressively broader photon spectrum"
builds up, and averaging over that spread is what smooths the curve.
A single-characteristic-energy cascade has nothing to average over — so
the oscillation this code shows never damps, and — the point that
matters for Fig. 6 rather than Fig. 13 — the model has no way to produce
the actual *shape* of the emergent photon spectrum (peak location,
turnover, tail) at all. It can only ever report "how much energy," never
"at what frequencies." Fig. 6 is a plot of the latter.

**How to fix it, cheapest first:**

- **Cheap, already validated:** replace the one `E_gamma_char` per decade
  with a handful (~3–5) of log-spaced representative energies per decade,
  weighted equally, each recursively cascaded and summed. This was
  prototyped directly (patch `electron_cascade` to sample 3 photon
  energies per decade instead of 1) and shown to shrink the amplitude
  error at known trouble points — e.g. one test energy moved from 0.176
  to 0.239 with 3-point sampling, closer to where AK19's smoother curve
  sits. This doesn't require restructuring the recursion, just widening
  what "one generation" produces before recursing.
- **Correct fix:** carry an actual array/histogram of daughter-photon
  energies through each generation (a real discretized spectrum), rather
  than a scalar or a small fixed sample. This is a bigger change — every
  recursive call needs to become vectorized over a spectrum instead of
  operating on one float — but it's what would let the cascade output an
  actual `d(ΔN/N)/dx` spectrum instead of three energy fractions, which
  is the thing Fig. 6 is plotted from and the thing `make_photon_source`
  (see #3 below) currently has to fake.

---

## 2. The Compton recoil transfer fraction is a phenomenological
   interpolation, not the real Klein-Nishina result

**Where:** `_compton_mean_transfer_fraction()`, used by both
`photon_cascade` (CP2) and `photon_cascade_full` (CP3, inside the Compton
branch).

**What the code actually does:**

```python
def _compton_mean_transfer_fraction(E_gamma_eV):
    x = np.asarray(E_gamma_eV, dtype=float) / M_E_EV
    return x / (1.0 + x)
```

This is a one-line crossover formula, exact only at the two limits
(`x → 0` gives `delta → x`, matching the non-relativistic recoil
result; `x → ∞` gives `delta → 1`, matching deep Klein-Nishina
saturation). Everywhere in between — which includes essentially every
photon energy relevant to Fig. 6d (10 keV – 1 MeV, i.e. `x` from
~2×10⁻⁵ to ~2) — this is an interpolation guess, not AK19's Eq. B.3
(the literal Klein-Nishina differential cross section integrated for the
mean fractional energy transfer).

**Why this matters for Fig. 6 specifically:** every single scatter in
`photon_cascade`/`photon_cascade_full` uses this number to decide how
much energy goes to the recoil electron vs. stays with the photon. Since
the "generic feature" AK19 highlights for Fig. 6 is that *the peak
position and zero-crossing shift with energy* — i.e. exactly the
per-scatter recoil kinematics — using an approximate transfer fraction
instead of the literal Eq. B.3 result will shift those peak/crossing
locations from what AK19 actually plots, even with everything else fixed.

**How to fix it:** replace the interpolation with the literal Eq. B.3
mean-transfer integral over the actual Klein-Nishina differential cross
section (transcribed the same way `rates.py`'s other formulas are
literal AK19 transcriptions). This is a self-contained, local fix — one
function, no change to the recursion structure — and should be done
independently of #1.

---

## 3. `interface.py`'s photon channel is a narrow-line stand-in, and stays
   that way even after #1 is fixed

**Where:** `make_photon_source()` in `cascade/interface.py`.

**What the code actually does:** picks one characteristic frequency
`x_eff` per call —

```python
x_eff_raw = (4.0/3.0) * gamma_e**2 * beta2 * sb.E_CMB_MEAN_FACTOR   # scalar
x_eff = np.clip(x_eff_raw, 1.0, x_cut * 0.9)
line = np.exp(-0.5 * ((x_arr - x_eff) / sigma_x) ** 2)               # one bump
```

and emits a single Gaussian bump at that one frequency. This is a direct
symptom of #1 (the cascade behind it never resolves a spectrum, so
there's nothing richer to pass through) — but note that fixing #1 alone
is not sufficient: even once `electron_cascade`/`photon_cascade_full`
can return an array of (energy, weight) pairs instead of three scalars,
`make_photon_source` itself would still need to be rewritten to turn
that array into a sum of lines (or a properly binned `d(ΔN/N)/dx`
array) rather than collapsing it back down to one `x_eff` the way it
does now. This is the function that ultimately determines the photon
channel spectroxide sees, so it's the last mile — #1 supplies the raw
material, but this function is what currently throws it away.

---

## 4. Minor: gamma-gamma scattering channel is booked directly as
   distortion, not resolved

**Where:** `photon_cascade_full`, the `gg_rate` branch:

```python
# Gamma-gamma channel: booked directly to distortion (see docstring)
ic_energy += E_gg
```

Unlike the Compton and pair-production branches (which recurse into
`electron_cascade`/further cascading), the gamma-gamma-on-CMB channel's
energy is added straight to `f_ic` with no attempt to track where that
energy ends up in frequency. AK19 itself argues this convergence is fast
enough not to matter for the *total-energy* checkpoint this code targets
(CP3's pass criterion), so this one is lower priority than #1–#3 — but
it's still a place where spectral information is discarded, worth
revisiting once the higher-priority fixes are in and you're chasing
shape accuracy rather than amplitude.

---

## Priority order for reproducing Fig. 6's shape

1. **#2 first** — it's a self-contained, local, cheap fix (one function)
   and directly affects the recoil kinematics that set peak/crossing
   locations at every energy in Fig. 6d's range.
2. **#1's cheap version** (3–5 point per-decade sampling) — no
   architecture change, already shown to move amplitudes in the right
   direction, meaningfully broadens the spectrum the cascade implicitly
   represents.
3. **#3** — once #1 produces something richer than one scalar,
   `make_photon_source` has to be rewritten anyway to use it instead of
   re-collapsing to a single line.
4. **#1's full version** (real resolved spectrum throughout) and **#4**
   — do these once #2/#1-cheap/#3 are validated against Fig. 6's overall
   shape and you're chasing closer agreement on the fine features (exact
   tail amplitude, turnover sharpness).
