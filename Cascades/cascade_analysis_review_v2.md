# Review of the previous Cascades analysis

I re-ran everything the previous analysis describes (test suite, both direct-script
and pytest invocations, `reproduce_ak19_plots.py`, `reproduce_fig3.py`) and then went
further and *instrumented* the energy-budget functions to trace exactly where energy
goes at each recursion step, rather than just re-observing the same top-level totals.
That changed the picture substantially for item 1.

**Bottom line: the previous analysis found a real bug but the wrong dominant cause.
The pair-production rest-mass leak it describes is real but small (~1-2% of energy).
There is a second, much larger bug in `electron_cascade` — never mentioned in the
previous analysis — that accounts for the bulk of the ~13% non-conservation, and
ironically it is almost exactly what `test_cp3`'s own (dismissed) comment already
said was happening.** Items 2 and 3 from the previous analysis both check out, with
one added nuance on item 2.

---

## 1. REVISED — the dominant energy leak is in `electron_cascade`, not pair production

### What I found

`electron_cascade` (CP3, `solve_burst.py`) walks an electron's cooling track from
`E0_eV` down to `E_min_eV` in log-spaced decades and sums each decade's heat/IC
split:

```python
n_decades = max(4, int(np.ceil(np.log10(E0_eV / E_min_eV) * 4)))
E_edges = np.geomspace(E0_eV, E_min_eV, n_decades + 1)
...
for i in range(n_decades):
    E_hi, E_lo = E_edges[i], E_edges[i + 1]
    dE = E_hi - E_lo
    heat_energy += f_heat_local * dE
    ...
f_heat = heat_energy / E0_eV
f_ic = ic_energy / E0_eV
return {"f_heat": ..., "f_ic": ..., "f_subthreshold": 0.0}
```

The loop only accounts for the energy the electron loses *while cooling from `E0_eV`
down to `E_min_eV`* — i.e. `sum(dE) = E0_eV - E_min_eV`, not `E0_eV`. The remaining
`E_min_eV` of kinetic energy the electron still carries once it hits the floor is
never added anywhere: `f_subthreshold` is hardcoded to `0.0` rather than accumulating
that leftover, unlike `photon_cascade_full`, which does correctly bank its own
leftover into `f_subthreshold`.

For a *directly injected* high-energy electron this is invisible (`E_min_eV/E0_eV` is
tiny). But `electron_cascade` is also called, recursively, on every Compton-recoil
electron generated inside `photon_cascade_full`'s scattering loop — and many of those
recoil electrons, especially in the later scatters of a cascade, land with energies
only modestly above the 1000 eV floor, where `E_min_eV/E0_eV` is *not* small. I traced
a single 30 keV photon's cascade and found its first Compton recoil electron
(1596 eV) alone loses 63% of its assigned energy to this gap:

```
electron_cascade(1596 eV): sum(dE over 4 decades) = 596 eV  (not 1596 eV)
                            heat+ic = 596 eV  ->  total = 0.373, not 1.0
```

Multiply that across the hundreds of recoil electrons a real cascade generates and it
compounds into exactly the ~13% deficit the previous analysis attributed entirely to
pair production.

### Isolating it from the pair-production bug

I confirmed this is a *separate* mechanism, not a restatement of item 1, three ways:

**a) It occurs with zero pair-production events.** `electron_cascade` called directly
(no photon injection at all) at energies with no pair production anywhere in its
recursive tree:

| E0 (eV) | reported total | pair events in this call's whole tree |
|---|---|---|
| 3.0e7 | 0.8985 | 0 |
| 6.0e7 | 0.8755 | 0 |
| 1.0e8 | 0.9003 | 0 |

**b) Fixing *only* this bug (booking the leftover `E_min_eV` as heat, consistent with
the function's own convention for `E0_eV <= E_min_eV`) restores exact conservation on
its own**, with the pair-production bug left untouched:

| species / E0 | original total | E_min-floor fix only |
|---|---|---|
| e, 3e7 | 0.8985 | 1.0000 |
| e, 6e7 | 0.8755 | 1.0000 |
| e, 1e8 | 0.9003 | 1.0000 |
| gamma, 6.58e7 | 0.8707 | 0.9845 (matches the pair-loss fraction exactly — see below) |
| gamma, 1e8 | 0.8906 | 0.9898 |

**c) Fixing *only* the pair-production bug (re-injecting `2*m_e c^2` as two
annihilation photons, exactly mirroring what `positron_burst_energy` already does)
recovers only 1-2 percentage points**, leaving the bulk of the deficit untouched:

| E0 (eV, gamma) | original | pair-fix only | missing after pair-fix |
|---|---|---|---|
| 6.58e7 | 0.8707 | 0.8855 | 11.4% |
| 1.00e8 | 0.8906 | 0.9004 | 9.96% |
| 1.00e9 | 0.9508 | 0.9517 | 4.8% |

**With both fixes applied together, every energy I tested (1e6 – 1e11 eV, both
species) conserves to 1.0000.**

### Why the previous analysis's table looked like it supported the pair-production
diagnosis

The previous analysis's own energy-scan table (2 MeV → 100 GeV) — which I
independently reproduced number-for-number — is real and its shape (worst around
30 MeV–1 GeV, recovering above ~10 GeV) is real. But that shape is actually a
signature of the `electron_cascade` floor bug, not pair production: it's the energy
range where (i) Compton recoil electrons are numerous and land near the E_min floor
often enough for the compounding to bite, while (ii) the cascade hasn't yet reached
the ultra-relativistic regime where a single scatter carries away most of the energy
in few-enough steps that the floor loss becomes negligible again. The traced
cumulative pair-production loss in that same range is only ~1-1.5% of the injected
energy — nowhere near enough to explain a 13% deficit by itself.

### An interesting side note: the codebase already half-knew this

`test_cp3_high_energy_cascade.py`'s tolerance comment says the leakage is from "the
recursive multi-generation solver drops residual energy below the E_min floor at each
recursion level." The previous analysis flagged this comment as a misattribution and
proposed pair production as "the actual mechanism" instead. Having traced it directly,
**the original comment was closer to the truth** — it just didn't distinguish that
this is true of `electron_cascade` specifically (which really does drop residual
floor energy, unconditionally) while `photon_cascade_full`'s own `f_subthreshold`
bookkeeping is fine (this part of the previous analysis was correct). The previous
analysis's replacement diagnosis (pair-production rest mass) is a real, separate,
worth-fixing bug — just not the dominant one, and not what that comment was
describing.

Separately, `cascade/ic_spectrum.py`'s own docstring already documents having
hit this same discrepancy independently — it notes that at `E0=1e9 eV, z=1e4`,
its own (non-recursive) decade sum disagrees sharply with
`solve_burst.burst_energy_budget`'s answer, and cross-checks against the latter as
the presumed "correctly-reprocessed" ground truth. That ground truth is itself
non-conservative (0.9413 at 1e9 eV per the table above) — so this cross-check
silently absorbs the bug rather than catching it, since it only compares its own
number against `burst_energy_budget`'s, and both are downstream of the same
`electron_cascade`.

### Suggested fix

In `electron_cascade`, before the final return:

```python
heat_energy += E_min_eV   # residual KE at the floor -- Coulomb-dominated by
                           # this function's own E0_eV <= E_min_eV convention
```

This is the minimal, convention-consistent fix (it matches what the function already
does for an electron injected *at* or below `E_min_eV`) and, combined with the
pair-production fix from the previous analysis, restores exact conservation across
every energy and species I tested. A more careful alternative would split the
residual using the last decade's local Coulomb/IC branching ratio rather than
crediting it 100% to heat, but that's a modeling refinement on top of fixing the
conservation bug itself, not a substitute for it.

---

## 2. CONFIRMED, with an added nuance — test suite import bug

Confirmed exactly as described: 8 of 9 checkpoint files use
`sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))` (two levels
up from `tests/`, landing one directory above `Cascades/`), while only
`test_cp7_interface.py` uses the correct one-level-up form. Running any of the 8
directly (`python3 test_cp0_cosmology.py`) fails with `ModuleNotFoundError: No module
named 'cascade'`.

**One thing worth adding**: this isn't just "run it the intended way and it's fine."
I tried the standard ways someone would actually run this suite:

| Invocation | Result |
|---|---|
| `python3 test_cpN....py` (any dir) | Fails (as previous analysis found) |
| `pytest tests/` from repo root | **Fails** — 7 collection errors |
| `pytest` from inside `tests/` | **Fails** — same errors |
| `python3 -m pytest` from inside `tests/` | **Fails** — same errors |
| `python3 -m pytest tests/` from repo root (`Cascades/Cascades/`) | **Passes**, all 45 tests |

The only invocation that works is `python -m pytest` run from exactly the repo-root
directory — and it only works by accident, because `python -m` prepends the current
working directory to `sys.path`, which happens to contain the real `cascade` package
and gets searched *after* the test file's own (wrong) `sys.path.insert(0, ...)`
entry fails to resolve. The bare `pytest` console command (no `-m`, what most people
actually type) fails regardless of directory, because it doesn't add the cwd this
way. So this is a somewhat narrower success case than "works if you run pytest from
the right directory" — it only works with one specific invocation form.

Practical fix: change the 8 broken files to match `test_cp7_interface.py`'s
`os.path.join(os.path.dirname(__file__), "..")`, or better, add a `conftest.py` at
the repo root (or a `pytest.ini` with `pythonpath = .`) so the test suite works under
any standard invocation.

## 3. CONFIRMED — inconsistent `high_energy` defaulting

Confirmed as described. `burst_energy_budget` and `y_distortion_estimate` hardcode
`high_energy=False`, while `burst_y_vs_redshift`, `convolve_generic_injection_y`,
`interface.make_dq_dz`, and `interface.make_photon_source` all use `high_energy=None`
with auto-select (`E0_eV > 2*M_E_EV`). I independently re-checked every call site in
the repo (`reproduce_ak19_plots.py`, `reproduce_fig3.py`, `dual_solver_harness.py`,
`ic_spectrum.py`, all test files) and confirm none currently trips this: every call
either passes `high_energy` explicitly or stays below the pair-production threshold.
Not live-broken today, but a real inconsistency and an easy trap for new call sites —
agreed with the previous analysis's assessment here.

## 4. NEW, minor — domain mismatch between the two Eq. A.6 transcriptions

The previous analysis notes `rates.ic_klein_nishina_double_diff` and
`ic_spectrum.klein_nishina_double_diff` "agree algebraically on the physics
bracket" — true for the bracket expression itself, but their validity domains
differ:

```python
# rates.py
valid = (q > 0.0) & (q < 1.0) & (E1 < ge * me)

# ic_spectrum.py
domain = (q >= 1.0 / (4.0 * ge ** 2)) & (q <= 1.0) & (denom > 0)
```

`ic_spectrum.py`'s version enforces the correct kinematic floor
`q >= 1/(4*gamma_e^2)` (standard Blumenthal & Gould); `rates.py`'s version only
requires `q > 0`, so it will return nonzero values in the kinematically-forbidden
region `0 < q < 1/(4*gamma_e^2)` where the true rate is zero. Low priority:
`rates.ic_klein_nishina_double_diff` is exercised only by its own regression test
(`test_cp1_rates.py`) and isn't called from anywhere in the actual cascade/figure
pipeline, so this doesn't affect any current result — but it means the two
"independent" transcriptions aren't quite as independently-agreeing as the previous
analysis states, and the unused one has a real, fixable inaccuracy for whenever it
does get used.

## 5. Everything else

Reran the full suite (`pytest tests/` from the repo root with correct pathing),
`reproduce_fig3.py`, and spot-checked `reproduce_ak19_plots.py`'s call sites — the
previous analysis's "everything else checks out" holds, including the
`cosmic_time` `IntegrationWarning` (still present, still non-fatal, still worth
tightening).

---

## Summary of changes to make

**Status as of this update: 1, 2, 3, 5, 6 done and verified (45/45 tests passing,
energy conservation to 1.0000 across 1e6–1e11 eV for both species). Item 4 was
comment-only and intentionally skipped. See `## Status log` below for details.**

1. **[DONE] — New, primary fix**: `electron_cascade` must book its residual
   `E_min_eV` of kinetic energy somewhere (e.g. as heat) instead of hardcoding
   `f_subthreshold: 0.0`. This is the dominant source of the ~13% energy
   non-conservation in the 30 MeV–1 GeV range, not pair production.
2. **[DONE]** Still fix the pair-production rest-mass leak in `photon_cascade_full` as
   the previous analysis describes (mirror `positron_burst_energy`'s re-injection of
   `2*m_e c^2` as two annihilation photons) — real, just smaller than previously
   estimated (~1-2% vs. the full ~13%).
3. **[DONE]** Fix the 8 test files' `sys.path.insert` (one level up, not two), and
   consider a `conftest.py`/`pytest.ini` so `pytest` works from any
   directory/invocation, not just `python -m pytest` from the exact repo root.
4. **[SKIPPED, by request]** Update `test_cp3_high_energy_cascade.py`'s comment to
   correctly attribute the leakage: it's `electron_cascade`'s dropped floor energy
   (dominant) plus `photon_cascade_full`'s untracked pair-production rest mass
   (minor) — not solely one or the other. Comment-only, no code/behavior impact;
   left as-is for now.
5. **[DONE]** Low priority: align `rates.ic_klein_nishina_double_diff`'s validity
   domain with `ic_spectrum.klein_nishina_double_diff`'s (`q >= 1/(4*gamma_e^2)`, not
   `q > 0`) before this unused function is ever wired into a live calculation.
6. **[DONE]** Unchanged from the previous analysis: fix the inconsistent
   `high_energy` default in `burst_energy_budget`/`y_distortion_estimate` for
   consistency with the rest of the module, even though no current call site hits it.

---

## Status log (fixes applied after this review)

1. **Item 1 — `electron_cascade` floor-energy fix.** Added
   `heat_energy += E_min_eV` immediately before the final return in
   `electron_cascade` (`solve_burst.py`), booking the residual kinetic energy at
   the `E_min_eV` floor as heat — consistent with the function's own convention
   for `E0_eV <= E_min_eV`. Verified: direct electron injections at 3e7/6e7/1e8 eV
   now conserve to exactly 1.0000 (previously 0.8985/0.8755/0.9003).
   `test_cp3_high_energy_cascade.py` passes.

2. **Item 2 — pair-production rest-mass re-injection.** In `photon_cascade_full`'s
   pair-production branch, the destroyed photon's `2*m_e*c^2` is now re-injected as
   two 511 keV annihilation photons and cascaded through `photon_cascade_full`
   (mirroring `positron_burst_energy`'s existing treatment of the same cost),
   instead of being silently dropped. Verified: combined with item 1, energy
   conservation holds to 1.0000 for every electron/photon energy tested from
   1e6–1e11 eV.

3. **Item 3 — test-suite import path.** Changed
   `sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))` to the
   one-level-up form (matching `test_cp7_interface.py`) in all 8 affected test
   files, and added a root-level `conftest.py` as a belt-and-braces fix. Verified
   all invocation forms the original review flagged as broken now work: direct
   script (`python3 test_cpN....py`), `pytest tests/` from repo root, `pytest`/
   `python3 -m pytest` from inside `tests/`, and bare `pytest` from repo root —
   all collect and run cleanly.

   **Collateral test fix (not in the original list):** fixing items 1-2 restored
   `f_total = f_heat + f_ic + f_subthreshold ≈ 1.0` for every species/energy, which
   broke `test_cp6_convolution.py::test_amplitude_sensitive_to_injected_species_energy`
   — that test's assertion (`y` differs measurably between 1 MeV and 100 MeV
   electrons) was unknowingly relying on the conservation bug as its source of
   energy-sensitivity. Renamed/rewrote it as
   `test_amplitude_insensitive_to_injected_species_energy_given_conservation`,
   asserting the (now correct) near-equality instead, with a comment explaining
   why and noting that real Fig. 14b energy/species sensitivity requires the
   DC/BR relaxation physics deferred to CP7. Full suite: **45/45 passing.**

4. **Skipped by explicit request** — comment-only, no functional effect.

5. **Item 5 — `ic_klein_nishina_double_diff` validity domain.** Changed
   `valid = (q > 0.0) & (q < 1.0) & (E1 < ge * me)` to
   `valid = (q >= 1.0 / (4.0 * ge ** 2)) & (q < 1.0) & (E1 < ge * me)` in
   `rates.py`, enforcing the standard Blumenthal & Gould kinematic floor and
   matching `ic_spectrum.klein_nishina_double_diff`'s domain. Verified: a `q`
   below the floor now correctly returns 0 (previously nonzero); `test_cp1_rates.py`
   and the full suite still pass (this function isn't exercised by the live
   cascade/figure pipeline, only its own regression test, so no downstream effect
   expected).

6. **Item 6 — `high_energy` default consistency.** Changed
   `burst_energy_budget(species, E0_eV, z, high_energy=False)` and
   `y_distortion_estimate(..., high_energy=False)` to `high_energy=None` with the
   same auto-select convention (`E0_eV > 2*M_E_EV`) used by `burst_y_vs_redshift`,
   `convolve_generic_injection_y`, `interface.make_dq_dz`, and
   `interface.make_photon_source`. Verified: full suite still 45/45 (no current
   call site relied on the old hardcoded default, as the original review noted).

**Net result: full test suite is 45/45 passing, and energy conservation
(`f_heat + f_ic + f_subthreshold = 1`) now holds to 1.0000 for both electron and
photon injections across 1e6–1e11 eV.**

