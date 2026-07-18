Run the following command in the terminal to do everything:
------------start-----------------------------------------------------

# ── 1. Get into the project and set up Python deps ─────────────────────────
cd Cascades/Cascades

pip install --break-system-packages numpy scipy matplotlib pytest

# ── 2. Run the full test suite ──────────────────────────────────────────────
# (conftest.py + the fixed sys.path.insert lines mean any of these work)
pytest -q                      # bare pytest, from repo root
python3 -m pytest tests/ -q    # equivalent, explicit tests/ dir

# Run an individual checkpoint file directly (also works standalone now):
python3 tests/test_cp0_cosmology.py
python3 tests/test_cp1_rates.py
python3 tests/test_cp2_low_energy_cascade.py
python3 tests/test_cp3_high_energy_cascade.py
python3 tests/test_cp4_positron_annihilation.py
python3 tests/test_cp5_redshift_marching.py
python3 tests/test_cp6_convolution.py
python3 tests/test_cp7_interface.py
python3 tests/test_cp8_mpr21.py
python3 tests/test_cp9_fig3.py
python3 tests/test_mckeen_bridge.py

# ── 3. Run the pure-cascade reproduction scripts (no spectroxide needed) ───
python3 reproduce_ak19_plots.py       # Figs. 1/2/13-style cascade-only plots
python3 reproduce_fig3.py             # Fig. 3 (electron-only, Eq. A.5/A.6)
python3 fig3_electron_ic_spectrum.py  # Actual Fig. 3

# ── 4. Require the spectroxide Rust binary ───────
python3 fig6_spectrum_via_spectroxide.py
python3 dual_solver_harness.py
python3 fig6_single_spectrum.py
python3 fig6_combine_points.py
python3 sweep_deltam_mckeen_corrected.py
# These will fail/degrade gracefully with a clear "spectroxide not available"
# error rather than crash silently — that's expected without the Rust build.

-----------------------------end-----------------------------------------------

NOTE: you have to build spectroxide separately and run the uncommented portions of the code for it to interact with spectroxide.


Test breakdown by file (Cascades/tests/) -- pytest-collected counts
=====================================================================
(corrected: accounts for parametrized tests, which the earlier
grep-based "def test_" count missed)
 
test_cp0_cosmology.py             7
test_cp1_rates.py                 9
test_cp2_low_energy_cascade.py    4
test_cp3_high_energy_cascade.py   4
test_cp4_positron_annihilation.py 4
test_cp5_redshift_marching.py     4
test_cp6_convolution.py           4
test_cp7_interface.py             4
test_cp8_mpr21.py                 5
test_cp9_fig3.py                  6
test_mckeen_bridge.py            48 
---------------------------------------------------------------------
TOTAL (all 11 files)              99
 

To increase the numerical resolution in all the non-spectroxide plots:

---------------------start------------------------------------------------------

## A. Fig. 3 (`reproduce_fig3.py` → `cascade/ic_spectrum.py`)

**`cascade/ic_spectrum.py`:**

1. **Line 129** — `mean_t_relativistic`'s internal quadrature grid:
```python
def mean_t_relativistic(p, n=4000):
```
→ `def mean_t_relativistic(p, n=30000):`

2. **Line 311** — seed-photon convolution grid (inside `electron_ic_spectrum`):
```python
x0_grid = np.geomspace(1e-3, 30.0, 300)  # seed-frequency integration grid for the convolution
```
→ `x0_grid = np.geomspace(1e-3, 30.0, 4000)  # seed-frequency integration grid for the convolution`

3. **Line 219** — cooling-track decade count, and **plumb it through** rather than just changing the default:
```python
def electron_ic_spectrum(E0_eV, z, x_grid, E_min_eV=100.0, min_decades_per_e_fold=4):
```
→ `def electron_ic_spectrum(E0_eV, z, x_grid, E_min_eV=100.0, min_decades_per_e_fold=60):`

4. **Line 382** — add the passthrough parameter to `electron_ic_delta_n`:
```python
def electron_ic_delta_n(species_E0_eV, z, injected_frac_rho_cmb, x_grid, E_min_eV=100.0):
```
→ `def electron_ic_delta_n(species_E0_eV, z, injected_frac_rho_cmb, x_grid, E_min_eV=100.0, min_decades_per_e_fold=60):`

5. **Line 392** — forward it in that same function's body:
```python
spec = electron_ic_spectrum(species_E0_eV, z, x_grid, E_min_eV=E_min_eV)
```
→ `spec = electron_ic_spectrum(species_E0_eV, z, x_grid, E_min_eV=E_min_eV, min_decades_per_e_fold=min_decades_per_e_fold)`

6. **Line 408** — add the same passthrough to `fig3_style_delta_I`:
```python
def fig3_style_delta_I(species_E0_eV, z, injected_frac_rho_cmb, x_grid, E_min_eV=100.0):
```
→ `def fig3_style_delta_I(species_E0_eV, z, injected_frac_rho_cmb, x_grid, E_min_eV=100.0, min_decades_per_e_fold=60):`

7. **Line 424** — forward it there too:
```python
delta_n, budget = electron_ic_delta_n(species_E0_eV, z, injected_frac_rho_cmb, x_grid, E_min_eV=E_min_eV)
```
→ `delta_n, budget = electron_ic_delta_n(species_E0_eV, z, injected_frac_rho_cmb, x_grid, E_min_eV=E_min_eV, min_decades_per_e_fold=min_decades_per_e_fold)`

**`reproduce_fig3.py`:**

8. **Line 54** — output/evaluation grid:
```python
X_MIN, X_MAX, N_X = 0.05, 20.0, 600
```
→ `X_MIN, X_MAX, N_X = 0.05, 20.0, 4000`

9. **Line 55** — cooling-track floor (lower it so the extra decades from #3 actually add resolution instead of just extending the tail):
```python
E_MIN_EV = 100.0  # cooling-track floor (well below all four injection energies)
```
→ `E_MIN_EV = 20.0  # cooling-track floor (well below all four injection energies)`

10. **Line 96** — pass the new kwarg through the actual call site:
```python
delta_I_SI, budget = ics.fig3_style_delta_I(E0, Z_INJ, INJECTED_FRAC, x_grid, E_min_eV=E_MIN_EV)
```
→ `delta_I_SI, budget = ics.fig3_style_delta_I(E0, Z_INJ, INJECTED_FRAC, x_grid, E_min_eV=E_MIN_EV, min_decades_per_e_fold=60)`

## B. Figs. 2 / 13 / 14a / 6-scan (`reproduce_ak19_plots.py` → `cascade/solve_burst.py`)

**`cascade/solve_burst.py`:**

11. **Line 236** — this is the same decade-staircasing issue as Fig. 3, but inside `electron_cascade`, and it's **hardcoded** (no kwarg to plumb — must edit the literal multiplier):
```python
n_decades = max(4, int(np.ceil(np.log10(E0_eV / E_min_eV) * 4)))
```
→ `n_decades = max(4, int(np.ceil(np.log10(E0_eV / E_min_eV) * 60)))`
(This affects every plot that touches `electron_cascade`/`burst_energy_budget` at `high_energy=True`: Fig. 13, the Fig. 6-style `plot_y_vs_energy` scan, and — once conservation matters at scale — `decay_convolution_y`.)

12. **Line 73** — the Coulomb/IC branching integral (feeds every decade split above and below the CP3 threshold):
```python
val, _ = integrate.quad(branch, E_min_eV, E0_eV, limit=200)
```
→ `val, _ = integrate.quad(branch, E_min_eV, E0_eV, limit=400, epsrel=1e-10)`

13. **Line 104** — `photon_cascade`'s scatter cap (CP2, low-energy path; already event-resolved rather than decade-binned, so this just guards against premature truncation, not smoothing):
```python
def photon_cascade(E0_eV, z, E_min_eV=1000.0, max_scatters=100000):
```
→ leave `max_scatters` as-is unless you see truncation warnings; it's already generous.

14. **Line 290** — `photon_cascade_full`'s scatter/recursion caps (CP3 path underlying Figs. 13/6-scan at high energy):
```python
def photon_cascade_full(E0_eV, z, E_min_eV=1000.0, max_scatters=2000, depth=0, max_depth=60):
```
→ `def photon_cascade_full(E0_eV, z, E_min_eV=1000.0, max_scatters=20000, depth=0, max_depth=120):`
(Raise both — `max_depth`/`max_scatters` become the effective ceiling once decade resolution goes up elsewhere; you want them well clear of anything that could actually get hit.)

15. **Line 208** — `electron_cascade`'s own `max_depth` default should be raised to match, for the same reason:
```python
def electron_cascade(E0_eV, z, E_min_eV=1000.0, depth=0, max_depth=60):
```
→ `def electron_cascade(E0_eV, z, E_min_eV=1000.0, depth=0, max_depth=120):`

**`reproduce_ak19_plots.py`:**

16. **Line 78** — Fig. 13 heat-fraction curve's energy-grid density:
```python
N_HEAT_FRACTION_POINTS = 100
```
→ `N_HEAT_FRACTION_POINTS = 1000`

17. **Line 79** — Fig. 6-style y-vs-energy scan's energy-grid density:
```python
N_Y_VS_ENERGY_POINTS = 60
```
→ `N_Y_VS_ENERGY_POINTS = 400`

18. **Line 236** — Fig. 2 cooling-ratio curve's energy grid:
```python
E_MeV = np.linspace(0.8, 6.0, 400)
```
→ `E_MeV = np.linspace(0.8, 6.0, 2000)`

19. **Line 282** — Fig. 14a injection-profile redshift grid:
```python
z_grid = np.logspace(np.log10(z_lo), np.log10(z_hi), 200)
```
→ `z_grid = np.logspace(np.log10(z_lo), np.log10(z_hi), 2000)`

## C. Global — feeds every plot indirectly (`cascade/cosmology.py`)

20. **Line 106** — `cosmic_time`'s integral, currently throwing `IntegrationWarning` (roundoff, not just resolution) — this underlies every redshift-dependent rate in the whole pipeline, so it's worth fixing once regardless of which plot you're chasing:
```python
val, _ = integrate.quad(integrand, z, 1.0e7, limit=200)
```
→ `val, _ = integrate.quad(integrand, z, 1.0e7, limit=500, epsrel=1e-10)`

## D. Any figure that uses `decay_convolution_y` (CP6/CP8 machinery, e.g. a future Fig. 14b-style plot)

21. **`cascade/solve_burst.py`, line 656** — injection-redshift grid density:
```python
def build_zinj_grid(tau_X_s, n_z=40, t_min_factor=1.0e-2, t_max_factor=30.0):
```
→ `def build_zinj_grid(tau_X_s, n_z=200, t_min_factor=1.0e-2, t_max_factor=30.0):`

22. **Line 707** — `decay_convolution_y`'s own default, same idea:
```python
def decay_convolution_y(species, E0_eV, z_X, tau_X_s, target_y_all_heat=1.5e-5,
                         n_z=40, high_energy=None):
```
→ `n_z=200` in place of `n_z=40`.

---


---------------------------------end------------------------------------------------------------------
