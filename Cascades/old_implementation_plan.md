# Fix Fig 6: Wire Cascade Through Spectroxide Instead of Scalar y-Approximation

## Problem Summary

Fig 6 (`fig6_y_vs_energy.png`) currently shows **two completely flat horizontal lines** at $y = 2.5 \times 10^{-6}$ for both electron and photon injection, instead of the energy-dependent curves AK19 Fig 6d/7a show. The root cause is documented in the codebase itself:

1. **100% heat efficiency / scalar-only model**: `y_distortion_estimate()` in [solve_burst.py](file:///home/batman/Documents/Work/CMBSD/Cascades/cascade/solve_burst.py#L517-L553) computes `f_distortion = f_heat + f_ic + f_subthreshold`. Since energy conservation guarantees this sum equals 1, the result is always `y = injected_frac / 4` — a flat constant regardless of injection energy or species.

2. **No spectroxide integration**: The [interface.py](file:///home/batman/Documents/Work/CMBSD/Cascades/cascade/interface.py) module was written to spectroxide's API but never actually run end-to-end because spectroxide was not available when the code was developed. However, **spectroxide is now installed** on the system at `/home/batman/Documents/Work/CMBSD/spectroxide`, accessible via `/home/batman/Documents/Work/.venv/bin/python`.

The fix is to **route the cascade output through spectroxide's PDE solver** for fig6, letting spectroxide's Kompaneets + DC/BR physics produce the actual energy-dependent $y$ values, instead of the `injected_frac/4` scalar approximation.

## Spectroxide API (Confirmed Working)

```python
from spectroxide import solve

# Custom heating + photon source:
result = solve(
    method="pde",
    dq_dz=my_dq_dz_func,           # z -> d(Δρ/ρ)/dz (scalar)
    photon_source=my_photon_src,    # (x, z) -> d(Δn)/dz
    z_start=3e5,
    z_end=1e3,
)
# result.mu, result.y, result.x, result.delta_n available

# Or with built-in Green's function (fast, no Rust needed):
from spectroxide import mu_from_heating, y_from_heating
y = y_from_heating(dq_dz, z_min=1e3, z_max=3e6)
```

Python with spectroxide: `/home/batman/Documents/Work/.venv/bin/python`

## Proposed Changes

> [!IMPORTANT]
> The user said **"don't change anything in the files for now"**. All changes go into a **new standalone script** that imports from the existing (unmodified) cascade modules and wires them through spectroxide.

### New Script

#### [NEW] `fig6_via_spectroxide.py`

A self-contained script that:

1. **Imports the existing cascade code** (`cascade.solve_burst`, `cascade.cosmology`, `cascade.rates`) unchanged
2. **Constructs `dq_dz(z)` and `photon_source(x, z)`** using the existing `interface.py` functions (`make_dq_dz`, `make_photon_source`) which are already written to spectroxide's conventions
3. **Runs `spectroxide.solve(method="pde", ...)`** for each energy/species combination in the Fig 6 grid
4. **Extracts `result.y`** from spectroxide's output (which properly accounts for spectral redistribution via Kompaneets + DC/BR)
5. **Also tries the Green's function fast path** (`spectroxide.y_from_heating`) as a cross-check
6. **Plots the results** as a proper energy-dependent y-vs-energy curve, alongside AK19's reference

The script will sweep over the same energy range as the current fig6 (10 keV – 1 MeV, both electron and photon injection at z=20000) but using spectroxide's PDE for the low-energy Kompaneets relaxation instead of the `Δρ/ρ = 4y` scalar approximation.

**Key technical details:**

- For each energy point E0 and species, `burst_energy_budget()` gives the cascade's `f_heat` (→ `dq_dz`) and `f_ic + f_subthreshold` (→ `photon_source`)
- The existing `make_dq_dz()` already constructs a proper `dq_dz(z)` in spectroxide's convention
- The existing `make_photon_source()` already constructs `photon_source(x, z)` using the Gaussian-line proxy
- Both functions are called with the spectroxide PDE solver to get proper $\mu$, $y$ decomposition
- The Green's function mode (`y_from_heating(dq_dz, ...)`) provides a fast cross-check that doesn't need the Rust binary

**Runtime expectation:** Each spectroxide PDE call takes ~seconds. With ~60 energy points × 2 species = ~120 PDE calls, expect ~5–15 minutes total. The user explicitly said "larger solving time... is fine."

**Two approaches will be tried and compared:**

| Approach | What it uses | Speed | Accuracy |
|----------|-------------|-------|----------|
| **PDE full** | `spectroxide.solve(method="pde", dq_dz=..., photon_source=...)` | ~seconds/point | Full Kompaneets + DC/BR |
| **Green's function** | `spectroxide.y_from_heating(dq_dz, ...)` heat-only (no photon_source) | ~instant | Approximate, heat-channel only |

## Open Questions

> [!IMPORTANT]
> **Cosmology consistency**: The cascade code uses h=0.71, Ω_b=0.044, Ω_m=0.26, Y_p=0.24, T_CMB=2.726 K (AK19/CosmoTherm defaults). Spectroxide's Rust defaults may differ (likely Planck 2015 or 2018). We should pass the cascade's cosmology to spectroxide via the `cosmo` parameter to ensure consistency. Will verify spectroxide's default cosmology parameters first.

> [!NOTE]
> **Photon source proxy quality**: `interface.py`'s `photon_source` is a narrow Gaussian line at a single characteristic frequency — this is a proxy for the actual spectral shape. For fig6's purposes (just extracting $y$), this should be adequate since the total energy is correct. For spectral shape comparisons, resolving the actual photon spectrum would be the next step.

## Verification Plan

### Automated Tests
- The existing test suite is untouched (no source files modified)
- The new script self-validates against known limits:
  - In the all-heat limit: spectroxide's $y$ should match the current `injected_frac/4` answer
  - Energy dependence: photon injection at low energy should give different $y$ than at high energy
  - Species dependence: electron vs photon should differ

### Manual Verification
- Compare the new fig6 against AK19 Fig 6d/7a in `acharya2019-2.pdf`
- Verify the Green's function and PDE answers are broadly consistent
