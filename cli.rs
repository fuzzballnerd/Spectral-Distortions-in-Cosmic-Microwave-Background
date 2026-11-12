//! CLI argument parsing with subcommands (zero dependencies).
//!
//! Supports both new-style subcommands and legacy flat-flag mode:
//!
//! ```text
//! spectroxide solve single-burst --z-h 2e5 --delta-rho 1e-5
//! spectroxide sweep --z-injections 1e3,1e4,1e5 --format csv
//! spectroxide greens --z-h 2e5 --delta-rho 1e-5
//! spectroxide info --cosmology planck2018
//! spectroxide help
//! ```
//!
//! If the first argument is not a known subcommand, falls back to legacy
//! flat-flag parsing with a deprecation warning.

use std::collections::HashMap;

use crate::cascade_photon_source::{
    build_decaying_particle_photon_table, DecayingParticlePhotonParams, TableGridSpec,
};
use crate::constants::{EV_IN_JOULES, K_BOLTZMANN};
use crate::energy_injection::InjectionScenario;
use crate::output::{
    GreensResult, OutputFormat, PhotonSweepBatchResult, PhotonSweepResult, PhotonSweepRow,
    SweepResult, SweepRow, CascadeTableResult,
};
use crate::recombination::RecombinationHistory;
use crate::prelude::*;

/// Top-level command parsed from CLI args.
#[derive(Debug)]
pub enum Command {
    /// Run a single PDE solve with a specific injection scenario.
    Solve(SolveOpts),
    /// Run a sweep over multiple injection redshifts.
    Sweep(SweepOpts),
    /// Run a Green's function calculation (no PDE).
    Greens(GreensOpts),
    /// Print cosmology/solver info.
    Info(InfoOpts),
    /// Print the compile-time physics-source hash and exit.
    PhysicsHash,
    /// Sweep over injection redshifts for a single photon injection frequency.
    PhotonSweep(PhotonSweepOpts),
    /// Batch sweep over multiple x_inj values, parallelizing all (x_inj, z_h) pairs.
    PhotonSweepBatch(PhotonSweepBatchOpts),
    /// Build a cascade-processed photon-source table for a single-photon
    /// decaying-particle channel (writes CSV consumable by `solve tabulated-photon`).
    CascadeTable(CascadeTableOpts),
    /// Print help text.
    Help,
}

/// Options for `spectroxide solve <injection-type> [flags]`.
#[derive(Debug)]
pub struct SolveOpts {
    /// Injection-scenario tag (positional, e.g. `single-burst`,
    /// `decaying-particle`, `photon`). Drives which `--params` keys are
    /// expected.
    pub injection_type: String,
    /// Free-form `key=value` parameters specific to the chosen injection
    /// (parsed downstream into the corresponding `InjectionScenario` variant).
    pub params: HashMap<String, String>,
    /// Solver tuning knobs (timestepping, grid, DC/BR diagnostics).
    pub solver: SolverOpts,
    /// Cosmology overrides (preset and/or individual parameters).
    pub cosmo: CosmoOpts,
    /// Output format and destination.
    pub output: OutputOpts,
}

/// Options for `spectroxide sweep [flags]`.
#[derive(Debug)]
pub struct SweepOpts {
    /// Comma-separated injection redshifts from `--z-injections`.
    /// `None` falls back to the built-in default grid.
    pub z_injections: Option<Vec<f64>>,
    /// Energy injection amplitude Δρ/ρ (`--delta-rho`).
    pub delta_rho: f64,
    /// Per-scenario `key=value` parameters (see [`SolveOpts::params`]).
    pub params: HashMap<String, String>,
    /// Solver tuning knobs.
    pub solver: SolverOpts,
    /// Cosmology overrides.
    pub cosmo: CosmoOpts,
    /// Output format and destination.
    pub output: OutputOpts,
}

/// Options for `spectroxide greens [flags]`.
#[derive(Debug)]
pub struct GreensOpts {
    /// Injection redshift `--z-h` (heating epoch).
    pub z_h: f64,
    /// Energy injection amplitude Δρ/ρ (`--delta-rho`).
    pub delta_rho: f64,
    /// Output format and destination.
    pub output: OutputOpts,
}

/// Options for `spectroxide photon-sweep [flags]`.
#[derive(Debug)]
pub struct PhotonSweepOpts {
    /// Injection frequency `--x-inj` in dimensionless units `x = hν/(kT_z)`.
    pub x_inj: f64,
    /// Photon-number injection amplitude `--delta-n-over-n`.
    pub delta_n_over_n: f64,
    /// Optional Gaussian width `--sigma-x` of the photon line in `x`.
    /// `None` requests a delta-function injection.
    pub sigma_x: Option<f64>,
    /// Injection redshifts `--z-injections`. `None` uses the default sweep grid.
    pub z_injections: Option<Vec<f64>>,
    /// Solver tuning knobs.
    pub solver: SolverOpts,
    /// Cosmology overrides.
    pub cosmo: CosmoOpts,
    /// Output format and destination.
    pub output: OutputOpts,
}

/// Options for `spectroxide photon-sweep-batch [flags]`.
#[derive(Debug)]
pub struct PhotonSweepBatchOpts {
    /// Injection frequencies `--x-inj-values` (one sweep per value, run in
    /// parallel across all `(x_inj, z_h)` pairs).
    pub x_inj_values: Vec<f64>,
    /// Photon-number injection amplitude `--delta-n-over-n`, shared across
    /// all `x_inj` values.
    pub delta_n_over_n: f64,
    /// Optional Gaussian width `--sigma-x` of the injected line.
    pub sigma_x: Option<f64>,
    /// Injection redshifts `--z-injections`. `None` uses the default grid.
    pub z_injections: Option<Vec<f64>>,
    /// Solver tuning knobs.
    pub solver: SolverOpts,
    /// Cosmology overrides.
    pub cosmo: CosmoOpts,
    /// Output format and destination.
    pub output: OutputOpts,
}

/// Options for `spectroxide cascade-table [flags]`.
///
/// Builds a cascade-processed photon-source table for a single-photon
/// decaying-particle channel (e.g. chi -> n + gamma), suitable for
/// `solve tabulated-photon --photon-table <path>` once written to CSV.
#[derive(Debug)]
pub struct CascadeTableOpts {
    /// Rest-frame injected photon energy `--omega-inj-0-ev`, in eV
    /// (e.g. for chi -> n + gamma: `(m_chi/2)(1 - m_n^2/m_chi^2)`).
    pub omega_inj_0_ev: f64,
    /// Vacuum decay rate `--gamma-x`, in s^-1.
    pub gamma_x: f64,
    /// `n_X,0 / n_H,0`, the initial parent-to-hydrogen number density ratio
    /// (`--n-x0-over-nh0`).
    pub n_x0_over_n_h0: f64,
    /// Redshift grid lower bound `--z-min` (default 1e2).
    pub z_min: f64,
    /// Redshift grid upper bound `--z-max` (default 1e7).
    pub z_max: f64,
    /// Number of redshift grid points `--n-z` (default 60).
    pub n_z: usize,
    /// Frequency grid lower bound `--x-min` (default 1.5e-4).
    pub x_min: f64,
    /// Frequency grid upper bound `--x-max` (default 80.0). Should match
    /// the `--x-max` used in the downstream `solve` call.
    pub x_max: f64,
    /// Number of frequency grid points `--n-x` (default 300).
    pub n_x: usize,
    /// Cascade energy-grid resolution per redshift `--n-cascade-points`
    /// (default 200).
    pub n_cascade_points: usize,
    /// Cosmology overrides.
    pub cosmo: CosmoOpts,
    /// Output format and destination.
    pub output: OutputOpts,
}

/// Options for `spectroxide info`.
#[derive(Debug)]
pub struct InfoOpts {
    /// Named cosmology preset to display (`default`, `planck2015`,
    /// `planck2018`).
    pub cosmology: String,
}

/// Solver tuning parameters.
#[derive(Debug)]
pub struct SolverOpts {
    /// Starting redshift `--z-start`. `None` lets each subcommand pick a default.
    pub z_start: Option<f64>,
    /// Ending redshift `--z-end` (default 500.0).
    pub z_end: f64,
    /// Cap on adaptive y-step `--dy-max`.
    pub dy_max: Option<f64>,
    /// Cap on adaptive optical-depth step `--dtau-max`.
    pub dtau_max: Option<f64>,
    /// Frequency-grid point count `--n-points`. Overrides the built-in
    /// `fast` / `production` presets.
    pub n_points: Option<usize>,
    /// Disable double-Compton + bremsstrahlung emission/absorption (`--no-dcbr`).
    pub disable_dcbr: bool,
    /// Split the DC/BR step into separate DC and BR substeps (`--split-dcbr`).
    pub split_dcbr: bool,
    /// Use Crank-Nicolson (instead of backward Euler) for the DC/BR solve
    /// (`--cn-dcbr`). Diagnostic only — known to fail at low x.
    pub cn_dcbr: bool,
    /// Keep the photon-number-conserving Kompaneets correction enabled
    /// (default true; disabled with `--no-number-conserving`).
    pub number_conserving: bool,
    /// Apply the number-conserving correction every `nc_stride` steps
    /// (`--nc-stride`).
    pub nc_stride: Option<usize>,
    /// Lower-z cutoff `--nc-z-min` below which the number-conserving
    /// correction is suppressed.
    pub nc_z_min: Option<f64>,
    /// Use the high-resolution `production` grid preset (`--production-grid`).
    pub production_grid: bool,
    /// Override the initial-condition Δn at z_start (`--dn-planck`); used
    /// for adiabatic / baseline diagnostics.
    pub dn_planck: Option<f64>,
    /// Disable automatic grid refinement for photon-injection scenarios
    /// (`--no-auto-refine`).
    pub no_auto_refine: bool,
    /// Thread count for parallel sweep execution (`--threads`).
    pub n_threads: Option<usize>,
    /// Maximum dtau per step during photon-source injection
    /// (`--dtau-max-photon-source`); tightens the timestep near a delta-line
    /// source.
    pub dtau_max_photon_source: Option<f64>,
}

/// Cosmology parameters.
#[derive(Debug)]
pub struct CosmoOpts {
    /// Fractional baryon density `Ω_b` (`--omega-b`). Converted internally
    /// to the physical density `ω_b = Ω_b h²` before the `Cosmology`
    /// struct is built.
    pub omega_b: Option<f64>,
    /// Fractional total matter density `Ω_m = Ω_b + Ω_cdm` (`--omega-m`).
    /// Combined with `omega_b` and `h` to compute the physical CDM density
    /// `ω_cdm = (Ω_m − Ω_b) h²`.
    pub omega_m: Option<f64>,
    /// Reduced Hubble parameter `h = H₀ / (100 km/s/Mpc)` (`--h`).
    pub h: Option<f64>,
    /// Effective number of relativistic species `N_eff` (`--n-eff`).
    pub n_eff: Option<f64>,
    /// Helium mass fraction `Y_p` (`--y-p`).
    pub y_p: Option<f64>,
    /// CMB temperature today, in K (`--t-cmb`).
    pub t_cmb: Option<f64>,
    /// Named preset (`--cosmology`) such as `default`, `planck2015`,
    /// `planck2018`. Individual flags above override preset values.
    pub preset: Option<String>,
}

/// Output formatting options.
#[derive(Debug)]
pub struct OutputOpts {
    /// Output format selected by `--format` (json, csv, table).
    pub format: OutputFormat,
    /// Optional file path for output (`--output`). If `None`, output goes
    /// to stdout.
    pub output_path: Option<String>,
}

impl Default for SolverOpts {
    fn default() -> Self {
        SolverOpts {
            z_start: None,
            z_end: 500.0,
            dy_max: None,
            dtau_max: None,
            n_points: None,
            disable_dcbr: false,
            split_dcbr: false,
            cn_dcbr: false,
            number_conserving: true,
            nc_stride: None,
            nc_z_min: None,
            production_grid: false,
            dn_planck: None,
            no_auto_refine: false,
            n_threads: None,
            dtau_max_photon_source: None,
        }
    }
}

impl Default for CosmoOpts {
    fn default() -> Self {
        CosmoOpts {
            omega_b: None,
            omega_m: None,
            h: None,
            n_eff: None,
            y_p: None,
            t_cmb: None,
            preset: None,
        }
    }
}

impl Default for OutputOpts {
    fn default() -> Self {
        OutputOpts {
            format: OutputFormat::Json,
            output_path: None,
        }
    }
}

/// Known subcommands.
const SUBCOMMANDS: &[&str] = &[
    "solve",
    "sweep",
    "photon-sweep",
    "photon-sweep-batch",
    "cascade-table",
    "greens",
    "info",
    "physics-hash",
    "help",
];

/// Parse CLI arguments into a Command.
pub fn parse_command(args: &[String]) -> Result<Command, String> {
    if args.is_empty() {
        return Ok(Command::Help);
    }

    let first = args[0].as_str();

    // Check if first arg is a known subcommand
    if !SUBCOMMANDS.contains(&first) {
        return Err(format!(
            "Unknown subcommand: '{first}'. Valid: solve, sweep, greens, info, help"
        ));
    }

    match first {
        "help" => Ok(Command::Help),
        "physics-hash" => Ok(Command::PhysicsHash),
        "info" => {
            let map = parse_flat_args(&args[1..]);
            Ok(Command::Info(InfoOpts {
                cosmology: map
                    .get("--cosmology")
                    .cloned()
                    .unwrap_or_else(|| "default".to_string()),
            }))
        }
        "greens" => {
            let map = parse_flat_args(&args[1..]);
            let z_h: f64 = map
                .get("--z-h")
                .ok_or("--z-h required for greens subcommand")?
                .parse()
                .map_err(|_| "Invalid --z-h")?;
            let delta_rho = parse_f64_or(&map, "--delta-rho", 1e-5)?;
            let output = parse_output_opts(&map)?;
            Ok(Command::Greens(GreensOpts {
                z_h,
                delta_rho,
                output,
            }))
        }
        "sweep" => {
            let map = parse_flat_args(&args[1..]);
            let z_injections = map
                .get("--z-injections")
                .map(|s| parse_float_list(s))
                .transpose()?;
            let delta_rho = parse_f64_or(&map, "--delta-rho", 1e-5)?;
            let solver = parse_solver_opts(&map)?;
            let cosmo = parse_cosmo_opts(&map)?;
            let output = parse_output_opts(&map)?;
            Ok(Command::Sweep(SweepOpts {
                z_injections,
                delta_rho,
                params: map,
                solver,
                cosmo,
                output,
            }))
        }
        "photon-sweep" => {
            let map = parse_flat_args(&args[1..]);
            let x_inj: f64 = map
                .get("--x-inj")
                .ok_or("--x-inj required for photon-sweep subcommand")?
                .parse()
                .map_err(|_| "Invalid --x-inj")?;
            let delta_n_over_n = parse_f64_or(&map, "--delta-n-over-n", 1e-5)?;
            let sigma_x: Option<f64> = map
                .get("--sigma-x")
                .map(|s| s.parse().map_err(|_| "Invalid --sigma-x"))
                .transpose()?;
            let z_injections = map
                .get("--z-injections")
                .map(|s| parse_float_list(s))
                .transpose()?;
            let solver = parse_solver_opts(&map)?;
            let cosmo = parse_cosmo_opts(&map)?;
            let output = parse_output_opts(&map)?;
            Ok(Command::PhotonSweep(PhotonSweepOpts {
                x_inj,
                delta_n_over_n,
                sigma_x,
                z_injections,
                solver,
                cosmo,
                output,
            }))
        }
        "photon-sweep-batch" => {
            let map = parse_flat_args(&args[1..]);
            let x_inj_str = map
                .get("--x-inj-values")
                .ok_or("--x-inj-values required for photon-sweep-batch subcommand")?;
            let x_inj_values = parse_float_list(x_inj_str)?;
            if x_inj_values.is_empty() {
                return Err("--x-inj-values must contain at least one value".to_string());
            }
            let delta_n_over_n = parse_f64_or(&map, "--delta-n-over-n", 1e-5)?;
            let sigma_x: Option<f64> = map
                .get("--sigma-x")
                .map(|s| s.parse().map_err(|_| "Invalid --sigma-x"))
                .transpose()?;
            let z_injections = map
                .get("--z-injections")
                .map(|s| parse_float_list(s))
                .transpose()?;
            let solver = parse_solver_opts(&map)?;
            let cosmo = parse_cosmo_opts(&map)?;
            let output = parse_output_opts(&map)?;
            Ok(Command::PhotonSweepBatch(PhotonSweepBatchOpts {
                x_inj_values,
                delta_n_over_n,
                sigma_x,
                z_injections,
                solver,
                cosmo,
                output,
            }))
        }
        "cascade-table" => {
            let map = parse_flat_args(&args[1..]);
            let omega_inj_0_ev: f64 = map
                .get("--omega-inj-0-ev")
                .ok_or("--omega-inj-0-ev required for cascade-table subcommand")?
                .parse()
                .map_err(|_| "Invalid --omega-inj-0-ev")?;
            let gamma_x: f64 = map
                .get("--gamma-x")
                .ok_or("--gamma-x required for cascade-table subcommand")?
                .parse()
                .map_err(|_| "Invalid --gamma-x")?;
            let n_x0_over_n_h0: f64 = map
                .get("--n-x0-over-nh0")
                .ok_or("--n-x0-over-nh0 required for cascade-table subcommand")?
                .parse()
                .map_err(|_| "Invalid --n-x0-over-nh0")?;
            let z_min = parse_f64_or(&map, "--z-min", 1.0e2)?;
            let z_max = parse_f64_or(&map, "--z-max", 1.0e7)?;
            let n_z: usize = map
                .get("--n-z")
                .map(|s| s.parse().map_err(|_| "Invalid --n-z"))
                .transpose()?
                .unwrap_or(60);
            let x_min = parse_f64_or(&map, "--x-min", 1.5e-4)?;
            let x_max = parse_f64_or(&map, "--x-max", 80.0)?;
            let n_x: usize = map
                .get("--n-x")
                .map(|s| s.parse().map_err(|_| "Invalid --n-x"))
                .transpose()?
                .unwrap_or(300);
            let n_cascade_points: usize = map
                .get("--n-cascade-points")
                .map(|s| s.parse().map_err(|_| "Invalid --n-cascade-points"))
                .transpose()?
                .unwrap_or(200);
            let cosmo = parse_cosmo_opts(&map)?;
            let output = parse_output_opts(&map)?;
            Ok(Command::CascadeTable(CascadeTableOpts {
                omega_inj_0_ev,
                gamma_x,
                n_x0_over_n_h0,
                z_min,
                z_max,
                n_z,
                x_min,
                x_max,
                n_x,
                n_cascade_points,
                cosmo,
                output,
            }))
        }
        "solve" => {
            // Next arg should be the injection type
            let injection_type = args
                .get(1)
                .ok_or("solve requires an injection type (e.g., single-burst, decaying-particle)")?
                .clone();
            if injection_type.starts_with("--") {
                return Err(format!(
                    "solve requires an injection type as second argument, got '{injection_type}'"
                ));
            }
            let map = parse_flat_args(&args[2..]);
            let solver = parse_solver_opts(&map)?;
            let cosmo = parse_cosmo_opts(&map)?;
            let output = parse_output_opts(&map)?;
            Ok(Command::Solve(SolveOpts {
                injection_type,
                params: map,
                solver,
                cosmo,
                output,
            }))
        }
        _ => unreachable!(),
    }
}

/// Parse flat --key value args into a HashMap (shared by all subcommands and legacy mode).
pub fn parse_flat_args(args: &[String]) -> HashMap<String, String> {
    let mut map = HashMap::new();
    let mut i = 0;
    while i < args.len() {
        if args[i].starts_with("--") {
            let key = args[i].clone();
            if i + 1 < args.len() && !args[i + 1].starts_with("--") {
                map.insert(key, args[i + 1].clone());
                i += 2;
            } else {
                map.insert(key, String::new());
                i += 1;
            }
        } else {
            i += 1;
        }
    }
    map
}

fn parse_f64_or(map: &HashMap<String, String>, key: &str, default: f64) -> Result<f64, String> {
    match map.get(key) {
        Some(s) => s.parse().map_err(|_| format!("Invalid {key}")),
        None => Ok(default),
    }
}

fn parse_float_list(s: &str) -> Result<Vec<f64>, String> {
    s.split(',')
        .filter(|s| !s.is_empty())
        .map(|s| {
            s.trim()
                .parse::<f64>()
                .map_err(|_| format!("Invalid number: '{s}'"))
        })
        .collect()
}

fn parse_solver_opts(map: &HashMap<String, String>) -> Result<SolverOpts, String> {
    Ok(SolverOpts {
        z_start: map
            .get("--z-start")
            .map(|s| s.parse().map_err(|_| "Invalid --z-start"))
            .transpose()?,
        z_end: parse_f64_or(map, "--z-end", 500.0)?,
        dy_max: map
            .get("--dy-max")
            .map(|s| s.parse().map_err(|_| "Invalid --dy-max"))
            .transpose()?,
        dtau_max: map
            .get("--dtau-max")
            .map(|s| s.parse().map_err(|_| "Invalid --dtau-max"))
            .transpose()?,
        n_points: map
            .get("--n-points")
            .map(|s| s.parse().map_err(|_| "Invalid --n-points"))
            .transpose()?,
        disable_dcbr: map.contains_key("--no-dcbr"),
        split_dcbr: map.contains_key("--split-dcbr"),
        cn_dcbr: map.contains_key("--cn-dcbr"),
        number_conserving: !map.contains_key("--no-number-conserving"),
        nc_stride: map
            .get("--nc-stride")
            .map(|s| s.parse().map_err(|_| "Invalid --nc-stride"))
            .transpose()?,
        nc_z_min: map
            .get("--nc-z-min")
            .map(|s| s.parse().map_err(|_| "Invalid --nc-z-min"))
            .transpose()?,
        production_grid: map.contains_key("--production-grid"),
        dn_planck: map
            .get("--dn-planck")
            .map(|s| s.parse().map_err(|_| "Invalid --dn-planck"))
            .transpose()?,
        no_auto_refine: map.contains_key("--no-auto-refine"),
        n_threads: map
            .get("--threads")
            .map(|s| s.parse().map_err(|_| "Invalid --threads"))
            .transpose()?,
        dtau_max_photon_source: map
            .get("--dtau-max-photon-source")
            .map(|s| s.parse().map_err(|_| "Invalid --dtau-max-photon-source"))
            .transpose()?,
    })
}

/// Parse an optional float CLI argument, returning a clear error on invalid values.
fn parse_optional_f64(map: &HashMap<String, String>, key: &str) -> Result<Option<f64>, String> {
    match map.get(key) {
        Some(s) => s
            .parse::<f64>()
            .map(Some)
            .map_err(|_| format!("Invalid value for {key}: '{s}' (expected a number)")),
        None => Ok(None),
    }
}

fn parse_cosmo_opts(map: &HashMap<String, String>) -> Result<CosmoOpts, String> {
    if map.contains_key("--omega-cdm") {
        return Err(
            "--omega-cdm is no longer accepted. Use --omega-m (fractional total \
             matter Ω_m); the CLI converts to ω_cdm = (Ω_m − Ω_b) h² internally."
                .to_string(),
        );
    }
    Ok(CosmoOpts {
        omega_b: parse_optional_f64(map, "--omega-b")?,
        omega_m: parse_optional_f64(map, "--omega-m")?,
        h: parse_optional_f64(map, "--h")?,
        n_eff: parse_optional_f64(map, "--n-eff")?,
        y_p: parse_optional_f64(map, "--y-p")?,
        t_cmb: parse_optional_f64(map, "--t-cmb")?,
        preset: map.get("--cosmology").cloned(),
    })
}

fn parse_output_opts(map: &HashMap<String, String>) -> Result<OutputOpts, String> {
    let format = match map.get("--format") {
        Some(s) => OutputFormat::from_str(s)?,
        None => OutputFormat::Json,
    };
    let output_path = map.get("--output").cloned();
    Ok(OutputOpts {
        format,
        output_path,
    })
}

/// Build a Cosmology from CosmoOpts.
pub fn build_cosmology(opts: &CosmoOpts) -> Result<crate::cosmology::Cosmology, String> {
    // Check for presets first
    if let Some(ref preset) = opts.preset {
        match preset.as_str() {
            "planck2015" => return Ok(crate::cosmology::Cosmology::planck2015()),
            "planck2018" => return Ok(crate::cosmology::Cosmology::planck2018()),
            "default" => return Ok(crate::cosmology::Cosmology::default()),
            _ => {
                return Err(format!(
                    "Unknown cosmology preset '{preset}'. Valid presets: default, planck2015, planck2018"
                ));
            }
        }
    }

    // CLI accepts fractional Ω_b and Ω_m; the internal Cosmology struct
    // wants physical ω_b = Ω_b h² and ω_cdm = (Ω_m − Ω_b) h². Convert here
    // so the same fractional convention is shared by the Python wrapper,
    // the Cosmology dataclass, and the CLI.
    match (opts.omega_b, opts.omega_m) {
        (Some(_), None) | (None, Some(_)) => {
            return Err(
                "--omega-b and --omega-m must be provided together (Ω_cdm is \
                 computed from the difference). Pass both, or neither (to keep \
                 the preset matter sector)."
                    .to_string(),
            );
        }
        _ => {}
    }
    let defaults = crate::cosmology::Cosmology::default();
    let h = opts.h.unwrap_or(defaults.h);
    let h2 = h * h;
    let (omega_b_phys, omega_cdm_phys) = match (opts.omega_b, opts.omega_m) {
        (Some(omega_b_frac), Some(omega_m_frac)) => {
            if !(0.0..=1.0).contains(&omega_b_frac) {
                return Err(format!(
                    "--omega-b is fractional Ω_b ∈ [0, 1]; got {omega_b_frac}. \
                     Pass the fraction (e.g. 0.044), not the physical density Ω_b h²."
                ));
            }
            if !(0.0..=1.0).contains(&omega_m_frac) {
                return Err(format!(
                    "--omega-m is fractional total matter Ω_m ∈ [0, 1]; got {omega_m_frac}."
                ));
            }
            if omega_m_frac < omega_b_frac {
                return Err(format!(
                    "Ω_m ({omega_m_frac}) must be ≥ Ω_b ({omega_b_frac})."
                ));
            }
            (omega_b_frac * h2, (omega_m_frac - omega_b_frac) * h2)
        }
        _ => (defaults.omega_b, defaults.omega_cdm),
    };
    crate::cosmology::Cosmology::new(
        opts.t_cmb.unwrap_or(defaults.t_cmb),
        omega_b_phys,
        omega_cdm_phys,
        h,
        opts.n_eff.unwrap_or(defaults.n_eff),
        opts.y_p.unwrap_or(defaults.y_p),
    )
}

/// Print help text to stderr.
pub fn print_help() {
    eprintln!("spectroxide: CMB spectral distortion solver");
    eprintln!();
    eprintln!("USAGE:");
    eprintln!("  spectroxide solve <injection-type> [options]");
    eprintln!("  spectroxide sweep [options]");
    eprintln!("  spectroxide photon-sweep --x-inj <x> [options]");
    eprintln!("  spectroxide photon-sweep-batch --x-inj-values <x1,x2,...> [options]");
    eprintln!("  spectroxide greens --z-h <z> [options]");
    eprintln!("  spectroxide info [--cosmology <preset>]");
    eprintln!("  spectroxide physics-hash");
    eprintln!("  spectroxide help");
    eprintln!();
    eprintln!("INJECTION TYPES:");
    eprintln!("  single-burst          --z-h, --delta-rho, [--sigma-z]");
    eprintln!("  decaying-particle     --f-x, --gamma-x");
    eprintln!("  annihilating-dm       --f-ann");
    eprintln!("  annihilating-dm-pwave --f-ann");
    eprintln!("  monochromatic-photon  --x-inj, --delta-n-over-n, --z-h");
    eprintln!("  decaying-particle-photon --x-inj-0, --f-inj, --gamma-x");
    eprintln!("  dark-photon-resonance --epsilon, --m-ev");
    eprintln!("  tabulated-heating     --heating-table PATH (CSV: z,dq_dz)");
    eprintln!("  tabulated-photon      --photon-table PATH (CSV: z,x1,...,xN)");
    eprintln!();
    eprintln!("SOLVER OPTIONS:");
    eprintln!("  --z-start <z>         Starting redshift");
    eprintln!("  --z-end <z>           Final redshift (default 500)");
    eprintln!("  --delta-rho <val>     Fractional energy injection (for solve)");
    eprintln!("  --dy-max <val>        Max Kompaneets step size");
    eprintln!(
        "  --dtau-max <val>      Max Compton optical depth per step (default 10; use 3 for <0.1% precision)"
    );
    eprintln!(
        "  --dtau-max-photon-source <val>  Max dtau per step near photon source (default 1.0)"
    );
    eprintln!("  --n-points <n>        Grid points");
    eprintln!("  --production-grid     Use 4000-point production grid");
    eprintln!("  --no-dcbr             Disable DC/BR");
    eprintln!("  --split-dcbr          Operator-split DC/BR");
    eprintln!("  --no-number-conserving  Disable NC T-shift subtraction (on by default)");
    eprintln!("  --nc-stride <n>       NC stripping stride (steps between strips)");
    eprintln!("  --nc-z-min <z>        NC stripping minimum redshift (default 5e4)");
    eprintln!("  --dn-planck <val>     Initial Planck perturbation amplitude");
    eprintln!("  --sigma-z <val>       Temporal width for burst injection");
    eprintln!("  --sigma-x <val>       Spectral width for photon injection");
    eprintln!("  --no-auto-refine      Disable automatic grid refinement");
    eprintln!("  --threads <n>         Number of threads for parallel sweeps");
    eprintln!("  --format json|csv|table  Output format (default json)");
    eprintln!("  --output <path>       Write output to file (default: stdout)");
    eprintln!();
    eprintln!("COSMOLOGY:");
    eprintln!("  --cosmology <preset>  default, planck2015, planck2018");
    eprintln!("  --omega-b <Ω_b>       Fractional baryon density (pass with --omega-m)");
    eprintln!("  --omega-m <Ω_m>       Fractional total matter density");
    eprintln!("  --h, --n-eff, --y-p, --t-cmb");
    eprintln!();
    eprintln!("EXAMPLES:");
    eprintln!("  spectroxide solve single-burst --z-h 2e5 --delta-rho 1e-5");
    eprintln!("  spectroxide solve decaying-particle --f-x 1e5 --gamma-x 2e5");
    eprintln!("  spectroxide solve dark-photon-resonance --epsilon 1e-9 --m-ev 1e-7");
    eprintln!("  spectroxide greens --z-h 2e5");
    eprintln!("  spectroxide info --cosmology planck2018");
}

/// Print cosmology info.
pub fn print_info(opts: &InfoOpts) -> Result<(), String> {
    let cosmo = build_cosmology(&CosmoOpts {
        preset: Some(opts.cosmology.clone()),
        ..CosmoOpts::default()
    })?;
    println!("Cosmology: {}", opts.cosmology);
    println!("  h       = {}", cosmo.h);
    println!("  Ω_b     = {:.6}", cosmo.omega_b_frac());
    println!(
        "  Ω_m     = {:.6}  (Ω_cdm = {:.6})",
        cosmo.omega_b_frac() + cosmo.omega_cdm_frac(),
        cosmo.omega_cdm_frac()
    );
    println!("  Y_p     = {:.4}", cosmo.y_p);
    println!("  T_CMB   = {:.4} K", cosmo.t_cmb);
    println!("  N_eff   = {:.3}", cosmo.n_eff);
    println!("  z_eq    = {:.1}", cosmo.z_eq());
    println!(
        "  H_0     = {:.2} km/s/Mpc",
        cosmo.hubble(0.0) * 3.086e22 / 1e3
    );
    Ok(())
}

/// Build an InjectionScenario from CLI arguments.
///
/// Returns `Err` for unknown injection types or missing/invalid parameters.
pub fn build_injection_scenario(
    injection_type: &str,
    args: &HashMap<String, String>,
    delta_rho: f64,
) -> Result<InjectionScenario, String> {
    let get_required = |key: &str| -> Result<f64, String> {
        args.get(key)
            .ok_or_else(|| format!("{key} required for {injection_type}"))?
            .parse()
            .map_err(|_| format!("Invalid {key}"))
    };
    let get_optional = |key: &str, default: f64| -> Result<f64, String> {
        match args.get(key) {
            Some(s) => s.parse().map_err(|_| format!("Invalid {key}")),
            None => Ok(default),
        }
    };

    match injection_type {
        "decaying-particle" => {
            let f_x = get_required("--f-x")?;
            let gamma_x = get_required("--gamma-x")?;
            Ok(InjectionScenario::DecayingParticle { f_x, gamma_x })
        }
        "annihilating-dm" => {
            let f_ann = get_required("--f-ann")?;
            Ok(InjectionScenario::AnnihilatingDM { f_ann })
        }
        "annihilating-dm-pwave" => {
            let f_ann = get_required("--f-ann")?;
            Ok(InjectionScenario::AnnihilatingDMPWave { f_ann })
        }
        "single-burst" => {
            let z_h = get_required("--z-h")?;
            let sigma_z = get_optional("--sigma-z", (z_h * 0.04_f64).max(100.0))?;
            Ok(InjectionScenario::SingleBurst {
                z_h,
                delta_rho_over_rho: delta_rho,
                sigma_z,
            })
        }
        "monochromatic-photon" => {
            let x_inj = get_required("--x-inj")?;
            let delta_n_over_n = get_required("--delta-n-over-n")?;
            let z_h = get_required("--z-h")?;
            let sigma_z = get_optional("--sigma-z", (z_h * 0.04_f64).max(100.0))?;
            let sigma_x = get_optional("--sigma-x", x_inj * 0.05)?;
            Ok(InjectionScenario::MonochromaticPhotonInjection {
                x_inj,
                delta_n_over_n,
                z_h,
                sigma_z,
                sigma_x,
            })
        }
        "decaying-particle-photon" => {
            let x_inj_0 = get_required("--x-inj-0")?;
            let f_inj = get_required("--f-inj")?;
            let gamma_x = get_required("--gamma-x")?;
            Ok(InjectionScenario::DecayingParticlePhoton {
                x_inj_0,
                f_inj,
                gamma_x,
            })
        }
        "dark-photon-resonance" => {
            let epsilon = get_required("--epsilon")?;
            let m_ev = get_required("--m-ev")?;
            Ok(InjectionScenario::DarkPhotonResonance { epsilon, m_ev })
        }
        "tabulated-heating" => {
            let path = args
                .get("--heating-table")
                .ok_or("--heating-table PATH required for tabulated-heating")?;
            crate::energy_injection::load_heating_table(path)
        }
        "tabulated-photon" => {
            let path = args
                .get("--photon-table")
                .ok_or("--photon-table PATH required for tabulated-photon")?;
            crate::energy_injection::load_photon_source_table(path)
        }
        _ => Err(format!(
            "Unknown injection type: '{injection_type}'. \
             Valid: single-burst, \
             decaying-particle, decaying-particle-photon, \
             annihilating-dm, annihilating-dm-pwave, monochromatic-photon, \
             tabulated-heating, tabulated-photon"
        )),
    }
}

/// `n` log-spaced points from `lo` to `hi` inclusive, ascending.
fn logspace(lo: f64, hi: f64, n: usize) -> Vec<f64> {
    if n <= 1 {
        return vec![lo];
    }
    let (log_lo, log_hi) = (lo.ln(), hi.ln());
    (0..n)
        .map(|i| {
            let t = i as f64 / (n - 1) as f64;
            (log_lo + t * (log_hi - log_lo)).exp()
        })
        .collect()
}

/// Execute a cascade-table build for a single-photon decaying-particle
/// channel, producing a `(z_table, x_grid, source_2d)` table suitable for
/// `solve tabulated-photon --photon-table <path>`.
pub fn execute_cascade_table(opts: &CascadeTableOpts) -> Result<CascadeTableResult, String> {
    if opts.z_min <= 0.0 || opts.z_max <= opts.z_min {
        return Err(format!(
            "invalid redshift range: z_min={}, z_max={}",
            opts.z_min, opts.z_max
        ));
    }
    if opts.n_z < 2 {
        return Err("--n-z must be at least 2".to_string());
    }
    if opts.x_min <= 0.0 || opts.x_max <= opts.x_min {
        return Err(format!(
            "invalid frequency range: x_min={}, x_max={}",
            opts.x_min, opts.x_max
        ));
    }
    if opts.n_x < 2 {
        return Err("--n-x must be at least 2".to_string());
    }

    let cosmo = build_cosmology(&opts.cosmo)?;
    let recomb = RecombinationHistory::new(&cosmo);

    let z_table = logspace(opts.z_min, opts.z_max, opts.n_z);

    // The cascade's native content lives at x >= x_max_solver (opts.x_max,
    // matching the downstream PDE's own ceiling), reaching up to
    // x0(z) = omega_inj_0 / (kT_z) at each z. Since omega_inj_0 is fixed
    // and T_z grows with z, x0 shrinks as z increases — so the *lowest* z
    // in the table has the largest x0, and the output x_grid must reach at
    // least that far or the cascade signal falls entirely outside the
    // table (silently returning all zeros on interpolation, not an error).
    // opts.x_max alone is NOT a safe upper bound for this grid — it's the
    // solver handoff point, not where the cascade signal lives.
    let x0_at_z_min =
        (opts.omega_inj_0_ev * EV_IN_JOULES) / (K_BOLTZMANN * cosmo.t_cmb * (1.0 + opts.z_min));
    let x_grid_max = opts.x_max.max(x0_at_z_min * 1.5);
    let x_grid = logspace(opts.x_min, x_grid_max, opts.n_x);

    let params = DecayingParticlePhotonParams {
        omega_inj_0: opts.omega_inj_0_ev * EV_IN_JOULES,
        gamma_x: opts.gamma_x,
        n_x0_over_n_h0: opts.n_x0_over_n_h0,
    };
    let spec = TableGridSpec {
        z_table,
        n_cascade_points: opts.n_cascade_points,
        x_max_solver: opts.x_max,
        x_grid,
    };

    let (z_table, x_grid, source_2d, dq_dz_table) =
        build_decaying_particle_photon_table(&params, &cosmo, &recomb, &spec)?;

    let result = CascadeTableResult {
        z_table,
        x_grid,
        source_2d,
        heating_rate_table: dq_dz_table,
    };

    // Auto-write the sibling heating-table CSV next to --output (Option 1
    // in the session notes: pair-production lepton energy at the cascade
    // boundary, previously unaccounted for, routed through
    // TabulatedHeating). Only when an output file was actually requested --
    // if the main table is going to stdout there's no sensible sibling path.
    if let Some(main_path) = &opts.output.output_path {
        let heating_path = format!("{}.heating.csv", main_path.trim_end_matches(".json"));
        use std::io::Write;
        let mut f = std::fs::File::create(&heating_path)
            .map_err(|e| format!("failed to create {heating_path}: {e}"))?;
        writeln!(f, "z,dq_dz").map_err(|e| e.to_string())?;
        for (z, dq) in result.z_table.iter().zip(result.heating_rate_table.iter()) {
            writeln!(f, "{z},{dq}").map_err(|e| e.to_string())?;
        }
        eprintln!("Wrote heating table to {heating_path}");
    }

    Ok(result)
}

/// Derive the sibling heating-table path from the main `--output` path:
/// `probe.json` -> `probe_heating.csv`, `probe.csv` -> `probe_heating.csv`,
/// `probe` (no extension) -> `probe_heating.csv`.
fn sibling_heating_path(main_path: &str) -> String {
    match main_path.rsplit_once('.') {
        Some((stem, _ext)) => format!("{stem}_heating.csv"),
        None => format!("{main_path}_heating.csv"),
    }
}

/// Execute a Green's function calculation. Returns result without doing I/O.
pub fn execute_greens(opts: &GreensOpts) -> Result<GreensResult, String> {
    let z_h = opts.z_h;
    let delta_rho = opts.delta_rho;
    let sigma = (z_h * 0.04_f64).max(100.0);

    let x_grid: Vec<f64> = {
        let gc = GridConfig::default();
        crate::grid::FrequencyGrid::new(&gc).x
    };

    let dq_dz = |z: f64| -> f64 {
        delta_rho * (-(z - z_h).powi(2) / (2.0 * sigma * sigma)).exp()
            / (2.0 * std::f64::consts::PI * sigma * sigma).sqrt()
    };

    let (mu, y) = greens::mu_y_from_heating(&dq_dz, 1e2, z_h * 5.0, 5000);
    let delta_n = greens::distortion_from_heating(&x_grid, &dq_dz, 1e2, z_h * 5.0, 5000);

    Ok(GreensResult {
        z_h,
        mu,
        y,
        x_grid,
        delta_n,
        warnings: Vec::new(),
    })
}

/// Build a GridConfig from CLI options.
///
/// If `production_grid` is set, uses `GridConfig::production()` as the base.
/// If `n_grid > 0`, overrides `n_points`. Otherwise uses the base defaults.
fn build_grid_config(n_grid: usize, production_grid: bool) -> GridConfig {
    let mut cfg = if production_grid {
        if n_grid > 0 {
            GridConfig {
                n_points: n_grid,
                ..GridConfig::production()
            }
        } else {
            GridConfig::production()
        }
    } else if n_grid > 0 {
        GridConfig {
            n_points: n_grid,
            ..GridConfig::default()
        }
    } else {
        GridConfig::default()
    };
    // SPECTROXIDE_X_MAX (x110 patch): extend the LINEAR part of the grid to a larger x_max with the SAME linear
    // spacing and the SAME number of log points, so nothing below the old x_max changes.  Unset -> unchanged.
    if let Ok(s) = std::env::var("SPECTROXIDE_X_MAX") {
        let xm: f64 = s
            .trim()
            .parse()
            .unwrap_or_else(|_| panic!("SPECTROXIDE_X_MAX={:?} is not a number", s));
        if xm > cfg.x_max {
            let n_log = (cfg.log_fraction * cfg.n_points as f64) as usize;
            let n_lin = cfg.n_points - n_log;
            let dx = (cfg.x_max - cfg.x_transition) / (n_lin - 1).max(1) as f64;
            let n_lin_new = ((xm - cfg.x_transition) / dx).ceil() as usize + 1;
            cfg.n_points = n_log + n_lin_new;
            cfg.log_fraction = (n_log as f64 + 0.5) / cfg.n_points as f64; // keeps exactly n_log log points
            cfg.x_max = xm;
        }
    }
    cfg
}

/// Build a SolverConfig from CLI solver options with the given z_start.
fn build_solver_config(solver_opts: &SolverOpts, z_start: f64, z_end: f64) -> SolverConfig {
    let effective_dy = solver_opts.dy_max.unwrap_or(SolverConfig::default().dy_max);
    let effective_dtau = solver_opts.dtau_max.unwrap_or(10.0);
    let defaults = SolverConfig::default();
    SolverConfig {
        z_start,
        z_end,
        dy_max: effective_dy,
        dtau_max: effective_dtau,
        nc_z_min: solver_opts.nc_z_min.unwrap_or(5.0e4),
        dtau_max_photon_source: solver_opts
            .dtau_max_photon_source
            .unwrap_or(defaults.dtau_max_photon_source),
        ..defaults
    }
}

/// Extract a human-readable message from a thread panic payload.
fn extract_panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    payload
        .downcast_ref::<String>()
        .map(|s| s.as_str())
        .or_else(|| payload.downcast_ref::<&str>().copied())
        .unwrap_or("unknown error")
        .to_string()
}

/// Generate the default log-spaced redshift array for photon sweep (150 points from 1e3 to 5e6).
fn default_photon_sweep_redshifts() -> Vec<f64> {
    let n = 150;
    (0..n)
        .map(|i| {
            let t = i as f64 / (n - 1) as f64;
            let log_max = (5e6_f64).log10();
            10f64.powf(3.0 + t * (log_max - 3.0))
        })
        .collect()
}

/// Deduplicate a `Vec<String>` while preserving first-occurrence order.
/// Used to compress repeated per-worker warnings (e.g. one identical
/// "z_start in O(theta_e^2) regime" message per sweep redshift) into
/// a single user-facing line.
fn dedup_keep_order(items: Vec<String>) -> Vec<String> {
    use std::collections::HashSet;
    let mut seen: HashSet<String> = HashSet::new();
    let mut out = Vec::with_capacity(items.len());
    for item in items {
        if seen.insert(item.clone()) {
            out.push(item);
        }
    }
    out
}

/// Diagnostic-flag warnings collected as plain strings for inclusion in the
/// CLI result `warnings` array. Each flag is intended for sensitivity probes,
/// not production runs; bury this in stderr too so notebook users see it.
fn diagnostic_flag_warnings(solver_opts: &SolverOpts) -> Vec<String> {
    let mut out = Vec::new();
    if solver_opts.disable_dcbr {
        out.push(
            "--no-dcbr disables double-Compton and bremsstrahlung. Photon number is no longer \
             adjusted downward, so μ and y lose physical meaning once Δρ/ρ grows beyond the \
             linear regime. Diagnostic flag — not for production runs."
                .to_string(),
        );
    }
    if solver_opts.split_dcbr {
        out.push(
            "--split-dcbr disables the coupled DC/BR Newton iteration. Photon-source spikes can \
             produce timestep-dependent wakes at intermediate x. Diagnostic flag — prefer the \
             coupled mode for production runs."
                .to_string(),
        );
    }
    if solver_opts.cn_dcbr {
        out.push(
            "--cn-dcbr selects Crank-Nicolson for DC/BR. The CN scheme can produce negative \
             diagonals at low x where DC/BR rates diverge (CLAUDE.md pitfall #3); the default \
             backward-Euler is the validated path. Diagnostic flag only."
                .to_string(),
        );
    }
    out
}

/// Apply common solver flags from CLI options to a solver instance.
fn apply_solver_flags(solver: &mut ThermalizationSolver, solver_opts: &SolverOpts) {
    solver.disable_dcbr = solver_opts.disable_dcbr;
    solver.number_conserving = solver_opts.number_conserving;
    for w in diagnostic_flag_warnings(solver_opts) {
        eprintln!("  Warning: {w}");
        solver.diag.warnings.push(w);
    }
    if let Some(stride) = solver_opts.nc_stride {
        solver.nc_stride = stride;
    }
    if solver_opts.split_dcbr {
        solver.coupled_dcbr = false;
    }
    if solver_opts.cn_dcbr {
        solver.config.cn_dcbr = true;
    }
}

/// Validate a (config, grid, injection) combination the same way
/// `SolverBuilder::build` does, returning the soft warnings that should
/// surface to the user. Hard errors are propagated as `Err`.
///
/// The CLI used to bypass this entirely by going through
/// `ThermalizationSolver::new` + `set_config`; this helper restores the
/// validation chain without forcing every call site to use the builder.
fn validate_and_collect_warnings(
    config: &SolverConfig,
    grid_config: &GridConfig,
    injection: &InjectionScenario,
    cosmo: &Cosmology,
) -> Result<Vec<String>, String> {
    cosmo.validate()?;
    grid_config.validate()?;
    config.validate()?;
    injection.validate()?;

    if let Some((_z_center, z_upper)) = injection.characteristic_redshift() {
        if config.z_start < z_upper {
            return Err(format!(
                "z_start={:.3e} is below the injection window upper bound z={:.3e}. \
                 The solver would miss part or all of the injection. Set z_start >= {:.3e}.",
                config.z_start, z_upper, z_upper
            ));
        }
        // Symmetric check for z_end above the injection window: the injection
        // never fires in any step (z_end > z_h + 5σ_z). User gets a finite
        // run with only adiabatic-cooling signal — flag as Err.
        if config.z_end > z_upper {
            return Err(format!(
                "z_end={:.3e} is above the injection window upper bound z={:.3e}. \
                 The solver would terminate before the injection epoch is reached. \
                 Set z_end < {:.3e}.",
                config.z_end, z_upper, z_upper
            ));
        }
    }

    // `warn_stimulated_emission` is intentionally omitted: `set_injection`
    // pushes those into the solver's `diag.warnings` directly, so emitting
    // them here would double-count.
    let mut warnings = config.soft_warnings();
    warnings.extend(injection.warn_strong_distortion());
    warnings.extend(injection.warn_tabulated_coverage(config.z_start, config.z_end));
    warnings.extend(injection.warn_dark_photon_range(cosmo));
    Ok(warnings)
}

/// Execute a single PDE solve. Returns result without doing I/O.
pub fn execute_solve(opts: &SolveOpts) -> Result<SolverResult, String> {
    let cosmo = build_cosmology(&opts.cosmo)?;
    let delta_rho: f64 = opts
        .params
        .get("--delta-rho")
        .map(|s| s.parse().map_err(|_| "Invalid --delta-rho".to_string()))
        .transpose()?
        .unwrap_or(1e-5);
    if !delta_rho.is_finite() {
        return Err(format!("--delta-rho must be finite, got {delta_rho}"));
    }

    let injection = build_injection_scenario(&opts.injection_type, &opts.params, delta_rho)?;
    injection.validate()?;
    eprintln!("Injection: {}", opts.injection_type);

    let n_grid = opts.solver.n_points.unwrap_or(2000);
    let effective_dy_max = opts.solver.dy_max.unwrap_or(SolverConfig::default().dy_max);
    let effective_dtau_max = opts.solver.dtau_max.unwrap_or(10.0);
    // For dark photon resonance, start at z_res if z_start isn't set
    // explicitly by the user. The IC Δn(x) is installed by the solver at
    // z_start via InjectionScenario::initial_delta_n.
    let default_z_start = match injection.dark_photon_params(&cosmo) {
        Some((_, z_res)) => z_res,
        None => 5e6,
    };
    let z_start = opts.solver.z_start.unwrap_or(default_z_start);
    let z_end = opts.solver.z_end;

    // DarkPhotonResonance: hard-error if NWA gives no resonance in the
    // supported redshift band. Without this the solver runs to mu=y=0 and
    // looks like a "successful" null result.
    if matches!(injection, InjectionScenario::DarkPhotonResonance { .. })
        && injection.dark_photon_params(&cosmo).is_none()
    {
        return Err(
            "DarkPhotonResonance: no resonance redshift z_res in [50, 3e6] for the given \
             (epsilon, m_ev). The plasma frequency never crosses the dark-photon mass in the \
             supported band, so no conversion occurs. Adjust m_ev or extend the supported range."
                .to_string(),
        );
    }

    let mut grid_config = build_grid_config(n_grid, opts.solver.production_grid);

    if !opts.solver.no_auto_refine {
        for zone in injection.refinement_zones() {
            grid_config.refinement_zones.push(zone);
        }
        if let Some(x_min) = injection.suggested_x_min() {
            if x_min < grid_config.x_min {
                grid_config.x_min = x_min;
            }
        }
    }

    let defaults_for_validate = SolverConfig::default();
    let probe_config = SolverConfig {
        z_start,
        z_end,
        dy_max: effective_dy_max,
        dtau_max: effective_dtau_max,
        nc_z_min: opts.solver.nc_z_min.unwrap_or(5.0e4),
        dtau_max_photon_source: opts
            .solver
            .dtau_max_photon_source
            .unwrap_or(defaults_for_validate.dtau_max_photon_source),
        ..defaults_for_validate
    };
    let preflight_warnings =
        validate_and_collect_warnings(&probe_config, &grid_config, &injection, &cosmo)?;

    let mut solver = ThermalizationSolver::new(cosmo, grid_config);
    apply_solver_flags(&mut solver, &opts.solver);
    solver.set_injection(injection)?;
    solver.diag.warnings.extend(preflight_warnings);

    if let Some(amp) = opts.solver.dn_planck {
        if !amp.is_finite() {
            return Err(format!("--dn-planck amplitude must be finite, got {amp}"));
        }
        let initial_dn: Vec<f64> = solver
            .grid
            .x
            .iter()
            .map(|&x| amp * crate::spectrum::planck(x))
            .collect();
        solver.set_initial_delta_n(initial_dn);
    }

    // Note: execute_solve uses explicit dy_max/dtau_max from its own defaults,
    // not the shared build_solver_config helper, because it has special is_continuous logic.
    let defaults = SolverConfig::default();
    solver.set_config(SolverConfig {
        z_start,
        z_end,
        dy_max: effective_dy_max,
        dtau_max: effective_dtau_max,
        nc_z_min: opts.solver.nc_z_min.unwrap_or(5.0e4),
        dtau_max_photon_source: opts
            .solver
            .dtau_max_photon_source
            .unwrap_or(defaults.dtau_max_photon_source),
        ..defaults
    });

    let result = solver.run_to_result(z_end);
    let last = &result.snapshot;
    eprintln!(
        "PDE result: mu={:.4e}, y={:.4e}, drho={:.4e}, steps={}",
        last.mu, last.y, last.delta_rho_over_rho, result.step_count
    );

    Ok(result)
}

/// Execute a sweep over multiple injection redshifts. Returns result without doing I/O.
pub fn execute_sweep(opts: &SweepOpts) -> Result<SweepResult, String> {
    let cosmo = build_cosmology(&opts.cosmo)?;
    let delta_rho = opts.delta_rho;
    if !delta_rho.is_finite() {
        return Err(format!("--delta-rho must be finite, got {delta_rho}"));
    }
    let z_end = opts.solver.z_end;
    let n_grid = opts.solver.n_points.unwrap_or(0);

    let injection_redshifts: Vec<f64> = opts.z_injections.clone().unwrap_or_else(|| {
        vec![
            2e3, 3e3, 5e3, 7e3, 1e4, 1.5e4, 2e4, 3e4, 5e4, 7e4, 1e5, 1.5e5, 2e5, 3e5, 5e5, 1e6, 3e6,
        ]
    });

    let n_threads = opts.solver.n_threads.unwrap_or_else(|| {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
    });
    let mut rows_all: Vec<SweepRow> = Vec::with_capacity(injection_redshifts.len());
    let mut warnings_all: Vec<String> = Vec::new();
    for chunk in injection_redshifts.chunks(n_threads) {
        let chunk_rows: Result<Vec<(SweepRow, Vec<String>)>, String> = std::thread::scope(|s| {
            let handles: Vec<_> = chunk
                .iter()
                .map(|&z_h| {
                    let cosmo = cosmo.clone();
                    let solver_opts = &opts.solver;
                    s.spawn(move || -> Result<(SweepRow, Vec<String>), String> {
                        let sigma: f64 = (z_h * 0.04_f64).max(100.0);
                        let z_start: f64 = solver_opts.z_start.unwrap_or(z_h + 7.0 * sigma);

                        let grid_config = build_grid_config(n_grid, solver_opts.production_grid);
                        let injection = InjectionScenario::SingleBurst {
                            z_h,
                            delta_rho_over_rho: delta_rho,
                            sigma_z: sigma,
                        };
                        let probe_config = build_solver_config(solver_opts, z_start, z_end);
                        let preflight = validate_and_collect_warnings(
                            &probe_config,
                            &grid_config,
                            &injection,
                            &cosmo,
                        )?;

                        let mut solver = ThermalizationSolver::new(cosmo, grid_config);
                        apply_solver_flags(&mut solver, solver_opts);
                        solver.set_injection(injection)?;
                        solver.set_config(probe_config);
                        solver.diag.warnings.extend(preflight);

                        solver.run_with_snapshots(&[z_end]);
                        let step_count = solver.step_count;
                        let x_grid = solver.grid.x.clone();
                        let row_warnings = solver.diag.warnings.clone();
                        let snapshot = solver
                            .snapshots
                            .last()
                            .ok_or_else(|| {
                                format!("sweep z_h={z_h:.3e}: solver produced no snapshots")
                            })?
                            .clone();

                        let gf_z_min = (z_h - 10.0 * sigma).max(1e2);
                        let gf_z_max = z_h + 10.0 * sigma;
                        let dq_dz = |z: f64| -> f64 {
                            delta_rho * (-(z - z_h).powi(2) / (2.0 * sigma * sigma)).exp()
                                / (2.0 * std::f64::consts::PI * sigma * sigma).sqrt()
                        };
                        let (gf_mu, gf_y) =
                            greens::mu_y_from_heating(&dq_dz, gf_z_min, gf_z_max, 5000);
                        let gf_delta_n = greens::distortion_from_heating(
                            &x_grid, &dq_dz, gf_z_min, gf_z_max, 5000,
                        );

                        Ok((
                            SweepRow {
                                z_h,
                                snapshot,
                                gf_mu,
                                gf_y,
                                gf_delta_n,
                                x_grid,
                                step_count,
                            },
                            row_warnings,
                        ))
                    })
                })
                .collect();
            let mut results = Vec::with_capacity(handles.len());
            for h in handles {
                match h.join() {
                    Ok(Ok(row)) => results.push(row),
                    Ok(Err(msg)) => return Err(format!("Sweep worker error: {msg}")),
                    Err(e) => {
                        return Err(format!(
                            "Sweep thread panicked: {}",
                            extract_panic_message(&e)
                        ));
                    }
                }
            }
            Ok(results)
        });
        for (row, ws) in chunk_rows? {
            rows_all.push(row);
            warnings_all.extend(ws);
        }
    }
    let rows = rows_all;

    Ok(SweepResult {
        delta_rho,
        rows,
        warnings: dedup_keep_order(warnings_all),
    })
}

/// Execute a photon injection sweep over multiple injection redshifts at a fixed x_inj.
/// Returns result without doing I/O.
pub fn execute_photon_sweep(opts: &PhotonSweepOpts) -> Result<PhotonSweepResult, String> {
    let cosmo = build_cosmology(&opts.cosmo)?;
    let x_inj = opts.x_inj;
    let delta_n_over_n = opts.delta_n_over_n;
    if !x_inj.is_finite() || x_inj <= 0.0 {
        return Err(format!("--x-inj must be positive and finite, got {x_inj}"));
    }
    if !delta_n_over_n.is_finite() {
        return Err(format!(
            "--delta-n-over-n must be finite, got {delta_n_over_n}"
        ));
    }
    let sigma_x = opts.sigma_x.unwrap_or(x_inj * 0.05);
    let z_end = opts.solver.z_end;
    let n_grid = opts.solver.n_points.unwrap_or(2000);

    let injection_redshifts: Vec<f64> = opts
        .z_injections
        .clone()
        .unwrap_or_else(default_photon_sweep_redshifts);

    eprintln!(
        "Photon sweep: x_inj={x_inj:.4e}, ΔN/N={delta_n_over_n:.4e}, {} redshifts",
        injection_redshifts.len()
    );

    let n_threads = opts.solver.n_threads.unwrap_or_else(|| {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
    });
    let mut rows_all: Vec<PhotonSweepRow> = Vec::with_capacity(injection_redshifts.len());
    let mut warnings_all: Vec<String> = Vec::new();
    for chunk in injection_redshifts.chunks(n_threads) {
        let chunk_rows: Result<Vec<(PhotonSweepRow, Vec<String>)>, String> =
            std::thread::scope(|s| {
                let handles: Vec<_> = chunk
                    .iter()
                    .map(|&z_h| {
                        let cosmo = cosmo.clone();
                        let solver_opts = &opts.solver;
                        s.spawn(move || -> Result<(PhotonSweepRow, Vec<String>), String> {
                            let sigma_z: f64 = (z_h * 0.04_f64).max(100.0);
                            let z_start_val: f64 =
                                solver_opts.z_start.unwrap_or(z_h + 7.0 * sigma_z);

                            let mut grid_config =
                                build_grid_config(n_grid, solver_opts.production_grid);

                            let injection = InjectionScenario::MonochromaticPhotonInjection {
                                x_inj,
                                delta_n_over_n,
                                z_h,
                                sigma_z,
                                sigma_x,
                            };

                            if !solver_opts.no_auto_refine {
                                for zone in injection.refinement_zones() {
                                    grid_config.refinement_zones.push(zone);
                                }
                                if let Some(x_min) = injection.suggested_x_min() {
                                    if x_min < grid_config.x_min {
                                        grid_config.x_min = x_min;
                                    }
                                }
                            }

                            let probe_config = build_solver_config(solver_opts, z_start_val, z_end);
                            let preflight = validate_and_collect_warnings(
                                &probe_config,
                                &grid_config,
                                &injection,
                                &cosmo,
                            )?;

                            let mut solver = ThermalizationSolver::new(cosmo, grid_config);
                            apply_solver_flags(&mut solver, solver_opts);
                            solver.set_injection(injection)?;
                            solver.set_config(probe_config);
                            solver.diag.warnings.extend(preflight);

                            solver.run_with_snapshots(&[z_end]);
                            let step_count = solver.step_count;
                            let x_grid = solver.grid.x.clone();
                            let row_warnings = solver.diag.warnings.clone();
                            let snapshot = solver
                                .snapshots
                                .last()
                                .ok_or_else(|| {
                                    format!("photon sweep z_h={z_h:.3e}: no snapshots produced")
                                })?
                                .clone();

                            Ok((
                                PhotonSweepRow {
                                    z_h,
                                    snapshot,
                                    x_grid,
                                    step_count,
                                },
                                row_warnings,
                            ))
                        })
                    })
                    .collect();
                let mut results = Vec::with_capacity(handles.len());
                for h in handles {
                    match h.join() {
                        Ok(Ok(row)) => results.push(row),
                        Ok(Err(msg)) => {
                            return Err(format!("Photon sweep worker error: {msg}"));
                        }
                        Err(e) => {
                            return Err(format!(
                                "Photon sweep thread panicked: {}",
                                extract_panic_message(&e)
                            ));
                        }
                    }
                }
                Ok(results)
            });
        for (row, ws) in chunk_rows? {
            rows_all.push(row);
            warnings_all.extend(ws);
        }
    }
    let rows = rows_all;

    Ok(PhotonSweepResult {
        x_inj,
        delta_n_over_n,
        rows,
        warnings: dedup_keep_order(warnings_all),
    })
}

/// Execute a batch photon injection sweep over multiple x_inj values.
///
/// Flattens all (x_inj, z_h) pairs into a single thread pool, avoiding
/// the overhead of spawning separate Rust processes per x_inj.
pub fn execute_photon_sweep_batch(
    opts: &PhotonSweepBatchOpts,
) -> Result<PhotonSweepBatchResult, String> {
    let cosmo = build_cosmology(&opts.cosmo)?;
    let delta_n_over_n = opts.delta_n_over_n;
    let z_end = opts.solver.z_end;
    let n_grid = opts.solver.n_points.unwrap_or(2000);

    let injection_redshifts: Vec<f64> = opts
        .z_injections
        .clone()
        .unwrap_or_else(default_photon_sweep_redshifts);

    let n_xinj = opts.x_inj_values.len();
    let n_zh = injection_redshifts.len();
    let total = n_xinj * n_zh;

    eprintln!(
        "Photon sweep batch: {} x_inj × {} z_h = {} PDE runs",
        n_xinj, n_zh, total
    );

    // Build flat list of (x_inj_index, z_h_index) tasks
    let mut tasks: Vec<(usize, usize)> = Vec::with_capacity(total);
    for xi_idx in 0..n_xinj {
        for zh_idx in 0..n_zh {
            tasks.push((xi_idx, zh_idx));
        }
    }

    // Run tasks in chunks bounded by available parallelism
    let n_threads = opts.solver.n_threads.unwrap_or_else(|| {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
    });
    let mut rows_all: Vec<(usize, usize, PhotonSweepRow, Vec<String>)> =
        Vec::with_capacity(tasks.len());
    for task_chunk in tasks.chunks(n_threads) {
        let chunk_rows: Result<Vec<(usize, usize, PhotonSweepRow, Vec<String>)>, String> =
            std::thread::scope(|s| {
                let handles: Vec<_> = task_chunk
                    .iter()
                    .map(|&(xi_idx, zh_idx)| {
                        let cosmo = cosmo.clone();
                        let solver_opts = &opts.solver;
                        let x_inj = opts.x_inj_values[xi_idx];
                        let sigma_x = opts.sigma_x.unwrap_or(x_inj * 0.05);
                        let z_h = injection_redshifts[zh_idx];
                        s.spawn(move || -> Result<(usize, usize, PhotonSweepRow, Vec<String>), String> {
                            let sigma_z: f64 = (z_h * 0.04_f64).max(100.0);
                            let z_start_val: f64 =
                                solver_opts.z_start.unwrap_or(z_h + 7.0 * sigma_z);

                            let mut grid_config =
                                build_grid_config(n_grid, solver_opts.production_grid);

                            let injection = InjectionScenario::MonochromaticPhotonInjection {
                                x_inj,
                                delta_n_over_n,
                                z_h,
                                sigma_z,
                                sigma_x,
                            };

                            if !solver_opts.no_auto_refine {
                                for zone in injection.refinement_zones() {
                                    grid_config.refinement_zones.push(zone);
                                }
                                if let Some(x_min) = injection.suggested_x_min() {
                                    if x_min < grid_config.x_min {
                                        grid_config.x_min = x_min;
                                    }
                                }
                            }

                            let probe_config =
                                build_solver_config(solver_opts, z_start_val, z_end);
                            let preflight = validate_and_collect_warnings(
                                &probe_config,
                                &grid_config,
                                &injection,
                                &cosmo,
                            )?;

                            let mut solver = ThermalizationSolver::new(cosmo, grid_config);
                            apply_solver_flags(&mut solver, solver_opts);
                            solver.set_injection(injection)?;
                            solver.set_config(probe_config);
                            solver.diag.warnings.extend(preflight);

                            solver.run_with_snapshots(&[z_end]);
                            let step_count = solver.step_count;
                            let x_grid = solver.grid.x.clone();
                            let row_warnings = solver.diag.warnings.clone();
                            let snapshot = solver
                                .snapshots
                                .last()
                                .ok_or_else(|| {
                                    format!(
                                        "photon sweep batch (x_inj={x_inj:.3e}, z_h={z_h:.3e}): \
                                         no snapshots produced"
                                    )
                                })?
                                .clone();

                            Ok((
                                xi_idx,
                                zh_idx,
                                PhotonSweepRow {
                                    z_h,
                                    snapshot,
                                    x_grid,
                                    step_count,
                                },
                                row_warnings,
                            ))
                        })
                    })
                    .collect();

                let mut results = Vec::with_capacity(handles.len());
                for h in handles {
                    match h.join() {
                        Ok(Ok(row)) => results.push(row),
                        Ok(Err(msg)) => {
                            return Err(format!("Photon sweep batch worker error: {msg}"));
                        }
                        Err(e) => {
                            return Err(format!(
                                "Photon sweep batch thread panicked: {}",
                                extract_panic_message(&e)
                            ));
                        }
                    }
                }
                Ok(results)
            });
        rows_all.extend(chunk_rows?);
    }
    let rows = rows_all;

    // Group results by x_inj index
    let mut per_xinj: Vec<Vec<PhotonSweepRow>> =
        (0..n_xinj).map(|_| Vec::with_capacity(n_zh)).collect();
    let mut per_xinj_warnings: Vec<Vec<String>> = (0..n_xinj).map(|_| Vec::new()).collect();
    for (xi_idx, _zh_idx, row, ws) in rows {
        per_xinj[xi_idx].push(row);
        per_xinj_warnings[xi_idx].extend(ws);
    }

    // Sort each group by z_h for consistent output
    for group in &mut per_xinj {
        group.sort_by(|a, b| a.z_h.total_cmp(&b.z_h));
    }

    let results: Vec<PhotonSweepResult> = per_xinj
        .into_iter()
        .zip(per_xinj_warnings)
        .enumerate()
        .map(|(i, (rows, ws))| PhotonSweepResult {
            x_inj: opts.x_inj_values[i],
            delta_n_over_n,
            rows,
            warnings: dedup_keep_order(ws),
        })
        .collect();

    let aggregated_warnings: Vec<String> = results
        .iter()
        .flat_map(|r| r.warnings.iter().cloned())
        .collect();

    Ok(PhotonSweepBatchResult {
        results,
        warnings: dedup_keep_order(aggregated_warnings),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_args(pairs: &[(&str, &str)]) -> HashMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    fn s(val: &str) -> String {
        val.into()
    }

    // ---- Command parsing ----

    #[test]
    fn test_parse_commands() {
        // help
        match parse_command(&[s("help")]).unwrap() {
            Command::Help => {}
            _ => panic!("Expected Help"),
        }

        // empty → help
        match parse_command(&[]).unwrap() {
            Command::Help => {}
            _ => panic!("Expected Help for empty args"),
        }

        // solve
        let args = vec![
            s("solve"),
            s("single-burst"),
            s("--z-h"),
            s("2e5"),
            s("--delta-rho"),
            s("1e-5"),
        ];
        match parse_command(&args).unwrap() {
            Command::Solve(opts) => {
                assert_eq!(opts.injection_type, "single-burst");
                assert_eq!(opts.params.get("--z-h").unwrap(), "2e5");
            }
            _ => panic!("Expected Solve"),
        }

        // sweep
        let args = vec![
            s("sweep"),
            s("--z-injections"),
            s("1e3,1e4"),
            s("--format"),
            s("csv"),
        ];
        match parse_command(&args).unwrap() {
            Command::Sweep(opts) => {
                assert_eq!(opts.z_injections.as_ref().unwrap().len(), 2);
                assert_eq!(opts.output.format, OutputFormat::Csv);
            }
            _ => panic!("Expected Sweep"),
        }

        // greens
        let args = vec![s("greens"), s("--z-h"), s("2e5")];
        match parse_command(&args).unwrap() {
            Command::Greens(opts) => assert!((opts.z_h - 2e5).abs() < 1.0),
            _ => panic!("Expected Greens"),
        }

        // info with cosmology
        let args = vec![s("info"), s("--cosmology"), s("planck2018")];
        match parse_command(&args).unwrap() {
            Command::Info(opts) => assert_eq!(opts.cosmology, "planck2018"),
            _ => panic!("Expected Info"),
        }

        // info default
        match parse_command(&[s("info")]).unwrap() {
            Command::Info(opts) => assert_eq!(opts.cosmology, "default"),
            _ => panic!("Expected Info"),
        }

        // bare --flags now error rather than falling back to legacy mode
        let args = vec![s("--injection"), s("single-burst"), s("--z-h"), s("2e5")];
        assert!(parse_command(&args).is_err());
    }

    #[test]
    fn test_parse_errors() {
        // Unknown subcommand
        let err = parse_command(&[s("foobar")]).unwrap_err();
        assert!(err.contains("Unknown subcommand") && err.contains("foobar"));

        // Solve missing injection type
        let err = parse_command(&[s("solve")]).unwrap_err();
        assert!(err.contains("injection type"));

        // Flag as injection type
        let err = parse_command(&[s("solve"), s("--z-h"), s("2e5")]).unwrap_err();
        assert!(err.contains("--z-h"));

        // Greens missing z_h
        let err = parse_command(&[s("greens")]).unwrap_err();
        assert!(err.contains("--z-h"));

        // Greens invalid z_h
        let err = parse_command(&[s("greens"), s("--z-h"), s("abc")]).unwrap_err();
        assert!(err.contains("Invalid"));

        // Invalid format
        let err = parse_command(&[s("sweep"), s("--format"), s("xml")]).unwrap_err();
        assert!(err.contains("Unknown"));

        // Invalid solver opt value
        assert!(parse_command(&[s("sweep"), s("--dy-max"), s("not_a_number")]).is_err());
    }

    #[test]
    fn test_parse_float_list() {
        let result = parse_float_list("1e3, 1e4, 1e5").unwrap();
        assert_eq!(result.len(), 3);
        assert!((result[0] - 1e3).abs() < 1.0);

        let err = parse_float_list("1e3,abc,1e5").unwrap_err();
        assert!(err.contains("Invalid number"));
    }

    #[test]
    fn test_solver_opts_all_flags() {
        let args: Vec<String> = vec![
            s("sweep"),
            s("--dy-max"),
            s("0.01"),
            s("--dtau-max"),
            s("0.5"),
            s("--n-points"),
            s("2000"),
            s("--z-start"),
            s("1e6"),
            s("--z-end"),
            s("500"),
            s("--no-dcbr"),
            s("--split-dcbr"),
            s("--nc-z-min"),
            s("5e4"),
            s("--production-grid"),
            s("--dn-planck"),
            s("1e-4"),
            s("--no-auto-refine"),
        ];
        match parse_command(&args).unwrap() {
            Command::Sweep(opts) => {
                assert!(opts.solver.disable_dcbr);
                assert!(opts.solver.split_dcbr);
                assert!(opts.solver.number_conserving);
                assert!(opts.solver.production_grid);
                assert!(opts.solver.no_auto_refine);
                assert!((opts.solver.dy_max.unwrap() - 0.01).abs() < 1e-10);
                assert!((opts.solver.nc_z_min.unwrap() - 5e4).abs() < 1.0);
                assert!((opts.solver.dn_planck.unwrap() - 1e-4).abs() < 1e-15);
            }
            _ => panic!("Expected Sweep"),
        }
    }

    // ---- Cosmology building ----

    #[test]
    fn test_build_cosmology_variants() {
        // Presets
        assert!((build_cosmology(&CosmoOpts::default()).unwrap().h - 0.71).abs() < 1e-10);
        assert!(
            (build_cosmology(&CosmoOpts {
                preset: Some("planck2018".into()),
                ..CosmoOpts::default()
            })
            .unwrap()
            .h - 0.6736)
                .abs()
                < 1e-10
        );
        assert!(
            (build_cosmology(&CosmoOpts {
                preset: Some("planck2015".into()),
                ..CosmoOpts::default()
            })
            .unwrap()
            .h - 0.6727)
                .abs()
                < 1e-10
        );
        // Unknown preset returns error (prevents silent use of wrong cosmology)
        let result = build_cosmology(&CosmoOpts {
            preset: Some("unknown".into()),
            ..CosmoOpts::default()
        });
        assert!(result.is_err(), "Unknown preset should return Err");

        // Custom params: CLI accepts fractional Ω_b, Ω_m and converts.
        let custom = build_cosmology(&CosmoOpts {
            h: Some(0.70),
            omega_b: Some(0.04),
            omega_m: Some(0.30),
            t_cmb: Some(2.73),
            ..CosmoOpts::default()
        })
        .unwrap();
        assert!((custom.h - 0.70).abs() < 1e-10);
        // Stored physical density: ω_b = Ω_b h² = 0.04 × 0.49 = 0.0196.
        assert!((custom.omega_b - 0.04 * 0.70 * 0.70).abs() < 1e-10);
        // ω_cdm = (Ω_m − Ω_b) h² = 0.26 × 0.49 = 0.1274.
        assert!((custom.omega_cdm - 0.26 * 0.70 * 0.70).abs() < 1e-10);
        assert!((custom.t_cmb - 2.73).abs() < 1e-10);

        // omega_b without omega_m (or vice versa) should error: ω_cdm
        // depends on both, and silently drifting is the bug we are
        // preventing by requiring them together.
        assert!(
            build_cosmology(&CosmoOpts {
                omega_b: Some(0.04),
                ..CosmoOpts::default()
            })
            .is_err()
        );

        // Cosmology opts from CLI
        let args: Vec<String> = vec![
            s("sweep"),
            s("--omega-b"),
            s("0.05"),
            s("--omega-m"),
            s("0.31"),
            s("--h"),
            s("0.67"),
            s("--n-eff"),
            s("3.044"),
            s("--y-p"),
            s("0.245"),
            s("--t-cmb"),
            s("2.7255"),
        ];
        match parse_command(&args).unwrap() {
            Command::Sweep(opts) => {
                assert!((opts.cosmo.omega_b.unwrap() - 0.05).abs() < 1e-10);
                assert!((opts.cosmo.omega_m.unwrap() - 0.31).abs() < 1e-10);
                assert!((opts.cosmo.h.unwrap() - 0.67).abs() < 1e-10);
                assert!((opts.cosmo.t_cmb.unwrap() - 2.7255).abs() < 1e-10);
            }
            _ => panic!("Expected Sweep"),
        }

        // --omega-cdm is gone — must error with a helpful message.
        let stale: Vec<String> = vec![s("sweep"), s("--omega-cdm"), s("0.12"), s("--h"), s("0.67")];
        let err = parse_command(&stale).unwrap_err();
        assert!(
            err.contains("--omega-cdm is no longer accepted"),
            "expected migration message, got: {err}"
        );
    }

    // ---- Injection scenario building ----

    #[test]
    fn test_build_injection_all_types() {
        // Each (type_name, args, expected_variant_check)
        let cases: Vec<(&str, Vec<(&str, &str)>, fn(&InjectionScenario) -> bool)> = vec![
            ("single-burst", vec![("--z-h", "2e5")], |i| {
                matches!(i, InjectionScenario::SingleBurst { .. })
            }),
            (
                "decaying-particle",
                vec![("--f-x", "1e-8"), ("--gamma-x", "1e4")],
                |i| matches!(i, InjectionScenario::DecayingParticle { .. }),
            ),
            ("annihilating-dm", vec![("--f-ann", "1e-23")], |i| {
                matches!(i, InjectionScenario::AnnihilatingDM { .. })
            }),
            ("annihilating-dm-pwave", vec![("--f-ann", "1e-23")], |i| {
                matches!(i, InjectionScenario::AnnihilatingDMPWave { .. })
            }),
            (
                "monochromatic-photon",
                vec![
                    ("--x-inj", "3.0"),
                    ("--delta-n-over-n", "1e-6"),
                    ("--z-h", "1e5"),
                ],
                |i| matches!(i, InjectionScenario::MonochromaticPhotonInjection { .. }),
            ),
            (
                "decaying-particle-photon",
                vec![
                    ("--x-inj-0", "3.0"),
                    ("--f-inj", "1e-6"),
                    ("--gamma-x", "1e4"),
                ],
                |i| matches!(i, InjectionScenario::DecayingParticlePhoton { .. }),
            ),
        ];

        for (type_name, arg_pairs, check) in &cases {
            let args = make_args(arg_pairs);
            let inj = build_injection_scenario(type_name, &args, 1e-5)
                .unwrap_or_else(|e| panic!("Failed to build {type_name}: {e}"));
            assert!(check(&inj), "Wrong variant for {type_name}");
        }

        // Verify specific field values for single burst
        let sb_args = make_args(&[("--z-h", "2e5")]);
        match build_injection_scenario("single-burst", &sb_args, 1e-5).unwrap() {
            InjectionScenario::SingleBurst {
                z_h,
                delta_rho_over_rho,
                ..
            } => {
                assert!((z_h - 2e5).abs() < 1.0);
                assert!((delta_rho_over_rho - 1e-5).abs() < 1e-14);
            }
            _ => unreachable!(),
        }
    }

    #[test]
    fn test_build_injection_errors() {
        // Unknown type
        match build_injection_scenario("foobar", &HashMap::new(), 1e-5) {
            Err(e) => assert!(e.contains("Unknown injection type") && e.contains("foobar")),
            Ok(_) => panic!("Expected error for unknown type"),
        }

        // Invalid value
        let args = make_args(&[("--f-ann", "not_a_number")]);
        assert!(build_injection_scenario("annihilating-dm", &args, 1e-5).is_err());

        // Missing required param
        let args = make_args(&[("--sigma-z", "100.0")]);
        assert!(build_injection_scenario("single-burst", &args, f64::NAN).is_err());
    }

    // ---- Helpers and misc ----

    #[test]
    fn test_parse_flat_args_boolean_flag() {
        let args: Vec<String> = vec![s("--no-dcbr"), s("--z-h"), s("1e5")];
        let map = parse_flat_args(&args);
        assert!(map.contains_key("--no-dcbr"));
        assert_eq!(map["--no-dcbr"], "");
        assert_eq!(map["--z-h"], "1e5");
    }
}
