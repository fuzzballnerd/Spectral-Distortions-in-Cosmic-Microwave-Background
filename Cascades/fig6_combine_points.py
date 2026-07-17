"""
fig6_combine_points.py -- stitch together the per-point .npz files produced
by repeated runs of fig6_single_spectrum.py into one final Fig. 6d-style
overlay plot.

Workflow this closes: fig6_single_spectrum.py runs one (species, E0_eV)
point at a time (edit SPECIES/E0_EV, rerun) and saves each point under
fig6_points/fig6_point_<species>_<E0_eV>eV_data.npz. Nothing previously
stitched those saved runs back together -- this script is that missing
combining step (the analogue of fig6_spectrum_via_spectroxide.py's own
plot_results(), but reading saved single-point runs instead of an in-memory
sweep).

Usage:
    # after collecting some subset of the 6 Fig. 6d points via
    # fig6_single_spectrum.py (10 keV/100 keV/1 MeV x e-/gamma):
    python fig6_combine_points.py

    # to combine only a specific subset of points, or a different
    # directory:
    python fig6_combine_points.py --points-dir fig6_points --pattern "*.npz"

Any number of points found in the directory are overlaid (not just 6) --
run fig6_single_spectrum.py for as many/few points as you want plotted
together, then run this.
"""

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from spectroxide.greens import delta_n_to_delta_I, y_shape

from fig6_single_spectrum import POINTS_DIR, JY_TO_PAPER_UNITS

COMBINED_PLOT_FILE = "fig6_combined.png"
COMBINED_DATA_FILE = "fig6_combined_data.npz"


def load_points(points_dir=POINTS_DIR, pattern="fig6_point_*_data.npz"):
    """Load every saved single-point .npz from points_dir, sorted by
    (species, E0_eV) for a stable legend/plot order."""
    paths = sorted(glob.glob(os.path.join(points_dir, pattern)))
    if not paths:
        raise FileNotFoundError(
            f"No point files matching {pattern!r} found in {points_dir!r}. "
            f"Run fig6_single_spectrum.py at least once first (edit its "
            f"SPECIES/E0_EV constants and rerun for each point you want)."
        )

    points = []
    for path in paths:
        with np.load(path, allow_pickle=True) as data:
            points.append({
                "path": path,
                "species": str(data["species"]),
                "E0_eV": float(data["E0_eV"]),
                "z_inj": float(data["z_inj"]),
                "x": data["x"],
                "delta_I_paper_units": data["delta_I_paper_units"],
                "y": float(data["y"]),
                "mu": float(data["mu"]),
            })

    # Stable order: species (e before gamma), then energy ascending -- so
    # reruns/subsets always produce the same legend ordering regardless of
    # filesystem glob order.
    points.sort(key=lambda p: (p["species"] != "e", p["E0_eV"]))
    return points


def combine_and_plot(points, plot_file=COMBINED_PLOT_FILE):
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "figure.dpi": 130, "savefig.dpi": 150, "font.size": 11,
        "axes.titlesize": 12, "axes.titleweight": "bold",
        "axes.labelsize": 11, "legend.fontsize": 9,
        "legend.frameon": False, "lines.linewidth": 1.8,
    })

    fig, ax = plt.subplots(figsize=(8, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(points), 1)))
    for p, c in zip(points, colors):
        species_label = "e-" if p["species"] == "e" else "gamma"
        label = f"{p['E0_eV']:.0f} eV {species_label}"
        ax.plot(p["x"], p["delta_I_paper_units"], color=c, label=label)

    # Scaled y-type distortion shape for comparison, as AK19 does ("we have
    # only shown a scaled y-distortion spectrum for comparison") -- use the
    # first point's y and x grid, same convention as
    # fig6_spectrum_via_spectroxide.py's plot_results().
    ref = points[0]
    x_ref = ref["x"]
    delta_n_y = ref["y"] * y_shape(x_ref)
    _, delta_I_y_jy = delta_n_to_delta_I(x_ref, delta_n_y)
    ax.plot(x_ref, delta_I_y_jy * JY_TO_PAPER_UNITS, "k--", lw=1.2,
            label=f"y-distortion shape (y={ref['y']:.2e})")

    ax.set_xlabel("x")
    ax.set_ylabel(r"$\delta I_\nu$  [$10^{-23}$ W m$^{-2}$ Hz$^{-1}$ sr$^{-1}$]")
    z_inj_vals = {p["z_inj"] for p in points}
    title_z = f"{ref['z_inj']:.0f}" if len(z_inj_vals) == 1 else "mixed"
    ax.set_title(f"Fig. 6-style spectrum overlay (via spectroxide), z={title_z}")
    ax.legend(fontsize=9, loc="best")
    ax.axhline(0, color="gray", lw=0.6)

    fig.tight_layout()
    fig.savefig(plot_file)
    print(f"Wrote {plot_file}")


def save_combined_data(points, data_file=COMBINED_DATA_FILE):
    save_dict = {
        "species_list": np.array([p["species"] for p in points]),
        "E0_eV_list": np.array([p["E0_eV"] for p in points]),
        "y_list": np.array([p["y"] for p in points]),
        "mu_list": np.array([p["mu"] for p in points]),
    }
    for i, p in enumerate(points):
        tag = f"pt{i:02d}_{p['species']}_{p['E0_eV']:.0e}"
        save_dict[f"{tag}_x"] = p["x"]
        save_dict[f"{tag}_delta_I_paper_units"] = p["delta_I_paper_units"]
    np.savez(data_file, **save_dict)
    print(f"Wrote {data_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points-dir", default=POINTS_DIR,
                         help=f"directory to read per-point .npz files from "
                              f"(default: {POINTS_DIR})")
    parser.add_argument("--pattern", default="fig6_point_*_data.npz",
                         help="glob pattern for point files within --points-dir")
    parser.add_argument("--plot-file", default=COMBINED_PLOT_FILE)
    parser.add_argument("--data-file", default=COMBINED_DATA_FILE)
    args = parser.parse_args()

    points = load_points(args.points_dir, args.pattern)
    print(f"Loaded {len(points)} point(s) from {args.points_dir}:")
    for p in points:
        print(f"  - {p['species']}, E0={p['E0_eV']:.3e} eV "
              f"(y={p['y']:.3e}, mu={p['mu']:.3e})  [{p['path']}]")

    combine_and_plot(points, args.plot_file)
    save_combined_data(points, args.data_file)
