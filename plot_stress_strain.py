"""Plot the stress-strain curves and report the elastic modulus and first yield peak at each temperature.

Usage:
    python plot_stress_strain.py                 # all stress_strain_*K.txt here
    python plot_stress_strain.py -o curves.png stress_strain_300K.txt ...

Files written by the old version of in.cu_tensile.txt have a strain column
that is ~0 on every row (see the README). For those the strain is rebuilt
from the row number: one row every 100 steps of 1 fs at erate 0.01 ps^-1,
i.e. 0.001 strain per row.
"""
import argparse
import glob
import os
import re
import sys
import warnings

import numpy as np

STRAIN_PER_ROW = 0.001  # 100 steps * 0.001 ps * 0.01 ps^-1


def load(path):
    """Return (strain, stress_GPa, rebuilt) for one fix-print file.

    Raises ValueError for a file with no data rows or fewer than two columns,
    e.g. the header-only file left behind when a LAMMPS run is stopped early."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # numpy's "Empty input file"
        data = np.loadtxt(path, comments="#", ndmin=2)
    if data.shape[0] == 0 or data.shape[1] < 2:
        raise ValueError(f"{path}: no strain/stress rows found")
    strain, stress = data[:, 0], data[:, 1]
    rebuilt = bool(np.all(np.abs(strain) < 1e-9)) and len(strain) > 1
    if rebuilt:
        strain = STRAIN_PER_ROW * np.arange(len(strain))
    return strain, stress, rebuilt


def first_yield(stress, drop_gpa=1.0):
    """Index of the first yield peak: the highest stress before the curve first
    falls more than drop_gpa below its running maximum. Later re-loading peaks
    (the 600 K and 900 K curves climb again after 0.17 strain) are ignored."""
    running = np.maximum.accumulate(stress)
    dropped = np.nonzero(stress < running - drop_gpa)[0]
    end = dropped[0] if len(dropped) else len(stress)
    return int(np.argmax(stress[:end]))


def elastic_modulus(strain, stress, max_strain=0.02):
    """Young's modulus in GPa: slope of a straight-line fit to the points with
    strain <= max_strain (the linear elastic part, well before the yield peak).
    Returns None when fewer than three points fall in that range."""
    mask = (strain >= 0) & (strain <= max_strain)
    if np.count_nonzero(mask) < 3:
        return None
    return float(np.polyfit(strain[mask], stress[mask], 1)[0])


def temperature(path):
    """Temperature in K from the file name, e.g. stress_strain_600K.txt -> 600.
    Only the base name is looked at, so a folder such as runs/300K_old/ does
    not hide the real temperature, and the last NNNK in the name wins."""
    found = re.findall(r"(\d+)K", os.path.basename(path))
    return int(found[-1]) if found else None


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("files", nargs="*", help="default: stress_strain_*K.txt")
    ap.add_argument("-o", "--out", default="stress_strain.png", help="output image")
    args = ap.parse_args()

    files = args.files or glob.glob("stress_strain_*K.txt")
    files.sort(key=lambda f: (temperature(f) is None, temperature(f) or 0, f))
    if not files:
        ap.error("no stress_strain_*K.txt files found")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    print(f"{'file':<26}{'E (GPa)':>9}{'yield peak (GPa)':>18}{'at strain':>11}")
    plotted = 0
    for f in files:
        try:
            strain, stress, rebuilt = load(f)
        except ValueError as e:
            print(f"skipped {e}", file=sys.stderr)
            continue
        plotted += 1
        i = first_yield(stress)
        e = elastic_modulus(strain, stress)
        e_txt = f"{e:>9.1f}" if e is not None else f"{'-':>9}"
        note = "  (strain rebuilt from row number)" if rebuilt else ""
        print(f"{f:<26}{e_txt}{stress[i]:>18.3f}{strain[i]:>11.3f}{note}")
        t = temperature(f)
        (line,) = ax.plot(strain, stress, lw=1.2, label=f"{t} K" if t else f)
        ax.plot(strain[i], stress[i], "o", ms=4, color=line.get_color())

    if not plotted:
        ap.error("none of the files had any data rows")

    ax.set_xlabel("Engineering strain")
    ax.set_ylabel("Stress (GPa)")
    ax.set_title("Cu tensile test, erate 0.01 ps$^{-1}$")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
