"""Plot the stress-strain curves and report the first yield peak at each temperature.

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
import re

import numpy as np

STRAIN_PER_ROW = 0.001  # 100 steps * 0.001 ps * 0.01 ps^-1


def load(path):
    """Return (strain, stress_GPa, rebuilt) for one fix-print file."""
    data = np.loadtxt(path, comments="#", ndmin=2)
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


def temperature(path):
    m = re.search(r"(\d+)K", path)
    return int(m.group(1)) if m else None


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
    print(f"{'file':<26}{'yield peak (GPa)':>18}{'at strain':>11}")
    for f in files:
        strain, stress, rebuilt = load(f)
        i = first_yield(stress)
        note = "  (strain rebuilt from row number)" if rebuilt else ""
        print(f"{f:<26}{stress[i]:>18.3f}{strain[i]:>11.3f}{note}")
        t = temperature(f)
        (line,) = ax.plot(strain, stress, lw=1.2, label=f"{t} K" if t else f)
        ax.plot(strain[i], stress[i], "o", ms=4, color=line.get_color())

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
