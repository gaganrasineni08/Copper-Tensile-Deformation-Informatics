"""Checks for plot_stress_strain.py. Run with `python -m pytest` or `python test_plot_stress_strain.py`."""
import os
import tempfile

import numpy as np

from plot_stress_strain import STRAIN_PER_ROW, first_yield, load, temperature


def _write(rows):
    fd, path = tempfile.mkstemp(suffix="_300K.txt")
    with os.fdopen(fd, "w") as f:
        f.write("# Fix print output for fix def_print\n")
        for s, p in rows:
            f.write(f"{s} {p}\n")
    return path


def test_load_rebuilds_all_zero_strain():
    path = _write([(0, 0.1), (1e-16, 0.2), (-5e-16, 0.3)])
    try:
        strain, stress, rebuilt = load(path)
    finally:
        os.remove(path)
    assert rebuilt
    assert np.allclose(strain, [0, STRAIN_PER_ROW, 2 * STRAIN_PER_ROW])
    assert np.allclose(stress, [0.1, 0.2, 0.3])


def test_load_keeps_real_strain():
    path = _write([(0, 0.1), (0.002, 0.2), (0.004, 0.3)])
    try:
        strain, _, rebuilt = load(path)
    finally:
        os.remove(path)
    assert not rebuilt
    assert np.allclose(strain, [0, 0.002, 0.004])


def test_first_yield_ignores_later_reloading_peak():
    # Rises to 9, drops to 3, then climbs to 10: the yield is the first peak.
    stress = np.array([0, 3, 6, 9, 8.5, 5, 3, 6, 10])
    assert first_yield(stress) == 3


def test_first_yield_without_a_drop_is_the_maximum():
    assert first_yield(np.array([0.0, 1.0, 2.0, 1.5])) == 2


def test_temperature_from_name():
    assert temperature("stress_strain_600K.txt") == 600
    assert temperature("curves.txt") is None


def test_temperature_ignores_the_folder():
    path = os.path.join("runs", "300K_old", "stress_strain_600K.txt")
    assert temperature(path) == 600
    assert temperature(os.path.join("900K", "curves.txt")) is None


def test_committed_data_yield_peaks():
    here = os.path.dirname(os.path.abspath(__file__))
    expected = {300: 9.46, 600: 7.45, 900: 5.07}  # table in the README
    for t, gpa in expected.items():
        _, stress, _ = load(os.path.join(here, f"stress_strain_{t}K.txt"))
        assert abs(stress[first_yield(stress)] - gpa) < 0.01, t


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
