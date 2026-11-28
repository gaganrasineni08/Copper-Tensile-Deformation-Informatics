"""Checks for plot_stress_strain.py. Run with `python -m pytest` or `python test_plot_stress_strain.py`."""
import contextlib
import io
import os
import sys
import tempfile

import numpy as np
import pytest

from plot_stress_strain import (
    STRAIN_PER_ROW,
    elastic_modulus,
    first_yield,
    load,
    main,
    temperature,
)


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


def test_load_rejects_header_only_file():
    # A run stopped before the first fix print leaves just the header line.
    path = _write([])
    try:
        with pytest.raises(ValueError, match="no strain/stress rows"):
            load(path)
    finally:
        os.remove(path)


def test_load_rejects_single_column():
    fd, path = tempfile.mkstemp(suffix="_300K.txt")
    with os.fdopen(fd, "w") as f:
        f.write("0.1\n0.2\n")
    try:
        with pytest.raises(ValueError):
            load(path)
    finally:
        os.remove(path)


def test_main_skips_empty_file():
    # An empty file next to good ones is reported and skipped, not a crash.
    here = os.path.dirname(os.path.abspath(__file__))
    good = os.path.join(here, "stress_strain_300K.txt")
    with tempfile.TemporaryDirectory() as tmp:
        empty = os.path.join(tmp, "stress_strain_600K.txt")
        with open(empty, "w") as f:
            f.write("# Fix print output for fix def_print\n")
        out = os.path.join(tmp, "curves.png")
        err = io.StringIO()
        argv = sys.argv
        sys.argv = ["plot", "-o", out, good, empty]
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
                main()
        finally:
            sys.argv = argv
        assert "skipped" in err.getvalue() and "600K" in err.getvalue()
        assert os.path.exists(out)


def test_first_yield_ignores_later_reloading_peak():
    # Rises to 9, drops to 3, then climbs to 10: the yield is the first peak.
    stress = np.array([0, 3, 6, 9, 8.5, 5, 3, 6, 10])
    assert first_yield(stress) == 3


def test_first_yield_without_a_drop_is_the_maximum():
    assert first_yield(np.array([0.0, 1.0, 2.0, 1.5])) == 2


def test_elastic_modulus_of_a_straight_line():
    strain = np.linspace(0, 0.05, 51)
    stress = 70.0 * strain
    stress[strain > 0.02] = 0.0  # past the fit window, must not count
    assert elastic_modulus(strain, stress) == pytest.approx(70.0)


def test_elastic_modulus_needs_three_points():
    assert elastic_modulus(np.array([0.0, 0.01]), np.array([0.0, 0.7])) is None


def test_committed_data_elastic_modulus():
    here = os.path.dirname(os.path.abspath(__file__))
    expected = {300: 70.2, 600: 57.2, 900: 50.1}  # table in the README
    for t, gpa in expected.items():
        strain, stress, _ = load(os.path.join(here, f"stress_strain_{t}K.txt"))
        assert abs(elastic_modulus(strain, stress) - gpa) < 0.1, t


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
