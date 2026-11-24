"""Tests for astrometry_corr_gradient.fid_shift: the gradient term applied to the fid centroids.

Small files in the aspect pipeline's formats (FIDPROPS, ACACENT, OBC
engineering, ASPSOL), written to a temporary folder, stand in for a real
observation.
"""

import numpy as np
import pytest
from astropy.io import fits
from Quaternion import Quat

from astrometry_corr_gradient import gradient_term
from astrometry_corr_gradient.fid_shift import (
    ARCSEC_PER_MM_DYDZ,
    DATA_NORMAL,
    EGAUSS,
    apply_fid_shift,
    fid_shift,
    fid_time_range,
    good_fid_slots,
    mean_gradients,
    predicted_event_shift,
    read_obc_gradients,
)

DATE = "2025-07-02T12:02:23"  # 2025Q3, which the reference table has


def write_fidprops(path, rows):
    """A FIDPROPS file: one row per fid slot (slot, id_string, id_status)."""
    columns = [
        fits.Column(name="slot", format="1J", array=[row[0] for row in rows]),
        fits.Column(name="id_string", format="10A", array=[row[1] for row in rows]),
        fits.Column(name="id_status", format="10A", array=[row[2] for row in rows]),
    ]
    table = fits.BinTableHDU.from_columns(columns, name="FIDPROPS")
    fits.HDUList([fits.PrimaryHDU(), table]).writeto(path)


def write_acacent(path, rows):
    """An ACACENT file: rows of (time, slot, alg, status, ang_y_sm, ang_z_sm), angles in degrees."""
    rows = list(rows)
    columns = [
        fits.Column(name="time", format="1D", unit="s", array=[row[0] for row in rows]),
        fits.Column(name="slot", format="1B", array=[row[1] for row in rows]),
        fits.Column(name="alg", format="1B", array=[row[2] for row in rows]),
        fits.Column(
            name="ang_y_sm", format="1E", unit="deg", array=[row[4] for row in rows]
        ),
        fits.Column(
            name="ang_z_sm", format="1E", unit="deg", array=[row[5] for row in rows]
        ),
        fits.Column(name="status", format="1B", array=[row[3] for row in rows]),
    ]
    table = fits.BinTableHDU.from_columns(columns, name="ACACENT")
    table.header["DATE-OBS"] = DATE
    fits.HDUList([fits.PrimaryHDU(), table]).writeto(path)


def write_obc(path, times, oobagrd3, oobagrd6, bad3=(), bad6=()):
    """An OBC engineering file: TIME, OOBAGRD3, OOBAGRD6 and a QUALITY bit per column."""
    quality = np.zeros((len(times), 3), dtype=bool)
    quality[list(bad3), 1] = True
    quality[list(bad6), 2] = True
    columns = [
        fits.Column(name="TIME", format="1D", array=times),
        fits.Column(name="OOBAGRD3", format="1E", array=oobagrd3),
        fits.Column(name="OOBAGRD6", format="1E", array=oobagrd6),
        fits.Column(name="QUALITY", format="3X", array=quality),
    ]
    table = fits.BinTableHDU.from_columns(columns, name="OBCENG")
    fits.HDUList([fits.PrimaryHDU(), table]).writeto(path)


def write_asol(path, times, attitude, dy_mm=0.0, dz_mm=0.0):
    """An ASPSOL file with one attitude throughout and constant dy and dz (mm)."""
    count = len(times)
    columns = [
        fits.Column(name="time", format="1D", array=times),
        fits.Column(name="q_att", format="4D", array=np.tile(attitude.q, (count, 1))),
        fits.Column(name="dy", format="1E", unit="mm", array=np.full(count, dy_mm)),
        fits.Column(name="dz", format="1E", unit="mm", array=np.full(count, dz_mm)),
    ]
    table = fits.BinTableHDU.from_columns(columns, name="ASPSOL")
    fits.HDUList([fits.PrimaryHDU(), table]).writeto(path)


def test_good_fid_slots_are_the_fidprops_rows_marked_good(tmp_path):
    write_fidprops(
        tmp_path / "fidprops.fits",
        [(0, "ACIS-S-2", "GOOD"), (1, "ACIS-S-4", "BAD"), (2, "ACIS-S-5", "GOOD")],
    )
    assert good_fid_slots(tmp_path / "fidprops.fits") == [0, 2]


def test_fid_time_range_uses_good_egauss_rows_of_fid_slots_only(tmp_path):
    rows = [
        (100.0, 0, EGAUSS, DATA_NORMAL, 0.1, 0.2),
        (50.0, 3, EGAUSS, DATA_NORMAL, 0.1, 0.2),  # a star slot
        (60.0, 0, EGAUSS, 1, 0.1, 0.2),  # not normal
        (70.0, 2, 1, DATA_NORMAL, 0.1, 0.2),  # another algorithm
        (300.0, 2, EGAUSS, DATA_NORMAL, 0.1, 0.2),
    ]
    write_acacent(tmp_path / "acen.fits", rows)
    assert fid_time_range(tmp_path / "acen.fits", [0, 2]) == (100.0, 300.0)


def test_read_obc_gradients_takes_each_columns_quality_bit(tmp_path):
    write_obc(
        tmp_path / "obc.fits",
        times=[1.0, 2.0, 3.0],
        oobagrd3=[0.1, 0.2, 0.3],
        oobagrd6=[0.01, 0.02, 0.03],
        bad3=[1],
        bad6=[2],
    )
    times, gradients, good = read_obc_gradients([tmp_path / "obc.fits"])
    assert list(times) == [1.0, 2.0, 3.0]
    assert list(good["OOBAGRD3"]) == [True, False, True]
    assert list(good["OOBAGRD6"]) == [True, True, False]
    assert gradients["OOBAGRD6"][1] == pytest.approx(0.02)


def test_mean_gradients_fill_bad_samples_like_aca_corr_fid(tmp_path):
    # in range 10-50: OOBAGRD3 0.0, bad, bad, 0.6, 0.8 -> 0.0, 0.2, 0.4, 0.6, 0.8
    # (linear in sample index)
    # OOBAGRD6: bad, 0.02, 0.03, 0.04, bad -> 0.02 (first good), ..., 0.04 (last good)
    write_obc(
        tmp_path / "obc.fits",
        times=[5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        oobagrd3=[9.0, 0.0, 9.0, 9.0, 0.6, 0.8, 9.0],
        oobagrd6=[9.0, 9.0, 0.02, 0.03, 0.04, 9.0, 9.0],
        bad3=[2, 3],
        bad6=[1, 5],
    )
    means = mean_gradients(read_obc_gradients([tmp_path / "obc.fits"]), 10.0, 50.0)
    assert means["OOBAGRD3"] == pytest.approx(np.mean([0.0, 0.2, 0.4, 0.6, 0.8]))
    assert means["OOBAGRD6"] == pytest.approx(np.mean([0.02, 0.02, 0.03, 0.04, 0.04]))


def test_mean_gradients_need_a_good_sample(tmp_path):
    write_obc(
        tmp_path / "obc.fits", [10.0, 20.0], [0.1, 0.2], [0.01, 0.02], bad3=[0, 1]
    )
    with pytest.raises(ValueError, match="OOBAGRD3"):
        mean_gradients(read_obc_gradients([tmp_path / "obc.fits"]), 0.0, 100.0)


def test_fid_shift_is_the_gradient_term_in_degrees():
    means = {"OOBAGRD3": -0.26719, "OOBAGRD6": 0.14234}
    expected = gradient_term.gradient_term(DATE, means)
    shift = fid_shift(DATE, means)
    assert shift["dy_arcsec"] == pytest.approx(expected["dy"][0])
    assert shift["dz_arcsec"] == pytest.approx(expected["dz"][0])
    assert shift["ang_y_deg"] == pytest.approx(expected["dy"][0] / 3600)
    assert shift["ang_z_deg"] == pytest.approx(expected["dz"][0] / 3600)
    assert shift["quarter"] == "2025Q3"


def test_fid_shift_refuses_a_quarter_without_a_reference():
    with pytest.raises(ValueError, match="reference"):
        fid_shift("2001-06-01T00:00:00", {"OOBAGRD3": 0.9, "OOBAGRD6": 0.04})


def test_apply_fid_shift_moves_fid_rows_only_and_records_it(tmp_path):
    rows = [
        (100.0, 0, EGAUSS, DATA_NORMAL, 0.5, -0.5),
        (100.0, 3, EGAUSS, DATA_NORMAL, 0.25, 0.25),  # a star slot
        (110.0, 0, 1, 1, 0.5, -0.5),  # a fid row of another algorithm, shifted too
    ]
    write_acacent(tmp_path / "acen.fits", rows)
    shift = {
        "ang_y_deg": 1e-4,
        "ang_z_deg": -2e-4,
        "dy_arcsec": 0.36,
        "dz_arcsec": -0.72,
        "quarter": "2025Q3",
    }
    apply_fid_shift(tmp_path / "acen.fits", tmp_path / "shifted.fits", [0], shift)
    with fits.open(tmp_path / "shifted.fits") as hdus:
        data, header = hdus[1].data, hdus[1].header
        assert data["ang_y_sm"] == pytest.approx(
            np.float32([0.5 - 1e-4, 0.25, 0.5 - 1e-4])
        )
        assert data["ang_z_sm"] == pytest.approx(
            np.float32([-0.5 + 2e-4, 0.25, -0.5 + 2e-4])
        )
        assert (
            data["ang_y_sm"].dtype.kind == "f" and data["ang_y_sm"].dtype.itemsize == 4
        )
        assert header["GTRM_DY"] == pytest.approx(0.36)
        assert header["GTRM_DZ"] == pytest.approx(-0.72)
        assert any("gradient term" in str(line) for line in header["HISTORY"])
    with fits.open(tmp_path / "acen.fits") as hdus:
        assert hdus[1].data["ang_y_sm"][0] == pytest.approx(
            0.5
        )  # the input is untouched


def test_predicted_event_shift_adds_the_attitude_turn_and_the_dydz_change(tmp_path):
    times = np.arange(10.0)
    before = Quat([150.0, 40.0, 30.0])
    after = before * Quat([0.3 / 3600, -0.2 / 3600, 0.0])  # yaw +0.3", pitch -0.2"
    write_asol(tmp_path / "before.fits", times, before)
    write_asol(tmp_path / "after.fits", times, after, dy_mm=0.01, dz_mm=-0.02)
    shift_y, shift_z = predicted_event_shift(
        tmp_path / "before.fits", tmp_path / "after.fits"
    )
    assert shift_y == pytest.approx(0.3 + ARCSEC_PER_MM_DYDZ * 0.01, abs=1e-4)
    assert shift_z == pytest.approx(-0.2 + ARCSEC_PER_MM_DYDZ * -0.02, abs=1e-4)
