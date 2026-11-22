"""Tests for astrometry_corr_gradient.calalign: which matrix each observation gets, and the rebase."""

import numpy as np
import pytest
from astropy.table import Table
from cxotime import CxoTime

from astrometry_corr_gradient.calalign import (
    calalign_table,
    current_caldb_matrices,
    latest_matrices,
    processing_matrices,
    rebase,
    release_tuple,
)


def secs(date: str) -> float:
    return float(CxoTime(date).secs)


def hand_table() -> Table:
    """ACIS-S: one N0008 and one N0009 matrix, two N0010 ones; HRC-S: one N0010."""
    rows = [
        ("4.4.4", "ACIS-S", "2000-01-01", "2050-01-01", 1.0, 10.0),
        ("4.6.2", "ACIS-S", "2005-01-01", "2050-01-01", 2.0, 20.0),
        ("4.10.0", "ACIS-S", "2013-01-01", "2014-01-01", 3.0, 30.0),
        ("4.10.0", "ACIS-S", "2014-01-01", "2050-01-01", 4.0, 40.0),
        ("4.10.0", "HRC-S", "2013-01-01", "2050-01-01", 5.0, 50.0),
    ]
    table = Table(
        rows=rows, names=["caldb_version", "detector", "start", "stop", "dy", "dz"]
    )
    table["start_secs"] = [secs(date) for date in table["start"]]
    table["stop_secs"] = [secs(date) for date in table["stop"]]
    return table


def test_release_tuple_ignores_a_trailing_dot():
    assert release_tuple("4.9.6.") == (4, 9, 6)
    assert release_tuple("4.10.0") > release_tuple("4.9.6")


def test_processing_matrix_is_the_latest_release_no_later_than_the_processing_one():
    times = np.array([secs("2015-06-01")] * 3)
    dy, dz = processing_matrices(
        np.array(["ACIS-S"] * 3),
        times,
        np.array(["4.5.0", "4.9.6", "4.10.2"]),
        hand_table(),
    )
    # 4.5.0 has only N0008; 4.9.6 has N0009 at the latest; 4.10.2 has N0010's latest start
    assert dy.tolist() == [1.0, 2.0, 4.0]
    assert dz.tolist() == [10.0, 20.0, 40.0]


def test_processing_matrix_must_be_valid_at_the_time():
    dy, _ = processing_matrices(
        np.array(["ACIS-S"]),
        np.array([secs("2013-06-01")]),
        np.array(["4.10.0"]),
        hand_table(),
    )
    assert dy.tolist() == [3.0]  # the 2014 matrix is not valid yet


def test_processing_matrix_raises_when_none_fits():
    with pytest.raises(ValueError, match="no CALALIGN matrix"):
        processing_matrices(
            np.array(["HRC-S"]),
            np.array([secs("2015-06-01")]),
            np.array(["4.9.6"]),
            hand_table(),
        )


def test_latest_matrices_are_each_detectors_latest_start():
    reference = latest_matrices(hand_table())
    assert reference["ACIS-S"] == {"start": "2014-01-01", "dy": 4.0, "dz": 40.0}
    assert reference["HRC-S"]["dy"] == 5.0


def test_latest_matrices_reject_a_tied_latest_start():
    table = hand_table()
    table.add_row(table[3])
    with pytest.raises(ValueError, match="latest start"):
        latest_matrices(table)


def test_rebase_swaps_the_processing_matrix_for_the_reference():
    dy_rebased, dz_rebased, reference = rebase(
        np.array([0.5]),
        np.array([-0.2]),
        np.array(["ACIS-S"]),
        np.array([secs("2015-06-01")]),
        np.array(["4.9.6"]),
        hand_table(),
    )
    # processed with N0009 (dy 2, dz 20); reference dy 4, dz 40
    assert dy_rebased[0] == pytest.approx(0.5 - (2.0 - 4.0))
    assert dz_rebased[0] == pytest.approx(-0.2 - (20.0 - 40.0))
    assert reference["ACIS-S"]["start"] == "2014-01-01"


def test_current_caldb_uses_the_newest_release_rule():
    dy, _ = current_caldb_matrices(
        np.array(["ACIS-S", "ACIS-S"]),
        np.array([secs("2010-06-01"), secs("2015-06-01")]),
        hand_table(),
    )
    assert dy.tolist() == [2.0, 4.0]


def test_the_data_file_holds_every_matrix_with_each_detectors_latest_on_2021_07_02():
    table = calalign_table()
    assert len(table) == 292
    assert len(set(table["file"])) == 73
    assert set(table["detector"]) == {"ACIS-S", "ACIS-I", "HRC-S", "HRC-I"}
    assert {row["start"] for row in latest_matrices(table).values()} == {"2021-07-02"}
