"""Tests for astrometry_corr_gradient.mock_n0011: quarterly CALALIGN matrices from 2021-07-02, made from the calibration set."""

import numpy as np
import pytest
from astropy.table import Table, vstack
from cxotime import CxoTime

from astrometry_corr_gradient.calalign import (
    calalign_table,
    current_caldb_matrices,
    latest_matrices,
    rebase,
)
from astrometry_corr_gradient.gradient_term import label_index
from astrometry_corr_gradient.mock_n0011 import (
    MOCK_RELEASE,
    detector_constants,
    mock_matrices,
    quarter_levels,
)


def made_up_calibration() -> Table:
    """Sources in 2021Q3 to 2022Q1 on ACIS-S and HRC-I, with a level per quarter and a detector offset."""
    rows = []
    levels = {"2021-08": 0.1, "2021-11": 0.3, "2022-02": 0.6}
    offsets = {"ACIS-S": 0.0, "HRC-I": 0.2}
    for month, level in levels.items():
        for detector, offset in offsets.items():
            for day in range(1, 13):
                rows.append(
                    {
                        "obsid": len(rows),
                        "date": f"{month}-{day:02d} 12:00:00",
                        "detector": detector,
                        "dz_rebased": level + offset + 0.01 * (day - 6.5),
                        "dy_rebased": -level - offset,
                    }
                )
    return Table(rows=rows)


Q3_2021, Q4_2021, Q1_2022, Q2_2022 = (
    label_index(label) for label in ("2021Q3", "2021Q4", "2022Q1", "2022Q2")
)


def test_quarter_levels_use_each_quarters_own_median_before_the_lagging_start():
    levels = quarter_levels(
        made_up_calibration(), last_complete_quarter=Q1_2022, lagging_from=Q2_2022
    )
    # own quarters: the median over both detectors; the quarter still filling lags
    assert levels["dz"][Q3_2021] == pytest.approx(0.1 + 0.1)
    assert levels["dz"][Q4_2021] == pytest.approx(0.3 + 0.1)
    assert levels["dz"][Q2_2022] == pytest.approx(levels["dz"][Q1_2022])


def test_quarter_levels_lag_from_the_lagging_start():
    levels = quarter_levels(
        made_up_calibration(), last_complete_quarter=Q1_2022, lagging_from=Q3_2021
    )
    assert Q3_2021 not in levels["dz"]  # nothing before it to lag from
    assert levels["dz"][Q4_2021] == pytest.approx(0.2)
    assert levels["dz"][Q1_2022] == pytest.approx(0.4)


def test_detector_constants_take_out_each_detectors_offset_and_exempt_hrc_i_in_dy():
    calibration = made_up_calibration()
    levels = quarter_levels(
        calibration, last_complete_quarter=Q1_2022, lagging_from=Q2_2022
    )
    constants = detector_constants(calibration, levels)
    assert constants["dz"]["ACIS-S"] == pytest.approx(-0.1)
    assert constants["dz"]["HRC-I"] == pytest.approx(0.1)
    assert constants["dy"]["HRC-I"] == 0.0


def test_mock_matrices_are_quarterly_epochs_from_2021_07_02():
    mock = mock_matrices(
        made_up_calibration(), last_complete_quarter=Q1_2022, lagging_from=Q2_2022
    )
    acis = mock[mock["detector"] == "ACIS-S"]
    assert list(acis["start"]) == [
        "2021-07-02",
        "2021-10-01",
        "2022-01-01",
        "2022-04-01",
    ]
    assert list(acis["stop"])[:-1] == ["2021-10-01", "2022-01-01", "2022-04-01"]
    assert set(mock["version"]) == {"N0011"}
    assert set(mock["caldb_version"]) == {MOCK_RELEASE}


def test_mock_matrices_zero_each_quarters_median_offset():
    calibration = made_up_calibration()
    mock = mock_matrices(
        calibration, last_complete_quarter=Q1_2022, lagging_from=Q2_2022
    )
    table = vstack([calalign_table(), mock])
    times = CxoTime(np.asarray(calibration["date"]).astype(str)).secs
    detectors = np.asarray(calibration["detector"]).astype(str)
    _, current_dz = current_caldb_matrices(detectors, times, table)
    reference = latest_matrices()
    latest_dz = np.array([reference[detector]["dz"] for detector in detectors])
    after = np.asarray(calibration["dz_rebased"]) + (current_dz - latest_dz)
    # processed with the mock N0011, each detector's offsets in each own quarter centre on zero
    for detector in ("ACIS-S", "HRC-I"):
        for month in ("2021-08", "2021-11", "2022-02"):
            rows = (detectors == detector) & np.char.startswith(
                np.asarray(calibration["date"]).astype(str), month
            )
            assert np.median(after[rows]) == pytest.approx(0.0, abs=1e-9)


def test_today_before_2021_07_02_keeps_the_shipped_matrices():
    mock = mock_matrices(
        made_up_calibration(), last_complete_quarter=Q1_2022, lagging_from=Q2_2022
    )
    with_mock = vstack([calalign_table(), mock])
    early = np.array([CxoTime("2015-06-01").secs])
    with_n0011 = current_caldb_matrices(np.array(["ACIS-S"]), early, with_mock)
    shipped = current_caldb_matrices(np.array(["ACIS-S"]), early, calalign_table())
    assert all(np.array_equal(a, b) for a, b in zip(with_n0011, shipped, strict=True))


def test_an_observation_processed_with_the_mock_rebases_onto_the_reference():
    calibration = made_up_calibration()
    mock = mock_matrices(
        calibration, last_complete_quarter=Q1_2022, lagging_from=Q2_2022
    )
    table = vstack([calalign_table(), mock])
    time = CxoTime("2021-11-05").secs
    epoch = mock[(mock["detector"] == "ACIS-S") & (mock["start"] == "2021-10-01")][0]
    # with N0011 shipped, the reference is its latest epoch
    reference = latest_matrices(table)["ACIS-S"]
    assert reference["start"] == "2022-04-01"
    # an offset of 0 measured with the mock matrix is reference - mock matrix once rebased
    dy_rebased, dz_rebased, _ = rebase(
        np.array([0.0]),
        np.array([0.0]),
        np.array(["ACIS-S"]),
        np.array([time]),
        np.array([MOCK_RELEASE]),
        table,
    )
    assert dz_rebased[0] == pytest.approx(reference["dz"] - epoch["dz"])
    assert dy_rebased[0] == pytest.approx(reference["dy"] - epoch["dy"])
