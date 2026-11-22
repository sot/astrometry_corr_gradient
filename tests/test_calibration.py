"""Tests for astrometry_corr_gradient.calibration: from the astromon database to the gradient term's numbers.

The database, kadi and the telemetry archive are outside these tests; each step
takes their output as an argument, so small made-up tables stand in for them.
"""

import numpy as np
import pytest
from astropy.table import Table, vstack
from cxotime import CxoTime

from astrometry_corr_gradient.calibration import (
    cached_series,
    calibration_set,
    dwell_means,
    solve,
)
from astrometry_corr_gradient.gradient_term import (
    fit_coefficients,
    quarter_index,
    quarter_references,
    within_quarter_sigma,
)


def made_up_matches() -> Table:
    """Five sources in four observations; obsid 3's two sources disagree by 2", obsid 4 has no dwell."""
    times = CxoTime(
        [
            "2008-05-01T00:00:00",
            "2008-02-01T00:00:00",
            "2008-02-01T00:00:00",
            "2008-08-01T00:00:00",
            "2008-08-01T00:00:00",
            "2008-09-01T00:00:00",
        ]
    )
    table = Table(
        {
            "obsid": [2, 1, 1, 3, 3, 4],
            "time": times,
            "detector": ["ACIS-S", "ACIS-I", "ACIS-I", "HRC-S", "HRC-S", "ACIS-S"],
            "target": [" B ", "A", "A", "C", "C", "D"],
            "catalog": ["Gaia", "Gaia", "Tycho", "Gaia", "Gaia", "Gaia"],
            "name": ["s2", "s1", "s9", "s3", "s4", "s5"],
            "dz_rebased": [0.1, 0.2, 0.25, 0.0, 2.0, 0.3],
            "dy_rebased": [-0.1, 0.0, 0.05, 0.0, 0.0, 0.1],
        }
    )
    table.meta["calalign_reference"] = {
        "ACIS-S": {"dz": 1.0, "dy": 2.0, "start": "2021-07-02"}
    }
    return table


def made_up_windows() -> dict[int, tuple[float, float]]:
    """Dwells of 1000 s for obsids 1 to 3 (none for 4)."""
    starts = {
        1: CxoTime("2008-02-01T00:00:00").secs,
        2: CxoTime("2008-05-01T00:00:00").secs,
        3: CxoTime("2008-08-01T00:00:00").secs,
    }
    return {obsid: (start, start + 1000.0) for obsid, start in starts.items()}


def made_up_series(level: float) -> tuple[np.ndarray, np.ndarray]:
    """A 5-minute series over each made-up dwell: the obsid's number times 0.1, plus level."""
    times, values = [], []
    for obsid, (start, stop) in made_up_windows().items():
        samples = np.arange(start, stop, 300.0)
        times.append(samples)
        values.append(np.full(samples.size, 0.1 * obsid + level))
    return np.concatenate(times), np.concatenate(values)


NO_BAD_INTERVALS = Table(
    {"start": np.array([], dtype=str), "stop": np.array([], dtype=str)}
)


def test_cached_series_reads_the_cache_file(tmp_path):
    np.savez(
        tmp_path / "OOBAGRD3_5min.npz",
        times=np.array([1.0, 2.0]),
        vals=np.array([0.3, 0.4]),
    )
    times, values = cached_series(tmp_path, "OOBAGRD3")
    assert times.tolist() == [1.0, 2.0]
    assert values.tolist() == [0.3, 0.4]


def test_dwell_means_average_each_dwell_and_mask_bad_telemetry():
    times, values = made_up_series(0.0)
    windows = made_up_windows()
    # obsid 2's whole dwell is inside a bad-telemetry interval
    start, stop = windows[2]
    bad = Table(
        {
            "start": [CxoTime(start - 1.0).isot],
            "stop": [CxoTime(stop + 1.0).isot],
        }
    )
    means = dwell_means(times, values, windows, np.array([1, 2, 3]), bad)
    assert means[0] == pytest.approx(0.1)
    assert np.isnan(means[1])
    assert means[2] == pytest.approx(0.3)


def test_calibration_set_keeps_one_dwell_drops_disagreeing_observations_and_sorts():
    series = {"OOBAGRD3": made_up_series(0.0), "OOBAGRD6": made_up_series(0.05)}
    table = calibration_set(
        made_up_matches(), made_up_windows(), series, NO_BAD_INTERVALS
    )
    # obsid 4 has no dwell; obsid 3's sources disagree by 2"; the rest in time order
    assert table["obsid"].tolist() == [1, 1, 2]
    assert table["date"].tolist() == [
        "2008-02-01 00:00:00",
        "2008-02-01 00:00:00",
        "2008-05-01 00:00:00",
    ]
    assert table["target"].tolist() == ["A", "A", "B"]
    assert table["OOBAGRD3"].tolist() == pytest.approx([0.1, 0.1, 0.2])
    assert table["OOBAGRD6"].tolist() == pytest.approx([0.15, 0.15, 0.25])
    assert table["year"][0] == pytest.approx(CxoTime("2008-02-01T00:00:00").decimalyear)
    assert table.meta["calalign_reference"]["ACIS-S"]["start"] == "2021-07-02"


def test_calibration_set_links_sources_of_one_observation():
    series = {"OOBAGRD3": made_up_series(0.0), "OOBAGRD6": made_up_series(0.05)}
    table = calibration_set(
        made_up_matches(), made_up_windows(), series, NO_BAD_INTERVALS
    )
    groups = table["linked_group"]
    assert groups[0] == groups[1]  # obsid 1's two sources
    assert groups[0] != groups[2]


def made_up_calibration() -> Table:
    """30 observations in each of 2008Q1 to 2008Q3, one source each."""
    rng = np.random.default_rng(3)
    months = np.repeat([2, 5, 8], 30)
    days = np.tile(np.arange(1, 31) % 28 + 1, 3)
    oobagrd3 = rng.normal(0.6, 0.1, 90)
    oobagrd6 = rng.normal(0.05, 0.007, 90)
    return Table(
        {
            "obsid": np.arange(90),
            "date": [
                f"2008-{month:02d}-{day:02d} 12:00:00"
                for month, day in zip(months, days, strict=True)
            ],
            "dz_rebased": 0.5 * oobagrd3 - 12.0 * oobagrd6 + rng.normal(0, 0.01, 90),
            "dy_rebased": 1.5 * oobagrd3 - 10.0 * oobagrd6 + rng.normal(0, 0.01, 90),
            "OOBAGRD3": oobagrd3,
            "OOBAGRD6": oobagrd6,
        }
    )


def test_solve_uses_the_gradient_term_functions():
    table = made_up_calibration()
    quarters = quarter_index(table["date"])
    gradients = {name: np.asarray(table[name]) for name in ("OOBAGRD3", "OOBAGRD6")}
    coefficients, sigmas, references = solve(
        table, last_complete_quarter=2008 * 4 + 2, lagging_from=2008 * 4 + 3
    )
    assert coefficients["dz"] == pytest.approx(
        fit_coefficients(quarters, np.asarray(table["dz_rebased"]), gradients)
    )
    assert sigmas["OOBAGRD6"] == pytest.approx(
        within_quarter_sigma(quarters, gradients["OOBAGRD6"])
    )
    expected = quarter_references(
        quarters,
        gradients,
        last_complete_quarter=2008 * 4 + 2,
        lagging_from=2008 * 4 + 3,
    )
    assert list(references["quarter"]) == list(expected["quarter"])
    assert list(references["OOBAGRD3"]) == pytest.approx(list(expected["OOBAGRD3"]))


def test_solve_takes_one_value_per_observation_for_the_references():
    # a second source of observation 0 shares its gradients and must not count twice
    table = made_up_calibration()
    with_second_source = vstack([table, table[:1]])
    _, _, once = solve(table, last_complete_quarter=2008 * 4 + 2, lagging_from=None)
    _, _, twice = solve(
        with_second_source, last_complete_quarter=2008 * 4 + 2, lagging_from=None
    )
    assert list(once["observations"]) == list(twice["observations"])
    assert list(once["OOBAGRD3"]) == pytest.approx(list(twice["OOBAGRD3"]))


def test_calibration_set_carries_what_todays_caldb_adds():
    from astrometry_corr_gradient.calalign import (
        current_caldb_matrices,
        latest_matrices,
    )

    series = {"OOBAGRD3": made_up_series(0.0), "OOBAGRD6": made_up_series(0.05)}
    table = calibration_set(
        made_up_matches(), made_up_windows(), series, NO_BAD_INTERVALS
    )
    detectors = np.asarray(table["detector"])
    current_dy, current_dz = current_caldb_matrices(
        detectors, CxoTime(table["date"]).secs
    )
    reference = latest_matrices()
    expected_dz = current_dz - np.array([reference[d]["dz"] for d in detectors])
    expected_dy = current_dy - np.array([reference[d]["dy"] for d in detectors])
    assert np.allclose(table["dz_caldb_shift"], expected_dz, rtol=0, atol=1e-9)
    assert np.allclose(table["dy_caldb_shift"], expected_dy, rtol=0, atol=1e-9)
    # in 2008 today's CALDB still applies an N0009 matrix, not the 2021-07-02 one
    assert np.all(np.abs(table["dz_caldb_shift"]) > 0)
