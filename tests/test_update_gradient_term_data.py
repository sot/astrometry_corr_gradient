"""Tests for scripts/update_gradient_term_data.py, which writes the data section of astrometry_corr_gradient/gradient_term.py.

A small made-up export (three quarters of observations, one without telemetry
and one before the table starts) goes through handoff_data and data_block, and
the written section reads back as the same data.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from astropy.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from update_gradient_term_data import data_block, handoff_data

from astrometry_corr_gradient import gradient_term


def made_up_export() -> Table:
    """12 observations in each of 2008Q1 to 2008Q3, one more in 2008Q2 without telemetry; made in 2008Q4."""
    rng = np.random.default_rng(2)
    months = np.repeat([2, 5, 8], 12)
    days = np.tile(np.arange(1, 13), 3)
    dates = [
        f"2008-{month:02d}-{day:02d} 12:00:00"
        for month, day in zip(months, days, strict=True)
    ]
    oobagrd3 = rng.normal(0.6, 0.1, 36)
    oobagrd6 = rng.normal(0.05, 0.007, 36)
    dz = 0.5 * oobagrd3 - 12.0 * oobagrd6 + rng.normal(0, 0.01, 36)
    dy = 1.5 * oobagrd3 - 10.0 * oobagrd6 + rng.normal(0, 0.01, 36)
    table = Table(
        {
            "obsid": np.arange(1000, 1036),
            "date": dates,
            "dz_rebased": dz,
            "dy_rebased": dy,
            "OOBAGRD3": oobagrd3,
            "OOBAGRD6": oobagrd6,
        }
    )
    table.add_row([2000, "2008-05-20 12:00:00", 0.1, 0.2, np.nan, np.nan])
    table.meta["made"] = "2008-11-08 23:07 UTC"
    return table


def test_handoff_data_covers_the_quarter_now_filling_and_its_vectors_reproduce():
    coefficients, sigmas, references, vectors = handoff_data(
        made_up_export(), test_quarters=("2008Q3",), lagging_from="2008Q4"
    )
    # complete quarters use their own median; the quarter now filling lags
    assert list(references["quarter"]) == ["2008Q1", "2008Q2", "2008Q3", "2008Q4"]
    assert list(references["from_quarter"]) == ["2008Q1", "2008Q2", "2008Q3", "2008Q3"]
    assert coefficients["dz"]["OOBAGRD3"] == pytest.approx(0.5, abs=0.1)
    # the made-up gradients swing by 0.1 and 0.007 K inside each quarter
    assert sigmas["OOBAGRD3"] == pytest.approx(0.1, abs=0.04)
    assert sigmas["OOBAGRD6"] == pytest.approx(0.007, abs=0.003)
    assert list(vectors["no_term"]) == ["-", "no telemetry"]
    term = gradient_term.gradient_term(
        vectors["date"],
        {name: vectors[name] for name in gradient_term.GRADIENT_MSIDS},
        references,
        coefficients,
    )
    for axis_name in gradient_term.AXES:
        assert np.allclose(term[axis_name], vectors[axis_name], rtol=0, atol=1e-4)


def test_data_block_reads_back_as_the_same_data():
    coefficients, sigmas, references, vectors = handoff_data(
        made_up_export(), test_quarters=("2008Q3",), lagging_from="2008Q4"
    )
    namespace: dict = {}
    block = data_block(coefficients, sigmas, references, vectors, "made up")
    exec(block, namespace)  # noqa: S102 -- the block under test
    assert namespace["COEFFICIENTS"] == coefficients
    assert namespace["SIGMAS"] == sigmas
    written = Table.read(namespace["REFERENCE_CSV"], format="ascii.csv")
    assert list(written["quarter"]) == list(references["quarter"])
    assert np.allclose(written["OOBAGRD3"], references["OOBAGRD3"], rtol=0, atol=1e-5)
    written_vectors = Table.read(namespace["TEST_VECTOR_CSV"], format="ascii.csv")
    assert list(written_vectors["obsid"]) == list(vectors["obsid"])
    assert np.allclose(written_vectors["dz"], vectors["dz"], rtol=0, atol=1e-4)
