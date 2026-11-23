"""Tests for astrometry_corr_gradient.gradient_term, the gradient term's handoff code and data.

Quarters and the term itself, on small hand-checked examples; then the
embedded data (the coefficients, sigmas, reference table and test vectors) for
consistency. How the numbers are made is tested in test_calibration.py.
"""

import numpy as np
import pytest
from astropy.table import Table
from astropy.time import Time

from astrometry_corr_gradient.calibration import quarter_start
from astrometry_corr_gradient.gradient_term import (
    COEFFICIENTS,
    GRADIENT_MSIDS,
    SIGMAS,
    coefficients_per_sigma,
    gradient_term,
    label_index,
    quarter_index,
    reference_table,
    vector_table,
)

QUARTER_2024Q3 = 2024 * 4 + 2


def one_row_table() -> Table:
    """A reference table with one quarter, 2024Q3."""
    return Table(
        rows=[
            {
                "quarter": "2024Q3",
                "start": "2024-07-01",
                "from_quarter": "2024Q2",
                "observations": 50,
                "OOBAGRD3": 0.1,
                "OOBAGRD6": 0.05,
            }
        ]
    )


HAND_COEFFICIENTS = {
    "dz": {"OOBAGRD3": 2.0, "OOBAGRD6": -10.0},
    "dy": {"OOBAGRD3": 1.0, "OOBAGRD6": 0.0},
}


def test_quarter_index_reads_iso_year_day_and_time():
    assert quarter_index(["2024-07-01 00:00:00", "2024-09-30 23:59:59"]).tolist() == [
        QUARTER_2024Q3,
        QUARTER_2024Q3,
    ]
    assert quarter_index("2024:274:23:59:59").tolist() == [QUARTER_2024Q3]  # 2024-09-30
    assert quarter_index(Time("2024-10-01T00:00:00")).tolist() == [QUARTER_2024Q3 + 1]


def test_gradient_term_by_hand():
    term = gradient_term(
        ["2024-08-01 00:00:00"],
        {"OOBAGRD3": [0.3], "OOBAGRD6": [0.04]},
        one_row_table(),
        HAND_COEFFICIENTS,
    )
    # dz: 2 x (0.3 - 0.1) - 10 x (0.04 - 0.05) = 0.5; dy: 1 x (0.3 - 0.1) = 0.2
    assert term["dz"] == pytest.approx([0.5])
    assert term["dy"] == pytest.approx([0.2])
    assert not term["no_telemetry"][0]
    assert not term["no_reference"][0]


def test_gradient_term_is_zero_and_flagged_without_telemetry_or_a_table_row():
    term = gradient_term(
        ["2024-08-01 00:00:00", "2024-08-02 00:00:00", "2024-12-01 00:00:00"],
        {"OOBAGRD3": [0.3, np.nan, 0.3], "OOBAGRD6": [0.04, 0.04, 0.04]},
        one_row_table(),
        HAND_COEFFICIENTS,
    )
    assert term["dz"] == pytest.approx([0.5, 0.0, 0.0])
    assert term["no_telemetry"].tolist() == [False, True, False]
    assert term["no_reference"].tolist() == [False, False, True]


def test_gradient_term_takes_single_values():
    term = gradient_term(
        "2024:214:00:00:00",  # 2024-08-01
        {"OOBAGRD3": 0.3, "OOBAGRD6": 0.04},
        one_row_table(),
        HAND_COEFFICIENTS,
    )
    assert term["dz"] == pytest.approx([0.5])


def test_gradient_term_needs_both_gradients():
    with pytest.raises(KeyError, match="OOBAGRD6"):
        gradient_term(
            ["2024-08-01 00:00:00"],
            {"OOBAGRD3": [0.3]},
            one_row_table(),
            HAND_COEFFICIENTS,
        )


def test_gradient_term_uses_the_embedded_data_by_default():
    vectors = vector_table()
    first = vectors[0]
    gradients = {name: first[name] for name in GRADIENT_MSIDS}
    default = gradient_term(first["date"], gradients)
    explicit = gradient_term(first["date"], gradients, reference_table(), COEFFICIENTS)
    assert default["dz"] == pytest.approx(explicit["dz"])
    assert default["dy"] == pytest.approx(explicit["dy"])


def test_embedded_reference_table_is_consecutive_quarters_in_kelvin():
    table = reference_table()
    indexes = [label_index(label) for label in table["quarter"]]
    assert indexes == list(range(indexes[0], indexes[0] + len(table)))
    assert list(table["start"]) == [quarter_start(index) for index in indexes]
    sources = [label_index(source) for source in table["from_quarter"]]
    # history uses its own quarter (or an earlier one); the last row, the quarter still filling, lags
    assert all(source <= index for source, index in zip(sources, indexes, strict=True))
    assert sources[-1] < indexes[-1]
    for name in GRADIENT_MSIDS:
        assert str(table[name].unit) == "K"
        assert np.all(np.isfinite(table[name]))


def test_embedded_coefficients_cover_both_axes_and_gradients():
    assert set(COEFFICIENTS) == {"dz", "dy"}
    for by_gradient in COEFFICIENTS.values():
        assert set(by_gradient) == set(GRADIENT_MSIDS)


def test_coefficients_per_sigma_is_each_coefficient_times_its_sigma():
    per_sigma = coefficients_per_sigma(
        HAND_COEFFICIENTS, {"OOBAGRD3": 0.1, "OOBAGRD6": 0.01}
    )
    assert per_sigma["dz"]["OOBAGRD3"] == pytest.approx(0.2)
    assert per_sigma["dz"]["OOBAGRD6"] == pytest.approx(-0.1)
    assert per_sigma["dy"]["OOBAGRD3"] == pytest.approx(0.1)


def test_normalized_form_gives_the_same_term():
    # k x (gradient - reference) / sigma, with k in arcsec per sigma, is the same term
    first = vector_table()[0]
    references = reference_table()
    row = references[list(references["quarter"]).index(first["date"][:4] + "Q2")]
    assert first["date"].startswith("2008-04")  # the first vector is in 2008Q2
    per_sigma = coefficients_per_sigma()
    term = gradient_term(first["date"], {name: first[name] for name in GRADIENT_MSIDS})
    for axis_name in ("dz", "dy"):
        normalized = sum(
            per_sigma[axis_name][name] * (first[name] - row[name]) / SIGMAS[name]
            for name in GRADIENT_MSIDS
        )
        assert term[axis_name][0] == pytest.approx(normalized)


def test_embedded_sigmas_are_positive_for_both_gradients():
    assert set(SIGMAS) == set(GRADIENT_MSIDS)
    assert all(sigma > 0 for sigma in SIGMAS.values())


def test_embedded_test_vectors_reproduce():
    vectors = vector_table()
    term = gradient_term(
        vectors["date"], {name: vectors[name] for name in GRADIENT_MSIDS}
    )
    for axis_name in ("dz", "dy"):
        assert np.allclose(term[axis_name], vectors[axis_name], rtol=0, atol=1e-4)
    reasons = np.where(
        term["no_telemetry"],
        "no telemetry",
        np.where(term["no_reference"], "no table row", "-"),
    )
    assert reasons.tolist() == [str(reason) for reason in vectors["no_term"]]
