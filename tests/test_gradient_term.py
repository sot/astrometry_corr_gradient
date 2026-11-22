"""Tests for astrometry_corr_gradient.gradient_term, the gradient term's handoff code and data.

Quarters, the reference table, the term itself and the one-time coefficient
fit, on small hand-checked examples; then the embedded data (the coefficients,
the reference table and the test vectors) for consistency.
"""

import numpy as np
import pytest
from astropy.table import Table
from astropy.time import Time

from astrometry_corr_gradient.gradient_term import (
    COEFFICIENTS,
    GRADIENT_MSIDS,
    SIGMAS,
    coefficients_per_sigma,
    fit_coefficients,
    gradient_term,
    label_index,
    quarter_index,
    quarter_label,
    quarter_references,
    quarter_start,
    reference_table,
    vector_table,
    within_quarter_sigma,
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


def test_quarter_labels_round_trip():
    assert quarter_label(QUARTER_2024Q3) == "2024Q3"
    assert quarter_start(QUARTER_2024Q3) == "2024-07-01"
    assert label_index("2024Q3") == QUARTER_2024Q3
    assert quarter_start(QUARTER_2024Q3 + 1) == "2024-10-01"


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


def test_quarter_references_uses_the_previous_quarter_and_falls_back():
    # 12 observations in quarter 8000, 3 (too few) in 8001, 12 in 8002; 8002 is the last complete quarter
    quarters = np.array([8000] * 12 + [8001] * 3 + [8002] * 12)
    oobagrd3 = np.array([0.1] * 12 + [5.0] * 3 + [0.3] * 12)
    oobagrd6 = np.array([0.01] * 12 + [5.0] * 3 + [0.03] * 12)
    table = quarter_references(
        quarters,
        {"OOBAGRD3": oobagrd3, "OOBAGRD6": oobagrd6},
        last_complete_quarter=8002,
        lagging_from=8000,  # every quarter lagging
    )
    assert list(table["quarter"]) == [
        quarter_label(index) for index in (8001, 8002, 8003)
    ]
    # 8002 uses 8001's observations, too few, so the latest earlier quarter with enough: 8000
    assert list(table["from_quarter"]) == [
        quarter_label(index) for index in (8000, 8000, 8002)
    ]
    assert list(table["OOBAGRD3"]) == pytest.approx([0.1, 0.1, 0.3])
    assert list(table["observations"]) == [12, 12, 12]


def test_quarter_references_uses_its_own_quarter_before_the_lagging_start():
    # history (before 8003) uses each quarter's own median; from 8003 on, the previous quarter's
    quarters = np.array([8000] * 12 + [8001] * 3 + [8002] * 12)
    oobagrd3 = np.array([0.1] * 12 + [5.0] * 3 + [0.3] * 12)
    table = quarter_references(
        quarters,
        {"OOBAGRD3": oobagrd3, "OOBAGRD6": oobagrd3 / 10},
        last_complete_quarter=8002,
        lagging_from=8003,
    )
    assert list(table["quarter"]) == [
        quarter_label(index) for index in (8000, 8001, 8002, 8003)
    ]
    # 8001 has too few of its own, so the latest earlier quarter with enough: 8000
    assert list(table["from_quarter"]) == [
        quarter_label(index) for index in (8000, 8000, 8002, 8002)
    ]
    assert list(table["OOBAGRD3"]) == pytest.approx([0.1, 0.1, 0.3, 0.3])


def test_quarter_references_switches_to_lagging_at_the_given_quarter():
    quarters = np.array([8000] * 12 + [8001] * 12 + [8002] * 12)
    values = np.array([0.1] * 12 + [0.2] * 12 + [0.3] * 12)
    table = quarter_references(
        quarters,
        {"OOBAGRD3": values, "OOBAGRD6": values},
        last_complete_quarter=8002,
        lagging_from=8002,
    )
    assert list(table["from_quarter"]) == [
        quarter_label(index) for index in (8000, 8001, 8001, 8002)
    ]


def test_quarter_references_ignores_the_quarter_still_filling():
    # the quarter still filling has no median of its own yet, so it lags even without a lagging start
    quarters = np.array([8000] * 12 + [8001] * 12)
    values = np.array([0.1] * 12 + [0.9] * 12)
    table = quarter_references(
        quarters,
        {"OOBAGRD3": values, "OOBAGRD6": values},
        last_complete_quarter=8000,
        lagging_from=None,
    )
    assert list(table["quarter"]) == [quarter_label(8000), quarter_label(8001)]
    assert list(table["OOBAGRD3"]) == pytest.approx([0.1, 0.1])


def test_quarter_references_needs_a_quarter_with_enough_observations():
    with pytest.raises(ValueError, match="observations"):
        quarter_references(
            np.array([8000, 8000]),
            {"OOBAGRD3": np.array([0.1, 0.2]), "OOBAGRD6": np.array([0.1, 0.2])},
            last_complete_quarter=8001,
            lagging_from=None,
        )


def test_fit_coefficients_follows_the_change_inside_quarters_not_the_drift():
    rng = np.random.default_rng(1)
    quarters = np.repeat(np.arange(8000, 8006), 40)
    # the gradients drift a lot between quarters, but the offset only follows their change inside a quarter
    drift = (quarters - 8000).astype(float)
    swing3, swing6 = (
        rng.normal(0, 0.1, quarters.size),
        rng.normal(0, 0.007, quarters.size),
    )
    level = rng.normal(0, 1.0, 6)[quarters - 8000]
    offset = 0.5 * swing3 - 12.0 * swing6 + level
    fitted = fit_coefficients(
        quarters,
        offset,
        {"OOBAGRD3": swing3 + 3.0 * drift, "OOBAGRD6": swing6 - 0.2 * drift},
    )
    assert fitted["OOBAGRD3"] == pytest.approx(0.5)
    assert fitted["OOBAGRD6"] == pytest.approx(-12.0)


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


def test_within_quarter_sigma_by_hand():
    # deviations from each quarter's mean: -1, 1 and -2, 2; 10 / (4 values - 2 quarters) = 5
    quarters = np.array([0, 0, 1, 1, 1])
    values = np.array([1.0, 3.0, 10.0, 14.0, np.nan])
    assert within_quarter_sigma(quarters, values) == pytest.approx(np.sqrt(5.0))


def test_within_quarter_sigma_ignores_drift_between_quarters():
    quarters = np.repeat(np.arange(4), 3)
    swing = np.tile([-0.1, 0.0, 0.1], 4)
    assert within_quarter_sigma(quarters, swing + 5.0 * quarters) == pytest.approx(
        within_quarter_sigma(quarters, swing)
    )


def test_within_quarter_sigma_needs_more_values_than_quarters():
    with pytest.raises(ValueError, match="sigma"):
        within_quarter_sigma(np.array([0, 1]), np.array([0.1, 0.2]))


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
