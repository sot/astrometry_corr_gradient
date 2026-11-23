"""Mock N0011: the CALALIGN matrices the plan would ship, made from the calibration set.

One matrix per calendar quarter and detector from 2021-07-02, where today's
CALDB stops. Each is set so that observations processed with it have their
quarter's median offset at zero, the old way: medians of the rebased offsets,
made without the gradient term. History (complete quarters before the release)
uses each quarter's own median; from the release on, and for the quarter still
filling, the previous quarter's. One constant per detector, as for the
stand-in in the validation; none for HRC-I in dy.

For an epoch's quarter level L and a detector's constant C, the matrix is
``reference - (L + C)`` on each axis, where the reference is the detector's
latest shipped matrix (the one the offsets are rebased onto), so
``rebased + (matrix - reference) = rebased - L - C``.

These are a mock: version N0011 with a placeholder CALDB release,
``MOCK_RELEASE``, later than any shipped one. They are kept apart from the
shipped matrices in ``data/calalign_matrices.ecsv``.
"""

import numpy as np
from astropy.table import Table
from cxotime import CxoTime

from astrometry_corr_gradient.calalign import latest_matrices
from astrometry_corr_gradient.gradient_term import (
    AXES,
    MIN_OBSERVATIONS,
    quarter_index,
    quarter_label,
    quarter_start,
)

MOCK_RELEASE = "4.13.0"  # a placeholder, later than every shipped release
FIRST_EPOCH = "2021-07-02"  # today's CALDB has no matrix after this
NO_STOP = "2050:001:00:00:00.00"
EXEMPT = {"dz": (), "dy": ("HRC-I",)}  # detectors that get no constant


def quarter_levels(
    calibration: Table,
    last_complete_quarter: int,
    lagging_from: int | None,
    min_sources: int = MIN_OBSERVATIONS,
) -> dict[str, dict[int, float]]:
    """Each quarter's level on each axis: the median rebased offset of every source in its source quarter.

    A quarter before ``lagging_from`` uses its own sources; from ``lagging_from``
    on, and for the quarter still filling, the latest earlier complete quarter
    with ``min_sources`` sources. Quarters run from the first with sources
    through the quarter after ``last_complete_quarter``.
    """
    quarters = quarter_index(np.asarray(calibration["date"]).astype(str))
    levels = {}
    for axis_name in AXES:
        offset = np.asarray(calibration[f"{axis_name}_rebased"], float)
        finite = np.isfinite(offset)
        medians = {
            int(index): float(np.median(offset[finite & (quarters == index)]))
            for index in np.unique(quarters[finite])
            if index <= last_complete_quarter
            and np.sum(finite & (quarters == index)) >= min_sources
        }
        sources = np.array(sorted(medians))
        by_quarter = {}
        for index in range(int(quarters.min()), last_complete_quarter + 2):
            own = (lagging_from is None or index < lagging_from) and index in medians
            earlier = sources[sources < index]
            if own:
                by_quarter[index] = medians[index]
            elif earlier.size:
                by_quarter[index] = medians[int(earlier.max())]
        levels[axis_name] = by_quarter
    return levels


def detector_constants(
    calibration: Table, levels: dict[str, dict[int, float]]
) -> dict[str, dict[str, float]]:
    """Each detector's median of what its quarter's level leaves, over every source with a level; HRC-I none in dy."""
    quarters = quarter_index(np.asarray(calibration["date"]).astype(str))
    detectors = np.asarray(calibration["detector"]).astype(str)
    constants = {}
    for axis_name in AXES:
        level = np.array(
            [levels[axis_name].get(int(index), np.nan) for index in quarters]
        )
        left = np.asarray(calibration[f"{axis_name}_rebased"], float) - level
        constants[axis_name] = {
            str(detector): 0.0
            if detector in EXEMPT[axis_name]
            else float(np.median(left[np.isfinite(left) & (detectors == detector)]))
            for detector in np.unique(detectors)
        }
    return constants


def mock_matrices(
    calibration: Table,
    last_complete_quarter: int,
    lagging_from: int | None,
    first_epoch: str = FIRST_EPOCH,
) -> Table:
    """The mock N0011 matrices, in ``calalign.calalign_table()``'s form: one row per quarter and detector.

    Epochs run from ``first_epoch`` (its quarter's first epoch starts there) to
    the quarter after ``last_complete_quarter``, whose epoch stays valid until a
    newer one; a quarter with no level has no epoch.
    """
    levels = quarter_levels(calibration, last_complete_quarter, lagging_from)
    constants = detector_constants(calibration, levels)
    reference = latest_matrices()
    first_quarter = int(quarter_index(first_epoch)[0])
    epochs = [
        index
        for index in range(first_quarter, last_complete_quarter + 2)
        if all(index in levels[axis_name] for axis_name in AXES)
    ]
    rows = []
    for position, index in enumerate(epochs):
        start = first_epoch if index == first_quarter else quarter_start(index)
        stop = (
            quarter_start(epochs[position + 1])
            if position + 1 < len(epochs)
            else NO_STOP
        )
        for detector in sorted(reference):
            rows.append(
                {
                    "file": f"mock N0011 {quarter_label(index)}",
                    "version": "N0011",
                    "caldb_version": MOCK_RELEASE,
                    "detector": detector,
                    "start": start,
                    "stop": stop,
                    **{
                        axis_name: reference[detector][axis_name]
                        - (
                            levels[axis_name][index]
                            + constants[axis_name].get(detector, 0.0)
                        )
                        for axis_name in ("dy", "dz")
                    },
                }
            )
    table = Table(rows=rows)
    for name in ("start", "stop"):
        table[f"{name}_secs"] = [CxoTime(str(value)).secs for value in table[name]]
    table["dy"].unit = table["dz"].unit = "arcsec"
    return table
