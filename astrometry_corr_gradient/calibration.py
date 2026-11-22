"""The gradient term's calibration: from an astromon database to its coefficients and references.

Four steps, each a function:

1. ``matched_sources()``: the X-ray to catalog cross matches from the astromon
   database, each rebased onto its detector's latest CALALIGN matrix (from this
   package's copy of the matrices), with the calibration's cuts (catalogs,
   detection quality, excluded targets, nominal SIM-Z), from 2004 on.
2. ``dwell_windows()``: each observation's dwell, from kadi's maneuvers: from
   its Kalman start to the next maneuver's start. An observation with more
   than one dwell is left out.
3. ``calibration_set()``: the matched sources with one dwell, without the
   observations whose sources disagree (a bad cross match), with each
   observation's OOBAGRD3 and OOBAGRD6 averaged over its dwell, bad telemetry
   masked, and what today's CALDB adds to each offset. One row per source, in
   time order.
4. ``solve()``: the coefficients, sigmas and reference table, with
   ``astrometry_corr_gradient.gradient_term``'s functions.

The only input is the astromon database. The telemetry comes from cheta
(``gradient_series()``) and the safing intervals from kadi and cheta
(``bad_intervals()``), each cached on disk; ``cached_series()`` and
``read_bad_intervals()`` read those caches without touching the archive.
"""

from pathlib import Path

import numpy as np
from astropy.table import Table

from astrometry_corr_gradient.calalign import current_caldb_matrices, latest_matrices
from astrometry_corr_gradient.crossmatch import inconsistent_obsid_mask
from astrometry_corr_gradient.gradient_term import (
    AXES,
    GRADIENT_MSIDS,
    fit_coefficients,
    quarter_index,
    quarter_references,
    within_quarter_sigma,
)
from astrometry_corr_gradient.groups import linked_group_ids
from astrometry_corr_gradient.telemetry import (
    align_msid_to_windows,
    cached_msid_fetch,
    get_bad_telemetry_intervals_cached,
    group_obsid_windows,
    mask_bad_telemetry_intervals,
)

FIRST_YEAR = 2004  # the calibration starts here
SAME_OBSID_THRESHOLD = 0.75  # arcsec; see crossmatch.inconsistent_obsid_mask
COLUMN_DESCRIPTIONS = {
    "obsid": "the observation",
    "date": "the observation's time (UTC)",
    "year": "the same time as a decimal year",
    "detector": "ACIS-S, ACIS-I, HRC-S or HRC-I",
    "target": "the observation's target, as astromon names it",
    "linked_group": "sources linked by a shared observation or catalog counterpart",
    "dz_rebased": "X-ray minus catalog position in dz, with the detector's latest CALALIGN matrix",
    "dy_rebased": "the same in dy",
    "OOBAGRD3": "periscope gradient RT 505 to RT 504, mean over the dwell, bad telemetry masked",
    "OOBAGRD6": "periscope gradient RT 507 to RT 506, mean over the dwell, bad telemetry masked",
    "dz_caldb_shift": "what today's CALDB adds to dz_rebased: its matrix at this time minus the latest",
    "dy_caldb_shift": "the same in dy",
}


def matched_sources(database: str | Path) -> Table:
    """Step 1: the cross matches from an astromon database, rebased, from 2004 on."""
    from astrometry_corr_gradient.crossmatch import (
        CALIBRATION_CATALOG_HIERARCHY,
        fetch_crossmatches,
    )

    matches = fetch_crossmatches(str(database), catalogs=CALIBRATION_CATALOG_HIERARCHY)
    year = np.array([int(date[:4]) for date in matches["date_obs"]])
    return matches[year >= FIRST_YEAR]


def dwell_windows(start: str, stop: str) -> dict[int, tuple[float, float]]:
    """Step 2: each observation's dwell (CXC seconds), from kadi's maneuvers; one dwell per observation."""
    from cxotime import CxoTime
    from kadi import events

    rows = list(
        events.manvrs.filter(start=start, stop=stop).values(
            "obsid", "kalman_start", "next_manvr_start"
        )
    )
    maneuvers = Table(rows)

    def seconds(value: object) -> float:
        return np.nan if value is None else CxoTime(value).secs

    maneuvers["kalman_start_secs"] = [
        seconds(value) for value in maneuvers["kalman_start"]
    ]
    maneuvers["next_manvr_start_secs"] = [
        seconds(value) for value in maneuvers["next_manvr_start"]
    ]
    return group_obsid_windows(maneuvers)


def gradient_series(
    msid: str, start: str, stop: str, cache_dir: str | Path
) -> tuple[np.ndarray, np.ndarray]:
    """An MSID's 5-minute means from cheta, cached in ``cache_dir`` (only new telemetry is fetched)."""

    def from_archive(name: str, first: str, last: str) -> tuple[np.ndarray, np.ndarray]:
        from cheta import fetch

        data = fetch.Msid(name, first, last, stat="5min")
        return data.times, data.vals

    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    return cached_msid_fetch(from_archive, cache_dir)(msid, start, stop)


def cached_series(cache_dir: str | Path, msid: str) -> tuple[np.ndarray, np.ndarray]:
    """An MSID's 5-minute series from the telemetry cache only (never the archive): times (CXC seconds), values."""
    cached = np.load(Path(cache_dir) / f"{msid}_5min.npz")
    return np.asarray(cached["times"]), np.asarray(cached["vals"])


def bad_intervals(
    start: str, stop: str, cache_path: str | Path, max_age_days: float = 30.0
) -> Table:
    """The safing intervals to mask, from kadi and cheta, cached at ``cache_path`` for ``max_age_days``."""
    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
    return get_bad_telemetry_intervals_cached(
        start, stop, cache_path, max_age_days=max_age_days
    )


def read_bad_intervals(path: str | Path) -> Table:
    """The safing intervals to mask, as saved (read only; never refreshed)."""
    return Table.read(path, format="ascii.ecsv")


def dwell_means(
    times: np.ndarray,
    values: np.ndarray,
    windows: dict[int, tuple[float, float]],
    obsids: np.ndarray,
    bad_intervals: Table,
) -> np.ndarray:
    """Each obsid's mean of the series over its dwell, with bad-telemetry intervals masked first."""
    masked = mask_bad_telemetry_intervals(times, values, bad_intervals)
    means = align_msid_to_windows(times, masked, windows)
    return np.array([means[int(obsid)] for obsid in obsids], dtype=float)


def calibration_set(
    matches: Table,
    windows: dict[int, tuple[float, float]],
    series: dict[str, tuple[np.ndarray, np.ndarray]],
    bad_intervals: Table,
    same_obsid_threshold: float = SAME_OBSID_THRESHOLD,
) -> Table:
    """Step 3: one row per matched source, ready for ``solve()``.

    Parameters
    ----------
    matches : astropy.table.Table
        Step 1's cross matches: ``obsid``, ``time``, ``detector``, ``target``,
        ``catalog``, ``name``, ``dz_rebased``, ``dy_rebased``.
    windows : dict
        Step 2's dwells, obsid -> (start, stop) in CXC seconds.
    series : dict
        OOBAGRD3 and OOBAGRD6, each (times, values).
    bad_intervals : astropy.table.Table
        The safing intervals to mask.
    same_obsid_threshold : float
        Arcsec; an observation whose sources disagree by more is left out.

    Returns
    -------
    astropy.table.Table
        ``obsid``, ``date`` (UTC), ``year`` (decimal), ``detector``,
        ``target``, ``linked_group`` (sources linked by an observation or a
        catalog counterpart), ``dz_rebased``, ``dy_rebased`` (arcsec), each
        gradient's dwell mean (K), and ``dz_caldb_shift``, ``dy_caldb_shift``
        (what today's CALDB adds to the rebased offsets), in time order.
    """
    matched = set(np.asarray(matches["obsid"]).tolist())
    windows = {obsid: windows[obsid] for obsid in matched if obsid in windows}
    matches = matches[np.isin(matches["obsid"], list(windows))]
    disagree = inconsistent_obsid_mask(
        np.asarray(matches["obsid"]),
        np.asarray(matches["dy_rebased"]),
        np.asarray(matches["dz_rebased"]),
        threshold=same_obsid_threshold,
    )
    matches = matches[~disagree]
    matches.sort("time")
    obsids = np.asarray(matches["obsid"])
    detectors = np.asarray(matches["detector"]).astype(str)
    source_names = np.array(
        [
            f"{catalog}|{name}"
            for catalog, name in zip(matches["catalog"], matches["name"], strict=True)
        ]
    )
    table = Table(
        {
            "obsid": obsids,
            "date": np.array([date[:19] for date in matches["time"].iso]),
            "year": np.asarray(matches["time"].decimalyear, float),
            "detector": detectors,
            "target": np.char.strip(np.asarray(matches["target"]).astype(str)),
            "linked_group": linked_group_ids(obsids, source_names),
            "dz_rebased": np.asarray(matches["dz_rebased"], float),
            "dy_rebased": np.asarray(matches["dy_rebased"], float),
        }
    )
    for name in GRADIENT_MSIDS:
        times, values = series[name]
        table[name] = dwell_means(times, values, windows, obsids, bad_intervals)
        table[name].unit = "K"
    current_dy, current_dz = current_caldb_matrices(
        detectors, np.asarray(matches["time"].secs)
    )
    reference = latest_matrices()
    for axis_name, current in (("dz", current_dz), ("dy", current_dy)):
        latest = np.array([reference[detector][axis_name] for detector in detectors])
        table[f"{axis_name}_caldb_shift"] = current - latest
    for axis_name in AXES:
        table[f"{axis_name}_rebased"].unit = "arcsec"
        table[f"{axis_name}_caldb_shift"].unit = "arcsec"
    for name, description in COLUMN_DESCRIPTIONS.items():
        table[name].description = description
    table.meta["calalign_reference"] = matches.meta.get("calalign_reference", reference)
    return table


def solve(
    calibration: Table, last_complete_quarter: int, lagging_from: int | None
) -> tuple[dict, dict, Table]:
    """Step 4: the coefficients (arcsec per K), sigmas (K) and reference table, unrounded.

    The coefficients and sigmas use every source with both gradients; the
    references use one value per observation (its sources share its gradients).
    ``last_complete_quarter`` and ``lagging_from`` are as for
    ``gradient_term.quarter_references``.
    """
    quarters = quarter_index(np.asarray(calibration["date"]).astype(str))
    gradients = {name: np.asarray(calibration[name], float) for name in GRADIENT_MSIDS}
    coefficients = {
        axis_name: fit_coefficients(
            quarters, np.asarray(calibration[f"{axis_name}_rebased"], float), gradients
        )
        for axis_name in AXES
    }
    both = np.all([np.isfinite(column) for column in gradients.values()], axis=0)
    sigmas = {
        name: within_quarter_sigma(quarters[both], column[both])
        for name, column in gradients.items()
    }
    _, first_rows = np.unique(np.asarray(calibration["obsid"]), return_index=True)
    references = quarter_references(
        quarters[first_rows],
        {name: column[first_rows] for name, column in gradients.items()},
        last_complete_quarter=last_complete_quarter,
        lagging_from=lagging_from,
    )
    return coefficients, sigmas, references
