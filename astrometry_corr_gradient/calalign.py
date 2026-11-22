"""CALALIGN matrices, from this package's data file, and the rebase that uses them.

Every CALALIGN matrix ever shipped, retired ones included, is copied into
``data/calalign_matrices.ecsv`` (one row per file and detector) by
``scripts/update_calalign_table.py``, so nothing here reads the CALDB.

- ``processing_matrices()``: the matrix each observation was processed with.
  Among its detector's matrices valid at its time, from a CALDB release no
  later than its processing one, the latest release, then the latest start
  (astromon's ``get_calalign_offsets`` rule).
- ``latest_matrices()``: each detector's latest-starting matrix, the rebase
  reference.
- ``rebase()``: each offset moved onto its detector's reference matrix,
  ``dy_rebased = dy - (processing dy - reference dy)``.
- ``current_caldb_matrices()``: the matrix today's CALDB applies at each time
  (the newest release's rule).
"""

from pathlib import Path

import numpy as np
from astropy.table import Table
from cxotime import CxoTime

MATRICES_PATH = Path(__file__).parent / "data" / "calalign_matrices.ecsv"
# each CALALIGN file version's CALDB release (as astromon's calalign_from_files has it)
CALDB_RELEASE_OF_FILE_VERSION = {"N0008": "4.4.4", "N0009": "4.6.2", "N0010": "4.10.0"}


def release_tuple(version: str) -> tuple[int, ...]:
    """A CALDB release as numbers for comparing; a trailing dot is ignored.

    >>> release_tuple("4.9.6.")
    (4, 9, 6)
    """
    return tuple(int(part) for part in str(version).strip().rstrip(".").split("."))


def calalign_table(path: str | Path = MATRICES_PATH) -> Table:
    """The CALALIGN matrices: one row per file and detector, with start and stop in CXC seconds."""
    table = Table.read(path, format="ascii.ecsv")
    # the headers mix ISO and year:day-of-year dates, so each is read on its own
    for name in ("start", "stop"):
        table[f"{name}_secs"] = [CxoTime(str(value)).secs for value in table[name]]
    return table


def processing_matrices(
    detectors: np.ndarray,
    times: np.ndarray,
    caldb_versions: np.ndarray,
    table: Table | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Each observation's processing matrix (dy, dz in arcsec).

    Parameters
    ----------
    detectors : numpy.ndarray
        Each row's detector.
    times : numpy.ndarray
        Each row's time, CXC seconds.
    caldb_versions : numpy.ndarray
        Each row's processing CALDB release, e.g. "4.9.6".
    table : astropy.table.Table, optional
        ``calalign_table()`` by default.

    Raises
    ------
    ValueError
        If a row has no matrix: none valid at its time from a release no later
        than its own.
    """
    table = calalign_table() if table is None else table
    table_detectors = np.asarray(table["detector"]).astype(str)
    starts, stops = np.asarray(table["start_secs"]), np.asarray(table["stop_secs"])
    releases = [release_tuple(version) for version in table["caldb_version"]]
    dy, dz = np.full(len(detectors), np.nan), np.full(len(detectors), np.nan)
    for row, (detector, time, version) in enumerate(
        zip(np.asarray(detectors).astype(str), times, caldb_versions, strict=True)
    ):
        processing = release_tuple(version)
        candidates = [
            index
            for index in np.flatnonzero(
                (table_detectors == detector) & (starts <= time) & (time < stops)
            )
            if releases[index] <= processing
        ]
        if not candidates:
            raise ValueError(
                f"no CALALIGN matrix for {detector} at {CxoTime(time).iso[:10]}"
                f" from a release no later than {version}"
            )
        chosen = max(candidates, key=lambda index: (releases[index], starts[index]))
        dy[row], dz[row] = table["dy"][chosen], table["dz"][chosen]
    return dy, dz


def latest_matrices(table: Table | None = None) -> dict[str, dict]:
    """Each detector's latest-starting matrix, the rebase reference: start date, dy, dz.

    Raises
    ------
    ValueError
        If two of a detector's matrices share its latest start.
    """
    table = calalign_table() if table is None else table
    detectors = np.asarray(table["detector"]).astype(str)
    starts = np.asarray(table["start_secs"])
    reference = {}
    for detector in np.unique(detectors):
        rows = np.flatnonzero(detectors == detector)
        latest = rows[starts[rows] == starts[rows].max()]
        if len(latest) > 1:
            raise ValueError(
                f"{detector}: {len(latest)} matrices share the latest start"
            )
        row = table[latest[0]]
        reference[str(detector)] = {
            "start": CxoTime(float(row["start_secs"])).iso[:10],
            "dy": float(row["dy"]),
            "dz": float(row["dz"]),
        }
    return reference


def rebase(
    dy: np.ndarray,
    dz: np.ndarray,
    detectors: np.ndarray,
    times: np.ndarray,
    caldb_versions: np.ndarray,
    table: Table | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, dict]]:
    """The offsets moved onto each detector's reference matrix, and that reference.

    ``dy_rebased = dy - (processing dy - reference dy)``, the same for dz.
    """
    table = calalign_table() if table is None else table
    processing_dy, processing_dz = processing_matrices(
        detectors, times, caldb_versions, table
    )
    reference = latest_matrices(table)
    detectors = np.asarray(detectors).astype(str)
    reference_dy = np.array([reference[detector]["dy"] for detector in detectors])
    reference_dz = np.array([reference[detector]["dz"] for detector in detectors])
    dy_rebased = np.asarray(dy, float) - (processing_dy - reference_dy)
    dz_rebased = np.asarray(dz, float) - (processing_dz - reference_dz)
    return dy_rebased, dz_rebased, reference


def current_caldb_matrices(
    detectors: np.ndarray, times: np.ndarray, table: Table | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """The matrix today's CALDB applies at each time (dy, dz): the newest release's rule."""
    table = calalign_table() if table is None else table
    newest = max(table["caldb_version"], key=release_tuple)
    return processing_matrices(detectors, times, np.full(len(detectors), newest), table)
