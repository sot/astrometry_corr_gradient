"""Apply the gradient term to the fid centroids, as aca_corr_fid does the periscope correction.

For testing in the aspect pipeline: run it through ``aca_corr_fid`` (the
relative, mean-centered periscope correction), stop, shift the fid centroids
in the ACACENT file by this observation's gradient term, then run the rest of
the pipeline. ``aca_corr_fid`` (ds15, ``tool_functions.c``) corrects each good
fid slot's ``ang_y_sm`` and ``ang_z_sm`` by::

    ang_y_sm -= peri_y_grd3 * (G3(t) - mean G3) + peri_y_grd6 * (G6(t) - mean G6)

The gradient term is the part that mean-centering leaves out, measured
against the quarter's reference instead of the observation's own mean. It is
one constant per observation, subtracted from every row of every good fid
slot::

    ang_y_sm -= (c_dy3 * (mean G3 - ref3) + c_dy6 * (mean G6 - ref6)) / 3600
    ang_z_sm -= (c_dz3 * (mean G3 - ref3) + c_dz6 * (mean G6 - ref6)) / 3600

with the coefficients and references of ``gradient_term``. That dy goes with
ang_y and dz with ang_z, with the same sign, is read from the periscope
coefficients (CALALIGN's PERIFIDCORR), which have the same signs as the
gradient term's on all four; ``predicted_event_shift()`` checks it on the
pipeline's output.

Mirrors ``aca_corr_fid``:

- Good fid slots: FIDPROPS rows whose ``id_status`` starts with GOOD.
- The time range: the first to the last ACACENT row of a good fid slot with
  status DATA_NORMAL and the EGAUSS algorithm.
- The mean gradients: OBC engineering samples in that range (inclusive);
  samples with bad quality are filled by linear interpolation over sample
  index (the nearest good value at either end), then all are averaged.

Assumed, not checked against a real file here: each gradient's quality is
the QUALITY bit at that column's position (the CXC level-0 convention).
Compare the means this prints with aca_corr_fid's verbose "Mean Gradient 3"
and "Mean Gradient 6".
"""

import logging
from pathlib import Path

import numpy as np
from astropy.io import fits

from astrometry_corr_gradient import gradient_term

LOGGER = logging.getLogger(__name__)

DATA_NORMAL = 0  # ACACENT status of a normal record (ds15 aspect_lib/acacent.h)
EGAUSS = 8  # ACACENT alg of the elliptical Gaussian centroids, aca_corr_fid's default
# How far events move in y (z) angle, arcsec, per mm added to the asol dy (dz),
# measured with acis_process_events (CIAO 4.17) on obsid 23059
ARCSEC_PER_MM_DYDZ = -20.484

GradientSamples = tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]


def good_fid_slots(fidprops: str | Path) -> list[int]:
    """The slots of the FIDPROPS rows whose id_status starts with GOOD."""
    with fits.open(fidprops) as hdus:
        data = hdus["FIDPROPS"].data
        return sorted(
            int(slot)
            for slot, status in zip(data["slot"], data["id_status"], strict=True)
            if str(status).strip().upper().startswith("GOOD")
        )


def fid_time_range(
    acacent: str | Path, slots: list[int], algorithm: int = EGAUSS
) -> tuple[float, float]:
    """First and last time (CXC seconds) of the usable fid rows.

    Usable: a slot in ``slots``, status DATA_NORMAL, alg ``algorithm``.
    """
    with fits.open(acacent) as hdus:
        data = hdus["ACACENT"].data
        usable = (
            np.isin(data["slot"], slots)
            & (data["status"] == DATA_NORMAL)
            & (data["alg"] == algorithm)
        )
        if not usable.any():
            raise ValueError(f"{acacent}: no normal fid centroids in slots {slots}")
        times = np.asarray(data["time"][usable], float)
    return float(times.min()), float(times.max())


def read_obc_gradients(paths: list[str | Path]) -> GradientSamples:
    """Times, OOBAGRD3 and OOBAGRD6, and their good-quality flags, in time order.

    From OBC engineering files (obcf*_eng0).
    """
    times, gradients, good = (
        [],
        {name: [] for name in gradient_term.GRADIENT_MSIDS},
        {name: [] for name in gradient_term.GRADIENT_MSIDS},
    )
    for path in paths:
        with fits.open(path) as hdus:
            table = hdus[1]
            names = [name.upper() for name in table.columns.names]
            quality = np.asarray(table.data["QUALITY"], bool)
            times.append(np.asarray(table.data["TIME"], float))
            for name in gradient_term.GRADIENT_MSIDS:
                gradients[name].append(np.asarray(table.data[name], float))
                good[name].append(~quality[:, names.index(name)])
    order = np.argsort(np.concatenate(times), kind="stable")
    return (
        np.concatenate(times)[order],
        {name: np.concatenate(values)[order] for name, values in gradients.items()},
        {name: np.concatenate(flags)[order] for name, flags in good.items()},
    )


def mean_gradients(
    samples: GradientSamples, start: float, stop: float
) -> dict[str, float]:
    """Each gradient's mean from ``start`` to ``stop`` (inclusive).

    Bad samples are filled as aca_corr_fid fills them.
    """
    times, gradients, good = samples
    in_range = (times >= start) & (times <= stop)
    means = {}
    for name in gradient_term.GRADIENT_MSIDS:
        values, flags = gradients[name][in_range], good[name][in_range]
        if not flags.any():
            raise ValueError(f"no good {name} samples from {start} to {stop}")
        index = np.arange(len(values))
        filled = np.interp(index, index[flags], values[flags])
        means[name] = float(np.mean(filled))
    return means


def fid_shift(date: str, means: dict[str, float]) -> dict:
    """This observation's gradient term (arcsec) and the matching fid shift (deg).

    Raises
    ------
    ValueError
        If the reference table has no row for the date's quarter, or a mean is missing.
    """
    term = gradient_term.gradient_term(date, means)
    if term["no_reference"][0]:
        raise ValueError(f"the reference table has no row for {date}")
    if term["no_telemetry"][0]:
        raise ValueError(f"missing gradient means: {means}")
    quarter = gradient_term.quarter_index(date)[0]
    return {
        "dy_arcsec": float(term["dy"][0]),
        "dz_arcsec": float(term["dz"][0]),
        "ang_y_deg": float(term["dy"][0]) / 3600,
        "ang_z_deg": float(term["dz"][0]) / 3600,
        "quarter": f"{quarter // 4}Q{quarter % 4 + 1}",
    }


def apply_fid_shift(
    acacent: str | Path, outfile: str | Path, slots: list[int], shift: dict
) -> int:
    """Write ``acacent`` with the shift subtracted from every row of ``slots``.

    Returns the number of rows shifted. ``outfile`` may be ``acacent`` itself.
    """
    with fits.open(acacent) as hdus:
        table = hdus["ACACENT"]
        rows = np.isin(table.data["slot"], slots)
        for column, key in (("ang_y_sm", "ang_y_deg"), ("ang_z_sm", "ang_z_deg")):
            values = table.data[column]
            values[rows] = values[rows] - np.asarray(shift[key], values.dtype)
        header = table.header
        header["GTRM_DY"] = (
            shift["dy_arcsec"],
            "[arcsec] gradient term dy, off ang_y_sm",
        )
        header["GTRM_DZ"] = (
            shift["dz_arcsec"],
            "[arcsec] gradient term dz, off ang_z_sm",
        )
        header["GTRM_QTR"] = (shift["quarter"], "gradient term reference quarter")
        header["HISTORY"] = (
            f"gradient term: fid slots {slots} ang_y_sm -= {shift['ang_y_deg']:.9f} deg,"
            f" ang_z_sm -= {shift['ang_z_deg']:.9f} deg"
        )
        hdus.writeto(outfile, overwrite=True)
    return int(rows.sum())


def predicted_event_shift(
    asol_before: str | Path, asol_after: str | Path
) -> tuple[float, float]:
    """How far X-ray events move in y and z (arcsec) when reprocessed with ``asol_after``.

    The median over the common times of the attitude turn (yaw moves events in
    y one for one, pitch in z) plus the change in dy and dz (ARCSEC_PER_MM_DYDZ
    per mm). For the gradient term this should be minus the term.

    Needs Ska's Quaternion package; nothing else in this module does.
    """
    from Quaternion import Quat

    columns = {}
    for label, path in (("before", asol_before), ("after", asol_after)):
        with fits.open(path) as hdus:
            data = hdus[1].data
            columns[label] = {
                name: np.asarray(data[name], float)
                for name in ("time", "q_att", "dy", "dz")
            }
    common, index_before, index_after = np.intersect1d(
        columns["before"]["time"], columns["after"]["time"], return_indices=True
    )
    if len(common) == 0:
        raise ValueError("the two aspect solutions share no times")
    turn = Quat(q=columns["before"]["q_att"][index_before]).inv() * Quat(
        q=columns["after"]["q_att"][index_after]
    )
    yaw = (np.atleast_1d(turn.ra) + 180) % 360 - 180
    pitch = np.atleast_1d(turn.dec)
    change_dy = (
        columns["after"]["dy"][index_after] - columns["before"]["dy"][index_before]
    )
    change_dz = (
        columns["after"]["dz"][index_after] - columns["before"]["dz"][index_before]
    )
    shift_y = np.median(yaw * 3600 + ARCSEC_PER_MM_DYDZ * change_dy)
    shift_z = np.median(pitch * 3600 + ARCSEC_PER_MM_DYDZ * change_dz)
    return float(shift_y), float(shift_z)
