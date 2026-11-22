"""Align telemetry (MSID) values to observation time windows.

An observation's "time window" is defined as kalman_start to
next_manvr_start of its Manvr row -- the period from when the aspect
solution locks after a maneuver to when the next maneuver begins. A
handful of obsids (interrupted/resumed observations) have more than one
Manvr row; those are excluded rather than merged, since reasoning about a
unioned window is more complex than the fraction of the sample it's worth
keeping.
"""

from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
from astropy.table import Table, vstack
from cxotime import CxoTime

OFP_SAFE_PAD_SECS = 30 * 60.0
OFP_OTHER_PAD_SECS = 5 * 60.0
NORMAL_SUN_BACKWARD_PAD_SECS = 90 * 60.0
# kadi command-sheet exclude entries about only these states say nothing about
# the thermistors' readings (Jean, 2026-10-29: off-nominal roll, obsid 28864)
IGNORED_SHEET_STATES = ("off_nom_roll",)


def mask_out_of_range(values: np.ndarray, low: float, high: float) -> np.ndarray:
    """Replace samples outside ``[low, high]`` with NaN.

    Meant to be applied to a raw MSID series (e.g. a 5-minute-stat fetch)
    before any per-obsid window-averaging, so a telemetry glitch drops the
    whole window's value to NaN rather than blending into a smaller,
    physically-plausible-looking but wrong mean (a real failure mode: an
    OBA thermistor glitch reading ~199 K or ~519128 K blended with mostly
    good samples can still land inside a naive range check on the
    averaged statistic). Confirmed present in this project's own OOBTHR06/
    13/40/41 telemetry, matching the old project's documented OBA-
    thermistor bad-value clusters exactly (PROGRESS.md section 59) -- not
    assumed from that precedent alone.

    Parameters
    ----------
    values : numpy.ndarray
        A telemetry series, in its own physical units (e.g. Kelvin).
    low, high : float
        The valid range, inclusive. Values outside it, and any existing
        NaN, become NaN in the result.

    Returns
    -------
    numpy.ndarray
        A new array, the same shape as ``values``, with out-of-range
        samples replaced by NaN. The input is not modified.

    Examples
    --------
    >>> import numpy as np
    >>> mask_out_of_range(np.array([198.9, 295.0, 519128.4]), low=280.0, high=340.0)
    array([ nan, 295.,  nan])
    """
    masked = values.astype(float).copy()
    masked[(masked < low) | (masked > high)] = np.nan
    return masked


def combine_bad_telemetry_intervals(
    ofp_states: Table,
    safe_suns: Table,
    normal_suns: Table,
    sheet_intervals: Table,
) -> Table:
    """Union the four sources of "telemetry may not be trustworthy" intervals.

    Root-cause investigation (PROGRESS.md, astrometry thermal scan project)
    found that every observed telemetry glitch in this project's driver
    MSIDs is attributable to a spacecraft safing event (Safe Mode, Normal
    Sun Mode, or a raw OFP-state transition away from ``NRML``) -- not
    independent per-MSID sensor faults. This combines the four kinds of
    evidence for such an event into one ``start``/``stop`` interval table,
    each padded according to what the evidence actually showed was needed
    (not a single one-size-fits-all margin):

    - ``ofp_states`` (e.g. from :func:`cheta.utils.get_ofp_states`): every
      row whose ``val`` is not ``"NRML"`` is included, padded forward by
      :data:`OFP_SAFE_PAD_SECS` (``val == "SAFE"``) or
      :data:`OFP_OTHER_PAD_SECS` (any other non-``NRML`` state) -- the same
      convention `kadi.commands.validate.ValidatePitchRollBase` itself
      uses.
    - ``safe_suns`` (e.g. from ``kadi.events.safe_suns``): included as-is,
      unpadded -- every verified episode inside a `safe_sun` interval
      needed no padding at all.
    - ``normal_suns`` (e.g. from ``kadi.events.normal_suns``): padded
      :data:`NORMAL_SUN_BACKWARD_PAD_SECS` *backward* on ``start`` only.
      The two episodes needing this category both preceded their nearest
      `normal_sun` interval by 43-62 minutes (a CTU reset anomaly and a
      momentum-monitor safe-mode recovery tail, both confirmed against
      `kadi.events.major_events`) -- the anomaly itself predates the
      formal state transition, not the other way around.
    - ``sheet_intervals`` (e.g. from
      :func:`kadi.commands.validate.get_command_sheet_exclude_intervals`):
      included as-is, unpadded (:func:`get_bad_telemetry_intervals` first
      drops the entries :func:`drop_ignored_sheet_states` ignores).

    Parameters
    ----------
    ofp_states : astropy.table.Table
        Columns ``datestart``, ``datestop``, ``val`` (CxoTime-parseable
        date strings and a state-code string), as returned by
        :func:`cheta.utils.get_ofp_states`.
    safe_suns, normal_suns, sheet_intervals : astropy.table.Table
        Each with columns ``start``, ``stop`` (CxoTime-parseable date
        strings).

    Returns
    -------
    astropy.table.Table
        Columns ``start``, ``stop`` -- the union of all four sources,
        padded as described above. Overlaps are not collapsed; a
        downstream overlap test (e.g. :func:`mask_bad_telemetry_intervals`)
        handles that directly.

    Examples
    --------
    >>> from astropy.table import Table
    >>> ofp = Table(
    ...     rows=[("2020:001:00:00:00.000", "2020:001:00:10:00.000", "SAFE")],
    ...     names=["datestart", "datestop", "val"],
    ... )
    >>> empty = Table(rows=[], names=["start", "stop"], dtype=["U23", "U23"])
    >>> combined = combine_bad_telemetry_intervals(ofp, empty, empty, empty)
    >>> len(combined)
    1
    """
    non_nrml = ofp_states[ofp_states["val"] != "NRML"]
    # CxoTime(...).date on a genuinely empty array falls back to a float64
    # dtype rather than a string one (nothing to format), which would break
    # the vstack below against the other, string-typed interval tables --
    # skip the round-trip entirely rather than pad zero rows.
    if len(non_nrml) == 0:
        ofp_intervals = Table(
            {"start": non_nrml["datestart"], "stop": non_nrml["datestop"]}
        )
    else:
        ofp_pad_secs = np.where(
            non_nrml["val"] == "SAFE", OFP_SAFE_PAD_SECS, OFP_OTHER_PAD_SECS
        )
        ofp_intervals = Table(
            {
                "start": CxoTime(non_nrml["datestart"]).date,
                "stop": CxoTime(CxoTime(non_nrml["datestop"]).secs + ofp_pad_secs).date,
            }
        )

    if len(normal_suns) == 0:
        padded_normal_suns = Table(
            {"start": normal_suns["start"], "stop": normal_suns["stop"]}
        )
    else:
        padded_normal_suns = Table(
            {
                "start": CxoTime(
                    CxoTime(normal_suns["start"]).secs - NORMAL_SUN_BACKWARD_PAD_SECS
                ).date,
                "stop": normal_suns["stop"],
            }
        )

    return vstack(
        [
            ofp_intervals,
            Table({"start": safe_suns["start"], "stop": safe_suns["stop"]}),
            padded_normal_suns,
            Table({"start": sheet_intervals["start"], "stop": sheet_intervals["stop"]}),
        ]
    )


def drop_ignored_sheet_states(
    sheet_intervals: Table, ignored_states: Sequence[str] = IGNORED_SHEET_STATES
) -> Table:
    """Drop the command-sheet exclude intervals that only concern ignored states.

    Each entry of kadi's command-sheet exclude list names the commanded
    states (``states``, space-separated) that kadi should not check against
    telemetry during it; an empty ``states`` means every state. An entry
    about only ``off_nom_roll`` (e.g. "Far tail sun > 177 deg", where roll
    is ill-defined) says nothing about whether the driver MSIDs' readings
    are valid, so it should not mask them.

    Parameters
    ----------
    sheet_intervals : astropy.table.Table
        Columns ``start``, ``stop``, ``states``, as returned by
        :func:`kadi.commands.validate.get_command_sheet_exclude_intervals`.
    ignored_states : sequence of str
        The states whose entries are dropped when they name nothing else.

    Returns
    -------
    astropy.table.Table
        The entries naming no state (every state) or at least one state not
        in ``ignored_states``. The input is not modified.

    Examples
    --------
    >>> from astropy.table import Table
    >>> sheet = Table(
    ...     rows=[
    ...         ("2024:100:01:51:26", "2024:100:01:54:24", "off_nom_roll"),
    ...         ("2021:243:11:50:00", "2021:245:20:37:00", ""),
    ...     ],
    ...     names=["start", "stop", "states"],
    ... )
    >>> len(drop_ignored_sheet_states(sheet))
    1
    """
    ignored = set(ignored_states)
    named = [set(str(states).split()) for states in sheet_intervals["states"]]
    keep = np.array(
        [not states or not states <= ignored for states in named], dtype=bool
    )
    return sheet_intervals[keep]


def mask_bad_telemetry_intervals(
    times: np.ndarray, values: np.ndarray, bad_intervals: Table
) -> np.ndarray:
    """Replace samples whose time falls inside any of ``bad_intervals`` with NaN.

    Time-based counterpart to :func:`mask_out_of_range`: rather than
    judging a sample by its value, this judges it by when it was taken,
    against known spacecraft safing intervals (see
    :func:`combine_bad_telemetry_intervals`). Meant to be applied to a raw
    MSID series before any window-averaging, for the same reason
    :func:`mask_out_of_range` is -- so a bad sample poisons its whole
    obsid's value instead of blending into a plausible-looking mean.

    Parameters
    ----------
    times : numpy.ndarray
        CXC seconds, one per sample.
    values : numpy.ndarray
        The telemetry values, same length as ``times``.
    bad_intervals : astropy.table.Table
        Columns ``start``, ``stop`` (CxoTime-parseable), as returned by
        :func:`combine_bad_telemetry_intervals`. Half-open: a sample at
        exactly ``stop`` is not masked.

    Returns
    -------
    numpy.ndarray
        A new array, the same shape as ``values``, with samples inside any
        bad interval replaced by NaN. The input is not modified.
    """
    masked = values.astype(float).copy()
    bad = np.zeros(len(times), dtype=bool)
    for interval in bad_intervals:
        start_secs = CxoTime(interval["start"]).secs
        stop_secs = CxoTime(interval["stop"]).secs
        bad |= (times >= start_secs) & (times < stop_secs)
    masked[bad] = np.nan
    return masked


def get_bad_telemetry_intervals(start: str, stop: str) -> Table:
    """Fetch and combine every known source of bad-telemetry intervals for a date range.

    Live-data counterpart to :func:`combine_bad_telemetry_intervals`: fetches
    the four inputs it needs (OFP states, safe suns, normal suns, the
    command-events sheet, less its entries about only
    :data:`IGNORED_SHEET_STATES`; see :func:`drop_ignored_sheet_states`) and
    combines them. Imports `kadi`/`cheta` locally
    rather than at module level, so importing :mod:`astrometry.telemetry`
    itself never requires ``SKA`` to be set -- only actually calling this
    function does.

    Parameters
    ----------
    start, stop : str
        The date range to fetch bad intervals for (CxoTime-parseable).

    Returns
    -------
    astropy.table.Table
        See :func:`combine_bad_telemetry_intervals`.
    """
    from cheta.utils import get_ofp_states
    from kadi import events
    from kadi.commands.validate import get_command_sheet_exclude_intervals

    ofp_states = get_ofp_states(CxoTime(start), CxoTime(stop))
    safe_suns = events.safe_suns.filter(start, stop).table
    normal_suns = events.normal_suns.filter(start, stop).table
    sheet_intervals = drop_ignored_sheet_states(get_command_sheet_exclude_intervals())
    return combine_bad_telemetry_intervals(
        ofp_states, safe_suns, normal_suns, sheet_intervals
    )


def get_bad_telemetry_intervals_cached(
    start: str,
    stop: str,
    cache_path: str | Path,
    fetch_fn: Callable[[str, str], Table] = get_bad_telemetry_intervals,
    max_age_days: float = 1.0,
) -> Table:
    """Cached wrapper around ``fetch_fn`` (default :func:`get_bad_telemetry_intervals`).

    The dominant cost of a fresh fetch is raw, ~1-second-cadence OFP-state
    telemetry across the whole requested range -- and that history, beyond
    the last few days, never changes. Rerunning the same notebook otherwise
    re-pays that cost every time for no reason. This caches the combined
    interval table to ``cache_path`` as plain-string-column ecsv (no
    CxoTime-mixin column, so it reads back cleanly -- see the ecsv gotcha
    this project hit before) and reuses it as long as it is both recent
    enough (``max_age_days``) and already covers the requested ``start``
    date; otherwise it calls ``fetch_fn`` fresh and overwrites the cache.

    Deliberately does NOT also require ``cached_stop >= stop``: this
    project's own callers compute ``stop`` as "now plus a few days'
    margin" (so a hardcoded stop date can't silently go stale), which
    means ``stop`` is a few seconds later on every single call, forever --
    checked directly against a real cache file and confirmed a
    stop-coverage check would never once be satisfied between two
    different calls, no matter how close together in time, defeating
    caching entirely. ``start`` is stable across calls (a fixed mission
    start date) and is the check that actually protects against reusing a
    cache built for a narrower historical range; ``max_age_days`` is what
    protects against a stale ``stop``.

    Parameters
    ----------
    start, stop : str
        The date range needed for this call (CxoTime-parseable).
    cache_path : str or pathlib.Path
        Where to read/write the cached interval table.
    fetch_fn : callable
        ``fetch_fn(start, stop) -> Table``, called on a cache miss.
        Injected (default :func:`get_bad_telemetry_intervals`) so this is
        testable without live `kadi`/`cheta` access.
    max_age_days : float
        A cache older than this is refetched even if it already covers the
        requested start date, in case an already-archived period gets
        revised, or ``stop`` has moved meaningfully further into the
        future since the cache was built.

    Returns
    -------
    astropy.table.Table
        See :func:`combine_bad_telemetry_intervals`.
    """
    cache_path = Path(cache_path)
    if cache_path.exists():
        cached = Table.read(cache_path, format="ascii.ecsv")
        age_days = (CxoTime.now() - CxoTime(cached.meta["computed_at"])).jd
        covers_start = CxoTime(cached.meta["cache_start"]).secs <= CxoTime(start).secs
        if age_days <= max_age_days and covers_start:
            return cached

    fresh = fetch_fn(start, stop)
    fresh.meta["computed_at"] = CxoTime.now().date
    fresh.meta["cache_start"] = start
    fresh.meta["cache_stop"] = stop
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    fresh.write(cache_path, format="ascii.ecsv", overwrite=True)
    return fresh


def cached_msid_fetch(
    fetch_fn: Callable[[str, str, str], tuple[np.ndarray, np.ndarray]],
    cache_dir: str | Path,
    overlap_days: float = 3.0,
    label: str = "5min",
) -> Callable[[str, str, str], tuple[np.ndarray, np.ndarray]]:
    """Wrap ``fetch_fn(msid, start, stop) -> (times, vals)`` with a per-MSID cache on disk.

    Fetching mission-long 5-minute telemetry for each driver MSID is most of
    the cost of loading the report, and that history never changes. The
    wrapper keeps each MSID's series in ``cache_dir/<msid>_<label>.npz``.
    On a later call it fetches only from ``overlap_days`` before the last
    cached sample to the requested stop, and appends that (replacing the
    overlap, in case late telemetry filled it in). So the newest
    observations always get their telemetry, unlike a cache that is simply
    reused until it is old. A request starting before the cached series
    refetches everything. Only the requested range is returned.

    Parameters
    ----------
    fetch_fn : callable
        ``fetch_fn(msid, start, stop) -> (times, vals)``, e.g. a wrapper of
        ``cheta.fetch.Msid(msid, start, stop, stat="5min")``.
    cache_dir : str or pathlib.Path
        Where the per-MSID files live.
    overlap_days : float
        How far before the last cached sample each refresh starts.
    label : str
        Part of the file name, so different stats keep separate files.

    Returns
    -------
    callable
        ``fetch(msid, start, stop) -> (times, vals)``, a drop-in for
        ``fetch_fn``.

    Examples
    --------
    >>> import tempfile
    >>> def archive(msid, start, stop):
    ...     times = np.arange(CxoTime(start).secs, CxoTime(stop).secs, 3600.0)
    ...     return times, np.zeros(len(times))
    >>> fetch = cached_msid_fetch(archive, tempfile.mkdtemp())
    >>> len(fetch("TEMP", "2020:001", "2020:002")[0])
    24
    """
    cache_dir = Path(cache_dir)

    def fetch(msid: str, start: str, stop: str) -> tuple[np.ndarray, np.ndarray]:
        start_secs, stop_secs = CxoTime(start).secs, CxoTime(stop).secs
        path = cache_dir / f"{msid}_{label}.npz"
        times = vals = None
        if path.exists():
            with np.load(path, allow_pickle=False) as cached:
                if float(cached["cache_start"]) <= start_secs:
                    times, vals, cache_start = (
                        cached["times"],
                        cached["vals"],
                        float(cached["cache_start"]),
                    )
        if times is None:
            cache_start = start_secs
            times, vals = (np.asarray(array) for array in fetch_fn(msid, start, stop))
        else:
            last = float(times[-1]) if len(times) else cache_start
            refresh_from = max(last - overlap_days * 86400.0, cache_start)
            if stop_secs > refresh_from:
                new_times, new_vals = (
                    np.asarray(array)
                    for array in fetch_fn(msid, CxoTime(refresh_from).date, stop)
                )
                keep = times < refresh_from
                times = np.concatenate([times[keep], new_times])
                vals = np.concatenate([vals[keep], new_vals])
        cache_dir.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(".partial.npz")
        np.savez(partial, times=times, vals=vals, cache_start=cache_start)
        partial.replace(path)  # a run stopped mid-write never leaves a broken cache
        selected = (times >= start_secs) & (times <= stop_secs)
        return times[selected], vals[selected]

    return fetch


def group_obsid_windows(manvrs: Table) -> dict[int, tuple[float, float]]:
    """Build obsid -> (start, stop) from Manvr rows.

    Parameters
    ----------
    manvrs : astropy.table.Table
        Must have ``obsid``, ``kalman_start_secs``, ``next_manvr_start_secs``
        (CxoTime seconds). Rows with a missing (NaN) bound are dropped
        before counting rows per obsid -- this happens for a handful of
        pre-launch/commissioning Manvr rows where ``kalman_start`` is
        unset.

    Returns
    -------
    dict
        obsid -> (start, stop) window in CxoTime seconds, for obsids with
        exactly one qualifying Manvr row. Obsids with more than one
        (interrupted or resumed observations) are excluded entirely.
    """
    valid = manvrs[
        ~np.isnan(manvrs["kalman_start_secs"])
        & ~np.isnan(manvrs["next_manvr_start_secs"])
    ]

    windows_by_obsid: dict[int, tuple[float, float]] = {}
    for obsid in np.unique(valid["obsid"]):
        rows = valid[valid["obsid"] == obsid]
        if len(rows) == 1:
            windows_by_obsid[int(obsid)] = (
                float(rows["kalman_start_secs"][0]),
                float(rows["next_manvr_start_secs"][0]),
            )

    return windows_by_obsid


def mean_in_windows(
    times: np.ndarray, values: np.ndarray, window: tuple[float, float]
) -> float:
    """Mean of ``values`` whose ``times`` fall inside ``window``.

    Parameters
    ----------
    times : numpy.ndarray
        Sample times, in the same units as ``window``'s bounds.
    values : numpy.ndarray
        Sample values, same length as ``times``.
    window : (float, float)
        Inclusive (start, stop) window.

    Returns
    -------
    float
        Mean of the samples inside ``window``, or ``nan`` if none fall
        inside it.
    """
    start, stop = window
    inside = (times >= start) & (times <= stop)

    if not np.any(inside):
        return np.nan
    return float(np.mean(values[inside]))


def align_msid_to_windows(
    times: np.ndarray,
    values: np.ndarray,
    windows: dict[int, tuple[float, float]],
) -> dict[int, float]:
    """Mean-in-window value of a telemetry series, for each requested obsid.

    Meant to be called once per MSID against the full telemetry series and
    the small set of obsids actually needed (e.g. the ~6000 matched
    obsids), not the ~47000-obsid mission-wide window table -- there's no
    reason to compute values for obsids we won't use.

    Uses ``searchsorted`` rather than a per-obsid boolean mask (as
    :func:`mean_in_windows` does): ``times`` is assumed already sorted
    ascending, true of any telemetry series from cheta, so each window's
    bounds can be found in O(log n) instead of scanning all n samples.
    That difference matters at this scale -- boolean-masking per obsid
    against a mission-length 5-minute-stat series (millions of samples)
    took ~12s for one MSID, about 2.5 hours for all 791.

    Parameters
    ----------
    times, values : numpy.ndarray
        A telemetry series (e.g. one MSID's 5-minute-stat means), with
        ``times`` sorted ascending.
    windows : dict
        obsid -> (start, stop), as returned by :func:`group_obsid_windows`
        (already narrowed to the obsids of interest).

    Returns
    -------
    dict
        obsid -> mean value in that obsid's window, or ``nan`` if no
        sample falls inside it.
    """
    obsids = list(windows)
    starts = np.array([windows[obsid][0] for obsid in obsids])
    stops = np.array([windows[obsid][1] for obsid in obsids])
    start_indices = np.searchsorted(times, starts, side="left")
    stop_indices = np.searchsorted(times, stops, side="right")

    result = {}
    for obsid, i0, i1 in zip(obsids, start_indices, stop_indices, strict=True):
        result[obsid] = float(np.mean(values[i0:i1])) if i1 > i0 else np.nan
    return result


def stitch_at_cutover(
    base_times: np.ndarray,
    base_vals: np.ndarray,
    wide_times: np.ndarray,
    wide_vals: np.ndarray,
    cutover: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Merge a narrow/"_WIDE" MSID pair into one series at a cutover time.

    Some MSIDs (e.g. ``OOBTHR41``/``OOBTHR41_WIDE``) had their telemetry
    format changed at a documented instant: the base name's values are
    only valid before the cutover, the ``_WIDE`` name's only at or after
    it (see :func:`astrometry.msids.classify_wide_variants`). This drops
    each series' samples on the wrong side of the cutover and
    concatenates what's left, in time order.

    Parameters
    ----------
    base_times, base_vals : numpy.ndarray
        The base MSID's telemetry, sorted ascending in time.
    wide_times, wide_vals : numpy.ndarray
        The ``_WIDE`` MSID's telemetry, sorted ascending in time.
    cutover : float
        The cutover time, in the same units as both series' times. Base
        samples strictly before it are kept; wide samples at or after it
        are kept.

    Returns
    -------
    times, vals : numpy.ndarray
        The merged series, sorted ascending in time.
    """
    base_keep = base_times < cutover
    wide_keep = wide_times >= cutover
    times = np.concatenate([base_times[base_keep], wide_times[wide_keep]])
    vals = np.concatenate([base_vals[base_keep], wide_vals[wide_keep]])
    return times, vals


def shift_windows(
    windows: dict[int, tuple[float, float]], lag: float
) -> dict[int, tuple[float, float]]:
    """Shift every obsid's window earlier by ``lag``, for a lag scan.

    A positive ``lag`` asks "does the target at an obsid's own time track
    the telemetry from ``lag`` earlier?" -- so the window telemetry gets
    pooled from moves backward in time by ``lag``. Re-run
    :func:`align_msid_to_windows` against the shifted windows to get
    each obsid's telemetry value at that lag.

    Parameters
    ----------
    windows : dict
        obsid -> (start, stop), as returned by :func:`group_obsid_windows`.
    lag : float
        Amount to shift both bounds earlier, in the same units as the
        windows' bounds. Negative shifts the windows later instead.

    Returns
    -------
    dict
        obsid -> (start - lag, stop - lag).
    """
    return {
        obsid: (start - lag, stop - lag) for obsid, (start, stop) in windows.items()
    }


def fetch_candidate_series(
    msid: str,
    stitch_pairs: dict[str, tuple[str, str]],
    fetch_fn: Callable[[str, str, str], tuple[np.ndarray, np.ndarray]],
    start: str,
    stop: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Fetch one candidate MSID's telemetry, stitching a "_WIDE" pair if needed.

    Centralizes the fetch-then-maybe-stitch logic so callers (the scan
    loop, a control check) don't each have to know which MSIDs are
    narrow/wide pairs -- see
    :func:`astrometry.msids.classify_wide_variants`.

    Parameters
    ----------
    msid : str
        A base MSID name (never a "_WIDE" name directly -- those are
        looked up via ``stitch_pairs``).
    stitch_pairs : dict
        base name -> (wide name, cutover date string), as returned by
        :func:`astrometry.msids.classify_wide_variants`. If ``msid`` is
        a key, both series are fetched and merged at the cutover; if
        not, ``msid`` is fetched plainly.
    fetch_fn : callable
        ``fetch_fn(msid, start, stop) -> (times, vals)``. Injected so
        this stays testable without a real telemetry archive -- in
        production, wraps ``cheta.fetch.Msid(msid, start, stop,
        stat="5min")``.
    start, stop : str
        Passed through to ``fetch_fn`` unchanged.

    Returns
    -------
    times, vals : numpy.ndarray
    """
    if msid not in stitch_pairs:
        return fetch_fn(msid, start, stop)

    wide_msid, cutover_date = stitch_pairs[msid]
    base_times, base_vals = fetch_fn(msid, start, stop)
    wide_times, wide_vals = fetch_fn(wide_msid, start, stop)
    cutover_secs = CxoTime(cutover_date).secs
    return stitch_at_cutover(base_times, base_vals, wide_times, wide_vals, cutover_secs)
