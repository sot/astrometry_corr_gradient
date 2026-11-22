"""Tests for astrometry_corr_gradient.telemetry.

Two independent pieces:

- group_obsid_windows: build obsid -> (start, stop) from Manvr rows
  (kalman_start to next_manvr_start). Obsids with more than one qualifying
  Manvr row (interrupted/resumed observations) are excluded rather than
  merged -- simpler than reasoning about unioned windows, at the cost of
  dropping the small fraction of obsids this happens to.
- mean_in_windows: average telemetry samples whose time falls inside a
  window.
"""

import numpy as np
from astropy.table import Table
from cxotime import CxoTime

from astrometry_corr_gradient.telemetry import (
    align_msid_to_windows,
    cached_msid_fetch,
    combine_bad_telemetry_intervals,
    drop_ignored_sheet_states,
    fetch_candidate_series,
    get_bad_telemetry_intervals_cached,
    group_obsid_windows,
    mask_bad_telemetry_intervals,
    mask_out_of_range,
    mean_in_windows,
    shift_windows,
    stitch_at_cutover,
)


def make_manvrs(rows):
    return Table(
        rows=rows, names=["obsid", "kalman_start_secs", "next_manvr_start_secs"]
    )


def test_single_manvr_row_gives_one_window():
    manvrs = make_manvrs([(1, 100.0, 200.0)])
    windows = group_obsid_windows(manvrs)
    assert windows == {1: (100.0, 200.0)}


def test_obsid_with_more_than_one_row_is_excluded():
    manvrs = make_manvrs([(1, 100.0, 200.0), (1, 500.0, 600.0)])
    windows = group_obsid_windows(manvrs)
    assert windows == {}


def test_overlapping_rows_for_same_obsid_are_also_excluded():
    manvrs = make_manvrs([(1, 100.0, 300.0), (1, 250.0, 400.0)])
    windows = group_obsid_windows(manvrs)
    assert windows == {}


def test_rows_with_missing_bounds_are_skipped_before_counting():
    # only one row has usable bounds, so this obsid keeps its one window
    # rather than being excluded as "more than one row".
    manvrs = make_manvrs([(1, np.nan, 200.0), (1, 100.0, 200.0)])
    windows = group_obsid_windows(manvrs)
    assert windows == {1: (100.0, 200.0)}


def test_distinct_obsids_stay_separate():
    manvrs = make_manvrs([(1, 100.0, 200.0), (2, 300.0, 400.0)])
    windows = group_obsid_windows(manvrs)
    assert windows == {1: (100.0, 200.0), 2: (300.0, 400.0)}


def test_mean_in_windows_averages_samples_inside_the_window():
    times = np.array([0.0, 10.0, 20.0, 30.0])
    values = np.array([1.0, 2.0, 3.0, 4.0])
    result = mean_in_windows(times, values, (5.0, 25.0))
    assert result == 2.5  # mean of values at t=10,20


def test_mean_in_windows_is_nan_when_no_samples_fall_inside():
    times = np.array([0.0, 10.0])
    values = np.array([1.0, 2.0])
    result = mean_in_windows(times, values, (100.0, 200.0))
    assert np.isnan(result)


def test_mean_in_windows_bounds_are_inclusive():
    times = np.array([5.0, 10.0, 15.0])
    values = np.array([1.0, 2.0, 3.0])
    result = mean_in_windows(times, values, (5.0, 10.0))
    assert result == 1.5


def test_align_msid_to_windows_computes_one_value_per_requested_obsid():
    times = np.array([0.0, 10.0, 20.0, 30.0, 40.0])
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    windows = {1: (5.0, 25.0), 2: (35.0, 45.0)}
    result = align_msid_to_windows(times, values, windows)
    assert result == {1: 2.5, 2: 5.0}


def test_align_msid_to_windows_only_computes_requested_obsids():
    # a huge windows dict shouldn't require touching obsids we don't ask
    # for -- this test just confirms the result is scoped to the input.
    times = np.array([0.0, 10.0])
    values = np.array([1.0, 2.0])
    windows = {1: (0.0, 10.0), 99: (1000.0, 2000.0)}
    result = align_msid_to_windows(times, values, windows)
    assert set(result) == {1, 99}
    assert result[1] == 1.5
    assert np.isnan(result[99])


def test_stitch_at_cutover_keeps_base_before_and_wide_at_or_after():
    base_times = np.array([0.0, 5.0, 10.0, 15.0])
    base_vals = np.array([1.0, 2.0, 3.0, 4.0])
    wide_times = np.array([8.0, 12.0, 20.0])
    wide_vals = np.array([30.0, 40.0, 50.0])
    times, vals = stitch_at_cutover(base_times, base_vals, wide_times, wide_vals, 10.0)
    # base's t=10,15 samples are dropped (>= cutover); wide's t=8 sample
    # is dropped (< cutover) -- merged series is base(<10) + wide(>=10).
    np.testing.assert_array_equal(times, [0.0, 5.0, 12.0, 20.0])
    np.testing.assert_array_equal(vals, [1.0, 2.0, 40.0, 50.0])


def test_stitch_at_cutover_drops_base_samples_at_or_after_cutover():
    base_times = np.array([0.0, 10.0])
    base_vals = np.array([1.0, 999.0])  # the t=10 sample is post-cutover
    wide_times = np.array([10.0, 20.0])
    wide_vals = np.array([100.0, 200.0])
    times, vals = stitch_at_cutover(base_times, base_vals, wide_times, wide_vals, 10.0)
    np.testing.assert_array_equal(times, [0.0, 10.0, 20.0])
    np.testing.assert_array_equal(vals, [1.0, 100.0, 200.0])


def test_stitch_at_cutover_drops_wide_samples_before_cutover():
    base_times = np.array([0.0, 5.0])
    base_vals = np.array([1.0, 2.0])
    wide_times = np.array([3.0, 20.0])  # the t=3 sample is pre-cutover
    wide_vals = np.array([888.0, 200.0])
    times, vals = stitch_at_cutover(base_times, base_vals, wide_times, wide_vals, 10.0)
    np.testing.assert_array_equal(times, [0.0, 5.0, 20.0])
    np.testing.assert_array_equal(vals, [1.0, 2.0, 200.0])


def test_stitch_at_cutover_handles_empty_wide():
    base_times = np.array([0.0, 5.0])
    base_vals = np.array([1.0, 2.0])
    times, vals = stitch_at_cutover(
        base_times, base_vals, np.array([]), np.array([]), 10.0
    )
    np.testing.assert_array_equal(times, base_times)
    np.testing.assert_array_equal(vals, base_vals)


def test_shift_windows_by_zero_is_unchanged():
    windows = {1: (100.0, 200.0), 2: (500.0, 600.0)}
    assert shift_windows(windows, lag=0.0) == windows


def test_shift_windows_moves_both_bounds_earlier_for_a_positive_lag():
    # a positive lag asks "does the target track the telemetry from
    # `lag` earlier?" -- so the window we sample telemetry from moves
    # backward in time by `lag`.
    windows = {1: (100.0, 200.0)}
    result = shift_windows(windows, lag=30.0)
    assert result == {1: (70.0, 170.0)}


def test_shift_windows_moves_bounds_later_for_a_negative_lag():
    windows = {1: (100.0, 200.0)}
    result = shift_windows(windows, lag=-30.0)
    assert result == {1: (130.0, 230.0)}


def test_shift_windows_handles_each_obsid_independently():
    windows = {1: (100.0, 200.0), 2: (1000.0, 2000.0)}
    result = shift_windows(windows, lag=10.0)
    assert result == {1: (90.0, 190.0), 2: (990.0, 1990.0)}


def test_fetch_candidate_series_fetches_plainly_when_not_stitched():
    fetched = {"1CBAT": (np.array([0.0, 1.0]), np.array([10.0, 20.0]))}

    def fake_fetch(msid, start, stop):
        return fetched[msid]

    times, vals = fetch_candidate_series(
        "1CBAT", stitch_pairs={}, fetch_fn=fake_fetch, start="x", stop="y"
    )
    np.testing.assert_array_equal(times, [0.0, 1.0])
    np.testing.assert_array_equal(vals, [10.0, 20.0])


def test_fetch_candidate_series_stitches_when_paired():
    fetched = {
        "OOBTHR41": (np.array([0.0, 5.0]), np.array([1.0, 2.0])),
        "OOBTHR41_WIDE": (np.array([5.0, 20.0]), np.array([100.0, 200.0])),
    }

    def fake_fetch(msid, start, stop):
        return fetched[msid]

    cutover_date = CxoTime(5.0).date
    stitch_pairs = {"OOBTHR41": ("OOBTHR41_WIDE", cutover_date)}
    times, vals = fetch_candidate_series(
        "OOBTHR41", stitch_pairs=stitch_pairs, fetch_fn=fake_fetch, start="x", stop="y"
    )
    # cutover at t=5: base < 5 kept, wide >= 5 kept.
    np.testing.assert_array_equal(times, [0.0, 5.0, 20.0])
    np.testing.assert_array_equal(vals, [1.0, 100.0, 200.0])


def test_mask_out_of_range_replaces_outliers_with_nan():
    vals = np.array([198.9, 290.0, 295.0, 256.6, 519128.4])
    masked = mask_out_of_range(vals, low=280.0, high=340.0)
    np.testing.assert_array_equal(
        np.isfinite(masked), [False, True, True, False, False]
    )
    np.testing.assert_allclose(masked[np.isfinite(masked)], [290.0, 295.0])


def test_mask_out_of_range_leaves_an_all_valid_array_unchanged():
    vals = np.array([285.0, 300.0, 310.0])
    masked = mask_out_of_range(vals, low=280.0, high=340.0)
    np.testing.assert_array_equal(masked, vals)


def test_mask_out_of_range_preserves_existing_nan():
    vals = np.array([np.nan, 300.0])
    masked = mask_out_of_range(vals, low=280.0, high=340.0)
    assert np.isnan(masked[0])
    assert masked[1] == 300.0


def test_mask_out_of_range_does_not_mutate_input():
    vals = np.array([198.9, 300.0])
    original = vals.copy()
    mask_out_of_range(vals, low=280.0, high=340.0)
    np.testing.assert_array_equal(vals, original)


def make_intervals(rows):
    return Table(rows=rows, names=["start", "stop"], dtype=["U23", "U23"])


def make_ofp_states(rows):
    return Table(
        rows=rows, names=["datestart", "datestop", "val"], dtype=["U23", "U23", "U8"]
    )


def test_combine_bad_telemetry_intervals_drops_nrml_and_pads_by_state():
    # SAFE gets 30 min padding, other non-NRML states get 5 min, NRML is dropped.
    # SAFE and STUP are placed hours apart so their padding windows can't
    # overlap and hide each other's boundary.
    ofp_states = Table(
        rows=[
            ("2020:001:00:00:00.000", "2020:001:00:10:00.000", "NRML"),
            ("2020:001:00:10:00.000", "2020:001:01:00:00.000", "SAFE"),
            ("2020:001:03:00:00.000", "2020:001:03:00:05.000", "STUP"),
            ("2020:001:03:00:05.000", "2020:001:04:00:00.000", "NRML"),
        ],
        names=["datestart", "datestop", "val"],
    )
    empty = make_intervals([])
    combined = combine_bad_telemetry_intervals(ofp_states, empty, empty, empty)

    assert len(combined) == 2
    times = CxoTime(
        [
            "2020:001:01:29:00.000",  # 29 min after SAFE's raw stop -- still padded
            "2020:001:01:31:00.000",  # 31 min after SAFE's raw stop -- past the 30 min pad
            "2020:001:03:04:05.000",  # 4 min after STUP's raw stop -- still padded
            "2020:001:03:06:05.000",  # 6 min after STUP's raw stop -- past the 5 min pad
        ]
    ).secs
    bad = mask_bad_telemetry_intervals(times, np.ones(4), combined)
    np.testing.assert_array_equal(np.isnan(bad), [True, False, True, False])


def test_combine_bad_telemetry_intervals_pads_normal_sun_backward_only():
    empty = make_intervals([])
    empty_ofp = make_ofp_states([])
    normal_suns = make_intervals([("2020:001:12:00:00.000", "2020:001:13:00:00.000")])
    combined = combine_bad_telemetry_intervals(empty_ofp, empty, normal_suns, empty)

    times = CxoTime(
        [
            "2020:001:10:31:00.000",  # 89 min before start -- inside the 90 min backward pad
            "2020:001:10:29:00.000",  # 91 min before start -- outside the 90 min backward pad
            "2020:001:12:30:00.000",  # inside the raw interval itself
            "2020:001:13:00:01.000",  # 1 s after stop -- not padded forward at all
        ]
    ).secs
    bad = mask_bad_telemetry_intervals(times, np.ones(4), combined)
    np.testing.assert_array_equal(np.isnan(bad), [True, False, True, False])


def test_combine_bad_telemetry_intervals_includes_safe_sun_and_sheet_unpadded():
    empty = make_intervals([])
    empty_ofp = make_ofp_states([])
    safe_suns = make_intervals([("2021:100:00:00:00.000", "2021:100:01:00:00.000")])
    sheet_intervals = make_intervals(
        [("2022:200:00:00:00.000", "2022:200:01:00:00.000")]
    )
    combined = combine_bad_telemetry_intervals(
        empty_ofp, safe_suns, empty, sheet_intervals
    )

    times = CxoTime(
        [
            "2021:100:00:30:00.000",  # inside safe_suns
            "2021:100:01:05:00.000",  # 5 min after safe_suns stop -- no padding at all
            "2022:200:00:30:00.000",  # inside the sheet interval
        ]
    ).secs
    bad = mask_bad_telemetry_intervals(times, np.ones(3), combined)
    np.testing.assert_array_equal(np.isnan(bad), [True, False, True])


def make_sheet():
    # one entry of each kind kadi's command-sheet exclude list has: roll only,
    # roll with another state, every state (empty), another state only
    return Table(
        rows=[
            (
                "2024:100:01:51:26.403",
                "2024:100:01:54:24.754",
                "off_nom_roll",
                "Far tail sun",
            ),
            (
                "2022:293:16:30:00.000",
                "2022:293:17:00:00.000",
                "pitch off_nom_roll",
                "NSM maneuver",
            ),
            ("2021:243:11:50:00.000", "2021:245:20:37:00.000", "", "LETG anomaly"),
            (
                "2022:016:00:05:23.000",
                "2022:018:18:43:48.000",
                "dpa_power",
                "ACIS reboot",
            ),
        ],
        names=["start", "stop", "states", "comment"],
    )


def test_drop_ignored_sheet_states_drops_only_roll_only_entries():
    sheet = make_sheet()
    kept = drop_ignored_sheet_states(sheet)

    assert list(kept["comment"]) == ["NSM maneuver", "LETG anomaly", "ACIS reboot"]
    assert len(sheet) == 4


def test_drop_ignored_sheet_states_takes_other_ignored_states():
    kept = drop_ignored_sheet_states(
        make_sheet(), ignored_states=("off_nom_roll", "dpa_power")
    )

    assert list(kept["comment"]) == ["NSM maneuver", "LETG anomaly"]


def test_mask_bad_telemetry_intervals_does_not_mutate_input():
    intervals = make_intervals([("2020:001:00:00:00.000", "2020:001:01:00:00.000")])
    times = CxoTime(["2020:001:00:30:00.000"]).secs
    values = np.array([5.0])
    original = values.copy()
    mask_bad_telemetry_intervals(times, values, intervals)
    np.testing.assert_array_equal(values, original)


def test_get_bad_telemetry_intervals_cached_reuses_a_fresh_covering_cache(tmp_path):
    cache_path = tmp_path / "bad_intervals.ecsv"
    fetch_calls = []

    def fake_fetch(start, stop):
        fetch_calls.append((start, stop))
        return make_intervals([("2020:001:00:00:00.000", "2020:001:01:00:00.000")])

    first = get_bad_telemetry_intervals_cached(
        "2004:001", "2020:100", cache_path, fetch_fn=fake_fetch, max_age_days=1.0
    )
    assert len(fetch_calls) == 1
    assert len(first) == 1

    # Same call again: cache file exists, was just written (fresh), and
    # already covers this exact request -- must NOT call fetch_fn again.
    second = get_bad_telemetry_intervals_cached(
        "2004:001", "2020:100", cache_path, fetch_fn=fake_fetch, max_age_days=1.0
    )
    assert len(fetch_calls) == 1
    np.testing.assert_array_equal(second["start"], first["start"])


def test_get_bad_telemetry_intervals_cached_reuses_cache_despite_a_later_stop(tmp_path):
    # stop is computed by real callers as "now plus a margin", so it is a
    # few seconds later on every single call, forever -- a cache must
    # still be reused despite that, or caching would never do anything.
    cache_path = tmp_path / "bad_intervals.ecsv"
    fetch_calls = []

    def fake_fetch(start, stop):
        fetch_calls.append((start, stop))
        return make_intervals([("2020:001:00:00:00.000", "2020:001:01:00:00.000")])

    get_bad_telemetry_intervals_cached(
        "2004:001", "2020:100", cache_path, fetch_fn=fake_fetch, max_age_days=1.0
    )
    assert len(fetch_calls) == 1

    get_bad_telemetry_intervals_cached(
        "2004:001", "2021:100", cache_path, fetch_fn=fake_fetch, max_age_days=1.0
    )
    assert len(fetch_calls) == 1


def test_get_bad_telemetry_intervals_cached_refetches_when_start_not_covered(tmp_path):
    cache_path = tmp_path / "bad_intervals.ecsv"
    fetch_calls = []

    def fake_fetch(start, stop):
        fetch_calls.append((start, stop))
        return make_intervals([("2020:001:00:00:00.000", "2020:001:01:00:00.000")])

    get_bad_telemetry_intervals_cached(
        "2010:001", "2020:100", cache_path, fetch_fn=fake_fetch, max_age_days=1.0
    )
    assert len(fetch_calls) == 1

    # An earlier start than what's cached -- must refetch, since the cache
    # was never asked to cover this much history.
    get_bad_telemetry_intervals_cached(
        "2004:001", "2020:100", cache_path, fetch_fn=fake_fetch, max_age_days=1.0
    )
    assert len(fetch_calls) == 2


def test_get_bad_telemetry_intervals_cached_refetches_when_stale(tmp_path):
    cache_path = tmp_path / "bad_intervals.ecsv"
    fetch_calls = []

    def fake_fetch(start, stop):
        fetch_calls.append((start, stop))
        return make_intervals([("2020:001:00:00:00.000", "2020:001:01:00:00.000")])

    # max_age_days=0 makes any existing cache immediately stale, regardless
    # of range coverage.
    get_bad_telemetry_intervals_cached(
        "2004:001", "2020:100", cache_path, fetch_fn=fake_fetch, max_age_days=0.0
    )
    get_bad_telemetry_intervals_cached(
        "2004:001", "2020:100", cache_path, fetch_fn=fake_fetch, max_age_days=0.0
    )
    assert len(fetch_calls) == 2


def make_archive(available_until):
    """A fake 5-minute telemetry archive: one sample every 300 s, value = time / 1000, up to `available_until`."""
    calls = []

    def fetch(msid, start, stop):
        calls.append((msid, CxoTime(start).secs, CxoTime(stop).secs))
        first, last = CxoTime(start).secs, min(CxoTime(stop).secs, available_until[0])
        times = np.arange(np.ceil(first / 300) * 300, last + 1, 300.0)
        offset = 0.0 if msid == "A" else 1.0
        return times, times / 1000 + offset

    return fetch, calls


def test_cached_msid_fetch_fetches_once_then_only_the_recent_overlap(tmp_path):
    now = [CxoTime("2020:010").secs]
    archive, calls = make_archive(now)
    fetch = cached_msid_fetch(archive, tmp_path, overlap_days=3.0)
    first_times, first_vals = fetch("A", "2020:001", "2020:015")
    assert len(calls) == 1
    second_times, second_vals = fetch("A", "2020:001", "2020:016")
    assert len(calls) == 2
    # the second fetch starts 3 days before the last cached sample, not at the requested start
    assert abs(calls[1][1] - (first_times[-1] - 3 * 86400)) < 1
    np.testing.assert_array_equal(second_times, first_times)
    np.testing.assert_array_equal(second_vals, first_vals)


def test_cached_msid_fetch_appends_newly_arrived_telemetry(tmp_path):
    now = [CxoTime("2020:010").secs]
    archive, _ = make_archive(now)
    fetch = cached_msid_fetch(archive, tmp_path, overlap_days=3.0)
    fetch("A", "2020:001", "2020:015")
    now[0] = CxoTime("2020:020").secs  # ten more days of telemetry arrive
    times, vals = fetch("A", "2020:001", "2020:025")
    direct_times, direct_vals = archive("A", "2020:001", "2020:025")
    np.testing.assert_array_equal(times, direct_times)
    np.testing.assert_array_equal(vals, direct_vals)
    # and a fresh wrapper (a new kernel) reads the cache from disk the same way
    times_again, _ = cached_msid_fetch(archive, tmp_path, overlap_days=3.0)(
        "A", "2020:001", "2020:025"
    )
    np.testing.assert_array_equal(times_again, direct_times)


def test_cached_msid_fetch_refetches_everything_for_an_earlier_start(tmp_path):
    now = [CxoTime("2020:010").secs]
    archive, calls = make_archive(now)
    fetch = cached_msid_fetch(archive, tmp_path)
    fetch("A", "2020:005", "2020:015")
    times, _ = fetch("A", "2020:001", "2020:015")
    assert abs(calls[-1][1] - CxoTime("2020:001").secs) < 1
    assert times[0] < CxoTime("2020:002").secs


def test_cached_msid_fetch_returns_only_the_requested_range(tmp_path):
    now = [CxoTime("2020:010").secs]
    archive, _ = make_archive(now)
    fetch = cached_msid_fetch(archive, tmp_path)
    fetch("A", "2020:001", "2020:015")
    times, _ = fetch("A", "2020:003", "2020:004")
    assert times.min() >= CxoTime("2020:003").secs
    assert times.max() <= CxoTime("2020:004").secs


def test_cached_msid_fetch_keeps_each_msid_apart(tmp_path):
    now = [CxoTime("2020:010").secs]
    archive, _ = make_archive(now)
    fetch = cached_msid_fetch(archive, tmp_path)
    _, a_vals = fetch("A", "2020:001", "2020:015")
    _, b_vals = fetch("B", "2020:001", "2020:015")
    np.testing.assert_allclose(b_vals - a_vals, 1.0)
