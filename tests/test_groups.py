"""Tests for astrometry_corr_gradient.groups (copied from astrometry's tests/test_evaluation.py)."""

import numpy as np

from astrometry_corr_gradient.groups import block_bootstrap_difference, linked_group_ids


def test_block_bootstrap_difference_favors_the_smaller_residual_series():
    rng = np.random.default_rng(0)
    n = 500
    years = np.sort(rng.uniform(2004, 2024, n))
    resid_a = rng.normal(scale=1.0, size=n)  # noisier model
    resid_b = rng.normal(scale=0.3, size=n)  # tighter model

    diff, low, _high = block_bootstrap_difference(
        resid_a, resid_b, years, statistic=lambda r: np.std(r), n_boot=200, seed=0
    )
    assert diff > 0  # a's spread is bigger than b's
    assert low > 0  # and the whole CI says so, not just the point estimate


def test_linked_group_ids_join_rows_sharing_any_label_transitively():
    obsids = np.array([1, 1, 2, 3, 3, 4])
    sources = np.array(["a", "b", "b", "c", "d", "d"])
    groups = linked_group_ids(obsids, sources)
    # rows 0-1 share obsid 1, rows 1-2 share source b; rows 3-4 share obsid 3, rows 4-5 share source d
    assert groups[0] == groups[1] == groups[2]
    assert groups[3] == groups[4] == groups[5]
    assert groups[0] != groups[3]


def test_linked_group_ids_with_one_label_array_is_just_that_labelling():
    labels = np.array([5, 7, 5, 9])
    groups = linked_group_ids(labels)
    assert groups[0] == groups[2]
    assert len(set(groups.tolist())) == 3
