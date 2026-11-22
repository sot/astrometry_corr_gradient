"""Tests for astrometry_corr_gradient.crossmatch.inconsistent_obsid_mask.

Cross-matches from the same obsid should report a very similar dy/dz --
one boresight offset per pointing, not a per-source quantity. A large
disagreement between an obsid's own matches means at least one of them
is a bad cross-match (found directly this session: obsid 25128, "HD
93129A", likely a false match to its close companion HD 93129B, sitting
in the crowded Trumpler 14 cluster field -- confirmed via the actual
X-ray event image, which shows dozens of distinct sources within an
arcminute).
"""

import numpy as np

from astrometry_corr_gradient.crossmatch import inconsistent_obsid_mask


def test_a_single_match_obsid_is_never_flagged():
    obsids = np.array([1, 2, 2])
    dy = np.array(
        [0.1, 0.1, 5.0]
    )  # obsid 2 has a huge spread, but obsid 1 has nothing to compare
    dz = np.array([0.1, 0.1, 0.1])
    mask = inconsistent_obsid_mask(obsids, dy, dz, threshold=0.75)
    assert not mask[0]  # obsid 1: lone match, never flagged


def test_consistent_matches_are_not_flagged():
    obsids = np.array([1, 1, 1])
    dy = np.array([0.10, 0.15, 0.05])
    dz = np.array([-0.05, 0.00, 0.05])
    mask = inconsistent_obsid_mask(obsids, dy, dz, threshold=0.75)
    np.testing.assert_array_equal(mask, [False, False, False])


def test_an_obsid_with_one_outlier_flags_every_row_in_that_obsid():
    # matches this project's actual finding: obsid 25128 had three matches
    # (dz = -2.65, +0.27, -0.58) -- x_id=16 is the clear outlier, but the
    # whole obsid gets flagged (Jean's call: safer than guessing which is bad).
    obsids = np.array([1, 1, 1, 2, 2])
    dy = np.array([0.50, 0.60, 0.45, 0.10, 0.12])
    dz = np.array([-2.65, 0.27, -0.58, -0.05, 0.05])
    mask = inconsistent_obsid_mask(obsids, dy, dz, threshold=0.75)
    np.testing.assert_array_equal(mask, [True, True, True, False, False])


def test_other_obsids_are_unaffected_by_one_bad_obsid():
    obsids = np.array([1, 1, 2, 2, 2])
    dy = np.array([0.1, 5.0, 0.1, 0.2, 0.15])  # obsid 1 is bad, obsid 2 is fine
    dz = np.array([0.1, 0.1, 0.0, 0.05, -0.05])
    mask = inconsistent_obsid_mask(obsids, dy, dz, threshold=0.75)
    np.testing.assert_array_equal(mask, [True, True, False, False, False])


def test_threshold_controls_sensitivity():
    # deviation from the 2-point median is half the total spread (0.45 here)
    obsids = np.array([1, 1])
    dy = np.array([0.0, 0.9])
    dz = np.array([0.0, 0.0])
    lenient = inconsistent_obsid_mask(obsids, dy, dz, threshold=0.5)
    strict = inconsistent_obsid_mask(obsids, dy, dz, threshold=0.4)
    assert not lenient.any()
    assert strict.all()


def test_flags_on_dz_deviation_alone_even_if_dy_agrees():
    obsids = np.array([1, 1])
    dy = np.array([0.1, 0.12])  # dy agrees fine
    dz = np.array([0.1, 2.0])  # dz disagrees badly
    mask = inconsistent_obsid_mask(obsids, dy, dz, threshold=0.75)
    assert mask.all()
