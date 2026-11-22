"""Tests for astrometry_corr_gradient.crossmatch.select_xray_catalog_matches.

Fixtures use only the columns the function reads, matching the real
astromon_obs / astromon_xray_src / astromon_cat_src schemas from
absolute_astrometry/data/astromon.h5. y_angle/z_angle/r_angle/dr are in
arcsec, matching the real table's units.
"""

import numpy as np
from astropy.table import Table

from astrometry_corr_gradient.crossmatch import (
    CALIBRATION_CATALOG_HIERARCHY,
    EXCLUDED_CATALOG_SOURCES,
    bad_catalog_source_mask,
    bad_obsid_mask,
    bad_target_mask,
    ensure_x_id_column,
    off_nominal_sim_z_mask,
    rebasable_rows,
    select_xray_catalog_matches,
)

CATALOGS = ["ICRF3", "RFC", "Tycho2"]

COMMON_KWARGS = {
    "catalogs": CATALOGS,
    "detect_method": "gaussian_detect",
    "snr_min": 9.5,
    "r_angle_max_arcsec": 160.0,
    "r_angle_grating_max_arcsec": 160.0,
    "dr_max_arcsec": 3.0,
    "near_neighbor_dist_min_arcsec": 6.0,
}

XRAY_SRC_DEFAULTS = {
    "y_angle": 0.0,
    "z_angle": 0.0,
    "r_angle": 5.0,
    "snr": 20.0,
    "near_neighbor_dist": 10.0,
    "acis_streak": False,
    "grating_arm": False,
    "brightest": False,
    "detect_method": "gaussian_detect",
}

CAT_SRC_DEFAULTS = {
    "y_angle": 0.0,
    "z_angle": 0.0,
}


def make_obs(rows):
    return Table(rows=rows)


def _angle_to_sky_position(row):
    """Fill in ra/dec from y_angle/z_angle (arcsec proxy) unless given.

    remap_candidate_x_id matches on ra/dec, not y_angle/z_angle, so distinct
    sources need distinct positions here even though these tests otherwise
    care only about the y_angle/z_angle-derived dy/dz/dr.
    """
    row = dict(row)
    row.setdefault("ra", row["y_angle"] / 3600.0)
    row.setdefault("dec", row["z_angle"] / 3600.0)
    return row


def xray_src_row(obsid, source_id, **overrides):
    row = {"obsid": obsid, "id": source_id, **XRAY_SRC_DEFAULTS, **overrides}
    return _angle_to_sky_position(row)


def make_xray_src(rows):
    return Table(rows=rows)


def cat_src_row(obsid, cat_id, x_id, catalog, **overrides):
    row = {
        "obsid": obsid,
        "id": cat_id,
        "x_id": x_id,
        "catalog": catalog,
        **CAT_SRC_DEFAULTS,
        **overrides,
    }
    return _angle_to_sky_position(row)


def make_cat_src(rows):
    return Table(rows=rows)


def test_prefers_earlier_catalog_in_hierarchy_over_smaller_separation():
    obs = make_obs([{"obsid": 1, "grating": "NONE"}])
    xray_src = make_xray_src([xray_src_row(1, 10)])
    cat_src = make_cat_src(
        [
            # closer separation but lower hierarchy priority than ICRF3
            cat_src_row(1, 100, 10, "RFC", z_angle=0.0002),
            cat_src_row(1, 101, 10, "ICRF3", z_angle=0.0005),
        ]
    )
    result = select_xray_catalog_matches(obs, xray_src, cat_src, **COMMON_KWARGS)
    assert len(result) == 1
    assert result["catalog"][0] == "ICRF3"


def test_excludes_source_below_snr_floor():
    obs = make_obs([{"obsid": 1, "grating": "NONE"}])
    xray_src = make_xray_src([xray_src_row(1, 10, snr=5.0)])
    cat_src = make_cat_src([cat_src_row(1, 100, 10, "RFC")])
    result = select_xray_catalog_matches(obs, xray_src, cat_src, **COMMON_KWARGS)
    assert len(result) == 0


def test_excludes_separation_at_or_beyond_dr_max():
    obs = make_obs([{"obsid": 1, "grating": "NONE"}])
    xray_src = make_xray_src([xray_src_row(1, 10)])
    # dz of 3.5" is beyond dr_max_arcsec=3.0
    cat_src = make_cat_src([cat_src_row(1, 100, 10, "RFC", z_angle=3.5)])
    result = select_xray_catalog_matches(obs, xray_src, cat_src, **COMMON_KWARGS)
    assert len(result) == 0


def test_r_angle_threshold_differs_for_grating_observations():
    obs = make_obs([{"obsid": 1, "grating": "NONE"}, {"obsid": 2, "grating": "HETG"}])
    xray_src = make_xray_src(
        [
            xray_src_row(1, 10, r_angle=150.0),
            xray_src_row(2, 20, r_angle=150.0),
        ]
    )
    cat_src = make_cat_src(
        [
            cat_src_row(1, 100, 10, "RFC"),
            cat_src_row(2, 200, 20, "RFC"),
        ]
    )
    # r_angle=150" clears the default 160" limit for both; tighten the
    # grating limit below 150" to see only the grating obsid excluded.
    kwargs = COMMON_KWARGS | {"r_angle_grating_max_arcsec": 24.0}
    result = select_xray_catalog_matches(obs, xray_src, cat_src, **kwargs)
    assert set(result["obsid"]) == {1}


def test_excludes_acis_streak_unless_brightest():
    obs = make_obs([{"obsid": 1, "grating": "NONE"}])
    xray_src = make_xray_src(
        [
            xray_src_row(1, 10, acis_streak=True),
            xray_src_row(
                1, 11, acis_streak=True, brightest=True, y_angle=100.0, z_angle=100.0
            ),
        ]
    )
    cat_src = make_cat_src(
        [
            cat_src_row(1, 100, 10, "RFC"),
            cat_src_row(1, 101, 11, "RFC", y_angle=100.0, z_angle=100.0),
        ]
    )
    result = select_xray_catalog_matches(obs, xray_src, cat_src, **COMMON_KWARGS)
    assert set(result["x_id"]) == {11}


def test_excludes_grating_arm_sources():
    obs = make_obs([{"obsid": 1, "grating": "NONE"}])
    xray_src = make_xray_src([xray_src_row(1, 10, grating_arm=True)])
    cat_src = make_cat_src([cat_src_row(1, 100, 10, "RFC")])
    result = select_xray_catalog_matches(obs, xray_src, cat_src, **COMMON_KWARGS)
    assert len(result) == 0


def test_excludes_near_neighbor_sources():
    obs = make_obs([{"obsid": 1, "grating": "NONE"}])
    xray_src = make_xray_src([xray_src_row(1, 10, near_neighbor_dist=5.0)])
    cat_src = make_cat_src([cat_src_row(1, 100, 10, "RFC")])
    result = select_xray_catalog_matches(obs, xray_src, cat_src, **COMMON_KWARGS)
    assert len(result) == 0


def test_filters_to_requested_detect_method():
    obs = make_obs([{"obsid": 1, "grating": "NONE"}])
    xray_src = make_xray_src([xray_src_row(1, 10, detect_method="celldetect")])
    cat_src = make_cat_src([cat_src_row(1, 100, 10, "RFC")])
    result = select_xray_catalog_matches(obs, xray_src, cat_src, **COMMON_KWARGS)
    assert len(result) == 0


def test_ignores_catalogs_not_in_hierarchy():
    obs = make_obs([{"obsid": 1, "grating": "NONE"}])
    xray_src = make_xray_src([xray_src_row(1, 10)])
    cat_src = make_cat_src([cat_src_row(1, 100, 10, "GaiaVarStar")])
    result = select_xray_catalog_matches(obs, xray_src, cat_src, **COMMON_KWARGS)
    assert len(result) == 0


def test_bad_target_mask_flags_only_excluded_targets():
    targets = np.array(["zeta Tau", "Perseus cluster", "zeta Tau", "Eta Car"])
    mask = bad_target_mask(targets)
    np.testing.assert_array_equal(mask, [True, False, True, False])


def test_bad_target_mask_flags_the_blended_counterparts():
    # RW Aur (a 1.4" binary) and HD 97950 C (the NGC 3603 core) are excluded;
    # NGC 3603 itself is not
    targets = np.array(["RW Aur", "HD 97950 C", "NGC 3603", "RW Aur"])
    mask = bad_target_mask(targets)
    np.testing.assert_array_equal(mask, [True, True, False, True])


def test_bad_catalog_source_mask_flags_only_the_listed_catalog_sources():
    # 4774-931-1 is the flagged Trapezium star; 4774-933-1, in the same field, is
    # not; the same name under another catalog is a different source
    catalogs = np.array(["Tycho2", "Tycho2", "GaiaAGN", "RFC"])
    names = np.array(["4774-931-1", "4774-933-1", "4774-931-1", "RFC J0615+7102"])
    mask = bad_catalog_source_mask(catalogs, names)
    np.testing.assert_array_equal(mask, [True, False, False, True])


def test_excluded_catalog_sources_are_catalog_name_pairs():
    assert all(
        isinstance(catalog, str) and isinstance(name, str)
        for catalog, name in EXCLUDED_CATALOG_SOURCES
    )
    assert len(EXCLUDED_CATALOG_SOURCES) == 27


def test_bad_catalog_source_mask_flags_the_point_source_review_entries():
    # Mrk 1216 and HCG 40 are not point sources; EX Hya's GaiaQSO entry is at Gaia's
    # epoch, not the observation's (proper motion); the same names elsewhere are kept
    catalogs = np.array(["MilliquasGaia", "GaiaQSO", "DESIV161", "GaiaAGN"])
    names = np.array(
        [
            "MCG -1-22-013",
            "GaiaQSO-6185040879503491584",
            "DESIV161-39627672468325895",
            "MCG -1-22-013",
        ]
    )
    np.testing.assert_array_equal(
        bad_catalog_source_mask(catalogs, names), [True, True, True, False]
    )


def test_calibration_catalog_hierarchy_drops_quaia_and_gaiavarstar_only():
    assert CALIBRATION_CATALOG_HIERARCHY == [
        "ICRF3",
        "RFC",
        "GaiaAGN",
        "DESIV161",
        "MilliquasGaia",
        "GaiaQSO",
        "Tycho2",
    ]


def test_select_falls_to_the_next_catalog_without_quaia():
    # one X-ray source with a Quaia and a GaiaQSO counterpart: without Quaia
    # in the list, GaiaQSO's match is kept
    obs = make_obs([{"obsid": 1, "grating": "NONE"}])
    xray_src = make_xray_src([xray_src_row(1, 10)])
    cat_src = make_cat_src(
        [cat_src_row(1, 100, 10, "Quaia"), cat_src_row(1, 101, 10, "GaiaQSO")]
    )
    kwargs = {**COMMON_KWARGS, "catalogs": CALIBRATION_CATALOG_HIERARCHY}
    result = select_xray_catalog_matches(obs, xray_src, cat_src, **kwargs)
    assert list(result["catalog"]) == ["GaiaQSO"]


def test_bad_catalog_source_mask_flags_eso_253_3s_milliquas_entry():
    mask = bad_catalog_source_mask(
        np.array(["MilliquasGaia", "MilliquasGaia"]),
        np.array(["ESO 253-3", "ESO 253-G003"]),
    )
    np.testing.assert_array_equal(mask, [True, False])


def test_bad_catalog_source_mask_flags_sn2026sqfs_gaiaqso_match_only():
    # a quasar candidate inside the SN's host galaxy; other GaiaQSO entries are kept
    mask = bad_catalog_source_mask(
        np.array(["GaiaQSO", "GaiaQSO"]),
        np.array(["GaiaQSO-849350162404966784", "GaiaQSO-4658434504170340352"]),
    )
    np.testing.assert_array_equal(mask, [True, False])


def test_bad_catalog_source_mask_flags_ngc_3393s_nucleus_but_not_its_field_star():
    # the nucleus sits in extended soft emission; the Tycho2 star in the same fields is point-like
    mask = bad_catalog_source_mask(
        np.array(["MilliquasGaia", "Tycho2"]), np.array(["NGC 3393", "6638-527-1"])
    )
    np.testing.assert_array_equal(mask, [True, False])


def test_bad_obsid_mask_flags_only_excluded_obsids():
    # 9688 is another Ar Lac pointing, processed with its own detector's entry
    obsids = np.array([9687, 9688, 9687, 1])
    mask = bad_obsid_mask(obsids)
    np.testing.assert_array_equal(mask, [True, False, True, False])


def test_bad_obsid_mask_flags_the_mkn_421_pointings_given_the_acis_s_entry():
    # 11963 and 11967 are at 9687's SIM position, labeled ACIS-I, and got the
    # ACIS-S entry; 11964 is not excluded
    obsids = np.array([11963, 11964, 11967])
    mask = bad_obsid_mask(obsids)
    np.testing.assert_array_equal(mask, [True, False, True])


def test_off_nominal_sim_z_mask_flags_rows_beyond_four_mm_of_each_detectors_nominal():
    # nominal SIM-Z: ACIS-I -233.587, ACIS-S -190.143, HRC-I 126.983, HRC-S 250.466
    detectors = np.array(["ACIS-I", "ACIS-I", "ACIS-S", "ACIS-S", "HRC-S", "HRC-I"])
    sim_z = np.array([-233.587, -225.9, -186.143, -200.1, 258.5, 127.0])
    mask = off_nominal_sim_z_mask(detectors, sim_z)
    np.testing.assert_array_equal(mask, [False, True, False, True, True, False])


def test_off_nominal_sim_z_mask_keeps_an_offset_of_exactly_the_limit():
    # astromon keeps |offset| <= limit
    detectors = np.array(["ACIS-S", "ACIS-S"])
    sim_z = np.array([-190.143 + 4.0, -190.143 - 4.0])
    np.testing.assert_array_equal(
        off_nominal_sim_z_mask(detectors, sim_z), [False, False]
    )


def test_off_nominal_sim_z_mask_takes_another_limit():
    detectors = np.array(["ACIS-I", "ACIS-I"])
    sim_z = np.array([-233.587 + 0.5, -233.587 + 2.0])
    mask = off_nominal_sim_z_mask(detectors, sim_z, max_offset_mm=1.0)
    np.testing.assert_array_equal(mask, [False, True])


def test_off_nominal_sim_z_mask_flags_a_missing_sim_z_and_an_unknown_detector():
    # neither can be checked, so neither is kept
    detectors = np.array(["ACIS-S", "MYSTERY"])
    sim_z = np.array([np.nan, 0.0])
    np.testing.assert_array_equal(
        off_nominal_sim_z_mask(detectors, sim_z), [True, True]
    )


def test_ensure_x_id_column_renames_celldetect_x_id():
    cat_src = Table(
        rows=[(1, 100, 10, "RFC")], names=["obsid", "id", "celldetect_x_id", "catalog"]
    )
    result = ensure_x_id_column(cat_src)
    assert "x_id" in result.colnames
    assert "celldetect_x_id" not in result.colnames
    assert list(result["x_id"]) == [10]


def test_ensure_x_id_column_leaves_existing_x_id_alone():
    cat_src = Table(
        rows=[(1, 100, 10, "RFC")], names=["obsid", "id", "x_id", "catalog"]
    )
    result = ensure_x_id_column(cat_src)
    assert list(result["x_id"]) == [10]
    assert "celldetect_x_id" not in result.colnames


EVERY_GOOD_ALIGN_FILE = [
    "pcadD2012-09-13alignN0009.fits",
    "pcadD2013-01-19alignN0009.fits",
    "pcadD2013-01-19alignN0010.fits",
]


def test_rebasable_rows_drops_unrecorded_and_excluded_versions():
    versions = np.array(["4.9.3", "0.0", "4.12.6", "4.12.6.", "4.12.5"])
    np.testing.assert_array_equal(
        rebasable_rows(versions, excluded_versions=["4.12.6"]),
        [True, False, False, False, True],
    )


def test_rebasable_rows_keeps_every_recorded_version_by_default():
    versions = np.array(["4.9.3", "0.0", "4.12.6"])
    np.testing.assert_array_equal(rebasable_rows(versions), [True, False, True])


PROCESSING_MATRICES = {1: (0.1, 0.3), 2: (9.0, 9.0), 3: (1.5, -1.0)}
