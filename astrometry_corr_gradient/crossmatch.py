"""X-ray source to catalog counterpart cross-matching, and the rebase onto each detector's latest CALALIGN matrix.

Copied from astrometry's ``astrometry/match.py``, without its astromon- and
CALDB-based rebase: :func:`fetch_crossmatches` rebases with this package's
own copy of the CALALIGN matrices (:mod:`astrometry_corr_gradient.calalign`),
so the only input is the astromon database.

``select_xray_catalog_matches`` is a self-contained reimplementation of the
"simple" cross-match algorithm in astromon_project's
``astromon.cross_match.simple_cross_match``, including the per-detect_method
``x_id`` remap (``remap_candidate_x_id``, copied from
``cross_match.remap_x_id_to_sources``) that step depends on, with the
astromon_27 selection parameters as named module-level constants. Parameter
values are copied from ``astromon_project/astromon/cross_match.py``
(``_MATCH_HIERARCHY`` and the ``CROSS_MATCHES_ARGS["astromon_27"]`` entry),
current as of 2026-08-31.
"""

from collections.abc import Iterable
from itertools import pairwise

import numpy as np
from astropy import table
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table

# Catalog priority order for the astromon_24-27 hierarchy: for each X-ray
# source, the match from the earliest-listed catalog wins over any other
# catalog's match, regardless of angular separation.
ASTROMON_27_CATALOG_HIERARCHY = [
    "ICRF3",
    "RFC",
    "GaiaAGN",
    "Quaia",
    "DESIV161",
    "MilliquasGaia",
    "GaiaQSO",
    "GaiaVarStar",
    "Tycho2",
]

# The calibration's own catalogs, in the same priority order: astromon_27's
# without Quaia and GaiaVarStar (Jean, 2026-11-04: "Those are silly in this
# context"). Quaia's quasars are Gaia DR3 quasar candidates that GaiaQSO,
# MilliquasGaia or DESIV161 mostly hold as well, and GaiaVarStar's stars are
# variables; a source either falls to the next catalog that has it or drops.
CALIBRATION_CATALOG_HIERARCHY = [
    name
    for name in ASTROMON_27_CATALOG_HIERARCHY
    if name not in {"Quaia", "GaiaVarStar"}
]

# The calibration's detection method: celldetect, which the report's fetch
# already uses (Jean, 2026-10-30; the ecsv path too since 2026-11-02, OPEN_QUESTIONS
# item 8). astromon's own astromon_27 recipe uses gaussian_detect at SNR > 9.5
# (ASTROMON_27_SNR_MIN); gaussian_detect is slightly more accurate where the two
# disagree but ties overall (PROGRESS.md section 87).
ASTROMON_27_DETECT_METHOD = "celldetect"

# SNR floors are method-specific: gaussian_detect SNR runs ~13x celldetect
# SNR for the same source (different noise definitions), so 9.5 here is the
# quantile-equivalent of celldetect's floor of 3.
ASTROMON_27_SNR_MIN = 9.5
ASTROMON_27_SNR_MIN_CELLDETECT = 3.0

# Off-axis limit extended to 160" (astromon_26/27) from the earlier 120"
# (astromon_21-25): the 160-180" bin shows elevated centroid scatter, so the
# limit stops at 160" rather than 180".
ASTROMON_27_R_ANGLE_MAX_ARCSEC = 160.0
ASTROMON_27_R_ANGLE_GRATING_MAX_ARCSEC = 160.0

ASTROMON_27_DR_MAX_ARCSEC = 3.0
ASTROMON_27_NEAR_NEIGHBOR_DIST_MIN_ARCSEC = 6.0


def remap_candidate_x_id(cat_src: Table, xray_src: Table) -> Table:
    """Re-point each catalog candidate's ``x_id`` at the nearest source in ``xray_src``.

    ``astromon_cat_src`` stores one ``x_id`` per catalog source, assigned once
    against celldetect positions when the rough match was first computed.
    Matching against any other detect_method (e.g. ``gaussian_detect``, which
    has its own, differently-numbered source list for the same obsid)
    requires re-matching each catalog candidate to that method's source ids by
    sky position first -- copied from astromon_project's
    ``cross_match.remap_x_id_to_sources``. The nearest-neighbor match is done
    independently per obsid, since source ids restart at 0 for each obsid.

    Parameters
    ----------
    cat_src : astropy.table.Table
        Catalog candidates with ``obsid``, ``ra``, ``dec``, ``x_id``.
    xray_src : astropy.table.Table
        X-ray sources of a single detect_method, with ``obsid``, ``id``,
        ``ra``, ``dec``.

    Returns
    -------
    astropy.table.Table
        A copy of ``cat_src`` with ``x_id`` re-pointed at ``xray_src``. Rows
        whose obsid has no ``xray_src`` entries get ``x_id = -1`` (so they
        will not join to anything downstream).
    """
    remapped = cat_src.copy()
    if len(remapped) == 0:
        return remapped

    x_id = np.full(len(remapped), -1, dtype=xray_src["id"].dtype)
    for obsid in np.unique(remapped["obsid"]):
        cat_mask = remapped["obsid"] == obsid
        src_mask = xray_src["obsid"] == obsid
        if not np.any(src_mask):
            continue
        candidates = remapped[cat_mask]
        sources = xray_src[src_mask]
        nearest, _, _ = SkyCoord(
            candidates["ra"] * u.deg, candidates["dec"] * u.deg
        ).match_to_catalog_sky(SkyCoord(sources["ra"] * u.deg, sources["dec"] * u.deg))
        x_id[cat_mask] = np.asarray(sources["id"])[nearest]

    remapped["x_id"] = x_id
    return remapped


def select_xray_catalog_matches(
    astromon_obs: Table,
    astromon_xray_src: Table,
    astromon_cat_src: Table,
    catalogs: list[str],
    detect_method: str,
    snr_min: float,
    r_angle_max_arcsec: float,
    r_angle_grating_max_arcsec: float,
    dr_max_arcsec: float,
    near_neighbor_dist_min_arcsec: float,
) -> Table:
    """Select one best catalog counterpart per X-ray source.

    Replicates astromon_project's ``simple_cross_match`` selection: filter
    X-ray sources by detection method, SNR, isolation, and streak/grating-arm
    flags; filter catalog counterparts to the requested catalogs; keep pairs
    within the separation and off-axis limits; and for X-ray sources with
    more than one surviving counterpart, keep only the one from the
    highest-priority catalog (ties broken by angular separation).

    Parameters
    ----------
    astromon_obs : astropy.table.Table
        Observation metadata, keyed by ``obsid``. Must have ``grating``.
    astromon_xray_src : astropy.table.Table
        Detected X-ray sources. Must have ``obsid``, ``id``, ``ra``, ``dec``,
        ``y_angle``, ``z_angle``, ``r_angle``, ``snr``, ``near_neighbor_dist``,
        ``acis_streak``, ``grating_arm``, ``brightest``, ``detect_method``.
    astromon_cat_src : astropy.table.Table
        Catalog counterparts, rough-matched to X-ray sources. Must have
        ``obsid``, ``id``, ``x_id``, ``catalog``, ``ra``, ``dec``, ``y_angle``,
        ``z_angle``. Its ``x_id`` is assumed to be celldetect-linked (as
        stored in astromon_cat_src) and is re-pointed at ``detect_method``'s
        source ids internally via :func:`remap_candidate_x_id` -- the caller
        does not need to do this.
    catalogs : list of str
        Catalog names in priority order; the earliest-listed catalog wins
        when an X-ray source has counterparts in more than one.
    detect_method : str
        Only X-ray sources with this ``detect_method`` are considered.
    snr_min : float
        Minimum signal-to-noise ratio (selects ``snr`` strictly above this).
    r_angle_max_arcsec : float
        Maximum off-axis angle in arcsec for non-grating observations.
    r_angle_grating_max_arcsec : float
        Maximum off-axis angle in arcsec for grating observations.
    dr_max_arcsec : float
        Maximum angular separation in arcsec between X-ray source and
        catalog counterpart.
    near_neighbor_dist_min_arcsec : float
        Minimum distance in arcsec to the nearest other X-ray source
        (excludes crowded/blended detections).

    Returns
    -------
    astropy.table.Table
        One row per selected (obsid, x_id) pair, with ``catalog``, ``dy``,
        ``dz``, ``dr`` (all in degrees for dy/dz/dr) among its columns.

    Examples
    --------
    >>> from astropy.table import Table
    >>> obs = Table(rows=[(1, "NONE")], names=["obsid", "grating"])
    >>> xray_src = Table(
    ...     rows=[(1, 10, 0.0, 0.0, 0.0, 0.0, 5.0, 20.0, 10.0, False, False, False, "gaussian_detect")],
    ...     names=["obsid", "id", "ra", "dec", "y_angle", "z_angle", "r_angle", "snr",
    ...            "near_neighbor_dist", "acis_streak", "grating_arm", "brightest",
    ...            "detect_method"],
    ... )
    >>> cat_src = Table(
    ...     rows=[(1, 100, 10, "RFC", 0.0, 0.0, 0.0, 0.0001)],
    ...     names=["obsid", "id", "x_id", "catalog", "ra", "dec", "y_angle", "z_angle"],
    ... )
    >>> result = select_xray_catalog_matches(
    ...     obs, xray_src, cat_src,
    ...     catalogs=["RFC"], detect_method="gaussian_detect", snr_min=9.5,
    ...     r_angle_max_arcsec=160.0, r_angle_grating_max_arcsec=160.0,
    ...     dr_max_arcsec=3.0, near_neighbor_dist_min_arcsec=6.0,
    ... )
    >>> len(result)
    1
    """
    xray_src_method = astromon_xray_src[
        astromon_xray_src["detect_method"] == detect_method
    ]

    cat_src = astromon_cat_src[np.isin(astromon_cat_src["catalog"], catalogs)]
    # astromon_cat_src's stored x_id is celldetect-linked; re-point it at this
    # detect_method's source ids before filtering/joining on x_id.
    cat_src = remap_candidate_x_id(cat_src, xray_src_method)

    arm_flag = xray_src_method["grating_arm"].astype(bool)
    brightest_flag = xray_src_method["brightest"].astype(bool)
    xray_src = xray_src_method[
        (xray_src_method["snr"] > snr_min)
        & (xray_src_method["near_neighbor_dist"] > near_neighbor_dist_min_arcsec)
        & (~xray_src_method["acis_streak"].astype(bool) | brightest_flag)
        & (~arm_flag)
    ]

    if len(xray_src) == 0 or len(cat_src) == 0:
        return astromon_xray_src[:0]

    xray_src = xray_src.copy()
    xray_src.rename_columns(
        ["id", "ra", "dec", "y_angle", "z_angle"],
        ["x_id", "x_ra", "x_dec", "x_y_angle", "x_z_angle"],
    )
    cat_src = cat_src.copy()
    cat_src.rename_columns(
        ["id", "ra", "dec", "y_angle", "z_angle"],
        ["c_id", "c_ra", "c_dec", "c_y_angle", "c_z_angle"],
    )

    matches = table.join(astromon_obs, xray_src, keys=["obsid"])
    matches = table.join(
        matches, cat_src, keys=["obsid", "x_id"], table_names=["xray", "cat"]
    )

    if len(matches) == 0:
        return matches

    matches["dz"] = matches["x_z_angle"] - matches["c_z_angle"]
    matches["dy"] = matches["x_y_angle"] - matches["c_y_angle"]
    matches["dr"] = np.sqrt(matches["dy"] ** 2 + matches["dz"] ** 2)

    catalog_priority = {name: i for i, name in enumerate(catalogs)}
    matches["cat_order"] = [catalog_priority[c] for c in matches["catalog"]]

    matches = matches[
        ((matches["grating"] == "NONE") & (matches["r_angle"] < r_angle_max_arcsec))
        | (
            (matches["grating"] != "NONE")
            & (matches["r_angle"] < r_angle_grating_max_arcsec)
        )
    ]
    matches = matches[matches["dr"] < dr_max_arcsec]

    if len(matches) == 0:
        return matches

    grouped = matches.group_by(["obsid", "x_id"])
    indices = grouped.groups.indices
    best_row_indices = []
    for i0, i1 in pairwise(indices):
        order = np.lexsort((grouped["dr"][i0:i1], grouped["cat_order"][i0:i1]))
        best_row_indices.append(i0 + order[0])

    return grouped[best_row_indices]


def inconsistent_obsid_mask(
    obsids: np.ndarray,
    dy: np.ndarray,
    dz: np.ndarray,
    threshold: float = 0.75,
) -> np.ndarray:
    """Flag every row belonging to an obsid whose cross-matches disagree.

    Multiple cross-matches from the SAME obsid should report a very
    similar dy/dz -- one boresight offset per pointing, not a
    per-source quantity. A large disagreement means at least one match
    is a bad cross-match: found directly this session, obsid 25128
    ("HD 93129A") had three matches with dz = -2.65, +0.27, -0.58 --
    likely a false match to its close companion star HD 93129B, sitting
    in the crowded Trumpler 14 cluster field (confirmed via the actual
    X-ray event image, which shows dozens of distinct sources within an
    arcminute of each other).

    Flags every row in an affected obsid, not just the specific outlier
    -- with only a handful of affected obsids in a large sample, it's
    safer to exclude all of an obsid's matches than to guess which one
    is the bad one.

    Parameters
    ----------
    obsids, dy, dz : numpy.ndarray
        Same length, one entry per cross-match row.
    threshold : float
        An obsid is flagged if any of its rows deviates from that
        obsid's own median dy or dz by more than this, in the same
        units as ``dy``/``dz`` (arcsec, in this project). Default 0.75
        arcsec is comfortably above this project's ~0.15-0.27 arcsec
        measurement floor from repeat measurements.

    Returns
    -------
    numpy.ndarray
        Boolean mask, True for every row belonging to a flagged obsid.
    """
    flagged = np.zeros(len(obsids), dtype=bool)
    for obsid in np.unique(obsids):
        rows = np.where(obsids == obsid)[0]
        if len(rows) < 2:
            continue
        dy_deviation = np.abs(dy[rows] - np.nanmedian(dy[rows]))
        dz_deviation = np.abs(dz[rows] - np.nanmedian(dz[rows]))
        if np.any(np.maximum(dy_deviation, dz_deviation) > threshold):
            flagged[rows] = True
    return flagged


# Targets excluded outright, regardless of obsid or catalog: found this
# session to dominate a handful of matches with values far outside the
# rest of their era, dragging binned medians on nothing but their own
# say-so. `select_xray_catalog_matches` has no category/target exclusion
# of its own (unlike astromon_project's `simple_cross_match`, which
# supports `exclude_categories`/`bad_sources.ecsv` -- neither is wired up
# here), so this is the one place that gap is closed for astromon_27.
#
# zeta Tau: a Be star with a known circumstellar disk -- variable,
# extended X-ray emission is expected, not a clean point source. Only 4
# rows in the 2025.5+ tail, but the most extreme single value there
# (dy median -1.14 arcsec, vs. -0.10 to -0.43 for every other target in
# the same window).
#
# RW Aur and HD 97950 C: both on astromon's `bad_sources.ecsv`, and both
# with an optical counterpart that is not the X-ray source (Jean,
# 2026-10-29). RW Aur is a T Tauri binary 1.4" apart whose secondary, B, is
# the brighter and variable X-ray source, so the X-ray centroid can follow
# B while the Tycho2 position follows A: its median corrected offset is
# 1.31" over 11 obsids (2015-2021), the 3rd largest of 419 targets with at
# least 4 rows, with dz changing sign between obsids. HD 97950 C sits at
# the center of NGC 3603's core of ~20 O and Wolf-Rayet stars, so its
# Tycho2 position is a blend: 1.12" over 4 HETG obsids (2019.8), the 6th
# largest. Targets named "NGC 3603" are not excluded.
EXCLUDED_TARGETS = {"zeta Tau", "RW Aur", "HD 97950 C"}


def bad_target_mask(targets: np.ndarray) -> np.ndarray:
    """Flag every row whose target is in :data:`EXCLUDED_TARGETS`.

    Parameters
    ----------
    targets : numpy.ndarray
        One target name per cross-match row.

    Returns
    -------
    numpy.ndarray
        Boolean mask, True for every row belonging to an excluded target.
    """
    return np.isin(targets, list(EXCLUDED_TARGETS))


# Obsids excluded outright, because the rebase cannot back them out
# correctly (Jean, 2026-09-30; PROGRESS.md section 68). Each obsid's aspect
# products (acal1) record the CALALIGN matrix its processing applied.
#
# 9687: Ar Lac, a calibration pointing at an off-nominal SIM position
# (sim_z -213.59 mm, 20 mm from the ACIS-I aimpoint). The database calls it
# ACIS-I, but its aspect processing applied the ACIS-S entry of
# pcadD2006-12-10alignN0009, so backing out the ACIS-I entry leaves it off
# by dy -0.92, dz +0.46 arcsec. It is the only such obsid among the 6314
# sample obsids checked against their acal1; Ar Lac's other pointings got
# their own detector's entry.
#
# 11963 and 11967: Mkn 421 LETG pointings (2010-02-06) at the same SIM
# position, also labeled ACIS-I and given the ACIS-S entry -- found by a
# database-wide check (2026-09-30). Neither sample keeps any of their rows
# (the 09-24 celldetect fetch, the gaussian ecsv), so excluding them changes
# nothing today; it protects any wider selection (Jean, 2026-10-01).
EXCLUDED_OBSIDS = {9687, 11963, 11967}


def bad_obsid_mask(obsids: np.ndarray) -> np.ndarray:
    """Flag every row whose obsid is in :data:`EXCLUDED_OBSIDS`.

    Parameters
    ----------
    obsids : numpy.ndarray
        One obsid per cross-match row.

    Returns
    -------
    numpy.ndarray
        Boolean mask, True for every row belonging to an excluded obsid.

    Examples
    --------
    >>> bad_obsid_mask(np.array([9687, 9688])).tolist()
    [True, False]
    """
    return np.isin(obsids, list(EXCLUDED_OBSIDS))


# Catalog sources excluded wherever they appear, keyed by (catalog, name):
# their X-ray position sits at a fixed place on the sky that is not the
# catalog position -- a wrong or blended counterpart, lensed images, or
# extended emission -- rather than anything the calibration should model
# (Jean, 2026-10-29). Found by a roll test on the workbench's out-of-sample
# corrected offsets (09-24 celldetect snapshot): rotated by each
# observation's roll into east/north, a source's offsets agree (chi-square
# p > 0.01, per-row sigma from the whole sample by SNR x off-axis x detector)
# while its dy/dz do not (p < 0.05 / 946 sources tested); its rolls differ
# enough to tell the two frames apart (3 sigma); and its mean sky offset is
# at least 0.5 arcsec. Thresholds were fixed before looking at which
# sources met them. The comment after each entry is its target, sky offset
# and observation count. Excluding a catalog source keeps the field's other
# sources (Orion's other Trapezium star, 4774-933-1, is not flagged).
#
# One more, not from the roll test: NGC 3393's nucleus is not enough of a
# point source (Jean, 2026-10-30). Its soft X-rays extend ~3" either side
# along NE-SW; corrected and stacked, its events sit on the Gaia position
# (within 0.03"), but celldetect's position wanders 0.29" rms around it and
# sometimes splits off a second source in the NE lobe. The Tycho2 star in the
# same fields (6638-527-1) is point-like and stays (PROGRESS.md section 88).
#
# From the review of psfratio-flagged rows (PROGRESS.md section 90), by eye
# on per-observation images: entries whose match is not a point source at the
# catalog position. E0102-72.3's GaiaQSO entry sits inside the supernova
# remnant: the matched detection is a knot of its diffuse emission 1.5" away,
# with no source at the catalog position (Jean, 2026-10-30: "obviously bad").
# 3C305's GaiaAGN entry: the source is extended (Jean, 2026-10-30); celldetect
# matched a soft knot 2.75" NE of the nucleus, which sits at the catalog
# position (the only counts above 2 keV). A1068's GaiaAGN entry (the central
# galaxy): the X-ray peak is the cluster core, a broad patch with no compact
# nucleus in hard X-rays; celldetect lands on it 0.75" north of the catalog
# position on average, 0.66-1.41" in all five observations (Jean, 2026-10-30).
# Three more from the same review, on the point-source test's evidence
# (PROGRESS.md section 91; Jean, 2026-11-02): Mrk 1216's MilliquasGaia entry
# (a compact core in a diffuse halo that pulls celldetect about 0.4" NE) and
# HCG 40's DESIV161 entry (faint diffuse emission), neither a point source;
# and EX Hya's GaiaQSO entry, which is EX Hya at Gaia's 2016 epoch with no
# proper motion applied, about 1" from where the star was in 2007.
# One from the 2026 Q3 look (Jean, 2026-11-03): SN2026sqf's GaiaQSO entry, a
# G = 19 quasar candidate inside the crowded host galaxy (the pointing is at
# NGC 3310's position, ~6 X-ray sources within 0.5' of the aimpoint),
# matched 1.4-1.8" off in both observations (32536, 33233), while the
# galaxy's other X-ray sources sit 0.3-0.6" from 2MASS/SDSS objects: the
# match is wrong, not the pointing.
# One from the 2022 Q2 look (Jean, 2026-11-04): ESO 253-3's MilliquasGaia
# entry, the host of ASASSN-14ko. The galaxy has two AGN 1.7" apart (Tucker
# et al. 2021, arXiv:2011.05998), both seen by Chandra; the northern one is
# the periodic flarer, so the blend's centroid moves with the flare cycle.
# Its 13 observations sit 0.26-1.57" off after the correction, varying from
# one to the next (nine in 2022-05/06 set 2022 Q2's p90: 0.86" -> 0.58").
EXCLUDED_CATALOG_SOURCES = {
    ("Tycho2", "8327-1691-1"),  # XTE J1701-462, 2.90", 4
    ("RFC", "RFC J0047-2517"),  # NGC 253, 2.54", 2
    ("GaiaQSO", "GaiaQSO-2344030839048367232"),  # NGC 253, 2.53", 3
    ("MilliquasGaia", "NGC 7552"),  # NGC 7552, 1.61", 2
    ("Tycho2", "2389-955-1"),  # RW Aur, 1.42", 11 (also in EXCLUDED_TARGETS)
    ("RFC", "RFC J0615+7102"),  # Markarian 3, 1.23", 8
    ("MilliquasGaia", "NGC 4388"),  # NGC 4388, 1.08", 4
    ("GaiaAGN", "GaiaAGN-3284740650964769024"),  # MG0414+0534 (lensed), 0.88", 2
    ("MilliquasGaia", "UGC 8696"),  # Mrk 273, 0.82", 2
    ("Tycho2", "4774-931-1"),  # Orion Trapezium, 0.78", 26
    ("MilliquasGaia", "UGC 8327"),  # UGC 08327, 0.71", 2
    ("Tycho2", "3550-356-1"),  # 2MASS J19094043+4930050, 0.66", 2
    ("DESIV161", "DESIV161-39633289706737241"),  # NGC 4102, 0.63", 6
    ("Tycho2", "8997-2337-1"),  # HD 113904, 0.62", 4
    ("MilliquasGaia", "ATLAS J025942.91-163543.3"),  # ATLAS J0259-16, 0.61", 5
    ("GaiaAGN", "GaiaAGN-2704542589921820288"),  # Q2237+0305 (lensed), 0.55", 30
    ("RFC", "RFC J1833-210A"),  # PKS 1830-211 (lensed), 0.54", 13
    ("GaiaAGN", "GaiaAGN-3800591481917387392"),  # SDSS J1138+0314 (lensed), 0.53", 5
    ("MilliquasGaia", "NGC 3393"),  # NGC 3393, extended (not the roll test), 10
    ("GaiaQSO", "GaiaQSO-4690512893287832320"),  # E0102-72.3, in the SNR (review), 1
    ("GaiaAGN", "GaiaAGN-1620307580994907392"),  # 3C305, extended (review), 1
    ("GaiaAGN", "GaiaAGN-779253444275109504"),  # A1068, cluster core (review), 5
    ("MilliquasGaia", "MCG -1-22-013"),  # Mrk 1216, core in a halo (review), 5
    ("GaiaQSO", "GaiaQSO-6185040879503491584"),  # EX Hya, proper motion (review), 4
    ("DESIV161", "DESIV161-39627672468325895"),  # HCG 40, diffuse (review), 1
    ("GaiaQSO", "GaiaQSO-849350162404966784"),  # SN2026sqf, wrong match in its host, 2
    ("MilliquasGaia", "ESO 253-3"),  # ASASSN-14ko host, two AGN 1.7" apart, 13
}


def bad_catalog_source_mask(catalogs: np.ndarray, names: np.ndarray) -> np.ndarray:
    """Flag every row whose (catalog, name) is in :data:`EXCLUDED_CATALOG_SOURCES`.

    Parameters
    ----------
    catalogs, names : numpy.ndarray
        Each cross-match row's catalog and catalog source name.

    Returns
    -------
    numpy.ndarray
        Boolean mask, True for every row matched to an excluded catalog source.

    Examples
    --------
    >>> bad_catalog_source_mask(
    ...     np.array(["Tycho2", "Tycho2"]), np.array(["4774-931-1", "4774-933-1"])
    ... ).tolist()
    [True, False]
    """
    pairs = zip(np.asarray(catalogs).astype(str), np.asarray(names).astype(str))
    return np.array([pair in EXCLUDED_CATALOG_SOURCES for pair in pairs], dtype=bool)


# Each detector's nominal SIM-Z (mm), astromon's own (`cross_match.SIM_Z`).
NOMINAL_SIM_Z_MM = {
    "ACIS-I": -233.587,
    "ACIS-S": -190.143,
    "HRC-I": 126.983,
    "HRC-S": 250.466,
}
# Rows further than this from their detector's nominal SIM-Z are dropped
# (Jean, 2026-10-20; PROGRESS.md section 83). It is the cut astromon's own
# consumers apply (`web/celmon.py`, `scripts/calc_astrometry_rms.py`), which
# the astromon_27 selection reproduced here does not carry. Off nominal,
# ACIS-I's dy grows about +0.10 arcsec per mm and HRC-S's about -0.11, so
# the observations up to 20 mm off added a SIM-Z error no time-based
# correction can remove.
SIM_Z_OFFSET_MAX_MM = 4.0


def off_nominal_sim_z_mask(
    detectors: np.ndarray,
    sim_z: np.ndarray,
    max_offset_mm: float = SIM_Z_OFFSET_MAX_MM,
) -> np.ndarray:
    """Flag every row further than `max_offset_mm` from its detector's nominal SIM-Z.

    The nominal positions are :data:`NOMINAL_SIM_Z_MM`; an offset of exactly
    `max_offset_mm` is kept, as in astromon. A row with no SIM-Z, or on a
    detector with no nominal position, is flagged too, since it cannot be
    checked (astromon instead keeps the unknown detector; the database has
    none).

    Parameters
    ----------
    detectors : numpy.ndarray
        One detector name per cross-match row.
    sim_z : numpy.ndarray
        The observation's SIM-Z (mm), one per row.
    max_offset_mm : float
        The largest offset kept.

    Returns
    -------
    numpy.ndarray
        Boolean mask, True for every row to drop.

    Examples
    --------
    >>> off_nominal_sim_z_mask(np.array(["ACIS-I", "ACIS-I"]), np.array([-233.6, -225.9])).tolist()
    [False, True]
    """
    nominal = np.array(
        [NOMINAL_SIM_Z_MM.get(str(detector), np.nan) for detector in detectors],
        dtype=float,
    )
    offset = np.asarray(sim_z, dtype=float) - nominal
    return ~(np.abs(offset) <= max_offset_mm)


def ensure_x_id_column(astromon_cat_src: Table) -> Table:
    """Normalize astromon_cat_src's celldetect-linked id column to ``x_id``.

    Some database snapshots (found directly: a ``astromon_gaussian_redo.h5``
    snapshot taken mid-rebuild) store this column as ``celldetect_x_id`` --
    arguably the more precise name for what it actually is, since it is
    always celldetect-linked regardless of which detect_method is being
    matched against (see :func:`select_xray_catalog_matches`'s docstring)
    -- rather than the plain ``x_id`` this module otherwise assumes. Makes
    either naming convention work transparently rather than failing on
    whichever one a given snapshot happens to use.

    Parameters
    ----------
    astromon_cat_src : astropy.table.Table
        Catalog counterparts, as loaded from an astromon.h5 file.

    Returns
    -------
    astropy.table.Table
        ``astromon_cat_src`` unchanged if it already has ``x_id``; a copy
        with ``celldetect_x_id`` renamed to ``x_id`` otherwise.

    Examples
    --------
    >>> from astropy.table import Table
    >>> cat_src = Table(rows=[(1, 100, 10, "RFC")],
    ...                 names=["obsid", "id", "celldetect_x_id", "catalog"])
    >>> int(ensure_x_id_column(cat_src)["x_id"][0])
    10
    """
    if (
        "x_id" in astromon_cat_src.colnames
        or "celldetect_x_id" not in astromon_cat_src.colnames
    ):
        return astromon_cat_src
    renamed = astromon_cat_src.copy()
    renamed.rename_column("celldetect_x_id", "x_id")
    return renamed


def rebasable_rows(
    caldb_versions: np.ndarray, excluded_versions: Iterable[str] = ()
) -> np.ndarray:
    """Rows the rebase can use: a processing CALDB recorded, and not excluded.

    ``"0.0"`` means no CALDB version was recorded (not yet reprocessed), so
    there is no processing matrix to back out. ``excluded_versions`` leaves
    out processing versions whose alignment files cannot be verified (e.g.
    an SDP-only release with no public notes); a trailing ``.`` is ignored
    on both sides, as in :func:`missing_calalign_files`.

    Parameters
    ----------
    caldb_versions : numpy.ndarray
        Each row's processing ``caldb_version``.
    excluded_versions : iterable of str
        Processing versions to leave out.

    Returns
    -------
    numpy.ndarray
        Boolean mask, True for rows to rebase.

    Examples
    --------
    >>> rebasable_rows(np.array(["4.9.3", "0.0", "4.12.6"]), ["4.12.6"]).tolist()
    [True, False, False]
    """
    excluded = {"0.0", *(str(v).strip().rstrip(".") for v in excluded_versions)}
    normalized = np.array([str(v).strip().rstrip(".") for v in caldb_versions])
    return ~np.isin(normalized, sorted(excluded))


def fetch_crossmatches(
    db_path: str,
    catalogs: list[str] | None = None,
    detect_method: str = ASTROMON_27_DETECT_METHOD,
    snr_min: float = ASTROMON_27_SNR_MIN_CELLDETECT,
    r_angle_max_arcsec: float = ASTROMON_27_R_ANGLE_MAX_ARCSEC,
    r_angle_grating_max_arcsec: float = ASTROMON_27_R_ANGLE_GRATING_MAX_ARCSEC,
    dr_max_arcsec: float = ASTROMON_27_DR_MAX_ARCSEC,
    near_neighbor_dist_min_arcsec: float = ASTROMON_27_NEAR_NEIGHBOR_DIST_MIN_ARCSEC,
    *,
    exclude_caldb_versions: Iterable[str] = (),
    sim_z_offset_max_mm: float | None = SIM_Z_OFFSET_MAX_MM,
) -> Table:
    """Cross-match and rebase, straight from an astromon database.

    Loads ``astromon_obs``, ``astromon_xray_src`` and ``astromon_cat_src``
    from ``db_path`` (read only), calls :func:`select_xray_catalog_matches`,
    drops rows with no CALDB release recorded (``caldb_version == "0.0"``) or
    an excluded one, rebases every row onto its detector's latest CALALIGN
    matrix with this package's copy of the CALALIGN matrices
    (:func:`astrometry_corr_gradient.calalign.rebase`), and drops
    :data:`EXCLUDED_TARGETS`, :data:`EXCLUDED_OBSIDS`,
    :data:`EXCLUDED_CATALOG_SOURCES` and observations off nominal SIM-Z.

    Parameters
    ----------
    db_path : str
        Path to an ``astromon.h5`` file.
    catalogs : list of str, optional
        Catalog names in priority order; :data:`CALIBRATION_CATALOG_HIERARCHY`
        by default.
    detect_method, snr_min, r_angle_max_arcsec, r_angle_grating_max_arcsec,
    dr_max_arcsec, near_neighbor_dist_min_arcsec
        As for :func:`select_xray_catalog_matches`.
    exclude_caldb_versions : iterable of str
        Processing releases whose rows are left out before rebasing.
    sim_z_offset_max_mm : float or None
        Rows further than this from their detector's nominal SIM-Z are
        dropped; None keeps every SIM-Z.

    Returns
    -------
    astropy.table.Table
        One row per selected, rebased cross-match, sorted by time, with
        ``time`` (a CxoTime), ``dy_rebased`` and ``dz_rebased`` added.
        ``meta["calalign_reference"]`` records each detector's reference
        matrix (start date, dy, dz).
    """
    import h5py
    from cxotime import CxoTime

    from astrometry_corr_gradient.calalign import rebase

    if catalogs is None:
        catalogs = CALIBRATION_CATALOG_HIERARCHY

    with h5py.File(db_path, "r") as handle:
        astromon_obs = Table(handle["astromon_obs"][:])
        astromon_xray_src = Table(handle["astromon_xray_src"][:])
        astromon_cat_src = Table(handle["astromon_cat_src"][:])
    astromon_obs["grating"] = astromon_obs["grating"].astype(str)
    astromon_obs["target"] = astromon_obs["target"].astype(str)
    astromon_xray_src["detect_method"] = astromon_xray_src["detect_method"].astype(str)
    astromon_cat_src["catalog"] = astromon_cat_src["catalog"].astype(str)
    astromon_cat_src = ensure_x_id_column(astromon_cat_src)

    matches = select_xray_catalog_matches(
        astromon_obs,
        astromon_xray_src,
        astromon_cat_src,
        catalogs=catalogs,
        detect_method=detect_method,
        snr_min=snr_min,
        r_angle_max_arcsec=r_angle_max_arcsec,
        r_angle_grating_max_arcsec=r_angle_grating_max_arcsec,
        dr_max_arcsec=dr_max_arcsec,
        near_neighbor_dist_min_arcsec=near_neighbor_dist_min_arcsec,
    )
    matches["time"] = CxoTime(matches["date_obs"])

    recorded = np.asarray(matches["caldb_version"]).astype(str)
    keep = rebasable_rows(recorded, exclude_caldb_versions)
    excluded_counts = {
        version: int(np.sum(~keep & (np.char.rstrip(recorded, ".") == version)))
        for version in sorted(
            {str(v).strip().rstrip(".") for v in exclude_caldb_versions}
        )
    }
    rebased = matches[keep]
    rebased["caldb_version"] = np.array(
        [
            str(version).strip().rstrip(".")
            for version in np.asarray(rebased["caldb_version"]).astype(str)
        ]
    )
    rebased["dy_rebased"], rebased["dz_rebased"], reference = rebase(
        np.asarray(rebased["dy"], float),
        np.asarray(rebased["dz"], float),
        np.asarray(rebased["detector"]).astype(str),
        np.asarray(rebased["time"].secs),
        np.asarray(rebased["caldb_version"]),
    )
    # the order the rows had in astrometry's fetch (by obsid, x_id, then by
    # time), so every downstream result is reproducible
    rebased.sort(["obsid", "x_id"])
    excluded = (
        bad_target_mask(np.asarray(rebased["target"]))
        | bad_obsid_mask(np.asarray(rebased["obsid"]))
        | bad_catalog_source_mask(
            np.asarray(rebased["catalog"]), np.asarray(rebased["name"])
        )
    )
    detectors = np.asarray(rebased["detector"]).astype(str)
    off_nominal = np.zeros(len(rebased), dtype=bool)
    if sim_z_offset_max_mm is not None:
        off_nominal = off_nominal_sim_z_mask(
            detectors, np.asarray(rebased["sim_z"]), sim_z_offset_max_mm
        )
    off_nominal_counts = {
        str(detector): int(np.sum(off_nominal & (detectors == detector)))
        for detector in np.unique(detectors)
    }
    rebased = rebased[~(excluded | off_nominal)]
    rebased.meta["calalign_reference"] = reference
    rebased.meta["excluded_caldb_versions"] = excluded_counts
    rebased.meta["off_nominal_sim_z"] = off_nominal_counts
    rebased.sort("time")
    return rebased
