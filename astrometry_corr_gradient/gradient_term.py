"""The gradient term: a prototype of what processing needs, for the developers.

The gradient term corrects each observation's position for the periscope
gradient monitors OOBAGRD3 and OOBAGRD6. It goes on top of the CALALIGN
alignment, which is made as today, without the term. For each observation, on
each axis (dz and dy)::

    term = c3 * (OOBAGRD3 - OOBAGRD3 reference) + c6 * (OOBAGRD6 - OOBAGRD6 reference)

The term is the expected X-ray minus catalog offset, in arcsec. Subtract it
from the X-ray position's dz and dy.

The same term, normalized::

    term = k3 * (OOBAGRD3 - OOBAGRD3 reference) / s3 + k6 * (OOBAGRD6 - OOBAGRD6 reference) / s6

- c3, c6: ``COEFFICIENTS``, arcsec per K. Fixed.
- s3, s6: ``SIGMAS``, how much each gradient typically moves inside a calendar
  quarter, in K.
- k3, k6: ``coefficients_per_sigma()``, c x s. Arcsec for that typical move,
  so the two gradients compare directly.
- The references: ``reference_table()``, one pair per calendar quarter.
  Observations in a quarter use its row. How the rows are made does not
  matter here; ``quarter_references()`` makes them.
- OOBAGRD3, OOBAGRD6: the observation's mean over its dwell, from the end of
  its maneuver to the start of the next, with bad telemetry masked.
- No term if a gradient is missing, or the table has no row for the quarter.
  The observation gets the CALALIGN value alone, and a flag.
- Units: K. The same numbers in degC. Divide degF by 1.8.

Needs only numpy and astropy. The data at the end of this file is written by
``scripts/update_gradient_term_data.py``. Don't edit it by hand.

Examples
--------
>>> term = gradient_term("2025:200:12:00:00", {"OOBAGRD3": -0.31, "OOBAGRD6": 0.141})
>>> sorted(term)
['dy', 'dz', 'no_reference', 'no_telemetry']
"""

import numpy as np
from astropy.table import Table
from astropy.time import Time

AXES = ("dz", "dy")
GRADIENT_MSIDS = ("OOBAGRD3", "OOBAGRD6")
MIN_OBSERVATIONS = 10  # a quarter needs this many observations to set a reference


def quarter_index(times) -> np.ndarray:
    """Each time's calendar quarter (UTC) as one integer: year * 4 + quarter - 1.

    Times can be ISO strings, Chandra's year:day-of-year strings, or an astropy Time.

    >>> quarter_index(["2024-07-01 00:00:00", "2024-09-30 23:59:59"])
    array([8098, 8098])
    >>> quarter_index("2024:274:23:59:59")  # 2024-09-30
    array([8098])
    """
    if not isinstance(times, Time):
        times = Time(np.atleast_1d(np.asarray(times, dtype=str)), scale="utc")
    calendar = times.utc.ymdhms
    years = np.atleast_1d(calendar["year"]).astype(int)
    months = np.atleast_1d(calendar["month"]).astype(int)
    return years * 4 + (months - 1) // 3


def quarter_label(index: int) -> str:
    """2024Q3 for the quarter that starts on 2024-07-01."""
    return f"{index // 4}Q{index % 4 + 1}"


def quarter_start(index: int) -> str:
    """The quarter's first day, e.g. 2024-07-01."""
    return f"{index // 4}-{3 * (index % 4) + 1:02d}-01"


def label_index(label: str) -> int:
    """The quarter index of a label such as 2024Q3."""
    return int(label[:4]) * 4 + int(label[5]) - 1


def reference_table() -> Table:
    """The reference table: observations in each quarter use its row's OOBAGRD3 and OOBAGRD6 (K)."""
    table = Table.read(REFERENCE_CSV, format="ascii.csv")
    for name in GRADIENT_MSIDS:
        table[name].unit = "K"
    return table


def vector_table() -> Table:
    """The test vectors: observations, their gradients (K), and the dz and dy terms they get (arcsec)."""
    table = Table.read(TEST_VECTOR_CSV, format="ascii.csv")
    for name in GRADIENT_MSIDS:
        table[name].unit = "K"
    for axis_name in AXES:
        table[axis_name].unit = "arcsec"
    return table


def gradient_term(
    times,
    gradients: dict,
    references: Table | None = None,
    coefficients: dict | None = None,
) -> dict[str, np.ndarray]:
    """The gradient term for each observation (arcsec), and why it is zero where it is.

    Parameters
    ----------
    times
        Each observation's time: ISO, Chandra's year:day-of-year, or an astropy Time.
    gradients : dict
        OOBAGRD3 and OOBAGRD6 per observation (K); both are required. NaN where
        the telemetry is missing.
    references : astropy.table.Table, optional
        The reference table; the embedded one (``reference_table()``) by default.
    coefficients : dict, optional
        Arcsec per K by axis and gradient; ``COEFFICIENTS`` by default.

    Returns
    -------
    dict
        ``dz`` and ``dy`` (subtract them from the X-ray position), and two flags:
        ``no_telemetry`` (a gradient is not finite) and ``no_reference`` (the
        table has no row for the observation's quarter). Flagged observations get 0.

    Raises
    ------
    KeyError
        If OOBAGRD3 or OOBAGRD6 is not given.
    """
    missing = [name for name in GRADIENT_MSIDS if name not in gradients]
    if missing:
        raise KeyError(f"gradients missing for {missing}")
    references = reference_table() if references is None else references
    coefficients = COEFFICIENTS if coefficients is None else coefficients
    quarters = quarter_index(times)
    columns = {
        name: np.broadcast_to(np.asarray(gradients[name], float), quarters.shape)
        for name in GRADIENT_MSIDS
    }
    row_of = {
        label_index(label): row for row, label in enumerate(references["quarter"])
    }
    rows = np.array([row_of.get(index, -1) for index in quarters])
    no_reference = rows < 0
    no_telemetry = ~np.all(
        [np.isfinite(columns[name]) for name in GRADIENT_MSIDS], axis=0
    )
    usable = ~no_reference & ~no_telemetry
    term = {}
    for axis_name in AXES:
        total = np.zeros(len(quarters))
        for name in GRADIENT_MSIDS:
            reference = np.asarray(references[name], float)[np.where(usable, rows, 0)]
            total += np.where(
                usable, coefficients[axis_name][name] * (columns[name] - reference), 0.0
            )
        term[axis_name] = total
    return term | {
        "no_telemetry": no_telemetry,
        "no_reference": no_reference & ~no_telemetry,
    }


def quarter_references(
    quarters: np.ndarray,
    gradients: dict,
    last_complete_quarter: int,
    *,
    lagging_from: int | None,
    min_observations: int = MIN_OBSERVATIONS,
) -> Table:
    """Make the reference table: one row per quarter Q, the OOBAGRD3 and OOBAGRD6 that observations in Q use.

    History, before ``lagging_from``, uses each quarter's own medians. From
    ``lagging_from`` (the release) on, a quarter uses the previous quarter's
    medians, and they never change. The quarter still filling always uses the
    previous one, having no medians of its own yet. Make the CALALIGN values
    the same way: the CALALIGN median and the reference come from the same
    quarter. Run it when a quarter ends, to add the next row.

    Parameters
    ----------
    quarters : numpy.ndarray
        Each observation's ``quarter_index()``; one value per observation, not per source.
    gradients : dict
        OOBAGRD3 and OOBAGRD6 per observation (K).
    last_complete_quarter : int
        The latest quarter that has ended; later observations set no reference.
    lagging_from : int or None
        The first quarter that lags (the release); None: none does, apart from
        the quarter still filling.
    min_observations : int
        A quarter needs this many observations with both gradients; with fewer,
        the latest earlier quarter that has enough is used.

    Returns
    -------
    astropy.table.Table
        From the first quarter that has a reference through the quarter after
        ``last_complete_quarter`` (the one now filling): its label and first
        day, the quarter the reference comes from, the number of observations
        behind it, and each gradient's median over them.

    Raises
    ------
    ValueError
        If no complete quarter has enough observations.
    """
    columns = {name: np.asarray(gradients[name], float) for name in GRADIENT_MSIDS}
    usable = np.all([np.isfinite(column) for column in columns.values()], axis=0)
    medians, counts = {}, {}
    for index in np.unique(quarters[usable]):
        rows = usable & (quarters == index)
        if index <= last_complete_quarter and rows.sum() >= min_observations:
            medians[int(index)] = {
                name: float(np.median(column[rows])) for name, column in columns.items()
            }
            counts[int(index)] = int(rows.sum())
    if not medians:
        raise ValueError(
            f"no complete quarter has {min_observations} observations with both gradients"
        )
    sources = np.array(sorted(medians))
    table_rows = []
    for index in range(int(sources[0]), last_complete_quarter + 2):
        own = lagging_from is None or index < lagging_from
        candidates = sources[sources <= index] if own else sources[sources < index]
        if candidates.size == 0:
            continue  # a lagging first quarter has nothing before it
        source = int(candidates.max())
        table_rows.append(
            {
                "quarter": quarter_label(index),
                "start": quarter_start(index),
                "from_quarter": quarter_label(source),
                "observations": counts[source],
            }
            | medians[source]
        )
    return Table(rows=table_rows)


def fit_coefficients(
    quarters: np.ndarray, offset: np.ndarray, gradients: dict
) -> dict[str, float]:
    """One axis's coefficients (arcsec per K), fit within calendar quarters. Done once; not part of processing.

    One row per matched source. Each quarter's mean is taken out of the offset
    and of both gradients, then one least-squares fit with no intercept, so the
    coefficients follow how the offset moves with the gradients inside a
    quarter, not their slow drift over the years.
    """
    columns = [np.asarray(gradients[name], float) for name in GRADIENT_MSIDS]
    usable = np.isfinite(offset) & np.all(
        [np.isfinite(column) for column in columns], axis=0
    )
    _, group = np.unique(quarters[usable], return_inverse=True)
    counts = np.bincount(group)

    def within_quarter(values: np.ndarray) -> np.ndarray:
        return (
            values[usable]
            - (np.bincount(group, weights=values[usable]) / counts)[group]
        )

    design = np.column_stack([within_quarter(column) for column in columns])
    fitted = np.linalg.lstsq(
        design, within_quarter(np.asarray(offset, float)), rcond=None
    )[0]
    return dict(zip(GRADIENT_MSIDS, (float(k) for k in fitted), strict=True))


def within_quarter_sigma(quarters: np.ndarray, values: np.ndarray) -> float:
    """A gradient's typical swing inside a calendar quarter (K), its sigma in the normalized form.

    The root mean square of each value's difference from its quarter's mean,
    with one degree of freedom taken out per quarter; NaN values are skipped.
    One row per matched source, as for the coefficients.

    >>> within_quarter_sigma(np.array([0, 0, 1, 1]), np.array([1.0, 3.0, 10.0, 14.0]))
    2.23606797749979
    """
    values = np.asarray(values, float)
    usable = np.isfinite(values)
    _, group = np.unique(np.asarray(quarters)[usable], return_inverse=True)
    counts = np.bincount(group)
    deviations = (
        values[usable] - (np.bincount(group, weights=values[usable]) / counts)[group]
    )
    degrees = int(usable.sum()) - len(counts)
    if degrees <= 0:
        raise ValueError("too few values for a within-quarter sigma")
    return float(np.sqrt(np.sum(deviations**2) / degrees))


def coefficients_per_sigma(
    coefficients: dict | None = None, sigmas: dict | None = None
) -> dict[str, dict[str, float]]:
    """The coefficients in arcsec per sigma (k = c x s): the term for a one-sigma swing of each gradient.

    ``COEFFICIENTS`` and ``SIGMAS`` by default.
    """
    coefficients = COEFFICIENTS if coefficients is None else coefficients
    sigmas = SIGMAS if sigmas is None else sigmas
    return {
        axis_name: {name: by_gradient[name] * sigmas[name] for name in GRADIENT_MSIDS}
        for axis_name, by_gradient in coefficients.items()
    }


# ---- the data: written by scripts/update_gradient_term_data.py, not by hand ----
# from gradient_term_calibration_set.ecsv, made 2026-11-20 18:52 UTC, astromon database astromon_gaussian_redo_20261105_1141.h5

COEFFICIENTS = {  # arcsec per K of each gradient
    "dz": {"OOBAGRD3": 0.462, "OOBAGRD6": -13.284},
    "dy": {"OOBAGRD3": 1.579, "OOBAGRD6": -11.185},
}

SIGMAS = {  # K: each gradient's typical swing inside a calendar quarter
    "OOBAGRD3": 0.10747,
    "OOBAGRD6": 0.00726,
}

# observations in each quarter use its row: each gradient's median (K) over from_quarter's
# observations: its own quarter for history, the previous quarter from the release on
REFERENCE_CSV = """\
quarter,start,from_quarter,observations,OOBAGRD3,OOBAGRD6
2004Q1,2004-01-01,2004Q1,47,0.85151,0.04103
2004Q2,2004-04-01,2004Q2,37,0.85104,0.04103
2004Q3,2004-07-01,2004Q3,30,0.83347,0.04103
2004Q4,2004-10-01,2004Q4,48,0.82879,0.04103
2005Q1,2005-01-01,2005Q1,58,0.83839,0.04107
2005Q2,2005-04-01,2005Q2,57,0.83153,0.04248
2005Q3,2005-07-01,2005Q3,36,0.81990,0.04219
2005Q4,2005-10-01,2005Q4,61,0.81237,0.04169
2006Q1,2006-01-01,2006Q1,65,0.77691,0.04416
2006Q2,2006-04-01,2006Q2,59,0.78264,0.04416
2006Q3,2006-07-01,2006Q3,46,0.77511,0.04416
2006Q4,2006-10-01,2006Q4,47,0.73879,0.04593
2007Q1,2007-01-01,2007Q1,52,0.67740,0.05355
2007Q2,2007-04-01,2007Q2,54,0.68311,0.05355
2007Q3,2007-07-01,2007Q3,54,0.67950,0.05355
2007Q4,2007-10-01,2007Q4,73,0.67174,0.05355
2008Q1,2008-01-01,2008Q1,101,0.67236,0.05355
2008Q2,2008-04-01,2008Q2,72,0.67892,0.05355
2008Q3,2008-07-01,2008Q3,48,0.65850,0.05385
2008Q4,2008-10-01,2008Q4,65,0.65136,0.05415
2009Q1,2009-01-01,2009Q1,77,0.62974,0.05667
2009Q2,2009-04-01,2009Q2,44,0.61008,0.05667
2009Q3,2009-07-01,2009Q3,45,0.60383,0.05667
2009Q4,2009-10-01,2009Q4,69,0.60569,0.05667
2010Q1,2010-01-01,2010Q1,84,0.57560,0.05926
2010Q2,2010-04-01,2010Q2,61,0.56648,0.05980
2010Q3,2010-07-01,2010Q3,71,0.53294,0.05981
2010Q4,2010-10-01,2010Q4,66,0.49463,0.06217
2011Q1,2011-01-01,2011Q1,79,0.47395,0.06293
2011Q2,2011-04-01,2011Q2,50,0.50084,0.06293
2011Q3,2011-07-01,2011Q3,58,0.49724,0.06293
2011Q4,2011-10-01,2011Q4,61,0.44444,0.06419
2012Q1,2012-01-01,2012Q1,55,0.36862,0.06606
2012Q2,2012-04-01,2012Q2,46,0.42439,0.06574
2012Q3,2012-07-01,2012Q3,57,0.31904,0.06606
2012Q4,2012-10-01,2012Q4,88,0.24247,0.06918
2013Q1,2013-01-01,2013Q1,58,0.14511,0.07210
2013Q2,2013-04-01,2013Q2,57,0.20297,0.07083
2013Q3,2013-07-01,2013Q3,60,0.11499,0.07231
2013Q4,2013-10-01,2013Q4,54,0.11394,0.07231
2014Q1,2014-01-01,2014Q1,49,0.05123,0.07365
2014Q2,2014-04-01,2014Q2,46,0.04896,0.07441
2014Q3,2014-07-01,2014Q3,55,-0.01748,0.07578
2014Q4,2014-10-01,2014Q4,50,0.03938,0.07310
2015Q1,2015-01-01,2015Q1,60,-0.00696,0.07544
2015Q2,2015-04-01,2015Q2,43,0.04305,0.07506
2015Q3,2015-07-01,2015Q3,58,0.02904,0.07516
2015Q4,2015-10-01,2015Q4,76,-0.04338,0.07654
2016Q1,2016-01-01,2016Q1,100,-0.12226,0.07856
2016Q2,2016-04-01,2016Q2,65,-0.08703,0.07856
2016Q3,2016-07-01,2016Q3,68,-0.15403,0.07856
2016Q4,2016-10-01,2016Q4,65,-0.21740,0.08036
2017Q1,2017-01-01,2017Q1,95,-0.27629,0.08460
2017Q2,2017-04-01,2017Q2,79,-0.25583,0.08169
2017Q3,2017-07-01,2017Q3,68,-0.22889,0.08100
2017Q4,2017-10-01,2017Q4,84,-0.31340,0.08794
2018Q1,2018-01-01,2018Q1,74,-0.31245,0.09004
2018Q2,2018-04-01,2018Q2,79,-0.29715,0.08794
2018Q3,2018-07-01,2018Q3,52,-0.29545,0.08704
2018Q4,2018-10-01,2018Q4,85,-0.28256,0.10384
2019Q1,2019-01-01,2019Q1,87,-0.28328,0.10669
2019Q2,2019-04-01,2019Q2,67,-0.27314,0.10053
2019Q3,2019-07-01,2019Q3,51,-0.27958,0.10433
2019Q4,2019-10-01,2019Q4,108,-0.28907,0.11305
2020Q1,2020-01-01,2020Q1,90,-0.25957,0.11458
2020Q2,2020-04-01,2020Q2,90,-0.25995,0.11880
2020Q3,2020-07-01,2020Q3,79,-0.27008,0.10669
2020Q4,2020-10-01,2020Q4,122,-0.32238,0.10914
2021Q1,2021-01-01,2021Q1,114,-0.32561,0.11986
2021Q2,2021-04-01,2021Q2,113,-0.33969,0.10669
2021Q3,2021-07-01,2021Q3,107,-0.32777,0.10480
2021Q4,2021-10-01,2021Q4,98,-0.32339,0.11936
2022Q1,2022-01-01,2022Q1,89,-0.33459,0.11607
2022Q2,2022-04-01,2022Q2,75,-0.34174,0.11277
2022Q3,2022-07-01,2022Q3,93,-0.32913,0.12703
2022Q4,2022-10-01,2022Q4,93,-0.30774,0.11801
2023Q1,2023-01-01,2023Q1,104,-0.32279,0.12821
2023Q2,2023-04-01,2023Q2,85,-0.32736,0.12420
2023Q3,2023-07-01,2023Q3,104,-0.35496,0.14103
2023Q4,2023-10-01,2023Q4,124,-0.37147,0.14010
2024Q1,2024-01-01,2024Q1,137,-0.33729,0.13601
2024Q2,2024-04-01,2024Q2,116,-0.32220,0.14195
2024Q3,2024-07-01,2024Q3,68,-0.30120,0.14030
2024Q4,2024-10-01,2024Q4,90,-0.35629,0.14388
2025Q1,2025-01-01,2025Q1,91,-0.31415,0.14282
2025Q2,2025-04-01,2025Q2,87,-0.30519,0.14028
2025Q3,2025-07-01,2025Q3,83,-0.30701,0.13906
2025Q4,2025-10-01,2025Q4,91,-0.33981,0.14079
2026Q1,2026-01-01,2026Q1,106,-0.36550,0.13792
2026Q2,2026-04-01,2026Q2,39,-0.35654,0.13974
2026Q3,2026-07-01,2026Q3,23,-0.23475,0.13481
2026Q4,2026-10-01,2026Q3,23,-0.23475,0.13481
"""

# matched observations, their gradients (K) and the dz and dy terms they get (arcsec); the last
# one gets no term
TEST_VECTOR_CSV = """\
obsid,date,OOBAGRD3,OOBAGRD6,dz,dy,no_term
7833,2008-04-01 04:58:27,0.70672,0.05355,0.0128,0.0439,-
13977,2012-07-01 02:51:34,0.43824,0.06293,0.0966,0.2232,-
19888,2016-10-01 15:29:43,-0.10007,0.07312,0.1504,0.2662,-
27455,2022-10-02 05:54:37,-0.51387,0.12805,-0.2286,-0.4378,-
30979,2025-07-02 12:02:23,-0.26719,0.14234,-0.0252,0.0262,-
33451,2026-10-03 19:48:41,-0.24373,0.13502,-0.0069,-0.0165,-
11749,2010-05-29 20:41:42,nan,nan,0.0000,0.0000,no telemetry
"""
