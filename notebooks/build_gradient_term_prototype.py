"""Generate notebooks/gradient_term_prototype.ipynb.

Run this script to (re)build the notebook from scratch; the notebook itself
is what gets committed and opened, this script is just how it's produced.
The gradient term's code and data live in astrometry_corr_gradient/gradient_term.py; the
notebook says what to implement, shows the coefficients, the reference table
and the test vectors, says how to apply the term to the fid centroids in
aca_corr_fid (with obsid 23059 as the worked example and test), then prints
the file. gradient_term_validation.ipynb checks the residuals with it.
"""

from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).parent

markdown_intro = """\
# Add an absolute astrometry correction using the periscope gradients OOBAGRD3 and OOBAGRD6

This correction is applied as a shift of the fid centroids, in `aca_corr_fid`.
Its size is the gradient term, which corrects each observation's dz and dy
for the periscope gradients OOBAGRD3 and OOBAGRD6. CALALIGN is unchanged; the
correction goes on top of it.

The pipeline already corrects the fid centroids with OOBAGRD3 and OOBAGRD6:
`aca_corr_fid`'s periscope correction, with CALALIGN's PERIFIDCORR
coefficients. That correction is relative: it removes temporal drift during
an observation, based on how the gradients change from their mean over the
observation. So it leaves each observation's average position about where it
was.

The new correction is absolute: one constant per observation, from how far
the observation's mean gradients are from the quarter's reference. It moves
each observation's average position, which the periscope correction leaves
alone.

Everything is in one file, `astrometry_corr_gradient/gradient_term.py`: the
code, the numbers and the test vectors. It needs only numpy and astropy.

**What to implement**, for each observation, on dz and dy:

`term = c3 * (OOBAGRD3 - ref3) + c6 * (OOBAGRD6 - ref6)`

1. **Gradients:** the observation's mean OOBAGRD3 and OOBAGRD6, in K. Each is a
   temperature difference between two periscope thermistors (RT 505 to RT 504,
   and RT 507 to RT 506, as the OBC files describe them), so these K are delta
   K. In the aspect pipeline, the means `aca_corr_fid` already computes for the
   periscope correction. The calibration averaged them over the dwell (from
   the end of its maneuver to the start of the next), with telemetry in safing
   intervals masked.
2. **Coefficients:** c3 and c6 for the axis, below.
3. **References:** ref3 and ref6, from the reference table's row for the
   calendar quarter (UTC) of the observation's start time (DATE-OBS). They
   remove the drift that fixed coefficients would otherwise give over the
   mission; see below.
4. **Result:** the term in arcsec, the expected X-ray minus catalog offset. The
   corrected offset is the observed one minus the term.
5. **Apply:** in `aca_corr_fid`, subtract the term, in degrees, from every good
   fid centroid: `ang_y_sm -= term_dy / 3600`, `ang_z_sm -= term_dz / 3600`.
   Details and a test below.
6. **No term** if a gradient is missing or the quarter has no row: the term is
   0 and the observation is flagged.
7. **Check:** match the test vectors below.

**Still open:**

- The averaging window: `aca_corr_fid`'s fid time range, the dwell (as in the
  calibration), or the observation's GTI span. On obsid 23059 the fid-range
  and dwell means give terms within 0.0005".
- The reference table starts at 2004Q1 for now, so observations before 2004
  get no term. Rows back to 2000Q1 will be added; there is no gradient
  telemetry before 2000.

Below: the coefficients, the reference table, the test vectors, how to apply
the term to the fid centroids, then the whole file.
`gradient_term_validation.ipynb` shows that the term works.
"""

code_setup = r"""
import inspect
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.table import Table
from IPython.display import Markdown, display

# this checkout's package, before any installed copy
sys.path.insert(0, str(Path.cwd().parent))
from astrometry_corr_gradient import gradient_term
"""

markdown_module = """\
## The whole file

`astrometry_corr_gradient/gradient_term.py`. Its docstring, at the top, says
what to implement; the numbers are at the end. After a new calibration set,
`python scripts/update_gradient_term_data.py` rewrites them.
"""

code_module = r"""
print(inspect.getsource(gradient_term))
"""

markdown_coefficients = """\
## Coefficients

Two ways to write the same term:

- **Per K:** `term = c3 * (OOBAGRD3 - ref3) + c6 * (OOBAGRD6 - ref6)`
- **Per sigma:** `term = k3 * (OOBAGRD3 - ref3) / s3 + k6 * (OOBAGRD6 - ref6) / s6`

**ref3** and **ref6** are the quarter's references, from the table below.
**s** is how much each gradient typically moves inside a quarter. **k** = c x s
is the arcsec for that typical move, so the two gradients compare directly.
"""

code_coefficients = r"""
per_sigma = gradient_term.coefficients_per_sigma()
header = "| axis | gradient | c (arcsec per K) | s (K) | k (arcsec per sigma) |"
rows = [
    f"| {axis_name} | {name} | {gradient_term.COEFFICIENTS[axis_name][name]}"
    f" | {gradient_term.SIGMAS[name]} | {per_sigma[axis_name][name]:.4f} |"
    for axis_name in gradient_term.AXES
    for name in gradient_term.GRADIENT_MSIDS
]
display(Markdown("\n".join([header, "|---|---|---|---|---|", *rows])))
print("COEFFICIENTS =", gradient_term.COEFFICIENTS)
print("SIGMAS =", gradient_term.SIGMAS)
print(
    "per sigma =",
    {
        axis_name: {name: round(k, 4) for name, k in by_gradient.items()}
        for axis_name, by_gradient in per_sigma.items()
    },
)
"""

markdown_drift = """\
## Why the references: removing the drift

With fixed coefficients for the whole mission, `c3 * OOBAGRD3 + c6 * OOBAGRD6`
drifts over the mission, because the gradients drift: OOBAGRD3 from about
0.85 K in 2004 to about -0.35 K now, and OOBAGRD6 from about 0.04 K to 0.14 K.
That comes to about 3" in dy and 2" in dz since 2004 (table below). This
correction should not take out that slow drift; CALALIGN handles the slow
changes in the offsets. The coefficients themselves were fit within each
quarter, so they only describe how the offsets follow the gradients over
weeks to months.

We compensate for the drift with time-based references. Each calendar
quarter's reference is the median of each gradient over that quarter's
observations, and the correction is measured from it:

`term = c3 * (OOBAGRD3 - ref3) + c6 * (OOBAGRD6 - ref6)`

This subtracts the correction's typical value in each quarter, so over a
quarter's observations its median is close to zero: typically 0.005", at most
0.08". (Not exactly zero, because the medians are taken of each gradient, not
of the correction.) The correction then removes only how each observation
differs from the quarter's typical one, which narrows the scatter; each
quarter's median offset stays where CALALIGN puts it.

Past quarters use their own observations. A new quarter's observations are
processed before the quarter is over, so from the release on each new quarter
uses the previous quarter's medians, and the table gains a row each quarter.

The drift the references remove: the correction at each quarter's typical
gradients, `c3 * ref3 + c6 * ref6`, every second year:
"""

code_drift = r"""
references = gradient_term.reference_table()
drift = references["quarter", "OOBAGRD3", "OOBAGRD6"]
for axis_name in gradient_term.AXES:
    drift[f"{axis_name} drift"] = np.round(
        sum(
            gradient_term.COEFFICIENTS[axis_name][name]
            * np.asarray(references[name], float)
            for name in gradient_term.GRADIENT_MSIDS
        ),
        2,
    )
    drift[f"{axis_name} drift"].unit = "arcsec"
rows = sorted(set(range(0, len(drift), 8)) | {len(drift) - 1})
drift[rows].pprint(max_width=-1)
"""

markdown_references = """\
## Reference table

One row per calendar quarter. Observations in that quarter use its OOBAGRD3
and OOBAGRD6, in K, as ref3 and ref6. `gradient_term()` only looks up the row;
how the row was made does not matter to it. `from_quarter` is the quarter
whose observations made the row; `observations` is how many. For now the
table starts at 2004Q1.
"""

code_references = r"""
gradient_term.reference_table().pprint(max_lines=-1, max_width=-1)
"""

markdown_vectors = """\
## Test vectors

Real observations and the term each should get, in arcsec, next to what
`gradient_term()` gives. The last one has no telemetry, so it gets no term.
Any other implementation should match these too.
"""

code_vectors = r"""
vectors = gradient_term.vector_table()
term = gradient_term.gradient_term(
    vectors["date"], {name: vectors[name] for name in gradient_term.GRADIENT_MSIDS}
)
for axis_name in gradient_term.AXES:
    vectors[f"{axis_name} computed"] = np.round(term[axis_name], 4)
    vectors[f"{axis_name} computed"].unit = "arcsec"
vectors.pprint(max_width=-1)
"""

markdown_fid = """\
## Applying it: shift the fid centroids

The term goes where the periscope correction already is: `aca_corr_fid`
(ds15 `tool_functions.c`, `apply_centroid_correction()`). That function loops
over the ACACENT rows and, for every row of every good fid slot, corrects
`ang_y_sm` and `ang_z_sm` (degrees) twice:

1. **ACIS thermal correction** (`apply_thermalcorr`): scales the fid positions
   about their center of expansion with the ACIS detector housing
   temperature. ACIS only; it turns itself off when the good fids are HRC's.
2. **Periscope correction** (`apply_pericorr`): removes how OOBAGRD3 and
   OOBAGRD6 change during the observation, relative to their mean (CALALIGN's
   PERIFIDCORR coefficients). ACIS and HRC.

The term is one more subtraction after those, the same for every row of the
observation. It needs the periscope correction's mean gradients, so it applies
wherever the periscope correction does, HRC included:

```
ang_y_sm -= term_dy / 3600
ang_z_sm -= term_dz / 3600
```

- **Axes and sign:** dy goes with `ang_y_sm` and dz with `ang_z_sm`, with the
  same sign. `gradient_term.fid_centroid_shift()` gives the two numbers, in
  degrees.
- **Gradients:** `info->obcinfo.meanGradient3` and `meanGradient6`, which
  `calc_periscope_corr_values()` already computes: the mean of the OBC
  engineering samples over the fid centroids' time range, with bad samples
  interpolated. Without them (no usable OBC data, or `apply_pericorr` off)
  there is no term.
- **Reference quarter:** the calendar quarter (UTC) of the observation's start
  (DATE-OBS).

With the periscope correction, each good fid centroid becomes

```
ang_y_sm -= peri_y_grd3 * (G3(t) - mean G3) + peri_y_grd6 * (G6(t) - mean G6) + term_dy / 3600
```

and likewise `ang_z_sm` with the z coefficients and `term_dz`. The periscope
correction removes how the gradients change during the observation; the term
removes how far their mean is from the quarter's reference.

Below: the whole correction as runnable Python, step by step; what changes in
`aca_corr_fid`'s C; an example; and the test in the aspect pipeline.
"""

markdown_prototype = """\
### The correction, step by step

`gradient_term_fid_correction()` below is the whole correction for one aspect
interval, in the order `aca_corr_fid` would run it, on plain ACACENT, FIDPROPS
and OBC columns. Steps 1 and 2 exist in `aca_corr_fid` already, for the
periscope correction; they are here so the prototype runs on its own. Steps 3
to 5 are new. `scripts/shift_fids.py` runs the same steps on real files; it is
what the aspect pipeline test used.
"""

code_prototype = r'''
DATA_NORMAL = 0  # ACACENT status of a normal record (ds15 aspect_lib/acacent.h)
EGAUSS = 8  # ACACENT alg of the Kalman filter's centroids (asp_l1_std.kalman_alg)


def gradient_term_fid_correction(
    date_obs: str,
    acacent: Table,
    good_fid_slots: list[int],
    obc: Table,
    kalman_alg: int = EGAUSS,
) -> tuple[Table, dict]:
    """The gradient term applied to one aspect interval's fid centroids, as aca_corr_fid would.

    date_obs: DATE-OBS of the ACACENT file.
    acacent: ACACENT columns time, slot, alg, status, ang_y_sm and ang_z_sm
        (deg), after the thermal and periscope corrections.
    good_fid_slots: the FIDPROPS slots whose id_status starts with GOOD (in
        aca_corr_fid, the slots with info->datndx[slot] >= 0).
    obc: OBC engineering columns TIME, OOBAGRD3 and OOBAGRD6 (K), and
        OOBAGRD3_GOOD and OOBAGRD6_GOOD: True where the sample's QUALITY bit
        is good (aca_corr_fid: *obcptr.quality.oobagrd3 == GOOD_NEW).

    Returns the corrected ACACENT and a record of what was done.
    """
    # 1. The fid time range. aca_corr_fid has it: info->sampdata.tmin, tmax.
    #    The first and last time of the good fid rows with status DATA_NORMAL
    #    and the Kalman filter's algorithm.
    usable = (
        np.isin(acacent["slot"], good_fid_slots)
        & (acacent["status"] == DATA_NORMAL)
        & (acacent["alg"] == kalman_alg)
    )
    if not usable.any():
        return acacent, {"applied": False, "status": "NOFIDS"}
    tmin = float(acacent["time"][usable].min())
    tmax = float(acacent["time"][usable].max())

    # 2. The mean gradients. aca_corr_fid has them, for the periscope
    #    correction: info->obcinfo.meanGradient3 and meanGradient6, from
    #    calc_periscope_corr_values(). The OBC samples with
    #    tmin <= TIME <= tmax; bad samples filled by linear interpolation over
    #    sample index (the nearest good value at either end); then the plain
    #    mean, before any smoothing.
    in_range = (obc["TIME"] >= tmin) & (obc["TIME"] <= tmax)
    means = {}
    for name in gradient_term.GRADIENT_MSIDS:
        values = np.asarray(obc[name][in_range], float)
        good = np.asarray(obc[f"{name}_GOOD"][in_range], bool)
        if not good.any():  # aca_corr_fid turns the periscope correction off too
            return acacent, {"applied": False, "status": "NOTELEM"}
        index = np.arange(len(values))
        means[name] = float(np.mean(np.interp(index, index[good], values[good])))

    # 3. NEW: the term (arcsec) and the fid shift (deg), once per aspect
    #    interval: COEFFICIENTS, and the reference table's row for DATE-OBS's
    #    calendar quarter (UTC).
    shift = gradient_term.fid_centroid_shift(date_obs, means)
    if shift["no_reference"][0]:  # warn, and leave the centroids as they are
        return acacent, {"applied": False, "status": "NOREF", "means": means}
    shift_y_deg = float(shift["ang_y_sm"][0])
    shift_z_deg = float(shift["ang_z_sm"][0])
    quarter = int(gradient_term.quarter_index(date_obs)[0])
    table = gradient_term.reference_table()
    row = table[
        [gradient_term.label_index(label) == quarter for label in table["quarter"]]
    ][0]

    # 4. NEW: subtract it from every row of every good fid slot, whatever its
    #    algorithm or status (as the other two corrections do), after the
    #    periscope correction. In aca_corr_fid's row loop:
    #    acacent.ang_y_sm -= (float)(term_dy / 3600.0), and likewise z.
    corrected = acacent.copy()
    new_y, new_z = gradient_term.shift_fid_centroids(
        acacent["slot"],
        acacent["ang_y_sm"],
        acacent["ang_z_sm"],
        good_fid_slots,
        shift_y_deg,
        shift_z_deg,
    )
    corrected["ang_y_sm"][:] = new_y
    corrected["ang_z_sm"][:] = new_z

    # 5. NEW, suggested: record what was done in the headers (below).
    return corrected, {
        "applied": True,
        "status": "APPLIED",
        "tmin": tmin,
        "tmax": tmax,
        "means": means,
        "references": {name: float(row[name]) for name in gradient_term.GRADIENT_MSIDS},
        "quarter": f"{quarter // 4}Q{quarter % 4 + 1}",
        "term_dy": shift_y_deg * 3600,
        "term_dz": shift_z_deg * 3600,
        "shift_y_deg": shift_y_deg,
        "shift_z_deg": shift_z_deg,
        "rows_shifted": int(np.isin(acacent["slot"], good_fid_slots).sum()),
    }
'''

markdown_c = """\
### What changes in `aca_corr_fid`

In `apply_centroid_correction()`, steps 3 and 4 above. `c_dy3` is
`COEFFICIENTS["dy"]["OOBAGRD3"]` (and so on), and `ref3`, `ref6` are OOBAGRD3
and OOBAGRD6 from the reference table's row for the quarter:

```c
/* once, before the row loop: the gradient term (arcsec) */
double g3 = info->obcinfo.meanGradient3 - ref3;
double g6 = info->obcinfo.meanGradient6 - ref6;
double term_dy = c_dy3 * g3 + c_dy6 * g6;
double term_dz = c_dz3 * g3 + c_dz6 * g6;

/* each row of a good fid slot, after the periscope correction */
acacent.ang_y_sm -= (float)(term_dy / 3600.0);
acacent.ang_z_sm -= (float)(term_dz / 3600.0);
```

- **Already in `aca_corr_fid`:** the good fid slots (`info->datndx`), the fid
  time range (`info->sampdata.tmin`, `tmax`) and the mean gradients
  (`info->obcinfo.meanGradient3`, `meanGradient6`).
- **New inputs:** the ACACENT file's DATE-OBS, for the calendar quarter (UTC);
  `COEFFICIENTS` and the reference table from `gradient_term.py`. The table
  gets a new row each quarter, from the previous quarter's observations.
- **No term** (warn, and leave the centroids as they are) when the mean
  gradients don't exist (the periscope correction is off or has no usable OBC
  data), or the reference table has no row for the quarter (before 2004Q1 for
  now, or a quarter not yet added).
- **More than one aspect interval:** `aca_corr_fid` runs on each interval's
  ACACENT, so each gets its own means and term. Which DATE-OBS sets the
  quarter matters only for an observation that crosses a quarter boundary.
- **Suggested, not decided:** a parameter to turn it off, like
  `apply_pericorr`; header keywords recording the term (step 5, and below).
- **Checks:** the term matches the test vectors above; on a real observation,
  `scripts/compare_asol.py` gives 1.0 on each axis.
"""

markdown_example = """\
### Example

A made-up aspect interval with obsid 23059's DATE-OBS and mean gradients
(ACIS-S, 2020-10-27): ACACENT rows every 2.05 s for 5 minutes, good fid slots
0 to 2 and star slots 3 to 7, and OBC samples every 32.8 s with one bad sample
in the range. The fids move by minus the term; the stars don't move.
"""

code_example = r"""
start = 720209953.786
times = start + np.arange(0.0, 300.0, 2.05)
positions = {  # made-up (ang_y_sm, ang_z_sm) of each slot, deg
    0: (0.6853, -0.4319),
    1: (-0.5902, -0.4317),
    2: (0.1207, 0.6612),
    3: (0.3101, 0.2203),
    4: (-0.4417, 0.5120),
    5: (0.0954, -0.6630),
    6: (-0.2381, -0.1179),
    7: (0.5560, 0.4021),
}
slots = np.repeat(np.array(list(positions), np.uint8), len(times))
acacent = Table(
    {
        "time": np.tile(times, len(positions)),
        "slot": slots,
        "alg": np.full(len(slots), EGAUSS, np.uint8),
        "status": np.full(len(slots), DATA_NORMAL, np.uint8),
        "ang_y_sm": np.float32([positions[slot][0] for slot in slots]),
        "ang_z_sm": np.float32([positions[slot][1] for slot in slots]),
    }
)
obc_times = start - 40.0 + np.arange(0.0, 400.0, 32.8)
obc = Table(
    {
        "TIME": obc_times,
        "OOBAGRD3": np.full(len(obc_times), -0.139041),  # 23059's mean, K
        "OOBAGRD6": np.full(len(obc_times), 0.083546),
        "OOBAGRD3_GOOD": np.ones(len(obc_times), bool),
        "OOBAGRD6_GOOD": np.ones(len(obc_times), bool),
    }
)
obc["OOBAGRD3"][4], obc["OOBAGRD3_GOOD"][4] = 99.0, False  # a bad sample, filled

corrected, record = gradient_term_fid_correction(
    "2020-10-27T18:19:13", acacent, [0, 1, 2], obc
)
for key, value in record.items():
    print(f"{key}: {value}")

# the first row of each slot, before and after (float32, as in ACACENT, so the
# change is exact to about 0.0002")
first = np.unique(acacent["slot"], return_index=True)[1]
summary = Table(
    {
        "slot": acacent["slot"][first],
        "good fid": np.isin(acacent["slot"][first], [0, 1, 2]),
        "ang_y_sm before": acacent["ang_y_sm"][first],
        "ang_y_sm after": corrected["ang_y_sm"][first],
        "ang_z_sm before": acacent["ang_z_sm"][first],
        "ang_z_sm after": corrected["ang_z_sm"][first],
        "change y (arcsec)": 3600
        * (corrected["ang_y_sm"][first].astype(float) - acacent["ang_y_sm"][first]),
        "change z (arcsec)": 3600
        * (corrected["ang_z_sm"][first].astype(float) - acacent["ang_z_sm"][first]),
    }
)
for name in ("change y (arcsec)", "change z (arcsec)"):
    summary[name].format = "+.4f"
summary.pprint(max_width=-1)
"""

markdown_headers = """\
### Suggested: record the term in the headers

Not needed for the correction itself, and little work: a few header keywords
saying whether the term was applied, by how much and from which numbers. They
would let anyone check or redo the term later without the OBC data or the code
that made it, and let a future calibration tell which observations already
have the term, so none is corrected twice.

| Keyword | Value | Why |
|---|---|---|
| `GTRMAPP` | `APPLIED`, or why not: `NOREF` (no reference row), `NOTELEM` (no mean gradients), `NOFIDS` (no normal fid centroids), `OFF` (turned off) | Tells "no term" from a term of 0 |
| `GTRM_DY`, `GTRM_DZ` | The term, arcsec | What was applied; the fid shift is this / 3600, in degrees |
| `GTRM_G3`, `GTRM_G6` | The mean gradients used, K | To check or redo the term without the OBC files |
| `GTRM_R3`, `GTRM_R6` | The reference values used, K | The same, and in case a table row is revised |
| `GTRM_QTR` | The reference quarter, such as `2020Q4` | Which row |
| `GTRM_VER` | A short version or date for the coefficients and reference table | The numbers change: a row each quarter, new coefficients after a recalibration |

Plus a `HISTORY` line in words.

- **Where:** the ACACENT header, where the correction happens, and the aspect
  solution's (`asol1`), with at least `GTRMAPP`, `GTRM_DY`, `GTRM_DZ` and
  `GTRM_VER`: users and astromon see the aspect solution, not ACACENT
  (`download_chandra_obsid` doesn't fetch ACACENT). Whether the pipeline's
  keyword pipes (`update_prim_keys`, `update_princ_keys`) can carry them there
  is not checked.
- **FITS limits:** names up to 8 characters, cards of 80, string values up to
  68 characters, `HISTORY` text up to 72 characters per card. All of these
  fit; keep `GTRM_VER` short. Longer names would need the non-standard
  HIERARCH convention.
- **Names and format** are only a suggestion; CXC's keyword conventions come
  first. `gradient_term.py` has no version constant yet; its numbers carry the
  date they were made, used below.

The cards for the example above, then for the same aspect interval moved to
2001, before the reference table starts:
"""

code_headers = r'''
def suggested_header(record: dict, version: str) -> fits.Header:
    """The suggested header keywords for one aspect interval, from gradient_term_fid_correction()'s record."""
    header = fits.Header()
    header["GTRMAPP"] = (record["status"], "gradient absolute corr applied")
    if record["applied"]:
        header["GTRM_DY"] = (round(record["term_dy"], 4), "[arcsec] gradient term dy")
        header["GTRM_DZ"] = (round(record["term_dz"], 4), "[arcsec] gradient term dz")
        for number, name in zip("36", gradient_term.GRADIENT_MSIDS, strict=True):
            header[f"GTRM_G{number}"] = (
                round(record["means"][name], 6),
                f"[K] mean {name} over the fid time range",
            )
        for number, name in zip("36", gradient_term.GRADIENT_MSIDS, strict=True):
            header[f"GTRM_R{number}"] = (
                record["references"][name],
                f"[K] reference {name}",
            )
        header["GTRM_QTR"] = (record["quarter"], "gradient term reference quarter")
    header["GTRM_VER"] = (version, "gradient term coefficients and reference table")
    if record["applied"]:
        header.add_history(
            f"gradient term: fid ang_y_sm -= {record['shift_y_deg']:.9f},"
            f" ang_z_sm -= {record['shift_z_deg']:.9f} deg"
        )
    return header


VERSION = "2026-11-23"  # the date the numbers in gradient_term.py were made
header = suggested_header(record, VERSION)
print(repr(header))
print()
_, record_2001 = gradient_term_fid_correction(
    "2001-06-01T00:00:00", acacent, [0, 1, 2], obc
)
print(repr(suggested_header(record_2001, VERSION)))
print()
print(
    "every card fits in 80 characters:",
    all(len(card.image) == 80 for card in header.cards),
)
'''

markdown_fid_test = """\
### Tested in the aspect pipeline

On obsid 23059 the aspect pipeline ran twice: as usual, and with the fid
centroids shifted as above, by `scripts/shift_fids.py` right after
`aca_corr_fid`. Its log matched the example: term dy +0.5928", dz +0.4364";
ang_y_sm -= 0.000164668 deg, ang_z_sm -= 0.000121215 deg, on 58,752 rows of 3
good fid slots. (The example starts from the means as the log printed them,
to 6 decimals, so its last digit in degrees can differ.)

- **Aspect solution:** the shifted run's aspect solution moves X-ray events by
  -0.591" in y and -0.435" in z, 0.997 times minus the term on both axes
  (`scripts/compare_asol.py`). A wrong sign would give -1; swapped axes, 0.74
  and 1.36. The shift shows up first in the aspect solution's dy and dz, then,
  after `asp_offaxis_corr` at the end of the pipeline, in its attitude.
- **Sources:** after chandra_repro with each aspect solution and celldetect,
  the three sources matched to catalog counterparts moved by the same amount.
  Their offsets from the catalog went from 0.61", 0.49" and 0.68" to 0.14",
  0.25" and 0.07".

23059 is in the calibration set, so this checks how the term is applied, not
the term itself; `gradient_term_validation.ipynb` checks the term.
"""

nb = nbf.v4.new_notebook()
nb.cells = [
    nbf.v4.new_markdown_cell(markdown_intro),
    nbf.v4.new_code_cell(code_setup.strip("\n")),
    nbf.v4.new_markdown_cell(markdown_coefficients),
    nbf.v4.new_code_cell(code_coefficients.strip("\n")),
    nbf.v4.new_markdown_cell(markdown_drift),
    nbf.v4.new_code_cell(code_drift.strip("\n")),
    nbf.v4.new_markdown_cell(markdown_references),
    nbf.v4.new_code_cell(code_references.strip("\n")),
    nbf.v4.new_markdown_cell(markdown_vectors),
    nbf.v4.new_code_cell(code_vectors.strip("\n")),
    nbf.v4.new_markdown_cell(markdown_fid),
    nbf.v4.new_markdown_cell(markdown_prototype),
    nbf.v4.new_code_cell(code_prototype.strip("\n")),
    nbf.v4.new_markdown_cell(markdown_c),
    nbf.v4.new_markdown_cell(markdown_example),
    nbf.v4.new_code_cell(code_example.strip("\n")),
    nbf.v4.new_markdown_cell(markdown_headers),
    nbf.v4.new_code_cell(code_headers.strip("\n")),
    nbf.v4.new_markdown_cell(markdown_fid_test),
    nbf.v4.new_markdown_cell(markdown_module),
    nbf.v4.new_code_cell(code_module.strip("\n")),
]

out_path = HERE / "gradient_term_prototype.ipynb"
nbf.write(nb, out_path)
print(f"wrote {out_path}")
