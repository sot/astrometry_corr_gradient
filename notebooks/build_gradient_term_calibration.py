"""Generate notebooks/gradient_term_calibration.ipynb.

Run this script to (re)build the notebook from scratch; the notebook itself
is what gets committed and opened, this script is just how it's produced.
The gradient term's calibration from an astromon database, with the steps in
astrometry_corr_gradient/calibration.py: the matched sources, the dwells, the
calibration set and the solve; then checks against the previous calibration
set and the numbers in astrometry_corr_gradient/gradient_term.py.
"""

from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).parent

markdown_intro = """\
# Gradient term: calibration from the astromon database

From an astromon database to the gradient term's numbers. The code is in
`astrometry_corr_gradient/calibration.py`, one function per step:

1. **Matched sources:** the X-ray to catalog cross matches, rebased onto each
   detector's latest CALALIGN matrix. Cuts: astromon's match cuts, the
   calibration catalogs, a few excluded targets and observations, SIM-Z within
   4 mm of nominal, and 2004 on.
2. **Dwells:** each observation's dwell, from the end of its maneuver to the
   start of the next, from kadi.
3. **Calibration set:** one row per source, with OOBAGRD3 and OOBAGRD6
   averaged over each dwell.
4. **Solve:** the coefficients, sigmas and reference table.

The only setting is `DATABASE`, the astromon database, which is read and never
written. The CALALIGN matrices are this repo's copy. The telemetry comes from
cheta, with telemetry in safing intervals (from kadi and cheta) masked. Both
are cached in `cache/`, so a rerun fetches only what is new.

The calibration set is saved as `gradient_term_calibration_set.ecsv`;
`python scripts/update_gradient_term_data.py` then writes the numbers into
the module.
"""

code_setup = r"""
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.table import Table
from astropy.time import Time

os.environ.setdefault("SKA", "/Users/jean/ska")
# this checkout's package and scripts, before any installed copy
sys.path.insert(0, str(Path.cwd().parent))
sys.path.insert(1, str(Path.cwd().parent / "scripts"))
from update_gradient_term_data import RELEASE_QUARTER

from astrometry_corr_gradient import calibration, gradient_term

DATABASE = "/Users/jean/git/astrometry_scratch_20260929/median_bins_telemetry/snapshots/astromon_gaussian_redo_20261122_2132.h5"
CACHE_DIR = Path.cwd().parent / "cache"
CALIBRATION_PATH = Path("gradient_term_calibration_set.ecsv")
START = "2004:001"
STOP = (
    Time.now() + 5 * u.day
).yday  # a few days past now, so the newest observations are covered
print(f"database: {DATABASE}")
print(f"cache: {CACHE_DIR}")
"""

markdown_matches = """\
## 1. Matched sources
"""

code_matches = r"""
matches = calibration.matched_sources(DATABASE)
print(f"{len(matches):,} matched sources from {calibration.FIRST_YEAR} on")
print(
    "rebased onto:",
    {
        detector: row["start"]
        for detector, row in matches.meta["calalign_reference"].items()
    },
)
"""

markdown_dwells = """\
## 2. Dwells
"""

code_dwells = r"""
windows = calibration.dwell_windows(START, STOP)
print(f"{len(windows):,} observations with one dwell, {START} to {STOP[:8]}")
"""

markdown_set = """\
## 3. Calibration set
"""

code_set = r"""
series = {
    name: calibration.gradient_series(name, START, STOP, CACHE_DIR)
    for name in gradient_term.GRADIENT_MSIDS
}
bad_intervals = calibration.bad_intervals(
    START, STOP, CACHE_DIR / "bad_telemetry_intervals.ecsv"
)
calibration_table = calibration.calibration_set(matches, windows, series, bad_intervals)
print(
    f"{len(calibration_table):,} sources in {len(np.unique(calibration_table['obsid'])):,} observations,"
    f" {calibration_table['date'][0][:10]} to {calibration_table['date'][-1][:10]}"
)
print(
    ", ".join(
        f"{detector} {int((calibration_table['detector'] == detector).sum()):,}"
        for detector in np.unique(calibration_table["detector"])
    )
)
calibration_table[:5].pprint(max_width=-1)
"""

markdown_previous = """\
## 3a. Check against the previous calibration set

The calibration set saved last time, if there is one: which sources came or
went, and whether the common ones changed. A new database adds sources; the
same database should give the same set.
"""

code_previous = r"""
if CALIBRATION_PATH.exists():
    previous = Table.read(CALIBRATION_PATH, format="ascii.ecsv")
    print(f"previous: {len(previous):,} sources, made {previous.meta.get('made')}")

    def keys(table: Table) -> list[tuple]:
        return [
            (int(row["obsid"]), str(row["date"]), round(float(row["dz_rebased"]), 6))
            for row in table
        ]

    new_keys, old_keys = keys(calibration_table), keys(previous)
    print(
        f"new: {len(set(new_keys) - set(old_keys))}, gone: {len(set(old_keys) - set(new_keys))}"
    )
    position = {key: index for index, key in enumerate(old_keys)}
    common = [
        (index, position[key]) for index, key in enumerate(new_keys) if key in position
    ]
    if common:
        here, there = (np.array(indexes) for indexes in zip(*common, strict=True))
        for name in ("dz_rebased", "dy_rebased", "OOBAGRD3", "OOBAGRD6"):
            now, before = (
                np.asarray(calibration_table[name], float)[here],
                np.asarray(previous[name], float)[there],
            )
            finite = np.isfinite(now) & np.isfinite(before)
            largest = (
                float(np.max(np.abs(now[finite] - before[finite])))
                if finite.any()
                else 0.0
            )
            same_missing = np.array_equal(np.isnan(now), np.isnan(before))
            print(
                f"{name}: {len(common):,} common sources, largest change {largest:.2e}, same missing values: {same_missing}"
            )
else:
    print("no previous calibration set")
"""

markdown_save = """\
## 3b. Save

The calibration set, with where it came from in its header.
"""

code_save = r"""
def git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], capture_output=True, text=True, check=True
    ).stdout.strip()


changed = git("status", "--porcelain", "--", "..")
calibration_table.meta.update(
    {
        "description": "the gradient term's calibration set, one row per matched source (gradient_term_calibration.ipynb)",
        "made": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        "database": str(DATABASE),
        "telemetry_last_sample": {
            name: Time(series[name][0][-1], format="cxcsec").iso[:16]
            for name in gradient_term.GRADIENT_MSIDS
        },
        "dwells": f"kadi maneuvers {START} to {STOP}",
        "git_commit": git("rev-parse", "HEAD"),
        "git_changes": changed.splitlines() if changed else "none",
    }
)
calibration_table.write(CALIBRATION_PATH, format="ascii.ecsv", overwrite=True)
reread = Table.read(CALIBRATION_PATH, format="ascii.ecsv")
for name in calibration_table.colnames:
    written, read_back = np.asarray(calibration_table[name]), np.asarray(reread[name])
    if written.dtype.kind == "f":
        assert np.array_equal(written, read_back, equal_nan=True), name
    else:
        assert list(written) == list(read_back), name
print(
    f"wrote {CALIBRATION_PATH}: {len(reread):,} sources, {len(reread.colnames)} columns; it reads back the same"
)
"""

markdown_solve = """\
## 4. Solve, and compare with the module

The coefficients and sigmas use every source with both gradients. The
reference table uses one value per observation, since an observation's
sources share its gradients. History uses each quarter's own medians; from the
release quarter on, the previous quarter's. The quarter still filling uses the
previous quarter's too.
"""

code_solve = r"""
last_complete_quarter = int(gradient_term.quarter_index(Time.now())[0]) - 1
coefficients, sigmas, references = calibration.solve(
    calibration_table,
    last_complete_quarter=last_complete_quarter,
    lagging_from=gradient_term.label_index(RELEASE_QUARTER),
)
rows = []
for axis_name in gradient_term.AXES:
    for name in gradient_term.GRADIENT_MSIDS:
        rows.append(
            {
                "axis": axis_name,
                "gradient": name,
                "c now (arcsec per K)": round(coefficients[axis_name][name], 3),
                "c in the module": gradient_term.COEFFICIENTS[axis_name][name],
                "s now (K)": round(sigmas[name], 5),
                "s in the module": gradient_term.SIGMAS[name],
            }
        )
Table(rows=rows).pprint(max_width=-1)
embedded = gradient_term.reference_table()
print(
    f"reference table: {len(references)} quarters now ({references['quarter'][0]} to {references['quarter'][-1]}),"
    f" {len(embedded)} in the module"
)
common_quarters = sorted(set(references["quarter"]) & set(embedded["quarter"]))
for name in gradient_term.GRADIENT_MSIDS:
    now = np.array(
        [
            references[name][list(references["quarter"]).index(q)]
            for q in common_quarters
        ]
    )
    module = np.array(
        [embedded[name][list(embedded["quarter"]).index(q)] for q in common_quarters]
    )
    print(
        f"{name}: largest difference over {len(common_quarters)} quarters {np.max(np.abs(now - module)):.1e} K"
    )
"""

markdown_notes = """\
## Notes

- **To make new numbers:** point `DATABASE` at a newer database and rerun,
  then `python scripts/update_gradient_term_data.py` and
  `python scripts/make_mock_n0011.py`.
- **A new CALALIGN file:** `python scripts/update_calalign_table.py
  /path/to/calalign` rewrites the repo's copy of the matrices.
- **Not here yet:**
  - averaging the gradients over each observation's GTI span instead of its
    dwell;
  - references back to 2000Q1. The data are there (about 1,080 matched
    sources in 2000-2003; cheta's OOBAGRD3 and OOBAGRD6 start on
    1999-12-31), but it needs `START` moved to 2000 and the references
    started separately from the fit. There is no gradient telemetry for
    1999.
"""

nb = nbf.v4.new_notebook()
nb.cells = [
    nbf.v4.new_markdown_cell(markdown_intro),
    nbf.v4.new_code_cell(code_setup.strip("\n")),
    nbf.v4.new_markdown_cell(markdown_matches),
    nbf.v4.new_code_cell(code_matches.strip("\n")),
    nbf.v4.new_markdown_cell(markdown_dwells),
    nbf.v4.new_code_cell(code_dwells.strip("\n")),
    nbf.v4.new_markdown_cell(markdown_set),
    nbf.v4.new_code_cell(code_set.strip("\n")),
    nbf.v4.new_markdown_cell(markdown_previous),
    nbf.v4.new_code_cell(code_previous.strip("\n")),
    nbf.v4.new_markdown_cell(markdown_save),
    nbf.v4.new_code_cell(code_save.strip("\n")),
    nbf.v4.new_markdown_cell(markdown_solve),
    nbf.v4.new_code_cell(code_solve.strip("\n")),
    nbf.v4.new_markdown_cell(markdown_notes),
]

out_path = HERE / "gradient_term_calibration.ipynb"
nbf.write(nb, out_path)
print(f"wrote {out_path}")
