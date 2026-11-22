"""Generate notebooks/gradient_term_validation.ipynb.

Run this script to (re)build the notebook from scratch; the notebook itself
is what gets committed and opened, this script is just how it's produced.
The residual check of astrometry_corr_gradient/gradient_term.py on the matched
sources (gradient_term_calibration_set.ecsv): the module's data against a fresh fit from the
calibration set, then the offsets left after the CALALIGN values alone and with the
gradient term, by period and over time, for the planned scheme (history
within each quarter, lagging from the release) against lagging throughout.
"""

from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).parent

markdown_intro = """\
# Gradient term: does it work?

The gradient term from `astrometry_corr_gradient/gradient_term.py`, checked on
the matched sources in `gradient_term_calibration_set.ecsv`, which
`gradient_term_calibration.ipynb` makes.

**The plan**

- **History:** each quarter is corrected with its own values.
- **From the 2026 release:** each new quarter uses the previous quarter's
  values. They never change.
- **Both parts follow this rule:** the CALALIGN value and the term's
  reference come from the same quarter.

The check compares the plan with **lagging throughout**: the previous
quarter's values everywhere.

**The CALALIGN values**

- **Before 2021-07-02:** today's CALDB.
- **After 2021-07-02:** today's CALDB has no values, so the check makes them
  the old way: each quarter's median offset, without the term, plus one
  constant per detector.
- **No source corrects itself:** its own-quarter median leaves out the source
  and the sources linked to it.

**What it shows**

- **The module's numbers** come back exactly from the calibration set.
- **The term helps in every period.** dr r68 / r90 drops from 0.349" / 0.578"
  to 0.287" / 0.472" before 2021-07-02, and from 0.594" / 0.879" to 0.396" /
  0.637" since.
- **The plan beats lagging throughout since 2021-07-02**, by 0.030" / 0.044".
  Before 2021-07-02 they tie.
- **The biggest difference is Safe Mode #10, in late 2022.** Lagging spikes
  there; the plan doesn't.

dr is the offset on the sky. r68 and r90 are the radii that hold 68% and 90%
of the sources, in arcsec. For dz and dy alone they are p68 and p90.
"""

code_setup = r"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
from astropy.time import Time

# this checkout's package, before any installed copy
sys.path.insert(0, str(Path.cwd().parent))
sys.path.insert(1, str(Path.cwd().parent / "scripts"))
from update_gradient_term_data import (
    RELEASE_QUARTER,
    handoff_data,
)

from astrometry_corr_gradient import gradient_term
from astrometry_corr_gradient.groups import block_bootstrap_difference

AXES = gradient_term.AXES
inputs = Table.read("gradient_term_calibration_set.ecsv", format="ascii.ecsv")
dates = np.asarray(inputs["date"]).astype(str)
years = np.asarray(inputs["year"], float)
assert np.all(np.diff(years) >= 0), "the sources should be in time order"
detectors = np.asarray(inputs["detector"]).astype(str)
groups = np.asarray(inputs["linked_group"])
offsets = {
    axis_name: np.asarray(inputs[f"{axis_name}_rebased"], float) for axis_name in AXES
}
gradients = {
    name: np.asarray(inputs[name], float) for name in gradient_term.GRADIENT_MSIDS
}
quarters = gradient_term.quarter_index(dates)
export_quarter = int(
    gradient_term.quarter_index(inputs.meta["made"].removesuffix(" UTC"))[0]
)
release_quarter = gradient_term.label_index(RELEASE_QUARTER)
print(
    f"{len(inputs):,} matched sources, {dates[0][:10]} to {dates[-1][:10]},"
    f" made {inputs.meta['made']}; lagging from {RELEASE_QUARTER}"
)
"""

markdown_data = """\
## 1. The module's numbers

Remake the coefficients, sigmas, reference table and test vectors from the
calibration set, `gradient_term_calibration_set.ecsv`. They should match the
module exactly.
"""

code_data = r"""
coefficients, sigmas, references, vectors = handoff_data(inputs)
assert coefficients == gradient_term.COEFFICIENTS
assert sigmas == gradient_term.SIGMAS
embedded = gradient_term.reference_table()
assert list(embedded["quarter"]) == list(references["quarter"])
assert list(embedded["from_quarter"]) == list(references["from_quarter"])
for name in gradient_term.GRADIENT_MSIDS:
    assert np.allclose(embedded[name], references[name], rtol=0, atol=1e-9)
assert list(gradient_term.vector_table()["obsid"]) == list(vectors["obsid"])
print(
    "the module's coefficients, sigmas, reference table and test vectors match a fresh fit from the calibration set"
)
print("coefficients (arcsec per K):", gradient_term.COEFFICIENTS)
print("sigmas (K):", gradient_term.SIGMAS)
"""

markdown_residuals = """\
## 2. The residuals

The offset left after the CALALIGN values, alone and with the term, by period.
A source without telemetry gets no term, as in processing.

"Better" means the difference holds up when the data are resampled in 30-day
blocks; "tie" means it doesn't.
"""

code_residuals = r'''
CUTOVER_DATE = "2021-07-02"  # today's last CALALIGN value
EXEMPT = {"dz": (), "dy": ("HRC-I",)}  # detectors with no constant
MINIMUM = gradient_term.MIN_OBSERVATIONS


def previous_level(offset: np.ndarray, index: int) -> float:
    """The previous quarter's median of all sources (the latest earlier ones if it has too few)."""
    before = np.isfinite(offset) & (quarters < index)
    previous = before & (quarters == index - 1)
    chosen = (
        previous if previous.sum() >= MINIMUM else np.flatnonzero(before)[-MINIMUM:]
    )
    return float(np.median(offset[chosen])) if np.size(offset[chosen]) else np.nan


def quarter_levels(offset: np.ndarray, own_before: int) -> np.ndarray:
    """Each source's level: before own_before, its own quarter's median leaving out its linked group;
    from then on, the previous quarter's median."""
    finite = np.isfinite(offset)
    level = np.full(len(offset), np.nan)
    for index in np.unique(quarters):
        in_quarter = quarters == index
        if index >= own_before:
            level[in_quarter] = previous_level(offset, index)
            continue
        for group in np.unique(groups[in_quarter]):
            others = finite & in_quarter & (groups != group)
            members = in_quarter & (groups == group)
            enough = others.sum() >= MINIMUM
            level[members] = (
                np.median(offset[others]) if enough else previous_level(offset, index)
            )
    return level


def after_calalign(own_before: int) -> dict[str, np.ndarray]:
    """The offsets after the CALALIGN values: today's CALDB to 2021-07-02, then the quarterly stand-in."""
    out = {}
    for axis_name in AXES:
        first = offsets[axis_name] - quarter_levels(offsets[axis_name], own_before)
        constants = {
            detector: 0.0
            if detector in EXEMPT[axis_name]
            else float(np.median(first[np.isfinite(first) & (detectors == detector)]))
            for detector in np.unique(detectors)
        }
        stand_in = first - np.array([constants[detector] for detector in detectors])
        today = offsets[axis_name] + np.asarray(
            inputs[f"{axis_name}_caldb_shift"], float
        )
        out[axis_name] = np.where(dates >= CUTOVER_DATE, stand_in, today)
    return out


# within chunk: own quarters before the release (and before the quarter still filling); lagging throughout:
# the previous quarter everywhere, for the medians and the references alike
own_before = min(release_quarter, export_quarter)
_, first_rows = np.unique(np.asarray(inputs["obsid"]), return_index=True)
lagging_table = gradient_term.quarter_references(
    quarters[first_rows],
    {name: column[first_rows] for name, column in gradients.items()},
    last_complete_quarter=export_quarter - 1,
    lagging_from=int(quarters.min()),
)
for name in gradient_term.GRADIENT_MSIDS:
    lagging_table[name] = np.round(lagging_table[name], 5)
terms = {
    "within chunk": gradient_term.gradient_term(dates, gradients),
    "lagging": gradient_term.gradient_term(
        dates, gradients, lagging_table, gradient_term.COEFFICIENTS
    ),
}
calaligns = {
    "within chunk": after_calalign(own_before),
    "lagging": after_calalign(int(quarters.min())),
}
VERSIONS = {}
for scheme in ("within chunk", "lagging"):
    VERSIONS[f"{scheme}: CALALIGN alone"] = calaligns[scheme]
    VERSIONS[f"{scheme}: CALALIGN + term"] = {
        axis_name: calaligns[scheme][axis_name] - terms[scheme][axis_name]
        for axis_name in AXES
    }
for by_axis in VERSIONS.values():
    by_axis["dr"] = np.hypot(by_axis["dz"], by_axis["dy"])
scored = np.logical_and.reduce(
    [np.isfinite(by_axis["dr"]) for by_axis in VERSIONS.values()]
    + [~term["no_reference"] for term in terms.values()]
)
print(
    f"{int(scored.sum()):,} sources, {int((scored & ~terms['within chunk']['no_telemetry']).sum()):,} with the term"
)


def decimal_year(date: str) -> float:
    return float(Time(date).decimalyear)


CUTOVER = decimal_year(CUTOVER_DATE)
WINDOWS = [
    ("mission", -np.inf, np.inf),
    ("before 2021-07-02", -np.inf, CUTOVER),
    (
        "2006-12-10 to 2012-09-13",
        decimal_year("2006-12-10"),
        decimal_year("2012-09-13"),
    ),
    ("2013-01-19 to 2021-07-02", decimal_year("2013-01-19"), CUTOVER),
    ("since 2021-07-02", CUTOVER, np.inf),
    ("2023+", 2023.0, np.inf),
]


def score_rows(axis_names: tuple[str, ...], names: list[str]) -> Table:
    rows = []
    for axis_name in axis_names:
        for name in names:
            row = {"offset": axis_name, "correction": name}
            for window, low, high in WINDOWS:
                chosen = np.abs(
                    VERSIONS[name][axis_name][scored & (years >= low) & (years < high)]
                )
                row[window] = (
                    f"{np.percentile(chosen, 68):.3f} / {np.percentile(chosen, 90):.3f}"
                )
            rows.append(row)
    return Table(rows=rows)


print("dr: 68% / 90% of |offset| (arcsec)")
score_rows(("dr",), list(VERSIONS)).pprint(max_width=-1)
print()
print("dz and dy, within chunk: 68% / 90% of |offset| (arcsec)")
score_rows(
    ("dz", "dy"), ["within chunk: CALALIGN alone", "within chunk: CALALIGN + term"]
).pprint(max_width=-1)
print()


def versus(name: str, reference: str) -> None:
    print(f"dr, {name} minus {reference}, paired 30-day-block bootstraps")
    for window, low, high in WINDOWS:
        both = scored & (years < high)
        verdicts = []
        for statistic, q in (("r68", 68), ("r90", 90)):
            difference, lower, upper = block_bootstrap_difference(
                np.where(both, VERSIONS[name]["dr"], np.nan),
                np.where(both, VERSIONS[reference]["dr"], np.nan),
                years,
                lambda chosen, q=q: np.percentile(np.abs(chosen), q),
                year_min=low,
            )
            verdict = "better" if upper < 0 else ("worse" if lower > 0 else "tie")
            verdicts.append(f"{statistic} {difference:+.3f} {verdict}")
        print(f"    {window}: " + "; ".join(verdicts))


versus("within chunk: CALALIGN + term", "within chunk: CALALIGN alone")
versus("within chunk: CALALIGN + term", "lagging: CALALIGN + term")
'''

markdown_time = """\
## 3. Over time

dr r68 and r90 in 90-day windows.
"""

code_time = r"""
HALF_WINDOW, STEP = 45 / 365.25, 15 / 365.25
centers = np.arange(years[scored].min() + HALF_WINDOW, years.max() - HALF_WINDOW, STEP)


def running(values: np.ndarray, q: float) -> np.ndarray:
    out = np.full(len(centers), np.nan)
    for index, center in enumerate(centers):
        side = (years < CUTOVER) if center < CUTOVER else (years >= CUTOVER)
        near = scored & side & (np.abs(years - center) < HALF_WINDOW)
        if near.sum() >= 20:
            out[index] = np.percentile(values[near], q)
    return out


STYLES = {
    "within chunk: CALALIGN alone": ("0.35", 2.0, "-"),
    "within chunk: CALALIGN + term": ("#1baf7a", 2.4, "-"),
    "lagging: CALALIGN + term": ("0.05", 1.3, (0, (1, 2))),
}
plt.rcParams.update({"font.size": 13})
figure, axes = plt.subplots(2, 1, figsize=(14, 8.5), sharex=True)
for axis, (statistic, q) in zip(axes, (("r68", 68), ("r90", 90)), strict=True):
    for name, (color, width, style) in STYLES.items():
        axis.plot(
            centers,
            running(VERSIONS[name]["dr"], q),
            color=color,
            linewidth=width,
            linestyle=style,
            label=name,
        )
    if statistic == "r90":
        axis.axhline(
            1.0,
            color="black",
            linewidth=1.0,
            linestyle="--",
            label='specification: 90% within 1.0"',
        )
    axis.axvline(CUTOVER, color="0.4", linewidth=1.2)
    axis.set_ylim(bottom=0)
    axis.set_ylabel(f"dr {statistic} (arcsec)")
    axis.set_title(
        f"dr {statistic} in 90-day windows", loc="left", fontsize=13, fontweight="bold"
    )
    axis.grid(color="0.93", linewidth=0.6)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    axis.legend(loc="upper left", fontsize=11, frameon=False)
axes[-1].set_xlim(2004.0, years.max() + 0.1)
axes[-1].set_xticks(range(2004, 2027, 2), [str(year) for year in range(2004, 2027, 2)])
figure.suptitle(
    "The matched sources after the CALALIGN values, alone and with the gradient term",
    x=0.01,
    ha="left",
)
figure.tight_layout(rect=(0, 0, 1, 0.96))
figure.savefig("gradient_term_validation.png", dpi=150)
plt.show()
"""

markdown_notes = """\
## Notes

- **In-sample:** the coefficients were fit on these same sources.
- **2026Q4** was still filling at the export, so it lags in both schemes.
- **The 2017-09 spike** is five M30 sources about 2.5" off, with or without
  the term.
"""

nb = nbf.v4.new_notebook()
nb.cells = [
    nbf.v4.new_markdown_cell(markdown_intro),
    nbf.v4.new_code_cell(code_setup.strip("\n")),
    nbf.v4.new_markdown_cell(markdown_data),
    nbf.v4.new_code_cell(code_data.strip("\n")),
    nbf.v4.new_markdown_cell(markdown_residuals),
    nbf.v4.new_code_cell(code_residuals.strip("\n")),
    nbf.v4.new_markdown_cell(markdown_time),
    nbf.v4.new_code_cell(code_time.strip("\n")),
    nbf.v4.new_markdown_cell(markdown_notes),
]

out_path = HERE / "gradient_term_validation.ipynb"
nbf.write(nb, out_path)
print(f"wrote {out_path}")
