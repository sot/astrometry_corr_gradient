"""Write the gradient term's data into astrometry_corr_gradient/gradient_term.py from the calibration set.

The calibration set, gradient_term_calibration_set.ecsv, is made by
gradient_term_calibration.ipynb from the astromon database.

The coefficients (fit within calendar quarters on every matched source), the
reference table (one row per calendar quarter, from one value per observation)
and the test vectors, all made with the module's own functions, replace
everything after the module's data marker. Run it after a new export, from
anywhere:

    python scripts/update_gradient_term_data.py
"""

import sys
from pathlib import Path

import numpy as np
from astropy.table import Table

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # this checkout's package, before any installed copy
from astrometry_corr_gradient import calibration
from astrometry_corr_gradient import gradient_term as module

MODULE_PATH = ROOT / "astrometry_corr_gradient" / "gradient_term.py"
INPUTS_PATH = ROOT / "notebooks" / "gradient_term_calibration_set.ecsv"
DATA_MARKER = "# ---- the data:"
# the first quarter processed with lagging references (the 2026 release); earlier quarters use their own
RELEASE_QUARTER = "2026Q4"
# one matched observation from each; the last one lags
TEST_QUARTERS = ("2008Q2", "2012Q3", "2016Q4", "2022Q4", "2025Q3", "2026Q4")


def handoff_data(
    inputs: Table,
    test_quarters: tuple[str, ...] = TEST_QUARTERS,
    lagging_from: str = RELEASE_QUARTER,
) -> tuple[dict, dict, Table, Table]:
    """The coefficients, the sigmas, the reference table and the test vectors, as the module holds them.

    The coefficients are rounded to 3 decimals, the sigmas and the references
    to 5, and the test vectors' terms are computed from those rounded values.
    The sigmas use the same sources as the coefficients (every matched source
    with both gradients). Quarters before ``lagging_from`` use their own
    median, later ones the previous quarter's; the export's own quarter
    (inputs.meta["made"]) was still filling, so it sets no reference.
    """
    dates = np.asarray(inputs["date"]).astype(str)
    obsids = np.asarray(inputs["obsid"])
    gradients = {
        name: np.asarray(inputs[name], float) for name in module.GRADIENT_MSIDS
    }
    quarters = module.quarter_index(dates)
    export_quarter = int(
        module.quarter_index(inputs.meta["made"].removesuffix(" UTC"))[0]
    )
    fitted, fitted_sigmas, references = calibration.solve(
        inputs,
        last_complete_quarter=export_quarter - 1,
        lagging_from=module.label_index(lagging_from),
    )
    coefficients = {
        axis_name: {name: round(k, 3) for name, k in by_gradient.items()}
        for axis_name, by_gradient in fitted.items()
    }
    sigmas = {name: round(sigma, 5) for name, sigma in fitted_sigmas.items()}
    for name in module.GRADIENT_MSIDS:
        references[name] = np.round(references[name], 5)
    both = np.all([np.isfinite(column) for column in gradients.values()], axis=0)

    # one observation from each test quarter, then one without telemetry
    rows = [
        int(np.flatnonzero(both & (quarters == module.label_index(label)))[0])
        for label in test_quarters
    ]
    rows.append(int(np.flatnonzero(~both)[0]))
    vectors = Table(
        {
            "obsid": obsids[rows],
            "date": dates[rows],
            **{
                name: np.round(gradients[name][rows], 5)
                for name in module.GRADIENT_MSIDS
            },
        }
    )
    term = module.gradient_term(
        vectors["date"],
        {name: vectors[name] for name in module.GRADIENT_MSIDS},
        references,
        coefficients,
    )
    for axis_name in module.AXES:
        vectors[axis_name] = np.round(term[axis_name], 4)
    vectors["no_term"] = np.where(
        term["no_telemetry"],
        "no telemetry",
        np.where(term["no_reference"], "no table row", "-"),
    )
    return coefficients, sigmas, references, vectors


def data_block(
    coefficients: dict,
    sigmas: dict,
    references: Table,
    vectors: Table,
    provenance: str,
) -> str:
    """The module's data section, as Python source."""
    coefficient_lines = [
        f'    "{axis_name}": {{'
        + ", ".join(f'"{name}": {by_gradient[name]}' for name in module.GRADIENT_MSIDS)
        + "},"
        for axis_name, by_gradient in coefficients.items()
    ]
    reference_lines = [
        f"{row['quarter']},{row['start']},{row['from_quarter']},{row['observations']},"
        f"{row['OOBAGRD3']:.5f},{row['OOBAGRD6']:.5f}"
        for row in references
    ]
    vector_lines = [
        f"{row['obsid']},{row['date']},{row['OOBAGRD3']:.5f},{row['OOBAGRD6']:.5f},"
        f"{row['dz']:.4f},{row['dy']:.4f},{row['no_term']}"
        for row in vectors
    ]
    return "\n".join(
        [
            f"{DATA_MARKER} written by scripts/update_gradient_term_data.py, not by hand ----",
            f"# {provenance}",
            "",
            "COEFFICIENTS = {  # arcsec per K of each gradient",
            *coefficient_lines,
            "}",
            "",
            "SIGMAS = {  # K: each gradient's typical swing inside a calendar quarter",
            *[f'    "{name}": {sigmas[name]},' for name in module.GRADIENT_MSIDS],
            "}",
            "",
            "# observations in each quarter use its row: each gradient's median (K) over from_quarter's",
            "# observations: its own quarter for history, the previous quarter from the release on",
            'REFERENCE_CSV = """\\',
            "quarter,start,from_quarter,observations,OOBAGRD3,OOBAGRD6",
            *reference_lines,
            '"""',
            "",
            "# matched observations, their gradients (K) and the dz and dy terms they get (arcsec); the last",
            "# one gets no term",
            'TEST_VECTOR_CSV = """\\',
            "obsid,date,OOBAGRD3,OOBAGRD6,dz,dy,no_term",
            *vector_lines,
            '"""',
            "",
        ]
    )


def main() -> None:
    inputs = Table.read(INPUTS_PATH, format="ascii.ecsv")
    coefficients, sigmas, references, vectors = handoff_data(inputs)
    provenance = (
        f"from gradient_term_calibration_set.ecsv, made {inputs.meta['made']}, astromon database"
        f" {inputs.meta['database'].split('/')[-1]}"
    )
    source = MODULE_PATH.read_text()
    marker = source.index(DATA_MARKER)
    MODULE_PATH.write_text(
        source[:marker]
        + data_block(coefficients, sigmas, references, vectors, provenance)
    )
    print(
        f"wrote {len(references)} quarters ({references['quarter'][0]} to {references['quarter'][-1]})"
    )
    print(f"and {len(vectors)} test vectors into {MODULE_PATH}")
    print("coefficients (arcsec per K):", coefficients)
    print("sigmas (K):", sigmas)


if __name__ == "__main__":
    main()
