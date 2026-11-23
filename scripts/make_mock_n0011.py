"""Write the mock N0011 CALALIGN matrices into astrometry_corr_gradient/data/calalign_mock_n0011.ecsv.

Made from the calibration set (notebooks/gradient_term_calibration_set.ecsv)
with astrometry_corr_gradient.mock_n0011: one matrix per calendar quarter and
detector from 2021-07-02, each quarter's own median for history and the
previous quarter's from the release (RELEASE_QUARTER) on. Run it after a new
calibration set::

    python scripts/make_mock_n0011.py
"""

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # this checkout's package
sys.path.insert(1, str(ROOT / "scripts"))
from astropy.table import Table
from update_gradient_term_data import INPUTS_PATH, RELEASE_QUARTER

from astrometry_corr_gradient import gradient_term
from astrometry_corr_gradient.mock_n0011 import MOCK_RELEASE, mock_matrices

LOGGER = logging.getLogger(__name__)
MOCK_PATH = ROOT / "astrometry_corr_gradient" / "data" / "calalign_mock_n0011.ecsv"


def build(calibration: Table) -> Table:
    """The mock matrices for this calibration set, with where they came from in the header."""
    made = calibration.meta["made"]
    last_complete = int(gradient_term.quarter_index(made.removesuffix(" UTC"))[0]) - 1
    mock = mock_matrices(
        calibration,
        last_complete_quarter=last_complete,
        lagging_from=gradient_term.label_index(RELEASE_QUARTER),
    )
    mock.remove_columns(["start_secs", "stop_secs"])
    mock.meta.update(
        {
            "description": (
                "mock N0011 CALALIGN matrices: one per calendar quarter and detector from 2021-07-02,"
                " each zeroing its quarter's median rebased offset (scripts/make_mock_n0011.py)"
            ),
            "caldb_release": f"{MOCK_RELEASE}, a placeholder",
            "lagging_from": RELEASE_QUARTER,
            "made_from": f"gradient_term_calibration_set.ecsv, made {made}, database {calibration.meta.get('database', '')}",
        }
    )
    return mock


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    mock = build(Table.read(INPUTS_PATH, format="ascii.ecsv"))
    mock.write(MOCK_PATH, format="ascii.ecsv", overwrite=True)
    LOGGER.info(
        "wrote %s: %d matrices, %s to %s",
        MOCK_PATH,
        len(mock),
        mock["start"][0],
        mock["start"][-1],
    )


if __name__ == "__main__":
    main()
