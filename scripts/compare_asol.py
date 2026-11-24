"""Check that a fid shift moved the aspect solution the way the gradient term intends.

Compare the aspect solution from the pipeline run without the fid shift with
the one from the run with it::

    python scripts/compare_asol.py ASOL_BEFORE ASOL_AFTER --acacent SHIFTED_ACACENT

Logs how far X-ray events would move in y and z angle (arcsec), from the
attitude turn and the dy/dz change between the two. With the shifted ACACENT
(its GTRM_DY and GTRM_DZ keywords), also logs that move as a fraction of
minus the term: 1.0 on each axis means the fid shift does what the term
intends; -1.0 means the sign is the other way.
"""

import argparse
import logging
import sys
from pathlib import Path

from astropy.io import fits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # this checkout's package
from astrometry_corr_gradient.fid_shift import predicted_event_shift

LOGGER = logging.getLogger("compare_asol")


def main(argv: list[str] | None = None) -> tuple[float, float] | None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("asol_before", help="aspect solution without the fid shift")
    parser.add_argument("asol_after", help="aspect solution with the fid shift")
    parser.add_argument("--acacent", help="the shifted ACACENT file, for the term")
    args = parser.parse_args(argv)
    shift_y, shift_z = predicted_event_shift(args.asol_before, args.asol_after)
    LOGGER.info(f'events would move y {shift_y:+.4f}", z {shift_z:+.4f}"')
    if not args.acacent:
        return None
    header = fits.getheader(args.acacent, "ACACENT")
    term_dy, term_dz = float(header["GTRM_DY"]), float(header["GTRM_DZ"])
    ratios = (shift_y / -term_dy, shift_z / -term_dz)
    LOGGER.info(
        f'term dy {term_dy:+.4f}", dz {term_dz:+.4f}": the move is {ratios[0]:.3f} and'
        f" {ratios[1]:.3f} times minus the term (1.0 is the intended correction)"
    )
    return ratios


if __name__ == "__main__":
    main()
