"""Shift the fid centroids in an ACACENT file by the observation's gradient term.

For testing in the aspect pipeline: run the pipeline through aca_corr_fid,
stop, run this on the ACACENT file, then run the rest of the pipeline::

    python scripts/shift_fids.py ACACENT FIDPROPS OBCFILE [OBCFILE ...] --in-place
    python scripts/shift_fids.py ACACENT FIDPROPS @obc.lis --outfile shifted_acen1.fits

OBC files are the obcf*_eng0 files given to aca_corr_fid (``@list`` works as
there). The script logs the time range, the mean gradients (compare them with
aca_corr_fid's verbose "Mean Gradient 3" and "Mean Gradient 6"), the term and
the shift. ``--means G3 G6`` uses given means instead of the OBC files;
``--in-place`` keeps the original as ``<name>.before_gradient_term``;
``--dry-run`` writes nothing. Afterwards, ``scripts/compare_asol.py`` checks
the new aspect solution.
"""

import argparse
import logging
import shutil
import sys
from pathlib import Path

from astropy.io import fits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # this checkout's package
from astrometry_corr_gradient.fid_shift import (
    EGAUSS,
    apply_fid_shift,
    fid_shift,
    fid_time_range,
    good_fid_slots,
    mean_gradients,
    read_obc_gradients,
)

LOGGER = logging.getLogger("shift_fids")


def expand_lists(names: list[str]) -> list[Path]:
    """File names, with each ``@list`` replaced by the files it lists.

    A listed file not found as given is taken relative to the list's folder.
    """
    paths = []
    for name in names:
        if not name.startswith("@"):
            paths.append(Path(name))
            continue
        listing = Path(name[1:])
        for line in listing.read_text().splitlines():
            if line.strip():
                path = Path(line.strip())
                paths.append(path if path.exists() else listing.parent / path)
    return paths


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("acacent", help="ACACENT file, after aca_corr_fid")
    parser.add_argument("fidprops", help="FIDPROPS file given to aca_corr_fid")
    parser.add_argument("obcfiles", nargs="*", help="OBC engineering files, or @list")
    parser.add_argument(
        "--means",
        nargs=2,
        type=float,
        metavar=("OOBAGRD3", "OOBAGRD6"),
        help="mean gradients (K) instead of the OBC files",
    )
    parser.add_argument(
        "--date",
        help="observation date for the reference quarter (default: the ACACENT DATE-OBS)",
    )
    parser.add_argument(
        "--algorithm",
        type=int,
        default=EGAUSS,
        help="centroid algorithm of the rows that set the time range (default 8, EGAUSS)",
    )
    output = parser.add_mutually_exclusive_group(required=True)
    output.add_argument("--outfile", help="write the shifted ACACENT here")
    output.add_argument(
        "--in-place", action="store_true", help="overwrite ACACENT, keeping a copy"
    )
    output.add_argument(
        "--dry-run", action="store_true", help="only log what would be done"
    )
    return parser


def main(argv: list[str] | None = None) -> dict:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = get_parser().parse_args(argv)
    acacent = Path(args.acacent)
    slots = good_fid_slots(args.fidprops)
    start, stop = fid_time_range(acacent, slots, args.algorithm)
    LOGGER.info(
        f"good fid slots {slots}; fid centroids from {start:.3f} to {stop:.3f}"
        f" ({stop - start:.0f} s)"
    )
    if args.means:
        means = dict(zip(("OOBAGRD3", "OOBAGRD6"), args.means, strict=True))
        LOGGER.info(
            f"given means: OOBAGRD3 {means['OOBAGRD3']:.6f} K, OOBAGRD6 {means['OOBAGRD6']:.6f} K"
        )
    else:
        if not args.obcfiles:
            raise SystemExit("give the OBC engineering files, or --means")
        means = mean_gradients(
            read_obc_gradients(expand_lists(args.obcfiles)), start, stop
        )
        LOGGER.info(
            f"mean gradients: OOBAGRD3 {means['OOBAGRD3']:.6f} K,"
            f" OOBAGRD6 {means['OOBAGRD6']:.6f} K"
        )
    date = args.date or fits.getheader(acacent, "ACACENT")["DATE-OBS"]
    shift = fid_shift(date, means)
    LOGGER.info(
        f'{date}, reference quarter {shift["quarter"]}: term dy {shift["dy_arcsec"]:+.4f}",'
        f' dz {shift["dz_arcsec"]:+.4f}"; fid ang_y_sm -= {shift["ang_y_deg"]:.9f} deg,'
        f" ang_z_sm -= {shift['ang_z_deg']:.9f} deg"
    )
    if args.dry_run:
        LOGGER.info("dry run: nothing written")
        return shift
    if args.in_place:
        backup = acacent.with_name(acacent.name + ".before_gradient_term")
        shutil.copyfile(acacent, backup)
        LOGGER.info(f"original kept as {backup}")
        outfile = acacent
    else:
        outfile = Path(args.outfile)
    rows = apply_fid_shift(acacent, outfile, slots, shift)
    LOGGER.info(f"wrote {outfile}: {rows} fid rows shifted")
    return shift


if __name__ == "__main__":
    main()
