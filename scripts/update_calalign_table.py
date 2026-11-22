"""Copy every CALALIGN matrix into astrometry_corr_gradient/data/calalign_matrices.ecsv.

Run it when a new CALALIGN file ships, with a folder holding every CALALIGN
file ever shipped, retired ones included::

    python scripts/update_calalign_table.py /path/to/calalign

One row per file and detector: the file, its version (N0008, ...), its CALDB
release, the detector, its validity start and stop (CVSD0001, CVED0001;
2050:001 when absent, as astromon has it), and the ACA misalignment as dy and
dz in arcsec (yaw and minus pitch, as astromon's ``get_offsets``). Needs the
Quaternion package; nothing else in this repo does.
"""

import re
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.table import Table

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # this checkout's package
from astrometry_corr_gradient.calalign import (
    CALDB_RELEASE_OF_FILE_VERSION,
    MATRICES_PATH,
)

FILE_NAME = re.compile(
    r"(?P<date>[0-9]{4}-[0-9]{2}-[0-9]{2})align(?P<version>N[0-9]{4})\.fits"
)
NO_DATE = "2050:001:00:00:00.00"


def offsets(aca_misalign: np.ndarray) -> tuple[float, float]:
    """dy, dz (arcsec) of one ACA misalignment matrix: yaw and minus pitch."""
    from Quaternion import Quat

    quaternion = Quat(transform=aca_misalign)
    return float(quaternion.yaw * 3600), float(-quaternion.pitch * 3600)


def read_calalign_files(calalign_dir: str | Path) -> Table:
    """Every matrix in the folder's CALALIGN files, one row per file and detector, in file-name order."""
    paths = sorted(Path(calalign_dir).glob("*.fits"))
    if not paths:
        raise FileNotFoundError(f"no CALALIGN files in {calalign_dir}")
    rows = []
    for path in paths:
        named = FILE_NAME.search(path.name)
        if named is None:
            raise ValueError(f"{path.name} is not a CALALIGN file name")
        version = named["version"]
        with fits.open(path) as hdus:
            header, data = hdus[1].header, hdus[1].data
            for entry in data:
                dy, dz = offsets(np.asarray(entry["ACA_MISALIGN"], float))
                rows.append(
                    {
                        "file": path.name,
                        "version": version,
                        "caldb_version": CALDB_RELEASE_OF_FILE_VERSION[version],
                        "detector": str(entry["INSTR_ID"]).strip(),
                        "start": str(header.get("CVSD0001", NO_DATE)),
                        "stop": str(header.get("CVED0001", NO_DATE)),
                        "dy": dy,
                        "dz": dz,
                    }
                )
    table = Table(rows=rows)
    table["dy"].unit = table["dz"].unit = "arcsec"
    descriptions = {
        "file": "the CALALIGN file",
        "version": "its file version",
        "caldb_version": "the CALDB release that version shipped in",
        "detector": "the detector",
        "start": "validity start (CVSD0001)",
        "stop": "validity stop (CVED0001)",
        "dy": "ACA misalignment yaw",
        "dz": "ACA misalignment, minus pitch",
    }
    for name, description in descriptions.items():
        table[name].description = description
    table.meta["description"] = (
        "every CALALIGN matrix, one row per file and detector (scripts/update_calalign_table.py)"
    )
    return table


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(
            "usage: python scripts/update_calalign_table.py /path/to/calalign"
        )
    table = read_calalign_files(sys.argv[1])
    table.meta["source"] = str(Path(sys.argv[1]).resolve())
    MATRICES_PATH.parent.mkdir(exist_ok=True)
    table.write(MATRICES_PATH, format="ascii.ecsv", overwrite=True)
    reread = Table.read(MATRICES_PATH, format="ascii.ecsv")
    for name in ("dy", "dz"):
        if not np.array_equal(np.asarray(reread[name]), np.asarray(table[name])):
            raise ValueError(f"{name} does not read back exactly")
    print(
        f"wrote {MATRICES_PATH}: {len(table)} matrices from {len(set(table['file']))} files"
    )


if __name__ == "__main__":
    main()
