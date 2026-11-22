"""Tests for scripts/update_calalign_table.py, which copies the CALALIGN matrices into the package's data file."""

import sys
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from update_calalign_table import NO_DATE, read_calalign_files


def write_calalign(path: Path, misalign: np.ndarray, start: str | None) -> None:
    """A one-detector CALALIGN file with the given ACA misalignment."""
    columns = [
        fits.Column(name="INSTR_ID", format="7A", array=np.array(["ACIS-S"])),
        fits.Column(
            name="ACA_MISALIGN", format="9D", dim="(3,3)", array=misalign[np.newaxis]
        ),
    ]
    table = fits.BinTableHDU.from_columns(columns)
    if start is not None:
        table.header["CVSD0001"] = start
    fits.HDUList([fits.PrimaryHDU(), table]).writeto(path)


def test_an_identity_misalignment_reads_as_zero_offsets(tmp_path):
    write_calalign(
        tmp_path / "pcadD2013-01-19alignN0010.fits", np.eye(3), "2013-01-19T00:00:00"
    )
    table = read_calalign_files(tmp_path)
    row = table[0]
    assert (row["version"], row["caldb_version"], row["detector"]) == (
        "N0010",
        "4.10.0",
        "ACIS-S",
    )
    assert row["start"] == "2013-01-19T00:00:00"
    assert row["stop"] == NO_DATE
    assert row["dy"] == pytest.approx(0.0, abs=1e-9)
    assert row["dz"] == pytest.approx(0.0, abs=1e-9)


def test_a_small_yaw_reads_as_dy(tmp_path):
    angle = np.radians(10.0 / 3600)  # 10 arcsec of yaw, about Z
    yaw = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    write_calalign(tmp_path / "pcadD2013-01-19alignN0010.fits", yaw, None)
    row = read_calalign_files(tmp_path)[0]
    assert abs(row["dy"]) == pytest.approx(10.0, rel=1e-6)
    assert row["dz"] == pytest.approx(0.0, abs=1e-6)
    assert row["start"] == NO_DATE


def test_a_folder_without_calalign_files_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_calalign_files(tmp_path)
