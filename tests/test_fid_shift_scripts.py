"""Tests for scripts/shift_fids.py and scripts/compare_asol.py, end to end on small files."""

import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from Quaternion import Quat

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import compare_asol
import shift_fids
from test_fid_shift import DATE, write_acacent, write_asol, write_fidprops, write_obc

from astrometry_corr_gradient import gradient_term
from astrometry_corr_gradient.fid_shift import ARCSEC_PER_MM_DYDZ, DATA_NORMAL, EGAUSS


@pytest.fixture
def pipeline_files(tmp_path: Path) -> dict[str, Path]:
    """An observation's FIDPROPS, ACACENT (one fid slot, one star slot) and two OBC files."""
    write_fidprops(tmp_path / "fidprops.fits", [(1, "ACIS-S-3", "GOOD")])
    write_acacent(
        tmp_path / "acen.fits",
        [
            (100.0, 1, EGAUSS, DATA_NORMAL, 0.4, 0.6),
            (200.0, 1, EGAUSS, DATA_NORMAL, 0.4, 0.6),
            (150.0, 4, EGAUSS, DATA_NORMAL, 0.1, 0.1),
        ],
    )
    write_obc(
        tmp_path / "obc_a.fits",
        [90.0, 100.0, 150.0],
        [9.0, -0.26, -0.27],
        [9.0, 0.142, 0.142],
    )
    write_obc(tmp_path / "obc_b.fits", [200.0, 210.0], [-0.28, 9.0], [0.143, 9.0])
    (tmp_path / "obc.lis").write_text("obc_a.fits\nobc_b.fits\n")
    return {
        "acacent": tmp_path / "acen.fits",
        "fidprops": tmp_path / "fidprops.fits",
        "obc_list": tmp_path / "obc.lis",
    }


def test_shift_fids_in_place_keeps_the_original_and_shifts_the_fids(pipeline_files):
    shift_fids.main(
        [
            str(pipeline_files["acacent"]),
            str(pipeline_files["fidprops"]),
            "@" + str(pipeline_files["obc_list"]),
            "--in-place",
        ]
    )
    means = {
        "OOBAGRD3": np.mean([-0.26, -0.27, -0.28]),
        "OOBAGRD6": np.mean([0.142, 0.142, 0.143]),
    }
    term = gradient_term.gradient_term(DATE, means)
    backup = pipeline_files["acacent"].with_name("acen.fits.before_gradient_term")
    with fits.open(backup) as hdus:
        assert list(hdus[1].data["ang_y_sm"]) == pytest.approx([0.4, 0.4, 0.1])
    with fits.open(pipeline_files["acacent"]) as hdus:
        data = hdus[1].data
        assert data["ang_y_sm"][:2] == pytest.approx(
            0.4 - term["dy"][0] / 3600, abs=1e-7
        )
        assert data["ang_z_sm"][:2] == pytest.approx(
            0.6 - term["dz"][0] / 3600, abs=1e-7
        )
        assert data["ang_y_sm"][2] == pytest.approx(0.1)  # the star slot
        # the OBC file holds float32 gradients, as the real ones do
        assert hdus[1].header["GTRM_DY"] == pytest.approx(term["dy"][0], abs=1e-6)


def test_shift_fids_dry_run_changes_nothing(pipeline_files, tmp_path):
    shutil.copyfile(pipeline_files["acacent"], tmp_path / "copy.fits")
    shift_fids.main(
        [
            str(pipeline_files["acacent"]),
            str(pipeline_files["fidprops"]),
            "--means",
            "-0.27",
            "0.142",
            "--dry-run",
        ]
    )
    assert (tmp_path / "copy.fits").read_bytes() == pipeline_files[
        "acacent"
    ].read_bytes()


def test_shift_fids_needs_an_output_choice(pipeline_files):
    with pytest.raises(SystemExit):
        shift_fids.main(
            [
                str(pipeline_files["acacent"]),
                str(pipeline_files["fidprops"]),
                "--means",
                "0",
                "0",
            ]
        )


def test_compare_asol_reports_the_events_moving_by_minus_the_term(tmp_path, caplog):
    times = np.arange(5.0)
    attitude = Quat([150.0, 40.0, 30.0])
    write_asol(tmp_path / "before.fits", times, attitude)
    # a term of +0.4" in dy and +0.2" in dz should move events by -0.4" and -0.2"
    write_asol(
        tmp_path / "after.fits",
        times,
        attitude,
        dy_mm=-0.4 / ARCSEC_PER_MM_DYDZ,
        dz_mm=-0.2 / ARCSEC_PER_MM_DYDZ,
    )
    write_acacent(tmp_path / "acen.fits", [(1.0, 1, EGAUSS, DATA_NORMAL, 0.4, 0.6)])
    with fits.open(tmp_path / "acen.fits", mode="update") as hdus:
        hdus[1].header["GTRM_DY"], hdus[1].header["GTRM_DZ"] = 0.4, 0.2
    caplog.set_level("INFO")
    ratios = compare_asol.main(
        [
            str(tmp_path / "before.fits"),
            str(tmp_path / "after.fits"),
            "--acacent",
            str(tmp_path / "acen.fits"),
        ]
    )
    assert ratios == pytest.approx((1.0, 1.0), abs=1e-3)
    assert "minus the term" in caplog.text
