# astrometry_corr_gradient

The gradient term for Chandra's absolute astrometry: a correction to each
observation's dz and dy from the periscope gradient monitors OOBAGRD3 and
OOBAGRD6, on top of the CALALIGN alignment. The code, its numbers, and the
calibration that makes them from an astromon database.

```
term = c3 * (OOBAGRD3 - OOBAGRD3 reference) + c6 * (OOBAGRD6 - OOBAGRD6 reference)
```

The term is the expected X-ray minus catalog offset, in arcsec; subtract it
from the X-ray position's dz and dy. The references are one pair per calendar
quarter.

## What is where

- `astrometry_corr_gradient/gradient_term.py`: the term itself, with its
  numbers at the end of the file (coefficients, sigmas, reference table, test
  vectors). Needs only numpy and astropy. This is the handoff.
- `astrometry_corr_gradient/calibration.py`: from an astromon database to the
  numbers: matched sources, dwells, calibration set, solve.
- `astrometry_corr_gradient/crossmatch.py`, `calalign.py`, `telemetry.py`,
  `groups.py`: the cross match, the rebase onto each detector's latest
  CALALIGN matrix, the telemetry helpers, and linked groups and bootstraps.
- `astrometry_corr_gradient/data/calalign_matrices.ecsv`: every CALALIGN
  matrix ever shipped, retired ones included, one row per file and detector.
- `astrometry_corr_gradient/mock_n0011.py` and
  `data/calalign_mock_n0011.ecsv`: a mock N0011, the quarterly CALALIGN
  matrices from 2021-07-02 that the plan would ship, with a placeholder CALDB
  release. Kept apart from the shipped matrices; the residual check uses them.
- `scripts/update_gradient_term_data.py`: writes the numbers into
  `gradient_term.py` from the calibration set.
- `scripts/update_calalign_table.py`: rewrites the CALALIGN data file from a
  folder of CALALIGN files, when a new one ships.
- `scripts/make_mock_n0011.py`: rewrites the mock N0011 file from the
  calibration set.
- `notebooks/`: the calibration, the prototype for the developers, and the
  residual check. Each notebook is built by its `build_*.py` script.

## Making new numbers

1. In `notebooks/gradient_term_calibration.ipynb`, set `DATABASE` to an
   astromon database and run it. It saves
   `notebooks/gradient_term_calibration_set.ecsv`. Telemetry and safing
   intervals are cached in `cache/`, so a rerun fetches only what is new.
2. `python scripts/update_gradient_term_data.py` writes the numbers into
   `gradient_term.py`.
3. `python scripts/make_mock_n0011.py` rewrites the mock N0011 matrices.
4. Rerun `gradient_term_prototype.ipynb` and `gradient_term_validation.ipynb`.

The astromon database is the only input from outside this repo. The
calibration also uses the Ska environment's cheta, kadi and cxotime.

## Tests

```
python -m pytest
```

## Where it came from

Copied from the astrometry repository, where the gradient term was chosen and
checked. On the 2026-11-05 astromon snapshot this repo reproduces astrometry's
calibration set exactly.
