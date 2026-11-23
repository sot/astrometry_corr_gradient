"""Generate notebooks/gradient_term_prototype.ipynb.

Run this script to (re)build the notebook from scratch; the notebook itself
is what gets committed and opened, this script is just how it's produced.
The gradient term's code and data live in astrometry_corr_gradient/gradient_term.py; the
notebook says what to implement, shows the coefficients, the reference table
and the test vectors, then prints the file. gradient_term_validation.ipynb
checks the residuals with it.
"""

from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).parent

markdown_intro = """\
# Gradient term: prototype

The gradient term corrects each observation's dz and dy for the periscope
gradients OOBAGRD3 and OOBAGRD6. CALALIGN is unchanged; the term goes on top
of it.

Everything is in one file, `astrometry_corr_gradient/gradient_term.py`: the
code, the numbers and the test vectors. It needs only numpy and astropy.

**What to implement**, for each observation, on dz and dy:

`term = c3 * (OOBAGRD3 - ref3) + c6 * (OOBAGRD6 - ref6)`

1. **Gradients:** the observation's mean OOBAGRD3 and OOBAGRD6, in K, over its
   dwell (from the end of its maneuver to the start of the next), with
   telemetry in safing intervals masked.
2. **Coefficients:** c3 and c6 for the axis, below.
3. **References:** ref3 and ref6, from the reference table's row for the
   calendar quarter (UTC) of the observation's start time (DATE-OBS).
4. **Result:** the term in arcsec, the expected X-ray minus catalog offset. The
   corrected offset is the observed one minus the term.
5. **No term** if a gradient is missing or the quarter has no row: the term is
   0 and the observation is flagged.
6. **Check:** match the test vectors below.

**Still open:**

- How the term enters processing: which positions or alignment it corrects,
  and with which sign there.
- The averaging window: the dwell, as here, or the observation's GTI span.

Below: the coefficients, the reference table, the test vectors, then the whole
file. `gradient_term_validation.ipynb` shows that the term works.
"""

code_setup = r"""
import inspect
import sys
from pathlib import Path

import numpy as np
from IPython.display import Markdown, display

# this checkout's package, before any installed copy
sys.path.insert(0, str(Path.cwd().parent))
from astrometry_corr_gradient import gradient_term
"""

markdown_module = """\
## The whole file

`astrometry_corr_gradient/gradient_term.py`. Its docstring, at the top, says
what to implement; the numbers are at the end. After a new calibration set,
`python scripts/update_gradient_term_data.py` rewrites them.
"""

code_module = r"""
print(inspect.getsource(gradient_term))
"""

markdown_coefficients = """\
## Coefficients

Two ways to write the same term:

- **Per K:** `term = c3 * (OOBAGRD3 - ref3) + c6 * (OOBAGRD6 - ref6)`
- **Per sigma:** `term = k3 * (OOBAGRD3 - ref3) / s3 + k6 * (OOBAGRD6 - ref6) / s6`

**ref3** and **ref6** are the quarter's references, from the table below.
**s** is how much each gradient typically moves inside a quarter. **k** = c x s
is the arcsec for that typical move, so the two gradients compare directly.
"""

code_coefficients = r"""
per_sigma = gradient_term.coefficients_per_sigma()
header = "| axis | gradient | c (arcsec per K) | s (K) | k (arcsec per sigma) |"
rows = [
    f"| {axis_name} | {name} | {gradient_term.COEFFICIENTS[axis_name][name]}"
    f" | {gradient_term.SIGMAS[name]} | {per_sigma[axis_name][name]:.4f} |"
    for axis_name in gradient_term.AXES
    for name in gradient_term.GRADIENT_MSIDS
]
display(Markdown("\n".join([header, "|---|---|---|---|---|", *rows])))
print("COEFFICIENTS =", gradient_term.COEFFICIENTS)
print("SIGMAS =", gradient_term.SIGMAS)
print(
    "per sigma =",
    {
        axis_name: {name: round(k, 4) for name, k in by_gradient.items()}
        for axis_name, by_gradient in per_sigma.items()
    },
)
"""

markdown_references = """\
## Reference table

One row per calendar quarter. Observations in that quarter use its OOBAGRD3
and OOBAGRD6, in K, as ref3 and ref6. `gradient_term()` only looks up the row;
how the row was made does not matter to it. `from_quarter` is the quarter
whose observations made the row; `observations` is how many.
"""

code_references = r"""
gradient_term.reference_table().pprint(max_lines=-1, max_width=-1)
"""

markdown_vectors = """\
## Test vectors

Real observations and the term each should get, in arcsec, next to what
`gradient_term()` gives. The last one has no telemetry, so it gets no term.
Any other implementation should match these too.
"""

code_vectors = r"""
vectors = gradient_term.vector_table()
term = gradient_term.gradient_term(
    vectors["date"], {name: vectors[name] for name in gradient_term.GRADIENT_MSIDS}
)
for axis_name in gradient_term.AXES:
    vectors[f"{axis_name} computed"] = np.round(term[axis_name], 4)
    vectors[f"{axis_name} computed"].unit = "arcsec"
vectors.pprint(max_width=-1)
"""

nb = nbf.v4.new_notebook()
nb.cells = [
    nbf.v4.new_markdown_cell(markdown_intro),
    nbf.v4.new_code_cell(code_setup.strip("\n")),
    nbf.v4.new_markdown_cell(markdown_coefficients),
    nbf.v4.new_code_cell(code_coefficients.strip("\n")),
    nbf.v4.new_markdown_cell(markdown_references),
    nbf.v4.new_code_cell(code_references.strip("\n")),
    nbf.v4.new_markdown_cell(markdown_vectors),
    nbf.v4.new_code_cell(code_vectors.strip("\n")),
    nbf.v4.new_markdown_cell(markdown_module),
    nbf.v4.new_code_cell(code_module.strip("\n")),
]

out_path = HERE / "gradient_term_prototype.ipynb"
nbf.write(nb, out_path)
print(f"wrote {out_path}")
