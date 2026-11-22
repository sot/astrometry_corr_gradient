"""Generate notebooks/gradient_term_prototype.ipynb.

Run this script to (re)build the notebook from scratch; the notebook itself
is what gets committed and opened, this script is just how it's produced.
The gradient term's code and data live in astrometry_corr_gradient/gradient_term.py; the
notebook imports it, prints it to review, and shows the coefficients, the
reference table and the test vectors. gradient_term_validation.ipynb checks
the residuals with it.
"""

from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).parent

markdown_intro = """\
# Gradient term: prototype for the developers

The gradient term corrects each observation's dz and dy for the periscope
gradients OOBAGRD3 and OOBAGRD6. It goes on top of the CALALIGN values.

All of it is in one file, `astrometry_corr_gradient/gradient_term.py`: the code, the numbers
and the test vectors. It needs only numpy and astropy.

This notebook prints that file, then shows:

1. the coefficients,
2. the reference table,
3. the test vectors.

`gradient_term_validation.ipynb` checks that the term works. After a new
export, `python scripts/update_gradient_term_data.py` remakes the numbers.
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
## The module

The whole file. Its docstring, at the top, explains the term.
"""

code_module = r"""
print(inspect.getsource(gradient_term))
"""

markdown_coefficients = """\
## Coefficients

Two ways to write the same term:

- **Per K:** `term = c3 * (OOBAGRD3 - ref) + c6 * (OOBAGRD6 - ref)`
- **Per sigma:** `term = k3 * (OOBAGRD3 - ref) / s3 + k6 * (OOBAGRD6 - ref) / s6`

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
and OOBAGRD6, in K. `gradient_term()` only looks up the row; how the row was
made does not matter to it. `from_quarter` and `observations` say where each
row's medians come from.
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
    nbf.v4.new_markdown_cell(markdown_module),
    nbf.v4.new_code_cell(code_module.strip("\n")),
    nbf.v4.new_markdown_cell(markdown_coefficients),
    nbf.v4.new_code_cell(code_coefficients.strip("\n")),
    nbf.v4.new_markdown_cell(markdown_references),
    nbf.v4.new_code_cell(code_references.strip("\n")),
    nbf.v4.new_markdown_cell(markdown_vectors),
    nbf.v4.new_code_cell(code_vectors.strip("\n")),
]

out_path = HERE / "gradient_term_prototype.ipynb"
nbf.write(nb, out_path)
print(f"wrote {out_path}")
