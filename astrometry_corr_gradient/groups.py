"""Linked groups and paired block-bootstrap differences.

Copied from astrometry's ``astrometry/evaluation.py``: the two functions the
calibration and its validation use.
"""

import numpy as np


def block_bootstrap_difference(
    residual_a: np.ndarray,
    residual_b: np.ndarray,
    years: np.ndarray,
    statistic,
    year_min: float = -np.inf,
    block_days: float = 30.0,
    n_boot: int = 2000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Paired block-bootstrap CI on ``statistic(a) - statistic(b)``.

    Both models are scored on the same resampled rows (paired), which is
    more powerful than comparing two separate confidence intervals.
    Resampling whole time-blocks rather than individual points preserves
    the short-term temporal correlation naive point resampling would
    ignore, which would otherwise be overconfident.

    Parameters
    ----------
    residual_a, residual_b : numpy.ndarray
        Two models' residuals, same length, aligned to the same rows.
    years : numpy.ndarray
        Decimal-year time of each row.
    statistic : callable
        A function of a residual array to a scalar (e.g. ``np.std``, or
        a closure wrapping :func:`residual_metrics`).
    year_min : float
        Only rows at or after this year are included.
    block_days : float
        Width of each resampling block, in days.
    n_boot : int
        Number of bootstrap resamples.
    seed : int
        RNG seed.

    Returns
    -------
    point_estimate, low, high : float
        ``statistic(a) - statistic(b)`` on the real data, and the 95%
        bootstrap interval for that difference. A CI that excludes zero
        means the difference is unlikely to be noise.
    """
    selected = (years >= year_min) & np.isfinite(residual_a) & np.isfinite(residual_b)
    a = residual_a[selected]
    b = residual_b[selected]
    t = years[selected]
    order = np.argsort(t)
    a, b, t = a[order], b[order], t[order]

    block_years = block_days / 365.25
    boundaries = np.arange(t.min(), t.max() + block_years, block_years)
    block_id = np.searchsorted(boundaries, t, side="right") - 1
    block_indices = [np.where(block_id == b_id)[0] for b_id in np.unique(block_id)]
    n_blocks = len(block_indices)

    point_estimate = float(statistic(a) - statistic(b))

    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        chosen_blocks = rng.integers(0, n_blocks, size=n_blocks)
        idx = np.concatenate([block_indices[c] for c in chosen_blocks])
        boot[i] = statistic(a[idx]) - statistic(b[idx])

    low, high = np.percentile(boot, [2.5, 97.5])
    return point_estimate, float(low), float(high)


def linked_group_ids(*label_arrays: np.ndarray) -> np.ndarray:
    """Group rows that share ANY label, transitively (connected components).

    For leave-out references: a row's reference should exclude not just its
    own obsid but every other row that shares something with it that could
    carry a common offset -- e.g. the same catalog counterpart observed in a
    different obsid (its catalog position error is common to both). Rows
    linked through a chain (obsid A shares a source with obsid B, which
    shares a source with obsid C) all land in one group.

    Parameters
    ----------
    *label_arrays : numpy.ndarray
        One or more label arrays, each one label per row, same length.

    Returns
    -------
    numpy.ndarray
        Integer group id per row; two rows share an id if and only if they
        are linked through any chain of shared labels.

    Examples
    --------
    >>> groups = linked_group_ids(np.array([1, 1, 2, 3]), np.array(["a", "b", "b", "c"]))
    >>> bool(groups[0] == groups[2]), bool(groups[0] == groups[3])
    (True, False)
    """
    n_rows = len(label_arrays[0])
    parent = np.arange(n_rows)

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for labels in label_arrays:
        first_row_with_label: dict = {}
        for row, label in enumerate(labels.tolist()):
            if label in first_row_with_label:
                a, b = root(row), root(first_row_with_label[label])
                if a != b:
                    parent[a] = b
            else:
                first_row_with_label[label] = row
    roots = np.array([root(row) for row in range(n_rows)])
    _unique_roots, group_ids = np.unique(roots, return_inverse=True)
    return group_ids.ravel()
