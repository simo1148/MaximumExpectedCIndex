"""Core computations for the EC index."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def ec_index(values: Iterable[float]) -> float:
    """Compute the EC index of ``values``.

    This is a placeholder implementation: replace the body with the real
    definition once it is settled.

    Parameters
    ----------
    values:
        Observations to summarise. Must be non-empty.

    Returns
    -------
    float
        The index value.

    Raises
    ------
    ValueError
        If ``values`` is empty.
    """
    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        raise ValueError("`values` must contain at least one observation")
    return float(array.mean())
