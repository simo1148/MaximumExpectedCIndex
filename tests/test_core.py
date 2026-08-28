import numpy as np
import pytest

from ecindex import ec_index


def test_ec_index_of_constant_series() -> None:
    assert ec_index([2.0, 2.0, 2.0]) == pytest.approx(2.0)


def test_ec_index_accepts_numpy_arrays() -> None:
    assert ec_index(np.array([0.0, 1.0])) == pytest.approx(0.5)


def test_ec_index_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="at least one observation"):
        ec_index([])
