# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np
import pytest  # noqa: F401

import awkward as ak


def test_bool():
    result = ak.full_like([True], 2, dtype=np.int64)
    assert ak.almost_equal(result, np.asarray([2], dtype=np.int64))


def test_empty():
    result = ak.full_like([], 2, dtype=np.int64)
    assert result.layout.is_unknown
    result = ak.full_like([], 2, dtype=np.int64, including_unknown=True)
    assert result.layout.is_numpy
    assert result.layout.dtype == np.dtype(np.int64)


def test_complex128():
    result = ak.full_like([1, 2], 4j, dtype=np.complex128)
    assert ak.almost_equal(result, np.asarray([0 + 4j, 0 + 4j], dtype=np.complex128))
