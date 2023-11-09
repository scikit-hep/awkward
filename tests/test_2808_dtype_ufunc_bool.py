# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np
import packaging.version
import pytest

import awkward as ak

if packaging.version.Version(np.__version__) < packaging.version.Version("1.24.0"):
    pytest.skip("Only NumPy>=1.24 supports `resolve_dtypes`", allow_module_level=True)


def test_bool():
    array = ak.Array([[True, False], [True, False, False]])
    assert (array == False).to_list() == [[False, True], [False, True, True]]  # noqa: E712


def test_int():
    array = ak.Array([[1, 2], [3, 4, 5]])
    assert (array == 2).to_list() == [[False, True], [False, False, False]]


def test_float():
    array = ak.Array([[1.0, 2.7], [2.0, 7.0, 8.8]])
    assert (array == 1.0).to_list() == [[True, False], [False, False, False]]


def test_complex():
    array = ak.Array([[1.0j, 2.7], [2.0 + 1j, 7.0, 8.8]])
    assert (array == 1.0j).to_list() == [[True, False], [False, False, False]]


def test_datetime64():
    array = ak.Array([[np.datetime64(1, "D"), np.datetime64(10, "D")]])
    assert (array == np.datetime64(10, "D")).to_list() == [[False, True]]
